/**
 * Starts `next start` on the build and checks what component tests cannot see: real status
 * codes, the Introduction at `/`, robots.txt, the robots header, and the refusal of AI agents.
 * Run with `pixi run -e viewer viewer-smoke` after `pixi run -e viewer viewer-build`.
 */
import { spawn } from 'node:child_process';
import process from 'node:process';

const PORT = Number(process.env.SMOKE_PORT ?? 3210);
const BASE = `http://127.0.0.1:${PORT}`;

// Its own process group, so `stop` reaches next-server too. Killing only the direct child can
// leave next-server running, and the script then never exits.
const server = spawn(
  process.execPath,
  ['node_modules/next/dist/bin/next', 'start', '--port', String(PORT), '--hostname', '127.0.0.1'],
  { stdio: ['ignore', 'ignore', 'inherit'], env: process.env, detached: true },
);

function stop(): void {
  try {
    process.kill(-server.pid!, 'SIGTERM');
  } catch {
    // Already gone.
  }
}

async function ready(): Promise<void> {
  for (let i = 0; i < 100; i += 1) {
    try {
      await fetch(`${BASE}/robots.txt`);
      return;
    } catch {
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw new Error('next start did not answer');
}

const failures: string[] = [];

async function expectStatus(path: string, status: number, init: RequestInit = {}): Promise<Response> {
  const response = await fetch(`${BASE}${path}`, { redirect: 'manual', ...init });
  if (response.status !== status) failures.push(`${path}: ${response.status}, expected ${status}`);
  return response;
}

try {
  await ready();
  for (const path of [
    '/suite?suite=nope',
    '/run?name=nope',
    '/transcript?run=nope&id=nope',
    '/record?run=nope&transcript=nope&kind=transcript',
    '/judge-call?run=nope&transcript=nope&seq=1&part=prompt',
    '/elsewhere',
  ]) {
    await expectStatus(path, 404);
  }
  const home = await expectStatus('/', 200);
  if (!(await home.text()).includes('The evaluation pipeline')) failures.push('/: no Introduction');
  const suites = await expectStatus('/suites', 200);
  if (!suites.headers.get('x-robots-tag')?.includes('noai')) failures.push('/suites: no X-Robots-Tag');
  await expectStatus('/robots.txt', 200, { headers: { 'user-agent': 'GPTBot/1.2' } });
  await expectStatus('/suites', 403, { headers: { 'user-agent': 'Mozilla/5.0 GPTBot/1.2' } });
} catch (error) {
  failures.push(String(error));
} finally {
  stop();
}

if (failures.length) {
  console.error(`smoke:\n  ${failures.join('\n  ')}`);
  process.exit(1);
}
console.log('smoke: OK.');
