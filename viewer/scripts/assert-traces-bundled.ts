/**
 * Fails a build whose functions would not ship the generated data, or would ship anything
 * from data-repository/.
 *
 * Every page and route renders on request from the generated data, so every route must carry
 * all of it. Reads the `.nft.json` trace manifests, which are what deployments package from.
 */
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { dirname, join, normalize, relative } from 'node:path';
import process from 'node:process';
import { filesUnder } from './leaks';

const viewer = process.cwd();
const repo = join(viewer, '..');
const server = join(viewer, '.next', 'server', 'app');
const manifests = existsSync(server)
  ? readdirSync(server, { recursive: true, encoding: 'utf-8' }).filter((name) => name.endsWith('.nft.json'))
  : [];
if (!manifests.length) {
  console.error(`assert-traces-bundled: found no trace manifests under ${server}; the build has not run, or Next moved them.`);
  process.exit(2);
}

const generated = filesUnder(join(viewer, 'src', 'generated')).map((p) => relative(repo, p));
const problems: string[] = [];

for (const manifest of manifests) {
  const parsed = JSON.parse(readFileSync(join(server, manifest), 'utf-8')) as { files?: unknown };
  if (!Array.isArray(parsed.files)) {
    console.error(`assert-traces-bundled: ${manifest} has no \`files\` array; the manifest format changed.`);
    process.exit(2);
  }
  const shipped = new Set(
    (parsed.files as string[]).map((f) => relative(repo, normalize(join(server, dirname(manifest), f)))),
  );
  if (manifest.startsWith('_')) continue;
  for (const file of generated) if (!shipped.has(file)) problems.push(`${manifest} lacks ${file}`);
  for (const file of shipped) if (file.startsWith('data-repository/')) problems.push(`${manifest} ships ${file}`);
}

if (problems.length) {
  console.error('assert-traces-bundled: the deployment would not ship the generated data, or ships results directly:\n');
  for (const problem of problems.slice(0, 20)) console.error(`  ${problem}`);
  process.exit(1);
}
console.log(`assert-traces-bundled: OK (${manifests.length} routes).`);
