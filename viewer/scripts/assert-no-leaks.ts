/**
 * Fails when the generated page data holds a home path, an endpoint address or an internal
 * field. Not part of the build: the Viewer workflow runs it with `pixi run -e viewer viewer-leaks`.
 */
import { join } from 'node:path';
import process from 'node:process';
import { generatedDir } from '../src/lib/artifacts';
import { scan } from './leaks';

const leaks = scan([generatedDir()], join(process.cwd(), '..'));
if (leaks.length) {
  console.error('assert-no-leaks: the generated data holds private values:\n');
  for (const leak of leaks.slice(0, 20)) console.error(`  ${leak.file}: ${leak.kind} (${leak.sample})`);
  if (leaks.length > 20) console.error(`  … and ${leaks.length - 20} more`);
  process.exit(1);
}
console.log('assert-no-leaks: OK.');
