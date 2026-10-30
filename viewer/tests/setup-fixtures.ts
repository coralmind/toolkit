/** Fails the run with the command to use when the Python-written fixtures are missing. */
import { existsSync } from 'node:fs';
import { FIXTURES } from './helpers';

export default function setup(): void {
  if (!existsSync(FIXTURES)) {
    throw new Error(
      `No test fixtures at ${FIXTURES}.\n\n` +
        'The page tests render data the Python ingest writes from the pytest record builders. Run:\n\n' +
        '    pixi run -e viewer viewer-test\n',
    );
  }
}
