/** Stops a build with nothing ingested, naming the command that writes the page data. */
import process from 'node:process';
import { MissingArtifact, loadIndex } from '../src/lib/artifacts';

try {
  loadIndex();
} catch (error) {
  if (!(error instanceof MissingArtifact)) throw error;
  console.error(
    'No ingested data to build from. Run the ingest first:\n\n' +
      '    pixi run -e viewer viewer-ingest                                         # data-repository/results\n' +
      '    pixi run -e viewer viewer-ingest --results data-repository/sample_results  # lab.coralmind.org\n',
  );
  process.exit(2);
}
