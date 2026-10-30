/**
 * The single place generated page data is read. Do not add a second reader, a fixture, or
 * inlined results anywhere in viewer/src.
 *
 * The dev server re-reads a file when the watcher has rewritten it; a production server
 * reads each file once. Paths are built at runtime, so the tracer is told to skip them;
 * `next.config.ts` names what each route ships.
 */
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import type {
  CaseData,
  IndexData,
  LibraryData,
  RecordsData,
  RunData,
  StatusData,
  SuiteFileData,
  TranscriptsData,
} from './types';

/** The command that writes the generated data, named in every missing-data error. */
export const INGEST_FIX = 'pixi run -e viewer viewer-dev';

export class MissingArtifact extends Error {}

/**
 * Anchored to the working directory rather than `import.meta.url`, because bundlers move
 * this module into build output. npm runs scripts from `viewer/`.
 */
export function generatedDir(): string {
  return process.env.LAB_GENERATED_DIR ?? join(/*turbopackIgnore: true*/ process.cwd(), 'src', 'generated');
}

const cache = new Map<string, { mtimeMs: number; value: unknown }>();

function readCached<T>(path: string): T | null {
  if (!existsSync(path)) return null;
  const mtimeMs = process.env.NODE_ENV === 'production' ? 0 : statSync(path).mtimeMs;
  const hit = cache.get(path);
  if (hit && hit.mtimeMs === mtimeMs) return hit.value as T;
  const value = JSON.parse(readFileSync(path, 'utf-8')) as T;
  cache.set(path, { mtimeMs, value });
  return value;
}

/** Throws with the fix when a file every page needs is missing, as on a fresh checkout. */
function required<T>(name: string): T {
  const path = join(/*turbopackIgnore: true*/ generatedDir(), name);
  const value = readCached<T>(path);
  if (value === null) {
    throw new MissingArtifact(
      `Could not read generated results data at ${path}.\n\n` +
        `The viewer reads results through an ingest step, which has to run first:\n\n` +
        `    ${INGEST_FIX}\n`,
    );
  }
  return value;
}

/** A page's own file, or null when the id names nothing. */
function optional<T>(...segments: string[]): T | null {
  return readCached<T>(join(/*turbopackIgnore: true*/ generatedDir(), ...segments));
}

/** The file key of a suite, run or transcript, as the ingest writes it. */
export function keyOf(...parts: string[]): string {
  return createHash('sha1').update(parts.join('\0'), 'utf8').digest('hex').slice(0, 20);
}

export function loadIndex(): IndexData {
  return required<IndexData>('index.json');
}

export function loadLibrary(): LibraryData {
  return required<LibraryData>('library.json');
}

export function loadTranscripts(): TranscriptsData {
  return required<TranscriptsData>('transcripts.json');
}

/** Why the last re-ingest failed, or null when it succeeded or no status was written. */
export function loadStatus(): StatusData | null {
  return optional<StatusData>('status.json');
}

/** A test case file's page data, by suite name. */
export function loadSuiteFile(name: string): SuiteFileData | null {
  return optional<SuiteFileData>('suite-files', `${keyOf(name)}.json`);
}

export function loadRun(name: string): RunData | null {
  return optional<RunData>('runs', `${keyOf(name)}.json`);
}

export function loadCase(run: string, transcriptId: string): CaseData | null {
  return optional<CaseData>('cases', `${keyOf(run, transcriptId)}.json`);
}

/** The records a transcript page links to, as the ingest copied them. */
export function loadRecords(run: string, transcriptId: string): RecordsData | null {
  const data = optional<RecordsData>('records', `${keyOf(run, transcriptId)}.json`);
  return data?.transcript_id === transcriptId ? data : null;
}
