/**
 * The transcript view's data: the transcript, its latest judge record, and the list it was
 * opened from, which previous, next and the breadcrumb follow.
 */
import { loadCase, loadTranscripts } from './artifacts';
import { type Notice, linkNotice } from './notices';
import { runLink, runView } from './run-view';
import { suiteFileLink } from './suite-file-view';
import { testCaseView } from './test-case-view';
import { type CaseData, type JudgeView, type LibraryData, NotFound, type Params } from './types';
import { url } from './url';

export type Crumb = [label: string, href: string | null];

/** Where a case was opened from: the list its previous and next follow. */
export interface ListContext {
  kind: 'run' | 'test-case';
  href: string;
  keys: [run: string, transcript: string][];
  crumbs: Crumb[];
}

/** Builds the link context from a `from` value. Null when it names nothing, or leaves the site. */
export function listContext(cameFrom: string, library: LibraryData | null): ListContext | null {
  if (!cameFrom || !cameFrom.startsWith('/') || cameFrom.startsWith('//')) return null;
  let parsed: URL;
  try {
    parsed = new URL(cameFrom, 'http://viewer.invalid');
  } catch {
    return null;
  }
  if (parsed.host !== 'viewer.invalid') return null;
  const params: Params = {};
  for (const [k, v] of parsed.searchParams) if (!(k in params) && v) params[k] = v;
  try {
    if (parsed.pathname === '/test-case') {
      const view = testCaseView(params);
      return {
        kind: 'test-case',
        href: cameFrom,
        keys: view.transcripts.map((t) => [t.run, t.id]),
        crumbs: [
          ['Suites', '/suites'],
          [view.file.name, suiteFileLink(view.file.name)],
          [view.record?.display_name || view.id, cameFrom],
        ],
      };
    }
    if (parsed.pathname === '/run') {
      const view = runView(params);
      return {
        kind: 'run',
        href: cameFrom,
        keys: view.rows.map((t) => [view.run.name, t.id]),
        crumbs: [
          ['Runs', '/runs'],
          [view.run.name, runLink(view)],
        ],
      };
    }
  } catch (error) {
    if (error instanceof NotFound) return null;
    throw error;
  }
  return null;
}

/** The transcript named by `transcript` (and `run`, when given). Throws `NotFound`. */
export function findCase(transcriptId: string | undefined, run: string | undefined): [CaseData, Notice[]] {
  if (!transcriptId) throw new NotFound('transcript');
  if (run) {
    const found = loadCase(run, transcriptId);
    if (!found || found.transcript_id !== transcriptId) throw new NotFound(`transcript ${transcriptId}`);
    return [found, []];
  }
  const places = [...(loadTranscripts().places[transcriptId] ?? [])].sort((a, b) => (a.run < b.run ? -1 : 1));
  const last = places.at(-1);
  const found = last ? loadCase(last.run, transcriptId) : null;
  if (!last || !found) throw new NotFound(`transcript ${transcriptId}`);
  const notes =
    places.length > 1
      ? [linkNotice(`This transcript id appears in ${places.length} runs; showing ${last.run}.`)]
      : [];
  return [found, notes];
}

/** The latest judge record in the run, as Run mode's rows count it. */
export function judgeFor(data: CaseData): JudgeView | null {
  return data.judges.reduce<JudgeView | null>(
    (best, j) => (best === null || j.evaluated_at > best.evaluated_at ? j : best),
    null,
  );
}

export function caseLink(run: string, transcript: string, cameFrom: string | null): string {
  return url('/transcript', { run, id: transcript, from: cameFrom });
}
