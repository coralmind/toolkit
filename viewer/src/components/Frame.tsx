/**
 * What every page reads before its own data: the index, the library and the ingest's status,
 * plus the page frame around the result.
 */
import { notFound } from 'next/navigation';
import type { ReactNode } from 'react';
import { INGEST_FIX, MissingArtifact, loadIndex, loadLibrary, loadStatus } from '../lib/artifacts';
import type { Notice } from '../lib/notices';
import { type IndexData, type LibraryData, NotFound } from '../lib/types';
import { type Section, Shell, type Stamp } from './Shell';

export interface PageContext {
  index: IndexData;
  library: LibraryData | null;
  stamp: Stamp;
  /** The stale notice when the watcher's last re-ingest failed. */
  top: Notice[];
}

export function pageContext(): PageContext {
  const index = loadIndex();
  const library = index.library ? loadLibrary() : null;
  const status = loadStatus();
  const stale = Boolean(status?.stale_since && status.error);
  const top: Notice[] = stale
    ? [{ kind: 'stale', text: `Results index is stale since ${status!.stale_since!.slice(11, 19)}: ${status!.error}` }]
    : [];
  return { index, library, stamp: { builtAt: index.built_at, root: index.root, stale }, top };
}

export interface PageResult {
  section: Section;
  notices?: Notice[];
  body: ReactNode;
}

function MissingData({ message }: { message: string }) {
  return (
    <>
      <h1>Could not read results</h1>
      <p className="vw-lede">The viewer reads results through an ingest step, which has to run first:</p>
      <pre className="vw-json">{INGEST_FIX}</pre>
      <p className="vw-muted">{message.split('\n')[0]}</p>
    </>
  );
}

/** Renders a page in its frame. Missing data shows its fix; an unknown id is a 404. */
export function renderPage(section: Section, build: (ctx: PageContext) => PageResult): ReactNode {
  let ctx: PageContext;
  try {
    ctx = pageContext();
  } catch (error) {
    if (!(error instanceof MissingArtifact)) throw error;
    return (
      <Shell section={section} stamp={null}>
        <MissingData message={error.message} />
      </Shell>
    );
  }
  let result: PageResult;
  try {
    result = build(ctx);
  } catch (error) {
    if (error instanceof NotFound) notFound();
    throw error;
  }
  return (
    <Shell section={result.section} stamp={ctx.stamp} notices={[...ctx.top, ...(result.notices ?? [])]}>
      {result.body}
    </Shell>
  );
}

export function title(name: string): string {
  return `${name} · CoralMind Evaluation Lab`;
}
