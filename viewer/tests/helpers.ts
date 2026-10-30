/** Loads a scenario written by tests/viewer/write_fixtures.py and renders pages from it. */
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { ReactNode } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { url } from '../src/lib/url';

export const FIXTURES = join(import.meta.dirname, 'fixtures', 'generated');

export interface Meta {
  /** Suite version ids, `<name>@<digest>`. */
  suites: string[];
  /** Suite names: test case files under test_case_records/. */
  names: string[];
  transcripts: { run: string; id: string }[];
}

/** Points the artifacts reader at what the ingest wrote for a scenario. */
export function use(name: string): Meta {
  process.env.LAB_GENERATED_DIR = join(FIXTURES, name, 'gen');
  return JSON.parse(readFileSync(join(FIXTURES, name, 'meta.json'), 'utf-8')) as Meta;
}

export function html(node: ReactNode): string {
  return renderToStaticMarkup(node);
}

type Route = (props: { searchParams: Promise<Record<string, string>> }) => Promise<ReactNode> | ReactNode;

/** Renders a page component as Next would for a URL query. */
export async function page(route: Route, params: Record<string, string> = {}): Promise<string> {
  return html(await route({ searchParams: Promise.resolve(params) }));
}

/** The digest Next's notFound() and redirect() throw, or null. */
export async function thrown(run: () => unknown): Promise<string | null> {
  try {
    await run();
  } catch (error) {
    return String((error as { digest?: string }).digest ?? (error as Error).message);
  }
  return null;
}

export function transcriptId(caseId: string, sample = 1, sut = 'azure-gpt-5-4-mini'): string {
  return `demo:${sut}:${caseId}:${sample}`;
}

export function caseParams(caseId: string, sample = 1, from = ''): Record<string, string> {
  const params: Record<string, string> = { run: 'run-a', id: transcriptId(caseId, sample) };
  if (from) params.from = from;
  return params;
}

export { url };

/** Undo the HTML escaping of an attribute value. */
export function unescape(value: string): string {
  return value.replaceAll('&amp;', '&').replaceAll('&quot;', '"').replaceAll('&#x27;', "'");
}
