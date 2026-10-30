/**
 * A Suite URL read against the suite's test case file: the test case filters.
 *
 * Values that name nothing are ignored with a notice; an unknown suite throws `NotFound`.
 */
import { loadSuiteFile } from './artifacts';
import { type Notice, linkNotice } from './notices';
import { type CaseRecordData, type LibraryData, NotFound, type Params, type SuiteFileData } from './types';
import { url } from './url';

export interface SuiteFileView {
  file: SuiteFileData;
  route: string | null;
  persona: string | null;
  /** The test cases the filters keep. */
  cases: CaseRecordData[];
  found: Notice[];
}

export function suiteFileLink(name: string, changes: Record<string, string | null> = {}): string {
  return url('/suite', { suite: name, ...changes });
}

export function suiteFileView(params: Params, library: LibraryData | null): SuiteFileView {
  const name = params.suite;
  const file = name ? loadSuiteFile(name) : null;
  if (!file || file.name !== name) throw new NotFound(`suite ${name ?? ''}`);
  const found: Notice[] = [];
  let route = params.route || null;
  if (route && !file.cases.some((c) => c.pathway_id === route)) {
    found.push(linkNotice(`Route ${route} not found; showing every route.`));
    route = null;
  }
  let persona = params.persona || null;
  if (persona && !file.cases.some((c) => c.persona_ref === persona)) {
    found.push(linkNotice(`Person ${persona} not found; showing every person.`));
    persona = null;
  }
  const cases = file.cases.filter(
    (c) => (route === null || c.pathway_id === route) && (persona === null || c.persona_ref === persona),
  );
  return { file, route, persona, cases, found };
}
