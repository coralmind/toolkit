/**
 * A Test case URL read against its suite: the record from the current file, and the
 * transcripts made from it in every run, narrowed by the model in each role.
 */
import { loadSuiteFile } from './artifacts';
import { type Notice, linkNotice } from './notices';
import { type CaseRecordData, type CaseTranscriptRow, NotFound, type Params, type SuiteFileData } from './types';
import { url } from './url';

export const ROLE_FILTERS = ['sut', 'simulator', 'checker', 'judge'] as const;
export type RoleFilter = (typeof ROLE_FILTERS)[number];

export interface TestCaseView {
  file: SuiteFileData;
  id: string;
  /** The test case as the current file has it; null when only older versions had it. */
  record: CaseRecordData | null;
  all: CaseTranscriptRow[];
  transcripts: CaseTranscriptRow[];
  filters: Record<RoleFilter, string | null>;
  options: Record<RoleFilter, string[]>;
  found: Notice[];
}

export function testCaseLink(suite: string, id: string, changes: Record<string, string | null> = {}): string {
  return url('/test-case', { suite, id, ...changes });
}

export function testCaseView(params: Params): TestCaseView {
  const file = params.suite ? loadSuiteFile(params.suite) : null;
  const id = params.id ?? '';
  if (!file || file.name !== params.suite) throw new NotFound(`suite ${params.suite ?? ''}`);
  const record = file.cases.find((c) => c.test_case_id === id) ?? null;
  const all = file.transcripts.filter((t) => t.test_case_id === id);
  if (!record && !all.length) throw new NotFound(`test case ${id}`);
  const found: Notice[] = [];
  const options = {} as Record<RoleFilter, string[]>;
  const filters = {} as Record<RoleFilter, string | null>;
  for (const role of ROLE_FILTERS) {
    options[role] = [...new Set(all.map((t) => t[role]).filter(Boolean))].sort();
    let wanted: string | null = params[role] || null;
    if (wanted && !options[role].includes(wanted)) {
      found.push(linkNotice(`${wanted} not found among this test case's transcripts; showing every model.`));
      wanted = null;
    }
    filters[role] = wanted;
  }
  const transcripts = all.filter((t) =>
    ROLE_FILTERS.every((role) => filters[role] === null || t[role] === filters[role]),
  );
  return { file, id, record, all, transcripts, filters, options, found };
}
