/** The Suite page: one test case file and its test cases. */
import type { ReactNode } from 'react';
import { BUCKETS, BUCKET_LABELS, levelLabel } from '../lib/labels';
import type { SuiteFileView } from '../lib/suite-file-view';
import { testCaseLink } from '../lib/test-case-view';
import type { LibraryData, SuiteFileData } from '../lib/types';
import { ContextLine } from './Shell';
import { BucketBar, Chip, DefinitionList, DirectionKeyItems, Ident, Key, PageHead } from './bits';

interface Props {
  view: SuiteFileView;
  library: LibraryData | null;
}

export function SuiteFilePage({ view, library }: Props) {
  const { file } = view;
  return (
    <>
      <SuiteHead file={file} />
      <CasesPanel view={view} library={library} />
    </>
  );
}

/** The levels of the file's test cases, in order. */
function caseLevels(file: SuiteFileData): number[] {
  return [...new Set(file.cases.map((c) => c.level ?? 0))].sort();
}

function SuiteHead({ file }: { file: SuiteFileData }) {
  const current = file.versions.find((v) => v.current);
  const levels = caseLevels(file);
  return (
    <PageHead eyebrow="Suite" title={file.name} context={<ContextLine kind="suite" />}>
      <DefinitionList
        items={[
          [
            'File',
            file.source ? (
              <>
                <Ident>{file.source}</Ident>
                {file.in_records ? null : (
                  <span className="vw-muted"> (a run&apos;s copy; the file is not in test_case_records/)</span>
                )}
              </>
            ) : (
              ''
            ),
          ],
          [
            'Test cases',
            <>
              {file.cases.length}
              {file.case_problems.length ? (
                <span className="vw-muted"> · {file.case_problems.length} line(s) unreadable</span>
              ) : null}
            </>,
          ],
          ...(levels.length === 1 ? [['Level', levelLabel(levels[0])] as [string, ReactNode]] : []),
          [
            'Runs',
            file.runs.length
              ? `${file.runs.length}, using ${file.versions.length} ${file.versions.length === 1 ? 'version' : 'versions'} of the file${
                  current ? '' : '; none used the current file'
                }`
              : 'none yet',
          ],
        ]}
      />
    </PageHead>
  );
}

function CasesPanel({ view, library }: { view: SuiteFileView; library: LibraryData | null }) {
  const { file } = view;
  const routes = [...new Set(file.cases.map((c) => c.pathway_id))];
  const people = [...new Set(file.cases.map((c) => c.persona_ref))].sort();
  const buckets = new Map<string, Map<string, number>>();
  for (const t of file.transcripts) {
    const counts = buckets.get(t.test_case_id) ?? new Map<string, number>();
    counts.set(t.bucket, (counts.get(t.bucket) ?? 0) + 1);
    buckets.set(t.test_case_id, counts);
  }
  const mixedLevels = caseLevels(file).length > 1;
  const routeLabel = (r: string) => {
    const entry = library?.routes[r];
    return entry ? `${entry.did} · ${entry.label}` : r;
  };
  return (
    <>
      <form className="vw-filters vw-picker" method="get" action="/suite">
        <input type="hidden" name="suite" value={file.name} />
        <label>
          <span>Route</span>
          <select name="route" defaultValue={view.route ?? ''}>
            <option value="">every route</option>
            {routes.map((r) => (
              <option key={r} value={r}>
                {routeLabel(r)}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>Person</span>
          <select name="persona" defaultValue={view.persona ?? ''}>
            <option value="">every person</option>
            {people.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <button type="submit" className="cm-btn">
          Show
        </button>
      </form>
      <p className="vw-meta">
        {view.cases.length} of {file.cases.length} test cases
      </p>
      <Key>
        <DirectionKeyItems />
      </Key>
      <table className="vw-table">
        <thead>
          <tr>
            <th scope="col">Test case</th>
            <th scope="col">Route</th>
            <th scope="col">Person</th>
            {mixedLevels ? <th scope="col">Level</th> : null}
            <th scope="col" className="num">
              Transcripts
            </th>
            <th scope="col">Transcript status</th>
          </tr>
        </thead>
        <tbody>
          {view.cases.map((c) => {
            const counts = buckets.get(c.test_case_id) ?? new Map<string, number>();
            const total = [...counts.values()].reduce((a, b) => a + b, 0);
            return (
              <tr key={c.test_case_id}>
                <th scope="row">
                  <a href={testCaseLink(file.name, c.test_case_id)}>{c.display_name || c.test_case_id}</a>
                  {c.display_name ? (
                    <span className="vw-rowmeta">
                      <Ident>{c.test_case_id}</Ident>
                    </span>
                  ) : null}
                </th>
                <td>
                  <span className="vw-routecell">
                    <Chip id={c.pathway_id} library={library} />
                    <span>{library?.names[c.pathway_id]?.label ?? ''}</span>
                  </span>
                </td>
                <td>
                  <Ident>{c.persona_ref}</Ident>
                </td>
                {mixedLevels ? <td>{levelLabel(c.level)}</td> : null}
                <td className="num">{total}</td>
                <td className="vw-transcriptstatus">
                  {total ? (
                    <>
                      <BucketBar counts={counts} total={total} />
                      {BUCKETS.filter((b) => counts.get(b))
                        .map((b) => `${BUCKET_LABELS[b]} ${counts.get(b)}`.replaceAll(' ', '\u00a0'))
                        .join(', ')}
                    </>
                  ) : (
                    <span className="vw-muted">none yet</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </>
  );
}

/** Which version of the file a run used, against the current file. */
