/**
 * The Test case page: the record as the current file has it, and every transcript made from
 * it across the suite's runs, narrowed by the model in each role.
 */
import { caseLink } from '../lib/case-view';
import { BUCKET_LABELS, OUTCOME_LABELS, words } from '../lib/labels';
import { suiteFileLink } from '../lib/suite-file-view';
import { ROLE_FILTERS, type TestCaseView, testCaseLink } from '../lib/test-case-view';
import type { LibraryData } from '../lib/types';
import { url } from '../lib/url';
import { CaseRecord } from './CasePage';
import { RoleKey, RolePills, when } from './Lists';
import { ROLE_LABELS } from './Pipeline';
import { ContextLine } from './Shell';
import { DefinitionList, Ident, PageHead, Section } from './bits';

interface Props {
  view: TestCaseView;
  library: LibraryData | null;
}

export function TestCasePage({ view, library }: Props) {
  const { file, record } = view;
  const version = record ? String(record.record.version ?? '') : '';
  return (
    <>
      <PageHead
        eyebrow="Test case"
        title={record?.display_name || view.id}
        context={<ContextLine kind="suite" />}
      >
        <DefinitionList
          items={[
            [
              'Suite',
              <a key="s" href={suiteFileLink(file.name)}>
                {file.name}
              </a>,
            ],
            ['Id', record?.display_name ? <Ident key="i">{view.id}</Ident> : ''],
            ['Version', version],
          ]}
        />
      </PageHead>
      <Section title="The test case" id="record">
        {record ? (
          <CaseRecord c={record} library={library} simulated={record.level === 3} />
        ) : (
          <p className="vw-muted">
            This test case is not in the current file. Its transcripts below come from runs that used an older version.
          </p>
        )}
      </Section>
      <Section title="Transcripts" id="transcripts">
        <TranscriptFilters view={view} />
        <p className="vw-meta">
          {view.transcripts.length} of {view.all.length} transcripts
        </p>
        {view.transcripts.length ? (
          <>
            <RoleKey />
            <table className="vw-table vw-casetranscripts">
              <thead>
                <tr>
                  <th scope="col">Transcript</th>
                  <th scope="col">Run</th>
                  <th scope="col">Models</th>
                  <th scope="col">Fidelity check</th>
                  <th scope="col">Result</th>
                  <th scope="col">Test case used</th>
                </tr>
              </thead>
              <tbody>
                {view.transcripts.map((t) => (
                  <tr key={`${t.run}:${t.id}`}>
                    <th scope="row">
                      <a href={caseLink(t.run, t.id, testCaseLink(file.name, view.id))}>Open</a>
                    </th>
                    <td>
                      <a href={url('/run', { name: t.run })}>{t.run}</a>
                      <br />
                      <span className="vw-muted">{when(t.created)}</span>
                    </td>
                    <td>
                      <RolePills
                        suts={[t.sut]}
                        simulators={t.simulator ? [t.simulator] : []}
                        checkers={t.checker ? [t.checker] : []}
                        judges={t.judge ? [t.judge] : []}
                      />
                    </td>
                    <td>{t.fidelity ? words(t.fidelity) : <span className="vw-muted">not checked</span>}</td>
                    <td>
                      {t.outcome ? (
                        <>
                          target pathway{' '}
                          <span className={`vw-outcome o-${t.outcome}`}>
                            {OUTCOME_LABELS[t.outcome] ?? words(t.outcome)}
                          </span>
                        </>
                      ) : (
                        (BUCKET_LABELS[t.bucket] ?? t.bucket)
                      )}
                    </td>
                    <td>{t.same_case ? 'as it is now' : <span className="vw-muted">an older version</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <p className="vw-muted">No transcript has been made from this test case yet.</p>
        )}
      </Section>
    </>
  );
}

function TranscriptFilters({ view }: { view: TestCaseView }) {
  if (ROLE_FILTERS.every((role) => view.options[role].length < 2)) return null;
  return (
    <form className="vw-filters vw-picker" method="get" action="/test-case">
      <input type="hidden" name="suite" value={view.file.name} />
      <input type="hidden" name="id" value={view.id} />
      {ROLE_FILTERS.filter((role) => view.options[role].length > 1).map((role) => (
        <label key={role}>
          <span>{ROLE_LABELS[role]}</span>
          <select name={role} defaultValue={view.filters[role] ?? ''}>
            <option value="">any</option>
            {view.options[role].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      ))}
      <button type="submit" className="cm-btn">
        Show
      </button>
    </form>
  );
}
