/** The Suites and Runs lists: one ruled table each. */
import type { ReactNode } from 'react';
import { BUCKETS, BUCKET_LABELS, levelLabel } from '../lib/labels';
import type { IndexData, RunListRow, SuiteListRow, SuiteRunRow } from '../lib/types';
import { url } from '../lib/url';
import { BucketBar, Ident, Key, PageHead, Section } from './bits';
import { type Role, RolePill } from './Pipeline';

/** A run's or suite's time as "YYYY-MM-DD HH:MM", read from ISO local time. */
export function when(iso: string): string {
  return iso ? `${iso.slice(0, 10)} ${iso.slice(11, 16)}` : '';
}

function NoResults({ index }: { index: IndexData }) {
  return (
    <p className="vw-lede">
      No results yet in <Ident>{index.root}</Ident>. Run <code>pixi run sim</code> or <code>pixi run run</code> to
      create some.
    </p>
  );
}

/** A run's models, one pill per model, in role order. */
export function RolePills({
  suts = [],
  simulators = [],
  checkers = [],
  judges = [],
}: {
  suts?: string[];
  simulators?: string[];
  checkers?: string[];
  judges?: string[];
}) {
  const pills: [Role, string][] = [
    ...suts.map((m): [Role, string] => ['sut', m]),
    ...simulators.map((m): [Role, string] => ['simulator', m]),
    ...checkers.map((m): [Role, string] => ['checker', m]),
    ...judges.map((m): [Role, string] => ['judge', m]),
  ];
  if (!pills.length) return <span className="vw-muted">none</span>;
  return (
    <span className="vw-pills">
      {pills.map(([role, model]) => (
        <RolePill key={`${role}-${model}`} role={role}>
          {model}
        </RolePill>
      ))}
    </span>
  );
}

/** The key for role pills: one pill per role, placed above the table that uses them. */
export function RoleKey() {
  return (
    <Key>
      <RolePill role="sut" />
      <RolePill role="simulator" />
      <RolePill role="checker" />
      <RolePill role="judge" />
    </Key>
  );
}

/** Problems a suite's runs share: mixed models in a role, or another Knowledge Library. */
export function suiteWarnings(suite: SuiteListRow): string[] {
  const out: string[] = [];
  if (suite.simulators.length > 1) out.push('runs used more than one simulator');
  if (suite.checkers.length > 1) out.push('runs used more than one fidelity checker');
  if (suite.other_library) out.push('judged with a different Knowledge Library');
  if (suite.unreadable_case_lines) out.push(`${suite.unreadable_case_lines} test case line(s) unreadable`);
  return out;
}

export function SuitesList({ index }: { index: IndexData }) {
  const outside = index.runs.filter((r) => !r.suite).length;
  return (
    <>
      <PageHead eyebrow="Results" title="Suites" />
      <p className="vw-lede">A suite is one test case file and every run made from it.</p>
      {!index.suites.length && !index.runs.length ? <NoResults index={index} /> : null}
      {index.suites.length ? (
        <table className="vw-table">
          <thead>
            <tr>
              <th scope="col">Suite</th>
              <th scope="col" className="num">
                Test cases
              </th>
              <th scope="col" className="num">
                Runs
              </th>
              <th scope="col">Systems tested</th>
              <th scope="col">Last run</th>
            </tr>
          </thead>
          <tbody>
            {index.suites.map((suite) => {
              const warnings = suiteWarnings(suite);
              return (
                <tr key={suite.name}>
                  <th scope="row">
                    <a href={url('/suite', { suite: suite.name })}>{suite.name}</a>
                    {warnings.length ? <span className="vw-rowflags">{warnings.join('; ')}</span> : null}
                  </th>
                  <td className="num">{suite.cases}</td>
                  <td className="num">{suite.runs}</td>
                  <td>
                    <RolePills suts={suite.suts} />
                  </td>
                  <td className="vw-nowrap">{when(suite.last_run)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
      {outside ? (
        <p className="vw-meta">
          <a href="/runs">
            {outside} {outside === 1 ? 'run is' : 'runs are'} not in a suite
          </a>
        </p>
      ) : null}
    </>
  );
}

type RunRow = RunListRow | SuiteRunRow;

/** How far a run's transcripts got: a bar, then the counts in words. */
function TranscriptStatus({ run }: { run: RunRow }) {
  const present = BUCKETS.filter((b) => run.buckets[b]);
  if (!present.length) return <span className="vw-muted">none yet</span>;
  return (
    <>
      <BucketBar counts={new Map(Object.entries(run.buckets))} total={run.transcripts} />
      {present.map((b) => `${BUCKET_LABELS[b]} ${run.buckets[b]}`.replaceAll(' ', '\u00a0')).join(', ')}
    </>
  );
}

/**
 * The runs table, the same on the Runs page and a suite's Runs tab. ``place`` is the third
 * column: the run's suite, or which version of the suite's file it used.
 */
export function RunsTable<T extends RunRow>({
  runs,
  place,
}: {
  runs: T[];
  place: { label: string; cell: (run: T) => ReactNode };
}) {
  return (
    <>
      <RoleKey />
      <table className="vw-table vw-runs">
        <thead>
          <tr>
            <th scope="col">Run</th>
            <th scope="col">Date</th>
            <th scope="col">{place.label}</th>
            <th scope="col">Models</th>
            <th scope="col" className="num">
              Transcripts
            </th>
            <th scope="col">Transcript status</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => (
            <tr key={run.name}>
              <th scope="row">
                <a href={url('/run', { name: run.name })}>{run.name}</a>
                <span className="vw-rowmeta">{levelLabel(run.level)}</span>
              </th>
              <td className="vw-nowrap">{when(run.created)}</td>
              <td>{place.cell(run)}</td>
              <td>
                <RolePills
                  suts={run.suts}
                  simulators={run.simulator ? [run.simulator] : []}
                  checkers={run.checker ? [run.checker] : []}
                  judges={run.judges}
                />
              </td>
              <td className="num">{run.transcripts}</td>
              <td className="vw-transcriptstatus">
                <TranscriptStatus run={run} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export function RunsList({ index }: { index: IndexData }) {
  return (
    <>
      <PageHead eyebrow="Results" title="Runs" />
      {!index.runs.length ? (
        <NoResults index={index} />
      ) : (
        <RunsTable
          runs={index.runs}
          place={{
            label: 'Suite',
            cell: (run) =>
              run.suite ? (
                <a href={url('/suite', { suite: run.suite })}>{run.suite}</a>
              ) : (
                <span className="vw-muted">none: its test cases were not copied</span>
              ),
          }}
        />
      )}
    </>
  );
}
