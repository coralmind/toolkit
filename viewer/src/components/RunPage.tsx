/** Run mode: one run, a row per transcript, problems first. */
import { caseLink } from '../lib/case-view';
import {
  BUCKET_LABELS,
  OUTCOME_LABELS,
  PROBLEM_ORDER,
  conditionLabel,
  failureLabel,
  fidelityReasonText,
  levelLabel,
  preemptedLabel,
  recorded,
  words,
} from '../lib/labels';
import { type RunView, runLink, shownRows } from '../lib/run-view';
import { suiteFileLink } from '../lib/suite-file-view';
import type { LibraryData, RunRow } from '../lib/types';
import { RolePills } from './Lists';
import { ContextLine } from './Shell';
import {
  Chip,
  DefinitionList,
  DirectionKeyItems,
  Ident,
  Key,
  KeyItem,
  PageHead,
  Section,
  TurnStrip,
} from './bits';

interface Props {
  view: RunView;
  library: LibraryData | null;
}

export function RunPage({ view, library }: Props) {
  const run = view.run;
  const counts = new Map<string, number>();
  for (const t of run.transcripts) counts.set(t.bucket, (counts.get(t.bucket) ?? 0) + 1);
  const fmt = (n: number) => n.toLocaleString('en-US');
  const tokens = run.judge_tokens;
  const judged = run.passes.length > 0;
  return (
    <>
      <PageHead eyebrow="Run" title={run.name} context={<ContextLine kind="run" />}>
        <DefinitionList
          items={[
            [
              'Suite',
              run.suite ? (
                <a href={suiteFileLink(run.suite)}>{run.suite}</a>
              ) : (
                <span className="vw-muted">none: the run&apos;s test cases were not copied into it</span>
              ),
            ],
            ['Manifest', run.manifest ? <Ident>{run.manifest}</Ident> : ''],
            ['Run id', run.run_id ? <Ident>{run.run_id}</Ident> : ''],
            ['Level', levelLabel(run.level)],
            ['Fact mode', recorded(run.fact_mode)],
          ]}
        />
        <RunSteps view={view} />
        <div className="cm-defgrid cm-defgrid--3 vw-health">
          <div>
            <p className="cm-defgrid__kicker">Duration and tokens</p>
            {judged ? (
              <p className="cm-defgrid__body">
                Judge time {fmt(Math.round(run.judge_elapsed_ms / 1000))} s<br />
                Judge tokens {fmt(tokens[0] + tokens[1])} (prompt {fmt(tokens[0])}, completion{' '}
                {fmt(tokens[1])})
              </p>
            ) : (
              <p className="cm-defgrid__body vw-muted">Not judged yet.</p>
            )}
          </div>
          <div>
            <p className="cm-defgrid__kicker">Call failures and retries</p>
            {judged ? (
              <p className="cm-defgrid__body">
                {run.call_failures} failed judge calls
                <br />
                {run.retries} retries
                <br />
                {run.judge_problems.length} unreadable judge lines
              </p>
            ) : (
              <p className="cm-defgrid__body vw-muted">Not judged yet.</p>
            )}
          </div>
          <div>
            <p className="cm-defgrid__kicker">Transcripts by status</p>
            <p className="cm-defgrid__body">{run.transcripts.length} transcripts</p>
            <p className="vw-filterchips">
              {PROBLEM_ORDER.filter((b) => counts.get(b)).map((b) => (
                <a
                  key={b}
                  className={`vw-filterchip b-${b}${view.bucket === b ? ' is-on' : ''}`}
                  href={runLink(view, { bucket: view.bucket === b ? null : b })}
                  aria-current={view.bucket === b ? 'true' : undefined}
                >
                  {BUCKET_LABELS[b]} {counts.get(b)}
                </a>
              ))}
            </p>
          </div>
        </div>
      </PageHead>
      <Transcripts view={view} library={library} />
    </>
  );
}

/** The run's steps in order: the simulation, its fidelity check, and each judge pass. */
function RunSteps({ view }: { view: RunView }) {
  const run = view.run;
  const total = run.transcripts.length;
  const failed = run.transcripts.filter((t) => t.bucket === 'sim_failure').length;
  const checked = Object.entries(run.fidelity_counts).filter(([status]) => status !== 'unchecked');
  return (
    <div className="cm-defgrid cm-defgrid--3 vw-runsteps">
      <div>
        <p className="cm-defgrid__kicker">Simulation</p>
        <RolePills suts={run.suts} simulators={run.person_model ? [run.person_model] : []} />
        <p className="cm-defgrid__body">
          {total} transcripts{failed ? `, ${failed} failed` : ''}
          {run.max_turns ? <> · up to {run.max_turns} turns</> : null}
        </p>
      </div>
      <div>
        <p className="cm-defgrid__kicker">Fidelity check</p>
        {run.checker ? (
          <>
            <RolePills checkers={[run.checker]} />
            <p className="cm-defgrid__body">
              {checked.length ? checked.map(([status, n]) => `${n} ${words(status)}`).join(', ') : 'no verdicts'}
              {run.checker_prompt ? <> · prompt {run.checker_prompt}</> : null}
            </p>
          </>
        ) : (
          <p className="cm-defgrid__body vw-muted">not run</p>
        )}
      </div>
      <div>
        <p className="cm-defgrid__kicker">{run.passes.length > 1 ? 'Judge passes' : 'Judge pass'}</p>
        {run.passes.length ? (
          run.passes.map((pass) => (
            <div key={pass.id} className="vw-runstep">
              <RolePills judges={[pass.judge_model]} />
              <p className="cm-defgrid__body">
                {pass.transcripts} transcripts · prompt {pass.prompt_version || pass.prompt_hash.slice(0, 8)} ·{' '}
                {pass.latest.slice(0, 10)}
              </p>
            </div>
          ))
        ) : (
          <p className="cm-defgrid__body vw-muted">not judged yet</p>
        )}
      </div>
    </div>
  );
}

function Pager({ view }: { view: RunView }) {
  if (view.pages === 1) return null;
  const first = (view.page - 1) * view.pageSize + 1;
  const last = first + shownRows(view).length - 1;
  return (
    <nav className="vw-sorts vw-pager" aria-label="Pages">
      {view.page > 1 ? (
        <a href={runLink(view, { page: String(view.page - 1) })}>← Previous {view.pageSize}</a>
      ) : null}
      <span className="vw-meta">
        rows {first}–{last} of {view.rows.length}
      </span>
      {view.page < view.pages ? <a href={runLink(view, { page: String(view.page + 1) })}>Next {view.pageSize} →</a> : null}
    </nav>
  );
}

function Transcripts({ view, library }: { view: RunView; library: LibraryData | null }) {
  if (!view.run.transcripts.length) {
    return (
      <Section id="transcripts" title="Transcripts">
        <p className="vw-empty">This run has no transcripts yet.</p>
      </Section>
    );
  }
  const cameFrom = runLink(view);
  return (
    <Section id="transcripts" title="Transcripts">
      <Pager view={view} />
      <Key>
        <KeyItem swatch="k-route">a step of the target pathway</KeyItem>
        <KeyItem swatch="k-onset">where an occurrence that fired begins</KeyItem>
        <KeyItem swatch="k-s1">the judge saw a behavior</KeyItem>
        <DirectionKeyItems />
      </Key>
      <div className="vw-matrix__frame">
        <table className="vw-table vw-runrows">
          <thead>
            <tr>
              <th scope="col">Test case</th>
              <th scope="col">Transcript status</th>
              <th scope="col">Strategies followed</th>
              <th scope="col">Target route</th>
              <th scope="col">Other routes that fired</th>
              <th scope="col">Turns</th>
            </tr>
          </thead>
          <tbody>
            {shownRows(view).map((t) => (
              <RunRowView key={`${t.batch}/${t.file}:${t.line}`} view={view} t={t} cameFrom={cameFrom} library={library} />
            ))}
          </tbody>
        </table>
      </div>
      <Pager view={view} />
    </Section>
  );
}

function RunRowView({
  view,
  t,
  cameFrom,
  library,
}: {
  view: RunView;
  t: RunRow;
  cameFrom: string;
  library: LibraryData | null;
}) {
  if (t.bucket === 'unreadable') {
    return (
      <tr className="is-problem">
        <th scope="row">
          <Ident>{t.file}</Ident> line {t.line}
        </th>
        <td>
          <span className="vw-bucket b-unreadable">unreadable</span> <span className="vw-meta">{t.error}</span>
        </td>
        <td colSpan={4} />
      </tr>
    );
  }
  let reasons = t.fidelity_reasons
    .map(([c, d]) => fidelityReasonText(c, d, library?.names[d]?.did))
    .join('; ');
  if (t.sim_failure) reasons = failureLabel(t.sim_failure);
  if (t.bucket === 'not_tested' && t.preempted_by) reasons = preemptedLabel(t.preempted_by);
  if (t.bucket === 'judged' && (t.level === 1 || t.level === 2)) reasons = 'fidelity n/a (L1/L2)';
  const followed = t.adherence.filter((a) => a === 'followed').length;
  const applicable = t.adherence.filter((a) => a !== 'trigger_absent').length;
  const href = caseLink(view.run.name, t.id, cameFrom);
  const problem = t.bucket !== 'judged' && t.bucket !== 'not_tested';
  return (
    <tr className={problem ? 'is-problem' : undefined}>
      <th scope="row">
        <a href={href}>{t.display_name || <Ident>{t.test_case_id}</Ident>}</a>
        {t.display_name ? (
          <span className="vw-rowmeta">
            <Ident>{t.test_case_id}</Ident>
          </span>
        ) : null}
        <span className="vw-meta">{t.sut}</span>
      </th>
      <td>
        <span className={`vw-bucket b-${t.bucket}`}>{BUCKET_LABELS[t.bucket]}</span>
        {reasons ? (
          <>
            {' '}
            <span className="vw-meta">{reasons}</span>
          </>
        ) : null}
      </td>
      <td>{t.adherence.length ? `${followed} of ${applicable}` : '—'}</td>
      <td>
        {t.target ? (
          <>
            <Chip id={t.target.route} library={library} />{' '}
            <span className={`vw-outcome o-${t.target.outcome}`}>{OUTCOME_LABELS[t.target.outcome]}</span>
            {t.target.support !== null ? (
              <>
                {' '}
                <span className="vw-meta">
                  {words(t.target.support)}, {conditionLabel(t.target.gate_status)}
                </span>
              </>
            ) : null}
          </>
        ) : t.judged ? (
          '—'
        ) : (
          <span className="vw-muted">not judged yet</span>
        )}
      </td>
      <td>
        {t.others.map((r, i) => (
          <span key={r.route}>
            {i ? ' ' : ''}
            <Chip id={r.route} library={library} did={r.did} />
          </span>
        ))}
      </td>
      <td className="vw-runrows__strip">
        <TurnStrip
          turns={t.strip.exchanges}
          marks={{
            routeTurns: t.strip.route_turns,
            onset: t.strip.onset,
            stage1Turns: t.strip.stage1_turns,
            fidelityTurns: t.strip.fidelity_turns,
          }}
          exitReason={t.strip.exit_reason}
          size="short"
          href={href}
        />
      </td>
    </tr>
  );
}
