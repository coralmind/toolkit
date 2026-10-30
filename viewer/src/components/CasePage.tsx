/**
 * The transcript view: the case on the left, the turns in the middle, and the judge,
 * the fidelity check and the raw records in the rail.
 */
import { Fragment, type ReactNode } from 'react';
import { type ListContext, caseLink } from '../lib/case-view';
import {
  BASIS,
  BUCKET_LABELS,
  CUE_READING,
  OUTCOME_LABELS,
  SENT_DEFINITION,
  SENT_TERM,
  failureLabel,
  fidelityReasonText,
  gateWord,
  levelLabel,
  partsNeededLabel,
  preemptedLabel,
  recorded,
  targetPartTag,
  targetSkipText,
  times,
  words,
} from '../lib/labels';
import type {
  CaseData,
  CaseRecordData,
  Cite,
  ConditionData,
  InstanceData,
  JudgeView,
  LibraryData,
  ObservationData,
  RailRoute,
  StepData,
} from '../lib/types';
import { url } from '../lib/url';
import { ContextLine } from './Shell';
import {
  Chip,
  Citation,
  Crumbs,
  DefinitionList,
  DirectionKeyItems,
  Ident,
  Key,
  KeyItem,
  Named,
  RoleTerm,
  Term,
  TurnTimeline,
  VisuallyHidden,
} from './bits';
import { ROLE_LABELS, type Role, RolePill } from './Pipeline';

interface Props {
  data: CaseData;
  judge: JudgeView | null;
  context: ListContext | null;
  library: LibraryData | null;
}

export function CasePage({ data, judge, context, library }: Props) {
  const bucket = judge ? judge.bucket : data.unjudged_bucket;
  const route = bucket === 'judged' ? (judge?.target_route ?? null) : null;
  const key: [string, string] = [data.run, data.transcript_id];
  const crumbs: [string, string | null][] = context
    ? [...context.crumbs]
    : [
        ['Runs', '/runs'],
        [data.run, url('/run', { name: data.run })],
      ];
  crumbs.push([data.test_case_id, null]);
  let nav: ReactNode = null;
  if (context) {
    const at = context.keys.findIndex(([r, t]) => r === key[0] && t === key[1]);
    if (at >= 0) {
      const prev = context.keys[at - 1];
      const next = context.keys[at + 1];
      nav = (
        <span className="vw-prevnext">
          {prev ? (
            <a className="vw-step" rel="prev" href={caseLink(prev[0], prev[1], context.href)}>
              ← Previous
            </a>
          ) : null}
          <span className="vw-meta">
            {at + 1} of {context.keys.length}
          </span>
          {next ? (
            <a className="vw-step" rel="next" href={caseLink(next[0], next[1], context.href)}>
              Next →
            </a>
          ) : null}
        </span>
      );
    }
  }
  const simulated = data.level !== 1 && data.level !== 2;
  let center: ReactNode;
  if (data.error !== null) {
    center = <p className="vw-notice vw-notice--info">Unreadable transcript line: {data.error}</p>;
  } else if (data.sim_failure !== null) {
    center = (
      <div className="vw-failure">
        <h2>Simulation failed</h2>
        <p>
          <strong>{failureLabel(data.sim_failure)}</strong> {data.failure_detail}
        </p>
      </div>
    );
  } else if (!data.turns || !data.turns.length) {
    center = <p className="vw-empty">This record holds no turns.</p>;
  } else {
    center = (
      <>
        <TurnTimeline
          turns={data.exchanges}
          marks={{
            routeTurns: route?.step_turns ?? [],
            onset: route?.onset ?? null,
            stage1Turns: judge?.stage1_turns ?? [],
            fidelityTurns: data.fidelity_turns,
          }}
          exitReason={data.exit_reason}
        />
        <Turns data={data} />
      </>
    );
  }
  return (
    <>
      <Crumbs items={crumbs} right={nav} />
      <ContextLine kind="case" />
      <div className="vw-case">
        <details className="vw-case__left" open>
          <summary>Case details</summary>
          <CaseColumn data={data} judge={judge} library={library} simulated={simulated} />
        </details>
        <div className="vw-case__center">
          <Verdict data={data} bucket={bucket} judge={judge} library={library} />
          <SystemSwitch data={data} context={context} />
          {center}
        </div>
        <aside className="vw-case__rail" aria-label="Judge, fidelity and raw records">
          <Rail data={data} judge={judge} bucket={bucket} library={library} />
        </aside>
      </div>
    </>
  );
}

function Verdict({
  data,
  bucket,
  judge,
  library,
}: {
  data: CaseData;
  bucket: string;
  judge: JudgeView | null;
  library: LibraryData | null;
}) {
  const parts: ReactNode[] = [];
  const target = data.case?.pathway_id ?? '';
  const route = bucket === 'judged' ? judge?.target_route : null;
  if (target) {
    const did = library?.routes[target]?.did ?? target;
    if (route) {
      parts.push(
        <>
          <Chip id={target} library={library} did={did} />{' '}
          <strong className={`vw-outcome o-${route.outcome}`}>{OUTCOME_LABELS[route.outcome]}</strong>
        </>,
      );
      if (route.support !== null) {
        parts.push(`support: ${words(route.support)}`);
        parts.push(
          <>
            <Term label="condition" />: {words(route.gate_status)}
          </>,
        );
        if (route.onset) parts.push(`begins at turn ${route.onset}`);
      } else {
        parts.push(route.assessment);
      }
    } else {
      parts.push(
        <>
          <Chip id={target} library={library} did={did} /> <span className="vw-muted">{BUCKET_LABELS[bucket]}</span>
        </>,
      );
    }
  }
  if (data.level === 1 || data.level === 2) parts.push('fidelity n/a (L1/L2)');
  else if (data.copied_turn !== null) parts.push(copiedText(data.copied_turn));
  else if (data.fidelity) parts.push(`fidelity: ${words(data.fidelity)}`);
  else if (data.sim_failure === null && data.error === null) parts.push('fidelity: not checked');
  parts.push(data.sut);
  return (
    <p className="vw-verdict">
      {parts.map((part, i) => (
        <span key={i}>
          {i ? ' · ' : ''}
          {part}
        </span>
      ))}
    </p>
  );
}

/** A part's result: present, ruled out by a turn that breaks it, or failed when nothing shows it. */
function partWord(part: { present: boolean; ruled_out: Cite | null }): string {
  if (part.present) return 'present';
  return part.ruled_out ? 'ruled out' : 'failed';
}

function copiedText(turn: number): string {
  return `the simulated person copied the AI's message at turn ${turn}`;
}

/**
 * The systems tested on this test case in the suite, this one marked, the others links to their
 * transcripts; and the counterfactual counterpart, when there is one.
 */
function SystemSwitch({ data, context }: { data: CaseData; context: ListContext | null }) {
  const others = data.compare.filter((link) => link.kind === 'other_sut');
  const pairs = data.compare.filter((link) => link.kind === 'counterpart');
  if (!others.length && !pairs.length) return null;
  const href = (link: (typeof data.compare)[number]) => caseLink(link.run, link.transcript, context?.href ?? null);
  return (
    <div className="vw-switch">
      {others.length ? (
        <p className="vw-switch__row">
          <span className="vw-switch__label">System tested</span>
          <span className="vw-switch__current" aria-current="page">
            <RolePill role="sut">{data.sut}</RolePill>
          </span>
          {others.map((link) => (
            <a key={`${link.run}\0${link.transcript}`} className="vw-switch__other" href={href(link)}>
              <RolePill role="sut">{link.sut}</RolePill>
            </a>
          ))}
        </p>
      ) : null}
      {pairs.map((link) => (
        <p key={`${link.run}\0${link.transcript}`} className="vw-switch__row">
          <span className="vw-switch__label">Paired test case</span>
          <a href={href(link)}>
            <Ident>{link.test_case_id}</Ident>
          </a>
        </p>
      ))}
    </div>
  );
}

function Turns({ data }: { data: CaseData }) {
  return (
    <div className="vw-turns">
      {data.turns!.map((exchange) => (
        <section
          key={exchange.number}
          className="vw-turn"
          id={`turn-${exchange.number}`}
          aria-label={`Turn ${exchange.number}`}
        >
          <p className="vw-turn__num">Turn {exchange.number}</p>
          {exchange.messages.map((message, i) => (
            <div key={i} className={`vw-msg vw-msg--${message.role}`} id={`turn-${exchange.number}-${message.role}`}>
              <p className="vw-msg__who">{message.role === 'user' ? 'Person' : 'AI'}</p>
              {message.copied_from !== null ? (
                <p className="vw-msg__flag">
                  This repeats the AI&apos;s message from turn {message.copied_from}. The simulated person did not write
                  it, so the transcript is not a valid test from here on.
                </p>
              ) : null}
              <div className="content-text">{message.content ?? ''}</div>
            </div>
          ))}
        </section>
      ))}
    </div>
  );
}

/** Marks text the simulated person receives. Unmarked at scripted levels. */
function Sent({ simulated, children }: { simulated: boolean; children: ReactNode }) {
  if (!simulated) return <>{children}</>;
  return (
    <div className="vw-sent">
      {children}
      <VisuallyHidden> ({SENT_TERM})</VisuallyHidden>
    </div>
  );
}

function PersonBlock({ c, library, simulated }: { c: CaseRecordData; library: LibraryData | null; simulated: boolean }) {
  if (!c.persona) return c.persona_error ? <p className="vw-muted">{c.persona_error}</p> : null;
  const p = c.persona;
  const prose: [string, string][] = [
    ['', p.description],
    ['Clinical context', p.clinical_context],
    ['Behavioral style', p.behavioral_description],
  ];
  const facts: [string, string][] = [
    ['Age', p.age],
    ['Gender', p.gender],
    ['Presentation', p.presentation],
    ['Treatment', p.treatment],
    ['Insight', p.insight],
    ['Engagement', p.engagement],
  ];
  const hasFacts = facts.some(([, v]) => v) || p.elevated.length > 0;
  return (
    <>
      {prose
        .filter(([, text]) => text)
        .map(([label, text]) => (
          <Sent key={label || 'description'} simulated={simulated}>
            {label ? <p className="vw-meta">{label}</p> : null}
            <p>{text}</p>
          </Sent>
        ))}
      {hasFacts && simulated ? (
        <p className="vw-meta">The fields below shaped the description above and are not sent.</p>
      ) : null}
      <DefinitionList items={facts} />
      {p.elevated.length ? (
        <>
          <p className="vw-meta">Facts about the person:</p>
          <ul className="vw-list">
            {p.elevated.map((v) => (
              <li key={v}>
                <Named id={v} library={library} />
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

/** The model in each role that made and read this transcript. */
function RunModels({ data, judge }: { data: CaseData; judge: JudgeView | null }) {
  const models: [Role, string][] = [
    ['sut', data.sut],
    ['simulator', data.simulator],
    ['checker', data.fidelity_view?.checker_model ?? ''],
    ['judge', judge?.judge_model ?? ''],
  ];
  return (
    <dl className="vw-runmodels">
      {models
        .filter(([, model]) => model)
        .map(([role, model]) => (
          <Fragment key={role}>
            <dt>{ROLE_LABELS[role]}</dt>
            <dd>
              <RolePill role={role}>{model}</RolePill>
            </dd>
          </Fragment>
        ))}
    </dl>
  );
}

function CaseColumn({
  data,
  judge,
  library,
  simulated,
}: {
  data: CaseData;
  judge: JudgeView | null;
  library: LibraryData | null;
  simulated: boolean;
}) {
  if (!data.case) return <p className="vw-muted">The test case is not in this run&apos;s inputs.</p>;
  return (
    <>
      <h2>
        Case <Ident>{data.case.test_case_id}</Ident>
      </h2>
      <RunModels data={data} judge={judge} />
      <CaseRecord
        c={data.case}
        library={library}
        simulated={simulated}
        factMode={data.fact_mode}
      />
    </>
  );
}

/**
 * A test case record: its target, person, scenario, opening, expected behaviors and strategies.
 * `factMode` comes from a transcript, and is left out without one.
 */
export function CaseRecord({
  c,
  library,
  simulated,
  factMode,
}: {
  c: CaseRecordData;
  library: LibraryData | null;
  simulated: boolean;
  factMode?: string;
}) {
  const conditions = library?.routes[c.pathway_id]?.parts.filter((p) => p.role === 'condition') ?? [];
  return (
    <>
      {simulated ? (
        <Key>
          <KeyItem swatch="k-sent">
            <Term label={SENT_TERM} definition={SENT_DEFINITION} />
          </KeyItem>
        </Key>
      ) : null}
      {c.display_name ? <p className="vw-meta">{c.display_name}</p> : null}
      {c.description ? <p className="vw-meta">{c.description}</p> : null}
      {c.assertions.length ? (
        <>
          <h3>Expected behaviors</h3>
          <ul className="vw-list">
            {c.assertions.map((a, i) => (
              <li key={i}>
                <Named id={a.behavior} library={library} />
                <Sent simulated={simulated}>
                  <span className="vw-meta">{a.assertion}</span>
                </Sent>
              </li>
            ))}
          </ul>
          {simulated ? <p className="vw-meta">The behavior ids are not sent; only each assertion is.</p> : null}
        </>
      ) : null}
      <DefinitionList
        items={[
          ['Target', <Named key="t" id={c.pathway_id} library={library} />],
          ['Person', c.persona_ref ? <Ident>{c.persona_ref}</Ident> : ''],
          [
            <Term key="c" label="condition" />,
            conditions.length
              ? conditions.map((p, i) => (
                  <span key={p.id}>
                    {i ? ', ' : ''}
                    <Named id={p.id} library={library} />
                  </span>
                ))
              : '',
          ],
          ['Fact mode', factMode === undefined ? '' : recorded(factMode)],
          ['Level', levelLabel(c.level)],
          ['Counterfactual of', c.counterpart_of ? <Ident>{c.counterpart_of}</Ident> : ''],
          [
            'Situation',
            c.situation.length
              ? c.situation.map(([k, v]) => (
                  <span key={k}>
                    {words(k)}: {v}
                    <br />
                  </span>
                ))
              : '',
          ],
          ['Design', c.design ? `${c.design[0]} row ${c.design[1]}` : ''],
        ]}
      />
      {c.persona || c.persona_error ? (
        <>
          <h3>Person</h3>
          <PersonBlock c={c} library={library} simulated={simulated} />
        </>
      ) : null}
      <h3>Scenario</h3>
      <Sent simulated={simulated}>
        <p>{c.scenario}</p>
      </Sent>
      {simulated && c.turns.length ? (
        <>
          <h3>Opening message</h3>
          <Sent simulated>
            <p>{c.turns[0]}</p>
          </Sent>
          <p className="vw-meta">Sent as the person&apos;s first message, word for word.</p>
        </>
      ) : null}
      {!simulated && c.turns.length ? (
        <>
          <h3>Scripted turns</h3>
          <ol className="vw-list">
            {c.turns.map((t, i) => (
              <li key={i}>{t}</li>
            ))}
          </ol>
        </>
      ) : null}
      {c.strategies.length ? (
        <>
          <h3>Strategies</h3>
          <ul className="vw-list">
            {c.strategies.map((s, i) => (
              <li key={i}>
                <Sent simulated={simulated}>
                  <p>
                    <span className="vw-cite__turn">{i + 1}</span> {s.strategy}
                  </p>
                  <p className="vw-role">
                    from turn {s.available_from_turn} · trigger: {s.trigger}
                  </p>
                </Sent>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

// ---------------------------------------------------------------------------
// Rail
// ---------------------------------------------------------------------------

function Rail({
  data,
  judge,
  bucket,
  library,
}: {
  data: CaseData;
  judge: JudgeView | null;
  bucket: string;
  library: LibraryData | null;
}) {
  const sections: [string, string, ReactNode][] = [
    ['fidelity', 'Fidelity', <FidelityTab key="f" data={data} library={library} />],
    ['pathways', 'Pathways', <PathwaysTab key="p" data={data} judge={judge} bucket={bucket} library={library} />],
    ['behaviors', 'Behaviors', <BehaviorsTab key="b" data={data} judge={judge} bucket={bucket} library={library} />],
    ['records', 'Records', <RecordsTab key="r" data={data} judge={judge} />],
  ];
  return (
    <>
      <nav className="vw-railtabs" aria-label="Rail">
        {sections.map(([key, label]) => (
          <a key={key} href={`#rail-${key}`} data-tab={key} id={`railtab-${key}`}>
            {label}
          </a>
        ))}
      </nav>
      {sections.map(([key, label, body]) => (
        <section key={key} className="vw-rail__panel" id={`rail-${key}`} aria-label={label}>
          <h2>{label}</h2>
          {body}
        </section>
      ))}
    </>
  );
}

function CiteView({ cite, showTurn = true }: { cite: Cite; showTurn?: boolean }) {
  return <Citation turn={cite.turn} quote={cite.quote} role={cite.role} found={cite.found} text={cite.text} showTurn={showTurn} />;
}

function Step({ step, library }: { step: StepData; library: LibraryData | null }) {
  return (
    <li>
      <p className="vw-step__head">
        <span className="vw-cite__turn">T{step.turn}</span>
        <Chip id={step.behavior_id} library={library} did={step.behavior_did} />
        <RoleTerm role={step.role} library={library} />
      </p>
      {step.why.length ? (
        <p className="vw-step__why">
          {step.why.map((segment, i) =>
            segment.cite ? <CiteView key={i} cite={segment.cite} showTurn={false} /> : <span key={i}>{segment.text}</span>,
          )}
        </p>
      ) : null}
      {step.others.map((cite, i) => (
        <p key={i} className="vw-step__quote">
          <CiteView cite={cite} showTurn={false} />
        </p>
      ))}
    </li>
  );
}

function Occurrence({ inst, library }: { inst: InstanceData; library: LibraryData | null }) {
  const window = inst.window;
  const turns = !window.length ? '' : window[0] === window.at(-1) ? `turn ${window[0]}` : `turns ${window[0]}–${window.at(-1)}`;
  const meta = [turns, words(inst.support), words(inst.structure)].filter(Boolean).join(' · ');
  return (
    <div className="vw-instance">
      <p className="vw-instance__head">
        <strong>Occurrence {inst.number}</strong> <span className="vw-role">{meta}</span>
      </p>
      <ul className="vw-steps">
        {inst.steps.map((step, i) => (
          <Step key={i} step={step} library={library} />
        ))}
      </ul>
      {inst.connection_evidence ? <p className="vw-meta">{inst.connection_evidence}</p> : null}
      {inst.rationale ? (
        <p>
          <strong>The judge&apos;s reasoning:</strong> {inst.rationale}
        </p>
      ) : null}
      {inst.evidence_gaps.length ? (
        <>
          <p className="vw-meta">What the judge found missing:</p>
          <ul className="vw-list">
            {inst.evidence_gaps.map((gap, i) => (
              <li key={i}>{gap}</li>
            ))}
          </ul>
        </>
      ) : null}
      {inst.structure_note || inst.uncertainty !== null || inst.adjustments.length ? (
        <details>
          <summary>More from the judge on this occurrence</summary>
          {inst.structure_note ? <p>How the parts fit: {inst.structure_note}</p> : null}
          {inst.uncertainty !== null ? (
            <p>The judge&apos;s uncertainty: {inst.uncertainty} (0 is certain, 1 is a guess).</p>
          ) : null}
          {inst.adjustments.length ? (
            <>
              <p className="vw-meta">Changed by code after the judge answered:</p>
              <ul className="vw-list">
                {inst.adjustments.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </>
          ) : null}
        </details>
      ) : null}
    </div>
  );
}

/** One line on why the route did or did not fire, from its occurrences and condition. */
export function routeReason(r: RailRoute): string {
  if (r.outcome === 'kl_mismatch') return 'Judged with a different Knowledge Library than the one loaded.';
  if (r.supported) {
    const occurred = `Occurred ${times(r.supported)}`;
    if (r.outcome === 'fires') return occurred + (r.has_condition ? ' and the condition held.' : '; it has no condition.');
    if (r.outcome === 'gate_unknown') return `${occurred}, but whether the condition held is unknown.`;
    return `${occurred}, but the condition did not hold.`;
  }
  if (r.partial) return `Partly supported ${times(r.partial)}; no occurrence was fully supported.`;
  if (r.instances.length) {
    const found = r.instances.length === 1 ? 'one possible occurrence' : `${r.instances.length} possible occurrences`;
    return `The judge considered ${found} and found none supported.`;
  }
  return 'No occurrence was found.';
}

function Condition({ gate, library }: { gate: ConditionData; library: LibraryData | null }) {
  const verdict = gateWord(gate.verdict);
  const basis = BASIS[gate.basis] ?? '';
  const record =
    gate.recorded === 'unknown' ? (
      'Test case record: says nothing on this.'
    ) : (
      <>
        Test case record: {gateWord(gate.recorded)}.
        {gate.fact ? (
          <>
            {' '}
            The person is recorded as <Named id={gate.fact} library={library} />.
          </>
        ) : null}
      </>
    );
  const lead = gate.conflict
    ? 'The judge read the transcript the other way, citing:'
    : gate.cues.length
      ? "The judge's cues from the transcript:"
      : '';
  return (
    <div className="vw-condition">
      <p>
        <Named id={gate.part_id} library={library} did={gate.part_did} label={gate.label} />
      </p>
      <p>
        <strong>{verdict.charAt(0).toUpperCase() + verdict.slice(1)}</strong>
        {basis ? `, ${basis}` : ''}.
      </p>
      <p className="vw-role">{record}</p>
      <p className="vw-role">Judge&apos;s reading of the transcript: {gateWord(gate.inferred)}.</p>
      {gate.cues.length ? (
        <>
          <p className="vw-meta">{lead}</p>
          <ul className="vw-steps">
            {gate.cues.map((cue, i) => (
              <li key={i}>
                <p className="vw-step__head">
                  <span className="vw-cite__turn">T{cue.cite.turn}</span>
                  <span>{({ user: 'person', assistant: 'AI' } as Record<string, string>)[cue.cite.role] ?? cue.cite.role}</span>
                  <span className="vw-role">{CUE_READING[cue.reading] ?? cue.reading}</span>
                </p>
                <p className="vw-step__quote">
                  <CiteView cite={cue.cite} showTurn={false} />
                </p>
              </li>
            ))}
          </ul>
        </>
      ) : null}
      {gate.rationale ? (
        <details>
          <summary>The judge&apos;s reasoning on the condition</summary>
          <p>{gate.rationale}</p>
        </details>
      ) : null}
    </div>
  );
}

/** A pathway's result as a summary line, with its occurrences and conditions folded under it. */
function RouteSection({ r, library }: { r: RailRoute; library: LibraryData | null }) {
  const direction = library?.routes[r.pathway_id]?.direction ?? '';
  return (
    <details
      className={`vw-railroute dir-${direction || 'unknown'}${r.is_target ? ' is-target' : ''}`}
      open={r.is_target}
    >
      <summary>
        <span className="vw-railroute__head">
          <Chip id={r.pathway_id} library={library} did={r.did} label={r.label} />{' '}
          <span className={`vw-outcome o-${r.outcome}`}>{OUTCOME_LABELS[r.outcome]}</span>
          {r.is_target ? <span className="vw-tag">target</span> : null}
        </span>
        <span className="vw-meta">{r.label}</span>
        <span className="vw-route__why">{routeReason(r)}</span>
      </summary>
      <div className="vw-route__body">
        {r.instances.length ? (
          <>
            <h4>
              {r.supported || r.partial ? 'Occurred' : 'Considered'} · {times(r.instances.length)}
            </h4>
            {r.instances.map((inst) => (
              <Occurrence key={inst.number} inst={inst} library={library} />
            ))}
          </>
        ) : (
          <>
            <h4>Occurred</h4>
            <p className="vw-muted">No occurrence found.</p>
          </>
        )}
        {r.missing.length ? (
          <p className="vw-meta">
            Never observed:{' '}
            {r.missing.map((m, i) => (
              <span key={m}>
                {i ? ', ' : ''}
                <Chip id={m} library={library} />
              </span>
            ))}
          </p>
        ) : null}
        {r.gates.length ? (
          <>
            <h4>{r.gates.length > 1 ? 'Conditions' : 'Condition'}</h4>
            {r.gates.map((g, i) => (
              <Condition key={i} gate={g} library={library} />
            ))}
          </>
        ) : null}
        {r.rationale ? (
          <details>
            <summary>The judge&apos;s reasoning on the route</summary>
            <p>{r.rationale}</p>
          </details>
        ) : null}
      </div>
    </details>
  );
}

/** Why a tab has nothing to show for this transcript, or null when the judge read it. */
function notJudged(data: CaseData, judge: JudgeView | null, bucket: string): ReactNode {
  if (data.sim_failure !== null) return <p className="vw-muted">Not judged: simulation failed.</p>;
  if (judge === null) return <p className="vw-muted">Not judged yet ({BUCKET_LABELS[bucket]}).</p>;
  if (!judge.ran) return <p className="vw-muted">The judge did not run: status {judge.status}.</p>;
  return null;
}

function PathwaysTab({
  data,
  judge,
  bucket,
  library,
}: {
  data: CaseData;
  judge: JudgeView | null;
  bucket: string;
  library: LibraryData | null;
}) {
  const none = notJudged(data, judge, bucket);
  if (none || judge === null) return none;
  // Harmful pathways first, then mixed, then protective, so each kind sits together.
  const order: Record<string, number> = { harmful: 0, mixed: 1, protective: 2 };
  const rank = (r: RailRoute) => order[library?.routes[r.pathway_id]?.direction ?? ''] ?? 3;
  const target = judge.routes.filter((r) => r.is_target);
  const others = judge.routes.filter((r) => !r.is_target).sort((a, b) => rank(a) - rank(b));
  const fired = others.filter((r) => r.outcome === 'fires');
  const rest = others.filter((r) => r.outcome !== 'fires');
  return (
    <>
      {judge.run_status !== 'success' || judge.run_error_detail ? (
        <p className="vw-notice vw-notice--info">
          Judge run {judge.run_status}
          {judge.run_error_detail ? `: ${judge.run_error_detail}` : ''}
        </p>
      ) : null}
      <p className="vw-meta">
        The judge assesses every pathway that has at least one behavior it saw in the transcript, so pathways other than
        the target appear here too. Open a pathway to see its occurrences and conditions.
      </p>
      <Key>
        <DirectionKeyItems />
      </Key>
      <h3>Target pathway</h3>
      {judge.target_missing !== null && data.case ? (
        <section className="vw-railroute is-target">
          <p className="vw-railroute__head">
            <Chip id={data.case.pathway_id} library={library} /> <span className="vw-outcome o-not">not assessed</span>
          </p>
          <p className="vw-meta">{targetSkipText(judge.target_missing)}</p>
          {judge.target_parts.length ? (
            <ul className="vw-list">
              {judge.target_parts.map((part) => (
                <li key={part.id}>
                  <Named id={part.id} library={library} />{' '}
                  <span className={`vw-tag ${part.seen ? 'vw-tag--yes' : 'vw-tag--no'}`}>{targetPartTag(part)}</span>
                  {part.reason ? <p className="vw-parts__ruling">Judge: {part.reason}</p> : null}
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}
      {target.map((r) => (
        <RouteSection key={r.pathway_id} r={r} library={library} />
      ))}
      {fired.length ? (
        <>
          <h3>Other pathways that fired · {fired.length}</h3>
          {fired.map((r) => (
            <RouteSection key={r.pathway_id} r={r} library={library} />
          ))}
        </>
      ) : null}
      {rest.length ? (
        <>
          <h3>Other pathways assessed that did not fire · {rest.length}</h3>
          {rest.map((r) => (
            <RouteSection key={r.pathway_id} r={r} library={library} />
          ))}
        </>
      ) : null}
    </>
  );
}

/** The first turn a behavior was seen in, for ordering; turns read as `T3` or `3`. */
function firstTurn(o: ObservationData): number {
  const numbers = o.turns.map((t) => Number.parseInt(t.replace(/\D+/g, ''), 10)).filter((n) => !Number.isNaN(n));
  return numbers.length ? Math.min(...numbers) : Number.POSITIVE_INFINITY;
}

function BehaviorsTab({
  data,
  judge,
  bucket,
  library,
}: {
  data: CaseData;
  judge: JudgeView | null;
  bucket: string;
  library: LibraryData | null;
}) {
  const none = notJudged(data, judge, bucket);
  if (none || judge === null) return none;
  if (!judge.observations.length)
    return <p className="vw-muted">The judge saw no behavior from the Knowledge Library.</p>;
  const groups: [string, ObservationData[]][] = [
    ['The person', judge.observations.filter((o) => o.speaker === 'user')],
    ['The AI', judge.observations.filter((o) => o.speaker !== 'user')],
  ];
  return (
    <>
      <p className="vw-meta">
        Behaviors from the Knowledge Library that the judge saw in the transcript, in the order they first appear.
      </p>
      {groups
        .filter(([, items]) => items.length)
        .map(([who, items]) => (
          <section key={who}>
            <h3>
              {who} · {items.length}
            </h3>
            <ul className="vw-list vw-observations">
              {[...items]
                .sort((a, b) => firstTurn(a) - firstTurn(b))
                .map((o, i) => (
                  <li key={i}>
                    <Named id={o.behavior_id} library={library} did={o.behavior_did} label={o.label} />{' '}
                    <span className="vw-meta">turns {o.turns.join(', ')}</span>
                    {o.evidence.length ? (
                      <details className="vw-quotes">
                        <summary>
                          {o.evidence.length} {o.evidence.length === 1 ? 'quote' : 'quotes'}
                        </summary>
                        {o.evidence.map((cite, j) => (
                          <p key={j}>
                            <CiteView cite={cite} />
                          </p>
                        ))}
                      </details>
                    ) : null}
                  </li>
                ))}
            </ul>
          </section>
        ))}
    </>
  );
}

/** What each adherence verdict says, and what its quotes are. */
const ADHERENCE_NOTES: Record<string, string> = {
  followed: 'The person acted on it, in the turn quoted.',
  not_followed: 'The trigger happened, in the turn quoted, and the person did not act on it.',
  trigger_absent: 'The trigger never happened, so the person had no cue to act on it.',
};

/** The tag style of a strategy's adherence: followed, not followed, or neither. */
export function adherenceTag(adherence: string): string {
  if (adherence === 'followed') return 'vw-tag--yes';
  if (adherence === 'not_followed') return 'vw-tag--no';
  return '';
}

function FidelityTab({ data, library }: { data: CaseData; library: LibraryData | null }) {
  if (data.level === 1 || data.level === 2) {
    return <p className="vw-muted">Not applicable: scripted levels have no fidelity check.</p>;
  }
  if (data.sim_failure !== null) return <p className="vw-muted">No fidelity check: simulation failed.</p>;
  const verdict = data.fidelity_view;
  if (!verdict) return <p className="vw-muted">Not checked: this transcript has no fidelity verdict.</p>;
  const missing = (behavior: string) =>
    verdict.behaviors
      .find((b) => b.behavior === behavior)
      ?.parts.filter((p) => !p.present && p.part !== 'all')
      .map((p) => `(${p.part}) ${partWord(p)}`) ?? [];
  const reasons = data.fidelity_reasons
    .map(([c, d]) => fidelityReasonText(c, d, library?.names[d]?.did, missing(d)))
    .join('; ');
  const cites = (items: Cite[]) =>
    items.map((cite, i) => (
      <span key={i}>
        <br />
        <CiteView cite={cite} />
      </span>
    ));
  return (
    <>
      {data.copied_turn !== null ? (
        <p className="vw-notice vw-notice--info">
          Counted as invalid: {copiedText(data.copied_turn)}. The fidelity check does not look for copied messages,
          so its verdict below does not count it.
        </p>
      ) : null}
      <p className="vw-lede">
        {data.copied_turn !== null ? "The check's verdict" : 'Status'}: <strong>{words(verdict.status)}</strong>
        {reasons ? ` — ${reasons}` : ''}
      </p>
      <p className="vw-meta">
        Checker {verdict.checker_model} · {verdict.checker_prompt_version}
      </p>
      {verdict.behaviors.length ? (
        <>
          <h3>Expected behaviors</h3>
          <ul className="vw-list">
            {verdict.behaviors.map((b, i) => (
              <li key={i}>
                <Named id={b.behavior} library={library} />{' '}
                <span className={`vw-tag ${b.catalog_occurred ? 'vw-tag--yes' : 'vw-tag--no'}`}>
                  {b.catalog_occurred ? 'occurred' : 'did not occur'}
                </span>{' '}
                <span className="vw-meta">as the test case states it: {b.assertion_occurred ? 'yes' : 'no'}</span>
                {b.miss === 'preempted' ? (
                  <p className="vw-meta">
                    Not tested: {preemptedLabel(b.preempted_by ?? '')}, so the person had no occasion for it.
                    {b.ai_citation ? (
                      <>
                        <br />
                        <CiteView cite={b.ai_citation} />
                      </>
                    ) : null}
                  </p>
                ) : null}
                {b.parts.length > 1 ? (
                  <>
                    {partsNeededLabel(b.parts_needed, b.parts.filter((p) => p.part !== 'all').length) ? (
                      <p className="vw-meta">
                        {partsNeededLabel(b.parts_needed, b.parts.filter((p) => p.part !== 'all').length)}
                      </p>
                    ) : null}
                    <ul className="vw-parts">
                      {b.parts.map((part) => (
                        <li key={part.part}>
                          <span className={`vw-tag ${part.present ? 'vw-tag--yes' : 'vw-tag--no'}`}>
                            ({part.part}) {partWord(part)}
                          </span>
                          {part.text ? <span className="vw-parts__text"> {part.text}</span> : null}
                          {part.ruled_out ? (
                            <p className="vw-parts__ruling">
                              Ruled out at T{part.ruled_out.turn}: {part.reason}
                              <br />
                              {cites([part.ruled_out])}
                            </p>
                          ) : part.reason ? (
                            <p className="vw-parts__ruling">Checker: {part.reason}</p>
                          ) : null}
                          {cites(part.citations)}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : (
                  cites(b.citations)
                )}
              </li>
            ))}
          </ul>
        </>
      ) : null}
      {verdict.facts.length ? (
        <>
          <h3>Facts</h3>
          <ul className="vw-list">
            {verdict.facts.map((f, i) => (
              <li key={i}>
                <Named id={f.value} library={library} />{' '}
                <span className={`vw-tag ${f.presented ? 'vw-tag--yes' : 'vw-tag--no'}`}>
                  {f.presented ? 'presented' : 'not presented'}
                </span>{' '}
                <span className="vw-meta">{f.mode}</span>
                {cites(f.citations)}
              </li>
            ))}
          </ul>
        </>
      ) : null}
      {verdict.strategies.length ? (
        <>
          <h3>Strategies</h3>
          <ul className="vw-list">
            {verdict.strategies.map((s) => (
              <li key={s.number}>
                <p>
                  <span className="vw-cite__turn">{s.number}</span> {s.text}
                </p>
                <p className="vw-role">trigger: {s.trigger || 'not recorded'}</p>
                <p>
                  <span className={`vw-tag ${adherenceTag(s.adherence)}`}>{words(s.adherence)}</span>{' '}
                  <span className="vw-meta">{ADHERENCE_NOTES[s.adherence] ?? ''}</span>
                </p>
                {s.reason ? <p className="vw-parts__ruling">Checker: {s.reason}</p> : null}
                <p>{cites(s.citations)}</p>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </>
  );
}

/** A record read on demand by `/record`; the link works without JavaScript. */
function OnDemand({ title, href }: { title: string; href: string }) {
  return (
    <details className="vw-record" data-src={href}>
      <summary>{title}</summary>
      <p>
        <a href={href} target="_blank" rel="noopener noreferrer">
          Open as text in a new tab
        </a>
      </p>
      <pre className="vw-json vw-record__text" hidden />
    </details>
  );
}

function RecordsTab({ data, judge }: { data: CaseData; judge: JudgeView | null }) {
  const base = { run: data.run, transcript: data.transcript_id };
  const fmt = (n: number) => n.toLocaleString('en-US');
  return (
    <>
      <DefinitionList
        items={[
          [
            'Transcript',
            <>
              <Ident>{data.file}</Ident> line {data.line}
            </>,
          ],
          [
            'Judge',
            judge ? (
              <>
                <Ident>{judge.file}</Ident> line {judge.line}
              </>
            ) : (
              ''
            ),
          ],
        ]}
      />
      {data.driver ? (
        <>
          <details className="vw-record">
            <summary>{data.driver.label}</summary>
            <pre className="vw-json">{data.driver.prompt}</pre>
          </details>
          {data.driver.system ? (
            <details className="vw-record">
              <summary>System prompt of the system tested</summary>
              <pre className="vw-json">{data.driver.system}</pre>
            </details>
          ) : null}
        </>
      ) : null}
      {data.error === null ? <OnDemand title="Transcript record" href={url('/record', { ...base, kind: 'transcript' })} /> : null}
      {data.case ? (
        <details className="vw-record">
          <summary>Test case record</summary>
          <pre className="vw-json">{JSON.stringify(data.case.record, null, 2)}</pre>
        </details>
      ) : null}
      {data.check_call ? (
        <>
          <h3>Fidelity check prompt</h3>
          <p className="vw-meta">
            {data.check_call.model} · {fmt(data.check_call.prompt_tokens)} + {fmt(data.check_call.completion_tokens)}{' '}
            tokens · {(data.check_call.latency_ms / 1000).toFixed(1)} s
          </p>
          <details className="vw-record">
            <summary>Prompt</summary>
            <pre className="vw-json">{data.check_call.prompt}</pre>
          </details>
          <details className="vw-record">
            <summary>Response</summary>
            <pre className="vw-json">{data.check_call.response}</pre>
          </details>
        </>
      ) : null}
      {judge ? (
        <>
          <OnDemand title="Judge record" href={url('/record', { ...base, kind: 'judge', judge: judge.ref })} />
          <h3>Judge prompts</h3>
          {judge.calls.length ? (
            ([1, 2] as const).map((stage) => {
              const calls = judge.calls.filter((c) => c.stage === stage);
              if (!calls.length) return null;
              return (
                <div key={stage}>
                  <h4>
                    {stage === 1 ? 'Stage 1: which behaviors occurred' : 'Stage 2: which pathways they form'}{' '}
                    <span className="vw-meta">{calls.length} calls</span>
                  </h4>
                  {calls.map((call) => {
                    const params = { ...base, judge: judge.ref, seq: call.seq };
                    const prompt = url('/judge-call', { ...params, part: 'prompt' });
                    const response = url('/judge-call', { ...params, part: 'response' });
                    return (
                      <details key={call.seq} className="vw-call" data-prompt={prompt} data-response={response}>
                        <summary>
                          {call.label}{' '}
                          <span className="vw-meta">
                            {fmt(call.prompt_tokens)} + {fmt(call.completion_tokens)} tokens ·{' '}
                            {(call.latency_ms / 1000).toFixed(1)} s
                          </span>
                        </summary>
                        <p>
                          <a href={prompt} target="_blank" rel="noopener noreferrer">
                            Prompt
                          </a>{' '}
                          ·{' '}
                          <a href={response} target="_blank" rel="noopener noreferrer">
                            Response
                          </a>{' '}
                          <span className="vw-meta">(each opens in a new tab)</span>
                        </p>
                        <pre className="vw-json vw-call__prompt" hidden />
                        <pre className="vw-json vw-call__response" hidden />
                      </details>
                    );
                  })}
                </div>
              );
            })
          ) : (
            <p className="vw-muted">
              No judge calls for this transcript in <Ident>{judge.trace.split('/').at(-1)}</Ident>.
            </p>
          )}
        </>
      ) : null}
    </>
  );
}
