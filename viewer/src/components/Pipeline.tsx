/**
 * The evaluation pipeline on the Introduction: the four phases, the step each box runs, the
 * model role behind it, what each step saves and where it shows in the viewer. On a wide screen
 * the boxes sit on a five-column grid of 4px rows with drawn connectors; below 1180px they
 * stack in reading order.
 */
import type { ReactNode } from 'react';
import { Key } from './bits';

export type Role = 'sut' | 'simulator' | 'checker' | 'judge';

export const ROLE_LABELS: Record<Role, string> = {
  sut: 'System tested',
  simulator: 'Simulator',
  checker: 'Fidelity checker',
  judge: 'Judge',
};

function RoleIcon({ role }: { role: Role }) {
  const paths: Record<Role, ReactNode> = {
    sut: <path d="M4 5h16v11H10l-6 4z" />,
    simulator: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21c1.5-4.5 4.5-7 8-7s6.5 2.5 8 7" />
      </>
    ),
    checker: (
      <>
        <rect x="5" y="4" width="14" height="17" rx="2" />
        <path d="M9 4V3h6v1" />
        <path d="M9 13l2 2 4-4" />
      </>
    ),
    judge: (
      <>
        <path d="M12 4v16" />
        <path d="M8 20h8" />
        <path d="M5 7h14" />
        <path d="M5 7l-3 6a3 3 0 0 0 6 0z" />
        <path d="M19 7l-3 6a3 3 0 0 0 6 0z" />
      </>
    ),
  };
  return (
    <svg
      className="vw-rolepill__icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[role]}
    </svg>
  );
}

/**
 * A model named by its role: an icon and a color per role. With no children it names the role.
 * `reasoning` is the model's reasoning level ('off' for none), shown on a second line.
 */
export function RolePill({ role, reasoning, children }: { role: Role; reasoning?: string; children?: ReactNode }) {
  const name = children ?? ROLE_LABELS[role];
  return (
    <span className={`vw-rolepill vw-rolepill--${role}`}>
      <RoleIcon role={role} />
      {reasoning ? (
        <span className="vw-rolepill__text">
          {name}
          <span className="vw-rolepill__reasoning">{reasoning === 'off' ? 'no reasoning' : `${reasoning} reasoning`}</span>
        </span>
      ) : (
        name
      )}
      {children ? <span className="vw-visually-hidden"> ({ROLE_LABELS[role]})</span> : null}
    </span>
  );
}

/** A box's top and bottom in px from the top of the content grid, and its grid column. */
type Place = { top: number; bottom: number; col: number };

/** The content grid is 4px rows, 336px tall. */
const ROW = 4;
const area = ({ top, bottom, col }: Place) => `${top / ROW + 1} / ${col} / ${bottom / ROW + 1} / ${col + 1}`;

/** Where the "Saves to" boxes begin, in px from the top of the content grid. */
const SAVES_TOP = 401;

/** Box centers, in px from the top of the content grid, that the connectors meet. */
const MAIN = 124;
const MANIFEST = 260;

type Segment = { x1: number; x2: number; y: number; arrow?: boolean } | { x: number; y1: number; y2: number };

/** Lines drawn in one gap column: x in percent of the gap, y in px from the top of the grid. */
function Gap({ column, segments }: { column: number; segments: Segment[] }) {
  return (
    <div className="vw-pipe__gap" style={{ gridColumn: column }} aria-hidden="true">
      {segments.map((s, i) =>
        'y' in s ? (
          <span
            key={i}
            className={`vw-pipe__h${s.arrow ? ' has-arrow' : ''}`}
            style={{ left: `${s.x1}%`, width: `${s.x2 - s.x1}%`, top: s.y }}
          />
        ) : (
          <span key={i} className="vw-pipe__v" style={{ left: `${s.x}%`, top: s.y1, height: s.y2 - s.y1 }} />
        ),
      )}
    </div>
  );
}

/** A dashed line from a box's bottom down to its "Saves to" box. */
function Drop({ column, from }: { column: number; from: number }) {
  return (
    <div className="vw-pipe__gap" style={{ gridColumn: column }} aria-hidden="true">
      <span className="vw-pipe__drop" style={{ top: from, height: SAVES_TOP - from }} />
    </div>
  );
}

function Box({
  tag,
  title,
  note,
  at,
  kind,
  children,
}: {
  tag: string;
  title: string;
  note?: string;
  at: Place;
  kind?: 'path';
  children?: ReactNode;
}) {
  return (
    <div className={`vw-pipe__box${kind ? ` is-${kind}` : ''}`} style={{ gridArea: area(at) }}>
      <span className="vw-pipe__tag">{tag}</span>
      <strong className="vw-pipe__title">{title}</strong>
      {note ? <span className="vw-pipe__note">{note}</span> : null}
      {children}
    </div>
  );
}

function Phase({ name, line, column, color }: { name: string; line: string; column: string; color: string }) {
  return (
    <div className="vw-pipe__phase" style={{ gridColumn: column, borderTopColor: color }}>
      <h3>{name}</h3>
      <p>{line}</p>
    </div>
  );
}

function PhaseArrow({ column }: { column: number }) {
  return <div className="vw-pipe__phasearrow" style={{ gridColumn: column }} aria-hidden="true" />;
}

export function Pipeline() {
  return (
    <figure className="vw-pipe">
      <Key>
        <RolePill role="sut" />
        <RolePill role="simulator" />
        <RolePill role="checker" />
        <RolePill role="judge" />
      </Key>

      <div className="vw-pipe__phases">
        <Phase
          name="Define"
          column="1"
          color="var(--cm-foundation-blue)"
          line="The pathways to test and the simulated people, written as test cases"
        />
        <PhaseArrow column={2} />
        <Phase
          name="Simulate"
          column="3 / 6"
          color="var(--cm-insight-blue)"
          line="Run the conversations, then check that the intended behaviors happened"
        />
        <PhaseArrow column={6} />
        <Phase
          name="Judge"
          column="7"
          color="var(--cm-mindful-purple)"
          line="Read each valid transcript for behaviors, then pathways"
        />
      </div>

      <div className="vw-pipe__suite">
        <span className="vw-pipe__scope">Suite · one test case file and every run of it</span>
        <div className="vw-pipe__grid">
          <p className="vw-pipe__inputs" style={{ gridArea: area({ top: 36, bottom: 52, col: 1 }) }}>
            Inputs, used together
          </p>
          <Box
            tag="Test case file"
            title="Write the test cases"
            note="each names a persona and the pathway it tests"
            at={{ top: 64, bottom: 184, col: 1 }}
          />
          <Box
            tag="Manifest · one per run"
            title="Choose the models"
            note="one for each role, and how long a conversation can run"
            at={{ top: 208, bottom: 312, col: 1 }}
          />

          <Box tag="Simulation · one per run" title="Run the conversations" note="Transcripts" at={{ top: 16, bottom: 232, col: 3 }} kind="path">
            <RolePill role="sut" reasoning="medium">gpt-6-luna</RolePill>
            <RolePill role="sut" reasoning="medium">Claude Sonnet 4.6</RolePill>
            <RolePill role="simulator" reasoning="default">gpt-5.5</RolePill>
          </Box>

          <Box
            tag="Fidelity check"
            title="Check that the intended behaviors happened"
            note="Valid transcripts"
            at={{ top: 52, bottom: 196, col: 5 }}
            kind="path"
          >
            <RolePill role="checker" reasoning="off">Claude Opus 4.6</RolePill>
          </Box>

          <Box tag="Judge" title="Judge each transcript" note="Detections" at={{ top: 52, bottom: 196, col: 7 }} kind="path">
            <RolePill role="judge" reasoning="high">gpt-6-sol</RolePill>
          </Box>

          <Gap
            column={2}
            segments={[
              { x1: 0, x2: 100, y: MAIN, arrow: true },
              { x1: 0, x2: 50, y: MANIFEST },
              { x: 50, y1: MAIN, y2: MANIFEST },
            ]}
          />
          <Gap column={4} segments={[{ x1: 0, x2: 100, y: MAIN, arrow: true }]} />
          <Gap column={6} segments={[{ x1: 0, x2: 100, y: MAIN, arrow: true }]} />

          <Drop column={3} from={232} />
          <Drop column={5} from={196} />
          <Drop column={7} from={196} />

          <div className="vw-pipe__runline" aria-hidden="true">
            <span className="vw-pipe__rl is-top" />
            <span className="vw-pipe__rl is-right" />
            <span className="vw-pipe__rl is-bottom" />
            <span className="vw-pipe__rl is-left" />
            <span className="vw-pipe__rl is-foot" />
            <span className="vw-pipe__rl is-inner" />
            <span className="vw-pipe__rc is-tl" />
            <span className="vw-pipe__rc is-tr" />
            <span className="vw-pipe__rc is-br" />
            <span className="vw-pipe__rc is-bl" />
            <span className="vw-pipe__rc is-ftl" />
            <span className="vw-pipe__scope is-run">Run</span>
          </div>
        </div>
      </div>

      <div className="vw-pipe__lane">
        <p className="vw-pipe__lanelabel">Saves to</p>
        <div className="vw-pipe__row">
          <p className="vw-pipe__save is-source">
            <span>
              test_case_records/
              <br />
              run_manifests/
            </span>
          </p>
          <p className="vw-pipe__save">
            <span>
              <b>results/&lt;run&gt;/</b>
              <br />
              inputs/
              <br />
              batch-NNN/
              <br />
              &lt;run&gt;__&lt;system&gt;-sim.jsonl
            </span>
          </p>
          <p className="vw-pipe__save">
            <span>
              beside the transcripts
              <br />
              batch-NNN/
              <br />
              &lt;run&gt;__&lt;system&gt;-check.jsonl
            </span>
          </p>
          <p className="vw-pipe__save">
            <span>
              beside the transcripts
              <br />
              batch-NNN/
              <br />
              &lt;run&gt;__&lt;system&gt;-judge.jsonl
            </span>
          </p>
        </div>
      </div>

      <div className="vw-pipe__lane is-viewer">
        <p className="vw-pipe__lanelabel">In the viewer</p>
        <div className="vw-pipe__row">
          <a className="vw-pipe__dest" href="/suites">
            <strong>Suites, then a suite</strong> the test cases in its file
          </a>
          <a className="vw-pipe__dest" href="/runs">
            <strong>A run, then its transcripts</strong> one per test case and system tested
          </a>
          <p className="vw-pipe__dest">
            <strong>Fidelity verdicts</strong> valid or invalid, on each transcript
          </p>
          <p className="vw-pipe__dest">
            <strong>Judge results</strong> the behaviors and pathways found in each transcript
          </p>
        </div>
      </div>

      <figcaption className="vw-pipe__caption">
        The models shown are one example setup. Every system tested in a run gets the same simulated person, fidelity
        check and judge.
      </figcaption>
    </figure>
  );
}
