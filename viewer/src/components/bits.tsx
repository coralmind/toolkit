/**
 * Small parts shared by the pages: terms with definitions, entry chips, bars, the turn strip,
 * citations and definition lists. Every link works without JavaScript; `public/viewer.js`
 * adds quote highlighting, pane focus and the rail's tabs.
 */
import { type CSSProperties, Fragment, type ReactNode } from 'react';
import { BUCKETS, BUCKET_LABELS, DIRECTION_MEANINGS, TERMS, endedLabel, endedNote } from '../lib/labels';
import { type Notice, ordered } from '../lib/notices';
import type { LibraryData } from '../lib/types';

/** The Knowledge Library site that entry chips link to. */
export const KL_SITE = 'https://map.coralmind.org';

export function klUrl(entryId: string): string | null {
  return /^kl-\d{6}$/.test(entryId) ? `${KL_SITE}/entry/${entryId}` : null;
}

export function VisuallyHidden({ children }: { children: ReactNode }) {
  return <span className="vw-visually-hidden">{children}</span>;
}

/** A label with its definition on hover and on keyboard focus. */
export function Term({ label, definition }: { label: string; definition?: string }) {
  const text = definition ?? TERMS[label] ?? '';
  if (!text) return <>{label}</>;
  return (
    <span className="vw-term" tabIndex={0} data-def={text}>
      {label}
      <VisuallyHidden>: {text}</VisuallyHidden>
    </span>
  );
}

/** A pathway part's role, with the Knowledge Library's definition on hover and focus. */
export function RoleTerm({ role, library }: { role: string; library: LibraryData | null }) {
  const shown = role.replaceAll('_', ' ');
  const definition = library?.roles[role];
  return <span className="vw-role">{definition ? <Term label={shown} definition={definition} /> : shown}</span>;
}

export function Ident({ children }: { children: ReactNode }) {
  return <code className="vw-id">{children}</code>;
}

function kindFrom(displayId: string): string {
  const prefix = displayId.split('-', 1)[0]!.toLowerCase();
  return ({ pathway: 'pathway', aibehavior: 'ai-behavior', userbehavior: 'user-behavior' } as Record<string, string>)[prefix] ?? 'entry';
}

/**
 * An entry chip in its Knowledge Library type color. Links to the entry on the Knowledge
 * Library site in a new tab, unless `link` is false because the chip sits inside a link.
 */
export function Chip({
  id,
  library,
  did = '',
  label = '',
  link = true,
}: {
  id: string;
  library: LibraryData | null;
  did?: string;
  label?: string;
  link?: boolean;
}) {
  const name = library?.names[id];
  const shown = name?.did || did || id;
  const title = name?.label || label;
  const kind = name?.type.replaceAll('_', '-') || kindFrom(shown);
  const href = link ? klUrl(id) : null;
  const direction = kind === 'pathway' ? (library?.routes[id]?.direction ?? '') : '';
  const className = `vw-chip vw-chip--${kind}${direction ? ` vw-chip--${direction}` : ''}`;
  if (href) {
    return (
      <a className={className} title={title} href={href} target="_blank" rel="noopener noreferrer">
        <DirectionIcon direction={direction} />
        {shown}
        <VisuallyHidden> (opens in a new tab)</VisuallyHidden>
      </a>
    );
  }
  return (
    <span className={className} title={title}>
      <DirectionIcon direction={direction} />
      {shown}
    </span>
  );
}

/** A chip followed by the entry's label. */
export function Named({
  id,
  library,
  did = '',
  label = '',
}: {
  id: string;
  library: LibraryData | null;
  did?: string;
  label?: string;
}) {
  const text = library?.names[id]?.label || label;
  return (
    <>
      <Chip id={id} library={library} did={did} label={label} /> {text}
    </>
  );
}

/** One thin stacked bar across the outcome buckets, with its counts in words. */
export function BucketBar({ counts, total }: { counts: Map<string, number>; total: number }) {
  if (!total) return <span className="vw-bucketbar vw-bucketbar--empty" aria-hidden="true" />;
  const present = BUCKETS.filter((b) => (counts.get(b) ?? 0) > 0);
  const words = present.map((b) => `${BUCKET_LABELS[b]} ${counts.get(b)}`).join(', ');
  return (
    <span className="vw-bucketbar" role="img" aria-label={words}>
      {present.map((b) => (
        <span key={b} className={`b-${b}`} style={{ width: `${(((counts.get(b) ?? 0) * 100) / total).toFixed(3)}%` }} />
      ))}
    </span>
  );
}

export function DefinitionList({ items }: { items: [ReactNode, ReactNode][] }) {
  const shown = items.filter(([, value]) => value !== null && value !== undefined && value !== '' && value !== false);
  if (!shown.length) return null;
  return (
    <dl className="vw-dl">
      {shown.map(([label, value], i) => (
        <Fragment key={i}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

export function Notices({ items }: { items: Notice[] }) {
  if (!items.length) return null;
  return (
    <div className="vw-notices" role="status">
      {ordered(items).map((n, i) => (
        <p key={i} className={`vw-notice vw-notice--${n.kind}`}>
          {n.text}
        </p>
      ))}
    </div>
  );
}

export function Section({ title, id, children }: { title: ReactNode; id?: string; children: ReactNode }) {
  return (
    <section className="vw-section" id={id}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function Crumbs({ items, right }: { items: [string, string | null][]; right?: ReactNode }) {
  return (
    <nav className="vw-crumbs" aria-label="Breadcrumb">
      <ol>
        {items.map(([label, href], i) => (
          <li key={i}>{href ? <a href={href}>{label}</a> : label}</li>
        ))}
      </ol>
      {right}
    </nav>
  );
}

/** The kind of page, its title, the summary list and the settings under it. */
export function PageHead({
  eyebrow,
  title,
  context,
  children,
}: {
  eyebrow: string;
  title: string;
  context?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="vw-head">
      <p className="vw-eyebrow">{eyebrow}</p>
      <h1>{title}</h1>
      {context}
      {children}
    </header>
  );
}

// ---------------------------------------------------------------------------
// Turn strip
// ---------------------------------------------------------------------------

/**
 * A key: what the marks, colors or pills in the thing directly below it mean. Every legend on
 * the site uses it, so a reader learns the one look.
 */
export function Key({ children }: { children: ReactNode }) {
  return (
    <div className="vw-key" role="note" aria-label="Key">
      <span className="vw-key__label">Key</span>
      {children}
    </div>
  );
}

const DIRECTION_ICONS: Record<string, ReactNode> = {
  harmful: (
    <>
      <path d="M8 1.6 15 14H1Z" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M8 6v4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <circle cx="8" cy="12" r=".95" fill="currentColor" />
    </>
  ),
  protective: (
    <>
      <path
        d="M8 1.5 13.5 3.6v4C13.5 11 11.2 13.4 8 14.6 4.8 13.4 2.5 11 2.5 7.6v-4Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <path
        d="m5.6 8 1.7 1.7L10.6 6.3"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </>
  ),
  mixed: (
    <>
      <circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <path d="M8 2a6 6 0 0 1 0 12Z" fill="currentColor" />
    </>
  ),
};

/** A pathway's direction as an icon: a warning sign, a shield, or a half-filled circle for mixed. */
export function DirectionIcon({ direction }: { direction: string }) {
  const icon = DIRECTION_ICONS[direction];
  if (!icon) return null;
  return (
    <svg className="vw-chip__dir" viewBox="0 0 16 16" role="img" aria-label={`${direction} pathway`}>
      {icon}
    </svg>
  );
}

/** Key items saying what a firing means for each kind of pathway. */
export function DirectionKeyItems() {
  return (
    <>
      {Object.entries(DIRECTION_MEANINGS).map(([direction, meaning]) => (
        <KeyItem key={direction}>
          <span className={`vw-chip vw-chip--pathway vw-chip--${direction} vw-chip--sample`}>
            <DirectionIcon direction={direction} />
          </span>{' '}
          {direction}: {meaning}
        </KeyItem>
      ))}
    </>
  );
}

export function KeyItem({ swatch, children }: { swatch?: string; children: ReactNode }) {
  return (
    <span className="vw-key__item">
      {swatch ? <span className={`vw-key__swatch ${swatch}`} aria-hidden="true" /> : null}
      {children}
    </span>
  );
}

export interface StripMarks {
  routeTurns: number[];
  onset: number | null;
  stage1Turns: number[];
  fidelityTurns: number[];
}

/**
 * One cell per exchange: a route bar, a judge tick and a fidelity mark, bottom up.
 *
 * Cells share the width down to 12px, then the strip scrolls in its own frame. On load
 * `viewer.js` scrolls it to `data-start` (the onset turn, or turn 1).
 */
export function TurnStrip({
  turns,
  marks,
  exitReason,
  size = 'tall',
  href = '',
}: {
  turns: number;
  marks: StripMarks;
  exitReason: string | null;
  size?: 'tall' | 'short';
  href?: string;
}) {
  if (turns <= 0) return null;
  const cells = [];
  for (let number = 1; number <= turns; number += 1) {
    const classes = ['vw-strip__cell'];
    const said: string[] = [];
    if (marks.routeTurns.includes(number)) {
      classes.push('has-route');
      said.push('route step');
    }
    if (number === marks.onset) {
      classes.push('is-onset');
      said.push('route begins');
    }
    if (marks.stage1Turns.includes(number)) {
      classes.push('has-s1');
      said.push('judge observation');
    }
    if (marks.fidelityTurns.includes(number)) {
      classes.push('has-fid');
      said.push('fidelity citation');
    }
    let label = `Turn ${number}${said.length ? `: ${said.join(', ')}` : ''}`;
    if (number === turns && exitReason) label += `. Ended: ${endedLabel(exitReason)}`;
    cells.push(
      <li key={number} className={classes.join(' ')} data-turn={number}>
        <a href={`${href}#turn-${number}`} aria-label={label} title={label}>
          <span className="m-fid" />
          <span className="m-s1" />
          <span className="m-route" />
          <span className="n">{number}</span>
        </a>
      </li>,
    );
  }
  return (
    <div
      className={`vw-strip vw-strip--${size}${turns > 30 ? ' is-narrow' : ''}`}
      data-start={marks.onset || 1}
      style={{ '--turns': turns } as CSSProperties}
    >
      <div className="vw-strip__frame">
        <ol className="vw-strip__cells">{cells}</ol>
      </div>
      {exitReason ? <span className="vw-strip__end">How it ended: {endedLabel(exitReason)}</span> : null}
    </div>
  );
}

/**
 * The transcript page's timeline: a column per turn and a labeled row for each kind of mark,
 * so marks never overlap. Uses the strip's classes, so `viewer.js` scrolls it to the onset.
 */
export function TurnTimeline({
  turns,
  marks,
  exitReason,
}: {
  turns: number;
  marks: StripMarks;
  exitReason: string | null;
}) {
  if (turns <= 0) return null;
  const cells = [];
  for (let number = 1; number <= turns; number += 1) {
    const classes = ['vw-strip__cell'];
    const said: string[] = [];
    if (marks.routeTurns.includes(number)) {
      classes.push('has-route');
      said.push('a step of the target pathway');
    }
    if (number === marks.onset) {
      classes.push('is-onset');
      said.push('an occurrence that fired begins');
    }
    if (marks.stage1Turns.includes(number)) {
      classes.push('has-s1');
      said.push('the judge saw a behavior');
    }
    if (marks.fidelityTurns.includes(number)) {
      classes.push('has-fid');
      said.push('the fidelity check quoted it');
    }
    const label = `Turn ${number}${said.length ? `: ${said.join(', ')}` : ''}`;
    cells.push(
      <li key={number} className={classes.join(' ')} data-turn={number}>
        <a href={`#turn-${number}`} aria-label={label} title={label}>
          <span className="m-route" />
          <span className="m-s1" />
          <span className="m-fid" />
          <span className="n">{number}</span>
        </a>
      </li>,
    );
  }
  return (
    <div
      className={`vw-strip vw-strip--tall vw-timeline${turns > 30 ? ' is-narrow' : ''}`}
      data-start={marks.onset || 1}
      style={{ '--turns': turns } as CSSProperties}
    >
      <Key>
        <KeyItem swatch="k-route">a step of the target pathway</KeyItem>
        <KeyItem swatch="k-onset">where an occurrence that fired begins</KeyItem>
        <KeyItem swatch="k-s1">the judge saw a behavior</KeyItem>
        <KeyItem swatch="k-fid">
          the fidelity check quoted the turn as evidence for an expected behavior, a strategy or a fact
        </KeyItem>
      </Key>
      <div className="vw-timeline__body">
        <ul className="vw-timeline__rows" aria-hidden="true">
          <li>Target pathway</li>
          <li>Judge</li>
          <li>Fidelity check</li>
        </ul>
        <div className="vw-strip__frame">
          <ol className="vw-strip__cells">{cells}</ol>
        </div>
      </div>
      {exitReason ? (
        <p className="vw-strip__end">
          How the conversation ended: <strong>{endedLabel(exitReason)}</strong>
          {endedNote(exitReason) ? ` (${endedNote(exitReason)})` : ''}
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Citations
// ---------------------------------------------------------------------------

/**
 * A link to `#turn-N` that lights its quote. Says so when the quote is not verbatim.
 *
 * `text` links words already on the page (an explanation that quotes the span) in place of
 * repeating the quote; `showTurn` drops the turn label where the context gives it.
 */
export function Citation({
  turn,
  quote,
  role = '',
  found = true,
  text,
  showTurn = true,
}: {
  turn: number;
  quote: string;
  role?: string;
  found?: boolean;
  text?: string | null;
  showTurn?: boolean;
}) {
  const speaker = ({ user: 'person', assistant: 'AI' } as Record<string, string>)[role] ?? '';
  return (
    <>
      <a
        className={`vw-cite${found ? '' : ' is-unmatched'}`}
        href={`#turn-${turn}`}
        data-turn={turn}
        data-role={role}
        data-quote={quote}
        data-found={found ? '1' : '0'}
      >
        {showTurn ? (
          <>
            <span className="vw-cite__turn">
              T{turn}
              {speaker ? ` ${speaker}` : ''}
            </span>{' '}
          </>
        ) : null}
        {text !== undefined && text !== null ? text : <q>{quote}</q>}
      </a>
      {found ? null : (
        <>
          {' '}
          <span className="vw-miss">not found verbatim</span>
        </>
      )}
    </>
  );
}
