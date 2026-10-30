/**
 * The frame of every page: the skip link, the header with the site's sections, the main column
 * with its notices, and a status line under it. Server-rendered; each page names its section,
 * so the active link is in the initial HTML.
 */
import logo from '@coralmind/brand/assets/logo/primary-color.svg';
import type { ReactNode } from 'react';
import type { Notice } from '../lib/notices';
import { KL_SITE, Notices } from './bits';

export type Section = 'intro' | 'suites' | 'runs' | null;

/** What the status line shows: when the results were read, and from where. */
export interface Stamp {
  builtAt: string;
  root: string;
  stale: boolean;
}

const NAV: { section: Exclude<Section, null>; label: string; href: string }[] = [
  { section: 'intro', label: 'Introduction', href: '/' },
  { section: 'suites', label: 'Suites', href: '/suites' },
  { section: 'runs', label: 'Runs', href: '/runs' },
];

const REPO_URL = 'https://github.com/coralmind/toolkit';

function GitHubIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
      <path
        fill="currentColor"
        d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"
      />
    </svg>
  );
}

/** The folder as its last two parts, so a screenshot shows no home folder. */
export function shortFolder(path: string): string {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length > 2 ? `…/${parts.slice(-2).join('/')}` : path;
}

export function SiteHeader({ section }: { section: Section }) {
  return (
    <header className="cm-site-header site-header vw-header">
      <div className="cm-site-header__inner">
        <a className="cm-site-header__brand site-header__brand" href="/">
          {/* Imports as a plain string URL in Next, not StaticImageData. */}
          <img src={logo} alt="CoralMind" width="150" height="36" />
          <span className="site-header__name">Evaluation Lab</span>
        </a>
        <nav className="cm-site-header__nav" aria-label="Main">
          {NAV.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className={item.section === section ? 'cm-is-active' : undefined}
              aria-current={item.section === section ? 'page' : undefined}
            >
              {item.label}
            </a>
          ))}
          <a href={KL_SITE} target="_blank" rel="noopener noreferrer" aria-label="Knowledge Library (opens in a new tab)">
            Knowledge Library ↗
          </a>
          <a
            href={REPO_URL}
            className="site-header__github"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="The CoralMind Toolkit on GitHub (opens in a new tab)"
          >
            <GitHubIcon />
          </a>
        </nav>
      </div>
    </header>
  );
}

export function Shell({
  section,
  stamp,
  notices = [],
  children,
}: {
  section: Section;
  stamp: Stamp | null;
  notices?: Notice[];
  children: ReactNode;
}) {
  return (
    <>
      <a className="vw-skip" href="#main">
        Skip to content
      </a>
      <SiteHeader section={section} />
      <main id="main" className="vw-main">
        <Notices items={notices} />
        {children}
      </main>
      {stamp ? <StatusLine stamp={stamp} /> : null}
    </>
  );
}

/** The note above the footer: when the results were read, and from where. */
function StatusLine({ stamp }: { stamp: Stamp }) {
  return (
    <div className="vw-statusbar">
      <p className="vw-status">
        <span>
          Results as of <time dateTime={stamp.builtAt}>{`${stamp.builtAt.slice(0, 10)} ${stamp.builtAt.slice(11, 19)} UTC`}</time>
          {stamp.stale ? ' · update failed' : ''}
        </span>
        <span className="vw-status__folder" title={stamp.root}>
          {shortFolder(stamp.root)}
        </span>
      </p>
    </div>
  );
}

/** The note under a results page's title. */
export function ContextLine({ kind }: { kind: 'suite' | 'run' | 'case' }) {
  const lead = {
    suite: 'Results from simulated conversations.',
    run: 'A run of simulated conversations.',
    case: 'A transcript of a simulated conversation.',
  }[kind];
  return (
    <p className="vw-context">
      {lead} Judgments compare against Knowledge Library definitions and do not establish that harm occurred.{' '}
      <a href="/">How to read this →</a>
    </p>
  );
}
