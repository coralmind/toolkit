/** The header, the Introduction, not-found pages, the footer and the lists. */
import { describe, expect, it } from 'vitest';
import IntroductionRoute from '../src/app/page';
import CaseRoute from '../src/app/transcript/page';
import NotFound from '../src/app/not-found';
import RunRoute from '../src/app/run/page';
import RunsRoute from '../src/app/runs/page';
import SuiteRoute from '../src/app/suite/page';
import SuitesRoute from '../src/app/suites/page';
import { SiteFooter } from '../src/components/SiteFooter';
import { conditionLabel, fidelityReasonText, partsNeededLabel } from '../src/lib/labels';
import { caseParams, html, page, url, use } from './helpers';

function current(out: string): string[] {
  const nav = /<nav class="cm-site-header__nav" aria-label="Main">(.*?)<\/nav>/s.exec(out)![1]!;
  return [...nav.matchAll(/<a [^>]*aria-current="page"[^>]*>([^<]+)</g)].map((m) => m[1]!);
}

describe('the header', () => {
  it('marks exactly one section per page', async () => {
    const meta = use('judged-one');
    const suite = meta.names[0]!;
    expect(current(await page(SuitesRoute))).toEqual(['Suites']);
    expect(current(await page(SuiteRoute, { suite }))).toEqual(['Suites']);
    expect(current(await page(RunsRoute))).toEqual(['Runs']);
    expect(current(await page(RunRoute, { name: 'run-a' }))).toEqual(['Runs']);
    const fromTestCase = url('/test-case', { suite, id: 'aimhef-0000000a' });
    expect(current(await page(CaseRoute, caseParams('aimhef-0000000a', 1, fromTestCase)))).toEqual(['Suites']);
    expect(current(await page(CaseRoute, caseParams('aimhef-0000000a', 1, '/run?name=run-a')))).toEqual(['Runs']);
    expect(current(await page(CaseRoute, caseParams('aimhef-0000000a')))).toEqual(['Runs']);
    expect(current(html(<NotFound />))).toEqual([]);
  });

  it('starts a case’s breadcrumb at the list it came from', async () => {
    const meta = use('judged-one');
    const fromTestCase = await page(
      CaseRoute,
      caseParams('aimhef-0000000a', 1, url('/test-case', { suite: meta.names[0]!, id: 'aimhef-0000000a' })),
    );
    expect(fromTestCase).toContain('<li><a href="/suites">Suites</a></li>');
    expect(fromTestCase).toContain('<a href="/suite?suite=suite">suite</a>');
    const fromRun = await page(CaseRoute, caseParams('aimhef-0000000a', 1, '/run?name=run-a'));
    expect(fromRun).toContain('<li><a href="/runs">Runs</a></li>');
  });

  it('opens the Knowledge Library in a new tab and says so', async () => {
    use('judged-one');
    const out = await page(SuitesRoute);
    expect(out).toContain(
      '<a href="https://map.coralmind.org" target="_blank" rel="noopener noreferrer" aria-label="Knowledge Library (opens in a new tab)">',
    );
    expect(out).toContain('Evaluation Lab');
  });

  it('shows when the results were read and their folder, relative to the repository', async () => {
    use('judged-one');
    const out = await page(SuitesRoute);
    expect(out).toContain('Results as of');
    expect(out).toContain('<span class="vw-status__folder" title="data/results">data/results</span>');
  });

  it('links the Introduction', async () => {
    use('judged-one');
    expect(await page(SuitesRoute)).toContain('>Introduction</a>');
  });
});

describe('the Introduction', () => {
  it('shows the pipeline and sends readers to the Suites list', async () => {
    use('judged-one');
    const out = html(await IntroductionRoute());
    expect(current(out)).toEqual(['Introduction']);
    expect(out).toContain('The evaluation pipeline');
    expect(out).toContain('How to read the results');
    expect(out).toContain('<a class="cm-btn" href="/suites">Explore suites</a>');
    expect(out).toContain('href="https://map.coralmind.org/guide/evaluation"');
    expect(out).toContain('It does not mean harm happened');
  });
});

describe('results pages', () => {
  it('put the context line under their title', async () => {
    const meta = use('judged-one');
    expect(await page(SuiteRoute, { suite: meta.names[0]! })).toContain('class="vw-context"');
    expect(await page(RunRoute, { name: 'run-a' })).toContain('class="vw-context"');
    expect(await page(CaseRoute, caseParams('aimhef-0000000a'))).toContain('How to read this →');
  });

  it('send a link that names nothing back to the lists', () => {
    use('judged-one');
    const out = html(<NotFound />);
    expect(out).toContain('Nothing with that id in these results.');
    expect(out).toContain('href="/suites"');
    expect(out).toContain('href="/runs"');
  });

  it('names the fix when the generated data is missing', async () => {
    use('judged-one');
    process.env.LAB_GENERATED_DIR = '/nonexistent/generated';
    const out = await page(SuitesRoute);
    expect(out).toContain('Could not read results');
    expect(out).toContain('pixi run -e viewer viewer-dev');
  });
});

describe('the footer', () => {
  it('carries the content license beside the copyright', () => {
    const out = html(<SiteFooter />);
    expect(out).toMatch(/CoralMind Institute · Content licensed <a href="https:\/\/creativecommons.org\/licenses\/by-nc\/4.0\/"[^>]*>CC BY-NC 4.0<\/a>/);
  });

  it('links entry chips to the Knowledge Library site', async () => {
    use('judged-one');
    expect(await page(CaseRoute, caseParams('aimhef-0000000a'))).toContain(
      'href="https://map.coralmind.org/entry/kl-900102" target="_blank"',
    );
  });
});

describe('the Suites and Runs lists', () => {
  it('lists only the test case files that runs used', async () => {
    use('lists');
    const out = await page(SuitesRoute);
    expect(out).toContain('>suite</a>');
    expect(out).not.toContain('>unused</a>');
    expect(out).not.toContain('Not run yet');
  });

  it('names the Runs table’s status column', async () => {
    use('judged-one');
    const table = /<table class="vw-table vw-runs">(.*?)<\/thead>/s.exec(await page(RunsRoute))![1]!;
    expect([...table.matchAll(/<th scope="col"[^>]*>([^<]+)</g)].map((m) => m[1]!)).toContain('Transcript status');
  });

  it('says whether each pathway is harmful and what firing means', async () => {
    const meta = use('judged-one');
    const run = await page(RunRoute, { name: 'run-a' });
    expect(run).toContain('vw-chip vw-chip--pathway vw-chip--harmful');
    expect(run).toContain('aria-label="harmful pathway"');
    expect(run).toContain('firing is the good result');
    const cases = await page(SuiteRoute, { suite: meta.names[0]! });
    expect(cases).toContain('vw-chip vw-chip--pathway vw-chip--harmful');
  });
});

describe('conditions', () => {
  it('words a route with no condition', () => {
    expect(conditionLabel('no_gates')).toBe('no condition');
    expect(conditionLabel('not_met')).toBe('condition did not hold');
  });
});

describe('parts needed', () => {
  it('says whether any one part counts or every part is needed', () => {
    expect(partsNeededLabel(1, 3)).toBe('Any one part counts.');
    expect(partsNeededLabel(3, 3)).toBe('All 3 parts are needed.');
    expect(partsNeededLabel(2, 4)).toBe('2 of 4 parts are needed.');
    expect(partsNeededLabel(0, 3)).toBe('');
  });
});

describe('fidelity reasons', () => {
  it('names a behavior that did not occur and the parts that failed or were ruled out', () => {
    expect(fidelityReasonText('simulator', 'kl-000320', 'Symptom-1')).toBe('Symptom-1 did not occur');
    expect(fidelityReasonText('simulator', 'kl-000320', 'Symptom-1', ['(b) ruled out'])).toBe(
      'Symptom-1 did not occur: part (b) ruled out',
    );
    expect(fidelityReasonText('simulator', 'kl-000320', 'Symptom-1', ['(a) failed', '(c) ruled out'])).toBe(
      'Symptom-1 did not occur: parts (a) failed, (c) ruled out',
    );
    expect(fidelityReasonText('simulator', 'the simulated person copied the AI’s message at turn 3')).toBe(
      'the simulated person copied the AI’s message at turn 3',
    );
  });
});
