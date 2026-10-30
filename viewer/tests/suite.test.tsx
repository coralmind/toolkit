/** The Suites and Runs lists, and a Suite page, ported from tests/viewer/test_pages.py and test_routes.py. */
import { describe, expect, it } from 'vitest';
import RunsRoute from '../src/app/runs/page';
import SuiteRoute from '../src/app/suite/page';
import SuitesRoute from '../src/app/suites/page';
import { page, use } from './helpers';

describe('the lists', () => {
  it('lists suites, and runs outside a suite on the Runs page', async () => {
    use('lists');
    const suites = await page(SuitesRoute);
    expect(suites).toContain('/suite?suite=suite"');
    expect(suites).toContain('href="/runs">1 run is not in a suite');
    expect(suites).toContain('class="vw-table"');
    expect(suites).not.toContain('cm-card');
    const runs = await page(RunsRoute);
    expect(runs).toContain('none: its test cases were not copied');
    expect(runs).toContain('/run?name=run-b');
  });

  it('says how to make results when the folder is empty', async () => {
    use('empty');
    const out = await page(SuitesRoute);
    expect(out).toContain('No results yet in');
    expect(out).toContain('pixi run sim');
  });
});

describe('a suite', () => {
  it('shows its test cases and no tabs', async () => {
    const meta = use('judged-one');
    const out = await page(SuiteRoute, { suite: meta.names[0]! });
    expect(out).toContain('aimhef-0000000a');
    expect(out).not.toContain('vw-pagetabs');
  });
});
