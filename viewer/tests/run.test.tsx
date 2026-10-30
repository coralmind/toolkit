/** Run mode, ported from tests/viewer/test_pages.py. */
import { describe, expect, it } from 'vitest';
import RunRoute from '../src/app/run/page';
import { RunPage } from '../src/components/RunPage';
import { runView } from '../src/lib/run-view';
import { html, page, thrown, url, use } from './helpers';

describe('run mode', () => {
  it('puts problems first and filters by bucket', async () => {
    use('run-mode');
    const out = await page(RunRoute, { name: 'run-a' });
    expect(out.indexOf('aimhef-0000000b')).toBeLessThan(out.indexOf('aimhef-0000000a'));
    const byCase = await page(RunRoute, { name: 'run-a', sort: 'case' });
    const rows = byCase.slice(byCase.indexOf('<tbody>'));
    expect(rows.indexOf('aimhef-0000000a')).toBeLessThan(rows.indexOf('aimhef-0000000b'));
    const invalid = await page(RunRoute, { name: 'run-a', bucket: 'invalid' });
    const body = invalid.slice(invalid.indexOf('<tbody>'));
    expect(body).toContain('aimhef-0000000b');
    expect(body).not.toContain('aimhef-0000000a');
    expect(out).toContain('vw-runsteps');
  });

  it('shows an unreadable line', async () => {
    use('unreadable-and-persona');
    expect(await page(RunRoute, { name: 'run-a' })).toContain('b-unreadable');
  });

  it('pages a long run', () => {
    use('long-run');
    const first = html(<RunPage view={runView({ name: 'run-a', sort: 'case' }, 2)} library={null} />);
    expect(first).toContain('rows 1–2 of 3');
    expect(first).toContain('page=2');
    const second = html(<RunPage view={runView({ name: 'run-a', sort: 'case', page: '2' }, 2)} library={null} />);
    expect(second).toContain('rows 3–3 of 3');
    expect(runView({ name: 'run-a', page: '9' }, 2).found.map((n) => n.text)).toContain(
      'Page 9 not found; showing page 1.',
    );
  });

  it('hides fact mode unless the run records one', async () => {
    use('fact-modes');
    expect(await page(RunRoute, { name: 'run-a' })).not.toContain('Fact mode');
    expect(await page(RunRoute, { name: 'run-b' })).toContain('Fact mode');
  });

  it('is not found for an unknown run', async () => {
    use('unjudged');
    expect(await thrown(() => page(RunRoute, { name: 'nope' }))).toContain('404');
  });

  it('builds case links that return to the run', async () => {
    use('unjudged');
    const out = await page(RunRoute, { name: 'run-a', sort: 'case' });
    expect(out).toContain(url('/transcript', { run: 'run-a', id: 'demo:azure-gpt-5-4-mini:aimhef-0000000a:1', from: '/run?name=run-a&sort=case' }).replaceAll('&', '&amp;'));
  });
});
