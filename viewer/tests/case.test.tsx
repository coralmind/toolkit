/** The transcript view, ported from tests/viewer/test_pages.py. */
import { describe, expect, it } from 'vitest';
import CaseRoute from '../src/app/transcript/page';
import RunRoute from '../src/app/run/page';
import { caseParams, page, thrown, transcriptId, unescape, url, use } from './helpers';

function caseHtml(caseId = 'aimhef-0000000a', sample = 1, from = ''): Promise<string> {
  return page(CaseRoute, caseParams(caseId, sample, from));
}

describe('content and failures', () => {
  it('escapes transcript content', async () => {
    use('escaped');
    const out = await caseHtml();
    expect(out).toContain('I feel &lt;terrible&gt; today');
    expect(out).not.toContain('<terrible>');
    expect(out).not.toContain('</script><script>alert(1)');
  });

  it('escapes a turn number from the record', async () => {
    use('escaped');
    expect(await caseHtml('aimhef-0000000a', 2)).not.toContain('<script>alert(1)</script>');
  });

  it('shows a failed simulation', async () => {
    use('sim-failure');
    const out = await caseHtml();
    expect(out).toContain('Simulation failed');
    expect(out).toContain('timed out');
    expect(out).toContain('took too long');
    expect(out).toContain('Not judged: simulation failed.');
    expect(out).not.toContain('class="vw-strip');
  });

  it('says in plain words that the person copied the AI', async () => {
    use('person-echo-exit');
    expect(await caseHtml()).toContain('How the conversation ended: <strong>the simulated person copied the AI</strong>');
    expect(await page(RunRoute, { name: 'run-a' })).toContain('How it ended: the simulated person copied the AI');
  });

  it('flags a person turn that copies an earlier AI turn', async () => {
    use('copied-turn');
    const out = await caseHtml();
    expect(out.split('class="vw-msg__flag"').length - 1).toBe(1);
    const flagged = out.indexOf('id="turn-3-user"');
    expect(out.indexOf('class="vw-msg__flag"', flagged)).toBeLessThan(out.indexOf('id="turn-3-assistant"'));
    expect(out).toContain('This repeats the AI&#x27;s message from turn 1.');
    expect(out).toContain('invalid</span></span><span> · the simulated person copied the AI&#x27;s message at turn 3');
    expect(out).not.toContain('fidelity: valid');
    expect(out).toContain('Counted as invalid: the simulated person copied the AI&#x27;s message at turn 3.');
    expect(out).toContain('The check&#x27;s verdict: <strong>valid</strong></p>');
    expect(await page(RunRoute, { name: 'run-a' })).toContain(
      'the simulated person copied the AI&#x27;s message at turn 3',
    );
  });

  it('survives a malformed failure and null content', async () => {
    use('malformed');
    expect(await caseHtml()).not.toContain('<b>x</b>');
    expect(await caseHtml('aimhef-0000000a', 2)).toContain('id="turn-1-user"');
  });

  it('says so when a transcript is empty', async () => {
    use('empty-turns');
    expect(await caseHtml()).toContain('This record holds no turns.');
  });

  it('shows the case column and the sim driver', async () => {
    use('driver');
    const out = await caseHtml();
    expect(out).toContain('Dana writes after a hard week.');
    expect(out).toContain('Pathway-902');
    expect(out).toContain('p-fix-01@1');
    expect(out).toContain('You are roleplaying as this person.');
  });

  it('names a missing persona', async () => {
    use('unreadable-and-persona');
    expect(await caseHtml('aimhef-0000000b')).toContain('persona record not found: p-gone@1');
  });

  it('is not found for an unknown transcript', async () => {
    use('unjudged');
    expect(await thrown(() => page(CaseRoute, { id: 'nope' }))).toContain('404');
  });

  it('finds a transcript by id alone', async () => {
    use('unjudged');
    expect(await page(CaseRoute, { id: transcriptId('aimhef-0000000a') })).toContain('hello 1');
  });
});

describe('the case column', () => {
  const sentBlocks = (out: string) => [...out.matchAll(/<div class="vw-sent">(.*?)<\/div>/gs)].map((m) => m[1]).join(' ');

  it('marks what the simulated person receives', async () => {
    use('sent-markers');
    const out = await caseHtml();
    const sent = sentBlocks(out);
    for (const text of [
      'Dana, 41, a night-shift nurse.',
      'Dana is in acute distress after a hard week.',
      'Writes short, flat messages.',
      'Dana writes after a hard week.',
      'I can&#x27;t sleep.',
      'Dana says she is lost',
      'asks for a hotline',
    ]) {
      expect(sent, text).toContain(text);
    }
    expect(out).toContain('given to the simulated person');
    for (const notSent of ['Age', 'Pathway-902', 'followed', 'p-fix-01@1']) expect(sent, notSent).not.toContain(notSent);
  });

  it.each(['scripted-l1', 'scripted-l2'])('has no simulated-person markers at %s', async (name) => {
    use(name);
    const out = await caseHtml();
    expect(out).not.toContain('class="vw-sent"');
    expect(out).toContain('Scripted turns');
    expect(out).toContain('Not applicable: scripted levels have no fidelity check.');
    expect(await page(RunRoute, { name: 'run-a' })).toContain('fidelity n/a (L1/L2)');
  });

  it('hides fact mode when no run records one', async () => {
    use('unjudged');
    expect(await caseHtml()).not.toContain('Fact mode');
  });
});

describe('the rail', () => {
  it('links citations to their turn and flags quotes not found', async () => {
    use('citations');
    const out = await caseHtml();
    const cites = [...out.matchAll(/<a class="vw-cite([^"]*)" href="#turn-(\d+)"[^>]*data-quote="([^"]*)"/g)];
    const found = new Map(cites.map(([, cls, turn, quote]) => [`${quote}|${turn}`, cls]));
    expect(found.get('reply 2|2')).toBe('');
    expect(found.get('never said|2')).toBe(' is-unmatched');
    expect(found.get('hello 1|1')).toBe(' is-unmatched');
    expect(out.split('not found verbatim').length - 1).toBeGreaterThanOrEqual(2);
  });

  it('orders steps as they happened and links quotes in place', async () => {
    use('steps');
    const out = await caseHtml();
    const start = out.indexOf('vw-steps');
    const steps = out.slice(start, out.indexOf('</ul>', start));
    expect(steps.indexOf('UserBehavior-901')).toBeLessThan(steps.indexOf('AiBehavior-901'));
    expect(steps).toContain('Opens the exchange the pathway runs through.');
    expect(out).not.toContain('<ol class="vw-steps">');
    expect(steps.split('<q>').length - 1).toBe(0);
    expect(steps).toContain('data-quote="So: hello 1 and more words here, I said"');
    expect(steps).toContain('>hello 1 and more words here</a>');
  });

  it('says why a route did not fire and shows its condition', async () => {
    use('condition');
    const out = await caseHtml();
    const rail = out.slice(out.indexOf('vw-railroute'));
    expect(rail).toContain('Occurred once, but the condition did not hold.');
    expect(rail.indexOf('<h4>Occurred · once</h4>')).toBeLessThan(rail.indexOf('<h4>Condition</h4>'));
    expect(rail).toContain('Occurrence 1');
    expect(rail).toContain('Did not hold</strong>, decided by the test case record.');
    expect(rail).toContain('The judge read the transcript the other way, citing:');
    expect(rail).toContain('Judge&#x27;s reading of the transcript: held.');
    expect(rail).toContain('Test case record: did not hold.');
    expect(rail).toContain('read as held');
    expect(rail).not.toContain('>supports<');
  });

  it('lists only this transcript’s judge calls', async () => {
    use('judge-prompts');
    const out = await caseHtml();
    expect(out).toContain('Stage 1: which behaviors occurred');
    expect(out).toContain('AI behaviors, group 1');
    expect(out).toContain('Pathway-902 <span');
    expect(out).not.toContain('seq=4');
    expect(await caseHtml('aimhef-0000000a', 2)).toContain('seq=4');
  });

  it('reaches the raw judge record', async () => {
    use('judged-one');
    const out = await caseHtml();
    expect(out).toContain('Judge record');
    expect(out).toContain('kind=judge');
  });

  it('keeps rail tabs as links without JavaScript', async () => {
    use('judged-one');
    const out = await caseHtml();
    expect(out).toContain('<nav class="vw-railtabs" aria-label="Rail">');
    expect(out).toContain('href="#rail-fidelity"');
    expect(out).not.toContain('role="tablist"');
    const tabs = [...out.matchAll(/data-tab="([a-z]+)"/g)].map((m) => m[1]);
    expect(tabs).toEqual(['fidelity', 'pathways', 'behaviors', 'records']);
  });

  it('folds each pathway under its result, the target open', async () => {
    use('judged-one');
    const out = await caseHtml();
    expect(out).toContain('<h3>Target pathway</h3>');
    expect(out).toMatch(/<details class="vw-railroute dir-[a-z]+ is-target" open="">/);
  });

  it('labels a row for each kind of mark on the timeline', async () => {
    use('judged-one');
    const out = await caseHtml();
    expect(out).toContain('<li>Target pathway</li><li>Judge</li><li>Fidelity check</li>');
    expect(out).toContain('<span class="vw-key__label">Key</span>');
  });
});

describe('the turn strip', () => {
  it('starts at the onset and narrows when long', async () => {
    use('strip');
    const ten = await caseHtml();
    expect(ten.match(/class="vw-strip__cell[ "]/g)).toHaveLength(10);
    expect(ten).toContain('data-start="4"');
    expect(ten).not.toContain('is-narrow');
    const eighty = await caseHtml('aimhef-0000000a', 2);
    expect(eighty.match(/class="vw-strip__cell[ "]/g)).toHaveLength(80);
    expect(eighty).toContain('data-start="63"');
    expect(eighty).toContain('vw-strip--tall vw-timeline is-narrow');
  });
});

describe('the list a case was opened from', () => {
  it('follows the test case page for previous and next and returns to it', async () => {
    const meta = use('prev-next');
    const cameFrom = url('/test-case', { suite: meta.names[0]!, id: 'aimhef-0000000b' });
    const first = await caseHtml('aimhef-0000000b', 1, cameFrom);
    expect(first).toContain('1 of 2');
    expect(first).not.toContain('rel="prev"');
    const next = /rel="next" href="([^"]+)"/.exec(first);
    expect(next).not.toBeNull();
    const target = new URL(unescape(next![1]!), 'http://x').searchParams;
    expect(target.get('id')).toBe(transcriptId('aimhef-0000000b', 2));
    expect(target.get('from')).toBe(cameFrom);
  });

  it.each(['javascript:/run?name=run-a&a=/;alert(1)//', '//elsewhere.example/run?name=run-a'])(
    'ignores a from link with a scheme or host: %s',
    async (cameFrom) => {
      use('unjudged');
      const out = await caseHtml('aimhef-0000000a', 1, cameFrom);
      expect(out).not.toContain('href="javascript');
      expect(out).not.toContain('href="//elsewhere');
    },
  );

  it('builds the run breadcrumb from the run', async () => {
    use('unjudged');
    const out = await caseHtml('aimhef-0000000a', 1, '/run?name=run-a&extra=%22x%22');
    expect(out).toContain('<a href="/run?name=run-a">run-a</a>');
  });

  it('pages a long run and keeps previous and next across pages', async () => {
    use('long-run');
    const out = await caseHtml('aimhef-0000000a', 3, url('/run', { name: 'run-a', sort: 'case', page: '2' }));
    expect(out).toContain('3 of 3');
  });
});

describe('the system-tested switcher', () => {
  it('marks this system and links the others', async () => {
    use('two-systems');
    const out = await caseHtml('aimhef-0000000a', 1);
    const row = /<div class="vw-switch">(.*?)<\/div>/s.exec(out)![1]!;
    expect(row).toContain('aria-current="page"');
    expect(row).toContain('class="vw-switch__other"');
    expect(row).toContain('other/model');
    expect(out).not.toContain('vw-menu');
  });
});
