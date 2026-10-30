/** Route handlers, redirects, the proxy and the artifacts reader. */
import { NextRequest } from 'next/server';
import { describe, expect, it } from 'vitest';
import { GET as judgeCall } from '../src/app/judge-call/route';
import JudgeRedirect from '../src/app/judge/page';
import { GET as record } from '../src/app/record/route';
import CaseRedirect from '../src/app/case/page';
import SuiteRoute from '../src/app/suite/page';
import TranscriptRedirect from '../src/app/transcript/page';
import { MissingArtifact, loadIndex } from '../src/lib/artifacts';
import { proxy } from '../src/proxy';
import { thrown, transcriptId, url, use } from './helpers';

function get(handler: (r: Request) => Response, path: string, params: Record<string, string | number>): Response {
  return handler(new Request(`http://localhost${url(path, params)}`));
}

describe('/judge-call', () => {
  const base = { run: 'run-a', transcript: transcriptId('aimhef-0000000a') };

  it('reads one call as the ingest copied it', async () => {
    use('judge-prompts');
    expect(await get(judgeCall, '/judge-call', { ...base, seq: 3, part: 'prompt' }).text()).toBe('first s2');
    const response = get(judgeCall, '/judge-call', { ...base, seq: 3, part: 'response' });
    expect(await response.text()).toBe('response 3');
    expect(response.headers.get('x-content-type-options')).toBe('nosniff');
  });

  it('is 404 for another transcript’s call, an unknown part or an unknown seq', () => {
    use('judge-prompts');
    expect(get(judgeCall, '/judge-call', { ...base, seq: 4, part: 'prompt' }).status).toBe(404);
    expect(get(judgeCall, '/judge-call', { ...base, seq: 3, part: 'secrets' }).status).toBe(404);
    expect(get(judgeCall, '/judge-call', { ...base, seq: 99, part: 'prompt' }).status).toBe(404);
  });

  it('serves the call the ingest read after the trace is rewritten', async () => {
    use('judge-prompts-rewritten');
    expect(await get(judgeCall, '/judge-call', { ...base, seq: 3, part: 'prompt' }).text()).toBe('first s2');
    expect(get(judgeCall, '/judge-call', { ...base, seq: 7, part: 'prompt' }).status).toBe(404);
  });

  it('reads calls only from the judge setup it names', async () => {
    use('judge-setup-shown');
    const params = { ...base, seq: 2, part: 'prompt' };
    expect(await get(judgeCall, '/judge-call', params).text()).toBe('old prompt');
    expect(get(judgeCall, '/judge-call', { ...params, judge: '00000000' }).status).toBe(404);
  });
});

describe('/record', () => {
  it('returns the raw judge record without its prompt snapshot', async () => {
    use('judged-one');
    const id = transcriptId('aimhef-0000000a');
    const response = get(record, '/record', { run: 'run-a', transcript: id, kind: 'judge', judge: 'run-a-judge.jsonl:1' });
    const text = await response.text();
    expect(response.status).toBe(200);
    expect(text).toContain('"coralmind-1"');
    expect(text).not.toContain('judge_prompt_snapshot');
  });

  it('returns the transcript record', async () => {
    use('judged-one');
    const text = await get(record, '/record', { run: 'run-a', transcript: transcriptId('aimhef-0000000a'), kind: 'transcript' }).text();
    expect(JSON.parse(text).turns[0].content).toBe('hello 1');
  });

  it('serves the transcript the ingest read after its file is rewritten', async () => {
    use('rewritten');
    const response = get(record, '/record', { run: 'run-a', transcript: transcriptId('aimhef-0000000a'), kind: 'transcript' });
    expect(response.status).toBe(200);
    expect(JSON.parse(await response.text()).turns[0].content).toBe('ALPHA');
  });

  it('is 404 for judge prompts the record did not hold', () => {
    use('judged-one');
    const id = transcriptId('aimhef-0000000a');
    const params = { run: 'run-a', transcript: id, kind: 'judge_prompts', judge: 'run-a-judge.jsonl:1' };
    expect(get(record, '/record', params).status).toBe(404);
  });

  it('is 404 for an unknown kind or transcript', () => {
    use('judged-one');
    expect(get(record, '/record', { run: 'run-a', transcript: 'nope', kind: 'transcript' }).status).toBe(404);
    expect(get(record, '/record', { run: 'run-a', transcript: transcriptId('aimhef-0000000a'), kind: 'x' }).status).toBe(404);
  });
});

describe('old links', () => {
  it('redirect to their new pages', async () => {
    use('unjudged');
    const toCase = await thrown(() =>
      TranscriptRedirect({
        searchParams: Promise.resolve({ run: 'run-a', batch: 'batch-001', file: 'run-a-sim.jsonl', index: '0' }),
      }),
    );
    expect(toCase).toContain(url('/transcript', { run: 'run-a', id: transcriptId('aimhef-0000000a') }));
    const fromCase = await thrown(() =>
      CaseRedirect({ searchParams: Promise.resolve({ run: 'run-a', transcript: transcriptId('aimhef-0000000a'), from: '/runs' }) }),
    );
    expect(fromCase).toContain(url('/transcript', { run: 'run-a', from: '/runs', id: transcriptId('aimhef-0000000a') }));
    const fromSuite = await thrown(() => SuiteRoute({ searchParams: Promise.resolve({ suite: 'suite@abc', tab: 'how' }) }));
    expect(fromSuite).toContain(url('/suite', { suite: 'suite' }));
    const toRun = await thrown(() =>
      JudgeRedirect({ searchParams: Promise.resolve({ run: 'run-a', batch: 'batch-001', file: 'run-a-judge.jsonl' }) }),
    );
    expect(toRun).toContain('/run?name=run-a&batch=batch-001');
  });

  it.each(['-1', '5'])('is not found outside the file: index %s', async (index) => {
    use('unjudged');
    const digest = await thrown(() =>
      TranscriptRedirect({ searchParams: Promise.resolve({ run: 'run-a', batch: 'batch-001', file: 'run-a-sim.jsonl', index }) }),
    );
    expect(digest).toContain('404');
  });
});

describe('the proxy', () => {
  it('refuses a self-identified AI agent', () => {
    const response = proxy(new NextRequest('http://localhost/suites', { headers: { 'user-agent': 'Mozilla/5.0 GPTBot/1.2' } }));
    expect(response.status).toBe(403);
    expect(response.headers.get('x-robots-tag')).toContain('noai');
  });

  it('lets a browser through', () => {
    const response = proxy(new NextRequest('http://localhost/suites', { headers: { 'user-agent': 'Mozilla/5.0 Safari' } }));
    expect(response.status).toBe(200);
  });
});

describe('the artifacts reader', () => {
  it('throws with the fix when the data is missing', () => {
    process.env.LAB_GENERATED_DIR = '/nonexistent/generated';
    expect(() => loadIndex()).toThrow(MissingArtifact);
    expect(() => loadIndex()).toThrow('pixi run -e viewer viewer-dev');
  });
});
