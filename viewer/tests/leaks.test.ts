/** The leak check that `pixi run -e viewer viewer-leaks` runs. */
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { leaksIn, scan } from '../scripts/leaks';

describe('the leak check', () => {
  it('finds a home path, an endpoint address and an internal field', () => {
    expect(leaksIn('a.json', '{"root": "/Users/someone/results"}').map((l) => l.kind)).toEqual(['home path']);
    expect(leaksIn('b.log', 'POST https://example-foundry.openai.azure.com/openai').map((l) => l.kind)).toEqual([
      'endpoint address',
    ]);
    expect(leaksIn('c.jsonl', '{"ok": 1}\n{"x": {"_note": "private"}}\n').map((l) => l.kind)).toEqual(['internal field']);
  });

  it('allows ordinary URLs and text', () => {
    expect(leaksIn('d.json', '{"content": "You can call or text 988, or visit https://988lifeline.org."}')).toEqual([]);
  });

  it('passes the committed samples', () => {
    const repo = join(import.meta.dirname, '..', '..');
    expect(scan([join(repo, 'data-repository', 'sample_results')], repo)).toEqual([]);
  });
});
