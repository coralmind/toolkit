/**
 * `/judge-call?run&transcript&judge&seq&part`: one judge call's prompt, response or system
 * prompt, as the ingest copied it from the judge trace.
 */
import { loadRecords } from '../../lib/artifacts';
import { textResponse } from '../../lib/records';

export const dynamic = 'force-dynamic';

const PARTS = new Set(['prompt', 'response', 'system'] as const);
type Part = 'prompt' | 'response' | 'system';

export function GET(request: Request): Response {
  const params = new URL(request.url).searchParams;
  const part = params.get('part') ?? '';
  const seq = Number(params.get('seq'));
  const wanted = params.get('judge');
  if (!PARTS.has(part as Part) || !Number.isInteger(seq)) return textResponse('Not found', 404);
  const data = loadRecords(params.get('run') ?? '', params.get('transcript') ?? '');
  if (!data) return textResponse('Not found', 404);
  for (const [ref, judge] of Object.entries(data.judges)) {
    if (wanted && ref !== wanted) continue;
    const call = judge.calls.find((c) => c.seq === seq);
    if (call) return textResponse(call[part as Part]);
  }
  return textResponse('Not found', 404);
}
