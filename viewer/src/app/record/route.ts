/**
 * `/record?run&transcript&kind`: the full transcript record, judge record or judge prompts,
 * as the ingest copied them, as indented JSON.
 */
import { loadRecords } from '../../lib/artifacts';
import { textResponse } from '../../lib/records';

export const dynamic = 'force-dynamic';

export function GET(request: Request): Response {
  const params = new URL(request.url).searchParams;
  const kind = params.get('kind') ?? '';
  const data = loadRecords(params.get('run') ?? '', params.get('transcript') ?? '');
  if (!data) return textResponse('Not found', 404);
  let shown: unknown;
  if (kind === 'transcript') {
    shown = data.transcript;
  } else if (kind === 'judge' || kind === 'judge_prompts') {
    const judge = data.judges[params.get('judge') ?? ''];
    shown = kind === 'judge' ? judge?.record : judge?.prompts;
  } else {
    return textResponse('Not found', 404);
  }
  if (!shown) return textResponse('Not found', 404);
  return textResponse(JSON.stringify(shown, null, 2), 200, { 'Content-Type': 'application/json; charset=utf-8' });
}
