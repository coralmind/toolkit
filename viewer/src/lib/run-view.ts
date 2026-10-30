/** A Run URL read against the run's data: filters, sort and page. */
import { loadRun } from './artifacts';
import { BUCKETS, PROBLEM_ORDER, RUN_SORTS } from './labels';
import { type Notice, linkNotice } from './notices';
import { NotFound, type Params, type RunData, type RunRow } from './types';
import { url } from './url';

/** Transcript rows per Run page. Previous and next on a case still follow the whole run. */
export const RUN_PAGE_SIZE = 200;

export interface RunView {
  run: RunData;
  params: Params;
  rows: RunRow[];
  batch: string | null;
  bucket: string | null;
  sort: string;
  page: number;
  pages: number;
  pageSize: number;
  found: Notice[];
}

/** A link to this Run page with some values changed. Any change but the page resets the page. */
export function runLink(view: RunView, changes: Record<string, string | null> = {}): string {
  const params: Record<string, string | null> = { ...view.params };
  if (Object.keys(changes).some((k) => k !== 'page')) delete params.page;
  return url('/run', { ...params, ...changes });
}

export function shownRows(view: RunView): RunRow[] {
  const start = (view.page - 1) * view.pageSize;
  return view.rows.slice(start, start + view.pageSize);
}

export function runView(params: Params, pageSize = RUN_PAGE_SIZE): RunView {
  const name = params.name;
  const run = name ? loadRun(name) : null;
  if (!run || run.name !== name) throw new NotFound(`run ${name ?? ''}`);
  const found: Notice[] = [];
  let batch: string | null = params.batch || null;
  if (batch !== null && !run.batches.includes(batch)) {
    found.push(linkNotice(`Batch ${batch} not found; showing every batch.`));
    batch = null;
  }
  let bucket: string | null = params.bucket || null;
  if (bucket !== null && !(BUCKETS as readonly string[]).includes(bucket)) {
    found.push(linkNotice(`Status ${bucket} not found; showing every status.`));
    bucket = null;
  }
  let sort = params.sort || 'problems';
  if (!Object.hasOwn(RUN_SORTS, sort)) {
    found.push(linkNotice(`Sort ${sort} not found.`));
    sort = 'problems';
  }
  const rows = run.transcripts.filter(
    (t) => (batch === null || t.batch === batch) && (bucket === null || t.bucket === bucket),
  );
  const caseOrder = (a: RunRow, b: RunRow) =>
    a.test_case_id < b.test_case_id ? -1 : a.test_case_id > b.test_case_id ? 1 : a.sut < b.sut ? -1 : a.sut > b.sut ? 1 : a.line - b.line;
  if (sort === 'problems') {
    const rank = (t: RunRow) => (PROBLEM_ORDER as readonly string[]).indexOf(t.bucket);
    rows.sort((a, b) => rank(a) - rank(b) || caseOrder(a, b));
  } else {
    rows.sort(caseOrder);
  }
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  const rawPage = params.page || '1';
  let page = /^[0-9]+$/.test(rawPage) ? Number(rawPage) : 0;
  if (!(page >= 1 && page <= pages)) {
    found.push(linkNotice(`Page ${rawPage} not found; showing page 1.`));
    page = 1;
  }
  const clean: Params = { name: run.name };
  if (batch) clean.batch = batch;
  if (bucket) clean.bucket = bucket;
  if (sort !== 'problems') clean.sort = sort;
  if (page > 1) clean.page = String(page);
  return { run, params: clean, rows, batch, bucket, sort, page, pages, pageSize, found };
}
