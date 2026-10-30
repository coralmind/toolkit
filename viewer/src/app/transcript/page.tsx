/**
 * `/transcript?run&id`: one transcript. An old `/transcript?run&batch&file&index` link
 * redirects to the transcript on that line.
 */
import type { Metadata } from 'next';
import { notFound, redirect } from 'next/navigation';
import { CasePage } from '../../components/CasePage';
import { renderPage, title } from '../../components/Frame';
import { loadRun } from '../../lib/artifacts';
import { findCase, judgeFor, listContext } from '../../lib/case-view';
import { firstValues, url } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  const id = firstValues(await searchParams).id ?? '';
  return { title: title(id.split(':').at(-2) ?? id) };
}

export default async function TranscriptRoute({ searchParams }: Props) {
  const params = firstValues(await searchParams);
  if (params.batch || params.file || params.index) {
    const run = params.run ? loadRun(params.run) : null;
    const ids = run ? run.files[`${params.batch ?? ''}/${params.file ?? ''}`] : undefined;
    const index = /^[0-9]+$/.test(params.index ?? '') ? Number(params.index) : -1;
    const id = ids?.[index];
    if (!run || id === undefined) notFound();
    redirect(url('/transcript', { run: run.name, id }));
  }
  return renderPage('runs', (ctx) => {
    const [data, notes] = findCase(params.id, params.run);
    const context = listContext(params.from ?? '', ctx.library);
    return {
      section: context && context.kind !== 'run' ? 'suites' : 'runs',
      notices: notes,
      body: <CasePage data={data} judge={judgeFor(data)} context={context} library={ctx.library} />,
    };
  });
}
