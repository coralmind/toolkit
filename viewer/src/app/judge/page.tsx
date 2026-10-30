/** `/judge?run&batch&file`, an old link: redirects to that run's page, narrowed to the batch. */
import { notFound, redirect } from 'next/navigation';
import { loadRun } from '../../lib/artifacts';
import { firstValues, url } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export default async function JudgeRedirect({ searchParams }: Props) {
  const params = firstValues(await searchParams);
  const run = params.run ? loadRun(params.run) : null;
  if (!run) notFound();
  redirect(url('/run', { name: run.name, batch: params.batch }));
}
