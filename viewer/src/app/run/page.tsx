import type { Metadata } from 'next';
import { renderPage, title } from '../../components/Frame';
import { RunPage } from '../../components/RunPage';
import { runView } from '../../lib/run-view';
import { firstValues } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  return { title: title(`${firstValues(await searchParams).name ?? ''} · Run`) };
}

export default async function RunRoute({ searchParams }: Props) {
  const params = firstValues(await searchParams);
  return renderPage('runs', (ctx) => {
    const view = runView(params);
    return {
      section: 'runs',
      notices: view.found,
      body: <RunPage view={view} library={ctx.library} />,
    };
  });
}
