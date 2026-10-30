import type { Metadata } from 'next';
import { renderPage, title } from '../../components/Frame';
import { TestCasePage } from '../../components/TestCasePage';
import { testCaseView } from '../../lib/test-case-view';
import { firstValues } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  return { title: title(`${firstValues(await searchParams).id ?? ''} · Test case`) };
}

export default async function TestCaseRoute({ searchParams }: Props) {
  const params = firstValues(await searchParams);
  return renderPage('suites', (ctx) => {
    const view = testCaseView(params);
    return {
      section: 'suites',
      notices: view.found,
      body: <TestCasePage view={view} library={ctx.library} />,
    };
  });
}
