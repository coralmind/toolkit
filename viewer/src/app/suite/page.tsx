import type { Metadata } from 'next';
import { redirect } from 'next/navigation';
import { renderPage, title } from '../../components/Frame';
import { SuiteFilePage } from '../../components/SuiteFilePage';
import type { Notice } from '../../lib/notices';
import { suiteFileView } from '../../lib/suite-file-view';
import { firstValues, url } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export async function generateMetadata({ searchParams }: Props): Promise<Metadata> {
  return { title: title(`${firstValues(await searchParams).suite ?? ''} · Suite`) };
}

export default async function SuiteRoute({ searchParams }: Props) {
  const params = firstValues(await searchParams);
  const [name, digest] = (params.suite ?? '').split('@');
  if (digest !== undefined) {
    // An old link named one version of the file: open the file's test cases.
    redirect(url('/suite', { suite: name }));
  }
  return renderPage('suites', (ctx) => {
    const view = suiteFileView(params, ctx.library);
    const notices: Notice[] = [];
    if (ctx.index.library_error) notices.push({ kind: 'kl', text: `Knowledge Library unavailable: ${ctx.index.library_error}` });
    notices.push(...view.found);
    return {
      section: 'suites',
      notices,
      body: <SuiteFilePage view={view} library={ctx.library} />,
    };
  });
}
