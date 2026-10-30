import { renderPage, title } from '../../components/Frame';
import { SuitesList } from '../../components/Lists';

export const dynamic = 'force-dynamic';

export const metadata = { title: title('Suites') };

export default function SuitesPage() {
  return renderPage('suites', (ctx) => ({ section: 'suites', body: <SuitesList index={ctx.index} /> }));
}
