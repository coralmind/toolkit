import { renderPage, title } from '../../components/Frame';
import { RunsList } from '../../components/Lists';

export const dynamic = 'force-dynamic';

export const metadata = { title: title('Runs') };

export default function RunsPage() {
  return renderPage('runs', (ctx) => ({ section: 'runs', body: <RunsList index={ctx.index} /> }));
}
