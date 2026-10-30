/** `/`: the Introduction, on both builds. */
import { renderPage, title } from '../components/Frame';
import { Introduction } from '../components/Introduction';

export const dynamic = 'force-dynamic';

export const metadata = { title: title('Introduction') };

export default function IntroductionPage() {
  return renderPage('intro', () => ({ section: 'intro', body: <Introduction /> }));
}
