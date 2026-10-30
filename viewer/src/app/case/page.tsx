/** `/case?run&transcript`, an old link: redirects to the same transcript at `/transcript`. */
import { redirect } from 'next/navigation';
import { firstValues, url } from '../../lib/url';

export const dynamic = 'force-dynamic';

type Props = { searchParams: Promise<Record<string, string | string[] | undefined>> };

export default async function CaseRedirect({ searchParams }: Props) {
  const { transcript, ...rest } = firstValues(await searchParams);
  redirect(url('/transcript', { ...rest, id: transcript }));
}
