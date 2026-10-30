/**
 * Refuses every request from a self-identified AI crawler with a 403.
 *
 * `robots.txt` stays reachable so a compliant crawler can read the disallow.
 */
import { NextResponse, type NextRequest } from 'next/server';
import { ROBOTS_DIRECTIVES, isAiUserAgent } from './lib/robots';

export function proxy(request: NextRequest) {
  if (isAiUserAgent(request.headers.get('user-agent'))) {
    return new NextResponse('Forbidden', {
      status: 403,
      headers: { 'Content-Type': 'text/plain', 'X-Robots-Tag': ROBOTS_DIRECTIVES },
    });
  }
  return NextResponse.next();
}

export const config = {
  matcher: '/((?!robots\\.txt$|_next/static/).*)',
};
