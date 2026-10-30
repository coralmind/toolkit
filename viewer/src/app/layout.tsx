/**
 * The brand stylesheet assigns everything to cascade layers, so the viewer's unlayered CSS
 * wins regardless of import order.
 *
 * This build MUST use Turbopack. Webpack rewrites the brand's `@import ... layer(...)` rules
 * into invalid `@media layer(...)`, and the page renders completely unstyled with no error.
 */
import '@coralmind/brand/styles/index.css';
import '../styles/viewer.css';

import type { Metadata } from 'next';
import Script from 'next/script';
import { SiteFooter } from '../components/SiteFooter';
import { ROBOTS_DIRECTIVES } from '../lib/robots';

export const metadata: Metadata = {
  title: 'CoralMind Evaluation Lab',
  robots: ROBOTS_DIRECTIVES,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="vw-page">
        {children}
        <SiteFooter />
        <Script src="/viewer.js" strategy="afterInteractive" />
      </body>
    </html>
  );
}
