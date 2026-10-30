import type { NextConfig } from 'next';
import { ROBOTS_DIRECTIVES } from './src/lib/robots';

const nextConfig: NextConfig = {
  async headers() {
    return [{ source: '/:path*', headers: [{ key: 'X-Robots-Tag', value: ROBOTS_DIRECTIVES }] }];
  },
  // Keep tracing inside this app. A root at the repository parent makes `vercel build`,
  // which runs from viewer/, look for viewer/viewer/.next. The bundled-trace check
  // still fails if a route ships anything from data-repository/.
  outputFileTracingRoot: import.meta.dirname,
  // lib/artifacts.ts builds its paths at runtime, so the tracer cannot follow them. Every
  // page and route renders on request, so every route ships the generated data.
  // Keys are matched by picomatch in `contains` mode; paths are relative to viewer/.
  outputFileTracingIncludes: {
    '/': ['./src/generated/**/*.json'],
  },
};

export default nextConfig;
