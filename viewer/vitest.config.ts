import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    // Pages render with react-dom/server; no DOM environment is needed.
    environment: 'node',
    include: ['tests/**/*.test.{ts,tsx}'],
    globalSetup: ['tests/setup-fixtures.ts'],
  },
});
