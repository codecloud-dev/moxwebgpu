import { defineConfig } from 'vitest/config';

// GPU micro-benchmarks only (see tests/gpu/bench.test.ts). Run via `pnpm bench`.
export default defineConfig({
  test: {
    include: ['tests/gpu/bench/bench.test.ts'],
    environment: 'node',
    testTimeout: 300_000,
    hookTimeout: 120_000,
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
  },
});
