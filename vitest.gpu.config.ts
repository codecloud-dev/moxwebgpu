import { defineConfig } from 'vitest/config';

// GPU end-to-end tests. These require a WebGPU-capable browser.
// Use `pnpm test:gpu` (wraps everything in xvfb-run + SwiftShader Vulkan ICD).
export default defineConfig({
  test: {
    include: ['tests/gpu/*.test.ts'],
    environment: 'node',
    testTimeout: 60_000,
    hookTimeout: 120_000,
    // Browser lifecycle is expensive: run all GPU tests in a single worker, sequentially.
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
  },
});
