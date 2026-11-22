import { defineConfig } from 'vitest/config';

// WebGL2 fallback backend tests. Run through scripts/run-gpu-tests.sh so the
// browser starts under xvfb with SwiftShader. No WebGPU ICD needed.
export default defineConfig({
  test: {
    include: ['tests/gpu/gl2.test.ts'],
    environment: 'node',
    testTimeout: 60_000,
    hookTimeout: 120_000,
    pool: 'forks',
    poolOptions: { forks: { singleFork: true } },
  },
});
