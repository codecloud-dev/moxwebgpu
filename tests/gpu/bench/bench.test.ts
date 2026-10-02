/**
 * GPU micro-benchmarks (end-to-end: dispatch + readback).
 *
 * Run with:  pnpm bench
 * (uses the SwiftShader/xvfb harness on GPU-less machines)
 *
 * NOTE: everything runs inside one serialized page callback — no outer
 * closures may be referenced from within `withGpu`.
 */
import { describe, it, afterAll } from 'vitest';
import { withGpu, closeBrowser } from '../harness.js';

afterAll(() => closeBrowser());

describe('moxwebgpu benchmarks (real GPU, dispatch+readback)', () => {
  it('reports elementwise / matmul / reduce timings', async () => {
    const r = await withGpu(async (gpu: any) => {
      // Median of 10 runs after 3 warmups, measured inside the page.
      const bench = async (fn: () => Promise<unknown>, warmup = 3, n = 10) => {
        for (let i = 0; i < warmup; i++) await fn();
        const times: number[] = [];
        for (let i = 0; i < n; i++) {
          const t0 = performance.now();
          await fn();
          times.push(performance.now() - t0);
        }
        times.sort((a, b) => a - b);
        return times[Math.floor(n / 2)];
      };

      const N = 1 << 20; // 1M elements
      const big = () => gpu.tensor(new Float32Array(N).fill(1));
      const mA = gpu.tensor(new Float32Array(512 * 512).fill(0.5), { shape: [512, 512] });
      const mB = gpu.tensor(new Float32Array(512 * 512).fill(2), { shape: [512, 512] });

      const addMs = await bench(() => gpu.tensor(new Float32Array(N).fill(1)).add(1).toArray());
      const matmulMs = await bench(() => mA.matmul(mB).toArray());
      const sumMs = await bench(() => big().sum().item());
      const chainMs = await bench(() => big().add(1).relu().mul(2).sum().item());

      return { addMs, matmulMs, sumMs, chainMs };
    });
    const gflops = ((512 * 512 * 512 * 2) / r.matmulMs / 1e6).toFixed(2);
    console.log(`
┌──────────────────────────────────────────────────────────┐
│  moxwebgpu micro-benchmarks (median, incl. readback)        │
├─────────────────────────────────┬────────────────────────┤
│  add 1M f32                     │  ${r.addMs.toFixed(2).padStart(9)} ms             │
│  matmul [512×512]·[512×512]     │  ${r.matmulMs.toFixed(2).padStart(9)} ms   ${gflops} GFLOP/s
│  sum 1M f32 (2-phase tree)      │  ${r.sumMs.toFixed(2).padStart(9)} ms             │
│  add→relu→mul→sum 1M            │  ${r.chainMs.toFixed(2).padStart(9)} ms             │
└─────────────────────────────────┴────────────────────────┘
  (SwiftShader is software rendering; on real hardware expect 10-100× faster.)`);
  }, 120_000);
});
