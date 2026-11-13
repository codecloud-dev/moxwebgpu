/**
 * Real-GPU regression for integer scalar ops.
 *
 * The uniform block stores `scalar` as f32. Before the 2026-11-13 fix,
 * `i32` / `u32` tensors produced WGSL like `a[i] + uniforms.scalar`, which
 * fails shader compilation. Unit tests only assert the generated source;
 * this file compiles and executes the shader on SwiftShader so the fix is
 * proven end-to-end, not just on paper.
 */

import { describe, it, expect, afterAll } from 'vitest';
import { withGpu, closeBrowser } from './harness.js';

afterAll(() => closeBrowser());

describe('integer scalar ops (real GPU / SwiftShader)', () => {
  it('i32 scalar add / sub / mul / div / rsub / clamp compile and compute', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3, -4], { dtype: 'i32' });
      return {
        add: Array.from(await a.add(10).toArray()),
        sub: Array.from(await a.sub(1).toArray()),
        mul: Array.from(await a.mul(3).toArray()),
        div: Array.from(await a.div(2).toArray()),
        rsub: Array.from(await a.rsub(10).toArray()),
        clamp: Array.from(await a.clamp(0, 2).toArray()),
      };
    });
    expect(r.add).toEqual([11, 12, 13, 6]);
    expect(r.sub).toEqual([0, 1, 2, -5]);
    expect(r.mul).toEqual([3, 6, 9, -12]);
    // WGSL integer division truncates toward zero.
    expect(r.div).toEqual([0, 1, 1, -2]);
    expect(r.rsub).toEqual([9, 8, 7, 14]);
    expect(r.clamp).toEqual([1, 2, 2, 0]);
  });

  it('u32 scalar add / mul / clamp compile and compute', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3], { dtype: 'u32' });
      return {
        add: Array.from(await a.add(10).toArray()),
        mul: Array.from(await a.mul(3).toArray()),
        clamp: Array.from(await a.clamp(0, 2).toArray()),
      };
    });
    expect(r.add).toEqual([11, 12, 13]);
    expect(r.mul).toEqual([3, 6, 9]);
    expect(r.clamp).toEqual([1, 2, 2]);
  });

  it('f32 scalar path is unchanged', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3]);
      return Array.from(await a.add(0.5).toArray());
    });
    expect(r).toEqual([1.5, 2.5, 3.5]);
  });

  it('argmax on an empty tensor is rejected in the shipped bundle', async () => {
    const msg = await withGpu(async (gpu: any) => {
      try {
        const t = gpu.tensor([], {});
        (t as any).argmax();
        return 'NO_ERROR';
      } catch (e: any) {
        return String(e && e.message ? e.message : e);
      }
    });
    expect(msg).toMatch(/non-empty tensor/);
  });
});
