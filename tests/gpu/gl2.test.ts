/**
 * WebGL2 fallback backend tests — run inside a real Chromium with SwiftShader.
 * Verifies float-texture elementwise, matmul and reduction against JS reference.
 */
import { describe, it, expect, afterAll } from 'vitest';
import { getGpuPage, closeBrowser } from './harness.js';

afterAll(async () => { await closeBrowser(); });

async function run(fn: (M: any) => Promise<any>) {
  const page = await getGpuPage();
  return page.evaluate(async (src) => {
    const M = (window as any).MoxWebGPU;
    if (!M) throw new Error('bundle missing');
    // eslint-disable-next-line no-new-func
    return (new Function('M', `return (${src})(M)`))(M);
  }, fn.toString());
}

describe('WebGL2 fallback backend', () => {
  it('detects WebGL2 with float render targets', async () => {
    const r = await run(async (M: any) => ({
      has: M.hasWebGl2(),
      forced: M.detectBackend('webgl2'),
    }));
    expect(r.has).toBe(true);
    expect(r.forced.kind).toBe('webgl2');
  });

  it('elementwise add / mul / scalar', async () => {
    const r = await run(async (M: any) => {
      const rt = new M.GlRuntime();
      const a = M.GlTensor.from(rt, [1, 2, 3, 4]);
      const b = M.GlTensor.from(rt, [10, 20, 30, 40]);
      const add = await a.add(b).toArray();
      const mul = await a.mul(2).toArray();
      const sub = await b.sub(a).toArray();
      rt.destroy();
      return { add: [...add], mul: [...mul], sub: [...sub] };
    });
    expect(r.add).toEqual([11, 22, 33, 44]);
    expect(r.mul).toEqual([2, 4, 6, 8]);
    expect(r.sub).toEqual([9, 18, 27, 36]);
  });

  it('unary exp / sqrt / relu', async () => {
    const r = await run(async (M: any) => {
      const rt = new M.GlRuntime();
      const a = M.GlTensor.from(rt, [0, 1, 4, -3]);
      const sqrt = await a.sqrt().toArray();
      const relu = await a.relu().toArray();
      const exp0 = await M.GlTensor.from(rt, [0]).exp().toArray();
      rt.destroy();
      return { sqrt: [...sqrt], relu: [...relu], exp0: exp0[0] };
    });
    expect(r.sqrt[0]).toBeCloseTo(0, 5);
    expect(r.sqrt[1]).toBeCloseTo(1, 5);
    expect(r.sqrt[2]).toBeCloseTo(2, 5);
    expect(Number.isNaN(r.sqrt[3])).toBe(true); // sqrt(-3)
    expect(r.relu).toEqual([0, 1, 4, 0]);
    expect(r.exp0).toBeCloseTo(1, 5);
  });

  it('matmul 2x3 @ 3x2', async () => {
    const r = await run(async (M: any) => {
      const rt = new M.GlRuntime();
      const a = M.GlTensor.from(rt, [[1, 2, 3], [4, 5, 6]]);
      const b = M.GlTensor.from(rt, [[7, 8], [9, 10], [11, 12]]);
      const c = await a.matmul(b).toArray();
      rt.destroy();
      return [...c];
    });
    // [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    expect(r).toEqual([58, 64, 139, 154]);
  });

  it('reduce sum / mean', async () => {
    const r = await run(async (M: any) => {
      const rt = new M.GlRuntime();
      const a = M.GlTensor.from(rt, [1, 2, 3, 4, 5]);
      const sum = a.sum();
      const mean = a.mean();
      rt.destroy();
      return { sum, mean };
    });
    expect(r.sum).toBeCloseTo(15, 4);
    expect(r.mean).toBeCloseTo(3, 4);
  });

  it('broadcast row vector over matrix', async () => {
    const r = await run(async (M: any) => {
      const rt = new M.GlRuntime();
      const a = M.GlTensor.from(rt, [[1, 2, 3], [4, 5, 6]]);
      const b = M.GlTensor.from(rt, [10, 20, 30]);
      const c = await a.add(b).toArray();
      rt.destroy();
      return [...c];
    });
    expect(r).toEqual([11, 22, 33, 14, 25, 36]);
  });
});
