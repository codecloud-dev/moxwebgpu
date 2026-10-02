import { describe, it, expect, afterAll } from 'vitest';
import { withGpu, closeBrowser } from './harness.js';

afterAll(() => closeBrowser());

describe('elementwise ops (real GPU)', () => {
  it('add / sub / mul / div two tensors', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3, 4]);
      const b = gpu.tensor([10, 20, 30, 40]);
      return {
        add: Array.from(await a.add(b).toArray()),
        sub: Array.from(await a.sub(b).toArray()),
        mul: Array.from(await a.mul(b).toArray()),
        div: Array.from(await b.div(a).toArray()),
      };
    });
    expect(r.add).toEqual([11, 22, 33, 44]);
    expect(r.sub).toEqual([-9, -18, -27, -36]);
    expect(r.mul).toEqual([10, 40, 90, 160]);
    expect(r.div).toEqual([10, 10, 10, 10]);
  });

  it('scalar ops + reversed variants', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3]);
      return {
        addS: Array.from(await a.add(10).toArray()),
        subS: Array.from(await a.sub(1).toArray()),
        rsubS: Array.from(await a.rsub(10).toArray()),
        mulS: Array.from(await a.mul(2).toArray()),
        divS: Array.from(await a.div(2).toArray()),
        rdivS: Array.from(await a.rdiv(6).toArray()),
      };
    });
    expect(r.addS).toEqual([11, 12, 13]);
    expect(r.subS).toEqual([0, 1, 2]);
    expect(r.rsubS).toEqual([9, 8, 7]);
    expect(r.mulS).toEqual([2, 4, 6]);
    expect(r.divS).toEqual([0.5, 1, 1.5]);
    expect(r.rdivS).toEqual([6, 3, 2]);
  });

  it('unary chain: relu / exp / neg / sigmoid', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([-2, -0.5, 0, 0.5, 2]);
      return {
        relu: Array.from<number>(await a.relu().toArray()),
        neg: Array.from<number>(await a.neg().toArray()),
        exp0: await gpu.tensor([0]).exp().item(),
        sig: Array.from<number>(await a.sigmoid().toArray()).map((v: number) => Math.round(v * 1000) / 1000),
      };
    });
    expect(r.relu.map((v: number) => v + 0)).toEqual([0, 0, 0, 0.5, 2]); // +0 normalization
    // neg flips the sign bit (IEEE 754): neg(0) === -0; normalize to +0.
    expect(r.neg.map((v: number) => v + 0)).toEqual([2, 0.5, 0, -0.5, -2]);
    expect(r.exp0).toBeCloseTo(1, 5);
    expect(r.sig[0]).toBeCloseTo(0.119, 2);
    expect(r.sig[4]).toBeCloseTo(0.881, 2);
  });

  it('lazy chain evaluates once: add -> relu -> mul -> sum', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3, 4]);
      return await a.add(1).relu().mul(10).sum().item();
    });
    // ((1+1,2+1,3+1,4+1) = 2,3,4,5) *10 = 20+30+40+50 = 140
    expect(r).toBe(140);
  });

  it('row broadcast: [2,3] + [3]', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3], [10, 20, 30]]);
      const b = gpu.tensor([1, 2, 3]);
      return Array.from(await a.add(b).toArray());
    });
    expect(r).toEqual([2, 4, 6, 11, 22, 33]);
  });

  it('column broadcast: [2,3] * [2]', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3], [10, 20, 30]]);
      const b = gpu.tensor([2, 3]);
      return Array.from(await a.mul(b).toArray());
    });
    expect(r).toEqual([2, 4, 6, 30, 60, 90]);
  });

  it('cast f32 -> i32', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1.7, 2.2, -3.9]);
      return Array.from(await a.cast('i32').toArray());
    });
    expect(r).toEqual([1, 2, -3]); // truncation toward zero
  });

  it('low-level Kernel API: raw WGSL dispatch + readback', async () => {
    const r = await withGpu(async (gpu: any) => {
      const wgsl = `
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<f32>;
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
          b[gid.x] = a[gid.x] * 2.0 + 1.0;
        }`;
      const k = gpu.kernel(wgsl, { workgroupSize: 64 });
      const a = gpu.tensor([1, 2, 3, 4, 5]);
      const out = gpu.pool.acquire(5, 'f32');
      const res = await k.run([a.data.buffer, out.buffer], { elements: 5 });
      return Array.from(res.slice(0, 5));
    });
    expect(r).toEqual([3, 5, 7, 9, 11]);
  });
});
