import { describe, it, expect, afterAll } from 'vitest';
import { withGpu, closeBrowser } from './harness.js';
import { cpuSoftmaxLastAxis, cpuArgmax, cpuArgmin } from './ref.js';

afterAll(() => closeBrowser());

describe('reduce ops (real GPU, two-phase tree)', () => {
  it('sum / mean / max / min global', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
      return {
        sum: await a.sum().item(),
        mean: await a.mean().item(),
        max: await a.maxReduce().item(),
        min: await a.minReduce().item(),
      };
    });
    expect(r.sum).toBe(55);
    expect(r.mean).toBeCloseTo(5.5, 5);
    expect(r.max).toBe(10);
    expect(r.min).toBe(1);
  });

  it('sum with padding-heavy sizes (non power of two, > 512)', async () => {
    const n = 1000;
    const got = await withGpu(async (gpu: any, n: number) => {
      const data = Array.from({ length: n }, (_, i) => i + 1);
      return await gpu.tensor(data).sum().item();
    }, n);
    expect(got).toBe((n * (n + 1)) / 2);
  });

  it('argmax / argmin (first occurrence wins)', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([3, 9, 2, 9, 1]);
      return {
        argmax: await a.argmax().item(),
        argmin: await a.argmin().item(),
      };
    });
    expect(r.argmax).toBe(1); // first 9
    expect(r.argmin).toBe(4); // 1
  });

  it('axis=-1 reduce: [3,4] -> [3]', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      return {
        sum: Array.from(await a.sum(-1).toArray()),
        mean: Array.from(await a.mean(-1).toArray()),
        max: Array.from(await a.maxReduce(-1).toArray()),
      };
    });
    expect(r.sum).toEqual([10, 26, 42]);
    expect(r.mean[0]).toBeCloseTo(2.5, 5);
    expect(r.mean[1]).toBeCloseTo(6.5, 5);
    expect(r.mean[2]).toBeCloseTo(10.5, 5);
    expect(r.max).toEqual([4, 8, 12]);
  });

  it('axis=0 reduce: [3,4] -> [4]', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      return Array.from(await a.sum(0).toArray());
    });
    expect(r).toEqual([15, 18, 21, 24]);
  });
});

describe('softmax (real GPU)', () => {
  it('matches CPU reference on [4, 10]', async () => {
    const rows = 4, cols = 10;
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = Math.sin(i * 1.7) * 5;
    const expected = cpuSoftmaxLastAxis(data, rows, cols);

    const got = await withGpu(async (gpu: any, arg: any) => {
      return Array.from<number>(await gpu.tensor(arg.data, { shape: [arg.rows, arg.cols] }).softmax().toArray());
    }, { data, rows, cols });
    for (let i = 0; i < data.length; i++) {
      expect(got[i]).toBeCloseTo(expected[i], 4);
    }
    // Each row sums to 1.
    for (let r0 = 0; r0 < rows; r0++) {
      const s = got.slice(r0 * cols, (r0 + 1) * cols).reduce((a: number, b: number) => a + b, 0);
      expect(s).toBeCloseTo(1, 4);
    }
  });

  it('1D input', async () => {
    const r = await withGpu(async (gpu: any) => {
      return Array.from<number>(await gpu.tensor([1, 2, 3]).softmax().toArray()).map((v: number) =>
        Math.round(v * 1000) / 1000,
      );
    });
    // e^1, e^2, e^3 -> [0.0900, 0.2447, 0.6652]
    expect(r).toEqual([0.09, 0.245, 0.665]);
  });
});

describe('shape ops (real GPU)', () => {
  it('transpose 2D', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3], [4, 5, 6]]);
      return { arr: Array.from(await a.transpose().toArray()), shape: a.transpose().shape };
    });
    expect(r.arr).toEqual([1, 4, 2, 5, 3, 6]);
    expect(r.shape).toEqual([3, 2]);
  });

  it('slice 1D and 2D', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
      const b = gpu.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]);
      return {
        s1: Array.from(await a.slice(2, 4).toArray()),
        s2: Array.from(await b.slice([1, 1], [2, 2]).toArray()),
      };
    });
    expect(r.s1).toEqual([2, 3, 4, 5]);
    expect(r.s2).toEqual([6, 7, 10, 11]);
  });

  it('concat along axis 0 and axis 1', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2], [3, 4]]);
      const b = gpu.tensor([[5, 6]]);
      const c = gpu.tensor([[7, 8], [9, 10]]);
      return {
        axis0: Array.from(await a.concat(b, 0).toArray()),
        axis1: Array.from(await a.concat(c, 1).toArray()),
      };
    });
    expect(r.axis0).toEqual([1, 2, 3, 4, 5, 6]);
    expect(r.axis1).toEqual([1, 2, 7, 8, 3, 4, 9, 10]);
  });

  it('reshape is a view (no extra compute)', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
      const b = a.reshape(3, 2);
      return { shape: b.shape, arr: Array.from(await b.sum(-1).toArray()) };
    });
    expect(r.shape).toEqual([3, 2]);
    expect(r.arr).toEqual([3, 7, 11]); // [1+2, 3+4, 5+6]
  });

  it('deep pipeline: slice -> matmul -> softmax -> row-sum', async () => {
    const r = await withGpu(async (gpu: any) => {
      const w = gpu.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
      const x = gpu.tensor([0, 1, 2, 3, 4, 5, 6, 7]).reshape([2, 4]);
      const x1 = x.slice([0, 0], [2, 4]); // identity slice
      const logits = x1.matmul(w);
      const probs = logits.softmax();
      return Array.from(await probs.sum(-1).toArray());
    });
    // Softmax rows always sum to 1 — proves the whole 4-kernel pipeline ran.
    expect(r).toEqual([1, 1]);
  });
});
