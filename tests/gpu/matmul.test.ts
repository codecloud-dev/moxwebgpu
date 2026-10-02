import { describe, it, expect, afterAll } from 'vitest';
import { withGpu, closeBrowser } from './harness.js';
import { cpuMatmul } from './ref.js';

afterAll(() => closeBrowser());

describe('matmul (real GPU, 16x16 tiling)', () => {
  it('small exact: [2,3] x [3,2]', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 2, 3], [4, 5, 6]]);
      const b = gpu.tensor([[7, 8], [9, 10], [11, 12]]);
      return Array.from(await a.matmul(b).toArray());
    });
    // [[58, 64], [139, 154]]
    expect(r).toEqual([58, 64, 139, 154]);
  });

  it('larger than one tile: 40x50 x 50x30 vs CPU', async () => {
    const M = 40, K = 50, N = 30;
    const ref = new Float32Array(M * K);
    const refB = new Float32Array(K * N);
    for (let i = 0; i < M * K; i++) ref[i] = ((i * 7) % 13) - 6;
    for (let i = 0; i < K * N; i++) refB[i] = ((i * 5) % 11) - 5;
    const expected = cpuMatmul(ref, refB, M, K, N);

    const got = await withGpu(async (gpu: any, arg: any) => {
      const a = gpu.tensor(arg.ref, { shape: [arg.M, arg.K] });
      const b = gpu.tensor(arg.refB, { shape: [arg.K, arg.N] });
      return Array.from(await a.matmul(b).toArray());
    }, { ref, refB, M, K, N });
    expect(got.length).toBe(M * N);
    for (let i = 0; i < M * N; i++) {
      expect(got[i]).toBeCloseTo(expected[i], 3);
    }
  });

  it('chained matmul: (A x B) x C', async () => {
    const r = await withGpu(async (gpu: any) => {
      const a = gpu.tensor([[1, 0], [0, 1]]);      // identity
      const b = gpu.tensor([[2, 3], [4, 5]]);
      const c = gpu.tensor([[1, 1], [0, 1]]);
      return Array.from(await b.matmul(c).toArray()); // (I x B) x C lazily
    });
    // [[2,3],[4,5]] x [[1,1],[0,1]] = [[2,5],[4,9]]
    expect(r).toEqual([2, 5, 4, 9]);
  });
});
