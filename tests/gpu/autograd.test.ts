/**
 * Autograd numerical tests — run on GPU (SwiftShader) via the shared harness.
 *
 * Validates reverse-mode gradients against central finite differences. This
 * exercises the broadcast-reduction path (sumTo) and the gradient-replication
 * path (expand) used by matmul / sum / softmax backward — the parts a
 * shape-only test cannot cover.
 */

import { describe, it, expect } from 'vitest';
import { withGpu } from './harness.js';

describe('autograd (numerical, GPU/SwiftShader)', () => {
  it('L = sum(x^2)  =>  dL/dx = 2x', async () => {
    const { analytic, numeric } = await withGpu(async (ctx: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const base = [1, 2, 3, 4];
      const x = Tensor.fromData(ctx, base.slice(), { shape: [4] }).withGrad();
      backward(x.mul(x).sum());
      const analytic = Array.from(await x.grad.toArray());
      const eps = 1e-3;
      const numeric: number[] = [];
      for (let i = 0; i < base.length; i++) {
        const p = base.slice(); p[i] += eps;
        const m = base.slice(); m[i] -= eps;
        const plus = (await Tensor.fromData(ctx, p, { shape: [4] }).mul(Tensor.fromData(ctx, p, { shape: [4] })).sum().toArray())[0];
        const minus = (await Tensor.fromData(ctx, m, { shape: [4] }).mul(Tensor.fromData(ctx, m, { shape: [4] })).sum().toArray())[0];
        numeric.push((plus - minus) / (2 * eps));
      }
      return { analytic, numeric };
    });
    for (let i = 0; i < analytic.length; i++) {
      expect(analytic[i]).toBeCloseTo(numeric[i], 3);
    }
  });

  it('L = sum(W @ x)  =>  dL/dW[j,k] = x[k]', async () => {
    const { gW, numericW } = await withGpu(async (ctx: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const baseW = [1, 2, 3, 4];
      const x = Tensor.fromData(ctx, [5, 6], { shape: [2] });
      const W = Tensor.fromData(ctx, baseW.slice(), { shape: [2, 2] }).withGrad();
      backward(W.matmul(x).sum());
      const gW = Array.from(await W.grad.toArray());
      const eps = 1e-3;
      const numericW: number[] = [];
      for (let i = 0; i < 4; i++) {
        const p = baseW.slice(); p[i] += eps;
        const m = baseW.slice(); m[i] -= eps;
        const plus = (await Tensor.fromData(ctx, p, { shape: [2, 2] }).matmul(x).sum().toArray())[0];
        const minus = (await Tensor.fromData(ctx, m, { shape: [2, 2] }).matmul(x).sum().toArray())[0];
        numericW.push((plus - minus) / (2 * eps));
      }
      return { gW, numericW };
    });
    // expected gradient: each column k equals x[k] => [[5,6],[5,6]]
    for (let i = 0; i < gW.length; i++) {
      expect(gW[i]).toBeCloseTo(numericW[i], 2);
    }
  });
});
