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
      // 2 digits, not 3: the *numeric* side is a central finite difference
      // evaluated in f32, which carries ~1e-3 absolute noise at these
      // magnitudes (the analytic value is exact). Same tolerance the
      // matmul finite-difference test below uses.
      expect(analytic[i]).toBeCloseTo(numeric[i], 2);
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

  it('ReLU gradient is exactly 0 on the negative half-axis', async () => {
    // Regression: reluDef.backward must differentiate relu(x), i.e. use
    // sign(x), not sign(relu(x)). Using the output's sign makes every
    // negative input receive a spurious 0.5 gradient (dead ReLUs still get
    // updated during training).
    const { analytic } = await withGpu(async (ctx: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const x = Tensor.fromData(ctx, [-2, -1, 0.5, 2], { shape: [4] }).withGrad();
      backward(x.relu().sum());
      return { analytic: Array.from(await x.grad.toArray()) };
    });
    expect(analytic[0]).toBeCloseTo(0, 4); // x = -2: relu' = 0 (buggy impl gives 0.5)
    expect(analytic[1]).toBeCloseTo(0, 4); // x = -1
    expect(analytic[2]).toBeCloseTo(1, 4); // x = 0.5
    expect(analytic[3]).toBeCloseTo(1, 4); // x = 2
  });

  it('row-broadcast gradient reduces along axis 0 even for a square matrix', async () => {
    // Regression: sumTo() must reduce a row-broadcast operand ([M,K] op [K])
    // along the leading axis (axis 0). The old heuristic compared the first
    // dimension of grad with the target and, when M == K (square), wrongly
    // reduced along the last axis instead -> wrong grads for b.
    const { analytic, numeric } = await withGpu(async (ctx: any, arg: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const aBase = arg.a;
      const bBase = arg.b;
      const A = Tensor.fromData(ctx, aBase.slice(), { shape: [2, 2] });
      const b = Tensor.fromData(ctx, bBase.slice(), { shape: [2] }).withGrad();
      backward(A.mul(b).sum());
      const analytic = Array.from(await b.grad.toArray());
      const eps = 1e-3;
      const numeric: number[] = [];
      for (let j = 0; j < 2; j++) {
        const p = bBase.slice(); p[j] += eps;
        const m = bBase.slice(); m[j] -= eps;
        const plus = (await Tensor.fromData(ctx, aBase.slice(), { shape: [2, 2] })
          .mul(Tensor.fromData(ctx, p, { shape: [2] })).sum().toArray())[0];
        const minus = (await Tensor.fromData(ctx, aBase.slice(), { shape: [2, 2] })
          .mul(Tensor.fromData(ctx, m, { shape: [2] })).sum().toArray())[0];
        numeric.push((plus - minus) / (2 * eps));
      }
      return { analytic, numeric };
    }, { a: [1, 2, 3, 4], b: [5, 6] });
    // dL/db_j = sum_i A[i,j] => [A[0,0]+A[1,0], A[0,1]+A[1,1]] = [4, 6]
    expect(analytic[0]).toBeCloseTo(numeric[0], 2);
    expect(analytic[1]).toBeCloseTo(numeric[1], 2);
    expect(analytic[0]).toBeCloseTo(4, 2);
    expect(analytic[1]).toBeCloseTo(6, 2);
  });

  it('pow(x, scalar) has a registered gradient (dL/dx = c*x^(c-1))', async () => {
    // Regression: pow was listed as an autograd-capable binary op but had no
    // backward fn, so backward() silently produced null and training with a
    // power term lost the gradient entirely.
    const { analytic, numeric } = await withGpu(async (ctx: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const base = [1, 2, 3, 4];
      const x = Tensor.fromData(ctx, base.slice(), { shape: [4] }).withGrad();
      backward(x.pow(3).sum());
      const analytic = Array.from(await x.grad.toArray());
      const eps = 1e-3;
      const numeric: number[] = [];
      for (let i = 0; i < base.length; i++) {
        const p = base.slice(); p[i] += eps;
        const m = base.slice(); m[i] -= eps;
        const plus = (await Tensor.fromData(ctx, p, { shape: [4] }).pow(3).sum().toArray())[0];
        const minus = (await Tensor.fromData(ctx, m, { shape: [4] }).pow(3).sum().toArray())[0];
        numeric.push((plus - minus) / (2 * eps));
      }
      return { analytic, numeric };
    });
    for (let i = 0; i < analytic.length; i++) {
      expect(analytic[i]).toBeCloseTo(numeric[i], 1);
    }
  });

  it('pow(x, y) (tensor exponent) has registered gradients on both operands', async () => {
    const { gx, gy, nx, ny } = await withGpu(async (ctx: any) => {
      const M = (window as any).MoxWebGPU;
      const { Tensor, backward } = M;
      const xBase = [2, 3];
      const yBase = [2, 3];
      const x = Tensor.fromData(ctx, xBase.slice(), { shape: [2] }).withGrad();
      const y = Tensor.fromData(ctx, yBase.slice(), { shape: [2] }).withGrad();
      backward(x.pow(y).sum());
      const gx = Array.from(await x.grad.toArray());
      const gy = Array.from(await y.grad.toArray());
      const eps = 1e-3;
      const nx: number[] = [], ny: number[] = [];
      for (let i = 0; i < 2; i++) {
        const xp = xBase.slice(); xp[i] += eps; const xm = xBase.slice(); xm[i] -= eps;
        const yp = yBase.slice(); yp[i] += eps; const ym = yBase.slice(); ym[i] -= eps;
        const fxPlus = (await Tensor.fromData(ctx, xp, { shape: [2] }).pow(Tensor.fromData(ctx, yBase.slice(), { shape: [2] })).sum().toArray())[0];
        const fxMinus = (await Tensor.fromData(ctx, xm, { shape: [2] }).pow(Tensor.fromData(ctx, yBase.slice(), { shape: [2] })).sum().toArray())[0];
        const fyPlus = (await Tensor.fromData(ctx, xBase.slice(), { shape: [2] }).pow(Tensor.fromData(ctx, yp, { shape: [2] })).sum().toArray())[0];
        const fyMinus = (await Tensor.fromData(ctx, xBase.slice(), { shape: [2] }).pow(Tensor.fromData(ctx, ym, { shape: [2] })).sum().toArray())[0];
        nx.push((fxPlus - fxMinus) / (2 * eps));
        ny.push((fyPlus - fyMinus) / (2 * eps));
      }
      return { gx, gy, nx, ny };
    });
    for (let i = 0; i < 2; i++) {
      expect(gx[i]).toBeCloseTo(nx[i], 1);
      expect(gy[i]).toBeCloseTo(ny[i], 1);
    }
  });
});
