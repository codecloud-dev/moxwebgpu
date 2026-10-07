/**
 * Autograd shape-level tests — no GPU required.
 *
 * `backward()` only builds the lazy gradient graph; it never touches the
 * device until a gradient is materialized. So we can validate gradient
 * *shapes* (incl. broadcast reduction) in plain Node. Numerical correctness
 * is covered by tests/gpu/autograd.test.ts (finite differences, CI).
 */

/**
 * Autograd shape-level tests — no GPU required.
 *
 * `backward()` only builds the lazy gradient graph; it never touches the
 * device until a gradient is materialized. So we can validate gradient
 * *shapes* (incl. broadcast reduction) in plain Node. Numerical correctness
 * is covered by tests/gpu/autograd.test.ts (finite differences, CI).
 *
 * We install the op surface manually (instead of importing src/index.js)
 * because index.js pulls in core/buffer.ts, which references the browser
 * global `GPUBufferUsage` at module load.
 */

import { describe, it, expect } from 'vitest';
import { Tensor } from '../../src/tensor/tensor.js';
import { installTensorOps } from '../../src/tensor/ops/index.js';
import { installBackward } from '../../src/tensor/ops/backward.js';
import { backward } from '../../src/graph/autograd.js';

installTensorOps(Tensor.prototype);
installBackward();

const ctx: any = {};

function leaf(shape: number[], reqGrad = true): Tensor {
  const t = new Tensor(ctx, shape, 'f32', null, null);
  t.requiresGrad = reqGrad;
  return t;
}

describe('autograd (shape-level, no GPU)', () => {
  it('backward populates .grad on requiresGrad leaves', () => {
    const a = leaf([2, 3]);
    const b = leaf([2, 3]);
    const loss = a.add(b).sum();
    loss.backward();
    expect(a.grad).not.toBeNull();
    expect(a.grad!.shape).toEqual([2, 3]);
    expect(b.grad!.shape).toEqual([2, 3]);
  });

  it('matmul backward shapes', () => {
    const a = leaf([4, 5]);
    const b = leaf([5, 6]);
    const loss = a.matmul(b).sum();
    loss.backward();
    expect(a.grad!.shape).toEqual([4, 5]);
    expect(b.grad!.shape).toEqual([5, 6]);
  });

  it('broadcast binary backward reduces grad to each operand shape', () => {
    const a = leaf([4, 5]); // [M,K]
    const b = leaf([5]); // row-broadcast
    const loss = a.mul(b).sum();
    loss.backward();
    expect(a.grad!.shape).toEqual([4, 5]);
    expect(b.grad!.shape).toEqual([5]);
  });

  it('softmax backward shape preserved', () => {
    const a = leaf([3, 4]);
    const loss = a.softmax().sum();
    loss.backward();
    expect(a.grad!.shape).toEqual([3, 4]);
  });

  it('gradient only computed for requiresGrad leaves', () => {
    const a = leaf([2], false);
    const b = leaf([2]);
    const loss = a.add(b).sum();
    loss.backward();
    expect(a.grad).toBeNull();
    expect(b.grad).not.toBeNull();
  });

  it('grad is itself a lazy, differentiable tensor', () => {
    const a = leaf([2, 3]);
    const loss = a.mul(a).sum();
    loss.backward();
    expect(a.grad).not.toBeNull();
    expect(a.grad!.shape).toEqual([2, 3]);
  });
});
