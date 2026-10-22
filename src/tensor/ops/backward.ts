/**
 * Attaches reverse-mode `backward` functions to the built-in OpDefs.
 *
 * Called once at module load (from index.ts) after `installTensorOps()`.
 * Every formula returns lazy Tensors, so the gradient graph composes with the
 * forward graph and only touches the GPU when a gradient is read.
 */

import type { Tensor } from '../tensor.js';
import {
  addDef, subDef, mulDef, divDef,
  addScalarDef, subScalarDef, rsubScalarDef,
  mulScalarDef, divScalarDef, rdivScalarDef,
  negDef, absDef, expDef, logDef, sqrtDef, squareDef, reluDef, sigmoidDef, tanhDef,
} from './elementwise.js';
import { sumDef, meanDef } from './reduce.js';
import { matmulDef } from './matmul.js';
import { softmaxDef } from './nn.js';
import { sumTo } from '../../graph/autograd.js';

/** Binary op gradient with broadcast-correct reduction to each operand. */
function binary(
  d0: (a: Tensor, b: Tensor, g: Tensor) => Tensor,
  d1: (a: Tensor, b: Tensor, g: Tensor) => Tensor,
) {
  return (inputs: Tensor[], _out: Tensor, g: Tensor) => {
    const [a, b] = inputs;
    return [sumTo(d0(a, b, g), a.shape), sumTo(d1(a, b, g), b.shape)];
  };
}

export function installBackward(): void {
  // ---- elementwise binary ----
  addDef.backward = binary((_a, _b, g) => g, (_a, _b, g) => g);
  subDef.backward = binary((_a, _b, g) => g, (_a, _b, g) => g.neg());
  mulDef.backward = binary((_a, b, g) => g.mul(b), (a, _b, g) => g.mul(a));
  divDef.backward = binary(
    (_a, b, g) => g.div(b),
    (a, b, g) => g.mul(a).div(b.mul(b)).neg(),
  );

  // ---- scalar variants ----
  addScalarDef.backward = (_i, _o, g) => [g];
  subScalarDef.backward = (_i, _o, g) => [g];
  // rsub: c - x  (scalar c, tensor x) -> d/dx = -g
  rsubScalarDef.backward = (_i, _o, g) => [g.neg()];
  mulScalarDef.backward = (_i, _o, g, attrs) => [g.mul(attrs.scalar as number)];
  divScalarDef.backward = (_i, _o, g, attrs) => [g.div(attrs.scalar as number)];
  // rdiv: c / x -> d/dx = -c g / x^2
  rdivScalarDef.backward = (_i, x, g, attrs) => [g.mul(attrs.scalar as number).div(x.mul(x)).neg()];

  // ---- unary ----
  negDef.backward = (_i, _o, g) => [g.neg()];
  absDef.backward = (inputs, _o, g) => [g.mul(inputs[0].sign())];
  expDef.backward = (_i, o, g) => [g.mul(o)];
  logDef.backward = (inputs, _o, g) => [g.div(inputs[0])];
  sqrtDef.backward = (_i, o, g) => [g.div(o.mul(2))];
  squareDef.backward = (inputs, _o, g) => [g.mul(inputs[0].mul(2))];
  // relu'(x) = step(x); approximated as (sign(x)+1)/2 (0.5 exactly at 0, harmless)
  reluDef.backward = (_i, o, g) => [g.mul(o.sign().add(1).div(2))];
  sigmoidDef.backward = (_i, o, g) => [g.mul(o.mul(o.neg().add(1)))];
  tanhDef.backward = (_i, o, g) => [g.mul(o.mul(o).neg().add(1))];

  // ---- reductions ----
  sumDef.backward = (inputs, _o, g) => [g.expand(inputs[0].shape)];
  meanDef.backward = (inputs, o, g) => {
    const n = inputs[0].size / Math.max(1, o.size);
    return [g.expand(inputs[0].shape).div(n)];
  };

  // ---- matmul ----
  matmulDef.backward = (inputs, o, g) => {
    const [a, b] = inputs;
    if (b.ndim === 1) {
      // W[M,K] @ x[K] -> out [M]: dL/dW = g ⊗ x (outer product), dL/dx = W^T g.
      // Built from 2-D matmuls via zero-copy reshapes (transpose is 2-D only).
      const m = o.shape[0];
      const gCol = g.reshape(m, 1);
      return [
        gCol.matmul(b.reshape(1, b.shape[0])),
        a.transpose().matmul(gCol).reshape(...b.shape),
      ];
    }
    return [g.matmul(b.transpose()), a.transpose().matmul(g)];
  };

  // ---- softmax (numerically stable, over the last axis) ----
  // dx = y * (g - sum(g*y, last axis)): the correction term must be the
  // per-row sum, NOT an elementwise product (sumTo with the full shape is
  // a no-op and silently yields a wrong gradient).
  softmaxDef.backward = (inputs, o, g) => {
    const y = o;
    const rowShape = y.ndim >= 2 ? y.shape.slice(0, -1) : [1];
    let s = sumTo(g.mul(y), rowShape);
    // A single-row (or 1-D) softmax has a scalar correction term; replicate
    // it so the subtraction is a plain elementwise op.
    if (s.size === 1) s = s.expand(y.shape);
    return [g.sub(s).mul(y)];
  };
}
