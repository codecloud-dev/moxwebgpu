/**
 * Installs all tensor ops onto the Tensor prototype: the unified, chainable
 * API surface of moxwebgpu. Every op is lazy — it returns a new Tensor backed
 * by a graph node; execution happens on toArray()/toBuffer().
 */

import type { OpDef } from '../../graph/lazy.js';
import { Tensor } from '../tensor.js';
import type { DType } from '../../core/dtype.js';
import {
  addDef, subDef, mulDef, divDef, powDef, minDef, maxDef,
  addScalarDef, subScalarDef, rsubScalarDef, mulScalarDef, divScalarDef,
  rdivScalarDef, powScalarDef, minScalarDef, maxScalarDef,
  negDef, absDef, expDef, logDef, sqrtDef, sinDef, cosDef, tanhDef,
  floorDef, ceilDef, reluDef, sigmoidDef, squareDef, signDef, castOpDef,
} from './elementwise.js';
import {
  sumDef, meanDef, maxReduceDef, minReduceDef, argmaxDef, argminDef,
  sumAxisDef, meanAxisDef, maxAxisDef, minAxisDef,
  sumAnyAxisDef, meanAnyAxisDef, maxAnyAxisDef, minAnyAxisDef,
  normAxis,
} from './reduce.js';
import { matmulDef } from './matmul.js';
import { transposeDef, sliceDef, concatDef } from './shape.js';
import { softmaxDef } from './nn.js';

function binaryMethod(def: OpDef, scalarDef: OpDef) {
  return function (this: Tensor, other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      return this.apply(scalarDef, [this], { scalar: other });
    }
    return this.apply(def, [this, other], {});
  };
}

function rbinaryMethod(def: OpDef, scalarDef: OpDef) {
  return function (this: Tensor, other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      // e.g. 1 - x  →  sub(scalar, x)
      return this.apply(scalarDef, [this], { scalar: other });
    }
    return other.apply(def, [other, this], {});
  };
}

function unaryMethod(def: OpDef) {
  return function (this: Tensor): Tensor {
    return this.apply(def, [this], {});
  };
}

/**
 * Axis-aware reduce methods:
 *   sum()      -> global scalar [1]
 *   sum(axis)  -> reduce along any axis (negative counts from the end);
 *                 the last axis uses the fast workgroup-tree kernel, other
 *                 axes use the generic any-axis kernel
 * For max/min the same method also accepts a Tensor to become elementwise:
 *   max(other) -> elementwise binary max
 */
function reduceMethod(
  globalDef: OpDef,
  axisDef: OpDef | null,
  genericAxisDef: OpDef | null,
  binaryDef?: OpDef,
) {
  return function (this: Tensor, other?: number | Tensor): Tensor {
    if (other === undefined) return this.apply(globalDef, [this], {});
    if (typeof other === 'number') {
      const rank = Math.max(1, this.ndim);
      const ax = normAxis(other, rank);
      if (ax === rank - 1 && axisDef) {
        return this.apply(axisDef, [this], {});
      }
      if (!genericAxisDef) {
        throw new Error(`moxwebgpu: axis ${other} reduce is not supported for this op (use no argument for a global reduce)`);
      }
      return this.apply(genericAxisDef, [this], { axis: ax });
    }
    if (binaryDef && other instanceof Tensor) {
      return this.apply(binaryDef, [this, other], {});
    }
    throw new Error('moxwebgpu: reduce methods accept no argument, an axis number, or (max/min) a Tensor');
  };
}

export function installTensorOps(proto: object): void {
  const p = proto as any;

  /* ---- elementwise binary (tensor or scalar operand) ----
   * NOTE: max/min elementwise forms live behind tensor.max(other)/tensor.min(other)
   * (installed in the reductions section below); scalar bounds go through clamp(). */
  p.add = binaryMethod(addDef, addScalarDef);
  p.sub = binaryMethod(subDef, subScalarDef);
  p.mul = binaryMethod(mulDef, mulScalarDef);
  p.div = binaryMethod(divDef, divScalarDef);
  p.pow = binaryMethod(powDef, powScalarDef);
  p.clamp = function (this: Tensor, lo: number, hi: number): Tensor {
    return this.apply(maxDef, [this.apply(minScalarDef, [this], { scalar: hi })], { scalar: lo });
  };

  /* ---- reversed scalar ops: 2.sub(x) etc. ---- */
  p.rsub = rbinaryMethod(subDef, rsubScalarDef);
  p.rdiv = rbinaryMethod(divDef, rdivScalarDef);

  /* ---- unary ---- */
  p.neg = unaryMethod(negDef);
  p.abs = unaryMethod(absDef);
  p.exp = unaryMethod(expDef);
  p.log = unaryMethod(logDef);
  p.sqrt = unaryMethod(sqrtDef);
  p.sin = unaryMethod(sinDef);
  p.cos = unaryMethod(cosDef);
  p.tanh = unaryMethod(tanhDef);
  p.floor = unaryMethod(floorDef);
  p.ceil = unaryMethod(ceilDef);
  p.relu = unaryMethod(reluDef);
  p.sigmoid = unaryMethod(sigmoidDef);
  p.square = unaryMethod(squareDef);
  p.sign = unaryMethod(signDef);

  /* ---- reductions (max/min: no arg = reduce; Tensor = elementwise) ---- */
  p.sum = reduceMethod(sumDef, sumAxisDef, sumAnyAxisDef);
  p.mean = reduceMethod(meanDef, meanAxisDef, meanAnyAxisDef);
  p.max = reduceMethod(maxReduceDef, maxAxisDef, maxAnyAxisDef, maxDef);
  p.min = reduceMethod(minReduceDef, minAxisDef, minAnyAxisDef, minDef);
  p.maxReduce = p.max; // explicit aliases
  p.minReduce = p.min;
  p.argmax = reduceMethod(argmaxDef, null, null);
  p.argmin = reduceMethod(argminDef, null, null);

  /* ---- matmul & shape ---- */
  p.matmul = function (this: Tensor, other: Tensor): Tensor {
    return this.apply(matmulDef, [this, other], {});
  };
  p.transpose = function (this: Tensor): Tensor {
    return this.apply(transposeDef, [this], {});
  };
  p.slice = function (this: Tensor, start: number | number[], size: number | number[]): Tensor {
    const ndim = this.ndim;
    let s = typeof start === 'number' ? [start] : start.slice();
    let z = typeof size === 'number' ? [size] : size.slice();
    while (s.length < ndim) s.push(0);
    while (z.length < ndim) z.push(this.shape[z.length]);
    for (let d = 0; d < ndim; d++) {
      if (s[d] < 0 || z[d] < 0 || s[d] + z[d] > this.shape[d]) {
        throw new Error(`moxwebgpu: slice range [${s}, ${z}] out of bounds for shape [${this.shape}]`);
      }
    }
    return this.apply(sliceDef, [this], { start: s, size: z });
  };
  p.concat = function (this: Tensor, other: Tensor, axis = 0): Tensor {
    return this.apply(concatDef, [this, other], { axis });
  };

  /* ---- nn ---- */
  p.softmax = unaryMethod(softmaxDef);

  /* ---- dtype conversion ---- */
  p.cast = function (this: Tensor, dtype: DType): Tensor {
    if (dtype === this.dtype) return this;
    return this.apply(castOpDef(dtype), [this], {});
  };
  p.toFloat = function (this: Tensor): Tensor {
    return this.cast('f32');
  };
}
