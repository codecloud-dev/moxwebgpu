/**
 * Shape ops: transpose (2D), slice (<=4D), concat (along any axis).
 * Reshape is a zero-copy view handled directly on the Tensor class.
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { numElements, type DType } from '../../core/dtype.js';
import {
  transpose2DWgsl,
  copyWgsl,
  encode2DUniform,
  encodeCopyUniform,
  contiguousStrides,
} from '../codegen.js';

const COPY_WG = 64;
const TILE = 16;

/* ------------------------------------------------------------------ */
/* Transpose (2D)                                                      */
/* ------------------------------------------------------------------ */

export const transposeDef: OpDef = {
  name: 'transpose',
  outShape: (shapes) => {
    const s = shapes[0];
    if (s.length !== 2) {
      throw new Error(`moxwebgpu: transpose currently supports 2D tensors, got rank ${s.length}`);
    }
    return [s[1], s[0]];
  },
  build: (shapes, dtypes) => {
    const dtype: DType = dtypes[0];
    const [rows, cols] = shapes[0];
    const step: KernelStep = {
      key: `transpose:${dtype}`,
      wgsl: transpose2DWgsl(dtype),
      bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
      uniforms: () => encode2DUniform(rows, cols),
      workgroups: () => [Math.ceil(cols / TILE), Math.ceil(rows / TILE), 1],
    };
    return { steps: [step] };
  },
};

/* ------------------------------------------------------------------ */
/* Slice (<=4D): extract a contiguous block                            */
/* ------------------------------------------------------------------ */

export interface SliceAttrs {
  start: number[];
  size: number[];
}

export const sliceDef: OpDef = {
  name: 'slice',
  outShape: (_shapes, attrs) => attrs.size as number[],
  build: (shapes, dtypes, attrs) => {
    const dtype: DType = dtypes[0];
    const { start, size } = attrs as SliceAttrs;
    const inShape = shapes[0];
    const inStrides = contiguousStrides(inShape);
    const inOffset = start.reduce((acc, s, d) => acc + s * inStrides[d], 0);
    const outStrides = contiguousStrides(size);
    const step: KernelStep = {
      key: `copy:${dtype}`,
      wgsl: copyWgsl(dtype),
      bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
      uniforms: () => encodeCopyUniform(size, inStrides, outStrides, inOffset, 0),
      workgroups: () => [Math.ceil(numElements(size) / COPY_WG), 1, 1],
    };
    return { steps: [step] };
  },
};

/* ------------------------------------------------------------------ */
/* Concat (two inputs, any axis)                                       */
/* ------------------------------------------------------------------ */

export const concatDef: OpDef = {
  name: 'concat',
  outShape: (shapes, attrs) => {
    const a = shapes[0];
    const b = shapes[1];
    if (a.length !== b.length) {
      throw new Error(`moxwebgpu: concat requires matching ranks, got ${a.length} vs ${b.length}`);
    }
    const axis = attrs.axis as number;
    for (let d = 0; d < a.length; d++) {
      if (d !== axis && a[d] !== b[d]) {
        throw new Error(`moxwebgpu: concat shape mismatch on non-axis dim ${d}: ${a[d]} vs ${b[d]}`);
      }
    }
    const out = a.slice();
    out[axis] = a[axis] + b[axis];
    return out;
  },
  build: (shapes, dtypes, attrs) => {
    const dtype: DType = dtypes[0];
    const axis = attrs.axis as number;
    const outShape = this_outShape(shapes, axis);
    const globalStrides = contiguousStrides(outShape);

    let cum = 0;
    const steps: KernelStep[] = shapes.map((chunkShape, k) => {
      const chunkStrides = contiguousStrides(chunkShape);
      // For the second chunk the input is dense (contiguous), and the
      // destination is strided: coords decode against the chunk shape but
      // walk the global output strides, offset along the concat axis.
      const outOffset = cum * globalStrides[axis];
      cum += chunkShape[axis];
      return {
        key: `copy:${dtype}`,
        wgsl: copyWgsl(dtype),
        bindings: [
          { kind: 'uniform' },
          { kind: 'read', input: k },
          { kind: 'rw', output: true },
        ],
        uniforms: () => encodeCopyUniform(chunkShape, chunkStrides, globalStrides, 0, outOffset),
        workgroups: () => [Math.ceil(numElements(chunkShape) / COPY_WG), 1, 1],
      };
    });
    return { steps };
  },
};

function this_outShape(shapes: number[][], axis: number): number[] {
  const out = shapes[0].slice();
  out[axis] = shapes.reduce((acc, s) => acc + s[axis], 0);
  return out;
}
