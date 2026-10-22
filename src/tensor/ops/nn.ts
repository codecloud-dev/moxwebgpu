/**
 * NN-flavoured composite ops: numerically stable softmax over the last axis.
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { numElements, type DType } from '../../core/dtype.js';
import { softmaxWgsl, encode2DUniform } from '../codegen.js';

export const softmaxDef: OpDef = {
  name: 'softmax',
  outShape: (shapes) => shapes[0],
  build: (shapes, dtypes) => {
    const dtype: DType = dtypes[0];
    // The shader seeds the row max from the f32 `identity` uniform slot and
    // accumulates in f32 — integer dtypes would produce a WGSL type error
    // (silent no-op dispatch). Same contract as mean: cast first.
    if (dtype !== 'f32') {
      throw new Error('moxwebgpu: softmax requires f32 input (cast first: t.toFloat())');
    }
    const shape = shapes[0];
    const rows = shape.length <= 1 ? 1 : numElements(shape) / shape[shape.length - 1];
    const cols = shape[shape.length - 1];
    const step: KernelStep = {
      key: `softmax:${dtype}`,
      wgsl: softmaxWgsl(dtype),
      bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
      uniforms: () => encode2DUniform(rows, cols, 0, -Infinity),
      workgroups: () => [rows, 1, 1],
    };
    return { steps: [step] };
  },
};
