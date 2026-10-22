/**
 * matmul: C[M,N] = A[M,K] x B[K,N], 16x16 workgroup tiling with
 * cooperative tile loading through workgroup shared memory.
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { encodeMatmulUniform, matmulWgsl } from '../codegen.js';

const TILE = 16;

export const matmulDef: OpDef = {
  name: 'matmul',
  outShape: (shapes) => {
    if (shapes[0].length !== 2) {
      throw new Error(`moxwebgpu: matmul left operand must be 2-D, got rank ${shapes[0].length}`);
    }
    const [m, k] = shapes[0];
    const b = shapes[1];
    // Matrix @ vector (numpy semantics): [M,K] @ [K] -> [M]. Without this,
    // a 1-D operand yields a shape containing `undefined` (NaN element
    // count) and the 16x16-tile kernel receives NaN workgroup counts.
    if (b.length === 1) {
      if (k !== b[0]) {
        throw new Error(`moxwebgpu: matmul shape mismatch (${shapes[0]} x ${shapes[1]}): inner dims ${k} vs ${b[0]}`);
      }
      return [m];
    }
    const [k2, n] = b;
    if (k !== k2) {
      throw new Error(`moxwebgpu: matmul shape mismatch (${shapes[0]} x ${shapes[1]}): inner dims ${k} vs ${k2}`);
    }
    return [m, n];
  },
  outDtype: (dtypes) => {
    if (dtypes[0] !== dtypes[1]) {
      throw new Error(`moxwebgpu: matmul requires matching dtypes, got ${dtypes[0]} vs ${dtypes[1]}`);
    }
    return dtypes[0];
  },
  build: (shapes, dtypes) => {
    const dtype = dtypes[0];
    const [m, k] = shapes[0];
    // A 1-D right operand is treated as a [K,1] column vector.
    const n = shapes[1].length === 1 ? 1 : shapes[1][1];
    const step: KernelStep = {
      key: `matmul:${dtype}`,
      wgsl: matmulWgsl(dtype),
      bindings: [
        { kind: 'uniform' },
        { kind: 'read', input: 0 },
        { kind: 'read', input: 1 },
        { kind: 'rw', output: true },
      ],
      uniforms: () => encodeMatmulUniform(m, n, k),
      // x covers output columns, y covers output rows.
      workgroups: () => [Math.ceil(n / TILE), Math.ceil(m / TILE), 1],
    };
    return { steps: [step] };
  },
};
