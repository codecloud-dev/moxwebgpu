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
    const [m, k] = shapes[0];
    const [k2, n] = shapes[1];
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
    const n = shapes[1][1];
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
