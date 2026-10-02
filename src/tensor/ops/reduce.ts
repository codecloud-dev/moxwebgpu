/**
 * Reduction ops: global (two-phase tree reduction) and per-axis
 * (one workgroup per row) variants.
 *
 * Identities (0, ±Infinity) are passed to the shaders through the uniform
 * `identity` slot — WGSL const-expressions may not produce ±Infinity, so
 * they cannot appear as literals in generated code.
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { numElements, type DType } from '../../core/dtype.js';
import {
  reducePhase1Wgsl,
  reducePhase2Wgsl,
  argReducePhase1Wgsl,
  argReducePhase2Wgsl,
  reduceLastAxisWgsl,
  encodeNUniform,
  encode2DUniform,
} from '../codegen.js';

const WG = 64;
const CHUNK = 8;

export interface ReduceSpec {
  /** JS number used as the initial accumulator (via uniforms.identity). */
  identityValue: number;
  combine: (acc: string, x: string) => string;
  /** Optional epilogue applied in phase 2 (e.g. mean divides by n). */
  epilogue?: (v: string) => string;
}

/** Global reduction to a single scalar: two-phase tree reduction. */
export function globalReduceOpDef(name: string, spec: ReduceSpec): OpDef {
  return {
    name,
    outShape: () => [1],
    outDtype: (dtypes) => dtypes[0],
    build: (shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      const n = numElements(shapes[0]);
      const nw1 = Math.max(1, Math.ceil(n / (WG * CHUNK)));
      const steps: KernelStep[] = [
        {
          key: `${name}-p1:${dtype}`,
          wgsl: reducePhase1Wgsl(dtype, spec.combine),
          bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', temp: true }],
          uniforms: () => encodeNUniform(n, 0, spec.identityValue),
          workgroups: () => [nw1, 1, 1],
          tempOutputElements: () => nw1,
        },
        {
          key: `${name}-p2:${dtype}`,
          wgsl: reducePhase2Wgsl(dtype, spec.combine, spec.epilogue),
          bindings: [{ kind: 'uniform' }, { kind: 'read', temp: true }, { kind: 'rw', output: true }],
          uniforms: () => encodeNUniform(nw1, spec.epilogue ? n : 0, spec.identityValue),
          workgroups: () => [1, 1, 1],
        },
      ];
      return { steps };
    },
  };
}

/** Global arg-reduction: index of the extreme value (first occurrence). */
export function argReduceOpDef(name: string, better: (candidate: string, best: string) => string): OpDef {
  return {
    name,
    outShape: () => [1],
    outDtype: () => 'u32',
    build: (shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      const n = numElements(shapes[0]);
      const nw1 = Math.max(1, Math.ceil(n / (WG * CHUNK)));
      const steps: KernelStep[] = [
        {
          key: `${name}-p1:${dtype}`,
          wgsl: argReducePhase1Wgsl(dtype, better),
          bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', temp: true }],
          uniforms: () => encodeNUniform(n),
          workgroups: () => [nw1, 1, 1],
          tempOutputElements: () => nw1,
          tempOutputDtype: 'u32',
        },
        {
          key: `${name}-p2:${dtype}`,
          wgsl: argReducePhase2Wgsl(dtype, better),
          bindings: [
            { kind: 'uniform' },
            { kind: 'read', temp: true },
            { kind: 'read', input: 0 },
            { kind: 'rw', output: true },
          ],
          uniforms: () => encodeNUniform(nw1),
          workgroups: () => [1, 1, 1],
        },
      ];
      return { steps };
    },
  };
}

/**
 * Reduce along the last axis: one workgroup per row, tree reduction inside.
 * [M, K] -> [M]. For a 1D input this degenerates to a [1] output.
 */
export function lastAxisReduceOpDef(name: string, spec: ReduceSpec): OpDef {
  return {
    name,
    outShape: (shapes) => (shapes[0].length <= 1 ? [1] : shapes[0].slice(0, -1)),
    outDtype: (dtypes) => dtypes[0],
    build: (shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      const shape = shapes[0];
      const rows = shape.length <= 1 ? 1 : numElements(shape) / shape[shape.length - 1];
      const cols = shape[shape.length - 1];
      const divideBy = spec.epilogue ? cols : 0;
      const step: KernelStep = {
        key: `${name}-axis:${dtype}`,
        wgsl: reduceLastAxisWgsl(dtype, spec.combine, spec.epilogue),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encode2DUniform(rows, cols, divideBy, spec.identityValue),
        workgroups: () => [rows, 1, 1],
      };
      return { steps: [step] };
    },
  };
}

/* ------------------------------------------------------------------ */
/* Concrete definitions                                                */
/* ------------------------------------------------------------------ */

export const sumDef = globalReduceOpDef('sum', {
  identityValue: 0,
  combine: (a, b) => `(${a} + ${b})`,
});
export const meanDef = globalReduceOpDef('mean', {
  identityValue: 0,
  combine: (a, b) => `(${a} + ${b})`,
  epilogue: (v) => `(${v}) / uniforms.scalar`,
});
export const maxReduceDef = globalReduceOpDef('max', {
  identityValue: -Infinity,
  combine: (a, b) => `max(${a}, ${b})`,
});
export const minReduceDef = globalReduceOpDef('min', {
  identityValue: Infinity,
  combine: (a, b) => `min(${a}, ${b})`,
});
export const argmaxDef = argReduceOpDef('argmax', (c, b) => `(${c} > ${b})`);
export const argminDef = argReduceOpDef('argmin', (c, b) => `(${c} < ${b})`);

export const sumAxisDef = lastAxisReduceOpDef('sum', {
  identityValue: 0,
  combine: (a, b) => `(${a} + ${b})`,
});
export const meanAxisDef = lastAxisReduceOpDef('mean', {
  identityValue: 0,
  combine: (a, b) => `(${a} + ${b})`,
  epilogue: (v) => `(${v}) / uniforms.scalar`,
});
export const maxAxisDef = lastAxisReduceOpDef('max', {
  identityValue: -Infinity,
  combine: (a, b) => `max(${a}, ${b})`,
});
export const minAxisDef = lastAxisReduceOpDef('min', {
  identityValue: Infinity,
  combine: (a, b) => `min(${a}, ${b})`,
});
