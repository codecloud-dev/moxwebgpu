/**
 * Reduction ops: global (two-phase tree reduction), per-axis (one
 * workgroup per row) and generic any-axis (one thread per output)
 * variants.
 *
 * Identities (0, ±Infinity, INT_MIN/MAX) are passed to the shaders through
 * the uniform `identity` slot — WGSL const-expressions may not produce
 * ±Infinity, so they cannot appear as literals in generated code. Int
 * dtypes convert the f32 slot (saturating) inside the shader.
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { numElements, type DType } from '../../core/dtype.js';
import {
  reducePhase1Wgsl,
  reducePhase2Wgsl,
  argReducePhase1Wgsl,
  argReducePhase2Wgsl,
  reduceLastAxisWgsl,
  axisReduceWgsl,
  encodeNUniform,
  encode2DUniform,
  encodeAxisUniform,
} from '../codegen.js';

const WG = 64;
const CHUNK = 8;

/** Typed identity value: a constant, or per-dtype (int boundaries differ). */
export type IdentityValue = number | ((dtype: DType) => number);

/** i32/u32 saturation boundaries (the shader-side conversion matches). */
export const INT32_MIN = -2147483648;
export const INT32_MAX = 2147483647;
export const UINT32_MAX = 4294967295;

function identityOf(spec: { identityValue: IdentityValue }, dtype: DType): number {
  return typeof spec.identityValue === 'function' ? spec.identityValue(dtype) : spec.identityValue;
}

export interface ReduceSpec {
  /** JS number(s) used as the initial accumulator (via uniforms.identity). */
  identityValue: IdentityValue;
  combine: (acc: string, x: string) => string;
  /** Optional epilogue applied in phase 2 (e.g. mean divides by n). */
  epilogue?: (v: string) => string;
  /** Set for ops (mean) whose epilogue only compiles for f32. */
  floatOnly?: boolean;
}

function checkFloatOnly(spec: ReduceSpec, dtype: DType, op: string): void {
  if (spec.floatOnly && dtype !== 'f32') {
    throw new Error(`moxwebgpu: ${op} requires f32 input (cast first: t.toFloat())`);
  }
}

/** Global reduction to a single scalar: two-phase tree reduction. */
export function globalReduceOpDef(name: string, spec: ReduceSpec): OpDef {
  return {
    name,
    outShape: () => [1],
    outDtype: (dtypes) => dtypes[0],
    build: (shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      checkFloatOnly(spec, dtype, name);
      const n = numElements(shapes[0]);
      const nw1 = Math.max(1, Math.ceil(n / (WG * CHUNK)));
      const steps: KernelStep[] = [
        {
          key: `${name}-p1:${dtype}`,
          wgsl: reducePhase1Wgsl(dtype, spec.combine),
          bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', temp: true }],
          uniforms: () => encodeNUniform(n, 0, identityOf(spec, dtype)),
          workgroups: () => [nw1, 1, 1],
          tempOutputElements: () => nw1,
        },
        {
          key: `${name}-p2:${dtype}`,
          wgsl: reducePhase2Wgsl(dtype, spec.combine, spec.epilogue),
          bindings: [{ kind: 'uniform' }, { kind: 'read', temp: true }, { kind: 'rw', output: true }],
          uniforms: () => encodeNUniform(nw1, spec.epilogue ? n : 0, identityOf(spec, dtype)),
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
      checkFloatOnly(spec, dtype, name);
      const shape = shapes[0];
      const rows = shape.length <= 1 ? 1 : numElements(shape) / shape[shape.length - 1];
      const cols = shape[shape.length - 1];
      const divideBy = spec.epilogue ? cols : 0;
      const step: KernelStep = {
        key: `${name}-axis:${dtype}`,
        wgsl: reduceLastAxisWgsl(dtype, spec.combine, spec.epilogue),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encode2DUniform(rows, cols, divideBy, identityOf(spec, dtype)),
        workgroups: () => [rows, 1, 1],
      };
      return { steps: [step] };
    },
  };
}

/** Normalize an axis number (negative allowed) into [0, rank). */
export function normAxis(axis: number, rank: number): number {
  const r = Math.max(1, rank);
  const a = Math.trunc(axis);
  return ((a % r) + r) % r;
}

/**
 * Generic reduction along ANY axis of ANY rank (attrs.axis selects it).
 * One thread per output element, serially folding along the axis.
 * Dispatch is along x only (65535 workgroup cap → ~4M output elements);
 * larger outputs can split the grid in a follow-up.
 */
export function axisReduceOpDef(name: string, spec: ReduceSpec): OpDef {
  return {
    name,
    outShape: (shapes, attrs) => {
      const s = shapes[0];
      if (s.length <= 1) return [1];
      const ax = normAxis(attrs.axis, s.length);
      return s.filter((_, d) => d !== ax);
    },
    outDtype: (dtypes) => dtypes[0],
    build: (shapes, dtypes, attrs) => {
      const dtype: DType = dtypes[0];
      checkFloatOnly(spec, dtype, name);
      const shape = shapes[0];
      const ax = normAxis(attrs.axis, shape.length);
      let outer = 1;
      let inner = 1;
      for (let d = 0; d < ax; d++) outer *= shape[d];
      for (let d = ax + 1; d < shape.length; d++) inner *= shape[d];
      const dim = shape[ax];
      const nOut = outer * inner;
      const divideBy = spec.epilogue ? dim : 0;
      const step: KernelStep = {
        key: `${name}-gaxis:${dtype}`,
        wgsl: axisReduceWgsl(dtype, spec.combine, spec.epilogue),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encodeAxisUniform(nOut, inner, dim, divideBy, identityOf(spec, dtype)),
        workgroups: () => [Math.max(1, Math.ceil(nOut / WG)), 1, 1],
      };
      return { steps: [step] };
    },
  };
}

/* ------------------------------------------------------------------ */
/* Concrete definitions                                                */
/* ------------------------------------------------------------------ */

const sumCombine = (a: string, b: string) => `(${a} + ${b})`;
const maxCombine = (a: string, b: string) => `max(${a}, ${b})`;
const minCombine = (a: string, b: string) => `min(${a}, ${b})`;
const meanEpilogue = (v: string) => `(${v}) / uniforms.scalar`;

export const sumDef = globalReduceOpDef('sum', { identityValue: 0, combine: sumCombine });
export const meanDef = globalReduceOpDef('mean', {
  identityValue: 0,
  combine: sumCombine,
  epilogue: meanEpilogue,
  floatOnly: true,
});
export const maxReduceDef = globalReduceOpDef('max', {
  identityValue: (dt) => (dt === 'f32' ? -Infinity : dt === 'i32' ? INT32_MIN : 0),
  combine: maxCombine,
});
export const minReduceDef = globalReduceOpDef('min', {
  identityValue: (dt) => (dt === 'f32' ? Infinity : dt === 'i32' ? INT32_MAX : UINT32_MAX),
  combine: minCombine,
});
export const argmaxDef = argReduceOpDef('argmax', (c, b) => `(${c} > ${b})`);
export const argminDef = argReduceOpDef('argmin', (c, b) => `(${c} < ${b})`);

export const sumAxisDef = lastAxisReduceOpDef('sum', { identityValue: 0, combine: sumCombine });
export const meanAxisDef = lastAxisReduceOpDef('mean', {
  identityValue: 0,
  combine: sumCombine,
  epilogue: meanEpilogue,
  floatOnly: true,
});
export const maxAxisDef = lastAxisReduceOpDef('max', {
  identityValue: (dt) => (dt === 'f32' ? -Infinity : dt === 'i32' ? INT32_MIN : 0),
  combine: maxCombine,
});
export const minAxisDef = lastAxisReduceOpDef('min', {
  identityValue: (dt) => (dt === 'f32' ? Infinity : dt === 'i32' ? INT32_MAX : UINT32_MAX),
  combine: minCombine,
});

export const sumAnyAxisDef = axisReduceOpDef('sum', { identityValue: 0, combine: sumCombine });
export const meanAnyAxisDef = axisReduceOpDef('mean', {
  identityValue: 0,
  combine: sumCombine,
  epilogue: meanEpilogue,
  floatOnly: true,
});
export const maxAnyAxisDef = axisReduceOpDef('max', {
  identityValue: (dt) => (dt === 'f32' ? -Infinity : dt === 'i32' ? INT32_MIN : 0),
  combine: maxCombine,
});
export const minAnyAxisDef = axisReduceOpDef('min', {
  identityValue: (dt) => (dt === 'f32' ? Infinity : dt === 'i32' ? INT32_MAX : UINT32_MAX),
  combine: minCombine,
});
