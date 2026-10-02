/**
 * Elementwise ops: unary, binary (with broadcast), and scalar variants.
 *
 * Broadcast rules (NumPy-flavoured subset, linearized for GPU):
 *  - same shape              -> plain elementwise
 *  - b.size == last dim of a -> row broadcast   ([M,K] op [K])
 *  - b.size == first dim of a (a.ndim >= 2)     -> column broadcast ([M,K] op [M])
 *  - scalar                  -> fused via uniform
 */

import type { OpDef, KernelStep } from '../../graph/lazy.js';
import { numElements, type DType } from '../../core/dtype.js';
import {
  binary1DWgsl,
  binary2DWgsl,
  binaryScalarWgsl,
  unaryWgsl,
  castWgsl,
  encodeNUniform,
  encode2DUniform,
  type BinaryExpr,
  type UnaryExpr,
  type BroadcastMode,
} from '../codegen.js';

const WG = 64;

/** Pick the broadcast mode / plain path for a binary op pair. */
export function broadcastMode(aShape: number[], bShape: number[]): BroadcastMode | 'plain' {
  if (aShape.length === bShape.length && aShape.every((d, i) => d === bShape[i])) return 'plain';
  const aSize = numElements(aShape);
  const bSize = numElements(bShape);
  const last = aShape[aShape.length - 1];
  if (aSize % last === 0 && bSize === last) return 'row';
  const first = aShape[0];
  if (aShape.length >= 2 && aSize % first === 0 && bSize === first) return 'col';
  throw new Error(
    `moxwebgpu: cannot broadcast shapes [${bShape}] against [${aShape}] (plain, row or col broadcast supported)`,
  );
}

/** Build a binary (tensor op tensor) op definition. */
export function binaryOpDef(name: string, expr: BinaryExpr): OpDef {
  return {
    name,
    outShape: (shapes) => {
      broadcastMode(shapes[0], shapes[1]); // validate early
      return shapes[0];
    },
    outDtype: (dtypes) => {
      if (dtypes[0] !== dtypes[1]) {
        throw new Error(`moxwebgpu: ${name} requires matching dtypes, got ${dtypes[0]} vs ${dtypes[1]}`);
      }
      return dtypes[0];
    },
    build: (shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      const aShape = shapes[0];
      const n = numElements(aShape);
      const mode = broadcastMode(aShape, shapes[1]);
      const wgsl =
        mode === 'plain'
          ? binary1DWgsl(dtype, expr)
          : binary2DWgsl(dtype, expr, mode as BroadcastMode);
      const step: KernelStep = {
        key: `${name}:${dtype}:${mode}`,
        wgsl,
        bindings: [
          { kind: 'uniform' },
          { kind: 'read', input: 0 },
          { kind: 'read', input: 1 },
          { kind: 'rw', output: true },
        ],
        uniforms: () =>
          mode === 'plain' ? encodeNUniform(n) : encode2DUniform(rowsOf(aShape, mode), colsOf(aShape, mode)),
        workgroups: () => [Math.ceil(n / WG), 1, 1],
      };
      return { steps: [step] };
    },
  };
}

/** Build a (tensor op scalar) op definition. */
export function scalarOpDef(name: string, expr: BinaryExpr): OpDef {
  return {
    name,
    outShape: (shapes) => shapes[0],
    build: (_shapes, dtypes, attrs) => {
      const dtype: DType = dtypes[0];
      const n = numElements(_shapes[0]);
      const step: KernelStep = {
        key: `${name}-scalar:${dtype}`,
        wgsl: binaryScalarWgsl(dtype, expr),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encodeNUniform(n, attrs.scalar as number),
        workgroups: () => [Math.ceil(n / WG), 1, 1],
      };
      return { steps: [step] };
    },
  };
}

/** Build a unary op definition. */
export function unaryOpDef(name: string, expr: UnaryExpr): OpDef {
  return {
    name,
    outShape: (shapes) => shapes[0],
    build: (_shapes, dtypes) => {
      const dtype: DType = dtypes[0];
      const n = numElements(_shapes[0]);
      const step: KernelStep = {
        key: `${name}:${dtype}`,
        wgsl: unaryWgsl(dtype, expr),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encodeNUniform(n),
        workgroups: () => [Math.ceil(n / WG), 1, 1],
      };
      return { steps: [step] };
    },
  };
}

function rowsOf(shape: number[], mode: BroadcastMode): number {
  return mode === 'row' ? numElements(shape) / shape[shape.length - 1] : shape[0];
}
function colsOf(shape: number[], mode: BroadcastMode): number {
  return mode === 'row' ? shape[shape.length - 1] : numElements(shape) / shape[0];
}

/* ------------------------------------------------------------------ */
/* Concrete op definitions                                             */
/* ------------------------------------------------------------------ */

export const addDef = binaryOpDef('add', (a, b) => `(${a} + ${b})`);
export const subDef = binaryOpDef('sub', (a, b) => `(${a} - ${b})`);
export const mulDef = binaryOpDef('mul', (a, b) => `(${a} * ${b})`);
export const divDef = binaryOpDef('div', (a, b) => `(${a} / ${b})`);
export const powDef = binaryOpDef('pow', (a, b) => `pow(${a}, ${b})`);
export const minDef = binaryOpDef('min', (a, b) => `min(${a}, ${b})`);
export const maxDef = binaryOpDef('max', (a, b) => `max(${a}, ${b})`);

export const addScalarDef = scalarOpDef('add', (a, b) => `(${a} + ${b})`);
export const subScalarDef = scalarOpDef('sub', (a, b) => `(${a} - ${b})`);
export const rsubScalarDef = scalarOpDef('sub', (a, b) => `(${b} - ${a})`);
export const mulScalarDef = scalarOpDef('mul', (a, b) => `(${a} * ${b})`);
export const divScalarDef = scalarOpDef('div', (a, b) => `(${a} / ${b})`);
export const rdivScalarDef = scalarOpDef('div', (a, b) => `(${b} / ${a})`);
export const powScalarDef = scalarOpDef('pow', (a, b) => `pow(${a}, ${b})`);
export const minScalarDef = scalarOpDef('min', (a, b) => `min(${a}, ${b})`);
export const maxScalarDef = scalarOpDef('max', (a, b) => `max(${a}, ${b})`);

export const negDef = unaryOpDef('neg', (a) => `(-${a})`);
export const absDef = unaryOpDef('abs', (a) => `abs(${a})`);
export const expDef = unaryOpDef('exp', (a) => `exp(${a})`);
export const logDef = unaryOpDef('log', (a) => `log(${a})`);
export const sqrtDef = unaryOpDef('sqrt', (a) => `sqrt(${a})`);
export const sinDef = unaryOpDef('sin', (a) => `sin(${a})`);
export const cosDef = unaryOpDef('cos', (a) => `cos(${a})`);
export const tanhDef = unaryOpDef('tanh', (a) => `tanh(${a})`);
export const floorDef = unaryOpDef('floor', (a) => `floor(${a})`);
export const ceilDef = unaryOpDef('ceil', (a) => `ceil(${a})`);
export const reluDef = unaryOpDef('relu', (a) => `max(${a}, 0.0)`);
export const sigmoidDef = unaryOpDef('sigmoid', (a) => `(1.0 / (1.0 + exp(-(${a}))))`);
export const squareDef = unaryOpDef('square', (a) => `(${a} * ${a})`);
export const signDef = unaryOpDef('sign', (a) => `sign(${a})`);

/* ------------------------------------------------------------------ */
/* Dtype conversion                                                    */
/* ------------------------------------------------------------------ */

/** Build a cast op definition toward `target`. */
export function castOpDef(target: DType): OpDef {
  return {
    name: 'cast',
    outShape: (shapes) => shapes[0],
    outDtype: () => target,
    build: (_shapes, dtypes) => {
      const from: DType = dtypes[0];
      const n = numElements(_shapes[0]);
      const step: KernelStep = {
        key: `cast:${from}->${target}`,
        wgsl: castWgsl(from, target),
        bindings: [{ kind: 'uniform' }, { kind: 'read', input: 0 }, { kind: 'rw', output: true }],
        uniforms: () => encodeNUniform(n),
        workgroups: () => [Math.ceil(n / WG), 1, 1],
      };
      return { steps: [step] };
    },
  };
}
