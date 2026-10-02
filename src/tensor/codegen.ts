/**
 * WGSL code generation templates.
 *
 * Every tensor op is materialized as one or two compute shaders generated
 * from these templates. Kernels take their dimensions through a small
 * uniform buffer, which keeps generated pipelines shape-independent and
 * therefore cache-friendly.
 *
 * Layout conventions:
 *  - data is row-major
 *  - binding 0 is always the op uniform block (dims / scalars)
 *  - remaining bindings are (input0, input1?, output)
 */

import type { DType } from '../core/dtype.js';
import { dtypeInfo } from '../core/dtype.js';

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

/** Format a JS number as a WGSL float literal (always with a decimal point).
 *  NOTE: ±Infinity/NaN cannot appear in WGSL const-expressions (they are
 *  shader-creation errors); route them through a uniform slot instead
 *  (see `OpUniforms.identity` and the reduce op definitions). */
export function fmtF32(v: number): string {
  if (Number.isNaN(v) || v === Infinity || v === -Infinity) {
    throw new Error(
      'moxwebgpu: NaN/±Infinity cannot be a WGSL const-expression literal — pass it through a uniform (OpUniforms.identity) instead',
    );
  }
  if (Number.isInteger(v) && Math.abs(v) < 1e15) return `${v}.0`;
  return `${v}`;
}

/** Format a JS number as a WGSL signed/unsigned int literal. */
export function fmtInt(v: number): string {
  return `${Math.trunc(v)}u`;
}

/* ------------------------------------------------------------------ */
/* Shared uniform block snippets                                       */
/* ------------------------------------------------------------------ */

/** Uniform block: element count + scalar scratch + runtime identity value.
 *  `identity` carries values like ±Infinity for max/min reduction — these
 *  are illegal as WGSL const-expressions, so they ride in at runtime. */
function uniN(): string {
  return `
struct OpUniforms { n: u32, _p0: u32, scalar: f32, identity: f32 };
@group(0) @binding(0) var<uniform> uniforms: OpUniforms;`;
}

/** Uniform block with 2D dims (rows, cols) + scalar scratch + identity. */
function uni2D(): string {
  return `
struct OpUniforms { rows: u32, cols: u32, scalar: f32, identity: f32 };
@group(0) @binding(0) var<uniform> uniforms: OpUniforms;`;
}

/**
 * 4D-capable copy uniform block (matches encodeCopyUniform layout).
 * Dimensions use vec4<u32> members: uniform-address-space *array* strides
 * are inconsistently implemented across backends, while vector member
 * layout is uniform and immutable — vec4 also keeps the block at 64 bytes.
 */
function uniCopy(): string {
  return `
struct OpUniforms {
  outShape: vec4<u32>,
  inStrides: vec4<u32>,
  outStrides: vec4<u32>,
  inOffset: u32,
  outOffset: u32,
  total: u32,
  rank: u32,
};
@group(0) @binding(0) var<uniform> uniforms: OpUniforms;`;
}

/* ------------------------------------------------------------------ */
/* Elementwise: 1D binary / unary / scalar                             */
/* ------------------------------------------------------------------ */

export type BinaryExpr = (a: string, b: string) => string;
export type UnaryExpr = (a: string) => string;

const BIND_READ = (name: string, t: string, i: number) =>
  `@group(0) @binding(${i}) var<storage, read> ${name}: array<${t}>;`;
const BIND_RW = (name: string, t: string, i: number) =>
  `@group(0) @binding(${i}) var<storage, read_write> ${name}: array<${t}>;`;

/** out[i] = f(a[i], b[i]) over n elements. */
export function binary1DWgsl(dtype: DType, expr: BinaryExpr): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('a', t, 1)}
${BIND_READ('b', t, 2)}
${BIND_RW('out', t, 3)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= uniforms.n) { return; }
  out[i] = ${expr('a[i]', 'b[i]')};
}`;
}

/** out[i] = f(a[i], scalar) — scalar arrives via uniform. */
export function binaryScalarWgsl(dtype: DType, expr: BinaryExpr): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('a', t, 1)}
${BIND_RW('out', t, 2)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= uniforms.n) { return; }
  out[i] = ${expr('a[i]', 'uniforms.scalar')};
}`;
}

/** out[i] = f(a[i]). */
export function unaryWgsl(dtype: DType, expr: UnaryExpr): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('a', t, 1)}
${BIND_RW('out', t, 2)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= uniforms.n) { return; }
  out[i] = ${expr('a[i]')};
}`;
}

/** Dtype conversion: reads `from`, converts each element, writes `to`. */
export function castWgsl(from: DType, to: DType): string {
  const ft = dtypeInfo(from).wgsl;
  const tt = dtypeInfo(to).wgsl;
  return `
${uniN()}
${BIND_READ('a', ft, 1)}
${BIND_RW('out', tt, 2)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= uniforms.n) { return; }
  out[i] = ${tt}(a[i]);
}`;
}

/* ------------------------------------------------------------------ */
/* Elementwise: 2D broadcast (row-vector or column-vector operand)     */
/* ------------------------------------------------------------------ */

export type BroadcastMode = 'none' | 'row' | 'col';

/**
 * 2D elementwise with broadcast:
 *  - 'row': b has `cols` elements, applied per column: out[i,j] = f(a[i,j], b[j])
 *  - 'col': b has `rows` elements, applied per row:    out[i,j] = f(a[i,j], b[i])
 *  - 'none': plain elementwise over rows*cols.
 */
export function binary2DWgsl(dtype: DType, expr: BinaryExpr, mode: BroadcastMode): string {
  const t = dtypeInfo(dtype).wgsl;
  const bIndex =
    mode === 'row' ? 'let bi = i % uniforms.cols;' : mode === 'col' ? 'let bi = i / uniforms.cols;' : 'let bi = i;';
  return `
${uni2D()}
${BIND_READ('a', t, 1)}
${BIND_READ('b', t, 2)}
${BIND_RW('out', t, 3)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  let total = uniforms.rows * uniforms.cols;
  if (i >= total) { return; }
  ${bIndex}
  out[i] = ${expr('a[i]', 'b[bi]')};
}`;
}

/* ------------------------------------------------------------------ */
/* Reduce: two-phase tree reduction                                     */
/* ------------------------------------------------------------------ */

const WORKGROUP_SIZE = 64;
const REDUCE_CHUNK = 8;

/**
 * Phase-1 shader: each workgroup reduces `WORKGROUP_SIZE * REDUCE_CHUNK`
 * contiguous-ish elements into a single value written to `partial[wg]`.
 * The initial accumulator comes from `uniforms.identity` (runtime value,
 * so ±Infinity identities are legal).
 */
export function reducePhase1Wgsl(dtype: DType, combine: (acc: string, x: string) => string): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('input', t, 1)}
${BIND_RW('partial', t, 2)}
var<workgroup> smem: array<${t}, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) g: vec3u,
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u,
) {
  let start = w.x * ${WORKGROUP_SIZE}u * ${REDUCE_CHUNK}u;
  var acc = uniforms.identity;
  for (var c: u32 = 0u; c < ${REDUCE_CHUNK}u; c++) {
    let idx = start + l.x + c * ${WORKGROUP_SIZE}u;
    if (idx < uniforms.n) {
      acc = ${combine('acc', 'input[idx]')};
    }
  }
  smem[l.x] = acc;
  workgroupBarrier();
  var size: u32 = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) {
      smem[l.x] = ${combine('smem[l.x]', 'smem[l.x + size]')};
    }
    workgroupBarrier();
    size = size / 2u;
  }
  if (l.x == 0u) {
    partial[w.x] = smem[0];
  }
}`;
}

/**
 * Phase-2 shader: one workgroup reduces the partial results to a single value.
 * For value reductions `combine` folds `partial[i]` values directly.
 * `final(expr)` writes the epilogue (e.g. division by n for mean).
 */
export function reducePhase2Wgsl(dtype: DType, combine: (acc: string, x: string) => string, epilogue: (v: string) => string = (v) => v): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('partial', t, 1)}
${BIND_RW('out', t, 2)}
var<workgroup> smem: array<${t}, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) g: vec3u,
  @builtin(local_invocation_id) l: vec3u,
) {
  var acc = uniforms.identity;
  // Sequential over partials (partials count is small).
  for (var i: u32 = l.x; i < uniforms.n; i += ${WORKGROUP_SIZE}u) {
    acc = ${combine('acc', 'partial[i]')};
  }
  smem[l.x] = acc;
  workgroupBarrier();
  var size: u32 = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) {
      smem[l.x] = ${combine('smem[l.x]', 'smem[l.x + size]')};
    }
    workgroupBarrier();
    size = size / 2u;
  }
  if (l.x == 0u) {
    out[0] = ${epilogue('smem[0]')};
  }
}`;
}

/**
 * Arg-max/min reduction (index of extreme value, first occurrence wins).
 * Phase-1 writes candidate indices; phase-2 picks among them by re-reading
 * the original values. `cmp(a, b)` must return true when a is "better" than b.
 */
export function argReducePhase1Wgsl(dtype: DType, better: (a: string, b: string) => string): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
${BIND_READ('input', t, 1)}
@group(0) @binding(2) var<storage, read_write> partial: array<u32>;
var<workgroup> smem: array<u32, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(global_invocation_id) g: vec3u,
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u,
) {
  let start = w.x * ${WORKGROUP_SIZE}u * ${REDUCE_CHUNK}u;
  // Clamp the initial candidate to a valid index: lanes beyond n would
  // otherwise seed the tree with out-of-bounds reads (which return 0 and
  // wrongly win argmin).
  var acc: u32 = min(start + l.x, uniforms.n - 1u); // candidate index
  for (var c: u32 = 0u; c < ${REDUCE_CHUNK}u; c++) {
    let idx = start + l.x + c * ${WORKGROUP_SIZE}u;
    if (idx < uniforms.n) {
      if (${better('input[idx]', 'input[acc]')}) { acc = idx; }
    }
  }
  smem[l.x] = acc;
  workgroupBarrier();
  var size: u32 = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) {
      let ai = smem[l.x];
      let bi = smem[l.x + size];
      if (${better('input[bi]', 'input[ai]')}) { smem[l.x] = bi; }
    }
    workgroupBarrier();
    size = size / 2u;
  }
  if (l.x == 0u) {
    partial[w.x] = smem[0];
  }
}`;
}

export function argReducePhase2Wgsl(dtype: DType, better: (a: string, b: string) => string): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniN()}
@group(0) @binding(1) var<storage, read> partial: array<u32>;
${BIND_READ('input', t, 2)}
@group(0) @binding(3) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) g: vec3u) {
  var acc: u32 = partial[0];
  for (var i: u32 = 1u; i < uniforms.n; i++) {
    let cand = partial[i];
    if (${better('input[cand]', 'input[acc]')}) { acc = cand; }
  }
  out[0] = acc;
}`;
}

/* ------------------------------------------------------------------ */
/* Matmul: 16x16 workgroup tiling                                      */
/* ------------------------------------------------------------------ */

/** C[m,n] = sum_k A[m,k] * B[k,n], all row-major, dimensions via uniform. */
export function matmulWgsl(dtype: DType): string {
  const t = dtypeInfo(dtype).wgsl;
  const TILE = 16;
  return `
struct Dims { m: u32, n: u32, k: u32, _p: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
${BIND_READ('a', t, 1)}
${BIND_READ('b', t, 2)}
${BIND_RW('c', t, 3)}
var<workgroup> tileA: array<${t}, ${TILE * TILE}>;
var<workgroup> tileB: array<${t}, ${TILE * TILE}>;
@compute @workgroup_size(${TILE}, ${TILE})
fn main(
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u,
) {
  let rowBase = w.y * ${TILE}u;
  let colBase = w.x * ${TILE}u;
  var acc = ${t === 'f32' ? '0.0' : `${t}(0)`};
  var tiles: u32 = (dims.k + ${TILE - 1}u) / ${TILE}u;
  for (var tile: u32 = 0u; tile < tiles; tile++) {
    // Cooperatively load one 16x16 tile of A and of B.
    let aRow = rowBase + l.y;
    let aCol = tile * ${TILE}u + l.x;
    let bRow = tile * ${TILE}u + l.y;
    let bCol = colBase + l.x;
    if (aRow < dims.m && aCol < dims.k) {
      tileA[l.y * ${TILE}u + l.x] = a[aRow * dims.k + aCol];
    } else {
      tileA[l.y * ${TILE}u + l.x] = ${t === 'f32' ? '0.0' : `${t}(0)`};
    }
    if (bRow < dims.k && bCol < dims.n) {
      tileB[l.y * ${TILE}u + l.x] = b[bRow * dims.n + bCol];
    } else {
      tileB[l.y * ${TILE}u + l.x] = ${t === 'f32' ? '0.0' : `${t}(0)`};
    }
    workgroupBarrier();
    for (var kk: u32 = 0u; kk < ${TILE}u; kk++) {
      acc = acc + tileA[l.y * ${TILE}u + kk] * tileB[kk * ${TILE}u + l.x];
    }
    workgroupBarrier();
  }
  let outRow = rowBase + l.y;
  let outCol = colBase + l.x;
  if (outRow < dims.m && outCol < dims.n) {
    c[outRow * dims.n + outCol] = acc;
  }
}`;
}

/* ------------------------------------------------------------------ */
/* Transpose 2D                                                        */
/* ------------------------------------------------------------------ */

/** out[j*rows + i] = in[i*cols + j]  (shape [rows, cols] -> [cols, rows]) */
export function transpose2DWgsl(dtype: DType): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uni2D()}
${BIND_READ('input', t, 1)}
${BIND_RW('out', t, 2)}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.y;
  let j = g.x;
  if (i < uniforms.rows && j < uniforms.cols) {
    out[j * uniforms.rows + i] = input[i * uniforms.cols + j];
  }
}`;
}

/**
 * Reduce along the last axis: one workgroup per row, tree reduction inside
 * the workgroup. [M, K] -> [M]. `epilogue` (optional) may reference
 * `uniforms.scalar` (typically cols, for mean).
 */
export function reduceLastAxisWgsl(
  dtype: DType,
  combine: (acc: string, x: string) => string,
  epilogue?: (v: string) => string,
): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uni2D()}
${BIND_READ('input', t, 1)}
${BIND_RW('out', t, 2)}
var<workgroup> smem: array<${t}, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u,
) {
  let cols = uniforms.cols;
  let base = w.x * cols;
  var acc = uniforms.identity;
  for (var i: u32 = l.x; i < cols; i += ${WORKGROUP_SIZE}u) {
    acc = ${combine('acc', 'input[base + i]')};
  }
  smem[l.x] = acc;
  workgroupBarrier();
  var size: u32 = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) {
      smem[l.x] = ${combine('smem[l.x]', 'smem[l.x + size]')};
    }
    workgroupBarrier();
    size = size / 2u;
  }
  if (l.x == 0u) {
    out[w.x] = ${epilogue ? epilogue('smem[0]') : 'smem[0]'};
  }
}`;
}

/* ------------------------------------------------------------------ */
/* Softmax (last axis, fused 3-pass within one workgroup per row)      */
/* ------------------------------------------------------------------ */

/** Numerically stable softmax over the last axis: one workgroup per row. */
export function softmaxWgsl(dtype: DType): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uni2D()}
${BIND_READ('input', t, 1)}
${BIND_RW('out', t, 2)}
var<workgroup> smem: array<${t}, ${WORKGROUP_SIZE}>;
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u,
) {
  let cols = uniforms.cols;
  let base = w.x * cols;
  // Pass 1: row max. -Infinity arrives via uniforms.identity (runtime value).
  var m = uniforms.identity;
  for (var i: u32 = l.x; i < cols; i += ${WORKGROUP_SIZE}u) {
    m = max(m, input[base + i]);
  }
  smem[l.x] = m;
  workgroupBarrier();
  var size: u32 = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) { smem[l.x] = max(smem[l.x], smem[l.x + size]); }
    workgroupBarrier();
    size = size / 2u;
  }
  let rowMax = smem[0];
  workgroupBarrier();
  // Pass 2: sum of exponentials.
  var s = 0.0;
  for (var i: u32 = l.x; i < cols; i += ${WORKGROUP_SIZE}u) {
    s = s + exp(input[base + i] - rowMax);
  }
  smem[l.x] = s;
  workgroupBarrier();
  size = ${WORKGROUP_SIZE / 2}u;
  loop {
    if (size == 0u) { break; }
    if (l.x < size) { smem[l.x] = smem[l.x] + smem[l.x + size]; }
    workgroupBarrier();
    size = size / 2u;
  }
  let denom = smem[0];
  // Pass 3: write normalized output.
  for (var i: u32 = l.x; i < cols; i += ${WORKGROUP_SIZE}u) {
    out[base + i] = exp(input[base + i] - rowMax) / denom;
  }
}`;
}

/* ------------------------------------------------------------------ */
/* Generic strided copy (powers slice / concat / pad)                  */
/* ------------------------------------------------------------------ */

/**
 * Generic gather/copy: for each output linear index `i`, decode ND coords
 * (rank <= 4) against `outShape`, then
 *   src = inOffset  + coord·inStrides
 *   dst = outOffset + coord·outStrides
 * and write `out[dst] = input[src]`.
 *
 * Covers slicing (input strided / output dense) and concat (input dense at
 * inOffset / output strided at outOffset) with one shader.
 */
export function copyWgsl(dtype: DType): string {
  const t = dtypeInfo(dtype).wgsl;
  return `
${uniCopy()}
${BIND_READ('input', t, 1)}
${BIND_RW('out', t, 2)}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x;
  if (i >= uniforms.total) { return; }
  var rem = i;
  var src: u32 = uniforms.inOffset;
  var dst: u32 = uniforms.outOffset;
  var d: u32 = uniforms.rank;
  loop {
    if (d == 0u) { break; }
    d = d - 1u;
    let coord = rem % uniforms.outShape[d];
    rem = rem / uniforms.outShape[d];
    src = src + coord * uniforms.inStrides[d];
    dst = dst + coord * uniforms.outStrides[d];
  }
  out[dst] = input[src];
}`;
}

/**
 * Encode the OpUniforms block consumed by `copyWgsl`.
 * Layout (64 bytes, matches uniCopy's vec4-based struct):
 *   outShape   @  0..15   inStrides @ 16..31   outStrides @ 32..47
 *   inOffset @ 48   outOffset @ 52   total @ 56   rank @ 60
 */
export function encodeCopyUniform(shape: number[], inStrides: number[], outStrides: number[], inOffset: number, outOffset: number): ArrayBuffer {
  const buf = new ArrayBuffer(64);
  const dv = new DataView(buf);
  const rank = Math.min(4, shape.length);
  for (let i = 0; i < rank; i++) {
    dv.setUint32(i * 4, shape[i], true);
    dv.setUint32(16 + i * 4, inStrides[i], true);
    dv.setUint32(32 + i * 4, outStrides[i], true);
  }
  dv.setUint32(48, inOffset, true);
  dv.setUint32(52, outOffset, true);
  dv.setUint32(56, numElementsOf(shape), true);
  dv.setUint32(60, rank, true);
  return buf;
}

function numElementsOf(shape: number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/* ------------------------------------------------------------------ */
/* Uniform encoders (must match the uniform structs above)             */
/* ------------------------------------------------------------------ */

/** OpUniforms { n, _p0, scalar, identity } — 16 bytes. */
export function encodeNUniform(n: number, scalar = 0, identity = 0): ArrayBuffer {
  const b = new ArrayBuffer(16);
  const dv = new DataView(b);
  dv.setUint32(0, n, true);
  dv.setFloat32(8, scalar, true);
  dv.setFloat32(12, identity, true);
  return b;
}

/**
 * OpUniforms { rows, cols, scalar, identity } — 16 bytes.
 * `scalar` rides in the third slot (e.g. mean divides by cols);
 * `identity` carries runtime-only values such as -Infinity for max.
 */
export function encode2DUniform(rows: number, cols: number, scalar = 0, identity = 0): ArrayBuffer {
  const b = new ArrayBuffer(16);
  const dv = new DataView(b);
  dv.setUint32(0, rows, true);
  dv.setUint32(4, cols, true);
  dv.setFloat32(8, scalar, true);
  dv.setFloat32(12, identity, true);
  return b;
}

/** matmul Dims { m, n, k, _p } — 16 bytes. */
export function encodeMatmulUniform(m: number, n: number, k: number): ArrayBuffer {
  const b = new ArrayBuffer(16);
  const dv = new DataView(b);
  dv.setUint32(0, m, true);
  dv.setUint32(4, n, true);
  dv.setUint32(8, k, true);
  return b;
}

/** Row-major contiguous strides for a shape. */
export function contiguousStrides(shape: number[]): number[] {
  const s = new Array<number>(shape.length);
  let acc = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    s[i] = acc;
    acc *= shape[i];
  }
  return s;
}
