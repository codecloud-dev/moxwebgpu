/**
 * Tensor: the lazy, chainable face of moxwebgpu.
 *
 * A tensor is either a *leaf* (owns GPU data, uploaded eagerly) or the root
 * of a lazy graph node (materialized on demand). All arithmetic returns new
 * lazy tensors; `toArray()` / `toBuffer()` trigger a single topological
 * execution of the whole subgraph.
 */

import type { MoxContext } from '../core/context.js';
import type { GpuDataBuffer } from '../core/buffer.js';
import type { LazyNode, OpDef } from '../graph/lazy.js';
import { createNode } from '../graph/lazy.js';
import { numElements, asTypedArray, type DType } from '../core/dtype.js';

export interface TensorOptions {
  shape?: number[];
  dtype?: DType;
}

export class Tensor {
  constructor(
    public readonly ctx: MoxContext,
    public shape: number[],
    public dtype: DType,
    /** Lazy graph node (null for leaves). */
    public node: LazyNode | null,
    /** GPU data (leaves and materialized tensors). */
    public data: GpuDataBuffer | null,
  ) {}

  get size(): number {
    return numElements(this.shape);
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** Create a leaf tensor from JS data (TypedArray, nested arrays or flat). */
  static fromData(ctx: MoxContext, data: TensorLike, options: TensorOptions = {}): Tensor {
    const { shape, flat } = inferShape(data);
    const dtype: DType = options.dtype ?? 'f32';
    const finalShape = options.shape ?? shape;
    const expected = numElements(finalShape);
    if (flat.length !== expected) {
      throw new Error(
        `moxwebgpu: data length ${flat.length} does not match shape [${finalShape}] (${expected} elements)`,
      );
    }
    const typed = asTypedArray(dtype, flat);
    const buf = ctx.pool.acquire(expected, dtype);
    ctx.queue.writeBuffer(buf.buffer, 0, typed.buffer, typed.byteOffset, typed.byteLength);
    return new Tensor(ctx, finalShape, dtype, null, buf);
  }

  /**
   * Attach an op to this tensor: returns a new lazy tensor. All prototype
   * methods funnel through here.
   */
  apply(op: OpDef, inputs: Tensor[], attrs: any = {}): Tensor {
    const node = createNode(op, inputs, attrs);
    return new Tensor(this.ctx, node.shape, node.dtype, node, null);
  }

  /**
   * Zero-copy reshape: shares the same GPU buffer / graph node, only the
   * shape metadata changes. Requires the same element count.
   */
  reshape(...shape: number[]): Tensor {
    // Allow both t.reshape(2, 3) and t.reshape([2, 3])
    const target = (shape.length === 1 && Array.isArray(shape[0]) ? shape[0] : shape) as number[];
    let hasMinusOne = false;
    let known = 1;
    for (const d of target) {
      if (d === -1) {
        if (hasMinusOne) throw new Error('moxwebgpu: reshape() supports at most one -1');
        hasMinusOne = true;
      } else {
        known *= d;
      }
    }
    const finalShape = hasMinusOne
      ? target.map((d) => (d === -1 ? this.size / known : d))
      : target;
    if (numElements(finalShape) !== this.size) {
      throw new Error(`moxwebgpu: cannot reshape [${this.shape}] (${this.size}) to [${finalShape}] (${numElements(finalShape)})`);
    }
    return new Tensor(this.ctx, finalShape, this.dtype, this.node, this.data);
  }

  /** Execute the underlying subgraph (once) and return the GPU buffer. */
  async toBuffer(): Promise<GpuDataBuffer> {
    return this.ctx.scheduler.materialize(this);
  }

  /** Execute (if needed) and copy the result back to the CPU. */
  async toArray(): Promise<Float32Array | Int32Array | Uint32Array> {
    const buf = await this.toBuffer();
    return this.ctx.scheduler.readback(buf);
  }

  /** Convenience for scalar results: first element as a JS number. */
  async item(): Promise<number> {
    const arr = await this.toArray();
    return arr[0];
  }

  /** Free the GPU buffer of a leaf tensor. Lazy tensors recycle via the pool. */
  destroy(): void {
    if (this.data) {
      this.ctx.pool.release(this.data);
      this.data = null;
    }
  }
}

/**
 * Op surface — installed on the prototype by installTensorOps()
 * (src/tensor/ops/index.ts, called from index.ts on module load).
 * Declared via interface merging: type-checkable, zero runtime footprint.
 */
export interface Tensor {
  // elementwise binary (tensor or scalar operand)
  add(other: Tensor | number): Tensor;
  sub(other: Tensor | number): Tensor;
  mul(other: Tensor | number): Tensor;
  div(other: Tensor | number): Tensor;
  pow(other: Tensor | number): Tensor;
  clamp(lo: number, hi: number): Tensor;
  // reversed scalar ops: 2.sub(x) ≙ x - 2
  rsub(x: number): Tensor;
  rdiv(x: number): Tensor;

  // unary
  neg(): Tensor;
  abs(): Tensor;
  exp(): Tensor;
  log(): Tensor;
  sqrt(): Tensor;
  sin(): Tensor;
  cos(): Tensor;
  tanh(): Tensor;
  floor(): Tensor;
  ceil(): Tensor;
  relu(): Tensor;
  sigmoid(): Tensor;
  square(): Tensor;
  sign(): Tensor;

  // reductions — one calling convention: no arg = global, number = axis,
  // and (max/min only) a Tensor = elementwise. Aliases: maxReduce/minReduce.
  sum(other?: number | Tensor): Tensor;
  mean(other?: number | Tensor): Tensor;
  max(other?: number | Tensor): Tensor;
  min(other?: number | Tensor): Tensor;
  maxReduce(other?: number | Tensor): Tensor;
  minReduce(other?: number | Tensor): Tensor;
  argmax(): Tensor;
  argmin(): Tensor;

  // matmul & shape
  matmul(other: Tensor): Tensor;
  transpose(): Tensor;
  slice(start: number | number[], size: number | number[]): Tensor;
  concat(other: Tensor, axis?: number): Tensor;

  // nn
  softmax(): Tensor;

  // dtype conversion
  cast(dtype: DType): Tensor;
  toFloat(): Tensor;
}

export type TensorLike = number | number[] | number[][] | number[][][] | Float32Array | Int32Array | Uint32Array | Float64Array;

/** Infer shape from nested arrays (TypedArrays are treated as 1D). */
function inferShape(data: TensorLike): { shape: number[]; flat: number[] } {
  if (typeof data === 'number') return { shape: [], flat: [data] };
  if (ArrayBuffer.isView(data)) {
    return { shape: [(data as Float32Array).length], flat: Array.from(data as Float32Array) };
  }
  const shape: number[] = [];
  let level: any = data;
  while (Array.isArray(level)) {
    shape.push(level.length);
    level = level[0];
  }
  const flat: number[] = [];
  flatten(data as number[], flat);
  return { shape, flat };
}

function flatten(arr: any[], out: number[]): void {
  for (const x of arr) {
    if (Array.isArray(x)) flatten(x, out);
    else out.push(x as number);
  }
}
