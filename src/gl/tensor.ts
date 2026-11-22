/**
 * Eager tensor facade for the WebGL2 fallback backend.
 *
 * Unlike the WebGPU path (lazy graph + scheduler), the WebGL2 path executes
 * each op immediately against float textures. The chainable API mirrors the
 * common subset so user code reads the same; advanced ops that need custom
 * WGSL are WebGPU-only and are reported clearly.
 */

import { GlRuntime, type Tex } from './runtime.js';

export type GlShape = number[];

function prod(shape: GlShape): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Broadcast a flat RHS to `outN` elements for row/col/plain patterns. */
function broadcastFlat(b: ArrayLike<number>, bShape: GlShape, aShape: GlShape): Float32Array {
  const outN = prod(aShape);
  const bN = b.length;
  if (bN === 1) return new Float32Array(outN).fill(b[0]);
  if (bShape.length === aShape.length && bShape.every((d, i) => d === aShape[i])) {
    return Float32Array.from(b);
  }
  const last = aShape[aShape.length - 1];
  if (bN === last && outN % last === 0) {
    const out = new Float32Array(outN);
    for (let i = 0; i < outN; i++) out[i] = b[i % last];
    return out;
  }
  const first = aShape[0];
  if (aShape.length >= 2 && bN === first && outN % first === 0) {
    const out = new Float32Array(outN);
    const stride = outN / first;
    for (let i = 0; i < outN; i++) out[i] = b[Math.floor(i / stride)];
    return out;
  }
  throw new Error(`moxwebgpu/gl2: cannot broadcast [${bShape}] against [${aShape}]`);
}

export class GlTensor {
  constructor(
    readonly rt: GlRuntime,
    readonly tex: Tex,
    readonly shape: GlShape,
  ) {}

  get size(): number { return prod(this.shape); }

  /** Build a tensor from nested arrays or a flat array + explicit shape. */
  static from(rt: GlRuntime, data: ArrayLike<number> | number[][], shape?: GlShape): GlTensor {
    let flat: ArrayLike<number>;
    if (Array.isArray(data) && Array.isArray((data as any)[0])) {
      const rows = data as number[][];
      flat = rows.flat();
      shape = shape ?? [rows.length, rows[0].length];
    } else {
      flat = data as ArrayLike<number>;
    }
    const finalShape = shape ?? [flat.length];
    if (prod(finalShape) !== flat.length) {
      throw new Error(`moxwebgpu/gl2: data length ${flat.length} != shape [${finalShape}]`);
    }
    return new GlTensor(rt, rt.upload(flat), finalShape);
  }

  private binaryOp(other: GlTensor | number, expr: (a: string, b: string) => string): GlTensor {
    if (typeof other === 'number') {
      // GLSL won't multiply a float by an int literal; force a float literal.
      const lit = Number.isFinite(other)
        ? (Number.isInteger(other) ? `${other}.0` : `${other}`)
        : String(other);
      const shader = this.rt.unary(this.tex, (a) => `(${expr(a, lit)})`);
      return new GlTensor(this.rt, shader, this.shape);
    }
    const bFlat = broadcastFlat(
      this.rt.readback(other.tex), other.shape, this.shape,
    );
    const bTex = this.rt.upload(bFlat);
    return new GlTensor(this.rt, this.rt.binary(this.tex, bTex, expr('a', 'b')), this.shape);
  }

  add(other: GlTensor | number): GlTensor { return this.binaryOp(other, (a, b) => `${a} + ${b}`); }
  sub(other: GlTensor | number): GlTensor { return this.binaryOp(other, (a, b) => `${a} - ${b}`); }
  mul(other: GlTensor | number): GlTensor { return this.binaryOp(other, (a, b) => `${a} * ${b}`); }
  div(other: GlTensor | number): GlTensor { return this.binaryOp(other, (a, b) => `${a} / ${b}`); }

  private unaryOp(expr: (v: string) => string): GlTensor {
    return new GlTensor(this.rt, this.rt.unary(this.tex, expr), this.shape);
  }

  neg(): GlTensor { return this.unaryOp((a) => `-${a}`); }
  abs(): GlTensor { return this.unaryOp((a) => `abs(${a})`); }
  exp(): GlTensor { return this.unaryOp((a) => `exp(${a})`); }
  log(): GlTensor { return this.unaryOp((a) => `log(${a})`); }
  sqrt(): GlTensor { return this.unaryOp((a) => `sqrt(${a})`); }
  sin(): GlTensor { return this.unaryOp((a) => `sin(${a})`); }
  cos(): GlTensor { return this.unaryOp((a) => `cos(${a})`); }
  square(): GlTensor { return this.unaryOp((a) => `(${a} * ${a})`); }
  relu(): GlTensor { return this.unaryOp((a) => `max(${a}, 0.0)`); }
  sigmoid(): GlTensor { return this.unaryOp((a) => `(1.0 / (1.0 + exp(-(${a}))))`); }

  /** Matrix multiply for 2D tensors: [M,K] @ [K,N] -> [M,N]. */
  matmul(other: GlTensor): GlTensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error('moxwebgpu/gl2: matmul requires two 2D tensors');
    }
    const [m, k] = this.shape;
    const [k2, n] = other.shape;
    if (k !== k2) {
      throw new Error(`moxwebgpu/gl2: matmul shape mismatch [${m},${k}] @ [${k2},${n}]`);
    }
    // Re-lay out both operands as explicit 2D textures so the shader's
    // texelFetch(ivec2(col,row)) addresses are stable on any device.
    const a2 = this.rt.upload2D(this.rt.readback(this.tex), m, k);
    const b2 = this.rt.upload2D(this.rt.readback(other.tex), k, n);
    const out = this.rt.matmul(a2, b2, m, k, n);
    return new GlTensor(this.rt, out, [m, n]);
  }

  /** Full reduction to scalar. */
  sum(): number { return this.rt.reduceSum(this.tex); }
  /** Mean of all elements. */
  mean(): number { return this.rt.reduceSum(this.tex) / Math.max(1, this.size); }

  /** Execute and copy back as Float32Array. */
  async toArray(): Promise<Float32Array> { return this.rt.readback(this.tex); }
  /** Scalar convenience. */
  async item(): Promise<number> { return (await this.toArray())[0]; }
}
