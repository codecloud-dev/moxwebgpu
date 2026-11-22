/**
 * WebGL2 compute runtime. Executes tensor ops as fragment shaders over
 * float textures (RGBA32F, one f32 per texel in the R channel). This is the
 * fallback path when WebGPU is unavailable; the high-level tensor API stays
 * identical so callers don't change code.
 *
 * Capability note: WebGL2 has no compute shaders, so reductions loop inside a
 * single fragment and matmul loops over K. This is correct but slower than a
 * WebGPU compute pass on large tensors — by design.
 */

import { VERT, unaryShader, binaryShader, MATMUL_SHADER, REDUCE_SUM_SHADER } from './shaders.js';

export type Tex = {
  tex: WebGLTexture;
  width: number;
  height: number;
  /** logical element count */
  n: number;
};

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) || 'unknown';
    gl.deleteShader(sh);
    throw new Error(`moxwebgpu/gl2: shader compile failed: ${log}`);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: WebGLShader, fs: WebGLShader): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, vs);
  gl.attachShader(p, fs);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p) || 'unknown';
    gl.deleteProgram(p);
    throw new Error(`moxwebgpu/gl2: program link failed: ${log}`);
  }
  return p;
}

/** Choose a 2D texture layout for `n` elements within the device limit. */
export function layout(n: number, maxWidth: number): { width: number; height: number } {
  const width = Math.max(1, Math.min(n, maxWidth));
  const height = Math.max(1, Math.ceil(n / width));
  return { width, height };
}

export class GlRuntime {
  readonly gl: WebGL2RenderingContext;
  private readonly maxWidth: number;
  private readonly programs = new Map<string, WebGLProgram>();
  private readonly quad: WebGLBuffer;
  private readonly vs: WebGLShader;

  constructor(canvas?: HTMLCanvasElement) {
    const c = canvas ?? document.createElement('canvas');
    const gl = c.getContext('webgl2', {
      antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true,
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error('moxwebgpu/gl2: WebGL2 context unavailable');
    if (!gl.getExtension('EXT_color_buffer_float')) {
      throw new Error('moxwebgpu/gl2: EXT_color_buffer_float unavailable (cannot store f32 tensors)');
    }
    this.gl = gl;
    this.maxWidth = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    this.vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const buf = gl.createBuffer();
    if (!buf) throw new Error('moxwebgpu/gl2: cannot create vertex buffer');
    this.quad = buf;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  }

  private program(key: string, fsSrc: string): WebGLProgram {
    let p = this.programs.get(key);
    if (p) return p;
    p = link(this.gl, this.vs, compile(this.gl, this.gl.FRAGMENT_SHADER, fsSrc));
    this.programs.set(key, p);
    return p;
  }

  private makeTex(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    return tex;
  }

  private fbo(tex: WebGLTexture): WebGLFramebuffer {
    const gl = this.gl;
    const fb = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      gl.deleteFramebuffer(fb);
      throw new Error(`moxwebgpu/gl2: framebuffer incomplete (0x${status.toString(16)})`);
    }
    return fb;
  }

  /** Upload a flat f32 array into a texture. */
  upload(data: ArrayLike<number>): Tex {
    const gl = this.gl;
    const n = data.length;
    const { width, height } = layout(n, this.maxWidth);
    const tex = this.makeTex(width, height);
    const pixels = new Float32Array(width * height * 4);
    for (let i = 0; i < n; i++) pixels[i * 4] = data[i] as number;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, pixels);
    return { tex, width, height, n };
  }

  /** Allocate an output texture with an explicit logical length. */
  alloc(n: number): Tex {
    const { width, height } = layout(n, this.maxWidth);
    return { tex: this.makeTex(width, height), width, height, n };
  }

  /** Allocate a texture with an explicit 2D layout (width=cols, height=rows). */
  alloc2D(rows: number, cols: number): Tex {
    const tex = this.makeTex(cols, rows);
    return { tex, width: cols, height: rows, n: rows * cols };
  }

  /**
   * Upload a flat row-major array as a 2D texture with an explicit column
   * count. Used by matmul so texelFetch(ivec2(col,row)) lands on the right
   * element regardless of the device texture width limit.
   */
  upload2D(flat: ArrayLike<number>, rows: number, cols: number): Tex {
    if (flat.length !== rows * cols) {
      throw new Error(`moxwebgpu/gl2: upload2D length ${flat.length} != ${rows}x${cols}`);
    }
    const gl = this.gl;
    const tex = this.makeTex(cols, rows);
    const pixels = new Float32Array(cols * rows * 4);
    for (let i = 0; i < flat.length; i++) pixels[i * 4] = flat[i] as number;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, cols, rows, gl.RGBA, gl.FLOAT, pixels);
    return { tex, width: cols, height: rows, n: rows * cols };
  }

  private bindProgram(p: WebGLProgram, out: Tex): void {
    const gl = this.gl;
    gl.useProgram(p);
    const loc = gl.getAttribLocation(p, 'aPos');
    if (loc >= 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo(out.tex));
    gl.viewport(0, 0, out.width, out.height);
  }

  private bindTex(p: WebGLProgram, name: string, tex: Tex, unit: number): void {
    const gl = this.gl;
    const loc = gl.getUniformLocation(p, name);
    if (loc === null) return;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex.tex);
    gl.uniform1i(loc, unit);
  }

  private draw(): void {
    this.gl.drawArrays(this.gl.TRIANGLES, 0, 3);
  }

  /** Elementwise unary op; `expr` is a GLSL expression in `a`. */
  unary(a: Tex, expr: (v: string) => string): Tex {
    const gl = this.gl;
    const key = `u:${expr('a')}`;
    const p = this.program(key, unaryShader(expr));
    const out = this.alloc(a.n);
    this.bindProgram(p, out);
    this.bindTex(p, 'uA', a, 0);
    gl.uniform1i(gl.getUniformLocation(p, 'uSize'), a.n);
    gl.uniform1i(gl.getUniformLocation(p, 'uWidth'), a.width);
    this.draw();
    return out;
  }

  /** Elementwise binary op on equal-length textures. */
  binary(a: Tex, b: Tex, op: string): Tex {
    const gl = this.gl;
    const p = this.program(`b:${op}`, binaryShader(op));
    const out = this.alloc(a.n);
    this.bindProgram(p, out);
    this.bindTex(p, 'uA', a, 0);
    this.bindTex(p, 'uB', b, 1);
    gl.uniform1i(gl.getUniformLocation(p, 'uSize'), a.n);
    gl.uniform1i(gl.getUniformLocation(p, 'uWidth'), a.width);
    this.draw();
    return out;
  }

  /** A[M,K] @ B[K,N] -> [M,N]. Both inputs must be 2D layouts (upload2D). */
  matmul(a: Tex, b: Tex, m: number, k: number, n: number): Tex {
    const gl = this.gl;
    const p = this.program('matmul', MATMUL_SHADER);
    const out = this.alloc2D(m, n);
    this.bindProgram(p, out);
    this.bindTex(p, 'uA', a, 0);
    this.bindTex(p, 'uB', b, 1);
    gl.uniform1i(gl.getUniformLocation(p, 'uM'), m);
    gl.uniform1i(gl.getUniformLocation(p, 'uK'), k);
    gl.uniform1i(gl.getUniformLocation(p, 'uN'), n);
    gl.uniform1i(gl.getUniformLocation(p, 'uWidth'), out.width);
    this.draw();
    return out;
  }

  /** Full sum of all logical elements; returns a single JS number. */
  reduceSum(a: Tex): number {
    const gl = this.gl;
    const p = this.program('reduce-sum', REDUCE_SUM_SHADER);
    const out = this.alloc(1);
    this.bindProgram(p, out);
    this.bindTex(p, 'uA', a, 0);
    gl.uniform1i(gl.getUniformLocation(p, 'uSize'), a.n);
    gl.uniform1i(gl.getUniformLocation(p, 'uWidth'), a.width);
    this.draw();
    // 1x1 readback
    const px = new Float32Array(4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo(out.tex));
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    return px[0];
  }

  /** Read back `n` logical elements in row-major order. */
  readback(tex: Tex): Float32Array {
    const gl = this.gl;
    const out = new Float32Array(tex.n * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo(tex.tex));
    gl.readPixels(0, 0, tex.width, tex.height, gl.RGBA, gl.FLOAT, out);
    const result = new Float32Array(tex.n);
    for (let i = 0; i < tex.n; i++) result[i] = out[i * 4];
    return result;
  }

  destroy(): void {
    const gl = this.gl;
    for (const p of this.programs.values()) gl.deleteProgram(p);
    this.programs.clear();
    gl.deleteShader(this.vs);
    gl.deleteBuffer(this.quad);
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext();
  }
}
