/**
 * 2026-11-13 全仓代码审计回归测试。
 *
 * 背景：外部 AI 分享清单列了 16 条 P0/P1/P2「问题」。逐条对照真实源码后，
 * 只有一部分成立（其余是误判，或本来就是有意的功能边界）。本文件只给
 * **真实修复**写回归断言，同时把几条最容易被误信的结论用测试固化下来，
 * 防止以后有人照着分享把正确实现改坏。
 *
 * 全部在 Node 下跑，不需要 GPU：Tensor 只构建 lazy DAG，readback / pool
 * 用最小 fake device 验证字节与生命周期，不执行任何真实 WGSL。
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { Tensor } from '../../src/tensor/tensor.js';
import { installTensorOps } from '../../src/tensor/ops/index.js';
import { installBackward } from '../../src/tensor/ops/backward.js';
import { GpuDataBuffer, BufferPool, DEFAULT_MAX_POOLED_BYTES } from '../../src/core/buffer.js';
import { Scheduler } from '../../src/graph/scheduler.js';
import { binaryScalarWgsl, argReducePhase1Wgsl } from '../../src/tensor/codegen.js';
import { matmulDef } from '../../src/tensor/ops/matmul.js';

installTensorOps(Tensor.prototype);
installBackward();

const noGpuCtx: any = {};
function leaf(shape: number[], dtype: any = 'f32'): Tensor {
  return new Tensor(noGpuCtx, shape, dtype, null, null);
}

/* ------------------------------------------------------------------ */
/* 1. readback 只拷逻辑字节数（真 bug）                                 */
/* ------------------------------------------------------------------ */
describe('scheduler.readback: copies only the logical payload', () => {
  let savedUsage: any;
  let savedMapMode: any;
  let hadUsage: boolean;
  let hadMapMode: boolean;

  beforeAll(() => {
    hadUsage = 'GPUBufferUsage' in globalThis;
    hadMapMode = 'GPUMapMode' in globalThis;
    savedUsage = (globalThis as any).GPUBufferUsage;
    savedMapMode = (globalThis as any).GPUMapMode;
    (globalThis as any).GPUBufferUsage = { MAP_READ: 1, COPY_DST: 8 };
    (globalThis as any).GPUMapMode = { READ: 1 };
  });

  afterAll(() => {
    if (hadUsage) (globalThis as any).GPUBufferUsage = savedUsage;
    else delete (globalThis as any).GPUBufferUsage;
    if (hadMapMode) (globalThis as any).GPUMapMode = savedMapMode;
    else delete (globalThis as any).GPUMapMode;
  });

  /** Minimal fake WebGPU device that records staging sizes and copy lengths. */
  function fakeCtx() {
    const copies: number[] = [];
    const stagingSizes: number[] = [];
    let stagingBytes = 0;
    const staging: any = {
      mapAsync: async () => {},
      getMappedRange: () => new ArrayBuffer(stagingBytes),
      unmap: () => {},
      destroy: () => {},
    };
    const device: any = {
      createBuffer: (d: { size: number; usage: number }) => {
        // MAP_READ bit marks the staging buffer created inside readback().
        if (d.usage & 1) {
          stagingBytes = d.size;
          stagingSizes.push(d.size);
          return staging;
        }
        return { size: d.size, destroy: () => {} };
      },
      createCommandEncoder: () => ({
        copyBufferToBuffer: (
          _src: unknown, _srcOff: number, _dst: unknown, _dstOff: number, size: number,
        ) => copies.push(size),
        finish: () => ({}),
      }),
    };
    const ctx: any = {
      device,
      queue: { submit: () => {} },
      deviceLost: new Promise<never>(() => {}),
    };
    return { ctx, copies, stagingSizes };
  }

  it('reads 5 f32 elements from a 32-byte bucket as 20 bytes, not 32', async () => {
    const { ctx, copies, stagingSizes } = fakeCtx();
    const pooled = new GpuDataBuffer({ size: 32, destroy() {} } as any, 5, 'f32');
    expect(pooled.byteSize).toBe(20);

    const out = await new Scheduler(ctx).readback(pooled);
    expect(copies).toEqual([20]);
    expect(stagingSizes).toEqual([20]);
    expect(out.length).toBe(5);
    expect(out).toBeInstanceOf(Float32Array);
  });

  it('keeps the integer dtype and returns exactly `elements` values', async () => {
    const { ctx, copies } = fakeCtx();
    const pooled = new GpuDataBuffer({ size: 64, destroy() {} } as any, 3, 'i32');
    const out = await new Scheduler(ctx).readback(pooled);
    expect(copies).toEqual([12]);
    expect(out).toBeInstanceOf(Int32Array);
    expect(out.length).toBe(3);
  });

  it('returns an empty array without allocating a staging buffer', async () => {
    const { ctx, copies, stagingSizes } = fakeCtx();
    const pooled = new GpuDataBuffer({ size: 16, destroy() {} } as any, 0, 'u32');
    const out = await new Scheduler(ctx).readback(pooled);
    expect(out.length).toBe(0);
    expect(out).toBeInstanceOf(Uint32Array);
    expect(copies).toEqual([]);
    expect(stagingSizes).toEqual([]);
  });
});

/* ------------------------------------------------------------------ */
/* 2. i32/u32 标量 op 的 WGSL 类型（真 bug）                            */
/* ------------------------------------------------------------------ */
describe('binaryScalarWgsl: scalar uniform is converted to the tensor dtype', () => {
  const add = (a: string, b: string) => `(${a} + ${b})`;
  const rsub = (a: string, b: string) => `(${b} - ${a})`;

  it('f32 keeps the raw f32 scalar', () => {
    const wgsl = binaryScalarWgsl('f32', add);
    expect(wgsl).toContain('out[i] = (a[i] + uniforms.scalar)');
    expect(wgsl).not.toContain('f32(uniforms.scalar)');
  });

  it('i32 emits i32(uniforms.scalar) instead of mixing i32 with f32', () => {
    const wgsl = binaryScalarWgsl('i32', add);
    expect(wgsl).toContain('out[i] = (a[i] + i32(uniforms.scalar))');
    expect(wgsl).not.toContain('a[i] + uniforms.scalar');
  });

  it('u32 emits u32(uniforms.scalar), also for reversed-scalar forms', () => {
    expect(binaryScalarWgsl('u32', add)).toContain('a[i] + u32(uniforms.scalar)');
    expect(binaryScalarWgsl('u32', rsub)).toContain('u32(uniforms.scalar) - a[i]');
  });

  it('the uniform struct still declares scalar as f32 (single layout)', () => {
    expect(binaryScalarWgsl('i32', add)).toContain('scalar: f32');
  });
});

/* ------------------------------------------------------------------ */
/* 3. argmax/argmin 空张量（真 bug：u32 下溢）                          */
/* ------------------------------------------------------------------ */
describe('arg-reduce on an empty tensor', () => {
  it('argmax()/argmin() reject shape [0] at the call site with a clear message', () => {
    expect(() => (leaf([0]) as any).argmax()).toThrow(/non-empty tensor/);
    expect(() => (leaf([0]) as any).argmin()).toThrow(/non-empty tensor/);
  });

  it('non-empty tensors still build a [1] u32 output', () => {
    expect((leaf([7]) as any).argmax().shape).toEqual([1]);
    expect((leaf([2, 3]) as any).argmin().dtype).toBe('u32');
  });

  it('phase-1 WGSL clamps n without unsigned underflow', () => {
    const wgsl = argReducePhase1Wgsl('f32', (c, b) => `(${c} > ${b})`);
    expect(wgsl).toContain('max(uniforms.n, 1u) - 1u');
    expect(wgsl).not.toContain('uniforms.n - 1u');
  });
});

/* ------------------------------------------------------------------ */
/* 4. BufferPool 字节上限（真 bug：无上限显存增长）                      */
/* ------------------------------------------------------------------ */
describe('BufferPool: bounded idle memory', () => {
  function fakeDevice() {
    const destroyed: number[] = [];
    const device: any = {
      createBuffer: ({ size }: { size: number }) => ({
        size,
        destroy: () => destroyed.push(size),
      }),
    };
    return { device, destroyed };
  }

  it('default ceiling is a finite, positive byte budget', () => {
    expect(DEFAULT_MAX_POOLED_BYTES).toBeGreaterThan(0);
    expect(Number.isFinite(DEFAULT_MAX_POOLED_BYTES)).toBe(true);
  });

  it('reuses a buffer and tracks checked-out count correctly', () => {
    const { device } = fakeDevice();
    const pool = new BufferPool(device, { maxPooledBytes: 1024 });
    const a = pool.acquire(5, 'f32');
    expect(pool.live).toBe(1);
    pool.release(a);
    expect(pool.live).toBe(0);
    expect(pool.pooled).toBe(1);
    expect(pool.pooledBytesUsed).toBe(a.buffer.size);

    const b = pool.acquire(5, 'f32');
    expect(pool.live).toBe(1);
    expect(pool.pooled).toBe(0);
    expect(pool.pooledBytesUsed).toBe(0);
    pool.release(b);
    expect(pool.live).toBe(0);
  });

  it('destroys a returned buffer instead of exceeding the budget', () => {
    const { device, destroyed } = fakeDevice();
    const pool = new BufferPool(device, { maxPooledBytes: 64 });
    const small = pool.acquire(5, 'f32'); // bucket 32
    pool.release(small);
    expect(pool.pooledBytesUsed).toBe(32);

    const big = pool.acquire(40, 'f32'); // bucket 256
    expect(big.buffer.size).toBe(256);
    pool.release(big);
    expect(destroyed).toContain(256);
    expect(pool.pooledBytesUsed).toBe(32);
    expect(pool.live).toBe(0);
    expect(pool.pooledBytesUsed).toBeLessThanOrEqual(pool.pooledBytesLimit);
  });

  it('maxPooledBytes: 0 disables pooling entirely', () => {
    const { device, destroyed } = fakeDevice();
    const pool = new BufferPool(device, { maxPooledBytes: 0 });
    const a = pool.acquire(5, 'f32');
    pool.release(a);
    expect(pool.pooled).toBe(0);
    expect(pool.pooledBytesUsed).toBe(0);
    expect(destroyed).toContain(32);
  });

  it('clear() resets the byte accounting', () => {
    const { device } = fakeDevice();
    const pool = new BufferPool(device, { maxPooledBytes: 1024 });
    pool.release(pool.acquire(5, 'f32'));
    expect(pool.pooledBytesUsed).toBeGreaterThan(0);
    pool.clear();
    expect(pool.pooled).toBe(0);
    expect(pool.pooledBytesUsed).toBe(0);
  });

  it('uniform buffers obey the same budget', () => {
    const { device } = fakeDevice();
    const pool = new BufferPool(device, { maxPooledBytes: 512 });
    const ubo = pool.acquireUniform(16); // bucket 256
    expect(ubo.size).toBe(256);
    pool.releaseUniform(ubo);
    expect(pool.pooledBytesUsed).toBe(256);
    const ubo2 = pool.acquireUniform(32); // bucket 256, over budget
    pool.releaseUniform(ubo2);
    expect(pool.pooledBytesUsed).toBe(256);
  });
});

/* ------------------------------------------------------------------ */
/* 5. reshape(-1) 整除校验（真 bug：小数维度 / 零除）                    */
/* ------------------------------------------------------------------ */
describe('reshape(-1) requires an exact integer solution', () => {
  it('infers a valid dimension', () => {
    expect(leaf([10]).reshape(2, -1).shape).toEqual([2, 5]);
    expect(leaf([12]).reshape(-1, 3).shape).toEqual([4, 3]);
    expect(leaf([0]).reshape(-1).shape).toEqual([0]);
  });

  it('rejects a non-divisible -1 with an actionable message', () => {
    expect(() => leaf([10]).reshape(3, -1)).toThrow(/not divisible by 3/);
    expect(() => leaf([10]).reshape(3, -1)).not.toThrow(/cannot reshape/);
  });

  it('rejects fractional / negative dimensions before they leak into shapes', () => {
    expect(() => leaf([10]).reshape(2.5, 4)).toThrow(/non-negative integers/);
    expect(() => leaf([10]).reshape(-2, 5)).toThrow(/non-negative integers/);
  });

  it('rejects a zero divisor and multiple -1 values', () => {
    expect(() => leaf([10]).reshape(0, -1)).toThrow(/not divisible by 0/);
    expect(() => leaf([10]).reshape(-1, -1)).toThrow(/at most one -1/);
  });
});

/* ------------------------------------------------------------------ */
/* 6. 操作数类型守卫（真 bug：晦涩内部错误）                             */
/* ------------------------------------------------------------------ */
describe('operand type guards', () => {
  it('binary ops reject null / string / plain objects at the entry point', () => {
    expect(() => (leaf([2]) as any).add(null)).toThrow(/expects a Tensor or a number, got null/);
    expect(() => (leaf([2]) as any).mul('x')).toThrow(/got string/);
    expect(() => (leaf([2]) as any).rsub({})).toThrow(/got object/);
  });

  it('valid Tensor / number operands still build lazy nodes', () => {
    expect((leaf([2]) as any).add(1).shape).toEqual([2]);
    expect((leaf([2]) as any).add(leaf([2])).shape).toEqual([2]);
  });

  it('clamp() validates finite bounds and ordering', () => {
    expect(() => (leaf([2]) as any).clamp(5, 1)).toThrow(/lo <= hi/);
    expect(() => (leaf([2]) as any).clamp(null, 1)).toThrow(/finite numeric/);
    expect(() => (leaf([2]) as any).clamp(0, Infinity)).toThrow(/finite numeric/);
    expect((leaf([2]) as any).clamp(0, 1).shape).toEqual([2]);
  });
});

/* ------------------------------------------------------------------ */
/* 7. 错误信息带形状上下文（P2，真实改进）                               */
/* ------------------------------------------------------------------ */
describe('matmul shape errors carry readable context', () => {
  it('prints A/B shapes as [MxK] instead of bare comma lists', () => {
    expect(() => matmulDef.outShape([[2, 3], [4, 5]], {}))
      .toThrow(/A\[2x3\] @ B\[4x5\]/);
  });

  it('rank-3 input is an explicit, well-described limitation (not a wrong shape)', () => {
    expect(() => matmulDef.outShape([[2, 2, 2], [2, 2, 2]], {}))
      .toThrow(/must be 2-D, got shape \[2,2,2\]/);
  });
});

/* ------------------------------------------------------------------ */
/* 8. 固化分享清单里的误判：这些实现本来就是对的，别照分享改坏           */
/* ------------------------------------------------------------------ */
describe('share-list claims that do NOT hold (guard against harmful "fixes")', () => {
  it('autograd accumulates gradients across branches instead of overwriting', () => {
    const a = leaf([2]);
    a.requiresGrad = true;
    (a.mul(a).sum() as any).backward();
    // Two grads flow into the same leaf -> `prev.add(gi)`, i.e. an add node.
    expect(a.grad).not.toBeNull();
    expect(a.grad!.node?.op.name).toBe('add');
  });

  it('fromData converts by the requested dtype, not "forced to Float32Array"', () => {
    const writes: any[] = [];
    const fakeCtx: any = {
      pool: {
        acquire: (elements: number, dtype: any) =>
          new GpuDataBuffer({ size: 64, destroy() {} } as any, elements, dtype),
      },
      queue: { writeBuffer: (...args: any[]) => writes.push(args) },
    };
    const t = Tensor.fromData(fakeCtx, new Int32Array([1, 2, 3]), { dtype: 'i32' });
    expect(t.dtype).toBe('i32');
    expect(writes.length).toBe(1);
    const uploaded = new Int32Array(writes[0][2]);
    expect(Array.from(uploaded)).toEqual([1, 2, 3]);
    expect(writes[0][4]).toBe(12); // 3 * 4 bytes
  });

  it('matmul rejects rank != 2 explicitly (a documented limitation, not silent corruption)', () => {
    expect(() => matmulDef.outShape([[2, 2, 2], [2, 2]], {})).toThrow(/must be 2-D/);
  });

  it('readback returns exactly `elements` values, never bucket padding', async () => {
    // Covered in detail above; this is the concise behavioural statement.
    const pooled = new GpuDataBuffer({ size: 32, destroy() {} } as any, 5, 'f32');
    expect(pooled.elements).toBe(5);
    expect(pooled.byteSize).toBeLessThan(pooled.buffer.size);
  });
});
