/**
 * GPU buffer management: a thin wrapper over GPUBuffer plus a size-bucketed
 * pool for recycling intermediate storage buffers between dispatches.
 */

import type { DType } from './dtype.js';
import { dtypeInfo } from './dtype.js';

/**
 * `GPUBufferUsage` 是浏览器 / WebGPU 全局，Node 环境下不存在。
 * 下面的常量是顶层定义，会在 `import` 时立即求值；若直接写 `GPUBufferUsage.STORAGE`
 * 会让整个包在 Node 下一 `require` 就 `ReferenceError` 崩溃（npm 包不可用）。
 * 这里用标准规范位值兜底：真机 WebGPU 运行时全局存在时走全局，否则用等价常量，
 * 保证包在 Node 下也能被 import（实际执行 GPU 命令仍需浏览器 / WebGPU 环境）。
 */
const GPUBufferUsageFlags = (typeof GPUBufferUsage !== 'undefined')
  ? GPUBufferUsage
  : {
      MAP_READ: 0x0001,
      MAP_WRITE: 0x0002,
      COPY_SRC: 0x0004,
      COPY_DST: 0x0008,
      INDEX: 0x0010,
      VERTEX: 0x0020,
      UNIFORM: 0x0040,
      STORAGE: 0x0080,
      INDIRECT: 0x0100,
      QUERY_RESOLVE: 0x0200,
    };

/** Usage flags shared by every storage buffer the framework creates. */
export const STORAGE_USAGE =
  GPUBufferUsageFlags.STORAGE | GPUBufferUsageFlags.COPY_SRC | GPUBufferUsageFlags.COPY_DST;

export const UNIFORM_USAGE = GPUBufferUsageFlags.UNIFORM | GPUBufferUsageFlags.COPY_DST;

export const STAGING_USAGE = GPUBufferUsageFlags.MAP_READ | GPUBufferUsageFlags.COPY_DST;

/** Round a byte size up to the WebGPU-required 4-byte alignment. */
export function align4(n: number): number {
  return (n + 3) & ~3;
}

/** Round a byte size up to 16 bytes (recommended uniform buffer alignment). */
export function align16(n: number): number {
  return (n + 15) & ~15;
}

/**
 * A GPU storage buffer owned by the framework. Wraps a raw GPUBuffer and
 * remembers its logical element type so kernels can bind it correctly.
 */
export class GpuDataBuffer {
  readonly buffer: GPUBuffer;
  /** Number of elements (of `dtype`) this buffer holds. */
  readonly elements: number;
  readonly dtype: DType;

  constructor(buffer: GPUBuffer, elements: number, dtype: DType) {
    this.buffer = buffer;
    this.elements = elements;
    this.dtype = dtype;
  }

  get byteSize(): number {
    return this.elements * dtypeInfo(this.dtype).bytes;
  }

  destroy(): void {
    this.buffer.destroy();
  }
}

/** Default ceiling for idle pooled buffers. Purely a memory budget, not a
 *  correctness limit: buffers beyond the budget are destroyed on release. */
export const DEFAULT_MAX_POOLED_BYTES = 64 * 1024 * 1024;

export interface BufferPoolOptions {
  /** Maximum total bytes kept idle in the pool. 0 disables pooling entirely. */
  maxPooledBytes?: number;
}

/**
 * Size-bucketed buffer pool. Buffers are recycled by *byte size bucket*
 * (power-of-two buckets), so repeated dispatch shapes hit the cache.
 *
 * The pool is bounded by a byte budget: without one, a long-running session
 * that allocates many distinct shapes would keep every returned buffer alive
 * forever (each distinct bucket grows its own unbounded list) until the GPU
 * process runs out of memory.
 */
export class BufferPool {
  private device: GPUDevice;
  /** Storage buckets and uniform buckets are kept strictly separate: a
   *  UNIFORM-only buffer must never be rebound as storage (and vice versa),
   *  which would be a validation error and a silent no-op dispatch. */
  private freeStorage = new Map<number, GPUBuffer[]>();
  private freeUniform = new Map<number, GPUBuffer[]>();
  /** Buffers currently checked out (not sitting idle in the pool). */
  private checkedOut = 0;
  private poolCount = 0;
  private pooledBytes = 0;
  private readonly maxPooledBytes: number;

  constructor(device: GPUDevice, options: BufferPoolOptions = {}) {
    this.device = device;
    const cap = options.maxPooledBytes ?? DEFAULT_MAX_POOLED_BYTES;
    this.maxPooledBytes = Number.isFinite(cap) ? Math.max(0, cap) : DEFAULT_MAX_POOLED_BYTES;
  }

  /** Acquire a storage buffer of at least `elements` elements of `dtype`. */
  acquire(elements: number, dtype: DType): GpuDataBuffer {
    const byteSize = align4(Math.max(4, elements * dtypeInfo(dtype).bytes));
    const bucket = nextPow2(byteSize);
    let bucketList = this.freeStorage.get(bucket);
    if (!bucketList || bucketList.length === 0) {
      const buffer = this.device.createBuffer({
        size: bucket,
        usage: STORAGE_USAGE,
        label: `moxwebgpu-pool-${bucket}`,
      });
      this.checkedOut++;
      return new GpuDataBuffer(buffer, elements, dtype);
    }
    const buffer = bucketList.pop()!;
    this.poolCount--;
    this.pooledBytes -= buffer.size;
    this.checkedOut++;
    return new GpuDataBuffer(buffer, elements, dtype);
  }

  /**
   * Acquire a raw uniform buffer of at least `bytes` bytes.
   * Uniform blocks are pooled too: reusing a UBO on the same queue timeline
   * is safe, and avoids churning buffer allocations per dispatch.
   */
  acquireUniform(bytes: number): GPUBuffer {
    const bucket = nextPow2(Math.max(256, bytes));
    let bucketList = this.freeUniform.get(bucket);
    if (!bucketList || bucketList.length === 0) {
      const buffer = this.device.createBuffer({
        size: bucket,
        usage: UNIFORM_USAGE,
        label: `moxwebgpu-ubo-${bucket}`,
      });
      this.checkedOut++;
      return buffer;
    }
    const buffer = bucketList.pop()!;
    this.poolCount--;
    this.pooledBytes -= buffer.size;
    this.checkedOut++;
    return buffer;
  }

  /**
   * Return a raw buffer. If the byte budget would be exceeded, destroy it
   * instead of keeping it idle — `checkedOut` drops either way.
   */
  releaseRaw(buffer: GPUBuffer, usage: 'storage' | 'uniform' = 'storage'): void {
    this.checkedOut = Math.max(0, this.checkedOut - 1);
    if (this.pooledBytes + buffer.size > this.maxPooledBytes) {
      buffer.destroy();
      return;
    }
    const map = usage === 'uniform' ? this.freeUniform : this.freeStorage;
    const bucket = nextPow2(buffer.size);
    let bucketList = map.get(bucket);
    if (!bucketList) {
      bucketList = [];
      map.set(bucket, bucketList);
    }
    bucketList.push(buffer);
    this.poolCount++;
    this.pooledBytes += buffer.size;
  }

  /** Return a buffer to the pool for recycling. */
  release(buf: GpuDataBuffer): void {
    this.releaseRaw(buf.buffer, 'storage');
  }

  /** Release a uniform buffer previously handed out by acquireUniform. */
  releaseUniform(buffer: GPUBuffer): void {
    this.releaseRaw(buffer, 'uniform');
  }

  /** Destroy every pooled buffer. */
  clear(): void {
    for (const list of this.freeStorage.values()) {
      for (const b of list) b.destroy();
    }
    for (const list of this.freeUniform.values()) {
      for (const b of list) b.destroy();
    }
    this.freeStorage.clear();
    this.freeUniform.clear();
    this.poolCount = 0;
    this.pooledBytes = 0;
  }

  /** Buffers currently checked out of the pool (not idle). */
  get live(): number {
    return this.checkedOut;
  }

  /** Buffers sitting idle in the pool. */
  get pooled(): number {
    return this.poolCount;
  }

  /** Bytes currently retained by idle buffers (never exceeds the budget). */
  get pooledBytesUsed(): number {
    return this.pooledBytes;
  }

  /** Configured idle-byte ceiling. */
  get pooledBytesLimit(): number {
    return this.maxPooledBytes;
  }

  destroy(): void {
    this.clear();
  }
}

function nextPow2(n: number): number {
  let p = 16;
  while (p < n) p *= 2;
  return p;
}
