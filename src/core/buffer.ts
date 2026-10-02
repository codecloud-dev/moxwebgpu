/**
 * GPU buffer management: a thin wrapper over GPUBuffer plus a size-bucketed
 * pool for recycling intermediate storage buffers between dispatches.
 */

import type { DType } from './dtype.js';
import { dtypeInfo } from './dtype.js';

/** Usage flags shared by every storage buffer the framework creates. */
export const STORAGE_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

export const UNIFORM_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

export const STAGING_USAGE = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

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

/**
 * Size-bucketed buffer pool. Buffers are recycled by *byte size bucket*
 * (power-of-two buckets), so repeated dispatch shapes hit the cache.
 */
export class BufferPool {
  private device: GPUDevice;
  /** Storage buckets and uniform buckets are kept strictly separate: a
   *  UNIFORM-only buffer must never be rebound as storage (and vice versa),
   *  which would be a validation error and a silent no-op dispatch. */
  private freeStorage = new Map<number, GPUBuffer[]>();
  private freeUniform = new Map<number, GPUBuffer[]>();
  private liveCount = 0;
  private poolCount = 0;

  constructor(device: GPUDevice) {
    this.device = device;
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
      this.liveCount++;
      return new GpuDataBuffer(buffer, elements, dtype);
    }
    const buffer = bucketList.pop()!;
    this.poolCount--;
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
      this.liveCount++;
      return buffer;
    }
    const buffer = bucketList.pop()!;
    this.poolCount--;
    return buffer;
  }

  /** Return a raw buffer to the pool. Usage must be given to pick the right bucket family. */
  releaseRaw(buffer: GPUBuffer, usage: 'storage' | 'uniform' = 'storage'): void {
    const map = usage === 'uniform' ? this.freeUniform : this.freeStorage;
    const bucket = nextPow2(buffer.size);
    let bucketList = map.get(bucket);
    if (!bucketList) {
      bucketList = [];
      map.set(bucket, bucketList);
    }
    bucketList.push(buffer);
    this.poolCount++;
  }

  /** Return a buffer to the pool for recycling. */
  release(buf: GpuDataBuffer): void {
    const bucket = nextPow2(buf.buffer.size);
    let bucketList = this.freeStorage.get(bucket);
    if (!bucketList) {
      bucketList = [];
      this.freeStorage.set(bucket, bucketList);
    }
    bucketList.push(buf.buffer);
    this.poolCount++;
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
  }

  /** Live buffers currently checked out of the pool. */
  get live(): number {
    return this.liveCount;
  }

  /** Buffers sitting idle in the pool. */
  get pooled(): number {
    return this.poolCount;
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
