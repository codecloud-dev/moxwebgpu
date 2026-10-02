/**
 * MoxContext: one GPU device + every cache the framework needs.
 * Created through `mox.init()` — the single entry point of the library.
 */

import { BufferPool } from './buffer.js';
import { PipelineCache } from '../graph/pipelineCache.js';
import { Scheduler } from '../graph/scheduler.js';
import { Tensor, type TensorLike, type TensorOptions } from '../tensor/tensor.js';
import { Kernel, type KernelOptions } from './kernel.js';
import type { GpuDataBuffer } from './buffer.js';
import type { DType } from './dtype.js';

export interface MoxInitOptions {
  /** Provide your own adapter (skip requestAdapter). */
  adapter?: GPUAdapter;
  powerPreference?: GPUPowerPreference;
  /** Extra options forwarded to requestAdapter. */
  requestAdapterOptions?: GPURequestAdapterOptions;
}

export class MoxContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly queue: GPUQueue;
  readonly pool: BufferPool;
  readonly pipelines: PipelineCache;
  readonly scheduler: Scheduler;

  private constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter;
    this.device = device;
    this.queue = device.queue;
    this.pool = new BufferPool(device);
    this.pipelines = new PipelineCache(device);
    this.scheduler = new Scheduler(this);
  }

  /** Request an adapter + device and build the context. */
  static async init(options: MoxInitOptions = {}): Promise<MoxContext> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) {
      throw new Error(
        'moxwebgpu: WebGPU is not available here. moxwebgpu targets browsers with WebGPU (Chrome/Edge 113+). ' +
          'For Node.js, run inside a WebGPU-enabled runtime.',
      );
    }
    const adapter =
      options.adapter ??
      (await navigator.gpu.requestAdapter({
        powerPreference: options.powerPreference,
        ...(options.requestAdapterOptions ?? {}),
      }));
    if (!adapter) {
      throw new Error('moxwebgpu: no suitable GPU adapter (WebGPU device unavailable or blocklisted)');
    }
    const device = await adapter.requestDevice();
    return new MoxContext(adapter, device);
  }

  /** Create a leaf tensor from JS data. */
  tensor(data: TensorLike, options: TensorOptions = {}): Tensor {
    return Tensor.fromData(this, data, options);
  }

  /** Wrap a raw WGSL compute kernel. */
  kernel(code: string, options: KernelOptions = {}): Kernel {
    return new Kernel(this, code, options);
  }

  /** Copy a GPU buffer back to the CPU. */
  async readback(buf: GpuDataBuffer): Promise<Float32Array | Int32Array | Uint32Array> {
    return this.scheduler.readback(buf);
  }

  /** Adapter summary for logs / about screens. */
  info(): { vendor?: string; architecture?: string; device?: string; description?: string } {
    const info = (this.adapter as any).info ?? {};
    return {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
    };
  }

  /** Release every pooled buffer and destroy the device. */
  destroy(): void {
    this.pool.destroy();
    this.device.destroy();
  }
}

/** Public entry: `const gpu = await mox.init()`. */
export const mox = {
  init: (options: MoxInitOptions = {}): Promise<MoxContext> => MoxContext.init(options),
};
