/**
 * Kernel: the low-level escape hatch. Wrap raw WGSL compute shaders and
 * dispatch them directly, with pipeline caching and an optional readback.
 *
 * Binding convention: your WGSL declares `@group(0) @binding(i)` entries in
 * the same order as the buffers you pass to `run()` / `dispatch()`.
 */

import type { MoxContext } from './context.js';
import type { GpuDataBuffer } from './buffer.js';
import { STAGING_USAGE } from './buffer.js';
import { DTYPES, type DType, makeTypedArray } from './dtype.js';

export interface KernelOptions {
  /** Workgroup size along x used for `run(elements)` auto-gridding. Default 64. */
  workgroupSize?: number;
  /** WGSL entry point. Default "main". */
  entryPoint?: string;
  /** Which binding's buffer `run()` reads back. Default: the last buffer. */
  output?: number;
  /** Typed array flavour returned by `run()`. Default "f32". */
  dtype?: DType;
}

export interface RunOptions {
  /** Total elements along x; grid becomes ceil(elements / workgroupSize). */
  elements?: number;
  /** Explicit workgroup grid — takes precedence over `elements`. */
  workgroups?: [number, number, number];
}

function raw(buffer: GPUBuffer | GpuDataBuffer): GPUBuffer {
  return buffer instanceof GPUBuffer ? buffer : buffer.buffer;
}

export class Kernel {
  constructor(
    private readonly ctx: MoxContext,
    /** Full WGSL source of the compute shader. */
    public readonly code: string,
    public readonly options: KernelOptions = {},
  ) {}

  private get workgroupSize(): number {
    return this.options.workgroupSize ?? 64;
  }

  private get entryPoint(): string {
    return this.options.entryPoint ?? 'main';
  }

  /**
   * Submit the kernel without reading anything back. Fires one compute pass.
   */
  dispatch(buffers: (GPUBuffer | GpuDataBuffer)[], workgroups: [number, number, number]): void {
    const { device } = this.ctx;
    const pipeline = this.ctx.pipelines.get(this.code, this.entryPoint);
    const entries = buffers.map((b, i) => ({ binding: i, resource: { buffer: raw(b) } }));
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(workgroups[0], workgroups[1] || 1, workgroups[2] || 1);
    pass.end();
    this.ctx.queue.submit([encoder.finish()]);
  }

  /**
   * Submit the kernel and read back the output binding as a typed array.
   * The read-back buffer must be GPUBufferUsage.COPY_SRC capable (all
   * moxwebgpu storage buffers are).
   */
  async run(buffers: (GPUBuffer | GpuDataBuffer)[], opts: RunOptions = {}): Promise<Float32Array | Int32Array | Uint32Array> {
    let wg: [number, number, number];
    if (opts.workgroups) {
      wg = opts.workgroups;
    } else if (opts.elements !== undefined) {
      wg = [Math.ceil(opts.elements / this.workgroupSize), 1, 1];
    } else {
      throw new Error('moxwebgpu: kernel.run() needs either `elements` or `workgroups`');
    }
    this.dispatch(buffers, wg);

    const outIndex = this.options.output ?? buffers.length - 1;
    const outBuf = buffers[outIndex];
    if (!outBuf) throw new Error(`moxwebgpu: no buffer at binding ${outIndex} to read back`);
    const gpuBuf = raw(outBuf);
    const dtype: DType = this.options.dtype ?? 'f32';
    // Prefer the logical element count recorded on pooled buffers: the raw
    // GPUBuffer is rounded up to a power-of-two bucket, so reading back its
    // full size would return padding elements. Bare GPUBuffers (no
    // `.elements`, e.g. ones created outside the pool) fall back to their
    // raw size. This matches scheduler.readback(), which always uses the
    // logical count.
    const logical =
      (outBuf as { elements?: number }).elements ??
      Math.floor(gpuBuf.size / DTYPES[dtype].bytes);
    const bytes = Math.min(gpuBuf.size, logical * DTYPES[dtype].bytes);
    const elements = Math.floor(bytes / DTYPES[dtype].bytes);

    const { device, queue } = this.ctx;
    const staging = device.createBuffer({ size: bytes, usage: STAGING_USAGE, label: 'moxwebgpu-kernel-staging' });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(gpuBuf, 0, staging, 0, bytes);
    queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const Ctor = DTYPES[dtype].array as any;
    const mapped = new Uint8Array(staging.getMappedRange());
    const out = makeTypedArray(dtype, elements);
    out.set(new Ctor(mapped.buffer, mapped.byteOffset, elements));
    staging.unmap();
    staging.destroy();
    return out;
  }
}
