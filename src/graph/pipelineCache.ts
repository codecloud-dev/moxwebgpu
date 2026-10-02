/**
 * Pipeline cache: WGSL -> GPUComputePipeline, memoized by shader hash so
 * repeated ops of the same kind & dtype reuse compiled pipelines.
 */

import { hashString } from '../util/hash.js';

export class PipelineCache {
  private device: GPUDevice;
  private cache = new Map<string, GPUComputePipeline>();

  constructor(device: GPUDevice) {
    this.device = device;
  }

  /**
   * Get (or compile) a pipeline for the given WGSL source + entry point.
   * The cache key is derived from the source text, so identical kernels
   * share one pipeline regardless of which op instance produced them.
   */
  get(wgsl: string, entryPoint = 'main'): GPUComputePipeline {
    const key = `${hashString(wgsl)}:${entryPoint}`;
    let pipeline = this.cache.get(key);
    if (!pipeline) {
      const module = this.device.createShaderModule({ code: wgsl });
      // WebGPU validation failures are silent by design (invalid object +
      // no-op dispatch). Surface shader compile errors loudly instead.
      module.getCompilationInfo?.().then((info) => {
        const errors = info.messages.filter((m) => m.type === 'error');
        if (errors.length > 0) {
          const detail = errors.map((m) => `  [${m.lineNum}:${m.linePos}] ${m.message}`).join('\n');
          console.error(`moxwebgpu: WGSL compile error(s):\n${detail}\n--- generated shader ---\n${wgsl}`);
        }
      }).catch(() => {});
      pipeline = this.device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint },
      });
      this.cache.set(key, pipeline);
    }
    return pipeline;
  }

  /** Number of compiled pipelines currently cached. */
  get size(): number {
    return this.cache.size;
  }
}
