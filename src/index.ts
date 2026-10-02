/**
 * moxwebgpu — a WebGPU compute (GPGPU) framework.
 *
 * Two layers, one call style:
 *   - High level: lazy, chainable tensors with a full op set
 *   - Low level: raw WGSL kernels with pipeline caching
 *
 *   const gpu = await mox.init();
 *   const c = await gpu.tensor([[1,2],[3,4]]).matmul(gpu.tensor([[5,6],[7,8]])).toArray();
 */

// ---- core ----
export { MoxContext, mox, type MoxInitOptions } from './core/context.js';
export { Kernel, type KernelOptions, type RunOptions } from './core/kernel.js';
export {
  BufferPool, GpuDataBuffer, STORAGE_USAGE, UNIFORM_USAGE, STAGING_USAGE,
} from './core/buffer.js';
export {
  DTYPES, dtypeInfo, makeTypedArray, asTypedArray, numElements,
  type DType, type DTypeInfo,
} from './core/dtype.js';

// ---- tensor ----
export { Tensor, type TensorLike, type TensorOptions } from './tensor/tensor.js';
export type { OpDef, KernelStep, LazyNode, OpPlan, BindingSpec } from './graph/lazy.js';
export { createNode, topoSort } from './graph/lazy.js';
export { Scheduler } from './graph/scheduler.js';
export { PipelineCache } from './graph/pipelineCache.js';

// ---- codegen (for building custom ops) ----
export * from './tensor/codegen.js';

// ---- version ----
export const MOXWEBGPU_VERSION = '0.2.0';

import { Tensor } from './tensor/tensor.js';
import { installTensorOps } from './tensor/ops/index.js';

// Install the unified chainable op API (add/matmul/sum/softmax/...) once.
installTensorOps(Tensor.prototype);
