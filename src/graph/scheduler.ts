/**
 * The scheduler: walks a lazy DAG topologically and submits GPU passes.
 *
 * Resource strategy:
 *  - input leaf tensors upload once (data lives on the Tensor)
 *  - intermediate buffers come from the pool and are recycled as soon as
 *    their consumer count drops to zero
 *  - executed nodes are "materialized": their buffer is attached to the
 *    node, so re-reading a tensor never recomputes
 */

import type { MoxContext } from '../core/context.js';
import type { GpuDataBuffer } from '../core/buffer.js';
import type { Tensor } from '../tensor/tensor.js';
import { numElements, makeTypedArray, DTYPES, type DType } from '../core/dtype.js';
import type { LazyNode, BindingSpec } from './lazy.js';
import { topoSort } from './lazy.js';

export class Scheduler {
  private ctx: MoxContext;

  constructor(ctx: MoxContext) {
    this.ctx = ctx;
  }

  /**
   * Execute the subgraph rooted at `t` (if needed) and return its GPU buffer.
   * Idempotent: already-materialized tensors return their existing buffer.
   */
  materialize(t: Tensor): GpuDataBuffer {
    if (t.data) return t.data;
    const root = t.node;
    if (!root) throw new Error('moxwebgpu: tensor has neither data nor a lazy node');
    if (root.buffer) return root.buffer;

    const order = topoSort(root);
    for (const n of order) {
      if (!n.buffer) this.runNode(n);
    }
    return root.buffer!;
  }

  private runNode(n: LazyNode): void {
    const { device, queue } = this.ctx;

    // Resolve input buffers (materializing leaf tensors on first use).
    // Shapes/dtypes follow the original input order — lazy nodes included.
    const inputTensors: Tensor[] = [];
    const inputBuffers: GpuDataBuffer[] = [];
    const nodeInputs: LazyNode[] = [];
    const inputShapes: number[][] = [];
    const inputDtypes: DType[] = [];
    for (const input of n.inputs) {
      if (input.kind === 'leaf') {
        const t = input.tensor as Tensor;
        inputTensors.push(t);
        inputBuffers.push(this.ctx.scheduler.materialize(t));
        inputShapes.push(t.shape);
        inputDtypes.push(t.dtype);
      } else {
        nodeInputs.push(input);
        inputBuffers.push(input.buffer!);
        inputShapes.push(input.shape);
        inputDtypes.push(input.dtype);
      }
    }
    const plan = n.op.build(inputShapes, inputDtypes, n.attrs);

    const output = this.ctx.pool.acquire(Math.max(1, numElements(n.shape)), n.dtype);
    let prevTemp: GpuDataBuffer | null = null;
    const scratchUbos: GPUBuffer[] = [];

    const encoder = device.createCommandEncoder();

    for (let s = 0; s < plan.steps.length; s++) {
      const step = plan.steps[s];
      const isLast = s === plan.steps.length - 1;
      // A step writes the final output if it is the last step, or if it
      // explicitly declares an `output` binding (e.g. concat writes every
      // chunk straight into the output at different offsets).
      const writesOutput =
        isLast || step.bindings.some((b) => b.kind === 'rw' && (b as any).output === true);
      const outBuf = writesOutput
        ? output
        : this.ctx.pool.acquire(
            Math.max(1, step.tempOutputElements!(n.shape)),
            step.tempOutputDtype ?? n.dtype,
          );

      const entries: GPUBindGroupEntry[] = [];
      for (const b of step.bindings as BindingSpec[]) {
        const bindingIndex = entries.length;
        if (b.kind === 'uniform') {
          const data = step.uniforms!();
          const ubo = this.ctx.pool.acquireUniform(data.byteLength);
          queue.writeBuffer(ubo, 0, data, 0, data.byteLength);
          scratchUbos.push(ubo);
          entries.push({ binding: bindingIndex, resource: { buffer: ubo } });
        } else if (b.kind === 'read') {
          const buf = b.temp ? prevTemp! : inputBuffers[b.input!];
          entries.push({ binding: bindingIndex, resource: { buffer: buf.buffer } });
        } else {
          if (b.temp) {
            prevTemp = outBuf; // this step writes the temp the next step reads
          }
          entries.push({ binding: bindingIndex, resource: { buffer: outBuf.buffer } });
        }
      }

      const pipeline = this.ctx.pipelines.get(step.wgsl, step.entryPoint ?? 'main');
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries,
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      const [wx, wy, wz] = step.workgroups(n.shape);
      pass.dispatchWorkgroups(wx, wy || 1, wz || 1);
      pass.end();
    }

    queue.submit([encoder.finish()]);

    // Recycle the per-dispatch uniform blocks (same-queue-timeline reuse is
    // safe) and inputs whose consumers are exhausted, plus the final
    // intermediate temp buffer.
    for (const ubo of scratchUbos) {
      this.ctx.pool.releaseUniform(ubo);
    }
    for (const nodeInput of nodeInputs) {
      nodeInput.consumers--;
      if (nodeInput.consumers === 0 && nodeInput.buffer) {
        this.ctx.pool.release(nodeInput.buffer);
        nodeInput.buffer = null;
      }
    }
    if (prevTemp) {
      this.ctx.pool.release(prevTemp);
      prevTemp = null;
    }

    n.buffer = output;
  }

  /** Copy a GPU buffer back to the CPU as a typed array. */
  async readback(buf: GpuDataBuffer): Promise<Float32Array | Int32Array | Uint32Array> {
    const { device, queue } = this.ctx;
    const bytes = buf.buffer.size;
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'moxwebgpu-staging',
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buf.buffer, 0, staging, 0, bytes);
    queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const Ctor = DTYPES[buf.dtype].array as any;
    const raw = new Uint8Array(staging.getMappedRange());
    const out = makeTypedArray(buf.dtype, buf.elements);
    out.set(new Ctor(raw.buffer, raw.byteOffset, buf.elements));
    staging.unmap();
    staging.destroy();
    return out;
  }
}
