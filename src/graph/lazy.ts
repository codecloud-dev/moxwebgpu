/**
 * Lazy compute graph.
 *
 * Tensor ops don't execute immediately: they build a LazyNode DAG.
 * Execution is deferred until `toArray()` / `toBuffer()` is called on the
 * root, at which point the scheduler walks the graph topologically and
 * submits the minimum number of GPU passes.
 */

import type { DType } from '../core/dtype.js';
import type { GpuDataBuffer } from '../core/buffer.js';
import type { Tensor } from '../tensor/tensor.js';

/**
 * Reverse-mode gradient of an op.
 *  - `inputs`  : the original input Tensors (leaves or node-backed), in order
 *  - `output`  : a Tensor view of this node's output
 *  - `gradOutput`: upstream gradient w.r.t. this node's output
 *  - returns one gradient Tensor (or null) per input, matching input order.
 * Gradients are themselves lazy tensors, so the whole backward graph reuses
 * the existing op set and only touches the GPU when materialized.
 */
export type BackwardFn = (
  inputs: Tensor[],
  output: Tensor,
  gradOutput: Tensor,
  attrs: any,
) => (Tensor | null)[];

/**
 * One GPU compute pass inside an op's execution plan.
 * `bindings` maps 1:1 to WGSL bindings, in binding order:
 *  - uniform: uploads the step's encoded uniform block
 *  - read:    `input: number` = n-th op input buffer, or `temp: true` =
 *             the previous step's intermediate output
 *  - rw:      `output: true` = the op's final output buffer, `temp: true` =
 *             a fresh intermediate buffer written by this step
 */
export type BindingSpec =
  | { kind: 'uniform' }
  | { kind: 'read'; input?: number; temp?: boolean }
  | { kind: 'rw'; output?: boolean; temp?: boolean };

export interface KernelStep {
  /** Stable key: identical steps share a compiled pipeline. */
  key: string;
  wgsl: string;
  entryPoint?: string;
  bindings: BindingSpec[];
  /** Encode the uniform block contents (for 'uniform' bindings). */
  uniforms?: () => ArrayBuffer;
  /** Workgroup grid for this step, given the op's output shape. */
  workgroups: (outShape: number[]) => [number, number, number];
  /**
   * For intermediate steps (all but the last), the element count of the
   * temporary buffer this step writes.
   */
  tempOutputElements?: (outShape: number[]) => number;
  tempOutputDtype?: DType;
}

export interface OpPlan {
  steps: KernelStep[];
}

/**
 * An operator: pure metadata + codegen. Ops are stateless so any node with
 * the same op/attrs can share compiled pipelines.
 */
export interface OpDef {
  name: string;
  /** Output shape given input shapes + attrs. */
  outShape: (inputShapes: number[][], attrs: any) => number[];
  /** Output dtype given input dtypes + attrs. */
  outDtype?: (inputDtypes: DType[], attrs: any) => DType;
  /** Build the GPU execution plan for this node. */
  build: (inputShapes: number[][], inputDtypes: DType[], attrs: any) => OpPlan;
  /** Optional reverse-mode gradient (see BackwardFn). Absent = no gradient. */
  backward?: BackwardFn;
}

/** A downstream reference: either another lazy node or a leaf tensor. */
export type NodeInput = LazyNode | { kind: 'leaf'; tensor: any };

/** A node in the lazy DAG. */
export interface LazyNode {
  /** Discriminator: always 'node' (leaves carry kind: 'leaf'). */
  kind: 'node';
  op: OpDef;
  attrs: any;
  inputs: NodeInput[];
  shape: number[];
  dtype: DType;
  /** Filled by the scheduler after execution. */
  buffer: GpuDataBuffer | null;
  /** Number of downstream consumers (set during graph collection). */
  consumers: number;
}

export function createNode(op: OpDef, inputs: any[], attrs: any = {}): LazyNode {
  const shapes = inputs.map((t) => t.shape);
  const dtypes = inputs.map((t) => t.dtype);
  const shape = op.outShape(shapes, attrs);
  const dtype = op.outDtype ? op.outDtype(dtypes, attrs) : dtypes[0];
  return {
    kind: 'node',
    op,
    attrs,
    // Lazy inputs reference their node; leaf tensors (node == null) wrap as leaves.
    inputs: inputs.map((t) => (t.node ? t.node : { kind: 'leaf', tensor: t })),
    shape,
    dtype,
    buffer: null,
    consumers: 0,
  };
}

/** Depth-first post-order topological sort of the subgraph under `root`. */
export function topoSort(root: LazyNode): LazyNode[] {
  const order: LazyNode[] = [];
  const visited = new Set<LazyNode>();
  const walk = (n: LazyNode): void => {
    if (visited.has(n)) return;
    visited.add(n);
    for (const input of n.inputs) {
      if (input.kind !== 'leaf') {
        input.consumers++;
        walk(input);
      }
    }
    order.push(n);
  };
  walk(root);
  return order;
}
