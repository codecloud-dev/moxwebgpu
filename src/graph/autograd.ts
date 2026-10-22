/**
 * Reverse-mode automatic differentiation.
 *
 * The gradient graph is built lazily on top of the existing lazy DAG: every
 * `BackwardFn` returns new lazy Tensors via the normal op set, so no GPU work
 * happens until a gradient is materialized with `.toArray()` / `.toBuffer()`.
 *
 * `backward(loss)` fills `.grad` on every leaf Tensor whose `requiresGrad` is
 * true. The loss is treated as a scalar (seed = 1); pass a pre-reduced scalar
 * loss for the usual "dL/dparams" semantics.
 */

import { Tensor } from '../tensor/tensor.js';
import type { MoxContext } from '../core/context.js';
import type { LazyNode } from './lazy.js';
import { numElements } from '../core/dtype.js';

/** Reverse topological order (parents before children); does not mutate state. */
function reverseTopo(root: LazyNode): LazyNode[] {
  const order: LazyNode[] = [];
  const seen = new Set<LazyNode>();
  const visit = (n: LazyNode): void => {
    if (seen.has(n)) return;
    seen.add(n);
    for (const inp of n.inputs) {
      if (inp.kind !== 'leaf') visit(inp);
    }
    order.push(n);
  };
  visit(root);
  return order;
}

/** A lazy ones-tensor shaped like `t` (no GPU upload until materialized). */
function onesLike(t: Tensor): Tensor {
  return t.toFloat().mul(0).add(1);
}

/**
 * Reduce `t` to `shape` by summing out broadcasted / leading dimensions.
 * Used to push a gradient back to a smaller broadcast operand's shape.
 *
 * Rank reduction direction matters: a row-broadcast operand's grad must sum
 * the *leading* axes ([M,K] -> [K]), a column-broadcast operand's grad must
 * sum the *trailing* axes ([M,K] -> [M]). The leading dim decides: if it
 * already matches the target, the extra axes are trailing, else leading.
 */
export function sumTo(t: Tensor, shape: number[]): Tensor {
  let cur = t;
  if (shape.length === 0) {
    // Scalar target (0-dim leaf): fold every axis out, then view as 0-dim.
    // (sum() bottoms out at rank 1, never 0, so reshape does the last step.)
    while (cur.shape.length > 1) cur = cur.sum(0);
    return cur.reshape();
  }
  while (cur.shape.length > shape.length) {
    const dropLast = shape.length > 0 && cur.shape[0] === shape[0];
    cur = dropLast ? cur.sum(cur.shape.length - 1) : cur.sum(0);
  }
  // Walk from the last axis so summing one axis never shifts the indices
  // of the axes still to be checked.
  const outShape: number[] = [];
  for (let d = shape.length - 1; d >= 0; d--) {
    if (cur.shape[d] !== shape[d]) {
      cur = cur.sum(d);
      outShape.unshift(1);
    } else {
      outShape.unshift(shape[d]);
    }
  }
  return cur.reshape(...outShape);
}

function wrap(node: LazyNode, ctx: MoxContext): Tensor {
  return new Tensor(ctx, node.shape, node.dtype, node, null);
}

export function backward(loss: Tensor): void {
  const root = loss.node;
  if (!root) return; // leaf loss: nothing to differentiate

  const order = reverseTopo(root);
  const gradOf = new Map<object, Tensor>();
  gradOf.set(root, onesLike(loss));

  for (let i = order.length - 1; i >= 0; i--) {
    const node = order[i];
    const g = gradOf.get(node);
    if (!g) continue;

    const inputs: Tensor[] = node.inputs.map((inp) => {
      if (inp.kind === 'leaf') return inp.tensor as Tensor;
      return wrap(inp as LazyNode, loss.ctx);
    });
    const output = wrap(node, loss.ctx);

    const fn = node.op.backward;
    const grads: (Tensor | null)[] = fn ? fn(inputs, output, g, node.attrs) : inputs.map(() => null);

    for (let j = 0; j < node.inputs.length; j++) {
      const gi = grads[j];
      if (!gi) continue;
      let target: object;
      if (node.inputs[j].kind === 'leaf') {
        target = (node.inputs[j] as { kind: 'leaf'; tensor: Tensor }).tensor;
      } else {
        target = node.inputs[j] as LazyNode;
      }
      const prev = gradOf.get(target);
      gradOf.set(target, prev ? prev.add(gi) : gi);
    }
  }

  for (const [key, g] of gradOf) {
    if (key instanceof Tensor && key.requiresGrad) {
      key.grad = g;
    }
  }
}
