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
 */
export function sumTo(t: Tensor, shape: number[]): Tensor {
  let cur = t;
  let curShape = cur.shape;
  while (curShape.length > shape.length) {
    cur = cur.sum(0);
    curShape = cur.shape;
  }
  const outShape: number[] = [];
  for (let d = 0; d < shape.length; d++) {
    if (curShape[d] !== shape[d]) {
      cur = cur.sum(d);
      curShape = cur.shape;
      outShape.push(1);
    } else {
      outShape.push(curShape[d]);
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
