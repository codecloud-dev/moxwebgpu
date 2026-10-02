import { describe, it, expect } from 'vitest';
import { broadcastMode, unaryOpDef, castOpDef } from '../../src/tensor/ops/elementwise.js';
import {
  encodeNUniform,
  encode2DUniform,
  encodeMatmulUniform,
  encodeCopyUniform,
  contiguousStrides,
  fmtF32,
} from '../../src/tensor/codegen.js';
import { createNode, topoSort, type OpDef } from '../../src/graph/lazy.js';
import { matmulDef } from '../../src/tensor/ops/matmul.js';
import { concatDef } from '../../src/tensor/ops/shape.js';
import { axisReduceOpDef, normAxis } from '../../src/tensor/ops/reduce.js';

describe('broadcastMode', () => {
  it('plain when shapes match', () => {
    expect(broadcastMode([2, 3], [2, 3])).toBe('plain');
    expect(broadcastMode([5], [5])).toBe('plain');
  });
  it('row broadcast when b.size == last dim', () => {
    expect(broadcastMode([2, 3], [3])).toBe('row');
    expect(broadcastMode([4, 5, 6], [6])).toBe('row');
  });
  it('col broadcast when b.size == first dim (rank>=2)', () => {
    expect(broadcastMode([2, 3], [2])).toBe('col');
  });
  it('throws on incompatible shapes', () => {
    expect(() => broadcastMode([2, 3], [4])).toThrow();
  });
});

describe('uniform encoders', () => {
  it('encodeNUniform packs n and scalar', () => {
    const dv = new DataView(encodeNUniform(42, 1.5));
    expect(dv.getUint32(0, true)).toBe(42);
    expect(dv.getFloat32(8, true)).toBeCloseTo(1.5, 5);
  });
  it('encode2DUniform packs rows/cols/scalar', () => {
    const dv = new DataView(encode2DUniform(3, 7, 0.25));
    expect(dv.getUint32(0, true)).toBe(3);
    expect(dv.getUint32(4, true)).toBe(7);
    expect(dv.getFloat32(8, true)).toBeCloseTo(0.25, 5);
  });
  it('encodeMatmulUniform packs m/n/k', () => {
    const dv = new DataView(encodeMatmulUniform(8, 9, 10));
    expect(dv.getUint32(0, true)).toBe(8);
    expect(dv.getUint32(4, true)).toBe(9);
    expect(dv.getUint32(8, true)).toBe(10);
  });
  it('encodeCopyUniform lays out vec4 dims + scalars in 64 bytes', () => {
    const shape = [2, 3];
    const inStrides = [3, 1];
    const outStrides = [3, 1];
    const buf = encodeCopyUniform(shape, inStrides, outStrides, 10, 20);
    expect(buf.byteLength).toBe(64);
    const dv = new DataView(buf);
    // outShape vec4 at 0..15, inStrides vec4 at 16..31, outStrides at 32..47
    expect(dv.getUint32(0, true)).toBe(2);
    expect(dv.getUint32(4, true)).toBe(3);
    expect(dv.getUint32(16, true)).toBe(3);
    expect(dv.getUint32(20, true)).toBe(1);
    expect(dv.getUint32(48, true)).toBe(10); // inOffset
    expect(dv.getUint32(52, true)).toBe(20); // outOffset
    expect(dv.getUint32(56, true)).toBe(6);  // total
    expect(dv.getUint32(60, true)).toBe(2);  // rank
  });
  it('contiguousStrides is row-major', () => {
    expect(contiguousStrides([2, 3, 4])).toEqual([12, 4, 1]);
    expect(contiguousStrides([5])).toEqual([1]);
  });
});

describe('fmtF32', () => {
  it('keeps the decimal point', () => {
    expect(fmtF32(1)).toBe('1.0');
    expect(fmtF32(-2)).toBe('-2.0');
    expect(fmtF32(0.5)).toBe('0.5');
  });
  it('rejects NaN/±Infinity (not legal WGSL const-expressions)', () => {
    expect(() => fmtF32(Infinity)).toThrow();
    expect(() => fmtF32(-Infinity)).toThrow();
    expect(() => fmtF32(NaN)).toThrow();
  });
});

describe('lazy graph', () => {
  const fakeOp: OpDef = {
    name: 'fake',
    outShape: (s) => s[0],
    build: () => ({ steps: [] }),
  };

  function leaf(shape: number[], name: string): any {
    return { shape, dtype: 'f32', node: null, ctx: {}, name };
  }

  it('topoSort orders parents before children and counts consumers', () => {
    const a = leaf([2], 'a');
    const b = leaf([2], 'b');
    const ab = createNode(fakeOp, [a, b]); // inputs: a, b
    // Wrap the node in a Tensor-like object the way Tensor.apply does.
    const abTensor = { ctx: {}, shape: [2], dtype: 'f32', node: ab, data: null };
    const abc = createNode(fakeOp, [abTensor, a]);

    const order = topoSort(abc);
    expect(order.length).toBe(2);
    expect(order[0]).toBe(ab);
    expect(order[1]).toBe(abc);
    expect(ab.consumers).toBe(1);
  });

  it('createNode derives shape/dtype from op', () => {
    const a = leaf([2, 3], 'a');
    const n = createNode(matmulDef, [a, leaf([3, 4], 'b')] as any);
    expect(n.shape).toEqual([2, 4]);
    expect(n.dtype).toBe('f32');
    expect(n.kind).toBe('node');
  });

  it('matmulDef rejects mismatched inner dims', () => {
    expect(() => matmulDef.outShape!([[2, 3], [4, 5]], {})).toThrow();
  });

  it('concatDef outShape sums the axis', () => {
    expect(concatDef.outShape!([[2, 3], [1, 3]], { axis: 0 })).toEqual([3, 3]);
    expect(concatDef.outShape!([[2, 3], [2, 2]], { axis: 1 })).toEqual([2, 5]);
  });

  it('castOpDef overrides outDtype', () => {
    expect(castOpDef('i32').outDtype!(["f32" as any], {})).toBe('i32');
  });

  it('unaryOpDef keeps input dtype', () => {
    const def = unaryOpDef('x', (a) => a);
    const n = createNode(def, [leaf([4], 'a')] as any);
    expect(n.dtype).toBe('f32');
    expect(n.shape).toEqual([4]);
  });
});

describe('axisReduceOpDef (generic any-axis)', () => {
  const def = axisReduceOpDef('sum', { identityValue: 0, combine: (a, b) => `(${a} + ${b})` });

  it('outShape drops the requested axis', () => {
    expect(def.outShape!([[2, 3, 4]], { axis: 1 })).toEqual([2, 4]);
    expect(def.outShape!([[2, 3, 4]], { axis: 0 })).toEqual([3, 4]);
    expect(def.outShape!([[2, 3, 4]], { axis: 2 })).toEqual([2, 3]);
  });

  it('negative axes normalize from the end', () => {
    expect(def.outShape!([[2, 3, 4]], { axis: -1 })).toEqual([2, 3]);
    expect(def.outShape!([[2, 3, 4]], { axis: -3 })).toEqual([3, 4]);
  });

  it('rank-1 input degenerates to [1]', () => {
    expect(def.outShape!([[5]], { axis: 0 })).toEqual([1]);
  });

  it('normAxis handles out-of-range negatives and truncation', () => {
    expect(normAxis(-1, 3)).toBe(2);
    expect(normAxis(-4, 3)).toBe(2);
    expect(normAxis(3, 3)).toBe(0);
    expect(normAxis(1.7, 3)).toBe(1);
  });
});
