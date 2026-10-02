/**
 * Supported tensor dtypes and their WebGPU mappings.
 */

export type DType = 'f32' | 'i32' | 'u32';

export interface DTypeInfo {
  /** WGSL type keyword */
  wgsl: string;
  /** Bytes per element */
  bytes: number;
  /** Corresponding JS TypedArray constructor */
  array: Float32ArrayConstructor | Int32ArrayConstructor | Uint32ArrayConstructor;
}

export const DTYPES: Record<DType, DTypeInfo> = {
  f32: { wgsl: 'f32', bytes: 4, array: Float32Array },
  i32: { wgsl: 'i32', bytes: 4, array: Int32Array },
  u32: { wgsl: 'u32', bytes: 4, array: Uint32Array },
};

export function dtypeInfo(dtype: DType): DTypeInfo {
  return DTYPES[dtype];
}

export function makeTypedArray(dtype: DType, length: number): Float32Array | Int32Array | Uint32Array {
  return new DTYPES[dtype].array(length) as Float32Array;
}

export function asTypedArray(dtype: DType, data: ArrayLike<number>): Float32Array | Int32Array | Uint32Array {
  const arr = makeTypedArray(dtype, data.length);
  for (let i = 0; i < data.length; i++) arr[i] = data[i] as number;
  return arr;
}

export function numElements(shape: readonly number[]): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}
