/** CPU reference implementations used to verify GPU results. */

export function cpuMatmul(a: Float32Array | number[], b: Float32Array | number[], m: number, k: number, n: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let p = 0; p < k; p++) acc += (a as any)[i * k + p] * (b as any)[p * n + j];
      out[i * n + j] = acc;
    }
  }
  return out;
}

export function cpuSoftmaxLastAxis(x: ArrayLike<number>, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let c = 0; c < cols; c++) max = Math.max(max, x[r * cols + c]);
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      const e = Math.exp(x[r * cols + c] - max);
      out[r * cols + c] = e;
      sum += e;
    }
    for (let c = 0; c < cols; c++) out[r * cols + c] /= sum;
  }
  return out;
}

export function cpuArgmax(x: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] > x[best]) best = i;
  }
  return best;
}

export function cpuArgmin(x: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i] < x[best]) best = i;
  }
  return best;
}
