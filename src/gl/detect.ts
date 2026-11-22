/**
 * Backend detection: pick WebGPU when available, otherwise fall back to
 * WebGL2, otherwise report unsupported. A three-state result keeps the
 * caller's decision explicit instead of throwing deep inside init().
 */

export type BackendKind = 'webgpu' | 'webgl2' | 'unsupported';

export interface BackendInfo {
  kind: BackendKind;
  /** Human-readable reason, useful for logs and "about" screens. */
  detail: string;
}

/** True when a WebGPU adapter can plausibly be requested. */
function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as any).gpu;
}

/** True when a WebGL2 context with float render targets can be created. */
export function hasWebGl2(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false });
    if (!gl) return false;
    // Float textures are required to store f32 tensor data. R32F/RGBA32F
    // renderability comes from EXT_color_buffer_float on WebGL2.
    const ok = !!gl.getExtension('EXT_color_buffer_float');
    const lose = gl.getExtension('WEBGL_lose_context');
    if (lose) lose.loseContext(); // release the probe context immediately
    return ok;
  } catch {
    return false;
  }
}

/**
 * Detect the best available backend at runtime.
 * `prefer` lets callers force a backend for testing; 'auto' picks WebGPU first.
 */
export function detectBackend(prefer: 'auto' | 'webgpu' | 'webgl2' = 'auto'): BackendInfo {
  if (prefer === 'webgpu') {
    return hasWebGpu()
      ? { kind: 'webgpu', detail: 'WebGPU adapter interface present' }
      : { kind: 'unsupported', detail: 'WebGPU requested but navigator.gpu is unavailable' };
  }
  if (prefer === 'webgl2') {
    return hasWebGl2()
      ? { kind: 'webgl2', detail: 'WebGL2 with float render targets' }
      : { kind: 'unsupported', detail: 'WebGL2 requested but float render targets unavailable' };
  }
  if (hasWebGpu()) return { kind: 'webgpu', detail: 'WebGPU adapter interface present' };
  if (hasWebGl2()) return { kind: 'webgl2', detail: 'WebGPU unavailable; fell back to WebGL2' };
  return { kind: 'unsupported', detail: 'Neither WebGPU nor WebGL2 float render targets are available' };
}
