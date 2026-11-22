/** WebGL2 fallback backend, public surface. */

export { detectBackend, hasWebGl2, type BackendKind, type BackendInfo } from './detect.js';
export { GlRuntime, layout } from './runtime.js';
export { GlTensor, type GlShape } from './tensor.js';
