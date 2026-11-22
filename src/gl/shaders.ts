/**
 * GLSL ES 3.00 fragment shaders for the WebGL2 fallback backend.
 *
 * Tensors are stored as RGBA32F textures, one element per texel (R channel).
 * `uSize` is the logical element count; `uWidth` the texture width. A fragment
 * maps itself to a linear index via gl_FragCoord and bails out past uSize.
 */

export const VERT = `#version 300 es
in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
`;

const PRELUDE = `#version 300 es
precision highp float;
precision highp int;
uniform int uSize;
uniform int uWidth;
uniform sampler2D uA;
out vec4 outColor;
int idxOf(){ return int(gl_FragCoord.y) * uWidth + int(gl_FragCoord.x); }
vec4 fetchA(){ int i = idxOf(); return texelFetch(uA, ivec2(i % uWidth, i / uWidth), 0); }
`;

/** Unary / elementwise expression over A, optional scalar uScalar. */
export function unaryShader(expr: (v: string) => string): string {
  return `${PRELUDE}
uniform float uScalar;
void main(){
  int i = idxOf();
  if (i >= uSize) { outColor = vec4(0.0); return; }
  float a = fetchA().r;
  outColor = vec4(${expr('a')}, 0.0, 0.0, 1.0);
}`;
}

/** Binary elementwise A op B, both same length (broadcast handled by CPU upload). */
export function binaryShader(op: string): string {
  return `${PRELUDE}
uniform sampler2D uB;
void main(){
  int i = idxOf();
  if (i >= uSize) { outColor = vec4(0.0); return; }
  float a = fetchA().r;
  float b = texelFetch(uB, ivec2(i % uWidth, i / uWidth), 0).r;
  outColor = vec4(${op}, 0.0, 0.0, 1.0);
}`;
}

/**
 * Matrix multiply: A[M,K] @ B[K,N] -> C[M,N].
 * uniforms uM/uK/uN; B is KxN. One fragment per output element, loops K.
 */
export const MATMUL_SHADER = `#version 300 es
precision highp float;
precision highp int;
uniform int uM, uK, uN;
uniform int uWidth;
uniform sampler2D uA;
uniform sampler2D uB;
out vec4 outColor;
void main(){
  int col = int(gl_FragCoord.x);
  int row = int(gl_FragCoord.y);
  if (row >= uM || col >= uN) { outColor = vec4(0.0); return; }
  float acc = 0.0;
  for (int k = 0; k < uK; k++) {
    float a = texelFetch(uA, ivec2(k, row), 0).r;
    float b = texelFetch(uB, ivec2(col, k), 0).r;
    acc += a * b;
  }
  outColor = vec4(acc, 0.0, 0.0, 1.0);
}`;

/**
 * Full reduction (sum) of A to a single value, rendered into a 1x1 target.
 * uSize logical elements laid out in uWidth columns.
 */
export const REDUCE_SUM_SHADER = `#version 300 es
precision highp float;
precision highp int;
uniform int uSize;
uniform int uWidth;
uniform sampler2D uA;
out vec4 outColor;
void main(){
  float acc = 0.0;
  for (int i = 0; i < uSize; i++) {
    acc += texelFetch(uA, ivec2(i % uWidth, i / uWidth), 0).r;
  }
  outColor = vec4(acc, 0.0, 0.0, 1.0);
}`;
