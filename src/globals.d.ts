// 构建期由 tsup 的 define 注入的真实版本号占位符（见 tsup.config.ts）。
// 仅用于源码里 `export const MOXWEBGPU_VERSION = __MOXWEBGPU_VERSION__;`，
// 运行期被替换成 package.json 的 version 字符串。.d.ts 只声明其类型，不会外泄。
declare const __MOXWEBGPU_VERSION__: string;
