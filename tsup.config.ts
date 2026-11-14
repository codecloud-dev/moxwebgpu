import { defineConfig } from 'tsup';
import { readFileSync } from 'node:fs';

// 构建期把版本注入产物，保证运行时 MOXWEBGPU_VERSION 永远等于 package.json 的 version，
// 不再手写出错（此前写死 1.0.1，与发布的 1.1.x 不一致）。
const { version } = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
) as { version: string };
const VERSION_DEFINE = { __MOXWEBGPU_VERSION__: JSON.stringify(version) };

export default defineConfig([
  {
    // Library builds: ESM + CJS with type declarations
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    sourcemap: true,
    clean: true,
    target: 'es2020',
    define: VERSION_DEFINE,
  },
  {
    // Single-file browser build (usable via <script> tag, sets window.MoxWebGPU)
    entry: { moxwebgpu: 'src/index.ts' },
    format: ['iife'],
    globalName: 'MoxWebGPU',
    outExtension: () => ({ js: '.browser.js' }),
    sourcemap: true,
    target: 'es2020',
    splitting: false,
    define: VERSION_DEFINE,
  },
]);
