import { defineConfig } from 'tsup';

export default defineConfig([
  {
    // Library builds: ESM + CJS with type declarations
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    sourcemap: true,
    clean: true,
    target: 'es2020',
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
  },
]);
