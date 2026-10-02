/**
 * GPU test harness: launches a WebGPU-capable Chromium (SwiftShader Vulkan
 * software rendering when no GPU is present), loads a secure-context page,
 * and injects the moxwebgpu browser bundle.
 *
 * Requirements (satisfied by `pnpm test:gpu` -> scripts/run-gpu-tests.sh):
 *   - run under xvfb-run on display-less machines
 *   - VK_ICD_FILENAMES pointing at a SwiftShader Vulkan ICD (auto-detected)
 */

import { chromium, type Browser, type Page } from 'playwright-core';
import { existsSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const CHROME_ARGS = [
  '--enable-unsafe-webgpu',
  '--enable-features=Vulkan',
  '--no-sandbox',
  '--disable-dev-shm-usage',
];

function findChrome(): string {
  const custom = process.env.MOXWEBGPU_CHROME;
  if (custom && existsSync(custom)) return custom;

  const abDir = '/root/.agent-browser/browsers';
  if (existsSync(abDir)) {
    const versions = readdirSync(abDir)
      .filter((d) => d.startsWith('chrome-'))
      .sort()
      .reverse();
    for (const v of versions) {
      const p = path.join(abDir, v, 'chrome');
      if (existsSync(p)) return p;
    }
  }
  for (const p of ['/usr/bin/google-chrome', '/usr/bin/google-chrome-stable', '/usr/bin/chromium', '/usr/bin/chromium-browser']) {
    if (existsSync(p)) return p;
  }
  throw new Error('moxwebgpu GPU tests: no Chromium found. Set MOXWEBGPU_CHROME=/path/to/chrome');
}

function ensureVulkanIcd(): void {
  if (process.env.VK_ICD_FILENAMES) return;
  const candidates: string[] = [];
  const abDir = '/root/.agent-browser/browsers';
  if (existsSync(abDir)) {
    for (const v of readdirSync(abDir).filter((d) => d.startsWith('chrome-')).sort().reverse()) {
      candidates.push(path.join(abDir, v, 'vk_swiftshader_icd.json'));
    }
  }
  candidates.push('/opt/google/chrome/vk_swiftshader_icd.json');
  for (const c of candidates) {
    if (existsSync(c)) {
      process.env.VK_ICD_FILENAMES = c;
      return;
    }
  }
  // Without an ICD the Vulkan loader may still find a system driver.
}

let browser: Browser | null = null;
let pagePromise: Promise<Page> | null = null;

/** Launch once per test run, return a page with `window.MoxWebGPU` loaded. */
export function getGpuPage(): Promise<Page> {
  if (!pagePromise) {
    ensureVulkanIcd();
    pagePromise = (async () => {
      browser = await chromium.launch({ executablePath: findChrome(), args: CHROME_ARGS });
      const page = await browser.newPage();
      // Forward page console (shader compile errors surface this way).
      page.on('console', (msg) => {
        if (msg.type() === 'error' || msg.type() === 'warning') {
          console.log(`[page:${msg.type()}]`, msg.text());
        }
      });
      page.on('pageerror', (err) => console.log('[page:error]', err.message));
      // WebGPU requires a secure context; example.com over https is enough.
      await page.goto('https://example.com/', { timeout: 30_000, waitUntil: 'domcontentloaded' });
      const bundle = path.resolve(__dirname, '../../dist/moxwebgpu.browser.js');
      await page.addScriptTag({ path: bundle });
      return page;
    })();
  }
  return pagePromise;
}

export async function closeBrowser(): Promise<void> {
  if (browser) {
    await browser.close();
    browser = null;
    pagePromise = null;
  }
}

/**
 * Run a piece of async code inside the page with a shared MoxContext.
 * The callback is serialized, so it must not close over outer variables —
 * pass the data explicitly through `arg` (structured-clone serializable),
 * available inside the callback as the second parameter.
 */
export async function withGpu<T>(fn: (gpu: any, arg: any) => Promise<T>, arg?: any): Promise<T> {
  const page = await getGpuPage();
  return page.evaluate(
    async ({ src, arg }: { src: string; arg: any }) => {
      if (!(window as any).__moxwebgpuCtx) {
        const M = (window as any).MoxWebGPU;
        if (!M) throw new Error('moxwebgpu bundle missing on page');
        (window as any).__moxwebgpuCtx = await M.mox.init();
      }
      (window as any).__moxwebgpuArg = arg;
      return eval(src);
    },
    { src: `(${fn.toString()})(window.__moxwebgpuCtx, window.__moxwebgpuArg)`, arg },
  ) as Promise<T>;
}
