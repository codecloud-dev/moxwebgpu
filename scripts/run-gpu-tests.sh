#!/usr/bin/env bash
# GPU end-to-end tests via SwiftShader (Vulkan) inside xvfb.
# Usage: pnpm test:gpu [-- <extra vitest args>]
set -euo pipefail
cd "$(dirname "$0")/.."

# Locate a SwiftShader Vulkan ICD (Chrome bundles one) unless already set.
if [ -z "${VK_ICD_FILENAMES:-}" ]; then
  for dir in /root/.agent-browser/browsers/chrome-* /opt/google/chrome; do
    icd="$dir/vk_swiftshader_icd.json"
    if [ -f "$icd" ]; then
      export VK_ICD_FILENAMES="$icd"
      break
    fi
  done
fi

CONFIG="${MOXWEBGPU_VITEST_CONFIG:-vitest.gpu.config.ts}"
RUN=(pnpm exec vitest run --config "$CONFIG" "$@")

if command -v xvfb-run >/dev/null 2>&1; then
  exec xvfb-run -a "${RUN[@]}"
else
  echo "[moxwebgpu] xvfb-run not found — assuming a display is available."
  exec "${RUN[@]}"
fi
