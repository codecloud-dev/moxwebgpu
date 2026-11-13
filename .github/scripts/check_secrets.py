#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描误提交的密钥 / 令牌。跨仓通用，无单仓专属路径假设。

用法: python3 .github/scripts/check_secrets.py [--self-test]
退出码: 发现疑似密钥 -> 1；干净 -> 0。
--self-test 会验证每条规则都能命中自己的样例、占位符不误报、过滤不吃真密钥。
"""
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXCLUDE_DIRS = {
    '.git', '.github', 'node_modules', 'target', 'build', 'dist', 'vendor',
    '__pycache__', '.gradle', '.idea', '.venv', 'venv', 'coverage', '.next',
    '.wrangler', '.cxx', '.externalNativeBuild', 'pods', 'DerivedData',
}
SKIP_EXT = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.avif', '.ico', '.bmp',
    '.jar', '.zip', '.gz', '.tar', '.7z', '.apk', '.aab', '.keystore', '.jks',
    '.so', '.o', '.a', '.dll', '.dylib', '.dex', '.class', '.wasm',
    '.ttf', '.otf', '.woff', '.woff2', '.eot', '.mp4', '.mp3', '.wav', '.ogg',
    '.pdf', '.lock', '.svg', '.min.js', '.map',
}
MAX_BYTES = 1_000_000

RULES = [
    ("github_pat", re.compile(r"github_pat_[0-9A-Za-z_]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[0-9A-Za-z]{36,}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{35}")),
    ("slack_token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
    ("stripe_live_key", re.compile(r"sk_live_[0-9a-zA-Z]{24,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("gitee_token", re.compile(r"(?i)GITEE_(?:TOKEN|PAT|ACCESS_TOKEN)\s*[:=]\s*['\"]?[A-Za-z0-9]{20,}")),
    ("cf_api_token", re.compile(r"(?i)CLOUDFLARE_API_TOKEN\s*[:=]\s*['\"]?[A-Za-z0-9_-]{30,}")),
    ("keystore_base64", re.compile(r"(?i)KEYSTORE_BASE64\s*[:=]\s*['\"]?[A-Za-z0-9+/=]{120,}")),
    # 宽规则只认「带引号的字符串字面量」：真硬编码密钥必然是字符串，
    # 而 kv.client_secret / app.client_id 这类属性访问不带引号，天然排除。
    ("generic_secret", re.compile(
        r"(?i)(?:client[_-]?secret|api[_-]?key|secret|private[_-]?key|"
        r"access[_-]?token|auth[_-]?token|password|passwd)\s*[:=]\s*"
        r"['\"]([A-Za-z0-9/+_@.#-]{16,})['\"]")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}")),
]

SELF_TEST_SAMPLES = {
    "github_pat": "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz1234567890",
    "github_token": "ghp_" + "a" * 36,
    "aws_access_key": "AKIA" + "A" * 16,
    "google_api_key": "AIza" + "a" * 35,
    "slack_token": "xoxb-123456789012-abcdefghijkl",
    "stripe_live_key": "sk_live_" + "a" * 24,
    "private_key_block": "-----BEGIN OPENSSH PRIVATE KEY-----",
    "jwt": "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwxyz12",
    "gitee_token": "GITEE_TOKEN = " + "a" * 32,
    "cf_api_token": "CLOUDFLARE_API_TOKEN = " + "a" * 40,
    "keystore_base64": "KEYSTORE_BASE64 = " + "A" * 160,
    "generic_secret": 'password = "supersecretvalue123456"',
    "bearer_token": "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123456",
}
SELF_TEST_SAFE = [
    "password = ${KEYSTORE_PASSWORD}",
    "password = REPLACE_WITH_YOUR_PASSWORD",
    "const token = process.env.GITHUB_TOKEN;",
    "client_secret = os.getenv('GITHUB_CLIENT_SECRET')",
    'private_key = "<YOUR_PRIVATE_KEY>"',
    'secret = "env.MOX_STORE_SECRET"',
    "api_key = secrets.API_KEY",
    # 属性访问：不带引号，宽规则不得命中（真实仓库里曾造成 6 处误报）
    "clientSecret: kv.client_secret, source: 'kv'",
    "apiKey: customCh.api_key || c.providerKey,",
    "access_token: consumed.device_token,",
    # 测试夹具固定假值
    "const CHALLENGE_SECRET = 'e2e-fixed-challenge-secret';",
    "GITEE_CLIENT_SECRET: 'gitee-secret-xyz',",
]
PLACEHOLDER_MARKERS = (
    "replace_with", "${", "<your", "your_", "changeme", "placeholder",
    "process.env", "os.getenv", "getenv(", "secrets.", "vars.", "env.",
    "example_", "dummy", "xxxx", "<redacted>", "todo",
    # 测试夹具里的固定假值（e2e-fixed-... / gitee-secret-xyz 等），不是真密钥
    "e2e-", "-test-", "test-", "-xyz", "fake", "sample_", "not-a-real",
)

def looks_like_placeholder(line):
    low = line.lower()
    return any(m in low for m in PLACEHOLDER_MARKERS)

def scan_line(line):
    if "${" in line or "replace_with" in line.lower():
        return None
    for label, rx in RULES:
        m = rx.search(line)
        if not m:
            continue
        if label == "generic_secret" and looks_like_placeholder(line):
            continue
        return (label, m)
    return None

def self_test():
    labels = {l for l, _ in RULES}
    missing = labels - set(SELF_TEST_SAMPLES)
    if missing:
        print("FAIL: 以下规则缺样例: %s" % sorted(missing)); return 1
    for label, sample in SELF_TEST_SAMPLES.items():
        if not any(name == label and rx.search(sample) for name, rx in RULES):
            print("FAIL: 规则 %s 未命中自己的样例" % label); return 1
    for label, sample in SELF_TEST_SAMPLES.items():
        if not scan_line(sample):
            print("FAIL: 占位符过滤误吃真实样例 %s" % label); return 1
    for safe in SELF_TEST_SAFE:
        hit = scan_line(safe)
        if hit:
            print("FAIL: 占位符被误报 %r -> %s" % (safe, hit[0])); return 1
    print("OK: %d 条规则全部命中样例，%d 条占位符无误报，过滤未误吃真密钥"
          % (len(RULES), len(SELF_TEST_SAFE)))
    return 0

def main():
    hits = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.git')]
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, REPO_ROOT).replace(os.sep, '/')
            # 扫描脚本自身含模式字符串，必须跳过
            if rel.endswith('check_secrets.py'):
                continue
            if os.path.splitext(name)[1].lower() in SKIP_EXT:
                continue
            try:
                if os.path.getsize(full) > MAX_BYTES:
                    continue
                with open(full, 'r', encoding='utf-8', errors='ignore') as f:
                    for ln, line in enumerate(f, 1):
                        hit = scan_line(line)
                        if hit:
                            hits.append((rel, ln, hit[0], hit[1].group(0)[:24]))
            except OSError:
                continue
    if not hits:
        print("OK: 未发现疑似密钥 / 令牌")
        return 0
    print("FAIL: 发现疑似密钥 / 令牌：")
    for rel, ln, label, snap in hits:
        print("  [%s] %s:%d -> %r..." % (label, rel, ln, snap))
    return 1

if __name__ == '__main__':
    sys.exit(self_test() if '--self-test' in sys.argv else main())
