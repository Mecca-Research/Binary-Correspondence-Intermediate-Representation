#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
EXAMPLE="$TRAINING_ROOT/07-optimization/examples/bolt-layout-demo.c"
OUT_DIR=${TMPDIR:-/tmp}/llvm-training-bolt-smoke
BIN="$OUT_DIR/bolt-layout-demo"
BASELINE_LAYOUT="$OUT_DIR/baseline-layout.txt"

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

if ! command -v llvm-bolt >/dev/null 2>&1; then
  printf '[skip] llvm-bolt not found; skipping optional BOLT smoke test.\n'
  exit 0
fi

if ! command -v clang >/dev/null 2>&1; then
  printf 'error: clang is required when llvm-bolt is present.\n' >&2
  exit 127
fi

if ! command -v llvm-objdump >/dev/null 2>&1; then
  printf 'error: llvm-objdump is required when llvm-bolt is present.\n' >&2
  exit 127
fi

if [ ! -f "$EXAMPLE" ]; then
  printf 'error: missing BOLT fixture: %s\n' "$(relpath "$EXAMPLE")" >&2
  exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

printf '[build] %s ... ' "$(relpath "$EXAMPLE")"
if output=$(clang -O2 -fno-omit-frame-pointer -Wl,--emit-relocs "$EXAMPLE" -o "$BIN" 2>&1); then
  printf 'ok\n'
else
  printf 'FAILED\n'
  printf '%s\n' "$output" | sed 's/^/    /'
  exit 1
fi

printf '[run] baseline executable ... '
if output=$("$BIN" 2>&1); then
  printf 'ok (%s)\n' "$output"
else
  printf 'FAILED\n'
  printf '%s\n' "$output" | sed 's/^/    /'
  exit 1
fi

printf '[inspect] baseline symbols and disassembly ... '
if llvm-objdump -t "$BIN" > "$BASELINE_LAYOUT" 2>&1 && \
   llvm-objdump -d --symbolize-operands "$BIN" >> "$BASELINE_LAYOUT" 2>&1; then
  printf 'ok (%s)\n' "$BASELINE_LAYOUT"
else
  printf 'FAILED\n'
  sed 's/^/    /' "$BASELINE_LAYOUT" 2>/dev/null || true
  exit 1
fi

printf '[check] expected symbols ... '
missing=0
for symbol in main hot_path cold_path checksum; do
  if ! grep -Eq "[[:space:]]${symbol}$|<${symbol}>:" "$BASELINE_LAYOUT"; then
    printf '\nerror: expected symbol not found in baseline layout: %s\n' "$symbol" >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 1
fi
printf 'ok\n'

llvm-bolt --version | sed 's/^/[llvm-bolt] /' | head -n 1

if ! command -v perf2bolt >/dev/null 2>&1; then
  printf '[skip] perf2bolt not found; baseline BOLT fixture smoke completed without profile conversion.\n'
  exit 0
fi

printf '[info] perf2bolt is available; see llvm-training/07-optimization/07-bolt-layout-walkthrough.md for the full profile-driven rewrite commands.\n'
