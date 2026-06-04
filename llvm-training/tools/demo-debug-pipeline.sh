#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
INPUT="$TRAINING_ROOT/07-optimization/examples/o2-pipeline-inspection.ll"
TMP_LOG=${TMPDIR:-/tmp}/llvm-training-debug-pass-manager.log

require_tool() {
  local tool=$1
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'error: required tool not found on PATH: %s\n' "$tool" >&2
    exit 127
  fi
}

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

run_cmd() {
  printf '\n$'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_tool opt

if [ ! -f "$INPUT" ]; then
  printf 'error: missing demo input: %s\n' "$(relpath "$INPUT")" >&2
  exit 1
fi

printf '# debug-pass-manager demo: record the default<O2> pass schedule\n'
printf '# Input: %s\n' "$(relpath "$INPUT")"
printf '# Pass-manager diagnostics will be written to %s and then printed to stdout.\n' "$TMP_LOG"

run_cmd opt -passes=verify "$INPUT" -o /dev/null
printf '\n$ opt -disable-output -debug-pass-manager -passes=default\\<O2\\> %q 2> %q\n' "$INPUT" "$TMP_LOG"
opt -disable-output -debug-pass-manager '-passes=default<O2>' "$INPUT" 2>"$TMP_LOG"
printf '\n# Contents of %s\n' "$TMP_LOG"
cat "$TMP_LOG"
