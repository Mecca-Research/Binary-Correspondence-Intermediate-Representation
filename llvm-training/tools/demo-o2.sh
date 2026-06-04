#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
INPUT="$TRAINING_ROOT/07-optimization/examples/o2-pipeline-inspection.ll"
TMP_IR=${TMPDIR:-/tmp}/llvm-training-o2-pipeline-inspection.after.ll

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

printf '# default<O2> demo: run the standard O2 optimization pipeline\n'
printf '# Input: %s\n' "$(relpath "$INPUT")"
printf '# Optimized IR will be written to %s and then printed to stdout.\n' "$TMP_IR"

run_cmd opt -passes=verify "$INPUT" -o /dev/null
run_cmd opt -S '-passes=default<O2>' "$INPUT" -o "$TMP_IR"
printf '\n# Contents of %s\n' "$TMP_IR"
cat "$TMP_IR"

if command -v llc >/dev/null 2>&1; then
  printf '\n# Optional codegen smoke check for the O2 result.\n'
  run_cmd llc -filetype=asm "$TMP_IR" -o /dev/null
else
  printf '\n# skip: optional tool not found on PATH: llc\n'
fi
