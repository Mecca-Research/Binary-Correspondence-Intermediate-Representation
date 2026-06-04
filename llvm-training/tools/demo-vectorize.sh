#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
IR_INPUT="$TRAINING_ROOT/09-vectorization/examples/sum-loop-before.ll"
C_INPUT="$TRAINING_ROOT/09-vectorization/examples/sum-loop.c"
TMP_OBJ=${TMPDIR:-/tmp}/llvm-training-sum-loop.o

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
require_tool clang

for input in "$IR_INPUT" "$C_INPUT"; do
  if [ ! -f "$input" ]; then
    printf 'error: missing demo input: %s\n' "$(relpath "$input")" >&2
    exit 1
  fi
done

printf '# vectorization demo: compare remarks from clang with IR from opt\n'
printf '# C input: %s\n' "$(relpath "$C_INPUT")"
printf '# IR input: %s\n' "$(relpath "$IR_INPUT")"
printf '# clang remarks are emitted on stderr; vectorized IR is printed to stdout.\n'

run_cmd clang -O3 -Rpass=loop-vectorize "$C_INPUT" -c -o "$TMP_OBJ"
run_cmd opt -passes=verify "$IR_INPUT" -o /dev/null
run_cmd opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1 "$IR_INPUT" -o -
