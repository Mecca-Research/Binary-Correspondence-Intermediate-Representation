#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
INPUT="$TRAINING_ROOT/07-optimization/examples/mem2reg-diamond-before.ll"

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

printf '# mem2reg demo: promote a stack slot to SSA\n'
printf '# Input: %s\n' "$(relpath "$INPUT")"
printf '# First verify the checked-in fixture, then print optimized IR to stdout.\n'

run_cmd opt -passes=verify "$INPUT" -o /dev/null
run_cmd opt -S -passes=mem2reg "$INPUT" -o -
