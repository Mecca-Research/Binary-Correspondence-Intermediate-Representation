#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

# Curated for portable assembly emission. This intentionally avoids examples
# that demonstrate non-default address spaces, target-specific intrinsics,
# GC/statepoint tokens, or analysis/vectorizer-only before/after artifacts.
EXAMPLES=(
  "00-foundations/examples/simple-add.ll"
  "00-foundations/examples/ssa-phi.ll"
  "01-syntax/examples/module-anatomy.ll"
  "03-constants/examples/constants-cookbook.ll"
  "05-control-flow/examples/control-flow-cookbook.ll"
  "07-optimization/examples/instcombine-before.ll"
  "07-optimization/examples/mem2reg-before.ll"
  "07-optimization/examples/loop-unroll-after.ll"
  "12-backend-jit/examples/codegen-input.ll"
)

SKIPPED=(
  "02-types/examples/opaque-pointer-after.ll::includes non-default address-space operations"
  "04-memory/examples/memory-cookbook.ll::demonstrates target-defined address spaces"
  "09-vectorization/examples/*.ll::vectorizer teaching inputs/outputs are better checked with opt"
  "13-advanced-ir/examples/token-outline.ll::uses GC/statepoint token intrinsics"
)

if ! command -v llc >/dev/null 2>&1; then
  printf 'error: required tool not found on PATH: llc\n' >&2
  exit 127
fi

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

status=0
count=0

for skipped in "${SKIPPED[@]}"; do
  pattern=${skipped%%::*}
  reason=${skipped#*::}
  printf '[skip] llvm-training/%s ... %s\n' "$pattern" "$reason"
done

for example in "${EXAMPLES[@]}"; do
  file="$TRAINING_ROOT/$example"
  count=$((count + 1))

  if [ ! -f "$file" ]; then
    printf '[llc] %s ... MISSING\n' "$(relpath "$file")"
    status=1
    continue
  fi

  printf '[llc] %s ... ' "$(relpath "$file")"
  if output=$(llc -filetype=asm "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    if [ -n "$output" ]; then
      printf '%s\n' "$output" | sed 's/^/    /'
    fi
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  printf 'llc smoke test emitted assembly for %d curated example(s).\n' "$count"
else
  printf 'One or more llc smoke examples failed.\n' >&2
fi

exit "$status"
