#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

find_tool() {
  local base=$1
  local tool

  if [ -n "${LLVM_SUFFIX:-}" ] && command -v "${base}${LLVM_SUFFIX}" >/dev/null 2>&1; then
    printf '%s' "${base}${LLVM_SUFFIX}"
    return 0
  fi

  if command -v "$base" >/dev/null 2>&1; then
    printf '%s' "$base"
    return 0
  fi

  while IFS= read -r tool; do
    if command -v "$tool" >/dev/null 2>&1; then
      printf '%s' "$tool"
      return 0
    fi
  done < <(compgen -c | grep -E "^${base}-[0-9]+$" | sort -V)

  return 1
}

MLIR_OPT=$(find_tool mlir-opt) || {
  printf 'mlir-opt not present; skipping llvm-training MLIR syntax validation\n'
  exit 0
}

status=0
count=0
mapfile -d '' mlir_examples < <(find "$TRAINING_ROOT" -path '*/examples/*.mlir' -type f -print0 | sort -z)

for file in "${mlir_examples[@]}"; do
  count=$((count + 1))
  printf '[mlir-opt] %s ... ' "$(relpath "$file")"
  if output=$("$MLIR_OPT" --allow-unregistered-dialect "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    if [ -n "$output" ]; then
      printf '%s\n' "$output" | sed 's/^/    /'
    fi
    status=1
  fi
done

if [ "$count" -eq 0 ]; then
  printf 'No MLIR examples found under %s\n' "$(relpath "$TRAINING_ROOT")"
  exit 0
fi

if [ "$status" -eq 0 ]; then
  printf 'Validated %d MLIR example(s).\n' "$count"
else
  printf 'One or more MLIR examples failed syntax validation.\n' >&2
fi

exit "$status"
