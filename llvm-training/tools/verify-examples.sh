#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

status=0
count=0
KNOWN_INVALID_SENTINEL="$TRAINING_ROOT/examples/broken-example.ll.txt"

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
  done < <(compgen -c | sed -n -E "s/^(${base}-[0-9]+)$/\\1/p" | sort -Vu)
  return 1
}

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

run_step() {
  local label=$1
  local file=$2
  shift 2

  local rel
  rel=$(relpath "$file")
  printf '[%s] %s ... ' "$label" "$rel"

  local output
  if output=$("$@" 2>&1); then
    printf 'ok\n'
    return 0
  fi

  printf 'FAILED\n'
  if [ -n "$output" ]; then
    printf '%s\n' "$output" | sed 's/^/    /'
  fi
  return 1
}

LLVM_AS=$(find_tool llvm-as) || {
  printf 'error: required tool not found on PATH: llvm-as (or llvm-as-N)\n' >&2
  exit 127
}
OPT=$(find_tool opt) || {
  printf 'error: required tool not found on PATH: opt (or opt-N)\n' >&2
  exit 127
}

mapfile -d '' examples < <(find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f ! -iname '*.ll.txt' ! -iname '*invalid*' -print0 | sort -z)

if [ ! -f "$KNOWN_INVALID_SENTINEL" ]; then
  printf 'error: missing invalid-example sentinel: %s\n' "$(relpath "$KNOWN_INVALID_SENTINEL")" >&2
  exit 1
fi

for file in "${examples[@]}"; do
  if [ "$file" = "$KNOWN_INVALID_SENTINEL" ]; then
    printf 'error: invalid-example sentinel entered the known-good manifest: %s\n' "$(relpath "$file")" >&2
    exit 1
  fi
done

printf '[tripwire] %s stays out of the known-good manifest ... ' "$(relpath "$KNOWN_INVALID_SENTINEL")"
if "$LLVM_AS" "$KNOWN_INVALID_SENTINEL" -o /dev/null >/dev/null 2>&1; then
  printf 'FAILED\n'
  printf 'error: invalid-example sentinel assembled successfully; refresh it so it remains a broken .ll.txt fixture.\n' >&2
  exit 1
fi
printf 'ok\n'

for file in "${examples[@]}"; do
  count=$((count + 1))
  run_step llvm-as "$file" "$LLVM_AS" "$file" -o /dev/null || status=1
  run_step verify "$file" "$OPT" -passes=verify "$file" -o /dev/null || status=1
done

if [ "$count" -eq 0 ]; then
  printf 'No standalone examples found under %s\n' "$(relpath "$TRAINING_ROOT")"
  exit 1
fi

if [ "$status" -eq 0 ]; then
  printf 'Verified %d standalone LLVM IR example(s).\n' "$count"
else
  printf 'One or more standalone LLVM IR examples failed verification.\n' >&2
fi

exit "$status"
