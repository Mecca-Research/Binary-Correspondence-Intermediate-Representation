#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

status=0
count=0

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

rejects_with_llvm_as_or_opt() {
  local file=$1
  local rel bitcode output
  rel=$(relpath "$file")
  bitcode=$(mktemp "${TMPDIR:-/tmp}/llvm-training-invalid.XXXXXX.bc")

  printf '[invalid] %s ... ' "$rel"
  if grep -q '^; llvm-training-invalid-kind: semantic-only$' "$file"; then
    if ! output=$(llvm-as "$file" -o "$bitcode" 2>&1); then
      printf 'FAILED\n'
      printf '    semantic-only invalid fixture should assemble, but llvm-as rejected it\n'
      printf '%s\n' "$output" | sed 's/^/    /'
      rm -f "$bitcode"
      return 1
    fi
    if ! output=$(opt -passes=verify "$bitcode" -o /dev/null 2>&1); then
      printf 'FAILED\n'
      printf '    semantic-only invalid fixture should verify syntactically, but opt rejected it\n'
      printf '%s\n' "$output" | sed 's/^/    /'
      rm -f "$bitcode"
      return 1
    fi
    printf 'accepted by verifier as semantic-only risk\n'
    rm -f "$bitcode"
    return 0
  fi

  if output=$(llvm-as "$file" -o "$bitcode" 2>&1); then
    if output=$(opt -passes=verify "$bitcode" -o /dev/null 2>&1); then
      printf 'FAILED\n'
      printf '    expected llvm-as or opt -passes=verify to reject this fixture, but both accepted it\n'
      rm -f "$bitcode"
      return 1
    fi
    printf 'rejected by opt\n'
  else
    printf 'rejected by llvm-as\n'
  fi

  if [ -n "${output:-}" ]; then
    printf '%s\n' "$output" | sed 's/^/    /'
  fi
  rm -f "$bitcode"
  return 0
}

require_tool llvm-as
require_tool opt

mapfile -d '' invalid_fixtures < <(
  find "$TRAINING_ROOT" \
    \( -name '*.invalid.ll.txt' -o -path "$TRAINING_ROOT/examples/broken-example.ll.txt" \) \
    -type f -print0 | sort -z
)

for file in "${invalid_fixtures[@]}"; do
  count=$((count + 1))
  rejects_with_llvm_as_or_opt "$file" || status=1
done

if [ "$count" -eq 0 ]; then
  printf 'No known invalid LLVM IR fixtures found under %s\n' "$(relpath "$TRAINING_ROOT")" >&2
  exit 1
fi

if [ "$status" -eq 0 ]; then
  printf 'Validated %d known invalid LLVM IR fixture(s).\n' "$count"
else
  printf 'One or more known invalid LLVM IR fixtures were accepted unexpectedly.\n' >&2
fi

exit "$status"
