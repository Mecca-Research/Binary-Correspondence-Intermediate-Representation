#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
UPDATE_OPT_DIFF=${UPDATE_OPT_DIFF:-0}

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

normalize_ir() {
  # Keep pass-output comparisons focused on semantically relevant IR. LLVM
  # versions and invocation paths can perturb the ModuleID banner, and host
  # builds may synthesize equivalent target datalayout spellings for examples
  # that only pin a target triple.
  sed \
    -e '/^; ModuleID = /d' \
    -e '/^target datalayout = /d' \
    "$1"
}

run_case() {
  local input=$1
  local passes=$2
  local golden=$3
  local temp actual expected output

  temp=$(mktemp "${TMPDIR:-/tmp}/llvm-training-opt-diff.XXXXXX.ll")
  actual=$(mktemp "${TMPDIR:-/tmp}/llvm-training-opt-diff.actual.XXXXXX.ll")
  expected=$(mktemp "${TMPDIR:-/tmp}/llvm-training-opt-diff.expected.XXXXXX.ll")

  printf '[opt-diff] opt -S -passes=%s %s ... ' "$passes" "$(relpath "$input")"
  if ! output=$(opt -S -passes="$passes" "$input" -o "$temp" 2>&1); then
    printf 'FAILED\n'
    if [ -n "$output" ]; then
      printf '%s\n' "$output" | sed 's/^/    /'
    fi
    rm -f "$temp" "$actual" "$expected"
    return 1
  fi

  if [ "$UPDATE_OPT_DIFF" = "1" ]; then
    cp "$temp" "$golden"
    printf 'updated %s\n' "$(relpath "$golden")"
    rm -f "$temp" "$actual" "$expected"
    return 0
  fi

  normalize_ir "$temp" > "$actual"
  normalize_ir "$golden" > "$expected"
  if diff -u --label "$(relpath "$golden")" --label "opt output" "$expected" "$actual"; then
    printf 'ok\n'
    rm -f "$temp" "$actual" "$expected"
    return 0
  fi

  printf 'FAILED\n'
  printf '    golden drift: %s\n' "$(relpath "$golden")"
  printf '    refresh intentionally with UPDATE_OPT_DIFF=1 %s\n' "$(relpath "$SCRIPT_DIR/verify-opt-diff.sh")"
  rm -f "$temp" "$actual" "$expected"
  return 1
}

require_tool opt
require_tool diff

status=0
count=0

CASES=(
  "07-optimization/examples/opt-diff-instcombine-before.ll|instcombine|07-optimization/examples/opt-diff-instcombine.after-instcombine.ll"
  "07-optimization/examples/opt-diff-loop-rotate-before.ll|loop-rotate|07-optimization/examples/opt-diff-loop-rotate.after-loop-rotate.ll"
)

for case in "${CASES[@]}"; do
  IFS='|' read -r input_rel passes golden_rel <<< "$case"
  input="$TRAINING_ROOT/$input_rel"
  golden="$TRAINING_ROOT/$golden_rel"
  if [ ! -f "$input" ]; then
    printf 'error: missing opt-diff input fixture: %s\n' "$(relpath "$input")" >&2
    status=1
    continue
  fi
  if [ ! -f "$golden" ]; then
    printf 'error: missing opt-diff golden fixture: %s\n' "$(relpath "$golden")" >&2
    status=1
    continue
  fi
  count=$((count + 1))
  run_case "$input" "$passes" "$golden" || status=1
done

if [ "$count" -eq 0 ]; then
  printf 'No opt-diff fixtures are registered.\n' >&2
  exit 1
fi

if [ "$status" -eq 0 ]; then
  printf 'Verified %d opt pipeline golden fixture(s).\n' "$count"
else
  printf 'One or more opt pipeline golden fixtures drifted.\n' >&2
fi

exit "$status"
