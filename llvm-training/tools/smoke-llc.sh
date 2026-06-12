#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

SKIP_POLICY="$SCRIPT_DIR/smoke-llc-skip.txt"

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

# Curated allowlist for portable assembly emission. This intentionally avoids
# examples that demonstrate non-default address spaces, target-specific
# intrinsics, GC/statepoint tokens, or analysis/vectorizer-only before/after
# artifacts. Document intentional exclusions in $SKIP_POLICY instead of adding
# skip rules here.
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

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

print_skips() {
  local lineno=0
  local line pattern reason

  if [ ! -f "$SKIP_POLICY" ]; then
    printf 'error: required skip policy not found: %s\n' "$(relpath "$SKIP_POLICY")" >&2
    return 1
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    lineno=$((lineno + 1))

    case "$line" in
      ''|'#'*)
        continue
        ;;
    esac

    if [[ "$line" != *$'\t'* ]]; then
      printf 'error: %s:%d: expected <path-or-glob><TAB><reason>\n' "$(relpath "$SKIP_POLICY")" "$lineno" >&2
      return 1
    fi

    pattern=${line%%$'\t'*}
    reason=${line#*$'\t'}

    if [ -z "$pattern" ] || [ -z "$reason" ]; then
      printf 'error: %s:%d: skip pattern and reason must be non-empty\n' "$(relpath "$SKIP_POLICY")" "$lineno" >&2
      return 1
    fi

    printf '[skip] llvm-training/%s ... %s\n' "$pattern" "$reason"
  done <"$SKIP_POLICY"
}

status=0
count=0

print_skips || exit 1

LLC=$(find_tool llc) || {
  printf 'error: required tool not found on PATH: llc (or llc-N)\n' >&2
  exit 127
}

for example in "${EXAMPLES[@]}"; do
  file="$TRAINING_ROOT/$example"
  count=$((count + 1))

  if [ ! -f "$file" ]; then
    printf '[llc] %s ... MISSING\n' "$(relpath "$file")"
    status=1
    continue
  fi

  printf '[llc] %s ... ' "$(relpath "$file")"
  if output=$("$LLC" -filetype=asm "$file" -o /dev/null 2>&1); then
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
