#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

# lli executes whole programs, not arbitrary library-style IR snippets.
# Most training examples are assembly-only because they expose functions that
# need caller-provided arguments, demonstrate optimization input/output, rely on
# target lowering, or include intrinsics/metadata meant for static verification.
# Keep this list to modules with a safe no-argument main, or entries explicitly
# documented here as runnable.
RUNNABLES=(
  "01-syntax/examples/module-anatomy.ll::main::safe no-argument main; prints a short message and returns 0"
)

if ! command -v lli >/dev/null 2>&1; then
  printf 'error: required tool not found on PATH: lli\n' >&2
  exit 127
fi

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

status=0
count=0

for runnable in "${RUNNABLES[@]}"; do
  example=${runnable%%::*}
  rest=${runnable#*::}
  entry=${rest%%::*}
  reason=${rest#*::}
  file="$TRAINING_ROOT/$example"
  count=$((count + 1))

  if [ ! -f "$file" ]; then
    printf '[lli] %s (%s) ... MISSING\n' "$(relpath "$file")" "$entry"
    status=1
    continue
  fi

  printf '[lli] %s (%s: %s) ... ' "$(relpath "$file")" "$entry" "$reason"
  if [ "$entry" = main ]; then
    if output=$(lli "$file" 2>&1); then
      printf 'ok\n'
      if [ -n "$output" ]; then
        printf '%s\n' "$output" | sed 's/^/    output: /'
      fi
    else
      printf 'FAILED\n'
      if [ -n "$output" ]; then
        printf '%s\n' "$output" | sed 's/^/    /'
      fi
      status=1
    fi
  else
    if output=$(lli -entry-function="$entry" "$file" 2>&1); then
      printf 'ok\n'
      if [ -n "$output" ]; then
        printf '%s\n' "$output" | sed 's/^/    output: /'
      fi
    else
      printf 'FAILED\n'
      if [ -n "$output" ]; then
        printf '%s\n' "$output" | sed 's/^/    /'
      fi
      status=1
    fi
  fi
done

if [ "$status" -eq 0 ]; then
  printf 'lli smoke test ran %d curated runnable example(s).\n' "$count"
  printf 'Note: most examples are intentionally assembly-only; see this script and llvm-training/examples/README.md for rationale.\n'
else
  printf 'One or more lli smoke examples failed.\n' >&2
fi

exit "$status"
