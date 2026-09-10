#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/../.." && pwd)

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

# The corpus spans more than one LLVM release, so an example may legitimately need a
# newer assembler than the host has -- `ptrtoaddr` parses on 23 and is a syntax error on
# 18. Such a file declares `; REQUIRES: llvm >= N` on its own first lines and is skipped
# below when this assembler is older. SEMVER.md has always allowed an example to state a
# newer requirement; this is that rule made executable instead of advisory.
ASSEMBLER_MAJOR=$("$LLVM_AS" --version 2>/dev/null | sed -n -E 's/.*LLVM version ([0-9]+)\..*/\1/p' | head -1)
if [ -z "$ASSEMBLER_MAJOR" ]; then
  printf 'error: could not read an LLVM major from %s --version; a version-gated example cannot be decided without one\n' "$LLVM_AS" >&2
  exit 1
fi
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

skipped=0
gated=0
for file in "${examples[@]}"; do
  required=$(sed -n -E '1,8s/^; REQUIRES: llvm >= ([0-9]+)[[:space:]]*$/\1/p' "$file" | head -1)
  if [ -n "$required" ]; then
    gated=$((gated + 1))
    if [ "$ASSEMBLER_MAJOR" -lt "$required" ]; then
      skipped=$((skipped + 1))
      printf '[skip] %s ... declares LLVM >= %s; this llvm-as is %s\n' \
        "$(relpath "$file")" "$required" "$ASSEMBLER_MAJOR"
      continue
    fi
  fi
  count=$((count + 1))
  run_step llvm-as "$file" "$LLVM_AS" "$file" -o /dev/null || status=1
  run_step verify "$file" "$OPT" -passes=verify "$file" -o /dev/null || status=1
done

if [ "$count" -eq 0 ]; then
  printf 'No standalone examples found under %s\n' "$(relpath "$TRAINING_ROOT")"
  exit 1
fi

# Anti-vacuity for the directive itself. A version requirement is a way to say "check this
# elsewhere", so it has to be checked SOMEWHERE: if every gated example skipped here, the
# CI job that installs the newer toolchain is the one that owns them, and a run where they
# ALL skip must not read as a run that verified them. Nothing is wrong with a local skip;
# what would be wrong is letting the count disappear.
if [ "$gated" -gt 0 ] && [ "$skipped" -eq "$gated" ]; then
  printf 'note: all %d version-gated example(s) skipped on this LLVM %s host; the job that installs the newer toolchain verifies them.\n' \
    "$gated" "$ASSEMBLER_MAJOR"
fi

if [ "$status" -eq 0 ]; then
  if [ "$skipped" -gt 0 ]; then
    printf 'Verified %d standalone LLVM IR example(s); %d skipped as needing a newer LLVM than %s.\n' \
      "$count" "$skipped" "$ASSEMBLER_MAJOR"
  else
    printf 'Verified %d standalone LLVM IR example(s).\n' "$count"
  fi
else
  printf 'One or more standalone LLVM IR examples failed verification.\n' >&2
fi

exit "$status"
