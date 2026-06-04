#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
MAPPING_EXAMPLES="$TRAINING_ROOT/bcir-mapping/examples"
BCIR_AS="$REPO_ROOT/tools/bcir-as/bcir-as"
UPDATE=${UPDATE_BCIR_MAPPING:-0}

status=0
count=0

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

require_tool() {
  local tool=$1
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'error: required tool not found on PATH: %s\n' "$tool" >&2
    exit 127
  fi
}

run_verify() {
  local file=$1
  printf '[llvm-as] %s ... ' "$(relpath "$file")"
  if output=$(llvm-as "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
    return 1
  fi

  printf '[verify] %s ... ' "$(relpath "$file")"
  if output=$(opt -passes=verify "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
    return 1
  fi
}

mapfile -d '' bcir_sources < <(find "$MAPPING_EXAMPLES" -type f -name '*.bcir' -print0 | sort -z)

if [ "${#bcir_sources[@]}" -eq 0 ]; then
  printf 'No .bcir source fixtures found under %s; skipping BCIR mapping output check.\n' "$(relpath "$MAPPING_EXAMPLES")"
  exit 0
fi

if [ ! -x "$BCIR_AS" ]; then
  printf 'error: required assembler is not executable: %s\n' "$(relpath "$BCIR_AS")" >&2
  exit 127
fi
require_tool llvm-as
require_tool opt

for source in "${bcir_sources[@]}"; do
  count=$((count + 1))
  expected="${source%.bcir}.generated.ll"

  if [ "$UPDATE" = 1 ]; then
    printf '[bcir-as:update] %s -> %s ... ' "$(relpath "$source")" "$(relpath "$expected")"
    if output=$("$BCIR_AS" "$source" -o "$expected" 2>&1); then
      printf 'ok\n'
    else
      printf 'FAILED\n'
      [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
      status=1
      continue
    fi
    run_verify "$expected" || status=1
    continue
  fi

  if [ ! -f "$expected" ]; then
    printf 'error: missing expected BCIR mapping output for %s: %s\n' "$(relpath "$source")" "$(relpath "$expected")" >&2
    status=1
    continue
  fi

  generated=$(mktemp "${TMPDIR:-/tmp}/llvm-training-bcir-mapping.XXXXXX.ll")
  printf '[bcir-as] %s ... ' "$(relpath "$source")"
  if output=$("$BCIR_AS" "$source" -o "$generated" 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
    rm -f "$generated"
    status=1
    continue
  fi

  printf '[compare] %s ... ' "$(relpath "$expected")"
  if diff -u "$expected" "$generated" >/dev/null; then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    diff -u "$expected" "$generated" | sed 's/^/    /'
    status=1
  fi
  run_verify "$generated" || status=1
  rm -f "$generated"
done

if [ "$status" -eq 0 ]; then
  printf 'Checked %d BCIR mapping source fixture(s).\n' "$count"
else
  printf 'One or more BCIR mapping source fixtures failed. Run UPDATE_BCIR_MAPPING=1 %s to refresh expected outputs when intentional.\n' "$(relpath "$SCRIPT_DIR/verify-bcir-mapping.sh")" >&2
fi

exit "$status"
