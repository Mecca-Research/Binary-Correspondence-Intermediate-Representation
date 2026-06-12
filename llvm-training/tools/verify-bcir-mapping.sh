#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
MAPPING_EXAMPLES="$TRAINING_ROOT/bcir-mapping/examples"
BCIR_AS=${BCIR_AS:-}
UPDATE=${UPDATE_BCIR_MAPPING:-0}

status=0
bcir_count=0
claim_count=0

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
  done < <(compgen -c | sed -n -E "s/^(${base}-[0-9]+)$/\\1/p" | sort -Vu)
  return 1
}


LLVM_AS=$(find_tool llvm-as || true)
OPT=$(find_tool opt || true)

run_verify() {
  local file=$1
  printf '[llvm-as] %s ... ' "$(relpath "$file")"
  if output=$("$LLVM_AS" "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
    return 1
  fi

  printf '[verify] %s ... ' "$(relpath "$file")"
  if output=$("$OPT" -passes=verify "$file" -o /dev/null 2>&1); then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    [ -n "$output" ] && printf '%s\n' "$output" | sed 's/^/    /'
    return 1
  fi
}

expected_lowered_outputs() {
  local fixture=$1
  local stem=${fixture%.bcir.txt}
  case "$(basename -- "$fixture")" in
    claim-resource-lookup.bcir.txt)
      printf '%s\0' "$MAPPING_EXAMPLES/claim-resource-lookup.ll"
      ;;
    graph-fragment.bcir.txt)
      printf '%s\0' "$MAPPING_EXAMPLES/graph-fragment-struct-gep.ll"
      ;;
    mixed-stride-graph.bcir.txt)
      printf '%s\0' "$MAPPING_EXAMPLES/mixed-stride-byte-offset.ll"
      ;;
    *)
      if [ -f "$stem.ll" ] || grep -Eq '(^|[[:space:]])lower([[:space:]]|$|[({])' "$fixture"; then
        printf '%s\0' "$stem.ll"
      fi
      ;;
  esac
}

require_claim_keywords() {
  local fixture=$1
  local description=$2
  shift 2

  local keyword
  for keyword in "$@"; do
    printf '[bcir.txt:%s] %s contains %s ... ' "$description" "$(relpath "$fixture")" "$keyword"
    if grep -Eq "(^|[[:space:]])${keyword}([[:space:]]|$|[({])" "$fixture"; then
      printf 'ok\n'
    else
      printf 'FAILED\n'
      printf 'error: %s is missing required BCIR marker or operation keyword: %s\n' "$(relpath "$fixture")" "$keyword" >&2
      return 1
    fi
  done
}

require_any_claim_keyword() {
  local fixture=$1
  shift

  local pattern keyword
  pattern=$(printf '%s|' "$@")
  pattern=${pattern%|}
  printf '[bcir.txt:generic] %s contains a BCIR marker or operation keyword ... ' "$(relpath "$fixture")"
  if grep -Eq "(^|[[:space:]])(${pattern})([[:space:]]|$|[({])" "$fixture"; then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    printf 'error: %s is missing all recognized BCIR markers or operation keywords:' "$(relpath "$fixture")" >&2
    for keyword in "$@"; do
      printf ' %s' "$keyword" >&2
    done
    printf '\n' >&2
    return 1
  fi
}

verify_claim_fixture() {
  local fixture=$1
  local fixture_status=0
  local lowered=()
  local lowered_file

  printf '[bcir.txt:non-empty] %s ... ' "$(relpath "$fixture")"
  if [ -s "$fixture" ]; then
    printf 'ok\n'
  else
    printf 'FAILED\n'
    printf 'error: BCIR claim fixture is empty: %s\n' "$(relpath "$fixture")" >&2
    return 1
  fi

  case "$(basename -- "$fixture")" in
    claim-resource-lookup.bcir.txt)
      require_claim_keywords "$fixture" claim resource claim lower || fixture_status=1
      ;;
    graph-fragment.bcir.txt)
      require_claim_keywords "$fixture" graph vertex edge query || fixture_status=1
      ;;
    mixed-stride-graph.bcir.txt)
      require_claim_keywords "$fixture" mixed-stride layout row_stride_bytes column_stride_bytes query || fixture_status=1
      ;;
    *)
      require_any_claim_keyword "$fixture" resource claim lower vertex edge layout query || fixture_status=1
      ;;
  esac

  mapfile -d '' lowered < <(expected_lowered_outputs "$fixture")
  for lowered_file in "${lowered[@]}"; do
    printf '[bcir.txt:lowered] %s -> %s ... ' "$(relpath "$fixture")" "$(relpath "$lowered_file")"
    if [ -f "$lowered_file" ]; then
      printf 'ok\n'
    else
      printf 'FAILED\n'
      printf 'error: missing lowered LLVM IR companion for %s: %s\n' "$(relpath "$fixture")" "$(relpath "$lowered_file")" >&2
      fixture_status=1
      continue
    fi

    if [ -n "$LLVM_AS" ] && [ -n "$OPT" ]; then
      run_verify "$lowered_file" || fixture_status=1
    else
      printf '[bcir.txt:verify] %s ... skipped (llvm-as and opt are not both available)\n' "$(relpath "$lowered_file")"
    fi
  done

  return "$fixture_status"
}

mapfile -d '' bcir_sources < <(find "$MAPPING_EXAMPLES" -type f -name '*.bcir' -print0 | sort -z)
mapfile -d '' claim_fixtures < <(find "$MAPPING_EXAMPLES" -type f -name '*.bcir.txt' -print0 | sort -z)

if [ "${#bcir_sources[@]}" -eq 0 ] && [ "${#claim_fixtures[@]}" -eq 0 ]; then
  printf 'No BCIR mapping fixtures found under %s; skipping BCIR mapping check.\n' "$(relpath "$MAPPING_EXAMPLES")"
  exit 0
fi

for fixture in "${claim_fixtures[@]}"; do
  claim_count=$((claim_count + 1))
  verify_claim_fixture "$fixture" || status=1
done

if [ "${#bcir_sources[@]}" -gt 0 ]; then
  if [ -z "$BCIR_AS" ] || [ ! -x "$BCIR_AS" ]; then
    printf 'No executable BCIR_AS was provided; skipping real .bcir source round trips.\n'
    printf 'Set BCIR_AS=/path/to/a/current/assembler to require those fixtures.\n'
    exit 0
  fi
  if [ -z "$LLVM_AS" ] || [ -z "$OPT" ]; then
    printf 'llvm-as/opt not present; skipping real .bcir source round trips.\n'
    exit 0
  fi
fi

for source in "${bcir_sources[@]}"; do
  bcir_count=$((bcir_count + 1))
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
  printf 'Checked %d BCIR mapping claim fixture(s) and %d BCIR mapping source fixture(s).\n' "$claim_count" "$bcir_count"
else
  printf 'One or more BCIR mapping fixtures failed. Run UPDATE_BCIR_MAPPING=1 %s to refresh expected outputs for .bcir sources when intentional.\n' "$(relpath "$SCRIPT_DIR/verify-bcir-mapping.sh")" >&2
fi

exit "$status"
