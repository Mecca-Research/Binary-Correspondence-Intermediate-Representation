#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
FIXTURE_DIR="$TRAINING_ROOT/exercises/adversarial"
LIST_ONLY=0

if [ "${1:-}" = "--list" ]; then
  LIST_ONLY=1
  shift
fi
if [ "$#" -ne 0 ]; then
  printf 'usage: %s [--list]\n' "${0##*/}" >&2
  exit 2
fi

relpath() {
  printf '%s' "${1#"$REPO_ROOT/"}"
}

find_tool() {
  local base=$1 tool
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

fixture_class() {
  local file=$1
  sed -n 's/^; adversarial-class: //p' "$file"
}

mapfile -d '' fixtures < <(
  find "$FIXTURE_DIR" -maxdepth 1 -type f \
    \( -name '*.ll' -o -name '*.ll.txt' \) -print0 | sort -z
)

if [ "${#fixtures[@]}" -eq 0 ]; then
  printf 'error: no adversarial fixtures found under %s\n' "$(relpath "$FIXTURE_DIR")" >&2
  exit 1
fi

status=0
for file in "${fixtures[@]}"; do
  mapfile -t classes < <(fixture_class "$file")
  if [ "${#classes[@]}" -ne 1 ]; then
    printf '[unclassified] %s ... FAILED (expected exactly one adversarial-class marker)\n' "$(relpath "$file")" >&2
    status=1
    continue
  fi
  case "${classes[0]}" in
    assemble-valid-semantically-risky|intentionally-invalid|target-specific|metadata-preservation)
      printf '[%s] %s\n' "${classes[0]}" "$(relpath "$file")"
      ;;
    *)
      printf '[unknown-class:%s] %s ... FAILED\n' "${classes[0]}" "$(relpath "$file")" >&2
      status=1
      ;;
  esac
done

if [ "$status" -ne 0 ] || [ "$LIST_ONLY" -eq 1 ]; then
  exit "$status"
fi

LLVM_AS=$(find_tool llvm-as) || {
  printf 'error: required tool not found on PATH: llvm-as (or llvm-as-N)\n' >&2
  exit 127
}
OPT=$(find_tool opt) || {
  printf 'error: required tool not found on PATH: opt (or opt-N)\n' >&2
  exit 127
}

validate_accepted() {
  local file=$1 bitcode output
  bitcode=$(mktemp "${TMPDIR:-/tmp}/llvm-training-adversarial.XXXXXX.bc")
  if ! output=$("$LLVM_AS" "$file" -o "$bitcode" 2>&1); then
    printf '    FAILED: expected llvm-as acceptance\n' >&2
    printf '%s\n' "$output" | sed 's/^/    /' >&2
    rm -f "$bitcode"
    return 1
  fi
  if ! output=$("$OPT" -passes=verify "$bitcode" -o /dev/null 2>&1); then
    printf '    FAILED: expected opt verifier acceptance\n' >&2
    printf '%s\n' "$output" | sed 's/^/    /' >&2
    rm -f "$bitcode"
    return 1
  fi
  rm -f "$bitcode"
}

validate_invalid() {
  local file=$1 bitcode
  bitcode=$(mktemp "${TMPDIR:-/tmp}/llvm-training-adversarial-invalid.XXXXXX.bc")
  if "$LLVM_AS" "$file" -o "$bitcode" >/dev/null 2>&1 && \
     "$OPT" -passes=verify "$bitcode" -o /dev/null >/dev/null 2>&1; then
    printf '    FAILED: intentionally invalid fixture was accepted\n' >&2
    rm -f "$bitcode"
    return 1
  fi
  rm -f "$bitcode"
}

validate_metadata_roundtrip() {
  local file=$1 marker output
  marker=$(sed -n 's/^; adversarial-preserve-metadata: //p' "$file")
  if [ -z "$marker" ]; then
    printf '    FAILED: metadata fixture lacks adversarial-preserve-metadata marker\n' >&2
    return 1
  fi
  if ! output=$("$OPT" -passes=verify -S "$file" -o - 2>&1); then
    printf '    FAILED: metadata verifier round trip failed\n' >&2
    printf '%s\n' "$output" | sed 's/^/    /' >&2
    return 1
  fi
  if ! grep -Fq "$marker" <<<"$output"; then
    printf '    FAILED: round trip dropped required metadata marker %s\n' "$marker" >&2
    return 1
  fi
}

counts_risky=0
counts_invalid=0
counts_target=0
counts_metadata=0
for file in "${fixtures[@]}"; do
  class=$(fixture_class "$file")
  printf '[validate:%s] %s ... ' "$class" "$(relpath "$file")"
  case "$class" in
    assemble-valid-semantically-risky)
      counts_risky=$((counts_risky + 1))
      if validate_accepted "$file"; then printf 'accepted; semantic review required\n'; else status=1; fi
      ;;
    intentionally-invalid)
      counts_invalid=$((counts_invalid + 1))
      if validate_invalid "$file"; then printf 'rejected as intended\n'; else status=1; fi
      ;;
    target-specific)
      counts_target=$((counts_target + 1))
      if validate_accepted "$file"; then printf 'accepted; backend execution remains target-gated\n'; else status=1; fi
      ;;
    metadata-preservation)
      counts_metadata=$((counts_metadata + 1))
      if validate_accepted "$file" && validate_metadata_roundtrip "$file"; then printf 'accepted; required metadata survived verifier round trip\n'; else status=1; fi
      ;;
  esac
done

printf 'Classified %d fixture(s): %d risky, %d intentionally invalid, %d target-specific, %d metadata-preservation.\n' \
  "${#fixtures[@]}" "$counts_risky" "$counts_invalid" "$counts_target" "$counts_metadata"
exit "$status"
