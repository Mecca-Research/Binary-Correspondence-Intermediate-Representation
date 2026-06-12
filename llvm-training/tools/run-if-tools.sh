#!/usr/bin/env bash
set -u -o pipefail

if [ "$#" -lt 4 ]; then
  printf 'usage: %s <label> <tool>... -- <command> [args...]\n' "$0" >&2
  exit 2
fi

label=$1
shift
missing=()

find_tool() {
  local base=$1
  local tool
  if command -v "$base" >/dev/null 2>&1; then
    return 0
  fi
  while IFS= read -r tool; do
    if command -v "$tool" >/dev/null 2>&1; then
      return 0
    fi
  done < <(compgen -c | sed -n -E "s/^(${base}-[0-9]+)$/\\1/p" | sort -Vu)
  return 1
}

while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
  find_tool "$1" || missing+=("$1")
  shift
done

if [ "$#" -eq 0 ]; then
  printf 'error: missing -- command separator\n' >&2
  exit 2
fi
shift

if [ "${#missing[@]}" -gt 0 ]; then
  printf '%s unavailable (%s); skipping cleanly\n' "$label" "${missing[*]}"
  exit 0
fi

exec "$@"
