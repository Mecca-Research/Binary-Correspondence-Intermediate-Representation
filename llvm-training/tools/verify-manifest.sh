#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
MANIFEST="$TRAINING_ROOT/examples/README.md"

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

if [ ! -f "$MANIFEST" ]; then
  printf 'error: missing example manifest: %s\n' "$(relpath "$MANIFEST")" >&2
  exit 1
fi

expected=$(mktemp "${TMPDIR:-/tmp}/llvm-training-manifest-expected.XXXXXX")
actual=$(mktemp "${TMPDIR:-/tmp}/llvm-training-manifest-actual.XXXXXX")
missing=$(mktemp "${TMPDIR:-/tmp}/llvm-training-manifest-missing.XXXXXX")
extra=$(mktemp "${TMPDIR:-/tmp}/llvm-training-manifest-extra.XXXXXX")
trap 'rm -f "$expected" "$actual" "$missing" "$extra"' EXIT

sed -n 's/^| `\(llvm-training\/.*\/examples\/[^`]*\.ll\)` |.*/\1/p' "$MANIFEST" | sort -u > "$expected"
find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f ! -iname '*.ll.txt' ! -iname '*invalid*' -print \
  | sed "s#^$REPO_ROOT/##" \
  | sort -u > "$actual"

comm -23 "$actual" "$expected" > "$missing"
comm -13 "$actual" "$expected" > "$extra"

if [ -s "$missing" ] || [ -s "$extra" ]; then
  printf 'Example manifest drift detected in %s\n' "$(relpath "$MANIFEST")" >&2
  if [ -s "$missing" ]; then
    printf '\nDiscovered examples missing from manifest:\n' >&2
    sed 's/^/  /' "$missing" >&2
  fi
  if [ -s "$extra" ]; then
    printf '\nManifest entries with no discovered example file:\n' >&2
    sed 's/^/  /' "$extra" >&2
  fi
  exit 1
fi

count=$(wc -l < "$actual" | tr -d ' ')
printf 'Manifest matches %s discovered standalone LLVM IR example(s).\n' "$count"
