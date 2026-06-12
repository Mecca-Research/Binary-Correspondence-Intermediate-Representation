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
artifact_expected=$(mktemp "${TMPDIR:-/tmp}/llvm-training-artifacts-expected.XXXXXX")
artifact_actual=$(mktemp "${TMPDIR:-/tmp}/llvm-training-artifacts-actual.XXXXXX")
artifact_missing=$(mktemp "${TMPDIR:-/tmp}/llvm-training-artifacts-missing.XXXXXX")
artifact_extra=$(mktemp "${TMPDIR:-/tmp}/llvm-training-artifacts-extra.XXXXXX")
trap 'rm -f "$expected" "$actual" "$missing" "$extra" "$artifact_expected" "$artifact_actual" "$artifact_missing" "$artifact_extra"' EXIT

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

# Every file below any examples/ directory must be named in the manifest, not
# only standalone .ll modules. This keeps MLIR, MIR-shaped text, Markdown/C++
# sketches, source-like BCIR fixtures, target-only artifacts, and invalid inputs
# explicitly classified instead of silently falling outside quality policy.
sed -n 's/^| `\([^`]*\)` |.*/\1/p' "$MANIFEST" \
  | grep -E '^llvm-training/(.*/)?examples/' \
  | sort -u > "$artifact_expected"
find "$TRAINING_ROOT" -path '*/examples/*' -type f -print \
  | sed "s#^$REPO_ROOT/##" \
  | sort -u > "$artifact_actual"

comm -23 "$artifact_actual" "$artifact_expected" > "$artifact_missing"
comm -13 "$artifact_actual" "$artifact_expected" > "$artifact_extra"

if [ -s "$artifact_missing" ] || [ -s "$artifact_extra" ]; then
  printf 'Example artifact inventory drift detected in %s\n' "$(relpath "$MANIFEST")" >&2
  if [ -s "$artifact_missing" ]; then
    printf '\nDiscovered example artifacts missing from manifest classification:\n' >&2
    sed 's/^/  /' "$artifact_missing" >&2
  fi
  if [ -s "$artifact_extra" ]; then
    printf '\nManifest artifact entries with no discovered file:\n' >&2
    sed 's/^/  /' "$artifact_extra" >&2
  fi
  exit 1
fi

count=$(wc -l < "$actual" | tr -d ' ')
artifact_count=$(wc -l < "$artifact_actual" | tr -d ' ')
printf 'Manifest matches %s standalone LLVM IR example(s) and classifies %s example artifact(s).\n' "$count" "$artifact_count"
