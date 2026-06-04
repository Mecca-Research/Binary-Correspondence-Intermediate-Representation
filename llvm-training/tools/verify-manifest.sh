#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
MANIFEST="$TRAINING_ROOT/examples/README.md"

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

if [ ! -f "$MANIFEST" ]; then
  printf 'error: example manifest not found: %s\n' "$(relpath "$MANIFEST")" >&2
  exit 1
fi

mapfile -d '' discovered_files < <(
  find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f \
    ! -iname '*.ll.txt' \
    ! -iname '*invalid*' \
    -print0 | sort -z
)

discovered_tmp=$(mktemp)
documented_tmp=$(mktemp)
trap 'rm -f "$discovered_tmp" "$documented_tmp"' EXIT

for file in "${discovered_files[@]}"; do
  relpath "$file"
  printf '\n'
done | sort >"$discovered_tmp"

python3 - "$MANIFEST" <<'PY' | sort >"$documented_tmp"
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
for path in sorted(set(re.findall(r"llvm-training/[^ `|)]*/examples/[^ `|)]*\.ll", text))):
    print(path)
PY

if ! cmp -s "$discovered_tmp" "$documented_tmp"; then
  printf 'Example manifest drift detected in %s\n' "$(relpath "$MANIFEST")" >&2
  missing=$(comm -13 "$documented_tmp" "$discovered_tmp")
  stale=$(comm -23 "$documented_tmp" "$discovered_tmp")
  if [ -n "$missing" ]; then
    printf '\nDiscovered examples missing from manifest:\n' >&2
    while IFS= read -r path; do
      [ -n "$path" ] && printf '  %s\n' "$path" >&2
    done <<<"$missing"
  fi
  if [ -n "$stale" ]; then
    printf '\nManifest entries without checked-in files:\n' >&2
    while IFS= read -r path; do
      [ -n "$path" ] && printf '  %s\n' "$path" >&2
    done <<<"$stale"
  fi
  exit 1
fi

printf 'Example manifest is in sync with %d standalone .ll example(s).\n' "${#discovered_files[@]}"
