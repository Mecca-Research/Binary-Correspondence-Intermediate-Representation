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

mapfile -d '' discovered < <(
  find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f \
    ! -iname '*.ll.txt' \
    ! -iname '*invalid*' \
    -print0 | sort -z
)

missing=()
for file in "${discovered[@]}"; do
  rel=$(relpath "$file")
  if ! grep -Fq "$rel" "$MANIFEST"; then
    missing+=("$rel")
  fi
done

mapfile -t listed < <(
  python3 - "$MANIFEST" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
for match in sorted(set(re.findall(r"llvm-training/[^ `|)]*/examples/[^ `|)]*\.ll", text))):
    print(match)
PY
)

stale=()
for rel in "${listed[@]}"; do
  if [ ! -f "$REPO_ROOT/$rel" ]; then
    stale+=("$rel")
  fi
done

if [ "${#missing[@]}" -ne 0 ] || [ "${#stale[@]}" -ne 0 ]; then
  printf 'Example manifest drift detected in %s\n' "$(relpath "$MANIFEST")" >&2
  if [ "${#missing[@]}" -ne 0 ]; then
    printf '\nDiscovered examples missing from manifest:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
  fi
  if [ "${#stale[@]}" -ne 0 ]; then
    printf '\nManifest entries without checked-in files:\n' >&2
    printf '  %s\n' "${stale[@]}" >&2
  fi
  exit 1
fi

printf 'Example manifest is in sync with %d standalone .ll example(s).\n' "${#discovered[@]}"
