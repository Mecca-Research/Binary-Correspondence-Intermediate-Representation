#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
SOURCE_MANIFEST="$TRAINING_ROOT/examples/MANIFEST.txt"
DOC_MANIFEST="$TRAINING_ROOT/examples/README.md"

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

if [ ! -f "$SOURCE_MANIFEST" ]; then
  printf 'error: source-of-truth example manifest not found: %s\n' "$(relpath "$SOURCE_MANIFEST")" >&2
  exit 1
fi

if [ ! -f "$DOC_MANIFEST" ]; then
  printf 'error: documented example manifest not found: %s\n' "$(relpath "$DOC_MANIFEST")" >&2
  exit 1
fi

mapfile -t manifest < <(
  python3 - "$SOURCE_MANIFEST" <<'PY'
import sys
from pathlib import Path
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith("#"):
        print(line)
PY
)

mapfile -t discovered < <(
  find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f \
    ! -iname '*.ll.txt' \
    ! -iname '*invalid*' \
    -printf '%P\n' | sed 's#^#llvm-training/#' | sort
)

mapfile -t documented < <(
  python3 - "$DOC_MANIFEST" <<'PY'
import re
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
for match in sorted(set(re.findall(r"llvm-training/[^ `|)]*/examples/[^ `|)]*\.ll", text))):
    print(match)
PY
)

manifest_tmp=$(mktemp)
discovered_tmp=$(mktemp)
documented_tmp=$(mktemp)
trap 'rm -f "$manifest_tmp" "$discovered_tmp" "$documented_tmp"' EXIT
printf '%s\n' "${manifest[@]}" | sort >"$manifest_tmp"
printf '%s\n' "${discovered[@]}" | sort >"$discovered_tmp"
printf '%s\n' "${documented[@]}" | sort >"$documented_tmp"

status=0

if ! cmp -s "$manifest_tmp" "$discovered_tmp"; then
  printf 'Example source manifest drift detected in %s\n' "$(relpath "$SOURCE_MANIFEST")" >&2
  missing=$(comm -13 "$manifest_tmp" "$discovered_tmp")
  stale=$(comm -23 "$manifest_tmp" "$discovered_tmp")
  if [ -n "$missing" ]; then
    printf '\nDiscovered examples missing from source manifest:\n' >&2
    printf '  %s\n' $missing >&2
  fi
  if [ -n "$stale" ]; then
    printf '\nSource manifest entries without checked-in files:\n' >&2
    printf '  %s\n' $stale >&2
  fi
  status=1
fi

if ! cmp -s "$manifest_tmp" "$documented_tmp"; then
  printf 'Example documentation manifest drift detected in %s\n' "$(relpath "$DOC_MANIFEST")" >&2
  missing=$(comm -13 "$documented_tmp" "$manifest_tmp")
  stale=$(comm -23 "$documented_tmp" "$manifest_tmp")
  if [ -n "$missing" ]; then
    printf '\nSource-manifest entries missing from README table:\n' >&2
    printf '  %s\n' $missing >&2
  fi
  if [ -n "$stale" ]; then
    printf '\nREADME table entries not present in source manifest:\n' >&2
    printf '  %s\n' $stale >&2
  fi
  status=1
fi

if [ "$status" -ne 0 ]; then
  exit "$status"
fi

printf 'Example manifests are in sync with %d standalone .ll example(s).\n' "${#manifest[@]}"
