#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

status=0
count=0

mapfile -d '' examples < <(find "$TRAINING_ROOT" -path '*/examples/*.ll' -type f ! -name '*.ll.txt' -print0 | sort -z)

for file in "${examples[@]}"; do
  count=$((count + 1))
  printf '[opaque-pointers] %s ... ' "$(relpath "$file")"
  if matches=$(awk '
    {
      line = $0
      sub(/;.*/, "", line)
      if (line ~ /(^|[[:space:],(=])((i[0-9]+)|(half)|(bfloat)|(float)|(double)|(x86_fp80)|(fp128)|(ppc_fp128)|(void)|(label)|(token)|(metadata)|(%[-A-Za-z$._0-9]+)|(@[-A-Za-z$._0-9]+)|(\[[^]]+\])|(<[^>]+>)|(\{[^}]+\}))[[:space:]]*\*/) {
        printf "%d:%s\n", NR, $0
      }
    }
  ' "$file"); then
    :
  else
    printf 'FAILED\n'
    printf '    awk failed while scanning fixture\n'
    status=1
    continue
  fi

  if [ -n "$matches" ]; then
    printf 'FAILED\n'
    printf '%s\n' "$matches" | sed 's/^/    typed pointer syntax: /'
    status=1
  else
    printf 'ok\n'
  fi
done

migration_count=$(find "$TRAINING_ROOT" -path '*/examples/*.ll.txt' -type f | wc -l | tr -d ' ')
printf '[opaque-pointers] allowing %s explicit .ll.txt migration/invalid fixture(s).\n' "$migration_count"

if [ "$count" -eq 0 ]; then
  printf 'No modern LLVM IR examples found under %s\n' "$(relpath "$TRAINING_ROOT")" >&2
  exit 1
fi

if [ "$status" -eq 0 ]; then
  printf 'Verified %d modern LLVM IR example(s) use opaque pointers.\n' "$count"
else
  printf 'Typed-pointer syntax was found in one or more modern .ll examples. Use ptr or move migration-only text to .ll.txt.\n' >&2
fi

exit "$status"
