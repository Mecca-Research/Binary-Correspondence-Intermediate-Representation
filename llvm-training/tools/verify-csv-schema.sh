#!/bin/sh
set -u

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TRAINING_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(CDPATH= cd -- "$TRAINING_ROOT/.." && pwd)
EXAMPLES_DIR="$TRAINING_ROOT/15-binary-analysis/examples"

relpath() {
  case "$1" in
    "$REPO_ROOT"/*) printf '%s' "${1#"$REPO_ROOT/"}" ;;
    *) printf '%s' "$1" ;;
  esac
}

expected_columns() {
  case "$1" in
    */bcsa-feature-sample.csv) printf '10' ;;
    */bcsa-feature-variant-wide.csv) printf '7' ;;
    */dynamic-trace-sample.csv) printf '5' ;;
    */perf-counter-sample.csv) printf '10' ;;
    */side-channel-trace-*.csv) printf '5' ;;
    *) return 1 ;;
  esac
}

if [ ! -d "$EXAMPLES_DIR" ]; then
  printf 'error: missing CSV examples directory: %s\n' "$(relpath "$EXAMPLES_DIR")" >&2
  exit 1
fi

csv_list=$(mktemp "${TMPDIR:-/tmp}/llvm-training-csv-schema.XXXXXX")
trap 'rm -f "$csv_list"' EXIT HUP INT TERM

if command -v git >/dev/null 2>&1 && git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$REPO_ROOT" ls-files 'llvm-training/15-binary-analysis/examples/*.csv' |
    sed "s#^#$REPO_ROOT/#" > "$csv_list"
else
  find "$EXAMPLES_DIR" -type f -name '*.csv' -print | sort > "$csv_list"
fi

if [ ! -s "$csv_list" ]; then
  printf 'error: no checked-in CSV examples found under %s\n' "$(relpath "$EXAMPLES_DIR")" >&2
  exit 1
fi

status=0
count=0

while IFS= read -r file; do
  [ -n "$file" ] || continue
  count=$((count + 1))

  expected=$(expected_columns "$file") || {
    printf 'error: %s: no expected schema column count is registered\n' "$(relpath "$file")" >&2
    status=1
    continue
  }

  printf '[csv-schema] %s ... ' "$(relpath "$file")"
  if awk -v expected="$expected" '
    function csv_count(line,    i, c, nextc, in_quotes, fields) {
      sub(/\r$/, "", line)
      fields = 1
      in_quotes = 0
      for (i = 1; i <= length(line); i++) {
        c = substr(line, i, 1)
        if (c == "\"") {
          if (in_quotes && i < length(line) && substr(line, i + 1, 1) == "\"") {
            i++
          } else {
            in_quotes = !in_quotes
          }
        } else if (c == "," && !in_quotes) {
          fields++
        }
      }
      if (in_quotes) {
        return -1
      }
      return fields
    }

    NR == 1 {
      header = $0
      sub(/\r$/, "", header)
      if (header == "") {
        printf("empty header row\n") > "/dev/stderr"
        failed = 1
        next
      }

      header_count = csv_count(header)
      if (header_count < 0) {
        printf("header row has unbalanced quotes\n") > "/dev/stderr"
        failed = 1
        next
      }
      if (header_count != expected) {
        printf("header has %d column(s); expected %d\n", header_count, expected) > "/dev/stderr"
        failed = 1
      }
      next
    }

    {
      row = $0
      sub(/\r$/, "", row)
      if (row == "") {
        next
      }

      data_rows++
      row_count = csv_count(row)
      if (row_count < 0) {
        printf("line %d has unbalanced quotes\n", NR) > "/dev/stderr"
        failed = 1
      } else if (header_count > 0 && row_count != header_count) {
        printf("line %d has %d column(s); header has %d\n", NR, row_count, header_count) > "/dev/stderr"
        failed = 1
      }
    }

    END {
      if (NR == 0) {
        printf("missing header row\n") > "/dev/stderr"
        failed = 1
      }
      if (data_rows == 0) {
        printf("no data rows found\n") > "/dev/stderr"
        failed = 1
      }
      exit failed ? 1 : 0
    }
  ' "$file" 2>"$csv_list.err"; then
    printf 'ok (%s columns)\n' "$expected"
  else
    printf 'FAILED\n'
    sed 's/^/    /' "$csv_list.err" >&2
    status=1
  fi
  rm -f "$csv_list.err"
done < "$csv_list"

if [ "$status" -eq 0 ]; then
  printf 'Validated %d CSV schema fixture(s).\n' "$count"
else
  printf 'One or more CSV schema fixtures failed validation.\n' >&2
fi

exit "$status"
