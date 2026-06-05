#!/usr/bin/env bash
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
EXERCISES_DIR="$TRAINING_ROOT/exercises"

status=0
ir_count=0
md_count=0
template_count=0

require_tool() {
  local tool=$1
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'error: required tool not found on PATH: %s\n' "$tool" >&2
    exit 127
  fi
}

relpath() {
  local path=$1
  printf '%s' "${path#"$REPO_ROOT/"}"
}

run_step() {
  local label=$1
  local file=$2
  shift 2

  printf '[%s] %s ... ' "$label" "$(relpath "$file")"

  local output
  if output=$("$@" 2>&1); then
    printf 'ok\n'
    return 0
  fi

  printf 'FAILED\n'
  if [ -n "$output" ]; then
    printf '%s\n' "$output" | sed 's/^/    /'
  fi
  return 1
}

check_markdown_solution() {
  local file=$1

  printf '[markdown] %s ... ' "$(relpath "$file")"

  if [ ! -s "$file" ]; then
    printf 'FAILED\n'
    printf '    markdown solution is missing or empty\n'
    return 1
  fi

  if ! sed -n '1p' "$file" | grep -Eq '^# '; then
    printf 'FAILED\n'
    printf '    markdown solution should start with a top-level heading\n'
    return 1
  fi

  printf 'ok\n'
  return 0
}

check_prompt_template() {
  local file=$1

  printf '[template] %s ... ' "$(relpath "$file")"

  if [ ! -s "$file" ]; then
    printf 'FAILED\n'
    printf '    prompt template is missing or empty\n'
    return 1
  fi

  if ! sed -n '1p' "$file" | grep -Eq '^# Agent template:'; then
    printf 'FAILED\n'
    printf '    prompt template should start with "# Agent template:"\n'
    return 1
  fi

  if ! grep -q '^## Verification checklist$' "$file"; then
    printf 'FAILED\n'
    printf '    prompt template should include a Verification checklist section\n'
    return 1
  fi

  printf 'ok\n'
  return 0
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
  done < <(compgen -c | grep -E "^${base}-[0-9]+$" | sort -V)

  return 1
}

LLVM_AS=$(find_tool llvm-as) || {
  printf 'error: required tool not found on PATH: llvm-as (or llvm-as-N)\n' >&2
  exit 127
}

OPT=$(find_tool opt) || {
  printf 'error: required tool not found on PATH: opt (or opt-N)\n' >&2
  exit 127
}


mapfile -d '' solutions < <(find "$EXERCISES_DIR" -maxdepth 1 -type f \
  \( -name '*.solution.ll' -o -name '*.good.ll' \) -print0 | sort -z)

for file in "${solutions[@]}"; do
  ir_count=$((ir_count + 1))
  run_step llvm-as "$file" "$LLVM_AS" "$file" -o /dev/null || status=1
  run_step verify "$file" "$OPT" -passes=verify "$file" -o /dev/null || status=1
done

mapfile -d '' markdown_solutions < <(find "$EXERCISES_DIR" -maxdepth 1 -type f -name '*.solution.md' -print0 | sort -z)

for file in "${markdown_solutions[@]}"; do
  md_count=$((md_count + 1))
  check_markdown_solution "$file" || status=1
done

mapfile -d '' prompt_templates < <(find "$EXERCISES_DIR/templates" -maxdepth 1 -type f -name '*.prompt.md' -print0 2>/dev/null | sort -z)

for file in "${prompt_templates[@]}"; do
  template_count=$((template_count + 1))
  check_prompt_template "$file" || status=1
done

if [ "$ir_count" -eq 0 ] && [ "$md_count" -eq 0 ] && [ "$template_count" -eq 0 ]; then
  printf 'No checked-in exercise solutions or templates found under %s\n' "$(relpath "$EXERCISES_DIR")"
  exit 1
fi

if [ "$status" -eq 0 ]; then
  printf 'Verified %d LLVM IR solution/review artifact(s), %d markdown solution(s), and %d prompt template(s).\n' "$ir_count" "$md_count" "$template_count"
else
  printf 'One or more checked-in exercise solutions, review artifacts, or prompt templates failed verification.\n' >&2
fi

exit "$status"
