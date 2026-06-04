#!/usr/bin/env bash
# Shared helpers for runtime/llvm validation scripts.
#
# The validation suite intentionally uses one concrete target triple so LLVM's
# verifier can infer a data layout even for interface-only modules. Newer opt
# builds reject modules that declare unknown-unknown-unknown with an empty data
# layout; keeping this default here makes local and CI failures reproducible.

set -euo pipefail

BCIR_VALIDATE_TRIPLE=${BCIR_VALIDATE_TRIPLE:-x86_64-unknown-linux-gnu}

bcir_repo_root() {
  local script_dir
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  cd -- "$script_dir/../.." && pwd
}

REPO_ROOT=${REPO_ROOT:-$(bcir_repo_root)}
BUILD_DIR=${BUILD_DIR:-$REPO_ROOT/build}

log() {
  printf '[validate] %s\n' "$*"
}

require_tool() {
  local tool=$1
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf 'error: missing required tool on PATH: %s\n' "$tool" >&2
    return 127
  fi
}

require_tools() {
  local tool
  for tool in "$@"; do
    require_tool "$tool"
  done
}

require_file() {
  local file=$1
  if [ ! -e "$REPO_ROOT/$file" ]; then
    printf 'error: required file does not exist: %s\n' "$file" >&2
    return 1
  fi
}

require_executable() {
  local file=$1
  require_file "$file"
  if [ ! -x "$REPO_ROOT/$file" ]; then
    printf 'error: required file is not executable: %s\n' "$file" >&2
    return 1
  fi
}

run_step() {
  local label=$1
  shift
  log "$label: $*"
  (cd "$REPO_ROOT" && "$@")
}

require_pattern() {
  local pattern=$1
  local file=$2
  local description=$3
  if ! grep -q -- "$pattern" "$REPO_ROOT/$file"; then
    printf 'error: expected %s in %s\n' "$description" "$file" >&2
    printf '       missing grep pattern: %s\n' "$pattern" >&2
    return 1
  fi
}

log_toolchain() {
  log "repo root: $REPO_ROOT"
  log "build dir: ${BUILD_DIR#"$REPO_ROOT/"}"
  log "validation triple: $BCIR_VALIDATE_TRIPLE"
  local tool
  for tool in "$@"; do
    log "$tool: $(command -v "$tool")"
    "$tool" --version 2>/dev/null | sed -n '1p' | sed "s/^/[validate] $tool version: /"
  done
}

init_build_dir() {
  mkdir -p "$BUILD_DIR"
}
