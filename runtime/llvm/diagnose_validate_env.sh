#!/usr/bin/env bash
# Print diagnostic context for runtime/llvm validation environment skew.
#
# This entry point is intentionally read-only: it reports paths, version banners,
# and prerequisite file availability without creating build outputs or running LLVM
# transforms. CI may invoke it with continue-on-error/`|| true` to attach useful
# environment context without changing validation pass/fail behavior.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=runtime/llvm/validate_common.sh
source "$SCRIPT_DIR/validate_common.sh"

print_diag() {
  printf '[diagnose] %s\n' "$*"
}

first_version_line() {
  local tool=$1
  local output status

  set +e
  output=$(cd "$REPO_ROOT" && "$tool" --version 2>&1)
  status=$?
  set -e

  if [ -n "$output" ]; then
    printf '%s' "$output" | sed -n '1p'
  else
    printf 'no --version output (exit %s)' "$status"
  fi
}

print_tool() {
  local tool=$1
  local resolved

  set +e
  resolved=$(cd "$REPO_ROOT" && command -v -- "$tool" 2>/dev/null)
  set -e

  if [ -n "$resolved" ]; then
    print_diag "$tool path: $resolved"
    print_diag "$tool version: $(first_version_line "$tool")"
  else
    print_diag "$tool path: not found"
    print_diag "$tool version: unavailable"
  fi
}

check_file() {
  local file=$1
  if [ -e "$REPO_ROOT/$file" ]; then
    print_diag "file ok: $file"
  else
    print_diag "file missing: $file"
  fi
}

check_executable() {
  local file=$1
  if [ -x "$REPO_ROOT/$file" ]; then
    print_diag "executable ok: $file"
  elif [ -e "$REPO_ROOT/$file" ]; then
    print_diag "executable missing bit: $file"
  else
    print_diag "executable missing: $file"
  fi
}

print_diag "repo root: $REPO_ROOT"
print_diag "build dir: $BUILD_DIR"
print_diag "BCIR_VALIDATE_TRIPLE: $BCIR_VALIDATE_TRIPLE"
print_diag "PATH: $PATH"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  print_diag "os: ${PRETTY_NAME:-unknown}"
else
  print_diag "os: /etc/os-release unavailable"
fi
print_diag "kernel: $(uname -srmo 2>/dev/null || uname -a)"

for tool in llvm-as llvm-link llvm-dis opt tools/bcir-as/bcir-as; do
  print_tool "$tool"
done

print_diag "checking validate_llvm_seed.sh prerequisites"
check_file runtime/llvm/bcir_master_reference_v2.ll

print_diag "checking validate_phase4.sh prerequisites"
check_executable tools/bcir-as/bcir-as
check_file examples/vector_add.bcir
for file in \
  runtime/llvm/bcir_claim_schema.ll \
  runtime/llvm/bcir_claim_accessors.ll \
  runtime/llvm/bcir_registry_schema.ll \
  runtime/llvm/bcir_ops.ll \
  runtime/llvm/bcir_phase_epoch.ll \
  runtime/llvm/bcir_claim_verify.ll \
  runtime/llvm/bcir_gem_seed.ll \
  runtime/llvm/bcir_worklist.ll \
  runtime/llvm/bcir_schedule_schema.ll \
  runtime/llvm/bcir_schedule_accessors.ll \
  runtime/llvm/bcir_prefetch_profiles.ll \
  runtime/llvm/bcir_lane_classifier.ll \
  runtime/llvm/bcir_batch_executor.ll \
  runtime/llvm/bcir_batch_verify.ll \
  runtime/llvm/bcir_stream_pack.ll \
  runtime/llvm/bcir_kbcost.ll \
  runtime/llvm/bcir_blob_schema.ll \
  runtime/llvm/bcir_blob_verify.ll \
  runtime/llvm/bcir_telemetry.ll \
  runtime/llvm/bcir_rehydrate.ll; do
  check_file "$file"
done
