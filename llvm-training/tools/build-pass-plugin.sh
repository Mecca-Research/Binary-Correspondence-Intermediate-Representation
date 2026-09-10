#!/usr/bin/env bash
# Build the out-of-tree LLVM pass plugin in
# llvm-training/17-new-pass-manager/examples/pass-plugin/ and exercise it
# through `opt`.
#
# This gate is OPTIONAL by construction. It needs LLVM development headers (the
# `llvm-dev` package or an installed LLVM tree), CMake, and a C++ compiler --
# none of which the core corpus gates assume. When any of them is missing, or
# when `opt` and `llvm-config` come from different LLVM major versions, the
# script prints why and exits 0. It never reports a skip as a pass: the final
# line always states which of the two it was.
set -u -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TRAINING_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
REPO_ROOT=$(cd -- "$TRAINING_ROOT/.." && pwd)
PLUGIN_DIR="$TRAINING_ROOT/17-new-pass-manager/examples/pass-plugin"
BUILD_DIR=${PASS_PLUGIN_BUILD_DIR:-"$REPO_ROOT/build/bcir-pass-plugin"}

status=0
checks=0

relpath() {
  printf '%s' "${1#"$REPO_ROOT/"}"
}

skip() {
  printf 'BCIR pass plugin gate: SKIPPED (%s)\n' "$1"
  exit 0
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
  done < <(compgen -c | sed -n -E "s/^(${base}-[0-9]+)$/\\1/p" | sort -Vu)
  return 1
}

major_version() {
  printf '%s' "${1%%.*}"
}

# --- preconditions ----------------------------------------------------------

LLVM_CONFIG=$(find_tool llvm-config || true)
OPT=$(find_tool opt || true)
CMAKE=$(command -v cmake || true)

[ -n "$LLVM_CONFIG" ] || skip "llvm-config not on PATH"
[ -n "$OPT" ] || skip "opt not on PATH"
[ -n "$CMAKE" ] || skip "cmake not on PATH"
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 ||
  command -v clang++ >/dev/null 2>&1 || skip "no C++ compiler on PATH"

CMAKE_DIR=$("$LLVM_CONFIG" --cmakedir 2>/dev/null || true)
[ -n "$CMAKE_DIR" ] && [ -f "$CMAKE_DIR/LLVMConfig.cmake" ] ||
  skip "LLVM development files not installed (no LLVMConfig.cmake under '${CMAKE_DIR:-?}'); install llvm-dev"

[ -f "$CMAKE_DIR/../../../include/llvm/Passes/PassBuilder.h" ] ||
  [ -f "$("$LLVM_CONFIG" --includedir)/llvm/Passes/PassBuilder.h" ] ||
  skip "LLVM headers not installed (no llvm/Passes/PassBuilder.h); install llvm-dev"

# A plugin is only loadable by an `opt` from the same LLVM major version: the
# plugin API version and the C++ ABI of the pass classes both track it.
CONFIG_VERSION=$("$LLVM_CONFIG" --version 2>/dev/null)
OPT_VERSION=$("$OPT" --version 2>/dev/null | sed -n -E 's/.*LLVM version ([0-9]+\.[0-9]+\.[0-9]+).*/\1/p' | head -1)
if [ -n "$OPT_VERSION" ] &&
   [ "$(major_version "$CONFIG_VERSION")" != "$(major_version "$OPT_VERSION")" ]; then
  skip "incoherent toolchain: llvm-config is $CONFIG_VERSION but $OPT is $OPT_VERSION"
fi

printf 'BCIR pass plugin gate: building against LLVM %s (%s)\n' "$CONFIG_VERSION" "$LLVM_CONFIG"

# --- build ------------------------------------------------------------------

build_log=$(mktemp "${TMPDIR:-/tmp}/bcir-pass-plugin-build.XXXXXX")
trap 'rm -f "$build_log"' EXIT

if ! "$CMAKE" -S "$PLUGIN_DIR" -B "$BUILD_DIR" \
      -DLLVM_DIR="$CMAKE_DIR" -DCMAKE_BUILD_TYPE=Release >"$build_log" 2>&1; then
  printf 'BCIR pass plugin gate: FAILED (cmake configure)\n' >&2
  sed 's/^/    /' "$build_log" >&2
  exit 1
fi

# Two workers: this repository keeps local build concurrency bounded.
if ! "$CMAKE" --build "$BUILD_DIR" -j "${PASS_PLUGIN_JOBS:-2}" >"$build_log" 2>&1; then
  printf 'BCIR pass plugin gate: FAILED (build)\n' >&2
  sed 's/^/    /' "$build_log" >&2
  exit 1
fi

PLUGIN=""
for candidate in \
  "$BUILD_DIR/libBCIRRegisterBinding.so" \
  "$BUILD_DIR/BCIRRegisterBinding.so" \
  "$BUILD_DIR/libBCIRRegisterBinding.dylib" \
  "$BUILD_DIR/BCIRRegisterBinding.dll"; do
  if [ -f "$candidate" ]; then
    PLUGIN=$candidate
    break
  fi
done

if [ -z "$PLUGIN" ]; then
  printf 'BCIR pass plugin gate: FAILED (built library not found under %s)\n' \
    "$(relpath "$BUILD_DIR")" >&2
  exit 1
fi

printf '[build] %s\n' "$(relpath "$PLUGIN")"

# --- assertions -------------------------------------------------------------
#
# Every check names the law it exists to test and fails loudly when the law
# stops holding. A check that cannot fail is not a check.

run_opt() {
  # usage: run_opt <passes> <file>
  "$OPT" -load-pass-plugin="$PLUGIN" -passes="$1" -disable-output "$2" 2>&1
}

check() {
  # usage: check <description> <expected-substring> <actual>
  local description=$1 expected=$2 actual=$3
  checks=$((checks + 1))
  printf '[check] %s ... ' "$description"
  case "$actual" in
    *"$expected"*)
      printf 'ok\n'
      ;;
    *)
      printf 'FAILED\n'
      printf '    expected to contain: %s\n' "$expected"
      printf '    actual output:\n'
      printf '%s\n' "$actual" | sed 's/^/      /'
      status=1
      ;;
  esac
}

OK_FIXTURE="$PLUGIN_DIR/binding-ok.ll"
COLLISION_FIXTURE="$PLUGIN_DIR/binding-collision.ll"
UNROLL_FIXTURE="$PLUGIN_DIR/unroll-breaks-binding.ll"

# 1. A conforming module reports no violation.
check "conforming module has zero violations" \
  "4 binding(s), 0 violation(s)" \
  "$(run_opt 'bcir-verify-bindings' "$OK_FIXTURE")"

# 2. The checker actually fires. This is the check that proves the gate can
#    fail; without it every other check passes against a pass that does nothing.
check "duplicate binding is reported" \
  "register 'r2' is bound to more than one value" \
  "$(run_opt 'bcir-verify-bindings' "$COLLISION_FIXTURE")"

check "duplicate binding is counted" \
  "4 binding(s), 1 violation(s)" \
  "$(run_opt 'bcir-verify-bindings' "$COLLISION_FIXTURE")"

# 3. Report-only mode still exits 0; strict mode does not. A verdict that
#    cannot change the exit status is a log line, not a gate.
checks=$((checks + 1))
printf '[check] report-only mode exits 0 ... '
if run_opt 'bcir-verify-bindings' "$COLLISION_FIXTURE" >/dev/null 2>&1; then
  printf 'ok\n'
else
  printf 'FAILED (expected exit 0)\n'
  status=1
fi

checks=$((checks + 1))
printf '[check] strict mode exits nonzero ... '
if run_opt 'bcir-verify-bindings<strict>' "$COLLISION_FIXTURE" >/dev/null 2>&1; then
  printf 'FAILED (expected a nonzero exit)\n'
  status=1
else
  printf 'ok\n'
fi

# 4. The analysis is reachable by name through the printer convention.
check "printer reports per-function bindings" \
  "bcir-bindings: function 'kernel': 3 binding(s), 0 collision(s)" \
  "$(run_opt 'function(print<bcir-bindings>)' "$OK_FIXTURE")"

# 5. The transform removes the bindings, and the analysis it abandoned is
#    recomputed rather than served stale.
check "strip-bindings clears the contract" \
  "0 binding(s), 0 violation(s)" \
  "$(run_opt 'function(bcir-strip-bindings),bcir-verify-bindings' "$COLLISION_FIXTURE")"

# 6. The point of the whole exercise: a stock LLVM pass breaks the invariant,
#    and the custom pass is what notices. Counts are LLVM-version dependent, so
#    assert the stable shape -- clean before, violations after.
check "annotated loop is clean before unrolling" \
  "3 binding(s), 0 violation(s)" \
  "$(run_opt 'bcir-verify-bindings' "$UNROLL_FIXTURE")"

checks=$((checks + 1))
printf '[check] full unrolling breaks the 1:1 contract ... '
unrolled=$(run_opt 'function(loop(loop-unroll-full)),bcir-verify-bindings' "$UNROLL_FIXTURE")
violations=$(printf '%s\n' "$unrolled" | sed -n -E 's/.*binding\(s\), ([0-9]+) violation\(s\).*/\1/p' | tail -1)
if [ -n "$violations" ] && [ "$violations" -gt 0 ] 2>/dev/null; then
  printf 'ok (%s violation(s) after unrolling)\n' "$violations"
else
  printf 'FAILED\n'
  printf '    expected a positive violation count after full unrolling\n'
  printf '%s\n' "$unrolled" | sed 's/^/      /'
  status=1
fi

# --- verdict ----------------------------------------------------------------

if [ "$status" -eq 0 ]; then
  printf 'BCIR pass plugin gate: PASSED (%s checks against LLVM %s)\n' "$checks" "$CONFIG_VERSION"
else
  printf 'BCIR pass plugin gate: FAILED (%s checks against LLVM %s)\n' "$checks" "$CONFIG_VERSION" >&2
fi

exit "$status"
