#!/usr/bin/env bash
# Judge this tree on the newest LLVM/Clang/MLIR 23 and Node 24: the local half of the policy in
# .claude/skills/bcir-latest-toolchain/SKILL.md. CI's oracle-llvm-latest, c-rails-llvm-latest,
# mlir-rail-validate and the cmake-build clang23 cell are the remote half. Every change is checked
# here as well as on the CI-default toolchain (Clang 18 / GCC 13) -- a pass there says nothing
# about 23.
#
#   bash tools/local/check_latest.sh                      every leg (about 35 minutes)
#   bash tools/local/check_latest.sh --legs cmake,gate    a subset, in the order below
#   bash tools/local/check_latest.sh --allow-outdated     record, not fail, a toolchain behind upstream
#   bash tools/local/check_latest.sh --cfront-sanitize    the gate leg runs the cfront ASan/UBSan sweep too
#   bash tools/local/check_latest.sh --out DIR            keep the logs in DIR (default: a new temp dir)
#
# Legs (serialized, two workers -- AGENTS.md):
#   status    tools/local/latest_toolchain.py status: the installed LLVM 23 and Node 24 are the newest
#   confirm   the oracle's resolver finds one coherent LLVM 23; node is 24; clang 23's fuzzer,
#             ASan/UBSan and TSan runtimes link and run (TSan through tools/build/sanitizer.py)
#   cmake     a fresh `clang23` preset tree, BCIR_REQUIRE_TSAN=ON: build, ctest -L section, -L build
#   fuzz      a fresh `fuzzer` preset tree on clang 23: ctest -L fuzz
#   gate      tools/c/check_runtime.sh on clang 23, BCIR_REQUIRE_TSAN=1
#   thorough  the thorough tier on LLVM 23 + Node 24, BCIR_REQUIRE_LLVM=1
#   mlir      a fresh `mlir` preset tree on MLIR 23: bcir-opt under the top-level project, -L mlir
#
# Each leg's log stays in the output directory. The summary names every leg PASS, FAIL or WAIVED
# (only `status`, only by --allow-outdated, and the summary says so); the exit status is 0 only
# when every selected leg passed or was waived, 1 when one failed, 2 on a usage error.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEGS_ALL="status confirm cmake fuzz gate thorough mlir"
PY="${PYTHON:-python3}"
legs="${LEGS_ALL}"; out=""; allow_outdated=0; cfront_sanitize=0
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --legs) [ $# -ge 2 ] || { usage >&2; exit 2; }; legs="${2//,/ }"; shift 2 ;;
    --out) [ $# -ge 2 ] || { usage >&2; exit 2; }; out="$2"; shift 2 ;;
    --allow-outdated) allow_outdated=1; shift ;;
    --cfront-sanitize) cfront_sanitize=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "check_latest: unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
done
selected=""
for leg in ${LEGS_ALL}; do  # the documented order, whatever order --legs named them in
  case " ${legs} " in *" ${leg} "*) selected="${selected} ${leg}" ;; esac
done
for leg in ${legs}; do
  case " ${LEGS_ALL} " in *" ${leg} "*) ;; *) echo "check_latest: unknown leg '${leg}' (legs: ${LEGS_ALL})" >&2; exit 2 ;; esac
done
[ -n "${selected}" ] || { echo "check_latest: no leg selected (legs: ${LEGS_ALL})" >&2; exit 2; }
out="${out:-$(mktemp -d "${TMPDIR:-/tmp}/bcir-check-latest.XXXXXX")}"
mkdir -p "${out}" || exit 2
cd "${ROOT}" || exit 2

# The toolchain every leg below runs on: LLVM_BIN and BCIR_NODE_BIN, found by the same tool that
# judges whether they are the newest.
if ! exports="$("${PY}" tools/local/latest_toolchain.py env)"; then
  echo "check_latest: FAIL: LLVM 23 or Node 24 is not installed here; see: ${PY} tools/local/latest_toolchain.py status"
  exit 1
fi
eval "${exports}"
LATEST_PATH="${PATH}"  # the env exports put LLVM_BIN and BCIR_NODE_BIN first
PATH="${PATH#"${LLVM_BIN}:${BCIR_NODE_BIN}:"}"  # the legs opt in to LATEST_PATH one command at a time

leg_status() {
  "${PY}" tools/local/latest_toolchain.py status; local rc=$?
  [ "${rc}" -eq 0 ] && return 0
  if [ "${allow_outdated}" -eq 1 ]; then
    echo "WAIVED by --allow-outdated (latest_toolchain.py exited ${rc}: not every tool is the newest release)"
    return 3
  fi
  return 1
}

leg_confirm() {
  (
    set -e
    export PATH="${LATEST_PATH}" LLVM_BIN
    case "$(clang --version)" in
      *"clang version 23."*) clang --version | head -1 ;;
      *) echo "clang on PATH is not 23: $(clang --version | head -1)"; exit 1 ;;
    esac
    case "$(node --version)" in
      v24.*) echo "node $(node --version)" ;;
      *) echo "node on PATH is not 24: $(node --version)"; exit 1 ;;
    esac
    "${PY}" - <<'PY'
from bcir.toolchain import resolve_llvm_tools

t = resolve_llvm_tools("clang", "llvm-as", "llc", "lli", "opt", "wasm-ld", pipeline="check_latest on LLVM 23")
assert t.ok and t.major == 23, t.message
print(t.message)
for name, path in t.paths.items():
    print(f"  {name}: {path}")
PY
    work="$(mktemp -d)"; trap 'rm -rf "${work}"' EXIT
    printf '#include <stdint.h>\n#include <stddef.h>\nint LLVMFuzzerTestOneInput(const uint8_t *d, size_t n) { return n > 0 && d[0] == 0 ? 0 : 0; }\n' > "${work}/fz.c"
    clang -std=c23 -g -fsanitize=fuzzer,address,undefined "${work}/fz.c" -o "${work}/fz"
    "${work}/fz" -runs=16 > "${work}/fz.out" 2>&1
    echo "fuzzer + ASan + UBSan runtimes: link and run"
    "${PY}" tools/build/sanitizer.py --cc "$(command -v clang)" thread
  )
}

leg_cmake() {
  local dir=build/cmake-clang23
  rm -rf "${dir}"
  PATH="${LATEST_PATH}" cmake --preset clang23 -DBCIR_BUILD_MLIR=OFF -DBCIR_REQUIRE_TSAN=ON || return 1
  PATH="${LATEST_PATH}" cmake --build --preset clang23 || return 1
  ctest --test-dir "${dir}" -L section -j 2 --output-on-failure || return 1
  ctest --test-dir "${dir}" -L build -j 2 --output-on-failure || return 1
}

leg_fuzz() {
  local dir=build/cmake-fuzzer23
  rm -rf "${dir}"
  PATH="${LATEST_PATH}" cmake --preset fuzzer -B "${dir}" -DCMAKE_C_COMPILER=clang-23 \
    -DCMAKE_CXX_COMPILER=clang++-23 -DBCIR_BUILD_MLIR=OFF || return 1
  PATH="${LATEST_PATH}" cmake --build "${dir}" -j 2 || return 1
  ctest --test-dir "${dir}" -L fuzz -j 2 --output-on-failure || return 1
}

leg_gate() {
  local skip=()
  [ "${cfront_sanitize}" -eq 1 ] || skip=(BCIR_SKIP_CFRONT_SANITIZE=1)
  env PATH="${LATEST_PATH}" CC="${LLVM_BIN}/clang" CLANG="${LLVM_BIN}/clang" "${skip[@]}" \
    BCIR_REQUIRE_TSAN=1 bash tools/c/check_runtime.sh
}

leg_thorough() {
  env PATH="${LATEST_PATH}" LLVM_BIN="${LLVM_BIN}" BCIR_THOROUGH=1 BCIR_REQUIRE_LLVM=1 \
    "${PY}" -m bcir.tests.run_all --tier thorough -j 2
}

leg_mlir() {
  local dir=build/cmake-mlir23 prefix
  prefix="$(dirname "${LLVM_BIN}")"
  if [ ! -f "${prefix}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    echo "no MLIR 23 package beside ${LLVM_BIN} (BCIR_LOCAL_FULL=1 bash tools/local/setup_mlir.sh, or apt's libmlir-23-dev)"
    return 1
  fi
  rm -rf "${dir}"
  # A conda-forge prefix carries the libstdc++ its MLIR was built against (tools/local/env_mlir.sh).
  local libs="${LD_LIBRARY_PATH:-}"
  [ -d "${prefix}/conda-meta" ] && libs="${prefix}/lib${libs:+:${libs}}"
  env PATH="${LATEST_PATH}" LD_LIBRARY_PATH="${libs}" MLIR_DIR="${prefix}/lib/cmake/mlir" \
    cmake --preset mlir -B "${dir}" -DCMAKE_C_COMPILER=clang-23 -DCMAKE_CXX_COMPILER=clang++-23 \
    -DLLVM_DIR="${prefix}/lib/cmake/llvm" || return 1
  env PATH="${LATEST_PATH}" LD_LIBRARY_PATH="${libs}" cmake --build "${dir}" -j 2 || return 1
  env LD_LIBRARY_PATH="${libs}" ctest --test-dir "${dir}" -L mlir -j 2 --output-on-failure || return 1
}

detail() {  # <leg> <log> -> the one line a reader needs from it
  case "$1" in
    status) grep -E '^latest-toolchain: [A-Z]+$' "$2" | tail -1 ;;
    confirm) grep -E 'resolved|^node v|runtimes|sanitizer thread' "$2" | tr '\n' ' ' | cut -c1-160 ;;
    cmake) printf '%s; %s; warnings=%s' "$(grep -E 'tests passed' "$2" | tr '\n' ' ')" \
      "$(grep -E '^section-parity: [0-9]' "${ROOT}/build/cmake-clang23/Testing/Temporary/LastTest.log" 2>/dev/null | head -1)" \
      "$(grep -c 'warning:' "$2")" ;;
    fuzz|mlir) grep -E 'tests passed' "$2" | tail -1 ;;
    gate) printf 'PASS lines=%s SKIP=%s FAIL=%s' "$(grep -c 'PASS' "$2")" "$(grep -c 'SKIP' "$2")" "$(grep -c 'FAIL' "$2")" ;;
    thorough) grep -E 'passed, ' "$2" | tail -1 ;;
  esac
}

failed=0
summary=()
echo "check_latest: $(git rev-parse --short HEAD 2>/dev/null || echo '?') on ${LLVM_BIN} + ${BCIR_NODE_BIN}; logs in ${out}"
for leg in ${selected}; do
  log="${out}/${leg}.log"
  started="$(date +%s)"
  echo "check_latest: $(date -u +%T) ${leg} ..."
  "leg_${leg}" > "${log}" 2>&1
  rc=$?
  case "${rc}" in
    0) verdict=PASS ;;
    3) verdict=WAIVED ;;
    *) verdict=FAIL; failed=1 ;;
  esac
  summary+=("$(printf '%-9s %-7s %5ss  %s' "${leg}" "${verdict}" "$(( $(date +%s) - started ))" "$(detail "${leg}" "${log}")")")
  echo "check_latest:   ${summary[-1]}"
done
echo "check_latest: summary (LLVM $("${LLVM_BIN}/clang" --version | sed -n 's/.*clang version \([0-9.]*\).*/\1/p' | head -1), Node $("${BCIR_NODE_BIN}/node" --version))"
for line in "${summary[@]}"; do echo "  ${line}"; done
if [ "${failed}" -ne 0 ]; then
  echo "check_latest: FAIL (logs: ${out})"
  exit 1
fi
echo "check_latest: PASS"
