#!/usr/bin/env bash
# check_solver.sh -- the gate of the hosted C++ solver library (runtime/cpp; PORT-5 laid it down with its first
# solver, the exact scheduler bcir_exact.cpp behind bcir_exact.h).
#
# The library is C++23 behind a C ABI (the native port audit's rules, docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md
# section 4.4). What is checked HERE:
#
#   1. THE BUILD. Warning-clean under C++23 with -Wpedantic and without RTTI, on every C++ compiler present: the
#      library uses no RTTI, and C++23 is the CI floor's (GCC 13, Clang 18 over libstdc++ 13 through the
#      bcir_expected.hpp shim).
#   2. THE FACADE. The object defines one strong symbol, `bcir_exact_schedule` -- no C++ name crosses the ABI -- and
#      no static initializer: the library has no global state to construct.
#   3. THE CONTRACTS. test_exact: every refusal, the budget's stops, determinism, and every allocation of a run
#      refused in turn (NOMEM, nothing left live).
#   4. ASan + UBSan over the contracts and over fuzz_exact's generated replay (its invariants: statuses, the
#      certificate's order, every placement a schedule, determinism, allocation discipline).
#   5. libFuzzer over the facade, bounded (FUZZ_EXACT_RUNS, default 20000), where a clang++ with compiler-rt is
#      present (CLANGXX names one).
#
# The PARITY with the oracle is not here: bcir/tests/test_cpp_exact.py holds `test_exact FILE` to `exact_schedule`
# over the generated corpus, and checks every placement it returns with code that shares nothing with the solver.
#
# Self-skips (exit 0) when no C++23 compiler is present, like the other C++ gates; a sanitizer or libFuzzer runtime
# the host lacks is a SKIP line naming it, never a pass.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="${ROOT}/runtime/c"
CPP="${ROOT}/runtime/cpp"

CXX="${CXX:-}"
if [ -z "${CXX}" ]; then
  for candidate in clang++ g++ c++; do
    if command -v "${candidate}" >/dev/null 2>&1; then CXX="${candidate}"; break; fi
  done
fi
if [ -z "${CXX}" ]; then
  echo "SKIP: no C++ compiler; the hosted solver library is optional by design"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

printf '#include <memory_resource>\nint main(){std::pmr::monotonic_buffer_resource r; return 0;}\n' > "${tmp}/probe.cpp"
if ! "${CXX}" -std=c++23 "${tmp}/probe.cpp" -o "${tmp}/probe" 2>/dev/null; then
  echo "SKIP: ${CXX} does not build C++23 with <memory_resource>; the hosted solver library needs both"
  exit 0
fi

strict=(-std=c++23 -O2 -Wall -Wextra -Wpedantic -Werror -I "${C}" -I "${CPP}")

# 1. warning-clean under C++23 and without RTTI, on every C++ compiler present
compilers=("${CXX}")
for other in g++ clang++; do
  if command -v "${other}" >/dev/null 2>&1 && [ "$(command -v "${other}")" != "$(command -v "${CXX}")" ] \
     && "${other}" -std=c++23 "${tmp}/probe.cpp" -o "${tmp}/probe" 2>/dev/null; then
    compilers+=("${other}")
  fi
done
for cxx in "${compilers[@]}"; do
  "${cxx}" "${strict[@]}" -fno-rtti -c "${CPP}/bcir_exact.cpp" -o "${tmp}/bcir_exact.o" 2>"${tmp}/build.err" \
    || { echo "FAIL: bcir_exact.cpp is not warning-clean under C++23 without RTTI (${cxx})"; cat "${tmp}/build.err"; exit 1; }
  echo "  ok: bcir_exact.cpp is warning-clean under C++23 -Wpedantic -fno-rtti (${cxx})"
done
"${CXX}" "${strict[@]}" -fno-rtti -c "${CPP}/bcir_exact.cpp" -o "${tmp}/bcir_exact.o"

# 2. the facade: one strong symbol, no static initializer
if command -v nm >/dev/null 2>&1; then
  strong="$(nm -g --defined-only "${tmp}/bcir_exact.o" | awk '$2 == "T" { print $3 }' | sort | tr '\n' ' ')"
  if [ "${strong}" != "bcir_exact_schedule " ]; then
    echo "FAIL: the library's object defines strong symbols '${strong}', not the facade's bcir_exact_schedule alone"
    exit 1
  fi
  inits="$(nm "${tmp}/bcir_exact.o" | awk '/_GLOBAL__sub_I|_GLOBAL__I_/ { n++ } END { print n + 0 }')"
  if [ "${inits}" -ne 0 ]; then
    echo "FAIL: the library has a static initializer (global state to construct)"
    exit 1
  fi
  echo "  ok: the object defines bcir_exact_schedule alone and constructs no global state"
else
  echo "  SKIP facade symbols: UNAVAILABLE (no nm on this host)"
fi

# 3. the contracts
"${CXX}" "${strict[@]}" "${CPP}/test_exact.cpp" "${tmp}/bcir_exact.o" -o "${tmp}/test_exact" 2>"${tmp}/build.err" \
  || { echo "FAIL: test_exact did not build"; cat "${tmp}/build.err"; exit 1; }
contracts="$("${tmp}/test_exact" 2>&1)" || { echo "FAIL: the facade's contracts"; echo "${contracts}" | tail -n 30; exit 1; }
[ "${contracts}" = "test_exact: OK" ] || { echo "FAIL: test_exact said '${contracts}'"; exit 1; }
echo "  PASS the facade's contracts: refusals, budget stops, determinism, every allocation refused in turn"

# 4. ASan + UBSan over the contracts and the fuzzer's replay
san=(-std=c++23 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -I "${C}" -I "${CPP}")
# the first C++ compiler present whose sanitizer runtimes work (a clang++ without compiler-rt has none)
printf 'int main(void){return 0;}\n' > "${tmp}/sprobe.cpp"
SANCXX=""
for cxx in "${compilers[@]}"; do
  if "${cxx}" -fsanitize=address,undefined "${tmp}/sprobe.cpp" -o "${tmp}/sprobe" 2>/dev/null && "${tmp}/sprobe"; then
    SANCXX="${cxx}"; break
  fi
done
if [ -n "${SANCXX}" ]; then
  "${SANCXX}" "${san[@]}" "${CPP}/test_exact.cpp" "${CPP}/bcir_exact.cpp" -o "${tmp}/test_exact_san" 2>"${tmp}/build.err" \
    || { echo "FAIL: sanitized test_exact build"; cat "${tmp}/build.err"; exit 1; }
  "${SANCXX}" "${san[@]}" "${CPP}/fuzz_exact.cpp" "${CPP}/bcir_exact.cpp" -o "${tmp}/fuzz_exact_san" 2>"${tmp}/build.err" \
    || { echo "FAIL: sanitized fuzz_exact build"; cat "${tmp}/build.err"; exit 1; }
  "${tmp}/test_exact_san" > "${tmp}/san.out" 2>&1 || { echo "FAIL: the contracts under ASan/UBSan"; tail -n 30 "${tmp}/san.out"; exit 1; }
  replay="$("${tmp}/fuzz_exact_san" 2>"${tmp}/san.err")" \
    || { echo "FAIL: fuzz_exact's replay under ASan/UBSan: ${replay}"; tail -n 30 "${tmp}/san.err"; exit 1; }
  echo "  PASS the contracts and the replay under ASan + UBSan, ${SANCXX} (${replay#fuzz_exact: })"
else
  echo "  SKIP ASan/UBSan: UNAVAILABLE (no C++ compiler here has a working sanitizer runtime)"
fi

# 5. libFuzzer over the facade, bounded
CLANGXX="${CLANGXX:-$(command -v clang++ || true)}"
runs="${FUZZ_EXACT_RUNS:-20000}"
case "${runs}" in ''|*[!0-9]*|0) echo "FAIL: FUZZ_EXACT_RUNS must be a positive count, not '${runs}'"; exit 1 ;; esac
if [ -n "${CLANGXX}" ] && printf 'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}\n' \
     | "${CLANGXX}" -std=c++23 -fsanitize=fuzzer,address -x c++ - -o "${tmp}/fprobe" 2>/dev/null; then
  "${CLANGXX}" -std=c++23 -O1 -g -DBCIR_LIBFUZZER -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
    -I "${C}" -I "${CPP}" "${CPP}/fuzz_exact.cpp" "${CPP}/bcir_exact.cpp" -o "${tmp}/fuzz_exact_lf" 2>"${tmp}/build.err" \
    || { echo "FAIL: the libFuzzer target did not build"; cat "${tmp}/build.err"; exit 1; }
  mkdir -p "${tmp}/corpus"
  (cd "${tmp}" && ./fuzz_exact_lf -runs="${runs}" -max_len=512 -seed=1 corpus > "${tmp}/fuzz.out" 2>&1) \
    || { echo "FAIL: libFuzzer found an input that breaks the facade"; tail -n 40 "${tmp}/fuzz.out"; exit 1; }
  done_line="$(awk '/^Done / { line = $0 } END { print line }' "${tmp}/fuzz.out")"
  [ -n "${done_line}" ] || { echo "FAIL: libFuzzer did not report its runs"; tail -n 20 "${tmp}/fuzz.out"; exit 1; }
  echo "  PASS libFuzzer over the facade (${done_line})"
else
  echo "  SKIP libFuzzer: UNAVAILABLE (no clang++ with compiler-rt; CLANGXX names one)"
fi
echo "PASS the hosted solver library: built, the facade alone exported, contracts, sanitizers, fuzzing"
