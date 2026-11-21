#!/usr/bin/env bash
# check_solver.sh -- the gate of the hosted C++ solver library (runtime/cpp): each solver behind its C ABI --
# PORT-5's exact scheduler (bcir_exact.cpp, bcir_exact.h) and PORT-6's finite-domain solver (bcir_csp.cpp,
# bcir_csp.h) -- over the library's shared memory and errors (bcir_solver.hpp).
#
# The library is C++23 behind a C ABI (the native port audit's rules, docs/research/BCIR_NATIVE_PORT_AUDIT_2026-11-09.md
# section 4.4). What is checked HERE:
#
#   1. THE BUILD. Warning-clean under C++23 with -Wpedantic and without RTTI, on every C++ compiler present: the
#      library uses no RTTI, and C++23 is the CI floor's (GCC 13, Clang 18 over libstdc++ 13 through the
#      bcir_expected.hpp shim).
#   2. THE FACADE. The library's objects define the facades' strong symbols and no other -- no C++ name crosses the
#      ABI -- and no static initializer: the library has no global state to construct.
#   3. THE CONTRACTS. Each solver's test_<solver>: every refusal, the budget's stops, determinism, and every
#      allocation of a run refused in turn (NOMEM, nothing left live).
#   4. ASan + UBSan over the contracts and over each fuzz_<solver>'s generated replay (its invariants: the statuses,
#      the certificate's order, every answer checked by the fuzzer's own reading of the problem, determinism,
#      allocation discipline).
#   5. libFuzzer over each facade, bounded (FUZZ_SOLVER_RUNS, default 20000), where a clang++ with compiler-rt is
#      present (CLANGXX names one).
#
# The PARITY with the oracle is not here: bcir/tests/test_cpp_exact.py and test_cpp_csp.py hold each solver's text
# driver to the oracle over its generated corpus, and check every answer with code that shares nothing with it.
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
# Each solver: its name (bcir_<name>.cpp, test_<name>.cpp, fuzz_<name>.cpp) and the facade's one symbol.
solvers=("exact:bcir_exact_schedule" "csp:bcir_csp_solve")

# 1. warning-clean under C++23 and without RTTI, on every C++ compiler present
compilers=("${CXX}")
for other in g++ clang++; do
  if command -v "${other}" >/dev/null 2>&1 && [ "$(command -v "${other}")" != "$(command -v "${CXX}")" ] \
     && "${other}" -std=c++23 "${tmp}/probe.cpp" -o "${tmp}/probe" 2>/dev/null; then
    compilers+=("${other}")
  fi
done
objects=()
facade=""
for entry in "${solvers[@]}"; do
  name="${entry%%:*}"; facade="${facade}${entry#*:} "
  for cxx in "${compilers[@]}"; do
    "${cxx}" "${strict[@]}" -fno-rtti -c "${CPP}/bcir_${name}.cpp" -o "${tmp}/bcir_${name}.o" 2>"${tmp}/build.err" \
      || { echo "FAIL: bcir_${name}.cpp is not warning-clean under C++23 without RTTI (${cxx})"; cat "${tmp}/build.err"; exit 1; }
    echo "  ok: bcir_${name}.cpp is warning-clean under C++23 -Wpedantic -fno-rtti (${cxx})"
  done
  "${CXX}" "${strict[@]}" -fno-rtti -c "${CPP}/bcir_${name}.cpp" -o "${tmp}/bcir_${name}.o"
  objects+=("${tmp}/bcir_${name}.o")
done
library=("${CPP}"/bcir_exact.cpp "${CPP}"/bcir_csp.cpp)

# 2. the facades: their strong symbols alone, no static initializer
if command -v nm >/dev/null 2>&1; then
  want="$(printf '%s\n' ${facade} | sort | tr '\n' ' ')"
  strong="$(nm -g --defined-only "${objects[@]}" | awk '$2 == "T" { print $3 }' | sort | tr '\n' ' ')"
  if [ "${strong}" != "${want}" ]; then
    echo "FAIL: the library's objects define strong symbols '${strong}', not the facades' '${want}' alone"
    exit 1
  fi
  inits="$(nm "${objects[@]}" | awk '/_GLOBAL__sub_I|_GLOBAL__I_/ { n++ } END { print n + 0 }')"
  if [ "${inits}" -ne 0 ]; then
    echo "FAIL: the library has a static initializer (global state to construct)"
    exit 1
  fi
  echo "  ok: the objects define ${want}alone and construct no global state"
else
  echo "  SKIP facade symbols: UNAVAILABLE (no nm on this host)"
fi

# 3. the contracts
for entry in "${solvers[@]}"; do
  name="${entry%%:*}"
  "${CXX}" "${strict[@]}" "${CPP}/test_${name}.cpp" "${objects[@]}" -o "${tmp}/test_${name}" 2>"${tmp}/build.err" \
    || { echo "FAIL: test_${name} did not build"; cat "${tmp}/build.err"; exit 1; }
  contracts="$("${tmp}/test_${name}" 2>&1)" \
    || { echo "FAIL: test_${name}'s contracts"; echo "${contracts}" | tail -n 30; exit 1; }
  [ "${contracts}" = "test_${name}: OK" ] || { echo "FAIL: test_${name} said '${contracts}'"; exit 1; }
  echo "  PASS test_${name}: refusals, budget stops, determinism, every allocation refused in turn"
done

# 4. ASan + UBSan over the contracts and the fuzzers' replays
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
  for entry in "${solvers[@]}"; do
    name="${entry%%:*}"
    "${SANCXX}" "${san[@]}" "${CPP}/test_${name}.cpp" "${library[@]}" -o "${tmp}/test_${name}_san" 2>"${tmp}/build.err" \
      || { echo "FAIL: sanitized test_${name} build"; cat "${tmp}/build.err"; exit 1; }
    "${SANCXX}" "${san[@]}" "${CPP}/fuzz_${name}.cpp" "${library[@]}" -o "${tmp}/fuzz_${name}_san" 2>"${tmp}/build.err" \
      || { echo "FAIL: sanitized fuzz_${name} build"; cat "${tmp}/build.err"; exit 1; }
    "${tmp}/test_${name}_san" > "${tmp}/san.out" 2>&1 \
      || { echo "FAIL: test_${name} under ASan/UBSan"; tail -n 30 "${tmp}/san.out"; exit 1; }
    replay="$("${tmp}/fuzz_${name}_san" 2>"${tmp}/san.err")" \
      || { echo "FAIL: fuzz_${name}'s replay under ASan/UBSan: ${replay}"; tail -n 30 "${tmp}/san.err"; exit 1; }
    echo "  PASS test_${name} and fuzz_${name}'s replay under ASan + UBSan, ${SANCXX} (${replay#fuzz_"${name}": })"
  done
else
  echo "  SKIP ASan/UBSan: UNAVAILABLE (no C++ compiler here has a working sanitizer runtime)"
fi

# 5. libFuzzer over each facade, bounded
CLANGXX="${CLANGXX:-$(command -v clang++ || true)}"
runs="${FUZZ_SOLVER_RUNS:-20000}"
case "${runs}" in ''|*[!0-9]*|0) echo "FAIL: FUZZ_SOLVER_RUNS must be a positive count, not '${runs}'"; exit 1 ;; esac
if [ -n "${CLANGXX}" ] && printf 'extern "C" int LLVMFuzzerTestOneInput(const unsigned char*, unsigned long){return 0;}\n' \
     | "${CLANGXX}" -std=c++23 -fsanitize=fuzzer,address -x c++ - -o "${tmp}/fprobe" 2>/dev/null; then
  for entry in "${solvers[@]}"; do
    name="${entry%%:*}"
    "${CLANGXX}" -std=c++23 -O1 -g -DBCIR_LIBFUZZER -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
      -I "${C}" -I "${CPP}" "${CPP}/fuzz_${name}.cpp" "${library[@]}" -o "${tmp}/fuzz_${name}_lf" 2>"${tmp}/build.err" \
      || { echo "FAIL: fuzz_${name}'s libFuzzer target did not build"; cat "${tmp}/build.err"; exit 1; }
    rm -rf "${tmp:?}/corpus"; mkdir -p "${tmp}/corpus"
    (cd "${tmp}" && "./fuzz_${name}_lf" -runs="${runs}" -max_len=512 -seed=1 corpus > "${tmp}/fuzz.out" 2>&1) \
      || { echo "FAIL: libFuzzer found an input that breaks bcir_${name}'s facade"; tail -n 40 "${tmp}/fuzz.out"; exit 1; }
    done_line="$(awk '/^Done / { line = $0 } END { print line }' "${tmp}/fuzz.out")"
    [ -n "${done_line}" ] || { echo "FAIL: libFuzzer did not report its runs"; tail -n 20 "${tmp}/fuzz.out"; exit 1; }
    echo "  PASS libFuzzer over bcir_${name}'s facade (${done_line})"
  done
else
  echo "  SKIP libFuzzer: UNAVAILABLE (no clang++ with compiler-rt; CLANGXX names one)"
fi
echo "PASS the hosted solver library: built, the facades alone exported, contracts, sanitizers, fuzzing"
