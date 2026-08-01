#!/usr/bin/env bash
# check_jer_index.sh -- J5's gate for the hosted structural index (#jerindex).
#
# The index rebuilds `bcir_jer_scan`'s DISPATCH on the exported `bcir_jer_scan_cursor` and
# reuses its token scanners verbatim, so it is a second dispatch loop rather than a second
# scanner. See runtime/cpp/bcir_jer_index.h and roadmap 7.4.1.
#
# What is checked HERE, and what is deliberately checked elsewhere:
#
#   1. THE BUILD. Warning-clean under C++17, and -- the clause that matters most -- the
#      freestanding C core still builds freestanding with the cursor exported. Exporting a
#      seam is where a hosted dependency most easily leaks into a core that must not have one,
#      and no test that RUNS the index would notice, because the hosted build has libc.
#   2. THE TIER IS REAL. The tier this host cannot execute is cross-compiled and disassembled,
#      so a vector pass that silently degraded to scalar fails where nothing executes it.
#      Same reasoning as #jersimd: a tier that never runs cannot be shown correct by running.
#
# The EQUIVALENCE differential is not here. It lives in bcir/tests/test_cpp_jer_index.py,
# where it can sweep 4.3's work ceiling across every failure position in every document --
# the one place the bulk whitespace charge and the scalar rail could disagree. A shell gate
# would have to reimplement that sweep to say anything useful.
#
# Self-skips (exit 0) when no C++ compiler is present, like the other C++ gates.
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
CC="${CC:-}"
if [ -z "${CC}" ]; then
  for candidate in clang gcc cc; do
    if command -v "${candidate}" >/dev/null 2>&1; then CC="${candidate}"; break; fi
  done
fi
if [ -z "${CXX}" ] || [ -z "${CC}" ]; then
  echo "SKIP: no C++/C compiler pair; the hosted index is optional by design"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

"${CXX}" -std=c++17 -O2 -Wall -Wextra -Werror -I "${C}" -I "${CPP}" \
  -c "${CPP}/bcir_jer_index.cpp" -o "${tmp}/bcir_jer_index.o" \
  || { echo "FAIL: bcir_jer_index.cpp is not warning-clean under C++17"; exit 1; }
echo "  ok: bcir_jer_index.cpp is warning-clean under C++17"

# The load-bearing one. The cursor exists so the hosted index can reach the scan's dispatch
# state; if exporting it ever drags a hosted header into the core, the core stops being
# freestanding and every embedded target loses it -- silently, because the hosted build works.
for std in c11 c23; do
  "${CC}" -ffreestanding -nostdlib -std=${std} -Wall -Wextra -Werror -I "${C}" \
    -c "${C}/bcir_jer.c" -o /dev/null \
    || { echo "FAIL: exporting the scan cursor cost bcir_jer.c its freestanding build (${std})"; exit 1; }
done
echo "  ok: the C core stays freestanding-clean with the cursor exported (C11 and C23)"

# The tier this host cannot run. Compiling it proves the code exists; disassembling it proves
# the compiler emitted the wide instructions rather than quietly lowering to scalar, which is
# the failure no run on this machine could ever surface.
cross_target=""
cross_needle=""
case "$(uname -m)" in
  aarch64|arm64) cross_target="x86_64-linux-gnu"; cross_needle="pmovmskb|pcmpeqb" ;;
  *)             cross_target="aarch64-linux-gnu"; cross_needle="uminv|cmeq" ;;
esac

if "${CXX}" --target="${cross_target}" -std=c++17 -O2 -Wall -Wextra -Werror -ffreestanding \
     -I "${C}" -I "${CPP}" -c "${CPP}/bcir_jer_index.cpp" -o "${tmp}/cross.o" \
     2>"${tmp}/cross.log"
then
  disasm=""
  for tool in llvm-objdump objdump; do
    if command -v "${tool}" >/dev/null 2>&1; then
      disasm="$("${tool}" -d "${tmp}/cross.o" 2>/dev/null || true)"
      [ -n "${disasm}" ] && break
    fi
  done
  if [ -z "${disasm}" ]; then
    echo "  note: ${cross_target} tier compiles, but no disassembler to prove it is not scalar"
  elif printf '%s' "${disasm}" | grep -qE "${cross_needle}"; then
    echo "  ok: the ${cross_target} whitespace pass compiles here AND emits its own SIMD"
  else
    echo "FAIL: ${cross_target} compiled without any ${cross_needle} instruction -- the tier"
    echo "      fell back to scalar rather than building, which no run on this host would show"
    exit 1
  fi
else
  # A clang without the other backend, or a gcc that cannot retarget. Skipped rather than
  # failed: the cross build is extra reach, and CI runs the tier natively either way.
  echo "  note: ${CXX} cannot target ${cross_target}; the other tier is unchecked here"
fi

echo "  note: equivalence across tiers is swept in bcir/tests/test_cpp_jer_index.py"
