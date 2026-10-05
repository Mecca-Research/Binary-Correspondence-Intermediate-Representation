#!/usr/bin/env bash
# cfront_abi: target-ABI matrix (bcir_cfront --target): sizeof data model == oracle (#abi)
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-cfront_abi` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical. It takes the `test_cfront` the `cfront` section takes.
#
#   usage: cfront_abi.sh <test_cfront>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_cfront>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Target-ABI matrix (#abi): the C twin's `--target` data model lays `long` / pointer / size_t-class
# types out exactly like the oracle's TargetABI, for every named target. The summary counts are
# target-invariant, so this compares the FOLDED sizeof constants (which carry the data-model widths):
# the C twin emits them as `= Nu;` literals; the oracle exposes them as the c.const immediates. The
# vectors must agree per target AND differ across LP64 / LLP64 / ILP32 (so the gate has teeth).
ABI_TARGETS="x86_64-linux aarch64-linux riscv64-linux x86_64-windows i386-linux"
# One python process for every target (was one per target): import compile_unit once.
"${PYTHON}" - "${C}" ${ABI_TARGETS} > "${tmp}/abi_sums.txt" <<'PY' || { echo "  FAIL: python ABI (batch)"; exit 1; }
import sys
from bcir.frontends.cfront import compile_unit
src = open(sys.argv[1] + "/cfront_abi.c").read()
for t in sys.argv[2:]:
    r = compile_unit(src, check_clang=False, target=t)
    lf = r.lowered.functions[next(reversed(r.lowered.functions))]
    print(f"{t}\t" + ','.join(str(c.imm[0]) for c in lf.claims if c.op == 'c.const'))
PY
abi_seen=""
for t in ${ABI_TARGETS}; do
  c_vals="$("${HARNESS}" --target "${t}" "${C}/cfront_abi.c" | sed -n '/----EMIT----/,$p' \
            | grep -oE '= [0-9]+u;' | grep -oE '[0-9]+' | paste -sd, -)" \
    || { echo "  FAIL: C ABI run ${t}"; exit 1; }
  py_vals="$(awk -F'\t' -v f="${t}" '$1==f{print $2; exit}' "${tmp}/abi_sums.txt")"
  [ "${c_vals}" = "${py_vals}" ] \
    && echo "  PASS ABI ${t} (sizeof model oracle == C: [${c_vals}])" \
    || { echo "  FAIL: ABI ${t} (C='[${c_vals}]' PY='[${py_vals}]')"; exit 1; }
  abi_seen="${abi_seen}${c_vals};"
done
# the three data models must produce distinct vectors (LP64 8/8/8, LLP64 long=4, ILP32 ptr=4 too).
echo "${abi_seen}" | grep -q "8,8,8,4,8;" && echo "${abi_seen}" | grep -q "4,8,8,4,8;" \
  && echo "${abi_seen}" | grep -q "4,4,4,4,8;" \
  && echo "  PASS ABI matrix spans LP64 / LLP64 / ILP32 (distinct data models)" \
  || { echo "  FAIL: ABI matrix did not span the three data models: ${abi_seen}"; exit 1; }
