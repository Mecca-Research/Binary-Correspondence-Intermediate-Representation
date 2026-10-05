#!/usr/bin/env bash
# runtime: build harness (C23) + Python->C ABI parity
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-runtime` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: runtime.sh <test_runtime>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_runtime>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

"${PYTHON}" -c "
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate
from bcir.abi import encode
m=vector_add(1024); pack=hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
open('${tmp}/pack.bin','wb').write(encode(pack))
" || { echo "  FAIL: python encode"; exit 1; }
out="$("${HARNESS}" "${tmp}/pack.bin")" || { echo "  FAIL: C decode"; echo "${out}"; exit 1; }
echo "${out}" | grep -q "^OK$" && echo "  PASS parity (Python encode -> C decode)" \
  || { echo "  FAIL: parity"; echo "${out}"; exit 1; }
