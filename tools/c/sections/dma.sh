#!/usr/bin/env bash
# dma: DMA programs from strided views (bcir_dma.c, N12's run-time half): test_dma's contracts, and its admissions
# and descriptor rings == the oracle's (bcir/kbcir/device_manifest.py's check_strided_view, bcir/kbcir/dma.py)
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-dma` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The comparison with the oracle is the Python test's own
# (bcir/tests/test_c_dma.py), run here against the built harness.
#
#   usage: dma.sh <test_dma>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_dma>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: DMA program contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_dma: OK" ] || { echo "  FAIL: unexpected DMA harness output"; echo "${out}"; exit 1; }
echo "  PASS DMA programs: a ring past its capacity, arithmetic past 64 bits, every NULL, dense and scattered copies"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_dma import check_driver
seen = check_driver(sys.argv[1])
print(f"{seen['admitted'] + seen['refused']} admissions ({seen['refused']} refused), {seen['rings']} rings of "
      f"{seen['descriptors']} descriptors, {sum(v for k, v in seen.items() if k.startswith('err'))} pair refusals")
PY
)" || { echo "  FAIL: test_dma != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_dma == the oracle: ${out}"
