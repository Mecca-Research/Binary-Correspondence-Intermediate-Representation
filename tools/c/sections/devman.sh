#!/usr/bin/env bash
# devman: device manifests bound at instantiation (bcir_devman.c, N11): test_devman's contracts, and its loads, probes
# and bound admissions == the oracle's (bcir/kbcir/device_manifest.py's load_device_manifest and probe_agree)
#
# One section of tools/c/check_runtime.sh, run against a binary built elsewhere: BCIR Make builds it from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds it
# from the same manifest and runs this script as the `c-section-devman` CTest entry, and tools/build/section_parity.py
# holds the two builds' outputs byte-identical. The comparison with the oracle is the Python test's own
# (bcir/tests/test_c_devman.py), run here against the built harness.
#
#   usage: devman.sh <test_devman>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_devman>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: device manifest contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_devman: OK" ] || { echo "  FAIL: unexpected device manifest harness output"; echo "${out}"; exit 1; }
echo "  PASS device manifests: a short workspace or bank table, a tile past 32 bits, a known envelope's digest, every NULL"
out="$("${PYTHON}" - "${HARNESS}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_devman import check_driver
seen = check_driver(sys.argv[1])
refused = sum(v for k, v in seen.items() if k.startswith("err"))
print(f"{seen['loaded'] + refused} envelopes ({refused} refused), {seen['probes']} probes ({seen['vetoed']} vetoed), "
      f"{seen['views']} views over the bound banks ({seen['admitted']} admitted)")
PY
)" || { echo "  FAIL: test_devman != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS test_devman == the oracle: ${out}"
