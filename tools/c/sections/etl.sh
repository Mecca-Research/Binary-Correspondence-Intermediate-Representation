#!/usr/bin/env bash
# etl: the event transducer (PORT-4): test_etl's contracts, and bcir-etl's verdicts == the oracle's (bcir/etl)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds them from the same manifest and runs this script as the `c-section-etl` CTest entry,
# and tools/build/section_parity.py holds the two builds' outputs byte-identical. The harness holds the
# library's own contracts (a golden parse and run, every output buffer one short, every NULL, each
# refusal's element, field and offset); the corpus comparison is the Python test's own
# (bcir/tests/test_c_etl.py::check_tool), run here against the built tool.
#
#   usage: etl.sh <test_etl> <bcir-etl>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_etl> <bcir-etl>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
TOOL="$(abs "$2")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: event transducer contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected event transducer harness output"; echo "${out}"; exit 1; }
echo "  PASS event transducer: a golden parse and run, every buffer one short, every NULL, each refusal's element and offset"
out="$("${PYTHON}" - "${TOOL}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_etl import check_tool
counts = check_tool(sys.argv[1])
print(f"{counts['programs']} mini-ROP programs, {counts['machines']} machines and their runs, {counts['events']} event descriptors")
PY
)" || { echo "  FAIL: bcir-etl != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS bcir-etl == the oracle: ${out}"
