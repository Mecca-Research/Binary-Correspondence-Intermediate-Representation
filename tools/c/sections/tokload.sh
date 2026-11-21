#!/usr/bin/env bash
# tokload: the native data loader (bcir_tokload.c): test_tokload's contracts, and bcir-tokload's shards, verdicts
# and batches == the oracle's (bcir/frontends/models/token_shards.py)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them from
# runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake project builds them
# from the same manifest and runs this script as the `c-section-tokload` CTest entry, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The comparison with the oracle is the
# Python test's own (bcir/tests/test_c_tokload.py), run here against the built tool.
#
#   usage: tokload.sh <test_tokload> <bcir-tokload>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_tokload> <bcir-tokload>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
TOOL="$(abs "$2")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: data loader contracts"; echo "${out}"; exit 1; }
[ "${out}" = "test_tokload: OK" ] || { echo "  FAIL: unexpected data loader harness output"; echo "${out}"; exit 1; }
echo "  PASS data loader: every shard refusal leaves the shard as it was, an epoch visits every window once, an exact resume"
out="$("${PYTHON}" - "${TOOL}" <<'PY' 2>&1
import sys
from bcir.tests.test_c_tokload import check_tool
seen = check_tool(sys.argv[1])
print(f"{seen['packed']} shards packed byte for byte, {seen['info']} verdicts ({seen['refused']} refusals), "
      f"{seen['batches']} batches, {seen['loader_refusals']} loader refusals")
PY
)" || { echo "  FAIL: bcir-tokload != the oracle"; echo "${out}" | tail -20; exit 1; }
echo "  PASS bcir-tokload == the oracle: ${out}"
