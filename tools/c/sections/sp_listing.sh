#!/usr/bin/env bash
# sp_listing: MC1 StreamPack listing (PORT-1): the record walk's laws + bcir-pack's octets == the oracle's
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds them from the same manifest and runs this script as the `c-section-sp_listing` CTest
# entry, and tools/build/section_parity.py holds the two builds' outputs byte-identical. The harness
# holds the walk's laws and the format calls' contracts; the corpus comparison is the Python test's
# own (bcir/tests/test_c_streampack_listing.py::check_tool), run here against the built tool.
#
#   usage: sp_listing.sh <test_sp_listing> <bcir-pack>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_sp_listing> <bcir-pack>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
TOOL="$(abs "$2")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

out="$("${HARNESS}" 2>&1)" || { echo "  FAIL: record walk / listing contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected listing harness output"; echo "${out}"; exit 1; }
echo "  PASS record walk: decoder laws (R10 left to the verifier), spans, sink, scratch and width contracts"

"${PYTHON}" - "${TOOL}" "${tmp}" <<'PY' || exit 1
import sys
from bcir.tests.test_c_streampack_listing import check_tool, corpus

failures = check_tool(sys.argv[1], sys.argv[2], mutations=24)
if failures:
    print("  FAIL: bcir-pack differs from bcir.abi.streampack_tool:")
    for line in failures[:20]:
        print("    " + line)
    sys.exit(1)
print(f"  PASS bcir-pack dis/hexdump == the oracle's octets over {len(corpus())} packs, "
      "and the same refusals over their resealed mutants")
PY
