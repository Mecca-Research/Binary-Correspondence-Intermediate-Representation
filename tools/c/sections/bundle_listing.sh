#!/usr/bin/env bash
# bundle_listing: MC1 BCAB listing (PORT-1): bcir-bundle's list/hexdump/select/extract/dis/mnemonics == the oracle's
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds them from the same manifest and runs this script as the `c-section-bundle_listing`
# CTest entry, and tools/build/section_parity.py holds the two builds' outputs byte-identical. The
# harness holds the listing's contracts over a bundle the oracle writes; the corpus comparison is the
# Python test's own (bcir/tests/test_c_bundle_listing.py::check_tool), run here against the built tool.
#
#   usage: bundle_listing.sh <test_ab_listing> <bcir-bundle>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_ab_listing> <bcir-bundle>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
TOOL="$(abs "$2")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

"${PYTHON}" - "${tmp}/many.bcab" <<'PY' || { echo "  FAIL: Python BCAB fixture"; exit 1; }
import sys
from bcir.abi import encode_bundle
from bcir.tests.test_c_bundle_listing import _many
open(sys.argv[1], "wb").write(encode_bundle(_many()))
PY
out="$("${HARNESS}" "${tmp}/many.bcab" 2>&1)" || { echo "  FAIL: BCAB listing contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected BCAB listing harness output"; echo "${out}"; exit 1; }
echo "  PASS BCAB listing: a corrupted bundle prints nothing; sink, width and text-view contracts"

"${PYTHON}" - "${TOOL}" "${tmp}" <<'PY' || exit 1
import sys
from bcir.tests.test_c_bundle_listing import _SELECTORS, check_tool, corpus

failures = check_tool(sys.argv[1], sys.argv[2], mutations=24)
if failures:
    print("  FAIL: bcir-bundle differs from bcir.abi.artifact_tool:")
    for line in failures[:20]:
        print("    " + line)
    sys.exit(1)
print(f"  PASS bcir-bundle == the oracle's octets over {len(corpus())} bundles: list, hexdump, "
      f"{len(_SELECTORS)} selections each, every extract, dis and mnemonics, resealed mutants")
PY
