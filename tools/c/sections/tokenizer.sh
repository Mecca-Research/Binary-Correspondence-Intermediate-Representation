#!/usr/bin/env bash
# tokenizer: the byte-level BPE tokenizer (PORT-2): bcir-tokenize's loads, ids and octets == the oracle's
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds them from the same manifest and runs this script as the `c-section-tokenizer` CTest
# entry, and tools/build/section_parity.py holds the two builds' outputs byte-identical. The harness
# holds the library's contracts (golden ids, the short-buffer paths, the laws a refusal names, every
# allocation failed in turn) over its own mini tokenizer and over one the oracle trains; the corpus
# comparison is the Python test's own (bcir/tests/test_c_bpe_tokenizer.py::check_tool), run here
# against the built tool.
#
#   usage: tokenizer.sh <test_bpe> <bcir-tokenize>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: $(basename "$0") <test_bpe> <bcir-tokenize>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
TOOL="$(abs "$2")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

"${PYTHON}" - "${tmp}/trained.json" <<'PY' || { echo "  FAIL: Python tokenizer fixture"; exit 1; }
import sys
from bcir.tests.test_c_bpe_tokenizer import tokenizers
open(sys.argv[1], "wb").write(dict(tokenizers())["trained"])
PY
out="$("${HARNESS}" "${tmp}/trained.json" 2>&1)" || { echo "  FAIL: BPE tokenizer contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected BPE tokenizer harness output"; echo "${out}"; exit 1; }
echo "  PASS BPE tokenizer: golden ids, short buffers, refusal laws, every allocation failed in turn"

"${PYTHON}" - "${TOOL}" "${tmp}" <<'PY' || exit 1
import sys
from bcir.tests.test_c_bpe_tokenizer import check_tool, law_documents, tokenizers

failures = check_tool(sys.argv[1], sys.argv[2], mutations=96)
if failures:
    print("  FAIL: bcir-tokenize differs from bcir.frontends.models.tokenizer:")
    for line in failures[:20]:
        print("    " + line)
    sys.exit(1)
print(f"  PASS bcir-tokenize == the oracle over {len(tokenizers())} tokenizers: limits, info, encode, "
      f"decode, {len(law_documents())} law documents and byte mutations")
PY
