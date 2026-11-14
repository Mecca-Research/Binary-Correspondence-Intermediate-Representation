#!/usr/bin/env bash
# tokenizer: the tokenizers (PORT-2): bcir-tokenize's BPE and SentencePiece loads, ids and octets == the oracle's
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds them
# from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the CMake
# project builds them from the same manifest and runs this script as the `c-section-tokenizer` CTest
# entry, and tools/build/section_parity.py holds the two builds' outputs byte-identical. The harnesses
# hold each library's contracts (golden ids, the short-buffer paths, the rules a refusal names, every
# allocation failed in turn) over their own inline models and over ones the oracle's tests build; the
# corpus comparisons are the Python tests' own (bcir/tests/test_c_bpe_tokenizer.py::check_tool and
# bcir/tests/test_c_spm_tokenizer.py::check_tool), run here against the built tool.
#
#   usage: tokenizer.sh <test_bpe> <test_spm> <bcir-tokenize>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 3 ] || { echo "usage: $(basename "$0") <test_bpe> <test_spm> <bcir-tokenize>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
BPE="$(abs "$1")"; [ -x "${BPE}" ] || { echo "  FAIL: HARNESS: ${BPE} is not an executable harness"; exit 2; }
SPM="$(abs "$2")"; [ -x "${SPM}" ] || { echo "  FAIL: HARNESS: ${SPM} is not an executable harness"; exit 2; }
TOOL="$(abs "$3")"; [ -x "${TOOL}" ] || { echo "  FAIL: TOOL: ${TOOL} is not an executable tool"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

"${PYTHON}" - "${tmp}" <<'PY' || { echo "  FAIL: Python tokenizer fixtures"; exit 1; }
import sys
from pathlib import Path
from bcir.tests.test_c_bpe_tokenizer import tokenizers
from bcir.tests.test_c_spm_tokenizer import models
Path(sys.argv[1], "trained.json").write_bytes(dict(tokenizers())["trained"])
Path(sys.argv[1], "trained.model").write_bytes(dict(models())["trained"])
PY
out="$("${BPE}" "${tmp}/trained.json" 2>&1)" || { echo "  FAIL: BPE tokenizer contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected BPE tokenizer harness output"; echo "${out}"; exit 1; }
echo "  PASS BPE tokenizer: golden ids, short buffers, refusal laws, every allocation failed in turn"
out="$("${SPM}" "${tmp}/trained.model" 2>&1)" || { echo "  FAIL: SentencePiece tokenizer contracts"; echo "${out}"; exit 1; }
[ "${out}" = "OK" ] || { echo "  FAIL: unexpected SentencePiece tokenizer harness output"; echo "${out}"; exit 1; }
echo "  PASS SentencePiece tokenizer: golden ids, replacement decoding, short buffers, refusal rules, every allocation failed in turn"

"${PYTHON}" - "${TOOL}" "${tmp}" <<'PY' || exit 1
import os
import sys
from bcir.tests import test_c_bpe_tokenizer as bpe
from bcir.tests import test_c_spm_tokenizer as spm

for name, module, work in (("BPE", bpe, "bpe"), ("SentencePiece", spm, "spm")):
    os.makedirs(os.path.join(sys.argv[2], work), exist_ok=True)
    failures = module.check_tool(sys.argv[1], os.path.join(sys.argv[2], work), mutations=96)
    if failures:
        print(f"  FAIL: bcir-tokenize ({name}) differs from the oracle:")
        for line in failures[:20]:
            print("    " + line)
        sys.exit(1)
print(f"  PASS bcir-tokenize == the oracle: {len(bpe.tokenizers())} BPE tokenizers and "
      f"{len(bpe.law_documents())} law documents; {len(spm.models())} SentencePiece models and "
      f"{len(spm.law_documents())} law documents; encode, decode and byte mutations")
PY
