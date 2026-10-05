#!/usr/bin/env bash
# encoder: StreamPack encoder: freestanding compile (C11 + C23) + byte-identical re-encode
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: the gate
# compiles them its own way and calls this script; the CMake project builds them from
# runtime/manifest.json and runs this script as the `c-section-encoder` CTest entry. The body is the
# gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text,
# and tools/build/section_parity.py holds the two builds' outputs byte-identical.
#
#   usage: encoder.sh <test_encode>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: $(basename "$0") <test_encode>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The executor section's fixture (exec.bin), regenerated here so this section stands alone.
"${PYTHON}" -c "
from bcir.examples import multi_histogram
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.gem import hydrate, execute
from bcir.abi import encode
m=multi_histogram(); r=optimize(m, TargetProfile.x86_avx512(), Theta.cool())
open('${tmp}/exec.bin','wb').write(encode(hydrate(m, r)))
open('${tmp}/exec.order','w').write(' '.join(str(i) for i in execute(m).order))
" || { echo "  FAIL: python encode"; exit 1; }
# bcir_encode.c is the C write-side twin of bcir/abi/streampack_abi.py -- the full C
# round-trip (a driver emits the artifact with no Python).
"${HARNESS}" "${tmp}/exec.bin" "${tmp}/reenc.bin" >/dev/null || { echo "  FAIL: C re-encode"; exit 1; }
if cmp -s "${tmp}/exec.bin" "${tmp}/reenc.bin"; then
  echo "  PASS encoder parity (C bcir_sp_reencode == Python encode, byte-identical)"
else
  echo "  FAIL: encoder bytes differ from the Python encoding"; exit 1
fi
