#!/usr/bin/env bash
# asn1fast: DER -> native StreamPack fast path: byte-identical reconstruction (#asn1fast)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-asn1fast` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binary is the
# test_asn1_streampack harness.
#
#   usage: asn1fast.sh <test_asn1_streampack>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 1 ] || { echo "usage: asn1fast.sh <test_asn1_streampack>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
HARNESS="$(abs "$1")"; [ -x "${HARNESS}" ] || { echo "  FAIL: HARNESS: ${HARNESS} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# DER -> native StreamPack fast path (#asn1fast, roadmap phase D): reconstruct the native
# artifact from its X.690 DER projection in freestanding C, with no Python anywhere in the
# reconstruction path, and assert BYTE IDENTITY against what the Python encoder produced.
# That is law A3 (additive: the native octets survive the round trip) proven on the C rail.
# Byte identity, not equivalence -- the fast path has to re-derive the StreamPack VERSION
# from content the way bcir/abi::encode does, emit the reserved stride_k the projection
# deliberately omits, and recompute the CRC.
"${PYTHON}" - "${tmp}" <<'ASN1FASTPY' || { echo "  FAIL: could not project the corpus"; exit 1; }
import os, sys
from bcir.abi import encode
from bcir.asn1.streampack import encode_pack
from bcir.examples import PROGRAMS
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
d = sys.argv[1]
host, theta = TargetProfile.x86_avx512(), Theta.cool()
for name, build in sorted(PROGRAMS.items()):
    module = build()
    pack = hydrate(module, optimize(module, host, theta))
    open(os.path.join(d, name + ".proj.der"), "wb").write(encode_pack(pack))
    open(os.path.join(d, name + ".native.bin"), "wb").write(encode(pack))
ASN1FASTPY
fast_ok=0; fast_bad=0
for proj in "${tmp}"/*.proj.der; do
  base="$(basename "${proj}" .proj.der)"
  if out="$("${HARNESS}" "${proj}" "${tmp}/${base}.native.bin" 2>&1)" \
       && [ "${out%% *}" = "OK" ]; then
    fast_ok=$((fast_ok + 1))
  else
    echo "  FAIL ${base}: ${out}"; fast_bad=$((fast_bad + 1))
  fi
done
# A malformed or BER-only projection must be refused, never partially reconstructed.
"${PYTHON}" - "${tmp}" <<'ASN1NEGPY'
import os, sys
d = sys.argv[1]
der = open(os.path.join(d, "vector_add.proj.der"), "rb").read()
for n in (0, 1, 5, len(der) // 2, len(der) - 1):
    open(os.path.join(d, f"neg{n}.bad.der"), "wb").write(der[:n])
if der[1] < 0x80:      # the same value with a non-minimal length: legal BER, not DER
    open(os.path.join(d, "nonmin.bad.der"), "wb").write(
        bytes([der[0], 0x81, der[1]]) + der[2:])
ASN1NEGPY
for bad in "${tmp}"/*.bad.der; do
  if "${HARNESS}" "${bad}" >/dev/null 2>&1; then
    echo "  FAIL: $(basename "${bad}") was accepted by the fast path"; fast_bad=$((fast_bad + 1))
  fi
done
if [ "${fast_bad}" -eq 0 ] && [ "${fast_ok}" -ge 10 ]; then
  echo "  PASS DER -> native byte-identical on ${fast_ok} corpus programs; malformed + BER-only refused"
else
  echo "  FAIL: fast path (${fast_ok} ok, ${fast_bad} bad)"; exit 1
fi
