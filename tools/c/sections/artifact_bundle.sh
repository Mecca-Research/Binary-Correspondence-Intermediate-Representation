#!/usr/bin/env bash
# artifact_bundle: BCAB artifact bundle: freestanding reader + Python/C selection parity
#
# One section of tools/c/check_runtime.sh, run against harness binaries built elsewhere: BCIR Make
# builds them from runtime/manifest.json and runs this script as a task, whose verdict the gate
# shows; the CMake project builds them from the same manifest and runs this script as the
# `c-section-artifact_bundle` CTest entry. The body is the gate's section text moved here (BUILD-2,
# docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and tools/build/section_parity.py holds the
# two builds' outputs byte-identical.
#
#   usage: artifact_bundle.sh <test_sha256> <test_artifact_bundle> <test_asn1>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 3 ] || { echo "usage: $(basename "$0") <test_sha256> <test_artifact_bundle> <test_asn1>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
SHA256="$(abs "$1")"; [ -x "${SHA256}" ] || { echo "  FAIL: SHA256: ${SHA256} is not an executable harness"; exit 2; }
BUNDLE="$(abs "$2")"; [ -x "${BUNDLE}" ] || { echo "  FAIL: BUNDLE: ${BUNDLE} is not an executable harness"; exit 2; }
ASN1="$(abs "$3")"; [ -x "${ASN1}" ] || { echo "  FAIL: ASN1: ${ASN1} is not an executable harness"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# The shared SHA-256 / HMAC-SHA256 (bcir_sha256.c) holds the standards' own worked examples:
# FIPS 180-4 and RFC 4231 (cases 6-7 take the hash-the-key-first path), one-shot and incremental.
sha_out="$("${SHA256}")" || { echo "  FAIL: SHA-256/HMAC vectors"; echo "${sha_out}"; exit 1; }
[ "${sha_out}" = "OK" ] && echo "  PASS shared SHA-256 / HMAC-SHA256 == FIPS 180-4 + RFC 4231 vectors" \
  || { echo "  FAIL: unexpected SHA-256/HMAC harness output"; echo "${sha_out}"; exit 1; }
"${PYTHON}" - "${tmp}/bundle.bcab" "${tmp}/bundle.der" "${tmp}/bundle.from-der.bcab" <<'PY' \
  || { echo "  FAIL: Python BCAB/ASN.1 fixture"; exit 1; }
import struct, sys
from bcir.abi import (ArtifactBundle, ArtifactFormat, ArtifactKind, ArtifactVariant,
                      Endianness, encode, encode_bundle)
from bcir.asn1.artifact_bundle import der_to_native, native_to_der
from bcir.examples import vector_add
from bcir.gem import hydrate
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
m = vector_add(8)
pack = hydrate(m, optimize(m, TargetProfile.x86_avx512(), Theta.cool()))
elf = bytearray(20); elf[:7] = b"\x7fELF\x02\x01\x01"; struct.pack_into("<H", elf, 16, 1); struct.pack_into("<H", elf, 18, 62)
variants = (
    ArtifactVariant("00-root", ArtifactKind.STREAM_PACK, ArtifactFormat.STREAM_PACK,
                    encode(pack), channel="host", portable=True),
    ArtifactVariant("portable-c", ArtifactKind.C_SOURCE, ArtifactFormat.TEXT,
                    b"int bcir_kernel(void){return 0;}\n", portable=True),
    ArtifactVariant("x86-avx2", ArtifactKind.ELF_OBJECT, ArtifactFormat.ELF, bytes(elf),
                    triple="x86_64-unknown-linux-gnu", architecture="x86_64",
                    os_abi="linux-gnu", channel="host", entry_symbol="bcir_kernel",
                    required_features=("avx2",), endianness=Endianness.LITTLE,
                    pointer_bits=64, e_machine=62, priority=9,
                    r12_attested=True, executable=True),
)
native = encode_bundle(ArtifactBundle(variants, "00-root", "portable-c", 123, 7))
open(sys.argv[1], "wb").write(native)
projection = native_to_der(native)
open(sys.argv[2], "wb").write(projection)
open(sys.argv[3], "wb").write(der_to_native(projection))
PY
about="$("${BUNDLE}" "${tmp}/bundle.bcab")" \
  || { echo "  FAIL: BCAB C parity harness"; exit 1; }
case "${about}" in
  OK\ entries=3*) echo "  PASS BCAB Python encode -> C checksum/select parity" ;;
  *) echo "  FAIL: unexpected BCAB result '${about}'"; exit 1 ;;
esac
cmp -s "${tmp}/bundle.bcab" "${tmp}/bundle.from-der.bcab" \
  || { echo "  FAIL: BCAB native -> ASN.1 DER -> native bytes differ"; exit 1; }
asn1_about="$("${ASN1}" "${tmp}/bundle.der")" \
  || { echo "  FAIL: BCAB ASN.1 projection C validation"; exit 1; }
printf '%s\n' "${asn1_about}" | grep -q '^der ok$' \
  && echo "  PASS BCAB native <-> DER byte identity + generic C X.690 validation" \
  || { echo "  FAIL: C X.690 rail rejected BCAB DER projection"; printf '%s\n' "${asn1_about}"; exit 1; }
