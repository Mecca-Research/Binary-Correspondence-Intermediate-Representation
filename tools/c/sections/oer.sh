#!/usr/bin/env bash
# oer: X.696 OER decoder: strict-warning and freestanding build (#oer)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-oer` CTest
# entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so
# the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binaries are the manifest variants test_oer_O0 and test_oer_O3: the twin's
# harness with its whole library closure rebuilt at -O0 and at -O3, whose answers must not depend on
# the optimiser.
#
#   usage: oer.sh <test_oer_O0> <test_oer_O3>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: oer.sh <test_oer_O0> <test_oer_O3>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
O0="$(abs "$1")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable variant"; exit 2; }
O3="$(abs "$2")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable variant"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT


"${PYTHON}" - <<'OERPY' > "${tmp}/oer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.oer import OerRules, encode_length, encode_oer
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Universal

for value in (0, 1, 126, 127, 128, 255, 256, 65535, 65536, 1 << 24):
    raw = encode_length(value)
    for pos in range(len(raw) + 2):
        print(f"length {raw.hex()} {pos}")
# The malformed and BASIC-OER spellings a decoder must tell apart.
for raw in ("80", "8100", "820080", "82ff", "ff"):
    print(f"length {raw} 0")

byte = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
word = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(-32768, 32767))
wide = Primitive(Universal.INTEGER, "INTEGER")
text = Primitive(Universal.UTF8_STRING, "UTF8String")
for kind, width, signed, values in ((byte, 1, 0, (0, 1, 255)),
                                    (word, 2, 1, (-32768, -1, 0, 32767)),
                                    (wide, 0, 1, (0, -1, 128, -129, 2 ** 40))):
    for value in values:
        raw = encode_oer(kind, value, rules=OerRules.CANONICAL)
        print(f"integer {raw.hex()} 0 {width} {signed}")
for raw in ("00", "ff", "0000", "ffff", "ffffffffffffffff"):
    for width in (1, 2, 4, 8, 0, 3):
        for signed in (0, 1):
            print(f"integer {raw} 0 {width} {signed}")

for raw in ("00", "80", "40", "c0", "ff", "81"):
    for count in (0, 1, 2, 3, 8):
        print(f"preamble {raw} 0 {count}")

record = Sequence((Component("id", byte), Component("delta", word),
                   Component("label", text), Component("note", text, optional=True)),
                  name="Record")
plan = "0:1:0:0:0,0:2:1:0:0,4:0:0:0:0,4:0:0:1:0"
for value in ({"id": 7, "delta": -3, "label": "abc", "note": "n"},
              {"id": 0, "delta": 0, "label": ""},
              {"id": 255, "delta": 32767, "label": "x" * 200}):
    raw = encode_oer(record, value, rules=OerRules.CANONICAL)
    for cut in range(len(raw) + 1):
        print(f"sequence {raw[:cut].hex() or '-'} {plan}")
# Plans the checker must refuse before reading an octet.
print(f"sequence 00 9:0:0:0:0")
print(f"sequence 00 0:3:0:0:0")
OERPY
"${O0}" < "${tmp}/oer_cases.txt" > "${tmp}/oer_O0.txt"
"${O3}" < "${tmp}/oer_cases.txt" > "${tmp}/oer_O3.txt"
if cmp -s "${tmp}/oer_O0.txt" "${tmp}/oer_O3.txt"; then
  echo "  PASS X.696 OER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/oer_cases.txt") cases)"
else
  echo "  FAIL: the OER twin's answers depend on the optimisation level"
  diff "${tmp}/oer_O0.txt" "${tmp}/oer_O3.txt" | head -10
  exit 1
fi
