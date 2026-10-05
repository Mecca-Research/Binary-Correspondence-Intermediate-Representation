#!/usr/bin/env bash
# jer: X.697 bounded JER reader: strict-warning and freestanding build (#jer)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-jer` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binaries are the
# manifest variants test_jer_O0 and test_jer_O3: the twin's harness with its whole library closure
# rebuilt at -O0 and at -O3, whose answers must not depend on the optimiser.
#
#   usage: jer.sh <test_jer_O0> <test_jer_O3>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: jer.sh <test_jer_O0> <test_jer_O3>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
O0="$(abs "$1")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable variant"; exit 2; }
O3="$(abs "$2")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable variant"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# X.697 bounded JER reader (#jer, JSON roadmap phase J3): the C twin of
# bcir/asn1/jer_bounded.py -- 4.3's limits, 7.6.2's encoding, and the ECMA-404 grammar as an
# event stream. Unlike the XER twin this one has no `json.loads` behind it, so it is a real
# parser and not only a lexer. The dual-rail differential lives in bcir/tests/test_c_jer.py;
# what is checked HERE is the discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
#
# The -O0 == -O3 comparison earns its place on this file specifically. The reader's hot path
# is signed/unsigned arithmetic on attacker-supplied lengths and a saturating exponent
# accumulator; if any of it were undefined behaviour the optimiser would be entitled to
# choose differently at -O3 than at -O0, and the answers would diverge exactly where a
# malicious document lives.
"${PYTHON}" - <<'JERPY' > "${tmp}/jer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.jer_bounded import STRICT_LIMITS, frame

# Documents that reach every branch of both the bounding pass and the parser, plus the
# forms permissive readers accept and ECMA-404 does not: trailing commas, missing
# separators, leading zeros, the non-JSON constants, and the surrogate cases.
docs = [b"", b" ", b"null", b"true", b"false", b"0", b"-0", b"10", b"01", b"1.5",
        b"-0.5e+3", b"1E-2", b"1.", b".5", b"-", b"1e", b"+1", b'""', b'"a"',
        rb'"\n\t\r\b\f\/\\\""', rb'"A"', '"\U0001f600"'.encode(), rb'"\ud800"',
        rb'"\udc00"', rb'"\uZZZZ"', rb'"\q"', b'"a', b"[]", b"{}", b"[1,2,3]",
        b'{"a":1,"b":2}', b'{"a":{"b":[1,[2,[3]]]}}', b"[1,]", b'{"a":1,}', b"[,]",
        b"[1 2]", b'{"a" 1}', b'{"a":}', b"[", b"]", b"{", b"}", b"1 2", b"[]]",
        b"nan", b"NaN", b"Infinity", b"undefined", b"'a'", b'{"a":1,"a":2}',
        b"\x80", b'"\x80"', b'"\xc0\x80"', b'"\xed\xa0\x80"', b"\xef\xbb\xbf{}",
        b'"\x00"', b'"\x1f"', "[\"é\", \"中\"]".encode(),
        b"[" * 80 + b"]" * 80, b"1" * (STRICT_LIMITS.integer_digits + 1),
        b"1e" + str(STRICT_LIMITS.exponent_magnitude + 1).encode()]
for doc in docs:
    payload = doc.hex() or "-"
    for strict in (0, 1):
        print(f"scan {strict} {payload}")
        print(f"parse {strict} {payload}")
    print(f"utf8doc {payload}")
    for at in range(4):
        print(f"refuse {at} {payload}")

for raw in (b"", b"a", rb"\n", rb"A", "\U0001f600".encode(), rb"\ud800", rb"\q",
            "é中".encode()):
    payload = raw.hex() or "-"
    for cap in (0, 3, 65536):
        print(f"unescape {cap} {payload}")

for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80", b"\xf4\x90\x80\x80"):
    for at in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {at}")

good = frame(b'{"a":1}', sequence=42, generation=7)
for cut in range(0, len(good) + 1):
    print(f"unframe {good[:cut].hex() or '-'}")
bad = bytearray(good)
bad[-1] ^= 1
print(f"unframe {bytes(bad).hex()}")

for field in ("input_bytes", "depth", "nodes", "members", "elements", "string_bytes",
              "number_bytes", "integer_digits", "exponent_magnitude", "work"):
    for value in (1, 1 << 40):
        print(f"tighten {field} {value}")
JERPY
"${O0}" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O0.txt"
"${O3}" < "${tmp}/jer_cases.txt" > "${tmp}/jer_O3.txt"
if cmp -s "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt"; then
  echo "  PASS X.697 JER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/jer_cases.txt") cases)"
else
  echo "  FAIL: the JER twin's answers depend on the optimisation level"
  diff "${tmp}/jer_O0.txt" "${tmp}/jer_O3.txt" | head -10
  exit 1
fi
