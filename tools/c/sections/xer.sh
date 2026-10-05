#!/usr/bin/env bash
# xer: X.693 XER lexical layer: strict-warning and freestanding build (#xer)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: BCIR Make builds
# them from runtime/manifest.json and runs this script as a task, whose verdict the gate shows; the
# CMake project builds them from the same manifest and runs this script as the `c-section-xer` CTest
# entry. The body is the gate's section text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so
# the two run one text, and tools/build/section_parity.py holds the two builds' outputs
# byte-identical. The binaries are the manifest variants test_xer_O0 and test_xer_O3: the twin's
# harness with its whole library closure rebuilt at -O0 and at -O3, whose answers must not depend on
# the optimiser.
#
#   usage: xer.sh <test_xer_O0> <test_xer_O3>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: xer.sh <test_xer_O0> <test_xer_O3>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
O0="$(abs "$1")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable variant"; exit 2; }
O3="$(abs "$2")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable variant"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# X.693 XER lexical layer (#xer, roadmap phase E-adjacent): the C twin of the tag scanner
# and the xmlcstring escaper. XER is text, so there is no bit cursor to get wrong -- but
# there is a byte cursor, and it is driven entirely by attacker-supplied content before any
# type is consulted. The dual-rail differential lives in bcir/tests/test_c_xer.py; what is
# checked HERE is the same discipline every other twin gets: strict warnings as errors, a
# genuinely freestanding translation unit, and answers that do not depend on the optimiser.
"${PYTHON}" - <<'XERPY' > "${tmp}/xer_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.xer import _CONTROL_ELEMENT

# Documents that reach every branch of the scanner, plus the truncations one octet short of
# each excluded construct -- the inputs where a bounds check that is off by one shows up.
docs = ["<a>", "</a>", "<a/>", "<PersonnelRecord>", "</ChildInformation>", "<_XMLThing/>",
        "<a >", "<a\t/>", "<nul/>", "<BIT_STRING>", "<x-y.z/>", "<!-- c -->",
        "<![CDATA[x]]>", "<!DOCTYPE a>", "<?xml?>", '<a b="1">', "<a:b>", "<", "</",
        "<a", "<a/", "<!", "<!-", "<![CDATA", "<?", "<1a>", "<>", "< a>", "a",
        "<a><b/></a>", "  <a>x</a>", "<a\xc3\xa9>"]
for doc in docs:
    raw = doc.encode("utf-8", "surrogatepass")
    for pos in range(len(raw) + 2):
        print(f"tag {raw.hex() or '-'} {pos}")
        print(f"space {raw.hex() or '-'} {pos}")

strings = ["", "a", "a<b>&c", "\t\n\r", "John P Smith", "&&&", "é中\U0001f600",
           "".join(chr(code) for code in sorted(_CONTROL_ELEMENT))]
for text in strings:
    raw = text.encode()
    print(f"escape {raw.hex() or '-'}")
    print(f"unescape 1 {raw.hex() or '-'}")
    print(f"unescape 0 {raw.hex() or '-'}")
for text in ("a&#233;b", "a&#xEE;b", "a&amp;b", "a&nbsp;b", "a<nul/>b", "a&#;b"):
    print(f"unescape 1 {text.encode().hex()}")
    print(f"unescape 0 {text.encode().hex()}")
for raw in (b"\xc0\x80", b"\xe0\x80\x80", b"\xed\xa0\x80", b"\xf5\x80\x80\x80", b"\x80",
            b"\xc3", b"\xc3\xa9", b"\xf0\x9f\x98\x80"):
    for pos in range(len(raw) + 1):
        print(f"utf8 {raw.hex()} {pos}")
XERPY
"${O0}" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O0.txt"
"${O3}" < "${tmp}/xer_cases.txt" > "${tmp}/xer_O3.txt"
if cmp -s "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt"; then
  echo "  PASS X.693 XER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/xer_cases.txt") cases)"
else
  echo "  FAIL: the XER twin's answers depend on the optimisation level"
  diff "${tmp}/xer_O0.txt" "${tmp}/xer_O3.txt" | head -10
  exit 1
fi
