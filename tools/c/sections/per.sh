#!/usr/bin/env bash
# per: X.691 PER primitives: strict-warning and freestanding build (#per)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-per` CTest entry. The body is the gate's section text
# moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binaries are the
# manifest variants test_per_O0 and test_per_O3: the twin's harness with its whole library closure
# rebuilt at -O0 and at -O3, whose answers must not depend on the optimiser.
#
#   usage: per.sh <test_per_O0> <test_per_O3>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: per.sh <test_per_O0> <test_per_O3>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
O0="$(abs "$1")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable variant"; exit 2; }
O3="$(abs "$2")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable variant"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# X.691 PER decoding primitives (#per, roadmap phase C): the C twin of clause 11. PER is
# NOT self-delimiting (X.691 7.2), so unlike the X.690 twin there is no schema-free
# structure walk -- clause 11's whole-number and length decoders ARE the schema-free layer,
# and they are the ones that take an attacker-supplied width, octet count or fragment header
# and move a cursor with it. The dual-rail differential lives in bcir/tests/test_c_per.py;
# what is checked HERE is the same discipline the other twins get: strict warnings as
# errors, and a genuinely freestanding translation unit.
"${PYTHON}" - "${tmp}" <<'PERPY' > "${tmp}/per_cases.txt"
import sys, random
sys.path.insert(0, ".")
from bcir.asn1.per import (BitWriter, PerVariant, _encode_constrained,
                           _encode_semi_constrained, _encode_unconstrained,
                           _encode_normally_small)
rng = random.Random(20260726)
for variant, flag in ((PerVariant.UNALIGNED, 0), (PerVariant.ALIGNED, 1)):
    for lb, ub in ((0, 255), (0, 256), (0, 65535), (0, 1 << 40), (-5, 5)):
        for _ in range(40):
            v = rng.randint(lb, ub)
            w = BitWriter(variant); _encode_constrained(w, v, lb, ub)
            print(f"constrained {lb} {ub} {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(-(1 << 40), 1 << 40)
        w = BitWriter(variant); _encode_unconstrained(w, v)
        print(f"unconstrained {flag} {w.to_bytes().hex()}")
    for _ in range(40):
        v = rng.randint(0, 1 << 20)
        w = BitWriter(variant); _encode_semi_constrained(w, v, 0)
        print(f"semi 0 {flag} {w.to_bytes().hex()}")
    for v in (0, 63, 64, 300):
        w = BitWriter(variant); _encode_normally_small(w, v)
        print(f"small {flag} {w.to_bytes().hex()}")
PERPY
"${O0}" < "${tmp}/per_cases.txt" > "${tmp}/per_O0.txt"
"${O3}" < "${tmp}/per_cases.txt" > "${tmp}/per_O3.txt"
if cmp -s "${tmp}/per_O0.txt" "${tmp}/per_O3.txt"; then
  echo "  PASS X.691 PER twin (freestanding, -Werror, -O0 == -O3 over $(wc -l < "${tmp}/per_cases.txt") cases)"
else
  echo "  FAIL: the PER twin's answers depend on the optimisation level"
  diff "${tmp}/per_O0.txt" "${tmp}/per_O3.txt" | head -10
  exit 1
fi
