#!/usr/bin/env bash
# asn1_emit: plan-driven ASN.1 encoder: strict-warning and freestanding build (#emit)
#
# One section of tools/c/check_runtime.sh, run against binaries built elsewhere: the gate compiles
# them its own way and calls this script; the CMake project builds them from runtime/manifest.json
# and runs this script as the `c-section-asn1_emit` CTest entry. The body is the gate's section
# text moved here (BUILD-2, docs/BCIR_BUILD_ROADMAP.md), so the two run one text, and
# tools/build/section_parity.py holds the two builds' outputs byte-identical. The binaries are the
# manifest variants test_emit_O0 and test_emit_O3: the twin's harness with its whole library
# closure rebuilt at -O0 and at -O3, whose answers must not depend on the optimiser.
#
#   usage: asn1_emit.sh <test_emit_O0> <test_emit_O3>
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
C="${ROOT}/runtime/c"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
[ $# -eq 2 ] || { echo "usage: asn1_emit.sh <test_emit_O0> <test_emit_O3>" >&2; exit 2; }
abs() { printf '%s/%s\n' "$(cd "$(dirname "$1")" && pwd)" "$(basename "$1")"; }
O0="$(abs "$1")"; [ -x "${O0}" ] || { echo "  FAIL: O0: ${O0} is not an executable variant"; exit 2; }
O3="$(abs "$2")"; [ -x "${O3}" ] || { echo "  FAIL: O3: ${O3} is not an executable variant"; exit 2; }
tmp="$(mktemp -d)"; trap 'rm -rf "${tmp}"' EXIT

# Plan-driven ASN.1 encoder (#emit, E2): the write-side twin of bcir/asn1/emit.py. #682
# established that only X.690 can be encoded WITHOUT a type -- X.697 22.2 puts member
# identifiers in a JER document and an identifier exists only in the schema -- so every
# emitter here is schema-DIRECTED and all four read one format-neutral value stream. That is
# what makes their costs comparable at all. -O0 == -O3 earns its place twice over: the file
# does long division on octet arrays for arbitrary-width integers, and it compares
# attacker-supplied counts against remaining stream lengths.
"${PYTHON}" - <<'EMITPY' > "${tmp}/emit_cases.txt"
import sys
sys.path.insert(0, ".")
from bcir.asn1.codec import NULL, Oid
from bcir.asn1.constraints import Extensible, Size, ValueRange
from bcir.asn1.emit import flatten
from bcir.asn1.encode_plan import compile_encode_plan
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf
from bcir.asn1.tags import Universal

I = Primitive(Universal.INTEGER)
S = Primitive(Universal.UTF8_STRING)
B = Primitive(Universal.BOOLEAN)
N = Primitive(Universal.NULL)
O = Primitive(Universal.OCTET_STRING)
D = Primitive(Universal.OBJECT_IDENTIFIER)


def seq(*components, name="X"):
    return Sequence(tuple(components), name=name)


CH = Choice((Component("num", I, tag=0), Component("txt", S, tag=1)), name="C")

# Version 4: an ENUMERATED needs its enumeration, because X.697 22.2 spells the value as the
# IDENTIFIER of its item and X.691 14.1 indexes the root. A hyphen exercises the descriptor's
# `name:number|...` field against an identifier X.680 12.4 permits.
ENUM = (("five", 5), ("two-hundred", 200), ("minus-one", -1))

CASES = [
    (seq(Component("v", I)), {"v": -1}),
    (seq(Component("v", I)), {"v": 2 ** 64 + 7}),
    (seq(Component("v", I)), {"v": 2 ** 400 + 12345}),
    (seq(Component("v", I)), {"v": 0}),
    (seq(Component("v", I)), {"v": 128}),
    (seq(Component("a", B), Component("b", B)), {"a": True, "b": False}),
    (seq(Component("v", N)), {"v": NULL}),
    (seq(Component("a", I), Component("v", N), Component("b", I)),
     {"a": 1, "v": NULL, "b": 2}),
    (seq(Component("v", O)), {"v": b"\x00\xff\x10"}),
    (seq(Component("v", O)), {"v": b""}),
    (seq(Component("v", S)), {"v": ""}),
    (seq(Component("v", S)), {"v": "a\nb\tc\"d\\e"}),
    (seq(Component("v", S)), {"v": "x" * 300}),
    (seq(Component("v", D)), {"v": Oid((1, 3, 6, 1, 4, 1, 62596, 1))}),
    (seq(Component("v", D)), {"v": Oid((2, 999, 1234567))}),
    (seq(Component("a", I), Component("b", I, optional=True)), {"a": 1, "b": 2}),
    (seq(Component("a", I), Component("b", I, optional=True)), {"a": 1}),
    (seq(Component("a", I), Component("b", B, default=False)), {"a": 1}),
    # Version 5: a DEFAULT component whose value EQUALS the default must be OMITTED (X.690
    # 11.5, X.696 31.9, CJER). Every row above supplies one that differs, which is exactly
    # how three emitters shipped emitting it.
    (seq(Component("a", I), Component("b", B, default=False)), {"a": 1, "b": False}),
    (seq(Component("a", I), Component("b", I, default=7)), {"a": 1, "b": 7}),
    (seq(Component("a", I), Component("b", S, default="hi")), {"a": 1, "b": "hi"}),
    (seq(*[Component("c%d" % i, I, default=i) for i in range(12)]),
     dict(("c%d" % i, (i if i % 2 else 99)) for i in range(12))),
    (seq(*[Component("c%d" % i, I, optional=True) for i in range(12)]),
     dict(("c%d" % i, i) for i in range(0, 12, 2))),
    (seq(*[Component("c%d" % i, I, optional=True) for i in range(12)]), {}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": []}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": [1, 2, 3]}),
    (seq(Component("v", SequenceOf(I, "SEQ"))), {"v": list(range(300))}),
    (seq(Component("in", seq(Component("a", I), name="In"))), {"in": {"a": 5}}),
    (seq(Component("a", I, tag=0), Component("b", S, tag=1)), {"a": 1, "b": "x"}),
    (seq(Component("a", I, tag=0, explicit=True)), {"a": 1}),
    (seq(Component("a", I, tag=100, explicit=True)), {"a": 1}),
    (seq(Component("a", I, tag=100)), {"a": 1}),
    (seq(Component("v", CH, tag=5, explicit=True)), {"v": ("num", 7)}),
    (seq(Component("v", CH, tag=5, explicit=True)), {"v": ("txt", "hi")}),
    (seq(Component("s", seq(Component("a", I), name="In"), tag=3)), {"s": {"a": 9}}),
    (seq(Component("v", SequenceOf(I, "SEQ"), tag=4)), {"v": [1, 2]}),
]

# Plan version 3's cases. Every row above is UNCONSTRAINED, and that is precisely how an OER
# emitter ignoring constraints passed this gate: X.696 10.3 gives a constrained INTEGER a
# fixed-width form with no length determinant, and nothing here ever asked for one. An
# ENUMERATED is here for the same reason -- 11 is not 10, and nothing asked for that either.
CASES += [
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, 255)))), {"v": 42}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, 2 ** 64 - 1)))), {"v": 2 ** 63}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(-128, 127)))), {"v": -5}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=ValueRange(0, None)))), {"v": 300}),
    (seq(Component("v", Primitive(Universal.INTEGER, "I",
                                  constraint=Extensible(ValueRange(0, 255))))), {"v": 42}),
    (seq(Component("v", Primitive(Universal.OCTET_STRING, "O",
                                  constraint=Size(ValueRange(3, 3))))), {"v": b"abc"}),
    (seq(Component("v", Primitive(Universal.IA5_STRING, "A",
                                  constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
    (seq(Component("v", Primitive(Universal.UTF8_STRING, "U",
                                  constraint=Size(ValueRange(3, 3))))), {"v": "abc"}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": 5}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": 200}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM))), {"v": -1}),
    (seq(Component("v", Primitive(Universal.ENUMERATED, "E", enumeration=ENUM,
                                  enum_extensible=True))), {"v": 5}),
    (Sequence((Component("a", I),), name="X", extensible=True), {"a": 1}),
]

for index, (kind, value) in enumerate(CASES):
    plan = compile_encode_plan(kind, module="Gate", type_name="c%d" % index)
    stream = flatten(plan, value)
    print("plan %s" % plan.serialize().hex())
    for rules in ("der", "ber", "jer", "coer", "cper-a", "cper-u", "bper-a", "bper-u"):
        print("emit %s %s" % (rules, stream.hex() or "-"))
EMITPY
"${O0}" < "${tmp}/emit_cases.txt" > "${tmp}/emit_O0.txt"
"${O3}" < "${tmp}/emit_cases.txt" > "${tmp}/emit_O3.txt"
if ! diff -q "${tmp}/emit_O0.txt" "${tmp}/emit_O3.txt" >/dev/null; then
  echo "  FAIL: the plan-driven encoder answers differently at -O0 and -O3"
  exit 1
fi
emit_cases=$(grep -c '^emit ' "${tmp}/emit_cases.txt")
if grep -q '^err ' "${tmp}/emit_O0.txt"; then
  echo "  FAIL: the encoder refused a case from its own reference corpus"
  exit 1
fi
# A section passes with a PASS line (tools/build/section_parity.py requires one); this verdict said
# `ok:` while it lived in the gate (BUILD-2j).
echo "  PASS plan-driven ASN.1 encoder: -O0 == -O3 over ${emit_cases} encode cases (8 candidates x 1 plan each)"
