"""A certificate must name the schema and the value, not the run that produced it.

Two defects in `select_certified`, found by asking what `sha256(repr(x))` actually hashes.
Both are the same mistake: `repr` is a *debugging* rendering, and it was being used as a
content address. Neither is a decoding bug — every certificate involved was internally
consistent — and that is exactly why they survived a green suite.

  * **`schema_digest` depended on a heap address.** `Component.default`'s sentinel was a bare
    `object()`, so every component that declares no DEFAULT reprs as
    `<object object at 0x7f...>`. Every SEQUENCE and every CHOICE therefore carried an
    address into `repr(kind)`. The sentinel is a module-level singleton, so the digest was
    stable *within* a process — which is why nothing caught it — and different on every run.
    Three runs of the same program over the same schema produced three certificates.

  * **`value_digest` depended on dict insertion order.** A SEQUENCE value is a mapping from
    component name to value, and a Python dict's `repr` follows insertion order. So
    `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` — equal as Python values, and encoding to the
    byte-identical `3006800101810102` — produced two different digests, and two certificates
    for one value.

The value fix is not "sort the dict": it is to digest the **canonical octets**, which is what
BCIR digests everywhere else and is canonical by construction.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys

from bcir.asn1.certified import Objective, build_table, select_certified
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf, SetOf
from bcir.asn1.tags import Universal
from bcir.asn1.tlv import encode_tlv
from bcir.frontends.asn1 import compile_module

_INT = Primitive(Universal.INTEGER, "INTEGER")
_ADDRESS = re.compile(r"0x[0-9a-f]{6,}")


def _certificate(kind, value):
    table = build_table(kind, value, target="host", cal_gen=1)
    return select_certified(kind, value, table, objective=Objective.WIRE_SIZE)


# --- schema identity -------------------------------------------------------------------


def test_no_schema_carries_a_heap_address_into_its_repr() -> None:
    """The property, over the shapes a real schema is built from.

    Asserted on `repr` rather than only on the digest because the digest is merely the
    first consumer; anything else that renders a schema for comparison, caching or a log
    inherits the same defect the moment it exists.
    """
    text = Primitive(Universal.UTF8_STRING, "UTF8String")
    shapes = {
        "Primitive": _INT,
        "SEQUENCE": Sequence((Component("a", _INT),), name="S"),
        "SEQUENCE with OPTIONAL": Sequence((Component("a", _INT, optional=True),), name="S"),
        "SEQUENCE with DEFAULT": Sequence((Component("a", _INT, default=1),), name="S"),
        "CHOICE": Choice((Component("a", _INT), Component("b", text, tag=1)), name="C"),
        "SEQUENCE OF": SequenceOf(_INT),
        "SET OF": SetOf(_INT),
        "nested": Sequence(
            (
                Component("inner", Sequence((Component("a", _INT),), name="I")),
                Component("b", text, tag=1, optional=True),
            ),
            name="N",
        ),
        "recursive": compile_module(
            "M DEFINITIONS ::= BEGIN Node ::= SEQUENCE { value INTEGER, next Node OPTIONAL } END"
        ).types["Node"],
    }
    for label, kind in shapes.items():
        found = _ADDRESS.findall(repr(kind))
        assert not found, f"{label}: repr carries {found}"


def test_the_schema_digest_is_reproducible_across_processes() -> None:
    """The failure as it would really be met: the same schema, two runs, two certificates.

    Run in subprocesses on purpose. Within one process the sentinel is a singleton, so its
    address never moves and an in-process test passes against the defect — which is exactly
    what happened.
    """
    script = (
        "import hashlib;"
        "from bcir.frontends.asn1 import compile_module;"
        "k = compile_module('M DEFINITIONS ::= BEGIN "
        "Node ::= SEQUENCE { value INTEGER, next Node OPTIONAL } END').types['Node'];"
        "print(hashlib.sha256(repr(k).encode()).hexdigest())"
    )
    digests = set()
    for _ in range(3):
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
        assert done.returncode == 0, done.stderr
        digests.add(done.stdout.strip())
    assert len(digests) == 1, f"three runs produced {len(digests)} distinct digests: {digests}"


def test_two_certificates_for_the_same_schema_agree() -> None:
    """Through the real entry point, with both schema objects alive at once."""
    src = "M DEFINITIONS ::= BEGIN S ::= SEQUENCE { a INTEGER, b UTF8String OPTIONAL } END"
    first = compile_module(src).types["S"]
    second = compile_module(src).types["S"]
    value = {"a": 1, "b": "x"}
    assert _certificate(first, value).schema_digest == _certificate(second, value).schema_digest


# --- value identity --------------------------------------------------------------------


def test_the_value_digest_follows_the_octets_not_the_dict_order() -> None:
    """Equal values, byte-identical encodings, one digest.

    The encoding equality is asserted first because it is what makes the digest inequality
    a defect rather than a preference: there is no sense in which these are two values.
    """
    kind = Sequence((Component("a", _INT, tag=0), Component("b", _INT, tag=1)), name="S")
    forward, reverse = {"a": 1, "b": 2}, {"b": 2, "a": 1}
    assert forward == reverse
    assert encode_tlv(kind.encode(forward)) == encode_tlv(kind.encode(reverse))

    one, two = _certificate(kind, forward), _certificate(kind, reverse)
    assert one.value_digest == two.value_digest
    assert one.schema_digest == two.schema_digest

    # And it is the canonical octets, so an independent computation reproduces it.
    assert one.value_digest == hashlib.sha256(encode_tlv(kind.encode(forward))).hexdigest()


def test_distinct_values_keep_distinct_digests() -> None:
    """The direction that matters for a digest: no collisions across values or shapes."""
    kind = Sequence((Component("a", _INT, tag=0), Component("b", _INT, tag=1)), name="S")
    values = [{"a": 1, "b": 2}, {"a": 2, "b": 1}, {"a": 0, "b": 0}, {"a": 1, "b": 3}]
    digests = {_certificate(kind, v).value_digest for v in values}
    assert len(digests) == len(values), "two distinct values share a digest"

    # Across types too: the same Python object under two schemas is two abstract values.
    enumerated = Primitive(Universal.ENUMERATED, "ENUMERATED", enumeration=(("a", 0), ("b", 1)))
    assert _certificate(_INT, 1).value_digest != _certificate(enumerated, 1).value_digest


def test_a_value_nothing_can_encode_still_gets_a_certificate() -> None:
    """The no-winner certificate has to remain issuable, and its digest must not collide.

    A certificate recording "no candidate could carry this" is a real answer, so the digest
    has to be computable when there are no canonical octets to digest. It is domain-separated
    from the encodable case rather than falling back to `repr`.
    """
    table = build_table(_INT, 5, target="host", cal_gen=1)
    certificate = select_certified(_INT, object(), table, objective=Objective.WIRE_SIZE)
    assert certificate.selected is None
    assert certificate.value_digest
    assert certificate.value_digest != _certificate(_INT, 5).value_digest
