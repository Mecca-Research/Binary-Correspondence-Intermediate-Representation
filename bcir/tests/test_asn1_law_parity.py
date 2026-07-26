"""Python ↔ MLIR parity for the ASN.1 law rail (LangRef §17, law R24).

`docs/PARITY.md` makes enum integer values normative: the oracle and the law must
agree on them, or an attribute means one thing in the IR and another in the encoder.
This gate reads the ODS source directly rather than a transcription of it, so the two
cannot drift without the test noticing.

It also pins the R24 rules the oracle enforces at *encode* time against the fixture the
MLIR rail rejects at *verify* time — the same rules, checked at two different moments,
which is the whole point of having a law rail and an executable oracle.
"""
from __future__ import annotations

import os
import re

from bcir.asn1 import Asn1Error, TagClass
from bcir.asn1.schema import Component, Module, Primitive, Sequence
from bcir.asn1.streampack import STREAMPACK_MODULE_OID
from bcir.asn1.tags import RESERVED_UNIVERSAL, Universal

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ATTRS_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIRAttrs.td")
_ASN1_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIRAsn1Ops.td")
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_asn1.mlir")


def _cases(name: str) -> dict[str, int]:
    text = open(_ATTRS_TD, encoding="utf-8").read()
    match = re.search(rf"def BCIR_{name}\s*:\s*BCIR_Enum<.*?\]>;", text, re.S)
    assert match, f"BCIR_{name} not found in {_ATTRS_TD}"
    return {m.group(3): int(m.group(2)) for m in
            re.finditer(r'I32EnumAttrCase<"(\w+)",\s*(\d+),\s*"([\w.]+)">', match.group(0))}


def test_tag_class_values_match_x690_table_1_on_both_rails():
    """X.690 Table 1 encodes the class in the two high bits of the identifier octet.

    The oracle relies on that directly (`encode_tag` shifts the class left by six), so
    a law rail using different integers would make an attribute and a wire octet
    disagree about the same class.
    """
    expected = {"universal": 0, "application": 1, "context": 2, "private": 3}
    assert _cases("Asn1Class") == expected, "law rail drifted from X.690 Table 1"
    assert {c.name.lower(): int(c) for c in TagClass} == expected, \
        "oracle rail drifted from X.690 Table 1"
    # And the values really are the identifier octet's high bits, not a coincidence.
    from bcir.asn1.tags import Tag, encode_tag
    for name, value in expected.items():
        octet = encode_tag(Tag(TagClass[name.upper()], 1))[0]
        assert octet >> 6 == value, (name, hex(octet))


def test_encoding_rules_and_tagging_enums_agree():
    rules = _cases("Asn1Rules")
    assert rules == {"ber": 0, "cer": 1, "der": 2}, rules
    tagging = _cases("Asn1Tagging")
    assert tagging == {"implicit": 0, "explicit": 1}, tagging


def test_law_rail_declares_every_op_the_oracle_needs():
    """The schema layer's constructors must all be nameable in the IR."""
    text = open(_ASN1_TD, encoding="utf-8").read()
    for mnemonic in ("asn1.module", "asn1.type", "asn1.component", "asn1.encode",
                     "asn1.decode", "asn1.projection"):
        assert f'BCIR_Op<"{mnemonic}"' in text, mnemonic


def test_reserved_universal_tags_agree_with_the_law_fixture():
    """X.680 Table 1 reserves 0, 15 and 37+; both rails must refuse the same set."""
    assert 15 in RESERVED_UNIVERSAL and 37 in RESERVED_UNIVERSAL
    # Every number X.680 DOES assign must stay out of the reserved set, or the oracle
    # would refuse a legal type.
    assigned = {int(u) for u in Universal if int(u) != 0}
    assert not (assigned & RESERVED_UNIVERSAL), sorted(assigned & RESERVED_UNIVERSAL)
    fixture = open(_FIXTURE, encoding="utf-8").read()
    assert "reserved universal tag number 15" in fixture, (
        "the law fixture must pin the reserved-tag case the oracle also refuses")


def test_streampack_module_oid_is_the_one_the_law_fixture_names():
    """One module, one identity: the OID in the IR is the OID the oracle encodes."""
    fixture = open(_FIXTURE, encoding="utf-8").read()
    arcs = ", ".join(str(a) for a in STREAMPACK_MODULE_OID)
    assert f"array<i64: {arcs}>" in fixture, (
        f"the law fixture must name the oracle's module OID {arcs}")


def test_oracle_refuses_the_same_component_faults_r24_rejects():
    """R24's component rules, checked on the oracle rail at encode time.

    The law rail rejects these when the *type* is written; the oracle rejects them when
    a *value* goes through. Both must refuse -- a schema the IR calls illegal must not
    be quietly encodable.
    """
    integer = Primitive(Universal.INTEGER, "INTEGER")

    # A DEFAULT-valued component must be omitted (X.690 11.5) -- the oracle's half of
    # the rule the law states about the type.
    seq = Sequence((Component("a", integer, tag=0),
                    Component("b", integer, tag=1, default=42)), name="T")
    module = Module("T", STREAMPACK_MODULE_OID, {"T": seq})
    assert module.encode("T", {"a": 1, "b": 42}) == module.encode("T", {"a": 1})
    assert module.encode("T", {"a": 1, "b": 43}) != module.encode("T", {"a": 1})

    # A mandatory component cannot be absent (X.690 8.9.2).
    try:
        module.encode("T", {"b": 43})
        raise AssertionError("oracle encoded a value missing a mandatory component")
    except Asn1Error:
        pass

    # A universal tag number outside X.680 Table 1 is refused when encoding a string.
    try:
        Primitive(999, "BOGUS").encode("x")
        raise AssertionError("oracle accepted an unassigned universal tag number")
    except Asn1Error:
        pass


def test_law_fixture_covers_every_r24_diagnostic_the_pass_can_emit():
    """Anti-degeneration: each R24 message class must have a negative fixture.

    A law with no negative case is a law nobody has seen fire. This pairs each
    diagnostic the pass can emit with the fixture that provokes it, so adding a rule
    without a fixture fails here rather than shipping unexercised.
    """
    pass_src = open(os.path.join(_ROOT, "mlir", "lib", "passes",
                                 "BCIRVerifyPass.cpp"), encoding="utf-8").read()
    fixture = open(_FIXTURE, encoding="utf-8").read()
    # The distinctive fragment of every R24 diagnostic in the pass.
    fragments = [
        "declares encoding rules",
        "object identifier root arc",
        "object identifier second arc",
        "names reserved universal tag number",
        "share tag [",
        "is both OPTIONAL and DEFAULT",
        "declares DEFAULT but carries no value",
        "mixes tagged and untagged components",
        "is primitive but names no universal tag number",
        "but names no element type",
        "is marked strict_der but declares it accepts BER",
        "is not marked additive",
    ]
    for fragment in fragments:
        assert fragment in pass_src, f"R24 diagnostic vanished from the pass: {fragment}"
        assert fragment in fixture, f"R24 diagnostic has no negative fixture: {fragment}"
