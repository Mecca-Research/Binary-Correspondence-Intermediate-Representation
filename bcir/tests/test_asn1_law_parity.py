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
from bcir.asn1.artifact_bundle import ARTIFACT_BUNDLE_MODULE_OID
from bcir.asn1.schema import Component, Module, Primitive, Sequence
from bcir.asn1.streampack import STREAMPACK_MODULE_OID
from bcir.asn1.tags import RESERVED_UNIVERSAL, Universal

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ATTRS_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIRAttrs.td")
_ASN1_TD = os.path.join(_ROOT, "mlir", "include", "BCIR", "BCIRAsn1Ops.td")
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_asn1.mlir")
_ARTIFACT_FIXTURE = os.path.join(
    _ROOT, "mlir", "test", "passes", "artifact_bundle_asn1.mlir",
)


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


#: The law rail's transfer-syntax enum, spelled once here so the three tests below can
#: each ask a different question of it. The X.690 three keep values 0/1/2 because the
#: extension had to be ADDITIVE: every artifact, bytecode file and fixture written before
#: the other families existed must still parse and still mean what it meant.
_EXPECTED_RULES = {
    "ber": 0, "cer": 1, "der": 2,
    "basic_per_aligned": 3, "basic_per_unaligned": 4,
    "canonical_per_aligned": 5, "canonical_per_unaligned": 6,
    "oer": 7, "coer": 8,
    "xer": 9, "cxer": 10,
    "jer": 11, "bcir_canonical_jer": 12,
}


def test_encoding_rules_and_tagging_enums_agree():
    rules = _cases("Asn1Rules")
    assert rules == _EXPECTED_RULES, rules
    # The X.690 members must keep their original integers, or an artifact encoded before
    # the enum grew would decode as a different syntax.
    assert (rules["ber"], rules["cer"], rules["der"]) == (0, 1, 2), rules
    tagging = _cases("Asn1Tagging")
    assert tagging == {"implicit": 0, "explicit": 1}, tagging


def test_the_law_rail_names_every_transfer_syntax_the_oracle_can_speak():
    """One list, two rails: the ODS enum and `selection.py`'s candidate table.

    The selection harness is the oracle's own inventory of what this repository can
    encode and decode — it is what phase H measures over — so it is the right thing to
    pin the law rail against. A syntax the IR can name but the oracle cannot produce is a
    law with no implementation; one the oracle produces but the IR cannot name cannot be
    reasoned about by R24 at all, which is worse, because it means an encoding path
    exists that no static law governs.

    XER is the one entry that is *deliberately* asymmetric: `xer.py` implements BASIC-XER
    and CXER, but `selection.py` does not offer them as candidates, because the harness
    measures encodings BCIR would choose between for a wire and XER is not one of them.
    The law rail still names them, because R24 must be able to govern an XER decode.
    """
    from bcir.asn1.selection import ALL_CANDIDATES

    rules = set(_cases("Asn1Rules"))
    # The mapping is read from `Candidate.rules`, not kept here: a transcription the
    # checker maintains privately is one the checker cannot catch drifting.
    spellings = {c.name: c.rules for c in ALL_CANDIDATES}
    for name, spelling in spellings.items():
        assert spelling, f"candidate {name} declares no law-rail spelling"
        assert spelling in rules, f"{name} names {spelling!r}, which the ODS enum lacks"
    # Only these three may be named without being a candidate: `cer` because R24 has to be
    # able to REFUSE it by name, and the two XER profiles for the reason above.
    assert rules - set(spellings.values()) == {"cer", "xer", "cxer"}, sorted(rules)


def test_canonicality_agrees_between_the_law_rail_and_the_measured_candidates():
    """R24's generalized law rests on ONE predicate, so both rails must classify alike.

    The law is no longer "BCIR emits DER"; it is "BCIR emits a transfer syntax whose
    octets are a function of the abstract value", because that is the property a digest
    actually needs. `selection.py` marks the same property as `Candidate.canonical` — it
    is what splits the five selectable candidates from the five decode-only ones — so a
    disagreement here would mean the verifier permits emitting something the harness
    knows is not replayable, or refuses something it measures.

    `cer` is the case worth stating out loud. Its NAME says canonical and it is not: X.690
    §9.1 makes the indefinite length form mandatory for constructed CER encodings, so a
    CER artifact is not byte-stable however canonically it chose among BER's options.
    """
    from bcir.asn1.selection import ALL_CANDIDATES

    source = open(os.path.join(_ROOT, "mlir", "lib", "BCIRDialect.cpp"), encoding="utf-8")
    body = re.search(r"bool isCanonicalAsn1Rules\(Asn1Rules rules\) \{.*?\n\}",
                     source.read(), re.S)
    assert body, "isCanonicalAsn1Rules not found; R24's law rests on it"
    returns_true = body.group(0).split("return true;")[0]
    law_canonical = set(re.findall(r"case Asn1Rules::(\w+):", returns_true))
    assert law_canonical == {"Der", "CanonicalPerAligned", "CanonicalPerUnaligned",
                             "Coer", "Cxer", "BcirCanonicalJer"}, sorted(law_canonical)
    # CER is classified, and classified as NOT canonical.
    assert "Cer" not in law_canonical, "CER is not byte-stable (X.690 9.1)"

    # snake_case ODS spelling -> the CamelCase enumerator the pass switches on.
    def camel(spelling: str) -> str:
        return "".join(part.capitalize() for part in spelling.split("_"))

    for candidate in ALL_CANDIDATES:
        assert (camel(candidate.rules) in law_canonical) == candidate.canonical, (
            f"{candidate.name}: the oracle says canonical={candidate.canonical}, "
            f"the law rail disagrees")


def test_law_rail_declares_every_op_the_oracle_needs():
    """The schema layer's constructors must all be nameable in the IR."""
    text = open(_ASN1_TD, encoding="utf-8").read()
    for mnemonic in ("asn1.module", "asn1.type", "asn1.component", "asn1.encode",
                     "asn1.decode", "asn1.projection", "asn1.transcode"):
        assert f'BCIR_Op<"{mnemonic}"' in text, mnemonic


def test_the_generalized_r24_laws_are_pinned_by_the_fixture():
    """Each new law needs a document that trips it, or it is a claim without a witness.

    The two holes the old `strict_der && rules == ber` test left open are called out by
    name: CER passed it, though CER is exactly as un-byte-stable as BER, and so did
    `strict_der` on a JER decode, which is a category error rather than a strict setting.
    """
    fixture = open(_FIXTURE, encoding="utf-8").read()
    for needle in (
        # the generalized canonicality law, in a family that is not X.690
        "basic_per_aligned (X.691), which is not canonical",
        # the two closed holes
        "accepts cer, which is not a canonical transfer syntax",
        "strict_der but declares the X.697 syntax bcir_canonical_jer",
        # the transcode laws
        "targets ber (X.690), which is not canonical; a transcode EMITS its target",
        "has the same source and target syntax der",
        "claims preserve_value but reads jer",
    ):
        assert needle in fixture, f"no fixture witnesses: {needle}"
    # And the positive direction: every canonical syntax must be emittable somewhere.
    for spelling in ("canonical_per_unaligned", "canonical_per_aligned", "coer", "cxer",
                     "bcir_canonical_jer"):
        assert f"rules = #bcir.asn1_rules<{spelling}>" in fixture, spelling


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


def test_artifact_bundle_module_oid_and_additive_projection_match_both_rails():
    """BCAB's second transfer syntax must keep one OID and R24's additive marker."""
    fixture = open(_ARTIFACT_FIXTURE, encoding="utf-8").read()
    arcs = ", ".join(str(arc) for arc in ARTIFACT_BUNDLE_MODULE_OID)
    assert f"array<i64: {arcs}>" in fixture
    assert 'native = "artifact_bundle"' in fixture
    assert "additive" in fixture


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


def test_the_law_rail_carries_the_effective_constraint_bounds_the_oracle_computes():
    """The bounds in the IR are the EFFECTIVE ones (X.696 8.2.7/8.2.8), so both rails
    must agree on what "effective" means -- most sharply for an extensible constraint,
    which reports NO bounds (X.696 8.2.2 g) and therefore can never trip R24's emptiness
    check however odd its root looks.
    """
    from bcir.asn1.constraints import (Extensible, Size, ValueRange, is_unsatisfiable)

    text = open(_ASN1_TD, encoding="utf-8").read()
    for attribute in ("constraint_low", "constraint_high", "size_low", "size_high"):
        assert f"${attribute}" in text, f"the law rail does not carry {attribute}"

    fixture = open(_FIXTURE, encoding="utf-8").read()
    # Each emptiness the law rejects, the oracle must also call unsatisfiable...
    assert is_unsatisfiable(ValueRange(10, 1)), "oracle accepts the value set R24 rejects"
    assert is_unsatisfiable(Size(ValueRange(5, 2))), "oracle accepts the SIZE R24 rejects"
    assert is_unsatisfiable(Size(ValueRange(-1, 4))), "oracle accepts a negative SIZE"
    assert "constraint_low = 10 : i64" in fixture and "size_low = 5 : i64" in fixture
    assert "size_low = -1 : i64" in fixture
    # ...and the extensible case must be legal on BOTH rails.
    assert not is_unsatisfiable(Extensible(ValueRange(10, 1)))
    assert "@extensible_ok" in fixture, (
        "the law fixture must pin the extensible case as a POSITIVE, since X.696 8.2.2 g "
        "makes it carry no effective bounds at all")


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
        # Was "is marked strict_der but declares it accepts BER". The old law tested
        # `strict_der && rules == ber` and so let two contradictions through: CER, which
        # X.690 9.1 makes just as un-byte-stable, and a `strict_der` decode in a family
        # that has no DER at all. Both now have their own diagnostic and their own
        # fixture below.
        "but declares it accepts",
        "strict_der but declares the",
        "is not marked additive",
        # The generalized canonicality law and the three transcode laws.
        "which is not canonical; BCIR emits only a transfer syntax",
        "which is not canonical; a transcode EMITS its target",
        "has the same source and target syntax",
        "claims preserve_value but reads",
        "has an empty value constraint",
        "has an empty SIZE constraint",
        "has a negative SIZE lower bound",
    ]
    for fragment in fragments:
        assert fragment in pass_src, f"R24 diagnostic vanished from the pass: {fragment}"
        assert fragment in fixture, f"R24 diagnostic has no negative fixture: {fragment}"
