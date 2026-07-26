"""X.691 PER conformance.

The load-bearing tests here are the Annex A ones. A round trip only proves the encoder and
decoder share an interpretation -- if both read a clause the same wrong way, it still
passes. Annex A gives the standard's OWN octets for a worked value, so comparing against
them is the check that a round trip cannot make: it caught a real bug in the front-end
(serial constraint application dropping a permitted alphabet) that every round-trip test in
this file was blind to.

A.1 (no constraints) and A.2 (subtype constraints) are reproduced for both variants. The
ASN.1 modules and expected octets below are the ones printed in X.691 (2021) Annex A.
"""

from __future__ import annotations

from bcir.asn1.codec import Asn1Error
from bcir.asn1.per import (
    BASIC_PER_ALIGNED_OID,
    BASIC_PER_UNALIGNED_OID,
    CANONICAL_PER_ALIGNED_OID,
    CANONICAL_PER_UNALIGNED_OID,
    PerRules,
    PerVariant,
    bits_for_range,
    decode_per,
    encode_per,
    rules_oid,
)
from bcir.asn1.schema import Component, Primitive, Sequence, SequenceOf
from bcir.asn1.tags import Universal
from bcir.frontends.asn1.lower import compile_module

# --- X.691 Annex A ---------------------------------------------------------------------

_A1_MODULE = """
AnnexA1 DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name,
      title [0] VisibleString,
      number EmployeeNumber,
      dateOfHire [1] Date,
      nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE OF ChildInformation DEFAULT {} }
  ChildInformation ::= SET { name Name, dateOfBirth [0] Date }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName VisibleString, initial VisibleString, familyName VisibleString }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER
  Date ::= [APPLICATION 3] IMPLICIT VisibleString
END
"""

# A.2 differs only in that the strings carry permitted-alphabet and size constraints.
_A2_MODULE = """
AnnexA2 DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name,
      title [0] VisibleString,
      number EmployeeNumber,
      dateOfHire [1] Date,
      nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE OF ChildInformation DEFAULT {} }
  ChildInformation ::= SET { name Name, dateOfBirth [0] Date }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName NameString, initial NameString (SIZE(1)), familyName NameString }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER
  Date ::= [APPLICATION 3] IMPLICIT VisibleString (FROM("0".."9") ^ SIZE(8))
  NameString ::= VisibleString (FROM("a".."z" | "A".."Z" | "-.") ^ SIZE(1..64))
END
"""

_PERSONNEL_RECORD = {
    "name": {"givenName": "John", "initial": "P", "familyName": "Smith"},
    "title": "Director",
    "number": 51,
    "dateOfHire": "19710917",
    "nameOfSpouse": {"givenName": "Mary", "initial": "T", "familyName": "Smith"},
    "children": [
        {"name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
         "dateOfBirth": "19571111"},
        {"name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
         "dateOfBirth": "19590717"},
    ],
}

#: A.1.3.1 / A.1.4.1 hexadecimal views.
_A1_ALIGNED = bytes.fromhex(
    "80044a6f686e015005536d6974680133"
    "084469726563746f7208313937313039"
    "3137044d617279015405536d69746802"
    "0552616c7068015405536d6974680831"
    "3935373131313105537573616e014205"
    "4a6f6e6573083139353930373137"
)
_A1_UNALIGNED = bytes.fromhex(
    "824adfa3700d005a7b74f4d002661113"
    "4f2cb8fa6fe410c5cb762c1cb16e0937"
    "0f2f20350169edd3d340102d2c3b3868"
    "01a80b4f6e9e9a0218b96add8b162c41"
    "69f5e787700c20595bf765e610c5cb57"
    "2c1bb16e"
)
#: A.2.3.1 / A.2.4.1 hexadecimal views.
_A2_ALIGNED = bytes.fromhex(
    "864a6f686e5010536d69746801330844"
    "69726563746f72197109170c4d617279"
    "5410536d697468021052616c70685410"
    "536d6974681957111110537573616e42"
    "104a6f6e657319590717"
)
_A2_UNALIGNED = bytes.fromhex(
    "865d51d2888a5125f180998444d3cb2e"
    "3e9bf90cb8848b867396e8a88a5125f1"
    "81089b93d71aa2294497c632ae222222"
    "985ce521885d54c170cac838b8"
)



# A.3 adds extension markers throughout: an extensible SET, extensible SEQUENCEs, an
# extensible INTEGER constraint, and extensible SIZE constraints.
_A3_MODULE = """
AnnexA3 DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name,
      title [0] VisibleString,
      number EmployeeNumber,
      dateOfHire [1] Date,
      nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE (SIZE(2, ...)) OF ChildInformation OPTIONAL,
      ... }
  ChildInformation ::= SET {
      name Name,
      dateOfBirth [0] Date,
      ...,
      sex [1] IMPLICIT ENUMERATED {male(1), female(2), unknown(3)} OPTIONAL }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName NameString, initial NameString (SIZE(1)), familyName NameString, ... }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER (0..9999, ...)
  Date ::= [APPLICATION 3] IMPLICIT VisibleString (FROM("0".."9") ^ SIZE(8, ..., 9..20))
  NameString ::= VisibleString (FROM("a".."z" | "A".."Z" | "-.") ^ SIZE(1..64, ...))
END
"""

_A3_RECORD = {
    "name": {"givenName": "John", "initial": "P", "familyName": "Smith"},
    "title": "Director",
    "number": 51,
    "dateOfHire": "19710917",
    "nameOfSpouse": {"givenName": "Mary", "initial": "T", "familyName": "Smith"},
    "children": [
        {"name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
         "dateOfBirth": "19571111"},
        {"name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
         "dateOfBirth": "19590717", "sex": 2},
    ],
}

# A.4 is the extension-addition-GROUP case: version brackets in a SEQUENCE and in a
# CHOICE, an extension marker PAIR with root components written after it, AUTOMATIC TAGS.
_A4_MODULE = """
AnnexA4 DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  Ax ::= SEQUENCE {
      a INTEGER (250..253),
      b BOOLEAN,
      c CHOICE { d INTEGER, ..., [[ e BOOLEAN, f IA5String ]], ... },
      ...,
      [[ g NumericString (SIZE(3)), h BOOLEAN OPTIONAL ]],
      ...,
      i BMPString OPTIONAL,
      j PrintableString OPTIONAL }
END
"""

_A4_RECORD = {"a": 253, "b": True, "c": {"e": True}, "g": "123", "h": True}

_A3_ALIGNED = bytes.fromhex(
    "40c04a6f686e5008536d697468000033"
    "084469726563746f720019710917034d"
    "6172795408536d697468010052616c70"
    "685408536d6974680019571111820053"
    "7573616e42084a6f6e65730019590717"
    "010140"
)
_A3_UNALIGNED = bytes.fromhex(
    "40cbaa3a5108a5125f180330889a7965"
    "c7d37f20cb8848b819ce5ba2a114a24b"
    "e30113727ae3542294497c6195711118"
    "22985ce521842eaa60b832b20e2e0202"
    "80"
)
_A4_ALIGNED = bytes.fromhex(
    "9e000180010291a4"
)
_A4_UNALIGNED = bytes.fromhex(
    "9e000600040a4690"
)



def _encode_annex(module: str, variant: PerVariant) -> bytes:
    lowered = compile_module(module, "<annex>")
    return encode_per(lowered.module.types["PersonnelRecord"], _PERSONNEL_RECORD,
                      variant=variant, rules=PerRules.CANONICAL)


def test_annex_a1_aligned_matches_the_specification_octets():
    """A.1.3: the unconstrained record, ALIGNED. 94 octets, as the clause states."""
    got = _encode_annex(_A1_MODULE, PerVariant.ALIGNED)
    assert len(got) == 94, f"A.1.3 is 94 octets, got {len(got)}"
    assert got == _A1_ALIGNED, f"\n got {got.hex()}\n exp {_A1_ALIGNED.hex()}"


def test_annex_a1_unaligned_matches_the_specification_octets():
    """A.1.4: the same value UNALIGNED is 84 octets -- ten fewer, all of it padding."""
    got = _encode_annex(_A1_MODULE, PerVariant.UNALIGNED)
    assert len(got) == 84, f"A.1.4 is 84 octets, got {len(got)}"
    assert got == _A1_UNALIGNED, f"\n got {got.hex()}\n exp {_A1_UNALIGNED.hex()}"


def test_annex_a2_aligned_matches_the_specification_octets():
    """A.2.3: constraints shrink the same value from 94 to 74 octets."""
    got = _encode_annex(_A2_MODULE, PerVariant.ALIGNED)
    assert len(got) == 74, f"A.2.3 is 74 octets, got {len(got)}"
    assert got == _A2_ALIGNED, f"\n got {got.hex()}\n exp {_A2_ALIGNED.hex()}"


def test_annex_a2_unaligned_matches_the_specification_octets():
    """A.2.4: 61 octets, against BER's 136. This is the compaction PER exists for."""
    got = _encode_annex(_A2_MODULE, PerVariant.UNALIGNED)
    assert len(got) == 61, f"A.2.4 is 61 octets, got {len(got)}"
    assert got == _A2_UNALIGNED, f"\n got {got.hex()}\n exp {_A2_UNALIGNED.hex()}"


def test_annex_records_round_trip_in_both_variants():
    for module in (_A1_MODULE, _A2_MODULE):
        lowered = compile_module(module, "<annex>")
        kind = lowered.module.types["PersonnelRecord"]
        for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
            data = encode_per(kind, _PERSONNEL_RECORD, variant=variant)
            assert decode_per(data, kind, variant=variant) == _PERSONNEL_RECORD


def test_serial_constraint_application_keeps_the_inner_permitted_alphabet():
    """X.680 §50.11 / X.691 §10.3.20, the bug Annex A.2 caught.

    `initial NameString (SIZE(1))` constrains an already-constrained type. The outer SIZE
    narrows the length; NameString's 54-character alphabet must survive, because it is what
    fixes each character at six bits instead of VisibleString's seven. Overwriting the
    inner constraint is invisible to DER and OER and silently changes every PER width.
    """
    lowered = compile_module(_A2_MODULE, "<annex>")
    name = lowered.module.types["Name"]
    initial = next(c for c in name.components if c.name == "initial")
    from bcir.asn1.constraints import effective_size_constraint

    low, high = effective_size_constraint(initial.type.constraint)
    assert (low, high) == (1, 1), "the outer SIZE(1) must win for the length"
    alphabet = initial.type.constraint.alphabet()
    assert alphabet is not None and len(set(alphabet)) == 54, (
        "NameString's permitted alphabet must survive serial application")


# --- clause 11: the whole-number and length machinery ------------------------------------


def test_bits_for_range_matches_the_aligned_table_in_11_5_7_1():
    """§11.5.6 and the §11.5.7.1 table have to agree; this pins both."""
    table = {2: 1, 3: 2, 4: 2, 5: 3, 8: 3, 9: 4, 16: 4, 17: 5, 32: 5,
             33: 6, 64: 6, 65: 7, 128: 7, 129: 8, 255: 8}
    for range_, expected in table.items():
        assert bits_for_range(range_) == expected, f"range {range_}"
    assert bits_for_range(1) == 0, "§11.5.4: a range of 1 is an empty bit-field"
    assert bits_for_range(256) == 8


def test_constrained_integer_widths_follow_the_variant():
    """§11.5.6 against §11.5.7: the same type costs different bits in each variant."""
    kind = Sequence((Component("v", _int_range(0, 255)),), name="S")
    assert len(encode_per(kind, {"v": 200}, variant=PerVariant.UNALIGNED)) == 1
    # range 256 is §11.5.7.2's one-octet case; the value still lands in a single octet.
    assert len(encode_per(kind, {"v": 200}, variant=PerVariant.ALIGNED)) == 1
    wide = Sequence((Component("v", _int_range(0, (1 << 64) - 1)),), name="S")
    # UNALIGNED spends the full 64-bit range width; ALIGNED takes §11.5.7.4's
    # length-prefixed minimum-octet form and is SMALLER for a small value.
    assert len(encode_per(wide, {"v": 1}, variant=PerVariant.UNALIGNED)) == 8
    assert len(encode_per(wide, {"v": 1}, variant=PerVariant.ALIGNED)) < 8


def test_single_value_constraint_encodes_nothing():
    """§13.2.1: a value set of one carries no information, so it occupies no field."""
    kind = Sequence((Component("v", _int_range(7, 7)),), name="S")
    data = encode_per(kind, {"v": 7})
    assert data == b"\x00", "§11.1.3.1: an empty encoding becomes one zero octet"
    assert decode_per(data, kind) == {"v": 7}


def test_unconstrained_length_uses_the_one_and_two_octet_forms():
    """§11.9.3.6 (n <= 127, bit 8 zero) and §11.9.3.7 (n < 16K, bits 8-7 = 10)."""
    octets = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    kind = Sequence((Component("v", octets),), name="S")
    short = encode_per(kind, {"v": b"\x01" * 100}, variant=PerVariant.ALIGNED)
    assert short[0] == 100 and len(short) == 101
    long_ = encode_per(kind, {"v": b"\x01" * 200}, variant=PerVariant.ALIGNED)
    assert long_[0] == 0x80 and long_[1] == 200 and len(long_) == 202


def test_length_fragmentation_round_trips_past_16k():
    """§11.9.3.8: past 16K the material is emitted in 16K blocks, each re-length-prefixed.

    The trailing short form matters as much as the blocks: §11.9.3.8.3's NOTE requires a
    final zero length when the last block exactly fills, and a decoder that stops at the
    last full block silently truncates.
    """
    octets = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    kind = Sequence((Component("v", octets),), name="S")
    for size in (16 * 1024, 16 * 1024 + 1, 40 * 1024, 64 * 1024):
        payload = bytes((i * 7 + 1) & 0xFF for i in range(size))
        for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
            data = encode_per(kind, {"v": payload}, variant=variant)
            assert decode_per(data, kind, variant=variant) == {"v": payload}, size


# --- clause 14: enumerated ---------------------------------------------------------------


def test_enumerated_encodes_the_index_not_the_value():
    """§14.1/§14.2: the enumeration INDEX, over a range of the enumeration count.

    This is the one place PER needs strictly more schema than DER/OER, both of which encode
    the value itself (X.690 §8.4, X.696 §11).
    """
    kind = Primitive(Universal.ENUMERATED, "E",
                     enumeration=(("red", 4), ("green", 9), ("blue", 25)))
    wrapper = Sequence((Component("v", kind),), name="S")
    # three enumerations -> indices 0..2 -> a 2-bit field, sorted ascending by value.
    assert encode_per(wrapper, {"v": 4}, variant=PerVariant.UNALIGNED) == b"\x00"
    assert encode_per(wrapper, {"v": 9}, variant=PerVariant.UNALIGNED) == b"\x40"
    assert encode_per(wrapper, {"v": 25}, variant=PerVariant.UNALIGNED) == b"\x80"
    for value in (4, 9, 25):
        data = encode_per(wrapper, {"v": value})
        assert decode_per(data, wrapper) == {"v": value}


def test_enumerated_without_an_enumeration_is_refused_rather_than_guessed():
    """A bare ENUMERATED is encodable under DER/OER and not under PER; say so."""
    kind = Sequence((Component("v", Primitive(Universal.ENUMERATED, "E")),), name="S")
    try:
        encode_per(kind, {"v": 1})
        raise AssertionError("a bare ENUMERATED must not be given an invented index")
    except Asn1Error as exc:
        assert "enumeration index" in str(exc)


def test_enumerated_rejects_a_value_outside_the_enumeration():
    kind = Sequence((Component("v", Primitive(
        Universal.ENUMERATED, "E", enumeration=(("a", 0), ("b", 1)))),), name="S")
    try:
        encode_per(kind, {"v": 7})
        raise AssertionError("7 is not an enumeration value of this type")
    except Asn1Error:
        pass


# --- clause 19: sequence extensibility ---------------------------------------------------


def _extensible_pair():
    base = Sequence((
        Component("a", _int_range(0, 255)),
        Component("b", _int_range(0, 255), optional=True),
    ), name="S", extensible=True)
    extended = Sequence((
        Component("a", _int_range(0, 255)),
        Component("b", _int_range(0, 255), optional=True),
        Component("c", _int_range(0, 255), optional=True, extension=True),
    ), name="S", extensible=True)
    return base, extended


def test_extension_bit_is_zero_when_no_addition_is_present():
    """§19.1: the extension bit says whether any addition is encoded, nothing more."""
    base, extended = _extensible_pair()
    without = encode_per(extended, {"a": 1, "b": 2})
    assert encode_per(base, {"a": 1, "b": 2}) == without, (
        "a type with additions, none present, must encode as the base type does")


def test_an_older_reader_skips_an_unknown_extension_addition():
    """§19.7-§19.9: the addition bitmap plus open-type wrapper is version tolerance.

    This is the property PER extension markers exist for: a peer built against the older
    type must recover every root component from a newer peer's encoding, and skip the rest
    rather than misparse it.
    """
    base, extended = _extensible_pair()
    newer = encode_per(extended, {"a": 1, "b": 2, "c": 3})
    assert decode_per(newer, base) == {"a": 1, "b": 2}
    assert decode_per(newer, extended) == {"a": 1, "b": 2, "c": 3}


def test_extension_additions_round_trip_in_both_variants():
    _, extended = _extensible_pair()
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        for value in ({"a": 1}, {"a": 1, "b": 2}, {"a": 1, "c": 9},
                      {"a": 1, "b": 2, "c": 3}):
            data = encode_per(extended, value, variant=variant)
            assert decode_per(data, extended, variant=variant) == value, (variant, value)


# --- trust boundary ----------------------------------------------------------------------


def test_trailing_octets_are_refused():
    """§7.2: a PER encoding is not self-delimiting, so slack is a fault, not a spelling."""
    kind = Sequence((Component("v", _int_range(0, 255)),), name="S")
    data = encode_per(kind, {"v": 9})
    try:
        decode_per(data + b"\x00", kind)
        raise AssertionError("an appended octet must not be ignored")
    except Asn1Error as exc:
        assert "trailing" in str(exc)


def test_non_zero_padding_is_refused():
    """§11.1.4 pads with ZERO bits; a one bit there is a second spelling of one value."""
    octets = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    kind = Sequence((Component("f", Primitive(Universal.BOOLEAN, "BOOLEAN")),
                     Component("v", octets)), name="S")
    data = bytearray(encode_per(kind, {"f": True, "v": b"\xaa"}, variant=PerVariant.ALIGNED))
    data[0] |= 0x40                                   # a pad bit after the boolean
    try:
        decode_per(bytes(data), kind, variant=PerVariant.ALIGNED)
        raise AssertionError("a non-zero pad bit must be refused")
    except Asn1Error:
        pass


def test_truncated_input_is_refused():
    octets = Primitive(Universal.OCTET_STRING, "OCTET STRING")
    kind = Sequence((Component("v", octets),), name="S")
    data = encode_per(kind, {"v": b"\x01" * 40})
    try:
        decode_per(data[:5], kind)
        raise AssertionError("a truncated encoding must be refused")
    except Asn1Error:
        pass


def test_rule_object_identifiers_are_the_four_of_clause_33_2():
    assert BASIC_PER_ALIGNED_OID == (2, 1, 3, 0, 0)
    assert BASIC_PER_UNALIGNED_OID == (2, 1, 3, 0, 1)
    assert CANONICAL_PER_ALIGNED_OID == (2, 1, 3, 1, 0)
    assert CANONICAL_PER_UNALIGNED_OID == (2, 1, 3, 1, 1)
    assert rules_oid(PerRules.CANONICAL, PerVariant.UNALIGNED) == (2, 1, 3, 1, 1)


# --- the BCAB projection -----------------------------------------------------------------


def test_artifact_bundle_per_projection_is_byte_identical_and_smaller_than_der():
    """The A1-A4 laws restated for PER, plus the roadmap's size law."""
    from bcir.abi.artifact_bundle import encode_bundle
    from bcir.asn1.artifact_bundle import (
        decode_bundle_per, native_to_der, native_to_per, per_to_native,
    )
    from bcir.tests.test_asn1_artifact_bundle import _three_bundle

    bundle = _three_bundle()
    native = encode_bundle(bundle)
    der = native_to_der(native)
    for aligned in (False, True):
        per = native_to_per(native, aligned=aligned)
        assert per_to_native(per, aligned=aligned) == native, (
            f"native -> PER -> native must be byte-identical (aligned={aligned})")
        assert decode_bundle_per(per, aligned=aligned) == bundle
        assert len(per) <= len(der), (
            f"PER must not be larger than DER: {len(per)} vs {len(der)}")


def test_the_two_per_variants_do_not_interwork():
    """§7.8. Decoding one variant's octets as the other must not quietly half-succeed."""
    from bcir.abi.artifact_bundle import encode_bundle
    from bcir.asn1.artifact_bundle import native_to_per, per_to_native
    from bcir.tests.test_asn1_artifact_bundle import _three_bundle

    native = encode_bundle(_three_bundle())
    unaligned = native_to_per(native, aligned=False)
    try:
        crossed = per_to_native(unaligned, aligned=True)
    except Exception:
        return                                        # refused, which is the good outcome
    assert crossed != native, "the variants must not silently interwork"


def _int_range(low: int, high: int) -> Primitive:
    from bcir.asn1.constraints import ValueRange

    return Primitive(Universal.INTEGER, "INTEGER", ValueRange(low, high))


def test_annex_a3_matches_the_specification_octets_in_both_variants():
    """A.3: extension markers on the SET, the SEQUENCEs, the INTEGER and the SIZEs.

    83 and 65 octets. This is the vector that pins X.680 50.11: `initial NameString
    (SIZE(1))` is serially constrained, so the parent's extension marker is erased and it
    encodes with NO extension bit and NO length -- while its sibling `givenName`, the same
    base type left unconstrained, keeps both.
    """
    for variant, expected, octets in (
        (PerVariant.ALIGNED, _A3_ALIGNED, 83),
        (PerVariant.UNALIGNED, _A3_UNALIGNED, 65),
    ):
        got = _encode_named(_A3_MODULE, "PersonnelRecord", _A3_RECORD, variant)
        assert len(got) == octets, f"{variant} is {octets} octets, got {len(got)}"
        assert got == expected, "A.3 " + variant.value + ": " + got.hex() + " != " + expected.hex()


def test_annex_a4_matches_the_specification_octets_in_both_variants():
    """A.4: extension addition GROUPS (version brackets) and an extension marker pair.

    Both variants are 8 octets. Three separate rules land at once: a group in a SEQUENCE
    encodes as one open type holding a SEQUENCE of its members (19.9); a bracket in a
    CHOICE has no effect and its members are ordinary extension alternatives (23.8 NOTE);
    and `i`/`j`, written after the SECOND marker, are extension ROOT components (19.9
    NOTE 2), not additions.
    """
    for variant, expected in ((PerVariant.ALIGNED, _A4_ALIGNED),
                              (PerVariant.UNALIGNED, _A4_UNALIGNED)):
        got = _encode_named(_A4_MODULE, "Ax", _A4_RECORD, variant)
        assert len(got) == 8, f"A.4 is 8 octets, got {len(got)}"
        assert got == expected, "A.4 " + variant.value + ": " + got.hex() + " != " + expected.hex()


def test_annex_a3_and_a4_round_trip_in_both_variants():
    for module, name, value in ((_A3_MODULE, "PersonnelRecord", _A3_RECORD),
                                (_A4_MODULE, "Ax", _A4_RECORD)):
        lowered = compile_module(module, "<annex>")
        kind = lowered.module.types[name]
        for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
            data = encode_per(kind, value, variant=variant)
            assert decode_per(data, kind, variant=variant) == value, (name, variant)


def test_serial_application_erases_the_parent_extension_marker():
    """X.680 50.11, isolated from the Annex A.3 vector that exposed it."""
    from bcir.asn1.constraints import root_size_bounds

    lowered = compile_module(_A3_MODULE, "<annex>")
    name = lowered.module.types["Name"]
    given = next(c for c in name.components if c.name == "givenName")
    initial = next(c for c in name.components if c.name == "initial")
    assert root_size_bounds(given.type.constraint) == ((1, 64), True), (
        "the plain reference keeps NameString's extensible SIZE(1..64, ...)")
    assert root_size_bounds(initial.type.constraint) == ((1, 1), False), (
        "the serially constrained one is fixed at 1 and NOT extensible")


def test_extensible_integer_outside_the_root_switches_to_unconstrained():
    """13.1: one bit says which side of the extension root the value fell on."""
    module = "M DEFINITIONS ::= BEGIN T ::= SEQUENCE { v INTEGER (0..9999, ...) } END"
    kind = compile_module(module, "<p>").module.types["T"]
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        inside = encode_per(kind, {"v": 51}, variant=variant)
        outside = encode_per(kind, {"v": 100000}, variant=variant)
        assert inside[0] & 0x80 == 0, "a root value sets the extension bit to zero"
        assert outside[0] & 0x80 != 0, "a value past the root sets it to one"
        for data, value in ((inside, 51), (outside, 100000)):
            assert decode_per(data, kind, variant=variant) == {"v": value}


def test_extensible_size_outside_the_root_drops_the_size_constraint():
    """17.3/30.4: outside the root the length is encoded as if unconstrained."""
    module = "M DEFINITIONS ::= BEGIN T ::= SEQUENCE { v OCTET STRING (SIZE(2, ...)) } END"
    kind = compile_module(module, "<p>").module.types["T"]
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        for payload in (b"ab", b"", b"abcdefgh"):
            data = encode_per(kind, {"v": payload}, variant=variant)
            assert decode_per(data, kind, variant=variant) == {"v": payload}


def test_half_open_integer_range_encodes_semi_constrained():
    """13.2.3. This branch was unreachable while a scalar bound was compared against the
    UNBOUNDED sentinel TUPLE, which sent `INTEGER (0..MAX)` into int(None)."""
    from bcir.asn1.constraints import ValueRange

    kind = Sequence((Component("v", Primitive(
        Universal.INTEGER, "INTEGER", ValueRange(0, None))),), name="S")
    for variant in (PerVariant.ALIGNED, PerVariant.UNALIGNED):
        for value in (0, 5, 300, 70000):
            data = encode_per(kind, {"v": value}, variant=variant)
            assert decode_per(data, kind, variant=variant) == {"v": value}


def _encode_named(module: str, name: str, value, variant: PerVariant) -> bytes:
    lowered = compile_module(module, "<annex>")
    return encode_per(lowered.module.types[name], value,
                      variant=variant, rules=PerRules.CANONICAL)
