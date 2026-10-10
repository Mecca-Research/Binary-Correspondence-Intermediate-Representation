"""X.696 Octet Encoding Rules (roadmap phase D).

The centrepiece is `test_annex_a_...`: the standard's OWN worked example, transcribed
from X.696 Annex A.3.1, encoded byte-for-byte. That is the only kind of test that can
establish a wire format is right — a round-trip test passes just as happily when the
encoder and decoder share the same wrong assumption, which is exactly the failure mode a
from-memory implementation produces. Everything else here pins one clause each.
"""

from __future__ import annotations

from bcir.asn1.oer import (
    OerRules,
    decode_length,
    decode_oer,
    decode_tag,
    encode_length,
    encode_oer,
    encode_tag,
)
from bcir.asn1.schema import Choice, Component, Primitive, Sequence, SequenceOf, Set, SetOf
from bcir.asn1.tags import Asn1Error, Tag, TagClass, Universal

_VIS = Primitive(Universal.VISIBLE_STRING, "VisibleString")
_INT = Primitive(Universal.INTEGER, "INTEGER")
_BOOL = Primitive(Universal.BOOLEAN, "BOOLEAN")
_ENUM = Primitive(Universal.ENUMERATED, "ENUMERATED")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")
_NULL = Primitive(Universal.NULL, "NULL")
_OCTETS = Primitive(Universal.OCTET_STRING, "OCTET STRING")
_APP = TagClass.APPLICATION


# --- X.696 Annex A: the standard's own reference encoding --------------------------------


def _annex_a_types():
    """Annex A.1, transcribed. The tags live on the components because that is where this
    type model carries them; X.696 §8.4.2 makes tagging invisible to OER anyway, EXCEPT
    for the SET ordering of §18.2 — which is precisely what this fixture exercises."""
    name = Sequence(
        (Component("givenName", _VIS), Component("initial", _VIS), Component("familyName", _VIS)),
        name="Name",
    )
    child = Set(
        (Component("name", name, tag=1, tag_class=_APP), Component("dateOfBirth", _VIS, tag=0)),
        name="ChildInformation",
    )
    personnel = Set(
        (
            Component("name", name, tag=1, tag_class=_APP),
            Component("title", _VIS, tag=0),
            Component("number", _INT, tag=2, tag_class=_APP),
            Component("dateOfHire", _VIS, tag=1),
            Component("nameOfSpouse", name, tag=2),
            Component(
                "children", SequenceOf(child, "SEQUENCE OF ChildInformation"), tag=3, default=[]
            ),
        ),
        name="PersonnelRecord",
    )
    return personnel


#: Annex A.2, the John Smith record.
_ANNEX_A_VALUE = {
    "name": {"givenName": "John", "initial": "P", "familyName": "Smith"},
    "title": "Director",
    "number": 51,
    "dateOfHire": "19710917",
    "nameOfSpouse": {"givenName": "Mary", "initial": "T", "familyName": "Smith"},
    "children": [
        {
            "name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
            "dateOfBirth": "19571111",
        },
        {
            "name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
            "dateOfBirth": "19590717",
        },
    ],
}

#: Annex A.3.1 hexadecimal view, transcribed exactly. A.3 states the length is 95 octets
#: in both BASIC-OER and CANONICAL-OER, and that the two produce the same encoding here.
_ANNEX_A_OCTETS = bytes.fromhex(
    "".join(
        """
80044A6F 686E0150 05536D69 74680133 08446972 6563746F 72083139 37313039
3137044D 61727901 5405536D 69746801 02055261 6C706801 5405536D 69746808
31393537 31313131 05537573 616E0142 054A6F6E 65730831 39353930 373137
""".split()
    )
)


def test_annex_a_transcription_is_the_length_the_standard_states():
    """Guards the fixture itself: X.696 A.3 says 95 octets, so a typo in the transcription
    fails here rather than silently becoming the thing the encoder is measured against."""
    assert len(_ANNEX_A_OCTETS) == 95, len(_ANNEX_A_OCTETS)


def test_annex_a_personnel_record_encodes_byte_for_byte():
    """THE gate for this phase: the standard's own reference octets.

    A round trip would pass even if the length determinant, the SET ordering, the
    preamble bitmap and the quantity field were all wrong in the same way on both sides.
    This cannot.
    """
    assert encode_oer(_annex_a_types(), _ANNEX_A_VALUE) == _ANNEX_A_OCTETS


def test_annex_a_octets_decode_to_the_annex_a_value():
    assert decode_oer(_annex_a_types(), _ANNEX_A_OCTETS, rules=OerRules.CANONICAL) == _ANNEX_A_VALUE


# --- §8.6 the length determinant --------------------------------------------------------


def test_length_determinant_short_form_boundary():
    """§8.6.4: the short form covers 0..127 in one octet; 128 needs the long form."""
    assert encode_length(0) == b"\x00"
    assert encode_length(127) == b"\x7f"
    assert encode_length(128) == b"\x81\x80"  # §8.6.5, one subsequent octet
    assert encode_length(255) == b"\x81\xff"
    assert encode_length(256) == b"\x82\x01\x00"


def test_canonical_length_determinant_uses_the_fewest_octets():
    """§31.2: the long form only above 127, and then minimally encoded."""
    for value in (0, 1, 127, 128, 255, 256, 65535, 65536, 1 << 32):
        octets = encode_length(value)
        assert decode_length(octets, 0) == (value, len(octets))
        if value < 128:
            assert len(octets) == 1, value
        else:
            assert octets[1] != 0, f"leading zero octet for {value} (X.696 31.2 NOTE)"


def test_decoder_accepts_the_basic_oer_redundant_length_form():
    """§3.7.12 NOTE: BASIC-OER permits leading zero octets. "BASIC in" means accepting
    them; only the ENCODER is held to §31.2."""
    assert decode_length(b"\x82\x00\x7f", 0) == (127, 3)


def test_a_long_form_length_with_no_subsequent_octets_is_refused():
    try:
        decode_length(b"\x80", 0)
        raise AssertionError("accepted a zero-octet long form")
    except Asn1Error as exc:
        assert "8.6.5" in str(exc), exc


# --- §8.7 tags (CHOICE alternatives only) -----------------------------------------------


def test_tag_encoding_low_and_high_forms():
    """§8.7.2.2 puts a number below 63 in the first octet; §8.7.2.3 spills above that."""
    assert encode_tag(Tag(TagClass.CONTEXT, 0)) == b"\x80"
    assert encode_tag(Tag(TagClass.APPLICATION, 1)) == b"\x41"
    assert encode_tag(Tag(TagClass.UNIVERSAL, 62)) == b"\x3e"
    high = encode_tag(Tag(TagClass.CONTEXT, 63))
    assert high == b"\xbf\x3f", high.hex()
    for number in (0, 1, 62, 63, 127, 128, 16383, 16384):
        for cls in TagClass:
            octets = encode_tag(Tag(cls, number))
            tag, cursor = decode_tag(octets, 0)
            assert (tag.cls, tag.number, cursor) == (cls, number, len(octets))


# --- §9 / §31.3 BOOLEAN -----------------------------------------------------------------


def test_canonical_boolean_true_is_255_and_basic_accepts_any_nonzero():
    assert encode_oer(_BOOL, True) == b"\xff"  # §31.3
    assert encode_oer(_BOOL, False) == b"\x00"
    assert decode_oer(_BOOL, b"\x01", rules=OerRules.BASIC) is True  # §9
    try:
        decode_oer(_BOOL, b"\x01", rules=OerRules.CANONICAL)
        raise AssertionError("CANONICAL-OER accepted TRUE encoded as 1")
    except Asn1Error as exc:
        assert "31.3" in str(exc), exc


# --- §10 INTEGER ------------------------------------------------------------------------


def test_unconstrained_integer_is_a_length_then_a_minimal_signed_number():
    """§10.4 e) for a type with no OER-visible constraint, plus §31.4's minimality.

    -128 in one octet and 128 in two is the two's-complement asymmetry; an encoder that
    sized by magnitude alone gets exactly this case wrong.
    """
    assert encode_oer(_INT, 0) == b"\x01\x00"
    assert encode_oer(_INT, 51) == b"\x01\x33"  # as in Annex A
    assert encode_oer(_INT, 127) == b"\x01\x7f"
    assert encode_oer(_INT, -128) == b"\x01\x80"
    assert encode_oer(_INT, 128) == b"\x02\x00\x80"
    assert encode_oer(_INT, -129) == b"\x02\xff\x7f"
    for value in (0, 1, -1, 127, 128, -128, -129, 1 << 70, -(1 << 70)):
        assert decode_oer(_INT, encode_oer(_INT, value)) == value


# --- §11 ENUMERATED ---------------------------------------------------------------------


def test_enumerated_short_form_below_128_and_signed_long_form_above():
    """§11.3 / §11.4. The long form's subsequent octets are a SIGNED number — unlike a
    length determinant's, which is unsigned (§11.4 NOTE 1). That difference is why the
    two cannot share an encoder."""
    assert encode_oer(_ENUM, 0) == b"\x00"
    assert encode_oer(_ENUM, 127) == b"\x7f"
    assert encode_oer(_ENUM, 128) == b"\x82\x00\x80"  # 2 octets; signed, so 0x00
    assert encode_oer(_ENUM, 255) == b"\x82\x00\xff"
    assert encode_oer(_ENUM, -1) == b"\x81\xff"
    for value in (0, 1, 127, 128, 255, -1, -128, -129):
        assert decode_oer(_ENUM, encode_oer(_ENUM, value)) == value


# --- §15 / §14 / §27 --------------------------------------------------------------------


def test_null_encodes_to_nothing():
    """§15: "The encoding of the null value shall be empty." Not a zero octet."""
    from bcir.asn1.codec import NULL

    assert encode_oer(_NULL, NULL) == b""
    kind = Sequence((Component("n", _NULL), Component("i", _INT)), name="T")
    assert encode_oer(kind, {"n": NULL, "i": 1}) == b"\x01\x01"


def test_octet_string_and_utf8_string_are_length_prefixed_when_unconstrained():
    assert encode_oer(_OCTETS, b"\x01\x02") == b"\x02\x01\x02"  # §14.2
    assert encode_oer(_UTF8, "hi") == b"\x02hi"  # §27.3
    # UTF8String is NOT a known-multiplier type (§27.1), so a multi-octet character makes
    # the octet count differ from the character count -- the length is in OCTETS.
    assert encode_oer(_UTF8, "é") == b"\x02\xc3\xa9"
    assert decode_oer(_UTF8, b"\x02\xc3\xa9") == "é"


def test_object_identifier_is_a_length_then_the_ber_contents_octets():
    """§21: OER reuses X.690's OID contents octets verbatim behind a length determinant.

    This path had no coverage until an AlgorithmIdentifier needed it -- neither the
    Annex A record nor the BCIR-StreamPack module contains an OBJECT IDENTIFIER, so a
    fault here would have shipped green.
    """
    from bcir.asn1.codec import Oid

    kind = Primitive(Universal.OBJECT_IDENTIFIER, "OBJECT IDENTIFIER")
    # X.690 §8.19's own example: {2 999 3} packs the first two arcs as 40*2 + 999.
    assert encode_oer(kind, Oid((2, 999, 3))) == b"\x03\x88\x37\x03"
    rsa = Oid((1, 2, 840, 113549, 1, 1, 11))
    assert decode_oer(kind, encode_oer(kind, rsa)) == rsa


def test_an_open_type_is_a_length_then_the_contained_encoding():
    """§30. The contained TYPE is unknown by definition, so the octets pass through
    unchanged -- re-encoding them would need the type an open type refuses to fix."""
    from bcir.asn1.schema import OpenType

    kind = OpenType()
    assert encode_oer(kind, bytes.fromhex("0500")) == b"\x02\x05\x00"
    assert decode_oer(kind, b"\x02\x05\x00") == bytes.fromhex("0500")


# --- §16 SEQUENCE preamble --------------------------------------------------------------


def test_a_sequence_with_no_optional_components_has_an_empty_preamble():
    """§16.2.4 NOTE: no extension marker and no OPTIONAL/DEFAULT means no preamble at all
    — not a zero octet. An encoder that always emitted one would add a spurious octet."""
    kind = Sequence((Component("a", _INT), Component("b", _INT)), name="T")
    assert encode_oer(kind, {"a": 1, "b": 2}) == b"\x01\x01\x01\x02"


def test_the_root_component_presence_bitmap_is_one_bit_per_optional_component():
    """§16.2.3, most significant bit first, padded to a whole octet by §16.2.4."""
    kind = Sequence(
        (Component("a", _INT, optional=True), Component("b", _INT, optional=True)), name="T"
    )
    assert encode_oer(kind, {"a": 1, "b": 2})[0] == 0b11000000
    assert encode_oer(kind, {"a": 1})[0] == 0b10000000
    assert encode_oer(kind, {"b": 2})[0] == 0b01000000
    assert encode_oer(kind, {}) == b"\x00"
    for value in ({"a": 1, "b": 2}, {"a": 1}, {"b": 2}, {}):
        assert decode_oer(kind, encode_oer(kind, value)) == value


def test_a_default_component_equal_to_its_default_is_encoded_absent():
    """§31.9 — the OER counterpart of X.690 §11.5."""
    kind = Sequence((Component("a", _INT), Component("b", _INT, default=7)), name="T")
    assert encode_oer(kind, {"a": 1, "b": 7}) == encode_oer(kind, {"a": 1})
    assert encode_oer(kind, {"a": 1, "b": 8}) != encode_oer(kind, {"a": 1})
    # ...and absence means the default on the way back.
    assert decode_oer(kind, encode_oer(kind, {"a": 1})) == {"a": 1, "b": 7}


# --- §16.2.2 / §16.4 / §16.5 extensibility ---------------------------------------------
#
# Every vector below except the version bracket's was cross-checked against an independent
# implementation (asn1tools 0.169.0's OER codec) when this rail was fixed; none comes from a
# round trip, because a round trip is what let four BCIR OER rails agree on a missing bit.


def _extensible_types():
    from bcir.frontends.asn1 import compile_module

    source = """T DEFINITIONS AUTOMATIC TAGS ::= BEGIN
      S  ::= SEQUENCE { a INTEGER (0..255) OPTIONAL, b INTEGER (0..255), ... }
      S2 ::= SEQUENCE { a INTEGER (0..255) OPTIONAL, b INTEGER (0..255) }
      S3 ::= SEQUENCE { b INTEGER (0..255), ... }
      Y  ::= SEQUENCE { a INTEGER (0..255), ..., c INTEGER (0..255) OPTIONAL,
                        d INTEGER (0..255) OPTIONAL }
      Y0 ::= SEQUENCE { a INTEGER (0..255), ... }
      W  ::= SEQUENCE { a INTEGER (0..255), ..., [[ d INTEGER (0..255), e BOOLEAN OPTIONAL ]] }
      C  ::= CHOICE { x INTEGER (0..255), ..., y INTEGER (0..255) }
      C0 ::= CHOICE { x INTEGER (0..255), ... }
      D  ::= SEQUENCE { a INTEGER (0..255), ..., c INTEGER (0..255) OPTIONAL,
                        d INTEGER (0..255) DEFAULT 3 }
      V  ::= SEQUENCE { a INTEGER (0..255), ..., b INTEGER (0..255) DEFAULT 5,
                        [[ c BOOLEAN OPTIONAL, d INTEGER (0..255) DEFAULT 3 ]],
                        f INTEGER (0..255) DEFAULT 9 }
      VS ::= SET { a INTEGER (0..255), ...,
                   [[ c BOOLEAN OPTIONAL, d INTEGER (0..255) DEFAULT 3 ]],
                   f INTEGER (0..255) DEFAULT 9 }
      WD ::= SEQUENCE { a INTEGER (0..255), ...,
                        [[ g INTEGER (0..255), d INTEGER (0..255) DEFAULT 3 ]] }
    END"""
    return compile_module(source, "t.asn1").module.types


def test_an_extensible_sequence_leads_its_preamble_with_the_extension_bit():
    """§16.2.2. The rail used to write the presence bitmap alone, so `S` with `a` present
    was `80 05 06`: to a conforming decoder, "extension additions present" followed by octets
    that are no addition bitmap. The non-extensible twin `S2` is the control -- same
    components, no marker, and its spelling must not move."""
    t = _extensible_types()
    assert encode_oer(t["S"], {"a": 5, "b": 6}) == bytes.fromhex("400506")
    assert encode_oer(t["S"], {"b": 6}) == bytes.fromhex("0006")
    assert encode_oer(t["S2"], {"a": 5, "b": 6}) == bytes.fromhex("800506")
    # An extensible type with no OPTIONAL component still has a preamble: the bit alone.
    assert encode_oer(t["S3"], {"b": 6}) == bytes.fromhex("0006")
    for name, value in (("S", {"a": 5, "b": 6}), ("S", {"b": 6}), ("S3", {"b": 6})):
        assert decode_oer(t[name], encode_oer(t[name], value)) == value
    # The old spelling now reads as "additions present", and no addition bitmap follows.
    for old in ("800506", "06"):
        try:
            decode_oer(t["S" if old != "06" else "S3"], bytes.fromhex(old))
            raise AssertionError(f"{old}: the pre-fix spelling decoded")
        except Asn1Error:
            pass


def test_extension_additions_follow_the_root_behind_a_bitmap_as_open_types():
    """§16.4: a length-prefixed BIT STRING, its first octet counting the unused bits, one bit
    per addition IN THE TYPE; §16.5: each present addition as an open type."""
    t = _extensible_types()
    vectors = {
        "80010206800102": {"a": 1, "c": 2},
        "80010206400103": {"a": 1, "d": 3},
        "80010206c001020103": {"a": 1, "c": 2, "d": 3},
        "0001": {"a": 1},
    }
    for octets, value in vectors.items():
        assert encode_oer(t["Y"], value).hex() == octets, value
        assert decode_oer(t["Y"], bytes.fromhex(octets), rules=OerRules.CANONICAL) == value


def test_a_version_bracket_is_one_addition_encoded_as_a_sequence():
    """A `[[ ]]` group is ONE addition: one bitmap bit and one open type holding the group's
    members as a SEQUENCE (with that SEQUENCE's own preamble for `e`) -- §16.5's reading, and
    the one X.691 §19.9 states for PER, where BCIR's PER rail and asn1tools' agree. asn1tools'
    OER codec flattens a group into separate additions instead, which its own PER codec does
    not; this rail keeps the two rules' structure parallel."""
    t = _extensible_types()
    assert encode_oer(t["W"], {"a": 1, "d": 3}) == bytes.fromhex("8001020780020003")
    assert encode_oer(t["W"], {"a": 1, "d": 3, "e": True}) == bytes.fromhex("8001020780038003ff")
    for value in ({"a": 1, "d": 3}, {"a": 1, "d": 3, "e": True}):
        assert decode_oer(t["W"], encode_oer(t["W"], value)) == value


def _typed_decode(kind, octets: bytes, strictness):
    """The typed BER/DER rail's decode of one complete encoding (as `Module.decode`)."""
    from bcir.asn1.codec import Strictness
    from bcir.asn1.der import require_der
    from bcir.asn1.tlv import decode_one

    tlv = decode_one(octets)
    if strictness is Strictness.DER:
        require_der(tlv)
    return kind.decode(tlv, strictness=strictness)


def _bracket_rails():
    """Every rule that reads `V`, as (name, encode, decode) over one abstract value: OER under
    both decoders, PER in both variants under both rule sets, JER, XER, and the typed BER/DER
    rail (DER out, read back as BER and as DER)."""
    from bcir.asn1.codec import Strictness
    from bcir.asn1.jer import decode_jer, encode_jer
    from bcir.asn1.per import PerRules, PerVariant, decode_per, encode_per
    from bcir.asn1.tlv import encode_tlv
    from bcir.asn1.xer import decode_xer, encode_xer

    rails = [
        (f"OER/{r.name}", encode_oer, lambda k, o, r=r: decode_oer(k, o, rules=r)) for r in OerRules
    ]
    for strictness in (Strictness.BER, Strictness.DER):
        rails.append(
            (
                f"typed/{strictness.name}",
                lambda k, v: encode_tlv(k.encode(v)),
                lambda k, o, s=strictness: _typed_decode(k, o, s),
            )
        )
    for a in PerVariant:
        for r in PerRules:
            rails.append(
                (
                    f"PER/{a.name}/{r.name}",
                    lambda k, v, a=a, r=r: encode_per(k, v, variant=a, rules=r),
                    lambda k, o, a=a, r=r: decode_per(o, k, variant=a, rules=r),
                )
            )
    rails.append(("JER", encode_jer, lambda k, o: decode_jer(o, k)))
    rails.append(("XER", encode_xer, lambda k, o: decode_xer(o, k)))
    return rails


def test_an_absent_bracket_leaves_its_default_members_at_their_defaults_on_every_rule():
    """X.680 §25.12: a DEFAULT component the encoding leaves out has its default -- and a
    version bracket's members are components of the SEQUENCE like any other, so a bracket left
    out whole leaves its DEFAULT members at theirs. PER and OER filled the defaults of single
    additions only (and PER, with the extension bit clear, took a path of its own), so
    `{"a": 1}` decoded to `{"a": 1, "b": 5, "f": 9}` on both while JER and XER, which flatten
    a bracket, returned `d` too: one abstract value, two decodings. Every rule now reads the
    one predicate, `schema.addition_defaults`. The typed BER/DER rail is a rail here too: it
    encoded the bracket as a nested SEQUENCE, refused `d` by name and decoded without it."""
    from bcir.asn1.per import PerVariant, encode_per

    kind = _extensible_types()["V"]
    full = {"a": 1, "b": 5, "d": 3, "f": 9}
    cases = [
        ({"a": 1}, full),
        ({"a": 1, "d": 3}, full),  # d at its default: the bracket is left out
        ({"a": 1, "b": 6}, dict(full, b=6)),  # an addition present, the bracket not
        ({"a": 1, "c": True}, dict(full, c=True)),  # the bracket present, d inside it absent
    ]
    # The paths the law lives on are the ones these values take: the bracket omitted whole,
    # with the extension bit clear and with it set.
    assert encode_oer(kind, {"a": 1, "d": 3}) == encode_oer(kind, {"a": 1}) == b"\x00\x01"
    assert not encode_per(kind, {"a": 1}, variant=PerVariant.UNALIGNED)[0] & 0x80
    assert encode_per(kind, {"a": 1, "b": 6}, variant=PerVariant.UNALIGNED)[0] & 0x80
    for name, encode, decode in _bracket_rails():
        for value, expected in cases:
            assert decode(kind, encode(kind, value)) == expected, (name, value)


#: DER of version-bracket values from two independent codecs, asn1tools 0.169.0 and pycrate
#: 0.8.1, which agree on every one: a bracket's members are components of the enclosing SEQUENCE
#: or SET, in the definition's order, tagged in the enclosing list's AUTOMATIC sequence (`V`: a
#: [0], b [1], c [2], d [3], f [4]), and DER leaves a member at its DEFAULT out (X.690 §11.5).
#: (type, value encoded, the octets, the value decoded with the DEFAULTs filled in)
_BRACKET_DER = (
    ("V", {"a": 1}, "3003800101", {"a": 1, "b": 5, "d": 3, "f": 9}),
    ("V", {"a": 1, "d": 3}, "3003800101", {"a": 1, "b": 5, "d": 3, "f": 9}),
    (
        "V",
        {"a": 1, "c": True, "d": 4},
        "30098001018201ff830104",
        {"a": 1, "b": 5, "c": True, "d": 4, "f": 9},
    ),
    ("V", {"a": 1, "b": 6, "c": False, "d": 3, "f": 10}, "300c80010181010682010084010a", None),
    ("W", {"a": 1}, "3003800101", None),
    ("W", {"a": 1, "d": 2}, "3006800101810102", None),
    ("W", {"a": 1, "d": 2, "e": False}, "3009800101810102820100", None),
    ("VS", {"a": 1}, "3103800101", {"a": 1, "d": 3, "f": 9}),
    ("VS", {"a": 1, "c": True, "d": 4, "f": 10}, "310c8001018101ff82010483010a", None),
    ("VS", {"a": 7, "d": 3, "f": 9}, "3103800107", None),
)


def test_a_version_bracket_is_flat_on_ber_and_der():
    """X.690 knows no version bracket: §8.9.2 encodes one value for each component type the
    SEQUENCE's definition lists, and a bracket lists its members as component types of the
    enclosing definition -- only PER and OER wrap a bracket into one open type (X.691 §19.9,
    X.696 §16.5). The typed rail encoded `[[ c, d ]]` as a nested SEQUENCE: it refused
    `{"a": 1, "d": 3}` (unknown component `d`), and decoded without the bracket's DEFAULT. The
    octets are checked against two independent codecs, not round-tripped, and include a SET,
    whose DER orders the members among the other components by tag (§8.11, §10.3). A member
    at its DEFAULT is a second spelling under DER (§11.5) and a sender's option under BER."""
    from bcir.asn1.codec import Strictness
    from bcir.asn1.tlv import encode_tlv

    t = _extensible_types()
    for name, value, octets, decoded in _BRACKET_DER:
        assert encode_tlv(t[name].encode(value)).hex() == octets, (name, value)
        for strictness in (Strictness.BER, Strictness.DER):
            got = _typed_decode(t[name], bytes.fromhex(octets), strictness)
            assert got == (value if decoded is None else decoded), (name, octets, got)
    explicit = bytes.fromhex("3006800101830103")  # `V` with d = 3 present, its DEFAULT
    assert _typed_decode(t["V"], explicit, Strictness.BER) == {"a": 1, "b": 5, "d": 3, "f": 9}
    try:
        _typed_decode(t["V"], explicit, Strictness.DER)
        raise AssertionError("DER read a bracket member equal to its DEFAULT")
    except Asn1Error as error:
        assert "DEFAULT" in str(error), error


def test_the_annex_a4_record_is_flat_on_der_in_the_definitions_order():
    """X.691 Annex A.4's type on the typed rail: its bracket holds a mandatory member (`g`), its
    CHOICE holds a bracket (presentational, X.691 §23.8 NOTE), and two root components follow
    the extension-marker pair. The record's DER is asn1tools' and pycrate's; with the bracket
    absent, both read the value, as every rule here does. Where the two codecs part is the third
    value: pycrate keeps the definition's order (§8.9.2, `g` before `i`), asn1tools emits the
    root first (`i` before `g`); this rail keeps the definition's order, with pycrate."""
    from bcir.asn1.codec import Strictness
    from bcir.frontends.asn1 import compile_module
    from bcir.tests.test_asn1_per import _A4_MODULE, _A4_RECORD

    module = compile_module(_A4_MODULE, "<annex>").module
    vectors = (
        (_A4_RECORD, "3014800200fd8101ffa2038101ff83033132338401ff"),
        (
            {"a": 253, "b": True, "c": ("d", 7), "i": "hi", "j": "pq"},
            "3016800200fd8101ffa20380010785040068006986027071",
        ),
        (
            {"a": 250, "b": False, "c": ("f", "x"), "g": "999", "i": "z"},
            "3015800200fa810100a20382017883033939398502007a",
        ),
    )
    for value, octets in vectors:
        assert module.encode("Ax", value).hex() == octets, value
        assert module.decode("Ax", bytes.fromhex(octets), strictness=Strictness.DER) == value


def test_a_bracket_is_absent_whole_or_present_with_its_mandatory_members_on_every_rule():
    """The lowering makes a bracket OPTIONAL as a whole, and X.691 §19.9 encodes a present one
    as a SEQUENCE of its members: a value carries a bracket entirely or not at all, and a
    carried bracket carries its mandatory members. JER and XER, which flatten a bracket, made a
    member mandatory outright, so `W` without its bracket -- a value PER, OER, asn1tools and
    pycrate all read -- was refused; and a carried bracket without its mandatory `d` must be
    refused by every rule, on encode and on decode. `WD`'s DEFAULT member does not carry its
    bracket: CXER writes it out for a value without the bracket, and that must read back. The
    bracket's own pseudo-name (`[[2]]`) is no component on any rule: PER and OER used to let
    it through, ignored."""
    from bcir.asn1.codec import Strictness
    from bcir.asn1.jer import decode_jer, encode_jer
    from bcir.asn1.tlv import encode_tlv
    from bcir.asn1.xer import decode_xer, encode_xer

    t = _extensible_types()
    w = t["W"]
    pseudo = next(comp.name for comp in w.components if comp.group is not None)
    for name, encode, decode in _bracket_rails():
        for kind, value in ((w, {"a": 1}), (w, {"a": 1, "d": 2}), (t["WD"], {"a": 1})):
            got = decode(kind, encode(kind, value))
            expected = dict(value, d=3) if kind is t["WD"] else value
            assert got == expected, (name, value, got)
        for bad in ({"a": 1, "e": True}, {"a": 1, pseudo: {"d": 2}}):
            try:
                encode(w, bad)
                raise AssertionError(f"{name}: {bad} was encoded")
            except Asn1Error as error:
                assert any(w in str(error) for w in ("mandatory", "missing", "unknown")), (
                    name,
                    error,
                )
    # The decoders: each rule's spelling of `e` without `d`.
    full = {"a": 1, "d": 2, "e": True}
    der = encode_tlv(w.encode(full)).replace(bytes.fromhex("810102"), b"")
    der = bytes([der[0], der[1] - 3]) + der[2:]
    jer = encode_jer(w, full).replace(b'"d":2,', b"")
    xer = encode_xer(w, full).replace(b"<d>2</d>", b"")
    refusals = (
        ("BER", lambda: _typed_decode(w, der, Strictness.BER)),
        ("DER", lambda: _typed_decode(w, der, Strictness.DER)),
        ("JER", lambda: decode_jer(jer, w)),
        ("XER", lambda: decode_xer(xer, w)),
    )
    for name, decode in refusals:
        try:
            got = decode()
            raise AssertionError(f"{name} read a bracket without its mandatory member: {got}")
        except Asn1Error as error:
            assert "version bracket [[d, e]]" in str(error), (name, error)


def test_a_jer_instruction_on_a_bracket_member_reaches_it_through_the_flat_view():
    """JER files an encoding instruction under the object it is assigned to, and assigns a NAME
    to a component (X.697 §9.9). A flat view that listed `W`'s mandatory `d` as a fresh OPTIONAL
    copy on every call looked the NAME up under another object and dropped it: `{"d":2}` went out
    where `{"dee":2}` was due, and the decoder agreed, so the round trip passed. The flat view is
    made once per bracket, and a copy files under the member it was made from (`origin`), so the
    member is reached whether it is named through the schema or through the flat view."""
    from bcir.asn1.jer import JerInstructions, Name, decode_jer, encode_jer
    from bcir.asn1.schema import flat_components

    w = _extensible_types()["W"]
    d, e = next(comp.group for comp in w.components if comp.group is not None)
    flat = flat_components(w.components)
    assert all(x is y for x, y in zip(flat, flat_components(w.components)))
    flat_d = next(comp for comp in flat if comp.name == "d")
    assert flat_d is not d and flat_d.origin is d and flat_d.optional and not d.optional
    value = {"a": 1, "d": 2, "e": True}
    for target in (d, flat_d):
        instructions = JerInstructions().assign(target, Name("dee")).assign(e, Name("ee"))
        octets = encode_jer(w, value, instructions=instructions)
        assert octets == b'{"a":1,"dee":2,"ee":true}', (target, octets)
        assert decode_jer(octets, w, instructions=instructions) == value


_FLAT_RULES_MODULE = """T DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  R  ::= SEQUENCE { a INTEGER (0..255), ..., [[ c INTEGER (0..255) OPTIONAL ]], ...,
                    z INTEGER (0..255) }
  VX ::= SET { a [5] INTEGER (0..255), ..., [[ c [1] BOOLEAN OPTIONAL ]] }
  PX ::= SET { a [5] INTEGER (0..255), ..., c [1] BOOLEAN OPTIONAL }
  AN ::= SEQUENCE { a INTEGER (0..255), ..., [[ n NULL, b BOOLEAN OPTIONAL ]] }
  PN ::= SEQUENCE { a INTEGER (0..255), ..., n NULL }
END"""


def test_a_bracket_member_is_an_extension_addition_on_jer_array_and_canonical_xer():
    """X.680 §25.1 lists a version bracket among the ExtensionAdditions, so its members are
    extension additions; the flat view kept each member's own `extension`, False inside a
    bracket. JER's ARRAY lists the root, then the additions (X.697 §27.2.1), and put `V`'s
    bracket members among the root (`[1,true,4,2,6]`) and `R`'s root `z`, after the marker pair,
    behind its bracket; canonical XER sorts a SET's root by tag and appends the additions in the
    definition's order (X.693 §9.6.2), and sorted `VX`'s bracket member into the root, so `VX`
    and `PX` -- one addition in a bracket, the same one without -- went out in two orders; and
    X.697 §14.2, which forbids a NULL as an extension component of an ARRAY sequence, let `AN`'s
    bracket member through where it refuses `PN`'s plain addition."""
    from bcir.asn1.jer import Array, JerInstructions, decode_jer, encode_jer
    from bcir.asn1.xer import decode_xer, encode_xer
    from bcir.frontends.asn1 import compile_module

    t = dict(_extensible_types(), **compile_module(_FLAT_RULES_MODULE, "t.asn1").module.types)
    rows = (
        ("V", {"a": 1, "b": 2, "c": True, "d": 4, "f": 6}, b"[1,2,true,4,6]", None),
        ("V", {"a": 1, "b": 2}, b"[1,2]", {"a": 1, "b": 2, "d": 3, "f": 9}),
        ("R", {"a": 1, "c": 3, "z": 26}, b"[1,26,3]", None),
    )
    for name, value, octets, decoded in rows:
        instructions = JerInstructions().assign(t[name], Array())
        assert encode_jer(t[name], value, instructions=instructions) == octets, (name, value)
        got = decode_jer(octets, t[name], instructions=instructions)
        assert got == (value if decoded is None else decoded), (name, octets, got)
    value = {"a": 1, "c": True}
    for name in ("VX", "PX"):
        assert encode_xer(t[name], value) == b"<SET><a>1</a><c><true/></c></SET>", name
        assert decode_xer(encode_xer(t[name], value), t[name]) == value, name
    for name in ("AN", "PN"):
        try:
            JerInstructions().assign(t[name], Array())
            raise AssertionError(f"{name}: ARRAY was assigned over a NULL extension component")
        except Asn1Error as error:
            assert "14.2" in str(error) and "'n' is a NULL" in str(error), (name, error)


_UNWRAPPED_MODULE = """T DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  U  ::= CHOICE { first SEQUENCE { x INTEGER (0..255), ..., [[ y INTEGER (0..255) ]] },
                  second SEQUENCE { x INTEGER (0..255), z BOOLEAN OPTIONAL } }
  U2 ::= CHOICE { first SEQUENCE { x INTEGER (0..255), ..., [[ y INTEGER (0..255) ]] },
                  second SEQUENCE { w INTEGER (0..255) } }
END"""


def test_a_bracket_member_separates_no_unwrapped_alternatives():
    """X.697 §19.2.3 tells the object-producing alternatives of an UNWRAPPED choice apart by a
    mandatory member name. A bracket's mandatory member is absent with its bracket, so it tells
    nothing apart: `{"x":1}` is `U`'s `first` with its bracket absent as much as its `second`
    without `z`, and the decoder reports the ambiguity. While `y` counted as mandatory, the
    decoder read that object as `second` without a word, and refused `U2`'s `first` without its
    bracket as matching nothing."""
    from bcir.asn1.jer import JerInstructions, Unwrapped, decode_jer
    from bcir.frontends.asn1 import compile_module

    t = compile_module(_UNWRAPPED_MODULE, "t.asn1").module.types
    rows = (
        ("U2", b'{"x":1}', ("first", {"x": 1})),
        ("U2", b'{"x":1,"y":2}', ("first", {"x": 1, "y": 2})),
        ("U2", b'{"w":3}', ("second", {"w": 3})),
    )
    for name, text, value in rows:
        instructions = JerInstructions().assign(t[name], Unwrapped())
        assert decode_jer(text, t[name], instructions=instructions) == value, (name, text)
    unwrapped = JerInstructions().assign(t["U"], Unwrapped())
    try:
        got = decode_jer(b'{"x":1}', t["U"], instructions=unwrapped)
        raise AssertionError(f"U: an object both alternatives read was decoded as {got}")
    except Asn1Error as error:
        assert "19.2.3" in str(error) and "['first', 'second']" in str(error), error


_MANDATORY_ADDITIONS = """T DEFINITIONS AUTOMATIC TAGS ::= BEGIN
  N  ::= SEQUENCE { a INTEGER (0..255), ..., b INTEGER (0..255) }
  NC ::= SEQUENCE { a INTEGER (0..255), ..., b INTEGER (0..255), c INTEGER (0..255) OPTIONAL }
  NS ::= SET { a INTEGER (0..255), ..., b INTEGER (0..255) }
END"""

#: Values without the mandatory addition `b`, as two independent codecs encode them. pycrate
#: 0.8.1 writes every row; asn1tools 0.169.0 writes the rows without `c` and, given `c`, drops it
#: without a word -- an addition after an absent mandatory one -- which this rail does not copy.
#: (type, value, rail, octets)
_WITHOUT_B = (
    ("N", {"a": 1}, "typed/DER", "3003800101"),
    ("NS", {"a": 1}, "typed/DER", "3103800101"),
    ("NC", {"a": 1, "c": 3}, "typed/DER", "3006800101820103"),
    ("N", {"a": 1}, "PER/UNALIGNED/BASIC", "0080"),
    ("NS", {"a": 1}, "PER/UNALIGNED/BASIC", "0080"),
    ("NC", {"a": 1, "c": 3}, "PER/UNALIGNED/BASIC", "80814040c0"),
    ("N", {"a": 1}, "PER/ALIGNED/BASIC", "0001"),
    ("NC", {"a": 1, "c": 3}, "PER/ALIGNED/BASIC", "800102800103"),
    ("N", {"a": 1}, "OER/BASIC", "0001"),
    ("NS", {"a": 1}, "OER/BASIC", "0001"),
    ("NC", {"a": 1, "c": 3}, "OER/BASIC", "80010206400103"),
)


def test_a_value_without_a_mandatory_addition_is_a_value_on_every_rule():
    """A value of an extensible type without one of its extension additions is a value of an
    earlier version -- mandatory or not, the addition was not there yet -- and X.691 §19.8 and
    X.696 §16.4 carry its absence, one presence bit per addition (LangRef §17.3). PER and OER
    read and wrote such values; the typed BER/DER rail, JER and XER refused them both ways
    ("component 'b' is mandatory"), so a relay that decoded `N` from a PER peer could not
    re-encode it under DER. Every rule now carries `N` without `b`, the SET `NS` without it,
    and `NC` with the later `c` but without `b`: each addition is judged on its own, as its
    presence bit is. The octets are two independent codecs', not a round trip, and so are the
    text spellings the decoders read. A root component stays mandatory on every rule."""
    from bcir.asn1.jer import decode_jer
    from bcir.asn1.xer import decode_xer
    from bcir.frontends.asn1 import compile_module

    t = compile_module(_MANDATORY_ADDITIONS, "t.asn1").module.types
    rails = {name: (encode, decode) for name, encode, decode in _bracket_rails()}
    values = (("N", {"a": 1}), ("N", {"a": 1, "b": 2}), ("NC", {"a": 1, "c": 3}), ("NS", {"a": 1}))
    for name, (encode, decode) in rails.items():
        for type_name, value in values:
            got = decode(t[type_name], encode(t[type_name], value))
            assert got == value, (name, type_name, value, got)
        try:
            encode(t["N"], {"b": 2})
            raise AssertionError(f"{name}: N was encoded without its root component a")
        except Asn1Error as error:
            assert "'a'" in str(error), (name, error)
    for type_name, value, rail, octets in _WITHOUT_B:
        encode, decode = rails[rail]
        assert encode(t[type_name], value).hex() == octets, (rail, type_name, value)
        assert decode(t[type_name], bytes.fromhex(octets)) == value, (rail, type_name, octets)
    # pycrate's JER and asn1tools' XER spellings of `N` without `b`.
    assert decode_jer('{\n "a": 1\n}', t["N"]) == {"a": 1}
    assert decode_jer('{\n "a": 1,\n "c": 3\n}', t["NC"]) == {"a": 1, "c": 3}
    assert decode_xer(b"<N><a>1</a></N>", t["N"], name="N") == {"a": 1}


def test_jer_carries_a_value_without_a_mandatory_addition_in_both_forms():
    """X.697 §27.2.1's ARRAY writes an absent addition as `null`, and the canonical profile drops
    the trailing ones (§27.2.2): `N` without `b` is `[1]`, `NC` without `b` is `[1,null,3]`. And
    the flat view's OPTIONAL stand-in for `b` files a JER instruction under `b` itself
    (`Component.origin`), so a NAME assigned to the component the schema holds still renames
    the member."""
    from bcir.asn1.jer import Array, JerInstructions, Name, decode_jer, encode_jer
    from bcir.asn1.schema import flat_components
    from bcir.frontends.asn1 import compile_module

    t = compile_module(_MANDATORY_ADDITIONS, "t.asn1").module.types
    rows = (
        ("N", {"a": 1}, b"[1]", (b"[1]", b"[1,null]")),
        ("N", {"a": 1, "b": 2}, b"[1,2]", (b"[1,2]",)),
        ("NC", {"a": 1, "c": 3}, b"[1,null,3]", (b"[1,null,3]",)),
    )
    for name, value, octets, spellings in rows:
        array = JerInstructions().assign(t[name], Array())
        assert encode_jer(t[name], value, instructions=array) == octets, (name, value)
        for text in spellings:
            assert decode_jer(text, t[name], instructions=array) == value, (name, text)
    b = t["N"].components[1]
    stand_in = flat_components(t["N"].components)[1]
    assert stand_in is not b and stand_in.origin is b and stand_in.optional and not b.optional
    named = JerInstructions().assign(b, Name("bee"))
    assert encode_jer(t["N"], {"a": 1, "b": 2}, instructions=named) == b'{"a":1,"bee":2}'
    assert decode_jer(b'{"a":1,"bee":2}', t["N"], instructions=named) == {"a": 1, "b": 2}
    assert encode_jer(t["N"], {"a": 1}, instructions=named) == b'{"a":1}'


def test_canonical_oer_refuses_a_present_bracket_that_carries_nothing():
    """§16.5 and §31.9 for a version bracket: the encoder leaves out a bracket whose members are
    all absent or at their DEFAULT (`_supplied`), so a peer that sends one -- the bitmap naming
    it, its open type holding a SEQUENCE with nothing in it -- spells the value without it a
    second way. The single-addition check (§31.9 for `D` above) never looked at a bracket."""
    kind = _extensible_types()["V"]
    full = {"a": 1, "b": 5, "d": 3, "f": 9}
    empty = "80010205400100"  # the bitmap names [[c, d]]; its open type: c and d absent
    assert decode_oer(kind, bytes.fromhex(empty), rules=OerRules.BASIC) == full
    assert "version bracket" in _refused_under(kind, empty, OerRules.CANONICAL)
    # Both halves: the value's canonical spelling, and a bracket that carries c, still read.
    assert decode_oer(kind, encode_oer(kind, {"a": 1}), rules=OerRules.CANONICAL) == full
    with_c = encode_oer(kind, {"a": 1, "c": True})
    assert with_c.hex() == "80010205400280ff"
    assert decode_oer(kind, with_c, rules=OerRules.CANONICAL) == dict(full, c=True)


def test_canonical_oer_spells_every_length_and_number_once():
    """§31.2 (a length determinant short below 128, else in the fewest octets), §31.4 (a
    variable-size number in the fewest octets), §11.3 (an ENUMERATED 0..127 in the short
    form), §8.7.2 (a tag in its shortest form) and §31.8 (a SET OF in its encodings' order).
    Each second spelling below is the canonical one padded at one of those sites -- among
    them the extension bitmap's and an addition's open-type lengths. BASIC reads each as the
    same value; CANONICAL accepted every one, so a peer chose the digest by choosing the
    spelling, and now refuses each. Both halves: every canonical spelling still decodes under
    CANONICAL, the forms a careless check would take for padding among them (the `00` that
    keeps 128 positive, the `ff` that keeps -129 negative)."""
    from bcir.frontends.asn1 import compile_module

    types = compile_module(
        """T DEFINITIONS AUTOMATIC TAGS ::= BEGIN
      L  ::= OCTET STRING
      I  ::= INTEGER
      U  ::= INTEGER (0..MAX)
      E  ::= ENUMERATED { a(1), big(200), neg(-5) }
      Q  ::= SEQUENCE OF BOOLEAN
      SO ::= SET OF OCTET STRING
      CH ::= CHOICE { x [5] BOOLEAN, y [70] BOOLEAN }
    END""",
        "t.asn1",
    ).module.types
    ext = _extensible_types()
    cases = [  # (type, value, canonical spelling, second spellings)
        ("L", b"\x01\x02", "020102", ["81020102", "8200020102"]),
        ("L", b"\xaa" * 200, "81c8" + "aa" * 200, ["8200c8" + "aa" * 200]),
        ("I", 5, "0105", ["020005", "810105"]),
        ("I", -1, "01ff", ["02ffff"]),
        ("I", 128, "020080", ["03000080"]),
        ("I", -129, "02ff7f", ["03ffff7f"]),
        ("U", 128, "0180", ["020080"]),
        ("U", 0, "0100", ["020000"]),
        ("E", 1, "01", ["8101"]),
        ("E", 200, "8200c8", ["830000c8"]),
        ("E", -5, "81fb", ["82fffb"]),
        ("Q", [True], "0101ff", ["020001ff", "810101ff"]),
        ("SO", [b"\x01", b"\x02"], "010201010102", ["010201020101"]),
        ("CH", ("x", True), "85ff", ["bf05ff"]),
        ("CH", ("y", True), "bf46ff", ["bf8046ff"]),
        ("Y", {"a": 1, "c": 2}, "80010206800102", ["8001810206800102", "8001020680810102"]),
        ("C", ("y", 2), "810102", ["81810102"]),
    ]

    def same(name, got, value):  # a SET OF value has no order
        return sorted(got) == sorted(value) if name == "SO" else got == value

    for name, value, canonical, seconds in cases:
        kind = types[name] if name in types else ext[name]
        assert encode_oer(kind, value).hex() == canonical, (name, value)
        assert same(
            name, decode_oer(kind, bytes.fromhex(canonical), rules=OerRules.CANONICAL), value
        )
        for second in seconds:
            got = decode_oer(kind, bytes.fromhex(second), rules=OerRules.BASIC)
            assert same(name, got, value), (name, second, got)
            assert "CANONICAL-OER" in _refused_under(kind, second, OerRules.CANONICAL), second


def test_an_older_decoder_skips_a_newer_peers_addition():
    """The point of the open-type wrapper: `Y0` is `Y` before its additions were written, and
    it reads `Y`'s encoding by skipping exactly the octets it cannot name."""
    t = _extensible_types()
    assert decode_oer(t["Y0"], encode_oer(t["Y"], {"a": 1, "c": 2, "d": 3})) == {"a": 1}


def test_an_extension_bit_with_no_addition_in_the_bitmap_is_refused():
    """§16.2.2 sets the bit only when an addition is present, so a set bit over an all-zero
    bitmap is a second spelling of "no additions" -- Class A, refused on both rule sets.
    So is a bitmap that cannot be one: no initial octet, or more than seven unused bits."""
    t = _extensible_types()
    for octets in ("8001020600", "80010201", "800100", "8001020880"):
        for rules in (OerRules.BASIC, OerRules.CANONICAL):
            try:
                decode_oer(t["Y"], bytes.fromhex(octets), rules=rules)
                raise AssertionError(f"{octets} decoded under {rules}")
            except Asn1Error:
                pass


def _refused_under(kind, octets: str, rules) -> str:
    try:
        decode_oer(kind, bytes.fromhex(octets), rules=rules)
    except Asn1Error as error:
        return str(error)
    raise AssertionError(f"{octets} decoded under {rules}")


def test_canonical_oer_refuses_a_set_padding_bit_in_either_bitmap():
    """§16.2.4 pads the root preamble with zero bits, and the additions' bitmap (§16.4) has
    zero unused bits. A set one leaves every presence bit as it was, so under CANONICAL each
    is a second spelling of one value: the C plan decoder reports the preamble's as
    non-canonical and the generated codecs refuse it, while this rail accepted both. BASIC
    still reads them, as the plan decoder does."""
    t = _extensible_types()
    cases = [
        (t["S2"], "800506", "810506", {"a": 5, "b": 6}),  # the root preamble's padding
        (t["Y"], "80010206800102", "80010206810102", {"a": 1, "c": 2}),  # the bitmap's
    ]
    for kind, good, bad, value in cases:
        assert decode_oer(kind, bytes.fromhex(good), rules=OerRules.CANONICAL) == value
        assert "not zero" in _refused_under(kind, bad, OerRules.CANONICAL)
        assert decode_oer(kind, bytes.fromhex(bad), rules=OerRules.BASIC) == value


def test_canonical_oer_refuses_a_present_addition_equal_to_its_default():
    """§31.9 for an extension addition as for a root component: the encoder leaves one equal
    to its DEFAULT out, so a peer that sends it anyway spells the same value a second way."""
    kind = _extensible_types()["D"]
    assert encode_oer(kind, {"a": 1, "d": 3}) == encode_oer(kind, {"a": 1}) == bytes.fromhex("0001")
    explicit = "80010206400103"  # the bitmap names d, and d's open type holds its DEFAULT, 3
    assert "DEFAULT" in _refused_under(kind, explicit, OerRules.CANONICAL)
    assert decode_oer(kind, bytes.fromhex(explicit), rules=OerRules.BASIC) == {"a": 1, "d": 3}
    other = encode_oer(kind, {"a": 1, "d": 4})
    assert other.hex() == "80010206400104"
    assert decode_oer(kind, other, rules=OerRules.CANONICAL) == {"a": 1, "d": 4}


# --- §17 SEQUENCE OF quantity field -----------------------------------------------------


def test_the_quantity_field_is_a_length_determinant_then_the_count():
    """§17.2. NOT a bare count and NOT a byte length: a length determinant giving the
    width of the count, then the count as a variable-size unsigned number. Three elements
    encode as `01 03`, not `03`."""
    kind = SequenceOf(_INT, "SEQUENCE OF INTEGER")
    assert encode_oer(kind, []) == b"\x01\x00"
    assert encode_oer(kind, [1, 2, 3]) == b"\x01\x03" + b"\x01\x01\x01\x02\x01\x03"
    big = list(range(300))
    octets = encode_oer(kind, big)
    assert octets[:3] == b"\x02\x01\x2c", octets[:3].hex()  # count 300 in two octets
    assert decode_oer(kind, octets) == big


# --- §18 SET ordering, §19/§31.8 SET OF ordering ----------------------------------------


def test_set_components_are_encoded_in_canonical_tag_order_not_textual_order():
    """§18.2 over X.680 §8.6: by tag CLASS first, then number. The Annex A record depends
    on this — its `name` is [APPLICATION 1] and sorts before `title`'s [0]."""
    kind = Set(
        (Component("ctx0", _INT, tag=0), Component("app1", _INT, tag=1, tag_class=_APP)), name="S"
    )
    # app1 (application) precedes ctx0 (context) regardless of how they were written.
    assert encode_oer(kind, {"ctx0": 1, "app1": 2}) == b"\x01\x02\x01\x01"
    assert decode_oer(kind, b"\x01\x02\x01\x01") == {"ctx0": 1, "app1": 2}


def test_set_of_is_sorted_ascending_in_canonical_oer():
    """§31.8: ascending as octet strings, the shorter zero-padded for the comparison."""
    kind = SetOf(_INT, "SET OF INTEGER")
    ascending = encode_oer(kind, [1, 2, 3])
    assert encode_oer(kind, [3, 1, 2]) == ascending
    assert decode_oer(kind, ascending) == [1, 2, 3]


# --- §20 CHOICE -------------------------------------------------------------------------


def test_choice_encodes_the_outermost_tag_then_the_value():
    """§20.1. The tag is the only thing that says which alternative was chosen — OER has
    no other discriminator, which is why §8.7 exists at all."""
    kind = Choice((Component("num", _INT, tag=0), Component("txt", _UTF8, tag=1)), name="C")
    assert encode_oer(kind, ("num", 5)) == b"\x80\x01\x05"
    assert encode_oer(kind, ("txt", "hi")) == b"\x81\x02hi"
    for value in (("num", 5), ("txt", "hi")):
        assert decode_oer(kind, encode_oer(kind, value)) == value


def test_an_extension_alternative_is_an_open_type_and_an_unknown_one_is_refused():
    """§20.2: an alternative after the `...` carries a length, so a decoder of an older
    version can step over it. The rail used to write it bare (`81 07`), leaving every octet
    after it unreadable to such a decoder. A tag no version known here defines is refused by
    name -- the PER rail's posture (X.691 23.8): a CHOICE has one value and this type has no
    name for it."""
    t = _extensible_types()
    assert encode_oer(t["C"], ("x", 7)) == bytes.fromhex("8007")
    assert encode_oer(t["C"], ("y", 7)) == bytes.fromhex("810107")
    assert decode_oer(t["C"], bytes.fromhex("810107")) == ("y", 7)
    try:
        decode_oer(t["C0"], bytes.fromhex("810107"))
        raise AssertionError("an unknown extension alternative decoded")
    except Asn1Error as exc:
        assert "unknown to this version" in str(exc), exc
    # A length that disagrees with the alternative's own encoding is not skipped past.
    try:
        decode_oer(t["C"], bytes.fromhex("81020700"))
        raise AssertionError("octets after an alternative inside its open type decoded")
    except Asn1Error as exc:
        assert "20.2" in str(exc), exc


def test_an_untagged_choice_alternative_is_refused_rather_than_guessed():
    inner = Choice((Component("a", _INT, tag=0),), name="Inner")
    outer = Choice((Component("nested", inner),), name="Outer")
    try:
        encode_oer(outer, ("nested", ("a", 1)))
        raise AssertionError("encoded an untagged CHOICE alternative")
    except Asn1Error as exc:
        assert "20.1" in str(exc), exc


# --- §6.2 self-delimiting only with the type -------------------------------------------


def test_trailing_octets_after_a_complete_encoding_are_an_error():
    """§6.2: the end of an OER encoding is knowable only from the type, so leftover
    octets mean the sender and this type disagree — silence would hide a real mismatch."""
    try:
        decode_oer(_INT, b"\x01\x05\x00")
        raise AssertionError("trailing octets were ignored")
    except Asn1Error as exc:
        assert "remain" in str(exc), exc


# --- the StreamPack projection under a second set of encoding rules ---------------------


def _corpus():
    from bcir.examples import PROGRAMS
    from bcir.gem import hydrate
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta

    host, theta = TargetProfile.x86_avx512(), Theta.cool()
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        yield name, hydrate(module, optimize(module, host, theta))


def test_the_streampack_projection_round_trips_under_oer_for_every_corpus_program():
    """The same module, a second transfer syntax. Nothing about `STREAM_PACK` changed to
    gain OER — which is the concrete form of the claim that encoding rules are a
    realization choice rather than part of the schema."""
    from bcir.asn1.streampack import decode_pack_oer, encode_pack_oer

    checked = 0
    for name, pack in _corpus():
        octets = encode_pack_oer(pack)
        recovered = decode_pack_oer(octets, canonical=True)
        assert recovered.source_plan == pack.source_plan, name
        assert len(recovered.segments) == len(pack.segments), name
        assert [s.claim_id for s in recovered.segments] == [s.claim_id for s in pack.segments], name
        assert encode_pack_oer(recovered) == octets, f"{name}: OER is not canonical"
        checked += 1
    assert checked >= 10, f"corpus shrank to {checked} programs"


def test_oer_is_smaller_than_der_on_the_whole_corpus():
    """Not a benchmark — a property. OER drops every tag and every length that the type
    already implies, so for this module it cannot be larger. Phase H needs the direction
    of this inequality to be a fact rather than an expectation."""
    from bcir.asn1.streampack import encode_pack, encode_pack_oer

    der_total = oer_total = 0
    for name, pack in _corpus():
        der, oer = len(encode_pack(pack)), len(encode_pack_oer(pack))
        assert oer < der, f"{name}: OER {oer} is not smaller than DER {der}"
        der_total += der
        oer_total += oer
    assert oer_total < der_total
    # Measured 0.764 at the time of writing; the bound is loose so a schema change that
    # shifts the ratio slightly does not fail the gate, but a regression to parity does.
    assert oer_total / der_total < 0.85, oer_total / der_total


def test_the_two_rule_sets_have_the_object_identifiers_the_standard_assigns():
    """§32.2. These name the transfer syntax in a protocol; a wrong arc is a wrong
    negotiation."""
    from bcir.asn1.oer import BASIC_OER_OID, CANONICAL_OER_OID

    assert BASIC_OER_OID == (2, 1, 6, 0)
    assert CANONICAL_OER_OID == (2, 1, 6, 1)
