"""X.696 Octet Encoding Rules (roadmap phase D).

The centrepiece is `test_annex_a_...`: the standard's OWN worked example, transcribed
from X.696 Annex A.3.1, encoded byte-for-byte. That is the only kind of test that can
establish a wire format is right — a round-trip test passes just as happily when the
encoder and decoder share the same wrong assumption, which is exactly the failure mode a
from-memory implementation produces. Everything else here pins one clause each.
"""
from __future__ import annotations

from bcir.asn1.oer import (OerRules, decode_length, decode_oer, decode_tag, encode_length,
                           encode_oer, encode_tag)
from bcir.asn1.schema import (Choice, Component, Primitive, Sequence, SequenceOf, Set,
                              SetOf)
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
    name = Sequence((Component("givenName", _VIS), Component("initial", _VIS),
                     Component("familyName", _VIS)), name="Name")
    child = Set((Component("name", name, tag=1, tag_class=_APP),
                 Component("dateOfBirth", _VIS, tag=0)), name="ChildInformation")
    personnel = Set((
        Component("name", name, tag=1, tag_class=_APP),
        Component("title", _VIS, tag=0),
        Component("number", _INT, tag=2, tag_class=_APP),
        Component("dateOfHire", _VIS, tag=1),
        Component("nameOfSpouse", name, tag=2),
        Component("children", SequenceOf(child, "SEQUENCE OF ChildInformation"),
                  tag=3, default=[]),
    ), name="PersonnelRecord")
    return personnel


#: Annex A.2, the John Smith record.
_ANNEX_A_VALUE = {
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

#: Annex A.3.1 hexadecimal view, transcribed exactly. A.3 states the length is 95 octets
#: in both BASIC-OER and CANONICAL-OER, and that the two produce the same encoding here.
_ANNEX_A_OCTETS = bytes.fromhex("".join("""
80044A6F 686E0150 05536D69 74680133 08446972 6563746F 72083139 37313039
3137044D 61727901 5405536D 69746801 02055261 6C706801 5405536D 69746808
31393537 31313131 05537573 616E0142 054A6F6E 65730831 39353930 373137
""".split()))


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
    assert decode_oer(_annex_a_types(), _ANNEX_A_OCTETS,
                      rules=OerRules.CANONICAL) == _ANNEX_A_VALUE


# --- §8.6 the length determinant --------------------------------------------------------

def test_length_determinant_short_form_boundary():
    """§8.6.4: the short form covers 0..127 in one octet; 128 needs the long form."""
    assert encode_length(0) == b"\x00"
    assert encode_length(127) == b"\x7f"
    assert encode_length(128) == b"\x81\x80"               # §8.6.5, one subsequent octet
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
    assert encode_oer(_BOOL, True) == b"\xff"              # §31.3
    assert encode_oer(_BOOL, False) == b"\x00"
    assert decode_oer(_BOOL, b"\x01", rules=OerRules.BASIC) is True     # §9
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
    assert encode_oer(_INT, 51) == b"\x01\x33"             # as in Annex A
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
    assert encode_oer(_ENUM, 128) == b"\x82\x00\x80"       # 2 octets; signed, so 0x00
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
    assert encode_oer(_OCTETS, b"\x01\x02") == b"\x02\x01\x02"           # §14.2
    assert encode_oer(_UTF8, "hi") == b"\x02hi"                          # §27.3
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
    kind = Sequence((Component("a", _INT, optional=True),
                     Component("b", _INT, optional=True)), name="T")
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
    assert octets[:3] == b"\x02\x01\x2c", octets[:3].hex()   # count 300 in two octets
    assert decode_oer(kind, octets) == big


# --- §18 SET ordering, §19/§31.8 SET OF ordering ----------------------------------------

def test_set_components_are_encoded_in_canonical_tag_order_not_textual_order():
    """§18.2 over X.680 §8.6: by tag CLASS first, then number. The Annex A record depends
    on this — its `name` is [APPLICATION 1] and sorts before `title`'s [0]."""
    kind = Set((Component("ctx0", _INT, tag=0),
                Component("app1", _INT, tag=1, tag_class=_APP)), name="S")
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
    kind = Choice((Component("num", _INT, tag=0), Component("txt", _UTF8, tag=1)),
                  name="C")
    assert encode_oer(kind, ("num", 5)) == b"\x80\x01\x05"
    assert encode_oer(kind, ("txt", "hi")) == b"\x81\x02hi"
    for value in (("num", 5), ("txt", "hi")):
        assert decode_oer(kind, encode_oer(kind, value)) == value


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
        assert [s.claim_id for s in recovered.segments] == \
            [s.claim_id for s in pack.segments], name
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
