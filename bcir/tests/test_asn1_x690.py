"""X.690 (02/2021) conformance for the ASN.1 codec.

The fixtures are the standard's **own worked examples**, transcribed with the clause
that produces them. That matters more than a self-consistent round trip: a codec can
round-trip its own private dialect forever, so the only evidence that these octets are
*ASN.1* is that ITU-T published them.

Erratum 1 (09/2021) is covered too — it redraws Figure 4 (the high-tag-number
identifier octets) and changes no normative rule, so the test asserts the encoding the
corrected figure describes.
"""
from __future__ import annotations

import math

from bcir.asn1 import Asn1Error, BitString, Tag, TagClass, Universal
from bcir.asn1.codec import NULL, Oid, RelativeOid, SetOf, Strictness, decode_der, encode_der
from bcir.asn1.der import der_violations, is_der, to_der
from bcir.asn1.tags import decode_tag, encode_tag
from bcir.asn1.tlv import Tlv, decode_one, encode_tlv
from bcir.asn1.values import (
    decode_integer,
    decode_oid,
    decode_real,
    decode_relative_oid,
    encode_integer,
    encode_oid,
    encode_real,
    encode_relative_oid,
)


# --- 8.1.2 identifier octets -------------------------------------------------------

def test_identifier_octets_match_the_standards_examples():
    """§8.1.2.2 (low tag number) and the §8.14 tagged-type worked example."""
    cases = [
        (Tag(TagClass.UNIVERSAL, 1), "01"),                    # BOOLEAN
        (Tag(TagClass.UNIVERSAL, 16, True), "30"),             # SEQUENCE, constructed
        (Tag(TagClass.UNIVERSAL, 17, True), "31"),             # SET, constructed
        (Tag(TagClass.UNIVERSAL, 26), "1a"),                   # VisibleString, §8.14
        (Tag(TagClass.APPLICATION, 3), "43"),                  # §8.14 Type2
        (Tag(TagClass.CONTEXT, 2, True), "a2"),                # §8.14 Type3
        (Tag(TagClass.APPLICATION, 7, True), "67"),            # §8.14 Type4
        (Tag(TagClass.CONTEXT, 2), "82"),                      # §8.14 Type5
    ]
    for tag, want in cases:
        assert encode_tag(tag).hex() == want, tag
        back, size = decode_tag(bytes.fromhex(want))
        assert back == tag and size == len(want) // 2, tag


def test_high_tag_number_form_round_trips_and_rejects_redundant_groups():
    """§8.1.2.4 (as redrawn by Erratum 1 09/2021) and §8.1.2.4.2 c)."""
    for number in (31, 127, 128, 16383, 16384, 0xFFFF):
        tag = Tag(TagClass.CONTEXT, number)
        raw = encode_tag(tag)
        assert raw[0] & 0x1F == 0x1F, "leading octet must signal the long form"
        assert all(o & 0x80 for o in raw[1:-1]) and not raw[-1] & 0x80
        assert decode_tag(raw)[0] == tag
    # §8.1.2.4.2 c): bits 7-1 of the first subsequent octet shall not all be zero.
    for bad in (b"\x9f\x80\x01", b"\x9f\x80\x80\x01"):
        try:
            decode_tag(bad)
            raise AssertionError(f"accepted a redundant leading group: {bad.hex()}")
        except Asn1Error:
            pass
    # A number that fits the short form must not use the long one.
    try:
        decode_tag(b"\x9f\x01")
        raise AssertionError("accepted the long form for a number below 31")
    except Asn1Error:
        pass


# --- 8.1.3 length octets -----------------------------------------------------------

def test_length_octet_examples_from_the_standard():
    """§8.1.3.4 (L = 38 -> 0x26) and §8.1.3.5 (L = 201 -> 0x81 0xC9)."""
    from bcir.asn1.length import decode_length, encode_length

    assert encode_length(38) == bytes([0b00100110])
    assert encode_length(201) == bytes([0b10000001, 0b11001001])
    assert decode_length(bytes([0x26]))[0].value == 38
    assert decode_length(bytes([0x81, 0xC9]))[0].value == 201
    # §8.1.3.5 c): 0xFF is reserved.
    try:
        decode_length(b"\xff\x00")
        raise AssertionError("accepted the reserved 0xFF initial length octet")
    except Asn1Error:
        pass
    # §8.1.3.5 NOTE 2: a non-minimal long form is legal BER, and must be flagged.
    assert decode_length(bytes([0x81, 0x05]))[0].non_minimal


# --- 8.3 integer -------------------------------------------------------------------

def test_integer_is_always_the_smallest_possible_number_of_octets():
    """§8.3.2 with its NOTE ("the smallest possible number of octets")."""
    pinned = {0: "00", 127: "7f", 128: "0080", 255: "00ff", 256: "0100",
              -1: "ff", -128: "80", -129: "ff7f", -256: "ff00"}
    for value, want in pinned.items():
        assert encode_integer(value).hex() == want, value
        assert decode_integer(bytes.fromhex(want)) == value, value
    for value in (0, 1, -1, 2 ** 63, -(2 ** 63), 2 ** 200, -(2 ** 200)):
        assert decode_integer(encode_integer(value)) == value
    # §8.3.2 a)/b): the padded forms are rejected on both rails, not just under DER.
    for bad in ("0001", "ff80", ""):
        try:
            decode_integer(bytes.fromhex(bad))
            raise AssertionError(f"accepted a padded integer: {bad}")
        except Asn1Error:
            pass


# --- 8.5 real ----------------------------------------------------------------------

def test_real_special_values_and_binary_canonical_form():
    """§8.5.2/§8.5.3/§8.5.9 specials, and §11.3.1's odd-mantissa canonical form."""
    assert encode_real(0.0) == b""                          # §8.5.2: plus zero
    assert encode_real(-0.0) == bytes([0x43])               # §8.5.9: minus zero
    assert encode_real(math.inf) == bytes([0x40])
    assert encode_real(-math.inf) == bytes([0x41])
    assert encode_real(math.nan) == bytes([0x42])
    assert math.isnan(decode_real(bytes([0x42])))
    assert decode_real(b"") == 0.0
    for value in (1.0, -1.0, 0.5, 3.14159265358979, 1e300, -1e-300, 2.0 ** 70):
        assert decode_real(encode_real(value)) == value, value
    # §11.3.1: the mantissa is odd, so 1.0 is 1 x 2^0 and not 2 x 2^-1.
    assert encode_real(1.0) == bytes([0x80, 0x00, 0x01])
    # §8.5.9: an unassigned special-value octet is reserved, not a silent success.
    try:
        decode_real(bytes([0x44]))
        raise AssertionError("accepted a reserved SpecialRealValue octet")
    except Asn1Error:
        pass


def test_real_accepts_the_base_8_and_base_16_sender_options():
    """§8.5.7.2: a BER sender may encode a base-2 abstract value in base 8 or 16."""
    # 1.0 spelled with base 16 and a scaling factor, per §8.5.7.2/§8.5.7.3.
    assert decode_real(bytes([0x80 | 0x20, 0x00, 0x01])) == 1.0     # base 16, E=0, N=1
    assert decode_real(bytes([0x80 | 0x10, 0x00, 0x01])) == 1.0     # base 8,  E=0, N=1
    try:
        decode_real(bytes([0x80 | 0x30, 0x00, 0x01]))
        raise AssertionError("accepted the reserved base bits 11")
    except Asn1Error:
        pass


# --- 8.6 bitstring, 8.7 octetstring ------------------------------------------------

def test_bitstring_worked_example_primitive_and_constructed():
    """§8.6.4.2: '0A3B5F291CD'H as a primitive and as a constructed encoding."""
    primitive = bytes.fromhex("0307040a3b5f291cd0")
    constructed = bytes.fromhex("23800303000a3b0305045f291cd00000")
    assert decode_one(primitive).content.hex() == "040a3b5f291cd0"
    both = decode_one(constructed)
    assert both.indefinite and len(both.children) == 2
    # The abstract value is the same; only the transfer form differs.
    assert encode_tlv(to_der(both)) == primitive


def test_octetstring_constructed_segments_concatenate():
    """§8.7.3.1: segment boundaries carry no significance."""
    raw = bytes.fromhex("2480" "0403 010203" "0402 0405" "0000".replace(" ", ""))
    tlv = decode_one(raw)
    assert tlv.flatten_content() == bytes.fromhex("0102030405")
    assert encode_tlv(to_der(tlv)) == bytes.fromhex("04050102030405")


# --- 8.9 sequence, 8.14 prefixed types ---------------------------------------------

def test_sequence_worked_example_is_byte_identical():
    """§8.9.3: SEQUENCE {name IA5String, ok BOOLEAN} = {"Smith", TRUE}."""
    want = bytes.fromhex("300a" "1605536d697468" "0101ff")
    built = Tlv(Tag(TagClass.UNIVERSAL, Universal.SEQUENCE, True), b"", [
        Tlv(Tag(TagClass.UNIVERSAL, Universal.IA5_STRING), b"Smith"),
        Tlv(Tag(TagClass.UNIVERSAL, Universal.BOOLEAN), b"\xff"),
    ])
    assert encode_tlv(built) == want
    assert is_der(decode_one(want))


def test_visible_string_jones_in_all_three_sender_option_forms():
    """§8.23.5's example: primitive, constructed-definite, constructed-indefinite.

    §8.23.6 requires receivers to handle all permitted forms; §10.2 requires DER to
    emit only the first. Both halves of "BER in, DER out" in one fixture.
    """
    primitive = bytes.fromhex("1a054a6f6e6573")
    definite = bytes.fromhex("3a09" "04034a6f6e" "04026573")
    indefinite = bytes.fromhex("3a80" "04034a6f6e" "04026573" "0000")
    for raw in (primitive, definite, indefinite):
        assert decode_one(raw).flatten_content() == b"Jones", raw.hex()
        assert encode_tlv(to_der(decode_one(raw))) == primitive, raw.hex()
    assert is_der(decode_one(primitive))
    assert not is_der(decode_one(definite))
    assert not is_der(decode_one(indefinite))


# --- 8.19 / 8.20 object identifiers ------------------------------------------------

def test_object_identifier_worked_examples():
    """§8.19.4's {2 999 3} -> 883703 and §8.20.5's relative {8571 3 2} -> C27B0302."""
    assert encode_oid((2, 999, 3)).hex() == "883703"
    assert decode_oid(bytes.fromhex("883703")) == (2, 999, 3)
    assert encode_relative_oid((8571, 3, 2)).hex() == "c27b0302"
    assert decode_relative_oid(bytes.fromhex("c27b0302")) == (8571, 3, 2)
    # A real-world OID: sha256WithRSAEncryption, as it appears in every X.509 chain.
    assert encode_oid((1, 2, 840, 113549, 1, 1, 11)).hex() == "2a864886f70d01010b"
    # §8.19.2: the fewest possible octets, i.e. no leading 0x80 in a subidentifier.
    try:
        decode_oid(bytes.fromhex("8003"))
        raise AssertionError("accepted a padded subidentifier")
    except Asn1Error:
        pass
    # §8.19.4 NOTE: only three values are allocated from the root node.
    try:
        encode_oid((3, 1))
        raise AssertionError("accepted a first component above 2")
    except Asn1Error:
        pass


# --- clause 10 + 11: DER ------------------------------------------------------------

def test_der_rejects_every_ber_sender_option_it_should():
    """One fixture per clause-10/11 rule a schema-free walk can see."""
    cases = {
        "10.1": bytes.fromhex("3a8004034a6f6e040265730000"),   # indefinite length
        "10.2": bytes.fromhex("3a0904034a6f6e04026573"),       # constructed string
        "11.1": bytes.fromhex("010101"),                       # boolean TRUE != 0xFF
        "11.2.1": bytes.fromhex("030403ffffff"),               # dirty unused bits
        "11.6": bytes.fromhex("3106020102020101"),             # set-of misordered
    }
    for clause, raw in cases.items():
        violations = der_violations(decode_one(raw))
        assert any(v.clause == clause for v in violations), (clause, violations)
        # to_der must repair it, and the repair must itself be DER.
        repaired = encode_tlv(to_der(decode_one(raw)))
        assert is_der(decode_one(repaired)), (clause, repaired.hex())


def test_non_minimal_length_is_ber_only():
    """§8.1.3.5 NOTE 2 permits it; §10.1 forbids it."""
    raw = bytes.fromhex("1a81054a6f6e6573")
    assert decode_one(raw).flatten_content() == b"Jones"
    assert [v.clause for v in der_violations(decode_one(raw))] == ["10.1"]
    assert encode_tlv(to_der(decode_one(raw))) == bytes.fromhex("1a054a6f6e6573")


def test_ber_to_der_conversion_is_idempotent():
    """Re-encoding DER returns identical octets — the property digests depend on."""
    from bcir.asn1 import reencode_as_der

    for raw in (bytes.fromhex("3a8004034a6f6e040265730000"),
                bytes.fromhex("010101"),
                bytes.fromhex("3106020102020101")):
        once = reencode_as_der(raw)
        assert reencode_as_der(once) == once, raw.hex()


def test_strict_der_decode_refuses_ber():
    """The trust-boundary posture: a peer must not choose among equivalent spellings."""
    ber = bytes.fromhex("3a8004034a6f6e040265730000")
    assert decode_der(ber, strictness=Strictness.BER) == "Jones"
    try:
        decode_der(ber, strictness=Strictness.DER)
        raise AssertionError("DER strictness accepted a BER-only encoding")
    except Asn1Error:
        pass


# --- the value mapping --------------------------------------------------------------

def test_python_value_round_trip_is_exact_and_canonical():
    values = [True, False, 0, 127, -128, 10 ** 40, 3.5, b"\x01\x02", "héllo ✓", NULL,
              Oid((1, 2, 840, 113549, 1, 1, 11)), RelativeOid((8571, 3, 2)),
              BitString(bytes.fromhex("0a3b5f291cd0"), 4),
              [1, "two", b"\x03"]]
    for value in values:
        raw = encode_der(value)
        assert decode_der(raw) == value, value
        assert encode_der(decode_der(raw)) == raw, value       # canonical
        assert is_der(decode_one(raw)), value


def test_set_of_is_encoded_in_ascending_order():
    """§11.6: components ascend as octet strings, shorter ones zero-padded."""
    raw = encode_der(SetOf([3, 1, 2]))
    assert raw == bytes.fromhex("3109020101020102020103")
    assert is_der(decode_one(raw))


# --- totality over hostile input ----------------------------------------------------

def test_decoder_is_total_over_malformed_input():
    """Every input either decodes or raises Asn1Error — never anything else."""
    corpus = [
        b"", b"\x30", b"\x30\x05", b"\x30\x80", b"\x00\x00", b"\x1f", b"\x1f\x80\x01",
        b"\x02\xff", b"\x30\x80\x01\x01\xff", b"\x05\x80", b"\x02\x01", b"\xff" * 32,
        b"\x30\x84\xff\xff\xff\xff", b"\x24\x80" + b"\x04\x01a" * 3,
    ]
    for raw in corpus:
        try:
            decode_one(raw)
        except Asn1Error:
            pass


def test_nesting_depth_is_bounded():
    """A decoder that recurses on attacker-chosen depth is a stack-exhaustion surface."""
    from bcir.asn1.tlv import DEFAULT_MAX_DEPTH

    bomb = b"\x30\x80" * (DEFAULT_MAX_DEPTH + 8) + b"\x00\x00" * (DEFAULT_MAX_DEPTH + 8)
    try:
        decode_one(bomb)
        raise AssertionError("accepted an encoding deeper than the decoder bound")
    except Asn1Error as exc:
        assert "nesting" in str(exc), exc
