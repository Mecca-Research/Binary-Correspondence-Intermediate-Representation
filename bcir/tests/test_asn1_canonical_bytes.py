"""One abstract value, one octet string — the invariant every BCIR digest rests on.

Eight defects found in an independent audit of `main` at `3decf69`, reproduced here before
they were fixed. They are not decoding errors in the usual sense: each one *accepted* an
encoding, and returned the right abstract value for it. What each one gave away is
**uniqueness** — a second byte string that a strict decoder accepted for a value that
already had one — or, in two cases, let the octets choose a type the schema did not ask for.

That distinction is the whole point. StreamPack digests the octets it receives and BCIR
compares artifacts byte-for-byte, so a canonical rule that admits a second spelling is not a
cosmetic conformance gap: it is two digests for one artifact, chosen by whoever sent the
bytes.

  * §8.1.2   — a schema-directed decode never compared the wire tag to the requested type,
               so an INTEGER schema returned `True` for `01 01 ff` and a SEQUENCE schema
               accepted a SET or a constructed context tag over identical contents.
  * §11.5    — DER "shall not" encode a component equal to its DEFAULT. The encoder obeyed;
               the decoder accepted both spellings.
  * §11.3.1  — DER REAL is base 2, F zero, odd mantissa, fewest octets. Bases 8 and 16 (a
               §8.5.4 sender's option), even mantissas and a zero mantissa were all accepted,
               so `80 01 02` and `80 02 01` were two encodings of 4.0.
  * §11.3.2  — the decimal form is ISO 6093 NR3 with five further restrictions. The field was
               handed to Python's `float()`, whose grammar is a strict superset: `1_0`,
               `inf`, `NaN`, `1e5` and surrounding whitespace all parsed. Two of those are
               second spellings of values §8.5.9 already encodes exactly once.
  * §8.7.3.2 — the segments of a constructed string are encodings of the same type. Any child
               tag was accepted and its contents concatenated, so a constructed VisibleString
               holding an INTEGER laundered arbitrary octets into a string type.
  * X.696
    §31.9    — CANONICAL-OER has the same DEFAULT rule as DER, and the same gap.
  * X.697
    §4.3     — the bounded scanner counted separators, so `elements=0` admitted `[0]`: a
               resource limit that did not hold at its own boundary.
  * X.692
    §24.7.4  — `INT-TO-CHARS` inverted with `int()`, which accepts PEP 515 underscores and
               every Unicode decimal digit. `4_2` and `۴۲` both decoded to 42.

Every case below asserts BOTH halves: the bad spelling is refused, and the encoder's own
output still round-trips. A canonicality check that also rejects valid encodings would be a
worse defect than the one it replaced.
"""

from __future__ import annotations

import math

from bcir.asn1.codec import Strictness, from_tlv, to_tlv
from bcir.asn1.der import der_violations, require_der
from bcir.asn1.ecn_transform import IntToChars
from bcir.asn1.jer_bounded import JerBoundedError, JerLimits, scan
from bcir.asn1.oer import OerRules, decode_oer, encode_oer
from bcir.asn1.schema import Component, Primitive, Sequence, Set
from bcir.asn1.tags import Tag, TagClass, Universal
from bcir.asn1.tlv import Tlv, decode_one, encode_tlv
from bcir.asn1.values import Asn1Error, decode_string

_INT = Primitive(Universal.INTEGER, "INTEGER")


def _refused(fn, what: str) -> str:
    try:
        result = fn()
    except Asn1Error as exc:
        return str(exc)
    raise AssertionError(f"{what}: accepted, returning {result!r}")


def _der(hexs: str):
    """Decode `hexs` under strict DER, the way a trust boundary would."""
    tlv = decode_one(bytes.fromhex(hexs))
    require_der(tlv)
    return from_tlv(tlv, strictness=Strictness.DER)


# --- §8.1.2: the requested type's tag is a claim to check ---------------------------------


def test_a_schema_directed_decode_refuses_another_types_encoding() -> None:
    """The octets do not get to choose the type when a schema was supplied.

    `Primitive.decode` fell through to the value-directed mapping in `codec`, which reads
    whatever tag arrived. That mapping is right for a schema-FREE walk and wrong here: the
    caller named INTEGER, so a BOOLEAN encoding is a fault, not a boolean.
    """
    for hexs, what in (
        ("0101ff", "a BOOLEAN encoding"),
        ("040141", "an OCTET STRING encoding"),
        ("0500", "a NULL encoding"),
    ):
        message = _refused(
            lambda h=hexs: _INT.decode(decode_one(bytes.fromhex(h)), strictness=Strictness.DER),
            f"{what} for an INTEGER schema",
        )
        assert "8.1.2" in message, message

    # The type it actually names still decodes.
    assert _INT.decode(decode_one(bytes.fromhex("020105")), strictness=Strictness.DER) == 5


def test_a_constructor_refuses_a_foreign_constructed_tag() -> None:
    """A SEQUENCE schema accepted SET and `[0]` over byte-identical contents.

    This is the StreamPack-critical case: a pack's root SEQUENCE tag could be replaced and
    the projection still decoded, so one pack had several accepted spellings.
    """
    seq = Sequence((Component("a", _INT),), name="S")
    canonical = encode_tlv(seq.encode({"a": 5}))
    assert canonical.hex() == "3003020105"
    assert seq.decode(decode_one(canonical), strictness=Strictness.DER) == {"a": 5}

    for first, what in ((0x31, "a SET tag"), (0xA0, "a constructed [0] tag")):
        swapped = bytes([first]) + canonical[1:]
        message = _refused(
            lambda r=swapped: seq.decode(decode_one(r), strictness=Strictness.DER),
            f"{what} for a SEQUENCE schema",
        )
        assert "8.1.2" in message, message


# --- §11.5 / X.696 §31.9: a DEFAULT that is present ---------------------------------------


def test_der_refuses_a_component_encoded_at_its_default() -> None:
    """§11.5, from the decoding side. Both spellings decoded to one value.

    BER still accepts it — §11.5 is a clause-11 restriction — and that asymmetry is the
    test: "BER in, DER out" means the two rails must differ here, not agree.
    """
    # Tagged, as StreamPack's own schema is: X.680 §24.4 needs distinct tags wherever
    # optionality makes the component boundary ambiguous, and an omitted DEFAULT does
    # exactly that -- two adjacent untagged INTEGERs would not be a legal SEQUENCE.
    kind = Sequence(
        (Component("version", _INT, tag=0, default=1), Component("body", _INT, tag=1)), name="D"
    )
    omitted = encode_tlv(kind.encode({"version": 1, "body": 7}))
    assert omitted.hex() == "3003810107", "the encoder has always omitted it"

    written = encode_tlv(
        Tlv(
            Tag(TagClass.UNIVERSAL, Universal.SEQUENCE, True),
            b"",
            [Tlv(Tag(TagClass.CONTEXT, 0), b"\x01"), Tlv(Tag(TagClass.CONTEXT, 1), b"\x07")],
        )
    )
    assert written != omitted

    message = _refused(
        lambda: kind.decode(decode_one(written), strictness=Strictness.DER),
        "a DEFAULT written out, under strict DER",
    )
    assert "11.5" in message

    # BER keeps accepting it, and both spellings still mean the same value there.
    assert kind.decode(decode_one(written), strictness=Strictness.BER) == {"version": 1, "body": 7}
    assert kind.decode(decode_one(omitted), strictness=Strictness.DER) == {"version": 1, "body": 7}
    # A non-default value in the same position is ordinary traffic.
    other = encode_tlv(kind.encode({"version": 2, "body": 7}))
    assert kind.decode(decode_one(other), strictness=Strictness.DER)["version"] == 2


def test_a_set_refuses_a_component_encoded_at_its_default() -> None:
    """SET reaches §11.5 through its own decode loop, so it needs its own case."""
    kind = Set((Component("v", _INT, tag=0, default=1), Component("b", _INT, tag=1)), name="T")
    omitted = encode_tlv(kind.encode({"v": 1, "b": 7}))
    assert kind.decode(decode_one(omitted), strictness=Strictness.DER) == {"v": 1, "b": 7}

    written = encode_tlv(
        Tlv(
            Tag(TagClass.UNIVERSAL, Universal.SET, True),
            b"",
            [Tlv(Tag(TagClass.CONTEXT, 0), b"\x01"), Tlv(Tag(TagClass.CONTEXT, 1), b"\x07")],
        )
    )
    assert "11.5" in _refused(
        lambda: kind.decode(decode_one(written), strictness=Strictness.DER),
        "a SET DEFAULT written out",
    )


def test_canonical_oer_refuses_a_component_encoded_at_its_default() -> None:
    """X.696 §31.9 is DER's §11.5 for OER, and had the identical gap.

    BASIC-OER must keep accepting it: §31 restricts the canonical rules only.
    """
    kind = Sequence((Component("v", _INT, default=1), Component("b", _INT)), name="D")
    omitted = encode_oer(kind, {"v": 1, "b": 7}, rules=OerRules.CANONICAL)
    assert omitted.hex() == "000107"
    assert decode_oer(kind, omitted, rules=OerRules.CANONICAL) == {"v": 1, "b": 7}

    written = bytes.fromhex("8001010107")
    assert "31.9" in _refused(
        lambda: decode_oer(kind, written, rules=OerRules.CANONICAL),
        "a present DEFAULT under CANONICAL-OER",
    )
    assert decode_oer(kind, written, rules=OerRules.BASIC) == {"v": 1, "b": 7}


# --- §11.3: the canonical REAL ------------------------------------------------------------


def test_der_real_is_base_two_with_an_odd_mantissa() -> None:
    """§11.3.1's NOTE says why: {M, 2, E} and {M/2^n, 2, E+n} are the SAME real value.

    Without "M is either 0 or is odd" a value has unboundedly many encodings. `0903800102`
    and `0903800201` are both 4.0, and both were accepted.
    """
    assert _der("0903800101") == 2.0, "the canonical spelling still decodes"

    for hexs, why in (
        ("0903900101", "base 8"),
        ("0903a00101", "base 16"),
        ("0903800102", "an even mantissa"),
        ("0903800104", "an even mantissa"),
        ("0903840101", "a non-zero scaling factor F"),
        ("0903800000", "a zero mantissa, which 8.5.2 spells as no octets"),
        ("0904800000" + "01", "a mantissa with a leading zero octet"),
    ):
        assert "11.3" in _refused(lambda h=hexs: _der(h), f"{why} under strict DER")

    # 8.5.4 makes the base a sender's option in BER, so BER still reads all of them.
    assert from_tlv(decode_one(bytes.fromhex("0903900101")), strictness=Strictness.BER) == 8.0


def test_der_real_decimal_is_iso_6093_not_a_python_literal() -> None:
    """§11.3.2.1: the NR3 form. The field was parsed by `float()`, which is far wider.

    `1_0` is the sharpest case — those octets are not a number in ISO 6093 at all, in any
    form, yet they decoded to 10.0. `inf` and `NaN` are the other kind of failure: §8.5.9
    already encodes both in a single octet, so accepting the spelled words hands one value
    a second encoding.
    """

    def decimal(form: int, text: str) -> str:
        body = bytes([form]) + text.encode()
        return "09" + f"{len(body):02x}" + body.hex()

    for text, why in (
        ("1_0", "a PEP 515 underscore"),
        ("1_000", "a PEP 515 underscore"),
        ("NaN", "a word 8.5.9 encodes as one octet"),
        ("inf", "a word 8.5.9 encodes as one octet"),
        ("-Infinity", "a Python spelling of infinity"),
        (" 1.E+1 ", "SPACE, which 11.3.2.2 forbids"),
        ("+1.E+1", "PLUS SIGN, which 11.3.2.3 forbids"),
        ("0.5E+1", "a leading mantissa zero (11.3.2.4)"),
        ("1.50E+1", "a trailing mantissa zero (11.3.2.4)"),
        ("1.5E+2", "PLUS SIGN on a non-zero exponent (11.3.2.6)"),
        ("1.5E01", "a leading exponent zero (11.3.2.6)"),
    ):
        assert "11.3" in _refused(
            lambda h=decimal(3, text): _der(h), f"{text!r} ({why}) under strict DER"
        )

    # The NR selector must MEAN something: NR1 is ISO 6093's integer form.
    assert "8.5.8" in _refused(
        lambda: _der(decimal(1, "1.5e3")), "a floating-point field declared NR1"
    )
    assert from_tlv(decode_one(bytes.fromhex(decimal(1, "42"))), strictness=Strictness.BER) == 42.0


def test_every_real_the_encoder_writes_survives_the_new_check() -> None:
    """The other half. A guard that rejects the encoder's own output is a worse bug."""
    values = [
        0.0,
        1.0,
        2.0,
        4.0,
        -4.0,
        0.5,
        0.125,
        1.5,
        1e10,
        1e-10,
        -1e-10,
        3.14159265358979,
        1e300,
        1e-300,
        math.inf,
        -math.inf,
        -0.0,
        2.0**53 + 1.0,
        123456.789,
    ]
    for value in values:
        raw = encode_tlv(to_tlv(value))
        tlv = decode_one(raw)
        assert not der_violations(tlv), (value, raw.hex())
        back = from_tlv(tlv, strictness=Strictness.DER)
        assert back == value or (math.isnan(back) and math.isnan(value)), value
        assert encode_tlv(to_tlv(back)) == raw, value  # and it is a fixed point


# --- §8.7.3.2: a constructed string's segments --------------------------------------------


def test_a_constructed_string_segment_must_be_the_same_type() -> None:
    """§8.23.3 makes a character string `[UNIVERSAL x] IMPLICIT OCTET STRING`.

    §8.14.4's implicit tag replaces only the OUTERMOST tag, so §8.7.3.2's recursion runs
    over an octetstring and its NOTE 2 holds verbatim: the segment tags "are always
    universal class, number 4". §8.23.5's own worked example encodes exactly that.

    The unchecked flatten concatenated any child's contents, so an INTEGER segment's
    contents octet became a character — and `to_der` then re-emitted it as a valid
    primitive string. Arbitrary octets entered any string type that way, including one
    whose repertoire forbids them.
    """
    for hexs in (
        "1a054a6f6e6573",  # primitive
        "3a0904034a6f6e04026573",  # 8.23.5's constructed-definite
        "3a8004034a6f6e040265730000",
    ):  # ... and indefinite
        assert decode_string(decode_one(bytes.fromhex(hexs))) == "Jones", hexs

    for hexs, why in (
        ("3a03020105", "an INTEGER segment"),
        ("3a051a03616263", "a VisibleString segment (8.7.3.2 says 4)"),
        ("3a052403020105", "an INTEGER nested under an OCTET STRING"),
    ):
        assert "8.7.3.2" in _refused(
            lambda h=hexs: decode_string(decode_one(bytes.fromhex(h))),
            f"{why} inside a constructed VisibleString",
        )

    # The repertoire check is downstream of the flatten, so it only ever saw laundered
    # octets that had already become a `str`. This is the octet that used to get through.
    assert "8.7.3.2" in _refused(
        lambda: decode_string(decode_one(bytes.fromhex("3a0302017f"))),
        "DEL smuggled into a VisibleString through an INTEGER segment",
    )


# --- X.697 §4.3: a bound that holds at its own boundary ------------------------------------


def test_a_zero_container_limit_admits_nothing() -> None:
    """The scanner counted COMMAS, so a one-child container never reached the ceiling.

    `elements=0` accepted `[0]` and `members=0` accepted `{"k":0}`. On the one code path
    whose entire job is bounding untrusted input, the limit did not hold at its own
    boundary — and zero is exactly the value a caller picks when it means "none".
    """

    def refuses(text: bytes, limits: JerLimits) -> bool:
        try:
            scan(text, limits=limits)
            return False
        except JerBoundedError:
            return True

    assert not refuses(b"[]", JerLimits(elements=0)), "an empty array has no elements"
    assert refuses(b"[0]", JerLimits(elements=0))
    assert refuses(b"[0,1]", JerLimits(elements=0))
    assert not refuses(b"{}", JerLimits(members=0)), "an empty object has no members"
    assert refuses(b'{"k":0}', JerLimits(members=0))

    # The ceiling still means "at most n" everywhere above zero, and still accepts n.
    for n in (1, 2, 3):
        full = b"[" + b",".join(b"0" * 1 for _ in range(n)) + b"]"
        assert not refuses(full, JerLimits(elements=n)), (n, full)
        assert refuses(full, JerLimits(elements=n - 1)), (n, full)

    # A nested container is counted against its OWN ceiling, not the outer one.
    assert not refuses(b"[[0,1]]", JerLimits(elements=2))
    assert refuses(b"[[0,1]]", JerLimits(elements=1))


# --- X.692 §24.7.4: DIGIT ZERO through DIGIT NINE -----------------------------------------


def test_int_to_chars_inverts_over_the_clauses_repertoire_only() -> None:
    """`int()` accepts PEP 515 underscores and every Unicode decimal digit.

    Neither is in §24.7.4's repertoire, so `4_2` and `۴۲` (ARABIC-INDIC DIGIT FOUR, TWO)
    were character strings the type does not contain, decoded as 42. The encoder would
    then re-emit ASCII `42`, so the round trip was not byte-preserving either.
    """
    transform = IntToChars()
    for text, want in (
        ("42", 42),
        ("  42  ", 42),
        ("+42", 42),
        ("-0042", -42),
        ("007", 7),
        ("0", 0),
        ("-0", 0),
    ):
        assert transform.inverse_one(text) == want, text

    for text, why in (
        ("4_2", "a PEP 515 underscore"),
        ("1_000", "a PEP 515 underscore"),
        ("۴۲", "ARABIC-INDIC digits"),
        ("４２", "FULLWIDTH digits"),
        ("4٠2", "an ARABIC-INDIC zero mid-string"),
    ):
        assert "24.7.4" in _refused(
            lambda t=text: transform.inverse_one(t), f"{text!r} ({why}) as an INT-TO-CHARS field"
        )

    # Round-tripping the encoder's own output is unaffected.
    for value in (0, 7, 42, -42, 1000, -1):
        assert transform.inverse_one(transform.apply_one(value)) == value, value
