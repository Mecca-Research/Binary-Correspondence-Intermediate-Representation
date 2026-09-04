"""CANONICAL-PER means one encoding per abstract value — three places it did not.

Review findings from PR #662, still live on main. Each is a case where two different octet
strings decoded to the same abstract value under CANONICAL-PER, which is exactly the property
the projection's byte-identity and digest claims rest on.

  * §22 -- SET OF is UNORDERED as an abstract value, so a canonical encoding must fix an
    order. The encoder preserved the caller's list order, as it does for the ORDERED
    SEQUENCE OF, so `[1, 2]` and `[2, 1]` produced different CANONICAL bytes for one value.
  * §18.2 -- a DEFAULT component equal to its default is OMITTED in canonical form. The
    encoder omitted it; the decoder accepted the long spelling anyway, so the canonical
    decoder admitted a non-canonical encoding.
  * §19.9 / §23.8 -- an extension addition is wrapped as a COMPLETE encoding, so §11.1
    governs the wrapper's contents. Nothing checked that the inner reader consumed the
    wrapper, so a peer could append octets or set the pad bits and still hand the caller the
    same component value.

The canonical-or-excluded discipline is the point: BASIC-PER permits both spellings and must
keep doing so, which is why every test here checks the rule set as well as the octets.
"""

from __future__ import annotations

from bcir.asn1.constraints import ValueRange
from bcir.asn1.per import (
    BitWriter,
    PerRules,
    PerVariant,
    _encode_constrained,
    _encode_length_and_payload,
    _encode_normally_small_length,
    decode_per,
    encode_per,
)
from bcir.asn1.schema import Component, Primitive, Sequence, SequenceOf, SetOf
from bcir.asn1.tags import Universal
from bcir.asn1.values import Asn1Error

_VARIANTS = (PerVariant.UNALIGNED, PerVariant.ALIGNED)
_BYTE = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255))
_BIT = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 1))


def _refused(fn, what: str) -> str:
    try:
        result = fn()
    except Asn1Error as exc:
        return str(exc)
    raise AssertionError(f"{what}: accepted, returning {result!r}")


def test_a_canonical_set_of_does_not_depend_on_the_callers_list_order() -> None:
    """§22: two peers holding the same SET must produce the same octets.

    SET OF is unordered, so preserving the caller's list order made the encoding depend on
    something the abstract value does not carry. Any permutation must give one answer.
    """
    kind = SetOf(_BYTE, name="S")
    for variant in _VARIANTS:
        encodings = {
            encode_per(kind, list(order), variant=variant, rules=PerRules.CANONICAL).hex()
            for order in ((1, 2, 3), (3, 2, 1), (2, 1, 3), (3, 1, 2))
        }
        assert len(encodings) == 1, f"{variant.value}: {len(encodings)} spellings of one set"
        raw = bytes.fromhex(encodings.pop())
        assert decode_per(raw, kind, variant=variant) == [1, 2, 3]

    # Elements of unequal encoded length sort with the shorter zero-padded, the rule DER's
    # §11.6 and OER's §31.8 already use -- so the three rails agree on what ascending means.
    wide = SetOf(Primitive(Universal.INTEGER, "INTEGER"), name="W")
    for variant in _VARIANTS:
        a = encode_per(wide, [1, 300], variant=variant, rules=PerRules.CANONICAL)
        b = encode_per(wide, [300, 1], variant=variant, rules=PerRules.CANONICAL)
        assert a == b
        assert sorted(decode_per(a, wide, variant=variant)) == [1, 300]


def test_sequence_of_and_basic_per_keep_the_order_they_were_given() -> None:
    """The other side of the same fix: SEQUENCE OF is ORDERED, and BASIC-PER is not canonical.

    Sorting either one would be a new bug -- SEQUENCE OF's order is part of the abstract
    value, and BASIC-PER is explicitly the rule set that permits more than one spelling.
    """
    ordered = SequenceOf(_BYTE, name="Q")
    unordered = SetOf(_BYTE, name="S")
    for variant in _VARIANTS:
        assert encode_per(ordered, [2, 1], variant=variant) != encode_per(
            ordered, [1, 2], variant=variant
        )
        assert decode_per(
            encode_per(ordered, [2, 1], variant=variant), ordered, variant=variant
        ) == [2, 1]
        assert decode_per(
            encode_per(unordered, [2, 1], variant=variant, rules=PerRules.BASIC),
            unordered,
            variant=variant,
            rules=PerRules.BASIC,
        ) == [2, 1]


def test_canonical_per_refuses_a_present_default_equal_to_its_default() -> None:
    """§18.2: the canonical form omits it, so its presence is a second spelling.

    The encoder already omitted such a component; the decoder took the long spelling without
    consulting the rule set, so CANONICAL-PER had two encodings of one abstract value.
    """
    kind = Sequence((Component("x", _BIT, default=0),), name="D")
    for variant in _VARIANTS:
        # The canonical spelling omits the component entirely: an empty field-list.
        canonical = encode_per(kind, {"x": 0}, variant=variant, rules=PerRules.CANONICAL)
        assert canonical == b"\x00"
        assert decode_per(canonical, kind, variant=variant) == {"x": 0}

        # b"\x80" is the BASIC spelling: presence bit set, then the value 0.
        assert "18.2" in _refused(
            lambda v=variant: decode_per(b"\x80", kind, variant=v, rules=PerRules.CANONICAL),
            "a present DEFAULT equal to its default under CANONICAL-PER",
        )
        # ... and BASIC-PER still accepts it, because that rule set permits both spellings.
        assert decode_per(b"\x80", kind, variant=variant, rules=PerRules.BASIC) == {"x": 0}

        # A DEFAULT component holding a NON-default value is present in both rule sets.
        for rules in (PerRules.CANONICAL, PerRules.BASIC):
            raw = encode_per(kind, {"x": 1}, variant=variant, rules=rules)
            assert decode_per(raw, kind, variant=variant, rules=rules) == {"x": 1}


def _extension_encoding(variant: PerVariant, payload: bytes) -> bytes:
    """`SEQUENCE { a INTEGER(0..255), ..., b INTEGER(0..1) }` with a chosen wrapper payload.

    Built from the module's own primitives rather than by splicing octets, so the OUTER
    encoding stays well-formed no matter what goes inside the §19.9 wrapper -- which is what
    makes this a test of the inner check rather than of the outer one.
    """
    writer = BitWriter(variant)
    writer.put_bit(1)  # §18.1: extension additions are present
    _encode_constrained(writer, 1, 0, 255)  # a = 1
    _encode_normally_small_length(writer, 1)  # §19.7: one addition in the type
    writer.put_bit(1)  # ... and it is present
    _encode_length_and_payload(
        writer,
        len(payload),
        0,
        None,
        lambda start, stop: (writer.align(), writer.put_octets(payload[start:stop])),
    )
    return writer.to_bytes()


def test_an_extension_wrapper_must_be_exactly_the_complete_encoding() -> None:
    """§19.9: the addition is wrapped as a COMPLETE encoding, so §11.1 governs its contents.

    `b INTEGER (0..1)` occupies one bit, so its complete encoding is one octet whose seven
    trailing pad bits are zero. A wrapper carrying more octets, or the same octet with a pad
    bit set, decoded to the identical component value before this check existed.
    """
    kind = Sequence(
        (Component("a", _BYTE), Component("b", _BIT, extension=True)), name="X", extensible=True
    )
    for variant in _VARIANTS:
        good = _extension_encoding(variant, b"\x80")  # b = 1, pad bits zero
        assert decode_per(good, kind, variant=variant) == {"a": 1, "b": 1}
        # The encoder produces exactly this, which is what makes the hand-built case fair.
        assert encode_per(kind, {"a": 1, "b": 1}, variant=variant) == good

        assert "19.9" in _refused(
            lambda v=variant: decode_per(_extension_encoding(v, b"\x80\xff"), kind, variant=v),
            "a trailing octet inside the extension wrapper",
        )
        assert "19.9" in _refused(
            lambda v=variant: decode_per(_extension_encoding(v, b"\x81"), kind, variant=v),
            "a non-zero pad bit inside the extension wrapper",
        )


def test_the_canonical_rules_did_not_narrow_the_corpus() -> None:
    """Round-trip a spread of the shapes these three fixes touch, under both rule sets.

    A canonical rule that also rejects conforming encodings is worse than the laxity it
    replaced, so each fix is re-checked from the accepting side.
    """
    setof = SetOf(_BYTE, name="S")
    seqof = SequenceOf(_BYTE, name="Q")
    defaulted = Sequence((Component("x", _BIT, default=0), Component("y", _BYTE)), name="D")
    extended = Sequence(
        (Component("a", _BYTE), Component("b", _BIT, extension=True)), name="X", extensible=True
    )
    cases = [
        (setof, ([], [7], [1, 2, 3], [3, 3, 3])),
        (seqof, ([], [7], [3, 1, 2])),
        (defaulted, ({"x": 1, "y": 9}, {"x": 0, "y": 9})),
        (extended, ({"a": 5}, {"a": 5, "b": 0}, {"a": 5, "b": 1})),
    ]
    for variant in _VARIANTS:
        for rules in (PerRules.CANONICAL, PerRules.BASIC):
            for kind, values in cases:
                for value in values:
                    raw = encode_per(kind, value, variant=variant, rules=rules)
                    back = decode_per(raw, kind, variant=variant, rules=rules)
                    if isinstance(value, list) and isinstance(kind, SetOf):
                        assert sorted(back) == sorted(value), (value, raw.hex())
                    else:
                        assert back == value, (kind.name, value, variant.value, raw.hex())
