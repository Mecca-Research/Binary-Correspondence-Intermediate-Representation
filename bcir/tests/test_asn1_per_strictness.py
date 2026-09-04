"""X.691 clause 11: the decoder rejects what no conforming encoder writes.

Five defects reported by review on PR #662 and still live on main. They share one shape --
the codec was permissive where the Recommendation is exact, so a peer had a second spelling
of a value, or a value outside the declared ASN.1 type reached the caller.

  * §11.5.3 -- a constrained whole number's offset must name one of the range's values. A
    range that is not a power of two leaves the widest bit patterns UNUSED; reading one back
    yielded a number outside the type (UNALIGNED `c0` for `INTEGER (0..2)` decoded as 3).
  * §11.8.2 / §11.7.4 -- an unconstrained or semi-constrained whole number occupies "the
    minimum number of octets", and no integer has a zero-octet spelling. `int.from_bytes(b"",
    signed=True)` is 0 in Python, so `b"\\x00"` decoded as a valid zero.
  * §11.8 -- the minimum octet count for a NEGATIVE value is not the one `bit_length()`
    gives, because `bit_length` measures the magnitude. -128 was emitted as `ff80` where `80`
    is minimal, from an encoder whose default is CANONICAL-PER.
  * §11.9 -- the unconstrained length forms carry no bound, so a SIZE whose upper endpoint is
    absent or at/past 64K had its LOWER endpoint enforced nowhere.
  * §11.1.4 -- the complete encoding of an empty field-list is exactly one zero octet. Both
    `b""` and `b"\\xff"` were accepted for a type that encodes to no bits.

Every test below states the encoding it is about, so a future reader can check the claim
against the clause rather than against this file.
"""

from __future__ import annotations

from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.per import PerRules, PerVariant, decode_per, encode_per
from bcir.asn1.schema import Primitive
from bcir.asn1.tags import Universal
from bcir.asn1.values import Asn1Error

_VARIANTS = (PerVariant.UNALIGNED, PerVariant.ALIGNED)


def _refused(fn, what: str) -> str:
    try:
        result = fn()
    except Asn1Error as exc:
        return str(exc)
    raise AssertionError(f"{what}: accepted, returning {result!r}")


def test_a_constrained_offset_outside_the_range_is_refused() -> None:
    """§11.5.3: the offset names one of `range` values, so the unused patterns are malformed.

    `INTEGER (0..2)` has three values and a two-bit field in UNALIGNED PER, which leaves the
    pattern `11` unused. A conforming encoder never writes it; reading it back returned 3, a
    number the type does not contain, at the point where untrusted octets become a value.
    """
    kind = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 2))
    message = _refused(
        lambda: decode_per(b"\xc0", kind, variant=PerVariant.UNALIGNED),
        "the unused bit pattern 11 for INTEGER (0..2)",
    )
    assert "11.5.3" in message

    # The three values the type DOES contain still round-trip, in both variants.
    for variant in _VARIANTS:
        for value in (0, 1, 2):
            assert (
                decode_per(encode_per(kind, value, variant=variant), kind, variant=variant) == value
            )


def test_a_whole_number_needs_at_least_one_contents_octet() -> None:
    """§11.8.2 and §11.7.4: "the minimum number of octets" is never zero of them.

    Python reads `int.from_bytes(b"", signed=True)` as 0 rather than raising, so a zero length
    determinant decoded as a perfectly good zero -- a second spelling of a value whose real
    encoding is `01 00`.
    """
    unconstrained = Primitive(Universal.INTEGER, "INTEGER")
    message = _refused(
        lambda: decode_per(b"\x00", unconstrained, variant=PerVariant.UNALIGNED),
        "a zero-length unconstrained INTEGER",
    )
    assert "11.8.2" in message

    # The genuine encoding of zero is one octet of contents, and it still decodes.
    assert encode_per(unconstrained, 0, variant=PerVariant.UNALIGNED) == b"\x01\x00"
    assert decode_per(b"\x01\x00", unconstrained, variant=PerVariant.UNALIGNED) == 0

    # §11.7.4's semi-constrained form has the same shape and the same fix.
    semi = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, None))
    message = _refused(
        lambda: decode_per(b"\x00", semi, variant=PerVariant.UNALIGNED),
        "a zero-length semi-constrained INTEGER",
    )
    assert "11.7.4" in message


def test_a_negative_unconstrained_integer_uses_the_minimum_octets() -> None:
    """§11.8: two's complement in the minimum octets -- which `bit_length()` mis-measures.

    `bit_length` reports the MAGNITUDE's width, so at exactly the signed boundaries it
    over-counts: 128 needs eight bits and two octets, but -128 needs eight bits and only ONE
    two's-complement octet. `encode_per` defaults to CANONICAL-PER, so the extra octet was a
    non-canonical encoding produced by the canonical encoder.
    """
    kind = Primitive(Universal.INTEGER, "INTEGER")
    U = PerVariant.UNALIGNED

    # The boundaries the old formula got wrong, with the minimal encoding spelled out.
    assert encode_per(kind, -128, variant=U).hex() == "0180"
    assert encode_per(kind, -32768, variant=U).hex() == "028000"
    # And the neighbours it got right, so the fix did not move the boundary the other way.
    assert encode_per(kind, -129, variant=U).hex() == "02ff7f"
    assert encode_per(kind, 127, variant=U).hex() == "017f"
    assert encode_per(kind, 128, variant=U).hex() == "020080"

    for variant in _VARIANTS:
        for value in (
            0,
            1,
            -1,
            127,
            128,
            -127,
            -128,
            -129,
            255,
            256,
            -255,
            -256,
            32767,
            -32768,
            -32769,
            1 << 40,
            -(1 << 40),
        ):
            assert (
                decode_per(encode_per(kind, value, variant=variant), kind, variant=variant) == value
            ), value


def test_an_unconstrained_length_still_honours_its_lower_bound() -> None:
    """§11.9: the unconstrained forms carry no bound, so the type's lower endpoint is ours.

    `_length_bounds` drops an upper endpoint that is absent or at/past 64K, because §11.9.1
    says such a bound is not usable as a constrained determinant. The LOWER endpoint is still
    part of the ASN.1 type, and nothing enforced it -- `OCTET STRING (SIZE(5..MAX))` emitted
    and admitted a one-octet value.
    """
    kind = Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(5, None)))
    U = PerVariant.UNALIGNED

    assert "11.9" in _refused(
        lambda: encode_per(kind, b"x", variant=U), "encoding one octet for SIZE(5..MAX)"
    )
    # b"\x01\x78" is the encoding the old encoder produced: a determinant of 1, then 'x'.
    assert "11.9" in _refused(
        lambda: decode_per(b"\x01\x78", kind, variant=U), "decoding one octet for SIZE(5..MAX)"
    )

    for variant in _VARIANTS:
        for payload in (b"abcde", b"abcdefghij"):
            assert (
                decode_per(encode_per(kind, payload, variant=variant), kind, variant=variant)
                == payload
            )

    # An upper endpoint at 64K is dropped by the same path, so it needs the same guard.
    wide = Primitive(
        Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(5, 1 << 16))
    )
    assert "11.9" in _refused(
        lambda: encode_per(wide, b"x", variant=U), "encoding one octet for SIZE(5..65536)"
    )


def test_an_empty_field_list_is_exactly_one_zero_octet() -> None:
    """§11.1.4: not zero octets, and not an octet the peer chooses.

    A type that contributes no bits -- NULL, or an INTEGER pinned to a single value -- has a
    complete encoding the clause fixes exactly. The old floor said only "at least one octet",
    which admitted `b""` (too short) and `b"\\xff"` (wrong octet) as well as the correct one.
    """
    pinned = Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(7, 7))
    null = Primitive(Universal.NULL, "NULL")

    for variant in _VARIANTS:
        assert encode_per(pinned, 7, variant=variant) == b"\x00"
        assert decode_per(b"\x00", pinned, variant=variant) == 7

        for bad, why in (
            (b"", "zero octets"),
            (b"\xff", "a non-zero octet"),
            (b"\x00\x00", "two octets"),
        ):
            assert "11.1.4" in _refused(
                lambda b=bad, v=variant: decode_per(b, pinned, variant=v),
                f"{why} for an empty field-list",
            )

        # NULL is the other type that reaches this path, and it behaves the same way.
        assert encode_per(null, None, variant=variant) == b"\x00"
        assert "11.1.4" in _refused(
            lambda v=variant: decode_per(b"\xff", null, variant=v), "a non-zero octet for NULL"
        )


def test_the_strictness_did_not_narrow_the_canonical_corpus() -> None:
    """A guard that also rejects valid encodings is a worse bug than the one it fixed.

    So the five clauses above are re-checked from the other side: a spread of types whose
    values sit at and around every boundary the new checks introduce must still round-trip in
    both variants and under both rule sets.
    """
    cases = [
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 2)), (0, 1, 2)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 3)), (0, 3)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 255)), (0, 255)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 256)), (0, 256)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 65535)), (0, 65535)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 1 << 20)), (0, 1 << 20)),
        (Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(7, 7)), (7,)),
        (Primitive(Universal.INTEGER, "INTEGER"), (0, -1, 1, -128, 128)),
        (
            Primitive(Universal.OCTET_STRING, "OCTET STRING", constraint=Size(ValueRange(0, 4))),
            (b"", b"ab", b"abcd"),
        ),
        (Primitive(Universal.OCTET_STRING, "OCTET STRING"), (b"", b"abcdefgh")),
    ]
    for variant in _VARIANTS:
        for rules in (PerRules.CANONICAL, PerRules.BASIC):
            for kind, values in cases:
                for value in values:
                    raw = encode_per(kind, value, variant=variant, rules=rules)
                    back = decode_per(raw, kind, variant=variant, rules=rules)
                    assert back == value, (kind.name, value, variant.value, raw.hex())
