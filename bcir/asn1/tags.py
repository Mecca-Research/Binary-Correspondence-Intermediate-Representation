"""ASN.1 tags and identifier octets — Rec. ITU-T X.690 (02/2021) §8.1.2, X.680 Table 1.

An identifier octet string encodes a tag: its *class* (X.690 Table 1), whether the
encoding is *constructed* (§8.1.2.5), and its *number*. Numbers 0..30 ride a single
octet (§8.1.2.2); 31 and above use the high-tag-number form — a leading octet whose
low five bits are all ones, followed by base-128 continuation octets (§8.1.2.4, as
corrected by Erratum 1 of 09/2021, which only redrew Figure 4).

Everything here is integer/byte work with no third-party dependency, matching the
oracle rail's determinism contract: the same tag always produces the same octets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Asn1Error(ValueError):
    """A malformed or non-conforming ASN.1 encoding (the codec's single error type).

    Carries the octet offset at which the fault was detected so a caller can report
    *where* an untrusted encoding went wrong, not merely that it did.
    """

    def __init__(self, message: str, offset: int = -1) -> None:
        super().__init__(message if offset < 0 else f"{message} (at octet {offset})")
        self.offset = offset


class TagClass(IntEnum):
    """X.690 Table 1 — encoding of class of tag (bits 8 and 7 of the leading octet)."""

    UNIVERSAL = 0b00
    APPLICATION = 0b01
    CONTEXT = 0b10          # "context-specific" in the text
    PRIVATE = 0b11


class Universal(IntEnum):
    """X.680 §8 Table 1 — universal class tag assignments.

    UNIVERSAL 0 is reserved for the encoding rules and is what the end-of-contents
    octets look like (X.690 §8.1.5 NOTE); UNIVERSAL 15 and 37+ are reserved and are
    rejected on decode rather than passed through as an unknown universal type.
    """

    END_OF_CONTENTS = 0
    BOOLEAN = 1
    INTEGER = 2
    BIT_STRING = 3
    OCTET_STRING = 4
    NULL = 5
    OBJECT_IDENTIFIER = 6
    OBJECT_DESCRIPTOR = 7
    EXTERNAL = 8            # also Instance-of (X.690 §8.16.1)
    REAL = 9
    ENUMERATED = 10
    EMBEDDED_PDV = 11
    UTF8_STRING = 12
    RELATIVE_OID = 13
    TIME = 14
    # 15 reserved for future editions
    SEQUENCE = 16           # also Sequence-of
    SET = 17                # also Set-of
    NUMERIC_STRING = 18
    PRINTABLE_STRING = 19
    TELETEX_STRING = 20     # T61String
    VIDEOTEX_STRING = 21
    IA5_STRING = 22
    UTC_TIME = 23
    GENERALIZED_TIME = 24
    GRAPHIC_STRING = 25
    VISIBLE_STRING = 26     # ISO646String
    GENERAL_STRING = 27
    UNIVERSAL_STRING = 28
    CHARACTER_STRING = 29   # the unrestricted character string type
    BMP_STRING = 30
    DATE = 31
    TIME_OF_DAY = 32
    DATE_TIME = 33
    DURATION = 34
    OID_IRI = 35
    RELATIVE_OID_IRI = 36


#: UNIVERSAL numbers X.680 leaves unassigned. Decoding one is a fault, not an
#: extension point: "reserved for future editions"/"reserved for addenda" means a
#: conforming sender never emits it, so accepting it would silently admit garbage.
RESERVED_UNIVERSAL: frozenset[int] = frozenset({15}) | frozenset(range(37, 1 << 20))


@dataclass(frozen=True, order=True)
class Tag:
    """An ASN.1 tag: class, number, and whether this encoding is constructed.

    Ordered so DER's set-component ordering (X.690 §10.3, via X.680 §8.6) and the
    canonical SET OF ordering can sort tags directly. The sort key is (class,
    number) — `constructed` is a property of the *encoding*, not of the tag's
    identity, so it deliberately sorts last.
    """

    cls: TagClass
    number: int
    constructed: bool = False

    def __post_init__(self) -> None:
        if self.number < 0:
            raise Asn1Error(f"tag number must be non-negative, got {self.number}")

    @property
    def is_universal(self) -> bool:
        return self.cls is TagClass.UNIVERSAL

    def as_primitive(self) -> "Tag":
        return Tag(self.cls, self.number, False)

    def as_constructed(self) -> "Tag":
        return Tag(self.cls, self.number, True)

    def __str__(self) -> str:
        if self.is_universal:
            try:
                name = Universal(self.number).name
            except ValueError:
                name = f"UNIVERSAL {self.number}"
            return name
        return f"[{self.cls.name} {self.number}]"


#: The end-of-contents encoding (X.690 §8.1.5): two zero octets, i.e. the identifier
#: octet of a primitive UNIVERSAL 0 followed by a zero length.
EOC = b"\x00\x00"


def encode_tag(tag: Tag) -> bytes:
    """Identifier octets for `tag` (X.690 §8.1.2), always in the fewest octets.

    The short form is used for numbers 0..30 (§8.1.2.2); above that the high-tag-number
    form (§8.1.2.4) emits base-128 groups, most significant first, with bit 8 set on
    every octet but the last. §8.1.2.4.2 c) forbids a leading subsequent octet of all
    zero bits, which is exactly the "no redundant leading group" rule that falls out of
    emitting the minimum number of groups.
    """
    lead = (int(tag.cls) << 6) | (0x20 if tag.constructed else 0x00)
    if tag.number <= 30:
        return bytes([lead | tag.number])
    groups = []
    value = tag.number
    while True:
        groups.append(value & 0x7F)
        value >>= 7
        if not value:
            break
    groups.reverse()
    out = bytearray([lead | 0x1F])
    for i, group in enumerate(groups):
        out.append(group | (0x80 if i < len(groups) - 1 else 0x00))
    return bytes(out)


def decode_tag(data: bytes, pos: int = 0) -> tuple[Tag, int]:
    """Decode identifier octets at `pos`; return (tag, next position).

    Rejects the three ways a high-tag-number form can be malformed: a truncated
    continuation chain, a first subsequent octet of 0x80 (§8.1.2.4.2 c) — a redundant
    leading zero group, the tag-number analogue of a non-minimal integer), and a number
    padded with leading zero groups. A tag number is also bounded: an encoding may name
    an arbitrarily large tag, but a decoder that keeps shifting one in is a denial-of-
    service surface, so anything wider than 32 bits is refused.
    """
    if pos >= len(data):
        raise Asn1Error("truncated identifier octets", pos)
    lead = data[pos]
    cls = TagClass((lead >> 6) & 0b11)
    constructed = bool(lead & 0x20)
    number = lead & 0x1F
    if number != 0x1F:
        return Tag(cls, number, constructed), pos + 1

    start = pos
    pos += 1
    if pos >= len(data):
        raise Asn1Error("truncated high-tag-number identifier octets", start)
    if data[pos] == 0x80:
        raise Asn1Error(
            "high-tag-number form has a redundant leading zero group "
            "(X.690 8.1.2.4.2 c: bits 7 to 1 of the first subsequent octet "
            "shall not all be zero)", start)
    number = 0
    while True:
        if pos >= len(data):
            raise Asn1Error("truncated high-tag-number identifier octets", start)
        octet = data[pos]
        number = (number << 7) | (octet & 0x7F)
        pos += 1
        if number > 0xFFFFFFFF:
            raise Asn1Error("tag number exceeds the 32-bit decode bound", start)
        if not octet & 0x80:
            break
    if number <= 30:
        raise Asn1Error(
            f"tag number {number} used the high-tag-number form but fits the short "
            f"form (X.690 8.1.2.2 applies for numbers 0 to 30)", start)
    return Tag(cls, number, constructed), pos
