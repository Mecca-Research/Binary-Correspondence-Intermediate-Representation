"""ASN.1 length octets — Rec. ITU-T X.690 (02/2021) §8.1.3.

Three forms exist and they are not interchangeable:

* **definite short** (§8.1.3.4) — one octet, bit 8 clear, length 0..127;
* **definite long** (§8.1.3.5) — an initial octet with bit 8 set naming the count of
  subsequent big-endian length octets; 0xFF is reserved and must never be sent;
* **indefinite** (§8.1.3.6) — the single octet 0x80, contents terminated by
  end-of-contents octets. Legal only for a constructed encoding (§8.1.3.2).

BER accepts all three. DER accepts only the definite form in the minimum number of
octets (§10.1); CER inverts that for constructed encodings (§9.1). This module encodes
the minimal definite form and decodes every form, tagging which one it saw so the
strictness layer can rule on it rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tags import Asn1Error

#: The initial octet of the indefinite form (X.690 §8.1.3.6.1).
INDEFINITE = 0x80

#: Refuse a declared length wider than this many octets before allocating anything.
#: A hostile encoding may claim a 2^64-octet body in eight bytes; a decoder that
#: believes it before checking the buffer is a memory-exhaustion surface.
_MAX_LENGTH_OCTETS = 8


@dataclass(frozen=True)
class Length:
    """A decoded length: `None` value means the indefinite form."""

    value: int | None
    #: Octets the length itself occupied — needed to re-derive minimality on decode.
    octets: int
    #: True when the encoder used more octets than X.690 §10.1 would allow.
    non_minimal: bool = False

    @property
    def indefinite(self) -> bool:
        return self.value is None


def encode_length(value: int) -> bytes:
    """The definite form in the fewest octets (X.690 §8.1.3.3 with §10.1's choice).

    This is the only form the encoder emits: DER requires it, and it is a legal BER
    sender's option, so one encoder satisfies both rails.
    """
    if value < 0:
        raise Asn1Error(f"length must be non-negative, got {value}")
    if value <= 127:
        return bytes([value])
    body = value.to_bytes((value.bit_length() + 7) // 8, "big")
    if len(body) > 126:
        # 0xFF is reserved (§8.1.3.5 c), so 0x80|126 is the largest legal initial octet.
        raise Asn1Error(f"length {value} needs more octets than the long form allows")
    return bytes([0x80 | len(body)]) + body


def decode_length(data: bytes, pos: int = 0) -> tuple[Length, int]:
    """Decode length octets at `pos`; return (length, next position).

    Records rather than rejects a non-minimal long form: BER permits it (§8.1.3.5
    NOTE 2 — "it is a sender's option whether to use more length octets than the
    minimum necessary"), and only the DER layer may refuse it. Refusing here would
    make the BER-in half of the contract impossible.
    """
    if pos >= len(data):
        raise Asn1Error("truncated length octets", pos)
    start = pos
    initial = data[pos]
    pos += 1
    if initial == INDEFINITE:
        return Length(None, 1), pos
    if not initial & 0x80:
        return Length(initial, 1), pos
    if initial == 0xFF:
        raise Asn1Error(
            "length initial octet 0xFF is reserved (X.690 8.1.3.5 c)", start)

    count = initial & 0x7F
    if count > _MAX_LENGTH_OCTETS:
        raise Asn1Error(
            f"long-form length declares {count} octets; the decoder bounds it at "
            f"{_MAX_LENGTH_OCTETS} to refuse a hostile size before allocating", start)
    if pos + count > len(data):
        raise Asn1Error("truncated long-form length octets", start)
    body = data[pos:pos + count]
    pos += count
    value = int.from_bytes(body, "big")
    # Minimality per X.690 §10.1: the short form would have sufficed, or the long form
    # carries leading zero octets.
    non_minimal = value <= 127 or (count > 1 and body[0] == 0)
    return Length(value, 1 + count, non_minimal), pos
