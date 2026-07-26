"""The public X.690 codec surface: DER out, BER in.

Two entry points carry the whole contract:

* `encode_der(value)` turns a Python value into DER octets — the only encoding this
  package emits, so a value has exactly one byte string on the wire;
* `decode_der(data, strictness=...)` turns octets back into a Python value, choosing
  how much of BER's sender's-option surface to admit.

`Strictness` is explicit rather than a boolean because "accept BER" and "accept BER
but tell me it was BER" are different operations, and BCIR needs the third one:
`reencode_as_der` accepts a peer's BER and returns the canonical octets, which is how
a foreign artifact enters a system that digests what it stores.
"""

from __future__ import annotations

from enum import Enum

from .der import require_der, to_der
from .tags import Asn1Error, Tag, TagClass, Universal
from .tlv import Tlv, decode_one, encode_tlv
from .values import (
    BitString,
    decode_bitstring,
    decode_boolean,
    decode_integer,
    decode_null,
    decode_octetstring,
    decode_oid,
    decode_real,
    decode_relative_oid,
    decode_string,
    encode_bitstring,
    encode_boolean,
    encode_integer,
    encode_oid,
    encode_real,
    encode_string,
)


class Strictness(Enum):
    """How much of BER's sender's-option surface a decode admits."""

    #: Accept every form X.690 clause 8 permits. Use for bytes from a foreign peer.
    BER = "ber"
    #: Accept only DER (clause 10 + 11). Use at a trust boundary that stores or
    #: digests what it receives — anything else would let a peer choose the digest.
    DER = "der"


# --- Python <-> ASN.1 value mapping -------------------------------------------------
#
# The mapping is deliberately narrow and total: each Python type has exactly one ASN.1
# universal type, so `encode_der` is deterministic without a schema. Anything richer
# (CHOICE, tagged components, OPTIONAL/DEFAULT) is the schema layer's job, because
# those need a type definition to be meaningful.


class Asn1Null:
    """The singleton standing for ASN.1 NULL (Python's `None` means "absent")."""

    _instance: "Asn1Null | None" = None

    def __new__(cls) -> "Asn1Null":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "NULL"


NULL = Asn1Null()


class Oid(tuple):
    """An OBJECT IDENTIFIER, so it is distinguishable from a plain tuple."""

    def __str__(self) -> str:
        return ".".join(str(arc) for arc in self)


class RelativeOid(tuple):
    """A RELATIVE-OID (X.690 §8.20): arcs with no X*40 + Y packing."""

    def __str__(self) -> str:
        return ".".join(str(arc) for arc in self)


class SetOf(list):
    """A SET OF, so encoding can apply the §11.6 ascending-order rule."""


def _universal(number: int, constructed: bool = False) -> Tag:
    return Tag(TagClass.UNIVERSAL, number, constructed)


def to_tlv(value) -> Tlv:
    """Build the `Tlv` tree for a Python value under the mapping above."""
    if value is NULL or isinstance(value, Asn1Null):
        return Tlv(_universal(Universal.NULL), b"")
    if isinstance(value, bool):                       # before int: bool is an int
        return Tlv(_universal(Universal.BOOLEAN), encode_boolean(value))
    if isinstance(value, int):
        return Tlv(_universal(Universal.INTEGER), encode_integer(value))
    if isinstance(value, float):
        return Tlv(_universal(Universal.REAL), encode_real(value))
    if isinstance(value, BitString):
        return Tlv(_universal(Universal.BIT_STRING), encode_bitstring(value))
    if isinstance(value, (bytes, bytearray)):
        return Tlv(_universal(Universal.OCTET_STRING), bytes(value))
    if isinstance(value, str):
        return Tlv(_universal(Universal.UTF8_STRING),
                   encode_string(Universal.UTF8_STRING, value))
    if isinstance(value, Oid):
        return Tlv(_universal(Universal.OBJECT_IDENTIFIER), encode_oid(value))
    if isinstance(value, RelativeOid):
        from .values import encode_relative_oid
        return Tlv(_universal(Universal.RELATIVE_OID), encode_relative_oid(value))
    if isinstance(value, SetOf):
        children = [to_tlv(item) for item in value]
        width = max((len(encode_tlv(c)) for c in children), default=0)
        children.sort(key=lambda c: encode_tlv(c).ljust(width, b"\x00"))   # §11.6
        return Tlv(_universal(Universal.SET, True), b"", children)
    if isinstance(value, (list, tuple)):
        return Tlv(_universal(Universal.SEQUENCE, True), b"",
                   [to_tlv(item) for item in value])
    raise Asn1Error(f"no ASN.1 universal type is mapped to {type(value).__name__}")


def from_tlv(tlv: Tlv, *, strictness: Strictness = Strictness.DER):
    """Recover a Python value from a `Tlv` tree."""
    der = strictness is Strictness.DER
    tag = tlv.tag
    if not tag.is_universal:
        raise Asn1Error(
            f"{tag} has no universal type; decode it through a schema", tlv.offset)
    number = tag.number

    if number == Universal.BOOLEAN:
        return decode_boolean(tlv.content, der=der)
    if number == Universal.INTEGER:
        return decode_integer(tlv.content)
    if number == Universal.ENUMERATED:
        return decode_integer(tlv.content)
    if number == Universal.REAL:
        return decode_real(tlv.content)
    if number == Universal.BIT_STRING:
        return decode_bitstring(tlv, der=der)
    if number == Universal.OCTET_STRING:
        return decode_octetstring(tlv, der=der)
    if number == Universal.NULL:
        decode_null(tlv.content)
        return NULL
    if number == Universal.OBJECT_IDENTIFIER:
        return Oid(decode_oid(tlv.content))
    if number == Universal.RELATIVE_OID:
        return RelativeOid(decode_relative_oid(tlv.content))
    if number == Universal.SEQUENCE:
        return [from_tlv(child, strictness=strictness) for child in tlv.children]
    if number == Universal.SET:
        return SetOf(from_tlv(child, strictness=strictness) for child in tlv.children)
    try:
        return decode_string(tlv, der=der)
    except Asn1Error:
        raise Asn1Error(
            f"{tag} has no decoder in the value mapping; decode it through a schema",
            tlv.offset) from None


# --- the public surface --------------------------------------------------------------


def encode_der(value) -> bytes:
    """DER octets for `value` (X.690 clause 8 restricted by clauses 10 and 11)."""
    return encode_tlv(to_tlv(value))


def decode_der(data: bytes, *, strictness: Strictness = Strictness.DER):
    """Decode exactly one encoding; return the Python value.

    Under `Strictness.DER` the encoding must already be canonical — a peer cannot
    choose among equivalent spellings of the same value, which is what makes a digest
    over the octets meaningful.
    """
    tlv = decode_one(data)
    if strictness is Strictness.DER:
        require_der(tlv)
    return from_tlv(tlv, strictness=strictness)


def decode_value(data: bytes, *, strictness: Strictness = Strictness.BER):
    """`decode_der` with BER admitted by default — for bytes from a foreign peer."""
    return decode_der(data, strictness=strictness)


def reencode_as_der(data: bytes) -> bytes:
    """Accept BER and return the canonical DER octets for the same abstract value.

    This is the BER-in/DER-out conversion as a single call: the point at which a
    foreign encoding becomes something BCIR can digest, store, and replay. Idempotent
    — re-encoding DER returns the identical octets, which the round-trip gate pins.
    """
    return encode_tlv(to_der(decode_one(data)))


__all__ = [
    "NULL", "Asn1Null", "Oid", "RelativeOid", "SetOf", "Strictness", "decode_der",
    "decode_value", "encode_der", "from_tlv", "reencode_as_der", "to_tlv",
]
