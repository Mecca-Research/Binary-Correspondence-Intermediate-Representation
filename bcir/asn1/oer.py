"""Octet Encoding Rules — Rec. ITU-T X.696 (02/2021) | ISO/IEC 8825-7:2021.

OER is the encoding rule in the ASN.1 suite with the **best decode cost**: everything is
octet-aligned, so a decoder never shifts bits, and most fields are fixed-width words the
target can load directly. That is why it is the right default for a driver-side or DMA-fed
path, and why roadmap phase D pairs it with the native fast path.

**COER out, BASIC-OER in** — the same posture X.690 gets. BCIR digests what it emits, so
it emits only CANONICAL-OER (clause 31), which removes every encoder's option; it accepts
BASIC-OER on input, because the interoperability half of the profile is that a peer built
against an ordinary OER toolkit can still talk to BCIR. X.696 §6.5 NOTE 2 makes this
sound: every CANONICAL-OER encoding is a legal BASIC-OER encoding.

WHAT OER DOES NOT HAVE, and why that shapes this module. There are no tags on the wire
except in a CHOICE (§8.7.1), and no lengths except where a clause asks for one. §6.2 is
explicit: *"without knowledge of the type of the value encoded, it is not possible to
determine the structure of the encoding"*. So unlike `der.py`, none of this can be done
schema-free — every function here takes the type. A decoder that guessed would not be a
lenient decoder, it would be a wrong one.

THE CONSTRAINT DEPENDENCY. Clauses 10, 13, 14 and 27 choose between a fixed-width and a
length-prefixed form based on the type's *effective value/size constraint* (§8.2.7,
§8.2.8). BCIR has no constraint model yet (roadmap phase B), so every integer here is
unconstrained and takes §10.4 e) — a length determinant followed by a variable-size
signed number — and every string takes the length-prefixed form. That is not a shortcut:
it is what X.696 specifies for a type with no OER-visible constraints, and phase B will
add the narrower forms without changing anything already emitted for an unconstrained
type.
"""

from __future__ import annotations

from enum import Enum

from .schema import (Asn1Type, Choice, Component, Primitive, Sequence, SequenceOf, Set,
                     SetOf)
from .tags import Asn1Error, Tag, TagClass, Universal

#: X.696 §32.2 — the object identifiers that name these encoding rules.
BASIC_OER_OID: tuple[int, ...] = (2, 1, 6, 0)
CANONICAL_OER_OID: tuple[int, ...] = (2, 1, 6, 1)


class OerRules(Enum):
    """Which of the two rule sets clause 31 distinguishes."""

    BASIC = 0
    CANONICAL = 1


# --- §8.6 the length determinant --------------------------------------------------------

def encode_length(value: int) -> bytes:
    """§8.6.3–§8.6.5, in the canonical form §31.2 requires.

    Short form for 0..127 (a single octet with bit 8 clear); long form otherwise, whose
    initial octet carries bit 8 set and the *count of subsequent octets* in bits 7..1,
    followed by the length as a variable-size unsigned number in the fewest octets.
    """
    if value < 0:
        raise Asn1Error(f"length determinant cannot be negative: {value}")
    if value < 0x80:                                       # §8.6.4 short form
        return bytes([value])
    octets = value.to_bytes((value.bit_length() + 7) // 8, "big")   # §31.2 smallest
    if len(octets) > 0x7F:
        raise Asn1Error("length determinant needs more than 127 octets (X.696 8.6.5)")
    return bytes([0x80 | len(octets)]) + octets


def decode_length(data: bytes, offset: int) -> tuple[int, int]:
    """Read a length determinant; return (value, next offset). Accepts BASIC-OER."""
    if offset >= len(data):
        raise Asn1Error("truncated length determinant", offset)
    first = data[offset]
    if not first & 0x80:                                   # §8.6.4
        return first, offset + 1
    count = first & 0x7F
    if count == 0:
        raise Asn1Error("long-form length determinant with zero subsequent octets "
                        "(X.696 8.6.5)", offset)
    end = offset + 1 + count
    if end > len(data):
        raise Asn1Error("truncated long-form length determinant", offset)
    # BASIC-OER permits redundant leading zero octets (§3.7.12 NOTE); CANONICAL-OER does
    # not (§31.2), but a decoder accepts both -- that is the whole point of "BASIC in".
    return int.from_bytes(data[offset + 1:end], "big"), end


# --- §3.7.11 / §3.7.12 variable-size numbers -------------------------------------------

def _encode_var_signed(value: int) -> bytes:
    """§3.7.11 as a variable-size signed number, in the fewest octets (§31.4).

    The `+ (value < 0)` is the two's-complement asymmetry: each width reaches one further
    negative than positive, so -128 fits one octet while 128 needs two.
    """
    width = ((value + (value < 0)).bit_length() // 8) + 1
    return value.to_bytes(width, "big", signed=True)


def _encode_var_unsigned(value: int) -> bytes:
    """§3.7.12 as a variable-size unsigned number, in the fewest octets (§31.4)."""
    if value < 0:
        raise Asn1Error(f"unsigned number cannot be negative: {value}")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


# --- §8.7 tag encoding (CHOICE alternatives only) --------------------------------------

def encode_tag(tag: Tag) -> bytes:
    """§8.7.2. Bits 8-7 are the class; the number goes in bits 6-1 when it is < 63,
    otherwise those bits are all ones and the number follows base-128, big-endian."""
    lead = int(tag.cls) << 6
    if tag.number < 63:                                    # §8.7.2.2
        return bytes([lead | tag.number])
    chunks = [tag.number & 0x7F]                           # §8.7.2.3
    remaining = tag.number >> 7
    while remaining:
        chunks.append(remaining & 0x7F)
        remaining >>= 7
    chunks.reverse()
    return bytes([lead | 0x3F]) + bytes(
        [chunk | 0x80 for chunk in chunks[:-1]] + [chunks[-1]])


def decode_tag(data: bytes, offset: int) -> tuple[Tag, int]:
    if offset >= len(data):
        raise Asn1Error("truncated tag", offset)
    first = data[offset]
    cls = TagClass(first >> 6)
    if (first & 0x3F) != 0x3F:                             # §8.7.2.2
        return Tag(cls, first & 0x3F), offset + 1
    number, cursor = 0, offset + 1                         # §8.7.2.3
    while True:
        if cursor >= len(data):
            raise Asn1Error("truncated high-tag-number form", offset)
        octet = data[cursor]
        cursor += 1
        number = (number << 7) | (octet & 0x7F)
        if not octet & 0x80:
            break
    return Tag(cls, number), cursor


# --- the primitive types ----------------------------------------------------------------

#: X.696 §27.1 — the known-multiplier character string types. Their per-character octet
#: count is fixed, which is what lets a *size-constrained* one drop its length
#: determinant (§27.2). UTF8String is deliberately absent: a character costs 1..4 octets,
#: so its length is never implied by the character count.
_KNOWN_MULTIPLIER = frozenset({
    Universal.IA5_STRING, Universal.VISIBLE_STRING, Universal.PRINTABLE_STRING,
    Universal.NUMERIC_STRING, Universal.BMP_STRING, Universal.UNIVERSAL_STRING,
})

_STRING_UNIVERSALS = _KNOWN_MULTIPLIER | {
    Universal.UTF8_STRING, Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING,
    Universal.GRAPHIC_STRING, Universal.GENERAL_STRING, Universal.OBJECT_DESCRIPTOR,
}


def _string_octets(universal: int, value: str) -> bytes:
    """§27.4 — the octets that encode a character string value of this type."""
    from .values import encode_string
    return encode_string(universal, value)


def _encode_primitive(kind: Primitive, value, rules: OerRules) -> bytes:
    universal = kind.universal

    if universal == Universal.BOOLEAN:                     # §9
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool, got {type(value).__name__}")
        return b"\xff" if value else b"\x00"               # §31.3: TRUE is 255

    if universal == Universal.INTEGER:                     # §10.4 e), unconstrained
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int, got {type(value).__name__}")
        octets = _encode_var_signed(value)
        return encode_length(len(octets)) + octets

    if universal == Universal.ENUMERATED:                  # §11
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int, got {type(value).__name__}")
        if 0 <= value < 0x80:                              # §11.3 / §31.5 short form
            return bytes([value])
        octets = _encode_var_signed(value)                 # §11.4 long form: SIGNED
        if len(octets) > 0x7F:
            raise Asn1Error("enumerated value needs more than 127 octets (X.696 11.4)")
        return bytes([0x80 | len(octets)]) + octets

    if universal == Universal.NULL:                        # §15
        return b""

    if universal == Universal.OCTET_STRING:                # §14.2, unconstrained
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
        return encode_length(len(value)) + bytes(value)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        # §21 / §22: the length determinant then the BER contents octets, unchanged.
        from .codec import Oid, RelativeOid
        from .values import encode_oid, encode_relative_oid
        arcs = tuple(value.arcs) if isinstance(value, (Oid, RelativeOid)) else tuple(value)
        octets = (encode_oid(arcs) if universal == Universal.OBJECT_IDENTIFIER
                  else encode_relative_oid(arcs))
        return encode_length(len(octets)) + octets

    if universal in _STRING_UNIVERSALS:                    # §27.3, unconstrained
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        octets = _string_octets(universal, value)
        return encode_length(len(octets)) + octets

    raise Asn1Error(f"{kind.name}: universal tag {universal} has no OER encoding here "
                    f"(X.696 clauses 9-30 cover the types this model can express)")


def _decode_primitive(kind: Primitive, data: bytes, offset: int,
                      rules: OerRules) -> tuple[object, int]:
    universal = kind.universal

    if universal == Universal.BOOLEAN:                     # §9
        if offset >= len(data):
            raise Asn1Error("truncated BOOLEAN", offset)
        octet = data[offset]
        if rules is OerRules.CANONICAL and octet not in (0x00, 0xFF):
            raise Asn1Error(f"CANONICAL-OER BOOLEAN must be 0 or 255, got {octet} "
                            f"(X.696 31.3)", offset)
        return octet != 0, offset + 1                      # §9: any non-zero is TRUE

    if universal == Universal.INTEGER:                     # §10.4 e)
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if length == 0 or end > len(data):
            raise Asn1Error("truncated INTEGER", offset)
        return int.from_bytes(data[cursor:end], "big", signed=True), end

    if universal == Universal.ENUMERATED:                  # §11
        if offset >= len(data):
            raise Asn1Error("truncated ENUMERATED", offset)
        first = data[offset]
        if not first & 0x80:                               # §11.3 short form
            return first, offset + 1
        count = first & 0x7F
        end = offset + 1 + count
        if count == 0 or end > len(data):
            raise Asn1Error("truncated ENUMERATED long form", offset)
        return int.from_bytes(data[offset + 1:end], "big", signed=True), end

    if universal == Universal.NULL:                        # §15
        from .codec import NULL
        return NULL, offset

    if universal == Universal.OCTET_STRING:                # §14.2
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if end > len(data):
            raise Asn1Error("truncated OCTET STRING", offset)
        return data[cursor:end], end

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        from .codec import Oid, RelativeOid
        from .values import decode_oid, decode_relative_oid
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if end > len(data):
            raise Asn1Error("truncated OBJECT IDENTIFIER", offset)
        body = data[cursor:end]
        if universal == Universal.OBJECT_IDENTIFIER:
            return Oid(decode_oid(body)), end
        return RelativeOid(decode_relative_oid(body)), end

    if universal in _STRING_UNIVERSALS:                    # §27.3
        # Reuse X.690's string decoder rather than a second copy of the repertoire and
        # width rules: X.696 §27.4 defines the octets by reference to the same character
        # abstract syntaxes, so the octets->str step is identical and only the framing
        # differs. Wrapping them in a primitive TLV is what lets the one implementation
        # serve both rails.
        from .tlv import Tlv
        from .values import decode_string
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if end > len(data):
            raise Asn1Error(f"truncated {kind.name}", offset)
        return decode_string(
            Tlv(Tag(TagClass.UNIVERSAL, universal), data[cursor:end])), end

    raise Asn1Error(f"{kind.name}: universal tag {universal} has no OER decoding here")


# --- §16 / §18 SEQUENCE and SET ---------------------------------------------------------

def _ordered(kind) -> tuple[Component, ...]:
    """§18.2: a SET's components are encoded in the canonical tag order of X.680 §8.6 —
    by tag CLASS first, then tag number. A SEQUENCE keeps its textual order (§16.3).

    This is the rule that makes a SET's encoding independent of how the module happened
    to be written, and it is why `Component` has to carry the tag's class and not just
    its number: `[APPLICATION 1]` sorts before `[0]`, however the author ordered them.
    """
    if not isinstance(kind, Set):
        return kind.components
    def key(comp: Component):
        tag = comp.outer_tag()
        if tag is None:                                    # §18.2: an untagged CHOICE
            alternatives = comp.type.alternative_tags()    # takes its smallest tag
            tag = min(alternatives, key=lambda t: (int(t.cls), t.number))
        return (int(tag.cls), tag.number)
    return tuple(sorted(kind.components, key=key))


def _encode_fields(kind, value: dict, rules: OerRules) -> bytes:
    components = _ordered(kind)
    unknown = set(value) - {c.name for c in kind.components}
    if unknown:
        raise Asn1Error(f"{kind.name}: unknown component(s) {sorted(unknown)}")

    # §16.2: the preamble is the root component presence bitmap, one bit per component
    # marked OPTIONAL or DEFAULT, most significant bit first, zero-padded to whole octets.
    # There is no extension bit because this type model has no extension markers (§16.2.2).
    optional = [c for c in components if c.optional or c.has_default]
    present: dict[str, bool] = {}
    body = bytearray()
    for comp in components:
        if comp.name not in value:
            if not (comp.optional or comp.has_default):
                raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
            present[comp.name] = False
            continue
        item = value[comp.name]
        # §31.9: a DEFAULT component is encoded as ABSENT when it equals its default.
        if comp.has_default and item == comp.default:
            present[comp.name] = False
            continue
        present[comp.name] = True
        body += encode_value(comp.type, item, rules=rules)

    preamble = bytearray()
    if optional:                                           # §16.2.3 / §16.2.4
        bits = "".join("1" if present[c.name] else "0" for c in optional)
        bits += "0" * (-len(bits) % 8)
        preamble += bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return bytes(preamble) + bytes(body)


def _decode_fields(kind, data: bytes, offset: int, rules: OerRules) -> tuple[dict, int]:
    components = _ordered(kind)
    optional = [c for c in components if c.optional or c.has_default]
    cursor = offset
    flags: dict[str, bool] = {c.name: True for c in components}
    if optional:
        width = (len(optional) + 7) // 8
        if cursor + width > len(data):
            raise Asn1Error(f"{kind.name}: truncated preamble (X.696 16.2)", offset)
        bits = "".join(f"{octet:08b}" for octet in data[cursor:cursor + width])
        for index, comp in enumerate(optional):
            flags[comp.name] = bits[index] == "1"
        cursor += width

    out: dict[str, object] = {}
    for comp in components:
        if flags[comp.name]:
            out[comp.name], cursor = decode_value(comp.type, data, cursor, rules=rules)
        elif comp.has_default:
            out[comp.name] = comp.default                  # absent MEANS the default
    return out, cursor


# --- the public entry points ------------------------------------------------------------

def encode_value(kind: Asn1Type, value, *, rules: OerRules = OerRules.CANONICAL) -> bytes:
    """Encode one value of `kind`. Emits CANONICAL-OER unless told otherwise."""
    if isinstance(kind, Primitive):
        return _encode_primitive(kind, value, rules)

    if isinstance(kind, (SequenceOf, SetOf)):              # §17 / §19
        items = list(value)
        encoded = [encode_value(kind.element, item, rules=rules) for item in items]
        if isinstance(kind, SetOf) and rules is OerRules.CANONICAL:
            # §31.8: ascending order as octet strings, the shorter zero-padded for the
            # comparison only -- the padding never appears in the encoding.
            width = max((len(e) for e in encoded), default=0)
            encoded.sort(key=lambda e: e.ljust(width, b"\x00"))
        # §17.2: the quantity field is a length determinant followed by the COUNT as a
        # variable-size unsigned number -- not a bare count, and not the byte length.
        count = _encode_var_unsigned(len(items))
        return encode_length(len(count)) + count + b"".join(encoded)

    if isinstance(kind, (Sequence, Set)):                  # §16 / §18
        if not isinstance(value, dict):
            raise Asn1Error(f"{kind.name}: expected a dict, got {type(value).__name__}")
        return _encode_fields(kind, value, rules)

    if isinstance(kind, Choice):                           # §20.1
        if not (isinstance(value, tuple) and len(value) == 2):
            raise Asn1Error(f"{kind.name}: value must be an (alternative, value) pair")
        chosen, payload = value
        for alt in kind.alternatives:
            if alt.name != chosen:
                continue
            tag = alt.outer_tag()
            if tag is None:                                # §20.1 NOTE 3
                raise Asn1Error(
                    f"{kind.name}: alternative {chosen!r} is an untagged CHOICE; OER "
                    f"needs its outermost tag (X.696 20.1)")
            return encode_tag(tag) + encode_value(alt.type, payload, rules=rules)
        raise Asn1Error(f"{kind.name}: {chosen!r} is not an alternative")

    raise Asn1Error(f"no OER encoding for {type(kind).__name__}")


def decode_value(kind: Asn1Type, data: bytes, offset: int = 0, *,
                 rules: OerRules = OerRules.BASIC) -> tuple[object, int]:
    """Decode one value of `kind` at `offset`; return (value, next offset).

    The default accepts BASIC-OER, which is the interoperability half of the profile.
    Pass `rules=OerRules.CANONICAL` at a trust boundary that stores or digests what it
    receives, since otherwise a peer picks the digest by picking a spelling.
    """
    if isinstance(kind, Primitive):
        return _decode_primitive(kind, data, offset, rules)

    if isinstance(kind, (SequenceOf, SetOf)):              # §17 / §19
        width, cursor = decode_length(data, offset)
        end = cursor + width
        if width == 0 or end > len(data):
            raise Asn1Error(f"{kind.name}: truncated quantity field (X.696 17.2)", offset)
        count = int.from_bytes(data[cursor:end], "big")
        cursor = end
        items = []
        for _ in range(count):
            item, cursor = decode_value(kind.element, data, cursor, rules=rules)
            items.append(item)
        return items, cursor

    if isinstance(kind, (Sequence, Set)):                  # §16 / §18
        return _decode_fields(kind, data, offset, rules)

    if isinstance(kind, Choice):                           # §20.1
        tag, cursor = decode_tag(data, offset)
        for alt in kind.alternatives:
            if any(t.cls is tag.cls and t.number == tag.number
                   for t in alt.expected_tags()):
                item, cursor = decode_value(alt.type, data, cursor, rules=rules)
                return (alt.name, item), cursor
        raise Asn1Error(f"{kind.name}: {tag} matches no alternative (X.696 20.1)", offset)

    raise Asn1Error(f"no OER decoding for {type(kind).__name__}")


def encode_oer(kind: Asn1Type, value, *,
               rules: OerRules = OerRules.CANONICAL) -> bytes:
    """§8.5.2: the complete encoding of an outermost type."""
    return encode_value(kind, value, rules=rules)


def decode_oer(kind: Asn1Type, data: bytes, *,
               rules: OerRules = OerRules.BASIC) -> object:
    """Decode a complete encoding. Trailing octets are an error, not ignored — §6.2 says
    the end of an OER encoding is known only from the type, so leftover octets mean the
    sender and this type disagree."""
    value, cursor = decode_value(kind, data, 0, rules=rules)
    if cursor != len(data):
        raise Asn1Error(
            f"{len(data) - cursor} octet(s) remain after a complete OER encoding", cursor)
    return value


__all__ = ["BASIC_OER_OID", "CANONICAL_OER_OID", "OerRules", "decode_length",
           "decode_oer", "decode_tag", "decode_value", "encode_length", "encode_oer",
           "encode_tag", "encode_value"]
