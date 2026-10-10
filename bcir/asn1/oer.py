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

CONSTRAINTS CHOOSE THE ENCODING. Clauses 10, 13, 14 and 27 select between a fixed-width
and a length-prefixed form from the type's *effective value/size constraint* (§8.2.7,
§8.2.8) — this is where `bcir/asn1/constraints.py` earns its place. `INTEGER (0..255)` is
ONE octet with no length determinant; unconstrained `INTEGER` is a length determinant plus
a variable-size signed number, four times the size for the same abstract value. An
unconstrained type still takes exactly the form it took before constraints existed, so
nothing already emitted moved.

The subtle case is the **extensible** constraint: `(0..255, ...)` is NOT OER-visible
(§8.2.2 g), so it encodes as though unbounded. The marker says the value set may grow in
a later protocol version, and a field sized from today's bounds could not carry
tomorrow's values.
"""

from __future__ import annotations

from enum import Enum

from .codec import NULL, require_null
from .constraints import effective_size_constraint, effective_value_constraint
from .schema import (
    Asn1Type,
    Choice,
    Component,
    OpenType,
    Primitive,
    Reference,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
    through,
)
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
    if value < 0x80:  # §8.6.4 short form
        return bytes([value])
    octets = value.to_bytes((value.bit_length() + 7) // 8, "big")  # §31.2 smallest
    if len(octets) > 0x7F:
        raise Asn1Error("length determinant needs more than 127 octets (X.696 8.6.5)")
    return bytes([0x80 | len(octets)]) + octets


def decode_length(data: bytes, offset: int) -> tuple[int, int]:
    """Read a length determinant; return (value, next offset). Accepts BASIC-OER."""
    if offset >= len(data):
        raise Asn1Error("truncated length determinant", offset)
    first = data[offset]
    if not first & 0x80:  # §8.6.4
        return first, offset + 1
    count = first & 0x7F
    if count == 0:
        raise Asn1Error(
            "long-form length determinant with zero subsequent octets (X.696 8.6.5)", offset
        )
    end = offset + 1 + count
    if end > len(data):
        raise Asn1Error("truncated long-form length determinant", offset)
    # BASIC-OER permits redundant leading zero octets (§3.7.12 NOTE); CANONICAL-OER does
    # not (§31.2), but a decoder accepts both -- that is the whole point of "BASIC in".
    return int.from_bytes(data[offset + 1 : end], "big"), end


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
    if tag.number < 63:  # §8.7.2.2
        return bytes([lead | tag.number])
    chunks = [tag.number & 0x7F]  # §8.7.2.3
    remaining = tag.number >> 7
    while remaining:
        chunks.append(remaining & 0x7F)
        remaining >>= 7
    chunks.reverse()
    return bytes([lead | 0x3F]) + bytes([chunk | 0x80 for chunk in chunks[:-1]] + [chunks[-1]])


def decode_tag(data: bytes, offset: int) -> tuple[Tag, int]:
    if offset >= len(data):
        raise Asn1Error("truncated tag", offset)
    first = data[offset]
    cls = TagClass(first >> 6)
    if (first & 0x3F) != 0x3F:  # §8.7.2.2
        return Tag(cls, first & 0x3F), offset + 1
    number, cursor = 0, offset + 1  # §8.7.2.3
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
_KNOWN_MULTIPLIER = frozenset(
    {
        Universal.IA5_STRING,
        Universal.VISIBLE_STRING,
        Universal.PRINTABLE_STRING,
        Universal.NUMERIC_STRING,
        Universal.BMP_STRING,
        Universal.UNIVERSAL_STRING,
    }
)

_STRING_UNIVERSALS = _KNOWN_MULTIPLIER | {
    Universal.UTF8_STRING,
    Universal.TELETEX_STRING,
    Universal.VIDEOTEX_STRING,
    Universal.GRAPHIC_STRING,
    Universal.GENERAL_STRING,
    Universal.OBJECT_DESCRIPTOR,
}


def _integer_form(kind) -> tuple[int | None, bool]:
    """§10.3 / §10.4: the fixed word width an integer's constraint selects, and its sign.

    Returns `(None, signed)` for the length-prefixed variable-size cases §10.3 e) and
    §10.4 e). The split between the two clauses is whether a lower bound EXISTS and is
    non-negative -- not whether the bounds happen to be small -- so a type with no lower
    bound is signed however tight its upper bound is.
    """
    low, high = effective_value_constraint(getattr(kind, "constraint", None))
    if low is not None and low >= 0:  # §10.2 a) -> §10.3, unsigned
        if high is None:
            return None, False
        for width, limit in ((1, 0xFF), (2, 0xFFFF), (4, 0xFFFFFFFF), (8, 0xFFFFFFFFFFFFFFFF)):
            if high <= limit:
                return width, False
        return None, False  # §10.3 e)
    # §10.2 b) -> §10.4, signed. Both bounds must be present and inside the word.
    if low is None or high is None:
        return None, True
    for width in (1, 2, 4, 8):
        bits = width * 8
        if -(1 << (bits - 1)) <= low and high <= (1 << (bits - 1)) - 1:
            return width, True
    return None, True  # §10.4 e)


def _fixed_size(kind) -> int | None:
    """The single length a SIZE constraint fixes, or None when it does not fix one.

    §13.1, §14.1 and §27.2 all turn on the SAME condition: the effective size
    constraint's lower and upper bounds being *identical*. A range is not enough -- only
    an exact length lets the decoder find the end of the field without a determinant.
    """
    low, high = effective_size_constraint(getattr(kind, "constraint", None))
    if low is not None and low == high:
        return low
    return None


def _string_octets(universal: int, value: str) -> bytes:
    """§27.4 — the octets that encode a character string value of this type."""
    from .values import encode_string

    return encode_string(universal, value)


def _encode_primitive(kind: Primitive, value, rules: OerRules) -> bytes:
    universal = kind.universal

    if universal == Universal.BOOLEAN:  # §9
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool, got {type(value).__name__}")
        return b"\xff" if value else b"\x00"  # §31.3: TRUE is 255

    if universal == Universal.INTEGER:  # §10
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int, got {type(value).__name__}")
        width, signed = _integer_form(kind)
        if width is None:  # §10.3 e) / §10.4 e)
            octets = _encode_var_signed(value) if signed else _encode_var_unsigned(value)
            return encode_length(len(octets)) + octets
        try:
            return value.to_bytes(width, "big", signed=signed)
        except OverflowError:
            raise Asn1Error(
                f"{kind.name}: {value} does not fit the {width}-octet "
                f"{'signed' if signed else 'unsigned'} word its constraint selected "
                f"(X.696 10.3/10.4)"
            ) from None

    if universal == Universal.ENUMERATED:  # §11
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int, got {type(value).__name__}")
        if 0 <= value < 0x80:  # §11.3 / §31.5 short form
            return bytes([value])
        octets = _encode_var_signed(value)  # §11.4 long form: SIGNED
        if len(octets) > 0x7F:
            raise Asn1Error("enumerated value needs more than 127 octets (X.696 11.4)")
        return bytes([0x80 | len(octets)]) + octets

    if universal == Universal.NULL:  # §15
        require_null(value, kind.name)
        return b""

    if universal == Universal.OCTET_STRING:  # §14
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
        fixed = _fixed_size(kind)
        if fixed is not None:  # §14.1: no length determinant
            if len(value) != fixed:
                raise Asn1Error(
                    f"{kind.name}: SIZE ({fixed}) requires exactly {fixed} octets, got "
                    f"{len(value)} (X.696 14.1)"
                )
            return bytes(value)
        return encode_length(len(value)) + bytes(value)  # §14.2

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        # §21 / §22: the length determinant then the BER contents octets, unchanged.
        from .values import encode_oid, encode_relative_oid

        # `Oid` and `RelativeOid` are tuple subclasses, so an arc sequence is exactly
        # what they already are -- no unwrapping step.
        arcs = tuple(value)
        octets = (
            encode_oid(arcs)
            if universal == Universal.OBJECT_IDENTIFIER
            else encode_relative_oid(arcs)
        )
        return encode_length(len(octets)) + octets

    if universal in _STRING_UNIVERSALS:  # §27
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        octets = _string_octets(universal, value)
        # §27.2 drops the length determinant only for a KNOWN-MULTIPLIER type whose
        # effective size constraint is a single value: only then does the character count
        # fix the octet count. UTF8String never qualifies (§27.1) -- a character costs
        # 1..4 octets there, so its length is never implied.
        if universal in _KNOWN_MULTIPLIER and _fixed_size(kind) is not None:
            fixed = _fixed_size(kind)
            if len(value) != fixed:
                raise Asn1Error(
                    f"{kind.name}: SIZE ({fixed}) requires exactly {fixed} characters, "
                    f"got {len(value)} (X.696 27.2)"
                )
            return octets
        return encode_length(len(octets)) + octets  # §27.3

    raise Asn1Error(
        f"{kind.name}: universal tag {universal} has no OER encoding here "
        f"(X.696 clauses 9-30 cover the types this model can express)"
    )


def _decode_primitive(
    kind: Primitive, data: bytes, offset: int, rules: OerRules
) -> tuple[object, int]:
    universal = kind.universal

    if universal == Universal.BOOLEAN:  # §9
        if offset >= len(data):
            raise Asn1Error("truncated BOOLEAN", offset)
        octet = data[offset]
        if rules is OerRules.CANONICAL and octet not in (0x00, 0xFF):
            raise Asn1Error(
                f"CANONICAL-OER BOOLEAN must be 0 or 255, got {octet} (X.696 31.3)", offset
            )
        return octet != 0, offset + 1  # §9: any non-zero is TRUE

    if universal == Universal.INTEGER:  # §10
        width, signed = _integer_form(kind)
        if width is not None:
            end = offset + width
            if end > len(data):
                raise Asn1Error("truncated fixed-width INTEGER", offset)
            return int.from_bytes(data[offset:end], "big", signed=signed), end
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if length == 0 or end > len(data):
            raise Asn1Error("truncated INTEGER", offset)
        return int.from_bytes(data[cursor:end], "big", signed=signed), end

    if universal == Universal.ENUMERATED:  # §11
        if offset >= len(data):
            raise Asn1Error("truncated ENUMERATED", offset)
        first = data[offset]
        if not first & 0x80:  # §11.3 short form
            return first, offset + 1
        count = first & 0x7F
        end = offset + 1 + count
        if count == 0 or end > len(data):
            raise Asn1Error("truncated ENUMERATED long form", offset)
        return int.from_bytes(data[offset + 1 : end], "big", signed=True), end

    if universal == Universal.NULL:  # §15
        return NULL, offset

    if universal == Universal.OCTET_STRING:  # §14
        fixed = _fixed_size(kind)
        if fixed is not None:  # §14.1
            end = offset + fixed
            if end > len(data):
                raise Asn1Error("truncated fixed-size OCTET STRING", offset)
            return data[offset:end], end
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

    if universal in _STRING_UNIVERSALS:  # §27.3
        # Reuse X.690's string decoder rather than a second copy of the repertoire and
        # width rules: X.696 §27.4 defines the octets by reference to the same character
        # abstract syntaxes, so the octets->str step is identical and only the framing
        # differs. Wrapping them in a primitive TLV is what lets the one implementation
        # serve both rails.
        from .tlv import Tlv
        from .values import decode_string

        fixed = _fixed_size(kind)
        if universal in _KNOWN_MULTIPLIER and fixed is not None:  # §27.2
            width = {Universal.BMP_STRING: 2, Universal.UNIVERSAL_STRING: 4}.get(universal, 1)
            end = offset + fixed * width
            if end > len(data):
                raise Asn1Error(f"truncated fixed-size {kind.name}", offset)
            return decode_string(Tlv(Tag(TagClass.UNIVERSAL, universal), data[offset:end])), end
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if end > len(data):
            raise Asn1Error(f"truncated {kind.name}", offset)
        return decode_string(Tlv(Tag(TagClass.UNIVERSAL, universal), data[cursor:end])), end

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
        if tag is None:  # §18.2: an untagged CHOICE
            alternatives = comp.type.alternative_tags()  # takes its smallest tag
            tag = min(alternatives, key=lambda t: (int(t.cls), t.number))
        return (int(tag.cls), tag.number)

    return tuple(sorted(kind.components, key=key))


def _split_root(kind) -> tuple[tuple[Component, ...], tuple[Component, ...]]:
    """(extension root, extension additions), each in encoding order (X.680 §25.1).

    The split is the same one PER's `_split_root` makes, and for the same reason: X.696
    encodes the two halves in different places. Root components follow the preamble;
    additions follow the root, behind their own presence bitmap, each wrapped as an open type.
    """
    components = _ordered(kind)
    root = tuple(c for c in components if not c.extension)
    additions = tuple(c for c in components if c.extension)
    return root, additions


def _supplied(comp: Component, value: dict) -> bool:
    """Whether a component's encoding is present. §31.9's DEFAULT rule applies on both rule
    sets here, as it always has on this rail's encoder (COER out); a version bracket is
    present when any of its members is (X.696 16.5, as X.691 19.9 says it for PER)."""
    if comp.group is not None:
        return any(_supplied(member, value) for member in comp.group)
    if comp.name not in value:
        return False
    return not (comp.has_default and value[comp.name] == comp.default)


def _bits_to_octets(bits: list[int]) -> bytes:
    padded = bits + [0] * (-len(bits) % 8)
    return bytes(
        int("".join(str(b) for b in padded[i : i + 8]), 2) for i in range(0, len(padded), 8)
    )


def _addition_value(comp: Component, value: dict):
    # A version bracket's members live FLAT in the parent value (as on the PER rail), and the
    # bracket is encoded as a SEQUENCE of them; a plain addition is its own value.
    if comp.group is not None:
        return {m.name: value[m.name] for m in comp.group if m.name in value}
    return value[comp.name]


def _encode_fields(kind, value: dict, rules: OerRules) -> bytes:
    """§16 (§18 for SET): the preamble, the root components, then the extension additions.

    **The extension bit is the preamble's FIRST bit.** §16.2.2: when the type has an extension
    marker the preamble begins with a bit saying whether any extension addition is present,
    and the root presence bitmap follows it. This rail used to write the bitmap alone,
    reasoning that "this type model has no extension markers" -- true when it was written,
    false once `Sequence.extensible` existed. From then on `SEQUENCE { a INTEGER (0..255)
    OPTIONAL, ... }` with `a` present encoded as `80 ..` where X.696 spells it `40 ..`, a
    well-formed document a conforming peer reads as an extension addition being present. The
    PER rail had the bit all along (X.691 §19.1), which is what made the omission visible.
    """
    root, additions = _split_root(kind)
    known = {c.name for c in root + additions}
    for comp in additions:
        if comp.group is not None:
            known.update(member.name for member in comp.group)
    unknown = set(value) - known
    if unknown:
        raise Asn1Error(f"{kind.name}: unknown component(s) {sorted(unknown)}")

    present_additions = [c for c in additions if _supplied(c, value)]
    if present_additions and not kind.extensible:  # pragma: no cover - the lowering guards it
        raise Asn1Error(f"{kind.name}: extension additions on a non-extensible type")

    bits: list[int] = []
    if kind.extensible:  # §16.2.2
        bits.append(1 if present_additions else 0)
    body = bytearray()
    for comp in root:
        if comp.name not in value:
            if not (comp.optional or comp.has_default):
                raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
            bits.append(0)
            continue
        if comp.optional or comp.has_default:  # §16.2.3: one bit per OPTIONAL/DEFAULT root
            # §31.9: a DEFAULT component is encoded as ABSENT when it equals its default.
            here = _supplied(comp, value)
            bits.append(1 if here else 0)
            if not here:
                continue
        body += encode_value(comp.type, value[comp.name], rules=rules)

    # §16.2.4: no bits, no preamble; otherwise zero-padded to whole octets.
    out = (_bits_to_octets(bits) if bits else b"") + bytes(body)
    if not present_additions:
        return out
    # §16.4: the extension addition presence bitmap, encoded as a variable-size BIT STRING
    # (§15.3: a length determinant over the initial octet and the bitmap, the initial octet
    # counting the unused trailing bits). It has one bit per addition IN THE TYPE, so a
    # decoder of an older version knows how many open types follow.
    flags = [1 if _supplied(c, value) else 0 for c in additions]
    bitmap = _bits_to_octets(flags)
    out += encode_length(1 + len(bitmap)) + bytes([(-len(flags)) % 8]) + bitmap
    for comp in present_additions:  # §16.5: each present addition as an open type (§30)
        inner = encode_value(comp.type, _addition_value(comp, value), rules=rules)
        out += encode_length(len(inner)) + inner
    return out


def _decode_open_type(data: bytes, offset: int, what: str) -> tuple[bytes, int]:
    """§30: a length determinant and that many octets, returned for a nested decode."""
    length, cursor = decode_length(data, offset)
    end = cursor + length
    if end > len(data):
        raise Asn1Error(f"{what}: truncated open type (X.696 30)", offset)
    return data[cursor:end], end


def _decode_fields(kind, data: bytes, offset: int, rules: OerRules) -> tuple[dict, int]:
    root, additions = _split_root(kind)
    optional = [c for c in root if c.optional or c.has_default]
    width_bits = len(optional) + (1 if kind.extensible else 0)
    cursor = offset
    flags: dict[str, bool] = {c.name: True for c in root}
    extended = False
    if width_bits:
        width = (width_bits + 7) // 8
        if cursor + width > len(data):
            raise Asn1Error(f"{kind.name}: truncated preamble (X.696 16.2)", offset)
        bits = "".join(f"{octet:08b}" for octet in data[cursor : cursor + width])
        at = 0
        if kind.extensible:  # §16.2.2: the extension bit leads
            extended = bits[0] == "1"
            at = 1
        for comp in optional:
            flags[comp.name] = bits[at] == "1"
            at += 1
        # §16.2.4 pads the preamble to an octet with ZERO bits. A set padding bit leaves every
        # presence bit the same, so under CANONICAL it is a second spelling of one value -- the
        # C plan decoder reports it as non-canonical and the generated codecs refuse it; this
        # rail accepted it as canonical. BASIC keeps accepting it, as the plan decoder does.
        if rules is OerRules.CANONICAL and "1" in bits[width_bits:]:
            raise Asn1Error(
                f"{kind.name}: the preamble's padding bits are not zero; CANONICAL-OER "
                f"writes them as zero (X.696 16.2.4)",
                offset,
            )
        cursor += width

    out: dict[str, object] = {}
    for comp in root:
        if flags[comp.name]:
            out[comp.name], cursor = decode_value(comp.type, data, cursor, rules=rules)
            # §31.9: under CANONICAL-OER "each component that is marked DEFAULT shall be
            # encoded as absent if its value is identical to the default value". The encoder
            # has always obeyed it; the decoder did not, so `80 01 01 01 07` and `00 01 07`
            # both decoded to the same abstract value under the CANONICAL rules. A canonical
            # encoding that admits two spellings of one value is not canonical, and BCIR
            # digests these octets. BASIC-OER keeps accepting it -- §31 is a clause-31
            # restriction, not a clause-16 one.
            if rules is OerRules.CANONICAL and comp.has_default and out[comp.name] == comp.default:
                raise Asn1Error(
                    f"{kind.name}: component {comp.name!r} is present and equal to its "
                    f"DEFAULT {comp.default!r}; CANONICAL-OER encodes it as absent "
                    f"(X.696 31.9)",
                    offset,
                )
        elif comp.has_default:
            out[comp.name] = comp.default  # absent MEANS the default

    if extended:
        # §16.4: the presence bitmap of the additions, as a variable-size BIT STRING.
        bitmap, cursor = _decode_open_type(data, cursor, f"{kind.name} addition bitmap")
        if not bitmap or bitmap[0] > 7 or (len(bitmap) == 1 and bitmap[0]):
            raise Asn1Error(
                f"{kind.name}: malformed extension addition presence bitmap (X.696 16.4)",
                cursor,
            )
        count = (len(bitmap) - 1) * 8 - bitmap[0]
        # The bitmap's unused trailing bits, like the preamble's padding: zero, or under
        # CANONICAL a second spelling of the same additions.
        if rules is OerRules.CANONICAL and bitmap[-1] & ((1 << bitmap[0]) - 1):
            raise Asn1Error(
                f"{kind.name}: the addition bitmap's {bitmap[0]} unused bit(s) are not zero; "
                f"CANONICAL-OER writes them as zero (X.696 16.4)",
                cursor,
            )
        present = [
            index for index in range(count) if bitmap[1 + index // 8] & (0x80 >> (index % 8))
        ]
        if not present:
            # The extension bit said an addition is present and the bitmap names none: two
            # spellings of "no additions", and §16.2.2 sets the bit only when one is there.
            raise Asn1Error(
                f"{kind.name}: the extension bit is set but the addition bitmap is empty "
                f"(X.696 16.2.2)",
                cursor,
            )
        for index in present:  # §16.5: each one an open type
            octets, cursor = _decode_open_type(data, cursor, f"{kind.name} addition {index}")
            if index >= len(additions):
                # A peer built against a NEWER version sent an addition this type does not
                # know. The open type's length is precisely what makes that recoverable: the
                # octets are skipped, not misparsed (the PER rail's posture, X.691 19.9).
                continue
            comp = additions[index]
            item, used = decode_value(comp.type, octets, 0, rules=rules)
            if used != len(octets):
                raise Asn1Error(
                    f"{kind.name}: {len(octets) - used} octet(s) after addition "
                    f"{comp.name!r} inside its open type (X.696 16.5)",
                    cursor,
                )
            if comp.group is not None:
                out.update(item)  # a version bracket's members are flat
            else:
                # §31.9 holds for an addition exactly as for a root component (checked above):
                # the encoder leaves one equal to its DEFAULT out, so sent anyway it is the
                # second spelling of the same value.
                if rules is OerRules.CANONICAL and comp.has_default and item == comp.default:
                    raise Asn1Error(
                        f"{kind.name}: addition {comp.name!r} is present and equal to its "
                        f"DEFAULT {comp.default!r}; CANONICAL-OER encodes it as absent "
                        f"(X.696 31.9)",
                        cursor,
                    )
                out[comp.name] = item
    for comp in additions:
        if comp.group is None and comp.name not in out and comp.has_default:
            out[comp.name] = comp.default
    return out, cursor


# --- the public entry points ------------------------------------------------------------


def encode_value(kind: Asn1Type, value, *, rules: OerRules = OerRules.CANONICAL) -> bytes:
    """Encode one value of `kind`. Emits CANONICAL-OER unless told otherwise."""
    if isinstance(kind, Primitive):
        return _encode_primitive(kind, value, rules)

    if isinstance(kind, (SequenceOf, SetOf)):  # §17 / §19
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

    if isinstance(kind, (Sequence, Set)):  # §16 / §18
        if not isinstance(value, dict):
            raise Asn1Error(f"{kind.name}: expected a dict, got {type(value).__name__}")
        return _encode_fields(kind, value, rules)

    if isinstance(kind, OpenType):  # §30
        # An open type is a length determinant then the contained value's encoding. The
        # contained TYPE is unknown here by definition, so the octets pass through
        # unchanged rather than being re-encoded -- re-encoding would require knowing the
        # type, and guessing it is the one thing an open type forbids.
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: an open type value must be bytes")
        return encode_length(len(value)) + bytes(value)

    if isinstance(kind, Choice):  # §20.1
        if not (isinstance(value, tuple) and len(value) == 2):
            raise Asn1Error(f"{kind.name}: value must be an (alternative, value) pair")
        chosen, payload = value
        for alt in kind.alternatives:
            if alt.name != chosen:
                continue
            tag = alt.outer_tag()
            if tag is None:  # §20.1 NOTE 3
                raise Asn1Error(
                    f"{kind.name}: alternative {chosen!r} is an untagged CHOICE; OER "
                    f"needs its outermost tag (X.696 20.1)"
                )
            inner = encode_value(alt.type, payload, rules=rules)
            if alt.extension:
                # §20.2: an extension alternative's value is an OPEN TYPE, so a decoder of
                # an older version can skip what it cannot read. Writing it bare (as this
                # rail did) made the octets after it unreadable to such a decoder.
                return encode_tag(tag) + encode_length(len(inner)) + inner
            return encode_tag(tag) + inner
        raise Asn1Error(f"{kind.name}: {chosen!r} is not an alternative")

    if isinstance(kind, Reference):  # a recursive definition: the shared resolve and bound
        return through("OER", kind, encode_value, kind.resolved(), value, rules=rules)
    raise Asn1Error(f"no OER encoding for {type(kind).__name__}")


def decode_value(
    kind: Asn1Type, data: bytes, offset: int = 0, *, rules: OerRules = OerRules.BASIC
) -> tuple[object, int]:
    """Decode one value of `kind` at `offset`; return (value, next offset).

    The default accepts BASIC-OER, which is the interoperability half of the profile.
    Pass `rules=OerRules.CANONICAL` at a trust boundary that stores or digests what it
    receives, since otherwise a peer picks the digest by picking a spelling.
    """
    if isinstance(kind, Primitive):
        return _decode_primitive(kind, data, offset, rules)

    if isinstance(kind, (SequenceOf, SetOf)):  # §17 / §19
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

    if isinstance(kind, (Sequence, Set)):  # §16 / §18
        return _decode_fields(kind, data, offset, rules)

    if isinstance(kind, OpenType):  # §30
        length, cursor = decode_length(data, offset)
        end = cursor + length
        if end > len(data):
            raise Asn1Error(f"{kind.name}: truncated open type (X.696 30)", offset)
        return data[cursor:end], end

    if isinstance(kind, Choice):  # §20.1
        tag, cursor = decode_tag(data, offset)
        for alt in kind.alternatives:
            if any(t.cls is tag.cls and t.number == tag.number for t in alt.expected_tags()):
                if not alt.extension:
                    item, cursor = decode_value(alt.type, data, cursor, rules=rules)
                    return (alt.name, item), cursor
                # §20.2: an extension alternative arrives as an open type.
                octets, cursor = _decode_open_type(data, cursor, f"{kind.name}.{alt.name}")
                item, used = decode_value(alt.type, octets, 0, rules=rules)
                if used != len(octets):
                    raise Asn1Error(
                        f"{kind.name}: {len(octets) - used} octet(s) after {alt.name!r} "
                        f"inside its open type (X.696 20.2)",
                        offset,
                    )
                return (alt.name, item), cursor
        if kind.extensible:
            # An alternative added by a newer version. Its open type could be skipped, but
            # a CHOICE has exactly one value and this type has no name for it, so the
            # posture is the PER rail's (X.691 23.8): refuse by name rather than invent.
            raise Asn1Error(
                f"{kind.name}: {tag} is an extension alternative unknown to this version "
                f"(X.696 20.2)",
                offset,
            )
        raise Asn1Error(f"{kind.name}: {tag} matches no alternative (X.696 20.1)", offset)

    if isinstance(kind, Reference):  # a recursive definition: the shared resolve and bound
        return through("OER", kind, decode_value, kind.resolved(), data, offset, rules=rules)
    raise Asn1Error(f"no OER decoding for {type(kind).__name__}")


def encode_oer(kind: Asn1Type, value, *, rules: OerRules = OerRules.CANONICAL) -> bytes:
    """§8.5.2: the complete encoding of an outermost type."""
    return encode_value(kind, value, rules=rules)


def decode_oer(kind: Asn1Type, data: bytes, *, rules: OerRules = OerRules.BASIC) -> object:
    """Decode a complete encoding. Trailing octets are an error, not ignored — §6.2 says
    the end of an OER encoding is known only from the type, so leftover octets mean the
    sender and this type disagree."""
    value, cursor = decode_value(kind, data, 0, rules=rules)
    if cursor != len(data):
        raise Asn1Error(
            f"{len(data) - cursor} octet(s) remain after a complete OER encoding", cursor
        )
    return value


__all__ = [
    "BASIC_OER_OID",
    "CANONICAL_OER_OID",
    "OerRules",
    "decode_length",
    "decode_oer",
    "decode_tag",
    "decode_value",
    "encode_length",
    "encode_oer",
    "encode_tag",
    "encode_value",
]
