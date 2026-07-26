"""X.691 (2021) Packed Encoding Rules — ALIGNED and UNALIGNED, BASIC and CANONICAL.

PER is the suite's first *compact* encoding, and it differs from BER/DER/OER in three
structural ways that shape this whole module:

1. **No tags.** X.691 §10.4.1/§10.6.3: type references and tagging "have no effect on the
   encoding and are invisible in the model", except for the canonical ordering of §10.2.
   So a `Component(tag=N)` in our schema contributes nothing to a PER encoding, and a PER
   encoding is not self-delimiting without the type (§7.2).
2. **Bits, not octets.** The unit of composition is a *field-list* of bit-fields (§10.5).
   Only the outermost value is padded out to an octet boundary (§11.1.3/§11.1.4).
3. **Constraints choose the encoding.** A PER-visible constraint (§10.3) is not a
   validation rule here, it is the *width*. `INTEGER (0..255)` occupies exactly eight bits
   with no length determinant; the same type with no constraint costs a length determinant
   plus a minimum-octets 2's-complement field.

The ALIGNED/UNALIGNED split is a genuine cost trade rather than a configuration: ALIGNED
inserts padding so that multi-octet fields start on octet boundaries (cheaper to read,
larger on the wire), UNALIGNED never pads (smaller, more bit-shifting). Both are emitted.

The canonical/basic split follows the same "DER out, BER in" discipline the rest of the
rail uses: BCIR emits CANONICAL-PER and accepts BASIC-PER. Per §7.5, CANONICAL-PER is a
restriction of BASIC-PER's implementation-dependent choices, so every canonical encoding is
already a legal basic one; the one place the two differ for our schemas is §19.5 (a DEFAULT
component whose value equals the default must be absent under CANONICAL-PER).
"""

from __future__ import annotations

from enum import Enum

from .codec import Asn1Error
from .constraints import UNBOUNDED, root_size_bounds, root_value_bounds
from .schema import (
    Asn1Type,
    Choice,
    Component,
    OpenType,
    Primitive,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
)
from .tags import Tag, Universal

# X.691 §33.2. joint-iso-itu-t(2) asn1(1) packed-encoding(3), then basic(0)/canonical(1)
# and aligned(0)/unaligned(1). Same shape as the OER pair in `oer.py`.
BASIC_PER_ALIGNED_OID: tuple[int, ...] = (2, 1, 3, 0, 0)
BASIC_PER_UNALIGNED_OID: tuple[int, ...] = (2, 1, 3, 0, 1)
CANONICAL_PER_ALIGNED_OID: tuple[int, ...] = (2, 1, 3, 1, 0)
CANONICAL_PER_UNALIGNED_OID: tuple[int, ...] = (2, 1, 3, 1, 1)

#: §11.9.3.8: fragmentation kicks in at 16K units and moves in multiples of 16K.
_FRAG_UNIT = 16 * 1024
#: §11.9.3.6/§11.9.3.7 thresholds for the one- and two-octet unconstrained length forms.
_LEN_1_OCTET_MAX = 127
_LEN_2_OCTET_MAX = 16 * 1024
#: §11.9.1/§20.2: 64K is the point past which an upper bound stops being usable as a
#: constrained length determinant and the unconstrained form is used instead.
_64K = 64 * 1024


class PerVariant(Enum):
    """§7.7: the two variants. They do not interwork (§7.8)."""

    ALIGNED = "aligned"
    UNALIGNED = "unaligned"


class PerRules(Enum):
    """§7.4/§7.5: BASIC-PER accepted on input, CANONICAL-PER emitted."""

    BASIC = "basic"
    CANONICAL = "canonical"


def rules_oid(rules: PerRules, variant: PerVariant) -> tuple[int, ...]:
    """The §33.2 object identifier naming one of the four algorithms."""
    return {
        (PerRules.BASIC, PerVariant.ALIGNED): BASIC_PER_ALIGNED_OID,
        (PerRules.BASIC, PerVariant.UNALIGNED): BASIC_PER_UNALIGNED_OID,
        (PerRules.CANONICAL, PerVariant.ALIGNED): CANONICAL_PER_ALIGNED_OID,
        (PerRules.CANONICAL, PerVariant.UNALIGNED): CANONICAL_PER_UNALIGNED_OID,
    }[(rules, variant)]


def bits_for_range(range_: int) -> int:
    """§11.5.6: the minimum number of bits that can represent `range` distinct values.

    The clause states it as an inequality -- "if 2^m < range <= 2^(m+1) then the number of
    bits = m + 1" -- which is ceil(log2(range)). §11.5.4 makes a range of 1 encode to an
    empty bit-field, i.e. zero bits. The intact ALIGNED table in §11.5.7.1 is the same
    function (range 2 -> 1 bit, 3..4 -> 2, 5..8 -> 3, ... 129..255 -> 8), which is what
    pins the reading: the two clauses have to agree on the bit-field case.
    """
    if range_ <= 1:
        return 0
    return (range_ - 1).bit_length()


class BitWriter:
    """A field-list under construction (§10.5): bits, plus the ALIGNED padding rule.

    `put_bits` appends an unaligned bit-field. `align()` implements §11.1.4 -- in the
    ALIGNED variant, zero bits are inserted so that an octet-aligned bit-field starts on an
    octet boundary; in the UNALIGNED variant it is a no-op (§11.1.3, "all fields shall be
    concatenated without padding").
    """

    __slots__ = ("bits", "variant")

    def __init__(self, variant: PerVariant) -> None:
        self.bits: list[int] = []
        self.variant = variant

    def __len__(self) -> int:
        return len(self.bits)

    def put_bit(self, bit: int) -> None:
        self.bits.append(1 if bit else 0)

    def put_bits(self, value: int, width: int) -> None:
        """§11.3: non-negative-binary-integer into a bit-field of `width` bits."""
        if width < 0:
            raise Asn1Error("PER: negative bit-field width")
        if width and (value < 0 or value >> width):
            raise Asn1Error(f"PER: value {value} does not fit in {width} bits")
        for shift in range(width - 1, -1, -1):
            self.bits.append((value >> shift) & 1)

    def put_octets(self, data: bytes) -> None:
        for byte in data:
            self.put_bits(byte, 8)

    def align(self) -> None:
        if self.variant is PerVariant.ALIGNED:
            while len(self.bits) % 8:
                self.bits.append(0)

    def to_bytes(self) -> bytes:
        """§11.1.3.1/§11.1.4: pad the complete encoding out to a whole number of octets.

        An empty encoding becomes a single zero octet -- an outermost value always occupies
        at least one octet, even when the type carries no information (a NULL, or an INTEGER
        constrained to a single value).
        """
        bits = self.bits
        if not bits:
            return b"\x00"
        pad = (-len(bits)) % 8
        packed = bits + [0] * pad
        out = bytearray(len(packed) // 8)
        for index, bit in enumerate(packed):
            if bit:
                out[index >> 3] |= 0x80 >> (index & 7)
        return bytes(out)


class BitReader:
    """The decoding counterpart of `BitWriter`, over a fixed octet string."""

    __slots__ = ("data", "pos", "variant")

    def __init__(self, data: bytes, variant: PerVariant) -> None:
        self.data = data
        self.pos = 0
        self.variant = variant

    @property
    def _limit(self) -> int:
        return len(self.data) * 8

    def get_bit(self) -> int:
        if self.pos >= self._limit:
            raise Asn1Error("PER: truncated encoding (ran out of bits)")
        byte = self.data[self.pos >> 3]
        bit = (byte >> (7 - (self.pos & 7))) & 1
        self.pos += 1
        return bit

    def get_bits(self, width: int) -> int:
        if width < 0:
            raise Asn1Error("PER: negative bit-field width")
        if self.pos + width > self._limit:
            raise Asn1Error("PER: truncated encoding (bit-field runs past the end)")
        value = 0
        for _ in range(width):
            value = (value << 1) | self.get_bit()
        return value

    def get_octets(self, count: int) -> bytes:
        if count < 0:
            raise Asn1Error("PER: negative octet count")
        return bytes(self.get_bits(8) for _ in range(count))

    def align(self) -> None:
        if self.variant is PerVariant.ALIGNED:
            while self.pos % 8:
                # §11.1.4 pads with ZERO bits. A non-zero pad bit is a malformed encoding,
                # not a spelling choice -- refusing it keeps the decoder from silently
                # accepting two distinct octet strings as the same abstract value.
                if self.get_bit():
                    raise Asn1Error("PER: non-zero pad bit before an octet-aligned field")


# --------------------------------------------------------------------------------------
# §11.5-§11.8: the whole-number encodings.
# --------------------------------------------------------------------------------------


def _encode_constrained(writer: BitWriter, value: int, lower: int, upper: int) -> None:
    """§11.5: a constrained whole number, in whichever of the five cases applies."""
    if value < lower or value > upper:
        raise Asn1Error(f"PER: {value} outside the constrained range {lower}..{upper}")
    range_ = upper - lower + 1                       # §11.5.3
    if range_ == 1:                                  # §11.5.4: empty bit-field
        return
    offset = value - lower
    if writer.variant is PerVariant.UNALIGNED:       # §11.5.6
        writer.put_bits(offset, bits_for_range(range_))
        return
    if range_ <= 255:                                # §11.5.7.1 the bit-field case
        writer.put_bits(offset, bits_for_range(range_))
    elif range_ == 256:                              # §11.5.7.2 the one-octet case
        writer.align()
        writer.put_bits(offset, 8)
    elif range_ <= _64K:                             # §11.5.7.3 the two-octet case
        writer.align()
        writer.put_bits(offset, 16)
    else:                                            # §11.5.7.4 the indefinite length case
        octets = max(1, (offset.bit_length() + 7) // 8)
        length_upper = max(1, ((range_ - 1).bit_length() + 7) // 8)
        # §13.2.6 a): the length is itself a constrained whole number, lb=1 and ub=the
        # octet count needed to hold the range.
        _encode_constrained(writer, octets, 1, length_upper)
        writer.align()
        writer.put_octets(offset.to_bytes(octets, "big"))


def _decode_constrained(reader: BitReader, lower: int, upper: int) -> int:
    range_ = upper - lower + 1
    if range_ == 1:
        return lower
    if reader.variant is PerVariant.UNALIGNED:
        return lower + reader.get_bits(bits_for_range(range_))
    if range_ <= 255:
        return lower + reader.get_bits(bits_for_range(range_))
    if range_ == 256:
        reader.align()
        return lower + reader.get_bits(8)
    if range_ <= _64K:
        reader.align()
        return lower + reader.get_bits(16)
    length_upper = max(1, ((range_ - 1).bit_length() + 7) // 8)
    octets = _decode_constrained(reader, 1, length_upper)
    reader.align()
    return lower + int.from_bytes(reader.get_octets(octets), "big")


def _encode_normally_small(writer: BitWriter, value: int) -> None:
    """§11.6: a small non-negative whole number whose size is potentially unlimited."""
    if value < 0:
        raise Asn1Error("PER: a normally small whole number cannot be negative")
    if value <= 63:                                  # §11.6.1
        writer.put_bit(0)
        writer.put_bits(value, 6)
        return
    writer.put_bit(1)                                # §11.6.2 -> semi-constrained, lb=0
    _encode_semi_constrained(writer, value, 0)


def _decode_normally_small(reader: BitReader) -> int:
    if reader.get_bit() == 0:
        return reader.get_bits(6)
    return _decode_semi_constrained(reader, 0)


def _encode_semi_constrained(writer: BitWriter, value: int, lower: int) -> None:
    """§11.7: offset from `lower` in the minimum octets, with a length determinant."""
    if value < lower:
        raise Asn1Error(f"PER: {value} is below the lower bound {lower}")
    offset = value - lower
    octets = max(1, (offset.bit_length() + 7) // 8)
    _encode_unconstrained_length(writer, octets)
    writer.align()
    writer.put_octets(offset.to_bytes(octets, "big"))


def _decode_semi_constrained(reader: BitReader, lower: int) -> int:
    octets = _decode_unconstrained_length(reader)
    reader.align()
    return lower + int.from_bytes(reader.get_octets(octets), "big")


def _encode_unconstrained(writer: BitWriter, value: int) -> None:
    """§11.8: 2's-complement into the minimum octets, with a length determinant."""
    octets = max(1, (value.bit_length() + 8) // 8)
    _encode_unconstrained_length(writer, octets)
    writer.align()
    writer.put_octets(value.to_bytes(octets, "big", signed=True))


def _decode_unconstrained(reader: BitReader) -> int:
    octets = _decode_unconstrained_length(reader)
    reader.align()
    return int.from_bytes(reader.get_octets(octets), "big", signed=True)


# --------------------------------------------------------------------------------------
# §11.9: length determinants.
# --------------------------------------------------------------------------------------


def _encode_unconstrained_length(writer: BitWriter, count: int) -> None:
    """§11.9.3.6/§11.9.3.7: the one- and two-octet unconstrained length forms.

    The fragmented form (§11.9.3.8) is handled by `_encode_length_and_payload`, which is
    the only caller that has the payload available to split. Reaching a count of 16K here
    would mean a caller tried to spell a single un-fragmentable length that large.
    """
    writer.align()
    if count <= _LEN_1_OCTET_MAX:
        writer.put_bits(count, 8)
    elif count < _LEN_2_OCTET_MAX:
        writer.put_bit(1)
        writer.put_bit(0)
        writer.put_bits(count, 14)
    else:
        raise Asn1Error(
            f"PER: length {count} needs the §11.9.3.8 fragmented form, which this call "
            f"site cannot produce")


def _decode_unconstrained_length(reader: BitReader) -> int:
    reader.align()
    first = reader.get_bits(8)
    if not first & 0x80:
        return first
    if not first & 0x40:
        return ((first & 0x3F) << 8) | reader.get_bits(8)
    raise Asn1Error(
        "PER: fragmented length determinant in a context that cannot be fragmented")


def _length_bounds(size) -> tuple[int, int | None]:
    """Turn an effective size constraint into the (lb, ub) pair §11.9 wants."""
    if size is None:
        return 0, None
    low, high = size
    lower = 0 if low is UNBOUNDED or low is None else int(low)
    upper = None if high is UNBOUNDED or high is None else int(high)
    if upper is not None and upper >= _64K:
        # §20.2/§11.9.1: an upper bound at or past 64K is not usable as a constrained
        # length determinant, so the type behaves as though it were unbounded.
        upper = None
    return lower, upper


def _encode_length_and_payload(
    writer: BitWriter, count: int, lower: int, upper: int | None, emit,
) -> None:
    """§11.9.3.3/§11.9.3.5-§11.9.3.8: a length determinant, then the material.

    `emit(start, stop)` appends the half-open unit range to the writer. Passing a range
    rather than pre-built bytes is what lets the fragmentation path in §11.9.3.8 hand out
    16K-unit slices for octets, characters *and* SEQUENCE OF components with one code path.
    """
    if upper is not None:
        # §11.9.3.3: constrained, ub < 64K -- the length is a constrained whole number and
        # there is never fragmentation.
        _encode_constrained(writer, count, lower, upper)
        if count:
            emit(0, count)
        return
    start = 0
    while True:
        remaining = count - start
        if remaining >= _FRAG_UNIT:
            blocks = min(4, remaining // _FRAG_UNIT)     # §11.9.3.8.1
            chunk = blocks * _FRAG_UNIT
            writer.align()
            writer.put_bits(0xC0 | blocks, 8)            # §11.9.3.8
            emit(start, start + chunk)
            start += chunk
            # §11.9.3.8.3 NOTE: a final fragment that exactly fills the last block is still
            # followed by a zero length, so the loop always emits a terminating short form.
            continue
        _encode_unconstrained_length(writer, remaining)
        if remaining:
            emit(start, count)
        return


def _decode_length_and_payload(
    reader: BitReader, lower: int, upper: int | None, consume,
) -> int:
    """The decoding counterpart. `consume(n)` reads `n` units; returns the total count."""
    if upper is not None:
        count = _decode_constrained(reader, lower, upper)
        if count:
            consume(count)
        return count
    total = 0
    while True:
        reader.align()
        first = reader.get_bits(8)
        if not first & 0x80:
            if first:
                consume(first)
            return total + first
        if not first & 0x40:
            count = ((first & 0x3F) << 8) | reader.get_bits(8)
            if count:
                consume(count)
            return total + count
        blocks = first & 0x3F
        if not 1 <= blocks <= 4:                          # §11.9.3.8 restricts m to 1..4
            raise Asn1Error(f"PER: fragment block count {blocks} outside 1..4")
        chunk = blocks * _FRAG_UNIT
        consume(chunk)
        total += chunk


# --------------------------------------------------------------------------------------
# Type dispatch.
# --------------------------------------------------------------------------------------

#: §30.1: the known-multiplier types. Only these have PER-visible size and permitted
#: alphabet constraints; everything else falls to §30.6 (base encoding + octet length).
_KNOWN_MULTIPLIER: dict[int, tuple[int, int]] = {
    Universal.NUMERIC_STRING: (32, 57),
    Universal.PRINTABLE_STRING: (32, 122),
    Universal.VISIBLE_STRING: (32, 126),
    Universal.IA5_STRING: (0, 127),
    Universal.BMP_STRING: (0, (1 << 16) - 1),
    Universal.UNIVERSAL_STRING: (0, (1 << 32) - 1),
}

#: X.680 §43 canonical order for the two types whose full range is not contiguous.
_NUMERIC_ALPHABET = " 0123456789"
_PRINTABLE_ALPHABET = (
    " '()+,-./0123456789:=?"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)


def _alphabet_of(kind: Primitive) -> str | None:
    """The effective permitted alphabet (§10.3.12) as an ordered string, or None."""
    from .constraints import Constraint

    constraint = kind.constraint
    if isinstance(constraint, Constraint):
        alphabet = constraint.alphabet()
        if alphabet:
            return "".join(sorted(set(alphabet)))
    if kind.universal == Universal.NUMERIC_STRING:
        return _NUMERIC_ALPHABET
    if kind.universal == Universal.PRINTABLE_STRING:
        return _PRINTABLE_ALPHABET
    return None


def _char_bits(kind: Primitive, variant: PerVariant) -> tuple[int, dict[str, int], dict[int, str]]:
    """§30.5.2-§30.5.4: bits per character, and the value map in force.

    §30.5.4 a) keeps the natural code value when it fits in `b` bits; otherwise b) renumbers
    the permitted characters 0..N-1 in canonical order. Doing the fit test explicitly is
    what makes IA5String cost 7 bits unaligned while a `FROM("0".."9")` alphabet costs 4.
    """
    alphabet = _alphabet_of(kind)
    low, high = _KNOWN_MULTIPLIER[kind.universal]
    if alphabet is None:
        count = high - low + 1
        bits = bits_for_range(count)
        if variant is PerVariant.ALIGNED:
            bits = 1 << max(0, (bits - 1).bit_length()) if bits else 0
        if high <= (1 << bits) - 1:
            return bits, {}, {}
        forward = {chr(low + i): i for i in range(count)}
        return bits, forward, {v: k for k, v in forward.items()}
    count = len(alphabet)
    bits = bits_for_range(count)
    if variant is PerVariant.ALIGNED and bits:
        bits = 1 << max(0, (bits - 1).bit_length())
    if max(ord(ch) for ch in alphabet) <= (1 << bits) - 1:
        return bits, {}, {}
    forward = {ch: index for index, ch in enumerate(alphabet)}
    return bits, forward, {v: k for k, v in forward.items()}


def _value_bounds(kind: Primitive) -> tuple[tuple[object, object], bool]:
    """The extension ROOT's value bounds plus the §13.1 extensibility flag."""
    from .constraints import Constraint

    if isinstance(kind.constraint, Constraint):
        return root_value_bounds(kind.constraint)
    return UNBOUNDED, False


def _size_bounds(kind) -> tuple[int, int | None, bool]:
    """Length bounds in §11.9's (lb, ub) shape, plus the extensibility flag.

    A type whose SIZE carries an extension marker is extensible for PER (§10.3.9 read with
    §17.3/§20.4/§30.4), and the bounds returned here are the ROOT's -- the width used when
    the extension bit says the value is inside it.
    """
    from .constraints import Constraint

    constraint = getattr(kind, "constraint", None)
    if isinstance(constraint, Constraint):
        bounds, extensible = root_size_bounds(constraint)
        lower, upper = _length_bounds(bounds)
        return lower, upper, extensible
    return 0, None, False


def _encode_integer_root(writer: BitWriter, value: int, low, high) -> None:
    """§13.2: the three non-extensible cases, chosen by what the constraint pins down."""
    if low is not None and high is not None:
        if low == high:
            if value != low:                             # §13.2.1: single value, no field
                raise Asn1Error(f"PER: {value} is not the single permitted value {low}")
            return
        _encode_constrained(writer, value, int(low), int(high))
    elif low is not None:
        _encode_semi_constrained(writer, value, int(low))   # §13.2.3
    else:
        _encode_unconstrained(writer, value)                # §13.2.4


def _decode_integer_root(reader: BitReader, low, high) -> int:
    if low is not None and high is not None:
        if low == high:
            return int(low)
        return _decode_constrained(reader, int(low), int(high))
    if low is not None:
        return _decode_semi_constrained(reader, int(low))
    return _decode_unconstrained(reader)


def _encode_integer(writer: BitWriter, kind: Primitive, value: int) -> None:
    """§13, including §13.1's extension bit.

    A half-open bound is `None`, not the UNBOUNDED sentinel -- the two were conflated here
    before, which made the §13.2.3 semi-constrained branch unreachable and sent
    `INTEGER (0..MAX)` into `int(None)`.
    """
    (low, high), extensible = _value_bounds(kind)
    if extensible:
        # §13.1: one bit, set when the value falls OUTSIDE the extension root. Outside, the
        # value is encoded unconstrained, because a future version may widen the root and a
        # root-sized field could not carry what it then admits.
        inside = (low is None or value >= low) and (high is None or value <= high)
        writer.put_bit(0 if inside else 1)
        if not inside:
            _encode_unconstrained(writer, value)
            return
    _encode_integer_root(writer, value, low, high)


def _decode_integer(reader: BitReader, kind: Primitive) -> int:
    (low, high), extensible = _value_bounds(kind)
    if extensible and reader.get_bit():
        return _decode_unconstrained(reader)
    return _decode_integer_root(reader, low, high)


def _encode_octets(writer: BitWriter, kind, data: bytes) -> None:
    """§17: octet strings. Fixed short lengths stay unaligned; the rest take a length."""
    lower, upper, extensible = _size_bounds(kind)
    count = len(data)
    if extensible:
        # §17.3: one bit, set when the LENGTH is outside the extension root. Outside, the
        # length becomes a semi-constrained whole number and the size constraint is ignored.
        inside = count >= lower and (upper is None or count <= upper)
        writer.put_bit(0 if inside else 1)
        if not inside:
            lower, upper = 0, None
    if upper is not None and lower == upper:
        if count != upper:
            raise Asn1Error(f"PER: octet string is {count} octets, fixed size is {upper}")
        if upper == 0:                                   # §17.5
            return
        if upper <= 2:                                   # §17.6: not octet-aligned
            writer.put_octets(data)
            return
        if upper < _64K:                                 # §17.7
            writer.align()
            writer.put_octets(data)
            return
    def emit(start: int, stop: int) -> None:
        writer.align()
        writer.put_octets(data[start:stop])
    _encode_length_and_payload(writer, count, lower, upper, emit)   # §17.8


def _decode_octets(reader: BitReader, kind) -> bytes:
    lower, upper, extensible = _size_bounds(kind)
    if extensible and reader.get_bit():
        lower, upper = 0, None
    if upper is not None and lower == upper:
        if upper == 0:
            return b""
        if upper <= 2:
            return reader.get_octets(upper)
        if upper < _64K:
            reader.align()
            return reader.get_octets(upper)
    chunks: list[bytes] = []
    def consume(count: int) -> None:
        reader.align()
        chunks.append(reader.get_octets(count))
    _decode_length_and_payload(reader, lower, upper, consume)
    return b"".join(chunks)


def _encode_known_multiplier(writer: BitWriter, kind: Primitive, text: str) -> None:
    """§30.5: one of the six known-multiplier types."""
    bits, forward, _ = _char_bits(kind, writer.variant)
    lower, upper, extensible = _size_bounds(kind)
    count = len(text)
    if extensible:
        # §30.4: zero when the value is within the extension root, one otherwise. Outside,
        # the encoding proceeds "as if there was no effective size constraint" -- the
        # permitted ALPHABET still applies, since an extensible alphabet is not PER-visible
        # at all (§10.3.11) and a non-extensible one is unaffected by the size marker.
        inside = count >= lower and (upper is None or count <= upper)
        writer.put_bit(0 if inside else 1)
        if not inside:
            lower, upper = 0, None

    def emit(start: int, stop: int) -> None:
        # §30.5.6/§30.5.7: align only when the whole field is wider than 16 bits.
        if upper is not None and upper * bits > 16:
            writer.align()
        elif upper is None:
            writer.align()
        for ch in text[start:stop]:
            writer.put_bits(forward[ch] if forward else ord(ch), bits)

    if upper is not None and lower == upper and upper < _64K:
        if count != upper:
            raise Asn1Error(f"PER: string is {count} characters, fixed size is {upper}")
        if count:
            emit(0, count)                               # §30.5.6
        return
    _encode_length_and_payload(writer, count, lower, upper, emit)   # §30.5.7


def _decode_known_multiplier(reader: BitReader, kind: Primitive) -> str:
    bits, _, back = _char_bits(kind, reader.variant)
    lower, upper, extensible = _size_bounds(kind)
    if extensible and reader.get_bit():
        lower, upper = 0, None
    out: list[str] = []

    def consume(count: int) -> None:
        if upper is None or upper * bits > 16:
            reader.align()
        for _ in range(count):
            raw = reader.get_bits(bits)
            out.append(back[raw] if back else chr(raw))

    if upper is not None and lower == upper and upper < _64K:
        if upper:
            consume(upper)
        return "".join(out)
    _decode_length_and_payload(reader, lower, upper, consume)
    return "".join(out)


def _encode_primitive(writer: BitWriter, kind: Primitive, value) -> None:
    universal = kind.universal
    if universal == Universal.BOOLEAN:                   # §12
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool")
        writer.put_bit(1 if value else 0)
        return
    if universal == Universal.NULL:                      # §18: no encoding at all
        return
    if universal == Universal.INTEGER:
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int")
        _encode_integer(writer, kind, value)
        return
    if universal == Universal.ENUMERATED:                # §14
        indices = kind.enum_indices()
        if int(value) not in indices:
            raise Asn1Error(
                f"{kind.name}: {value} is not an enumeration value of this type")
        index = indices[int(value)]
        if kind.enum_extensible:                         # §14.3
            writer.put_bit(0)
        _encode_constrained(writer, index, 0, len(indices) - 1)   # §14.2
        return
    if universal == Universal.OCTET_STRING:
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: expected bytes")
        _encode_octets(writer, kind, bytes(value))
        return
    if universal == Universal.OBJECT_IDENTIFIER:         # §24
        from .values import encode_oid

        octets = encode_oid(value)
        _encode_length_and_payload(
            writer, len(octets), 0, None,
            lambda start, stop: (writer.align(), writer.put_octets(octets[start:stop])),
        )
        return
    if universal in _KNOWN_MULTIPLIER:
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str")
        _encode_known_multiplier(writer, kind, value)
        return
    if universal in _OTHER_STRINGS:                      # §30.6
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str")
        from .values import encode_string

        octets = encode_string(universal, value)
        _encode_length_and_payload(
            writer, len(octets), 0, None,
            lambda start, stop: (writer.align(), writer.put_octets(octets[start:stop])),
        )
        return
    raise Asn1Error(f"PER: no encoding for universal type {universal}")


def _decode_primitive(reader: BitReader, kind: Primitive):
    universal = kind.universal
    if universal == Universal.BOOLEAN:
        return bool(reader.get_bit())
    if universal == Universal.NULL:
        return None
    if universal == Universal.INTEGER:
        return _decode_integer(reader, kind)
    if universal == Universal.ENUMERATED:
        indices = kind.enum_indices()
        if kind.enum_extensible and reader.get_bit():
            raise Asn1Error(
                f"{kind.name}: extension-addition enumeration values are not supported")
        index = _decode_constrained(reader, 0, len(indices) - 1)
        for number, position in indices.items():
            if position == index:
                return number
        raise Asn1Error(f"{kind.name}: enumeration index {index} is out of range")
    if universal == Universal.OCTET_STRING:
        return _decode_octets(reader, kind)
    if universal == Universal.OBJECT_IDENTIFIER:
        from .values import decode_oid

        chunks: list[bytes] = []

        def consume(count: int) -> None:
            reader.align()
            chunks.append(reader.get_octets(count))

        _decode_length_and_payload(reader, 0, None, consume)
        return decode_oid(b"".join(chunks))
    if universal in _KNOWN_MULTIPLIER:
        return _decode_known_multiplier(reader, kind)
    if universal in _OTHER_STRINGS:
        from .tags import TagClass
        from .tlv import Tlv
        from .values import decode_string

        chunks = []

        def consume(count: int) -> None:
            reader.align()
            chunks.append(reader.get_octets(count))

        _decode_length_and_payload(reader, 0, None, consume)
        # §30.6.1: the "base encoding" is X.690 §8.23.5's octet string, so the repertoire
        # check belongs to the X.690 rail. Rewrapping as a TLV reuses it rather than
        # decoding the octets here and losing the check.
        tlv = Tlv(Tag(TagClass.UNIVERSAL, universal, False), b"".join(chunks))
        return decode_string(tlv, der=True)
    raise Asn1Error(f"PER: no decoding for universal type {universal}")


#: The restricted character strings that are NOT known-multiplier (§30.6). UTF8String is
#: the one that matters for the BCIR modules: its constraints are never PER-visible, so it
#: always costs an unconstrained octet length plus the X.690 §8.23.5 base encoding.
_OTHER_STRINGS = frozenset({
    Universal.UTF8_STRING, Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING,
    Universal.GRAPHIC_STRING, Universal.GENERAL_STRING, Universal.OBJECT_DESCRIPTOR,
})


def _root_components(kind) -> tuple[Component, ...]:
    """The root component list, in the order PER encodes it.

    SEQUENCE/SET name the list `components`; CHOICE names it `alternatives`. §21 makes a
    SET encode "as if it had been declared a sequence type" once its root is sorted into
    the X.680 §8.6 canonical tag order, and §23.2 orders CHOICE alternatives the same way
    to assign the index -- which is the same rule OER's `_ordered` already implements for
    X.696 §18.2, so it is imported rather than restated.
    """
    from .oer import _ordered

    if isinstance(kind, Choice):
        return _ordered_alternatives(kind)
    if isinstance(kind, Set):
        return _ordered(kind)
    return tuple(kind.components)


def _ordered_alternatives(kind: Choice) -> tuple[Component, ...]:
    """§23.2: CHOICE indices follow the X.680 §8.6 canonical order of the alternatives."""
    def key(comp: Component):
        tag = comp.outer_tag()
        if tag is None:                                   # §23.3: an untagged CHOICE
            nested = comp.type.alternative_tags()         # takes its smallest tag
            tag = min(nested, key=lambda t: (int(t.cls), t.number))
        return (int(tag.cls), tag.number)

    return tuple(sorted(kind.alternatives, key=key))


def _split_root(kind) -> tuple[tuple[Component, ...], tuple[Component, ...]]:
    """(extension root, extension additions) in encoding order (X.680 §25.1)."""
    components = _root_components(kind)
    root = tuple(c for c in components if not c.extension)
    additions = tuple(c for c in components if c.extension)
    return root, additions


def _supplied(comp: Component, value: dict, rules: PerRules) -> bool:
    """Whether a component's encoding is present, applying the §19.5 DEFAULT rule."""
    if comp.group is not None:
        # §19.9: "If all components values of the ExtensionAdditionGroup are missing then
        # the ExtensionAdditionGroup shall be encoded as a missing extension addition."
        return any(_supplied(member, value, rules) for member in comp.group)
    if comp.name not in value:
        return False
    if comp.has_default and rules is PerRules.CANONICAL:
        # §19.5: under CANONICAL-PER a DEFAULT component whose value IS the default must be
        # absent, so the bitmap bit is zero and no field is emitted. BASIC-PER leaves it to
        # the sender, which is exactly the implementation freedom §7.5 removes.
        return value[comp.name] != comp.default
    return True


def _encode_sequence(writer: BitWriter, kind, value: dict, rules: PerRules) -> None:
    """§19: an optional extension bit, the root presence bitmap, then the components."""
    if not isinstance(value, dict):
        raise Asn1Error(f"{kind.name}: expected a mapping")
    root, additions = _split_root(kind)
    known = {comp.name for comp in root + additions}
    for comp in additions:
        if comp.group is not None:
            known.update(member.name for member in comp.group)
    unknown = set(value) - known
    if unknown:
        raise Asn1Error(f"{kind.name}: unknown components {sorted(unknown)}")

    present_additions = [c for c in additions if _supplied(c, value, rules)]
    if kind.extensible:                                   # §19.1
        writer.put_bit(1 if present_additions else 0)
    elif present_additions:                               # pragma: no cover - guarded
        raise Asn1Error(f"{kind.name}: extension additions on a non-extensible type")

    present: dict[str, bool] = {}
    for comp in root:                                     # §19.2
        if comp.optional or comp.has_default:
            present[comp.name] = _supplied(comp, value, rules)
            writer.put_bit(1 if present[comp.name] else 0)
        elif comp.name not in value:
            raise Asn1Error(f"{kind.name}: missing component {comp.name!r}")

    for comp in root:                                     # §19.4
        if (comp.optional or comp.has_default) and not present[comp.name]:
            continue
        _encode(writer, comp.type, value[comp.name], rules)

    if not present_additions:
        return                                            # §19.6
    # §19.8: the addition bitmap is preceded by its length as a NORMALLY SMALL length,
    # and §19.7 sizes the bitmap by the number of additions in the type, not the number
    # present -- a decoder built against an older version relies on that width.
    _encode_normally_small_length(writer, len(additions))
    for comp in additions:                                # §19.7
        writer.put_bit(1 if _supplied(comp, value, rules) else 0)
    for comp in additions:                                # §19.9: each as an open type
        if not _supplied(comp, value, rules):
            continue
        inner = BitWriter(writer.variant)
        if comp.group is not None:
            # A version bracket is encoded as a SEQUENCE of its members (§19.2-§19.6), then
            # wrapped as one open type. Its members live FLAT in the parent value, so the
            # subset is projected here rather than expecting a nested mapping.
            subset = {m.name: value[m.name] for m in comp.group if m.name in value}
            _encode(inner, comp.type, subset, rules)
        else:
            _encode(inner, comp.type, value[comp.name], rules)
        octets = inner.to_bytes()
        _encode_length_and_payload(
            writer, len(octets), 0, None,
            lambda start, stop, _o=octets: (writer.align(), writer.put_octets(_o[start:stop])),
        )


def _decode_sequence(reader: BitReader, kind, rules: PerRules) -> dict:
    root, additions = _split_root(kind)
    has_additions = bool(reader.get_bit()) if kind.extensible else False
    present: dict[str, bool] = {}
    for comp in root:
        if comp.optional or comp.has_default:
            present[comp.name] = bool(reader.get_bit())
    out: dict = {}
    for comp in root:
        if (comp.optional or comp.has_default) and not present[comp.name]:
            if comp.has_default:
                out[comp.name] = comp.default
            continue
        out[comp.name] = _decode(reader, comp.type, rules)
    if not has_additions:
        for comp in additions:
            if comp.has_default:
                out[comp.name] = comp.default
        return out
    count = _decode_normally_small_length(reader)
    flags = [bool(reader.get_bit()) for _ in range(count)]
    for index, flag in enumerate(flags):
        if not flag:
            continue
        chunks: list[bytes] = []
        # Bind `chunks` as a default rather than closing over the loop variable: the
        # closure would otherwise resolve it at CALL time, so a later iteration's list
        # would be the one appended to if the callback ever outlived its iteration.
        def consume(n: int, _into: list = chunks) -> None:
            reader.align()
            _into.append(reader.get_octets(n))

        _decode_length_and_payload(reader, 0, None, consume)
        if index >= len(additions):
            # A peer built against a NEWER version sent an addition this type does not
            # know. §19.7's bitmap plus the open-type wrapper is precisely what makes that
            # recoverable: the octets are skipped, not misparsed.
            continue
        comp = additions[index]
        inner = BitReader(b"".join(chunks), reader.variant)
        decoded = _decode(inner, comp.type, rules)
        if comp.group is not None:
            out.update(decoded)                          # the bracket's members are flat
        else:
            out[comp.name] = decoded
    for comp in additions:
        if comp.group is None and comp.name not in out and comp.has_default:
            out[comp.name] = comp.default
    return out


def _encode_normally_small_length(writer: BitWriter, count: int) -> None:
    """§11.9.3.4: a normally small LENGTH -- note the n-1 bias, unlike §11.6's plain n."""
    if count <= 64:
        writer.put_bit(0)
        writer.put_bits(count - 1, 6)
        return
    writer.put_bit(1)
    _encode_unconstrained_length(writer, count)


def _decode_normally_small_length(reader: BitReader) -> int:
    if reader.get_bit() == 0:
        return reader.get_bits(6) + 1
    return _decode_unconstrained_length(reader)


def _encode_sequence_of(writer: BitWriter, kind, value, rules: PerRules) -> None:
    """§20: a count (unless fixed), then the components' field-lists concatenated."""
    if not isinstance(value, (list, tuple)):
        raise Asn1Error(f"{kind.name}: expected a list")
    items = list(value)
    lower, upper, extensible = _size_bounds(kind)
    if extensible:
        # §20.4: one bit, set when the COUNT is outside the extension root; outside, the
        # length determinant becomes a semi-constrained whole number.
        inside = len(items) >= lower and (upper is None or len(items) <= upper)
        writer.put_bit(0 if inside else 1)
        if not inside:
            lower, upper = 0, None
    if upper is not None and lower == upper and upper < _64K:
        if len(items) != upper:                           # §20.5: no length determinant
            raise Asn1Error(f"{kind.name}: expected exactly {upper} components")
        for item in items:
            _encode(writer, kind.element, item, rules)
        return

    def emit(start: int, stop: int) -> None:
        for item in items[start:stop]:
            _encode(writer, kind.element, item, rules)

    _encode_length_and_payload(writer, len(items), lower, upper, emit)   # §20.6


def _decode_sequence_of(reader: BitReader, kind, rules: PerRules) -> list:
    lower, upper, extensible = _size_bounds(kind)
    if extensible and reader.get_bit():
        lower, upper = 0, None
    out: list = []
    if upper is not None and lower == upper and upper < _64K:
        for _ in range(upper):
            out.append(_decode(reader, kind.element, rules))
        return out

    def consume(count: int) -> None:
        for _ in range(count):
            out.append(_decode(reader, kind.element, rules))

    _decode_length_and_payload(reader, lower, upper, consume)
    return out


def _encode_choice(writer: BitWriter, kind, value, rules: PerRules) -> None:
    """§23: an index over the alternatives, then the chosen alternative's fields."""
    if not isinstance(value, dict) or len(value) != 1:
        raise Asn1Error(f"{kind.name}: a CHOICE value is a single-entry mapping")
    (name, inner), = value.items()
    root, additions = _split_root(kind)
    for index, comp in enumerate(root):
        if comp.name == name:
            if kind.extensible:                           # §23.5
                writer.put_bit(0)
            if len(root) > 1:                             # §23.4: one alternative, no index
                _encode_constrained(writer, index, 0, len(root) - 1)           # §23.6/§23.7
            _encode(writer, comp.type, inner, rules)
            return
    for index, comp in enumerate(additions):
        if comp.name != name:
            continue
        if not kind.extensible:                           # pragma: no cover - guarded
            raise Asn1Error(f"{kind.name}: extension alternative on a non-extensible CHOICE")
        writer.put_bit(1)                                 # §23.5
        # §23.8: the index is a normally small non-negative whole number with lb=0, and the
        # alternative itself is wrapped as an open type so a reader that does not know it
        # can still skip exactly the right number of octets.
        _encode_normally_small(writer, index)
        payload = BitWriter(writer.variant)
        _encode(payload, comp.type, inner, rules)
        octets = payload.to_bytes()
        _encode_length_and_payload(
            writer, len(octets), 0, None,
            lambda start, stop, _o=octets: (writer.align(), writer.put_octets(_o[start:stop])),
        )
        return
    raise Asn1Error(f"{kind.name}: {name!r} is not an alternative of this CHOICE")


def _decode_choice(reader: BitReader, kind, rules: PerRules) -> dict:
    root, additions = _split_root(kind)
    if kind.extensible and reader.get_bit():              # §23.8
        index = _decode_normally_small(reader)
        chunks: list[bytes] = []

        def consume(n: int) -> None:
            reader.align()
            chunks.append(reader.get_octets(n))

        _decode_length_and_payload(reader, 0, None, consume)
        if index >= len(additions):
            raise Asn1Error(
                f"{kind.name}: extension alternative {index} is unknown to this version")
        comp = additions[index]
        inner = BitReader(b"".join(chunks), reader.variant)
        return {comp.name: _decode(inner, comp.type, rules)}
    index = 0
    if len(root) > 1:
        index = _decode_constrained(reader, 0, len(root) - 1)
    if not 0 <= index < len(root):
        raise Asn1Error(f"{kind.name}: CHOICE index {index} is out of range")
    comp = root[index]
    return {comp.name: _decode(reader, comp.type, rules)}


def _encode(writer: BitWriter, kind: Asn1Type, value, rules: PerRules) -> None:
    if isinstance(kind, Primitive):
        _encode_primitive(writer, kind, value)
    elif isinstance(kind, (Sequence, Set)):
        _encode_sequence(writer, kind, value, rules)
    elif isinstance(kind, (SequenceOf, SetOf)):
        _encode_sequence_of(writer, kind, value, rules)
    elif isinstance(kind, Choice):
        _encode_choice(writer, kind, value, rules)
    elif isinstance(kind, OpenType):
        # §11.2: an open type is a complete encoding of the inner value, carried as an
        # unconstrained octet count. The inner encoding is padded to an octet boundary in
        # its own right, which is why it cannot simply be inlined into the field-list.
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: an open type value is raw bytes")
        octets = bytes(value)
        _encode_length_and_payload(
            writer, len(octets), 0, None,
            lambda start, stop: (writer.align(), writer.put_octets(octets[start:stop])),
        )
    else:
        raise Asn1Error(f"PER: no encoding for schema type {type(kind).__name__}")


def _decode(reader: BitReader, kind: Asn1Type, rules: PerRules):
    if isinstance(kind, Primitive):
        return _decode_primitive(reader, kind)
    if isinstance(kind, (Sequence, Set)):
        return _decode_sequence(reader, kind, rules)
    if isinstance(kind, (SequenceOf, SetOf)):
        return _decode_sequence_of(reader, kind, rules)
    if isinstance(kind, Choice):
        return _decode_choice(reader, kind, rules)
    if isinstance(kind, OpenType):
        chunks: list[bytes] = []

        def consume(count: int) -> None:
            reader.align()
            chunks.append(reader.get_octets(count))

        _decode_length_and_payload(reader, 0, None, consume)
        return b"".join(chunks)
    raise Asn1Error(f"PER: no decoding for schema type {type(kind).__name__}")


def encode_per(
    kind: Asn1Type, value, *,
    variant: PerVariant = PerVariant.UNALIGNED,
    rules: PerRules = PerRules.CANONICAL,
) -> bytes:
    """Encode `value` as a complete PER encoding of `kind` (§11.1)."""
    writer = BitWriter(variant)
    _encode(writer, kind, value, rules)
    return writer.to_bytes()


def decode_per(
    data: bytes, kind: Asn1Type, *,
    variant: PerVariant = PerVariant.UNALIGNED,
    rules: PerRules = PerRules.CANONICAL,
) -> object:
    """Decode a complete PER encoding of `kind`.

    §7.2: a PER encoding is not self-delimiting without the type, so the trailing padding
    of §11.1 is the only slack permitted -- anything more is a malformed encoding rather
    than a longer spelling, and the bound below is what stops a decoder from silently
    ignoring appended bytes.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise Asn1Error("PER: expected bytes")
    reader = BitReader(bytes(data), variant)
    value = _decode(reader, kind, rules)
    # §11.1.3.1/§11.1.4: an EMPTY field-list is not zero octets, it is "a single octet with
    # all bits set to 0" -- so a type that encodes to nothing (a NULL, an INTEGER pinned to
    # one value) legitimately consumes no bits while occupying one octet. Without the
    # floor, that one octet reads as trailing garbage and the decode fails on a correct
    # encoding.
    consumed_octets = max(1, (reader.pos + 7) // 8)
    if consumed_octets < len(data):
        raise Asn1Error(
            f"PER: {len(data) - consumed_octets} trailing octet(s) after the encoding")
    if reader.pos % 8:
        # The pad bits of §11.1 must be zero, exactly as `BitReader.align` requires.
        while reader.pos % 8:
            if reader.get_bit():
                raise Asn1Error("PER: non-zero trailing pad bit")
    return value


__all__ = [
    "BASIC_PER_ALIGNED_OID", "BASIC_PER_UNALIGNED_OID",
    "CANONICAL_PER_ALIGNED_OID", "CANONICAL_PER_UNALIGNED_OID",
    "BitReader", "BitWriter", "PerRules", "PerVariant",
    "bits_for_range", "decode_per", "encode_per", "rules_oid",
]
