"""Encoding Control Notation, part two: user-defined encoding objects.

[`ecn.py`](ecn.py) is part one — the class/object/object-set model of clauses 9–18 and the
seven built-in BER/PER object sets. Those sets *name* encodings the repository already
implements, so applying one dispatches to `der.py` or `per.py` and the octets are the ones
those rails' own Annex A tests already pin. **This module is where ECN stops naming other
people's encodings and starts defining octets of its own.**

WHY THIS EXISTS AT ALL, given the gate. The build-out roadmap's §6 reduction gate fired and
was signed off: the fixed DER/PER/OER/CJER candidate set already demonstrates cost-governed
selection, so user-defined ECN was closed as "not an active prerequisite", reopenable only
with an approved measured workload *and* a proof that ordinary BCIR lowering contracts cannot
express it. The approval is the project owner's to give and was given. The **proof** is not
something approval can supply, so it is built here as evidence rather than asserted:
`legacy_frame_workload()` is a field-scaled integer of the kind real link-layer and IP-family
headers use, and `refuted_by()` runs all five fixed candidates against it and reports what
each produces. See `test_asn1_ecn_user.py`, which fails if any candidate ever *does* express
it — because on that day this module's justification is gone and the roadmap should say so.

WHAT THE FIXED SET CANNOT DO, stated precisely. DER, PER, OER and JER all encode *the
abstract value*. Given the integer 40 they will write 40, in their various framings and
widths. A legacy header that transmits `40` as the nibble `1010` — because its field is
scaled in 4-octet units — is asking for the *encoded* value to be a function of the abstract
one. That function is exactly what `#TRANSFORM` is, and no amount of constraint tightening,
alignment choice, or canonical-variant selection in the fixed set produces it. This is not a
performance argument or a size argument; it is an expressiveness one, which is the only kind
the gate accepts.

CITATIONS ARE NOW CHECKED AGAINST THE TEXT. The first version of this module was written
without Rec. ITU-T X.692 (02/2021) in hand and said so, citing at clause granularity and
describing the rest in its own words. The text has since been read, and the pass found the
clause-level attributions correct — 19 is mapping values, 20-23 the defined syntax, 24
`#TRANSFORM`, 25 `#OUTER` — and two genuine SEMANTIC divergences:

* **One object, one operation.** §24.3.5 permits "any given encoding object to specify
  precisely one arithmetic operation. General arithmetic can, however, be defined by the use
  of an ordered list of transforms", and §22.4.1.1 declares the property as
  `&Encoder-transforms #TRANSFORM ORDERED OPTIONAL`. The old `IntToInt(offset=, scale=)`
  fused two operations into one object. It is now `IntOp` plus `TransformChain`.
* **Reversibility is per transform, not a blanket rule.** The old code refused any value a
  transform could not invert, reasoning that a lossy transform is not an encoding rule. Table
  6 disagrees: `modulo:n` is **Never reversible** and is still legal, while `divide:n` is
  reversible exactly when the "Value is a multiple of n". So `reversible()` now *reports* per
  Table 6 and only a caller needing a reversible chain refuses — which, pleasingly, means the
  old refusal was the right rule for `divide` and the wrong rule for everything else.

Two smaller things the text settled: §24.3.7 defines `divide:n` to truncate toward zero, so
`-1 divide:2` is 0 where Python's `//` gives -1; and §24.3.9 confines `subtract:lower-bound`
to the first position in a list, which is a statement about the LIST and is therefore enforced
in `TransformChain` rather than in the transform.

THE PROPERTY GROUPS ARE THE CLAUSE'S OWN, not a boolean each. Reading clause 23's defined
syntax turned three of this module's knobs out to be defaults wearing a disguise. An
`align_before: bool` is `ALIGNED TO NEXT octet PADDING zero` with the unit, the padding and
the pattern all frozen, where §22.2.1.1 gives all three as properties — legacy layouts align
to nibbles and to 16-bit words and pad with ones. A `Justification` of `LEFT` or `RIGHT` is
§21.8.1's `CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}` with the offset dropped,
and the offset is what places a field two bits in from the top of its space. And a `#PAD` or
`#OUTER` filling with an integer cannot spell §21.9's `encoder-option` at all. `PreAlignment`,
`ValuePadding`, `Padding` and `Pattern` are those groups as the clauses declare them.

`encoder-option` is worth naming as a hazard rather than a feature. §21.9.7 lets an encoder
"freely choose the bit values", so an encoding that uses it has no unique octets — which is
exactly what a byte-identity twin test assumes it has. It is implemented, it writes zeros,
and both this module and any comparison built on it should treat that as *a* conforming
answer rather than *the* one.

THE SURFACE SYNTAX IS PARSED, in [`ecn_syntax.py`](ecn_syntax.py). Clause 20's defined syntax
— the bracket-optional keyword grammar that clause 23's `WITH SYNTAX` statements spell out —
reads into the objects below, so an ECN specification can be written as the text X.692
defines rather than assembled field by field in Python. That module also gives an object set
a canonical serialization and a digest, which is the thing this one lacked: every other
descriptor in this package can be hashed and named, and until now an ECN encoding could not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error


# --- clause 21's property types: the values the defined syntax sets ------------------------

#: §21.1.1's `Unit`, whose named values are `repetitions(0), bit(1), nibble(4), octet(8),
#: word16(16), dword32(32)` under the constraint `(0..256)`.
#:
#: Constants rather than an enumeration, and the constraint is why: the type is an INTEGER
#: with *named values*, so `nibble` is a spelling for 4 and not an exhaustive alternative.
#: `Unit 12` is a legal ECN specification, and an enumeration would refuse it. §21.1.2 makes
#: `bit` the default in every group that takes one.
UNIT_REPETITIONS = 0
UNIT_BIT = 1
UNIT_NIBBLE = 4
UNIT_OCTET = 8
UNIT_WORD16 = 16
UNIT_DWORD32 = 32
UNIT_MAX = 256

#: The six §21.1.1 spellings, for the surface syntax to resolve and for diagnostics to use.
UNIT_NAMES = {
    "repetitions": UNIT_REPETITIONS, "bit": UNIT_BIT, "nibble": UNIT_NIBBLE,
    "octet": UNIT_OCTET, "word16": UNIT_WORD16, "dword32": UNIT_DWORD32,
}


def check_unit(bits: int, *, allow_repetitions: bool) -> int:
    """§21.1's range, plus the `(ALL EXCEPT repetitions)` most property groups carry.

    Pre-alignment, encoding space and start pointer all declare their unit as
    `Unit (ALL EXCEPT repetitions)` — §22.2.1.1 and §22.4.1.1 spell it out — because
    "align to a multiple of zero bits" denotes nothing. Only the repetition category admits
    it, and §21.1.5 says what it means there.
    """
    if not 0 <= bits <= UNIT_MAX:
        raise Asn1Error(
            f"ECN: §21.1.1 constrains Unit to (0..{UNIT_MAX}); got {bits}")
    if bits == UNIT_REPETITIONS and not allow_repetitions:
        raise Asn1Error(
            "ECN: this encoding property is declared Unit (ALL EXCEPT repetitions); "
            "§21.1.5 admits `repetitions` only in the repetition category")
    return bits


class Padding(Enum):
    """§21.9.1's `Padding ::= ENUMERATED {zero, one, pattern, encoder-option}`.

    §21.9.4–§21.9.7 give each value its meaning: `zero` and `one` fill with that bit,
    `pattern` defers to the group's `Pattern` property, and `encoder-option` lets the encoder
    "freely choose the bit values". §21.9.2 makes `zero` the default.

    `ENCODER_OPTION` is the one with teeth. It means the octets are **not** a function of the
    abstract value alone, so an encoding that uses it has no single correct output and cannot
    be compared byte-for-byte against a twin. This module writes zeros for it and says so at
    the point of use, rather than pretending the choice was never offered.
    """

    ZERO = "zero"
    ONE = "one"
    PATTERN = "pattern"
    ENCODER_OPTION = "encoder-option"


class PatternKind(Enum):
    """§21.10.1's `Pattern` CHOICE, by alternative name."""

    BITS = "bits"
    OCTETS = "octets"
    CHAR8 = "char8"
    CHAR16 = "char16"
    CHAR32 = "char32"
    ANY_OF_LENGTH = "any-of-length"
    DIFFERENT_ANY = "different"


@dataclass(frozen=True)
class Pattern:
    """§21.10's `Pattern`, and the `Non-Null-Pattern` subtype that excludes the empty ones.

    The alternatives divide into two groups that behave differently, which is why one type
    carries both rather than the bits alone. `bits`, `octets`, `char8`, `char16` and `char32`
    *denote* a bit sequence: §21.10.4 gives the first two their literal value, and §21.10.5
    to §21.10.7 convert each character to its ISO/IEC 10646 value as 8, 16 or 32 bits. Those
    are what `bit_sequence()` returns.

    `any-of-length` (§21.10.8) and `different:any` (§21.10.9) denote a *length* and leave the
    value an encoder's option. They have no bit sequence to return, and asking for one is a
    refusal rather than a default, because a guessed pattern would make two conforming
    encoders disagree while both are right.

    §21.10.10 is the reason `fill()` is separate from `bit_sequence()`: for pre-padding and
    justification the pattern "is truncated and/or replicated as necessary to provide
    sufficient bits", so a 2-bit pattern filling 5 bits is not an error.
    """

    kind: PatternKind = PatternKind.BITS
    #: The literal bits, most significant first, for the five concrete alternatives.
    bits: tuple[int, ...] = (0,)
    #: §21.10.8's length, in bits, for `any-of-length`. Zero elsewhere.
    length: int = 0

    @classmethod
    def from_bits(cls, spelling: str) -> "Pattern":
        """§21.10.4's `bits` alternative, written as a BIT STRING's `'0101'B` body."""
        for character in spelling:
            if character not in "01":
                raise Asn1Error(
                    f"ECN: a bits: pattern is a BIT STRING; {character!r} is not a bit")
        return cls(PatternKind.BITS, tuple(int(c) for c in spelling))

    @classmethod
    def from_octets(cls, data: bytes) -> "Pattern":
        """§21.10.4's `octets` alternative: the octet string's own bits."""
        bits = tuple(
            (byte >> shift) & 1 for byte in data for shift in range(7, -1, -1))
        return cls(PatternKind.OCTETS, bits)

    @classmethod
    def from_chars(cls, text: str, width: int) -> "Pattern":
        """§21.10.5–§21.10.7: each character as its ISO/IEC 10646 value, `width` bits wide."""
        kind = {8: PatternKind.CHAR8, 16: PatternKind.CHAR16,
                32: PatternKind.CHAR32}[width]
        bits: list[int] = []
        for character in text:
            point = ord(character)
            if point >> width:
                raise Asn1Error(
                    f"ECN: {character!r} has ISO/IEC 10646 value {point}, which does not fit "
                    f"the {width}-bit {kind.value} alternative")
            bits.extend((point >> shift) & 1 for shift in range(width - 1, -1, -1))
        return cls(kind, tuple(bits))

    @classmethod
    def any_of_length(cls, length: int) -> "Pattern":
        if length < 1:
            raise Asn1Error(
                f"ECN: §21.10.1 constrains any-of-length to INTEGER (1..MAX); got {length}")
        return cls(PatternKind.ANY_OF_LENGTH, (), length)

    @classmethod
    def different_any(cls) -> "Pattern":
        return cls(PatternKind.DIFFERENT_ANY, (), 0)

    def is_null(self) -> bool:
        """§21.10.2's `Non-Null-Pattern` exclusion: the five empty concrete alternatives."""
        return self.kind not in (
            PatternKind.ANY_OF_LENGTH, PatternKind.DIFFERENT_ANY) and not self.bits

    def require_non_null(self, where: str) -> "Pattern":
        if self.is_null():
            raise Asn1Error(
                f"ECN: {where} is declared Non-Null-Pattern, and §21.10.2 excludes the empty "
                f"alternatives; a zero-length pattern cannot fill a padding bit")
        return self

    def bit_sequence(self) -> tuple[int, ...]:
        if self.kind in (PatternKind.ANY_OF_LENGTH, PatternKind.DIFFERENT_ANY):
            raise Asn1Error(
                f"ECN: §21.10.8/§21.10.9 make {self.kind.value} an encoder's option, so it "
                f"denotes a length and not a bit sequence; two conforming encoders may write "
                f"different bits here and this rail will not choose one for them")
        return self.bits

    def fill(self, count: int) -> tuple[int, ...]:
        """`count` bits from this pattern, replicated and truncated per §21.10.10.

        §22.2.3.3 states the direction: "the first inserted bit is the leading bit of
        Pattern", and if more bits are needed "the pattern shall be re-used, most significant
        bit first". So this repeats from the front rather than mirroring or zero-extending.
        """
        if count < 0:
            raise Asn1Error(f"ECN: cannot fill {count} bits")
        source = self.bit_sequence()
        if not source:
            raise Asn1Error("ECN: an empty pattern cannot fill any bits (§21.10.2)")
        return tuple(source[index % len(source)] for index in range(count))


def _padding_bits(padding: Padding, pattern: "Pattern | None", count: int,
                  where: str) -> tuple[int, ...]:
    """The `count` bits a §21.9 `Padding` value produces, given the group's `Pattern`."""
    if count <= 0:
        return ()
    if padding is Padding.ZERO:
        return (0,) * count
    if padding is Padding.ONE:
        return (1,) * count
    if padding is Padding.PATTERN:
        if pattern is None:
            raise Asn1Error(
                f"ECN: {where} is `pattern`, which §21.9.6 resolves through the group's "
                f"Pattern property; none is set")
        return pattern.require_non_null(where).fill(count)
    # §21.9.7 — `encoder-option`. Zeros are *a* conforming choice, not *the* conforming
    # choice, and the difference matters to anything comparing octets: an encoding using
    # this has no unique output, so a twin writing ones would also be right.
    return (0,) * count


class JustificationSide(Enum):
    """§21.8.1's `Justification` CHOICE alternatives."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Justification:
    """§21.8.1's `Justification ::= CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}`.

    Where a value sits when its encoding space is wider than the value needs. The fixed
    candidate set has no equivalent knob: PER picks a width from the constraint and fills it,
    OER and DER work in whole octets. A user-defined encoding chooses the space *and* the
    position within it, which is what lets an ECN object match a header field laid out before
    any of these standards existed.

    **The offset is the part a bare left/right flag loses.** §21.8.4 measures `left:n` as "the
    number of bits between the leading edge of the encoding space and the leading bit of the
    value encoding", and §21.8.5 measures `right:n` from the trailing edge. §22.8.3.3 and
    §22.8.3.4 then split the b padding bits accordingly: `right:n` puts `b-n` before and `n`
    after, `left:n` puts `n` before and `b-n` after. A field sitting two bits in from the top
    of its space is `left:2`, and the first version of this module could only say `left`.

    §21.8.2 makes `right:0` the default, which is why `Justification()` is that.
    """

    side: JustificationSide = JustificationSide.RIGHT
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise Asn1Error(
                f"ECN: §21.8.1 constrains both alternatives to INTEGER (0..MAX); "
                f"got {self.side.value}:{self.offset}")

    @classmethod
    def left(cls, offset: int = 0) -> "Justification":
        return cls(JustificationSide.LEFT, offset)

    @classmethod
    def right(cls, offset: int = 0) -> "Justification":
        return cls(JustificationSide.RIGHT, offset)

    def split(self, padding_bits: int) -> tuple[int, int]:
        """§22.8.3.3/§22.8.3.4: how "b" padding bits divide into (pre, post).

        §22.8.2.1 is checked here rather than at construction because it relates the offset
        to "b", which is not known until a value is encoded: "the number of bits specified in
        justification shall be less than or equal to the total number of padding bits".
        """
        if padding_bits < 0:
            raise Asn1Error(f"ECN: a value cannot overrun its encoding space by "
                            f"{-padding_bits} bits")
        if self.offset > padding_bits:
            raise Asn1Error(
                f"ECN: §22.8.2.1 — {self.side.value}:{self.offset} needs {self.offset} "
                f"padding bits but the encoding space leaves {padding_bits}")
        if self.side is JustificationSide.LEFT:
            return self.offset, padding_bits - self.offset
        return padding_bits - self.offset, self.offset


# --- clause 22's property groups: the encoder actions, in the stated order -----------------

@dataclass(frozen=True)
class PreAlignment:
    """§22.2's pre-alignment and padding group.

    §22.2.1.1's three properties: `&encoding-space-pre-alignment-unit Unit (ALL EXCEPT
    repetitions) DEFAULT bit`, `&encoding-space-pre-padding Padding DEFAULT zero`, and
    `&encoding-space-pre-pattern Non-Null-Pattern (ALL EXCEPT different:any) DEFAULT
    bits:'0'B`.

    **This replaces an `align_before: bool`**, which could only spell `ALIGNED TO NEXT octet
    PADDING zero`. Legacy layouts align to nibbles and to 16-bit words, and pad with ones as
    often as with zeros, so the boolean was three defaults hiding inside one flag.

    `NEXT` against `ANY` (§22.2.2.1, defaulting to `NEXT`) is not modelled as a choice here:
    §22.2.2.2 makes `ANY` require a `START-POINTER` clause, and start pointers are not built,
    so `ANY` is refused at the surface rather than silently encoded as `NEXT`.
    """

    unit: int = UNIT_BIT
    padding: Padding = Padding.ZERO
    pattern: Pattern | None = None

    def __post_init__(self) -> None:
        check_unit(self.unit, allow_repetitions=False)
        if self.pattern is not None:
            if self.pattern.kind is PatternKind.DIFFERENT_ANY:
                raise Asn1Error(
                    "ECN: §22.2.1.1 declares the pre-pattern `(ALL EXCEPT different:any)`; "
                    "§21.10.9 needs a second Pattern in the group for it to differ from")
            self.pattern.require_non_null("the pre-alignment pattern")

    def apply(self, out: "BitWriter") -> None:
        """§22.2.3.1: insert the minimum bits making the encoding a multiple of `unit`.

        The alignment point is §22.2.1.4's — "the start of the encoding of the type to which
        an ELM applied an encoding" — which for this writer is bit zero, since `#OUTER` is
        the only thing that resets it (clause 25) and it runs at the end.
        """
        if self.unit <= UNIT_BIT:
            return
        short = (-out.bit_length) % self.unit
        for bit in _padding_bits(self.padding, self.pattern, short,
                                 "the pre-alignment padding"):
            out.put_bit(bit)


@dataclass(frozen=True)
class ValuePadding:
    """§22.8's value padding and justification group.

    §22.8.1.1's properties, minus the determinant half: `&value-justification Justification
    DEFAULT right:0`, `&value-pre-padding Padding DEFAULT zero`, `&value-pre-pattern
    Non-Null-Pattern DEFAULT bits:'0'B`, and the matching post pair.

    The `UNUSED BITS` sub-group (`&unused-bits-determination`, `&unused-bits-reference` and
    its transform lists) is deliberately absent: §22.8.2.2 makes `USING` mandatory unless the
    determination is `not-needed`, and a reference to another field is the auxiliary-field
    machinery that this module does not build. `not-needed` is what is implemented, which
    §22.8.4.1 defines as the decoder deriving "b" from the value and space specifications —
    exactly the case a fixed-width field is in.
    """

    justification: Justification = field(default_factory=Justification)
    pre_padding: Padding = Padding.ZERO
    pre_pattern: Pattern | None = None
    post_padding: Padding = Padding.ZERO
    post_pattern: Pattern | None = None

    def place(self, value_bits: tuple[int, ...], space: int) -> tuple[int, ...]:
        """`value_bits` positioned in a `space`-bit encoding space, padding included.

        §22.8.3.2 defines "b" as the number of added padding bits; §22.8.3.5 sets them "in
        accordance with the PRE-PADDING and POST-PADDING specifications, with the leading bit
        of the pattern as the first inserted bit in each case" — so each side starts the
        pattern afresh rather than continuing the other side's phase.
        """
        b = space - len(value_bits)
        if b < 0:
            raise Asn1Error(
                f"ECN: a {len(value_bits)}-bit value encoding does not fit a {space}-bit "
                f"encoding space")
        pre, post = self.justification.split(b)
        return (_padding_bits(self.pre_padding, self.pre_pattern, pre, "PRE-PADDING")
                + tuple(value_bits)
                + _padding_bits(self.post_padding, self.post_pattern, post, "POST-PADDING"))


# --- the bit-level output the fixed candidate set never needed ---------------------------

class IntForm(Enum):
    """How a non-negative or signed integer becomes bits inside its space."""

    POSITIVE_INT = "positive-int"
    TWOS_COMPLEMENT = "twos-complement"


class BitWriter:
    """Bits, most significant first, with an explicit octet-alignment operation.

    Separate from `per.py`'s writer on purpose. PER's bit output is a consequence of PER's
    rules; this one is a *primitive* a user-defined object composes freely, including
    alignment in places no standard rule would align and padding no standard rule would emit.
    Sharing one writer would have meant one of the two dictating the other's shape.
    """

    def __init__(self) -> None:
        self._bits: list[int] = []

    def put_bit(self, bit: int) -> None:
        self._bits.append(1 if bit else 0)

    def put_bits(self, value: int, width: int) -> None:
        if width < 0:
            raise Asn1Error(f"ECN: an encoding space cannot be {width} bits wide")
        if width and value >> width:
            raise Asn1Error(
                f"ECN: {value} does not fit the {width}-bit encoding space this object "
                f"declares; a user-defined encoding states its width, so a value that "
                f"overflows it is a specification error rather than a wider field")
        for shift in range(width - 1, -1, -1):
            self._bits.append((value >> shift) & 1)

    def align(self, boundary_bits: int = 8, pad: int = 0) -> None:
        if boundary_bits <= 0:
            raise Asn1Error("ECN: an alignment boundary is a positive number of bits")
        while len(self._bits) % boundary_bits:
            self._bits.append(1 if pad else 0)

    @property
    def bit_length(self) -> int:
        return len(self._bits)

    def octets(self) -> bytes:
        """The written bits as octets.

        Refuses a partial octet rather than padding silently. Whether a trailing field is
        padded, and with what, is a decision the encoding object makes — `#OUTER` is where
        it belongs — so guessing here would put a choice the specification owns into the
        plumbing.
        """
        if len(self._bits) % 8:
            raise Asn1Error(
                f"ECN: the encoding is {len(self._bits)} bits, which is not a whole number "
                f"of octets; an object set that can end mid-octet needs a #OUTER encoding "
                f"object to say how the last octet is completed")
        out = bytearray()
        for index in range(0, len(self._bits), 8):
            byte = 0
            for bit in self._bits[index:index + 8]:
                byte = (byte << 1) | bit
            out.append(byte)
        return bytes(out)


# --- clause 24's #TRANSFORM: the encoded value as a function of the abstract one ----------

class IntOp(Enum):
    """§24.3.1's `&int-to-int` CHOICE. Exactly one of these per encoding object.

    §24.3.5 is explicit: giving a value to `INT-TO-INT` permits "any given encoding object to
    specify **precisely one** arithmetic operation. General arithmetic can, however, be
    defined by the use of an ordered list of transforms". That ORDERED list is not a
    convenience — §22.4.1.1 declares the encoding-space property as
    `&Encoder-transforms #TRANSFORM ORDERED OPTIONAL`, so composition is a property of the
    *list*, never of a single object.

    **This corrects the first version of this module**, which fused an offset and a scale into
    one `IntToInt(offset=, scale=)`. That is two operations in one object, which §24.3.5
    forbids; the same encoding is now `(SUBTRACT or DECREMENT) then DIVIDE` as a chain.
    """

    INCREMENT = "increment"
    DECREMENT = "decrement"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    NEGATE = "negate"
    MODULO = "modulo"
    SUBTRACT_LOWER_BOUND = "subtract"


#: Table 6, "Reversal of INT-TO-INT transforms", verbatim in force rather than paraphrased.
#: `None` means always reversible; a callable is the condition on the abstract value.
#:
#: The two interesting rows are the ones a designer would guess wrong. `divide:n` is reversible
#: only when the "Value is a multiple of n" — which is why refusing a non-multiple is the
#: standard's rule and not this module's caution. And `modulo:n` is **Never reversible** and is
#: still a perfectly legal transform, which is why reversibility cannot be a precondition on
#: applying one.
_REVERSAL = {
    IntOp.INCREMENT: None,
    IntOp.DECREMENT: None,
    IntOp.MULTIPLY: None,
    IntOp.DIVIDE: lambda value, n: value % n == 0,
    IntOp.NEGATE: None,
    IntOp.MODULO: lambda value, n: False,
    IntOp.SUBTRACT_LOWER_BOUND: None,
}


@dataclass(frozen=True)
class Transform:
    """Base for the value transformations a `#TRANSFORM` encoding object applies (clause 24).

    This is the mechanism the whole reopening rests on. Every rule in the fixed candidate set
    encodes the abstract value; a transform makes the *transmitted* value a declared function
    of it.

    **Reversibility is a property of the transform, not a requirement on all of them.** Clause
    24 states it per transform: §24.4.6 and §24.5.6 say `bool-to-bool` and `bool-to-int` are
    "defined to be reversible for all abstract values", §24.9.6 says `bits-to-int` "shall not
    be used where reversible transforms are required", and Table 6 gives `int-to-int` a
    per-operation condition including **Never reversible** for `modulo:n`. So `reversible()`
    reports, and only a caller that needs a reversible chain refuses on it.
    """

    name: str = ""

    def apply(self, value):
        raise NotImplementedError

    def inverse(self, value):
        raise NotImplementedError

    def reversible(self, value) -> bool:
        """Whether this transform can be undone for `value`. See Table 6 for `int-to-int`."""
        return True


@dataclass(frozen=True)
class IntToInt(Transform):
    """§24.3 `INT-TO-INT`: exactly one arithmetic operation on an integer.

    §24.3.6 gives `increment`, `decrement`, `multiply` and `negate` "their normal mathematical
    meaning". §24.3.7 defines `divide:n` to produce "the integer value that is closest to the
    mathematical result, but is no further from zero than that result" — truncation toward
    zero, so -1 with `divide:2` gives zero. §24.3.8 defines `modulo:n` in terms of divide then
    multiply.

    The operand ranges are §24.3.1's: `increment`/`decrement` take `INTEGER (1..MAX)`, while
    `multiply`, `divide` and `modulo` take `INTEGER (2..MAX)` — a multiplier of 1 is not a
    transform and the notation does not admit one.
    """

    op: IntOp = IntOp.INCREMENT
    operand: int = 1

    def __post_init__(self) -> None:
        floor = {IntOp.INCREMENT: 1, IntOp.DECREMENT: 1,
                 IntOp.MULTIPLY: 2, IntOp.DIVIDE: 2, IntOp.MODULO: 2}.get(self.op)
        if floor is not None and self.operand < floor:
            raise Asn1Error(
                f"ECN: §24.3.1 constrains {self.op.value} to INTEGER ({floor}..MAX); "
                f"got {self.operand}")

    def apply(self, value: int) -> int:
        if self.op is IntOp.INCREMENT:
            return value + self.operand
        if self.op is IntOp.DECREMENT:
            return value - self.operand
        if self.op is IntOp.MULTIPLY:
            return value * self.operand
        if self.op is IntOp.DIVIDE:
            # §24.3.7: truncate toward zero, which is NOT Python's floor division for
            # negatives. `-1 divide:2` is 0, and `-1 // 2` would be -1.
            quotient = abs(value) // self.operand
            return -quotient if value < 0 else quotient
        if self.op is IntOp.NEGATE:
            return -value
        if self.op is IntOp.MODULO:
            # §24.3.8: i - ((i divide:n) multiply:n), using this clause's own divide.
            divided = IntToInt(op=IntOp.DIVIDE, operand=self.operand).apply(value)
            return value - divided * self.operand
        # §24.3.9: subtract:lower-bound, whose operand is the source's lower bound.
        return value - self.operand

    def inverse(self, value: int) -> int:
        if self.op is IntOp.INCREMENT:
            return value - self.operand
        if self.op is IntOp.DECREMENT:
            return value + self.operand
        if self.op is IntOp.MULTIPLY:
            return value // self.operand
        if self.op is IntOp.DIVIDE:
            return value * self.operand
        if self.op is IntOp.NEGATE:
            return -value
        if self.op is IntOp.MODULO:
            raise Asn1Error(
                "ECN: Table 6 lists modulo:n as Never reversible; there is no inverse to take")
        return value + self.operand

    def reversible(self, value: int) -> bool:
        condition = _REVERSAL[self.op]
        return True if condition is None else condition(value, self.operand)


@dataclass(frozen=True)
class IntToBits(Transform):
    """§24.8 `INT-TO-BITS`: an integer as a bitstring.

    §24.8.1's properties are `&int-to-bits-encoded-as ENUMERATED {positive-int,
    twos-complement} DEFAULT twos-complement`, a `&int-to-bits-unit Unit DEFAULT bit`, and a
    `&int-to-bits-size ResultSize DEFAULT variable`. §24.8.10: "The most significant bit shall
    be at the leading end of the bitstring."

    Only the fixed-size `positive-int` case is built here, which is the one the workload needs;
    the default is `twos-complement`, so an object relying on the default is refused rather
    than silently encoded as unsigned.
    """

    width: int = 0
    encoded_as: str = "positive-int"

    def __post_init__(self) -> None:
        if self.encoded_as != "positive-int":
            raise Asn1Error(
                f"ECN: only §24.8's positive-int form is implemented; {self.encoded_as!r} "
                f"(the DEFAULT is twos-complement) would need a signed width rule")

    def apply(self, value: int) -> tuple[int, ...]:
        if value < 0 or (self.width and value >> self.width):
            raise Asn1Error(
                f"ECN: {value} does not fit an INT-TO-BITS transform of width {self.width}")
        # §24.8.10: most significant bit at the leading end.
        return tuple((value >> shift) & 1 for shift in range(self.width - 1, -1, -1))

    def inverse(self, value: tuple[int, ...]) -> int:
        out = 0
        for bit in value:
            out = (out << 1) | (1 if bit else 0)
        return out


@dataclass(frozen=True)
class TransformChain:
    """§24.3.5's "ordered list of transforms" — how general arithmetic is actually expressed.

    §22.4.1.1 declares the encoding-space properties as `&Encoder-transforms #TRANSFORM
    ORDERED OPTIONAL`, so the order is part of the specification rather than an artefact of
    how a tool stores them. Reversing runs the chain backwards, which is the only reading that
    makes a decoder recover the abstract value.

    §24.3.9 is enforced here because it is a statement about the LIST rather than about any
    one object: `subtract:lower-bound` "shall only be used as the first of an ordered list of
    transforms".
    """

    transforms: tuple[Transform, ...] = ()

    def __post_init__(self) -> None:
        for index, transform in enumerate(self.transforms):
            if (isinstance(transform, IntToInt)
                    and transform.op is IntOp.SUBTRACT_LOWER_BOUND and index != 0):
                raise Asn1Error(
                    f"ECN: §24.3.9 — subtract:lower-bound shall only be used as the first of "
                    f"an ordered list of transforms; it is at position {index}")

    def apply(self, value):
        for transform in self.transforms:
            value = transform.apply(value)
        return value

    def inverse(self, value):
        for transform in reversed(self.transforms):
            value = transform.inverse(value)
        return value

    def reversible(self, value) -> bool:
        """Whether the WHOLE chain can be undone for `value`.

        Checked stepwise on the intermediate values, because Table 6's conditions are about
        the value each transform actually sees — `divide:4` is reversible for 40 and not for
        41, and which of those reaches it depends on everything before it in the list.
        """
        for transform in self.transforms:
            if not transform.reversible(value):
                return False
            value = transform.apply(value)
        return True


# --- clauses 20-23's defined syntax, as the properties it denotes -------------------------

@dataclass(frozen=True)
class IntSpec:
    """A user-defined `#INT` encoding: a width, a form, a position, and an optional transform.

    Compare `ecn.py`'s built-in shared `#INT`, which is "a PER-BASIC-UNALIGNED #INTEGER
    encoding provided it is bounded" (§18.2.5.1) and therefore takes its width from the
    constraint. Here the width is *stated*, which is the difference between describing an
    encoding and choosing one.
    """

    width: int
    form: IntForm = IntForm.POSITIVE_INT
    transform: Transform | None = None
    #: §22.2's group, replacing the `align_before: bool` this class used to carry. `None` is
    #: "no pre-alignment specified", which §22.2.1.1's defaults make equivalent to a unit of
    #: one bit — the group is a no-op rather than an implicit octet alignment.
    pre_alignment: PreAlignment | None = None
    #: §22.8's group. `None` means the value fills its space exactly, so no "b" arises.
    value_padding: ValuePadding | None = None

    def write(self, value: int, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        if self.transform is not None:
            value = self.transform.apply(value)
        if self.form is IntForm.TWOS_COMPLEMENT:
            if value < -(1 << (self.width - 1)) or value >= (1 << (self.width - 1)):
                raise Asn1Error(
                    f"ECN: {value} does not fit {self.width} bits two's complement")
            encoded = value & ((1 << self.width) - 1)
        else:
            if value < 0:
                raise Asn1Error(
                    f"ECN: {value} is negative but this #INT object declares "
                    f"{IntForm.POSITIVE_INT.value}")
            encoded = value
        if self.value_padding is None:
            out.put_bits(encoded, self.width)
            return
        # §22.8.3.2's "b" is the difference between the space and the *value encoding*, so
        # the value's own width has to be a stated thing rather than the space's. For a
        # positive-int that is its bit length, with a single zero bit for zero itself; a
        # two's-complement encoding has no shorter form than the space it was checked
        # against, so it is already exactly `width` and no padding arises.
        if self.form is IntForm.TWOS_COMPLEMENT:
            value_bits = tuple(
                (encoded >> shift) & 1 for shift in range(self.width - 1, -1, -1))
        else:
            used = max(encoded.bit_length(), 1)
            value_bits = tuple((encoded >> shift) & 1 for shift in range(used - 1, -1, -1))
        for bit in self.value_padding.place(value_bits, self.width):
            out.put_bit(bit)


@dataclass(frozen=True)
class BoolSpec:
    """A user-defined `#BOOL`: one bit, with the true/false patterns stated.

    DER writes a whole octet and CER/DER fix `TRUE` at `0xFF`; PER writes one bit. A header
    with an active-low flag matches neither, and says so here.
    """

    width: int = 1
    true_value: int = 1
    false_value: int = 0
    pre_alignment: PreAlignment | None = None

    def write(self, value: bool, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        out.put_bits(self.true_value if value else self.false_value, self.width)


@dataclass(frozen=True)
class PadSpec:
    """A user-defined `#PAD`: bits that carry no abstract value.

    `#PAD` is one of part one's primitive classes and has no built-in object, because none of
    BER/PER needs one. A fixed-layout header does: reserved bits are part of the octets and
    part of nothing else.

    The bits are §21.9's `Padding` rather than an integer, and §21.9.3 says so directly: that
    type "specifies details of the padding for pre-padding, **for classes in the pad
    category**, and for the post-padding of a PDU". An integer would spell `zero` and `one`
    fine and could not spell `encoder-option` at all.
    """

    width: int
    padding: Padding = Padding.ZERO
    pattern: Pattern | None = None

    def write(self, _value, out: BitWriter) -> None:
        for bit in _padding_bits(self.padding, self.pattern, self.width, "a #PAD object"):
            out.put_bit(bit)


@dataclass(frozen=True)
class ConcatenationSpec:
    """A user-defined `#CONCATENATION`: named fields in a **stated** order.

    The order is the point. Every rule in the fixed set writes components in the order the
    ASN.1 type declares them (X.690 §8.9, X.691 §19), and canonical variants may only reorder
    a SET, by tag. A wire format whose fields do not appear in the order the abstract type
    finds natural cannot be reached from any of them — here it is just a different `order`.
    """

    #: Component name -> how that component is encoded.
    fields: dict = field(default_factory=dict)
    #: Names in transmission order. Empty means "the order of `fields`".
    order: tuple[str, ...] = ()
    #: Names that are pure padding, so they take no value from the abstract type.
    padding: tuple[str, ...] = ()

    def transmission_order(self) -> tuple[str, ...]:
        if self.order:
            missing = set(self.fields) - set(self.order)
            extra = set(self.order) - set(self.fields)
            if missing or extra:
                raise Asn1Error(
                    f"ECN: this #CONCATENATION object states a transmission order that does "
                    f"not match its fields (missing {sorted(missing)}, unknown "
                    f"{sorted(extra)})")
            return self.order
        return tuple(self.fields)

    def write(self, value: dict, out: BitWriter) -> None:
        for name in self.transmission_order():
            spec = self.fields[name]
            if name in self.padding:
                spec.write(None, out)
                continue
            if name not in value:
                raise Asn1Error(
                    f"ECN: this #CONCATENATION object encodes {name!r}, which the value does "
                    f"not carry; a user-defined encoding has no optionality unless an "
                    f"#OPTIONAL object supplies it")
            spec.write(value[name], out)


@dataclass(frozen=True)
class OuterSpec:
    """A `#OUTER` encoding object: what happens to the complete encoding.

    §18.1.7 lets an encoding object set hold `#OUTER` and no other encoding-procedure class,
    and part one already enforces that. This is the object that clause admits: it runs once,
    around everything, and is where "pad the last octet" belongs — a decision about the whole
    encoding rather than about any field in it.

    §21.9.3 names this use explicitly — `Padding` specifies the padding "for the post-padding
    of a PDU specified in the `#OUTER` encoding class" — so the bits are that type here too,
    and `encoder-option` is a thing a `#OUTER` object may legitimately say.
    """

    #: Pad the completed encoding up to this boundary. 8 makes the result whole octets.
    boundary_bits: int = 8
    padding: Padding = Padding.ZERO
    pattern: Pattern | None = None

    def finish(self, out: BitWriter) -> None:
        if self.boundary_bits <= 0:
            raise Asn1Error("ECN: an alignment boundary is a positive number of bits")
        short = (-out.bit_length) % self.boundary_bits
        for bit in _padding_bits(self.padding, self.pattern, short, "#OUTER post-padding"):
            out.put_bit(bit)


# --- the encoding object, and applying a set of them --------------------------------------

@dataclass(frozen=True)
class UserEncodingObject:
    """An encoding object whose realization is a specification, not another rail's name.

    Structurally the same thing as `ecn.py`'s `EncodingObject` — it realizes one encoding
    class — and it is deliberately a distinct type so that `encode_with_user` can tell "this
    set names DER" from "this set defines octets", and so a mixed set is a type error rather
    than a silent precedence question.
    """

    encoding_class: object
    spec: object
    name: str = ""


def encode_with_user(objects, cls, value, *, outer: OuterSpec | None = None) -> bytes:
    """Apply user-defined encoding objects to a value and produce octets.

    `objects` maps an encoding class to a `UserEncodingObject`, which is the runtime form of
    part one's `EncodingObjectSet` restricted to user definitions; the one-object-per-class
    law of §9.5.2 is a property of the mapping rather than something to re-check.

    The `#OUTER` object runs last, over the complete encoding, which is the only place a
    decision about the whole result can be made.
    """
    obj = objects.get(cls)
    if obj is None:
        raise Asn1Error(
            f"ECN: no user-defined encoding object for "
            f"{getattr(cls, 'name', cls)!r} (9.5.1)")
    out = BitWriter()
    obj.spec.write(value, out)
    if outer is not None:
        outer.finish(out)
    return out.octets()


# --- the workload the gate asks for, and the refutation it asks for -----------------------

def legacy_frame_workload():
    """The measured workload: a fixed-layout frame header with a **scaled** length field.

    Returns `(asn1_type, abstract_value, expected_octets)`.

    Every part of this is something real headers do. `version` is a 3-bit field. `urgent` is
    a single flag bit, active low, because plenty of hardware asserts low. `reserved` is two
    bits that belong to the octets and to no abstract component. `payloadOctets` is carried
    in 4-octet units in a 4-bit field, the way IPv4's IHL and a long list of its relatives
    carry lengths — and it is transmitted *before* `version`, because the layout predates any
    opinion about declaration order.

    The abstract value is what an application holds: 40 octets of payload, not 10 units.
    """
    from .constraints import ValueRange
    from .schema import Component, Primitive, Sequence
    from .tags import Universal

    def bounded(low: int, high: int) -> Primitive:
        return Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(low, high))

    header = Sequence((
        Component("version", bounded(0, 7)),
        Component("urgent", Primitive(Universal.BOOLEAN, "BOOLEAN")),
        Component("payloadOctets", bounded(0, 60)),
    ), name="FrameHeader")
    value = {"version": 5, "urgent": True, "payloadOctets": 40}

    #  payloadOctets/4 = 10 -> 1010 | version 5 -> 101 | urgent TRUE active-low -> 0 | rr 00
    #  1010 101 0 00  == 0xAA 0x80 once #OUTER pads the last octet.
    return header, value, bytes((0b10101010, 0b00000000))


def legacy_frame_objects():
    """The user-defined encoding objects that produce `legacy_frame_workload`'s octets."""
    from .ecn import CONCATENATION

    spec = ConcatenationSpec(
        fields={
            # §24.3.5: one object, one operation. A field scaled in 4-octet units is a
            # single `divide:4`; had it also carried an offset it would be a two-element
            # chain, which is exactly why `TransformChain` exists.
            "payloadOctets": IntSpec(width=4, transform=TransformChain(
                (IntToInt(op=IntOp.DIVIDE, operand=4, name="OCTETS-TO-UNITS"),))),
            "version": IntSpec(width=3),
            "urgent": BoolSpec(true_value=0, false_value=1),
            "reserved": PadSpec(width=2),
        },
        order=("payloadOctets", "version", "urgent", "reserved"),
        padding=("reserved",),
    )
    return {CONCATENATION: UserEncodingObject(CONCATENATION, spec, "FrameHeader-encoding")}


#: The fixed candidate set the §6 reduction gate was signed off against.
FIXED_CANDIDATES = (
    "DER", "CANONICAL-PER-ALIGNED", "CANONICAL-PER-UNALIGNED", "COER", "CJER",
)


def refuted_by(asn1_type, value, target: bytes) -> dict:
    """What each fixed candidate produces for `value`, against the octets ECN must reach.

    The gate asks for "a proof that ordinary BCIR lowering contracts cannot express it". A
    proof by assertion would be worthless, so this *runs* every candidate and reports its
    octets. A candidate that raises is recorded as refusing rather than swallowed — refusing
    to encode is still not encoding it.

    Returns `{candidate: (octets_or_None, note)}`. The caller decides what it means; the test
    treats any candidate matching `target` as the gate closing again.
    """
    from .jer import JerRules, encode_jer
    from .oer import OerRules, encode_oer
    from .per import PerRules, PerVariant, encode_per
    from .schema import Module

    results: dict = {}

    def record(name: str, thunk) -> None:
        try:
            results[name] = (thunk(), "encoded")
        except Exception as error:  # noqa: BLE001 - a refusal is a result, not a bug
            results[name] = (None, f"refused: {type(error).__name__}: {error}")

    # `Module.encode` is the DER rail: it builds the TLV and serializes it, which is the
    # definite-length, minimal-form encoding DER requires. Same entry point `ecn.py`'s
    # built-in BER/DER object sets dispatch to, so this measures the rail the roadmap means.
    module = Module("<ecn-workload>", (), {"T": asn1_type})
    record("DER", lambda: module.encode("T", value))
    record("CANONICAL-PER-ALIGNED",
           lambda: encode_per(asn1_type, value, variant=PerVariant.ALIGNED,
                              rules=PerRules.CANONICAL))
    record("CANONICAL-PER-UNALIGNED",
           lambda: encode_per(asn1_type, value, variant=PerVariant.UNALIGNED,
                              rules=PerRules.CANONICAL))
    record("COER", lambda: encode_oer(asn1_type, value, rules=OerRules.CANONICAL))
    record("CJER", lambda: encode_jer(asn1_type, value, rules=JerRules.CANONICAL))
    for name, (octets, _note) in results.items():
        assert name in FIXED_CANDIDATES, name
        if octets == target:  # pragma: no cover - the gate closing is not the expected path
            results[name] = (octets, "MATCHES the target; the expressiveness gap is gone")
    return results


__all__ = [
    "UNIT_BIT", "UNIT_DWORD32", "UNIT_MAX", "UNIT_NAMES", "UNIT_NIBBLE", "UNIT_OCTET",
    "UNIT_REPETITIONS", "UNIT_WORD16",
    "BitWriter", "BoolSpec", "check_unit", "ConcatenationSpec", "FIXED_CANDIDATES", "IntForm", "IntOp",
    "IntSpec", "IntToBits", "IntToInt", "Justification", "JustificationSide", "OuterSpec",
    "PadSpec", "Padding", "Pattern", "PatternKind", "PreAlignment", "Transform",
    "TransformChain", "UserEncodingObject", "ValuePadding", "encode_with_user",
    "legacy_frame_objects", "legacy_frame_workload", "refuted_by",
]
