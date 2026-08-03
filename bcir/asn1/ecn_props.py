"""X.692 clause 21: the property types the defined syntax sets.

Clause 21 is the vocabulary the rest of ECN is written in. `Unit`, `Padding`, `Pattern`,
`Justification`, `Comparison`, `RangeCondition` and `ReversalSpecification` are the ASN.1
types whose *values* fill the encoding properties that clauses 22 to 25 declare — so this
module is the bottom of the ECN layering and imports nothing but the error type.

Split out of [`ecn_user.py`](ecn_user.py) when clause 24 grew: these types are what both the
property groups (clause 22) and the transforms (clause 24) are built from, and having them
underneath both is what lets the two be separate modules rather than one that cannot be read.
Everything here is re-exported from `ecn_user`, so no existing import moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error


class IntForm(Enum):
    """How a non-negative or signed integer becomes bits inside its space.

    X.690 §8.3.2 and §8.3.3 define both, and §24.8.9 points at those clauses rather than
    restating them — so this is the ASN.1 integer encoding, named once for every ECN property
    that selects between the two.
    """

    POSITIVE_INT = "positive-int"
    TWOS_COMPLEMENT = "twos-complement"


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


class Comparison(Enum):
    """§21.12.1's `Comparison ::= ENUMERATED {equal-to, not-equal-to, greater-than,
    less-than, greater-than-or-equal-to, less-than-or-equal-to}`.

    §21.12.2 gives this type **no default**, which is the unusual part: every other clause 21
    type names one, so a `Comparison` that is absent is a specification that did not say what
    it meant rather than one taking a sensible default.
    """

    EQUAL_TO = "equal-to"
    NOT_EQUAL_TO = "not-equal-to"
    GREATER_THAN = "greater-than"
    LESS_THAN = "less-than"
    GREATER_THAN_OR_EQUAL_TO = "greater-than-or-equal-to"
    LESS_THAN_OR_EQUAL_TO = "less-than-or-equal-to"

    def holds(self, value: int, comparator: int) -> bool:
        """§21.12.4's six conditions, tested against the comparator."""
        return {
            Comparison.EQUAL_TO: value == comparator,
            Comparison.NOT_EQUAL_TO: value != comparator,
            Comparison.GREATER_THAN: value > comparator,
            Comparison.LESS_THAN: value < comparator,
            Comparison.GREATER_THAN_OR_EQUAL_TO: value >= comparator,
            Comparison.LESS_THAN_OR_EQUAL_TO: value <= comparator,
        }[self]


class RangeCondition(Enum):
    """§21.11.1's `RangeCondition`, the predicate over an integer class's bounds.

    §21.11.3: it "tests the existence and nature of bounds on the integer values associated
    with an encoding class in the integer category". This is what makes ECN's integer
    encodings *schema-directed* in a way the encoding objects alone are not — one object set
    encodes `INTEGER (0..255)` and `INTEGER` differently because the **bounds** differ, not
    because any value does.

    The first five are §21.11.4's bound shapes, and its NOTE is load-bearing: "For any given
    set of bounds, exactly one predicate will be satisfied." So they partition, and
    `exactly_one_shape` asserts they still do — a partition that quietly stopped covering
    would make §23.6.3.1's "the first whose conditions are satisfied" select nothing.

    The last three take a `Comparison` and an integer comparator (§21.11.5), and unlike the
    first five they overlap freely: `test-lower-bound greater-than -10` and `test-range
    less-than-or-equal-to 20` can both hold, which is what `IF-ALL` composes.
    """

    UNBOUNDED_OR_NO_LOWER_BOUND = "unbounded-or-no-lower-bound"
    SEMI_BOUNDED_WITH_NEGATIVES = "semi-bounded-with-negatives"
    BOUNDED_WITH_NEGATIVES = "bounded-with-negatives"
    SEMI_BOUNDED_WITHOUT_NEGATIVES = "semi-bounded-without-negatives"
    BOUNDED_WITHOUT_NEGATIVES = "bounded-without-negatives"
    TEST_LOWER_BOUND = "test-lower-bound"
    TEST_UPPER_BOUND = "test-upper-bound"
    TEST_RANGE = "test-range"

    def needs_comparison(self) -> bool:
        """§21.11.5: the last three take a `Comparison` and a comparator; the rest take none.

        A predicate rather than an inline check because the clause works in both directions —
        supplying one where it is not wanted is as much an error as omitting one where it is.
        """
        return self in (RangeCondition.TEST_LOWER_BOUND, RangeCondition.TEST_UPPER_BOUND,
                        RangeCondition.TEST_RANGE)


@dataclass(frozen=True)
class IntegerBounds:
    """The bounds §21.11 tests: the effective constraint on an integer encoding class.

    `None` on either side is "no bound", which is a different statement from a bound that
    happens to be large — §21.11.4 a) turns on the *existence* of a lower bound, never on
    its value.
    """

    low: int | None = None
    high: int | None = None

    def satisfies(self, condition: RangeCondition, comparison: "Comparison | None" = None,
                  comparator: int | None = None) -> bool:
        """§21.11.4's five shapes and §21.11.5's three comparisons."""
        if condition.needs_comparison() != (comparison is not None):
            raise Asn1Error(
                f"ECN: §21.11.5 — {condition.value} "
                f"{'requires' if condition.needs_comparison() else 'does not admit'} a "
                f"Comparison and an integer comparator")
        if comparison is not None:
            if comparator is None:
                raise Asn1Error(
                    f"ECN: §21.11.5 gives {condition.value} a Comparison *and* an integer "
                    f"comparator; the comparator is missing")
            if condition is RangeCondition.TEST_LOWER_BOUND:
                # A bound that does not exist cannot compare. §21.11.4's shapes are how a
                # specification asks "is there one at all", so a missing bound fails the test
                # rather than standing in as an infinity that would satisfy half the
                # comparisons and none of the ones a reader expected.
                return self.low is not None and comparison.holds(self.low, comparator)
            if condition is RangeCondition.TEST_UPPER_BOUND:
                return self.high is not None and comparison.holds(self.high, comparator)
            # test-range compares the WIDTH of the value set, which exists only when both
            # bounds do. X.680 ranges are inclusive, so the width is high - low + 1.
            if self.low is None or self.high is None:
                return False
            return comparison.holds(self.high - self.low + 1, comparator)
        has_low = self.low is not None
        has_high = self.high is not None
        if condition is RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND:
            return not has_low
        if condition is RangeCondition.SEMI_BOUNDED_WITH_NEGATIVES:
            return has_low and self.low < 0 and not has_high
        if condition is RangeCondition.BOUNDED_WITH_NEGATIVES:
            return has_low and self.low < 0 and has_high
        if condition is RangeCondition.SEMI_BOUNDED_WITHOUT_NEGATIVES:
            return has_low and self.low >= 0 and not has_high
        return has_low and self.low >= 0 and has_high

    def exactly_one_shape(self) -> RangeCondition:
        """The one §21.11.4 shape these bounds satisfy, per its NOTE.

        The NOTE says "For any given set of bounds, exactly one predicate will be satisfied",
        and this asserts it rather than trusting it: the five branches above are written by
        hand, and a partition that stopped partitioning would make integer selection pick the
        wrong object, or none, with no other symptom.
        """
        shapes = [condition for condition in RangeCondition
                  if not condition.needs_comparison() and self.satisfies(condition)]
        if len(shapes) != 1:  # pragma: no cover - the clause's NOTE says this cannot happen
            raise Asn1Error(
                f"ECN: §21.11.4's NOTE says exactly one predicate holds for any bounds; "
                f"{self} satisfies {[shape.value for shape in shapes]}")
        return shapes[0]


class ReversalSpecification(Enum):
    """§21.14.1's `ReversalSpecification`, in the enumeration's own order.

    **The text disagrees with itself about which name means which action, and the names win.**
    §21.14.1 lists `{no-reversal, reverse-bits-in-units, reverse-half-units,
    reverse-bits-in-half-units}`. §21.14.6 then claims to describe them "in the order of
    enumerations listed above" and gives: no reversal, reversal of half-units, reversal of
    bits in each half-unit, reversal of bits in each unit — a *different* order. §22.12.3.2
    describes them in the enumeration's order and matches the names exactly: "no reversal
    ..., or shall reverse the bits in each unit, or shall reverse the half-units (without
    changing the order of bits in each half-unit) or shall reverse the bits within each
    half-unit". Two readings agree with each other and with what the names say, so §21.14.6's
    listing is taken as the typo — and recorded here rather than silently resolved, because
    the other reading would produce well-formed octets of the wrong shape.

    §21.14.5 and §22.12.2.2 make the two half-unit values need an even `Unit`; §22.12.2.3
    forbids any reversal when the unit is one bit, since reversing one bit is the identity.
    """

    NO_REVERSAL = "no-reversal"
    REVERSE_BITS_IN_UNITS = "reverse-bits-in-units"
    REVERSE_HALF_UNITS = "reverse-half-units"
    REVERSE_BITS_IN_HALF_UNITS = "reverse-bits-in-half-units"

    def check_unit(self, unit: int) -> None:
        if self is ReversalSpecification.NO_REVERSAL:
            return
        if unit <= 1:
            raise Asn1Error(
                f"ECN: §22.12.2.3 — BIT-REVERSAL shall not be set unless MULTIPLE OF is "
                f"greater than one bit; reversing a {unit}-bit unit is the identity")
        if self in (ReversalSpecification.REVERSE_HALF_UNITS,
                    ReversalSpecification.REVERSE_BITS_IN_HALF_UNITS) and unit % 2:
            raise Asn1Error(
                f"ECN: §21.14.5 / §22.12.2.2 — {self.value} needs an even Unit; {unit} is "
                f"odd and has no half")

    def apply(self, bits: tuple[int, ...], unit: int) -> tuple[int, ...]:
        """§22.12.3's reversal over an encoding space's contents.

        §22.12.3.1 divides the contents into `unit`-bit units; §21.14.7 makes a length that
        is not an integral multiple of `Unit` a specification error rather than a short final
        unit to be handled leniently.

        §22.12.1.4's NOTE 2 draws the boundary this operates on: reversal "applies to the
        contents of an encoding space or repetition space (including any value pre-padding or
        post-padding), but does not apply to any pre-alignment padding" — so callers hand it
        the placed value, never the aligned stream.
        """
        self.check_unit(unit)
        if self is ReversalSpecification.NO_REVERSAL:
            return tuple(bits)
        if len(bits) % unit:
            raise Asn1Error(
                f"ECN: §21.14.7 — {len(bits)} bits is not an integral multiple of the "
                f"{unit}-bit Unit that BIT-REVERSAL divides them into")
        out: list[int] = []
        half = unit // 2
        for start in range(0, len(bits), unit):
            chunk = tuple(bits[start:start + unit])
            if self is ReversalSpecification.REVERSE_BITS_IN_UNITS:
                out.extend(reversed(chunk))
            elif self is ReversalSpecification.REVERSE_HALF_UNITS:
                out.extend(chunk[half:] + chunk[:half])
            else:
                out.extend(tuple(reversed(chunk[:half])) + tuple(reversed(chunk[half:])))
        return tuple(out)


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
