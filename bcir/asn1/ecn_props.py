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

class SizeRangeCondition(Enum):
    """§21.13.1's `SizeRangeCondition`: §21.11's sibling, over the **size** constraint.

    §21.13.3: it "is used to test properties of the bounds in an effective size constraint
    associated with a class in the repetition or characterstring category". So where
    `RangeCondition` asks about an integer's *value* bounds, this asks about a string's or a
    repetition's *length* bounds.

    **The two NOTEs differ, and that is the interesting part.** §21.11.4's says "For any given
    set of bounds, exactly one predicate will be satisfied" — the five shapes partition.
    §21.13.4's says "Only the `fixed-size` case overlaps with other predicates" — so these five
    do **not** partition, and `fixed-size` deliberately co-occurs with `ub-with-zero-lb` (for
    `SIZE(0)`) and with `ub-with-non-zero-lb` (for any other fixed size). An implementation
    that assumed the sibling's exhaustiveness here would pick the wrong encoding whenever a
    size is fixed, which is the common case.
    """

    NO_UB_WITH_ZERO_LB = "no-ub-with-zero-lb"
    UB_WITH_ZERO_LB = "ub-with-zero-lb"
    NO_UB_WITH_NON_ZERO_LB = "no-ub-with-non-zero-lb"
    UB_WITH_NON_ZERO_LB = "ub-with-non-zero-lb"
    FIXED_SIZE = "fixed-size"
    TEST_LOWER_BOUND = "test-lower-bound"
    TEST_UPPER_BOUND = "test-upper-bound"
    TEST_RANGE = "test-range"

    def needs_comparison(self) -> bool:
        """§21.13.5, worded identically to §21.11.5: the last three take a Comparison."""
        return self in (SizeRangeCondition.TEST_LOWER_BOUND,
                        SizeRangeCondition.TEST_UPPER_BOUND,
                        SizeRangeCondition.TEST_RANGE)


@dataclass(frozen=True)
class SizeBounds:
    """The effective size constraint §21.13 tests. `high=None` is "no upper bound".

    A size has a lower bound always — X.680 sizes are `INTEGER (0..MAX)` constrained — so
    `low` is an int rather than an optional, and §21.13.4 a) turns on it being *zero* where
    §21.11.4 a) turned on a bound *existing*. Another place the two siblings diverge.
    """

    low: int = 0
    high: int | None = None

    def satisfies(self, condition: SizeRangeCondition,
                  comparison: "Comparison | None" = None,
                  comparator: int | None = None) -> bool:
        """§21.13.4's five shapes and §21.13.5's three comparisons."""
        if condition.needs_comparison() != (comparison is not None):
            raise Asn1Error(
                f"ECN: §21.13.5 — {condition.value} "
                f"{'requires' if condition.needs_comparison() else 'does not admit'} a "
                f"Comparison and an integer comparator")
        if comparison is not None:
            if comparator is None:
                raise Asn1Error(
                    f"ECN: §21.13.5 gives {condition.value} a Comparison *and* an integer "
                    f"comparator; the comparator is missing")
            if condition is SizeRangeCondition.TEST_LOWER_BOUND:
                return comparison.holds(self.low, comparator)
            if condition is SizeRangeCondition.TEST_UPPER_BOUND:
                return self.high is not None and comparison.holds(self.high, comparator)
            if self.high is None:
                return False
            return comparison.holds(self.high - self.low + 1, comparator)
        has_high = self.high is not None
        if condition is SizeRangeCondition.NO_UB_WITH_ZERO_LB:
            return not has_high and self.low == 0
        if condition is SizeRangeCondition.UB_WITH_ZERO_LB:
            return has_high and self.low == 0
        if condition is SizeRangeCondition.NO_UB_WITH_NON_ZERO_LB:
            return not has_high and self.low != 0
        if condition is SizeRangeCondition.UB_WITH_NON_ZERO_LB:
            return has_high and self.low != 0
        return has_high and self.high == self.low          # §21.13.4 e)

    def shapes(self) -> tuple[SizeRangeCondition, ...]:
        """Every §21.13.4 shape these bounds satisfy — plural, unlike §21.11.4's.

        §21.13.4's NOTE says `fixed-size` overlaps, so this returns a tuple where the integer
        sibling returns one value. Asserting a single answer here would fail on `SIZE(4)`.
        """
        return tuple(condition for condition in SizeRangeCondition
                     if not condition.needs_comparison() and self.satisfies(condition))


class OptionalityDetermination(Enum):
    """§21.5.1's `OptionalityDetermination ::= ENUMERATED {field-to-be-set, field-to-be-used,
    container, handle, pointer}`.

    §21.5.3: it "specifies the way in which a decoder determines whether an optional component
    is present in an encoding". Five answers, and they are the five ways real formats say
    "this field may not be here": a presence bit written by the encoder (`field-to-be-set`),
    a presence bit the application supplies (`field-to-be-used`), running out of container
    (`container`), recognizing what comes next (`handle`), and a pointer that is zero when the
    thing is absent (`pointer`, §21.5.9).

    §21.5.2 makes `field-to-be-set` the default, as it is for every other determination type.
    """

    FIELD_TO_BE_SET = "field-to-be-set"
    FIELD_TO_BE_USED = "field-to-be-used"
    CONTAINER = "container"
    HANDLE = "handle"
    POINTER = "pointer"


class AlternativeDetermination(Enum):
    """§21.6.1's `AlternativeDetermination ::= ENUMERATED {field-to-be-set, field-to-be-used,
    handle}`.

    §21.6.3: how "a decoder determines which alternative is present in an encoding of a class
    in the alternatives category". Three, not five — a CHOICE always encodes exactly one
    alternative, so neither `container` nor `pointer` has anything to say here, and §21.6.1
    simply does not list them.

    The conceptual value the first two carry is §22.6.3.2's `alternative-index`: zero for the
    first alternative, one for the next, in whatever order `ORDER` fixes. That indirection is
    what lets a two-bit selector field and a CHOICE of four be related without either knowing
    the other's spelling.
    """

    FIELD_TO_BE_SET = "field-to-be-set"
    FIELD_TO_BE_USED = "field-to-be-used"
    HANDLE = "handle"


class ComponentOrder(Enum):
    """The order property §22.6 and §22.10 both carry — with **different value sets**.

    §22.10.1.1 declares concatenation's as `ENUMERATED {textual, tag, random}`; §22.6.1.1
    declares the alternatives one as `ENUMERATED {textual, tag}` and stops there. One
    enumeration covers both because the three values mean the same thing in each
    (§22.6.3.4 and §22.10.3.1–§22.10.3.3 are worded identically), and `random` is refused at
    the point of use rather than by having two nearly identical types.

    `RANDOM` is the one with a prerequisite: §22.10.2.1 makes it require an identification
    handle exhibited by *every* component, with disjoint value sets — an encoder free to
    reorder is only decodable if each component announces which one it is.
    """

    TEXTUAL = "textual"
    TAG = "tag"
    RANDOM = "random"


class ConcatenationAlignment(Enum):
    """§22.10.1.1's `&concatenation-alignment ENUMERATED {none, aligned} DEFAULT aligned`.

    **The default is `aligned`**, which is worth stating because it is the only defaulted
    property in clause 22 that inserts bits when nobody asked. §22.10.2.2: "If `ALIGNMENT` is
    `aligned`, then the pre-alignment specification assumes the default value unless set" —
    and §22.2.1.1's default unit is one bit, so the default-on-default is a no-op. A
    concatenation only gains padding here when it also states a pre-alignment unit.
    """

    NONE = "none"
    ALIGNED = "aligned"


class HandleValueKind(Enum):
    """§21.16.1's `HandleValueSet` CHOICE, by alternative name."""

    BITS = "bits"
    OCTETS = "octets"
    NUMBER = "number"
    TAG_ANY = "tag"
    RANGE = "range"
    RANGES = "ranges"


@dataclass(frozen=True)
class HandleValueSet:
    """§21.16's `HandleValueSet`: the bit patterns an exhibited handle is allowed to take.

    §21.16.2: it "is used to specify the set of bit patterns (the handle value set)
    characterizing the encodings produced by an encoding object that exhibits an identification
    handle". Six alternatives, which reduce to two ideas — a single pattern (`bits`, `octets`,
    `number`, and `tag:any` once its tag number is known) and a set of integer ranges (`range`,
    `ranges`).

    **Everything here is expressed as ranges over the conceptual handle field's integer
    value**, because that is the only representation in which the question the clause actually
    asks — "are these two sets disjoint?" (§21.5.7, §21.6.6, §21.7.10, §22.10.2.1) — is
    answerable without enumerating 2^n patterns. §22.9.1.7 fixes the integer reading: "the bit
    in the conceptual handle field nearest to the zero position is the high-order bit", and the
    number "is right-justified within this field".

    `tag:any` is the one alternative that carries no set of its own. §21.16.5 makes its value
    "determined by the number specified in an ECN encoding structure for a class in the tag
    category, or by the tag number mapped from an ASN.1 tag construction" — so it is resolved
    against a tag number before it can be tested, and `ranges_over` refuses it rather than
    guessing. §22.9.1.9 confines it to `#TAG` objects for the same reason.
    """

    kind: HandleValueKind = HandleValueKind.TAG_ANY
    #: The literal bits, most significant first, for `bits` and `octets`.
    bits: tuple[int, ...] = ()
    #: The `number` alternative's value.
    number: int = 0
    #: Inclusive `(low, high)` pairs for `range` and `ranges`.
    ranges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if self.kind is HandleValueKind.NUMBER and self.number < 0:
            raise Asn1Error(
                f"ECN: §21.16.1 constrains the `number` alternative to INTEGER (0..MAX); "
                f"got {self.number}")
        if self.kind in (HandleValueKind.RANGE, HandleValueKind.RANGES):
            if not self.ranges:
                raise Asn1Error(
                    "ECN: §21.16.1 gives `ranges` SIZE(1..MAX), and `range` is a single "
                    "SEQUENCE; an empty handle value set matches no encoding at all")
            if self.kind is HandleValueKind.RANGE and len(self.ranges) != 1:
                raise Asn1Error(
                    f"ECN: §21.16.1's `range` alternative is one SEQUENCE {{low, high}}; "
                    f"{len(self.ranges)} were given, which is the `ranges` alternative")
            for low, high in self.ranges:
                if low < 0:
                    raise Asn1Error(
                        f"ECN: §21.16.1 constrains a range's bounds to INTEGER (0..MAX); "
                        f"got low {low}")
                if high < low:
                    raise Asn1Error(
                        f"ECN: §21.16.6/§21.16.7 require high greater than or equal to low; "
                        f"got {low}..{high}")

    @classmethod
    def from_bits(cls, spelling: str) -> "HandleValueSet":
        """§21.16.4's `bits` alternative, written as a BIT STRING's `'0101'B` body."""
        for character in spelling:
            if character not in "01":
                raise Asn1Error(
                    f"ECN: a handle value set's `bits` alternative is a BIT STRING; "
                    f"{character!r} is not a bit")
        return cls(HandleValueKind.BITS, tuple(int(c) for c in spelling))

    @classmethod
    def from_octets(cls, data: bytes) -> "HandleValueSet":
        """§21.16.4's `octets` alternative: the octet string's own bits."""
        return cls(HandleValueKind.OCTETS,
                   tuple((byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)))

    @classmethod
    def of_number(cls, number: int) -> "HandleValueSet":
        return cls(HandleValueKind.NUMBER, number=number)

    @classmethod
    def tag_any(cls) -> "HandleValueSet":
        """§21.16.5's `tag:any`, which is also §22.9.1.1's DEFAULT for the property."""
        return cls(HandleValueKind.TAG_ANY)

    @classmethod
    def of_range(cls, low: int, high: int) -> "HandleValueSet":
        return cls(HandleValueKind.RANGE, ranges=((low, high),))

    @classmethod
    def of_ranges(cls, pairs) -> "HandleValueSet":
        return cls(HandleValueKind.RANGES, ranges=tuple((low, high) for low, high in pairs))

    def resolve_tag(self, tag_number: int) -> "HandleValueSet":
        """§21.16.5 / §22.9.1.9: give `tag:any` the tag number that determines its value.

        The clause also makes a *stated* set that disagrees with the tag number an error
        rather than an override: "If, however, a value is specified by `HandleValueSet` and
        differs from that assigned in an ECN specification of a tag class or in an ASN.1 tag
        that maps to an ECN tag, that is an ECN specification error." So a non-`tag:any` set
        is checked here instead of being replaced.
        """
        if self.kind is not HandleValueKind.TAG_ANY:
            if not self.contains_number(tag_number):
                raise Asn1Error(
                    f"ECN: §22.9.1.9 — this #TAG object's handle value set does not admit its "
                    f"own tag number {tag_number}; a stated set that differs from the tag "
                    f"number is an ECN specification error, not an override")
            return self
        return HandleValueSet.of_number(tag_number)

    def contains_number(self, value: int) -> bool:
        """Membership without a field width, for checks that have no encoding in hand."""
        if self.kind is HandleValueKind.NUMBER:
            return value == self.number
        if self.kind in (HandleValueKind.BITS, HandleValueKind.OCTETS):
            return value == _int_of_bits(self.bits)
        if self.kind in (HandleValueKind.RANGE, HandleValueKind.RANGES):
            return any(low <= value <= high for low, high in self.ranges)
        raise Asn1Error(
            "ECN: §21.16.5 — a `tag:any` handle value set has no value of its own until a tag "
            "number determines it; resolve it against the #TAG object's number first")

    def ranges_over(self, width: int) -> tuple[tuple[int, int], ...]:
        """This set as inclusive integer ranges over a `width`-bit conceptual handle field.

        Where the two width rules live. §21.16.4 and §22.9.1.8 are separate sentences saying
        the same thing from opposite sides: a `bits` or `octets` value "shall have the same
        number of bits as those specified for the identification handle by `AT`", and a
        `number` or range bound that "cannot be encoded within the number of bits specified
        for the identification handle" is a specification error. Both are checked here rather
        than at construction, because the width belongs to the *handle* and the set is written
        without it.
        """
        if width < 0:
            raise Asn1Error(f"ECN: a handle field cannot be {width} bits wide")
        limit = 1 << width
        if self.kind in (HandleValueKind.BITS, HandleValueKind.OCTETS):
            if len(self.bits) != width:
                raise Asn1Error(
                    f"ECN: §22.9.1.8 — a `{self.kind.value}` handle value has to have the same "
                    f"number of bits as the handle's AT positions; the value is "
                    f"{len(self.bits)} bits and the handle is {width}")
            point = _int_of_bits(self.bits)
            return ((point, point),)
        if self.kind is HandleValueKind.NUMBER:
            if self.number >= limit:
                raise Asn1Error(
                    f"ECN: §22.9.1.7 / §21.16.4 — the handle value {self.number} does not fit "
                    f"the {width}-bit conceptual handle field")
            return ((self.number, self.number),)
        if self.kind is HandleValueKind.TAG_ANY:
            raise Asn1Error(
                "ECN: §21.16.5 — a `tag:any` handle value set is determined by a tag number; "
                "it has no range until it is resolved against the #TAG object's number")
        for low, high in self.ranges:
            if high >= limit:
                raise Asn1Error(
                    f"ECN: §21.16.4 — the range bound {high} does not fit the {width}-bit "
                    f"conceptual handle field")
        return _normalize_ranges(self.ranges)

    def contains(self, value: int, width: int) -> bool:
        """§22.9.2.2's membership test, over the conceptual handle field's integer value."""
        return any(low <= value <= high for low, high in self.ranges_over(width))

    def disjoint_from(self, other: "HandleValueSet", width: int) -> bool:
        """§21.5.7 / §21.6.6 / §21.7.10 / §22.10.2.1's disjointness, at a given width."""
        mine = self.ranges_over(width)
        theirs = other.ranges_over(width)
        return not any(low <= other_high and other_low <= high
                       for low, high in mine for other_low, other_high in theirs)

    def describe(self) -> str:  # pragma: no cover - diagnostics only
        if self.kind in (HandleValueKind.BITS, HandleValueKind.OCTETS):
            return f"{self.kind.value}:'{''.join(str(bit) for bit in self.bits)}'B"
        if self.kind is HandleValueKind.NUMBER:
            return f"number:{self.number}"
        if self.kind is HandleValueKind.TAG_ANY:
            return "tag:any"
        body = ", ".join(f"{low}..{high}" for low, high in self.ranges)
        return f"{self.kind.value}:{{{body}}}"


def _int_of_bits(bits: tuple[int, ...]) -> int:
    """The integer a conceptual handle field denotes, per §22.9.1.7's high-order-first rule."""
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    return value


def _normalize_ranges(
        ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    """Sorted, coalesced, non-overlapping — so disjointness is a linear scan and not a set.

    §21.16.7 lets `ranges` be any set of ranges and does not require them to be disjoint from
    each other, only that each has `high` at least `low`. Coalescing them here means the
    disjointness test between two *different* sets never has to reason about a set that
    overlaps itself.
    """
    out: list[tuple[int, int]] = []
    for low, high in sorted(ranges):
        if out and low <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], high))
        else:
            out.append((low, high))
    return tuple(out)


class RepetitionSpaceDetermination(Enum):
    """§21.7.1's eight ways a decoder finds the end of a repetition.

    §21.7.3 says what it is for: it "specifies the way in which a decoder determines the end of
    the encoding space in an encoding of a class in the repetition category. It **replaces**
    use of an encoding property of type `EncodingSpaceDetermination`" — so this is §21.3's
    counterpart and not an extension of it, which is why the two are separate enums with
    different members rather than one with eight.

    Five of the eight are recognizable protocol shapes: a count field (`field-to-be-set` /
    `field-to-be-used`), a terminator (`pattern`), a per-element continuation flag
    (`flag-to-be-set` / `flag-to-be-used`), a container, and an identification handle.
    """

    FIELD_TO_BE_SET = "field-to-be-set"
    FIELD_TO_BE_USED = "field-to-be-used"
    FLAG_TO_BE_SET = "flag-to-be-set"
    FLAG_TO_BE_USED = "flag-to-be-used"
    CONTAINER = "container"
    PATTERN = "pattern"
    HANDLE = "handle"
    NOT_NEEDED = "not-needed"
