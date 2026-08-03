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

from dataclasses import dataclass, field, replace
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
    #: §22.2.2.1's `ANY` against `NEXT`. `NEXT` (False) inserts §22.2.3.1's *minimum* bits;
    #: `ANY` (True) lets the encoder insert "an encoder-dependent number", subject to the same
    #: multiple-of-Unit rule. §22.2.2.2 then requires a START-POINTER, since nothing else
    #: could tell a decoder how many bits the encoder chose — which is why the flag lives here
    #: and the requirement is checked where both groups are visible.
    encoder_chosen_offset: bool = False

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


class UnusedBitsDetermination(Enum):
    """§21.4's `UnusedBitsDetermination ::= ENUMERATED {field-to-be-set, field-to-be-used,
    not-needed}`, quoted verbatim at §22.8.1.3."""

    FIELD_TO_BE_SET = "field-to-be-set"
    FIELD_TO_BE_USED = "field-to-be-used"
    NOT_NEEDED = "not-needed"


class EncodingSpaceDetermination(Enum):
    """§21.3.1's `EncodingSpaceDetermination ::= ENUMERATED {field-to-be-set,
    field-to-be-used, container}`.

    §21.3.4 and §21.3.5 are two different relationships to the same field. `field-to-be-set`
    means the *encoder* computes the size and writes it there. `field-to-be-used` means the
    field's value "may be set from the abstract syntax (i.e., a corresponding field appears
    within the ASN.1 specification)" and the encoder's job is to check it agrees — the value
    is the application's, not the encoder's, and a mismatch is an error rather than a
    correction.

    `container` (§21.3.6) is not built: it needs either another field "whose encoding class
    (the container) has a length determinant and whose contents include this encoding space",
    or the end of the PDU via `#OUTER`. Containment is a structural relationship this rail's
    flat concatenation does not have, so it is refused with the clause rather than
    approximated by "the rest of the encoding".
    """

    FIELD_TO_BE_SET = "field-to-be-set"
    FIELD_TO_BE_USED = "field-to-be-used"
    CONTAINER = "container"


def _determinant_value(chain: "TransformChain | None", value: int, where: str) -> int:
    """Apply an encoder-transform list to a determinant, refusing an irreversible one.

    §22.8.2.4 and §22.3.2.3 both say it in the same words: "It is an ECN specification or
    application error if any transform in the ENCODER-TRANSFORMS is not reversible for the
    abstract value to which it is applied." That is stricter than the value path, where
    Table 6 lets `modulo:n` be legal and lossy — a determinant a decoder cannot invert is a
    length nobody can read, so here reversibility really is a precondition.
    """
    if chain is None:
        return value
    if not chain.reversible(value):
        raise Asn1Error(
            f"ECN: §22.8.2.4 / §22.3.2.3 — {where}'s ENCODER-TRANSFORMS are not reversible "
            f"for {value}, so a decoder could not recover it; a determinant has to be "
            f"invertible even where a value need not be")
    return chain.apply(value)


@dataclass(frozen=True)
class UnusedBits:
    """§22.8's `UNUSED BITS` sub-group: how a decoder learns how much padding there was.

    §22.8.1.6: "`USING` is a reference that enables a decoder to determine the number of
    padding bits inserted." The three determinations differ in *who owns* the number.
    `not-needed` (§22.8.4.1) means it follows from the space and value specifications, which
    is the fixed-width case. `field-to-be-set` (§22.8.3.7) makes the encoder write it. And
    `field-to-be-used` (§22.8.3.8) makes the encoder *check* an application-supplied field.

    Three restrictions are enforced because each is a way to write something meaningless:
    §22.8.2.2 makes `USING` present exactly when the determination is not `not-needed`;
    §22.8.2.3 confines `ENCODER-TRANSFORMS` to `field-to-be-set`; §22.8.2.5 confines
    `DECODER-TRANSFORMS` to `field-to-be-used`. A transform list on the wrong determination
    would never run, which is worse than an error because it reads as though it did.
    """

    determination: UnusedBitsDetermination = UnusedBitsDetermination.NOT_NEEDED
    #: §22.8.1.1's `&unused-bits-reference REFERENCE OPTIONAL` — an earlier field's name.
    reference: str = ""
    encoder_transforms: "TransformChain | None" = None
    decoder_transforms: "TransformChain | None" = None

    def __post_init__(self) -> None:
        needed = self.determination is not UnusedBitsDetermination.NOT_NEEDED
        if needed != bool(self.reference):
            raise Asn1Error(
                f"ECN: §22.8.2.2 — USING shall be specified if and only if DETERMINED BY is "
                f"not `not-needed`; {self.determination.value} "
                f"{'has no' if needed else 'has a'} reference")
        if (self.encoder_transforms is not None
                and self.determination is not UnusedBitsDetermination.FIELD_TO_BE_SET):
            raise Asn1Error(
                "ECN: §22.8.2.3 — ENCODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is `field-to-be-set`")
        if (self.decoder_transforms is not None
                and self.determination is not UnusedBitsDetermination.FIELD_TO_BE_USED):
            raise Asn1Error(
                "ECN: §22.8.2.5 — DECODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is `field-to-be-used`")

    def record(self, out: "BitWriter", padding_bits: int) -> None:
        """§22.8.3.6–§22.8.3.8, given the "b" the justification actually inserted."""
        if self.determination is UnusedBitsDetermination.NOT_NEEDED:
            return  # §22.8.3.6: this completes the encoder's actions.
        if self.determination is UnusedBitsDetermination.FIELD_TO_BE_SET:
            out.patch(self.reference,
                      _determinant_value(self.encoder_transforms, padding_bits,
                                         f"UNUSED BITS USING {self.reference}"))
            return
        # §22.8.3.8: the encoder CHECKS rather than writes. "It is an application error if
        # this condition is not met, and encoding shall not proceed."
        carried = out.value_of(self.reference)
        recovered = (carried if self.decoder_transforms is None
                     else self.decoder_transforms.apply(carried))
        if recovered != padding_bits:
            raise Asn1Error(
                f"ECN: §22.8.3.8 — the field {self.reference!r} carries {carried}, which "
                f"reduces to {recovered} unused bits, but the encoding inserted "
                f"{padding_bits}; encoding shall not proceed")


@dataclass(frozen=True)
class ValuePadding:
    """§22.8's value padding and justification group.

    §22.8.1.1's properties: `&value-justification Justification DEFAULT right:0`,
    `&value-pre-padding Padding DEFAULT zero`, `&value-pre-pattern Non-Null-Pattern DEFAULT
    bits:'0'B`, the matching post pair, and the `UNUSED BITS` sub-group.

    §22.8.2.7 decides what "set" means for this group: "This specification is considered set
    if the `VALUE-PADDING` keyword is used." So a `ValuePadding()` with every default is a
    different statement from no `ValuePadding` at all, and the specs below keep `None` for
    the second.
    """

    justification: Justification = field(default_factory=Justification)
    pre_padding: Padding = Padding.ZERO
    pre_pattern: Pattern | None = None
    post_padding: Padding = Padding.ZERO
    post_pattern: Pattern | None = None
    #: §22.8.1.1's determinant sub-group. Absent is §22.8.4.1's `not-needed`.
    unused_bits: UnusedBits | None = None

    def place(self, value_bits: tuple[int, ...], space: int,
              out: "BitWriter | None" = None) -> tuple[int, ...]:
        """`value_bits` positioned in a `space`-bit encoding space, padding included.

        §22.8.3.2 defines "b" as the number of added padding bits; §22.8.3.5 sets them "in
        accordance with the PRE-PADDING and POST-PADDING specifications, with the leading bit
        of the pattern as the first inserted bit in each case" — so each side starts the
        pattern afresh rather than continuing the other side's phase.

        `out` is taken only so the `UNUSED BITS` group can reach the auxiliary field it
        references; the placement itself does not touch the stream.
        """
        b = space - len(value_bits)
        if b < 0:
            raise Asn1Error(
                f"ECN: a {len(value_bits)}-bit value encoding does not fit a {space}-bit "
                f"encoding space")
        pre, post = self.justification.split(b)
        if self.unused_bits is not None:
            if out is None:
                raise Asn1Error(
                    "ECN: an UNUSED BITS determination refers to another field, so it can "
                    "only be resolved while writing into a stream")
            self.unused_bits.record(out, b)
        return (_padding_bits(self.pre_padding, self.pre_pattern, pre, "PRE-PADDING")
                + tuple(value_bits)
                + _padding_bits(self.post_padding, self.post_pattern, post, "POST-PADDING"))


@dataclass(frozen=True)
class SpaceDeterminant:
    """§21.3 / §22.4's encoding-space determination: `DETERMINED BY ... USING ...`.

    §21.2.5 and §21.2.6 make this mandatory when the size is `variable-with-determinant` or
    `encoder-option-with-determinant`, and §21.2's NOTE permits it in every other case too,
    "to support encodings (similar to BER) that use length determinants even when they are
    redundant. **Any difference between the two determinations is an error.**" So a stated
    width and a determinant can coexist, and when they do this checks them against each other
    rather than trusting either.
    """

    determination: EncodingSpaceDetermination = EncodingSpaceDetermination.FIELD_TO_BE_SET
    #: §22.4.1.1's `&encoding-space-reference REFERENCE OPTIONAL`.
    reference: str = ""
    #: The `MULTIPLE OF` unit the count is in — §21.2.4's multiplier, not a bit count.
    unit: int = UNIT_BIT
    encoder_transforms: "TransformChain | None" = None
    decoder_transforms: "TransformChain | None" = None

    def __post_init__(self) -> None:
        if self.determination is EncodingSpaceDetermination.CONTAINER:
            raise Asn1Error(
                "ECN: §21.3.6's `container` determination needs a field whose encoding class "
                "has a length determinant and whose contents include this encoding space, or "
                "the end of the PDU through #OUTER. Containment is a structural relationship "
                "this rail's flat concatenation does not have, so it is refused rather than "
                "approximated by `the rest of the encoding`")
        if not self.reference:
            raise Asn1Error(
                "ECN: §21.3.4/§21.3.5 — both determinations require a REFERENCE to the field "
                "carrying the length")
        check_unit(self.unit, allow_repetitions=False)

    def record(self, out: "BitWriter", space_bits: int) -> None:
        """§21.3.4's set, or §21.3.5's use, given the space this field actually took."""
        if space_bits % self.unit:
            raise Asn1Error(
                f"ECN: a {space_bits}-bit encoding space is not a whole number of "
                f"{self.unit}-bit units, so no determinant can state its size")
        count = space_bits // self.unit
        if self.determination is EncodingSpaceDetermination.FIELD_TO_BE_SET:
            out.patch(self.reference,
                      _determinant_value(self.encoder_transforms, count,
                                         f"ENCODING-SPACE USING {self.reference}"))
            return
        carried = out.value_of(self.reference)
        recovered = (carried if self.decoder_transforms is None
                     else self.decoder_transforms.apply(carried))
        if recovered != count:
            raise Asn1Error(
                f"ECN: §21.3.5 — the field {self.reference!r} carries {carried}, which "
                f"reduces to {recovered} units, but the encoding space is {count}; a "
                f"conforming encoder shall not produce encodings whose determinant does not "
                f"identify the end of the encoding space")


@dataclass(frozen=True)
class StartPointer:
    """§22.3's start-pointer group: an earlier field carrying where this one begins.

    §22.3.1.4: "If the start of the encoding space for the element is an offset of `n`
    `MULTIPLE OF` units, then the value placed in the field referenced by the `START-POINTER`
    encoding property is the value obtained by applying `ENCODER-TRANSFORMS` to `n`."

    §22.3.3.1 fixes both ends of the measurement precisely, and neither is obvious: it counts
    "from the start of the encoding of the `START-POINTER` field (**after any pre-alignment of
    that field**) to the start of the encoding of the element with the start-pointer
    specification (**after any pre-alignment of that element**)". So pre-alignment padding is
    outside the span at both ends, which is why the writer marks a field's start after
    aligning it.
    """

    reference: str = ""
    unit: int = UNIT_BIT
    encoder_transforms: "TransformChain | None" = None

    def __post_init__(self) -> None:
        if not self.reference:
            raise Asn1Error("ECN: §22.3.1.1 — START-POINTER names the field it sets")
        check_unit(self.unit, allow_repetitions=False)

    def record(self, out: "BitWriter") -> None:
        """Called once this element's own pre-alignment is done and its space is about to start."""
        offset = out.bit_length - out.start_of(self.reference)
        if offset % self.unit:
            raise Asn1Error(
                f"ECN: §22.3.3.1 — the offset from {self.reference!r} is {offset} bits, which "
                f"is not an integral number of {self.unit}-bit units; the clause makes a "
                f"non-integral `n` a specification error")
        out.patch(self.reference,
                  _determinant_value(self.encoder_transforms, offset // self.unit,
                                     f"START-POINTER {self.reference}"))


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
        #: Reserved auxiliary fields: name -> (bit offset, width, IntForm).
        self._slots: dict[str, tuple[int, int, "IntForm"]] = {}
        #: Which reserved fields have been given a value. A slot written twice is a fault
        #: rather than a last-writer-wins, per §21.3.4's rule about a field set more than once.
        self._patched: set[str] = set()
        #: Bit offset where each named field's encoding *space* begins, which is what §22.3's
        #: start pointer measures from. Distinct from the slot table: an ordinary field has a
        #: position but reserves nothing.
        self._starts: dict[str, int] = {}
        #: Abstract values of ordinary fields, which §21.3.5's `field-to-be-used` reads.
        self._values: dict[str, int] = {}

    def put_bit(self, bit: int) -> None:
        self._bits.append(1 if bit else 0)

    # --- auxiliary fields: the mechanism every REFERENCE in clause 22 needs ----------------
    #
    # §22.8.3.7's NOTE states the problem exactly: "The encoding of the USING reference in
    # this case appears earlier in the encoding than the encoding of this field, and an
    # encoder will need to SUSPEND the encoding of that field until the value to be encoded
    # has been determined by the encoding of this field." A single forward pass cannot do
    # that, and neither can a two-pass encoder that re-runs everything — the second pass
    # would have to reproduce every encoder's-option decision the first one made. Reserving
    # the bits and patching them is the mechanism that makes one pass sufficient, and it is
    # shared by all three clauses that take a REFERENCE: §21.3's encoding-space determinant,
    # §22.3's start pointer and §22.8's unused-bits count.

    def reserve(self, name: str, width: int, form: "IntForm") -> None:
        """Set `width` bits aside for an auxiliary field whose value is not yet known."""
        if name in self._slots:
            raise Asn1Error(f"ECN: the field {name!r} is reserved twice in one encoding")
        self._slots[name] = (len(self._bits), width, form)
        self._bits.extend([0] * width)

    def patch(self, name: str, value: int) -> None:
        """Write `value` into a previously reserved field.

        §21.3.4 is the reason a second write is refused rather than overwriting: "If a field
        is set more than once through the use of `field-to-be-set` or `flag-to-be-set`, then
        it is an ECN specification or an application error if different values are produced
        by the different encoding procedures, and encoders shall not generate encodings in
        this case." Comparing the two values would implement the clause exactly; refusing the
        second write implements it conservatively, and the difference only shows up for a
        specification that sets one field from two places, which nothing here builds.
        """
        if name not in self._slots:
            raise Asn1Error(
                f"ECN: nothing reserved a field named {name!r}, so there is no room to set "
                f"it; a REFERENCE names a field that appears EARLIER in the encoding")
        if name in self._patched:
            raise Asn1Error(
                f"ECN: §21.3.4 — the field {name!r} is set more than once, and an encoder "
                f"shall not generate an encoding whose determinants disagree")
        at, width, form = self._slots[name]
        if form is IntForm.TWOS_COMPLEMENT:
            if value < -(1 << (width - 1)) or value >= (1 << (width - 1)):
                raise Asn1Error(
                    f"ECN: the determinant {value} does not fit the {width}-bit field "
                    f"{name!r} as two's complement")
            encoded = value & ((1 << width) - 1)
        else:
            if value < 0 or (width and value >> width):
                raise Asn1Error(
                    f"ECN: the determinant {value} does not fit the {width}-bit field "
                    f"{name!r}; §21.3.4 leaves the field's range a specification decision, "
                    f"so a value that overruns it is a specification error and not a "
                    f"wider field")
            encoded = value
        for index, shift in enumerate(range(width - 1, -1, -1)):
            self._bits[at + index] = (encoded >> shift) & 1
        self._patched.add(name)

    def mark_start(self, name: str) -> None:
        """Record where a named field's encoding space begins, for §22.3.3.1's count."""
        self._starts[name] = len(self._bits)

    def start_of(self, name: str) -> int:
        if name not in self._starts:
            raise Asn1Error(
                f"ECN: {name!r} has no recorded position, so no offset can be measured from "
                f"it; §22.3.3.1 counts from the start of the START-POINTER field's own "
                f"encoding, which therefore has to come first")
        return self._starts[name]

    def record_value(self, name: str, value: int) -> None:
        """Remember an ordinary field's abstract value, for §21.3.5's `field-to-be-used`.

        That determination reads a field "whose value may be set from the abstract syntax
        (i.e., a corresponding field appears within the ASN.1 specification)", so the number
        the encoder has to check against is the application's value and not the bits — the
        transforms between them are exactly what §21.3.5 makes the encoder reverse.
        """
        self._values[name] = value

    def value_of(self, name: str) -> int:
        if name not in self._values:
            raise Asn1Error(
                f"ECN: {name!r} carries no abstract value, so a `field-to-be-used` "
                f"determination has nothing to read; §21.3.5's field is one that appears in "
                f"the ASN.1 specification, not an auxiliary field the encoder sets")
        return self._values[name]

    def unpatched(self) -> tuple[str, ...]:
        """Reserved fields nobody set. Every one is a determinant that was never determined."""
        return tuple(sorted(set(self._slots) - self._patched))

    def reverse_all(self, reversal, unit: int) -> None:
        """§22.12.3.1's `#OUTER` reversal, over the entire encoding after padding.

        In place, and after every slot has been patched, because reversal moves the bits an
        auxiliary field occupies: patching a slot afterwards would write a determinant into
        positions the reversal had already relocated.
        """
        missing = self.unpatched()
        if missing:
            raise Asn1Error(
                f"ECN: {', '.join(missing)} still need their determinants, and a #OUTER bit "
                f"reversal would move the bits they occupy")
        self._bits = list(reversal.apply(tuple(self._bits), unit))

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

    **The encoder actions run in §23.3.3.1's order**, which is stated once per category and is
    the same list every time: replacement, pre-alignment and padding, start pointer, encoding
    space, value encoding, value padding and justification, identification handle, bit
    reversal. `write` below follows it step for step, and the order is not interchangeable —
    the start pointer is measured after pre-alignment (§22.3.3.1) and bit reversal is applied
    to the placed value including its padding but not to the alignment bits (§22.12.1.4's
    NOTE 2), so moving either would produce different octets.

    **The decoder's order is NOT the encoder's reversed**, which is worth stating because the
    obvious implementation gets it wrong. §23.3.4.1 gives it as pre-alignment, start pointer,
    encoding space, **bit reversal**, value padding, value decoding — bit reversal comes
    before value padding on the way in, where it came after on the way out.
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
    #: §22.3's group: an earlier field carrying this element's offset.
    start_pointer: StartPointer | None = None
    #: §21.3/§22.4's determinant: an earlier field carrying this space's size.
    space_determinant: SpaceDeterminant | None = None
    #: §22.12's group. `NO_REVERSAL` is §21.14.2's default, so this is the quiet case.
    bit_reversal: ReversalSpecification = ReversalSpecification.NO_REVERSAL
    #: The `MULTIPLE OF` unit §22.12.3.1 divides the space into. Only read when reversing.
    reversal_unit: int = UNIT_BIT

    def __post_init__(self) -> None:
        # §22.12.2.2/§22.12.2.3 relate the reversal to the space's unit, so the pair is
        # checked where both are stated rather than at write time — an object that could
        # never encode anything is invalid when it is written, not when it is first used.
        if self.bit_reversal is not ReversalSpecification.NO_REVERSAL:
            self.bit_reversal.check_unit(self.reversal_unit)

    def write(self, value: int, out: BitWriter) -> None:
        # b) Pre-alignment and padding.
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        # c) Start pointer. Measured here, after this element's own pre-alignment, which is
        #    exactly where §22.3.3.1 puts the second end of the span.
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        # e) Value encoding.
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
            space_bits = tuple(
                (encoded >> shift) & 1 for shift in range(self.width - 1, -1, -1))
            if encoded >> self.width:
                raise Asn1Error(
                    f"ECN: {value} does not fit the {self.width}-bit encoding space this "
                    f"object declares")
        else:
            # f) Value padding and justification. §22.8.3.2's "b" is the difference between
            #    the space and the *value encoding*, so the value's own width has to be a
            #    stated thing rather than the space's. For a positive-int that is its bit
            #    length, with a single zero bit for zero itself; a two's-complement encoding
            #    has no shorter form than the space it was checked against, so it is already
            #    exactly `width` and no padding arises.
            if self.form is IntForm.TWOS_COMPLEMENT:
                value_bits = tuple(
                    (encoded >> shift) & 1 for shift in range(self.width - 1, -1, -1))
            else:
                used = max(encoded.bit_length(), 1)
                value_bits = tuple(
                    (encoded >> shift) & 1 for shift in range(used - 1, -1, -1))
            space_bits = self.value_padding.place(value_bits, self.width, out)
        # h) Bit reversal, over the encoding space's contents and nothing else.
        space_bits = self.bit_reversal.apply(space_bits, self.reversal_unit)
        for bit in space_bits:
            out.put_bit(bit)
        # d) The encoding space's own determinant, which can only be written once the space
        #    is known. §21.2's NOTE permits a determinant even for a fixed size, and calls a
        #    disagreement between the two determinations an error — so it is checked here
        #    against the width this object actually wrote rather than against the declared one.
        if self.space_determinant is not None:
            self.space_determinant.record(out, len(space_bits))


@dataclass(frozen=True)
class AuxIntSpec:
    """An **auxiliary field**: an integer in the encoding that no abstract value supplies.

    §22.1.2.6 names the category — "All fields of the replacement structure that are not part
    of the encoding class parameter are auxiliary fields, and shall be set by the encoding of
    the replacement structure" — and §21.3.4's `field-to-be-set`, §22.3's start pointer and
    §22.8's `UNUSED BITS` all write into one. It is what a length field, an offset field or an
    unused-bit count *is*: bits the encoder owns, carrying a fact about the encoding rather
    than a fact about the value.

    Distinct from `PadSpec`, which also takes no abstract value: pad bits are *inert*, and
    these are determined later. The writer reserves the space and some other field's
    determinant patches it, so an auxiliary field nobody sets is a fault rather than zeros —
    `encode_with_user` checks that before returning octets.
    """

    width: int
    form: IntForm = IntForm.POSITIVE_INT
    pre_alignment: PreAlignment | None = None

    def reserve(self, name: str, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        out.mark_start(name)
        out.reserve(name, self.width, self.form)


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
    start_pointer: StartPointer | None = None
    space_determinant: SpaceDeterminant | None = None
    bit_reversal: ReversalSpecification = ReversalSpecification.NO_REVERSAL
    reversal_unit: int = UNIT_BIT

    def __post_init__(self) -> None:
        # §22.12.2.2/§22.12.2.3 relate the reversal to the space's unit, so the pair is
        # checked where both are stated rather than at write time — an object that could
        # never encode anything is invalid when it is written, not when it is first used.
        if self.bit_reversal is not ReversalSpecification.NO_REVERSAL:
            self.bit_reversal.check_unit(self.reversal_unit)

    def write(self, value: bool, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        pattern = self.true_value if value else self.false_value
        space_bits = tuple(
            (pattern >> shift) & 1 for shift in range(self.width - 1, -1, -1))
        if pattern >> self.width:
            raise Asn1Error(
                f"ECN: the pattern {pattern} does not fit this #BOOL object's "
                f"{self.width}-bit encoding space")
        space_bits = self.bit_reversal.apply(space_bits, self.reversal_unit)
        for bit in space_bits:
            out.put_bit(bit)
        if self.space_determinant is not None:
            self.space_determinant.record(out, len(space_bits))


@dataclass(frozen=True)
class ConditionalIntSpec:
    """§23.7's `#CONDITIONAL-INT`: an integer encoding **guarded by the type's bounds**.

    This is the piece that makes an ECN integer encoding schema-directed. §23.6.3.1: the
    encoder "shall select and apply the first `#CONDITIONAL-INT` encoding object in
    `ENCODING(S)` whose conditions are satisfied", and §21.11.3 makes those conditions tests
    "on the bounds of the integer" rather than on any value. So one object set encodes
    `INTEGER (0..255)` in eight bits and `INTEGER (0..65535)` in sixteen, from the schema,
    with no value involved in the choice.

    §23.7.2.4 permits at most one of `IF`, `IF-ALL` and `ELSE`; §23.7.2.2 makes `ELSE` and the
    omission of all three mean the same thing, "that there is no condition". An unconditional
    object is therefore represented by an empty `conditions` tuple, and `IF-ALL`'s several
    conditions by several entries — §23.7.2.2's three parallel lists are folded into one list
    of triples here, since the clause itself says they "shall be interpreted as a list of
    predicates using the values in corresponding positions in the three lists".

    §23.7.2.7 is enforced because it is a statement relating the transforms to the condition
    rather than to either alone: `subtract:lower-bound` "shall be included only if the `IF` or
    `IF-ALL` condition restricts the application of this encoding to classes of the integer
    category with a lower bound". A subtraction of a bound that may not exist is not a
    narrower encoding, it is an undefined one.
    """

    spec: IntSpec = None
    #: `(RangeCondition, Comparison | None, comparator | None)`, all of which must hold.
    conditions: tuple[tuple[RangeCondition, "Comparison | None", int | None], ...] = ()

    def __post_init__(self) -> None:
        if self.spec is None:
            raise Asn1Error("ECN: a #CONDITIONAL-INT object carries an integer encoding")
        # §23.7.2.9: the encoding space "shall not be set to self-delimiting-values", and its
        # NOTE says the default therefore "always has to be overridden". `IntSpec` states a
        # width, so that is already true by construction and there is nothing to check here.
        if self._subtracts_lower_bound() and not self._guarantees_a_lower_bound():
            raise Asn1Error(
                "ECN: §23.7.2.7 — the INT-TO-INT transform `subtract:lower-bound` shall be "
                "included only if the IF or IF-ALL condition restricts this encoding to "
                "classes with a lower bound; none of these conditions does")

    def _subtracts_lower_bound(self) -> bool:
        chain = self.spec.transform
        steps = chain.transforms if isinstance(chain, TransformChain) else (
            (chain,) if chain is not None else ())
        return any(isinstance(step, IntToInt)
                   and step.op is IntOp.SUBTRACT_LOWER_BOUND for step in steps)

    def _guarantees_a_lower_bound(self) -> bool:
        """Whether every set of bounds satisfying these conditions has a lower bound.

        Four of §21.11.4's five shapes do, and `test-lower-bound` does by construction — a
        bound that does not exist compares against nothing, so `satisfies` already reports
        False for it. `unbounded-or-no-lower-bound` is precisely the one that does not.
        """
        guaranteeing = {
            RangeCondition.SEMI_BOUNDED_WITH_NEGATIVES,
            RangeCondition.BOUNDED_WITH_NEGATIVES,
            RangeCondition.SEMI_BOUNDED_WITHOUT_NEGATIVES,
            RangeCondition.BOUNDED_WITHOUT_NEGATIVES,
            RangeCondition.TEST_LOWER_BOUND,
        }
        return any(condition in guaranteeing for condition, _c, _v in self.conditions)

    def applies(self, bounds: IntegerBounds) -> bool:
        """§23.7.2.2: no condition means it always applies; `IF-ALL` means all of them."""
        return all(bounds.satisfies(condition, comparison, comparator)
                   for condition, comparison, comparator in self.conditions)


@dataclass(frozen=True)
class IntSelector:
    """§23.6's `#INT`: a list of `#CONDITIONAL-INT` objects and the type's bounds.

    §23.6.2.2 permits exactly one of `ENCODING` and `ENCODINGS`; both name objects of the
    conditional class, so one is the single-element case of the other and only the list is
    modelled. §23.6.3.1 selects the first whose conditions hold, and makes it "an ECN
    specification error if none of the conditional encodings have conditions that are
    satisfied" — so falling off the end is a refusal, never a default encoding.

    The bounds live here rather than on the conditional objects because they are a property of
    the *type this object set is applied to*, and the same object set is meant to be applied
    to many types. §23.7.2.6's NOTE makes the same point from the other side: the condition is
    tested "on the bounds of the original value, and is not affected by these transforms".
    """

    encodings: tuple[ConditionalIntSpec, ...] = ()
    bounds: IntegerBounds = field(default_factory=IntegerBounds)

    def select(self) -> IntSpec:
        for candidate in self.encodings:
            if candidate.applies(self.bounds):
                return candidate.spec
        raise Asn1Error(
            f"ECN: §23.6.3.1 — no #CONDITIONAL-INT object's conditions are satisfied by "
            f"{self.bounds}; the clause makes that an ECN specification error rather than a "
            f"reason to fall back on a default encoding")

    def write(self, value: int, out: BitWriter) -> None:
        self.select().write(value, out)


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


class ReplaceAction(Enum):
    """§22.1.1.7's five replacement actions, which differ in *what* gets replaced.

    a) `REPLACE STRUCTURE` — "the encoding class to which this encoding object is applied is
       to be replaced completely".
    b) `REPLACE COMPONENT` / `REPLACE ALL COMPONENTS` — every component, with the same
       action. §22.1.1.8 makes the singular "a synonym for" the plural, "normal but not
       required" when there is one component, so they are one value here and not two.
    c) `REPLACE OPTIONALS`, d) `REPLACE NON-OPTIONALS` — which need the optionality category.
    """

    STRUCTURE = "structure"
    ALL_COMPONENTS = "all-components"
    OPTIONALS = "optionals"
    NON_OPTIONALS = "non-optionals"


@dataclass(frozen=True)
class ReplacementStructure:
    """§22.1.2.2's replacement structure: parameterized, with a single encoding class parameter.

    "The `WITH` replacement structures shall be parameterized encoding structures with a
    single encoding class parameter." So a replacement is a small structure with a hole in it
    — `#Length-prefixed{#D} ::= #CONCATENATION { length #INT, value #D }` — and replacing a
    component instantiates that hole with the component's own class (§22.1.3.1).

    `dummy` names the field that IS the hole. §22.1.2.6 classifies the rest: "All fields of
    the replacement structure that are not part of the encoding class parameter are auxiliary
    fields, and shall be set by the encoding of the replacement structure" — which is why
    `auxiliary` holds `AuxIntSpec`s and why they are reserved rather than written.

    `determinant` is the `ENCODED BY` object's contribution (§22.1.1.9, §22.1.2.4): the thing
    that connects the auxiliary field to the instantiated one. Its `reference` is a
    *structure-local* field name, which `expand` qualifies — otherwise replacing two
    components with the same structure would give both length fields the same name, and the
    second `reserve` would be the collision it should be.
    """

    name: str
    #: Field names in textual order, one of which is `dummy`.
    order: tuple[str, ...] = ()
    dummy: str = ""
    #: The non-dummy fields, which §22.1.2.6 makes auxiliary.
    auxiliary: dict = field(default_factory=dict)
    #: The `ENCODED BY` object's determinant, attached to the instantiated field.
    determinant: SpaceDeterminant | None = None

    def __post_init__(self) -> None:
        if self.dummy not in self.order:
            raise Asn1Error(
                f"ECN: §22.1.2.2 gives {self.name} a single encoding class parameter, and "
                f"{self.dummy!r} is not one of its fields")
        holes = [name for name in self.order if name not in self.auxiliary]
        if holes != [self.dummy]:
            raise Asn1Error(
                f"ECN: §22.1.2.2 — {self.name} shall have exactly one encoding class "
                f"parameter; {sorted(holes)} have no encoding object")

    def expand(self, field_name: str, spec) -> tuple[tuple[str, object], ...]:
        """Instantiate this structure around `spec`, per §22.1.3.1 and §22.1.3.5.

        Returns `(name, spec)` pairs in the replacement structure's textual order. The
        instantiated field keeps the *original* field's name so the abstract value still
        reaches it — §22.1.3.5: "All abstract values and tag numbers of the original
        structure or component shall be mapped to corresponding abstract values and tag
        numbers in the actual parameter of the replacement structure."
        """
        out: list[tuple[str, object]] = []
        for member in self.order:
            if member == self.dummy:
                out.append((field_name, self._instantiate(field_name, spec)))
            else:
                out.append((f"{field_name}${member}", self.auxiliary[member]))
        return tuple(out)

    def _instantiate(self, field_name: str, spec):
        if self.determinant is None:
            return spec
        if not hasattr(spec, "space_determinant"):
            raise Asn1Error(
                f"ECN: {type(spec).__name__} has no encoding space, so {self.name}'s "
                f"ENCODED BY object has nothing to determine")
        if spec.space_determinant is not None:
            raise Asn1Error(
                f"ECN: {field_name!r} already carries an encoding-space determinant, and "
                f"{self.name} would add a second; §21.2's NOTE calls a disagreement between "
                f"two determinations an error, so two are refused rather than raced")
        return replace(spec, space_determinant=replace(
            self.determinant, reference=f"{field_name}${self.determinant.reference}"))


@dataclass(frozen=True)
class HeadEndStructure:
    """§22.1.2.7's head-end insertion, which is a replacement structure **without the hole**.

    "The `INSERT AT HEAD` encoding structures shall not have dummy parameters. All their
    fields are auxiliary fields, and shall be set by the `ENCODED BY` encoding object through
    its `REFERENCE` parameter." So it is a distinct type rather than a `ReplacementStructure`
    with an optional dummy: the two differ in exactly the property that makes one of them a
    parameterized structure, and §22.1's NOTE says what the difference buys — "These
    structures will normally be a simple integer field providing a location determinant for
    the field being replaced."
    """

    name: str
    order: tuple[str, ...] = ()
    auxiliary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        unbacked = [name for name in self.order if name not in self.auxiliary]
        if unbacked:
            raise Asn1Error(
                f"ECN: §22.1.2.7 — all fields of the head-end structure {self.name} are "
                f"auxiliary fields; {sorted(unbacked)} have no encoding object")

    def expand(self, field_name: str) -> tuple[tuple[str, object], ...]:
        return tuple((f"{field_name}^{member}", self.auxiliary[member])
                     for member in self.order)


@dataclass(frozen=True)
class Replacement:
    """§22.1's replacement group, as the `REPLACE ... WITH ... ENCODED BY ...` syntax gives it.

    §22.1.2.8 is checked because it is a rule about the *combination*: "If an encoding object
    has a `REPLACE STRUCTURE` clause, it shall not have an `INSERT AT HEAD` clause and shall
    have an `ENCODED BY` clause."

    `REPLACE OPTIONALS` and `REPLACE NON-OPTIONALS` are refused rather than approximated:
    §22.1.1.7 c) and d) sort components by whether they are optional, and the optionality
    category is not built here, so "which components are optional" has no answer to sort by.
    """

    action: ReplaceAction
    structure: ReplacementStructure
    #: §22.1.1.10's head-end insertion: a structure "inserted before all components of the
    #: (constructor) class performing the replacement", one per replaced component, "in the
    #: textual order of the original components".
    head_end: "HeadEndStructure | None" = None

    def __post_init__(self) -> None:
        if self.action in (ReplaceAction.OPTIONALS, ReplaceAction.NON_OPTIONALS):
            raise Asn1Error(
                f"ECN: §22.1.1.7 sorts REPLACE {self.action.value.upper()} by whether each "
                f"component is optional, and the optionality category is not built on this "
                f"rail, so there is nothing to sort by")
        if self.action is ReplaceAction.STRUCTURE:
            if self.head_end is not None:
                raise Asn1Error(
                    "ECN: §22.1.2.8 — an encoding object with a REPLACE STRUCTURE clause "
                    "shall not have an INSERT AT HEAD clause")
            if self.structure.determinant is None:
                raise Asn1Error(
                    "ECN: §22.1.2.8 — an encoding object with a REPLACE STRUCTURE clause "
                    "shall have an ENCODED BY clause")


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
    #: §22.1's group. Applied when the fields are laid out, not when they are written, so a
    #: replacement is visible to `transmission_order` and therefore to the digest.
    replacement: Replacement | None = None

    def transmission_order(self) -> tuple[str, ...]:
        return tuple(name for name, _spec in self._laid_out())

    def _laid_out(self) -> tuple[tuple[str, object], ...]:
        """The fields as they actually appear, replacement expanded.

        §22.1.3.6 fixes where a head-end insertion goes: "the encoder shall insert the
        head-end structure **before all components** of the structure whose encoding object is
        performing the replacement. Head-end insertions shall be inserted in the same textual
        order as the components being replaced." So the insertions are hoisted to the front as
        a block, in component order — not interleaved with the components they belong to,
        which is the reading that first suggests itself and is what makes them useful as
        location determinants.
        """
        if self.order:
            missing = set(self.fields) - set(self.order)
            extra = set(self.order) - set(self.fields)
            if missing or extra:
                raise Asn1Error(
                    f"ECN: this #CONCATENATION object states a transmission order that does "
                    f"not match its fields (missing {sorted(missing)}, unknown "
                    f"{sorted(extra)})")
            names = self.order
        else:
            names = tuple(self.fields)
        if self.replacement is None:
            return tuple((name, self.fields[name]) for name in names)
        if self.replacement.action is ReplaceAction.STRUCTURE:
            raise Asn1Error(
                "ECN: §22.1.3.2 replaces the ENTIRE construction for a class in the encoding "
                "constructor category, so a #CONCATENATION object with REPLACE STRUCTURE "
                "describes a structure other than this one; apply the replacement object to "
                "the field whose class it replaces instead")
        heads: list[tuple[str, object]] = []
        body: list[tuple[str, object]] = []
        for name in names:
            if self.replacement.head_end is not None:
                heads.extend(self.replacement.head_end.expand(name))
            body.extend(self.replacement.structure.expand(name, self.fields[name]))
        return tuple(heads) + tuple(body)

    def write(self, value: dict, out: BitWriter) -> None:
        for name, spec in self._laid_out():
            # An auxiliary field's bits are reserved where it sits and written later, by
            # whichever determinant references it. §22.8.3.7's NOTE describes exactly this
            # suspension, and it is the only way a length can precede what it measures in a
            # single pass.
            if isinstance(spec, AuxIntSpec):
                spec.reserve(name, out)
                continue
            if name in self.padding:
                out.mark_start(name)
                spec.write(None, out)
                continue
            if name not in value:
                raise Asn1Error(
                    f"ECN: this #CONCATENATION object encodes {name!r}, which the value does "
                    f"not carry; a user-defined encoding has no optionality unless an "
                    f"#OPTIONAL object supplies it")
            out.mark_start(name)
            # Recorded before writing so §21.3.5's `field-to-be-used` can read the abstract
            # value rather than re-derive it from bits it would have to un-transform.
            if isinstance(value[name], int) and not isinstance(value[name], bool):
                out.record_value(name, value[name])
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
    #: §22.12.2.1 makes bit reversal available "within `#OUTER`", and §22.12.3.1 gives it a
    #: different subject there: "the entire encoding (**after any PADDING has been applied**)
    #: shall be divided into MULTIPLE OF units". So a `#OUTER` reversal runs over the padded
    #: whole, where a field's reversal runs over that field's space alone.
    bit_reversal: ReversalSpecification = ReversalSpecification.NO_REVERSAL
    reversal_unit: int = UNIT_BIT

    def finish(self, out: BitWriter) -> None:
        if self.boundary_bits <= 0:
            raise Asn1Error("ECN: an alignment boundary is a positive number of bits")
        short = (-out.bit_length) % self.boundary_bits
        for bit in _padding_bits(self.padding, self.pattern, short, "#OUTER post-padding"):
            out.put_bit(bit)
        if self.bit_reversal is not ReversalSpecification.NO_REVERSAL:
            out.reverse_all(self.bit_reversal, self.reversal_unit)


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
    # Every auxiliary field is a determinant, and a determinant nobody wrote is not zero — it
    # is a length, an offset or an unused-bit count that no clause ever produced. Checked
    # before `#OUTER` runs, since a reversal there would relocate the reserved bits.
    missing = out.unpatched()
    if missing:
        raise Asn1Error(
            f"ECN: the auxiliary field{'s' if len(missing) > 1 else ''} "
            f"{', '.join(repr(name) for name in missing)} "
            f"{'were' if len(missing) > 1 else 'was'} reserved and never set; an auxiliary "
            f"field exists to carry a determinant, so one that no ENCODING-SPACE, "
            f"START-POINTER or UNUSED BITS group references would transmit zeros as though "
            f"they meant something")
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
    "AuxIntSpec", "BitWriter", "BoolSpec", "Comparison", "ConcatenationSpec",
    "ConditionalIntSpec", "EncodingSpaceDetermination", "FIXED_CANDIDATES", "IntForm",
    "IntOp", "IntSelector", "IntSpec", "IntToBits", "IntToInt", "IntegerBounds",
    "Justification", "JustificationSide", "OuterSpec", "PadSpec", "Padding", "Pattern",
    "PatternKind", "PreAlignment", "RangeCondition", "ReplaceAction",
    "Replacement", "ReplacementStructure", "HeadEndStructure",
    "ReversalSpecification",
    "SpaceDeterminant", "StartPointer", "Transform", "TransformChain", "UnusedBits",
    "UnusedBitsDetermination", "UserEncodingObject", "ValuePadding", "check_unit",
    "encode_with_user", "legacy_frame_objects", "legacy_frame_workload", "refuted_by",
]
