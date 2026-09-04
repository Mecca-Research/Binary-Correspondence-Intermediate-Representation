"""X.692 clause 24: the `#TRANSFORM` encoding class.

A transform makes the *transmitted* value a declared function of the abstract one, which is
the whole mechanism the §6 reduction gate was reopened for. Every rule in the fixed candidate
set encodes the abstract value; §24.2.1 lets an ECN specification interpose a procedure that
"transform[s] input abstract values (the source) into output abstract values of the same or a
different type (the result)".

Clause 24 defines nineteen of them and this module implements them. They divide into three
groups that behave differently, and telling them apart is most of the work:

* **Value transforms** — `int-to-int`, `bool-to-bool`, `bool-to-int`, `int-to-bool`,
  `int-to-chars`, `int-to-bits`, `bits-to-int`, `char-to-bits`, `bits-to-char`, `bit-to-bits`,
  `bits-to-bits`. Each takes one abstract value and produces one.
* **Composite constructors** — `chars-to-composite-char`, `bits-to-composite-bits`,
  `octets-to-composite-bits`. These take a *string* and produce §24.2.1's transform composite:
  an ordered list whose elements the value transforms then apply to one at a time.
* **Composite collapsers** — `composite-char-to-chars`, `composite-bits-to-bits`,
  `composite-bits-to-octets`. The inverse direction.

THE COMPOSITE RULE IS UNIFORM AND IS IMPLEMENTED ONCE. Every value transform states it the
same way — §24.4.4 is representative: "If the source is a boolean, the result is a boolean. If
the source is a boolean composite, the result is a boolean composite in which each element of
the source has been transformed as specified". So `Transform.apply` maps over a `Composite`
and each transform only implements the scalar case. Writing that per transform would be
eleven chances to get it wrong in a way no test distinguishes.

REVERSIBILITY IS PER TRANSFORM AND THE CLAUSE SAYS SO EACH TIME. Most are "defined to be
reversible for all abstract values". Three are not, and they are the interesting ones:
§24.9.6 says `bits-to-int` "shall not be used where reversible transforms are required" at
all; §24.6.9 makes `int-to-bool` reversible "if and only if both TRUE-IS and FALSE-IS are set,
and they each specify a single integer value"; and Table 6 gives `int-to-int` a per-operation
condition. So `reversible()` reports rather than refuses, and only a caller needing a
reversible chain — a determinant, per §22.8.2.4 — turns the report into an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .ecn_props import (
    UNIT_BIT,
    IntForm,
    Pattern,
    PatternKind,
    check_unit,
)
from .tags import Asn1Error


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
class Composite:
    """§24.2.1's *transform composite*: an ordered list a value transform maps over.

    Clause 24 has three kinds of transform, and this type is what distinguishes them. A value
    transform takes one abstract value; a composite constructor (§24.14–§24.16) turns a string
    into a list of its elements; a composite collapser (§24.17–§24.19) puts one back together.
    In between, every value transform applies **elementwise** — which is why the composite is
    a type here rather than a loop written eleven times.

    `unit` is the bits per element for a bitstring composite (§24.15.5's `UNIT`). It is
    load-bearing at the far end: §24.19.1 makes it "an ECN specification error if this is
    applied to a bitstring composite that has a unit size which is not 8", and §24.18.5's NOTE
    explains why the round trip works at all — "the units used in its generation are specified
    in the transform that produced the bitstring composite, **and are associated with that
    composite**". So the composite carries its own unit; nothing downstream has to guess.

    `kind` records what the elements are, because the clauses type their sources: a
    `bits-to-char` applied to a *character* composite is a specification error, not a coercion.
    """

    elements: tuple = ()
    kind: str = "bits"  # "bits", "char", "bool", "int"
    unit: int = 0  # bits per element, for bitstring composites only

    def of(self, elements, kind: str | None = None) -> "Composite":
        """A composite of the same shape carrying transformed elements."""
        return Composite(tuple(elements), self.kind if kind is None else kind, self.unit)


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

    #: What this transform's result elements are, when applied to a composite. `None` keeps
    #: the source's kind, which is what a same-type transform like `int-to-int` does.
    result_kind: str | None = None

    def apply(self, value):
        """§24.x.4's uniform composite rule, applied once for every transform.

        Every value transform in clause 24 states it the same way — §24.4.4 is representative:
        "If the source is a boolean, the result is a boolean. If the source is a boolean
        composite, the result is a boolean composite in which each element of the source has
        been transformed as specified." So the mapping lives here and each transform writes
        only `apply_one`. Eleven hand-written loops would be eleven chances to differ in a way
        no single test distinguishes.

        The composite constructors and collapsers override this, because for them the
        composite is the *subject* rather than a container to map through.
        """
        if isinstance(value, Composite):
            return value.of(
                [self.apply_one(element) for element in value.elements], self.result_kind
            )
        return self.apply_one(value)

    def inverse(self, value):
        if isinstance(value, Composite):
            return value.of([self.inverse_one(element) for element in value.elements])
        return self.inverse_one(value)

    def apply_one(self, value):
        raise NotImplementedError

    def inverse_one(self, value):
        raise NotImplementedError

    def reversible(self, value) -> bool:
        """Whether this transform can be undone for `value`. See Table 6 for `int-to-int`.

        Reported per ELEMENT for a composite and then conjoined, because the conditions clause
        24 states are about abstract values: §24.6.9's `int-to-bool` and Table 6's `divide:n`
        both turn on the value in hand, so a composite is reversible exactly when every element
        of it is.
        """
        if isinstance(value, Composite):
            return all(self.reversible_one(element) for element in value.elements)
        return self.reversible_one(value)

    def reversible_one(self, value) -> bool:
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
        floor = {
            IntOp.INCREMENT: 1,
            IntOp.DECREMENT: 1,
            IntOp.MULTIPLY: 2,
            IntOp.DIVIDE: 2,
            IntOp.MODULO: 2,
        }.get(self.op)
        if floor is not None and self.operand < floor:
            raise Asn1Error(
                f"ECN: §24.3.1 constrains {self.op.value} to INTEGER ({floor}..MAX); "
                f"got {self.operand}"
            )

    def apply_one(self, value: int) -> int:
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

    def inverse_one(self, value: int) -> int:
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
                "ECN: Table 6 lists modulo:n as Never reversible; there is no inverse to take"
            )
        return value + self.operand

    def reversible_one(self, value: int) -> bool:
        condition = _REVERSAL[self.op]
        return True if condition is None else condition(value, self.operand)


#: §21.15.1's `ResultSize ::= INTEGER {variable(-1), fixed-to-max(0)} (-1..MAX)`, which
#: §24.7, §24.8 and §24.10 all take. Named so the two special values are not bare magic.
RESULT_SIZE_VARIABLE = -1
RESULT_SIZE_FIXED_TO_MAX = 0


def _bits_of(value: int, width: int, form: IntForm) -> tuple[int, ...]:
    """`width` bits of `value`, most significant first (§24.8.10)."""
    if form is IntForm.TWOS_COMPLEMENT:
        value &= (1 << width) - 1
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


def _minimal_bits(value: int, form: IntForm) -> tuple[int, ...]:
    """§24.8.11's *initial* bitstring: "the minimum number of bits necessary".

    The clause spells the minimality condition rather than a width, and the two forms differ:
    "a positive integer encoding shall not have zero as the leading bit (unless there is a
    single zero bit in the encoding), and a 2's-complement encoding shall not have two
    successive leading zero bits or two successive leading one bits."
    """
    if form is IntForm.POSITIVE_INT:
        if value < 0:
            raise Asn1Error(
                f"ECN: §24.8.12 — INT-TO-BITS AS positive-int cannot transform the negative "
                f"value {value}; encoders shall not encode such values"
            )
        return _bits_of(value, max(value.bit_length(), 1), form)
    # Two's complement: grow until the value fits with its sign bit, which is exactly the
    # "no two successive leading zero or one bits" condition read as a width.
    width = 1
    while not (-(1 << (width - 1)) <= value < (1 << (width - 1))):
        width += 1
    return _bits_of(value, width, form)


@dataclass(frozen=True)
class IntToBits(Transform):
    """§24.8 `INT-TO-BITS`: an integer as a bitstring.

    §24.8.1's properties: `&int-to-bits-encoded-as ENUMERATED {positive-int, twos-complement}
    DEFAULT twos-complement`, `&int-to-bits-unit Unit (1..MAX) DEFAULT bit`, and
    `&int-to-bits-size ResultSize DEFAULT variable`.

    The size rule has three arms and they are not interchangeable. §24.8.13's `variable` makes
    the *initial* bitstring the result — the minimum width of §24.8.11. §24.8.14's positive
    size makes it `MULTIPLE OF × SIZE` bits. §24.8.15's `fixed-to-max` is "the smallest
    multiple of MULTIPLE OF that is large enough to receive the encoding of any abstract value
    of the class", which needs the class's bounds and so takes them as an argument rather than
    guessing.

    §24.8.17's padding is sign-aware, which is the part a naive implementation gets wrong: a
    positive-int encoding "shall have zero bits prefixed", and a two's-complement one "shall
    have bits prefixed **equal in value to the original leading bit**". Zero-extending a
    negative two's-complement value changes its sign.
    """

    encoded_as: IntForm = IntForm.TWOS_COMPLEMENT
    size: int = RESULT_SIZE_VARIABLE
    unit: int = UNIT_BIT
    #: The source class's bounds, needed only by `fixed-to-max` (§24.8.8, §24.8.15).
    bounds: tuple[int, int] | None = None
    result_kind: str | None = "bits"

    def __post_init__(self) -> None:
        check_unit(self.unit, allow_repetitions=False)
        if self.size < RESULT_SIZE_VARIABLE:
            raise Asn1Error(f"ECN: §21.15.1 constrains ResultSize to (-1..MAX); got {self.size}")
        if self.size == RESULT_SIZE_FIXED_TO_MAX and self.bounds is None:
            raise Asn1Error(
                "ECN: §24.8.8 — SIZE shall not be set to `fixed-to-max` unless the source "
                "class has both lower and upper bounds; none are stated"
            )

    def width(self) -> int | None:
        """The result width in bits, or `None` for §24.8.13's variable case."""
        if self.size == RESULT_SIZE_VARIABLE:
            return None
        if self.size > 0:
            return self.size * self.unit  # §24.8.14
        low, high = self.bounds  # §24.8.15
        widest = max(
            len(_minimal_bits(low, self.encoded_as)), len(_minimal_bits(high, self.encoded_as))
        )
        units = -(-widest // self.unit)  # smallest multiple of the unit
        return units * self.unit

    def apply_one(self, value: int) -> tuple[int, ...]:
        initial = _minimal_bits(value, self.encoded_as)
        width = self.width()
        if width is None:
            return initial  # §24.8.13
        if len(initial) > width:
            raise Asn1Error(
                f"ECN: §24.8.16 — the initial bitstring for {value} is {len(initial)} bits, "
                f"too large for the fixed size of {width}; encoders shall not encode such "
                f"values"
            )
        # §24.8.17: zero for positive-int, the ORIGINAL LEADING BIT for two's complement.
        fill = 0 if self.encoded_as is IntForm.POSITIVE_INT else initial[0]
        return (fill,) * (width - len(initial)) + initial

    def inverse_one(self, value: tuple[int, ...]) -> int:
        return _int_from_bits(tuple(value), self.encoded_as)

    def self_delimiting(self) -> bool:
        """§24.8.18: self-delimiting "if and only if SIZE is not variable"."""
        return self.size != RESULT_SIZE_VARIABLE


def _int_from_bits(bits: tuple[int, ...], form: IntForm) -> int:
    """X.690 §8.3.2/§8.3.3, which §24.9.5 and §24.8.9 both point at rather than restate."""
    if not bits:
        raise Asn1Error("ECN: an empty bitstring encodes no integer")
    out = 0
    for bit in bits:
        out = (out << 1) | (1 if bit else 0)
    if form is IntForm.TWOS_COMPLEMENT and bits[0]:
        out -= 1 << len(bits)
    return out


@dataclass(frozen=True)
class BoolToBool(Transform):
    """§24.4 `BOOL-TO-BOOL`. §24.4.5: "There is only one value ... `AS logical:not`".

    A transform with exactly one possible setting looks pointless until you notice what it
    buys: an active-low flag written as a *transform* rather than as swapped patterns keeps
    the boolean's encoding object saying `TRUE-PATTERN '1'B`, which is what every reader
    expects, and puts the inversion where the specification can see it.
    """

    result_kind: str | None = "bool"

    def apply_one(self, value: bool) -> bool:
        return not value

    def inverse_one(self, value: bool) -> bool:
        return not value  # §24.4.6: reversible for all values


@dataclass(frozen=True)
class BoolToInt(Transform):
    """§24.5 `BOOL-TO-INT AS {true-zero, true-one}`, default `true-one`.

    §24.5.3: "The result has no associated bounds" — so a `fixed-to-max` downstream of this
    has nothing to size itself from, which is why §24.8.8 requires the bounds to come from the
    source class and not from an intermediate result.
    """

    true_zero: bool = False  # §24.5.1's DEFAULT is true-one
    result_kind: str | None = "int"

    def apply_one(self, value: bool) -> int:
        if self.true_zero:
            return 0 if value else 1  # §24.5.5
        return 1 if value else 0

    def inverse_one(self, value: int) -> bool:
        return (value == 0) if self.true_zero else (value != 0)


@dataclass(frozen=True)
class IntToBool(Transform):
    """§24.6 `INT-TO-BOOL`, whose four settings are not four spellings of one thing.

    §24.6.4 states the combinations: "Either one of `AS`, `TRUE-IS` and `FALSE-IS` is set, or
    both `TRUE-IS` and `FALSE-IS` are set (and `AS` is not set), or none are set."

    §24.6.9 is the reason this transform exists as a separate case rather than as a `bits-to-`
    style lossy one: it is "reversible if and only if both `TRUE-IS` and `FALSE-IS` are set,
    and they each specify a single integer value". Anything else maps many integers onto two
    booleans, and a decoder cannot pick which integer it was.
    """

    zero_true: bool = False  # §24.6.1's DEFAULT is zero-false
    true_is: tuple[int, ...] | None = None
    false_is: tuple[int, ...] | None = None
    result_kind: str | None = "bool"

    def __post_init__(self) -> None:
        if self.true_is is not None and self.false_is is not None:
            overlap = set(self.true_is) & set(self.false_is)
            if overlap:
                raise Asn1Error(
                    f"ECN: §24.6.8 — the integer values in TRUE-IS and FALSE-IS shall be "
                    f"disjoint; {sorted(overlap)} are in both"
                )

    def apply_one(self, value: int) -> bool:
        if self.true_is is not None and self.false_is is not None:
            if value in self.true_is:
                return True
            if value in self.false_is:
                return False
            # §24.6.8: "it is an ECN specification or application error if abstract values
            # which are not included in either TRUE-IS or FALSE-IS are included in the source,
            # and encoders shall not generate encodings for such values."
            raise Asn1Error(
                f"ECN: §24.6.8 — {value} is in neither TRUE-IS nor FALSE-IS, and encoders "
                f"shall not generate encodings for such values"
            )
        if self.true_is is not None:
            return value in self.true_is  # §24.6.6
        if self.false_is is not None:
            return value not in self.false_is  # §24.6.7
        return (value == 0) if self.zero_true else (value != 0)  # §24.6.5

    def inverse_one(self, value: bool) -> int:
        if not self.reversible_one(value):
            raise Asn1Error(
                "ECN: §24.6.9 — INT-TO-BOOL is reversible only when TRUE-IS and FALSE-IS are "
                "both set to a single integer each; this one maps many integers onto two "
                "booleans and no decoder can choose among them"
            )
        return self.true_is[0] if value else self.false_is[0]

    def reversible_one(self, value) -> bool:
        return (
            self.true_is is not None
            and self.false_is is not None
            and len(self.true_is) == 1
            and len(self.false_is) == 1
        )


@dataclass(frozen=True)
class IntToChars(Transform):
    """§24.7 `INT-TO-CHARS`: an integer as its decimal characters.

    §24.7.9 fixes the spelling exactly: "converted to a decimal representation with no leading
    zeros and with a pre-fixed `-` (HYPHEN-MINUS) if it is negative. If, and only if,
    `PLUS-SIGN` is set to true, positive values have a `+` (PLUS SIGN) pre-fixed."

    **§24.7.13 puts the padding in a surprising place, and the literal reading is implemented.**
    The clause pads "with either ` ` (SPACE) or `0` (DIGIT ZERO), determined by the value of
    `PADDING`, **pre-fixed** to produce the specified size" — and what it pre-fixes to is
    §24.7.9's representation, which already carries the sign. So `-7` in a four-character field
    with `PADDING zeros` is `00-7`, not the `-007` a reader expects from every printf in
    existence. Nothing in clause 24 says the pad goes after the sign; saying so would have
    taken a sentence, and the clause does not spend it.

    Written down because it is exactly the kind of divergence that survives review: `-007`
    looks right, round-trips through most parsers, and would be wrong. The inverse here reads
    both spellings so a decoder is not made brittle by the choice, but the encoder emits only
    what the clause specifies.
    """

    size: int = RESULT_SIZE_VARIABLE
    plus_sign: bool = False
    pad_with_spaces: bool = False  # §24.7.1's DEFAULT is zeros
    bounds: tuple[int, int] | None = None
    result_kind: str | None = "char"

    def __post_init__(self) -> None:
        if self.size < RESULT_SIZE_VARIABLE:
            raise Asn1Error(f"ECN: §21.15.1 constrains ResultSize to (-1..MAX); got {self.size}")
        if self.size == RESULT_SIZE_FIXED_TO_MAX and self.bounds is None:
            raise Asn1Error(
                "ECN: §24.7.8 — SIZE shall not be set to `fixed-to-max` unless the source "
                "class has both lower and upper bounds"
            )

    def _digits(self, value: int) -> str:
        text = str(abs(value))
        if value < 0:
            return "-" + text
        return ("+" + text) if self.plus_sign else text

    def width(self) -> int | None:
        if self.size == RESULT_SIZE_VARIABLE:
            return None
        if self.size > 0:
            return self.size
        low, high = self.bounds
        return max(len(self._digits(low)), len(self._digits(high)))

    def apply_one(self, value: int) -> str:
        text = self._digits(value)
        width = self.width()
        if width is None:
            return text  # §24.7.11
        if len(text) > width:
            raise Asn1Error(
                f"ECN: §24.7.12 — {text!r} is {len(text)} characters, too large for the "
                f"fixed size of {width}; encoders shall not generate encodings for such "
                f"abstract values"
            )
        pad = " " if self.pad_with_spaces else "0"
        return pad * (width - len(text)) + text  # §24.7.13: prefixed

    def inverse_one(self, value: str) -> int:
        # Strips SPACE and DIGIT ZERO padding from either side of the sign, so a decoder reads
        # both `00-7` (what §24.7.13 specifies) and `-007` (what most other formats write).
        # Being liberal about the PADDING costs nothing; being liberal about what counts as a
        # DIGIT is not the same thing, and that is what handing the remainder to `int()` did.
        # Python's `int` accepts PEP 515 underscores and every Unicode decimal digit, so
        # `4_2` and `۴۲` (ARABIC-INDIC FOUR TWO) both arrived as 42 -- character strings that
        # §24.7.4's repertoire, DIGIT ZERO through DIGIT NINE, does not contain at all. This
        # is a decoder over bytes a peer chose, so its accepted language has to be the
        # clause's, not the host language's.
        text = value.strip().lstrip("0") or "0"
        sign = 1
        if text[:1] in ("-", "+"):
            sign = -1 if text[0] == "-" else 1
            text = text[1:].lstrip("0") or "0"
        if not text.isascii() or not text.isdigit():
            raise Asn1Error(
                f"ECN: §24.7.4 — {value!r} is not a string of DIGIT ZERO..DIGIT NINE; "
                f"{text!r} contains a character the repertoire does not hold"
            )
        return sign * int(text)


@dataclass(frozen=True)
class BitsToInt(Transform):
    """§24.9 `BITS-TO-INT`, the one transform clause 24 forbids outright where reversibility
    is required.

    §24.9.6: "This transform shall not be used where reversible transforms are required." Not
    "is reversible under conditions" — never. §24.9.3 says why in passing: "There are no bounds
    associated with the result", and the bit width is not recoverable from the integer, so
    `0011` and `11` both decode to 3 and nothing distinguishes them.
    """

    decoded_assuming: IntForm = IntForm.TWOS_COMPLEMENT
    result_kind: str | None = "int"

    def apply_one(self, value: tuple[int, ...]) -> int:
        return _int_from_bits(tuple(value), self.decoded_assuming)

    def inverse_one(self, value: int) -> tuple[int, ...]:
        raise Asn1Error(
            "ECN: §24.9.6 — BITS-TO-INT shall not be used where reversible transforms are "
            "required; the source bitstring's width is not recoverable from the integer"
        )

    def reversible_one(self, value) -> bool:
        return False


@dataclass(frozen=True)
class CharToBits(Transform):
    """§24.10 `CHAR-TO-BITS`: one character as bits, three ways.

    §24.10.11's `iso10646` takes the character's own numerical value. §24.10.12's `compact`
    takes its *index in the effective permitted alphabet*, which is how PER shrinks a
    constrained string — and §24.10.12 makes it "an ECN specification error if there is no
    effective permitted alphabet constraint", because an index into nothing is not a number.
    §24.10.10's `mapped` reads two parallel lists.

    Both `iso10646` and `compact` then defer to §24.8's `INT-TO-BITS AS positive-int` with this
    transform's own `SIZE` and `MULTIPLE OF` — §24.10.11.4 spells that out as a nested
    transform rather than restating the bit rules, and this follows it literally by building
    one.
    """

    encoded_as: str = "compact"  # §24.10.1's DEFAULT
    alphabet: str = ""  # the effective permitted alphabet
    chars: tuple[str, ...] = ()  # §24.10.10.1's CHAR-LIST
    bit_values: tuple[tuple[int, ...], ...] = ()  # §24.10.10.1's BITS-LIST
    size: int = RESULT_SIZE_VARIABLE
    unit: int = UNIT_BIT
    result_kind: str | None = "bits"

    def __post_init__(self) -> None:
        if self.encoded_as not in ("iso10646", "compact", "mapped"):
            raise Asn1Error(
                f"ECN: §24.10.1's &char-to-bits-encoded-as is ENUMERATED "
                f"{{iso10646, compact, mapped}}; got {self.encoded_as!r}"
            )
        if self.encoded_as == "mapped":
            # §24.10.8: the two lists "are only used if AS is set to `mapped`, in which case
            # their presence is mandatory, and they shall then contain at least one element".
            if not self.chars or not self.bit_values:
                raise Asn1Error(
                    "ECN: §24.10.8 — CHAR-LIST and BITS-LIST are mandatory for "
                    "`AS mapped` and shall contain at least one element"
                )
            if len(self.chars) != len(self.bit_values):
                raise Asn1Error(
                    f"ECN: §24.10.10.2 — there shall be an equal number of values in each "
                    f"list; CHAR-LIST has {len(self.chars)} and BITS-LIST "
                    f"{len(self.bit_values)}"
                )
            if len(set(self.chars)) != len(self.chars):
                raise Asn1Error(
                    "ECN: §24.10.10.2 — all character values in CHAR-LIST shall be distinct"
                )
        elif self.encoded_as == "compact" and not self.alphabet:
            raise Asn1Error(
                "ECN: §24.10.12 — `AS compact` is an ECN specification error if there is no "
                "effective permitted alphabet constraint; a character's index into an "
                "alphabet that does not exist is not a number"
            )
        else:
            check_unit(self.unit, allow_repetitions=False)

    def _nested(self, high: int) -> IntToBits:
        """§24.10.11.4/§24.10.12.3's nested `INT-TO-BITS AS positive-int`, built rather than
        restated — the clause defines this transform *in terms of* that one."""
        return IntToBits(
            encoded_as=IntForm.POSITIVE_INT,
            size=self.size,
            unit=self.unit,
            bounds=(0, high) if self.size == RESULT_SIZE_FIXED_TO_MAX else None,
        )

    def apply_one(self, value: str) -> tuple[int, ...]:
        if len(value) != 1:
            raise Asn1Error(f"ECN: §24.10.4 — CHAR-TO-BITS takes a single character; got {value!r}")
        if self.encoded_as == "mapped":
            try:
                return tuple(self.bit_values[self.chars.index(value)])  # §24.10.10.3
            except ValueError:
                raise Asn1Error(
                    f"ECN: §24.10.10.4 — {value!r} is not in the CHAR-LIST, which is an ECN "
                    f"specification or application error"
                ) from None
        if self.encoded_as == "compact":
            # §24.10.12.1: canonical order by ISO/IEC 10646 value, lowest first, then indexed
            # from zero. §24.10.12.2 gives the integer bounds 0..n-1.
            order = sorted(set(self.alphabet))
            if value not in order:
                raise Asn1Error(
                    f"ECN: {value!r} is not in the effective permitted alphabet, so "
                    f"§24.10.12.1 gives it no index"
                )
            return self._nested(len(order) - 1).apply_one(order.index(value))
        # §24.10.11.1-3: the ISO/IEC 10646 value, bounded by the alphabet when there is one
        # and by 0..32767 when there is not.
        point = ord(value)
        if self.alphabet:
            high = max(ord(character) for character in self.alphabet)
        else:
            high = 32767
        return self._nested(high).apply_one(point)

    def inverse_one(self, value: tuple[int, ...]) -> str:
        bits = tuple(value)
        if self.encoded_as == "mapped":
            for index, candidate in enumerate(self.bit_values):
                if tuple(candidate) == bits:
                    return self.chars[index]
            raise Asn1Error(f"ECN: {bits} is not in the BITS-LIST")
        number = _int_from_bits(bits, IntForm.POSITIVE_INT)
        if self.encoded_as == "compact":
            return sorted(set(self.alphabet))[number]
        return chr(number)

    def reversible_one(self, value) -> bool:
        if self.encoded_as != "mapped":
            return True  # §24.10.11.5, §24.10.12.4
        # §24.10.10.5: reversible "if and only if the set of all bitstring values in BITS-LIST
        # are distinct".
        return len({tuple(bits) for bits in self.bit_values}) == len(self.bit_values)


@dataclass(frozen=True)
class BitsToChar(Transform):
    """§24.11 `BITS-TO-CHAR`, `CHAR-TO-BITS` read backwards — with one asymmetry.

    §24.11.6.2 requires **both** lists distinct here, where §24.10.10.2 required only the
    characters. That is not an oversight in either: going char-to-bits, two characters mapping
    to one bitstring makes the transform lossy but still well defined; going bits-to-char, two
    identical bitstrings in the source list would make the transform itself ambiguous.
    """

    decoded_assuming: str = "iso10646"  # §24.11.1's DEFAULT
    bit_values: tuple[tuple[int, ...], ...] = ()
    chars: tuple[str, ...] = ()
    result_kind: str | None = "char"

    def __post_init__(self) -> None:
        if self.decoded_assuming not in ("iso10646", "mapped"):
            raise Asn1Error(
                f"ECN: §24.11.1's &bits-to-char-decoded-assuming is ENUMERATED "
                f"{{iso10646, mapped}}; got {self.decoded_assuming!r}"
            )
        if self.decoded_assuming == "mapped":
            if len(self.chars) != len(self.bit_values) or not self.chars:
                raise Asn1Error(
                    "ECN: §24.11.6.2 — there shall be an equal number of values in each list"
                )
            if len(set(self.chars)) != len(self.chars) or len(
                {tuple(bits) for bits in self.bit_values}
            ) != len(self.bit_values):
                raise Asn1Error(
                    "ECN: §24.11.6.2 — all character values AND all bitstring values in the "
                    "list shall be distinct"
                )

    def apply_one(self, value: tuple[int, ...]) -> str:
        bits = tuple(value)
        if self.decoded_assuming == "mapped":
            for index, candidate in enumerate(self.bit_values):
                if tuple(candidate) == bits:
                    return self.chars[index]  # §24.11.6.3
            raise Asn1Error(
                f"ECN: §24.11.6.4 — the bitstring {bits} is not in the BITS-LIST, which is an "
                f"ECN specification or application error"
            )
        number = _int_from_bits(bits, IntForm.POSITIVE_INT)
        if number > 32767:
            raise Asn1Error(
                f"ECN: §24.11.5 — it is an ECN specification error if the integer value "
                f"exceeds 32767; got {number}"
            )
        return chr(number)

    def inverse_one(self, value: str) -> tuple[int, ...]:
        if self.decoded_assuming == "mapped":
            return tuple(self.bit_values[self.chars.index(value)])
        return _minimal_bits(ord(value), IntForm.POSITIVE_INT)


@dataclass(frozen=True)
class BitToBits(Transform):
    """§24.12 `BIT-TO-BITS`: one bit becomes a stated pattern.

    §24.12.9's restriction is the sharp one, and it is stronger than "different": it is an ECN
    specification error "if `ZERO-PATTERN` and `ONE-PATTERN` are the same, **or if one is an
    initial sub-string of the other**". Two patterns that merely differ can still be
    undecodable when read from a stream — `01` and `011` share a prefix, so a decoder reaching
    `011` cannot tell one pattern followed by something from the other.
    """

    zero_pattern: Pattern = field(default_factory=lambda: Pattern.from_bits("0"))
    one_pattern: Pattern = field(default_factory=lambda: Pattern.from_bits("1"))
    result_kind: str | None = "bits"

    def __post_init__(self) -> None:
        for pattern, where in (
            (self.zero_pattern, "ZERO-PATTERN"),
            (self.one_pattern, "ONE-PATTERN"),
        ):
            if pattern.kind is PatternKind.ANY_OF_LENGTH:
                raise Asn1Error(
                    f"ECN: §24.12.7 — the `any-of-length` alternative shall not be used for {where}"
                )
            pattern.require_non_null(where)
        different = [
            pattern
            for pattern in (self.zero_pattern, self.one_pattern)
            if pattern.kind is PatternKind.DIFFERENT_ANY
        ]
        if len(different) > 1:
            raise Asn1Error(
                "ECN: §24.12.6 — at most one of ZERO-PATTERN and ONE-PATTERN shall be "
                "`different:any`"
            )
        if different:
            return  # the other's value is an encoder's option
        zero = self.zero_pattern.bit_sequence()
        one = self.one_pattern.bit_sequence()
        if zero == one or zero[: len(one)] == one or one[: len(zero)] == zero:
            raise Asn1Error(
                f"ECN: §24.12.9 — it is an ECN specification error if ZERO-PATTERN and "
                f"ONE-PATTERN are the same, or if one is an initial sub-string of the other; "
                f"{zero} and {one} are not distinguishable in a stream"
            )

    def apply_one(self, value: int) -> tuple[int, ...]:
        pattern = self.one_pattern if value else self.zero_pattern
        return pattern.bit_sequence()  # §24.12.8

    def inverse_one(self, value: tuple[int, ...]) -> int:
        bits = tuple(value)
        if bits == self.one_pattern.bit_sequence():
            return 1
        if bits == self.zero_pattern.bit_sequence():
            return 0
        raise Asn1Error(f"ECN: {bits} is neither the ZERO-PATTERN nor the ONE-PATTERN")


@dataclass(frozen=True)
class BitsToBits(Transform):
    """§24.13 `BITS-TO-BITS`: a stated substitution table.

    §24.13.7 requires the *sources* distinct — without that the transform is not a function.
    §24.13.11 then makes reversibility turn on the *results* being distinct, which is a
    separate property and the one a decoder needs.
    """

    source_values: tuple[tuple[int, ...], ...] = ()
    result_values: tuple[tuple[int, ...], ...] = ()
    result_kind: str | None = "bits"

    def __post_init__(self) -> None:
        if not self.source_values or not self.result_values:
            raise Asn1Error(
                "ECN: §24.13.5 — SOURCE-LIST and RESULT-LIST are required, and shall contain "
                "at least one element in the ordered list"
            )
        if len(self.source_values) != len(self.result_values):
            raise Asn1Error(
                f"ECN: §24.13.7 — there shall be an equal number of bitstring values in each "
                f"list; got {len(self.source_values)} and {len(self.result_values)}"
            )
        sources = [tuple(bits) for bits in self.source_values]
        if len(set(sources)) != len(sources):
            raise Asn1Error(
                "ECN: §24.13.7 — all bitstring values in SOURCE-LIST shall be distinct, or "
                "the transform is not a function"
            )

    def apply_one(self, value: tuple[int, ...]) -> tuple[int, ...]:
        bits = tuple(value)
        for index, candidate in enumerate(self.source_values):
            if tuple(candidate) == bits:
                return tuple(self.result_values[index])  # §24.13.8
        raise Asn1Error(
            f"ECN: §24.13.10 — the source bitstring {bits} is not in the SOURCE-LIST, which "
            f"is an ECN specification or application error"
        )

    def inverse_one(self, value: tuple[int, ...]) -> tuple[int, ...]:
        bits = tuple(value)
        for index, candidate in enumerate(self.result_values):
            if tuple(candidate) == bits:
                return tuple(self.source_values[index])
        raise Asn1Error(f"ECN: {bits} is not in the RESULT-LIST")

    def reversible_one(self, value) -> bool:
        results = [tuple(bits) for bits in self.result_values]
        return len(set(results)) == len(results)  # §24.13.11


# --- §24.14-§24.19: the composite constructors and collapsers ------------------------------
#
# These six are where a STRING becomes a list the value transforms map over, and back. They
# override `apply` rather than `apply_one` because for them the composite is the subject
# rather than a container -- which is exactly the distinction §24.2.1 draws when it separates
# "procedures that map a characterstring, octetstring or bitstring source into a transform
# composite" from the transforms that act on values.


@dataclass(frozen=True)
class CharsToCompositeChar(Transform):
    """§24.14: a characterstring to a single-character composite."""

    def apply(self, value: str) -> Composite:
        return Composite(tuple(value), "char")  # §24.14.4

    def inverse(self, value: Composite) -> str:
        return "".join(value.elements)


@dataclass(frozen=True)
class BitsToCompositeBits(Transform):
    """§24.15: a bitstring to a bitstring composite of equal-sized elements.

    §24.15.6 makes a source that is not a whole number of `UNIT` bits an error rather than a
    short final element, and §24.18.5's NOTE says why the pair round-trips: the unit "is
    specified in the transform that produced the bitstring composite, and [is] associated with
    that composite" — so `Composite.unit` carries it and the collapser needs no argument.
    """

    unit: int = UNIT_BIT

    def __post_init__(self) -> None:
        check_unit(self.unit, allow_repetitions=False)

    def apply(self, value: tuple[int, ...]) -> Composite:
        bits = tuple(value)
        if len(bits) % self.unit:
            raise Asn1Error(
                f"ECN: §24.15.6 — the source bitstring is {len(bits)} bits, which is not a "
                f"multiple of the {self.unit}-bit UNIT; this is an ECN specification or "
                f"application error"
            )
        elements = tuple(bits[at : at + self.unit] for at in range(0, len(bits), self.unit))
        return Composite(elements, "bits", self.unit)

    def inverse(self, value: Composite) -> tuple[int, ...]:
        return tuple(bit for element in value.elements for bit in element)


@dataclass(frozen=True)
class OctetsToCompositeBits(Transform):
    """§24.16: an octetstring to a bitstring composite of size 8. No properties (§24.16.2)."""

    def apply(self, value: bytes) -> Composite:
        elements = tuple(
            tuple((octet >> shift) & 1 for shift in range(7, -1, -1)) for octet in value
        )
        return Composite(elements, "bits", 8)

    def inverse(self, value: Composite) -> bytes:
        return bytes(_int_from_bits(element, IntForm.POSITIVE_INT) for element in value.elements)


@dataclass(frozen=True)
class CompositeCharToChars(Transform):
    """§24.17: a single-character composite back to a characterstring."""

    def apply(self, value: Composite) -> str:
        return "".join(value.elements)  # §24.17.4

    def inverse(self, value: str) -> Composite:
        return Composite(tuple(value), "char")


@dataclass(frozen=True)
class CompositeBitsToBits(Transform):
    """§24.18: a bitstring composite back to a bitstring.

    §24.18.5: "The result bitstring is not self-delimiting." Concatenating equal-width elements
    loses the boundaries, and the NOTE explains why the transform is still reversible — the
    unit came from the composite rather than from the result.
    """

    def apply(self, value: Composite) -> tuple[int, ...]:
        return tuple(bit for element in value.elements for bit in element)

    def inverse(self, value: tuple[int, ...]) -> Composite:
        raise Asn1Error(
            "ECN: §24.18 collapses a composite whose unit came from the transform that built "
            "it; inverting needs that unit, so reverse the pair rather than this alone"
        )


@dataclass(frozen=True)
class CompositeBitsToOctets(Transform):
    """§24.19: a bitstring composite of unit 8 to an octetstring.

    §24.19.1 makes any other unit "an ECN specification error", which is the one place a
    composite's recorded unit is checked rather than used.
    """

    def apply(self, value: Composite) -> bytes:
        if value.unit != 8:
            raise Asn1Error(
                f"ECN: §24.19.1 — it is an ECN specification error to apply "
                f"COMPOSITE-BITS-TO-OCTETS to a bitstring composite whose unit size is "
                f"{value.unit} rather than 8"
            )
        return bytes(_int_from_bits(element, IntForm.POSITIVE_INT) for element in value.elements)

    def inverse(self, value: bytes) -> Composite:
        return OctetsToCompositeBits().apply(value)


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
            if (
                isinstance(transform, IntToInt)
                and transform.op is IntOp.SUBTRACT_LOWER_BOUND
                and index != 0
            ):
                raise Asn1Error(
                    f"ECN: §24.3.9 — subtract:lower-bound shall only be used as the first of "
                    f"an ordered list of transforms; it is at position {index}"
                )

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
