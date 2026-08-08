"""Encoding Control Notation, part three: the defined syntax, read from text.

[`ecn.py`](ecn.py) is the class/object/object-set model; [`ecn_user.py`](ecn_user.py) is the
bit-level semantics a user-defined object denotes. Both are reachable only from Python. This
module closes that gap: it reads the notation X.692 actually defines — clause 20's *defined
syntax*, spelled out by the `WITH SYNTAX` statements of clauses 23 to 25 — and produces the
objects `ecn_user` executes.

WHY A PARSER IS NOT COSMETIC HERE. An encoding specification assembled field by field in
Python has no canonical form, and therefore no digest. Every other descriptor in this package
does: `jer_plan` and `encode_plan` both serialize by hand so that byte identity is by
construction, because a descriptor that names a landed artifact has to be nameable itself.
Until now an ECN encoding could not be hashed, compared or shipped — two specifications could
only be diffed as Python source. `EcnModule.serialize()` and `.sha256()` are the point of this
module at least as much as the grammar is.

WHAT THE GRAMMAR IS. Clause 20.4 says the defined syntax "shall be the syntax specified by the
WITH SYNTAX statements in clauses 23 to 25", and those statements are a keyword grammar with
`[...]` marking optional groups::

    ENCODING-SPACE
      [ SIZE &encoding-space-size
        [MULTIPLE OF &encoding-space-unit]]
      [ DETERMINED BY &encoding-space-determination]

So `ENCODING-SPACE SIZE 4 MULTIPLE OF bit` sets two properties, and the nesting is real: a
`MULTIPLE OF` without a preceding `SIZE` is not a thing the syntax admits. §20.5 notes that
the `WITH SYNTAX` statements "impose constraints on the values of some encoding properties, in
conjunction with the values of other encoding properties, to enforce some (but not all)
semantic constraints" — so the bracket structure is load-bearing and the parser follows it
rather than accepting keywords in any order.

WHAT IS SUPPORTED, and everything else is refused by name. The module grammar is X.692's own:
an `ENCODING-DEFINITIONS` module holding class assignments, one concatenation structure, and
encoding object assignments for `#TRANSFORM`, `#CONDITIONAL-INT`, `#INT`, `#BOOL`, `#PAD`,
`#CONCATENATION` and `#OUTER`. Every other class, and every property group this repository has
not built — `REPLACE`, `START-POINTER`, `IF`/`IF-ALL` conditions, `DETERMINED BY` anything but
the default, `USING`, `UNUSED BITS`, `EXHIBITS HANDLE`, `BIT-REVERSAL` — parses far enough to
be *recognized* and is then refused with the clause that defines it. A parser that skipped an
unimplemented keyword would produce octets that silently disagree with the specification it
was handed, which is the failure mode the whole triple-rail design exists to prevent.

THE ONE-OBJECT-PER-CLASS LAW IS WHY CLASSES GET DEFINED. §9.5.2 permits an encoding object set
at most one object per encoding class, and `encode_with_user` relies on it. A frame header with
two integer fields of different widths therefore cannot use `#INT` twice — it defines
`#Scaled-length ::= #INT` and `#Version ::= #INT` and gives each its own object. That is not a
workaround; it is what the notation is for, and clause 11 spells the assignment (`#My-int ::=
#INT`) as an ordinary thing to write.

THE COMPONENT ORDER LIVES IN THE STRUCTURE, which reading the text corrected. `ecn_user`'s
`ConcatenationSpec` takes a free tuple of field names as its transmission order, and §22.10.1.1
has no such property: `&concatenation-order` is `ENUMERATED {textual, tag, random}` and nothing
else. §22.10.3.1 then defines `textual` as "the textual order in the ASN.1 type specification
**or the ECN structure definition**" — so a wire order that differs from the abstract type's is
reached by declaring a §16.5 `ConcatenationStructure` whose fields are in that order, and the
concatenation object stays `textual`. The free tuple is a faithful *denotation* of that; this
module is where it gets its stated source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .ecn_param import (
    ActualKind, ActualParameter, ActualParameterList, AssignmentKind, GovernorKind, Parameter,
    ParameterizedAssignment, ParameterKind, ParameterList, ReplacementParameterization,
)
from .ecn_user import (
    UNIT_BIT, UNIT_NAMES, AuxIntSpec, BoolSpec, Comparison, ConcatenationSpec,
    ConditionalIntSpec, EncodingSpaceDetermination, HeadEndStructure, IntForm, IntOp,
    RESULT_SIZE_FIXED_TO_MAX, RESULT_SIZE_VARIABLE, BitsToBits, BitsToChar,
    BitsToCompositeBits, BitsToInt, BitToBits, BoolToBool, BoolToInt, CharsToCompositeChar,
    CharToBits, CompositeBitsToBits, CompositeBitsToOctets, CompositeCharToChars,
    HandleValueSet, IdentificationHandle,
    IntSelector, IntSpec, IntToBits, IntToBool, IntToChars, IntToInt, IntegerBounds,
    Justification, OctetsToCompositeBits, OuterSpec,
    PadSpec, Padding, Pattern, PreAlignment, RangeCondition, ReplaceAction, Replacement,
    ReplacementStructure, ReversalSpecification, SpaceDeterminant, StartPointer,
    TransformChain, UnusedBits, UnusedBitsDetermination, UserEncodingObject, ValuePadding,
    check_unit,
)
from .tags import Asn1Error

#: The serialization's version. Its own counter, not `encode_plan`'s: see this module's
#: `serialize` for why an ECN encoding is a separate compilation rather than a plan version.
SYNTAX_VERSION = 5
SYNTAX_COMPILER = "bcir-ecn-syntax/1"

#: The bit-field and constructor classes whose defined syntax clause 23 gives, restricted to
#: the ones `ecn_user` can execute. A class outside this set is refused by name rather than
#: parsed permissively, so "ECN accepted my specification" and "ECN will encode my
#: specification" stay the same statement.
_BUILTIN_CLASSES = {
    "#TRANSFORM": "transform",
    "#CONDITIONAL-INT": "conditional-int",
    "#INT": "int",
    "#BOOL": "bool",
    "#PAD": "pad",
    "#CONCATENATION": "concatenation",
    "#OUTER": "outer",
}

#: Property groups that clause 23 gives every bit-field class and this repository has not
#: built. Recognized so the refusal can cite; never silently dropped.
_UNSUPPORTED_KEYWORDS = {
    "CONTAINED": "§22.11's contained-type specification is not implemented",
    # §23.1's `#ALTERNATIVES` and §23.11's `#OPTIONAL` are built in `ecn_user` and reachable
    # from Python; what this grammar cannot read is the STRUCTURE side of them. §16.2.12 names
    # the three: `AlternativesStructure` is §16.3, `RepetitionStructure` is §16.4 and
    # `ConcatenationStructure` — the one this parser models — is §16.5. So alternatives are a
    # second constructor shape beside the concatenation, and an optional component is §16.5's
    # own `ConcatComponentPresence` tail on a `ConcatComponent`. Both are structure notation
    # rather than object notation, which is why the objects arrived first.
    #
    # (These two citations were each other's until Annex C was read for slice F: §16.3 and
    # §16.5 were swapped, and the optional marker was called `OPTIONAL` when §16.5.1 spells it
    # `OPTIONAL-ENCODING` followed by an `OptionalClass`.)
    "ALTERNATIVE": "§23.1's alternatives objects are built, but §16.3's AlternativesStructure "
                   "is a constructor shape this grammar does not read; assemble "
                   "`AlternativesSpec` in Python",
    "PRESENCE": "§23.11's optionality objects are built, but marking a structure component "
                "`OPTIONAL-ENCODING` is §16.5's ConcatComponentPresence, structure notation "
                "this grammar does not read; assemble `OptionalSpec` in Python",
    # §22.1's replacement SEMANTICS are built in `ecn_user`, and its PARAMETERIZATION model —
    # what a dummy may stand for, which actual fits, and §22.1.2's rules about the definitions
    # a REPLACE names — is built in `ecn_param`. Both are reachable from Python. What this
    # grammar cannot read is the notation: §22.1.2.2 makes the `WITH` structures parameterized
    # encoding structures with a single encoding class parameter, and §22.1.2.4 makes the
    # `ENCODED BY` objects parameterized encoding objects whose governor is that structure
    # instantiated with the dummy.
    #
    # The shape is `#Length-prefixed{<#D>} ::= ...`, and the delimiters are the point: Annex
    # C.1 rewrites X.683 §8.3's `ParameterList` to use `{<` and `>}`, so an X.683 parser reads
    # the one spelling ECN does not have and refuses the only one it does. (This comment said
    # `{#D}` — ASN.1's braces — until Annex C was read for slice F.)
    "REPLACE": "§22.1.1.2/§22.1.1.4/§22.1.1.6's defined syntax is not read here. The "
               "parameterized structures and objects it names now PARSE — see `{<`/`>}` "
               "above — and §22.1.2's restrictions on them are checked as they are declared; "
               "what is missing is the clause that binds an auxiliary field's encoding and "
               "its determinant to the instantiated one (§22.1.2.6, §22.1.1.9). Assemble "
               "`Replacement` in Python until that lands",
}


# --- lexing -------------------------------------------------------------------------------

#: Characters that are a token on their own. Everything else accumulates into a word, so
#: `payloadOctets  #INT,` lexes as three tokens without the source needing spaces around the
#: comma. `:` is NOT here: §21.8's `left:2` and §24.3.1's `divide:4` are single tokens, since
#: the CHOICE alternative and its value are one value notation.
_PUNCTUATION = "{},"

#: Annex C.1 and C.4's parameter-list brackets, which are **two characters**:
#: `ParameterList ::= "{<" Parameter "," + ">}"`, against X.683 §8.3's `"{" ... "}"`.
#:
#: They are lexed as single tokens ahead of `_PUNCTUATION`, and the order matters in both
#: directions. `{<` has to win over `{` or a parameter list would open an object body; `>` is
#: not punctuation at all, so `>}` has to win over word accumulation or it would glue itself
#: to the dummy before it. Nothing else in the ECN surface uses either character — §21.12's
#: comparisons are spelled `greater-than`, not `>` — so there is no third reading to lose.
_PARAM_BRACKETS = ("{<", ">}")


@dataclass(frozen=True)
class Token:
    text: str
    line: int

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.text!r} on line {self.line}"


def tokenize(source: str) -> list[Token]:
    """Split ECN text into tokens, dropping X.680 §12.6 comments.

    §7.2 of X.692 makes ECN's lexical items those of X.680 clause 12 plus its own additions,
    so a comment runs from `--` "to the next `--` or to the end of the line, whichever comes
    first". Both terminations are honoured: a `-- note --` inside a line is not a truncation
    of the rest of it.

    Quoted forms — `'1010'B`, `'FF'H` and `"text"` — are held together, because a bit string
    containing a brace would otherwise lex as punctuation.
    """
    tokens: list[Token] = []
    line = 1
    index = 0
    length = len(source)
    word = ""
    word_line = 1

    def flush() -> None:
        nonlocal word
        if word:
            tokens.append(Token(word, word_line))
            word = ""

    while index < length:
        character = source[index]
        if character == "\n":
            flush()
            line += 1
            index += 1
            continue
        if character.isspace():
            flush()
            index += 1
            continue
        if source.startswith("--", index):
            flush()
            index += 2
            while index < length and source[index] != "\n":
                if source.startswith("--", index):
                    index += 2
                    break
                index += 1
            continue
        if character in "'\"":
            close = source.find(character, index + 1)
            if close < 0:
                raise Asn1Error(f"ECN: unterminated {character} literal on line {line}")
            # A BIT STRING is `'0101'B` and a hex string `'FF'H`; the radix letter belongs to
            # the token, so take one more character when it is there.
            end = close + 1
            if end < length and source[end] in "BH":
                end += 1
            # A quoted form usually FOLLOWS an alternative name — `bits:'0'B` is one value
            # notation, not two tokens — so it joins the word in progress rather than
            # flushing it. On its own it starts a word, which the next flush emits.
            if not word:
                word_line = line
            word += source[index:end]
            index = end
            continue
        bracket = next((b for b in _PARAM_BRACKETS if source.startswith(b, index)), None)
        if bracket is not None:
            flush()
            tokens.append(Token(bracket, line))
            index += len(bracket)
            continue
        if character in _PUNCTUATION:
            flush()
            tokens.append(Token(character, line))
            index += 1
            continue
        if not word:
            word_line = line
        word += character
        index += 1
    flush()
    return tokens


class _Cursor:
    """A token stream with the lookahead a bracket grammar needs.

    Clause 23's syntax is `LL(k)` only in the loose sense that every optional group starts
    with a distinct keyword, so a single-token peek decides every branch except the two-word
    ones (`ALIGNED TO`, `MULTIPLE OF`, `DETERMINED BY`), which is what `accept_words` is for.
    """

    def __init__(self, tokens: list[Token], where: str) -> None:
        self._tokens = tokens
        self._at = 0
        self._where = where

    def eof(self) -> bool:
        return self._at >= len(self._tokens)

    def peek(self) -> str | None:
        return None if self.eof() else self._tokens[self._at].text

    def next(self) -> Token:
        if self.eof():
            raise Asn1Error(f"ECN: {self._where} ends early; more tokens were expected")
        token = self._tokens[self._at]
        self._at += 1
        return token

    def accept(self, text: str) -> bool:
        if self.peek() == text:
            self._at += 1
            return True
        return False

    def accept_words(self, *texts: str) -> bool:
        """Accept a multi-word keyword atomically, or leave the cursor untouched."""
        if self._tokens[self._at:self._at + len(texts)] and all(
                self._tokens[self._at + offset].text == text
                for offset, text in enumerate(texts)):
            self._at += len(texts)
            return True
        return False

    def expect(self, text: str) -> Token:
        token = self.next()
        if token.text != text:
            raise Asn1Error(f"ECN: {self._where} expected {text!r} but found {token}")
        return token

    def expect_words(self, *texts: str) -> None:
        for text in texts:
            self.expect(text)


# --- clause 21's value notations -----------------------------------------------------------

def _parse_unit(cursor: _Cursor, *, allow_repetitions: bool = False) -> int:
    """§21.1.1's `Unit`: one of the six named values, or any integer the constraint admits."""
    token = cursor.next()
    if token.text in UNIT_NAMES:
        return check_unit(UNIT_NAMES[token.text], allow_repetitions=allow_repetitions)
    try:
        bits = int(token.text)
    except ValueError:
        raise Asn1Error(
            f"ECN: {token} is not a Unit; §21.1.1 names "
            f"{', '.join(sorted(UNIT_NAMES))} and admits any integer in (0..256)") from None
    return check_unit(bits, allow_repetitions=allow_repetitions)


_PADDING_NAMES = {value.value: value for value in Padding}
_SPACE_DETERMINATIONS = {value.value: value for value in EncodingSpaceDetermination}
_REVERSALS = {value.value: value for value in ReversalSpecification}
_RANGE_CONDITIONS = {value.value: value for value in RangeCondition}
_COMPARISONS = {value.value: value for value in Comparison}


def _parse_padding(cursor: _Cursor) -> Padding:
    """§21.9.1's `Padding ::= ENUMERATED {zero, one, pattern, encoder-option}`."""
    token = cursor.next()
    try:
        return _PADDING_NAMES[token.text]
    except KeyError:
        raise Asn1Error(
            f"ECN: {token} is not a Padding value; §21.9.1 gives "
            f"{', '.join(sorted(_PADDING_NAMES))}") from None


def _parse_pattern(cursor: _Cursor) -> Pattern:
    """§21.10.1's `Pattern` CHOICE, in ASN.1 value notation: `alternative:value`.

    The five concrete alternatives are built here. `any-of-length` and `different:any` parse —
    they are legal values and refusing to *read* them would misreport the specification — and
    fail later, at the point where a bit sequence is actually needed, which is where §21.10.8
    and §21.10.9 put the encoder's freedom.
    """
    token = cursor.next()
    text = token.text
    if ":" not in text:
        raise Asn1Error(
            f"ECN: {token} is not a Pattern; §21.10.1 is a CHOICE, so a value is written "
            f"`alternative:value` — for example bits:'0'B")
    alternative, _, body = text.partition(":")
    if alternative == "bits":
        if not (body.startswith("'") and body.endswith("'B")):
            raise Asn1Error(f"ECN: {token} — a bits: pattern is a BIT STRING like '1010'B")
        return Pattern.from_bits(body[1:-2])
    if alternative == "octets":
        if not (body.startswith("'") and body.endswith("'H")):
            raise Asn1Error(f"ECN: {token} — an octets: pattern is a hex string like 'FF'H")
        digits = body[1:-2]
        if len(digits) % 2:
            raise Asn1Error(
                f"ECN: {token} — an OCTET STRING takes an even number of hex digits")
        try:
            return Pattern.from_octets(bytes.fromhex(digits))
        except ValueError:
            raise Asn1Error(f"ECN: {token} is not a hex string") from None
    if alternative in ("char8", "char16", "char32"):
        if not (body.startswith('"') and body.endswith('"')):
            raise Asn1Error(f"ECN: {token} — a {alternative}: pattern is a quoted string")
        return Pattern.from_chars(body[1:-1], {"char8": 8, "char16": 16, "char32": 32}[
            alternative])
    if alternative == "any-of-length":
        try:
            return Pattern.any_of_length(int(body))
        except ValueError:
            raise Asn1Error(f"ECN: {token} — any-of-length takes INTEGER (1..MAX)") from None
    if alternative == "different" and body == "any":
        return Pattern.different_any()
    raise Asn1Error(
        f"ECN: {token} names no alternative of §21.10.1's Pattern CHOICE")


def _parse_justification(cursor: _Cursor) -> Justification:
    """§21.8.1's `Justification ::= CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}`."""
    token = cursor.next()
    alternative, _, body = token.text.partition(":")
    if alternative not in ("left", "right"):
        raise Asn1Error(
            f"ECN: {token} is not a Justification; §21.8.1's alternatives are left and right, "
            f"each taking an offset — `right:0` is §21.8.2's default")
    try:
        offset = int(body)
    except ValueError:
        raise Asn1Error(
            f"ECN: {token} — §21.8.1 gives both alternatives INTEGER (0..MAX), so the offset "
            f"is written explicitly: `{alternative}:0` for none") from None
    return (Justification.left if alternative == "left" else Justification.right)(offset)


# --- clause 22's property groups, as the bracket structure the WITH SYNTAX gives them -------

def _parse_pre_alignment(cursor: _Cursor) -> PreAlignment:
    """§22.2.1.2's group, entered on `ALIGNED TO` having been accepted.

        [ALIGNED TO [NEXT] [ANY] &unit [PADDING &padding [PATTERN &pattern]]]
    """
    # §22.2.2.1: at most one of NEXT and ANY, and NEXT is assumed when neither is written.
    # §22.2.2.2 then ties ANY to the start-pointer group, which the caller checks once it has
    # parsed both — the requirement relates two groups, so neither alone can enforce it.
    if cursor.accept("NEXT") and cursor.accept("ANY"):
        raise Asn1Error("ECN: §22.2.2.1 — at most one of NEXT and ANY shall be specified")
    any_offset = cursor.accept("ANY")
    unit = _parse_unit(cursor)
    padding = Padding.ZERO
    pattern: Pattern | None = None
    if cursor.accept("PADDING"):
        padding = _parse_padding(cursor)
        if cursor.accept("PATTERN"):
            pattern = _parse_pattern(cursor)
    return PreAlignment(unit=unit, padding=padding, pattern=pattern,
                        encoder_chosen_offset=any_offset)


def _parse_value_padding(cursor: _Cursor, module: "EcnModule") -> ValuePadding:
    """§22.8.1.2's group, entered on `VALUE-PADDING` having been accepted.

        [VALUE-PADDING [JUSTIFIED &j] [PRE-PADDING &p [PATTERN &pp]]
                       [POST-PADDING &q [PATTERN &qp]] [UNUSED BITS ...]]
    """
    justification = Justification()
    pre_padding = post_padding = Padding.ZERO
    pre_pattern: Pattern | None = None
    post_pattern: Pattern | None = None
    if cursor.accept("JUSTIFIED"):
        justification = _parse_justification(cursor)
    if cursor.accept("PRE-PADDING"):
        pre_padding = _parse_padding(cursor)
        if cursor.accept("PATTERN"):
            pre_pattern = _parse_pattern(cursor)
    if cursor.accept("POST-PADDING"):
        post_padding = _parse_padding(cursor)
        if cursor.accept("PATTERN"):
            post_pattern = _parse_pattern(cursor)
    unused: UnusedBits | None = None
    if cursor.accept_words("UNUSED", "BITS"):
        unused = _parse_unused_bits(cursor, module)
    return ValuePadding(justification=justification, pre_padding=pre_padding,
                        pre_pattern=pre_pattern, post_padding=post_padding,
                        post_pattern=post_pattern, unused_bits=unused)


_UNUSED_DETERMINATIONS = {value.value: value for value in UnusedBitsDetermination}


def _parse_unused_bits(cursor: _Cursor, module: "EcnModule") -> UnusedBits:
    """§22.8.1.2's `UNUSED BITS` sub-group, entered on the two keywords having been accepted.

        [UNUSED BITS [DETERMINED BY &d]
                     [USING &ref [ENCODER-TRANSFORMS &e] [DECODER-TRANSFORMS &d]]]
    """
    determination = UnusedBitsDetermination.FIELD_TO_BE_SET  # §22.8.1.1's DEFAULT.
    if cursor.accept_words("DETERMINED", "BY"):
        token = cursor.next()
        try:
            determination = _UNUSED_DETERMINATIONS[token.text]
        except KeyError:
            raise Asn1Error(
                f"ECN: {token} names no value of §21.4's UnusedBitsDetermination "
                f"({', '.join(sorted(_UNUSED_DETERMINATIONS))})") from None
    reference = ""
    encoder = decoder = None
    if cursor.accept("USING"):
        reference = cursor.next().text
        if cursor.accept("ENCODER-TRANSFORMS"):
            encoder = TransformChain(module.transform_list(_parse_reference_list(cursor)))
        if cursor.accept("DECODER-TRANSFORMS"):
            decoder = TransformChain(module.transform_list(_parse_reference_list(cursor)))
    # UnusedBits itself enforces §22.8.2.2/§22.8.2.3/§22.8.2.5, which is where those rules
    # belong: they constrain the combination and hold however the object was built.
    return UnusedBits(determination=determination, reference=reference,
                      encoder_transforms=encoder, decoder_transforms=decoder)


@dataclass(frozen=True)
class _EncodingSpace:
    """§22.4's group, reduced to what a fixed-size field needs.

        ENCODING-SPACE [SIZE &size [MULTIPLE OF &unit]] [DETERMINED BY ...] [USING ...]

    `size` is in `unit`s, so the width in bits is their product — §21.2.4: "a fixed size for
    the encoding space, as the value of type Unit multiplied by the value of type
    EncodingSpaceSize, in bits".
    """

    size: int | None = None
    unit: int = 1
    determinant: SpaceDeterminant | None = None

    @property
    def width(self) -> int:
        if self.size is None:
            raise Asn1Error(
                "ECN: §21.2.2 defaults the encoding space to `self-delimiting-values`, which "
                "§21.2.7 defines by matching candidate encodings rather than by a width; this "
                "rail writes fixed spaces, so ENCODING-SPACE SIZE has to be stated")
        return self.size * self.unit


def _parse_encoding_space(cursor: _Cursor, module: "EcnModule") -> _EncodingSpace:
    """Entered on `ENCODING-SPACE` having been accepted. §22.4.1.2's bracket nesting."""
    size: int | None = None
    unit = 1
    if cursor.accept("SIZE"):
        token = cursor.next()
        named = ("encoder-option-with-determinant", "variable-with-determinant",
                 "self-delimiting-values", "fixed-to-max")
        if token.text in named:
            raise Asn1Error(
                f"ECN: §21.2 gives `{token.text}` a size this rail cannot write — the "
                f"negative values need a determinant field (§21.2.5/§21.2.6) and "
                f"`fixed-to-max` needs the widest encoding of the whole value set "
                f"(§21.2.8). State a positive SIZE")
        try:
            size = int(token.text)
        except ValueError:
            raise Asn1Error(
                f"ECN: {token} is not an EncodingSpaceSize; §21.2.1 gives INTEGER "
                f"(-3..MAX)") from None
        if size < 0:
            raise Asn1Error("ECN: §21.2.1 constrains EncodingSpaceSize to (-3..MAX)")
        if cursor.accept_words("MULTIPLE", "OF"):
            unit = _parse_unit(cursor)
    determination = None
    if cursor.accept_words("DETERMINED", "BY"):
        token = cursor.next()
        try:
            determination = _SPACE_DETERMINATIONS[token.text]
        except KeyError:
            raise Asn1Error(
                f"ECN: {token} names no value of §21.3.1's EncodingSpaceDetermination "
                f"({', '.join(sorted(_SPACE_DETERMINATIONS))})") from None
    determinant = None
    if cursor.accept("USING"):
        reference = cursor.next().text
        encoder = decoder = None
        if cursor.accept("ENCODER-TRANSFORMS"):
            encoder = TransformChain(module.transform_list(_parse_reference_list(cursor)))
        if cursor.accept("DECODER-TRANSFORMS"):
            decoder = TransformChain(module.transform_list(_parse_reference_list(cursor)))
        determinant = SpaceDeterminant(
            determination=determination or EncodingSpaceDetermination.FIELD_TO_BE_SET,
            reference=reference, unit=unit, encoder_transforms=encoder,
            decoder_transforms=decoder)
    elif determination is not None:
        # §21.3.4 and §21.3.5 both "require the specification of a REFERENCE"; §21.3.6's
        # `container` requires one too, or #OUTER. So a determination with no USING names no
        # field, and there is nothing for the encoder to set or the decoder to read.
        raise Asn1Error(
            f"ECN: §21.3.4/§21.3.5 — `DETERMINED BY {determination.value}` requires a USING "
            f"reference to the field carrying the length")
    return _EncodingSpace(size=size, unit=unit, determinant=determinant)


# --- clause 24's #TRANSFORM ----------------------------------------------------------------

_INT_OP_NAMES = {op.value: op for op in IntOp}


_INT_FORMS = {"positive-int": IntForm.POSITIVE_INT,
              "twos-complement": IntForm.TWOS_COMPLEMENT}


def _parse_int_form(cursor: _Cursor) -> IntForm:
    """X.690 §8.3.2/§8.3.3's two integer encodings, which §24.8.9 and §24.9.5 both name."""
    token = cursor.next()
    if token.text in ("reverse-positive-int", "reverse-twos-complement"):
        raise Asn1Error(
            f"ECN: §23.7.1 admits `{token.text}`, whose bits run from the least significant "
            f"end; this rail writes the forward forms only")
    try:
        return _INT_FORMS[token.text]
    except KeyError:
        raise Asn1Error(
            f"ECN: {token} is neither positive-int nor twos-complement") from None


def _parse_result_size(cursor: _Cursor) -> int:
    """§21.15.1's `ResultSize ::= INTEGER {variable(-1), fixed-to-max(0)} (-1..MAX)`."""
    token = cursor.next()
    if token.text == "variable":
        return RESULT_SIZE_VARIABLE
    if token.text == "fixed-to-max":
        return RESULT_SIZE_FIXED_TO_MAX
    try:
        size = int(token.text)
    except ValueError:
        raise Asn1Error(
            f"ECN: {token} is not a ResultSize; §21.15.1 names `variable` and `fixed-to-max` "
            f"and admits any positive count") from None
    if size < RESULT_SIZE_VARIABLE:
        raise Asn1Error(f"ECN: §21.15.1 constrains ResultSize to (-1..MAX); got {size}")
    return size


def _parse_boolean(cursor: _Cursor) -> bool:
    token = cursor.next()
    if token.text not in ("TRUE", "FALSE"):
        raise Asn1Error(f"ECN: a BOOLEAN is TRUE or FALSE; got {token}")
    return token.text == "TRUE"


def _parse_int_list(cursor: _Cursor) -> tuple[int, ...]:
    out = []
    for text in _parse_reference_list(cursor):
        try:
            out.append(int(text))
        except ValueError:
            raise Asn1Error(f"ECN: {text!r} is not an INTEGER") from None
    return tuple(out)


def _parse_char_list(cursor: _Cursor) -> tuple[str, ...]:
    """§24.10.10.1's `CHAR-LIST`: an ordered list of `UniversalString (SIZE(1))` values."""
    out = []
    for text in _parse_reference_list(cursor):
        if not (text.startswith('"') and text.endswith('"') and len(text) == 3):
            raise Asn1Error(
                f"ECN: §24.10.1 declares CHAR-LIST as UniversalString (SIZE(1)), so each "
                f"entry is one quoted character; got {text!r}")
        out.append(text[1])
    return tuple(out)


def _parse_bits_list(cursor: _Cursor) -> tuple[tuple[int, ...], ...]:
    """An ordered list of BIT STRING values, each written `'0101'B`."""
    out = []
    for text in _parse_reference_list(cursor):
        if not (text.startswith("'") and text.endswith("'B")):
            raise Asn1Error(
                f"ECN: each entry of a BITS-LIST is a BIT STRING like '1010'B; got {text!r}")
        out.append(Pattern.from_bits(text[1:-2]).bit_sequence())
    return tuple(out)


def _parse_int_to_bits(cursor: _Cursor) -> dict:
    """§24.8.2's `[INT-TO-BITS [AS &as] [SIZE &size] [MULTIPLE OF &unit]]`."""
    encoded_as = IntForm.TWOS_COMPLEMENT        # §24.8.1's DEFAULT
    size, unit = RESULT_SIZE_VARIABLE, UNIT_BIT
    if cursor.accept("AS"):
        encoded_as = _parse_int_form(cursor)
    if cursor.accept("SIZE"):
        size = _parse_result_size(cursor)
    if cursor.accept_words("MULTIPLE", "OF"):
        unit = _parse_unit(cursor)
    return {"encoded_as": encoded_as, "size": size, "unit": unit}


def _parse_transform_body(cursor: _Cursor, name: str):
    """§24.1.1's `WITH SYNTAX`, whose comment says "Only one of the following clauses can be
    used" — so this reads exactly one and refuses a second."""
    if cursor.accept("INT-TO-INT"):
        token = cursor.next()
        alternative, _, body = token.text.partition(":")
        if alternative not in _INT_OP_NAMES:
            raise Asn1Error(
                f"ECN: {token} names no alternative of §24.3.1's &int-to-int CHOICE "
                f"({', '.join(sorted(_INT_OP_NAMES))})")
        op = _INT_OP_NAMES[alternative]
        if op is IntOp.NEGATE:
            if body != "value":
                raise Asn1Error(
                    f"ECN: §24.3.1 spells negate as `negate:value` (ENUMERATED{{value}}); "
                    f"got {token}")
            operand = 0
        elif op is IntOp.SUBTRACT_LOWER_BOUND:
            if body != "lower-bound":
                raise Asn1Error(
                    f"ECN: §24.3.1 spells subtract as `subtract:lower-bound` "
                    f"(ENUMERATED{{lower-bound}}); got {token}")
            # §24.3.9's operand is the source class's lower bound, which is a property of the
            # ASN.1 type rather than of the object. Recorded as zero and supplied by whoever
            # applies the object; stating it in the notation would be inventing syntax.
            operand = 0
        else:
            try:
                operand = int(body)
            except ValueError:
                raise Asn1Error(
                    f"ECN: §24.3.1 gives {alternative} an INTEGER operand; got {token}"
                ) from None
        transform = IntToInt(name=name, op=op, operand=operand)
    elif cursor.accept("INT-TO-BITS"):
        transform = IntToBits(name=name, **_parse_int_to_bits(cursor))
    elif cursor.accept("BOOL-TO-BOOL"):
        # §24.4.5: "There is only one value ... AS logical:not, which may be omitted."
        if cursor.accept("AS"):
            token = cursor.next()
            if token.text != "logical:not":
                raise Asn1Error(
                    f"ECN: §24.4.1's &bool-to-bool CHOICE has the single alternative "
                    f"`logical:not`; got {token}")
        transform = BoolToBool(name=name)
    elif cursor.accept_words("BOOL-TO-INT", "AS"):
        token = cursor.next()
        if token.text not in ("true-zero", "true-one"):
            raise Asn1Error(
                f"ECN: §24.5.1's &bool-to-int is ENUMERATED {{true-zero, true-one}}; "
                f"got {token}")
        transform = BoolToInt(name=name, true_zero=token.text == "true-zero")
    elif cursor.accept("INT-TO-BOOL"):
        zero_true = False
        true_is = false_is = None
        if cursor.accept("AS"):
            token = cursor.next()
            if token.text not in ("zero-true", "zero-false"):
                raise Asn1Error(
                    f"ECN: §24.6.1's &int-to-bool is ENUMERATED {{zero-true, zero-false}}; "
                    f"got {token}")
            zero_true = token.text == "zero-true"
        if cursor.accept("TRUE-IS"):
            true_is = _parse_int_list(cursor)
        if cursor.accept("FALSE-IS"):
            false_is = _parse_int_list(cursor)
        # §24.6.4: "Either one of AS, TRUE-IS and FALSE-IS is set, or both TRUE-IS and
        # FALSE-IS are set (and AS is not set), or none are set."
        if zero_true and (true_is is not None or false_is is not None):
            raise Asn1Error(
                "ECN: §24.6.4 — AS shall not be set alongside TRUE-IS or FALSE-IS")
        transform = IntToBool(name=name, zero_true=zero_true, true_is=true_is,
                              false_is=false_is)
    elif cursor.accept("INT-TO-CHARS"):
        size = RESULT_SIZE_VARIABLE
        plus_sign = False
        pad_with_spaces = False
        if cursor.accept("SIZE"):
            size = _parse_result_size(cursor)
        if cursor.accept("PLUS-SIGN"):
            plus_sign = _parse_boolean(cursor)
        if cursor.accept("PADDING"):
            token = cursor.next()
            if token.text not in ("spaces", "zeros"):
                raise Asn1Error(
                    f"ECN: §24.7.1's &int-to-chars-pad is ENUMERATED {{spaces, zeros}}; "
                    f"got {token}")
            pad_with_spaces = token.text == "spaces"
        transform = IntToChars(name=name, size=size, plus_sign=plus_sign,
                               pad_with_spaces=pad_with_spaces)
    elif cursor.accept("BITS-TO-INT"):
        decoded = IntForm.TWOS_COMPLEMENT
        if cursor.accept("AS"):
            decoded = _parse_int_form(cursor)
        transform = BitsToInt(name=name, decoded_assuming=decoded)
    elif cursor.accept("CHAR-TO-BITS"):
        encoded_as = "compact"                     # §24.10.1's DEFAULT
        chars = ()
        bit_values = ()
        size, unit = RESULT_SIZE_VARIABLE, UNIT_BIT
        if cursor.accept("AS"):
            encoded_as = cursor.next().text
        if cursor.accept("CHAR-LIST"):
            chars = _parse_char_list(cursor)
        if cursor.accept("BITS-LIST"):
            bit_values = _parse_bits_list(cursor)
        if cursor.accept("SIZE"):
            size = _parse_result_size(cursor)
        if cursor.accept_words("MULTIPLE", "OF"):
            unit = _parse_unit(cursor)
        transform = CharToBits(name=name, encoded_as=encoded_as, chars=chars,
                               bit_values=bit_values, size=size, unit=unit)
    elif cursor.accept("BITS-TO-CHAR"):
        decoded_assuming = "iso10646"              # §24.11.1's DEFAULT
        chars = ()
        bit_values = ()
        if cursor.accept("AS"):
            decoded_assuming = cursor.next().text
        if cursor.accept("BITS-LIST"):
            bit_values = _parse_bits_list(cursor)
        if cursor.accept("CHAR-LIST"):
            chars = _parse_char_list(cursor)
        transform = BitsToChar(name=name, decoded_assuming=decoded_assuming, chars=chars,
                               bit_values=bit_values)
    elif cursor.accept("BIT-TO-BITS"):
        zero = Pattern.from_bits("0")              # §24.12.1's DEFAULTs
        one = Pattern.from_bits("1")
        if cursor.accept("ZERO-PATTERN"):
            zero = _parse_pattern(cursor)
        if cursor.accept("ONE-PATTERN"):
            one = _parse_pattern(cursor)
        transform = BitToBits(name=name, zero_pattern=zero, one_pattern=one)
    elif cursor.accept("BITS-TO-BITS"):
        cursor.expect("SOURCE-LIST")
        sources = _parse_bits_list(cursor)
        cursor.expect("RESULT-LIST")
        results = _parse_bits_list(cursor)
        transform = BitsToBits(name=name, source_values=sources, result_values=results)
    elif cursor.accept("CHARS-TO-COMPOSITE-CHAR"):
        transform = CharsToCompositeChar(name=name)
    elif cursor.accept("BITS-TO-COMPOSITE-BITS"):
        unit = UNIT_BIT                            # §24.15.2's DEFAULT
        if cursor.accept("UNIT"):
            unit = _parse_unit(cursor)
        transform = BitsToCompositeBits(name=name, unit=unit)
    elif cursor.accept("OCTETS-TO-COMPOSITE-BITS"):
        transform = OctetsToCompositeBits(name=name)
    elif cursor.accept("COMPOSITE-CHAR-TO-CHARS"):
        transform = CompositeCharToChars(name=name)
    elif cursor.accept("COMPOSITE-BITS-TO-BITS"):
        transform = CompositeBitsToBits(name=name)
    elif cursor.accept("COMPOSITE-BITS-TO-OCTETS"):
        transform = CompositeBitsToOctets(name=name)
    else:
        token = cursor.peek()
        raise Asn1Error(
            f"ECN: {token!r} starts no #TRANSFORM clause; §24.1.1's WITH SYNTAX defines "
            f"nineteen and all of them are read here")
    if not cursor.eof():
        raise Asn1Error(
            f"ECN: §24.1.1's WITH SYNTAX says only one transform clause can be used, and "
            f"{cursor.peek()!r} follows one that was already read")
    return transform


# --- clause 23's bit-field classes ---------------------------------------------------------

def _parse_start_pointer(cursor: _Cursor, module: "EcnModule") -> StartPointer:
    """§22.3.1.2's group, entered on `START-POINTER` having been accepted.

        [START-POINTER &ref [MULTIPLE OF &unit] [ENCODER-TRANSFORMS &transforms]]
    """
    reference = cursor.next().text
    unit = UNIT_BIT  # §22.3.1.1's DEFAULT.
    if cursor.accept_words("MULTIPLE", "OF"):
        unit = _parse_unit(cursor)
    transforms = None
    if cursor.accept("ENCODER-TRANSFORMS"):
        transforms = TransformChain(module.transform_list(_parse_reference_list(cursor)))
    return StartPointer(reference=reference, unit=unit, encoder_transforms=transforms)


def _parse_bit_reversal(cursor: _Cursor) -> ReversalSpecification:
    """§22.12.1.2's `[BIT-REVERSAL &bit-reversal]`."""
    token = cursor.next()
    try:
        return _REVERSALS[token.text]
    except KeyError:
        raise Asn1Error(
            f"ECN: {token} names no value of §21.14.1's ReversalSpecification "
            f"({', '.join(sorted(_REVERSALS))})") from None


def _parse_conditions(cursor: _Cursor, all_of: bool):
    """§23.7.1's `[IF &c [&comparison &comparator]]` and `[IF-ALL {..} [{..} {..}]]`.

    §23.7.2.2 gives `IF-ALL` either one list or three: "shall be used with three lists if one
    or more of the size-range-conditions require a comparison, and shall be used with one list
    otherwise", and the three "shall be interpreted as a list of predicates using the values
    in corresponding positions in the three lists". So the three-list form is transposed here
    into one list of triples, which is what the clause says it means.
    """
    if not all_of:
        condition = _named(cursor.next(), _RANGE_CONDITIONS, "§21.11.1's RangeCondition")
        if condition.needs_comparison():
            comparison = _named(cursor.next(), _COMPARISONS, "§21.12.1's Comparison")
            token = cursor.next()
            try:
                return ((condition, comparison, int(token.text)),)
            except ValueError:
                raise Asn1Error(
                    f"ECN: §21.11.5 gives {condition.value} an integer comparator; "
                    f"got {token}") from None
        return ((condition, None, None),)
    conditions = [_named(Token(name, 0), _RANGE_CONDITIONS, "§21.11.1's RangeCondition")
                  for name in _parse_reference_list(cursor)]
    if cursor.peek() != "{":
        if any(condition.needs_comparison() for condition in conditions):
            raise Asn1Error(
                "ECN: §23.7.2.2 — IF-ALL shall be used with three lists if one or more of "
                "the conditions requires a comparison")
        return tuple((condition, None, None) for condition in conditions)
    comparisons = [_named(Token(name, 0), _COMPARISONS, "§21.12.1's Comparison")
                   for name in _parse_reference_list(cursor)]
    comparators = []
    for text in _parse_reference_list(cursor):
        try:
            comparators.append(int(text))
        except ValueError:
            raise Asn1Error(
                f"ECN: §21.11.5's comparator is an integer; got {text!r}") from None
    if not len(comparisons) == len(comparators):
        raise Asn1Error(
            f"ECN: §23.7.2.2's three lists are read by position, so the comparison list "
            f"({len(comparisons)}) and the comparator list ({len(comparators)}) have to "
            f"match")
    if len(comparisons) > len(conditions):
        raise Asn1Error(
            f"ECN: §23.7.2.2 gives {len(conditions)} conditions but {len(comparisons)} "
            f"comparisons; a comparison with no condition in the same position tests nothing")
    # §23.7.2.2: "size-range-conditions that do not require a comparison or comparator (if
    # any) shall follow all those that require a comparison, and shall have no corresponding
    # entry in the second and third lists." So the short lists are a prefix, not a sparse map.
    out = []
    for index, condition in enumerate(conditions):
        has = index < len(comparisons)
        if condition.needs_comparison() != has:
            raise Asn1Error(
                f"ECN: §23.7.2.2 — {condition.value} at position {index} "
                f"{'needs' if condition.needs_comparison() else 'admits'} no comparison in "
                f"that position; conditions taking one come first")
        out.append((condition, comparisons[index] if has else None,
                    comparators[index] if has else None))
    return tuple(out)


def _named(token, table: dict, what: str):
    try:
        return table[token.text]
    except KeyError:
        raise Asn1Error(
            f"ECN: {token} names no value of {what} "
            f"({', '.join(sorted(table))})") from None


@dataclass(frozen=True)
class _CommonGroups:
    """The groups clause 23 gives every bit-field class, parsed once."""

    pre_alignment: PreAlignment | None = None
    space: _EncodingSpace = field(default_factory=_EncodingSpace)
    value_padding: ValuePadding | None = None
    start_pointer: StartPointer | None = None
    bit_reversal: ReversalSpecification = ReversalSpecification.NO_REVERSAL


def _refuse_unsupported(cursor: _Cursor) -> None:
    keyword = cursor.peek()
    if keyword in _UNSUPPORTED_KEYWORDS:
        raise Asn1Error(
            f"ECN: `{keyword}` is a property group this rail does not build — "
            f"{_UNSUPPORTED_KEYWORDS[keyword]}. It is recognized rather than skipped, because "
            f"an encoder that ignored it would write octets the specification does not "
            f"describe")


def _parse_common(cursor: _Cursor, module: "EcnModule", *,
                  space_required: bool) -> _CommonGroups:
    """§23.x's shared prefix and suffix, in the order the WITH SYNTAX gives them.

    The order is not this parser's convenience. §23.3.3.1 lists the encoder actions —
    replacement, pre-alignment, start pointer, encoding space, value encoding, value padding,
    handle, bit reversal — and the syntax follows that order, so reading it in the same order
    keeps one sequence rather than two that have to be kept in step.
    """
    _refuse_unsupported(cursor)
    pre_alignment = None
    if cursor.accept_words("ALIGNED", "TO"):
        pre_alignment = _parse_pre_alignment(cursor)
    _refuse_unsupported(cursor)
    start_pointer = None
    if cursor.accept("START-POINTER"):
        start_pointer = _parse_start_pointer(cursor, module)
    if (pre_alignment is not None and pre_alignment.encoder_chosen_offset
            and start_pointer is None):
        raise Asn1Error(
            "ECN: §22.2.2.2 — if `ALIGNED TO ANY` is specified, then the encoding object "
            "specification shall include the START-POINTER clause; nothing else could tell a "
            "decoder how many bits the encoder chose to insert")
    _refuse_unsupported(cursor)
    space = _EncodingSpace()
    if cursor.accept("ENCODING-SPACE"):
        space = _parse_encoding_space(cursor, module)
    elif space_required:
        raise Asn1Error(
            "ECN: clause 23's WITH SYNTAX gives `ENCODING-SPACE` without brackets for this "
            "class, so it is mandatory rather than optional")
    return _CommonGroups(pre_alignment=pre_alignment, space=space,
                         start_pointer=start_pointer)


def _parse_value_padding_tail(cursor: _Cursor, module: "EcnModule") -> ValuePadding | None:
    _refuse_unsupported(cursor)
    if cursor.accept("VALUE-PADDING"):
        return _parse_value_padding(cursor, module)
    return None


def _parse_handle_tail(cursor: _Cursor) -> "IdentificationHandle | None":
    """§22.9.1.2's `[EXHIBITS HANDLE &exhibited-handle AT &Handle-positions
    [AS &handle-value-set]]`.

    It sits between `VALUE-PADDING` and `BIT-REVERSAL` because §23.3.3.1's encoder actions put
    it there — value padding and justification, **identification handle**, bit reversal — and
    this grammar follows that list rather than a second order of its own.

    `AT` is inside the outer bracket rather than in one of its own, so a handle always has
    positions. `AS` is optional, and its DEFAULT is §21.16.5's `tag:any`, which only a `#TAG`
    object can resolve — so omitting it anywhere else is a refusal that arrives at write time
    with §22.9.1.9's words rather than a silent match-nothing.
    """
    _refuse_unsupported(cursor)
    if not cursor.accept_words("EXHIBITS", "HANDLE"):
        return None
    named = cursor.next().text
    cursor.expect("AT")
    positions = _parse_int_list(cursor)
    value_set = HandleValueSet.tag_any()
    if cursor.accept("AS"):
        value_set = _parse_handle_value_set(cursor)
    return IdentificationHandle(name=named, positions=positions, value_set=value_set)


def _parse_handle_value_set(cursor: _Cursor) -> HandleValueSet:
    """§21.16.1's CHOICE, written as its ASN.1 value notation: `bits:'01'B`, `number:5`,
    `tag:any`, `range:{4, 7}`, `ranges:{{0, 1}, {6, 7}}`.

    `bits:` and `octets:` reuse `Pattern`'s spellings, because §21.10 and §21.16 write a
    BIT STRING and an OCTET STRING the same way — the difference between them is what the
    bits are *for*, not how they are written.
    """
    if cursor.peek() == "ranges:":
        cursor.next()
        cursor.expect("{")
        pairs = [_parse_int_pair(cursor)]
        while cursor.accept(","):
            pairs.append(_parse_int_pair(cursor))
        cursor.expect("}")
        return HandleValueSet.of_ranges(pairs)
    if cursor.peek() == "range:":
        cursor.next()
        low, high = _parse_int_pair(cursor)
        return HandleValueSet.of_range(low, high)
    token = cursor.next()
    alternative, _, body = token.text.partition(":")
    if alternative == "tag" and body == "any":
        return HandleValueSet.tag_any()
    if alternative == "bits":
        if not (body.startswith("'") and body.endswith("'B")):
            raise Asn1Error(
                f"ECN: {token} — §21.16.1's `bits` alternative is a BIT STRING like '1010'B")
        return HandleValueSet.from_bits(body[1:-2])
    if alternative == "octets":
        if not (body.startswith("'") and body.endswith("'H")):
            raise Asn1Error(
                f"ECN: {token} — §21.16.1's `octets` alternative is a hex string like 'FF'H")
        try:
            return HandleValueSet.from_octets(bytes.fromhex(body[1:-2]))
        except ValueError:
            raise Asn1Error(f"ECN: {token} is not a hex string") from None
    if alternative == "number":
        try:
            return HandleValueSet.of_number(int(body))
        except ValueError:
            raise Asn1Error(
                f"ECN: §21.16.1's `number` alternative is an INTEGER (0..MAX); got "
                f"{token}") from None
    raise Asn1Error(
        f"ECN: {token} names no alternative of §21.16.1's HandleValueSet "
        f"(bits, octets, number, tag, range, ranges)")


def _parse_int_pair(cursor: _Cursor) -> tuple[int, int]:
    """§21.16.1's `SEQUENCE {low INTEGER(0..MAX), high INTEGER(0..MAX)}`."""
    values = _parse_int_list(cursor)
    if len(values) != 2:
        raise Asn1Error(
            f"ECN: §21.16.1 gives a range a `low` and a `high`; {list(values)} has "
            f"{len(values)}")
    return values[0], values[1]


def _parse_reversal_tail(cursor: _Cursor) -> ReversalSpecification:
    """Clause 23's `[BIT-REVERSAL &bit-reversal]`, which every bit-field class ends with."""
    _refuse_unsupported(cursor)
    if cursor.accept("BIT-REVERSAL"):
        return _parse_bit_reversal(cursor)
    return ReversalSpecification.NO_REVERSAL


def _finish(cursor: _Cursor, what: str) -> None:
    _refuse_unsupported(cursor)
    if not cursor.eof():
        raise Asn1Error(
            f"ECN: {cursor.peek()!r} is not part of {what}'s defined syntax, or appears out "
            f"of the order clause 23 gives its property groups")


def _parse_bool_body(cursor: _Cursor, name: str, module: "EcnModule") -> BoolSpec:
    """§23.3.1's `#BOOL`, restricted to the groups this rail writes."""
    common = _parse_common(cursor, module, space_required=True)
    true_pattern = Pattern.from_bits("1")   # §23.3.1's DEFAULT bits:'1'B
    false_pattern = Pattern.from_bits("0")  # §23.3.1's DEFAULT bits:'0'B
    if cursor.accept("TRUE-PATTERN"):
        true_pattern = _parse_pattern(cursor)
    if cursor.accept("FALSE-PATTERN"):
        false_pattern = _parse_pattern(cursor)
    if (true_pattern.kind.value == "different" and false_pattern.kind.value == "different"):
        raise Asn1Error(
            "ECN: §23.3.2.3 — at most one of TRUE-PATTERN and FALSE-PATTERN may be "
            "`different:any`; §21.10.9 needs the other one to differ from")
    padding = _parse_value_padding_tail(cursor, module)
    exhibits = _parse_handle_tail(cursor)
    reversal = _parse_reversal_tail(cursor)
    _finish(cursor, name)
    true_bits = true_pattern.bit_sequence()
    false_bits = false_pattern.bit_sequence()
    width = common.space.width
    if len(true_bits) > width or len(false_bits) > width:
        raise Asn1Error(
            f"ECN: {name}'s patterns are {len(true_bits)} and {len(false_bits)} bits, which "
            f"do not fit a {width}-bit encoding space")
    if padding is None and (len(true_bits) != width or len(false_bits) != width):
        raise Asn1Error(
            f"ECN: §23.3.2.7 — {name} leaves unused bits in its encoding space, so "
            f"VALUE-PADDING shall be set")
    return BoolSpec(
        width=width,
        true_value=_bits_to_int(_place(true_bits, width, padding)),
        false_value=_bits_to_int(_place(false_bits, width, padding)),
        pre_alignment=common.pre_alignment, start_pointer=common.start_pointer,
        space_determinant=common.space.determinant, exhibits=exhibits,
        bit_reversal=reversal, reversal_unit=common.space.unit)


def _place(bits: tuple[int, ...], width: int, padding: ValuePadding | None) -> tuple[int, ...]:
    return bits if padding is None else padding.place(bits, width)


def _bits_to_int(bits: tuple[int, ...]) -> int:
    out = 0
    for bit in bits:
        out = (out << 1) | (1 if bit else 0)
    return out


_INT_ENCODINGS = {
    "positive-int": IntForm.POSITIVE_INT,
    "twos-complement": IntForm.TWOS_COMPLEMENT,
}


def _parse_conditional_int_body(cursor: _Cursor, name: str,
                                module: "EcnModule") -> ConditionalIntSpec:
    """§23.7.1's `#CONDITIONAL-INT`, which is where an integer's encoding actually lives.

    §23.7.2.4: "At most one of `IF`, `IF-ALL` and `ELSE` shall be present", and §23.7.2.2
    makes `ELSE` and the omission of all three mean the same thing — no condition.
    """
    conditions = ()
    if cursor.accept("IF"):
        conditions = _parse_conditions(cursor, all_of=False)
    elif cursor.accept("IF-ALL"):
        conditions = _parse_conditions(cursor, all_of=True)
    elif not cursor.accept("ELSE"):
        pass
    if cursor.peek() in ("IF", "IF-ALL", "ELSE"):
        raise Asn1Error(
            f"ECN: §23.7.2.4 — at most one of IF, IF-ALL and ELSE shall be present; {name} "
            f"has a second")
    common = _parse_common(cursor, module, space_required=True)
    chain: TransformChain | None = None
    if cursor.accept("TRANSFORMS"):
        chain = TransformChain(module.transform_list(_parse_reference_list(cursor)))
    form = IntForm.TWOS_COMPLEMENT  # §23.7.1's DEFAULT.
    if cursor.accept("ENCODING"):
        token = cursor.next()
        if token.text in ("reverse-positive-int", "reverse-twos-complement"):
            raise Asn1Error(
                f"ECN: §23.7.1 admits `{token.text}`, whose bits run from the least "
                f"significant end; this rail writes the forward forms only")
        try:
            form = _INT_ENCODINGS[token.text]
        except KeyError:
            raise Asn1Error(
                f"ECN: {token} names no alternative of §23.7.1's &encoding ENUMERATED") from None
    padding = _parse_value_padding_tail(cursor, module)
    exhibits = _parse_handle_tail(cursor)
    reversal = _parse_reversal_tail(cursor)
    _finish(cursor, name)
    # §23.7.2.7's `subtract:lower-bound` rule relates the transforms to the CONDITION, so it
    # is checked by ConditionalIntSpec where both are in hand rather than here.
    return ConditionalIntSpec(
        spec=IntSpec(width=common.space.width, form=form, transform=chain,
                     pre_alignment=common.pre_alignment, value_padding=padding,
                     start_pointer=common.start_pointer,
                     space_determinant=common.space.determinant, exhibits=exhibits,
                     bit_reversal=reversal, reversal_unit=common.space.unit),
        conditions=conditions)


def _parse_int_body(cursor: _Cursor, name: str, module: "EcnModule"):
    """§23.6.1's `#INT`: `[ENCODINGS &Integer-encodings] [ENCODING &integer-encoding]`.

    §23.6.2.2: "Exactly one of ENCODING and ENCODINGS shall be set." Both name
    `#CONDITIONAL-INT` objects, and §23.6.3.1 selects "the first ... whose conditions are
    satisfied" against the bounds of the type the object set is applied to.

    `BOUNDS` is this rail's stand-in for those bounds, and it is a **deviation stated rather
    than hidden** — now a *fallback* rather than the only source. In X.692 the bounds arrive
    from the ASN.1 type through the encoding link module — **clause 12**, whose own NOTE
    separates it from clause 14's encoding *definition* module: "There are two top-level
    productions in ECN, the `ELMDefinition` specified in this clause and the `EDMDefinition`
    specified in clause 14."

    [`ecn_link.py`](ecn_link.py) builds that link: `LinkedStructure.bounds_for` reads the
    bounds off the ASN.1 component's own constraint, which is where §21.11.3 and §23.7.2.6's
    NOTE both put them. `BOUNDS` remains accepted because a specification written against this
    rail may still have no ASN.1 type in hand — a value dict is what it links against then —
    but where there is a link, the link is authoritative and this clause is redundant.
    """
    if cursor.accept("AUXILIARY"):
        # A DEVIATION, stated. X.692 has no keyword for "this field is auxiliary" because
        # §22.1.2.6's classification comes from the encoding link module: a structure field
        # with no ASN.1 component behind it is auxiliary, and clause 12's ELM is what decides
        # that. §19.3.1 says the same thing from clause 19's side -- a target structure "has
        # fields corresponding to the components of the type, but also has added fields for
        # determinants" -- which is why `MatchingFields.added` in ecn_mapping.py names them
        # too. This rail links against a value dict rather than an ASN.1 type, so the fact
        # has nowhere else to be written. It is spelled on the object, where the width already
        # lives, and named here rather than inferred from "the value did not carry it" —
        # which would make a typo in a field name silently produce a determinant field.
        common = _parse_common(cursor, module, space_required=True)
        _finish(cursor, name)
        return AuxIntSpec(width=common.space.width, form=IntForm.POSITIVE_INT,
                          pre_alignment=common.pre_alignment)
    single = cursor.accept("ENCODING")
    plural = not single and cursor.accept("ENCODINGS")
    if not (single or plural):
        raise Asn1Error(
            f"ECN: §23.6.2.2 — {name} must set exactly one of ENCODING and ENCODINGS")
    if single:
        names = (cursor.next().text,)
    else:
        names = _parse_reference_list(cursor)
    if cursor.accept("ENCODING") or cursor.accept("ENCODINGS"):
        raise Asn1Error(
            f"ECN: §23.6.2.2 — {name} sets both ENCODING and ENCODINGS; exactly one is "
            f"permitted")
    bounds = IntegerBounds()
    if cursor.accept("BOUNDS"):
        bounds = IntegerBounds(_parse_bound(cursor), _parse_bound(cursor))
    _finish(cursor, name)
    encodings = tuple(module.conditional_int(reference) for reference in names)
    if plural and len(encodings) == 1:
        raise Asn1Error(
            f"ECN: §23.6.2.2 gives ENCODINGS a list; {name} lists one object, which is what "
            f"ENCODING spells")
    return IntSelector(encodings=encodings, bounds=bounds)


def _parse_bound(cursor: _Cursor) -> int | None:
    """One bound, `-` for absent. §21.11.4 turns on EXISTENCE, so absent is not a number."""
    token = cursor.next()
    if token.text == "-":
        return None
    try:
        return int(token.text)
    except ValueError:
        raise Asn1Error(
            f"ECN: a bound is an integer, or `-` for none; got {token}") from None


def _parse_pad_body(cursor: _Cursor, name: str, module: "EcnModule") -> PadSpec:
    """§23.12.1's `#PAD`."""
    common = _parse_common(cursor, module, space_required=True)
    pattern: Pattern | None = None
    padding = Padding.ZERO
    if cursor.accept("PAD-PATTERN"):
        pattern = _parse_pattern(cursor)
        padding = Padding.PATTERN
        # §23.12.2.2: with a positive ENCODING-SPACE SIZE the pattern "shall not be of zero
        # length, and is replicated and truncated to fill the encoding space".
        pattern.require_non_null(f"{name}'s PAD-PATTERN")
    exhibits = _parse_handle_tail(cursor)
    _finish(cursor, name)
    return PadSpec(width=common.space.width, padding=padding, pattern=pattern,
                   exhibits=exhibits)


def _parse_outer_body(cursor: _Cursor, name: str) -> OuterSpec:
    """Clause 25's `#OUTER`, restricted to the post-padding this rail applies.

    §25 gives `#OUTER` more than this — a bit-reversal and a contained-type reset among them —
    and the one property built is the one §21.9.3 names: `Padding` specifies "the post-padding
    of a PDU specified in the #OUTER encoding class".
    """
    padding = Padding.ZERO
    pattern: Pattern | None = None
    boundary = 8
    if cursor.accept_words("MULTIPLE", "OF"):
        boundary = _parse_unit(cursor)
    if cursor.accept("POST-PADDING"):
        padding = _parse_padding(cursor)
        if cursor.accept("PATTERN"):
            pattern = _parse_pattern(cursor)
    _finish(cursor, name)
    return OuterSpec(boundary_bits=boundary, padding=padding, pattern=pattern)


def _parse_concatenation_body(cursor: _Cursor, name: str, module: "EcnModule"
                              ) -> ConcatenationSpec:
    """§23.5.1's `#CONCATENATION`, whose components come from the §16.5 structure.

    §22.10.1.1 gives this class no property naming its components: the fields, and their
    textual order, are the encoding structure's. So the object is parsed for its own
    properties and the structure supplies the rest.
    """
    common = _parse_common(cursor, module, space_required=False)
    if common.space.size is not None:
        raise Asn1Error(
            f"ECN: §21.2.8 forbids `fixed-to-max` for a concatenation, and a stated SIZE on "
            f"{name} would fix the whole structure's width; the components determine it")
    order = "textual"
    if cursor.accept("CONCATENATION"):
        if cursor.accept("ORDER"):
            order = cursor.next().text
        if cursor.accept("ALIGNMENT"):
            alignment = cursor.next().text
            if alignment not in ("none", "aligned"):
                raise Asn1Error(
                    f"ECN: §22.10.1.1's &concatenation-alignment is ENUMERATED "
                    f"{{none, aligned}}; got {alignment!r}")
            if alignment == "aligned":
                raise Asn1Error(
                    "ECN: §22.10.2.2 makes ALIGNMENT aligned pre-align every component with "
                    "the concatenation's own pre-alignment defaults; this rail applies "
                    "pre-alignment per component, so state `ALIGNMENT none` and put "
                    "`ALIGNED TO` on the components that need it")
        if cursor.accept("HANDLE"):
            cursor.next()
    if order != "textual":
        raise Asn1Error(
            "ECN: §22.10.1.1's &concatenation-order is ENUMERATED {textual, tag, random}; "
            "`tag` needs every component to start with a class in the tag category "
            "(§22.10.2.4) and `random` needs disjoint identification handles (§22.10.2.1). "
            "`textual` is built, and §22.10.3.1 takes it from the ECN structure definition")
    _finish(cursor, name)
    structure = module.require_structure()
    fields = {}
    for field_name, class_name in structure:
        fields[field_name] = module.spec_for_class(class_name, field_name)
    padding = tuple(
        field_name for field_name, class_name in structure
        if isinstance(fields[field_name], PadSpec))
    return ConcatenationSpec(fields=fields, order=tuple(name for name, _ in structure),
                             padding=padding)


def _parse_parameter(text: str) -> Parameter:
    """X.683 §8.3's `Parameter ::= [ParamGovernor ":"] DummyReference`, as C.1 governs it.

    `:` is not punctuation in this lexer — §21.8's `left:2` and §24.3.1's `divide:4` are single
    tokens — so a governed parameter arrives whole and splits here. That is a happy accident of
    ECN's own value notation rather than a design, and it is worth saying: if `:` were ever
    made punctuation for some other clause, this function is what would break.

    The governor determines the dummy's kind only when C.1 does not share it (see
    `ecn_param.kinds_for`). A shared governor leaves `kind` `None`, which is not a gap: X.683
    §9.6 settles it from the actual parameter's form instead.
    """
    governor_text, _, dummy = text.rpartition(":")
    if not dummy:
        raise Asn1Error(f"ECN: X.683 8.3 — {text!r} is a governor with no DummyReference")
    if not governor_text:
        # C.1 a): "an encoding class, in which case there shall be no ParamGovernor".
        return Parameter(dummy, ParameterKind.ENCODING_CLASS)
    if governor_text == "REFERENCE":
        return Parameter(dummy, ParameterKind.IDENTIFIER, GovernorKind.REFERENCE)
    if governor_text == "#ENCODINGS":
        return Parameter(dummy, ParameterKind.ENCODING_OBJECT_SET, GovernorKind.ENCODINGS)
    if not governor_text.startswith("#"):
        raise Asn1Error(
            f"ECN: C.1's Governor is `EncodingClassFieldType | REFERENCE | "
            f"DefinedOrBuiltinEncodingClass | #ENCODINGS | Type`; {governor_text!r} is none of "
            f"them. (`Type` is in that production and in none of C.1's a)-d) rules, so a "
            f"dummy governed by an ASN.1 type stands for nothing and is refused here.)")
    if ".&" in governor_text:
        # C.1 b): a type extracted from an encoding class governs a value, a value set or a
        # fixed-type ordered value list — three kinds, one governor, so the kind stays open.
        return Parameter(dummy, None, GovernorKind.ENCODING_CLASS_FIELD_TYPE, governor_text)
    # C.1 c): an encoding class governs an encoding object or an ordered encoding object list.
    return Parameter(dummy, None, GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS,
                     governor_text)


def _parse_parameter_list(cursor: _Cursor) -> ParameterList:
    """C.1's `ParameterList ::= "{<" Parameter "," + ">}"`."""
    cursor.expect("{<")
    parameters: list[Parameter] = []
    while not cursor.accept(">}"):
        if cursor.accept(","):
            continue
        if cursor.eof():
            raise Asn1Error("ECN: C.1 — a parameter list opened with `{<` has no `>}`")
        parameters.append(_parse_parameter(cursor.next().text))
    return ParameterList(tuple(parameters))


def _parse_actual(text: str, parameter: Parameter | None) -> ActualParameter:
    """One entry of C.4's `ActualParameterList`, classified as far as spelling allows.

    C.4's a)-h) run from the dummy's kind to the actual's alternative, so the actual's own
    spelling settles only the cases ECN gave a distinct shape: `STRUCTURE` and `OUTER` are
    keywords, a `#` opens a class reference, and a `.` makes §15.3.1's `ComponentIdList`. A
    bare name is none of those, and what it denotes depends on the dummy it is being supplied
    to — which is X.683 §9.6's direction of fit, so `parameter` is consulted rather than
    guessed at.
    """
    if text == "STRUCTURE":
        return ActualParameter(ActualKind.STRUCTURE)
    if text == "OUTER":
        return ActualParameter(ActualKind.OUTER)
    if text.startswith("#"):
        return ActualParameter(ActualKind.ENCODING_CLASS, text)
    if "." in text:
        return ActualParameter(ActualKind.COMPONENT_ID_LIST, text)
    candidates = () if parameter is None else parameter.candidates()
    bare = {
        ParameterKind.IDENTIFIER: ActualKind.IDENTIFIER,
        ParameterKind.ENCODING_OBJECT: ActualKind.ENCODING_OBJECT,
        ParameterKind.ENCODING_OBJECT_SET: ActualKind.ENCODING_OBJECT_SET,
    }
    resolved = {bare[kind] for kind in candidates if kind in bare}
    if len(resolved) == 1:
        return ActualParameter(next(iter(resolved)), text)
    raise Asn1Error(
        f"ECN: C.4 — {text!r} is a bare name, which is the spelling shared by an identifier, "
        f"an encoding object and an encoding object set; the dummy it is supplied to does not "
        f"narrow it to one, so nothing here can say which alternative was meant")


def _parse_actual_parameter_list(cursor: _Cursor,
                                 parameters: ParameterList | None = None
                                 ) -> ActualParameterList:
    """C.4's `ActualParameterList ::= "{<" ActualParameter "," + ">}"`, and C.3's empty form.

    Empty is accepted here and refused by `ParameterList`, which is C.3's doing: it makes
    `Reference "{<" ">}"` a way to *name* a definition, so the same two brackets mean "no
    actuals" here and are not a parameter list at all there.
    """
    cursor.expect("{<")
    actuals: list[ActualParameter] = []
    while not cursor.accept(">}"):
        if cursor.accept(","):
            continue
        if cursor.eof():
            raise Asn1Error("ECN: C.4 — an actual parameter list opened with `{<` has no `>}`")
        supplied = None
        if parameters is not None and len(actuals) < len(parameters):
            supplied = parameters.parameters[len(actuals)]
        actuals.append(_parse_actual(cursor.next().text, supplied))
    return ActualParameterList(tuple(actuals))


def _parse_reference_list(cursor: _Cursor) -> tuple[str, ...]:
    """`{ ref, ref, ... }` — X.681's object set / ordered list notation."""
    cursor.expect("{")
    names: list[str] = []
    while True:
        token = cursor.next()
        if token.text == "}":
            break
        if token.text == ",":
            continue
        names.append(token.text)
    return tuple(names)


# --- the module -----------------------------------------------------------------------------

@dataclass
class EcnModule:
    """A parsed `ENCODING-DEFINITIONS` module: classes, one structure, and the objects.

    Mutable during parsing because assignments resolve references to earlier ones — an
    `#INT` object names a `#CONDITIONAL-INT` object, and a `#CONCATENATION` object reads the
    structure. §9.5.2's one-object-per-class law is enforced as the objects land, which is
    what makes `object_set()` well defined.
    """

    name: str = ""
    #: A defined class name -> the built-in class it was assigned from (clause 11).
    classes: dict[str, str] = field(default_factory=dict)
    #: The §16.5 concatenation structure: `(field name, class name)` in textual order.
    structure: tuple[tuple[str, str], ...] = ()
    structure_name: str = ""
    #: Object reference -> `(class name, spec)`.
    objects: dict[str, tuple[str, object]] = field(default_factory=dict)
    #: Object reference -> the transform it defines.
    transforms: dict[str, object] = field(default_factory=dict)
    #: C.2's parameterized assignments, by name. Held UNINSTANTIATED, which is X.683 §9.7's
    #: doing: instantiation is substitution of actuals for dummies, so there is nothing to
    #: resolve until a reference supplies them, and any body built for a dummy now would be
    #: wrong for some later actual. `frontends/asn1/ast.py` keeps ASN.1's the same way.
    #:
    #: These live beside `structure` rather than in it. A §22.1 replacement structure is a
    #: separate definition that a REPLACE instantiates *around* a component — it is not the
    #: application point §13.2 walks, and putting it in `structure` would make the module look
    #: like it declared two.
    parameterized: dict[str, object] = field(default_factory=dict)

    def builtin_of(self, class_name: str) -> str:
        """The built-in class a name resolves to, following clause 11's assignments."""
        seen = set()
        current = class_name
        while current not in _BUILTIN_CLASSES:
            if current in seen:
                raise Asn1Error(f"ECN: the class assignment for {current} is circular")
            seen.add(current)
            if current not in self.classes:
                raise Asn1Error(
                    f"ECN: {current} is not a built-in encoding class and no assignment "
                    f"defines it")
            current = self.classes[current]
        return _BUILTIN_CLASSES[current]

    def transform_list(self, names: tuple[str, ...]) -> tuple:
        out = []
        for name in names:
            if name not in self.transforms:
                raise Asn1Error(
                    f"ECN: TRANSFORMS names {name!r}, which no #TRANSFORM object in this "
                    f"module defines; §24.2.4.1 orders a list of objects that must exist")
            out.append(self.transforms[name])
        return tuple(out)

    def conditional_int(self, name: str) -> ConditionalIntSpec:
        entry = self.objects.get(name)
        if entry is None or self.builtin_of(entry[0]) != "conditional-int":
            raise Asn1Error(
                f"ECN: §23.6.1 gives #INT's ENCODING a #CONDITIONAL-INT object; {name!r} is "
                f"{'undefined' if entry is None else 'a ' + entry[0] + ' object'}")
        return entry[1]

    def require_structure(self) -> tuple[tuple[str, str], ...]:
        if not self.structure:
            raise Asn1Error(
                "ECN: a #CONCATENATION object takes its components from a §16.5 "
                "ConcatenationStructure, and this module declares none")
        return self.structure

    def spec_for_class(self, class_name: str, field_name: str):
        """§9.5.2: the module's one object for `class_name`.

        The error names the law rather than the symbol, because "no object for #Version" is
        the same shape of mistake whether the object is missing or was written for a
        different class, and a specification is not applicable until every class in its
        structure has exactly one.
        """
        matches = [spec for name, (cls, spec) in self.objects.items() if cls == class_name]
        if len(matches) != 1:
            raise Asn1Error(
                f"ECN: §9.5.1 requires an encoding object for every class in the structure, "
                f"and §9.5.2 permits at most one per class; the field {field_name!r} has "
                f"class {class_name} with {len(matches)} objects in this module")
        return matches[0]

    def object_set(self) -> dict:
        """The §9.5.1 set this module forms, keyed by class, in `encode_with_user`'s shape.

        Built from the classes the structure actually uses rather than from every object in
        the module, because those are different collections and §9.5.2 governs only the
        first. `#TRANSFORM` and `#CONDITIONAL-INT` objects are referenced by name from other
        objects and are not applied to anything, so a module may hold several of each while
        the set it forms still holds at most one object per class.
        """
        applied: dict[str, object] = {}
        for _field_name, class_name in self.structure:
            applied[class_name] = self.spec_for_class(class_name, _field_name)
        for name, (class_name, spec) in self.objects.items():
            if isinstance(spec, (ConcatenationSpec, OuterSpec)):
                if class_name in applied:
                    raise Asn1Error(
                        f"ECN: §9.5.2 permits a set at most one object per class, and "
                        f"{class_name} already has one; {name} would be a second")
                applied[class_name] = spec
        return {
            class_name: UserEncodingObject(class_name, spec, self._name_of(spec))
            for class_name, spec in applied.items()
        }

    def _name_of(self, spec) -> str:
        for name, (_class_name, candidate) in self.objects.items():
            if candidate is spec:
                return name
        return ""

    def outer(self) -> OuterSpec | None:
        for class_name, spec in self.objects.values():
            if isinstance(spec, OuterSpec):
                return spec
        return None

    def concatenation(self) -> ConcatenationSpec:
        for class_name, spec in self.objects.values():
            if isinstance(spec, ConcatenationSpec):
                return spec
        raise Asn1Error("ECN: this module defines no #CONCATENATION encoding object")

    def serialize(self) -> bytes:
        """A canonical byte form, written by hand so byte identity is by construction.

        **This is deliberately not `encode_plan` version 6.** That plan describes an ASN.1
        *type* — tag, members, enumeration, constraint, extensibility — and five emitters read
        the same node and apply their own rule to it. An ECN encoding is not a sixth rule over
        those facts: `legacy_frame_objects` transmits `payloadOctets` before `version` and
        emits a `reserved` field that corresponds to no ASN.1 component at all. `EncodeNode`
        has no slot for either, because both are properties of an encoding structure rather
        than of a type. Carrying them would make a node's meaning depend on which candidate
        read it, which is the one thing that plan's design rules out.

        So this is a third compilation of the same schema, and `encode_plan`'s own docstring
        already made the argument for the second one: a write plan is not a read plan, and
        "saying so is cheaper than pretending one plan serves both directions". The same holds
        here, and the version counter is this module's own.
        """
        out = [
            f"ecn-syntax-version {SYNTAX_VERSION}",
            f"compiler {SYNTAX_COMPILER}",
            f"module {self.name}",
        ]
        for defined in sorted(self.classes):
            out.append(f"class {defined} from {self.classes[defined]}")
        for defined in sorted(self.parameterized):
            assignment = self.parameterized[defined]
            governor = assignment.governor or "-"
            if assignment.governor_actuals is not None:
                governor += assignment.governor_actuals.render()
            # The BODY is in the digest, not just the signature. Two modules whose replacement
            # structures differ only inside the braces describe different octets, and a name
            # they shared would be a name that meant two things.
            out.append(
                f"parameterized {defined} {assignment.kind.value} "
                f"{assignment.parameters.render_declaration()} governor {governor} "
                f"body {' '.join(assignment.body)}")
        if self.structure:
            out.append(f"structure {self.structure_name} fields {len(self.structure)}")
            for index, (field_name, class_name) in enumerate(self.structure):
                out.append(f"field {index} name {field_name} class {class_name}")
        for name in sorted(self.transforms):
            out.append(f"transform {name} {_describe(self.transforms[name])}")
        for name in sorted(self.objects):
            class_name, spec = self.objects[name]
            out.append(f"object {name} class {class_name} {_describe(spec)}")
        return ("\n".join(out) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()


def _describe(spec) -> str:
    """One line per object, naming every property that reaches the octets.

    Written out per type rather than derived from the dataclass fields, for the reason the
    other serializers in this package give: a field added tomorrow must force a deliberate
    decision about whether it belongs in the digest, and `asdict` would quietly change every
    existing hash instead.
    """
    if isinstance(spec, IntToInt):
        return f"int-to-int {spec.op.value} {spec.operand}"
    if isinstance(spec, IntToBits):
        return (f"int-to-bits {spec.encoded_as.value} size {spec.size} unit {spec.unit} "
                f"bounds {_bound(None if spec.bounds is None else spec.bounds[0])}.."
                f"{_bound(None if spec.bounds is None else spec.bounds[1])}")
    if isinstance(spec, BoolToBool):
        return "bool-to-bool logical:not"
    if isinstance(spec, BoolToInt):
        return f"bool-to-int {'true-zero' if spec.true_zero else 'true-one'}"
    if isinstance(spec, IntToBool):
        return (f"int-to-bool {'zero-true' if spec.zero_true else 'zero-false'} "
                f"true {_ints(spec.true_is)} false {_ints(spec.false_is)}")
    if isinstance(spec, IntToChars):
        return (f"int-to-chars size {spec.size} plus {int(spec.plus_sign)} "
                f"pad {'spaces' if spec.pad_with_spaces else 'zeros'}")
    if isinstance(spec, BitsToInt):
        return f"bits-to-int {spec.decoded_assuming.value}"
    if isinstance(spec, CharToBits):
        return (f"char-to-bits {spec.encoded_as} alphabet {spec.alphabet or '-'} "
                f"chars {'/'.join(spec.chars) or '-'} bits {_bits_list(spec.bit_values)} "
                f"size {spec.size} unit {spec.unit}")
    if isinstance(spec, BitsToChar):
        return (f"bits-to-char {spec.decoded_assuming} bits {_bits_list(spec.bit_values)} "
                f"chars {'/'.join(spec.chars) or '-'}")
    if isinstance(spec, BitToBits):
        return (f"bit-to-bits zero {_pattern(spec.zero_pattern)} "
                f"one {_pattern(spec.one_pattern)}")
    if isinstance(spec, BitsToBits):
        return (f"bits-to-bits source {_bits_list(spec.source_values)} "
                f"result {_bits_list(spec.result_values)}")
    if isinstance(spec, BitsToCompositeBits):
        return f"bits-to-composite-bits unit {spec.unit}"
    # The four property-free composite transforms (24.14, 24.16, 24.17, 24.18, 24.19) carry
    # nothing but their identity, which the name already is.
    for kind, mnemonic in ((CharsToCompositeChar, "chars-to-composite-char"),
                           (OctetsToCompositeBits, "octets-to-composite-bits"),
                           (CompositeCharToChars, "composite-char-to-chars"),
                           (CompositeBitsToBits, "composite-bits-to-bits"),
                           (CompositeBitsToOctets, "composite-bits-to-octets")):
        if isinstance(spec, kind):
            return mnemonic
    if isinstance(spec, IntSpec):
        return (f"int width {spec.width} form {spec.form.value} "
                f"transform {_chain(spec.transform)} "
                f"pre {_pre(spec.pre_alignment)} pad {_pad(spec.value_padding)} "
                f"ptr {_ref(spec.start_pointer)} det {_det(spec.space_determinant)} "
                f"handle {_handle(spec.exhibits)} "
                f"rev {spec.bit_reversal.value}:{spec.reversal_unit}")
    if isinstance(spec, ConditionalIntSpec):
        conditions = "/".join(
            f"{condition.value}"
            + (f":{comparison.value}:{comparator}" if comparison is not None else "")
            for condition, comparison, comparator in spec.conditions) or "-"
        return f"conditional-int if {conditions} then {_describe(spec.spec)}"
    if isinstance(spec, IntSelector):
        chosen = "/".join(_name_of_condition(entry) for entry in spec.encodings) or "-"
        return (f"int-selector bounds {_bound(spec.bounds.low)}..{_bound(spec.bounds.high)} "
                f"encodings {chosen}")
    if isinstance(spec, AuxIntSpec):
        return (f"aux width {spec.width} form {spec.form.value} "
                f"pre {_pre(spec.pre_alignment)}")
    if isinstance(spec, BoolSpec):
        return (f"bool width {spec.width} true {spec.true_value} false {spec.false_value} "
                f"pre {_pre(spec.pre_alignment)} handle {_handle(spec.exhibits)}")
    if isinstance(spec, PadSpec):
        return (f"pad width {spec.width} padding {spec.padding.value} "
                f"pattern {_pattern(spec.pattern)} handle {_handle(spec.exhibits)}")
    if isinstance(spec, OuterSpec):
        return (f"outer boundary {spec.boundary_bits} padding {spec.padding.value} "
                f"pattern {_pattern(spec.pattern)}")
    if isinstance(spec, ConcatenationSpec):
        group = spec.concatenation
        return (f"concatenation order {'/'.join(spec.transmission_order())} "
                f"replace {_replacement(spec.replacement)} "
                f"group {'-' if group is None else group.order.value}:"
                f"{'-' if group is None else group.alignment.value}:"
                f"{'-' if group is None else group.handle_id} "
                f"handle {_handle(spec.exhibits)}")
    raise Asn1Error(f"ECN: {type(spec).__name__} has no canonical serialization")


def _name_of_condition(entry: ConditionalIntSpec) -> str:
    return _describe(entry).replace(" ", "~")


def _bound(value: int | None) -> str:
    return "-" if value is None else str(value)


def _ints(values) -> str:
    return "-" if values is None else ("/".join(str(v) for v in values) or "-")


def _bits_list(values) -> str:
    return "/".join("".join(str(bit) for bit in bits) for bits in values) or "-"


def _ref(pointer) -> str:
    if pointer is None:
        return "-"
    return (f"{pointer.reference}:{pointer.unit}:"
            f"{_chain(pointer.encoder_transforms)}")


def _det(determinant) -> str:
    if determinant is None:
        return "-"
    return (f"{determinant.determination.value}:{determinant.reference}:{determinant.unit}:"
            f"{_chain(determinant.encoder_transforms)}:"
            f"{_chain(determinant.decoder_transforms)}")


def _replacement(replacement) -> str:
    if replacement is None:
        return "-"
    structure = replacement.structure
    head = replacement.head_end
    return (f"{replacement.action.value}:{structure.name}:"
            f"{'/'.join(structure.order)}:{structure.dummy}:{_det(structure.determinant)}:"
            f"{'-' if head is None else head.name + '/' + '/'.join(head.order)}")


def _chain(chain) -> str:
    if chain is None:
        return "-"
    if isinstance(chain, TransformChain):
        return "/".join(_describe(step) for step in chain.transforms) or "-"
    return _describe(chain)


def _pre(pre_alignment: PreAlignment | None) -> str:
    if pre_alignment is None:
        return "-"
    return (f"{pre_alignment.unit}:{pre_alignment.padding.value}:"
            f"{_pattern(pre_alignment.pattern)}:"
            f"{'any' if pre_alignment.encoder_chosen_offset else 'next'}")


def _pad(padding: ValuePadding | None) -> str:
    if padding is None:
        return "-"
    unused = padding.unused_bits
    tail = "-" if unused is None else (
        f"{unused.determination.value}:{unused.reference}:"
        f"{_chain(unused.encoder_transforms)}:{_chain(unused.decoder_transforms)}")
    return (f"{padding.justification.side.value}:{padding.justification.offset}:"
            f"{padding.pre_padding.value}:{_pattern(padding.pre_pattern)}:"
            f"{padding.post_padding.value}:{_pattern(padding.post_pattern)}:{tail}")


def _handle(handle: "IdentificationHandle | None") -> str:
    """§22.9's group, in the digest. A handle changes what a decoder reads, so two modules
    differing only in one are two specifications and have to hash differently."""
    if handle is None:
        return "-"
    positions = "/".join(str(position) for position in handle.ordered())
    return f"{handle.name}@{positions}={handle.value_set.describe()}"


def _pattern(pattern: Pattern | None) -> str:
    if pattern is None:
        return "-"
    if pattern.kind.value in ("any-of-length", "different"):
        return f"{pattern.kind.value}:{pattern.length}"
    return f"{pattern.kind.value}:" + ("".join(str(bit) for bit in pattern.bits) or "-")


# --- parsing a module -----------------------------------------------------------------------

def parse_module(source: str) -> EcnModule:
    """Read an `ENCODING-DEFINITIONS` module.

    The wrapper is X.692 §14's::

        <ModuleName> ENCODING-DEFINITIONS ::= BEGIN <assignments> END

    and the assignments are the three kinds this rail reads: a class assignment
    (`#Name ::= #BASE`), a §16.5 concatenation structure (`Name ::= #CONCATENATION {...}`),
    and an encoding object assignment (`name #Class ::= { defined syntax }`).
    """
    tokens = tokenize(source)
    cursor = _Cursor(tokens, "the encoding definition module")
    module = EcnModule(name=cursor.next().text)
    cursor.expect("ENCODING-DEFINITIONS")
    cursor.expect("::=")
    cursor.expect("BEGIN")
    while True:
        if cursor.eof():
            raise Asn1Error("ECN: the module has no END")
        if cursor.accept("END"):
            break
        _parse_assignment(cursor, module)
    if not cursor.eof():
        raise Asn1Error(f"ECN: {cursor.peek()!r} follows the module's END")
    return module


def _collect_braced(cursor: _Cursor) -> list[Token]:
    """The tokens between a `{` and its matching `}`, exclusive."""
    cursor.expect("{")
    depth = 1
    body: list[Token] = []
    while True:
        token = cursor.next()
        if token.text == "{":
            depth += 1
        elif token.text == "}":
            depth -= 1
            if depth == 0:
                return body
        body.append(token)


def _parse_parameterized(cursor: _Cursor, module: EcnModule, name: str) -> None:
    """C.2's three parameterized assignments, told apart by what follows the parameter list.

    C.2 adds them to X.683 §8.2 as `ParameterizedEncodingClassAssignment`,
    `ParameterizedEncodingObjectAssignment` and `ParameterizedEncodingObjectSetAssignment` —
    three, not X.683's own set, because ECN assigns classes, objects and object sets and leaves
    types to ASN.1.

    The object form is the one with a governor between the parameter list and the `::=`, and
    that governor may itself be a parameterized reference — `#New-component{<#Any-class>}` —
    using a dummy declared to its left. C.2's modification to X.683 §8.4 is what permits that,
    and it permits it for this form alone; `ParameterizedAssignment` refuses the others.

    The body is collected and NOT parsed. §9.7 makes instantiation a substitution, so a body
    read against dummies would have to invent an encoding for each one, and any encoding it
    invented would be wrong for some instantiation.
    """
    parameters = _parse_parameter_list(cursor)
    if name in module.parameterized:
        raise Asn1Error(f"ECN: {name} is assigned twice in this module")

    if cursor.accept("::="):
        # No governor: a class assignment (`#Length-prefixed{<#D>} ::= #CONCATENATION {...}`)
        # or an object-set one, told apart by whether the name is a class reference.
        kind = (AssignmentKind.ENCODING_CLASS if name.startswith("#")
                else AssignmentKind.ENCODING_OBJECT_SET)
        # §16.2.12's `EncodingStructureDefn` is a class followed by a braced field list, so a
        # class assignment's body is *two* things and taking only the first would leave the
        # braces to be read as the next assignment. An object set's body is the braces alone.
        body: list[Token] = []
        if cursor.peek() != "{":
            body.append(cursor.next())
        if cursor.peek() == "{":
            body.append(Token("{", 0))
            body.extend(_collect_braced(cursor))
            body.append(Token("}", 0))
        if not body:
            raise Asn1Error(f"ECN: the parameterized assignment {name} has no body")
        module.parameterized[name] = ParameterizedAssignment(
            name=name, kind=kind, parameters=parameters,
            body=tuple(token.text for token in body))
        return

    governor = cursor.next().text
    governor_actuals = None
    if cursor.peek() == "{<":
        governor_actuals = _parse_actual_parameter_list(cursor)
    cursor.expect("::=")
    body = _collect_braced(cursor)
    module.parameterized[name] = ParameterizedAssignment(
        name=name, kind=AssignmentKind.ENCODING_OBJECT, parameters=parameters,
        governor=governor, governor_actuals=governor_actuals,
        body=tuple(token.text for token in body))
    _check_replacement_pair(module, name)


def _check_replacement_pair(module: EcnModule, object_name: str) -> None:
    """§22.1.2.2 and §22.1.2.4, applied when an object's governor is a parameterized structure.

    §22.1.2.4 makes the `ENCODED BY` object's governor "the corresponding `WITH` parameterized
    encoding structure, instantiated with `#D`", so an object whose governor names a
    parameterized structure in this module *is* a candidate replacement pair — and the two
    clauses' restrictions can be checked the moment both halves are present, rather than
    waiting for a `REPLACE` to name them.

    Checking early is the point. A module may define a replacement pair and apply it from an
    ELM that this rail never reads, and a pair that could never be instantiated is invalid on
    its own terms whether or not anything here instantiates it.
    """
    assignment = module.parameterized[object_name]
    structure = module.parameterized.get(assignment.governor)
    if structure is None or structure.kind is not AssignmentKind.ENCODING_CLASS:
        return
    ReplacementParameterization(
        structure=structure.parameters,
        encoded_by=assignment.parameters,
        governor_actuals=assignment.governor_actuals,
        # §22.1.2.5's biconditional is read off the object's own dummies: a REFERENCE
        # parameter is present exactly when the group has an INSERT AT HEAD, so the presence
        # of the dummy is what says the pair expects one.
        insert_at_head=ParameterKind.IDENTIFIER in assignment.parameters.kinds())


def _parse_assignment(cursor: _Cursor, module: EcnModule) -> None:
    head = cursor.next()
    name = head.text

    if cursor.peek() == "{<":
        _parse_parameterized(cursor, module, name)
        return

    if name.startswith("#"):
        # Clause 11's class assignment: `#My-int ::= #INT`, or a §16.5 structure written
        # against a built-in class directly.
        cursor.expect("::=")
        base = cursor.next().text
        if cursor.peek() == "{":
            _parse_structure(cursor, module, name, base)
            return
        if name in module.classes or name in _BUILTIN_CLASSES:
            raise Asn1Error(f"ECN: {name} is assigned twice in this module")
        module.classes[name] = base
        module.builtin_of(name)  # Resolve now, so a bad base is reported where it is written.
        return

    if cursor.peek() == "::=":
        # `Name ::= #CONCATENATION { ... }` — a §16.5 structure with an ordinary type name.
        cursor.expect("::=")
        base = cursor.next().text
        _parse_structure(cursor, module, name, base)
        return

    class_name = cursor.next().text
    cursor.expect("::=")
    body = _collect_braced(cursor)
    inner = _Cursor(body, f"the encoding object {name}")
    builtin = module.builtin_of(class_name)

    if name in module.objects or name in module.transforms:
        raise Asn1Error(f"ECN: {name} is assigned twice in this module")

    if builtin == "transform":
        module.transforms[name] = _parse_transform_body(inner, name)
        return

    if builtin == "conditional-int":
        spec = _parse_conditional_int_body(inner, name, module)
    elif builtin == "int":
        spec = _parse_int_body(inner, name, module)
    elif builtin == "bool":
        spec = _parse_bool_body(inner, name, module)
    elif builtin == "pad":
        spec = _parse_pad_body(inner, name, module)
    elif builtin == "outer":
        spec = _parse_outer_body(inner, name)
    else:
        spec = _parse_concatenation_body(inner, name, module)

    # §9.5.2 is NOT checked here, and the clause is why: the rule is about "encoding object
    # SET construction", and a module is not a set. A `#CONDITIONAL-INT` object is reached by
    # name from a `#INT` object (§23.6.1) and never enters the set, so a module may hold as
    # many as its integer fields need. The law is enforced in `spec_for_class`, where the set
    # is actually formed from the structure's classes.
    module.objects[name] = (class_name, spec)


def _parse_structure(cursor: _Cursor, module: EcnModule, name: str, base: str) -> None:
    """§16.5's `ConcatenationStructure ::= ConcatenationClass "{" NamedFields "}"`.

    §16.3.1's `NamedField ::= identifier EncodingStructure` is the shape read here, with the
    `EncodingStructure` restricted to a `DefinedEncodingClass` — a field that is itself a
    nested structure is legal ECN and is refused rather than flattened.
    """
    if module.builtin_of(base) != "concatenation":
        raise Asn1Error(
            f"ECN: §16.5 builds a ConcatenationStructure from a class in the concatenation "
            f"category; {base} is not one")
    if module.structure:
        raise Asn1Error(
            f"ECN: this module already declares the structure {module.structure_name}; "
            f"one application point is what §13.2 walks")
    body = _collect_braced(cursor)
    inner = _Cursor(body, f"the encoding structure {name}")
    fields: list[tuple[str, str]] = []
    while not inner.eof():
        if inner.accept(","):
            continue
        field_name = inner.next().text
        if inner.eof() or inner.peek() == ",":
            raise Asn1Error(
                f"ECN: §16.3.1's NamedField is an identifier followed by an EncodingStructure; "
                f"{field_name!r} has no class")
        class_name = inner.next().text
        if inner.peek() == "{":
            raise Asn1Error(
                f"ECN: {field_name!r} is a nested EncodingStructure. §16.2.1 admits them and "
                f"this rail walks one flat concatenation, so a nested structure is refused "
                f"rather than flattened into its parent's field order")
        module.builtin_of(class_name)
        fields.append((field_name, class_name))
    if not fields:
        raise Asn1Error(f"ECN: the structure {name} has no fields")
    seen: set[str] = set()
    for field_name, _class in fields:
        if field_name in seen:
            raise Asn1Error(f"ECN: the structure {name} names {field_name!r} twice")
        seen.add(field_name)
    module.structure = tuple(fields)
    module.structure_name = name


#: The gate's workload as an encoding definition module, shipped as source next to the ASN.1
#: modules for the same reason those are: it is what a test compiles, not documentation about
#: one. See `pyproject.toml`'s package-data note.
FRAME_HEADER_MODULE = "BCIR-FrameHeader.ecn"


def frame_header_source() -> str:
    """The text of `BCIR-FrameHeader.ecn`.

    Read through `importlib.resources` rather than by path arithmetic so it works from an
    installed wheel, where `__file__` may sit inside a zip and `Path(__file__).parent` is not
    a directory anything can be opened from.
    """
    from importlib import resources

    return (resources.files(__package__) / FRAME_HEADER_MODULE).read_text(encoding="utf-8")


def frame_header_module() -> EcnModule:
    """`BCIR-FrameHeader.ecn`, parsed."""
    return parse_module(frame_header_source())


__all__ = [
    "FRAME_HEADER_MODULE", "SYNTAX_COMPILER", "SYNTAX_VERSION", "EcnModule", "Token",
    "frame_header_module", "frame_header_source", "parse_module", "tokenize",
]
