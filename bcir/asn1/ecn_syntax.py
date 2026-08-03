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

from .ecn_user import (
    UNIT_BIT, UNIT_NAMES, AuxIntSpec, BoolSpec, Comparison, ConcatenationSpec,
    ConditionalIntSpec, EncodingSpaceDetermination, HeadEndStructure, IntForm, IntOp,
    IntSelector, IntSpec, IntToBits, IntToInt, IntegerBounds, Justification, OuterSpec,
    PadSpec, Padding, Pattern, PreAlignment, RangeCondition, ReplaceAction, Replacement,
    ReplacementStructure, ReversalSpecification, SpaceDeterminant, StartPointer,
    TransformChain, UnusedBits, UnusedBitsDetermination, UserEncodingObject, ValuePadding,
    check_unit,
)
from .tags import Asn1Error

#: The serialization's version. Its own counter, not `encode_plan`'s: see this module's
#: `serialize` for why an ECN encoding is a separate compilation rather than a plan version.
SYNTAX_VERSION = 2
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
    "EXHIBITS": "§22.9's identification handles are read only by ORDER random and by the "
                "alternatives category, neither of which is built",
    "CONTAINED": "§22.11's contained-type specification is not implemented",
    # §22.1's replacement SEMANTICS are built in `ecn_user` and reachable from Python. What
    # this grammar cannot read is the notation around them: §22.1.2.2 makes the `WITH`
    # structures parameterized encoding structures with a single encoding class parameter, and
    # §22.1.2.4 makes the `ENCODED BY` objects parameterized encoding objects whose governor is
    # that structure instantiated with the dummy. That is X.683's parameterization applied to
    # ECN, and `#Length-prefixed{#D} ::= ...` is not a shape this parser reads.
    "REPLACE": "§22.1.2.2 and §22.1.2.4 build a replacement from a PARAMETERIZED encoding "
               "structure and a PARAMETERIZED encoding object (X.683 applied to ECN), which "
               "this grammar does not read; the replacement semantics themselves are built, "
               "so assemble `Replacement` in Python",
}


# --- lexing -------------------------------------------------------------------------------

#: Characters that are a token on their own. Everything else accumulates into a word, so
#: `payloadOctets  #INT,` lexes as three tokens without the source needing spaces around the
#: comma. `:` is NOT here: §21.8's `left:2` and §24.3.1's `divide:4` are single tokens, since
#: the CHOICE alternative and its value are one value notation.
_PUNCTUATION = "{},"


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
        encoded_as = "twos-complement"  # §24.8.1's DEFAULT.
        size = 0
        if cursor.accept("AS"):
            encoded_as = cursor.next().text
        if cursor.accept("SIZE"):
            token = cursor.next()
            try:
                size = int(token.text)
            except ValueError:
                raise Asn1Error(
                    f"ECN: §24.8's ResultSize is `variable` or a fixed count; {token} is "
                    f"neither, and `variable` has no width for this rail to write") from None
        unit = 1
        if cursor.accept_words("MULTIPLE", "OF"):
            unit = _parse_unit(cursor)
        transform = IntToBits(name=name, width=size * unit, encoded_as=encoded_as)
    else:
        token = cursor.peek()
        raise Asn1Error(
            f"ECN: {token!r} starts no #TRANSFORM clause this rail implements; §24.1.1 "
            f"defines seventeen and INT-TO-INT (§24.3) and INT-TO-BITS (§24.8) are built")
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
        space_determinant=common.space.determinant,
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
    reversal = _parse_reversal_tail(cursor)
    _finish(cursor, name)
    # §23.7.2.7's `subtract:lower-bound` rule relates the transforms to the CONDITION, so it
    # is checked by ConditionalIntSpec where both are in hand rather than here.
    return ConditionalIntSpec(
        spec=IntSpec(width=common.space.width, form=form, transform=chain,
                     pre_alignment=common.pre_alignment, value_padding=padding,
                     start_pointer=common.start_pointer,
                     space_determinant=common.space.determinant,
                     bit_reversal=reversal, reversal_unit=common.space.unit),
        conditions=conditions)


def _parse_int_body(cursor: _Cursor, name: str, module: "EcnModule"):
    """§23.6.1's `#INT`: `[ENCODINGS &Integer-encodings] [ENCODING &integer-encoding]`.

    §23.6.2.2: "Exactly one of ENCODING and ENCODINGS shall be set." Both name
    `#CONDITIONAL-INT` objects, and §23.6.3.1 selects "the first ... whose conditions are
    satisfied" against the bounds of the type the object set is applied to.

    `BOUNDS` is this rail's stand-in for those bounds, and it is a **deviation stated rather
    than hidden**. In X.692 the bounds arrive from the ASN.1 type through an encoding link
    module (clause 14), which this rail does not have — a value dict is what it links against.
    Writing them on the `#INT` object keeps §21.11's predicates testable against real numbers;
    it does not make them a property the notation gives that object.
    """
    if cursor.accept("AUXILIARY"):
        # A DEVIATION, stated. X.692 has no keyword for "this field is auxiliary" because
        # §22.1.2.6's classification comes from the encoding link module: a structure field
        # with no ASN.1 component behind it is auxiliary, and clause 14's ELM is what decides
        # that. This rail links against a value dict rather than an ASN.1 type, so the fact
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
    _finish(cursor, name)
    return PadSpec(width=common.space.width, padding=padding, pattern=pattern)  # noqa: E501


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
        return f"int-to-bits {spec.encoded_as} {spec.width}"
    if isinstance(spec, IntSpec):
        return (f"int width {spec.width} form {spec.form.value} "
                f"transform {_chain(spec.transform)} "
                f"pre {_pre(spec.pre_alignment)} pad {_pad(spec.value_padding)} "
                f"ptr {_ref(spec.start_pointer)} det {_det(spec.space_determinant)} "
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
                f"pre {_pre(spec.pre_alignment)}")
    if isinstance(spec, PadSpec):
        return (f"pad width {spec.width} padding {spec.padding.value} "
                f"pattern {_pattern(spec.pattern)}")
    if isinstance(spec, OuterSpec):
        return (f"outer boundary {spec.boundary_bits} padding {spec.padding.value} "
                f"pattern {_pattern(spec.pattern)}")
    if isinstance(spec, ConcatenationSpec):
        return (f"concatenation order {'/'.join(spec.transmission_order())} "
                f"replace {_replacement(spec.replacement)}")
    raise Asn1Error(f"ECN: {type(spec).__name__} has no canonical serialization")


def _name_of_condition(entry: ConditionalIntSpec) -> str:
    return _describe(entry).replace(" ", "~")


def _bound(value: int | None) -> str:
    return "-" if value is None else str(value)


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


def _parse_assignment(cursor: _Cursor, module: EcnModule) -> None:
    head = cursor.next()
    name = head.text

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
