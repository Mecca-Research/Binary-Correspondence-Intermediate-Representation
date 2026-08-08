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

from .ecn_props import (
    UNIT_BIT, UNIT_DWORD32, UNIT_MAX, UNIT_NAMES, UNIT_NIBBLE, UNIT_OCTET,
    UNIT_REPETITIONS, UNIT_WORD16, AlternativeDetermination, Comparison, ComponentOrder,
    ConcatenationAlignment, HandleValueKind, HandleValueSet, IntForm, IntegerBounds,
    Justification, JustificationSide, OptionalityDetermination, Padding, Pattern, PatternKind,
    RangeCondition, RepetitionSpaceDetermination, ReversalSpecification, SizeBounds,
    SizeRangeCondition, _padding_bits, check_unit,
)
from .ecn_transform import (
    RESULT_SIZE_FIXED_TO_MAX, RESULT_SIZE_VARIABLE, BitsToBits, BitsToChar,
    BitsToCompositeBits, BitsToInt, BitToBits, BoolToBool, BoolToInt, CharsToCompositeChar,
    CharToBits, CompositeBitsToBits, CompositeBitsToOctets, CompositeCharToChars,
    Composite, IntOp, IntToBits, IntToBool, IntToChars, IntToInt, OctetsToCompositeBits,
    Transform, TransformChain,
)
from .tags import Asn1Error

# Clause 21's property types and clause 24's transforms are re-exported here because that is
# where every caller already expects them. The split is a file-layout decision -- clause 24 alone is
# nineteen transforms -- and not a change to the public surface.


#: The name a `container` determination's REFERENCE takes when the container is the PDU.
#:
#: §21.3.6, §21.5.6 and §21.7.8 each offer two forms — "a REFERENCE to another field whose
#: encoding class (the container) has a length determinant", "or of a specification that the
#: end of the PDU determines the end ... (**using `OUTER`**)". The notation does **not** give
#: the second one a keyword: §22.4.1.2's syntax is `USING &encoding-space-reference` in every
#: case, and §22.4.1.6 says that reference is "to an auxiliary field or to a field carrying
#: abstract values, **or to a container**, depending on the value of `DETERMINED BY`". So the
#: PDU is simply the outermost container and clause 25's `#OUTER` is what names it — which is
#: why this is a reserved reference rather than a flag on the group, and why the surface
#: grammar needed no new keyword to read it.
OUTER_CONTAINER = "#OUTER"


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
        if not self.reference:
            raise Asn1Error(
                "ECN: §21.3.4/§21.3.5/§21.3.6 — every determination requires a REFERENCE; "
                "§22.4.1.6 says it is one \"to an auxiliary field or to a field carrying "
                "abstract values, or to a container, depending on the value of DETERMINED BY\"")
        if (self.determination is EncodingSpaceDetermination.CONTAINER
                and (self.encoder_transforms is not None
                     or self.decoder_transforms is not None)):
            # §22.4.2.3 and §22.4.2.4 confine the transform lists to the two field
            # determinations, and the reason is structural rather than a restriction: a
            # container's end is a position, not a number carried through a field, so there is
            # nothing for a transform to convert.
            raise Asn1Error(
                "ECN: §22.4.2.3/§22.4.2.4 — a `container` determination reads no field's "
                "value, so there is nothing for ENCODER-TRANSFORMS or DECODER-TRANSFORMS to "
                "convert; §21.3.4 and §21.3.5 are the determinations that carry a length")
        check_unit(self.unit, allow_repetitions=False)

    def record(self, out: "BitWriter", space_bits: int) -> None:
        """§21.3.4's set, or §21.3.5's use, given the space this field actually took."""
        if self.determination is EncodingSpaceDetermination.CONTAINER:
            # §21.3.6 gives the encoder nothing to write: "The encoding space terminates when
            # the specified container terminates or when the end of the PDU is encountered."
            # What it does give the encoder is a rule to CHECK — "This specification can only
            # be used if the encoding space of the element being encoded is the last encoding
            # to be placed in the container" — and that is what this registers. The check
            # itself fires when the container closes, because until then there is no end to
            # compare against.
            out.claim_container_end(self.reference, "the ENCODING-SPACE of this element")
            return
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


# --- clause 22.9: identification handles, and the three determinations that read them ------

@dataclass(frozen=True)
class IdentificationHandle:
    """§22.9's `EXHIBITS HANDLE ... AT ... AS ...`: bits an encoding announces itself by.

    §22.9.1.4 gives the three parts: "the name of the handle", "the bit positions that form
    the handle", and "the possible bit patterns ... occurring in the encodings produced by this
    encoding object (the handle value set)".

    **This is the mechanism ECN offers instead of a discriminant field.** Everything else in
    clause 22 that resolves a choice — optionality, alternatives, repetition end, random
    concatenation order — can also be resolved by *looking at the bits that are there anyway*,
    which is how BER's tag, IP's version nibble and a hundred other formats actually work. A
    handle is a declaration that a particular window into this object's own encoding is a
    reliable discriminator, plus a statement of what it can hold.

    Two subtleties in the positions, both of which change the answer:

    * §22.9.1.5 measures them "in the final encoding, **after any pre-alignment has been
      applied**, and after any encoder bit-reversal actions have occurred, except those
      bit-reversals that result from ... `#OUTER`". So position zero is where this object's
      encoding *space* starts, not where its pre-alignment padding starts, and a §22.12
      reversal has already moved the bits by the time the handle is read.
    * §22.9.1.6 makes them "a set of integer values (not necessarily contiguous, and not
      necessarily in ascending order in the ECN specification)" which "shall be ordered by
      encoders and decoders from the zero position ... upwards". A handle can therefore be
      bits 0, 3 and 7 written in any order, and the conceptual field is always their
      ascending-order concatenation.

    §22.9.1.7 then reads that field as an integer with "the bit ... nearest to the zero
    position" as the high-order bit.
    """

    name: str = "default-handle"
    #: §22.9.1.1's `&Handle-positions INTEGER (0..MAX) OPTIONAL`, in the order written.
    positions: tuple[int, ...] = ()
    value_set: HandleValueSet = field(default_factory=HandleValueSet.tag_any)

    def __post_init__(self) -> None:
        if not self.name:
            raise Asn1Error(
                "ECN: §22.9.1.1 gives &exhibited-handle a DEFAULT of \"default-handle\"; a "
                "handle with an empty name cannot be the one §22.10.2.1 names by default")
        if not self.positions:
            raise Asn1Error(
                "ECN: §22.9.1.2's syntax makes `AT` part of `EXHIBITS HANDLE` rather than an "
                "optional tail; a handle with no positions is a zero-bit field that no value "
                "set could characterize")
        for position in self.positions:
            if position < 0:
                raise Asn1Error(
                    f"ECN: §22.9.1.1 constrains &Handle-positions to INTEGER (0..MAX); "
                    f"got {position}")
        if len(set(self.positions)) != len(self.positions):
            raise Asn1Error(
                f"ECN: §22.9.1.6 calls the positions \"a set of integer values\"; "
                f"{sorted(self.positions)} repeats one, and a bit cannot appear twice in the "
                f"conceptual handle field")

    @property
    def width(self) -> int:
        """The conceptual handle field's width — the number of positions, not their span."""
        return len(self.positions)

    def ordered(self) -> tuple[int, ...]:
        """§22.9.1.6's ordering: "from the zero position ... upwards"."""
        return tuple(sorted(self.positions))

    def resolved_for(self, value_set: "HandleValueSet | None") -> "IdentificationHandle":
        """This handle with `tag:any` replaced by a concrete set, for §22.9.1.9's `#TAG` case."""
        if value_set is None or value_set is self.value_set:
            return self
        return replace(self, value_set=value_set)

    def value_in(self, out: "BitWriter", start: int) -> int:
        """The conceptual handle field's value, read out of what has just been written."""
        value = 0
        for position in self.ordered():
            index = start + position
            if index >= out.bit_length:
                raise Asn1Error(
                    f"ECN: the handle {self.name!r} names bit {position} of an encoding that "
                    f"is only {out.bit_length - start} bits long; §22.9.1.5 counts positions "
                    f"in the final encoding of the object exhibiting the handle")
            value = (value << 1) | out.bit_at(index)
        return value

    def check(self, out: "BitWriter", start: int,
              value_set: "HandleValueSet | None" = None) -> int:
        """§22.9.3.1, the one encoder action a handle has.

        "The encoder shall check that the value of the identification handle occurring in the
        encoding produced is a member of the specified handle value set, and shall diagnose a
        specification or application error otherwise." Not a hint and not a decoder-only
        concern: an object whose encodings can leave its declared set has made every
        determination that reads the handle unsound, and §22.9.2.2 says so from the other side.
        """
        effective = value_set if value_set is not None else self.value_set
        value = self.value_in(out, start)
        if not effective.contains(value, self.width):
            raise Asn1Error(
                f"ECN: §22.9.3.1 — this encoding puts {value} in the handle {self.name!r}, "
                f"which declares {effective.describe()}; an encoder shall diagnose a value "
                f"outside its own handle value set rather than transmit it")
        return value


def _exhibited(spec) -> "IdentificationHandle | None":
    """The handle an arbitrary spec exhibits, with a `#TAG`'s `tag:any` already resolved.

    §22.9.1.9 and §21.16.5 make `tag:any` mean "whatever this tag class's number is", so a
    `#TAG` object's declared set is not usable until the number is folded in. Every caller
    that compares handles across objects — the disjointness rules — needs the resolved form,
    so resolution happens here rather than at each of the four call sites.
    """
    handle = getattr(spec, "exhibits", None)
    if handle is None:
        return None
    if isinstance(spec, TagSpec):
        return handle.resolved_for(handle.value_set.resolve_tag(spec.number))
    if handle.value_set.kind is HandleValueKind.TAG_ANY:
        raise Asn1Error(
            f"ECN: §22.9.1.9 — the handle value set shall not be specified as `tag:any` "
            f"unless the specification is for an encoding object of the #TAG class; "
            f"{type(spec).__name__} is not one")
    return handle


def _component_tag(spec) -> "int | None":
    """The component-tag §22.6.2.10 and §22.10.2.4 both name, or `None` if there is not one.

    Both clauses word it the same way — the component "shall start with an encoding class in
    the tag category. The tag number associated with this class is called the component-tag" —
    so one function answers for alternatives and for concatenations alike. An `#OPTIONAL`
    wrapper is transparent to it: §23.11.3.2 says the replacement of an optional component
    covers "the entire component (including any classes in the tag category, but excluding
    classes in the optionality category)", which is the same statement that the tag is inside
    the optionality wrapper rather than outside it.
    """
    if isinstance(spec, TagSpec):
        return spec.number
    if isinstance(spec, OptionalSpec) and spec.component is not None:
        return _component_tag(spec.component)
    return None


def _exhibit(spec, out: "BitWriter", start: int) -> None:
    """Run §22.9.3.1's check for whatever handle `spec` exhibits, if it exhibits one.

    Routed through `_exhibited` rather than reading `spec.exhibits` directly so that a `#TAG`
    object's `tag:any` is resolved and every other category's is refused with §22.9.1.9's own
    words, instead of failing later as "this set has no range".
    """
    handle = _exhibited(spec)
    if handle is not None:
        handle.check(out, start)


class HandleRegistry:
    """§22.9.2.1 and §22.9.2.3: the two rules that are about a *specification*, not an object.

    Neither can be checked where a handle is written, because both relate one `EXHIBITS
    HANDLE` clause to every other one sharing its name. §22.9.2.1: "In any ECN specification,
    all identification handles with the same name shall specify the same set of bit
    positions." §22.9.2.3: "All encoding objects that exhibit the same identification handle
    shall either have no pre-alignment specification, or shall align to the same pre-alignment
    unit", whose NOTE gives the reason — "so that decoders can move to the alignment position
    before looking for the handle".

    **"No pre-alignment specification" and "align to bit" are treated as one case here**, and
    that is a reading rather than a quotation: §22.2.1.1's default unit is `bit`, and aligning
    to a one-bit boundary inserts nothing, so the two are the same operation written two ways.
    Refusing their mixture would reject specifications that differ only in whether they spelled
    a default out.
    """

    def __init__(self) -> None:
        self._positions: dict[str, tuple[int, ...]] = {}
        self._alignment: dict[str, int] = {}
        self._sets: dict[str, list[tuple[str, HandleValueSet]]] = {}

    def declare(self, handle: IdentificationHandle, *, where: str,
                alignment_unit: "int | None" = None) -> None:
        unit = UNIT_BIT if alignment_unit is None else alignment_unit
        positions = handle.ordered()
        seen = self._positions.get(handle.name)
        if seen is not None and seen != positions:
            raise Asn1Error(
                f"ECN: §22.9.2.1 — all identification handles named {handle.name!r} shall "
                f"specify the same set of bit positions; {where} says {list(positions)} where "
                f"an earlier one said {list(seen)}")
        aligned = self._alignment.get(handle.name)
        if aligned is not None and aligned != unit:
            raise Asn1Error(
                f"ECN: §22.9.2.3 — objects exhibiting {handle.name!r} shall align to the same "
                f"pre-alignment unit so a decoder can reach the handle; {where} aligns to "
                f"{unit} where an earlier one aligned to {aligned}")
        self._positions[handle.name] = positions
        self._alignment[handle.name] = unit
        self._sets.setdefault(handle.name, []).append((where, handle.value_set))

    def require_disjoint(self, name: str, clause: str) -> None:
        """§21.5.7 / §21.6.6 / §21.7.10 / §22.10.2.1's shared "shall all be disjoint"."""
        entries = self._sets.get(name, ())
        width = len(self._positions.get(name, ()))
        for index, (where, value_set) in enumerate(entries):
            for other_where, other_set in entries[index + 1:]:
                if not value_set.disjoint_from(other_set, width):
                    raise Asn1Error(
                        f"ECN: {clause} — the handle value sets of the objects exhibiting "
                        f"{name!r} shall all be disjoint; {where} declares "
                        f"{value_set.describe()} and {other_where} declares "
                        f"{other_set.describe()}, which overlap")


def _handles_of(named_specs, *, handle_id: str, clause: str, what: str,
                alignment_unit: "int | None" = None) -> HandleRegistry:
    """Collect and validate the handles a handle-driven determination depends on.

    One helper for four clauses because they ask for exactly the same three things: every
    participant exhibits the named handle, they agree about its positions, and their value
    sets are disjoint. What differs between §21.5.7, §21.6.6, §21.7.10 and §22.10.2.1 is only
    *which* objects participate, which is the caller's business.
    """
    registry = HandleRegistry()
    for name, spec in named_specs:
        handle = _exhibited(spec)
        if handle is None or handle.name != handle_id:
            exhibits = "none" if handle is None else repr(handle.name)
            raise Asn1Error(
                f"ECN: {clause} — {what} determined by the handle {handle_id!r} requires every "
                f"participant to exhibit it; {name!r} exhibits {exhibits}")
        registry.declare(handle, where=repr(name), alignment_unit=alignment_unit)
    registry.require_disjoint(handle_id, clause)
    return registry


@dataclass(frozen=True)
class Optionality:
    """§22.5's `PRESENCE` group: how a decoder learns whether an optional component is there.

    §22.5.1.6 is unusual and is enforced: this specification "is considered set if the
    `PRESENCE` keyword is used, and **it is mandatory for it to be set** in all places in the
    defined syntax where it is allowed. Defaulting all other parts of this defined syntax
    (e.g., use of `PRESENCE` alone) would not satisfy the above constraints." So an
    `#OPTIONAL` object without this group is not an object taking defaults — it is incomplete.

    The five determinations divide by *who owns the fact*. `field-to-be-set` has the encoder
    write a presence bit; `field-to-be-used` has the application supply one and the encoder
    check it (§22.5.3.4: "It is an application error if this condition is not met, and encoding
    shall not proceed"); `container` says absence is the container running out; `handle` says
    absence is recognizing what comes next; `pointer` says a start pointer of zero means
    absent (§21.5.9).
    """

    determination: OptionalityDetermination = OptionalityDetermination.FIELD_TO_BE_SET
    #: §22.5.1.1's `&optionality-reference REFERENCE OPTIONAL`.
    reference: str = ""
    encoder_transforms: "TransformChain | None" = None
    decoder_transforms: "TransformChain | None" = None
    handle_id: str = "default-handle"
    #: Whether `HANDLE` was written, which §22.5.2.2 constrains independently of its value —
    #: the property has a DEFAULT, so "absent" and "set to the default" are different facts.
    handle_set: bool = False

    def __post_init__(self) -> None:
        by_handle = self.determination is OptionalityDetermination.HANDLE
        if self.handle_set and not by_handle:
            raise Asn1Error(
                f"ECN: §22.5.2.2 — HANDLE shall not be specified unless DETERMINED BY is "
                f"`handle`; this object says `{self.determination.value}`")
        uses_field = self.determination in (OptionalityDetermination.FIELD_TO_BE_SET,
                                            OptionalityDetermination.FIELD_TO_BE_USED,
                                            OptionalityDetermination.CONTAINER)
        if self.reference and not uses_field:
            raise Asn1Error(
                f"ECN: §22.5.2.3 — USING shall not be specified if DETERMINED BY is `handle` "
                f"or `pointer`; this object says `{self.determination.value}`")
        if uses_field and not self.reference:
            raise Asn1Error(
                f"ECN: §21.5.4/§21.5.5/§21.5.6 — `{self.determination.value}` requires a USING "
                f"reference to the field that carries the presence information")
        if (self.encoder_transforms is not None
                and self.determination is not OptionalityDetermination.FIELD_TO_BE_SET):
            raise Asn1Error(
                "ECN: §22.5.2.6 — ENCODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is set to (or defaults to) `field-to-be-set`")
        if (self.decoder_transforms is not None
                and self.determination is not OptionalityDetermination.FIELD_TO_BE_USED):
            raise Asn1Error(
                "ECN: §22.5.2.8 — DECODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is set to `field-to-be-used`")

    def record(self, out: "BitWriter", present: bool) -> None:
        """§22.5.3.2–§22.5.3.7, given the `element-is-present` the application implied.

        The conceptual value is a **boolean** (§22.5.3.2), and §22.5.2.7 makes the first
        transform's source boolean to match. With no transforms the boolean itself is what
        goes in the field, and `int(present)` is the bridge to an auxiliary field that holds
        an integer — a `#BOOL` auxiliary field would encode the same one bit, so the two
        spellings agree on the octets.
        """
        if self.determination is OptionalityDetermination.FIELD_TO_BE_SET:
            carried = _determinant_value(self.encoder_transforms, present,
                                         f"PRESENCE USING {self.reference}")
            out.patch(self.reference, int(carried))
            return
        if self.determination is OptionalityDetermination.FIELD_TO_BE_USED:
            # §22.5.3.4: the encoder CHECKS. The application owns this field's value, and a
            # disagreement is its error rather than something to correct silently.
            carried = out.value_of(self.reference)
            recovered = (carried if self.decoder_transforms is None
                         else self.decoder_transforms.apply(carried))
            if bool(recovered) != present:
                raise Asn1Error(
                    f"ECN: §22.5.3.4 — the field {self.reference!r} carries {carried}, which "
                    f"reduces to `element-is-present` {bool(recovered)}, but the component is "
                    f"{'present' if present else 'absent'}")
            return
        if self.determination is OptionalityDetermination.CONTAINER and present:
            # §22.5.3.5 gives the encoder no value to write and one thing to *detect*: it must
            # "cease encoding if the application requests the encoding of further components
            # in the USING container when the conceptual value `element-is-present` is false".
            # §21.5.6's NOTE generalizes it — "no further encodings are to be placed in the
            # container" — so a present component determined this way still has to be the last
            # thing in it, or the absence of the next one is undetectable.
            out.claim_container_end(self.reference, "an optional component whose presence")
        # §22.5.3.6 (handle) and §22.5.3.7 (pointer) say there is no further encoder action.
        # The handle's bits and the start pointer's zero are already in the encoding.


@dataclass(frozen=True)
class AlternativeSelection:
    """§22.6's `ALTERNATIVE` group: how a decoder learns which alternative was encoded.

    §22.6.3.2's conceptual value is an integer, `alternative-index`, and §22.6.3.3 fixes it:
    "zero for the first alternative, one for the next, and so on, where the order of the
    alternatives is determined by `ORDER`". That indirection is the whole design — a two-bit
    selector field and a four-way CHOICE are related through an index neither of them spells,
    so the same `#ALTERNATIVES` object works over a differently-named CHOICE.

    §22.6.1.1 declares `&alternative-ordering ENUMERATED {textual, tag}` — **two values, not
    concatenation's three**. `random` would be meaningless here: a CHOICE encodes exactly one
    alternative, so there is no order to randomize.
    """

    determination: AlternativeDetermination = AlternativeDetermination.FIELD_TO_BE_SET
    reference: str = ""
    encoder_transforms: "TransformChain | None" = None
    decoder_transforms: "TransformChain | None" = None
    handle_id: str = "default-handle"
    handle_set: bool = False
    ordering: ComponentOrder = ComponentOrder.TEXTUAL

    def __post_init__(self) -> None:
        by_handle = self.determination is AlternativeDetermination.HANDLE
        if self.handle_set and not by_handle:
            raise Asn1Error(
                f"ECN: §22.6.2.2 — HANDLE shall not be specified unless DETERMINED BY is "
                f"`handle`; this object says `{self.determination.value}`")
        if by_handle and self.reference:
            raise Asn1Error(
                "ECN: §22.6.2.3 — USING shall not be specified if DETERMINED BY is `handle`")
        if not by_handle and not self.reference:
            raise Asn1Error(
                f"ECN: §21.6.4/§21.6.5 — `{self.determination.value}` requires a USING "
                f"reference to the field that carries the alternative's identity")
        if (self.encoder_transforms is not None
                and self.determination is not AlternativeDetermination.FIELD_TO_BE_SET):
            raise Asn1Error(
                "ECN: §22.6.2.5 — ENCODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is set to (or defaults to) `field-to-be-set`")
        if (self.decoder_transforms is not None
                and self.determination is not AlternativeDetermination.FIELD_TO_BE_USED):
            raise Asn1Error(
                "ECN: §22.6.2.7 — DECODER-TRANSFORMS shall be present only if DETERMINED BY "
                "is set to `field-to-be-used`")
        if self.ordering is ComponentOrder.RANDOM:
            raise Asn1Error(
                "ECN: §22.6.1.1 declares &alternative-ordering as ENUMERATED {textual, tag}; "
                "`random` belongs to §22.10's concatenation order, where there are several "
                "components to permute")

    def record(self, out: "BitWriter", index: int) -> None:
        """§22.6.3.5–§22.6.3.7, given the `alternative-index` the ordering produced."""
        if self.determination is AlternativeDetermination.FIELD_TO_BE_SET:
            out.patch(self.reference,
                      _determinant_value(self.encoder_transforms, index,
                                         f"ALTERNATIVE USING {self.reference}"))
            return
        if self.determination is AlternativeDetermination.FIELD_TO_BE_USED:
            carried = out.value_of(self.reference)
            recovered = (carried if self.decoder_transforms is None
                         else self.decoder_transforms.apply(carried))
            if recovered != index:
                raise Asn1Error(
                    f"ECN: §22.6.3.6 — the field {self.reference!r} carries {carried}, which "
                    f"reduces to alternative-index {recovered}, but the alternative being "
                    f"encoded has index {index}")
            return
        # §22.6.3.7: `handle` needs no encoder action. The alternative's own encoding already
        # carries the bits, and §22.9.3.1 has checked they are in its declared set.


@dataclass(frozen=True)
class ContainedType:
    """§22.11's `CONTENTS-ENCODING` group: which rules encode a type held inside another.

    §22.11.1.3 states the purpose in two halves, and the second is the one with teeth: "to
    determine the encoding of a contained type, **and whether an ASN.1 `ENCODED BY` contents
    constraint associated with that contained type shall be overridden**". So this group is
    where an ECN specification and an X.682 clause 11 contents constraint meet, and it says
    which of them wins.

    **§22.11.2's decision is a five-row table, and THE TEXT CONTRADICTS ITSELF ABOUT THE LAST
    ROW.** Written out because the obvious two-way reading ("use `CONTENTS-ENCODING` if set")
    gets half of it wrong, and because the disagreement has to be recorded rather than quietly
    resolved:

    | `CONTENTS-ENCODING` | `ENCODED BY` | `OVERRIDE` | what encodes the contained type |
    |---|---|---|---|
    | not set | absent  | —     | the set applied to the container (§22.11.2.1, §13.2.10.6 d) |
    | not set | present | —     | the rules `ENCODED BY` names (§22.11.2.1, §13.2.10.6 a) |
    | set     | absent  | —     | this group's combined set (§22.11.2.2, §13.2.10.6 c) |
    | set     | present | TRUE  | this group's combined set (§22.11.2.2, §13.2.10.6 b) |
    | set     | present | FALSE | **the rules `ENCODED BY` names** (§13.2.10.6 a) |

    **The last row is where the two clauses disagree.** §22.11.2.2's closing sentence says
    "Otherwise the combined encoding set applied to the **containing type** shall be applied to
    the contained type". §13.2.10.6 a) says the opposite for the same case: an object that
    "either does not contain a specification of the encoding of the contained type, **or
    specifies that it should not override an `ENCODED BY`**" leaves it that "the `ENCODED BY`
    specification **shall be used** for the contained type".

    §13.2.10.6 a) is taken as correct, and three independent readings are why:

    * §22.11.1.3 states this group's *purpose* as deciding "whether an ASN.1 `ENCODED BY`
      contents constraint … shall be **overridden**". Declining to override should leave the
      constraint standing; §22.11.2.2's reading would have `OVERRIDE FALSE` discard it outright,
      which is the opposite of declining.
    * §22.11.2.1 gives the parallel unset case to the `ENCODED BY`. §13.2.10.6 a) folds both
      into one rule; §22.11.2.2 would make them differ with no reason stated anywhere.
    * §13.2.10.6 is the *application-point algorithm* — the operative procedure that says what
      an encoder actually does — and it cites "(see 22.11)" as though it agreed.

    So `containing` is reached by exactly one row, not two. This is the same family of defect
    as §21.14.6's ordering, recorded in `ReversalSpecification`: two passages of X.692 that
    cannot both be followed, resolved by weight of agreement and written down so the reading is
    auditable rather than assumed.

    §22.11.1.5's "considered set if the `CONTENTS-ENCODING` keyword is used" is what makes the
    first column a fact about the notation rather than about whether `primary` happens to be
    empty.

    §22.11.1.4's combination is §9.23.2's, quoted there in full: the combined set is formed
    "by adding to the first set encoding objects for any encoding class for which the first
    set is lacking an encoding object and the second set contains one". A left-biased merge —
    primary wins, `COMPLETED BY` fills gaps, and never the other way round.
    """

    #: §22.11.1.1's `&Primary-encoding-object-set #ENCODINGS OPTIONAL`.
    primary: dict = field(default_factory=dict)
    #: `COMPLETED BY` — §9.23.1's second set. `None` is "not supplied", which is different
    #: from an empty one only in what a diagnostic can say about it.
    secondary: "dict | None" = None
    #: §22.11.1.1's `&over-ride-encoded-by BOOLEAN DEFAULT FALSE`.
    override: bool = False

    def combined(self) -> dict:
        """§13.2 / §9.23.2's combined encoding object set."""
        out = dict(self.primary)
        for cls, obj in (self.secondary or {}).items():
            out.setdefault(cls, obj)
        return out

    def select(self, *, encoded_by: "dict | None", containing: dict) -> dict:
        """§22.11.2's table, given the ASN.1 contents constraint and the container's own set.

        `encoded_by` is the rules an X.682 §11 `ENCODED BY` names, or `None` when the contents
        constraint states none. `containing` is "the combined encoding object set applied to
        the containing type", which is what two of the five rows fall back to.
        """
        if encoded_by is None or self.override:
            return self.combined()
        # §13.2.10.6 a): an object that "specifies that it should not override an ENCODED BY"
        # leaves it that "the ENCODED BY specification shall be used for the contained type".
        # §22.11.2.2's closing sentence says the CONTAINING type's set instead; see the class
        # docstring for why that sentence is the outlier and this is the reading taken.
        # `containing` is unused on this path and is kept in the signature because §13.2.10.6 d)
        # — the group unset with no ENCODED BY — is the one row that does reach it.
        return encoded_by


@dataclass(frozen=True)
class ContainerSpec:
    """A bit-field class whose encoding space holds another type's complete encoding.

    X.680 spells this `OCTET STRING (CONTAINING Inner)` and X.682 clause 11 gives it the
    `ENCODED BY` half; §22.11 is ECN's side of the same relationship. What makes it a distinct
    class here rather than a property on `IntSpec` is that the contained type is encoded by a
    **different object set** — §22.11.2.2's whole point — so it is a separate encoding that
    happens to be placed inside this one.

    **The contained encoding gets its own reference scope — and not its own bit buffer.**
    §9.24.2's application point moves into the contained type, so a start pointer or an
    auxiliary field reaching across the boundary would be measuring an offset in one encoding
    against a field in another, which no clause defines; an auxiliary field left unset inside
    is therefore refused *inside*, with its own diagnostic. The bits, though, go straight into
    the containing encoding, because the relationship §21.3.6 measures — "the last encoding
    placed in the container" — is a question about offsets in one stream. An implementation
    that isolated the buffer too would make that rule unanswerable, which is how this started
    and why the distinction is written down.

    `contained_by_container` is the *other* direction — this class as the container that
    §21.3.6's, §21.5.6's and §21.7.8's `container` determinations point at. Both directions
    exist in the clause and they are not the same relationship: one is "my contents are
    another type", the other is "my end is what bounds a component".
    """

    #: The encoding class of the contained type, looked up in whichever set §22.11.2 selects.
    contained_class: object = None
    #: §22.11's group. `None` is §22.11.1.5's "not set", which selects the container's own set.
    contents: "ContainedType | None" = None
    #: The X.682 §11 `ENCODED BY` rules, when the contents constraint names any.
    encoded_by: "dict | None" = None
    #: The container's own encoding space, in bits, when it is stated rather than determined.
    width: int = 0
    pre_alignment: PreAlignment | None = None
    start_pointer: StartPointer | None = None
    space_determinant: SpaceDeterminant | None = None
    #: Components written *after* the contained value, inside this container. Present so the
    #: "last encoding in the container" rule has something to be violated by.
    trailer: object = None
    exhibits: "IdentificationHandle | None" = None
    #: The name this container answers to for a `container` determination's REFERENCE.
    name: str = "container"

    def __post_init__(self) -> None:
        if self.contained_class is None:
            raise Asn1Error(
                "ECN: a container class encodes a contained type; §22.11.1.3 makes this "
                "group's purpose \"to determine the encoding of a contained type\", and there "
                "is none here")

    def objects_for(self, containing: dict) -> dict:
        """§22.11.2 and §13.2.10.6's selection, against the set applied to this container."""
        if self.contents is None:
            # §22.11.2.1 and §13.2.10.6 a)/d), which agree here: with the group unset an
            # ENCODED BY wins outright, and with neither the container's own set is used. That
            # last case is the ONLY row that reads `containing` — §13.2.10.6 d) in full.
            return self.encoded_by if self.encoded_by is not None else containing
        return self.contents.select(encoded_by=self.encoded_by, containing=containing)

    def write(self, value, out: "BitWriter") -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        containing = out.objects()
        objects = self.objects_for(containing)
        obj = objects.get(self.contained_class)
        if obj is None:
            named = getattr(self.contained_class, "name", self.contained_class)
            fell_back = objects is containing
            raise Asn1Error(
                f"ECN: §9.5.1 — the encoding object set §22.11.2 selected for this contained "
                f"type has no object for {named!r}"
                + (", and the set it selected is the one applied to the containing type "
                   "(§22.11.2's fallback), which is where an ENCODED BY that OVERRIDE did "
                   "not claim sends it" if fell_back else ""))
        out.open_container(self.name)
        scope = out.push_reference_scope()
        # §9.24.2's application point moves with the set, so a container nested inside this
        # contained type inherits *this* type's set as "the set applied to the containing
        # type" — not the PDU's. Restored by `pop_reference_scope`, which saves it alongside
        # the reference tables for exactly this reason.
        out.set_objects(objects)
        obj.spec.write(value, out)
        out.pop_reference_scope(scope, "the contained type")
        if self.trailer is not None:
            self.trailer.write(value, out)
        size = out.close_container(self.name)
        if self.width and size > self.width:
            raise Asn1Error(
                f"ECN: the container's contents are {size} bits and its stated encoding space "
                f"is {self.width}")
        if self.space_determinant is not None:
            self.space_determinant.record(out, size)
        _exhibit(self, out, handle_start)


@dataclass(frozen=True)
class Concatenation:
    """§22.10's `CONCATENATION` group: component order, inter-component alignment, handle.

    §22.10.2.6 is why this can be `None` on a spec and still mean something definite: "This
    specification is considered set if the `CONCATENATION` keyword is used. If it is not set
    then encoders and decoders act as if it was set with each encoding property taking its
    default value." So an absent group is `textual`, `aligned`, `default-handle` — and since
    §22.2.1.1's default alignment unit is one bit, `aligned` on its own inserts nothing.

    `random` is the value with a prerequisite. §22.10.2.1: it makes `HANDLE` "assume the
    default value of `default-handle` if not set", requires "the encoding objects applied to
    **all** components" to exhibit that handle, and requires their value sets to be disjoint.
    §22.10.3.3 then lets the encoder "determine the order of concatenation without constraint"
    — which, like §21.9.7's `encoder-option` padding, means the encoding has no unique octets.
    """

    order: ComponentOrder = ComponentOrder.TEXTUAL
    alignment: ConcatenationAlignment = ConcatenationAlignment.ALIGNED
    handle_id: str = "default-handle"


# --- the bit-level output the fixed candidate set never needed ---------------------------

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
        #: Open container scopes, innermost last: `(name, bit offset where its contents began)`.
        #: §21.3.6, §21.5.6 and §21.7.8 all end something "when the specified container
        #: terminates", and a stack is what lets a container inside a container mean what it
        #: says.
        self._containers: list[tuple[str, int]] = []
        #: Claims that an element's extent runs to a container's end: `name -> (offset, what)`.
        #: An empty name is §21.3.6's `OUTER` form, whose container is the PDU.
        self._container_claims: dict[str, tuple[int, str]] = {}
        #: The encoding object set in force. §22.11.2 twice falls back to "the combined
        #: encoding object set applied to the containing type", and §9.24's application point
        #: is what travels with an encoding — so the set lives with the writer rather than
        #: being re-derived at each nesting level from a structure's field table, which is a
        #: different namespace and would answer a different question.
        self._objects: dict = {}

    def put_bit(self, bit: int) -> None:
        self._bits.append(1 if bit else 0)

    def set_objects(self, objects: dict) -> None:
        self._objects = objects

    def objects(self) -> dict:
        return self._objects

    def bit_at(self, index: int) -> int:
        """One written bit, for §22.9's handles to read back out of the encoding.

        Reading is what makes a handle different from every other property group: the others
        decide what to write, and this one *inspects what was written*. §22.9.1.5 places the
        positions "in the final encoding, after any pre-alignment has been applied, and after
        any encoder bit-reversal actions have occurred" — so the check has to happen against
        the buffer rather than against the value the object started with.
        """
        if not 0 <= index < len(self._bits):
            raise Asn1Error(
                f"ECN: bit {index} is outside the {len(self._bits)}-bit encoding written so "
                f"far; §22.9.1.5 counts handle positions within the encoding that exhibits "
                f"the handle")
        return self._bits[index]

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

    # --- containment: the relationship §21.3.6, §21.5.6 and §21.7.8 all end something with ---
    #
    # Three clauses phrase it identically — "a REFERENCE to another field whose encoding class
    # (the container) has a length determinant and whose contents include this encoding space,
    # or ... that the end of the PDU determines the end" — and all three add the same rule:
    # the contained element has to be the LAST thing in the container. That rule is the whole
    # of the encoder's work for these determinations. §21.3.6's NOTE says why it is worth
    # checking rather than trusting: "It is an ECN encoder's error ... if additional encodings
    # are placed in the container", and the symptom otherwise is a decoder reading one field's
    # bits as another's.

    def open_container(self, name: str) -> None:
        if any(open_name == name for open_name, _at in self._containers):
            raise Asn1Error(
                f"ECN: the container {name!r} is already open; a container cannot contain "
                f"itself, and two with one name make every REFERENCE to it ambiguous")
        self._containers.append((name, len(self._bits)))

    def close_container(self, name: str) -> int:
        """End the innermost container, checking any claim made against it. Returns its size."""
        if not self._containers or self._containers[-1][0] != name:
            raise Asn1Error(
                f"ECN: {name!r} is not the innermost open container, so closing it would "
                f"cross a containment boundary")
        _name, at = self._containers.pop()
        claim = self._container_claims.pop(name, None)
        if claim is not None:
            offset, what = claim
            if offset != len(self._bits):
                raise Asn1Error(
                    f"ECN: §21.3.6/§21.5.6/§21.7.8 — {what} is determined by the container "
                    f"{name!r}, so it has to be the last encoding placed in it; "
                    f"{len(self._bits) - offset} further bits were written inside {name!r} "
                    f"afterwards. The three clauses word this rule identically; which one "
                    f"applies is what the description above names")
        return len(self._bits) - at

    def claim_container_end(self, name: str, what: str) -> None:
        """Register that `what` just ended, and that its end must be its container's end."""
        if name != OUTER_CONTAINER:
            if not any(open_name == name for open_name, _at in self._containers):
                raise Asn1Error(
                    f"ECN: a `container` determination names {name!r}, which is not an open "
                    f"container here; §21.3.6's REFERENCE is to a field \"whose contents "
                    f"include this encoding space\"")
        elif self._containers:
            raise Asn1Error(
                f"ECN: this element is determined by {OUTER_CONTAINER}, the end of the PDU, "
                f"but it sits inside the container {self._containers[-1][0]!r}; the PDU's end "
                f"is not this element's container's end, and §21.3.6's rule is about the "
                f"container that immediately holds it")
        self._container_claims[name] = (len(self._bits), what)

    def open_containers(self) -> tuple[str, ...]:
        return tuple(name for name, _at in self._containers)

    def push_reference_scope(self) -> tuple:
        """§9.24.2's application point moving into a contained type.

        "The combined encoding object set is applied to a generated encoding structure, and it
        is the encodings defined for the abstract values of **this** encoding structure that
        encode the abstract values of the ASN.1 type." A contained type is encoded by its own
        object set, so its REFERENCEs resolve among its own fields — a start pointer inside it
        measuring to a field of the container would be an offset in one encoding against a
        field in another, which no clause defines.

        The encoding object set is saved and restored with them, because §9.24.2 moves the
        *application point* and not merely the name resolution: a container nested inside a
        contained type inherits that type's set as §22.11.2's "set applied to the containing
        type", and reading the PDU's would answer a question one level too far out.

        What is scoped is those tables, and deliberately not the bit buffer. The
        contained encoding's bits are placed in the containing one, so the containment
        relationship §21.3.6 measures — "the last encoding placed in the container" — is a
        question about offsets in a single stream, and isolating those would make it
        unanswerable.
        """
        saved = (self._slots, self._patched, self._starts, self._values, self._objects)
        self._slots, self._patched, self._starts, self._values = {}, set(), {}, {}
        return saved

    def pop_reference_scope(self, saved: tuple, what: str) -> None:
        missing = self.unpatched()
        if missing:
            raise Asn1Error(
                f"ECN: {what}'s auxiliary field{'s' if len(missing) > 1 else ''} "
                f"{', '.join(repr(name) for name in missing)} "
                f"{'were' if len(missing) > 1 else 'was'} never set. §9.24.2 moves the "
                f"application point into a contained type, so its determinants resolve inside "
                f"it and cannot be supplied by the container")
        self._slots, self._patched, self._starts, self._values, self._objects = saved

    def outer_claim(self) -> "tuple[int, str] | None":
        """The `#OUTER` claim, for `encode_with_user` to check against the whole PDU."""
        return self._container_claims.get(OUTER_CONTAINER)

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

    def bits(self) -> tuple[int, ...]:
        """The written bits, whatever their number.

        Distinct from `octets()`, which refuses a partial octet: a contained type's encoding
        is placed *inside* another and has no reason to end on an octet boundary, so the whole
        -octets rule belongs to the PDU and not to every encoding on the way to it.
        """
        return tuple(self._bits)

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
    #: §22.9's group. Checked after the space is written, which §22.9.1.5's "final encoding"
    #: and step g) of §23.3.3.1's order both require.
    exhibits: "IdentificationHandle | None" = None

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
        # §22.9.1.5 puts position zero here: after pre-alignment, at the start of the space.
        handle_start = out.bit_length
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
        # g) Identification handle. §22.9.3.1's check runs against the bits as written, which
        #    is why it comes after the reversal rather than against `encoded`.
        _exhibit(self, out, handle_start)


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
    exhibits: "IdentificationHandle | None" = None

    def __post_init__(self) -> None:
        # §22.12.2.2/§22.12.2.3 relate the reversal to the space's unit, so the pair is
        # checked where both are stated rather than at write time — an object that could
        # never encode anything is invalid when it is written, not when it is first used.
        if self.bit_reversal is not ReversalSpecification.NO_REVERSAL:
            self.bit_reversal.check_unit(self.reversal_unit)

    def write(self, value: bool, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
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
        _exhibit(self, out, handle_start)


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
    exhibits: "IdentificationHandle | None" = None

    def write(self, _value, out: BitWriter) -> None:
        handle_start = out.bit_length
        for bit in _padding_bits(self.padding, self.pattern, self.width, "a #PAD object"):
            out.put_bit(bit)
        # §23.12.1's `#PAD` carries the identification-handle group like every other
        # bit-field category, and it is the one place a handle is *entirely* determined by
        # the specification: pad bits take no abstract value, so a #PAD object's handle value
        # set is a constant the specifier wrote down twice.
        _exhibit(self, out, handle_start)


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

    `REPLACE OPTIONALS` and `REPLACE NON-OPTIONALS` sort components by whether they are
    optional (§22.1.1.7 c) and d)), and became answerable when §23.11's optionality category
    was built: a component is optional exactly when its encoding object is an `OptionalSpec`.
    They were refused before that with those words, which is why `selects` is a method here
    rather than a condition inlined at the one call site.
    """

    action: ReplaceAction
    structure: ReplacementStructure
    #: §22.1.1.10's head-end insertion: a structure "inserted before all components of the
    #: (constructor) class performing the replacement", one per replaced component, "in the
    #: textual order of the original components".
    head_end: "HeadEndStructure | None" = None

    def selects(self, spec) -> bool:
        """§22.1.1.7 b)–d): whether this action replaces the component encoded by `spec`."""
        if self.action is ReplaceAction.OPTIONALS:
            return isinstance(spec, OptionalSpec)
        if self.action is ReplaceAction.NON_OPTIONALS:
            return not isinstance(spec, OptionalSpec)
        return True

    def __post_init__(self) -> None:
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
    #: §22.10's group. `None` is §22.10.2.6's "act as if it was set with each encoding
    #: property taking its default value", which is `textual`, `aligned`, `default-handle`.
    concatenation: "Concatenation | None" = None
    #: §22.2's group, which a `#CONCATENATION` object uses for **two** things: its own
    #: pre-alignment, and — per §22.10.3.5 — the alignment applied before each component when
    #: `ALIGNMENT` is `aligned`. §23.5.1 declares only one such group, so the two uses share it.
    pre_alignment: PreAlignment | None = None
    exhibits: "IdentificationHandle | None" = None
    #: The name this structure answers to as a *container*, if it is one. §22.5.2.10 says the
    #: `container` reference "shall be to a concatenation or to a repetition (or to a bitstring
    #: or octetstring with a contained type) in which the element being encoded is a
    #: component" — so a plain concatenation is a legitimate container, and naming it is how a
    #: REFERENCE reaches it. Empty means "not referred to as a container", which costs nothing.
    container_name: str = ""

    def __post_init__(self) -> None:
        group = self.concatenation or Concatenation()
        if group.order is ComponentOrder.RANDOM:
            # §22.10.2.1's prerequisite, checked when the object is written rather than when
            # it first encodes something: an object that can never be decoded is invalid on
            # its own terms.
            _handles_of(self._laid_out(), handle_id=group.handle_id,
                        clause="§22.10.2.1", what="a concatenation whose ORDER is `random`",
                        alignment_unit=self._component_alignment())
        elif group.order is ComponentOrder.TAG:
            self._tag_order()

    def _component_alignment(self) -> "int | None":
        """The unit §22.10.3.5 aligns each component to, or `None` when `ALIGNMENT` is `none`."""
        group = self.concatenation or Concatenation()
        if group.alignment is ConcatenationAlignment.NONE or self.pre_alignment is None:
            return None
        return self.pre_alignment.unit

    def _tag_order(self) -> tuple[tuple[str, object], ...]:
        """§22.10.3.2's ordering: "the order shall be that of the tag numbers ... lowest first".

        §22.10.2.4 and §22.10.2.5 are the preconditions, and both are checked: every component
        "shall start with an encoding class in the tag category", and "the component-tags of
        each alternative shall be distinct". A duplicate tag is the one that matters — the
        order would be arbitrary between the two, and so would the decoder's reading.
        """
        laid_out = self._laid_out()
        tagged: list[tuple[int, str, object]] = []
        for name, spec in laid_out:
            number = _component_tag(spec)
            if number is None:
                raise Asn1Error(
                    f"ECN: §22.10.2.4 — when ORDER is `tag`, every component shall start with "
                    f"an encoding class in the tag category; {name!r} does not")
            tagged.append((number, name, spec))
        numbers = [number for number, _name, _spec in tagged]
        if len(set(numbers)) != len(numbers):
            raise Asn1Error(
                f"ECN: §22.10.2.5 — the component-tags shall be distinct; {sorted(numbers)} "
                f"repeats one, so `ORDER tag` would not fix an order")
        return tuple((name, spec) for _number, name, spec in sorted(tagged))

    def transmission_order(self) -> tuple[str, ...]:
        return tuple(name for name, _spec in self._ordered())

    def _ordered(self) -> tuple[tuple[str, object], ...]:
        """The components in the order §22.10.3.1–§22.10.3.3 puts them on the wire.

        `random` is implemented as "the textual order", and that is *a* conforming choice
        rather than *the* one: §22.10.3.3 lets the encoder "determine the order of
        concatenation without constraint", so an encoding using it has no unique octets — the
        same hazard §21.9.7's `encoder-option` padding carries, and worth the same warning to
        anything comparing this rail's output byte for byte against a twin.
        """
        group = self.concatenation or Concatenation()
        if group.order is ComponentOrder.TAG:
            return self._tag_order()
        return self._laid_out()

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
            spec = self.fields[name]
            # §22.1.1.7 c) and d) replace only the optional or only the non-optional
            # components; everything the action does not select passes through untouched, and
            # takes no head-end insertion either — §22.1.3.6 orders the insertions by "the
            # components being replaced", so a component that is not replaced contributes none.
            if not self.replacement.selects(spec):
                body.append((name, spec))
                continue
            if self.replacement.head_end is not None:
                heads.extend(self.replacement.head_end.expand(name))
            # §22.1.3.4 strips the optionality class: the component is replaced "with a
            # **non-optional** instantiation of the replacement structure", and the actual
            # parameter "shall de-reference to the entire original optional component
            # (including any classes in the tag category) **except for any class in the
            # optionality category**". So replacing an optional component makes it mandatory,
            # and its PRESENCE determinant has nothing left to determine — which surfaces as
            # an unset auxiliary field rather than as silence.
            if isinstance(spec, OptionalSpec):
                spec = spec.component
            body.extend(self.replacement.structure.expand(name, spec))
        return tuple(heads) + tuple(body)

    def write(self, value: dict, out: BitWriter) -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.container_name:
            out.open_container(self.container_name)
        group = self.concatenation or Concatenation()
        aligned = group.alignment is ConcatenationAlignment.ALIGNED
        for name, spec in self._ordered():
            # §22.10.3.5: with `ALIGNMENT aligned` the concatenation's own pre-alignment
            # specification runs before *each* component, and §22.10.3.5's exception folds
            # `ALIGNED TO ANY` down to `ALIGNED TO NEXT` here — NOTE 1 gives the reason,
            # "there can only be a single start pointer for ALIGNED TO ANY". NOTE 2 then puts
            # the component's own pre-alignment after this, which is where each spec runs it.
            if aligned and self.pre_alignment is not None:
                self.pre_alignment.apply(out)
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
            if isinstance(spec, OptionalSpec):
                # §23.11 makes absence the component's own business, so a missing key is a
                # value rather than the fault it is for a mandatory component.
                out.mark_start(name)
                spec.write(value.get(name), out)
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
        if self.container_name:
            out.close_container(self.container_name)
        _exhibit(self, out, handle_start)


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


# --- clause 22.7 and clause 23.13/23.14: repetition ----------------------------------------

@dataclass(frozen=True)
class RepetitionSpace:
    """§22.7's repetition space: how a decoder finds the end of a repetition.

    §21.7.3 is the sentence that places this group: it "**replaces** use of an encoding
    property of type `EncodingSpaceDetermination` in the encoding of repetitions". So a
    repetition does not have a §22.4 encoding space with a §21.3 determinant — it has this,
    with §21.7's eight-valued determination, and the two are siblings rather than one being a
    special case of the other.

    Five determinations are built, and they cover how essentially every real repeated format
    works:

    * `field-to-be-set` — a count field, written by the encoder from the number of repetitions.
    * `field-to-be-used` — the same field, supplied by the application and *checked*.
    * `pattern` — §22.7.1.1's `&termination-pattern`, emitted after the last element. This is
      the NUL-terminated string, and it is why the group carries a `Pattern` at all.
    * `handle` — §21.7.10's identification handle, exhibited by the repeated element and by
      whatever can follow it. §22.7.3.11 gives the encoder no action for it beyond §22.9.3.1's
      own check, which is why this became buildable the moment handles did.
    * `not-needed` — §21.7.11's fixed count, which the abstract syntax already carries.

    The remaining three are refused with what each would need. `flag-to-be-set` and
    `flag-to-be-used` (§21.7.6/§21.7.7) put a continuation flag **inside the repeated
    element**, which needs the element's own structure to have a field reserved for it;
    `container` (§21.7.8) needs containment.
    """

    determination: RepetitionSpaceDetermination = RepetitionSpaceDetermination.FIELD_TO_BE_SET
    #: §22.7.1.1's `&main-reference REFERENCE OPTIONAL` — the count field.
    reference: str = ""
    #: The `MULTIPLE OF` unit the count is in. §21.1.5 admits `repetitions` here and only
    #: here: "the associated count gives the number of repetitions in the encoding".
    unit: int = UNIT_REPETITIONS
    #: §22.7.1.1's `&termination-pattern`, for the `pattern` determination.
    termination_pattern: Pattern | None = None
    encoder_transforms: "TransformChain | None" = None
    decoder_transforms: "TransformChain | None" = None
    #: §21.7.10's identification handle, for the `handle` determination.
    handle_id: str = "default-handle"

    _BUILT = (RepetitionSpaceDetermination.FIELD_TO_BE_SET,
              RepetitionSpaceDetermination.FIELD_TO_BE_USED,
              RepetitionSpaceDetermination.PATTERN,
              RepetitionSpaceDetermination.HANDLE,
              RepetitionSpaceDetermination.CONTAINER,
              RepetitionSpaceDetermination.NOT_NEEDED)

    def __post_init__(self) -> None:
        if self.determination not in RepetitionSpace._BUILT:
            needs = {
                RepetitionSpaceDetermination.FLAG_TO_BE_SET:
                    "§21.7.6 puts a continuation flag INSIDE the repeated element, which needs "
                    "the element's structure to reserve a field for it",
                RepetitionSpaceDetermination.FLAG_TO_BE_USED:
                    "§21.7.7 reads a continuation flag from inside the repeated element",
            }[self.determination]
            raise Asn1Error(
                f"ECN: the repetition-space determination `{self.determination.value}` is not "
                f"implemented — {needs}")
        if self.determination is RepetitionSpaceDetermination.PATTERN:
            if self.termination_pattern is None:
                raise Asn1Error(
                    "ECN: §22.7.1.1's `pattern` determination ends the repetition with the "
                    "&termination-pattern, and none is set")
            self.termination_pattern.require_non_null("the termination pattern")
        elif self.determination is RepetitionSpaceDetermination.HANDLE:
            if self.reference:
                raise Asn1Error(
                    "ECN: §21.7.10's `handle` determination reads the identification handle "
                    "the elements already exhibit; it takes no USING reference to a count")
        elif self.determination is not RepetitionSpaceDetermination.NOT_NEEDED:
            if not self.reference:
                raise Asn1Error(
                    f"ECN: §21.7.4/§21.7.5 — `{self.determination.value}` requires a USING "
                    f"reference to the field carrying the count")
        check_unit(self.unit, allow_repetitions=True)

    def count_in_units(self, repetitions: int, space_bits: int) -> int:
        """§21.7.4's "size (in repetition space units)".

        §21.1.5 is what makes this two cases rather than one: with `repetitions` as the unit
        the count IS the number of elements, and with any other unit it is the space's size in
        those units. A format carrying "how many" and one carrying "how many octets" are both
        ordinary, and they are not the same number.
        """
        if self.unit == UNIT_REPETITIONS:
            return repetitions
        if space_bits % self.unit:
            raise Asn1Error(
                f"ECN: a {space_bits}-bit repetition space is not a whole number of "
                f"{self.unit}-bit units, so no determinant can state its size")
        return space_bits // self.unit

    def record(self, out: "BitWriter", repetitions: int, space_bits: int) -> None:
        """Set or check the count field, after the elements have been written."""
        if self.determination is RepetitionSpaceDetermination.CONTAINER:
            # §22.7.3.6: "there is no further encoder action". §21.7.8's NOTE is the rule that
            # remains — "This specification can only be used if the encoding of the
            # (repetition category) class is the last encoding to be placed in the container"
            # — and it is checked when the container closes.
            out.claim_container_end(self.reference, "a repetition whose end")
            return
        # §22.7.3.11: "If DETERMINED BY is `handle` there is no further action needed by the
        # encoder." The elements' own §22.9.3.1 checks have already run by the time this does.
        if self.determination in (RepetitionSpaceDetermination.NOT_NEEDED,
                                  RepetitionSpaceDetermination.PATTERN,
                                  RepetitionSpaceDetermination.HANDLE):
            return
        count = self.count_in_units(repetitions, space_bits)
        if self.determination is RepetitionSpaceDetermination.FIELD_TO_BE_SET:
            out.patch(self.reference,
                      _determinant_value(self.encoder_transforms, count,
                                         f"REPETITION-SPACE USING {self.reference}"))
            return
        carried = out.value_of(self.reference)
        recovered = (carried if self.decoder_transforms is None
                     else self.decoder_transforms.apply(carried))
        if recovered != count:
            raise Asn1Error(
                f"ECN: §21.7.5 — the field {self.reference!r} carries {carried}, which reduces "
                f"to {recovered} repetition-space units, but the encoding used {count}")

    def terminate(self, out: "BitWriter") -> None:
        """Emit §22.7.1.1's termination pattern, if that is how this space ends."""
        if self.determination is RepetitionSpaceDetermination.PATTERN:
            for bit in self.termination_pattern.bit_sequence():
                out.put_bit(bit)


@dataclass(frozen=True)
class ConditionalRepetitionSpec:
    """§23.14's `#CONDITIONAL-REPETITION`: a repetition encoding guarded by the SIZE bounds.

    The integer story again, one clause over. §23.13.3.1 selects "the first
    `#CONDITIONAL-REPETITION` encoding object in `ENCODING(S)` whose conditions are satisfied",
    and §21.13.3 makes those conditions tests on the *effective size constraint* — so an object
    set encodes `SEQUENCE (SIZE(4)) OF` with no length field and `SEQUENCE OF` with one, from
    the schema.
    """

    element: object = None
    space: RepetitionSpace = field(default_factory=RepetitionSpace)
    pre_alignment: PreAlignment | None = None
    #: `(SizeRangeCondition, Comparison | None, comparator | None)`, all of which must hold.
    conditions: tuple = ()

    def __post_init__(self) -> None:
        if self.element is None:
            raise Asn1Error(
                "ECN: a #CONDITIONAL-REPETITION object encodes a repeated element")
        if self.space.determination is RepetitionSpaceDetermination.HANDLE:
            # §21.7.10 requires the handle from two sides: "the encoding object applied to the
            # component being repeated, **and** the encoding object applied to each possible
            # (taking account of optionality) following encoding class". Only the first is
            # visible here — what follows a repetition belongs to the enclosing structure — so
            # this checks the element and the enclosing concatenation is where the other half
            # would go. Half a rule enforced is worth more than none, and the half that is
            # missing is named rather than silently skipped.
            _handles_of((("the repeated element", self.element),),
                        handle_id=self.space.handle_id, clause="§21.7.10",
                        what="a repetition whose end is",
                        alignment_unit=None if self.pre_alignment is None
                        else self.pre_alignment.unit)

    def applies(self, bounds: SizeBounds) -> bool:
        return all(bounds.satisfies(condition, comparison, comparator)
                   for condition, comparison, comparator in self.conditions)

    def write(self, values, out: "BitWriter") -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        start = out.bit_length
        for value in values:
            self.element.write(value, out)
        self.space.terminate(out)
        self.space.record(out, len(values), out.bit_length - start)


@dataclass(frozen=True)
class RepetitionSpec:
    """§23.13's `#REPETITION`: the conditional encodings and the type's size bounds.

    §23.13.2.2 permits exactly one of `REPETITION-ENCODING` and `REPETITION-ENCODINGS`, and
    §23.13.2's NOTE gives the reason the singular exists at all — it avoids "a double
    curly-bracket (`{{`) in the common case of a single encoding object", and using the plural
    for one object "is deprecated but is allowed". A syntactic convenience, not a semantic
    distinction, so only the list is modelled.

    §23.13.2.3 is a real ordering rule and is enforced: "If an encoding object in the
    `REPETITION-ENCODINGS` ordered list is defined using `IF` or `IF-ALL`, then all **preceding**
    encoding objects in that list shall be defined using `IF` or `IF-ALL`." An unconditional
    object matches everything, so anything after it is unreachable — the clause forbids writing
    dead alternatives rather than leaving them to be discovered.
    """

    encodings: tuple = ()
    bounds: SizeBounds = field(default_factory=SizeBounds)

    def __post_init__(self) -> None:
        seen_unconditional = False
        for index, candidate in enumerate(self.encodings):
            if seen_unconditional and candidate.conditions:
                raise Asn1Error(
                    f"ECN: §23.13.2.3 — a conditional #CONDITIONAL-REPETITION object at "
                    f"position {index} follows an unconditional one, which always matches; "
                    f"every object preceding a conditional one shall itself be conditional")
            if not candidate.conditions:
                seen_unconditional = True

    def select(self) -> ConditionalRepetitionSpec:
        for candidate in self.encodings:
            if candidate.applies(self.bounds):
                return candidate
        raise Asn1Error(
            f"ECN: §23.13.3.1 — no #CONDITIONAL-REPETITION object's conditions are satisfied "
            f"by {self.bounds}; the clause makes that an ECN specification error")

    def write(self, values, out: "BitWriter") -> None:
        self.select().write(values, out)


# --- clause 23's string, null and tag categories -------------------------------------------

@dataclass(frozen=True)
class StringSpec:
    """§23.2's `#BITS`, §23.9's `#OCTETS` and §23.4's `#CHARS`, which share one shape.

    **None of the three has an `ENCODING-SPACE` group**, and noticing that is the whole point.
    Their `WITH SYNTAX` gives pre-alignment, a start pointer, `VALUE-REVERSAL`, `TRANSFORMS`,
    and `REPETITION-ENCODING(S)` — so a string's size comes from the *repetition* machinery of
    §22.7, not from a stated width. That is why these three could not be built before
    repetition was, and why they are one class here rather than three.

    `value_reversal` is §23.2.1's `&value-reversal BOOLEAN DEFAULT FALSE`, which reverses the
    order of the *elements* — distinct from §22.12's bit reversal, which reverses bits within
    a unit. A format that sends a string backwards and a format that sends each octet's bits
    backwards are different formats, and ECN spells them with different properties.
    """

    #: How one element encodes: a bit, an octet or a character.
    element: object = None
    repetition: RepetitionSpec = None
    transform: "Transform | None" = None
    value_reversal: bool = False
    pre_alignment: PreAlignment | None = None
    start_pointer: StartPointer | None = None
    exhibits: "IdentificationHandle | None" = None

    def __post_init__(self) -> None:
        if self.repetition is None:
            raise Asn1Error(
                "ECN: §23.2.1 gives a string category no ENCODING-SPACE; its size comes from "
                "the §22.7 repetition space, so a REPETITION-ENCODING is required")

    def write(self, value, out: "BitWriter") -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        if self.transform is not None:
            value = self.transform.apply(value)
        elements = list(value.elements if isinstance(value, Composite) else value)
        if self.value_reversal:
            elements.reverse()
        self.repetition.write(elements, out)
        _exhibit(self, out, handle_start)


@dataclass(frozen=True)
class BitFieldSpec:
    """One element of a `#BITS` repetition: a bitstring written as it stands.

    §23.2's repeated element is a *bit*, and §24.15's composite makes it a bitstring of the
    composite's unit — so after a transform has run, a string's elements are tuples of bits
    rather than integers. This writes those, which `IntSpec` cannot: an `IntSpec` takes an
    abstract integer and chooses bits for it, where here the bits already are the value.
    """

    width: int = 0

    def write(self, value, out: "BitWriter") -> None:
        bits = tuple(value)
        if self.width and len(bits) != self.width:
            raise Asn1Error(
                f"ECN: this bit-field element is {self.width} bits wide; the value carries "
                f"{len(bits)}")
        for bit in bits:
            out.put_bit(bit)


@dataclass(frozen=True)
class NullSpec:
    """§23.8's `#NUL`: a class with exactly one abstract value.

    It has an encoding space like any bit-field class, and nothing to put in it — X.680's NULL
    carries no information, so every bit of the space is padding. That makes it the one
    category where `VALUE-PADDING` is the entire value encoding.
    """

    width: int = 0
    padding: Padding = Padding.ZERO
    pattern: Pattern | None = None
    pre_alignment: PreAlignment | None = None
    start_pointer: StartPointer | None = None
    exhibits: "IdentificationHandle | None" = None

    def write(self, _value, out: "BitWriter") -> None:
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        for bit in _padding_bits(self.padding, self.pattern, self.width, "a #NUL object"):
            out.put_bit(bit)
        _exhibit(self, out, handle_start)


@dataclass(frozen=True)
class TagSpec:
    """§23.15's `#TAG`: an identifier written ahead of what it tags.

    §20.2 is why this class is not like the others: "The defined syntax for each category can
    also be used to define encoding objects for structures which are classes of that category,
    **preceded by one or more instances of a class in the tag category**." A tag is a prefix
    that composes with any bit-field encoding rather than a category of its own value — which
    is exactly how BER's identifier octet relates to its contents.
    """

    width: int
    number: int = 0
    form: IntForm = IntForm.POSITIVE_INT
    #: What the tag precedes. `None` writes the tag alone, which §20.2's composition allows.
    tagged: object = None
    pre_alignment: PreAlignment | None = None
    value_padding: ValuePadding | None = None
    #: §22.9's group, and the **only** category where `tag:any` is legal (§22.9.1.9). A tag is
    #: a discriminator by construction, so a `#TAG` object exhibiting a handle usually has
    #: nothing to state: the number it writes is the handle value.
    exhibits: "IdentificationHandle | None" = None

    def write(self, value, out: "BitWriter") -> None:
        # Pre-alignment is applied here rather than handed to the inner `IntSpec` so that the
        # handle's position zero can be taken afterwards. §22.9.1.5 puts the positions "after
        # any pre-alignment has been applied", and the octets are identical either way.
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        IntSpec(width=self.width, form=self.form,
                value_padding=self.value_padding).write(self.number, out)
        if self.tagged is not None:
            self.tagged.write(value, out)
        # Checked over the tag *and* what it tags, because §22.9.1.5 counts positions in the
        # final encoding of the object exhibiting the handle — which lets a handle reach a bit
        # of the tagged content, as BER's constructed flag effectively does.
        _exhibit(self, out, handle_start)


# --- clause 23.1 and 23.11: the constructor categories -------------------------------------

@dataclass(frozen=True)
class OptionalSpec:
    """§23.11's `#OPTIONAL`: a component that may or may not be in the encoding.

    **This is the category the rest of the module has been refusing to need.** Until now every
    field a `#CONCATENATION` object named had to be in the value, because "a user-defined
    encoding has no optionality unless an `#OPTIONAL` object supplies it" — this is that
    object. §23.11.2.1: "This syntax is used to define the encoding of a class in the
    optionality category."

    Its shape is small and its consequences are not. An `#OPTIONAL` object owns only §22.5's
    `PRESENCE` group plus replacement, pre-alignment and a start pointer; the *component's*
    own encoding object still encodes the component. What this adds is the fact that a decoder
    needs and the component itself cannot carry: whether it is there at all.

    §22.5.1.6 makes `PRESENCE` mandatory rather than defaulted, and that is enforced in
    `__post_init__` — an `#OPTIONAL` object without it is not one taking defaults, it is one
    that never said how absence is detected.

    Absence writes nothing here. §23.11.3.1 lists pre-alignment among the encoder actions, but
    aligning before a component that is not going to be encoded would put bits in the stream
    that no decoder is looking for, and §22.10.2.7 shows the clause thinking in the same terms
    when it forbids pre-alignment on a concatenation that "has no bits in its encoding".
    """

    #: The encoding object for the component itself, when it is present.
    component: object = None
    #: §22.5's group. No default, because §22.5.1.6 gives it none.
    presence: "Optionality | None" = None
    pre_alignment: PreAlignment | None = None
    start_pointer: StartPointer | None = None
    replacement: Replacement | None = None
    exhibits: "IdentificationHandle | None" = None

    def __post_init__(self) -> None:
        if self.component is None:
            raise Asn1Error(
                "ECN: an #OPTIONAL object wraps the encoding of the component it makes "
                "optional; there is nothing here to encode when the component is present")
        if self.presence is None:
            raise Asn1Error(
                "ECN: §22.5.1.6 — the PRESENCE specification is mandatory wherever the defined "
                "syntax allows it, and defaulting every part of it "
                "\"would not satisfy the above constraints\"; this #OPTIONAL object states "
                "none, so nothing says how a decoder detects absence")
        if self.presence.determination is OptionalityDetermination.POINTER:
            if self.start_pointer is None:
                raise Asn1Error(
                    "ECN: §22.5.2.4 — if DETERMINED BY is `pointer`, there shall be a "
                    "START-POINTER specification in the same encoding object")
            # §22.5.2.4's NOTE stays a diagnostic rather than a rule: "A start pointer
            # specification normally also needs a pre-alignment specification with ALIGNED TO
            # ANY". `normally` is not `shall`, and it is the ANY case that makes the pointer
            # carry information a fixed layout would not need.
        if self.replacement is not None:
            if self.replacement.action is not ReplaceAction.STRUCTURE:
                raise Asn1Error(
                    "ECN: §23.11.1 gives `#OPTIONAL` structure-only replacement — only "
                    "`&#Replacement-structure`, with no COMPONENT, OPTIONALS or NON-OPTIONALS "
                    "actions; those belong to the concatenation category")
            raise Asn1Error(
                "ECN: §23.11.3.2's REPLACE STRUCTURE hands \"the entire component (including "
                "any classes in the tag category, but excluding classes in the optionality "
                "category)\" to the replacement structure as its actual parameter, and the "
                "component \"becomes a mandatory component\". That is X.683 parameterization "
                "applied to an optional class, which is not built; applying the replacement "
                "object to the component's own class expresses the same encoding")

    def write(self, value, out: "BitWriter") -> None:
        present = value is not None
        if not present:
            if self.presence.determination is OptionalityDetermination.POINTER:
                # §21.5.9: "If that field is zero, then this component is absent." Zero is
                # available as a sentinel because the pointer field is itself encoded before
                # the offset it measures, so a genuine offset is never zero.
                out.patch(self.start_pointer.reference, 0)
            self.presence.record(out, False)
            return
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        self.component.write(value, out)
        self.presence.record(out, True)
        _exhibit(self, out, handle_start)


@dataclass(frozen=True)
class AlternativesSpec:
    """§23.1's `#ALTERNATIVES`: a construction of which exactly one component is encoded.

    The CHOICE, and the first category in this module whose encoder has to *decide* something
    about structure rather than about bits. §22.6.3.2: the encoder "shall determine which
    alternative the application wishes to be encoded, and shall create a conceptual integer
    value `alternative-index` to identify that alternative".

    §22.6.3.3 defines that index positionally — zero for the first, one for the next — and
    §22.6.3.4 says which order counts: `textual` is "the textual order in the ASN.1 type
    specification or the ECN structure definition", `tag` is "the order of the tag numbers in
    the component-tags (lowest tag number first)". So renaming an alternative changes nothing
    and reordering the ECN structure changes the wire format, which is the right way round for
    a notation whose job is to describe a layout.

    §23.1.2.3 is worth stating because the obvious implementation gets it wrong: this object
    "does not exhibit an identification handle unless `EXHIBITS HANDLE` is set (**even if the
    components of the defined construction exhibit an identification handle**)". The
    alternatives' handles are what `DETERMINED BY handle` reads; they are not inherited.
    """

    #: Alternative name -> the encoding object for that alternative.
    alternatives: dict = field(default_factory=dict)
    #: §22.6's group. Defaults to `field-to-be-set`, which then needs a `USING` reference.
    selection: "AlternativeSelection | None" = None
    #: Names in textual order. Empty means "the order of `alternatives`".
    order: tuple[str, ...] = ()
    pre_alignment: PreAlignment | None = None
    start_pointer: StartPointer | None = None
    exhibits: "IdentificationHandle | None" = None

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise Asn1Error(
                "ECN: an #ALTERNATIVES object encodes a construction in the alternatives "
                "category, and a construction with no alternatives has no encodings")
        if self.selection is None:
            raise Asn1Error(
                "ECN: §22.6.2.9 — the ALTERNATIVE specification \"is mandatory for it to be "
                "set in all places in the defined syntax where it is allowed\"; this "
                "#ALTERNATIVES object states none")
        if self.order:
            missing = set(self.alternatives) - set(self.order)
            extra = set(self.order) - set(self.alternatives)
            if missing or extra:
                raise Asn1Error(
                    f"ECN: this #ALTERNATIVES object states a textual order that does not "
                    f"match its alternatives (missing {sorted(missing)}, unknown "
                    f"{sorted(extra)})")
        if self.selection.ordering is ComponentOrder.TAG:
            self._tag_order()
        if self.selection.determination is AlternativeDetermination.HANDLE:
            # §21.6.6: the handle "shall be exhibited by the encoding objects applied to each
            # of the alternatives ... The handle value sets specified by those encoding
            # objects shall all be disjoint." Both halves are checked; without the second the
            # decoder's §22.6.4.4 lookup would have more than one answer.
            _handles_of(tuple(self._textual()), handle_id=self.selection.handle_id,
                        clause="§21.6.6", what="an alternatives construction",
                        alignment_unit=None if self.pre_alignment is None
                        else self.pre_alignment.unit)

    def _textual(self):
        names = self.order or tuple(self.alternatives)
        return [(name, self.alternatives[name]) for name in names]

    def _tag_order(self) -> tuple[str, ...]:
        """§22.6.3.4's `tag` ordering, with §22.6.2.10 and §22.6.2.11 as its preconditions."""
        tagged: list[tuple[int, str]] = []
        for name, spec in self._textual():
            number = _component_tag(spec)
            if number is None:
                raise Asn1Error(
                    f"ECN: §22.6.2.10 — when ORDER is `tag`, every alternative shall start "
                    f"with an encoding class in the tag category; {name!r} does not")
            tagged.append((number, name))
        numbers = [number for number, _name in tagged]
        if len(set(numbers)) != len(numbers):
            raise Asn1Error(
                f"ECN: §22.6.2.11 — the component-tags of each alternative shall be distinct; "
                f"{sorted(numbers)} repeats one")
        return tuple(name for _number, name in sorted(tagged))

    def ordering(self) -> tuple[str, ...]:
        """The alternatives in the order §22.6.3.3 counts them from."""
        if self.selection.ordering is ComponentOrder.TAG:
            return self._tag_order()
        return tuple(name for name, _spec in self._textual())

    def index_of(self, name: str) -> int:
        """§22.6.3.3's `alternative-index`."""
        order = self.ordering()
        if name not in order:
            raise Asn1Error(
                f"ECN: {name!r} is not one of this #ALTERNATIVES object's alternatives "
                f"({', '.join(order)})")
        return order.index(name)

    def write(self, value, out: "BitWriter") -> None:
        """Encode the chosen alternative.

        `value` is the pair `(name, value)` or a single-key mapping, which is what a CHOICE
        value is: the alternative's identity is part of the value and not something the
        encoding object can recover from the payload.
        """
        if isinstance(value, dict):
            if len(value) != 1:
                raise Asn1Error(
                    f"ECN: a class in the alternatives category encodes exactly one "
                    f"alternative; this value carries {len(value)}")
            name, chosen = next(iter(value.items()))
        else:
            name, chosen = value
        index = self.index_of(name)
        if self.pre_alignment is not None:
            self.pre_alignment.apply(out)
        handle_start = out.bit_length
        if self.start_pointer is not None:
            self.start_pointer.record(out)
        self.alternatives[name].write(chosen, out)
        # d) Alternative determination, after the alternative is written — the field it sets
        #    is earlier in the encoding, which is exactly what §22.6.3.5's NOTE describes as
        #    the encoder having to "suspend the encoding of that field".
        self.selection.record(out, index)
        _exhibit(self, out, handle_start)


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
    out.set_objects(objects)
    obj.spec.write(value, out)
    still_open = out.open_containers()
    if still_open:  # pragma: no cover - every writer closes what it opens
        raise Asn1Error(
            f"ECN: the container{'s' if len(still_open) > 1 else ''} "
            f"{', '.join(repr(name) for name in still_open)} "
            f"{'were' if len(still_open) > 1 else 'was'} opened and never closed")
    # §21.3.6's second form: "a specification that the end of the PDU determines the end of
    # the encoding space (using OUTER)". Its check can only run here, because the PDU's end is
    # what this function is producing — and it runs BEFORE `#OUTER` pads, since padding to an
    # octet boundary is not "further encodings placed in the container".
    claim = out.outer_claim()
    if claim is not None and claim[0] != out.bit_length:
        raise Asn1Error(
            f"ECN: §21.3.6 — {claim[1]} is determined by {OUTER_CONTAINER}, the end of the "
            f"PDU, so it "
            f"has to be the last encoding in it; {out.bit_length - claim[0]} further bits "
            f"follow")
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
    "BitToBits", "BitsToBits", "BitsToChar", "BitsToCompositeBits", "BitsToInt",
    "BoolToBool", "BoolToInt", "CharToBits", "CharsToCompositeChar", "Composite",
    "CompositeBitsToBits", "CompositeBitsToOctets", "CompositeCharToChars",
    "IntOp", "IntSelector", "IntSpec", "IntToBits", "IntToBool", "IntToChars", "IntToInt",
    "IntegerBounds", "OctetsToCompositeBits", "RESULT_SIZE_FIXED_TO_MAX",
    "RESULT_SIZE_VARIABLE",
    "Justification", "JustificationSide", "OuterSpec", "PadSpec", "Padding", "Pattern",
    "PatternKind", "PreAlignment", "RangeCondition", "ReplaceAction",
    "Replacement", "ReplacementStructure", "HeadEndStructure",
    "ReversalSpecification",
    "BitFieldSpec", "ConditionalRepetitionSpec", "NullSpec", "RepetitionSpaceDetermination",
    "RepetitionSpace", "RepetitionSpec", "SizeBounds", "SizeRangeCondition",
    "SpaceDeterminant", "StartPointer", "StringSpec", "TagSpec", "Transform",
    "TransformChain", "UnusedBits",
    # The constructor categories and the identification handle four clauses read (§21.5,
    # §21.6, §21.16, §22.5, §22.6, §22.9, §22.10, §23.1, §23.11).
    "AlternativeDetermination", "AlternativeSelection", "AlternativesSpec",
    "ComponentOrder", "Concatenation", "ConcatenationAlignment", "HandleRegistry",
    "HandleValueKind", "HandleValueSet", "IdentificationHandle", "Optionality",
    "OptionalityDetermination", "OptionalSpec",
    # Containment, in both directions (§22.11, §21.3.6/§21.5.6/§21.7.8).
    "ContainedType", "ContainerSpec", "OUTER_CONTAINER",
    "UnusedBitsDetermination", "UserEncodingObject", "ValuePadding", "check_unit",
    "encode_with_user", "legacy_frame_objects", "legacy_frame_workload", "refuted_by",
]
