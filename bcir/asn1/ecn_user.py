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

CITATION HONESTY. Part one cites sub-clauses exactly (§9.6.7, §18.1.7, §18.2.5.1) because
each was checked against the text. The text of clauses 19–25 was **not** available when this
module was written. So it cites at the granularity the repository has already established —
clauses 19–25 are the user-defined encodings, of which 20–23 are the defined syntax, and
`#TRANSFORM`/`#OUTER`/`#CONDITIONAL-INT`/`#CONDITIONAL-REPETITION` are the encoding-procedure
classes part one already declares — and describes every other rule in its own words rather
than attaching a sub-clause number it cannot support. A precise-looking citation that is
wrong is worse in this repository than an honest description, because the whole point of the
correspondence is that a reader can check it.

WHAT IS DELIBERATELY NOT HERE. The ECN *surface syntax* is not parsed: there is no
`EncodingObjectDefinition` grammar reading `WITH SYNTAX`-style defined syntax from text. The
model below is the semantics that syntax denotes, reachable directly from Python, which is
the same posture `ecn.py` takes for EDM/ELM. Nothing in the encoding path depends on a parser
existing, so adding one later changes how objects are *written*, not what they *mean*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .tags import Asn1Error


# --- the bit-level output the fixed candidate set never needed ---------------------------

class Justification(Enum):
    """Where a value sits when its encoding space is wider than the value needs.

    The fixed candidate set has no equivalent knob. PER picks a width from the constraint and
    fills it; OER and DER work in whole octets. A user-defined encoding chooses the space
    *and* the position within it, which is what lets an ECN object match a header field that
    was laid out before any of these standards existed.
    """

    RIGHT = "right"
    LEFT = "left"


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


# --- clause 21's #TRANSFORM: the encoded value as a function of the abstract one ----------

@dataclass(frozen=True)
class Transform:
    """Base for the value transformations a `#TRANSFORM` encoding object applies.

    This is the mechanism the whole reopening rests on. Every rule in the fixed candidate set
    encodes the abstract value; a transform makes the *transmitted* value a declared function
    of it, and `inverse` is what keeps that honest — a transform with no inverse would be an
    encoding a decoder cannot undo, which is a lossy channel rather than an encoding rule.
    """

    name: str = ""

    def apply(self, value):
        raise NotImplementedError

    def inverse(self, value):
        raise NotImplementedError


@dataclass(frozen=True)
class IntToInt(Transform):
    """`INT-TO-INT`: an affine map on an integer, `(value - offset) // scale`.

    The shape real headers actually use. IPv4's IHL is a length in 4-octet units; a great many
    link-layer length fields transmit `n - 1` so that a 4-bit field can express 1..16. Both are
    this transform with different constants.

    `scale` must divide the value exactly. A transform that rounded would make two abstract
    values share one encoding, so a decoder could not recover which was sent — and an encoding
    rule that cannot be inverted is not an encoding rule.
    """

    offset: int = 0
    scale: int = 1

    def __post_init__(self) -> None:
        if self.scale == 0:
            raise Asn1Error("ECN: an INT-TO-INT transform cannot scale by zero")

    def apply(self, value: int) -> int:
        shifted = value - self.offset
        if shifted % self.scale:
            raise Asn1Error(
                f"ECN: {value} is not expressible under this INT-TO-INT transform: "
                f"({value} - {self.offset}) is not a multiple of {self.scale}, so the "
                f"transform would not be invertible for it")
        return shifted // self.scale

    def inverse(self, value: int) -> int:
        return value * self.scale + self.offset


@dataclass(frozen=True)
class IntToBits(Transform):
    """`INT-TO-BITS`: an integer as a fixed-width bit field, so a later object sees bits."""

    width: int = 0

    def apply(self, value: int) -> tuple[int, ...]:
        if value < 0 or (self.width and value >> self.width):
            raise Asn1Error(
                f"ECN: {value} does not fit an INT-TO-BITS transform of width {self.width}")
        return tuple((value >> shift) & 1 for shift in range(self.width - 1, -1, -1))

    def inverse(self, value: tuple[int, ...]) -> int:
        out = 0
        for bit in value:
            out = (out << 1) | (1 if bit else 0)
        return out


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
    justification: Justification = Justification.RIGHT
    transform: Transform | None = None
    #: Octet-align before writing this field. Legacy headers align in places no standard
    #: rule does, which is one of the things the fixed set cannot be talked into.
    align_before: bool = False

    def write(self, value: int, out: BitWriter) -> None:
        if self.align_before:
            out.align()
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
        # Justification only bites when a value is narrower than its space; RIGHT is the
        # ordinary case and LEFT is what a field padded on the low side looks like.
        if self.justification is Justification.LEFT:
            used = max(encoded.bit_length(), 1)
            encoded <<= (self.width - used)
        out.put_bits(encoded, self.width)


@dataclass(frozen=True)
class BoolSpec:
    """A user-defined `#BOOL`: one bit, with the true/false patterns stated.

    DER writes a whole octet and CER/DER fix `TRUE` at `0xFF`; PER writes one bit. A header
    with an active-low flag matches neither, and says so here.
    """

    width: int = 1
    true_value: int = 1
    false_value: int = 0
    align_before: bool = False

    def write(self, value: bool, out: BitWriter) -> None:
        if self.align_before:
            out.align()
        out.put_bits(self.true_value if value else self.false_value, self.width)


@dataclass(frozen=True)
class PadSpec:
    """A user-defined `#PAD`: bits that carry no abstract value.

    `#PAD` is one of part one's primitive classes and has no built-in object, because none of
    BER/PER needs one. A fixed-layout header does: reserved bits are part of the octets and
    part of nothing else.
    """

    width: int
    value: int = 0

    def write(self, _value, out: BitWriter) -> None:
        out.put_bits(self.value, self.width)


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
    """

    #: Pad the completed encoding up to this boundary. 8 makes the result whole octets.
    boundary_bits: int = 8
    pad_value: int = 0

    def finish(self, out: BitWriter) -> None:
        out.align(self.boundary_bits, self.pad_value)


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
            "payloadOctets": IntSpec(width=4, transform=IntToInt(offset=0, scale=4,
                                                                 name="OCTETS-TO-UNITS")),
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
    "BitWriter", "BoolSpec", "ConcatenationSpec", "FIXED_CANDIDATES", "IntForm", "IntSpec",
    "IntToBits", "IntToInt", "Justification", "OuterSpec", "PadSpec", "Transform",
    "UserEncodingObject", "encode_with_user", "legacy_frame_objects",
    "legacy_frame_workload", "refuted_by",
]
