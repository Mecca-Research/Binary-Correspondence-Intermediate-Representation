"""E1 — the write-side schema plan, and a plan-driven emitter for every candidate.

#682 established, against the oracle's own encoders, that a **schema-free** encode column is
not the decode column's mirror: `encode_der` takes a value and no type, and every other
encoder takes the type first, because X.697 §22.2 puts member *identifiers* in a JER document
and an identifier exists only in the schema. So a schema-free harness would yield a two-row
table with JER missing — an unfinished implementation, not the law it is.

The fix is to make **every** emitter schema-directed, so the work compared is the same work.
This module is that: one plan, one value stream, N emitters.

**Why not reuse `jer_plan`.** J2's plan was compiled for *reading* JER. It carries
`json_kinds` discriminators and dispatch tables a decoder needs, and it does not carry ASN.1
tags — a JER document has no tags, so a JER read plan had no reason to. DER and OER cannot be
emitted without them. Extending the read plan would also move `instruction_hash` and its
`sha256`, which name landed artifacts. A write plan is a different compilation of the same
schema, and saying so is cheaper than pretending one plan serves both directions.

**The value stream is the load-bearing idea.** Emitters cannot be compared while they consume
different inputs: handing DER its own octets and JER a Python object measures the adapters.
So a value is flattened, *against the plan*, into a format-neutral stream — presence bytes,
counts, choice indices and abstract leaves — and every emitter consumes exactly that. No
candidate is privileged, and the input is identical by construction rather than by being
"the same document in different encodings".

The stream, in plan traversal order::

    boolean      1 octet, 0 or 1
    integer      <len:u8><two's-complement big-endian, minimal>   the abstract integer
    enumerated   as integer
    null         nothing at all
    octetstring  <len:u32><octets>
    string       <len:u32><UTF-8>
    oid          <len:u32><dotted ASCII>      text, so JER needs no base-128 decode
    sequence     per component: a presence octet iff OPTIONAL or DEFAULT, then the value
    sequence-of  <count:u32> then that many elements
    choice       <index:u32> then the chosen alternative

**Refusal is by construction.** A construct the plan compiler does not understand raises at
*compile* time with the clause that governs it. A plan that silently skipped a construct
would produce an emitter disagreeing with the oracle, and the parity test would report that
as an unexplained byte difference rather than as the missing feature it is.

**Version 3 records subtype constraints, and that is a bug fix rather than a feature.**
Version 2 dropped them, which is harmless for DER, BER and JER — X.690 encodes the value
either way and X.697 §7.2.2 l)/h) hide integer and string constraints from JER outright —
and *wrong* for OER, where X.696 §10.3 gives a constrained INTEGER a fixed-width form with
no length determinant. The version 2 OER emitter therefore wrote the unconstrained spelling
for every type: a well-formed document of a different value, which every parity test passed
over because the corpus contained no constrained type. `EncodeConstraint` says what each
rule reads; see its docstring for why the OER and PER bound pairs are separate facts.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .constraints import (
    Constraint, effective_size_constraint, effective_value_constraint, root_alphabet,
    root_size_bounds, root_value_bounds,
)
from .schema import Asn1Type, Choice, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, TagClass, Universal

PLAN_VERSION = 4
PLAN_COMPILER = "bcir-encode-plan/1"

#: The widest bound this plan will record, both signs. A descriptor is read by a
#: freestanding C twin with no bignum, so the format states its arithmetic range rather than
#: letting one rail silently truncate what the other wrote. 2**64-1 is not an arbitrary
#: ceiling: it is exactly X.696 §10.3 d)'s widest fixed word, so every bound that selects an
#: OER width is representable and anything past it falls to the length-prefixed form anyway.
BOUND_MAX = (1 << 64) - 1

#: The longest permitted alphabet recorded, in UTF-8 octets. X.691 §30.5's alphabet is a
#: per-node fixed buffer in the C twin, so §5.1's "state your scratch bound" applies. 128
#: covers IA5String's whole repertoire, which is the largest a known-multiplier type has.
ALPHABET_MAX = 128

#: Universal tag -> the plan's leaf kind. Written out rather than derived so an unlisted
#: type is a refusal naming itself, not a default that encodes wrongly.
_LEAF_KIND: dict[int, str] = {
    Universal.BOOLEAN: "boolean",
    Universal.INTEGER: "integer",
    Universal.ENUMERATED: "enumerated",
    Universal.NULL: "null",
    Universal.OCTET_STRING: "octetstring",
    Universal.OBJECT_IDENTIFIER: "oid",
    Universal.UTF8_STRING: "string",
    Universal.PRINTABLE_STRING: "string",
    Universal.IA5_STRING: "string",
    Universal.NUMERIC_STRING: "string",
    Universal.VISIBLE_STRING: "string",
}

_TAG_CLASS_NAME = {
    TagClass.UNIVERSAL: "universal", TagClass.APPLICATION: "application",
    TagClass.CONTEXT: "context", TagClass.PRIVATE: "private",
}
_TAG_CLASS_BY_NAME = {name: value for value, name in _TAG_CLASS_NAME.items()}


@dataclass(frozen=True)
class EncodeMember:
    """One component, with everything all three emitters need from it.

    `name` and `identifier` are both kept for the same reason J2 keeps both: X.697 §16.1.4's
    NAME instruction can rename a member on the wire, and a JER document must carry the
    renamed spelling while the value keys by the schema's.
    """

    name: str
    identifier: str
    index: int
    optional: bool
    has_default: bool
    tag: int | None
    tag_class: str
    explicit: bool
    node: "EncodeNode"


@dataclass(frozen=True)
class EncodeConstraint:
    """What the encoding rules read out of a subtype constraint — recorded, not re-derived.

    A constraint restricts a value SET. DER, BER and JER encode a value identically whether
    or not one exists, and X.697 §7.2.2 l)/h) hide integer and string constraints from JER
    outright. **OER and PER do not**: X.696 §10.3 gives a constrained INTEGER a fixed-width
    form with no length determinant, so `INTEGER (0..255)` holding 42 is `2A` where the
    unconstrained type is `01 2A`. Version 2 of this plan dropped constraints, and the OER
    emitter it drove therefore emitted the unconstrained spelling for every type — a
    well-formed document of a different value, hidden by a corpus with no constrained type.

    **Four bound pairs, not two, because the two rules disagree about extensibility.**
    X.696 §8.2.2 g) makes an extensible constraint invisible to OER: the marker says the
    value set may grow, so sizing a field from today's bounds would emit octets a future
    peer cannot read. X.691 §13.1/§17.3/§20.4/§30.4 make the opposite choice — one bit
    saying whether the value is inside the extension root, then the root's own width. So the
    OER-effective bounds and the PER root bounds are different facts about the same
    constraint, and they genuinely differ: intersecting an extensible `(0..255, ...)` with a
    plain `(0..1000)` leaves OER reading `0..1000` and PER's root reading `0..255`.

    Deriving one pair from the other would therefore be a guess. Both are recorded, which is
    the same discipline the rest of this compiler follows: a descriptor states what each rule
    reads, and a rule this plan cannot state is refused rather than approximated.
    """

    #: X.696 §8.2.7 / §8.2.8 — what OER reads. `None` is "no finite bound".
    value_low: int | None = None
    value_high: int | None = None
    size_low: int | None = None
    size_high: int | None = None
    #: X.691 §13.1 / §17.3 — the extension ROOT's bounds, which PER encodes against.
    root_value_low: int | None = None
    root_value_high: int | None = None
    root_size_low: int | None = None
    root_size_high: int | None = None
    #: X.691 §13.1's extension bit: whether one is emitted at all, per dimension.
    value_extensible: bool = False
    size_extensible: bool = False
    #: X.691 §30.5's permitted alphabet in canonical order, or "" when unrestricted. Empty
    #: rather than None because a *constraint* that permits no character at all is
    #: unsatisfiable and `require_satisfiable` refuses it before this is ever built.
    alphabet: str = ""

    def is_trivial(self) -> bool:
        """True when this records nothing an encoder would read.

        A node with a trivial constraint serializes no constraint line, so a plan for an
        unconstrained schema is byte-identical to what version 2 wrote apart from its
        version. That keeps the format's cost proportional to what it actually says.
        """
        return self == EncodeConstraint()


@dataclass(frozen=True)
class EncodeNode:
    """The compiled write-side form of one type."""

    kind: str
    #: The base universal tag number, which DER needs and JER never mentions.
    universal: int = 0
    members: tuple[EncodeMember, ...] = ()
    element: "EncodeNode | None" = None
    #: The enumeration ROOT as `(identifier, number)` pairs, in source order. **Both halves
    #: are load-bearing and to different rules.** X.690 §8.4 and X.696 §11 encode the
    #: *number*, which the value stream already carries. X.697 §22.2 encodes the
    #: *identifier* — "the identifier of the chosen enumeration item" — and says so
    #: explicitly because it cannot be derived from the number. X.691 §14.1 encodes neither:
    #: it encodes the value's INDEX into the root sorted ascending. One field, three
    #: readings, and a plan that carried only names could not serve the third.
    enumeration: tuple[tuple[str, int], ...] = ()
    type_name: str = ""
    #: None where the type carries no encoding-visible constraint. Version 3 is this field.
    constraint: EncodeConstraint | None = None
    #: X.680 §25.4's `...` on a SEQUENCE or CHOICE, or X.680 §20.4's on an ENUMERATED.
    #: X.691 §19.1, §23.5 and §14.3 each emit a leading bit for it, so two schemas differing
    #: only here encode differently under PER and identically everywhere else. Version 4.
    extensible: bool = False


@dataclass(frozen=True)
class EncodePlan:
    """A write-side descriptor. Data only — no callables, no live type references.

    The `_kind` back-reference exists so parity tests can call the oracle with the same
    type; it is excluded from the serialization for §5.1's reason, that a descriptor must
    contain no process pointers.
    """

    module: str
    type_name: str
    source_sha256: str
    root: EncodeNode
    plan_version: int = PLAN_VERSION
    compiler: str = PLAN_COMPILER
    _kind: Asn1Type | None = field(default=None, repr=False, compare=False)

    def serialize(self) -> bytes:
        """A canonical byte form, written by hand so byte-identity is by construction."""
        out: list[str] = [
            f"plan-version {self.plan_version}",
            f"compiler {self.compiler}",
            f"module {self.module}",
            f"type {self.type_name}",
            f"source-sha256 {self.source_sha256}",
        ]
        _serialize_node(self.root, "", out)
        return ("\n".join(out) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()


def _serialize_node(node: EncodeNode, path: str, out: list[str]) -> None:
    here = path or "."
    # `members` and `element` are counts, and E2's freestanding reader is why. The node
    # lines arrive depth first, so a reader without counts must rebuild the tree from the
    # PATH strings — and a member name containing `/` splits a path in the wrong place.
    # Counting makes the descriptor parseable with a fixed stack and no string arithmetic.
    # Version 2 was exactly this addition; version 3 adds the optional `constraint` line
    # below, which a node emits only when it has something an encoder would read.
    enumeration = "|".join(f"{name}:{number}" for name, number in node.enumeration)
    out.append(
        f"node {here} kind={node.kind} universal={node.universal} "
        f"members={len(node.members)} element={'1' if node.element is not None else '0'} "
        f"type={node.type_name or '-'} enum={enumeration or '-'} "
        f"ext={'1' if node.extensible else '0'}")
    if node.constraint is not None:
        out.append(_serialize_constraint(node.constraint, here))
    for member in node.members:
        out.append(
            f"member {here} {member.index} name={member.name} id={member.identifier} "
            f"opt={'1' if member.optional else '0'} "
            f"def={'1' if member.has_default else '0'} "
            f"tag={'-' if member.tag is None else member.tag} "
            f"class={member.tag_class} exp={'1' if member.explicit else '0'}")
        _serialize_node(member.node, f"{here}/{member.name}", out)
    if node.element is not None:
        _serialize_node(node.element, f"{here}[]", out)


def _bound(value: int | None) -> str:
    """One bound, `-` for unbounded. Decimal with a leading `-` for negatives.

    Written as text rather than as a fixed word because the range a bound may take is a
    property of ASN.1, not of a machine: `INTEGER (0..18446744073709551615)` is X.696
    §10.3 d)'s eight-octet unsigned word, and its upper bound does not fit int64. The C
    reader parses sign and magnitude separately for exactly that reason.
    """
    return "-" if value is None else str(value)


def _serialize_constraint(constraint: EncodeConstraint, path: str) -> str:
    # The alphabet is hex-encoded UTF-8: it may contain a space, a `=`, or the `-` that
    # spells "absent" everywhere else in this format, and a quoting scheme the C reader has
    # to unpick is more surface than a fixed two-nibble encoding.
    alphabet = constraint.alphabet.encode("utf-8").hex() or "-"
    return (
        f"constraint {path} "
        f"vlo={_bound(constraint.value_low)} vhi={_bound(constraint.value_high)} "
        f"slo={_bound(constraint.size_low)} shi={_bound(constraint.size_high)} "
        f"rvlo={_bound(constraint.root_value_low)} rvhi={_bound(constraint.root_value_high)} "
        f"rslo={_bound(constraint.root_size_low)} rshi={_bound(constraint.root_size_high)} "
        f"vext={'1' if constraint.value_extensible else '0'} "
        f"sext={'1' if constraint.size_extensible else '0'} "
        f"alpha={alphabet}")


# --- compilation ------------------------------------------------------------------------------


def compile_encode_plan(kind: Asn1Type, *, module: str, type_name: str,
                        source: bytes = b"") -> EncodePlan:
    """Compile one type into a write-side plan, refusing what it cannot emit."""
    return EncodePlan(module=module, type_name=type_name,
                      source_sha256=hashlib.sha256(source).hexdigest(),
                      root=_compile_node(kind, type_name, 0), _kind=kind)


def _compile_node(kind: Asn1Type, path: str, depth: int) -> EncodeNode:
    if depth > 32:
        raise Asn1Error(
            f"{path}: the write plan refuses recursion beyond depth 32; a descriptor whose "
            f"depth is unbounded cannot state a scratch bound, and §5.1 requires one")
    constraint = _record_constraint(kind, path)
    if isinstance(kind, Primitive):
        leaf = _LEAF_KIND.get(kind.universal)
        if leaf is None:
            raise Asn1Error(
                f"{path}: the write plan has no leaf rule for universal tag "
                f"{kind.universal} ({kind.name}); refusing rather than guessing a spelling "
                f"that three encoders would each get differently")
        return EncodeNode(kind=leaf, universal=int(kind.universal), type_name=kind.name,
                          constraint=constraint,
                          enumeration=_record_enumeration(kind, path),
                          extensible=bool(getattr(kind, "enum_extensible", False)))
    if isinstance(kind, Sequence):
        return EncodeNode(kind="sequence", universal=int(Universal.SEQUENCE),
                          type_name=getattr(kind, "name", "") or "",
                          members=_compile_members(kind, path, depth),
                          constraint=constraint,
                          extensible=bool(getattr(kind, "extensible", False)))
    if isinstance(kind, SequenceOf):
        return EncodeNode(kind="sequence-of", universal=int(Universal.SEQUENCE),
                          type_name=getattr(kind, "name", "") or "",
                          element=_compile_node(kind.element, f"{path}[]", depth + 1),
                          constraint=constraint)
    if isinstance(kind, Choice):
        return EncodeNode(kind="choice", type_name=getattr(kind, "name", "") or "",
                          members=_compile_members(kind, path, depth),
                          constraint=constraint,
                          extensible=bool(getattr(kind, "extensible", False)))
    raise Asn1Error(
        f"{path}: the write plan does not compile {type(kind).__name__}; SET, SET OF and "
        f"OPEN TYPE each need a rule of their own (X.690 §11.6 orders a SET's components by "
        f"tag, and X.697 §41 gives an open type no JER fallback), and inventing one here "
        f"would produce an emitter that silently disagrees with the oracle")


def _record_constraint(kind, path: str) -> EncodeConstraint | None:
    """Project a live constraint onto the facts the encoding rules read.

    The constraint object itself is not stored: §5.1 makes a descriptor data, and a
    `Constraint` is a Python object graph with `permits()` on it. What OER and PER read out
    of one is four bound pairs, two extension flags and an alphabet — arithmetic, and
    therefore expressible in a text descriptor a freestanding reader can parse.

    Returns None when the projection is empty, so an unconstrained schema's plan is what
    version 2 wrote apart from its version line.
    """
    constraint = getattr(kind, "constraint", None)
    if not isinstance(constraint, Constraint):
        return None
    value_low, value_high = effective_value_constraint(constraint)
    size_low, size_high = effective_size_constraint(constraint)
    (root_value_low, root_value_high), value_extensible = root_value_bounds(constraint)
    (root_size_low, root_size_high), size_extensible = root_size_bounds(constraint)
    alphabet = root_alphabet(constraint)
    recorded = EncodeConstraint(
        value_low=value_low, value_high=value_high,
        size_low=size_low, size_high=size_high,
        root_value_low=root_value_low, root_value_high=root_value_high,
        root_size_low=root_size_low, root_size_high=root_size_high,
        value_extensible=value_extensible, size_extensible=size_extensible,
        alphabet="".join(sorted(alphabet)) if alphabet else "")
    for name in ("value_low", "value_high", "size_low", "size_high",
                 "root_value_low", "root_value_high", "root_size_low", "root_size_high"):
        bound = getattr(recorded, name)
        if bound is not None and abs(bound) > BOUND_MAX:
            raise Asn1Error(
                f"{path}: the constraint's {name} is {bound}, past the ±{BOUND_MAX} this "
                f"plan format records; X.696 §10.3 d)'s widest fixed word is eight octets, "
                f"so a bound beyond it selects the length-prefixed form anyway — but "
                f"writing it down would truncate in the C reader, and a silently truncated "
                f"bound is a different type")
    if len(recorded.alphabet.encode("utf-8")) > ALPHABET_MAX:
        raise Asn1Error(
            f"{path}: the permitted alphabet is {len(recorded.alphabet)} characters, past "
            f"the {ALPHABET_MAX}-octet buffer this plan format states; X.691 §30.5 makes "
            f"the alphabet decide bits-per-character, so a truncated one encodes a "
            f"different document rather than a slightly wrong one")
    return None if recorded.is_trivial() else recorded


def _record_enumeration(kind: Primitive, path: str) -> tuple[tuple[str, int], ...]:
    """The enumeration root, refusing an ENUMERATED that has none.

    A bare ENUMERATED is encodable under X.690 and X.696, which read the *number* the value
    already carries. It is NOT encodable under X.697 — §22.2 spells an enumerated value as
    "the identifier of the chosen enumeration item", and no identifier can be derived from a
    number — nor under X.691, whose §14.1 needs the whole root to compute an index. The
    oracle's `encode_jer` refuses it for exactly that reason.

    Version 3 compiled such a type happily and the JER emitter wrote the bare number: a
    document the oracle would not have produced and a JER decoder cannot map back. Refusing
    at compile time keeps the failure where the missing information is, rather than three
    emitters downstream.
    """
    if kind.universal != Universal.ENUMERATED:
        return ()
    enumeration = getattr(kind, "enumeration", None)
    if not enumeration:
        raise Asn1Error(
            f"{path}: ENUMERATED has no enumeration. X.690 §8.4 and X.696 §11 encode the "
            f"number, so this would look encodable — but X.697 §22.2 encodes the "
            f"IDENTIFIER and X.691 §14.1 encodes the index into the root, and neither can "
            f"be derived from a number alone. One plan drives all four, so it is refused "
            f"here rather than producing a JER document of the wrong shape")
    for name, _number in enumeration:
        if not name or any(character in name for character in "|: \n"):
            raise Asn1Error(
                f"{path}: the enumeration identifier {name!r} contains a character the "
                f"descriptor's `name:number|...` field uses as a separator; X.680 §12.4 "
                f"gives an identifier no such character, so this is a malformed schema "
                f"rather than a format limitation")
    return tuple((str(name), int(number)) for name, number in enumeration)


def _compile_members(kind, path: str, depth: int) -> tuple[EncodeMember, ...]:
    # A CHOICE names its arms `alternatives`, not `components`, because X.680 §29.1 gives a
    # CHOICE no tag of its own — the arms are not components of anything.
    parts = kind.alternatives if isinstance(kind, Choice) else kind.components
    out: list[EncodeMember] = []
    for index, component in enumerate(parts):
        if getattr(component, "extension", False):
            raise Asn1Error(
                f"{path}/{component.name}: an extension addition is refused by the write "
                f"plan; X.691 §19.7 splits root from additions and X.690 does not, so one "
                f"plan cannot describe both until PER joins this harness")
        # `Component.has_default` is the schema's own property against its `_NO_DEFAULT`
        # sentinel. Re-deriving it here from `default is not None` would call a component
        # whose declared default IS None undefaulted, and X.690 §11.5 makes that the
        # difference between emitting the component and omitting it.
        out.append(EncodeMember(
            name=component.name, identifier=component.name, index=index,
            optional=bool(getattr(component, "optional", False)),
            has_default=bool(component.has_default),
            tag=component.tag,
            tag_class=_TAG_CLASS_NAME[getattr(component, "tag_class", TagClass.CONTEXT)],
            explicit=bool(getattr(component, "explicit", False)),
            node=_compile_node(component.type, f"{path}/{component.name}", depth + 1)))
    return tuple(out)



__all__ = [
    "ALPHABET_MAX", "BOUND_MAX", "PLAN_COMPILER", "PLAN_VERSION", "EncodeConstraint",
    "EncodeMember", "EncodeNode", "EncodePlan", "compile_encode_plan",
]
