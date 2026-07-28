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
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .schema import Asn1Type, Choice, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, TagClass, Universal

PLAN_VERSION = 1
PLAN_COMPILER = "bcir-encode-plan/1"

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
class EncodeNode:
    """The compiled write-side form of one type."""

    kind: str
    #: The base universal tag number, which DER needs and JER never mentions.
    universal: int = 0
    members: tuple[EncodeMember, ...] = ()
    element: "EncodeNode | None" = None
    enumeration: tuple[str, ...] = ()
    type_name: str = ""


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
    out.append(
        f"node {here} kind={node.kind} universal={node.universal} "
        f"type={node.type_name or '-'} enum={'|'.join(node.enumeration) or '-'}")
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
    if isinstance(kind, Primitive):
        leaf = _LEAF_KIND.get(kind.universal)
        if leaf is None:
            raise Asn1Error(
                f"{path}: the write plan has no leaf rule for universal tag "
                f"{kind.universal} ({kind.name}); refusing rather than guessing a spelling "
                f"that three encoders would each get differently")
        return EncodeNode(kind=leaf, universal=int(kind.universal), type_name=kind.name)
    if isinstance(kind, Sequence):
        return EncodeNode(kind="sequence", universal=int(Universal.SEQUENCE),
                          type_name=getattr(kind, "name", "") or "",
                          members=_compile_members(kind, path, depth))
    if isinstance(kind, SequenceOf):
        return EncodeNode(kind="sequence-of", universal=int(Universal.SEQUENCE),
                          type_name=getattr(kind, "name", "") or "",
                          element=_compile_node(kind.element, f"{path}[]", depth + 1))
    if isinstance(kind, Choice):
        return EncodeNode(kind="choice", type_name=getattr(kind, "name", "") or "",
                          members=_compile_members(kind, path, depth))
    raise Asn1Error(
        f"{path}: the write plan does not compile {type(kind).__name__}; SET, SET OF and "
        f"OPEN TYPE each need a rule of their own (X.690 §11.6 orders a SET's components by "
        f"tag, and X.697 §41 gives an open type no JER fallback), and inventing one here "
        f"would produce an emitter that silently disagrees with the oracle")


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
    "PLAN_COMPILER", "PLAN_VERSION", "EncodeMember", "EncodeNode", "EncodePlan",
    "compile_encode_plan",
]
