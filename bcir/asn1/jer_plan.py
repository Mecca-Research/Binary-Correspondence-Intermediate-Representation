"""J2 — the schema-plan compiler: a deterministic descriptor for a JER root type.

`docs/BCIR_ASN1_JSON_ROADMAP.md` §5.1 asks the X.680 front end to compile each supported
root type into a descriptor carrying schema identity, the JER family/profile/instruction
hash, member dispatch tables, required/default/extension metadata, the duplicate policy,
constraints and open-type selectors, recursion bounds, statically derivable capacity, and a
compiler identity — with three properties: "Repeated compilation is byte-identical. Unknown
required descriptor features fail closed. Descriptors are data; they contain no process
pointers or executable callbacks when serialized."

This module is that compiler, and it exists to be handed to J3: a C twin cannot walk a
Python type model, but it can walk a table. Everything a decode needs is resolved here,
once, so the hot path is a lookup rather than a traversal of `schema.py`.

**THE BOUND DERIVATION FINDING, because it changes what J3 can assume.** §5.1 asks for
"exact scratch/output upper bounds where statically derivable", and for JER the honest
answer is *almost nowhere*. X.697 §7.2.2 l) makes an integer's value constraint invisible,
§7.2.2 h) does the same for a SIZE on an octet or character string, and §7.2.2 g) removes
any extensible constraint — so a schema that pins `INTEGER (0..255)` and `OCTET STRING
(SIZE (4))` tells a JER encoder *nothing at all* about width. Only four things end up
bounded: BOOLEAN (5 octets), NULL (4), ENUMERATED (its longest identifier), and a BIT STRING
whose effective size constraint is a single value, which is the one constraint §7.2.1 a)
lets through. Everything else is `None`, and a container is bounded only if every member is.

That is not a gap in this compiler; it is what JER *is*, and it is the argument for J1's
runtime limits. A binary rail can size a buffer from the schema. A JER rail cannot, so it
must be told. `bounded_octets` reports `None` rather than a guess, and J3's C interface
takes its capacity from the caller for exactly this reason.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from .jer import (
    Array,
    Base64,
    JerRules,
    Name,
    ObjectAs,
    Text,
    Unwrapped,
    _bitstring_size,
    _flatten,
    _json_kinds,
    _member_name,
    _Opts,
)
from .jer_bounded import JerErrorCode, JerLimits, _fail, decode_bounded
from .schema import Asn1Type, Choice, OpenType, Primitive, Sequence, SequenceOf, Set, SetOf
from .tags import Asn1Error, Universal

#: §5.1's "deterministic schema-plan version and compiler identity". Both are serialized,
#: so a descriptor produced by a different compiler is recognisable as such rather than
#: silently trusted.
PLAN_VERSION: int = 1
PLAN_COMPILER: str = "bcir-asn1c/jer-plan/1"

#: §5.3's family/profile naming, which the MLIR attribute will mirror.
FAMILY: str = "jer"
PROFILE_BASIC: str = "basic"
PROFILE_CANONICAL: str = "bcir_canonical_v1"


@dataclass(frozen=True)
class PlanMember:
    """One member of a sequence, set or choice, with everything a dispatch needs."""

    #: The JSON member name, *after* any NAME instruction (§16.1.4). This is what a
    #: decoder matches on, which is why the rename is resolved at compile time.
    name: str
    #: The ASN.1 component identifier, kept so a decoded value keys by the schema's name
    #: rather than the wire's.
    identifier: str
    index: int
    required: bool
    has_default: bool
    extension: bool
    node: "PlanNode"


@dataclass(frozen=True)
class PlanNode:
    """The compiled form of one type."""

    kind: str
    #: Which JSON value kinds this node can produce (X.697 §19.2.2's discriminator). A
    #: decoder uses it to reject a wrongly-shaped value before looking at content, and an
    #: unwrapped choice uses it to identify the alternative at all.
    json_kinds: tuple[str, ...]
    members: tuple[PlanMember, ...] = ()
    element: "PlanNode | None" = None
    #: name -> member index, in name order. Sorted so the serialization is deterministic
    #: even though `members` is in schema order; a C twin can binary-search it.
    dispatch: tuple[tuple[str, int], ...] = ()
    #: §5.1's "duplicate policy". One value today, named rather than implied so a future
    #: relaxation is a descriptor change a reader can see.
    duplicate_policy: str = "refuse"
    instructions: tuple[str, ...] = ()
    #: The JER-*visible* constraint only (§7.2.1). Rendered as text: a descriptor is data.
    constraint: str = ""
    extensible: bool = False
    #: §5.1's "recursion bounds": the deepest JSON nesting a value of this type can reach.
    max_depth: int = 1
    #: §5.1's "exact scratch/output upper bounds where statically derivable", or None. See
    #: the module docstring for why None is the usual answer.
    bounded_octets: int | None = None
    enumeration: tuple[str, ...] = ()


@dataclass(frozen=True)
class JerSchemaPlan:
    """§5.1's descriptor. Data only: no callables, no live type references."""

    module: str
    type_name: str
    source_sha256: str
    family: str
    profile: str
    instruction_hash: str
    root: PlanNode
    plan_version: int = PLAN_VERSION
    compiler: str = PLAN_COMPILER
    #: §5.1's "optional direct-builder contract for a named BCIR consumer".
    direct_builder: str = ""
    #: Carried so a plan-driven decode can reach the live type model. Excluded from the
    #: serialization -- it is a process pointer, and §5.1 forbids those in a descriptor.
    _kind: Asn1Type | None = field(default=None, repr=False, compare=False)
    _instructions: object | None = field(default=None, repr=False, compare=False)

    def serialize(self) -> bytes:
        """A canonical byte form. Field order is declared, never sorted or reflected.

        Written by hand rather than delegated so byte-identity is by construction rather
        than by hoping a general serializer is stable across versions -- §5.1 requires
        "Repeated compilation is byte-identical", and a descriptor whose bytes move breaks
        every hash that names it.
        """
        out: list[str] = [
            f"plan-version {self.plan_version}",
            f"compiler {self.compiler}",
            f"module {self.module}",
            f"type {self.type_name}",
            f"source-sha256 {self.source_sha256}",
            f"family {self.family}",
            f"profile {self.profile}",
            f"instruction-sha256 {self.instruction_hash}",
            f"direct-builder {self.direct_builder}",
        ]
        _serialize_node(self.root, "", out)
        return ("\n".join(out) + "\n").encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.serialize()).hexdigest()


def _serialize_node(node: PlanNode, path: str, out: list[str]) -> None:
    here = path or "."
    out.append(
        f"node {here} kind={node.kind} json={'|'.join(node.json_kinds)} "
        f"depth={node.max_depth} octets={'?' if node.bounded_octets is None else node.bounded_octets} "
        f"dup={node.duplicate_policy} ext={'1' if node.extensible else '0'} "
        f"instr={'|'.join(node.instructions)} constraint={node.constraint or '-'} "
        f"enum={'|'.join(node.enumeration) or '-'}"
    )
    for name, index in node.dispatch:
        out.append(f"dispatch {here} {name} {index}")
    for member in node.members:
        out.append(
            f"member {here} {member.index} name={member.name} id={member.identifier} "
            f"req={'1' if member.required else '0'} "
            f"def={'1' if member.has_default else '0'} "
            f"ext={'1' if member.extension else '0'}"
        )
        _serialize_node(member.node, f"{here}/{member.name}", out)
    if node.element is not None:
        _serialize_node(node.element, f"{here}[]", out)


# --- compilation --------------------------------------------------------------------------

_PRIMITIVE_KIND = {
    Universal.BOOLEAN: "boolean",
    Universal.INTEGER: "integer",
    Universal.ENUMERATED: "enumerated",
    Universal.REAL: "real",
    Universal.NULL: "null",
    Universal.BIT_STRING: "bitstring",
    Universal.OCTET_STRING: "octetstring",
    Universal.OBJECT_IDENTIFIER: "oid",
    Universal.RELATIVE_OID: "relative-oid",
}


def _instruction_names(opts: _Opts, target) -> tuple[str, ...]:
    if opts.instructions is None:
        return ()
    found = []
    for category, label in (
        (Array, "ARRAY"),
        (Base64, "BASE64"),
        (Name, "NAME"),
        (ObjectAs, "OBJECT"),
        (Text, "TEXT"),
        (Unwrapped, "UNWRAPPED"),
    ):
        if opts.instructions.get(target, category) is not None:
            found.append(label)
    return tuple(found)


def _bounded(node_kind: str, kind, opts: _Opts) -> int | None:
    """§5.1's static output bound, or None. See the module docstring for why None wins.

    The four bounded cases are exactly the ones whose width JER does not take from a
    constraint it cannot see: a boolean and a null have fixed spellings, an enumerated is
    one of a known set of identifiers, and a bitstring is the single type whose SIZE §7.2.1 a)
    admits.
    """
    if node_kind == "boolean":
        return 5  # `false`
    if node_kind == "null":
        return 4  # `null`
    if node_kind == "enumerated" and kind.enumeration:
        text = _instruction_names(opts, kind)
        if "TEXT" in text:
            return None  # a TEXT rewrite is unbounded
        return max(len(name) for name, _n in kind.enumeration) + 2
    if node_kind == "bitstring" and kind.contains is None:
        low, high = _bitstring_size(kind)
        if low is not None and low == high:  # §24.2, a fixed-size string
            return 2 + 2 * ((low + 7) // 8)
    return None


def _compile_node(kind: Asn1Type, opts: _Opts, path: str, depth: int) -> PlanNode:
    if depth > 64:
        raise _fail(
            JerErrorCode.SCHEMA,
            path=path,
            detail="schema recursion beyond the 5.1 recursion bound of 64",
        )
    if isinstance(kind, OpenType):
        # Decided before `_json_kinds`, which refuses an open type with §19.2.4's message
        # about unwrapped choices -- true in that context and misleading in this one. The
        # reason a *plan* cannot hold an open type is §41: it encodes AS its contained
        # type, and the table (X.682 §10.19) picks that from a sibling's value at decode
        # time, which a static descriptor does not have. JER offers no hexadecimal fallback
        # the way XER's §8.5 does, so this is fail-closed per §5.1 rather than a gap.
        raise _fail(
            JerErrorCode.SCHEMA,
            path=path,
            detail="41 encodes an open type AS its contained type, which is "
            "chosen by a sibling's value at decode time; a static plan "
            "cannot name it",
        )
    kinds = tuple(sorted(_json_kinds(kind, opts)))
    instructions = _instruction_names(opts, kind)

    if isinstance(kind, Primitive):
        node_kind = _PRIMITIVE_KIND.get(kind.universal, "string")
        if node_kind == "enumerated" and not kind.enumeration:
            # Fail closed (§5.1): X.697 §22.2 encodes the identifier, which does not exist.
            raise _fail(
                JerErrorCode.SCHEMA,
                path=path,
                detail="an ENUMERATED with no enumeration has no JER spelling "
                "(22.2); the plan refuses rather than deferring the fault",
            )
        constraint = ""
        if node_kind == "bitstring":
            low, high = _bitstring_size(kind)
            constraint = f"size={low}..{high}" if low is not None else "size=variable"
        if kind.contains is not None and kind.encoded_by is None:
            constraint = (constraint + ";" if constraint else "") + "containing"
        return PlanNode(
            node_kind,
            kinds,
            instructions=instructions,
            constraint=constraint,
            bounded_octets=_bounded(node_kind, kind, opts),
            enumeration=tuple(name for name, _n in (kind.enumeration or ())),
        )

    if isinstance(kind, (Sequence, Set)):
        members: list[PlanMember] = []
        deepest = 0
        total: int | None = 2  # the braces
        for index, comp in enumerate(_flatten(kind.components)):
            child = _compile_node(comp.type, opts, f"{path}/{comp.name}", depth + 1)
            name = _member_name(opts, comp)
            members.append(
                PlanMember(
                    name,
                    comp.name,
                    index,
                    not (comp.optional or comp.has_default),
                    comp.has_default,
                    comp.extension,
                    child,
                )
            )
            deepest = max(deepest, child.max_depth)
            if total is not None and child.bounded_octets is not None:
                total += child.bounded_octets + len(name) + 4  # "name": plus a comma
            else:
                total = None
        dispatch = tuple(sorted((m.name, m.index) for m in members))
        if len({name for name, _i in dispatch}) != len(dispatch):
            raise _fail(
                JerErrorCode.SCHEMA,
                path=path,
                detail="two members share a JSON name after NAME instructions (16.2)",
            )
        node_kind = (
            "array" if "ARRAY" in instructions else ("set" if isinstance(kind, Set) else "sequence")
        )
        return PlanNode(
            node_kind,
            kinds,
            tuple(members),
            None,
            dispatch,
            instructions=instructions,
            extensible=kind.extensible,
            max_depth=deepest + 1,
            bounded_octets=total,
        )

    if isinstance(kind, (SequenceOf, SetOf)):
        element = _compile_node(kind.element, opts, f"{path}[]", depth + 1)
        node_kind = (
            "object-map"
            if "OBJECT" in instructions
            else ("set-of" if isinstance(kind, SetOf) else "sequence-of")
        )
        # No bound: §7.2.2 h) hides a SIZE from JER, so the occurrence count is unbounded
        # as far as an encoder can see.
        return PlanNode(
            node_kind,
            kinds,
            element=element,
            instructions=instructions,
            max_depth=element.max_depth + 1,
        )

    if isinstance(kind, Choice):
        members = []
        deepest = 0
        for index, alt in enumerate(_flatten(kind.alternatives)):
            child = _compile_node(alt.type, opts, f"{path}/{alt.name}", depth + 1)
            members.append(
                PlanMember(
                    _member_name(opts, alt), alt.name, index, False, False, alt.extension, child
                )
            )
            deepest = max(deepest, child.max_depth)
        dispatch = tuple(sorted((m.name, m.index) for m in members))
        unwrapped = "UNWRAPPED" in instructions
        if unwrapped:
            # §19.2.2 at compile time rather than at decode time: a plan is a contract, and
            # a contract that cannot be decoded is better refused when it is written.
            seen: dict[str, str] = {}
            for member in members:
                for shape in member.node.json_kinds:
                    if shape == "object":
                        continue
                    if shape in seen:
                        raise _fail(
                            JerErrorCode.SCHEMA,
                            path=path,
                            detail=f"19.2.2 admits at most one alternative producing a "
                            f"JSON {shape}; {seen[shape]} and {member.identifier} "
                            f"both do, so this choice cannot carry UNWRAPPED",
                        )
                    seen[shape] = member.identifier
        return PlanNode(
            "unwrapped-choice" if unwrapped else "choice",
            kinds,
            tuple(members),
            None,
            dispatch,
            instructions=instructions,
            extensible=kind.extensible,
            max_depth=deepest + (0 if unwrapped else 1),
        )

    raise _fail(
        JerErrorCode.SCHEMA,
        path=path,
        detail=f"{type(kind).__name__} has no JER plan representation",
    )


def compile_plan(
    kind: Asn1Type,
    *,
    module: str,
    type_name: str,
    source: str | bytes = b"",
    rules: JerRules = JerRules.CANONICAL,
    instructions=None,
    direct_builder: str = "",
) -> JerSchemaPlan:
    """Compile one root type into a §5.1 descriptor.

    `source` is the ASN.1 module text, hashed into the descriptor so a plan names the
    schema it came from rather than merely a type's shape — two modules can define the same
    structure and mean different things.
    """
    octets = source.encode("utf-8") if isinstance(source, str) else bytes(source)
    opts = _Opts(rules, instructions)
    root = _compile_node(kind, opts, "", 0)
    return JerSchemaPlan(
        module=module,
        type_name=type_name,
        source_sha256=hashlib.sha256(octets).hexdigest(),
        family=FAMILY,
        profile=PROFILE_CANONICAL if rules is JerRules.CANONICAL else PROFILE_BASIC,
        instruction_hash=_instruction_hash(root),
        root=root,
        direct_builder=direct_builder,
        _kind=kind,
        _instructions=instructions,
    )


def _instruction_hash(root: PlanNode) -> str:
    """A hash over every instruction the plan resolved, in tree order.

    §5.1 wants the "instruction set, and instruction hash" in the descriptor. Hashing the
    *resolved* instructions rather than the assignment calls is what makes it meaningful:
    two different assignment orders that produce the same final set (§13.3) hash the same,
    which is precisely the equivalence §7.5.4 says holds.
    """
    parts: list[str] = []

    def walk(node: PlanNode, path: str) -> None:
        if node.instructions:
            parts.append(f"{path}={'|'.join(node.instructions)}")
        for member in node.members:
            walk(member.node, f"{path}/{member.name}")
        if node.element is not None:
            walk(node.element, f"{path}[]")

    walk(root, "")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# --- the table-driven decode, and its trace ------------------------------------------------


def trace_of(plan: JerSchemaPlan, value) -> tuple[str, ...]:
    """The event trace a decode of `value` under `plan` visits, in order.

    §5.2 requires a generated fixed-schema wrapper to "produce the same event trace and
    diagnostics as the table-driven scalar implementation". This is the J2 form of that
    contract: the plan-driven walk and the direct decode must visit the same members in the
    same order, so a J3 wrapper has something concrete to be equal to.
    """
    events: list[str] = []
    _trace(plan.root, value, "", events)
    return tuple(events)


def _trace(node: PlanNode, value, path: str, events: list[str]) -> None:
    here = path or "."
    events.append(f"enter {here} {node.kind}")
    if node.kind in ("sequence", "set", "array"):
        for member in node.members:
            if member.identifier in value:
                events.append(f"member {here}/{member.name}")
                _trace(member.node, value[member.identifier], f"{here}/{member.name}", events)
    elif node.kind in ("sequence-of", "set-of", "object-map"):
        for index, item in enumerate(value):
            events.append(f"element {here}[{index}]")
            _trace(node.element, item, f"{here}[]", events)
    elif node.kind in ("choice", "unwrapped-choice"):
        chosen, payload = value
        member = next(m for m in node.members if m.identifier == chosen)
        events.append(f"alternative {here}/{member.name}")
        _trace(member.node, payload, f"{here}/{member.name}", events)
    else:
        events.append(f"value {here} {node.kind}")
    events.append(f"leave {here}")


def decode_with_plan(plan: JerSchemaPlan, data: bytes | str, *, limits: JerLimits = JerLimits()):
    """Decode through the plan, returning `(value, trace)`.

    The plan carries the dispatch tables a J3 twin will walk; this rail still delegates the
    lexical work to `decode_bounded`, because J2's contract is the *descriptor* and J3's is
    the parser. What is proven here is that the descriptor is sufficient — the trace is
    produced from the plan alone, and `test_asn1_jer_plan.py` requires it to equal the one
    the direct decode produces.
    """
    if plan._kind is None:
        raise _fail(
            JerErrorCode.SCHEMA,
            detail="this plan was deserialized and carries no type model; a "
            "descriptor is data (5.1) and cannot decode by itself",
        )
    rules = JerRules.CANONICAL if plan.profile == PROFILE_CANONICAL else JerRules.BASIC
    value = decode_bounded(
        data, plan._kind, rules=rules, limits=limits, instructions=plan._instructions
    )
    return value, trace_of(plan, value)


__all__ = [
    "FAMILY",
    "PLAN_COMPILER",
    "PLAN_VERSION",
    "PROFILE_BASIC",
    "PROFILE_CANONICAL",
    "JerSchemaPlan",
    "PlanMember",
    "PlanNode",
    "compile_plan",
    "decode_with_plan",
    "trace_of",
]
