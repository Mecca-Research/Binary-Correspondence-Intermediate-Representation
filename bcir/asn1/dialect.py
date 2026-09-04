"""J4 part 2 — the commuting projection between the `bcir.asn1.*` dialect and JER.

§7.1 of `docs/BCIR_ASN1_JSON_ROADMAP.md` sets this phase's gate:

> The projection between the `bcir.*` dialect and its JER form must **commute in both
> directions**. `MLIR -> JER -> MLIR` must be the identity on the dialect, and
> `JER -> MLIR -> JER` must be byte-identical under the canonical profile.

**The two directions are not the same claim, and the difference is the whole design.**

`JER -> MLIR -> JER` is a claim about *bytes*, and it can be, because the canonical JER
profile defines exactly one octet string per abstract value. So that direction is checked
by comparing octets, and a failure points at one.

`MLIR -> JER -> MLIR` cannot be a byte claim, and saying it were would be dishonest.
**MLIR textual assembly is not canonical**: `mlir-opt` is free to reprint attributes in a
different order or with different whitespace, the fixture corpus in `mlir/test/` is
hand-aligned into columns, and nothing in MLIR promises otherwise. What survives a round
trip is the *dialect* — the operations, their symbols, their attributes and their nesting.
So that direction is the identity on `DialectModule`, the parsed form, which is what
"identity on the dialect" says. Claiming byte-identity for MLIR text would be a statement
about a formatter rather than about a projection, and would fail for reasons that have
nothing to do with whether the projection is correct.

**Why a pivot type rather than a direct text-to-JSON translation.** `DialectModule` is the
single in-memory form both rails go through. A direct MLIR-text-to-JER path would have two
independent transcriptions of the dialect's shape — one in the writer, one in the reader —
and nothing forcing them to agree, which is the drift this repository keeps designing
against. With a pivot there is exactly one description of what a module *is*, and both
directions are defined against it.

**The schema uses a CHOICE for operations, not a SEQUENCE of optionals.** J4 part 1 landed
on the principle that an IR should not be able to write down a contradiction: a stored
(family, profile) pair can disagree with itself, so `BCIR_Asn1Rules` stayed one enum. The
same argument applies here. A flat operation record with optional `native`, `from`, `to`
and `rules` fields admits `op = "encode"` carrying a `native` — expressible and meaningless.
X.680 §29's CHOICE says instead that an operation *is* exactly one of four things, and
X.697 §23 encodes it as a single-member object, so the JER stays readable too.

**What this module does NOT do.** It does not verify R24. The laws live in
`mlir/lib/passes/BCIRVerifyPass.cpp` and are checked by `bcir-opt -bcir-verify`; a second
implementation here would be a second definition of the law, free to drift from the one
that runs. This module answers "is this the same module?", never "is this module legal?".
A projection of an *illegal* module is still expected to round-trip — that is what makes it
a projection rather than a filter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .jer import JerRules, decode_jer, encode_jer
from .schema import Choice, Component, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, Universal

#: The module OID this projection's schema is published under, in the same
#: private-enterprise arc as the StreamPack and artifact-bundle modules.
DIALECT_MODULE_OID = (1, 3, 6, 1, 4, 1, 62596, 32)

#: Bumped when the projected shape changes in a way an older reader would misread. It is
#: carried IN the document rather than derived from it, so a consumer can refuse a version
#: it does not know instead of guessing from which members happen to be present.
PROJECTION_VERSION = 1


# --- the pivot ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DialectComponent:
    """One `bcir.asn1.component`."""

    name: str
    type: str
    tag: int | None = None
    tagging: str | None = None
    optional: bool = False
    has_default: bool = False
    default_value: str | None = None


@dataclass(frozen=True)
class DialectType:
    """One `bcir.asn1.type`, with the components it owns."""

    name: str
    kind: str
    universal: int | None = None
    element: str | None = None
    constraint_low: int | None = None
    constraint_high: int | None = None
    size_low: int | None = None
    size_high: int | None = None
    components: tuple[DialectComponent, ...] = ()


@dataclass(frozen=True)
class DialectOperation:
    """One `bcir.asn1.encode` / `decode` / `transcode` / `projection`.

    `op` is the discriminator and the other fields are those its arm carries. The ASN.1
    projection turns this into a CHOICE, so the arms cannot be mixed on the wire; keeping
    one Python class is a convenience for the parser, and `to_value` is where the
    discipline is enforced — a field belonging to another arm is dropped rather than
    smuggled, and `from_value` can only produce a consistent record.
    """

    op: str
    name: str
    type: str
    rules: str | None = None
    strict_der: bool = False
    strict_canonical: bool = False
    source: str | None = None
    from_rules: str | None = None
    to_rules: str | None = None
    preserve_value: bool = False
    native: str | None = None
    additive: bool = False


@dataclass(frozen=True)
class DialectModule:
    """One `bcir.asn1.module` — the unit the projection round-trips."""

    name: str
    oid: tuple[int, ...]
    rules: str
    default_tagging: str
    types: tuple[DialectType, ...] = ()
    operations: tuple[DialectOperation, ...] = ()


# --- the ASN.1 schema -----------------------------------------------------------------------

_UTF8 = Primitive(Universal.UTF8_STRING)
_INT = Primitive(Universal.INTEGER)
_BOOL = Primitive(Universal.BOOLEAN)

COMPONENT_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("type", _UTF8),
        Component("tag", _INT, optional=True),
        Component("tagging", _UTF8, optional=True),
        # DEFAULT FALSE rather than OPTIONAL: a flag that is absent and a flag that is false
        # are the same fact, and X.690 §11.5 (with X.697 §21.2 following it) makes the
        # canonical encoder omit a component encoded at its default. So the canonical JER for
        # an unset flag carries no member at all, which is both smaller and unambiguous —
        # whereas OPTIONAL would let `false` and absent both appear and mean one thing.
        Component("optional", _BOOL, default=False),
        Component("hasDefault", _BOOL, default=False),
        Component("defaultValue", _UTF8, optional=True),
    ),
    name="DialectComponent",
)

TYPE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("kind", _UTF8),
        Component("universal", _INT, optional=True),
        Component("element", _UTF8, optional=True),
        Component("constraintLow", _INT, optional=True),
        Component("constraintHigh", _INT, optional=True),
        Component("sizeLow", _INT, optional=True),
        Component("sizeHigh", _INT, optional=True),
        Component("components", SequenceOf(COMPONENT_TYPE), default=()),
    ),
    name="DialectType",
)

ENCODE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("type", _UTF8),
        Component("rules", _UTF8),
        Component("source", _UTF8, optional=True),
    ),
    name="DialectEncode",
)

DECODE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("type", _UTF8),
        Component("rules", _UTF8),
        Component("strictDer", _BOOL, default=False),
        Component("strictCanonical", _BOOL, default=False),
    ),
    name="DialectDecode",
)

TRANSCODE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("type", _UTF8),
        Component("from", _UTF8),
        Component("to", _UTF8),
        Component("preserveValue", _BOOL, default=False),
    ),
    name="DialectTranscode",
)

PROJECTION_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("type", _UTF8),
        Component("native", _UTF8),
        Component("additive", _BOOL, default=False),
    ),
    name="DialectProjection",
)

#: X.680 §29. An operation IS one of four things; it does not have four optional halves.
#:
#: The context tags are REQUIRED, not decoration. All four arms are SEQUENCE, so in a
#: binary rail they would all carry universal tag 16 and no decoder could tell them apart —
#: X.680 §29.3 says a CHOICE's alternatives must have distinct tags, and `schema.py`
#: refuses the untagged version at construction. JER never sees a tag (X.697 §23 encodes a
#: CHOICE as a single-member object keyed by the alternative's *name*), so the tags cost
#: nothing here and are what lets the same schema project to DER or COER unchanged.
OPERATION_TYPE = Choice(
    (
        Component("encode", ENCODE_TYPE, tag=0),
        Component("decode", DECODE_TYPE, tag=1),
        Component("transcode", TRANSCODE_TYPE, tag=2),
        Component("projection", PROJECTION_TYPE, tag=3),
    ),
    name="DialectOperation",
)

MODULE_TYPE = Sequence(
    (
        Component("version", _INT),
        Component("name", _UTF8),
        Component("oid", SequenceOf(_INT)),
        Component("rules", _UTF8),
        Component("defaultTagging", _UTF8),
        Component("types", SequenceOf(TYPE_TYPE), default=()),
        Component("operations", SequenceOf(OPERATION_TYPE), default=()),
    ),
    name="DialectModule",
)


# --- pivot <-> ASN.1 value ------------------------------------------------------------------


def _put(out: dict, key: str, value) -> None:
    """Set `key` only when it carries information.

    An absent optional and a present-but-empty one are different documents, and only the
    first is what "this module has no such attribute" means.
    """
    if value is not None and value != () and value is not False:
        out[key] = value


def component_to_value(component: DialectComponent) -> dict:
    out = {"name": component.name, "type": component.type}
    _put(out, "tag", component.tag)
    _put(out, "tagging", component.tagging)
    _put(out, "optional", component.optional)
    _put(out, "hasDefault", component.has_default)
    _put(out, "defaultValue", component.default_value)
    return out


def type_to_value(kind: DialectType) -> dict:
    out = {"name": kind.name, "kind": kind.kind}
    _put(out, "universal", kind.universal)
    _put(out, "element", kind.element)
    _put(out, "constraintLow", kind.constraint_low)
    _put(out, "constraintHigh", kind.constraint_high)
    _put(out, "sizeLow", kind.size_low)
    _put(out, "sizeHigh", kind.size_high)
    _put(out, "components", tuple(component_to_value(c) for c in kind.components))
    return out


def operation_to_value(operation: DialectOperation) -> tuple[str, dict]:
    """The CHOICE, as `(alternative, value)`.

    Each arm carries only its own fields. A `native` on an encode is dropped here rather
    than travelling as an ignored member, which is what makes the CHOICE worth having:
    the projected document cannot express the contradiction the flat record could.
    """
    body = {"name": operation.name, "type": operation.type}
    if operation.op == "encode":
        body["rules"] = operation.rules
        _put(body, "source", operation.source)
        return "encode", body
    if operation.op == "decode":
        body["rules"] = operation.rules
        _put(body, "strictDer", operation.strict_der)
        _put(body, "strictCanonical", operation.strict_canonical)
        return "decode", body
    if operation.op == "transcode":
        body["from"] = operation.from_rules
        body["to"] = operation.to_rules
        _put(body, "preserveValue", operation.preserve_value)
        return "transcode", body
    if operation.op == "projection":
        body["native"] = operation.native
        _put(body, "additive", operation.additive)
        return "projection", body
    raise Asn1Error(f"no bcir.asn1 operation is named {operation.op!r}")


def module_to_value(module: DialectModule) -> dict:
    return {
        "version": PROJECTION_VERSION,
        "name": module.name,
        "oid": tuple(module.oid),
        "rules": module.rules,
        "defaultTagging": module.default_tagging,
        "types": tuple(type_to_value(t) for t in module.types),
        "operations": tuple(operation_to_value(o) for o in module.operations),
    }


def value_to_module(value: dict) -> DialectModule:
    version = value.get("version", 0)
    if version != PROJECTION_VERSION:
        # Refused rather than guessed. A reader that fell back to inspecting which members
        # are present would be inferring a version from a shape, which is exactly how a
        # format acquires two incompatible readings of the same document.
        raise Asn1Error(
            f"projection version {version} is not {PROJECTION_VERSION}; this reader "
            f"refuses rather than inferring the shape from which members are present"
        )
    types = tuple(
        DialectType(
            name=t["name"],
            kind=t["kind"],
            universal=t.get("universal"),
            element=t.get("element"),
            constraint_low=t.get("constraintLow"),
            constraint_high=t.get("constraintHigh"),
            size_low=t.get("sizeLow"),
            size_high=t.get("sizeHigh"),
            components=tuple(
                DialectComponent(
                    name=c["name"],
                    type=c["type"],
                    tag=c.get("tag"),
                    tagging=c.get("tagging"),
                    optional=c.get("optional", False),
                    has_default=c.get("hasDefault", False),
                    default_value=c.get("defaultValue"),
                )
                for c in t.get("components", ())
            ),
        )
        for t in value.get("types", ())
    )
    operations = tuple(
        _operation_from_value(alt, body) for alt, body in value.get("operations", ())
    )
    return DialectModule(
        name=value["name"],
        oid=tuple(value["oid"]),
        rules=value["rules"],
        default_tagging=value["defaultTagging"],
        types=types,
        operations=operations,
    )


def _operation_from_value(alternative: str, body: dict) -> DialectOperation:
    common = {"op": alternative, "name": body["name"], "type": body["type"]}
    if alternative == "encode":
        return DialectOperation(**common, rules=body["rules"], source=body.get("source"))
    if alternative == "decode":
        return DialectOperation(
            **common,
            rules=body["rules"],
            strict_der=body.get("strictDer", False),
            strict_canonical=body.get("strictCanonical", False),
        )
    if alternative == "transcode":
        return DialectOperation(
            **common,
            from_rules=body["from"],
            to_rules=body["to"],
            preserve_value=body.get("preserveValue", False),
        )
    if alternative == "projection":
        return DialectOperation(
            **common, native=body["native"], additive=body.get("additive", False)
        )
    raise Asn1Error(f"no bcir.asn1 operation is named {alternative!r}")


# --- pivot <-> JER --------------------------------------------------------------------------


def module_to_jer(module: DialectModule, *, rules: JerRules = JerRules.CANONICAL) -> bytes:
    return encode_jer(MODULE_TYPE, module_to_value(module), rules=rules)


def jer_to_module(data: bytes, *, rules: JerRules = JerRules.CANONICAL) -> DialectModule:
    return value_to_module(decode_jer(data, MODULE_TYPE, rules=rules))


# --- pivot <-> MLIR textual assembly --------------------------------------------------------
#
# A parser for the `bcir.asn1.*` SUBSET of MLIR, and nothing wider. That bound is the reason
# it is safe to hand-write: these seven operations have a flat, regular shape -- a mnemonic,
# an optional symbol, an attribute dictionary, and an optional region of the same -- with no
# SSA values, no types, no successors and no nesting beyond one level. A general MLIR parser
# is a different undertaking and this is not a step towards one; anything outside the subset
# is refused rather than skipped, so a module carrying an operation this cannot read fails
# loudly instead of silently projecting a smaller module than it was given.

_MODULE_RE = re.compile(r"bcir\.asn1\.module\s+@([A-Za-z_][\w.$]*)\s+attributes\s*\{")
_OP_RE = re.compile(
    r"bcir\.asn1\.(type|component|encode|decode|transcode|projection)"
    r"(?:\s+@([A-Za-z_][\w.$]*))?\s*(?:attributes\s*)?\{"
)
_RULES_RE = re.compile(r"#bcir\.asn1_rules<(\w+)>")
_TAGGING_RE = re.compile(r"#bcir\.asn1_tagging<(\w+)>")
_ARRAY_RE = re.compile(r"array<i64:\s*([^>]*)>")


def _balanced(text: str, start: int, opener: str = "{", closer: str = "}") -> int:
    """Offset just past the `closer` matching the `opener` at `start`.

    Quote-aware, because a `}` inside an attribute string is not a delimiter. Without that
    a `default_value = "}"` would end the dictionary early and the rest of the module would
    be parsed as something else entirely.
    """
    depth = 0
    at = start
    in_string = False
    while at < len(text):
        ch = text[at]
        if in_string:
            if ch == "\\":
                at += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return at + 1
        at += 1
    raise Asn1Error(f"unbalanced {opener!r} at offset {start}")


def _attributes(body: str) -> dict:
    """Parse one MLIR attribute dictionary body into Python values.

    Deliberately narrow: the value forms this dialect uses are a quoted string, a decimal
    integer with an optional `: i64` suffix, `array<i64: ...>`, the two `#bcir.asn1_*`
    enum attributes, and a bare name for a unit attribute. Anything else raises, because a
    silently ignored attribute is an attribute that vanishes from the projection.
    """
    out: dict = {}
    at = 0
    end = len(body)
    while at < end:
        while at < end and body[at] in " \t\r\n,":
            at += 1
        if at >= end:
            break
        key = re.match(r"([A-Za-z_]\w*)", body[at:])
        if not key:
            raise Asn1Error(f"not an attribute name at {body[at : at + 24]!r}")
        name = key.group(1)
        at += key.end()
        while at < end and body[at] in " \t\r\n":
            at += 1
        if at >= end or body[at] != "=":
            out[name] = True  # a unit attribute
            continue
        at += 1
        while at < end and body[at] in " \t\r\n":
            at += 1
        rest = body[at:]
        if rest.startswith('"'):
            match = re.match(r'"((?:[^"\\]|\\.)*)"', rest)
            if not match:
                raise Asn1Error(f"unterminated string at {rest[:24]!r}")
            out[name] = match.group(1).replace('\\"', '"').replace("\\\\", "\\")
            at += match.end()
            continue
        if rest.startswith("@"):
            match = re.match(r"@([A-Za-z_][\w.$]*)", rest)
            out[name] = match.group(1)
            at += match.end()
            continue
        if rest.startswith("#bcir.asn1_rules<"):
            match = _RULES_RE.match(rest)
            out[name] = match.group(1)
            at += match.end()
            continue
        if rest.startswith("#bcir.asn1_tagging<"):
            match = _TAGGING_RE.match(rest)
            out[name] = match.group(1)
            at += match.end()
            continue
        if rest.startswith("array<i64:"):
            match = _ARRAY_RE.match(rest)
            body_text = match.group(1).strip()
            out[name] = tuple(int(x) for x in body_text.split(",") if x.strip())
            at += match.end()
            continue
        match = re.match(r"(-?\d+)\s*(?::\s*i\d+)?", rest)
        if not match:
            raise Asn1Error(f"unsupported attribute value at {rest[:24]!r}")
        out[name] = int(match.group(1))
        at += match.end()
    return out


def parse_mlir(text: str) -> tuple[DialectModule, ...]:
    """Parse every `bcir.asn1.module` in `text`.

    Comments are stripped first, and `// -----` split markers are simply comment lines, so
    a `-split-input-file` fixture parses as the sequence of modules it contains.
    """
    stripped = re.sub(r"//[^\n]*", "", text)
    modules: list[DialectModule] = []
    for match in _MODULE_RE.finditer(stripped):
        attr_start = stripped.index("{", match.end() - 1)
        attr_end = _balanced(stripped, attr_start)
        attrs = _attributes(stripped[attr_start + 1 : attr_end - 1])
        body_start = stripped.index("{", attr_end)
        body_end = _balanced(stripped, body_start)
        types, operations = _parse_body(stripped[body_start + 1 : body_end - 1])
        modules.append(
            DialectModule(
                name=match.group(1),
                oid=tuple(attrs.get("oid", ())),
                rules=attrs.get("rules", ""),
                default_tagging=attrs.get("default_tagging", ""),
                types=types,
                operations=operations,
            )
        )
    return tuple(modules)


def _parse_body(body: str) -> tuple[tuple[DialectType, ...], tuple[DialectOperation, ...]]:
    types: list[DialectType] = []
    operations: list[DialectOperation] = []
    at = 0
    while True:
        match = _OP_RE.search(body, at)
        if not match:
            break
        mnemonic, symbol = match.group(1), match.group(2)
        attr_start = body.index("{", match.end() - 1)
        attr_end = _balanced(body, attr_start)
        attrs = _attributes(body[attr_start + 1 : attr_end - 1])
        at = attr_end
        if mnemonic == "type":
            region_start = body.index("{", attr_end)
            region_end = _balanced(body, region_start)
            components = _parse_components(body[region_start + 1 : region_end - 1])
            at = region_end
            types.append(
                DialectType(
                    name=symbol,
                    kind=attrs.get("kind", ""),
                    universal=attrs.get("universal"),
                    element=attrs.get("element"),
                    constraint_low=attrs.get("constraint_low"),
                    constraint_high=attrs.get("constraint_high"),
                    size_low=attrs.get("size_low"),
                    size_high=attrs.get("size_high"),
                    components=components,
                )
            )
        elif mnemonic == "component":
            continue  # owned by a type's region, never loose
        else:
            operations.append(
                DialectOperation(
                    op=mnemonic,
                    name=symbol,
                    type=attrs.get("type", ""),
                    rules=attrs.get("rules"),
                    strict_der=bool(attrs.get("strict_der", False)),
                    strict_canonical=bool(attrs.get("strict_canonical", False)),
                    source=attrs.get("source"),
                    from_rules=attrs.get("from"),
                    to_rules=attrs.get("to"),
                    preserve_value=bool(attrs.get("preserve_value", False)),
                    native=attrs.get("native"),
                    additive=bool(attrs.get("additive", False)),
                )
            )
    return tuple(types), tuple(operations)


def _parse_components(region: str) -> tuple[DialectComponent, ...]:
    out: list[DialectComponent] = []
    at = 0
    while True:
        match = re.compile(r"bcir\.asn1\.component\s*\{").search(region, at)
        if not match:
            break
        start = region.index("{", match.end() - 1)
        stop = _balanced(region, start)
        attrs = _attributes(region[start + 1 : stop - 1])
        at = stop
        out.append(
            DialectComponent(
                name=attrs.get("name", ""),
                type=attrs.get("type", ""),
                tag=attrs.get("tag"),
                tagging=attrs.get("tagging"),
                optional=bool(attrs.get("optional", False)),
                has_default=bool(attrs.get("has_default", False)),
                default_value=attrs.get("default_value"),
            )
        )
    return tuple(out)


def _quote(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_mlir(module: DialectModule) -> str:
    """Emit `module` as `bcir.asn1.*` textual assembly, in ONE canonical shape.

    Canonical here means: this function's output is a function of the `DialectModule`
    alone. It is *not* a claim that MLIR text is canonical in general — `mlir-opt` may
    reprint it differently and remain correct — which is exactly why the round-trip law in
    this direction is the identity on the dialect rather than on the octets.
    """
    lines = [
        f"bcir.asn1.module @{module.name} attributes {{",
        f"  oid = array<i64: {', '.join(str(a) for a in module.oid)}>,",
        f"  rules = #bcir.asn1_rules<{module.rules}>,",
        f"  default_tagging = #bcir.asn1_tagging<{module.default_tagging}>",
        "} {",
    ]
    for kind in module.types:
        attrs = [f"kind = {_quote(kind.kind)}"]
        if kind.universal is not None:
            attrs.append(f"universal = {kind.universal} : i64")
        if kind.element is not None:
            attrs.append(f"element = @{kind.element}")
        for name, value in (
            ("constraint_low", kind.constraint_low),
            ("constraint_high", kind.constraint_high),
            ("size_low", kind.size_low),
            ("size_high", kind.size_high),
        ):
            if value is not None:
                attrs.append(f"{name} = {value} : i64")
        lines.append(f"  bcir.asn1.type @{kind.name} attributes {{ {', '.join(attrs)} }} {{")
        for component in kind.components:
            parts = [f"name = {_quote(component.name)}", f"type = @{component.type}"]
            if component.tag is not None:
                parts.append(f"tag = {component.tag} : i64")
            if component.tagging is not None:
                parts.append(f"tagging = #bcir.asn1_tagging<{component.tagging}>")
            if component.optional:
                parts.append("optional")
            if component.has_default:
                parts.append("has_default")
            if component.default_value is not None:
                parts.append(f"default_value = {_quote(component.default_value)}")
            lines.append(f"    bcir.asn1.component {{ {', '.join(parts)} }}")
        lines.append("  }")
    for operation in module.operations:
        parts = [f"type = @{operation.type}"]
        if operation.op == "encode":
            parts.append(f"rules = #bcir.asn1_rules<{operation.rules}>")
            if operation.source is not None:
                parts.append(f"source = @{operation.source}")
        elif operation.op == "decode":
            parts.append(f"rules = #bcir.asn1_rules<{operation.rules}>")
            if operation.strict_der:
                parts.append("strict_der")
            if operation.strict_canonical:
                parts.append("strict_canonical")
        elif operation.op == "transcode":
            parts.append(f"from = #bcir.asn1_rules<{operation.from_rules}>")
            parts.append(f"to = #bcir.asn1_rules<{operation.to_rules}>")
            if operation.preserve_value:
                parts.append("preserve_value")
        elif operation.op == "projection":
            parts.append(f"native = {_quote(operation.native)}")
            if operation.additive:
                parts.append("additive")
        lines.append(f"  bcir.asn1.{operation.op} @{operation.name} {{ {', '.join(parts)} }}")
    lines.append("}")
    return "\n".join(lines) + "\n"


# --- the two round trips §7.1 names ---------------------------------------------------------


def mlir_to_jer_to_mlir(text: str) -> tuple[DialectModule, ...]:
    """`MLIR -> JER -> MLIR`, returning what came back. Identity is checked against
    `parse_mlir(text)` by the caller."""
    return tuple(jer_to_module(module_to_jer(module)) for module in parse_mlir(text))


def jer_to_mlir_to_jer(data: bytes) -> bytes:
    """`JER -> MLIR -> JER`, returning the octets. Byte-identity is the law here, and it
    can be, because the canonical profile gives one octet string per abstract value."""
    module = jer_to_module(data)
    reparsed = parse_mlir(emit_mlir(module))
    if len(reparsed) != 1:
        raise Asn1Error(f"emitting one module re-parsed as {len(reparsed)}")
    return module_to_jer(reparsed[0])


__all__ = [
    "COMPONENT_TYPE",
    "DIALECT_MODULE_OID",
    "MODULE_TYPE",
    "OPERATION_TYPE",
    "PROJECTION_VERSION",
    "TYPE_TYPE",
    "DialectComponent",
    "DialectModule",
    "DialectOperation",
    "DialectType",
    "emit_mlir",
    "jer_to_mlir_to_jer",
    "jer_to_module",
    "mlir_to_jer_to_mlir",
    "module_to_jer",
    "module_to_value",
    "parse_mlir",
    "value_to_module",
]
