"""Lower a parsed X.680 module onto the `bcir.asn1.schema` encoder model.

Three X.680 rules live here rather than in the parser, because each needs the whole
module in view:

* **§31.2.1 the tag default.** A `[0] Type` with no IMPLICIT/EXPLICIT keyword means
  whichever the module's `DEFINITIONS ... TAGS` header declared. The parser keeps
  `mode=None` so the printer can reproduce the source; this pass resolves it.
* **§31.2.7 the CHOICE exception.** In an IMPLICIT or AUTOMATIC module a tag is still
  EXPLICIT when it is applied to a CHOICE or an open type. An implicit tag REPLACES
  the base tag, and a CHOICE has no tag of its own (§29.1) -- replacing it would erase
  the only marker of which alternative was chosen. This is a correctness rule, not a
  style one: getting it wrong produces octets a conforming peer cannot decode.
* **§12.3 / Annex automatic tagging.** In an AUTOMATIC TAGS module, a SEQUENCE / SET /
  CHOICE whose components bear NO tags at all has context tags 0, 1, 2, … assigned in
  order. The "none of them are tagged" precondition matters -- a partially tagged list
  keeps the author's tags untouched.

Recursion is handled with a lazy reference (`_LazyType`) rather than refused: real
modules define mutually recursive types, and the encoder model is eager dataclasses
that cannot be built bottom-up for a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bcir.asn1.schema import (Asn1Type, Choice, Component, Module, OpenType, Primitive,
                              Sequence, SequenceOf, Set, SetOf)
from bcir.asn1.tags import Asn1Error, Tag, Universal

from . import ast
from .lexer import Asn1SyntaxError

#: X.680 Table 1: built-in type name -> universal class tag number.
UNIVERSAL_OF = {
    "BOOLEAN": Universal.BOOLEAN, "INTEGER": Universal.INTEGER,
    "BIT STRING": Universal.BIT_STRING, "OCTET STRING": Universal.OCTET_STRING,
    "NULL": Universal.NULL, "OBJECT IDENTIFIER": Universal.OBJECT_IDENTIFIER,
    "ObjectDescriptor": Universal.OBJECT_DESCRIPTOR, "EXTERNAL": Universal.EXTERNAL,
    "REAL": Universal.REAL, "ENUMERATED": Universal.ENUMERATED,
    "EMBEDDED PDV": Universal.EMBEDDED_PDV, "UTF8String": Universal.UTF8_STRING,
    "RELATIVE-OID": Universal.RELATIVE_OID, "TIME": Universal.TIME,
    "NumericString": Universal.NUMERIC_STRING,
    "PrintableString": Universal.PRINTABLE_STRING,
    "TeletexString": Universal.TELETEX_STRING, "T61String": Universal.TELETEX_STRING,
    "VideotexString": Universal.VIDEOTEX_STRING, "IA5String": Universal.IA5_STRING,
    "UTCTime": Universal.UTC_TIME, "GeneralizedTime": Universal.GENERALIZED_TIME,
    "GraphicString": Universal.GRAPHIC_STRING,
    "VisibleString": Universal.VISIBLE_STRING,
    "ISO646String": Universal.VISIBLE_STRING,
    "GeneralString": Universal.GENERAL_STRING,
    "UniversalString": Universal.UNIVERSAL_STRING,
    "CHARACTER STRING": Universal.CHARACTER_STRING,
    "BMPString": Universal.BMP_STRING, "DATE": Universal.DATE,
    "TIME-OF-DAY": Universal.TIME_OF_DAY, "DATE-TIME": Universal.DATE_TIME,
    "DURATION": Universal.DURATION, "OID-IRI": Universal.OID_IRI,
    "RELATIVE-OID-IRI": Universal.RELATIVE_OID_IRI,
}


class Asn1SemanticError(Exception):
    """A module that parses but does not describe a usable type."""


@dataclass
class _LazyType(Asn1Type):
    """A forward reference to a type still being built (a recursive definition)."""

    target_name: str
    registry: dict
    name: str = "?"

    def _resolved(self) -> Asn1Type:
        try:
            resolved = self.registry[self.target_name]
        except KeyError:                                   # pragma: no cover - guarded
            raise Asn1SemanticError(
                f"unresolved forward reference to {self.target_name!r}") from None
        if resolved is self:                               # pragma: no cover - guarded
            raise Asn1SemanticError(
                f"type {self.target_name!r} is defined as itself")
        return resolved

    def base_tag(self) -> Tag:
        return self._resolved().base_tag()

    def alternative_tags(self) -> tuple[Tag, ...]:
        return self._resolved().alternative_tags()

    def encode(self, value):
        return self._resolved().encode(value)

    def decode(self, tlv, *, strictness):
        return self._resolved().decode(tlv, strictness=strictness)


@dataclass
class LoweredModule:
    """A compiled module plus the metadata the encoder model does not carry."""

    module: Module
    tag_default: str
    #: type name -> {enumeration item name: number}, for reading DEFAULT values back.
    enumerations: dict[str, dict[str, int]] = field(default_factory=dict)
    node: ast.ModuleNode | None = None

    def __getattr__(self, item):                           # convenience delegation
        return getattr(self.module, item)


class Lowerer:
    def __init__(self, node: ast.ModuleNode, imports: dict[str, Module] | None = None):
        self.node = node
        self.assignments = node.type_assignments()
        #: X.681 §9 class definitions, by name. Needed because `CLASS.&field` resolves to
        #: the field's DECLARED type when it is a value field, and only to an open type
        #: when it is a type field -- the class definition is the only place that says
        #: which, so a front-end without this table has to guess.
        self.classes = {a.name: a for a in node.assignments
                        if isinstance(a, ast.ClassAssignment)}
        self.imported = imports or {}
        self.types: dict[str, Asn1Type] = {}
        self.enumerations: dict[str, dict[str, int]] = {}
        self._in_progress: set[str] = set()

    # --- entry point ------------------------------------------------------------------

    def run(self) -> LoweredModule:
        for name in self.assignments:
            self._type_by_name(name)
        oid = self._oid(self.node.oid) if self.node.oid else ()
        module = Module(self.node.name, oid, dict(self.types))
        return LoweredModule(module, self.node.tag_default, self.enumerations, self.node)

    def _oid(self, value: ast.OidValue) -> tuple[int, ...]:
        arcs: list[int] = []
        for arc in value.arcs:
            if arc.number is None:
                number = _WELL_KNOWN_ARCS.get(arc.name)
                if number is None:
                    raise Asn1SemanticError(
                        f"module {self.node.name}: object identifier arc {arc.name!r} "
                        f"has no number and is not a well-known arc name; write it as "
                        f"{arc.name}(n)")
                arcs.append(number)
            else:
                arcs.append(arc.number)
        return tuple(arcs)

    # --- types ------------------------------------------------------------------------

    def _type_by_name(self, name: str) -> Asn1Type:
        if name in self.types:
            return self.types[name]
        if name in self._in_progress:
            return _LazyType(name, self.types, name)       # a recursive definition
        if name not in self.assignments:
            for module in self.imported.values():
                if name in module.types:
                    return module.types[name]
            raise Asn1SemanticError(
                f"module {self.node.name}: type {name!r} is referenced but never "
                f"assigned, and no IMPORTS provides it")
        self._in_progress.add(name)
        try:
            built = self._type(self.assignments[name], name)
        finally:
            self._in_progress.discard(name)
        self.types[name] = built
        return built

    def _type(self, node, label: str) -> Asn1Type:
        if isinstance(node, ast.TypeRef):
            if node.module is not None:
                target = self.imported.get(node.module)
                if target is None or node.name not in target.types:
                    raise Asn1SemanticError(
                        f"{label}: external reference {node.module}.{node.name} is not "
                        f"available; pass the module in `imports`")
                return target.types[node.name]
            return self._type_by_name(node.name)

        if isinstance(node, ast.OpenTypeNode):
            return self._open_type(node, label)

        if isinstance(node, ast.Builtin):
            return self._builtin(node, label)

        if isinstance(node, ast.Tagged):
            # A tagged type at ASSIGNMENT level (`T ::= [0] INTEGER`) is a distinct type
            # in X.680, but the encoder model carries tags on components. Only the
            # component form is supported, and saying so beats silently dropping a tag.
            raise Asn1SemanticError(
                f"{label}: a tagged type outside a component list is not supported; "
                f"write the tag on the component that references this type")

        if isinstance(node, ast.SequenceOfType):
            element = self._type(node.element, f"{label} element")
            return SequenceOf(element, f"SEQUENCE OF {_render_name(node.element)}")

        if isinstance(node, ast.SetOfType):
            element = self._type(node.element, f"{label} element")
            return SetOf(element, f"SET OF {_render_name(node.element)}")

        if isinstance(node, ast.SequenceType):
            return Sequence(self._components(node.components, label), label)

        if isinstance(node, ast.SetType):
            return Set(self._components(node.components, label), label)

        if isinstance(node, ast.ChoiceType):
            return Choice(self._components(node.alternatives, label, choice=True), label)

        raise Asn1SemanticError(f"{label}: unsupported type node {type(node).__name__}")

    def _open_type(self, node: ast.OpenTypeNode, label: str) -> Asn1Type:
        """Resolve `ANY [DEFINED BY x]` and `CLASS.&field` (X.681 §14/§15).

        A `CLASS.&Type` reference is genuinely open. A `CLASS.&id` reference is NOT: the
        class declared a type for that value field, so lowering it to an open type would
        throw away information the module supplied and turn a checkable OBJECT IDENTIFIER
        into opaque octets.
        """
        if node.object_class is not None:
            declared = self.classes.get(node.object_class)
            if declared is None:
                raise Asn1SemanticError(
                    f"{label}: information object class {node.object_class!r} is "
                    f"referenced but never defined in this module")
            for field in declared.fields:
                if field.name != node.field:
                    continue
                if field.is_type_field:
                    return OpenType(f"{node.object_class}{node.field}")
                if field.type is None:
                    raise Asn1SemanticError(
                        f"{label}: {node.object_class}{node.field} is a value field with "
                        f"no declared type, so it has no encoding")
                return self._type(field.type, label)
            raise Asn1SemanticError(
                f"{label}: {node.object_class} has no field {node.field!r}")
        governed = f" DEFINED BY {node.governed_by}" if node.governed_by else ""
        return OpenType(f"ANY{governed}")

    def _builtin(self, node: ast.Builtin, label: str) -> Asn1Type:
        universal = UNIVERSAL_OF.get(node.name)
        if universal is None:
            raise Asn1SemanticError(f"{label}: unknown built-in type {node.name!r}")
        if node.named:
            self.enumerations[label] = {n.name: n.number for n in node.named}
        return Primitive(int(universal), node.name)

    def _components(self, nodes, label: str, choice: bool = False):
        entries = [n for n in nodes if isinstance(n, ast.ComponentNode)]
        automatic = self._use_automatic_tags(entries)
        out: list[Component] = []
        for position, item in enumerate(entries):
            inner, tag, mode = _peel_tag(item.type)
            if tag is None and automatic:
                tag, mode = position, None                 # §12.3 automatic assignment
            built = self._type(inner, f"{label}.{item.name}")
            explicit = self._explicit(mode, built)
            default, has_default = self._default(item, built, f"{label}.{item.name}")
            out.append(Component(
                name=item.name, type=built, tag=tag, explicit=explicit,
                optional=item.optional,
                **({"default": default} if has_default else {})))
        if choice:
            return tuple(out)
        return tuple(out)

    def _use_automatic_tags(self, entries) -> bool:
        """§12.3: automatic tagging applies only when NO component carries a tag."""
        if self.node.tag_default != "AUTOMATIC":
            return False
        return all(_peel_tag(item.type)[1] is None for item in entries)

    def _explicit(self, mode: str | None, built: Asn1Type) -> bool:
        """Resolve IMPLICIT/EXPLICIT for a component tag (§31.2.1 + §31.2.7)."""
        if mode == "EXPLICIT":
            return True
        if mode == "IMPLICIT":
            if _needs_explicit_tag(built):
                raise Asn1SemanticError(
                    "IMPLICIT cannot tag a CHOICE or an open type: an implicit tag "
                    "replaces the base tag and neither has one (X.680 29.1/31.2.7)")
            return False
        # No keyword: the module's default decides -- except that §31.2.7 forces
        # EXPLICIT over a CHOICE or an open type even in an IMPLICIT/AUTOMATIC module.
        if _needs_explicit_tag(built):
            return True
        return self.node.tag_default == "EXPLICIT"

    # --- values -----------------------------------------------------------------------

    def _default(self, item: ast.ComponentNode, built: Asn1Type, label: str):
        if not item.has_default:
            return None, False
        return self.value(item.default, built, label), True

    def value(self, node, built: Asn1Type, label: str):
        """Turn a parsed value into the Python object the encoder model expects."""
        if isinstance(node, ast.IntValue):
            return node.value
        if isinstance(node, ast.StrValue):
            return node.value
        if isinstance(node, ast.BoolValue):
            return node.value
        if isinstance(node, ast.NullValue):
            from bcir.asn1.codec import NULL
            return NULL
        if isinstance(node, ast.BitsValue):
            return node.data
        if isinstance(node, ast.OidValue):
            from bcir.asn1.codec import Oid
            return Oid(self._oid(node))
        if isinstance(node, ast.BracedValue):
            return self._braced(node, built, label)
        if isinstance(node, ast.RefValue):
            return self._named_value(node.name, built, label)
        raise Asn1SemanticError(f"{label}: unsupported value {type(node).__name__}")

    def _braced(self, node: ast.BracedValue, built: Asn1Type, label: str):
        """Resolve a `{ ... }` value against the type that governs it.

        The parser deliberately left this open (see `ast.BracedValue`): `{ 1 }` reads as
        both a one-element SEQUENCE OF value and a one-arc OBJECT IDENTIFIER, and only
        the type here can say which. A type that admits neither reading is an error
        rather than a default, because a DEFAULT the encoder can never compare equal to
        would quietly disable X.690 §11.5 for that component.
        """
        element = _element_of(built)
        if element is not None or isinstance(built, (SequenceOf, SetOf)):
            if node.items is None:
                raise Asn1SemanticError(
                    f"{label}: {built.name} needs a value list, but the braced value "
                    f"is an object identifier arc list")
            return [self.value(v, element, label) for v in node.items]
        if isinstance(built, Primitive) and built.universal in (
                int(Universal.OBJECT_IDENTIFIER), int(Universal.RELATIVE_OID)):
            from bcir.asn1.codec import Oid, RelativeOid
            if node.arcs is None:
                raise Asn1SemanticError(
                    f"{label}: {built.name} needs object identifier arcs, but the "
                    f"braced value is a comma-separated value list")
            wrap = Oid if built.universal == int(Universal.OBJECT_IDENTIFIER) \
                else RelativeOid
            return wrap(self._oid(ast.OidValue(node.arcs)))
        if node.items == ():
            # `{}` against a constructed type with no element: an empty SEQUENCE/SET.
            return {}
        raise Asn1SemanticError(
            f"{label}: a braced value needs a SEQUENCE OF / SET OF or OBJECT "
            f"IDENTIFIER type, not {built.name}")

    def _named_value(self, name: str, built: Asn1Type, label: str):
        """A bare identifier as a value: an ENUMERATED/INTEGER item, or a
        valuereference assigned elsewhere in the module."""
        target = built.name if isinstance(built, Primitive) else None
        for holder in ([target] if target else []) + [label.split(".")[0]]:
            table = self.enumerations.get(holder or "")
            if table and name in table:
                return table[name]
        # The enumeration is registered under the ASSIGNED type name, which for a
        # component like `lane [3] Lane` is `Lane` -- look through every table whose
        # type the component actually resolved to.
        for holder, table in self.enumerations.items():
            if name in table and holder in self.types and self.types[holder] is built:
                return table[name]
        assignment = self.node.value_assignments().get(name)
        if assignment is not None:
            return self.value(assignment.value, built, label)
        raise Asn1SemanticError(
            f"{label}: DEFAULT {name!r} is neither an enumeration item of "
            f"{built.name} nor a value assigned in this module")


#: Arc names X.660 fixes, so `{ iso 3 6 1 }` resolves without a number in parentheses.
_WELL_KNOWN_ARCS = {
    "itu-t": 0, "ccitt": 0, "iso": 1, "joint-iso-itu-t": 2, "joint-iso-ccitt": 2,
}


def _peel_tag(node):
    """Split `[n] IMPLICIT Type` into (Type, n, mode); (node, None, None) if untagged."""
    if isinstance(node, ast.Tagged):
        if node.tag_class:
            raise Asn1SemanticError(
                f"[{node.tag_class} {node.number}] is not a context-specific tag; the "
                f"encoder model carries context tags only")
        return node.inner, node.number, node.mode
    return node, None, None


def _needs_explicit_tag(built: Asn1Type) -> bool:
    """X.680 §31.2.7: a tag over a CHOICE or an OPEN TYPE is always EXPLICIT.

    Both have no tag of their own -- a CHOICE shows the chosen alternative's tag (§29.1),
    an open type the contained value's -- so an implicit tag would have nothing to
    replace and would erase the only discriminator on the wire.
    """
    if isinstance(built, (Choice, OpenType)):
        return True
    if isinstance(built, _LazyType):
        try:
            return isinstance(built._resolved(), (Choice, OpenType))
        except Asn1SemanticError:                          # pragma: no cover - guarded
            return False
    return False


def _element_of(built: Asn1Type | None):
    return built.element if isinstance(built, (SequenceOf, SetOf)) else None


def _render_name(node) -> str:
    if isinstance(node, ast.TypeRef):
        return node.name
    if isinstance(node, ast.Builtin):
        return node.name
    if isinstance(node, ast.SequenceOfType):
        return f"SEQUENCE OF {_render_name(node.element)}"
    if isinstance(node, ast.SetOfType):
        return f"SET OF {_render_name(node.element)}"
    return {ast.SequenceType: "SEQUENCE", ast.SetType: "SET",
            ast.ChoiceType: "CHOICE"}.get(type(node), "TYPE")


def lower(node: ast.ModuleNode, imports: dict[str, Module] | None = None
          ) -> LoweredModule:
    return Lowerer(node, imports).run()


def compile_module(text: str, source: str = "<asn1>",
                   imports: dict[str, Module] | None = None) -> LoweredModule:
    """Parse and lower one ASN.1 module in a single call."""
    from .parser import parse_module
    return lower(parse_module(text, source), imports)


__all__ = ["Asn1SemanticError", "Asn1SyntaxError", "LoweredModule", "Lowerer",
           "UNIVERSAL_OF", "compile_module", "lower"]
