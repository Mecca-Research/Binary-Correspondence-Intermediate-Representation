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

Recursion is handled with a lazy reference (`schema.Reference`) rather than refused: real
modules define mutually recursive types, and the encoder model is eager dataclasses
that cannot be built bottom-up for a cycle. Every encoding rule follows the reference through
`schema.resolve` and holds a value to `schema.MAX_RECURSION` levels of it; the containment
graph (`containment_graph`, `recursive_types`) names each cycle for a rail that cannot carry one.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from dataclasses import dataclass, field, replace

from bcir.asn1 import constraints as _constraints

from bcir.asn1.schema import (
    MAX_RECURSION,
    Asn1Type,
    Choice,
    Component,
    Module,
    ObjectSetTable,
    OpenType,
    Primitive,
    Reference,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
    resolve,
)
from bcir.asn1.tags import Asn1Error, TagClass, Universal

from . import ast
from .lexer import Asn1SyntaxError

#: X.680 Table 1: built-in type name -> universal class tag number.
UNIVERSAL_OF = {
    "BOOLEAN": Universal.BOOLEAN,
    "INTEGER": Universal.INTEGER,
    "BIT STRING": Universal.BIT_STRING,
    "OCTET STRING": Universal.OCTET_STRING,
    "NULL": Universal.NULL,
    "OBJECT IDENTIFIER": Universal.OBJECT_IDENTIFIER,
    "ObjectDescriptor": Universal.OBJECT_DESCRIPTOR,
    "EXTERNAL": Universal.EXTERNAL,
    "REAL": Universal.REAL,
    "ENUMERATED": Universal.ENUMERATED,
    "EMBEDDED PDV": Universal.EMBEDDED_PDV,
    "UTF8String": Universal.UTF8_STRING,
    "RELATIVE-OID": Universal.RELATIVE_OID,
    "TIME": Universal.TIME,
    "NumericString": Universal.NUMERIC_STRING,
    "PrintableString": Universal.PRINTABLE_STRING,
    "TeletexString": Universal.TELETEX_STRING,
    "T61String": Universal.TELETEX_STRING,
    "VideotexString": Universal.VIDEOTEX_STRING,
    "IA5String": Universal.IA5_STRING,
    "UTCTime": Universal.UTC_TIME,
    "GeneralizedTime": Universal.GENERALIZED_TIME,
    "GraphicString": Universal.GRAPHIC_STRING,
    "VisibleString": Universal.VISIBLE_STRING,
    "ISO646String": Universal.VISIBLE_STRING,
    "GeneralString": Universal.GENERAL_STRING,
    "UniversalString": Universal.UNIVERSAL_STRING,
    "CHARACTER STRING": Universal.CHARACTER_STRING,
    "BMPString": Universal.BMP_STRING,
    "DATE": Universal.DATE,
    "TIME-OF-DAY": Universal.TIME_OF_DAY,
    "DATE-TIME": Universal.DATE_TIME,
    "DURATION": Universal.DURATION,
    "OID-IRI": Universal.OID_IRI,
    "RELATIVE-OID-IRI": Universal.RELATIVE_OID_IRI,
}


class Asn1SemanticError(Exception):
    """A module that parses but does not describe a usable type."""


class _Forward(Reference):
    """A forward reference to a type still being built (a recursive definition).

    It is the one placeholder every encoding rule follows (`bcir.asn1.schema.Reference`),
    and it also names the lowerer building its target. A tag over the reference has to be
    decided before that target exists (X.680 §31.2.7: EXPLICIT over a CHOICE or an open
    type), and only that lowerer can read the definition it is building
    (`Lowerer._forward_choice_or_open`). The link is dropped once the module is complete."""

    def __init__(self, target_name: str, registry: dict, name: str, cycle, lowerer, tags=()):
        super().__init__(target_name, registry, name, cycle, tags)
        self.lowerer = lowerer


@dataclass
class LoweredModule:
    """A compiled module plus the metadata the encoder model does not carry."""

    module: Module
    tag_default: str
    #: type name -> {enumeration item name: number}, for reading DEFAULT values back.
    enumerations: dict[str, dict[str, int]] = field(default_factory=dict)
    node: ast.ModuleNode | None = None
    #: type name -> (TagClass, number, mode) for a type ASSIGNED a tag
    #: (`Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {...}`), as written. The type itself
    #: carries the tag (`Asn1Type.tags`), so a direct encode, a component and an element
    #: naming it all put it on the wire; this is the record of what the source said.
    assigned_tags: dict[str, tuple] = field(default_factory=dict)
    #: The modules this one was lowered against, so a parameterized assignment it defines
    #: can be instantiated by another module in THIS module's environment (X.683 §9.8).
    imports: dict = field(default_factory=dict)

    def __getattr__(self, item):  # convenience delegation
        if item in ("module", "imports", "node"):  # never delegate the fields themselves
            raise AttributeError(item)
        return getattr(self.module, item)

    @property
    def recursive(self) -> dict[str, tuple[str, ...]]:
        """Every recursive type, with the strongly connected component of the containment
        graph it belongs to (`recursive_types`): `{"A": ("A", "B"), "B": ("A", "B")}`.
        Computed when asked, so lowering a module does not pay for it."""
        cached = self.__dict__.get("_recursive")
        if cached is None:
            cached = recursive_types(self.node) if self.node is not None else {}
            self.__dict__["_recursive"] = cached
        return cached


class Lowerer:
    def __init__(self, node: ast.ModuleNode, imports: dict[str, Module] | None = None):
        self.node = node
        self.assignments = node.type_assignments()
        #: X.681 §9 class definitions, by name. Needed because `CLASS.&field` resolves to
        #: the field's DECLARED type when it is a value field, and only to an open type
        #: when it is a type field -- the class definition is the only place that says
        #: which, so a front-end without this table has to guess.
        self.classes = {a.name: a for a in node.assignments if isinstance(a, ast.ClassAssignment)}
        self.imported = imports or {}
        self.types: dict[str, Asn1Type] = {}
        self.enumerations: dict[str, dict[str, int]] = {}
        self.assigned_tags: dict[str, tuple] = {}
        self.objects = {a.name: a for a in node.assignments if isinstance(a, ast.ObjectAssignment)}
        self.object_sets = {
            a.name: a for a in node.assignments if isinstance(a, ast.ObjectSetAssignment)
        }
        self._tables: dict[str, ObjectSetTable] = {}
        #: X.683 §8.2 parameterized assignments, by name. They are NOT lowered eagerly:
        #: §9.7 makes instantiation a substitution of actuals for dummy references, so
        #: there is nothing to build until a reference supplies them.
        self.parameterized = {
            a.name: a for a in node.assignments if isinstance(a, ast.ParameterizedAssignment)
        }
        #: Instances of parameterized types, by their structural key (X.683 §9.7). The key
        #: is built from actuals in which every enclosing dummy has already been replaced,
        #: so it names what the instance MEANS; nothing is ever installed in the module's
        #: own tables under a dummy's name, which is what made the old memo capture names.
        self._instances: dict[str, Asn1Type] = {}
        self._instances_in_progress: set[str] = set()
        #: The substituted body of each instance being built, by key: what a tag over a
        #: recursive reference to that instance is decided from (§31.2.7).
        self._instance_bodies: dict[str, object] = {}
        #: Lowerers of imported modules whose templates this module instantiates, by name.
        self._children: dict[str, Lowerer] = {}
        #: Rows of objects lowered in ANOTHER module and handed in as actuals (§9.8).
        self._prebuilt_rows: dict[str, dict] = {}
        self._in_progress: set[str] = set()
        #: The types and instances being built, in the order their definitions were entered,
        #: as (key, name): the path a recursive reference's containment cycle is read from.
        self._building: list[tuple[str, str]] = []
        #: Every recursive reference this lowerer made, checked once the module is complete.
        self._forwards: list[_Forward] = []
        #: The CHOICEs whose distinct-tag check waited for a recursive alternative's type.
        self._deferred_choices: list[Choice] = []
        #: Tags decided over a reference before its type was built, with the decision:
        #: confirmed against the built type once the module is complete.
        self._decided_early: list[tuple[_Forward, bool]] = []
        #: Value assignments being lowered: `a INTEGER ::= b` with `b INTEGER ::= a` names
        #: itself, and is refused rather than recursed (`_assigned_value`).
        self._values_in_progress: set[str] = set()
        #: What each template's dummies stand for, by (template, position): it depends on the
        #: template alone, so it is decided once rather than at every instantiation.
        self._dummy_kinds: dict[tuple[str, int], str] = {}

    # --- entry point ------------------------------------------------------------------

    def run(self) -> LoweredModule:
        for name in self.assignments:
            self._type_by_name(name)
        self._finish(self.node.name)
        oid = self._oid(self.node.oid) if self.node.oid else ()
        module = Module(self.node.name, oid, dict(self.types))
        return LoweredModule(
            module,
            self.node.tag_default,
            self.enumerations,
            self.node,
            dict(self.assigned_tags),
            dict(self.imported),
        )

    def _finish(self, module: str) -> None:
        """What waited for the module to be complete, decided now -- over this lowerer and the
        lowerers of imported modules whose templates it instantiated (§9.8), each check over
        all of them before the next:

          * every recursive reference names a type: `A ::= B` with `B ::= A` passes through
            no constructor, so it names no type at all and has no value to encode;
          * a tag decided over a reference before its type existed agrees with the type built
            (§31.2.7), so a definition the syntax was misread from is refused, never encoded;
          * X.680 §29.3 for every CHOICE whose alternatives' tags needed a type still being
            built when the CHOICE was."""
        lowerers: list[Lowerer] = []
        pending = [self]
        while pending:
            lowerer = pending.pop()
            if all(lowerer is not done for done in lowerers):
                lowerers.append(lowerer)
                pending.extend(lowerer._children.values())
        for forward in (f for lowerer in lowerers for f in lowerer._forwards):
            try:
                resolve(forward)
            except Asn1Error:
                raise Asn1SemanticError(
                    f"module {module}: {forward.name} is defined only as a reference to itself "
                    f"({' -> '.join(forward.cycle)}); a recursive definition has to pass "
                    f"through a SEQUENCE, SET, CHOICE, SEQUENCE OF or SET OF, or it names no "
                    f"type"
                ) from None
        for reference, decided in (d for lowerer in lowerers for d in lowerer._decided_early):
            if _needs_explicit_tag(reference) != decided:
                raise Asn1SemanticError(
                    f"module {module}: a tag over {reference.name} was decided "
                    f"{'EXPLICIT' if decided else 'IMPLICIT'} before {reference.name} was "
                    f"built, and the type built disagrees (X.680 31.2.7); tag the reference "
                    f"EXPLICITly"
                )
        for kind in (c for lowerer in lowerers for c in lowerer._deferred_choices):
            try:
                kind.check_tags()
            except Asn1Error as exc:
                raise Asn1SemanticError(f"module {module}: {exc}") from None
            kind.tags_checked = True
        for lowerer in lowerers:
            for forward in lowerer._forwards:
                forward.lowerer = None  # the schema does not keep a lowerer alive
            lowerer._forwards, lowerer._decided_early, lowerer._deferred_choices = [], [], []

    def _oid(self, value: ast.OidValue) -> tuple[int, ...]:
        arcs: list[int] = []
        for arc in value.arcs:
            if arc.number is None:
                number = _WELL_KNOWN_ARCS.get(arc.name)
                if number is None:
                    raise Asn1SemanticError(
                        f"module {self.node.name}: object identifier arc {arc.name!r} "
                        f"has no number and is not a well-known arc name; write it as "
                        f"{arc.name}(n)"
                    )
                arcs.append(number)
            else:
                arcs.append(arc.number)
        return tuple(arcs)

    # --- types ------------------------------------------------------------------------

    def _type_by_name(self, name: str) -> Asn1Type:
        if name in self.types:
            return self.types[name]
        if name in self._in_progress:
            return self._forward(name, self.types, name)  # a recursive definition
        if name not in self.assignments:
            for module in self.imported.values():
                if name in module.types:
                    return module.types[name]
            raise Asn1SemanticError(
                f"module {self.node.name}: type {name!r} is referenced but never "
                f"assigned, and no IMPORTS provides it"
            )
        self._in_progress.add(name)
        self._building.append((name, name))
        try:
            built = self._type(self.assignments[name], name)
        finally:
            self._in_progress.discard(name)
            self._building.pop()
        self.types[name] = built
        return built

    def _forward(self, key: str, registry: dict, name: str) -> _Forward:
        """A reference to the type or instance `key`, still being built: it closes the
        containment cycle from `key` through everything entered since, back to `key`."""
        keys = [entry[0] for entry in self._building]
        start = keys.index(key) if key in keys else len(keys)
        cycle = tuple(entry[1] for entry in self._building[start:]) + (name,)
        forward = _Forward(key, registry, name, cycle, self)
        self._forwards.append(forward)
        return forward

    def _type(self, node, label: str) -> Asn1Type:
        if isinstance(node, ast.PreLowered):
            if node.kind != _TYPE:
                raise Asn1SemanticError(
                    f"{label}: a {node.kind} actual is used where a type is required"
                )
            if node.tag is None:
                return node.payload
            # A tagged actual standing where no component peels its tag -- the whole body of
            # `Id {T} ::= T`, an element of SEQUENCE OF T: the tag goes with the type, as
            # resolved in the module that wrote it (X.683 9.8).
            cls, number, explicit = node.tag
            return self._with_tag(node.payload, cls, number, explicit)
        if isinstance(node, ast.TypeRef):
            if node.module is not None:
                target = self.imported.get(node.module)
                if target is None or node.name not in target.types:
                    raise Asn1SemanticError(
                        f"{label}: external reference {node.module}.{node.name} is not "
                        f"available; pass the module in `imports`"
                    )
                return target.types[node.name]
            return self._type_by_name(node.name)

        if isinstance(node, ast.ParameterizedRef):
            return self._instantiate(node, label)

        if isinstance(node, ast.Constrained):
            built = self._type(node.inner, label)
            return self._apply_constraints(built, node.constraints, label)

        if isinstance(node, ast.OpenTypeNode):
            return self._open_type(node, label)

        if isinstance(node, ast.Builtin):
            return self._builtin(node, label)

        if isinstance(node, ast.Tagged):
            # A tagged type (`Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {...}`, or an
            # element's `SEQUENCE OF [0] INTEGER`). X.680 §31 makes it a type, so the tag is
            # the built type's own (`Asn1Type.tags`) and goes wherever the type goes: a
            # direct encode, every component and element naming it, and the tag-ordered SETs
            # and CHOICEs of X.691 §21 / X.696 §18.2. The type model used to carry tags on
            # components only, so a component's own tag over a tagged type, an element's tag
            # and an outermost encode all dropped one (audit 2026-10-06 §4.4).
            inner, number, mode, cls = _peel_tag(node)
            self.assigned_tags[label] = (cls, number, mode)
            built = self._type(inner, label)
            return self._with_tag(built, cls, number, self._explicit(mode, built))

        if isinstance(node, ast.SequenceOfType):
            element = self._type(node.element, f"{label} element")
            return SequenceOf(element, f"SEQUENCE OF {_render_name(node.element)}")

        if isinstance(node, ast.SetOfType):
            element = self._type(node.element, f"{label} element")
            return SetOf(element, f"SET OF {_render_name(node.element)}")

        if isinstance(node, ast.SequenceType):
            comps, ext = self._components(node.components, label)
            return Sequence(comps, label, ext)

        if isinstance(node, ast.SetType):
            comps, ext = self._components(node.components, label)
            return Set(comps, label, ext)

        if isinstance(node, ast.ChoiceType):
            alts, ext = self._components(node.alternatives, label, choice=True)
            built = Choice(alts, label, ext)
            if not built.tags_checked:  # a recursive alternative: decided once the module is
                self._deferred_choices.append(built)
            return built

        raise Asn1SemanticError(f"{label}: unsupported type node {type(node).__name__}")

    def _apply_constraints(self, built: Asn1Type, applied, label: str) -> Asn1Type:
        """Attach a constraint to the type it constrains.

        Only the types whose ENCODING depends on a constraint carry one: an integer or a
        string (X.696 §10/§13/§14/§27) and a sequence-of/set-of (its SIZE bounds the
        occurrence count). A constraint on anything else is dropped, because there is no
        encoding decision for it to inform -- and it is still checked for satisfiability
        first, so an empty value set is not silently discarded along with it.
        """
        from bcir.asn1.constraints import Intersection, require_satisfiable

        # X.682 §11 CONTAINING / ENCODED BY and §9 CONSTRAINED BY are not element set specs
        # and never reach the value-set machinery: §11 says what the contents octets ARE,
        # and §9 is explicitly "a special form of ASN.1 comment" (§9 NOTE 1). Both are split
        # off here so `Intersection` is only ever handed real element set specs.
        contents = [c for c in applied if isinstance(c, ast.ContentsConstraintNode)]
        applied = [
            c
            for c in applied
            if not isinstance(c, (ast.ContentsConstraintNode, ast.UserDefinedConstraintNode))
        ]
        if contents:
            spec = contents[0]
            if not isinstance(built, Primitive) or built.universal not in (
                Universal.OCTET_STRING,
                Universal.BIT_STRING,
            ):
                raise Asn1SemanticError(
                    f"{label}: a contents constraint applies only to OCTET STRING and to "
                    f"BIT STRING without a NamedBitList (X.682 11.3)"
                )
            built = replace(
                built,
                contains=(
                    self._type(spec.contained, f"{label} CONTAINING")
                    if spec.contained is not None
                    else None
                ),
                encoded_by=self._encoded_by(spec.encoded_by, label),
            )
        if not applied:
            return built
        applied = [self._resolve_constraint(c, built, label) for c in applied]
        combined = applied[0] if len(applied) == 1 else Intersection(tuple(applied))
        inner = getattr(built, "constraint", None)
        if inner is not None:
            # SERIAL APPLICATION (X.680 §50.11, X.691 §10.3.20/§10.3.21). `NameString
            # (SIZE(1))` constrains a type that is ALREADY constrained, and the effective
            # constraint is the intersection of the two, not the outer one: the outer SIZE
            # narrows the length while NameString's permitted alphabet survives untouched.
            # Overwriting here silently widened the alphabet back to the base type's, which
            # is invisible to BER/DER/OER -- they encode the value identically either way --
            # and changes the WIDTH of every character under PER (X.691 §30.5.2).
            #
            # §50.11 also DROPS the parent's extension marker: the serially applied
            # constraint behaves "as if the constraint had been applied to the parent type
            # without its extension marker". So the parent is stripped before intersecting,
            # and the result is extensible only if the OUTER constraint says so.
            from bcir.asn1.constraints import without_extension

            combined = Intersection((without_extension(inner), combined))
        require_satisfiable(combined, label)
        if isinstance(built, (Primitive, SequenceOf, SetOf)):
            return replace(built, constraint=combined)
        return built

    def _resolve_constraint(self, constraint, built: Asn1Type, label: str):
        """Resolve every name a subtype constraint mentions (X.680 §51).

        A value reference becomes the value it names -- a value of this module, of a module it
        imports, or an item of the constrained type -- and a contained subtype becomes its
        type's own constraint. A name that resolves to nothing is a refusal: the constraint is
        never attached with one left in it, because under PER and OER dropping it changes the
        octets (`constraints.require_satisfiable` refuses any that survives)."""
        if not isinstance(constraint, _constraints.Constraint) or not _constraints.references(
            constraint
        ):
            return constraint

        def value_of(reference, context):
            return self._constraint_value(reference.name, context, built, label)

        def subtype_of(reference, context):
            return self._contained_subtype(reference.name, context, built, label)

        return _constraints.resolve_references(constraint, value_of, subtype_of)

    def _constraint_value(self, name: str, context: str, built: Asn1Type, label: str):
        governor = _INTEGER if context == "size" else built
        value = self._named_value(name, governor, label, what="the constraint names")
        if context == "size":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Asn1SemanticError(
                    f"{label}: SIZE bound {name!r} is {value!r}, not a non-negative integer "
                    f"(X.680 51.5)"
                )
        elif context == "alphabet":
            if not isinstance(value, str):
                raise Asn1SemanticError(
                    f"{label}: FROM bound {name!r} is not a character string (X.680 51.7)"
                )
        elif isinstance(built, Primitive) and built.universal == Universal.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                raise Asn1SemanticError(
                    f"{label}: value bound {name!r} is {value!r}, not an INTEGER value (X.680 51.4)"
                )
        return value

    def _contained_subtype(self, name: str, context: str, built: Asn1Type, label: str):
        """§51.3: the element is every value of the named type -- its constraint, or MIN..MAX."""
        target = self._type_by_name(name)
        if isinstance(target, Reference):
            if not target.ready():
                raise Asn1SemanticError(
                    f"{label}: {name} is used as a contained subtype of itself (X.680 51.3)"
                )
            target = resolve(target)
        if (
            context == "value"
            and isinstance(built, Primitive)
            and isinstance(target, Primitive)
            and built.universal != target.universal
        ):
            raise Asn1SemanticError(
                f"{label}: contained subtype {name} is a {target.name}, not a subtype of "
                f"{built.name} (X.680 51.3.1)"
            )
        constraint = getattr(target, "constraint", None)
        if constraint is None:
            return _constraints.ValueRange(None, None)
        if context == "alphabet":
            if isinstance(constraint, _constraints.PermittedAlphabet):
                return constraint.inner
            raise Asn1SemanticError(
                f"{label}: contained subtype {name} inside FROM states no permitted alphabet"
            )
        return constraint

    def _open_type(self, node: ast.OpenTypeNode, label: str) -> Asn1Type:
        """Resolve `ANY [DEFINED BY x]` and `CLASS.&field` (X.681 §14/§15).

        A `CLASS.&Type` reference is genuinely open. A `CLASS.&id` reference is NOT: the
        class declared a type for that value field, so lowering it to an open type would
        throw away information the module supplied and turn a checkable OBJECT IDENTIFIER
        into opaque octets.
        """
        if node.object_class is not None:
            declared = self._class(node.object_class)
            if declared is None:
                raise Asn1SemanticError(
                    f"{label}: information object class {node.object_class!r} is "
                    f"referenced but never defined in this module"
                )
            for field in declared.fields:
                if field.name != node.field:
                    continue
                table = self._table_for(node, label)
                if field.is_type_field:
                    governing, columns = self._governing(node, declared, label)
                    return OpenType(
                        f"{node.object_class}{node.field}",
                        table=table,
                        field=node.field,
                        governing=governing,
                        governing_fields=columns,
                    )
                if field.type is None:
                    raise Asn1SemanticError(
                        f"{label}: {node.object_class}{node.field} is a value field with "
                        f"no declared type, so it has no encoding"
                    )
                built = self._type(field.type, label)
                if table is not None and isinstance(built, Primitive):
                    # X.682 §10.6 b): a value field is restricted to its column. This rides
                    # on `table_values`, NOT `constraint` -- X.691 §10.3.4 makes a table
                    # constraint invisible to PER, so narrowing `constraint` here would
                    # change the field's encoded WIDTH from a constraint the encoder is
                    # required not to see.
                    column = table.column(node.field)
                    if column:
                        built = replace(built, table_values=tuple(column))
                return built
            raise Asn1SemanticError(f"{label}: {node.object_class} has no field {node.field!r}")
        governed = f" DEFINED BY {node.governed_by}" if node.governed_by else ""
        return OpenType(f"ANY{governed}")

    # --- X.683 parameterization ----------------------------------------------------------
    #
    # Hygiene is the whole design. An instance is lowered from its template's body with every
    # dummy reference replaced structurally (`_substitute`) by what the instantiation bound
    # it to, so a name left in the body is resolved where the TEMPLATE was written -- this
    # module's assignments, or an imported module's -- and never through another template's
    # bindings. Nothing is installed in the module's tables under a dummy's name, and an
    # actual written in braces is registered once under a content-addressed name that no
    # assignment can share. The memo key is built from actuals with every enclosing dummy
    # already replaced, so it names what the instance means.

    def _instantiate(self, node: ast.ParameterizedRef, label: str) -> Asn1Type:
        """X.683 §9.7: build the type a parameterized reference denotes."""
        template, owner = self._template(node, label)
        self._check_arity(template, node, label)
        if not isinstance(template.body, ast.TypeAssignment):
            raise Asn1SemanticError(
                f"{label}: only a parameterized TYPE assignment can be referenced as a "
                f"type; {node.name} assigns a {type(template.body).__name__}"
            )
        if owner is not self:
            bindings = owner._bind(
                template, self._prelower_all(owner, template, node, label), label
            )
            return owner._lower_instance(template, bindings, label)
        return self._lower_instance(template, self._bind(template, node.actuals, label), label)

    def _check_arity(self, template, node, label: str) -> None:
        if len(node.actuals) != len(template.params):
            raise Asn1SemanticError(
                f"{label}: {node.name} takes {len(template.params)} parameter(s), "
                f"{len(node.actuals)} supplied (X.683 9.6)"
            )

    def _template(self, node: ast.ParameterizedRef, label: str):
        """The parameterized assignment a reference names, and the lowerer of its module."""
        if node.module is None and node.name in self.parameterized:
            return self.parameterized[node.name], self
        for module_name, imported in self.imported.items():
            if node.module is not None and module_name != node.module:
                continue
            inner = getattr(imported, "node", None)
            if inner is None:
                continue
            for assignment in inner.assignments:
                if (
                    isinstance(assignment, ast.ParameterizedAssignment)
                    and assignment.name == node.name
                ):
                    return assignment, self._child(module_name, imported)
        raise Asn1SemanticError(
            f"{label}: {node.name!r} is referenced with actual parameters but is not a "
            f"parameterized assignment of this module or of a module it imports (X.683 9.2)"
        )

    def _child(self, module_name: str, imported) -> "Lowerer":
        """A lowerer for an imported module, so its templates lower in its own environment."""
        child = self._children.get(module_name)
        if child is None:
            child = Lowerer(imported.node, getattr(imported, "imports", None) or {})
            child.types.update(imported.module.types)
            child.enumerations.update(getattr(imported, "enumerations", {}) or {})
            child.assigned_tags.update(getattr(imported, "assigned_tags", {}) or {})
            self._children[module_name] = child
        return child

    def _governor(self, template, index: int):
        governors = template.governors or ()
        return governors[index] if index < len(governors) else None

    def _dummy_kind(self, template, index: int) -> str:
        """X.683 §8.3/§8.4: what the dummy at `index` stands for, from its governor."""
        key = (template.name, index)
        kind = self._dummy_kinds.get(key)
        if kind is None:
            kind = self._dummy_kinds[key] = self._classify_dummy(template, index)
        return kind

    def _classify_dummy(self, template, index: int) -> str:
        dummy, governor = template.params[index], self._governor(template, index)
        if governor is None:
            return _CLASS if _used_as_class(template.body, dummy) else _TYPE
        if isinstance(governor, ast.TypeRef) and governor.module is None:
            other = governor.name
            if other in template.params:  # a DummyGovernor
                governed = self._dummy_kind(template, template.params.index(other))
                if governed == _CLASS:
                    return _OBJECT if dummy[:1].islower() else _OBJECT_SET
                return _VALUE if dummy[:1].islower() else _VALUE_SET
            if self._class(other) is not None:
                return _OBJECT if dummy[:1].islower() else _OBJECT_SET
        return _VALUE if dummy[:1].islower() else _VALUE_SET

    def _class(self, name: str):
        """A class definition visible here: this module's, or an imported module's."""
        found = self.classes.get(name)
        if found is not None:
            return found
        for imported in self.imported.values():
            inner = getattr(imported, "node", None)
            if inner is None:
                continue
            for assignment in inner.assignments:
                if isinstance(assignment, ast.ClassAssignment) and assignment.name == name:
                    return assignment
        return None

    def _bind(self, template, actuals, label: str) -> dict:
        """One binding per dummy, each checked against what its dummy stands for (§9.6)."""
        out = {}
        for index, (dummy, actual) in enumerate(zip(template.params, actuals)):
            out[dummy] = self._binding(template, index, actual, f"{label}[{template.name}]")
        return out

    def _binding(self, template, index: int, actual, label: str) -> _Binding:
        dummy = template.params[index]
        kind = self._dummy_kind(template, index)
        governor = self._governor(template, index)
        where = f"{label}: the actual for {dummy}"
        if isinstance(actual, ast.PreLowered):
            return self._adopt(kind, governor, actual, where)
        if kind == _TYPE:
            if isinstance(actual, str):
                actual = ast.TypeRef(actual)
            if isinstance(actual, ast.BracedActual) or isinstance(actual, _VALUE_NODES):
                raise Asn1SemanticError(f"{where} must be a type (X.683 9.6)")
            return _Binding(_TYPE, actual, key="type:" + _key(actual))
        if kind == _VALUE:
            node = actual
            if isinstance(node, str):
                node = ast.RefValue(node)
            elif isinstance(node, ast.BracedActual):
                node = self._reparse(node.raw, "value", where)
            elif isinstance(node, ast.Builtin) and node.name == "NULL":
                node = ast.NullValue()  # NULL is both a type and a value; the governor decides
            elif not isinstance(node, _VALUE_NODES):
                raise Asn1SemanticError(f"{where} must be a value (X.683 9.6)")
            value = self.value(node, self._type(governor, f"{where} governor"), where)
            return _Binding(
                _VALUE, ast.PreLowered(_VALUE, value, key=_key(value)), key="value:" + _key(value)
            )
        if kind == _VALUE_SET:
            if isinstance(actual, ast.BracedActual):
                element = self._reparse(actual.raw, "value-set", where)
            elif isinstance(actual, (str, ast.TypeRef)):
                element = _constraints.TypeReference(
                    actual if isinstance(actual, str) else actual.name
                )
            else:
                raise Asn1SemanticError(f"{where} must be a value set (X.683 9.6)")
            if element is None:
                element = _constraints.ValueRange(None, None)  # the model cannot narrow it
            return _Binding(
                _VALUE_SET,
                ast.Constrained(governor, (element,)),
                element,
                key="valueset:" + _key(element) + "/" + _key(governor),
            )
        if kind == _CLASS:
            name = (
                actual.name if isinstance(actual, ast.TypeRef) and actual.module is None else actual
            )
            if not isinstance(name, str) or self._class(name) is None:
                raise Asn1SemanticError(f"{where} must be an information object class (X.683 9.6)")
            return _Binding(_CLASS, name, key="class:" + name)
        # an object or an object set
        class_name = governor.name if isinstance(governor, ast.TypeRef) else None
        if isinstance(actual, ast.BracedActual):
            name = self._register_inline(kind, actual.raw, class_name, where)
        elif isinstance(actual, ast.ParameterizedRef):
            name = self._object_instance(actual, where)
        elif isinstance(actual, (str, ast.TypeRef)):
            name = actual if isinstance(actual, str) else actual.name
            known = self.object_sets if kind == _OBJECT_SET else self.objects
            if name not in known and name not in self._tables and name not in self._prebuilt_rows:
                what = "an object set" if kind == _OBJECT_SET else "an object"
                raise Asn1SemanticError(f"{where} must be {what}; {name!r} is not one (X.683 9.6)")
        else:
            raise Asn1SemanticError(f"{where} must be an object or object set (X.683 9.6)")
        return _Binding(kind, name, key=f"{kind}:{name}")

    def _adopt(self, kind: str, governor, actual: ast.PreLowered, where: str) -> _Binding:
        """Bind an actual another module lowered (§9.8) under a content-addressed name."""
        if actual.kind != kind:
            raise Asn1SemanticError(f"{where} must be a {kind}, not a {actual.kind} (X.683 9.6)")
        key = f"pre:{kind}:{actual.key}"
        if kind in (_TYPE, _VALUE):
            return _Binding(kind, actual, key=key)
        if kind == _VALUE_SET:
            return _Binding(
                kind, ast.Constrained(governor, (actual.payload,)), actual.payload, key=key
            )
        name = self._synthetic("Imported", key)
        if kind == _CLASS:
            self.classes.setdefault(name, replace(actual.payload, name=name))
        elif kind == _OBJECT_SET:
            self._tables.setdefault(name, actual.payload)
        else:
            self._prebuilt_rows.setdefault(name, actual.payload)
        return _Binding(kind, name, key=key)

    def _prelower_all(self, owner: "Lowerer", template, node: ast.ParameterizedRef, label: str):
        """Lower each actual HERE, where it is written, for a template of another module."""
        out = []
        for index, actual in enumerate(node.actuals):
            kind = owner._dummy_kind(template, index)
            governor = owner._governor(template, index)
            out.append(self._prelower(kind, governor, owner, actual, f"{label}[{node.name}]"))
        return tuple(out)

    def _prelower(
        self, kind: str, governor, owner: "Lowerer", actual, label: str
    ) -> ast.PreLowered:
        here = self.node.name
        if kind == _TYPE:
            node = ast.TypeRef(actual) if isinstance(actual, str) else actual
            inner, number, mode, cls = _peel_tag(node)
            built = self._type(inner, label)
            tag = None if number is None else (cls, number, self._explicit(mode, built))
            return ast.PreLowered(_TYPE, built, key=f"{here}:{_key(node)}", tag=tag)
        if kind == _VALUE:
            node = actual
            if isinstance(node, str):
                node = ast.RefValue(node)
            elif isinstance(node, ast.BracedActual):
                node = self._reparse(node.raw, "value", label)
            elif isinstance(node, ast.Builtin) and node.name == "NULL":
                node = ast.NullValue()
            value = self.value(node, owner._type(governor, f"{label} governor"), label)
            return ast.PreLowered(_VALUE, value, key=_key(value))
        if kind == _VALUE_SET:
            if isinstance(actual, ast.BracedActual):
                element = self._reparse(actual.raw, "value-set", label)
            else:
                element = _constraints.TypeReference(
                    actual if isinstance(actual, str) else actual.name
                )
            element = self._resolve_constraint(element, owner._type(governor, label), label)
            return ast.PreLowered(_VALUE_SET, element, key=f"{here}:{_key(element)}")
        if kind == _CLASS:
            name = actual.name if isinstance(actual, ast.TypeRef) else actual
            found = self._class(name) if isinstance(name, str) else None
            if found is None:
                raise Asn1SemanticError(
                    f"{label}: {name!r} is not an information object class (X.683 9.6)"
                )
            return ast.PreLowered(_CLASS, found, key=f"{here}:class:{name}")
        class_name = governor.name if isinstance(governor, ast.TypeRef) else None
        if isinstance(actual, ast.BracedActual):
            name = self._register_inline(kind, actual.raw, class_name, label)
        elif isinstance(actual, ast.ParameterizedRef):
            name = self._object_instance(actual, label)
        else:
            name = actual.name if isinstance(actual, ast.TypeRef) else actual
        if kind == _OBJECT_SET:
            return ast.PreLowered(
                _OBJECT_SET, self._object_set_table(name, label), key=f"{here}:set:{name}"
            )
        obj = self.objects.get(name)
        if obj is None:
            raise Asn1SemanticError(f"{label}: {name!r} is not an object (X.683 9.6)")
        return ast.PreLowered(
            _OBJECT, self._row(obj.settings, obj.object_class, label), key=f"{here}:obj:{name}"
        )

    def _lower_instance(self, template, bindings: dict, label: str) -> Asn1Type:
        key = (
            f"{self.node.name}.{template.name}{{" + ",".join(b.key for b in bindings.values()) + "}"
        )
        built = self._instances.get(key)
        if built is not None:
            return built
        if key in self._instances_in_progress:
            # A template that refers to itself with the same actuals is a recursive TYPE,
            # which X.680 permits; it is a lazy reference to the instance being built.
            return self._forward(key, self._instances, f"{template.name}{{...}}")
        if len(self._instances_in_progress) >= _MAX_INSTANCE_DEPTH:
            raise Asn1SemanticError(
                f"{label}: instantiating {template.name} nests {_MAX_INSTANCE_DEPTH} templates "
                f"deep; a template whose recursive reference changes its own actuals denotes "
                f"an infinite family of types (X.683 9.7)"
            )
        substituted = _substitute(template.body.type, bindings)
        self._instances_in_progress.add(key)
        self._instance_bodies[key] = substituted
        self._building.append((key, f"{template.name}{{...}}"))
        try:
            built = self._type(substituted, f"{label}[{template.name}]")
        finally:
            self._instances_in_progress.discard(key)
            del self._instance_bodies[key]
            self._building.pop()
        self._instances[key] = built
        return built

    def _object_instance(self, ref: ast.ParameterizedRef, label: str) -> str:
        """Instantiate a parameterized object or object set; its content-addressed name."""
        template, owner = self._template(ref, label)
        self._check_arity(template, ref, label)
        body = template.body
        if not isinstance(body, (ast.ObjectSetAssignment, ast.ObjectAssignment)):
            raise Asn1SemanticError(
                f"{label}: {ref.name} is a parameterized {type(body).__name__}, used where an "
                f"object or object set is required (X.683 9.2)"
            )
        if owner is not self:
            bindings = owner._bind(template, self._prelower_all(owner, template, ref, label), label)
            name = owner._object_instance_bound(template, bindings, label)
            key = f"{owner.node.name}:{name}"
            if isinstance(body, ast.ObjectSetAssignment):
                return self._adopt(
                    _OBJECT_SET,
                    None,
                    ast.PreLowered(_OBJECT_SET, owner._object_set_table(name, label), key=key),
                    label,
                ).actual
            obj = owner.objects[name]
            row = owner._row(obj.settings, obj.object_class, label)
            return self._adopt(_OBJECT, None, ast.PreLowered(_OBJECT, row, key=key), label).actual
        return self._object_instance_bound(
            template, self._bind(template, ref.actuals, label), label
        )

    def _object_instance_bound(self, template, bindings: dict, label: str) -> str:
        key = f"{template.name}{{" + ",".join(b.key for b in bindings.values()) + "}"
        name = self._synthetic(template.name, key)
        body = _substitute(template.body, bindings)
        if isinstance(body, ast.ObjectSetAssignment):
            if name not in self.object_sets:
                elements = tuple(
                    self._set_element(e, body.object_class, label) for e in body.elements
                )
                self.object_sets[name] = replace(body, name=name, elements=elements)
        elif name not in self.objects:
            self.objects[name] = replace(body, name=name)
        return name

    def _set_element(self, element, class_name: str, label: str):
        """A parameterized reference inside an object set becomes the instance's name."""
        if isinstance(element, ast.ParameterizedRef):
            return self._object_instance(element, label)
        return element

    def _synthetic(self, prefix: str, key: str) -> str:
        """A content-addressed name for something written in place: a lexable reference (it
        may be re-read inside a braced actual) that no assignment of the module can share."""
        stem = "".join(c if c.isalnum() else "-" for c in prefix).strip("-") or "X"
        while "--" in stem:
            stem = stem.replace("--", "-")
        name = f"Bcir-{stem}-{_digest(key)}"
        if name in self.assignments or name in self.parameterized:
            raise Asn1SemanticError(f"the module assigns {name!r}, a name the front end reserves")
        return name

    def _register_inline(self, kind: str, raw: str, class_name: str | None, label: str) -> str:
        """An object or object set written as a braced actual, registered once by content."""
        if class_name is None or self._class(class_name) is None:
            raise Asn1SemanticError(f"{label}: a braced {kind} needs its class as the governor")
        name = self._synthetic("Inline", f"{kind}\x00{class_name}\x00{raw}")
        if kind == _OBJECT_SET and name not in self.object_sets:
            elements, extensible = self._reparse(raw, "object-set", label, class_name)
            elements = tuple(self._set_element(e, class_name, label) for e in elements)
            self.object_sets[name] = ast.ObjectSetAssignment(
                name, class_name, (), raw, elements, extensible
            )
        elif kind == _OBJECT and name not in self.objects:
            settings = self._reparse(raw, "object", label, class_name)
            self.objects[name] = ast.ObjectAssignment(name, class_name, raw, tuple(settings))
        return name

    def _reparse(self, raw: str, production: str, label: str, class_name: str | None = None):
        """Read a braced actual in the production its dummy names (§9.6)."""
        from .parser import Parser

        parser = Parser(raw, f"{label}<actual>")
        cls = self._class(class_name) if class_name else None
        try:
            if production == "object-set":
                out = parser._parse_object_set_body(cls)
            elif production == "object":
                out = parser._parse_object_body(cls)
            elif production == "value":
                out = parser.parse_value()
            else:
                out = parser._value_set()
        except Asn1SyntaxError as exc:
            raise Asn1SemanticError(
                f"{label}: the braced actual {raw!r} is not a {production}: {exc}"
            ) from None
        if parser.current.kind != "end":
            raise Asn1SemanticError(
                f"{label}: the braced actual {raw!r} has text after the {production}"
            )
        return out

    def _encoded_by(self, node, label: str) -> tuple | None:
        """X.682 §11.2: the ENCODED BY value, which shall be an OBJECT IDENTIFIER.

        `{2 1 1}` is ambiguous in isolation -- the parser reads a braced literal as a
        `BracedValue` carrying BOTH readings -- so the arcs are taken from whichever shape
        arrived rather than assuming one. Anything that is not an object identifier is a
        specification error under §11.2 and is refused instead of being dropped.
        """
        if node is None:
            return None
        if isinstance(node, ast.OidValue):
            return self._oid(node)
        arcs = getattr(node, "arcs", None)
        if arcs:
            return self._oid(ast.OidValue(arcs))
        raise Asn1SemanticError(
            f"{label}: ENCODED BY takes an object identifier value (X.682 11.2)"
        )

    def _table_for(self, node: ast.OpenTypeNode, label: str):
        """The associated table (X.681 §13) of the object set a table constraint names."""
        if node.table is None:
            return None
        if node.table.spec is not None:
            return self._object_set_table(
                self._spec_set(node.table.spec, node.object_class, label), label
            )
        return self._object_set_table(node.table.object_set, label)

    def _spec_set(self, spec: ast.ObjectSetSpec, class_name: str, label: str) -> str:
        """X.682 §10.3: a table constraint's ObjectSet written in place, registered by content."""
        elements = tuple(self._set_element(e, class_name, label) for e in spec.elements)
        name = self._synthetic("Spec", _key((class_name, elements, spec.extensible)))
        if name not in self.object_sets:
            self.object_sets[name] = ast.ObjectSetAssignment(
                name, class_name, (), "", elements, spec.extensible
            )
        return name

    def _object_set_table(self, name: str, label: str):
        """Build one object set's associated table, resolving references and unions."""
        if name in self._tables:
            return self._tables[name]
        assignment = self.object_sets.get(name)
        if assignment is None:
            raise Asn1SemanticError(
                f"{label}: table constraint names object set {name!r}, which this module "
                f"does not define (X.682 10.4)"
            )
        rows: list[dict] = []
        extensible = assignment.extensible
        self._tables[name] = ObjectSetTable(assignment.object_class, (), extensible)
        for element in assignment.elements:
            if isinstance(element, ast.ParameterizedRef):
                element = self._object_instance(element, label)
            if isinstance(element, str):
                if element in self._prebuilt_rows:  # an object lowered in another module
                    rows.append(self._prebuilt_rows[element])
                    continue
                if element in self._tables and element not in self.object_sets:
                    inner = self._tables[element]  # a set lowered in another module
                    rows.extend(inner.rows)
                    extensible = extensible or inner.extensible
                    continue
                # §12.5: a referenced object set is spliced in and its extension marker is
                # inherited; a referenced OBJECT contributes its single row.
                nested = self.object_sets.get(element)
                if nested is not None:
                    inner = self._object_set_table(element, label)
                    rows.extend(inner.rows)
                    extensible = extensible or inner.extensible
                    continue
                obj = self.objects.get(element)
                if obj is None:
                    raise Asn1SemanticError(
                        f"{label}: object set {name!r} references {element!r}, which is "
                        f"neither a defined object nor a defined object set"
                    )
                rows.append(self._row(obj.settings, assignment.object_class, label))
                continue
            rows.append(self._row(element, assignment.object_class, label))
        table = ObjectSetTable(assignment.object_class, tuple(rows), extensible)
        self._tables[name] = table
        return table

    def _row(self, settings, class_name: str, label: str) -> dict:
        """One row of the associated table: a field name -> cell mapping (§13.4 a)).

        A type field's cell is a lowered `Asn1Type`; a value field's cell is a Python value.
        That asymmetry is §13.1's, not an implementation shortcut -- the columns of a class
        genuinely hold different kinds of thing.
        """
        declared = self._class(class_name)
        by_name = {f.name: f for f in declared.fields} if declared else {}
        row: dict = {}
        for setting in settings:
            field = by_name.get(setting.name)
            is_type = (
                field.is_type_field
                if field is not None
                else len(setting.name) > 1 and setting.name[1].isupper()
            )
            if is_type:
                row[setting.name] = self._type(setting.value, f"{label}.{setting.name}")
            else:
                governor = (
                    self._type(field.type, label)
                    if field is not None and field.type is not None
                    else None
                )
                row[setting.name] = self.value(setting.value, governor, f"{label}.{setting.name}")
        return row

    def _governing(self, node: ast.OpenTypeNode, declared, label: str):
        """Turn §10.7's AtNotation list into (component paths, matching class columns).

        §10.10's level counting is deliberately NOT resolved to an absolute path here: the
        dots count enclosing constructions, and the decoder walks the value it is building,
        so the path is kept relative and the leading dots are recorded by stripping them.
        The column each path matches comes from §10.15 -- the referenced components are
        ObjectClassFieldTypes of the same class, so their FIELD is what names the column,
        and it is looked up when the constrained type is assembled (see `_bind_governing`).
        """
        if node.table is None or not node.table.at_notations:
            return (), ()
        paths: list[tuple[str, ...]] = []
        for at in node.table.at_notations:
            body = at.lstrip("@").lstrip(".")
            paths.append(tuple(body.split(".")))
        return tuple(paths), ()

    def _builtin(self, node: ast.Builtin, label: str) -> Asn1Type:
        universal = UNIVERSAL_OF.get(node.name)
        if universal is None:
            raise Asn1SemanticError(f"{label}: unknown built-in type {node.name!r}")
        if node.named or node.extension_named:
            # The module-level side table answers "what does this DEFAULT identifier mean",
            # which an extension item can be as legitimately as a root one -- so both halves
            # go in here, unlike the codec-facing lists below.
            self.enumerations[label] = {n.name: n.number for n in node.named + node.extension_named}
        if universal == Universal.ENUMERATED:
            # Carry the enumeration ONTO the type, not just into the module-level side
            # table. The side table answers "what does this DEFAULT identifier mean"; PER
            # asks a different question -- X.691 §14.1 encodes the enumeration INDEX, so
            # the codec needs the whole root list at the point of use. BER/DER/OER never
            # needed it because they encode the value itself.
            return Primitive(
                int(universal),
                node.name,
                enumeration=tuple((n.name, n.number) for n in node.named),
                enum_extensible=bool(node.extensible),
                enum_extension=tuple((n.name, n.number) for n in node.extension_named) or None,
            )
        return Primitive(int(universal), node.name)

    def _components(self, nodes, label: str, choice: bool = False):
        # X.680 §25.1: `...` splits the list into the extension ROOT and the extension
        # ADDITIONS. BER/DER/OER encode both alike, so this pass used to drop the marker;
        # PER does not (X.691 §19.1/§19.7), so the split is recorded on each component.
        # X.680 §25.1 / X.691 §19.9 NOTE 2. One `...` opens the extension additions; a
        # SECOND `...` closes the pair, and anything written after it is part of the
        # extension ROOT again ("encoded as if they were defined immediately before the
        # extension marker pair"). Tracking a bare boolean would put those trailing root
        # components in the additions and shift every bit after the preamble.
        entries, markers, extension_of, groups = [], 0, {}, {}
        for node in nodes:
            if isinstance(node, ast.ExtensionMarker):
                markers += 1
                continue
            if isinstance(node, ast.ExtensionGroup):
                # The group is one addition; it is lowered below into a single Component
                # carrying its members.
                marker = object()
                extension_of[id(marker)] = True
                groups[id(marker)] = node
                entries.append(marker)
                continue
            if isinstance(node, ast.ComponentNode):
                extension_of[id(node)] = markers == 1
                entries.append(node)
        automatic = self._use_automatic_tags(
            [e for e in entries if isinstance(e, ast.ComponentNode)]
        )
        out: list[Component] = []
        position = -1
        for item in entries:
            position += 1
            if id(item) in groups:
                members, _ = self._components(
                    groups[id(item)].components, f"{label}.group{position}", choice
                )
                if choice:
                    # X.691 §23.8 NOTE: "Version brackets in the definition of choice
                    # extension additions have no effect on how ExtensionAdditionAlternatives
                    # are encoded." A bracket in a CHOICE is presentational only -- each
                    # member is its own alternative with its own index (§23.2), so grouping
                    # them would invent a nesting the encoding does not have.
                    for i, m in enumerate(members):
                        out.append(
                            replace(
                                m,
                                extension=True,
                                **(
                                    {"tag": position + i, "tag_class": TagClass.CONTEXT}
                                    if automatic
                                    else {}
                                ),
                            )
                        )
                    position += len(members) - 1
                    continue
                if automatic:
                    # §12.3 numbers a group's members in the enclosing list's sequence, so
                    # the group consumes as many tag numbers as it holds.
                    members = tuple(
                        replace(m, tag=position + i, tag_class=TagClass.CONTEXT)
                        for i, m in enumerate(members)
                    )
                    position += len(members) - 1
                out.append(
                    Component(
                        name=f"[[{position}]]",
                        type=Sequence(members, f"{label}.group"),
                        extension=True,
                        optional=True,
                        group=members,
                    )
                )
                continue
            inner, tag, mode, tag_cls = _peel_tag(item.type)
            if tag is None and automatic:
                tag, mode, tag_cls = position, None, TagClass.CONTEXT  # §12.3
            # A type ASSIGNED a tag carries it itself (`Asn1Type.tags`), so a component that
            # names one shows it untagged, and a tag of the component's own is one more layer
            # (or, IMPLICIT, replaces the type's outermost).
            built = self._type(inner, f"{label}.{item.name}")
            explicit = self._explicit(mode, built)
            default, has_default = self._default(item, built, f"{label}.{item.name}")
            out.append(
                Component(
                    name=item.name,
                    type=built,
                    tag=tag,
                    explicit=explicit,
                    optional=item.optional,
                    extension=extension_of[id(item)],
                    **({"tag_class": tag_cls} if tag_cls is not None else {}),
                    **({"default": default} if has_default else {}),
                )
            )
        return (
            self._bind_governing(tuple(out), entries, label),
            any(isinstance(n, ast.ExtensionMarker) for n in nodes),
        )

    def _bind_governing(self, built: tuple, entries, label: str) -> tuple:
        """Fill in each table-constrained open type's governing COLUMNS (X.682 §10.15).

        §10.15 requires the referenced components to be ObjectClassFieldTypes of the same
        class as the referencing one, so a referenced component's own `&field` is what names
        the column its value has to match. That is knowable only once the sibling list
        exists, which is why it happens here and not in `_open_type`.

        A path this list cannot resolve is left unbound rather than guessed: `OpenType.resolve`
        returns None for an incomplete binding, so the open type keeps its octets instead of
        being decoded as the wrong type.
        """
        fields: dict[str, str] = {}
        for item in entries:
            if not isinstance(item, ast.ComponentNode):
                continue  # an extension-group marker
            node = item.type
            while isinstance(node, (ast.Tagged, ast.Constrained)):
                node = node.inner
            if isinstance(node, ast.OpenTypeNode) and node.field:
                fields[item.name] = node.field
        out = []
        for comp in built:
            inner = comp.type
            if isinstance(inner, OpenType) and inner.governing and not inner.governing_fields:
                columns = tuple(fields.get(path[-1], "") for path in inner.governing)
                if all(columns):
                    inner = replace(inner, governing_fields=columns)
                    comp = replace(comp, type=inner)
            out.append(comp)
        return tuple(out)

    def _use_automatic_tags(self, entries) -> bool:
        """§12.3: automatic tagging applies only when NO component carries a tag."""
        if self.node.tag_default != "AUTOMATIC":
            return False
        return all(_peel_tag(item.type)[1] is None for item in entries)

    def _needs_explicit_tag(self, built: Asn1Type) -> bool:
        """§31.2.7 for `built`, including a recursive reference whose type is not built yet.

        That one is read from the definition it is being built from, by the lowerer building
        it, and the decision is confirmed against the type built once the module is complete
        (`_finish`), so a definition the syntax is misread from is refused, never encoded.
        Taking "not built yet" for "not a CHOICE" made `neg [1] Expr` inside
        `Expr ::= CHOICE {...}` an IMPLICIT tag over a CHOICE, and the module failed to lower."""
        if not isinstance(built, Reference):
            return isinstance(built, (Choice, OpenType)) and not built.tags
        decided = self._built_choice_or_open(built, frozenset())
        self._decided_early.append((built, decided))
        return decided

    def _built_choice_or_open(self, built: Asn1Type, seen: frozenset) -> bool:
        """Whether `built` is a CHOICE or an open type, through any chain of references -- one
        still being built is answered by the lowerer building it."""
        for _ in range(MAX_RECURSION):
            if built.tags:  # a tagged type has a tag an IMPLICIT one can replace
                return False
            if not isinstance(built, Reference):
                return isinstance(built, (Choice, OpenType))
            if not built.ready():
                lowerer = getattr(built, "lowerer", None)
                return lowerer is not None and lowerer._forward_choice_or_open(built, seen)
            built = built.resolved()
        return False

    def _forward_choice_or_open(self, forward: Reference, seen: frozenset) -> bool:
        """Whether the type or instance this lowerer is still building under `forward`'s name
        is a CHOICE or an open type, read from its definition."""
        key = (id(self), forward.target_name)
        if key in seen:
            return False  # no constructor on the way round: refused when the module completes
        if forward.registry is self._instances:
            definition = self._instance_bodies.get(forward.target_name)
        else:
            definition = self.assignments.get(forward.target_name)
        return self._reads_as_choice_or_open(definition, {}, seen | {key})

    def _reads_as_choice_or_open(self, node, dummies: dict, seen: frozenset) -> bool:
        """Whether the type the syntax `node` denotes in this module is a CHOICE or an open
        type, following what `_type` builds from it: constraints and an assignment's own tag
        are looked through (the type built is the inner one), names are followed to their
        definitions, and a template's dummies -- `dummies` maps each to (the lowerer that
        wrote the actual, the actual, that lowerer's dummies) -- are read as their actuals."""
        while True:
            if isinstance(node, ast.Constrained):
                node = node.inner
            elif isinstance(node, ast.Tagged):  # a tagged type is built with a tag of its own
                return False
            elif isinstance(node, ast.ChoiceType):
                return True
            elif isinstance(node, ast.OpenTypeNode):
                # `ANY` and `CLASS.&Type` are open; `CLASS.&id` is the type the class
                # declared for that value field, as `_open_type` builds it.
                declared = self._class(node.object_class) if node.object_class else None
                field = next(
                    (f for f in getattr(declared, "fields", ()) if f.name == node.field), None
                )
                if field is None or field.is_type_field or field.type is None:
                    return True
                node, dummies = field.type, {}
            elif isinstance(node, ast.PreLowered):
                if node.kind != _TYPE or node.tag is not None:  # a tagged actual is tagged
                    return False
                return self._built_choice_or_open(node.payload, seen)
            elif isinstance(node, (str, ast.TypeRef)):
                ref = ast.TypeRef(node) if isinstance(node, str) else node
                if ref.module is None and ref.name in dummies:
                    lowerer, actual, outer = dummies[ref.name]
                    return lowerer._reads_as_choice_or_open(actual, outer, seen)
                if ref.module is not None:
                    found = getattr(self.imported.get(ref.module), "types", {}).get(ref.name)
                    return found is not None and self._built_choice_or_open(found, seen)
                if ref.name in self.types:
                    return self._built_choice_or_open(self.types[ref.name], seen)
                if ref.name not in self.assignments:
                    found = next(
                        (m.types[ref.name] for m in self.imported.values() if ref.name in m.types),
                        None,
                    )
                    return found is not None and self._built_choice_or_open(found, seen)
                key = (id(self), ref.name)
                if key in seen:
                    return False
                node, dummies, seen = self.assignments[ref.name], {}, seen | {key}
            elif isinstance(node, ast.ParameterizedRef):
                try:
                    template, owner = self._template(node, "X.680 31.2.7")
                except Asn1SemanticError:
                    return False  # refused where the reference itself is lowered
                key = (id(owner), "template", template.name)
                if (
                    key in seen
                    or not isinstance(template.body, ast.TypeAssignment)
                    or len(template.params) != len(node.actuals)
                ):
                    return False
                inner = {
                    param: (self, actual, dummies)
                    for param, actual in zip(template.params, node.actuals)
                }
                return owner._reads_as_choice_or_open(template.body.type, inner, seen | {key})
            else:  # SEQUENCE, SET, their OFs and every builtin have a base tag
                return False

    def _with_tag(self, built: Asn1Type, cls, number: int, explicit: bool) -> Asn1Type:
        """`built` under one more tag of its own, outermost (X.680 §31), as a copy: the type
        it was built from keeps its own tags. IMPLICIT over a tagged type replaces that type's
        outermost tag and keeps the form of the layer it replaces (X.690 §8.14.4); over an
        untagged type it replaces the universal tag."""
        own = built.tags
        if explicit or not own:
            tags = ((cls, number, explicit),) + tuple(own)
        else:
            tags = ((cls, number, own[0][2]),) + tuple(own[1:])
        if isinstance(built, Reference):  # a type still being built: tag the reference
            forward = _Forward(
                built.target_name,
                built.registry,
                built.name,
                built.cycle,
                getattr(built, "lowerer", None),
                tags,
            )
            self._forwards.append(forward)
            return forward
        tagged = replace(built, tags=tags)
        if isinstance(tagged, Choice) and not tagged.tags_checked:
            self._deferred_choices.append(tagged)
        return tagged

    def _explicit(self, mode: str | None, built: Asn1Type) -> bool:
        """Resolve IMPLICIT/EXPLICIT for a component tag (§31.2.1 + §31.2.7)."""
        if mode == "EXPLICIT":
            return True
        if mode == "IMPLICIT":
            if self._needs_explicit_tag(built):
                raise Asn1SemanticError(
                    "IMPLICIT cannot tag a CHOICE or an open type: an implicit tag "
                    "replaces the base tag and neither has one (X.680 29.1/31.2.7)"
                )
            return False
        # No keyword: the module's default decides -- except that §31.2.7 forces
        # EXPLICIT over a CHOICE or an open type even in an IMPLICIT/AUTOMATIC module.
        if self._needs_explicit_tag(built):
            return True
        return self.node.tag_default == "EXPLICIT"

    # --- values -----------------------------------------------------------------------

    def _default(self, item: ast.ComponentNode, built: Asn1Type, label: str):
        if not item.has_default:
            return None, False
        return self.value(item.default, built, label), True

    def value(self, node, built: Asn1Type, label: str):
        """Turn a parsed value into the Python object the encoder model expects."""
        if isinstance(node, ast.PreLowered):
            if node.kind != _VALUE:
                raise Asn1SemanticError(
                    f"{label}: a {node.kind} actual is used where a value is required"
                )
            return node.payload  # lowered where the instantiation is written (X.683 9.8)
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
                    f"is an object identifier arc list"
                )
            return [self.value(v, element, label) for v in node.items]
        if isinstance(built, Primitive) and built.universal in (
            int(Universal.OBJECT_IDENTIFIER),
            int(Universal.RELATIVE_OID),
        ):
            from bcir.asn1.codec import Oid, RelativeOid

            if node.arcs is None:
                raise Asn1SemanticError(
                    f"{label}: {built.name} needs object identifier arcs, but the "
                    f"braced value is a comma-separated value list"
                )
            wrap = Oid if built.universal == int(Universal.OBJECT_IDENTIFIER) else RelativeOid
            return wrap(self._oid(ast.OidValue(node.arcs)))
        if node.items == ():
            # `{}` against a constructed type with no element: an empty SEQUENCE/SET.
            return {}
        raise Asn1SemanticError(
            f"{label}: a braced value needs a SEQUENCE OF / SET OF or OBJECT "
            f"IDENTIFIER type, not {built.name}"
        )

    def _assigned_value(self, assignment: ast.ValueAssignment, label: str):
        """The value a value assignment denotes, lowered against the type IT was assigned
        (X.680 §16.2): `x C ::= red` is C's `red` wherever `x` is used, so it cannot be read
        against the type that names it. A value that names itself on the way is refused."""
        name = assignment.name
        if name in self._values_in_progress:
            raise Asn1SemanticError(
                f"{label}: value {name!r} is defined in terms of itself (X.680 16.2)"
            )
        self._values_in_progress.add(name)
        try:
            governor = self._type(assignment.type, f"{label} (value {name})")
            return self.value(assignment.value, governor, label)
        finally:
            self._values_in_progress.discard(name)

    def _named_value(self, name: str, built: Asn1Type, label: str, what: str = "DEFAULT"):
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
            return self._assigned_value(assignment, label)
        for module_name, imported in self.imported.items():
            inner = getattr(imported, "node", None)
            found = inner.value_assignments().get(name) if inner is not None else None
            if found is not None:  # lowered in ITS module: its names, its enumerations
                return self._child(module_name, imported)._assigned_value(found, label)
        raise Asn1SemanticError(
            f"{label}: {what} {name!r}, which is neither an enumeration item of "
            f"{built.name} nor a value assigned in this module or one it imports"
            if what != "DEFAULT"
            else f"{label}: DEFAULT {name!r} is neither an enumeration item of "
            f"{built.name} nor a value assigned in this module or one it imports"
        )


#: Arc names X.660 fixes, so `{ iso 3 6 1 }` resolves without a number in parentheses.
_WELL_KNOWN_ARCS = {
    "itu-t": 0,
    "ccitt": 0,
    "iso": 1,
    "joint-iso-itu-t": 2,
    "joint-iso-ccitt": 2,
}


#: X.680 §8.1 tag classes, as spelled in the notation. An empty class name is the default
#: context-specific case (`[0]`), which is why it maps to CONTEXT rather than UNIVERSAL.
_TAG_CLASSES = {
    "": TagClass.CONTEXT,
    "UNIVERSAL": TagClass.UNIVERSAL,
    "APPLICATION": TagClass.APPLICATION,
    "PRIVATE": TagClass.PRIVATE,
}


def _peel_tag(node):
    """Split `[class n] IMPLICIT Type` into (Type, n, mode, class); 4x None if untagged.

    A type actual lowered in another module keeps the tag it was written with, its mode
    already resolved there (X.683 §9.8), so it is peeled with that mode spelled out."""
    if isinstance(node, ast.PreLowered) and node.tag is not None:
        cls, number, explicit = node.tag
        return replace(node, tag=None), number, ("EXPLICIT" if explicit else "IMPLICIT"), cls
    if isinstance(node, ast.Tagged):
        cls = _TAG_CLASSES.get(node.tag_class)
        if cls is None:
            raise Asn1SemanticError(f"unknown tag class {node.tag_class!r}")
        return node.inner, node.number, node.mode, cls
    return node, None, None, None


def containment_graph(node: ast.ModuleNode) -> dict[str, tuple[str, ...]]:
    """Each type (or parameterized type) the module assigns -> the ones its definition names.

    The edges are the references a value's encoding follows: components, elements,
    alternatives, tagged and constrained types, and the template a parameterized reference
    instantiates. A name in a constraint (a contained subtype) is not containment, and a
    reference into another module cannot close a cycle in this one."""
    names = set(node.type_assignments())
    names |= {a.name for a in node.assignments if isinstance(a, ast.ParameterizedAssignment)}
    graph: dict[str, tuple[str, ...]] = {}
    for assignment in node.assignments:
        if isinstance(assignment, ast.TypeAssignment):
            body = assignment.type
        elif isinstance(assignment, ast.ParameterizedAssignment) and isinstance(
            assignment.body, ast.TypeAssignment
        ):
            body = assignment.body.type
        else:
            continue
        found: set[str] = set()
        stack = [body]
        while stack:
            item = stack.pop()
            if isinstance(item, (ast.TypeRef, ast.ParameterizedRef)):
                if item.module is None and item.name in names:
                    found.add(item.name)
                if isinstance(item, ast.ParameterizedRef):
                    stack.extend(item.actuals)
            elif isinstance(item, ast.Constrained):
                stack.append(item.inner)  # the constraints are not containment
            elif isinstance(item, (tuple, list)):
                stack.extend(item)
            elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                stack.extend(getattr(item, f.name) for f in dataclasses.fields(item))
        graph[assignment.name] = tuple(sorted(found))
    return graph


def recursive_types(node: ast.ModuleNode) -> dict[str, tuple[str, ...]]:
    """Every type on a cycle of the containment graph, with its strongly connected component
    (Tarjan's algorithm, iterative so a long chain of definitions cannot exhaust the stack).
    A type is recursive when its component has more than one member or it names itself."""
    graph = containment_graph(node)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    out: dict[str, tuple[str, ...]] = {}
    counter = 0
    for root in sorted(graph):
        if root in index:
            continue
        work = [(root, iter(graph[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            name, edges = work[-1]
            advanced = False
            for nxt in edges:
                if nxt not in graph:
                    continue
                if nxt not in index:
                    index[nxt] = low[nxt] = counter
                    counter += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(graph[nxt])))
                    advanced = True
                    break
                if nxt in on_stack:
                    low[name] = min(low[name], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[name])
            if low[name] == index[name]:
                members = []
                while True:
                    top = stack.pop()
                    on_stack.discard(top)
                    members.append(top)
                    if top == name:
                        break
                component = tuple(sorted(members))
                if len(component) > 1 or name in graph[name]:
                    for member in component:
                        out[member] = component
    return dict(sorted(out.items()))


def _needs_explicit_tag(built: Asn1Type) -> bool:
    """X.680 §31.2.7: a tag over a CHOICE or an OPEN TYPE is always EXPLICIT.

    Both have no tag of their own -- a CHOICE shows the chosen alternative's tag (§29.1),
    an open type the contained value's -- so an implicit tag would have nothing to
    replace and would erase the only discriminator on the wire.
    """
    for _ in range(MAX_RECURSION):
        if built.tags:  # a tagged CHOICE has a tag an IMPLICIT one can replace
            return False
        if not isinstance(built, Reference):
            break
        built = built.resolved()
    return isinstance(built, (Choice, OpenType))


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
    return {ast.SequenceType: "SEQUENCE", ast.SetType: "SET", ast.ChoiceType: "CHOICE"}.get(
        type(node), "TYPE"
    )


def lower(node: ast.ModuleNode, imports: dict[str, Module] | None = None) -> LoweredModule:
    return Lowerer(node, imports).run()


#: The governor of a SIZE bound: a length is an INTEGER value (X.680 §51.5).
_INTEGER = Primitive(Universal.INTEGER, "INTEGER")

#: X.683 §8.3-§8.4: what a dummy reference stands for.
_TYPE, _VALUE, _VALUE_SET, _CLASS, _OBJECT, _OBJECT_SET = (
    "type",
    "value",
    "value-set",
    "class",
    "object",
    "object-set",
)

#: How deep instantiations may nest before the definition is called non-terminating. A
#: template whose recursive reference grows its own actuals (`T {X} ::= ... T {SEQUENCE OF
#: X}`) denotes an infinite family; it is refused rather than allowed to exhaust the stack.
_MAX_INSTANCE_DEPTH = 64

_VALUE_NODES = (
    ast.IntValue,
    ast.StrValue,
    ast.BoolValue,
    ast.NullValue,
    ast.BitsValue,
    ast.OidValue,
    ast.BracedValue,
    ast.RefValue,
)


@dataclass(frozen=True)
class _Binding:
    """One dummy reference bound by an instantiation (X.683 §9.7).

    `actual` is the replacement in AST form: a type node for a type, a `PreLowered` value for
    a value (lowered against the governor where the instantiation is written), a constrained
    governor for a value set, a module-level or synthetic NAME for a class, object or object
    set. `element` is a value set's constraint element. `key` is the structural identity the
    instance is memoised under."""

    kind: str
    actual: object
    element: object = None
    key: str = ""

    def as_type(self, dummy: str):
        if self.kind in (_TYPE, _VALUE_SET):
            return self.actual
        raise Asn1SemanticError(f"dummy {dummy} stands for a {self.kind}, not a type (X.683 9.6)")

    def as_value(self, dummy: str):
        if self.kind == _VALUE:
            return self.actual
        raise Asn1SemanticError(f"dummy {dummy} stands for a {self.kind}, not a value (X.683 9.6)")

    def as_name(self, dummy: str, kinds: tuple[str, ...]) -> str:
        if self.kind in kinds and isinstance(self.actual, str):
            return self.actual
        raise Asn1SemanticError(
            f"dummy {dummy} stands for a {self.kind}, where {' or '.join(kinds)} is required "
            f"(X.683 9.6)"
        )

    def as_actual(self, dummy: str):
        """The binding passed on as an actual parameter of a nested reference."""
        if self.kind == _VALUE_SET:
            return ast.PreLowered(_VALUE_SET, self.element, key=self.key)
        return self.actual


def _key(node) -> str:
    """A structural identity for memoising an instance: equal for actuals that mean the same
    thing, built from content -- never from `repr`, which is not a content address."""
    if node is None or isinstance(node, (bool, int, float, str)):
        return json.dumps(node)
    if isinstance(node, (bytes, bytearray)):
        return "x" + bytes(node).hex()
    if isinstance(node, ast.PreLowered):
        return f"pre({node.kind}:{node.key})"
    if isinstance(node, enum.Enum):
        return f"{type(node).__name__}.{node.name}"
    if isinstance(node, (tuple, list)):
        return "[" + ",".join(_key(item) for item in node) + "]"
    if isinstance(node, (set, frozenset)):
        return "{" + ",".join(sorted(_key(item) for item in node)) + "}"
    if isinstance(node, dict):
        items = sorted((_key(k), _key(v)) for k, v in node.items())
        return "{" + ",".join(f"{k}:{v}" for k, v in items) + "}"
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        inner = ",".join(
            f"{f.name}={_key(getattr(node, f.name))}" for f in dataclasses.fields(node) if f.compare
        )
        return f"{type(node).__name__}({inner})"
    state = getattr(node, "__dict__", None)
    if state is not None:
        return f"{type(node).__name__}{_key(dict(state))}"
    slots = [s for klass in type(node).__mro__ for s in getattr(klass, "__slots__", ())]
    if slots:
        return (
            f"{type(node).__name__}{_key({s: getattr(node, s) for s in slots if hasattr(node, s)})}"
        )
    raise Asn1SemanticError(f"an actual of kind {type(node).__name__} has no structural identity")


def _digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _used_as_class(body, dummy: str) -> bool:
    """Whether a template body uses `dummy` as an information object class (`dummy.&f`)."""
    if isinstance(body, ast.OpenTypeNode):
        return body.object_class == dummy or (
            body.table is not None and _used_as_class(body.table, dummy)
        )
    if isinstance(body, (tuple, list)):
        return any(_used_as_class(item, dummy) for item in body)
    if dataclasses.is_dataclass(body) and not isinstance(body, type):
        return any(_used_as_class(getattr(body, f.name), dummy) for f in dataclasses.fields(body))
    return False


def _substitute(node, bindings: dict):
    """Replace every dummy reference in `node` by what its binding denotes (X.683 §9.7).

    The walk is structural over the AST, which §9.8's NOTE asks for: the actual is re-lowered
    in its own right rather than having its spelling pasted in. It is also total over the
    places a dummy can stand -- a type, a value, a value set, a class, an object or object set
    -- including the actual parameters of a nested reference and the elements of an object
    set, so an instance is lowered with no dummy left in it and needs no help from the
    module's tables. Strings are names only in the positions handled here; a string field
    anywhere else is data and is never rewritten."""
    if isinstance(node, (str, ast.PreLowered)) or node is None:
        return node
    if isinstance(node, ast.TypeRef):
        binding = bindings.get(node.name) if node.module is None else None
        return node if binding is None else binding.as_type(node.name)
    if isinstance(node, ast.RefValue):
        binding = bindings.get(node.name)
        return node if binding is None else binding.as_value(node.name)
    if isinstance(node, ast.ParameterizedRef):
        return replace(node, actuals=tuple(_substitute_actual(a, bindings) for a in node.actuals))
    if isinstance(node, ast.TableConstraintNode):
        object_set = node.object_set
        if object_set in bindings:
            object_set = bindings[object_set].as_name(object_set, (_OBJECT_SET,))
        spec = node.spec
        if spec is not None:
            spec = replace(
                spec, elements=tuple(_substitute_element(e, bindings) for e in spec.elements)
            )
        return replace(node, object_set=object_set, spec=spec)
    if isinstance(node, ast.OpenTypeNode):
        object_class = node.object_class
        if object_class is not None and object_class in bindings:
            object_class = bindings[object_class].as_name(object_class, (_CLASS,))
        table = _substitute(node.table, bindings)
        return replace(node, object_class=object_class, table=table)
    if isinstance(node, ast.Constrained):
        return replace(
            node,
            inner=_substitute(node.inner, bindings),
            constraints=tuple(_substitute_constraint(c, bindings) for c in node.constraints),
        )
    if isinstance(node, ast.ObjectSetAssignment):
        return replace(
            node, elements=tuple(_substitute_element(e, bindings) for e in node.elements)
        )
    if isinstance(node, ast.ObjectAssignment):
        return replace(node, settings=tuple(_substitute(f, bindings) for f in node.settings))
    if isinstance(node, ast.FieldSetting):
        return replace(node, value=_substitute(node.value, bindings))
    if isinstance(node, (tuple, list)):
        return type(node)(_substitute(item, bindings) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        changes = {}
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            if isinstance(value, (tuple, list)) or (
                dataclasses.is_dataclass(value) and not isinstance(value, type)
            ):
                new = _substitute(value, bindings)
                if new is not value:
                    changes[f.name] = new
        return replace(node, **changes) if changes else node
    return node


def _substitute_actual(actual, bindings: dict):
    """An actual parameter of a NESTED reference: a dummy name is replaced by what it is bound
    to, so the nested instance's memo key names its meaning, not the enclosing spelling."""
    if isinstance(actual, str):
        binding = bindings.get(actual)
        return actual if binding is None else binding.as_actual(actual)
    if isinstance(actual, ast.BracedActual):
        return _substitute_braced(actual, bindings)
    return _substitute(actual, bindings)


def _substitute_element(element, bindings: dict):
    """An element of an object set: a reference, a parameterized reference, an inline object."""
    if isinstance(element, str):
        binding = bindings.get(element)
        return element if binding is None else binding.as_name(element, (_OBJECT, _OBJECT_SET))
    if isinstance(element, tuple):  # an inline object: its settings may name dummies
        return tuple(_substitute(setting, bindings) for setting in element)
    return _substitute(element, bindings)


def _substitute_constraint(constraint, bindings: dict):
    """A subtype constraint: a value dummy (`SIZE (1..ub)`) becomes the value it is bound to,
    and a value-set dummy used as a contained subtype becomes that set's element."""
    if isinstance(constraint, (ast.ContentsConstraintNode, ast.UserDefinedConstraintNode)):
        return _substitute(constraint, bindings)
    if not isinstance(constraint, _constraints.Constraint):
        return constraint

    def value_of(reference, context):
        binding = bindings.get(reference.name)
        if binding is None:
            return reference  # a module value: resolved when the constraint is attached
        if binding.kind != _VALUE:
            raise Asn1SemanticError(
                f"dummy {reference.name} stands for a {binding.kind}, but the constraint "
                f"uses it as a value (X.683 9.6)"
            )
        return binding.actual.payload

    def subtype_of(reference, context):
        binding = bindings.get(reference.name)
        if binding is None:
            return reference
        if binding.kind == _VALUE_SET:
            return binding.element
        if binding.kind == _TYPE:
            return (
                reference
                if not isinstance(binding.actual, ast.TypeRef)
                else _constraints.TypeReference(binding.actual.name)
            )
        raise Asn1SemanticError(
            f"dummy {reference.name} stands for a {binding.kind}, but the constraint uses it "
            f"as a contained subtype (X.683 9.6)"
        )

    return _constraints.resolve_references(constraint, value_of, subtype_of)


def _substitute_braced(actual: ast.BracedActual, bindings: dict) -> ast.BracedActual:
    """A braced actual inside a template body is kept as tokens, so a dummy it mentions is
    replaced token by token -- the whole token, never a substring -- by the NAME or literal
    value it is bound to. A dummy bound to a type or a structured value has no token
    spelling, and is refused rather than spliced as text."""
    from .lexer import Token, tokenize
    from .parser import render_tokens

    tokens = tokenize(actual.raw, "<actual>")
    if not any(t.text in bindings for t in tokens if t.kind in ("identifier", "typereference")):
        return actual
    out = []
    for tok in tokens:
        binding = bindings.get(tok.text) if tok.kind in ("identifier", "typereference") else None
        if binding is None:
            out.append(tok)
            continue
        if binding.kind in (_CLASS, _OBJECT, _OBJECT_SET) and isinstance(binding.actual, str):
            out.append(Token("typereference", binding.actual, tok.line, tok.column))
            continue
        value = binding.actual.payload if binding.kind == _VALUE else None
        if isinstance(value, bool):
            out.append(Token("reserved", "TRUE" if value else "FALSE", tok.line, tok.column))
        elif isinstance(value, int):
            out.append(Token("number", str(value), tok.line, tok.column))
        elif isinstance(value, str):
            out.append(Token("cstring", value, tok.line, tok.column))
        else:
            raise Asn1SemanticError(
                f"the braced actual {actual.raw!r} names dummy {tok.text}, bound to a "
                f"{binding.kind} that has no single-token spelling; name it instead (X.683 9.7)"
            )
    return ast.BracedActual(render_tokens(out))


def compile_module(
    text: str, source: str = "<asn1>", imports: dict[str, Module] | None = None
) -> LoweredModule:
    """Parse and lower one ASN.1 module in a single call."""
    from .parser import parse_module

    return lower(parse_module(text, source), imports)


__all__ = [
    "Asn1SemanticError",
    "Asn1SyntaxError",
    "LoweredModule",
    "Lowerer",
    "UNIVERSAL_OF",
    "compile_module",
    "containment_graph",
    "lower",
    "recursive_types",
]
