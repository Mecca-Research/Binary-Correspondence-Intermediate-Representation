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

from dataclasses import dataclass, field, replace

from bcir.asn1.schema import (Asn1Type, Choice, Component, Module, ObjectSetTable,
                              OpenType, Primitive, Sequence, SequenceOf, Set, SetOf)
from bcir.asn1.tags import Asn1Error, Tag, TagClass, Universal

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
    #: type name -> (TagClass, number, mode) for a type ASSIGNED a tag
    #: (`Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {...}`). Components that reference the
    #: name already carry the tag; this is the record for an outermost direct encode, which
    #: the component-shaped type model cannot express.
    assigned_tags: dict[str, tuple] = field(default_factory=dict)

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
        self.assigned_tags: dict[str, tuple] = {}
        self.objects = {a.name: a for a in node.assignments
                        if isinstance(a, ast.ObjectAssignment)}
        self.object_sets = {a.name: a for a in node.assignments
                            if isinstance(a, ast.ObjectSetAssignment)}
        self._tables: dict[str, ObjectSetTable] = {}
        #: X.683 §8.2 parameterized assignments, by name. They are NOT lowered eagerly:
        #: §9.7 makes instantiation a substitution of actuals for dummy references, so
        #: there is nothing to build until a reference supplies them.
        self.parameterized = {a.name: a for a in node.assignments
                              if isinstance(a, ast.ParameterizedAssignment)}
        self._instantiations: dict[tuple, Asn1Type] = {}
        self._in_progress: set[str] = set()

    # --- entry point ------------------------------------------------------------------

    def run(self) -> LoweredModule:
        for name in self.assignments:
            self._type_by_name(name)
        oid = self._oid(self.node.oid) if self.node.oid else ()
        module = Module(self.node.name, oid, dict(self.types))
        return LoweredModule(module, self.node.tag_default, self.enumerations, self.node,
                            dict(self.assigned_tags))

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
            # A tagged type at ASSIGNMENT level (`Name ::= [APPLICATION 1] IMPLICIT
            # SEQUENCE {...}`). The encoder model carries tags on COMPONENTS, so the tag is
            # recorded against the assigned name and picked up by every component that
            # references it (see `_components`). That is where the tag is observable: it is
            # what X.690 puts on the wire for such a component and what X.691 §21 / X.696
            # §18.2 sort a SET on.
            #
            # The one place it is NOT reinstated is a direct encode of the assigned type as
            # an outermost value under a tag-visible rule, which would need a wrapper the
            # type model does not have. That is recorded rather than papered over:
            # `LoweredModule.assigned_tags` exposes it, and PER is unaffected either way
            # because X.691 §10.4.1/§10.6.3 make tagging invisible to it.
            inner, number, mode, cls = _peel_tag(node)
            built = self._type(inner, label)
            self.assigned_tags[label] = (cls, number, mode)
            return built

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
            return Choice(alts, label, ext)

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
        applied = [c for c in applied
                   if not isinstance(c, (ast.ContentsConstraintNode,
                                         ast.UserDefinedConstraintNode))]
        if contents:
            spec = contents[0]
            if not isinstance(built, Primitive) or built.universal not in (
                    Universal.OCTET_STRING, Universal.BIT_STRING):
                raise Asn1SemanticError(
                    f"{label}: a contents constraint applies only to OCTET STRING and to "
                    f"BIT STRING without a NamedBitList (X.682 11.3)")
            built = replace(
                built,
                contains=(self._type(spec.contained, f"{label} CONTAINING")
                          if spec.contained is not None else None),
                encoded_by=self._encoded_by(spec.encoded_by, label))
        if not applied:
            return built
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
                table = self._table_for(node, label)
                if field.is_type_field:
                    governing, columns = self._governing(node, declared, label)
                    return OpenType(f"{node.object_class}{node.field}", table=table,
                                    field=node.field, governing=governing,
                                    governing_fields=columns)
                if field.type is None:
                    raise Asn1SemanticError(
                        f"{label}: {node.object_class}{node.field} is a value field with "
                        f"no declared type, so it has no encoding")
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
            raise Asn1SemanticError(
                f"{label}: {node.object_class} has no field {node.field!r}")
        governed = f" DEFINED BY {node.governed_by}" if node.governed_by else ""
        return OpenType(f"ANY{governed}")

    def _instantiate(self, node: ast.ParameterizedRef, label: str) -> Asn1Type:
        """X.683 §9.7: build the type a parameterized reference denotes.

        The actual parameters replace the dummy references throughout the assignment's body
        and the result is lowered. §9.8's NOTE warns that this is "not exactly textual
        substitution" -- the ACTUAL parameter's tagging environment applies, not the dummy's
        -- which only differs when the actual crosses a module boundary with a different
        tag default. This front-end lowers one module at a time, so the two coincide; a
        cross-module instantiation with differing tag defaults is a known gap, not a claim.
        """
        target = self.parameterized.get(node.name)
        if target is None:
            raise Asn1SemanticError(
                f"{label}: {node.name!r} is referenced with actual parameters but is not a "
                f"parameterized assignment (X.683 9.2)")
        if len(node.actuals) != len(target.params):
            raise Asn1SemanticError(
                f"{label}: {node.name} takes {len(target.params)} parameter(s), "
                f"{len(node.actuals)} supplied (X.683 9.6)")
        key = (node.name, tuple(map(_actual_key, node.actuals)))
        if key in self._instantiations:
            return self._instantiations[key]
        bindings = dict(zip(target.params, node.actuals))
        body = target.body
        if not isinstance(body, ast.TypeAssignment):
            raise Asn1SemanticError(
                f"{label}: only a parameterized TYPE assignment can be referenced as a "
                f"type; {node.name} assigns a {type(body).__name__}")
        substituted = _substitute(body.type, bindings)
        # Object sets carried as actuals have to be visible to the table machinery under the
        # dummy's name for the duration, because a table constraint inside the body names
        # the DUMMY (`{Supported}`), not the actual.
        saved = {name: self.object_sets.get(name) for name in bindings}
        for dummy, actual in bindings.items():
            if isinstance(actual, str) and actual in self.object_sets:
                self.object_sets[dummy] = self.object_sets[actual]
        try:
            built = self._type(substituted, f"{label}[{node.name}]")
        finally:
            for name, previous in saved.items():
                if previous is None:
                    self.object_sets.pop(name, None)
                else:
                    self.object_sets[name] = previous
        self._instantiations[key] = built
        return built

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
            f"{label}: ENCODED BY takes an object identifier value (X.682 11.2)")

    def _table_for(self, node: ast.OpenTypeNode, label: str):
        """The associated table (X.681 §13) of the object set a table constraint names."""
        if node.table is None:
            return None
        return self._object_set_table(node.table.object_set, label)

    def _object_set_table(self, name: str, label: str):
        """Build one object set's associated table, resolving references and unions."""
        if name in self._tables:
            return self._tables[name]
        assignment = self.object_sets.get(name)
        if assignment is None:
            raise Asn1SemanticError(
                f"{label}: table constraint names object set {name!r}, which this module "
                f"does not define (X.682 10.4)")
        rows: list[dict] = []
        extensible = assignment.extensible
        self._tables[name] = ObjectSetTable(assignment.object_class, (), extensible)
        for element in assignment.elements:
            if isinstance(element, str):
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
                        f"neither a defined object nor a defined object set")
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
        declared = self.classes.get(class_name)
        by_name = {f.name: f for f in declared.fields} if declared else {}
        row: dict = {}
        for setting in settings:
            field = by_name.get(setting.name)
            is_type = (field.is_type_field if field is not None
                       else len(setting.name) > 1 and setting.name[1].isupper())
            if is_type:
                row[setting.name] = self._type(setting.value, f"{label}.{setting.name}")
            else:
                governor = (self._type(field.type, label)
                            if field is not None and field.type is not None else None)
                row[setting.name] = self.value(
                    setting.value, governor, f"{label}.{setting.name}")
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
        if node.named:
            self.enumerations[label] = {n.name: n.number for n in node.named}
        if universal == Universal.ENUMERATED:
            # Carry the enumeration ONTO the type, not just into the module-level side
            # table. The side table answers "what does this DEFAULT identifier mean"; PER
            # asks a different question -- X.691 §14.1 encodes the enumeration INDEX, so
            # the codec needs the whole root list at the point of use. BER/DER/OER never
            # needed it because they encode the value itself.
            return Primitive(
                int(universal), node.name,
                enumeration=tuple((n.name, n.number) for n in node.named),
                enum_extensible=bool(node.extensible))
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
            [e for e in entries if isinstance(e, ast.ComponentNode)])
        out: list[Component] = []
        position = -1
        for item in entries:
            position += 1
            if id(item) in groups:
                members, _ = self._components(
                    groups[id(item)].components, f"{label}.group{position}", choice)
                if choice:
                    # X.691 §23.8 NOTE: "Version brackets in the definition of choice
                    # extension additions have no effect on how ExtensionAdditionAlternatives
                    # are encoded." A bracket in a CHOICE is presentational only -- each
                    # member is its own alternative with its own index (§23.2), so grouping
                    # them would invent a nesting the encoding does not have.
                    for i, m in enumerate(members):
                        out.append(replace(
                            m, extension=True,
                            **({"tag": position + i, "tag_class": TagClass.CONTEXT}
                               if automatic else {})))
                    position += len(members) - 1
                    continue
                if automatic:
                    # §12.3 numbers a group's members in the enclosing list's sequence, so
                    # the group consumes as many tag numbers as it holds.
                    members = tuple(
                        replace(m, tag=position + i, tag_class=TagClass.CONTEXT)
                        for i, m in enumerate(members))
                    position += len(members) - 1
                out.append(Component(
                    name=f"[[{position}]]", type=Sequence(members, f"{label}.group"),
                    extension=True, optional=True, group=members))
                continue
            inner, tag, mode, tag_cls = _peel_tag(item.type)
            if tag is None and automatic:
                tag, mode, tag_cls = position, None, TagClass.CONTEXT   # §12.3
            built = self._type(inner, f"{label}.{item.name}")
            if tag is None and isinstance(inner, ast.TypeRef):
                # The component is untagged, but the type it names may have been ASSIGNED
                # a tag (`Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {...}`). X.680 §31
                # makes that tag part of the referenced type, so the component carries it
                # -- and it is what X.691 §21 / X.696 §18.2 sort a SET on, which is why
                # dropping it would silently reorder a SET's components.
                assigned = self.assigned_tags.get(inner.name)
                if assigned is not None:
                    tag_cls, tag, mode = assigned
            explicit = self._explicit(mode, built)
            default, has_default = self._default(item, built, f"{label}.{item.name}")
            out.append(Component(
                name=item.name, type=built, tag=tag, explicit=explicit,
                optional=item.optional, extension=extension_of[id(item)],
                **({"tag_class": tag_cls} if tag_cls is not None else {}),
                **({"default": default} if has_default else {})))
        return (self._bind_governing(tuple(out), entries, label),
                any(isinstance(n, ast.ExtensionMarker) for n in nodes))

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
                continue                                   # an extension-group marker
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


#: X.680 §8.1 tag classes, as spelled in the notation. An empty class name is the default
#: context-specific case (`[0]`), which is why it maps to CONTEXT rather than UNIVERSAL.
_TAG_CLASSES = {
    "": TagClass.CONTEXT,
    "UNIVERSAL": TagClass.UNIVERSAL,
    "APPLICATION": TagClass.APPLICATION,
    "PRIVATE": TagClass.PRIVATE,
}


def _peel_tag(node):
    """Split `[class n] IMPLICIT Type` into (Type, n, mode, class); 4x None if untagged."""
    if isinstance(node, ast.Tagged):
        cls = _TAG_CLASSES.get(node.tag_class)
        if cls is None:
            raise Asn1SemanticError(f"unknown tag class {node.tag_class!r}")
        return node.inner, node.number, node.mode, cls
    return node, None, None, None


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


def _actual_key(actual) -> str:
    """A stable identity for one actual parameter, so instantiations can be memoised."""
    return actual if isinstance(actual, str) else repr(actual)


def _substitute(node, bindings: dict):
    """Replace every dummy reference in `node` with its actual parameter (X.683 §9.7).

    The walk is structural over the AST dataclasses rather than textual, which is what
    §9.8's NOTE asks for: a `TypeRef` naming a dummy becomes the actual's NODE, so the
    actual is re-lowered in its own right instead of having its spelling pasted in.
    """
    import dataclasses

    if isinstance(node, ast.TypeRef) and node.module is None and node.name in bindings:
        actual = bindings[node.name]
        return ast.TypeRef(actual) if isinstance(actual, str) else actual
    if isinstance(node, ast.ParameterizedRef):
        return dataclasses.replace(
            node, actuals=tuple(_substitute(a, bindings) for a in node.actuals))
    if isinstance(node, ast.TableConstraintNode):
        # `{Supported}` inside the body names a DUMMY object set; rewriting it to the
        # actual's name is what lets the table constraint resolve after instantiation.
        actual = bindings.get(node.object_set)
        if isinstance(actual, str):
            return dataclasses.replace(node, object_set=actual)
        return node
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        changes = {}
        for f in dataclasses.fields(node):
            value = getattr(node, f.name)
            if isinstance(value, tuple):
                changes[f.name] = tuple(_substitute(v, bindings) for v in value)
            elif dataclasses.is_dataclass(value) and not isinstance(value, type):
                changes[f.name] = _substitute(value, bindings)
        return dataclasses.replace(node, **changes) if changes else node
    return node


def compile_module(text: str, source: str = "<asn1>",
                   imports: dict[str, Module] | None = None) -> LoweredModule:
    """Parse and lower one ASN.1 module in a single call."""
    from .parser import parse_module
    return lower(parse_module(text, source), imports)


__all__ = ["Asn1SemanticError", "Asn1SyntaxError", "LoweredModule", "Lowerer",
           "UNIVERSAL_OF", "compile_module", "lower"]
