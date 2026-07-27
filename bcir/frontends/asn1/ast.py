"""The parse tree for the X.680 subset the BCIR ASN.1 rail consumes.

This is deliberately a *syntactic* tree, not the encoder's type model: it keeps
unresolved type references as references, keeps the module's tag default un-applied,
and keeps DEFAULT values as written. Resolution and tag-default application happen in
`lower.py`, so a fault in either is attributable to one place and the printer can
reproduce the module as the author wrote it (the round-trip law in `printer.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NamedNumber:
    """X.680 §20.3 / §22.3: `identifier(number)` in ENUMERATED, INTEGER, BIT STRING."""

    name: str
    number: int


@dataclass(frozen=True)
class OidArc:
    """One arc of an OBJECT IDENTIFIER value: a number, a name, or `name(number)`."""

    name: str | None
    number: int | None


# --- values (X.680 clause 17) ---------------------------------------------------------

@dataclass(frozen=True)
class IntValue:
    value: int


@dataclass(frozen=True)
class StrValue:
    value: str


@dataclass(frozen=True)
class BoolValue:
    value: bool


@dataclass(frozen=True)
class NullValue:
    pass


@dataclass(frozen=True)
class BitsValue:
    """A bstring/hstring value; `bits` is the bit count, `data` the octets."""

    data: bytes
    bits: int


@dataclass(frozen=True)
class RefValue:
    """A bare identifier used as a value: an ENUMERATED item, or a valuereference."""

    name: str


@dataclass(frozen=True)
class OidValue:
    arcs: tuple[OidArc, ...]


@dataclass(frozen=True)
class BracedValue:
    """`{ ... }` — a SEQUENCE OF / SET OF value, or an OBJECT IDENTIFIER value.

    X.680 does not tell these apart syntactically. `{ 1 }` is a one-element
    SEQUENCE OF INTEGER value AND a one-arc OBJECT IDENTIFIER value; `{}` is an empty
    value of either. Only the governing type decides, so the parser records every
    reading the text admits and `lower.value` picks using the component's type.

    Guessing from shape instead is not a small inaccuracy: reading
    `strides [2] SEQUENCE OF INTEGER DEFAULT { 1 }` as an OID makes the DEFAULT
    incomparable to any encoded value, so X.690 §11.5 never omits it and the module
    silently produces different octets than the same schema written by hand.
    """

    #: The value-list reading, or None when the text cannot be one.
    items: tuple[object, ...] | None = ()
    #: The object-identifier reading, or None when the text cannot be one.
    arcs: tuple[OidArc, ...] | None = ()


# --- types (X.680 clauses 16-31) ------------------------------------------------------

@dataclass(frozen=True)
class TypeRef:
    """A reference to a type assigned elsewhere, possibly in another module."""

    name: str
    module: str | None = None


@dataclass(frozen=True)
class Builtin:
    """A built-in type named by its X.680 keyword, e.g. INTEGER or UTF8String."""

    name: str
    #: NamedNumbers for INTEGER (§19.1), ENUMERATED (§20.1), BIT STRING (§22.1).
    named: tuple[NamedNumber, ...] = ()
    #: True when an ENUMERATED carries an extension marker `...` (§20.1).
    extensible: bool = False


@dataclass(frozen=True)
class Tagged:
    """A tagged type (§31): `[class number] IMPLICIT/EXPLICIT Type`.

    `mode` is None when the module's TagDefault applies -- keeping it unresolved is what
    lets the printer reproduce the source and lets `lower.py` own the §31.2.7 rule.
    """

    tag_class: str            # "" (context) | UNIVERSAL | APPLICATION | PRIVATE
    number: int
    inner: object
    mode: str | None = None   # None | IMPLICIT | EXPLICIT


@dataclass(frozen=True)
class Constrained:
    """A type with one or more X.680 clause 51 subtype constraints applied.

    Kept as a WRAPPER rather than a field on every type node so that a serial application
    of constraints (§49.9) round-trips in the order written — and so that the printer can
    reproduce `INTEGER (0..255) (0..100)` rather than silently folding it to one.
    """

    inner: object
    constraints: tuple[object, ...] = ()


@dataclass(frozen=True)
class OpenTypeNode:
    """An open type — X.681 §14, reached two ways.

    `ANY` / `ANY DEFINED BY x` is the withdrawn X.680:1988 spelling that RFC 5280's
    1988 module still uses; `CLASS.&Field` where the field is a TYPE field is the modern
    one. Both mean the same thing: the schema does not fix this component's type.

    `governed_by` records the sibling component (`DEFINED BY x`) or the object set
    (`({Algorithms}{@algorithm})`) that a peer uses to work out the contained type. It is
    kept because it is real information a caller can act on, and dropped from the
    encoding because X.690 encodes the contained value the same way regardless.
    """

    governed_by: str | None = None
    #: The information object class the field came from, when reached via `CLASS.&Field`.
    object_class: str | None = None
    field: str | None = None
    #: The X.682 §10 table constraint applied to this ObjectClassFieldType, when one was
    #: written. It is what turns an opaque open type into a RESOLVABLE one: the object set
    #: names the candidate rows and the AtNotation names the sibling that selects among them.
    table: object | None = None


@dataclass(frozen=True)
class ClassField:
    """One field of an information object class (X.681 §9)."""

    name: str                      # including the leading `&`
    #: A TYPE field (`&Type`, capitalised) versus a VALUE field (`&id`).
    is_type_field: bool
    type: object | None = None     # the declared type of a value field
    optional: bool = False
    unique: bool = False


@dataclass(frozen=True)
class ClassAssignment:
    """`NAME ::= CLASS { ... } WITH SYNTAX { ... }` (X.681 §9.1)."""

    name: str
    fields: tuple[ClassField, ...] = ()
    #: The WITH SYNTAX list as its full token sequence, fields AND literal words, in order
    #: (X.681 §10). It changes only how an OBJECT is spelled, never an encoding -- but §11.4
    #: makes the spelling mandatory: with a WithSyntaxSpec an object MUST use DefinedSyntax.
    #: The literals have to be kept, not just the `&field`s: `{&Type IDENTIFIED BY &id}`
    #: puts two words between the settings, and an object body that is matched positionally
    #: against fields alone stops at the first literal it did not expect.
    with_syntax: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldSetting:
    """One `PrimitiveFieldName Setting` of an object (X.681 §11.5).

    `value` is a value node for a value field and a TYPE node for a type field -- which is
    the whole point of the machinery: a type field's setting IS a type, and that is what an
    open type governed by this object resolves to.
    """

    name: str                      # including the leading `&`
    value: object


@dataclass(frozen=True)
class ObjectAssignment:
    """`obj CLASS ::= { ... }` (X.681 §11.1).

    `raw` is the object's body as a normalized token stream, kept so the printer can
    reproduce the assignment and the round-trip law keeps covering these modules.
    `settings` is the interpreted form: dropping it is what left table constraints
    unresolvable, because an object with no readable field settings contributes no row to
    the associated table of §13.
    """

    name: str
    object_class: str
    raw: str = ""
    settings: tuple[FieldSetting, ...] = ()


@dataclass(frozen=True)
class ObjectSetAssignment:
    """`Set CLASS ::= { obj | obj, ... }` (X.681 §12.1).

    `elements` holds each member: either an inline `ObjectAssignment`-shaped body or the
    name of a defined object / object set to splice in (§12.5 inherits an extension marker
    through such a reference). `extensible` records the `...` of §12.3.
    """

    name: str
    object_class: str
    objects: tuple[str, ...] = ()
    raw: str = ""
    elements: tuple[object, ...] = ()
    extensible: bool = False


@dataclass(frozen=True)
class ParameterizedAssignment:
    """`Name {P1, P2} ::= <body>` — X.683 §8.2.

    The body is kept as an UNRESOLVED assignment node. §9.7 makes instantiation a
    substitution of actual parameters for dummy references, so there is nothing to lower
    until a reference supplies them: lowering the body eagerly would have to invent types
    for the dummies, and any type it invented would be wrong for some instantiation.

    `governors` records each parameter's optional `Governor ":"` prefix (§8.3). It is not
    consulted when substituting -- §9.6 says the ACTUAL parameter's form is what has to fit
    -- but dropping it would lose the module's own statement of intent.
    """

    name: str
    params: tuple[str, ...]
    body: object
    governors: tuple[object, ...] = ()


@dataclass(frozen=True)
class ParameterizedRef:
    """`Name {Actual1, Actual2}` — X.683 §9.2, a reference that supplies actuals."""

    name: str
    actuals: tuple[object, ...] = ()
    module: str | None = None


@dataclass(frozen=True)
class ContentsConstraintNode:
    """X.682 §11 `CONTAINING Type [ENCODED BY oid]` / `ENCODED BY oid`.

    Unlike a value-set constraint this one says what the contents octets ARE, so it cannot
    be discarded: §11.4 makes the octet/bit string's abstract value the ENCODING of another
    type. It is only valid on OCTET STRING and on BIT STRING without a NamedBitList (§11.3).
    """

    contained: object | None = None
    encoded_by: object | None = None


@dataclass(frozen=True)
class UserDefinedConstraintNode:
    """X.682 §9 `CONSTRAINED BY { ... }`.

    §9 NOTE 1 calls this "a special form of ASN.1 comment": it is explicitly not fully
    machine-processable, and X.691 §10.3.3 makes it not PER-visible. So it is RECORDED and
    never consulted by an encoder -- keeping it is what lets a module round-trip through the
    printer without the front-end having silently deleted part of the author's intent.
    """

    raw: str = ""


@dataclass(frozen=True)
class TableConstraintNode:
    """X.682 §10.3 `({ObjectSet})` / §10.7 `({ObjectSet}{@a,@.b})`.

    `at_notations` is empty for a SimpleTableConstraint. Each entry is the raw AtNotation
    text (`@errorCategory`, `@.errorCode`, `@...errorId`), kept verbatim because §10.10's
    level counting is positional and re-deriving it from a parsed form would lose the dots.
    """

    object_set: str
    at_notations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SequenceOfType:
    element: object
    #: The element's identifier when written `SEQUENCE OF name Type` (§25.1).
    element_name: str | None = None


@dataclass(frozen=True)
class SetOfType:
    element: object
    element_name: str | None = None


@dataclass(frozen=True)
class ComponentNode:
    name: str
    type: object
    optional: bool = False
    default: object | None = None
    has_default: bool = False


@dataclass(frozen=True)
class ExtensionMarker:
    """`...` in a component list (§25.1): the extension root ends here.

    A SECOND marker closes an "extension marker pair": X.691 §19.9 NOTE 2 makes components
    written after it part of the extension ROOT again, "encoded as if they were defined
    immediately before the extension marker pair".
    """


@dataclass(frozen=True)
class ExtensionGroup:
    """`[[ a, b ]]` — a version bracket (X.680 §25.1), one extension addition GROUP.

    A group is present or absent as a unit and, per X.691 §19.9, is encoded as a SEQUENCE
    of its members which is then wrapped as a single open type field -- so one bit in the
    addition bitmap covers the whole bracket, however many components it holds.
    """

    components: tuple[object, ...] = ()


@dataclass(frozen=True)
class SequenceType:
    components: tuple[object, ...] = ()


@dataclass(frozen=True)
class SetType:
    components: tuple[object, ...] = ()


@dataclass(frozen=True)
class ChoiceType:
    alternatives: tuple[object, ...] = ()


# --- module structure (X.680 clause 13) -----------------------------------------------

@dataclass(frozen=True)
class TypeAssignment:
    name: str
    type: object


@dataclass(frozen=True)
class ValueAssignment:
    name: str
    type: object
    value: object


@dataclass(frozen=True)
class SymbolsFromModule:
    symbols: tuple[str, ...]
    module: str
    oid: OidValue | None = None


@dataclass
class ModuleNode:
    name: str
    oid: OidValue | None = None
    #: EXPLICIT (§13.4 default), IMPLICIT, or AUTOMATIC.
    tag_default: str = "EXPLICIT"
    extensibility_implied: bool = False
    imports: tuple[SymbolsFromModule, ...] = ()
    exports: tuple[str, ...] | None = None          # None = EXPORTS ALL
    assignments: list[object] = field(default_factory=list)

    def type_assignments(self) -> dict[str, object]:
        return {a.name: a.type for a in self.assignments
                if isinstance(a, TypeAssignment)}

    def value_assignments(self) -> dict[str, ValueAssignment]:
        return {a.name: a for a in self.assignments
                if isinstance(a, ValueAssignment)}


__all__ = [
    "BitsValue", "BoolValue", "BracedValue", "Builtin", "ChoiceType", "ClassAssignment",
    "ClassField", "ComponentNode", "Constrained", "ExtensionMarker", "IntValue", "ModuleNode",
    "NamedNumber", "NullValue", "ObjectAssignment", "ObjectSetAssignment", "OidArc",
    "OidValue", "OpenTypeNode", "RefValue", "SequenceOfType", "SequenceType",
    "SetOfType", "SetType", "StrValue", "SymbolsFromModule", "Tagged", "TypeAssignment",
    "TypeRef", "ValueAssignment",
]
