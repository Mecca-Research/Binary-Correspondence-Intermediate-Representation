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
    """`...` in a component list (§25.1): the extension root ends here."""


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
    "BitsValue", "BoolValue", "BracedValue", "Builtin", "ChoiceType",
    "ComponentNode", "ExtensionMarker", "IntValue", "ModuleNode", "NamedNumber",
    "NullValue", "OidArc", "OidValue", "RefValue", "SequenceOfType", "SequenceType",
    "SetOfType", "SetType", "StrValue", "SymbolsFromModule", "Tagged", "TypeAssignment",
    "TypeRef", "ValueAssignment",
]
