"""A minimal ASN.1 type model — enough to own the rules a schema-free walk cannot.

`der.py` checks every clause 10/11 rule that is visible in the octets alone. Two are
not, because they are stated in terms of the *type definition*:

* **§8.9.3** — a component referenced with OPTIONAL or DEFAULT may be absent, and if
  present must appear at the position the definition gives it. Without the definition
  a decoder cannot tell an omitted optional component from a misplaced one.
* **§11.5** — a DER encoder shall not emit a component whose value equals its DEFAULT.
  Without the definition there is no DEFAULT to compare against.

So this module carries just enough of X.680: a component list with tags, optionality
and defaults, and the SEQUENCE / SEQUENCE OF / CHOICE constructors the BCIR ABI
projection needs. It is not a general ASN.1 compiler and does not try to be — X.680's
information object classes (X.681), constraints (X.682) and parameterization (X.683)
are out of scope, and a type this module cannot express is a type the projection does
not use.

Tagging follows an IMPLICIT TAGS environment (X.690 §8.14.4): a context tag replaces
the base type's tag and the encoding stays primitive unless the base is constructed.
`explicit=True` on a component selects §8.14.3 instead, wrapping the base encoding in
a constructed tag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .codec import Strictness, from_tlv, to_tlv
from .tags import Asn1Error, Tag, TagClass, Universal
from .tlv import Tlv, decode_one, encode_tlv

#: Sentinel distinguishing "no DEFAULT" from "DEFAULT is None".
_NO_DEFAULT = object()


@dataclass(frozen=True)
class Component:
    """One component of a SEQUENCE, with its X.680 optionality."""

    name: str
    type: "Asn1Type"
    #: Context-specific tag number, or None to keep the base type's tag.
    tag: int | None = None
    #: §8.14.3 (explicit) versus §8.14.4 (implicit) tagging for this component.
    explicit: bool = False
    optional: bool = False
    default: object = _NO_DEFAULT

    @property
    def has_default(self) -> bool:
        return self.default is not _NO_DEFAULT

    def tagged(self, base: Tag) -> Tag:
        if self.tag is None:
            return base
        # §8.14.4 a): implicit tagging keeps the base's constructed bit; §8.14.3:
        # explicit tagging is always constructed, since it nests a full encoding.
        return Tag(TagClass.CONTEXT, self.tag, True if self.explicit else base.constructed)


class Asn1Type:
    """Base of the type model. Subclasses encode/decode a Python value."""

    name: str = "?"

    def encode(self, value) -> Tlv:                      # pragma: no cover - abstract
        raise NotImplementedError

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> object:  # pragma: no cover
        raise NotImplementedError

    def base_tag(self) -> Tag:                            # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class Primitive(Asn1Type):
    """A universal type named by the schema, encoded *tag-first*.

    The value mapping in `codec` is value-directed: a Python `int` becomes INTEGER, a
    `str` becomes UTF8String. A schema knows better — it may declare ENUMERATED or
    PrintableString for the same Python value — so the declared tag wins here and the
    mapping is only consulted for types where it is unambiguous.

    Two equivalences make this sound rather than a loose retag:
      * §8.4 — "the encoding of an enumerated value shall be that of the integer value
        with which it is associated", so ENUMERATED and INTEGER share contents octets;
      * §8.23.3 — every restricted character string is encoded "as if it had been
        declared [UNIVERSAL x] IMPLICIT OCTET STRING", so a `str` can be re-encoded
        into whichever string type the schema names, with that type's repertoire
        enforced by `encode_string`.
    Anything else must map exactly, and a mismatch is an error rather than a coercion.
    """

    universal: int
    name: str = "PRIMITIVE"

    def base_tag(self) -> Tag:
        constructed = self.universal in (Universal.SEQUENCE, Universal.SET)
        return Tag(TagClass.UNIVERSAL, self.universal, constructed)

    def encode(self, value) -> Tlv:
        from .values import encode_string

        if self.universal in _STRING_UNIVERSALS:
            if not isinstance(value, str):
                raise Asn1Error(
                    f"{self.name}: expected str, got {type(value).__name__}")
            return Tlv(self.base_tag(), encode_string(self.universal, value))

        tlv = to_tlv(value)
        if tlv.tag.is_universal and tlv.tag.number == self.universal:
            return tlv
        # §8.4: ENUMERATED borrows the integer contents octets.
        if (self.universal == Universal.ENUMERATED
                and tlv.tag.is_universal and tlv.tag.number == Universal.INTEGER):
            return Tlv(self.base_tag(), tlv.content)
        raise Asn1Error(
            f"{self.name}: value maps to {tlv.tag}, not "
            f"{Tag(TagClass.UNIVERSAL, self.universal)}")

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> object:
        return from_tlv(tlv, strictness=strictness)


#: The restricted character string types of X.680 §41, which §8.23.3 unifies as
#: implicitly-tagged octet strings and which therefore all accept a Python `str`.
_STRING_UNIVERSALS = frozenset({
    Universal.UTF8_STRING, Universal.NUMERIC_STRING, Universal.PRINTABLE_STRING,
    Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING, Universal.IA5_STRING,
    Universal.GRAPHIC_STRING, Universal.VISIBLE_STRING, Universal.GENERAL_STRING,
    Universal.UNIVERSAL_STRING, Universal.BMP_STRING, Universal.OBJECT_DESCRIPTOR,
})


@dataclass
class SequenceOf(Asn1Type):
    """SEQUENCE OF (X.690 §8.10): order is significant and is preserved."""

    element: Asn1Type
    name: str = "SEQUENCE OF"

    def base_tag(self) -> Tag:
        return Tag(TagClass.UNIVERSAL, Universal.SEQUENCE, True)

    def encode(self, value) -> Tlv:
        return Tlv(self.base_tag(), b"", [self.element.encode(v) for v in value])

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> list:
        if not tlv.constructed:
            raise Asn1Error(
                f"{self.name} must be constructed (X.690 8.10.1)", tlv.offset)
        return [self.element.decode(c, strictness=strictness) for c in tlv.children]


@dataclass
class Sequence(Asn1Type):
    """SEQUENCE (X.690 §8.9): components in definition order, OPTIONAL/DEFAULT aware.

    Values are dicts keyed by component name — the projection is a data mapping, not a
    code generator, so there is no generated class to keep in sync with the module.
    """

    components: tuple[Component, ...]
    name: str = "SEQUENCE"

    def base_tag(self) -> Tag:
        return Tag(TagClass.UNIVERSAL, Universal.SEQUENCE, True)

    def encode(self, value: dict) -> Tlv:
        unknown = set(value) - {c.name for c in self.components}
        if unknown:
            raise Asn1Error(f"{self.name}: unknown component(s) {sorted(unknown)}")
        children: list[Tlv] = []
        for comp in self.components:
            if comp.name not in value:
                if comp.optional or comp.has_default:
                    continue
                raise Asn1Error(f"{self.name}: component {comp.name!r} is mandatory")
            item = value[comp.name]
            # §11.5: DER shall not encode a component equal to its DEFAULT.
            if comp.has_default and item == comp.default:
                continue
            children.append(_apply_tag(comp, comp.type.encode(item)))
        return Tlv(self.base_tag(), b"", children)

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> dict:
        if not tlv.constructed:
            raise Asn1Error(
                f"{self.name} must be constructed (X.690 8.9.1)", tlv.offset)
        out: dict[str, object] = {}
        children = list(tlv.children)
        index = 0
        for comp in self.components:
            expected = comp.tagged(comp.type.base_tag())
            if index < len(children) and _matches(children[index], expected):
                out[comp.name] = comp.type.decode(
                    _strip_tag(comp, children[index], expected),
                    strictness=strictness)
                index += 1
            elif comp.has_default:
                out[comp.name] = comp.default          # §11.5: absent means DEFAULT
            elif comp.optional:
                continue
            else:
                raise Asn1Error(
                    f"{self.name}: mandatory component {comp.name!r} ({expected}) is "
                    f"missing or out of order (X.690 8.9.2)", tlv.offset)
        if index != len(children):
            raise Asn1Error(
                f"{self.name}: {len(children) - index} unexpected trailing "
                f"component(s)", tlv.offset)
        return out


def _matches(tlv: Tlv, expected: Tag) -> bool:
    return tlv.tag.cls is expected.cls and tlv.tag.number == expected.number


def _apply_tag(comp: Component, base: Tlv) -> Tlv:
    if comp.tag is None:
        return base
    if comp.explicit:                                     # §8.14.3
        return Tlv(Tag(TagClass.CONTEXT, comp.tag, True), b"", [base])
    return Tlv(Tag(TagClass.CONTEXT, comp.tag, base.tag.constructed),  # §8.14.4
               base.content, base.children, offset=base.offset)


def _strip_tag(comp: Component, tlv: Tlv, expected: Tag) -> Tlv:
    if comp.tag is None:
        return tlv
    if comp.explicit:                                     # §8.14.3: unwrap the nesting
        if len(tlv.children) != 1:
            raise Asn1Error(
                f"explicitly tagged [{comp.tag}] must wrap exactly one encoding "
                f"(X.690 8.14.3)", tlv.offset)
        return tlv.children[0]
    base = comp.type.base_tag()                           # §8.14.4: restore the base tag
    return Tlv(Tag(base.cls, base.number, tlv.tag.constructed), tlv.content,
               tlv.children, offset=tlv.offset)


@dataclass
class Module:
    """A named collection of types, with the OID that identifies it on the wire."""

    name: str
    oid: tuple[int, ...]
    types: dict[str, Asn1Type] = field(default_factory=dict)

    def encode(self, type_name: str, value) -> bytes:
        return encode_tlv(self.types[type_name].encode(value))

    def decode(self, type_name: str, data: bytes, *,
               strictness: Strictness = Strictness.DER):
        tlv = decode_one(data)
        if strictness is Strictness.DER:
            from .der import require_der
            require_der(tlv)
        return self.types[type_name].decode(tlv, strictness=strictness)


__all__ = ["Asn1Type", "Component", "Module", "Primitive", "Sequence", "SequenceOf"]
