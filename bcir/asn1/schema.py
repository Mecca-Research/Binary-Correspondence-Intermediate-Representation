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
    #: Tag number, or None to keep the base type's tag.
    tag: int | None = None
    #: The tag's class. X.680 §31 allows [APPLICATION n] and [PRIVATE n] as well as the
    #: bare [n] that means context-specific, and the class is not cosmetic: X.680 §8.6
    #: orders tags by class first, which is what fixes the component order of a SET in
    #: every canonical encoding (X.690 §11.6, X.696 §18.2).
    tag_class: TagClass = TagClass.CONTEXT
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
        return Tag(self.tag_class, self.tag, True if self.explicit else base.constructed)

    def outer_tag(self) -> Tag | None:
        """The single tag this component shows on the wire, or None if it has no one tag.

        None means an UNTAGGED CHOICE, which shows whichever alternative was chosen
        (X.680 §29.1). Note that an EXPLICIT tag is computed WITHOUT consulting the base
        type: §8.14.3 wraps the base encoding in a new constructed tag, so the base tag
        moves inside and stops being observable from outside. That distinction is what
        lets `[4] Name` work when `Name` is a CHOICE and has no base tag to ask for.
        """
        if self.tag is None:
            if isinstance(self.type, (Choice, OpenType)):
                return None
            return self.type.base_tag()
        if self.explicit:                                  # §8.14.3
            return Tag(self.tag_class, self.tag, True)
        return Tag(self.tag_class, self.tag,               # §8.14.4
                   self.type.base_tag().constructed)

    def expected_tags(self) -> tuple[Tag, ...]:
        """Every tag this component may present on the wire.

        Empty means "any tag": an untagged open type does not constrain what arrives.
        """
        tag = self.outer_tag()
        if tag is None:
            if isinstance(self.type, OpenType):
                return ()
            return self.type.alternative_tags()
        return (tag,)


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
    #: The X.680 clause 51 subtype constraint, when the module stated one. Invisible to
    #: BER/DER -- a constraint restricts the value set, and X.690 encodes a value the same
    #: way regardless -- but load-bearing for OER and PER, which CHOOSE the encoding from
    #: it (X.696 §8.2.7/§8.2.8).
    constraint: object | None = None

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
class OpenType(Asn1Type):
    """An OPEN TYPE — X.681 §14: a field whose type is not fixed by the schema.

    This is the construct that X.509 cannot be written without. `AlgorithmIdentifier`'s
    `parameters` component holds *whatever the algorithm identified by the sibling
    `algorithm` component says it holds*: NULL for RSA, an OID for most EC curves, a
    SEQUENCE for RSASSA-PSS. No fixed type can describe it.

    An open type value is carried here as the **complete encoding of the contained
    value** (X.690: an open type field is the contained type's encoding, unchanged).
    Keeping it as octets rather than a decoded object is the honest representation: this
    layer does not know the contained type, and inventing one would be worse than
    admitting it. A caller who *does* know -- because it read the governing field -- can
    decode the octets with the right type as a second step.

    Two consequences the rest of the model has to respect:

    * **No tag of its own.** Like a CHOICE (X.680 §29.1), an open type shows whichever
      tag the contained value has, so `base_tag` raises and matching is by position.
    * **IMPLICIT tagging is impossible on it.** An implicit tag replaces the base tag,
      and there is no base tag to replace (X.680 §31.2.7 names open types alongside
      CHOICE for exactly this reason).
    """

    name: str = "OPEN TYPE"

    def base_tag(self) -> Tag:
        raise Asn1Error(
            f"{self.name}: an open type has no tag of its own (X.681 14); it shows the "
            f"tag of whatever value it contains")

    def encode(self, value) -> Tlv:
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(
                f"{self.name}: an open type value is the contained value's complete "
                f"encoding, so it must be bytes, not {type(value).__name__}")
        return decode_one(bytes(value))

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> bytes:
        return encode_tlv(tlv)


@dataclass
class SequenceOf(Asn1Type):
    """SEQUENCE OF (X.690 §8.10): order is significant and is preserved."""

    element: Asn1Type
    name: str = "SEQUENCE OF"
    #: A SIZE constraint on the number of occurrences (X.680 §51.5).
    constraint: object | None = None

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
            if index < len(children) and _matches_any(children[index], comp):
                out[comp.name] = comp.type.decode(
                    _strip_tag(comp, children[index]), strictness=strictness)
                index += 1
            elif comp.has_default:
                out[comp.name] = comp.default          # §11.5: absent means DEFAULT
            elif comp.optional:
                continue
            else:
                shown = " | ".join(str(t) for t in comp.expected_tags())
                raise Asn1Error(
                    f"{self.name}: mandatory component {comp.name!r} ({shown}) is "
                    f"missing or out of order (X.690 8.9.2)", tlv.offset)
        if index != len(children):
            raise Asn1Error(
                f"{self.name}: {len(children) - index} unexpected trailing "
                f"component(s)", tlv.offset)
        return out


@dataclass
class SetOf(Asn1Type):
    """SET OF (X.690 §8.12): unordered as an abstract value.

    DER fixes the order the octets appear in -- §11.6 requires ascending order of the
    component encodings -- so the encoder SORTS rather than preserving input order.
    That is the whole point of a canonical encoding: two peers holding the same set
    must produce the same octets, and therefore the same digest.
    """

    element: Asn1Type
    name: str = "SET OF"
    #: A SIZE constraint on the number of occurrences (X.680 §51.5).
    constraint: object | None = None

    def base_tag(self) -> Tag:
        return Tag(TagClass.UNIVERSAL, Universal.SET, True)

    def encode(self, value) -> Tlv:
        children = [self.element.encode(v) for v in value]
        return Tlv(self.base_tag(), b"", _sorted_set_of(children))

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> list:
        if not tlv.constructed:
            raise Asn1Error(
                f"{self.name} must be constructed (X.690 8.12.1)", tlv.offset)
        return [self.element.decode(c, strictness=strictness) for c in tlv.children]


@dataclass
class Set(Asn1Type):
    """SET (X.690 §8.11): components identified by tag, not by position.

    Because the wire order carries no information, DER fixes it: the components are
    emitted in ascending tag order. Decoding accepts them in any order, which is what
    makes a SET a SET rather than a SEQUENCE with a different tag.
    """

    components: tuple[Component, ...]
    name: str = "SET"

    def base_tag(self) -> Tag:
        return Tag(TagClass.UNIVERSAL, Universal.SET, True)

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
            if comp.has_default and item == comp.default:  # §11.5
                continue
            children.append(_apply_tag(comp, comp.type.encode(item)))
        return Tlv(self.base_tag(), b"", _sorted_set_of(children))

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> dict:
        if not tlv.constructed:
            raise Asn1Error(
                f"{self.name} must be constructed (X.690 8.11.1)", tlv.offset)
        remaining = list(tlv.children)
        out: dict[str, object] = {}
        for comp in self.components:
            for position, child in enumerate(remaining):
                if _matches_any(child, comp):
                    out[comp.name] = comp.type.decode(
                        _strip_tag(comp, child), strictness=strictness)
                    del remaining[position]
                    break
            else:
                if comp.has_default:
                    out[comp.name] = comp.default
                elif not comp.optional:
                    raise Asn1Error(
                        f"{self.name}: mandatory component {comp.name!r} is missing "
                        f"(X.690 8.11.2)", tlv.offset)
        if remaining:
            raise Asn1Error(
                f"{self.name}: {len(remaining)} component(s) match no alternative",
                tlv.offset)
        return out


@dataclass
class Choice(Asn1Type):
    """CHOICE (X.680 §29): exactly one alternative, identified by its tag.

    A CHOICE has NO TAG OF ITS OWN (§29.1) -- the encoding is the chosen alternative's
    encoding, unchanged. Two consequences the model has to carry:

    * `base_tag` cannot answer, so it raises rather than inventing one. A CHOICE
      component is matched through `Component.expected_tags` instead.
    * X.680 §31.2.7 forbids an IMPLICIT tag on a CHOICE: an implicit tag REPLACES the
      base tag, and replacing the tag of an untagged type would erase the only thing
      that says which alternative was chosen. A context tag on a CHOICE must therefore
      be explicit, and `Component` is checked for that here.

    Values are `(alternative_name, value)` pairs -- a bare value would be ambiguous
    whenever two alternatives accept the same Python type.
    """

    alternatives: tuple[Component, ...]
    name: str = "CHOICE"

    def __post_init__(self) -> None:
        for alt in self.alternatives:
            if alt.tag is not None and not alt.explicit and isinstance(alt.type, Choice):
                raise Asn1Error(
                    f"{self.name}: alternative {alt.name!r} tags a CHOICE implicitly; "
                    f"X.680 31.2.7 requires EXPLICIT")
        tags = [t for alt in self.alternatives for t in alt.expected_tags()]
        duplicate = {t for t in tags if tags.count(t) > 1}
        if duplicate:
            raise Asn1Error(
                f"{self.name}: alternatives share tag {sorted(map(str, duplicate))}; "
                f"X.680 29.3 requires distinct tags")

    def base_tag(self) -> Tag:
        raise Asn1Error(
            f"{self.name}: a CHOICE has no tag of its own (X.680 29.1); tag the "
            f"component that references it, EXPLICITly (X.680 31.2.7)")

    def alternative_tags(self) -> tuple[Tag, ...]:
        return tuple(t for alt in self.alternatives for t in alt.expected_tags())

    def encode(self, value) -> Tlv:
        if not (isinstance(value, tuple) and len(value) == 2):
            raise Asn1Error(
                f"{self.name}: value must be an (alternative, value) pair, got "
                f"{type(value).__name__}")
        chosen, payload = value
        for alt in self.alternatives:
            if alt.name == chosen:
                return _apply_tag(alt, alt.type.encode(payload))
        raise Asn1Error(f"{self.name}: {chosen!r} is not an alternative")

    def decode(self, tlv: Tlv, *, strictness: Strictness) -> tuple:
        for alt in self.alternatives:
            if _matches_any(tlv, alt):
                return (alt.name,
                        alt.type.decode(_strip_tag(alt, tlv), strictness=strictness))
        raise Asn1Error(
            f"{self.name}: {tlv.tag} matches no alternative (X.680 29.1)", tlv.offset)


def _sorted_set_of(children: list[Tlv]) -> list[Tlv]:
    """X.690 §11.6 ordering: ascending, shorter encodings padded with zero octets.

    The padding is what makes the comparison a total order on encodings of unequal
    length; it mirrors `der.py`, so what the encoder emits is exactly what
    `der_violations` accepts.
    """
    if len(children) < 2:
        return children
    encoded = [encode_tlv(c) for c in children]
    width = max(len(e) for e in encoded)
    return [c for _, c in sorted(zip(encoded, children),
                                 key=lambda pair: pair[0].ljust(width, b"\x00"))]


def _matches(tlv: Tlv, expected: Tag) -> bool:
    return tlv.tag.cls is expected.cls and tlv.tag.number == expected.number


def _matches_any(tlv: Tlv, comp: Component) -> bool:
    # An untagged open type accepts ANY tag: the schema does not fix its contained type,
    # so there is nothing to compare against (X.681 14). This is what lets
    # `parameters ANY OPTIONAL` sit at the end of AlgorithmIdentifier and absorb whatever
    # the algorithm actually carries.
    if comp.tag is None and isinstance(comp.type, OpenType):
        return True
    return any(_matches(tlv, tag) for tag in comp.expected_tags())


def _apply_tag(comp: Component, base: Tlv) -> Tlv:
    if comp.tag is None:
        return base
    if comp.explicit:                                     # §8.14.3
        return Tlv(Tag(comp.tag_class, comp.tag, True), b"", [base])
    return Tlv(Tag(comp.tag_class, comp.tag, base.tag.constructed),    # §8.14.4
               base.content, base.children, offset=base.offset)


def _strip_tag(comp: Component, tlv: Tlv) -> Tlv:
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


__all__ = ["Asn1Type", "Choice", "Component", "Module", "OpenType", "Primitive",
           "Sequence", "SequenceOf", "Set", "SetOf"]
