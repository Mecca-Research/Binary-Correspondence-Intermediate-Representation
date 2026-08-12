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
    #: True when this component appears AFTER the `...` extension marker, i.e. it is an
    #: extension addition rather than part of the extension root (X.680 §25.1). Invisible
    #: to BER/DER/OER, which encode a component the same way wherever it sits; PER splits
    #: the two hard (X.691 §19.1/§19.7-§19.9: the root gets a presence bitmap, additions
    #: get a separate bitmap and are each wrapped as an open type).
    extension: bool = False
    #: For an extension addition GROUP (`[[ a, b ]]`, X.680 §25.1): the members of the
    #: bracket. X.691 §19.9 encodes the whole group as ONE open type holding a SEQUENCE of
    #: these, so a single bit in the addition bitmap covers the bracket however many
    #: components it holds -- and a group whose members are all absent is itself absent.
    group: tuple["Component", ...] | None = None

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

    def require_tag(self, tlv: Tlv) -> None:
        """X.690 §8.1.2: refuse an encoding whose identifier octets are not this type's.

        A schema-DIRECTED decode is told which type to expect, so the tag on the wire is a
        claim to check, not information to follow. Without this the decoder fell through to
        the value-directed path and returned whatever the octets happened to spell: an
        INTEGER schema handed `01 01 ff` returned `True`, and a SEQUENCE schema accepted a
        SET or a constructed context tag over identical contents. That is not a decode of
        the requested type -- it is the octets choosing the type, at the exact boundary
        where untrusted bytes become a value.

        Class and number only. Whether the constructed bit is legal is a per-type question
        that each `decode` already answers: §8.9.1 requires a SEQUENCE to be constructed,
        while §8.23.6 lets BER spell a character string either way, so a shared check here
        would have to be wrong for one of them.

        Types that genuinely have no tag of their own -- CHOICE (§8.13/X.680 §29.1) and an
        open type (X.681 §14) -- override or do not call this: a CHOICE checks the tag
        against its alternatives instead, and an open type accepts what it contains.
        """
        want = self.base_tag()
        if tlv.tag.cls is not want.cls or tlv.tag.number != want.number:
            raise Asn1Error(
                f"{self.name}: expected {want} but the encoding carries {tlv.tag} "
                f"(X.690 8.1.2)", tlv.offset)


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
    #: For ENUMERATED: the `identifier(number)` list of the enumeration ROOT, in source
    #: order. Like `constraint`, this is invisible to BER/DER/OER -- all three encode the
    #: enumeration *value* (X.690 §8.4, X.696 §11) -- but PER encodes the enumeration
    #: *INDEX* (X.691 §14.1/§14.2: sort the root ascending by value, number from zero, then
    #: encode as a constrained whole number with lb=0 and ub=the largest index). Without the
    #: enumeration a PER codec cannot know either the index or the bit width, so a bare
    #: ENUMERATED is encodable under the other three rules and not under this one.
    #: X.682 §10.6 b): the permitted values of a table-constrained VALUE field, i.e. one
    #: column of an associated table. Deliberately NOT `constraint`: X.691 §10.3.4 and
    #: §10.3.5 make table and component relation constraints NOT PER-visible, and X.696 is
    #: the same, so putting these in `constraint` would narrow a PER field's WIDTH from a
    #: constraint the standard says the encoder must not see. A verifier reads this; no
    #: encoder does.
    table_values: tuple | None = None
    #: X.682 §11.4: the type whose ENCODING this octet/bit string's value is. Set by a
    #: `CONTAINING` constraint. Like a table-constrained open type this is resolvable -- the
    #: octets are a complete encoding of `contains` -- and like it, resolution is an
    #: enrichment: the octets stay exactly as they arrived.
    contains: object | None = None
    #: §11.5's `ENCODED BY`: the object identifier of the rules that produced the contents.
    #: Recorded, not dispatched on -- naming a rule this rail does not implement is a
    #: statement about the data, not licence to guess at it.
    encoded_by: tuple | None = None
    enumeration: tuple[tuple[str, int], ...] | None = None
    #: §20.1's items declared AFTER the `...`. Held apart from `enumeration`, which is the
    #: extension ROOT: X.691 §14.2 encodes a root value as an INDEX into the sorted root, so
    #: an extension item placed in that list would be encoded as -- and would mean -- a
    #: different enumerator. Invisible to BER/DER/OER, which encode the value itself.
    enum_extension: tuple[tuple[str, int], ...] | None = None
    #: True when the "Enumerations" production carried an extension marker (X.680 §20.1).
    #: X.691 §10.3.22 a) makes such a type extensible for PER, which adds the §14.3 bit.
    enum_extensible: bool = False

    def enum_indices(self) -> dict[int, int]:
        """X.691 §14.1: enumeration value -> enumeration index, root sorted ascending."""
        if not self.enumeration:
            raise Asn1Error(
                f"{self.name}: ENUMERATED has no enumeration; PER encodes the enumeration "
                f"index (X.691 §14.1), which cannot be derived from the value alone")
        ordered = sorted({number for _name, number in self.enumeration})
        return {number: index for index, number in enumerate(ordered)}

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
        self.require_tag(tlv)
        return from_tlv(tlv, strictness=strictness)


#: The restricted character string types of X.680 §41, which §8.23.3 unifies as
#: implicitly-tagged octet strings and which therefore all accept a Python `str`.
_STRING_UNIVERSALS = frozenset({
    Universal.UTF8_STRING, Universal.NUMERIC_STRING, Universal.PRINTABLE_STRING,
    Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING, Universal.IA5_STRING,
    Universal.GRAPHIC_STRING, Universal.VISIBLE_STRING, Universal.GENERAL_STRING,
    Universal.UNIVERSAL_STRING, Universal.BMP_STRING, Universal.OBJECT_DESCRIPTOR,
})


@dataclass(frozen=True)
class ObjectSetTable:
    """The ASSOCIATED TABLE of an information object set — X.681 §13.

    §13.1: "Every information object or information object set can be viewed as a table."
    Columns are the class's fields, rows are its objects. A cell holds a VALUE for a value
    field and an `Asn1Type` for a type field, which is the whole point: the type-field
    column is a column of types, and selecting a row selects a type.

    This is what makes an open type resolvable. X.682 §10.19/§10.20 select rows by matching
    the referenced components' values, then constrain the referencing component to the
    selected rows -- so for a type field, a selected row names the type the open type's
    octets actually are.

    `extensible` records §12.3's `...`. It matters at resolution time rather than encoding
    time: §12.9 lets a conforming peer use an object outside an extensible set, so an
    unmatched row is a legitimate "cannot resolve", not a malformed value.
    """

    object_class: str
    rows: tuple[dict, ...] = ()
    extensible: bool = False

    def column(self, field: str) -> tuple:
        """Every non-empty cell of one column, in row order (§13.2 a))."""
        return tuple(row[field] for row in self.rows if field in row)

    def select(self, criteria: dict) -> tuple[dict, ...]:
        """§10.19: the rows whose referenced columns all equal the given values."""
        return tuple(
            row for row in self.rows
            if all(field in row and row[field] == want for field, want in criteria.items()))


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
    #: The X.682 §10 constraining table, when a table constraint named one.
    table: ObjectSetTable | None = None
    #: The class field this open type IS (`&Type`), i.e. which column of `table` to select.
    field: str | None = None
    #: The AtNotation component paths of §10.7, as dotted paths relative to the constrained
    #: type. `("errorCategory",)` for `@errorCategory`. Empty for a SimpleTableConstraint,
    #: which selects no rows and therefore cannot resolve to a single type.
    governing: tuple[tuple[str, ...], ...] = ()
    #: The class fields the governing paths correspond to, positionally -- §10.15 requires
    #: the referenced components to be ObjectClassFieldTypes of the same class, so each
    #: governing path has a column that its value must match.
    governing_fields: tuple[str, ...] = ()

    def resolve(self, context: dict):
        """The contained type, per §10.19/§10.20, or None when it cannot be determined.

        `context` maps a governing path to the value already decoded for it. Returning None
        rather than raising is deliberate: §12.9 permits a peer to use an object outside an
        extensible set, and §10.21 only requires exactly one selected row when a referenced
        component is an identifier field. An unresolvable open type keeps its octets, which
        is what this type models anyway -- resolution is an enrichment, not a precondition.
        """
        if self.table is None or self.field is None or not self.governing:
            return None
        criteria = {}
        for path, column in zip(self.governing, self.governing_fields):
            if path not in context:
                return None                                # §10.18: a referenced component
            criteria[column] = context[path]               #         is absent
        selected = self.table.select(criteria)
        if len(selected) != 1:
            # §10.21 wants exactly one row when the field is a type field and a referenced
            # component is an identifier field. Zero rows is an unknown object; more than
            # one is genuinely ambiguous. Neither is a type this layer may invent.
            return None
        return selected[0].get(self.field)

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
        self.require_tag(tlv)
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
    #: True when the component list carried a `...` extension marker (X.680 §25.1).
    extensible: bool = False

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
        self.require_tag(tlv)
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
                _refuse_present_default(self, comp, out[comp.name], strictness,
                                        children[index].offset)
                _resolve_open_type(comp, out, strictness)
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
        self.require_tag(tlv)
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
    #: True when the component list carried a `...` extension marker (X.680 §25.1).
    extensible: bool = False

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
        self.require_tag(tlv)
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
                    _refuse_present_default(self, comp, out[comp.name], strictness,
                                            child.offset)
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
    #: True when the alternative list carried a `...` extension marker (X.680 §25.1).
    extensible: bool = False

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


def _resolve_open_type(comp: Component, decoded: dict, strictness: Strictness) -> None:
    """X.682 §10.19/§10.20: decode a table-constrained open type as its selected type.

    An open type's octets are the contained value's COMPLETE encoding, so once the governing
    siblings are known the contained type is decodable in place. Because a SEQUENCE decodes
    its components in definition order and §10.15 requires the referenced components to be
    in the same enclosing type, the governing values are already in `decoded` by the time the
    referencing component arrives -- which is the whole reason this can happen during the
    walk rather than in a second pass.

    The resolved value is added ALONGSIDE the octets rather than replacing them, under
    `<name>.resolved`. Replacing them would break the encode/decode round trip (the encoder
    needs the octets back) and would also throw away the one honest representation when the
    row is unknown. §12.9 permits a peer to use an object outside an extensible set, so an
    unresolvable open type is ordinary traffic, not a fault.
    """
    inner = comp.type
    if isinstance(inner, Primitive) and inner.contains is not None:
        # X.682 §11.4: the octets ARE an encoding of `contains`, so no sibling is needed.
        octets = decoded.get(comp.name)
        if isinstance(octets, (bytes, bytearray)):
            try:
                decoded[f"{comp.name}.resolved"] = inner.contains.decode(
                    decode_one(bytes(octets)), strictness=strictness)
            except Asn1Error:
                pass
        return
    if not isinstance(inner, OpenType) or inner.table is None or not inner.governing:
        return
    octets = decoded.get(comp.name)
    if not isinstance(octets, (bytes, bytearray)):
        return
    context = {path: decoded.get(path[-1]) for path in inner.governing
               if path[-1] in decoded}
    contained = inner.resolve(context)
    if contained is None:
        return
    try:
        decoded[f"{comp.name}.resolved"] = contained.decode(
            decode_one(bytes(octets)), strictness=strictness)
    except Asn1Error:
        # The row said one type and the octets are another. That is a real disagreement, but
        # it belongs to the caller that chose to trust the table, not to the structural
        # decode -- the octets are still exactly what arrived, and they stay available.
        return


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


def _refuse_present_default(kind, comp: Component, value, strictness: Strictness,
                            offset: int) -> None:
    """X.690 §11.5: under DER a component equal to its DEFAULT shall not be encoded.

    The encoders have always honoured this -- `encode` skips such a component -- but the
    decoders did not, so `30 06 02 01 01 02 01 07` and `30 03 02 01 07` BOTH decoded to
    the same abstract value under strict DER. Two accepted byte strings for one value is
    exactly what a canonical encoding exists to prevent, and StreamPack digests the octets
    it receives, so the looser half of the pair was a second spelling of an attested
    artifact.

    Only under DER. §11.5 is a clause-11 restriction; BER permits the redundant component
    and the BER decode path must keep accepting it.
    """
    if strictness is not Strictness.DER or not comp.has_default:
        return
    if value == comp.default:
        raise Asn1Error(
            f"{kind.name}: component {comp.name!r} is present and equal to its DEFAULT "
            f"{comp.default!r}; DER shall not encode it (X.690 11.5)", offset)


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
