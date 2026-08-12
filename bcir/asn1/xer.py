"""XML Encoding Rules — Rec. ITU-T X.693 (02/2021) | ISO/IEC 8825-4:2021.

XER is the odd member of the suite: it is the only encoding rule whose output a human
reads, and the only one that is *not* a compact binary form. X.693 Annex A says so with a
number — the same `PersonnelRecord` value is **653 octets** in BASIC-XER, 136 in BER and
**84** in UNALIGNED PER. So XER is here for interchange with peers that speak XML, never
as a candidate for a digested artifact, and the roadmap says exactly that.

**BASIC-XER out and in; CXER out and in; EXTENDED-XER neither.** The posture matches
`oer.py`: BCIR digests what it emits, so the emit path defaults to the canonical variant
(clause 9), which removes every encoder's option. The decode path accepts both, because
the interoperability half of the profile is that a peer built against an ordinary XER
toolkit can still talk to BCIR — and every CXER encoding is a legal BASIC-XER encoding.

WHERE THE RULES ACTUALLY LIVE. Clause 8 is four pages long, and most of it is a
redirection: §8.3 says the XML document element "shall be an `XMLTypedValue` as specified
in Rec. ITU-T X.680 ... 16.2". The per-type notation — `XMLBooleanValue`,
`XMLSequenceOfValue`, the `xmlasn1typename` table — is X.680's, not X.693's. What X.693
adds on top is a short list of *restrictions* (§8.3.5-§8.3.10) that delete the encoder's
options X.680 allows in a module, and clause 9, which deletes the rest.

The restrictions are load-bearing and are implemented here as the only forms emitted:

* §8.3.5 — BOOLEAN is `<true/>`/`<false/>`, never the text `true`. The text form is an
  EXTENDED-XER addition, so a BASIC-XER decoder has no reason to accept it.
* §8.3.6 — INTEGER is an `XMLSignedNumber`. The `<identifier/>` form a `NamedNumberList`
  would allow is not produced, so a decoder never has to resolve a name to a number.
* §8.3.7 — ENUMERATED is `<identifier/>`. Note the asymmetry with §8.3.6: an enumerated
  value has *no* numeric spelling in XER at all, which is why `Primitive.enumeration` is
  required here exactly as it is for PER's §14.1 index.
* §8.3.9 — BIT STRING is an `xmlbstring`; the `<flagName/>` identifier list is out.

WHAT MAKES CXER CANONICAL, and why each rule needed code rather than a comment:

* §9.6.1 sorts a SET's root components into X.680 §8.6 canonical tag order — and adds a
  rule found nowhere else in the suite: an **untagged CHOICE component is ordered as
  though it carried the smallest tag among its alternatives**, recursively. Annex A.4 is
  the proof that this is not decorative: `PersonnelRecord`'s components move from source
  order to `name, number, title, ...` because APPLICATION 1 and APPLICATION 2 sort ahead
  of CONTEXT 0.
* §9.5/§9.6.3 invert DER. X.690 §11.5 forbids encoding a component equal to its DEFAULT;
  CXER *requires* it, textually present. Two canonical rule sets, opposite answers, same
  abstract value — which is the cleanest illustration in the whole suite that "canonical"
  is a property of a rule set and not of a value.
* §9.7 sorts SET OF by the CXER encoding of each element, compared as ISO/IEC 10646 code
  points with a conceptual pad character that "precedes all other characters" (§9.7.3).
  Python's `str` comparison is code-point-wise and ranks a prefix before its extensions,
  which is that rule exactly — no padding needed.
* §9.2 normalizes REAL to one non-zero integer digit, a fraction with no trailing zeros
  after the first, an upper-case `E`, and no `+` anywhere.
* §9.3.1/§9.4 make the `XMLTypedValue` alternative **mandatory** for a BIT STRING or
  OCTET STRING carrying a CONTAINING constraint, so a contents constraint stops being
  documentation and starts choosing the encoding — the X.682 §11 model earns its place.
* §9.12 forbids the `xmlhstring` alternative for an open type, so under CXER an open type
  must be emitted as the type its table selects, or not at all.

NOT BUILT, and recorded rather than approximated:

* **EXTENDED-XER** (clause 10 and the ~25 encoding instructions of clauses 18-39). It is
  a different language — XER type prefixes, an encoding control section, XML namespaces,
  attributes — and none of it changes a BASIC-XER or CXER encoding. The decoder refuses
  XML attributes and namespace-qualified names by name rather than ignoring them.
* **§9.13, the TIME type and the useful time types.** Its rules b)-d) delete zero
  components from a *parsed* duration and time-difference components from the end point
  of an interval. `schema.py` carries a TIME value as the value-notation string, and a
  string is not a parsed interval; approximating would produce encodings that are wrong
  in exactly the cases the clause exists for. CANONICAL refuses them; BASIC passes them
  through unchanged.
* **§9.3.2**, no trailing zero bits when the bitstring has a `NamedBitList`. The type
  model carries no named-bit list, so there is no list to consult (§8.3.9 already removes
  the only other place one would be used).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from .codec import Strictness
from .schema import (Asn1Type, Choice, Component, Module, OpenType, Primitive,
                     Sequence, SequenceOf, Set, SetOf, _resolve_open_type)
from .tags import Asn1Error, TagClass, Universal
from .tlv import decode_one, encode_tlv
from .values import BitString, is_ascii_digits, is_number_form

#: §40.2 — the object identifiers that name these encoding rules.
BASIC_XER_OID: tuple[int, ...] = (2, 1, 5, 0)
CANONICAL_XER_OID: tuple[int, ...] = (2, 1, 5, 1)
EXTENDED_XER_OID: tuple[int, ...] = (2, 1, 5, 2)
#: §40.3/§16.9 — the ASN.1 namespace, `urn:oid:2.1.5.2.0.1`, recommended prefix `asn1`.
ASN1_NAMESPACE_OID: tuple[int, ...] = (2, 1, 5, 2, 0, 1)
ASN1_NAMESPACE: str = "urn:oid:2.1.5.2.0.1"


class XerRules(Enum):
    """Which of the two rule sets clause 5 distinguishes and this module implements."""

    BASIC = 0
    CANONICAL = 1


#: §8.2.1 b) — the only XML prolog a conforming encoder may emit, with the single SPACE
#: separators §8.2.2 requires.
XML_PROLOG: str = '<?xml version="1.0" encoding="UTF-8"?>'

#: §8.1.4 — "white-space" for the purposes of this Recommendation. Note that this is a
#: *smaller* set than XML's own S production would allow through, which is why the reader
#: below checks against this and not against `str.isspace`.
_WHITESPACE = frozenset("\t\n\r ")

#: The characters an XER element name is built from. Every name XER can produce is a
#: `typereference` (X.680 §12.2), an `identifier` (§12.3) or an `xmlasn1typename` (Table 4),
#: so the repertoire is letters, digits and HYPHEN-MINUS, plus the LOW LINE that Table 4 and
#: §14.2's "XML" guard introduce. COLON is scanned so that a namespace-qualified name can be
#: refused by name rather than by a confusing syntax error.
_NAME_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAME_CHARS = _NAME_START | frozenset("0123456789-.:")

#: X.680 Table 4 (§12.36.2) — the `xmlasn1typename` for each built-in type, plus §12.36.3's
#: rule that a useful type is named by the `typereference` used in its definition and
#: §41's rule that a restricted character string type is named by its own type name.
_XMLASN1TYPENAME: dict[int, str] = {
    Universal.BIT_STRING: "BIT_STRING",
    Universal.BOOLEAN: "BOOLEAN",
    Universal.INTEGER: "INTEGER",
    Universal.NULL: "NULL",
    Universal.OBJECT_IDENTIFIER: "OBJECT_IDENTIFIER",
    Universal.OCTET_STRING: "OCTET_STRING",
    Universal.REAL: "REAL",
    Universal.RELATIVE_OID: "RELATIVE_OID",
    Universal.OID_IRI: "OID_IRI",
    Universal.RELATIVE_OID_IRI: "RELATIVE_OID_IRI",
    Universal.ENUMERATED: "ENUMERATED",
    Universal.DATE: "DATE",
    Universal.TIME_OF_DAY: "TIME_OF_DAY",
    Universal.DATE_TIME: "DATE_TIME",
    Universal.DURATION: "DURATION",
    Universal.TIME: "TIME",
    # Table 4: the three types whose XER tag name is SEQUENCE because their XER encoding
    # is the encoding of an associated sequence type, not of the type as written.
    Universal.EXTERNAL: "SEQUENCE",
    Universal.EMBEDDED_PDV: "SEQUENCE",
    Universal.CHARACTER_STRING: "SEQUENCE",
    # §12.36.3: the useful types keep their own typereference.
    Universal.UTC_TIME: "UTCTime",
    Universal.GENERALIZED_TIME: "GeneralizedTime",
    Universal.OBJECT_DESCRIPTOR: "ObjectDescriptor",
    # X.680 §41: "The type name (e.g. IA5String)".
    Universal.UTF8_STRING: "UTF8String",
    Universal.NUMERIC_STRING: "NumericString",
    Universal.PRINTABLE_STRING: "PrintableString",
    Universal.TELETEX_STRING: "TeletexString",
    Universal.VIDEOTEX_STRING: "VideotexString",
    Universal.IA5_STRING: "IA5String",
    Universal.GRAPHIC_STRING: "GraphicString",
    Universal.VISIBLE_STRING: "VisibleString",
    Universal.GENERAL_STRING: "GeneralString",
    Universal.UNIVERSAL_STRING: "UniversalString",
    Universal.BMP_STRING: "BMPString",
}

#: X.680 §41 — the restricted character string types, whose XMLValue is an `xmlcstring`.
_STRING_UNIVERSALS = frozenset({
    Universal.UTF8_STRING, Universal.NUMERIC_STRING, Universal.PRINTABLE_STRING,
    Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING, Universal.IA5_STRING,
    Universal.GRAPHIC_STRING, Universal.VISIBLE_STRING, Universal.GENERAL_STRING,
    Universal.UNIVERSAL_STRING, Universal.BMP_STRING, Universal.OBJECT_DESCRIPTOR,
})

#: X.680 §38/§39 — the TIME type and the useful time types, which §9.13 canonicalizes with
#: rules this rail cannot apply to an unparsed string. See the module docstring.
_TIME_UNIVERSALS = frozenset({
    Universal.TIME, Universal.DATE, Universal.TIME_OF_DAY, Universal.DATE_TIME,
    Universal.DURATION,
})

#: X.680 Table 3 (§12.15.5) — the control characters that have no direct spelling in an
#: `xmlcstring` and are written as empty elements instead. Note what is NOT here: 9, 10
#: and 13 appear literally (the Table's own NOTE), which is why `_WHITESPACE` and this
#: table are disjoint.
_CONTROL_ELEMENT: dict[int, str] = {
    0: "nul", 1: "soh", 2: "stx", 3: "etx", 4: "eot", 5: "enq", 6: "ack", 7: "bel",
    8: "bs", 11: "vt", 12: "ff", 14: "so", 15: "si", 16: "dle", 17: "dc1", 18: "dc2",
    19: "dc3", 20: "dc4", 21: "nak", 22: "syn", 23: "etb", 24: "can", 25: "em",
    26: "sub", 27: "esc", 28: "is4", 29: "is3", 30: "is2", 31: "is1",
}
_CONTROL_CHARACTER: dict[str, str] = {
    name: chr(code) for code, name in _CONTROL_ELEMENT.items()}


def rules_oid(rules: XerRules) -> tuple[int, ...]:
    """§40.2 — the object identifier for a rule set."""
    return BASIC_XER_OID if rules is XerRules.BASIC else CANONICAL_XER_OID


# --- §14.2 the XML tag name of a type ---------------------------------------------------

class XerTypeNames:
    """The `typereference` that names a type in an XML tag (X.680 §14.2, §26.10).

    XER's element names come from two different places and the difference is visible in
    Annex A: `<ChildInformation>` is a *typereference*, while a SEQUENCE OF an inline
    `INTEGER` would produce `<INTEGER>` from Table 4. §26.10 states the rule — if the
    component is a `typereference` use it, otherwise use the `xmlasn1typename`.

    The type model has no "I am a reference" bit, because a reference and its referent are
    the same object once lowered. So the mapping is by *identity*: a module's `types` dict
    already holds one object per assigned name, and the lowerer hands out that same object
    everywhere the name is used. Registration keeps a strong reference to each type so an
    `id()` can never be recycled underneath the table.

    When two names alias one type, the first registered wins, which for a `Module` built
    by the front-end is the one that appears first in the source.
    """

    __slots__ = ("_by_id", "_keep")

    def __init__(self, types=None) -> None:
        self._by_id: dict[int, str] = {}
        self._keep: list[Asn1Type] = []
        if types is not None:
            mapping = types.types if isinstance(types, Module) else types
            for name, kind in mapping.items():
                self.add(name, kind)

    def add(self, name: str, kind: Asn1Type) -> "XerTypeNames":
        if id(kind) not in self._by_id:
            self._by_id[id(kind)] = name
            self._keep.append(kind)
        return self

    def name_of(self, kind: Asn1Type) -> str | None:
        return self._by_id.get(id(kind))


def _guarded(name: str) -> str:
    """§14.2: a tag name beginning with "XML" gets a LOW LINE pre-pended.

    XML 1.0 reserves names starting with "xml" in any case combination, so a type called
    `XMLDocument` cannot lend its name to an element unchanged.
    """
    return "_" + name if name.startswith("XML") else name


def _builtin_xml_name(kind: Asn1Type) -> str:
    if isinstance(kind, Primitive):
        name = _XMLASN1TYPENAME.get(kind.universal)
        if name is None:
            raise Asn1Error(
                f"XER: UNIVERSAL {int(kind.universal)} has no xmlasn1typename "
                f"(X.680 Table 4)")
        return name
    if isinstance(kind, Sequence):
        return "SEQUENCE"
    if isinstance(kind, Set):
        return "SET"
    if isinstance(kind, SequenceOf):
        return "SEQUENCE_OF"
    if isinstance(kind, SetOf):
        return "SET_OF"
    if isinstance(kind, Choice):
        return "CHOICE"
    raise Asn1Error(
        f"XER: {type(kind).__name__} has no XML tag name; an open type is named by the "
        f"type its table selects, not by itself (X.681 14)")


def xml_type_name(kind: Asn1Type, names: XerTypeNames | None = None) -> str:
    """X.680 §14.2/§26.10 — the `NonParameterizedTypeName` for a type."""
    if names is not None:
        reference = names.name_of(kind)
        if reference is not None:
            return _guarded(reference)
    return _guarded(_builtin_xml_name(kind))


# --- §12.15 the xmlcstring lexical item -------------------------------------------------

def escape_xmlcstring(value: str) -> str:
    """X.680 §12.15.4/§12.15.5 — the escapes an `xmlcstring` requires.

    Only two mechanisms are used, and CXER is the reason: §9.1.3 forbids the numeric
    `&#n;`/`&#xn;` escapes outright, so a canonical encoder is left with `&amp;`/`&lt;`/
    `&gt;` and the Table 3 empty elements. Emitting only those keeps one code path for both
    rule sets — there is no `rules` argument because every string this produces is legal
    under either, and the asymmetry lives entirely on the decode side.

    A character outside the `xmlcstring` repertoire of §12.15.1 — a lone surrogate, FFFE,
    FFFF — is refused rather than escaped. X.680 §41.10's NOTE is explicit that such values
    "cannot be transferred using XML Encoding Rules"; there is no spelling to fall back to.
    """
    out: list[str] = []
    for character in value:
        code = ord(character)
        if character == "&":
            out.append("&amp;")                              # §12.15.4
        elif character == "<":
            out.append("&lt;")
        elif character == ">":
            out.append("&gt;")
        elif code in _CONTROL_ELEMENT:
            out.append(f"<{_CONTROL_ELEMENT[code]}/>")       # §12.15.5
        elif code in (9, 10, 13) or 32 <= code <= 0xD7FF \
                or 0xE000 <= code <= 0xFFFD or 0x10000 <= code <= 0x10FFFF:
            out.append(character)                            # §12.15.1
        else:
            raise Asn1Error(
                f"XER: U+{code:04X} is not an xmlcstring character (X.680 12.15.1) and "
                f"has no escape; the value cannot be transferred in XER (X.680 41.10)")
    return "".join(out)


# --- clause 9 canonical spellings -------------------------------------------------------

def canonical_realnumber(value: float) -> str:
    """§9.2.3-§9.2.5 — one non-zero integer digit, a trimmed fraction, `E`, no `+`.

    The fraction keeps at least one digit even when it is zero (§9.2.3 says so explicitly),
    so `1.0` is `1.0E0` and not `1E0`. Trailing zeros are removed only *after* that first
    digit, which is why the trim and the pad are two separate steps rather than one.
    """
    if value == 0:
        return "0"                                           # §9.2.1
    number = Decimal(repr(abs(value)))
    _sign, digits, exponent = number.as_tuple()
    # `as_tuple` gives the digit string and a base-10 exponent; the scientific exponent is
    # the one that leaves a single digit ahead of the point.
    scientific = exponent + len(digits) - 1
    trimmed = list(digits)
    while len(trimmed) > 1 and trimmed[-1] == 0:
        trimmed.pop()
    if len(trimmed) == 1:
        trimmed.append(0)                                    # §9.2.3: at least one digit
    body = f"{trimmed[0]}.{''.join(str(d) for d in trimmed[1:])}E{scientific}"
    return ("-" + body) if value < 0 else body                # §9.2.5: no "+"


def _canonical_time(text: str, universal: int) -> str:
    """§9.10/§9.11 — the canonical spelling of a GeneralizedTime or UTCTime.

    Nothing here converts a time: an encoding that does not already terminate in "Z" is
    refused rather than shifted, because shifting is a calendar operation on a value this
    layer holds as an opaque string, and a wrong shift is a wrong *time* rather than a
    wrong spelling. Trailing zeros in the fraction, and a comma used as the decimal sign,
    are pure spelling and are fixed in place.
    """
    what = "GeneralizedTime" if universal == Universal.GENERALIZED_TIME else "UTCTime"
    clause = "9.10" if universal == Universal.GENERALIZED_TIME else "9.11"
    if not text.endswith("Z"):
        raise Asn1Error(
            f"XER: CXER requires a {what} to terminate with \"Z\" ({clause}.1); "
            f"{text!r} carries a local time or an offset, and converting it is a "
            f"calendar operation on a value this layer holds as text")
    body = text[:-1].replace(",", ".")                       # §9.10.4
    integer, _, fraction = body.partition(".")
    wanted = 14 if universal == Universal.GENERALIZED_TIME else 12
    if len(integer) != wanted or not is_ascii_digits(integer):
        raise Asn1Error(
            f"XER: CXER requires the seconds of a {what} to be present ({clause}.2); "
            f"{text!r} is not the {wanted}-digit form")
    if universal == Universal.UTC_TIME:
        if fraction:
            raise Asn1Error(
                f"XER: UTCTime has no fractional seconds ({clause}.2); got {text!r}")
        return integer + "Z"
    fraction = fraction.rstrip("0")                           # §9.10.3
    return (f"{integer}.{fraction}Z" if fraction else f"{integer}Z")


def _canonical_key(comp: Component) -> tuple[int, int]:
    """§9.6.1 — X.680 §8.6 canonical order, with the untagged-CHOICE rule.

    §8.6 orders by tag class first (UNIVERSAL, APPLICATION, context-specific, PRIVATE) and
    then by number, which `TagClass`'s own values already spell. The addition §9.6.1 makes
    is for a component that has no single tag: an untagged CHOICE is ordered "as though it
    has a tag equal to that of the smallest tag in the RootAlternativeTypeList of that
    choice type or any such choice types nested within it" — and `expected_tags` already
    flattens the nesting, because a nested untagged CHOICE contributes its own
    alternatives' tags to the outer one.
    """
    tags = comp.expected_tags()
    if not tags:
        raise Asn1Error(
            f"XER: component {comp.name!r} shows no tag, so CXER cannot place it in the "
            f"canonical order a SET requires (9.6.1)")
    return min((int(tag.cls), tag.number) for tag in tags)


# --- the component list ------------------------------------------------------------------

def _flatten(components: tuple[Component, ...]) -> tuple[Component, ...]:
    """Expand X.680 §25.1 version brackets into their members.

    A `[[ a, b ]]` group is one bit in PER's addition bitmap (X.691 §19.9) and therefore
    one component in the type model. XER has no such wrapper — each member is its own XML
    element — so the bracket is transparent here and only its members are encoded.
    """
    out: list[Component] = []
    for comp in components:
        if comp.group is not None:
            out.extend(comp.group)
        else:
            out.append(comp)
    return tuple(out)


def _ordered(kind, rules: XerRules) -> tuple[Component, ...]:
    """The order the components are emitted in.

    §9.6.1/§9.6.2: a canonical SET sorts its *root* into canonical tag order and then
    appends the extension additions in the order they are defined — the additions are not
    merged into the sort, because a later version adding a component must not be able to
    move an existing one.
    """
    components = _flatten(kind.components)
    if rules is not XerRules.CANONICAL or not isinstance(kind, Set):
        return components
    root = [c for c in components if not c.extension]
    additions = [c for c in components if c.extension]
    root.sort(key=_canonical_key)
    return tuple(root + additions)


# --- encoding ----------------------------------------------------------------------------

def _typed(name: str, kind: Asn1Type, value, rules: XerRules,
           names: XerTypeNames | None, context: dict | None = None) -> str:
    """X.680 §16.2's `XMLTypedValue`, with the §17.8/§9.1.4 empty-element form.

    §17.8 lets a start tag immediately followed by an end tag collapse to `<name/>`;
    §9.1.4 makes that mandatory under CXER. Emitting it under both rule sets is legal
    either way, and it is the only spelling for a NULL, whose XMLValue is `empty`.
    """
    body = _xml_value(kind, value, rules, names, context)
    return f"<{name}/>" if body == "" else f"<{name}>{body}</{name}>"


def _contained_typed(kind: Primitive, octets: bytes, rules: XerRules,
                     names: XerTypeNames | None) -> str:
    """§9.3.1/§9.4 with X.680 §22.11/§23.4 — a contents constraint chooses the encoding.

    The octets of a CONTAINING string *are* a complete encoding of the constrained type
    (X.682 §11.4), so the XMLTypedValue alternative is reachable: decode them with that
    type and emit the value as XML. §9.3.1 and §9.4 both say that when the alternative can
    be used it *shall* be used, which is what turns this from an option into a branch.
    """
    contained = kind.contains
    value = contained.decode(decode_one(octets), strictness=Strictness.BER)
    return _typed(xml_type_name(contained, names), contained, value, rules, names)


def _governing_context(kind: OpenType, siblings: dict | None) -> dict:
    """X.682 §10.19's row-selection criteria, keyed the way `OpenType.resolve` wants them.

    `OpenType.governing` holds component *paths*, and a SEQUENCE's value is keyed by
    component name, so the two need translating. §10.15 requires the referenced components
    to sit in the same enclosing type as the referencing one, which is what makes the
    sibling dict the right and only place to look.
    """
    if not siblings:
        return {}
    return {path: siblings[path[-1]] for path in kind.governing
            if path[-1] in siblings}


def _open_type(kind: OpenType, octets: bytes, rules: XerRules,
               names: XerTypeNames | None, context: dict | None) -> str:
    """§8.5 with X.681 §14.6, and §9.12's deletion of one of the two alternatives.

    BASIC-XER may spell an open type as an `xmlhstring`, and §8.5's NOTE is candid that
    this is a poor choice — nothing in the encoding says which rules produced those octets.
    CXER removes it (§9.12), which leaves only the typed form, which needs the contained
    type. That is precisely what X.682 §10.19's row selection produces, so an open type is
    canonically encodable exactly when its table resolves it.
    """
    if rules is not XerRules.CANONICAL:
        return octets.hex().upper()
    contained = kind.resolve(_governing_context(kind, context))
    if contained is None:
        raise Asn1Error(
            f"XER: CXER forbids the xmlhstring alternative for an open type (9.12), and "
            f"{kind.name} could not be resolved to a type by its table (X.682 10.19); "
            f"there is no canonical encoding for these octets")
    value = contained.decode(decode_one(octets), strictness=Strictness.BER)
    return _typed(xml_type_name(contained, names), contained, value, rules, names)


def _xml_value(kind: Asn1Type, value, rules: XerRules,
               names: XerTypeNames | None, context: dict | None = None) -> str:
    """X.680 §17.7's `XMLValue` — everything *between* a type's tags."""
    if isinstance(kind, Primitive):
        return _primitive_value(kind, value, rules, names)
    if isinstance(kind, (Sequence, Set)):
        return _components_value(kind, value, rules, names)
    if isinstance(kind, (SequenceOf, SetOf)):
        return _list_value(kind, value, rules, names)
    if isinstance(kind, Choice):
        return _choice_value(kind, value, rules, names)
    if isinstance(kind, OpenType):
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(
                f"{kind.name}: an open type value is the contained value's complete "
                f"encoding, so it must be bytes, not {type(value).__name__}")
        return _open_type(kind, bytes(value), rules, names, context)
    raise Asn1Error(f"XER: no encoding for schema type {type(kind).__name__}")


def _primitive_value(kind: Primitive, value, rules: XerRules,
                     names: XerTypeNames | None) -> str:
    universal = kind.universal

    if universal == Universal.BOOLEAN:                       # §8.3.5
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool, got {type(value).__name__}")
        return "<true/>" if value else "<false/>"

    if universal == Universal.INTEGER:                       # §8.3.6
        if isinstance(value, bool) or not isinstance(value, int):
            raise Asn1Error(f"{kind.name}: expected int, got {type(value).__name__}")
        # X.680 §19.13 forbids "-" & number when the number is zero; `str` of an int
        # never produces "-0", so the rule holds by construction.
        return str(value)

    if universal == Universal.ENUMERATED:                    # §8.3.7
        return _enumerated_value(kind, value)

    if universal == Universal.REAL:
        return _real_value(value)

    if universal == Universal.NULL:
        if value is not None:
            raise Asn1Error(f"{kind.name}: a NULL value is None, got {value!r}")
        return ""                                            # X.680 §24.3: `empty`

    if universal == Universal.BIT_STRING:                    # §8.3.9, §9.3
        return _bitstring_value(kind, value, rules, names)

    if universal == Universal.OCTET_STRING:                  # §9.4
        return _octetstring_value(kind, value, rules, names)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        return _oid_value(kind, value)

    if universal in (Universal.UTC_TIME, Universal.GENERALIZED_TIME):
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        if rules is XerRules.CANONICAL:
            return _canonical_time(value, universal)
        return escape_xmlcstring(value)

    if universal in _TIME_UNIVERSALS:
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        if rules is XerRules.CANONICAL:
            raise Asn1Error(
                f"XER: 9.13 canonicalizes {kind.name} by deleting components from a "
                f"parsed duration and interval, and this rail carries the value as its "
                f"value-notation string; BASIC-XER encodes it unchanged")
        return escape_xmlcstring(value)

    if universal in (Universal.OID_IRI, Universal.RELATIVE_OID_IRI) \
            or universal in _STRING_UNIVERSALS:
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        return escape_xmlcstring(value)                # X.680 §41.9

    raise Asn1Error(
        f"XER: no XMLValue notation for UNIVERSAL {int(universal)} in this rail")


def _enumerated_value(kind: Primitive, value) -> str:
    if not kind.enumeration:
        raise Asn1Error(
            f"{kind.name}: ENUMERATED has no enumeration; XER encodes the enumeration "
            f"IDENTIFIER (X.680 20.8), which — unlike BER's value (X.690 8.4) — cannot "
            f"be derived from the number alone")
    if isinstance(value, str):
        known = {name for name, _number in kind.enumeration}
        if value not in known:
            raise Asn1Error(f"{kind.name}: {value!r} is not an enumeration identifier")
        return f"<{value}/>"
    if isinstance(value, bool) or not isinstance(value, int):
        raise Asn1Error(f"{kind.name}: expected int or str, got {type(value).__name__}")
    for name, number in kind.enumeration:
        if number == value:
            return f"<{name}/>"
    raise Asn1Error(
        f"{kind.name}: {value} names no enumeration item, and XER has no numeric "
        f"spelling to fall back to (X.680 20.8)")


def _real_value(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Asn1Error(f"REAL: expected a number, got {type(value).__name__}")
    number = float(value)
    if number != number:                                     # §8.3.8: EmptyElementReal
        return "<NOT-A-NUMBER/>"
    if number == float("inf"):
        return "<PLUS-INFINITY/>"
    if number == float("-inf"):
        return "<MINUS-INFINITY/>"
    # §9.2's normalized form is a legal `realnumber` (X.680 §12.9), so it is emitted under
    # both rule sets rather than kept as a canonical-only spelling.
    return canonical_realnumber(number)


def _bitstring_value(kind: Primitive, value, rules: XerRules,
                     names: XerTypeNames | None) -> str:
    if kind.contains is not None and kind.encoded_by is None \
            and rules is XerRules.CANONICAL:                 # §9.3.1 with X.680 §22.11
        if not isinstance(value, BitString):
            raise Asn1Error(f"{kind.name}: expected BitString, got "
                            f"{type(value).__name__}")
        if value.unused:
            raise Asn1Error(
                f"{kind.name}: a CONTAINING bitstring's bits are a complete encoding "
                f"(X.682 11.4), so it cannot end {value.unused} bits into an octet")
        return _contained_typed(kind, bytes(value.octets), rules, names)
    if not isinstance(value, BitString):
        raise Asn1Error(f"{kind.name}: expected BitString, got {type(value).__name__}")
    # X.680 §12.11's `xmlbstring`, with no white-space (§9.1.2/§9.3.1).
    return "".join(str(value[index]) for index in range(value.bit_length))


def _octetstring_value(kind: Primitive, value, rules: XerRules,
                       names: XerTypeNames | None) -> str:
    if not isinstance(value, (bytes, bytearray)):
        raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
    if kind.contains is not None and kind.encoded_by is None \
            and rules is XerRules.CANONICAL:                 # §9.4 with X.680 §23.4
        return _contained_typed(kind, bytes(value), rules, names)
    return bytes(value).hex().upper()                        # §9.4: upper-case, no space


def _oid_value(kind: Primitive, value) -> str:
    if isinstance(value, str):
        arcs = tuple(int(part) for part in value.split(".") if part != "")
    elif isinstance(value, (tuple, list)):
        arcs = tuple(value)
    else:
        raise Asn1Error(
            f"{kind.name}: expected a tuple of arcs, got {type(value).__name__}")
    if not arcs:
        raise Asn1Error(f"{kind.name}: an object identifier has at least one arc")
    for arc in arcs:
        if isinstance(arc, bool) or not isinstance(arc, int) or arc < 0:
            raise Asn1Error(f"{kind.name}: {arc!r} is not an arc number")
    # §9.8/§9.9: the XMLNumberForm, so no `name(number)` spelling is emitted.
    return ".".join(str(arc) for arc in arcs)


def _components_value(kind, value, rules: XerRules, names: XerTypeNames | None) -> str:
    if not isinstance(value, dict):
        raise Asn1Error(f"{kind.name}: expected a dict, got {type(value).__name__}")
    components = _ordered(kind, rules)
    known = {comp.name for comp in components}
    unknown = {name for name in value if name not in known and not name.endswith(
        ".resolved")}
    if unknown:
        raise Asn1Error(f"{kind.name}: unknown component(s) {sorted(unknown)}")
    parts: list[str] = []
    for comp in components:
        if comp.name in value:
            item = value[comp.name]
            # §9.5/§9.6.3 invert X.690 §11.5: a component holding its DEFAULT is present
            # under CXER, and omitted under BASIC exactly as DER omits it.
            if comp.has_default and item == comp.default and rules is XerRules.BASIC:
                continue
        elif comp.has_default:
            if rules is not XerRules.CANONICAL:
                continue
            item = comp.default
        elif comp.optional:
            continue
        else:
            raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
        # X.680 §25.18's XMLNamedValue is `<identifier>XMLValue</identifier>`, so the
        # element is named by the COMPONENT, never by its type.
        parts.append(_typed(comp.name, comp.type, item, rules, names, value))
    return "".join(parts)


def _uses_value_list(element: Asn1Type) -> bool:
    """X.680 Table 5 with §26.6/§26.7 — `XMLValueList` or `XMLDelimitedItemList`.

    Table 5 sends CHOICE and NULL to `XMLValueList` outright, and makes BOOLEAN and
    ENUMERATED conditional on the empty-element form being used. X.693 §8.3.5 and §8.3.7
    make that form the *only* one a BASIC-XER encoder may produce, so the condition is
    always met here and those two join the list unconditionally.
    """
    if isinstance(element, Choice):
        return True
    if isinstance(element, Primitive):
        return element.universal in (
            Universal.NULL, Universal.BOOLEAN, Universal.ENUMERATED)
    return False


def _list_value(kind, value, rules: XerRules, names: XerTypeNames | None) -> str:
    if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
        raise Asn1Error(
            f"{kind.name}: expected a sequence of elements, got "
            f"{type(value).__name__}")
    element = kind.element
    element_name = xml_type_name(element, names)
    parts: list[str] = []
    for item in value:
        if _uses_value_list(element):
            body = _xml_value(element, item, rules, names)
            # §26.4: an `empty` XMLValue — only SEQUENCE OF NULL reaches this — takes the
            # `<NonParameterizedTypeName/>` alternative rather than vanishing.
            parts.append(body if body else f"<{element_name}/>")
        else:
            parts.append(_typed(element_name, element, item, rules, names))
    if isinstance(kind, SetOf) and rules is XerRules.CANONICAL:
        # §9.7.2/§9.7.3: sort by ISO/IEC 10646 code point with a conceptual pad character
        # that precedes every real one. Python compares `str` by code point and ranks a
        # prefix ahead of any extension of it, which is that rule with no padding needed.
        parts.sort()
    return "".join(parts)


def _choice_value(kind: Choice, value, rules: XerRules,
                  names: XerTypeNames | None) -> str:
    if not (isinstance(value, tuple) and len(value) == 2):
        raise Asn1Error(
            f"{kind.name}: value must be an (alternative, value) pair, got "
            f"{type(value).__name__}")
    chosen, payload = value
    for alt in _flatten(kind.alternatives):
        if alt.name == chosen:
            # X.680 §29.11: `<identifier>XMLValue</identifier>`, the alternative's name.
            return _typed(alt.name, alt.type, payload, rules, names)
    raise Asn1Error(
        f"{kind.name}: {chosen!r} is not an alternative; an unknown extension "
        f"alternative arrives as raw XML on decode (8.6.3) and is not re-encodable")


def encode_xer(kind: Asn1Type, value, *, name: str | None = None,
               rules: XerRules = XerRules.CANONICAL,
               names: XerTypeNames | None = None,
               prolog: bool = False) -> bytes:
    """Encode `value` as a complete XER encoding of `kind` (§8.1).

    The result is UTF-8 octets, because §8.1.3 says the XML document "shall be encoded
    using UTF-8 to produce a string of octets which forms the encoding" — the encoding is
    the octets, not the character string, which is why this returns `bytes` like every
    other rail here.

    `name` is the document element's `NonParameterizedTypeName` (§8.3); it defaults to the
    type's own name via `names`, falling back to X.680 Table 4.
    """
    if prolog and rules is XerRules.CANONICAL:
        raise Asn1Error("XER: CXER requires an empty XML prolog (9.1.1)")
    document = _typed(name or xml_type_name(kind, names), kind, value, rules, names)
    if prolog:
        document = XML_PROLOG + document
    return document.encode("utf-8")


# --- decoding ----------------------------------------------------------------------------

class _Reader:
    """A scanner for the XML subset clause 8 permits, and nothing else.

    Writing this by hand rather than reaching for a general XML parser is a decision, not
    an omission. §8.1.2's NOTE says a conforming BASIC-XER encoder never produces
    processing instructions, comments, DOCTYPE declarations or CDATA sections, and
    BASIC-XER never produces an attribute either — attributes and namespaces arrive only
    with EXTENDED-XER's ATTRIBUTE and NAMESPACE instructions. A general parser would
    accept all of them and hand back something that looks decoded; this one names each
    construct and the clause that excludes it.

    The other reason is `xmlcstring`. Inside a character string value, `<nul/>` is a
    *character*, not a child element (X.680 §12.15.5), and `&#233;` is legal under
    BASIC-XER but forbidden under CXER (§9.1.3). Which escapes are legal depends on the
    rule set and on the type being read — knowledge a schema-blind parser does not have.
    """

    __slots__ = ("text", "pos", "rules")

    def __init__(self, text: str, rules: XerRules) -> None:
        self.text = text
        self.pos = 0
        self.rules = rules

    def error(self, message: str) -> "Asn1Error":
        return Asn1Error(f"XER: {message}", self.pos)

    def skip_space(self) -> None:
        """§8.1.4 — the four characters that count as white-space here."""
        while self.pos < len(self.text) and self.text[self.pos] in _WHITESPACE:
            self.pos += 1

    def at_end(self) -> bool:
        return self.pos >= len(self.text)

    def read_prolog(self) -> bool:
        """§8.2 — an empty prolog, or exactly the one character sequence §8.2.1 b) gives."""
        if not self.text.startswith(XML_PROLOG, self.pos):
            if self.text.startswith("<?", self.pos):
                raise self.error(
                    "the only XML processing instruction a conforming encoder produces "
                    "is the 8.2.1 prolog, verbatim and separated by single SPACEs (8.2.2)")
            return False
        if self.rules is XerRules.CANONICAL:
            raise self.error("CXER requires an empty XML prolog (9.1.1)")
        self.pos += len(XML_PROLOG)
        self.skip_space()
        return True

    def peek_tag(self) -> tuple[str, str] | None:
        """The next tag as (kind, name) without consuming it, or None at end of input."""
        save = self.pos
        try:
            self.skip_space()
            # None means "no tag here", which includes plain text content — a `<` that is
            # present but malformed still raises, so a bad construct is never mistaken for
            # the absence of one.
            if self.at_end() or self.text[self.pos] != "<":
                return None
            return self._scan_tag()
        finally:
            self.pos = save

    def read_tag(self) -> tuple[str, str]:
        self.skip_space()
        return self._scan_tag()

    def _scan_tag(self) -> tuple[str, str]:
        text = self.text
        if self.at_end() or text[self.pos] != "<":
            raise self.error(f"expected a tag, found {text[self.pos:self.pos + 16]!r}")
        if text.startswith("<!--", self.pos):
            raise self.error(
                "XML comments are not part of a BASIC-XER encoding (8.1.2 NOTE)")
        if text.startswith("<![CDATA[", self.pos):
            raise self.error(
                "CDATA sections are not part of a BASIC-XER encoding (8.1.2 NOTE)")
        if text.startswith("<!", self.pos):
            raise self.error(
                "a document type declaration is not part of a BASIC-XER encoding "
                "(8.1.2 NOTE)")
        if text.startswith("<?", self.pos):
            raise self.error(
                "an XML processing instruction is not part of a BASIC-XER encoding "
                "(8.1.2 NOTE); only the 8.2.1 prolog is permitted, and only first")
        start = self.pos + 1
        closing = text.startswith("</", self.pos)
        if closing:
            start += 1
        cursor = start
        while cursor < len(text) and text[cursor] in _NAME_CHARS:
            cursor += 1
        name = text[start:cursor]
        if not name or name[0] not in _NAME_START:
            raise self.error(f"{name!r} is not an XML element name")
        if cursor < len(text) and text[cursor] not in "/> \t\n\r":
            # An XML Name may hold letters far outside ASCII; an XER element name may not,
            # because every name it can carry comes from a `typereference` (X.680 §12.2),
            # an `identifier` (§12.3) or Table 4, and all three are ASCII. Saying so is
            # what lets the C twin in `runtime/c/bcir_xer.c` agree character for character.
            raise self.error(
                f"{text[cursor]!r} is not a character of an XER element name; X.680 12.2 "
                f"and 12.3 admit only letters, digits and HYPHEN-MINUS, and X.680 14.2 "
                f"adds LOW LINE")
        if ":" in name:
            raise self.error(
                f"{name!r} is a namespace-qualified name; namespaces arrive with the "
                f"EXTENDED-XER NAMESPACE instruction (clause 29), which is not "
                f"implemented")
        self.pos = cursor
        self.skip_space()
        if self.pos < len(self.text) and self.text[self.pos] not in "/>":
            raise self.error(
                f"element {name!r} carries an XML attribute; BASIC-XER produces no "
                f"attributes, they arrive with the EXTENDED-XER ATTRIBUTE instruction "
                f"(clause 20), which is not implemented")
        if self.text.startswith("/>", self.pos):
            if closing:
                raise self.error(f"</{name}/> is not a tag")
            self.pos += 2
            return ("empty", name)
        if self.pos < len(self.text) and self.text[self.pos] == ">":
            self.pos += 1
            return ("end" if closing else "start", name)
        raise self.error(f"unterminated tag for element {name!r}")

    def expect_start(self, name: str) -> bool:
        """Consume `<name>` or `<name/>`; True when it was the empty-element form."""
        kind, found = self.read_tag()
        if found != name:
            raise self.error(f"expected element {name!r}, found {found!r}")
        if kind == "empty":
            return True
        if kind != "start":
            raise self.error(f"expected the start tag of {name!r}, found an end tag")
        return False

    def expect_end(self, name: str) -> None:
        kind, found = self.read_tag()
        if kind != "end" or found != name:
            raise self.error(f"expected </{name}>, found a {kind} tag for {found!r}")

    def read_raw_element(self) -> str:
        """Consume one complete element and return its text, tags included.

        This is how §8.6.2/§8.6.3's unknown extensions are handled: an element this
        version's type does not name is skipped whole rather than parsed, which is only
        safe because XML is self-delimiting where PER needs an explicit open-type wrapper
        to achieve the same thing.
        """
        self.skip_space()
        start = self.pos
        kind, name = self._scan_tag()
        if kind == "empty":
            return self.text[start:self.pos]
        if kind != "start":
            raise self.error(f"expected an element, found </{name}>")
        depth = 1
        while depth:
            index = self.text.find("<", self.pos)
            if index < 0:
                raise self.error(f"unterminated element {name!r}")
            self.pos = index
            inner, _found = self._scan_tag()
            if inner == "start":
                depth += 1
            elif inner == "end":
                depth -= 1
        return self.text[start:self.pos]

    def read_plain_text(self, name: str) -> str:
        """The text content of an element that is not a character string.

        White-space around the value is stripped: X.680 §16.2 permits it around an
        `XMLValue` in an `XMLTypedValue`, and §8.3.4 restricts which characters may be
        used for it. A restricted character string is the exception §41.9 carves out, and
        is read by `read_string_text` instead.
        """
        index = self.text.find("<", self.pos)
        if index < 0:
            raise self.error(f"unterminated element {name!r}")
        body = self.text[self.pos:index]
        self.pos = index
        if not self.text.startswith("</", self.pos):
            # Something other than the end tag interrupts the value. Scanning it here is
            # what makes the diagnosis specific: `_scan_tag` names comments, processing
            # instructions, CDATA and DOCTYPE against the clause that excludes them,
            # whereas letting the text through would report the far less useful "'' is not
            # an XMLSignedNumber".
            save = self.pos
            self._scan_tag()
            self.pos = save
            raise self.error(
                f"an element interrupts the value of {name!r}, whose XMLValue is text")
        return body.strip("".join(_WHITESPACE))

    def read_string_text(self, name: str) -> str:
        """An `xmlcstring` (X.680 §12.15), unescaped.

        Two things make this more than a `find('<')`: the Table 3 control characters are
        spelled as empty elements *inside* the content, and §9.1.3 forbids the numeric
        escapes under CXER, so which spellings are legal depends on the rule set.
        """
        out: list[str] = []
        while True:
            index = self.text.find("<", self.pos)
            if index < 0:
                raise self.error(f"unterminated element {name!r}")
            out.append(self._unescape(self.text[self.pos:index]))
            self.pos = index
            if self.text.startswith("</", self.pos):
                return "".join(out)
            save = self.pos
            kind, found = self._scan_tag()
            character = _CONTROL_CHARACTER.get(found)
            if kind != "empty" or character is None:
                self.pos = save
                raise self.error(
                    f"element {found!r} inside the character string value of {name!r} is "
                    f"not one of the X.680 Table 3 control-character escapes")
            out.append(character)

    def _unescape(self, body: str) -> str:
        out: list[str] = []
        index = 0
        while index < len(body):
            character = body[index]
            if character == ">":
                raise self.error(
                    "\">\" appears literally; X.680 12.15.2 admits it only as \"&gt;\" "
                    "or a numeric escape")
            if character != "&":
                out.append(character)
                index += 1
                continue
            stop = body.find(";", index)
            if stop < 0:
                raise self.error("an \"&\" begins an escape that is never terminated")
            entity = body[index + 1:stop]
            index = stop + 1
            if entity == "amp":
                out.append("&")
            elif entity == "lt":
                out.append("<")
            elif entity == "gt":
                out.append(">")
            elif entity.startswith("#"):
                # X.680 §12.15.8, deleted from CXER by §9.1.3. Accepting it under BASIC
                # and refusing it under CANONICAL is the whole difference between the two
                # here, so it is checked rather than tolerated.
                if self.rules is XerRules.CANONICAL:
                    raise self.error(
                        f"the numeric escape \"&{entity};\" is forbidden in CXER (9.1.3)")
                digits = entity[1:]
                # §12.15.8's escape is `&#` + decimal digits or `&#x` + hex digits. Handing
                # the remainder to `int()` accepted a far wider language: PEP 515
                # underscores, a leading PLUS SIGN, surrounding whitespace, every Unicode
                # decimal digit, and -- for the hex form -- a second `0x` prefix, since
                # `int(s, 16)` strips one. `&#65;`, `&#6_5;`, `&#+65;`, `&# 65 ;`,
                # `&#\u0666\u0665;`, `&#\uff16\uff15;`, `&#0065;`, `&#x41;` and `&#x0x41;`
                # were nine accepted spellings of the character `A`.
                hexform = digits[:1] in ("x", "X")
                body_digits = digits[1:] if hexform else digits
                legal = (all(character in "0123456789abcdefABCDEF"
                             for character in body_digits) and body_digits
                         if hexform else is_ascii_digits(body_digits))
                if not legal:
                    raise self.error(
                        f"\"&{entity};\" is not a numeric escape; X.680 12.15.8 spells it "
                        f"with DIGIT ZERO..DIGIT NINE (or hexadecimal digits after \"x\")")
                code = int(body_digits, 16 if hexform else 10)
                if not 0 <= code <= 0x10FFFF:
                    raise self.error(f"\"&{entity};\" is not an ISO/IEC 10646 character")
                out.append(chr(code))
            else:
                raise self.error(
                    f"\"&{entity};\" is not one of the escapes X.680 12.15.4 permits; "
                    f"XER defines no general entity mechanism")
        return "".join(out)


def _decode_typed(reader: _Reader, name: str, kind: Asn1Type, rules: XerRules,
                  names: XerTypeNames | None, context: dict | None = None):
    """Read one `XMLTypedValue` named `name` and return the value it denotes."""
    empty = reader.expect_start(name)
    if empty:
        # §17.8: `<x/>` stands for `<x></x>`, so the value is whatever an empty XMLValue
        # denotes for this type — "" for a string, the empty list, NULL.
        value = _decode_empty(kind, name, reader)
    else:
        value = _decode_value(reader, kind, name, rules, names, context)
        reader.expect_end(name)
    return value


def _decode_empty(kind: Asn1Type, name: str, reader: _Reader):
    if isinstance(kind, Primitive):
        if kind.universal == Universal.NULL:
            return None
        if kind.universal in _STRING_UNIVERSALS or kind.universal in (
                Universal.OID_IRI, Universal.RELATIVE_OID_IRI):
            return ""
        if kind.universal == Universal.OCTET_STRING:
            return b""
        if kind.universal == Universal.BIT_STRING:
            return BitString(b"", 0)
        raise reader.error(f"<{name}/> is not a value of {kind.name}")
    if isinstance(kind, (SequenceOf, SetOf)):
        return []
    if isinstance(kind, (Sequence, Set)):
        return _finish_components(kind, {}, reader)
    raise reader.error(f"<{name}/> is not a value of {kind.name}")


def _decode_value(reader: _Reader, kind: Asn1Type, name: str, rules: XerRules,
                  names: XerTypeNames | None, context: dict | None):
    if isinstance(kind, Primitive):
        return _decode_primitive(reader, kind, name, rules, names)
    if isinstance(kind, (Sequence, Set)):
        return _decode_components(reader, kind, rules, names)
    if isinstance(kind, (SequenceOf, SetOf)):
        return _decode_list(reader, kind, rules, names)
    if isinstance(kind, Choice):
        return _decode_choice(reader, kind, rules, names)
    if isinstance(kind, OpenType):
        return _decode_open_type(reader, kind, name, rules, names, context)
    raise reader.error(f"no decoding for schema type {type(kind).__name__}")


def _decode_primitive(reader: _Reader, kind: Primitive, name: str, rules: XerRules,
                      names: XerTypeNames | None):
    universal = kind.universal

    if universal == Universal.BOOLEAN:                       # §8.3.5
        tag = reader.peek_tag()
        if tag is None or tag[0] != "empty" or tag[1] not in ("true", "false"):
            raise reader.error(
                "a BASIC-XER boolean is <true/> or <false/> (8.3.5); the text spelling "
                "arrives with the EXTENDED-XER TEXT instruction (clause 31)")
        reader.read_tag()
        return tag[1] == "true"

    if universal == Universal.ENUMERATED:                    # §8.3.7
        tag = reader.peek_tag()
        if tag is None or tag[0] != "empty":
            raise reader.error(
                "a BASIC-XER enumerated value is <identifier/> (8.3.7), and X.680 20.8 "
                "gives an enumerated value no numeric spelling at all")
        _tag, found = reader.read_tag()
        if not kind.enumeration:
            raise reader.error(
                f"{kind.name}: ENUMERATED has no enumeration, so <{found}/> cannot be "
                f"mapped to a value (X.680 20.8)")
        for item, number in kind.enumeration:
            if item == found:
                return number
        raise reader.error(
            f"{found!r} is not an enumeration identifier of {kind.name}")

    if universal == Universal.REAL:
        tag = reader.peek_tag()
        if tag is not None and tag[0] == "empty":            # §8.3.8
            _kind, found = reader.read_tag()
            special = {"PLUS-INFINITY": float("inf"),
                       "MINUS-INFINITY": float("-inf"),
                       "NOT-A-NUMBER": float("nan")}
            if found not in special:
                raise reader.error(f"<{found}/> is not an XMLSpecialRealValue (X.680 21.6)")
            return special[found]
        return _parse_real(reader, reader.read_plain_text(name))

    if universal == Universal.NULL:
        body = reader.read_plain_text(name)
        if body:
            raise reader.error(f"a NULL value has no content (X.680 24.3); got {body!r}")
        return None

    if universal == Universal.INTEGER:                       # §8.3.6
        tag = reader.peek_tag()
        if tag is not None and tag[0] != "end":
            raise reader.error(
                f"<{tag[1]}> is not an XMLSignedNumber; 8.3.6 removes both identifier "
                f"forms X.680 19.9 would otherwise allow for an integer")
        return _parse_integer(reader, reader.read_plain_text(name))

    if universal == Universal.BIT_STRING:
        if kind.contains is not None and kind.encoded_by is None:
            tag = reader.peek_tag()
            if tag is not None and tag[0] in ("start", "empty"):
                return BitString(_decode_contained(reader, kind, rules, names), 0)
        return _parse_bitstring(reader, reader.read_plain_text(name))

    if universal == Universal.OCTET_STRING:
        if kind.contains is not None and kind.encoded_by is None:
            tag = reader.peek_tag()
            if tag is not None and tag[0] in ("start", "empty"):
                return _decode_contained(reader, kind, rules, names)
        return _parse_hex(reader, reader.read_plain_text(name))

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        return _parse_oid(reader, reader.read_plain_text(name))

    if universal in (Universal.UTC_TIME, Universal.GENERALIZED_TIME):
        return reader.read_plain_text(name)

    if universal in _TIME_UNIVERSALS:
        return reader.read_plain_text(name)

    if universal in _STRING_UNIVERSALS or universal in (
            Universal.OID_IRI, Universal.RELATIVE_OID_IRI):
        return reader.read_string_text(name)                 # X.680 §41.9

    raise reader.error(
        f"no XMLValue notation for UNIVERSAL {int(universal)} in this rail")


def _parse_integer(reader: _Reader, body: str) -> int:
    """X.680 §19.9's `XMLSignedNumber`, which §8.3.6 makes the only integer spelling."""
    digits = body[1:] if body.startswith("-") else body
    if not is_ascii_digits(digits):
        raise reader.error(
            f"{body!r} is not an XMLSignedNumber (X.680 19.9); 8.3.6 admits no other "
            f"spelling of an integer")
    if digits[0] == "0" and len(digits) > 1:
        raise reader.error(f"{body!r} has a leading zero (X.680 12.8)")
    if body.startswith("-") and digits == "0":
        raise reader.error("\"-0\" is forbidden by X.680 19.13")
    return int(body)


def _parse_real(reader: _Reader, body: str) -> float:
    if not body:
        raise reader.error("an XMLNumericRealValue may not be empty (X.680 21.6)")
    text = body[1:] if body.startswith("-") else body
    mantissa, marker, exponent = text.partition("E")
    if not marker:
        mantissa, marker, exponent = text.partition("e")
    integer, _dot, fraction = mantissa.partition(".")
    if not is_ascii_digits(integer) or (fraction and not is_ascii_digits(fraction)):
        raise reader.error(f"{body!r} is not a realnumber (X.680 12.9)")
    if marker:
        signless = exponent[1:] if exponent.startswith("-") else exponent
        if not is_ascii_digits(signless):
            raise reader.error(f"{body!r} has no valid exponent (X.680 12.9)")
        if exponent.startswith("+"):
            raise reader.error(f"{body!r} carries a \"+\"; X.680 12.9 admits only \"-\"")
    return float(body)


def _parse_bitstring(reader: _Reader, body: str) -> BitString:
    """X.680 §12.11's `xmlbstring`, with §8.3.4's white-space removed."""
    bits = "".join(character for character in body if character not in _WHITESPACE)
    if any(character not in "01" for character in bits):
        raise reader.error(f"{body!r} is not an xmlbstring (X.680 12.11)")
    padded = bits + "0" * (-len(bits) % 8)
    octets = bytes(int(padded[at:at + 8], 2) for at in range(0, len(padded), 8))
    return BitString(octets, (-len(bits)) % 8)


def _parse_hex(reader: _Reader, body: str) -> bytes:
    """X.680 §12.13's `xmlhstring`, with §8.3.4's white-space removed."""
    digits = "".join(character for character in body if character not in _WHITESPACE)
    if len(digits) % 2:
        raise reader.error(
            f"an xmlhstring denotes whole octets, so it has an even number of "
            f"characters; got {len(digits)}")
    try:
        return bytes.fromhex(digits)
    except ValueError:
        raise reader.error(f"{body!r} is not an xmlhstring (X.680 12.13)") from None


def _parse_oid(reader: _Reader, body: str) -> tuple[int, ...]:
    parts = body.split(".")
    # §12.26 spells the arc out: characters "in the range 0 (DIGIT ZERO) to 9 (DIGIT NINE)",
    # and "shall not commence with a 0 (DIGIT ZERO) character unless it has only a single
    # character". `str.isdigit()` answered True for ARABIC-INDIC and FULLWIDTH digits and
    # said nothing about leading zeros, so `1.2.٨٤٠` and `1.2.0840` both decoded to
    # (1, 2, 840) -- three spellings of one object identifier. `_parse_integer` above
    # already applied the leading-zero half; this production needs it just as much.
    if not body or any(not is_number_form(part) for part in parts):
        raise reader.error(
            f"{body!r} is not an XMLObjectIdentifierValue; 9.8 requires every component "
            f"to be an XMLNumberForm of DIGIT ZERO..DIGIT NINE with no leading zero "
            f"(X.680 12.26)")
    return tuple(int(part) for part in parts)


def _decode_contained(reader: _Reader, kind: Primitive, rules: XerRules,
                      names: XerTypeNames | None) -> bytes:
    """The `XMLTypedValue` alternative of a CONTAINING string (X.680 §22.11/§23.4).

    The model holds such a value as the contained value's *octets*, so decoding the XML
    form means re-encoding what it denotes. That is sound because those octets are a
    canonical (DER) encoding by construction, and the XML — not some earlier octet string
    — is what the peer actually sent.
    """
    contained = kind.contains
    name = xml_type_name(contained, names)
    value = _decode_typed(reader, name, contained, rules, names)
    return encode_tlv(contained.encode(value))


def _decode_open_type(reader: _Reader, kind: OpenType, name: str, rules: XerRules,
                      names: XerTypeNames | None, context: dict | None) -> bytes:
    tag = reader.peek_tag()
    if tag is not None and tag[0] in ("start", "empty"):     # X.681 §14.6's XMLTypedValue
        contained = kind.resolve(_governing_context(kind, context))
        if contained is None:
            raise reader.error(
                f"{kind.name} is spelled as a typed value, but its table selects no row "
                f"for the governing components (X.682 10.19), so there is no type to "
                f"decode <{tag[1]}> with")
        expected = xml_type_name(contained, names)
        if tag[1] != expected:
            raise reader.error(
                f"the table selects {expected} for this open type, but the value is "
                f"spelled <{tag[1]}>")
        value = _decode_typed(reader, expected, contained, rules, names)
        return encode_tlv(contained.encode(value))
    if rules is XerRules.CANONICAL:
        raise reader.error(
            "CXER forbids the xmlhstring alternative for an open type (9.12)")
    return _parse_hex(reader, reader.read_plain_text(name))


def _finish_components(kind, out: dict, reader: _Reader) -> dict:
    """Fill in the components no element supplied, or refuse if one was mandatory."""
    for comp in _flatten(kind.components):
        if comp.name in out:
            continue
        if comp.has_default:
            out[comp.name] = comp.default                    # X.680 §25.12
        elif not comp.optional:
            raise reader.error(
                f"{kind.name}: mandatory component {comp.name!r} is missing "
                f"(X.680 25.20)")
    return out


def _decode_components(reader: _Reader, kind, rules: XerRules,
                       names: XerTypeNames | None) -> dict:
    """A SEQUENCE in definition order (X.680 §25.20) or a SET in any order (§27.9).

    §8.6.1-§8.6.2 are what the `extensible` branch implements: a decoder "shall accept as
    a valid XML document" an encoding carrying extensions it does not know, and those
    arrive as element names distinct from every expected one. Skipping the element whole
    is safe here in a way it is not in PER, where an unknown addition is only recoverable
    because §19.9 wrapped it in an explicit open type — XML is self-delimiting.
    """
    components = _flatten(kind.components)
    unordered = isinstance(kind, Set)
    out: dict = {}
    index = 0
    while True:
        tag = reader.peek_tag()
        if tag is None or tag[0] == "end":
            break
        element = tag[1]
        if unordered:
            position = next((at for at, comp in enumerate(components)
                             if comp.name == element and comp.name not in out), None)
        else:
            position = next((at for at in range(index, len(components))
                             if components[at].name == element), None)
        if position is None:
            if kind.extensible:
                reader.read_raw_element()                    # §8.6.2
                continue
            raise reader.error(
                f"{kind.name}: <{element}> matches no component, and the type carries no "
                f"extension marker (8.6.2)")
        if not unordered:
            for skipped in components[index:position]:
                if skipped.has_default:
                    out[skipped.name] = skipped.default
                elif not skipped.optional:
                    raise reader.error(
                        f"{kind.name}: mandatory component {skipped.name!r} is missing "
                        f"or out of order (X.680 25.20)")
            index = position + 1
        comp = components[position]
        out[comp.name] = _decode_typed(reader, comp.name, comp.type, rules, names, out)
        _resolve_open_type(comp, out, Strictness.BER)
    return _finish_components(kind, out, reader)


def _decode_list(reader: _Reader, kind, rules: XerRules,
                 names: XerTypeNames | None) -> list:
    element = kind.element
    element_name = xml_type_name(element, names)
    values: list = []
    while True:
        tag = reader.peek_tag()
        if tag is None or tag[0] == "end":
            return values
        if _uses_value_list(element):
            if isinstance(element, Choice):
                values.append(_decode_choice(reader, element, rules, names))
                continue
            if isinstance(element, Primitive) \
                    and element.universal == Universal.NULL:
                # §26.4: SEQUENCE OF NULL spells each element `<NULL/>` because its
                # XMLValue is `empty` and there would otherwise be nothing to count.
                found_kind, found = reader.read_tag()
                if found_kind != "empty" or found != element_name:
                    raise reader.error(
                        f"expected <{element_name}/> for a NULL element (X.680 26.4); "
                        f"found a {found_kind} tag for {found!r}")
                values.append(None)
                continue
            values.append(_decode_primitive(reader, element, element_name, rules, names))
            continue
        if tag[1] != element_name:
            raise reader.error(
                f"expected <{element_name}> for an element of {kind.name} "
                f"(X.680 26.10); found <{tag[1]}>")
        values.append(_decode_typed(reader, element_name, element, rules, names))


def _decode_choice(reader: _Reader, kind: Choice, rules: XerRules,
                   names: XerTypeNames | None) -> tuple:
    tag = reader.peek_tag()
    if tag is None or tag[0] == "end":
        raise reader.error(f"{kind.name}: a CHOICE value is one element (X.680 29.11)")
    for alt in _flatten(kind.alternatives):
        if alt.name == tag[1]:
            return (alt.name,
                    _decode_typed(reader, alt.name, alt.type, rules, names))
    if kind.extensible:
        # §8.6.3: an unknown extension alternative is a single unexpected element, and the
        # decoder "shall accept" it. There is no type to decode it with, so the raw XML is
        # kept — the same posture as an unresolvable open type, whose octets also stay
        # exactly as they arrived. `encode_xer` refuses such a pair rather than emitting
        # text it did not build.
        return (tag[1], reader.read_raw_element())
    raise reader.error(
        f"{kind.name}: <{tag[1]}> matches no alternative and the CHOICE carries no "
        f"extension marker (8.6.3)")


def decode_xer(data: bytes | str, kind: Asn1Type, *, name: str | None = None,
               rules: XerRules = XerRules.CANONICAL,
               names: XerTypeNames | None = None) -> object:
    """Decode a complete XER encoding of `kind` (§8.1.1).

    `rules` selects what the decoder *accepts*, not what it expects to see: CANONICAL is
    the stricter of the two, refusing an XML prolog (§9.1.1), the numeric character
    escapes (§9.1.3) and the `xmlhstring` open-type alternative (§9.12). BASIC accepts all
    of those. Every CXER encoding is a legal BASIC-XER encoding, so `XerRules.BASIC` reads
    both.
    """
    if isinstance(data, str):
        text = data
    elif isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")               # §8.1.3
        except UnicodeDecodeError as error:
            raise Asn1Error(
                f"XER: the encoding is a UTF-8 XML document (8.1.3): {error}") from None
    else:
        raise Asn1Error("XER: expected bytes or str")
    if text.startswith("﻿"):
        text = text[1:]                                      # a UTF-8 byte order mark
    reader = _Reader(text, rules)
    reader.read_prolog()
    value = _decode_typed(reader, name or xml_type_name(kind, names), kind, rules, names)
    reader.skip_space()
    if not reader.at_end():
        raise reader.error(
            f"{len(text) - reader.pos} character(s) follow the XML document element; "
            f"an XER encoding is one document element (8.1.1)")
    return value


__all__ = [
    "ASN1_NAMESPACE", "ASN1_NAMESPACE_OID",
    "BASIC_XER_OID", "CANONICAL_XER_OID", "EXTENDED_XER_OID", "XML_PROLOG",
    "XerRules", "XerTypeNames", "canonical_realnumber", "decode_xer", "encode_xer",
    "escape_xmlcstring", "rules_oid", "xml_type_name",
]
