"""JSON Encoding Rules — Rec. ITU-T X.697 (02/2021) | ISO/IEC 8825-8:2021.

JER is the rail that makes a BCIR artifact readable by tooling that will never speak
ASN.1. Like XER it is a text encoding and a poor fit for a digested artifact; unlike XER,
JSON is what the surrounding world actually reads, which is why the roadmap puts JER in the
*selection candidate set* and keeps XER out of it.

**ONE FINDING FIRST, because it changes what this module can honestly claim.**
X.697 defines **no canonical variant**. §42.2 assigns exactly one object identifier —
`{joint-iso-itu-t asn1(1) jer-encoding(7)}` — and there is no second name, no "CJER" clause,
and no canonicalization annex. What the standard *does* have is encoder's options, several
of them load-bearing:

* §27.3.3 — "The components of the sequence value may be added to the encoding in any
  order."
* §30.3.3 — the same for the items of a set-of value.
* §24.1 b) and §25.1 b) — a sender's option between two forms when a contents constraint is
  present.
* §25.3 and §24.2.1 — the hexadecimal digits are `0123456789abcdefABCDEF`, so case is free.
* §7.6.3 — "The use of any of the escapes specified in ECMA-404, clause 9, is permitted in
  any JSON string produced by these encoding rules."
* §6.3 — "Alternative encodings are permitted … as encoder's options. Decoders that claim
  conformance to JER shall support all JER encoding alternatives."

So a value has many legal JER encodings, and the repo's standing rule — *no encoding rule
ships without a canonical variant on the emit path* — cannot be satisfied by citing X.697.
`JerRules.CANONICAL` is therefore **BCIR's own profile, not a standardized one**, and it
deliberately has **no object identifier**: inventing an arc under `jer-encoding(7)` would
claim a registration that does not exist. A peer is told "JER, with the BCIR canonical
profile", never "CJER" as though ITU-T had defined it. Every option the profile pins is
listed on `JerRules` with the clause that left it open.

WHAT JER READS FROM A TYPE, and why that is the interesting contrast. PER's whole size
advantage comes from §10.3's PER-visible constraints. JER's §7.2 list is far shorter, and
§7.2.2 l) is the striking entry: **value and value range constraints on integer types are
NOT JER-visible**. `INTEGER (0..255)` and plain `INTEGER` produce identical JSON. Only four
things actually reach the encoder:

* §7.2.1 a) — a non-extensible SIZE on a **bitstring** type, which chooses between the
  fixed-size string of §24.2 and the `{"value", "length"}` object of §24.3;
* §7.2.1 b)-d) — constraints on a real type's base, which choose between §23.3's JSON
  number and §23.4's `{"base10Value": …}` object;
* §7.2.1 e) — a contents constraint with CONTAINING but without ENCODED BY, which unlocks
  the `{"containing": …}` form of §24.4 and §25.4;
* §7.2.1 f) — a contained subtype constraint whose constraining type carries one of those.

That is the whole of it. A rail that consulted a SIZE on an OCTET STRING, or a permitted
alphabet, would be reading constraints §7.2.2 h) and j) explicitly exclude.

NOT BUILT, and recorded rather than approximated: the **JER encoding instructions** of
clauses 14-19 (`ARRAY`, `BASE64`, `NAME`, `OBJECT`, `TEXT`, `UNWRAPPED`). They are the JER
analogue of XER's encoding instructions and are assigned the same way — a type prefix or an
encoding control section — and, like EXTENDED-XER, none of them changes an encoding that
does not carry one. Their absence has three visible consequences, all of them defaults the
standard itself specifies: a sequence is object-based (§27.1 with §27.3), a set-of is
array-based (§30.1 with §30.2), a choice is wrapped (§31.1 with §31.3), and an octetstring
is hexadecimal rather than Base64 (§25.1 c) with §25.3).
"""

from __future__ import annotations

import json
from enum import Enum

from .codec import Strictness
from .schema import (Asn1Type, Choice, Component, OpenType, Primitive, Sequence,
                     SequenceOf, Set, SetOf, _resolve_open_type)
from .tags import Asn1Error, Universal
from .tlv import decode_one, encode_tlv
from .values import BitString

#: §42.2 — the ONE object identifier X.697 assigns, and its object descriptor.
JER_OID: tuple[int, ...] = (2, 1, 7)
JER_OID_DESCRIPTOR: str = "JER encoding of a single ASN.1 type"


class JerRules(Enum):
    """Which profile to emit or accept.

    `BASIC` is X.697 as written. `CANONICAL` is **BCIR's** profile — X.697 registers no
    canonical variant (see the module docstring), so this name is local and carries no
    object identifier. It pins exactly the encoder's options the standard leaves open:

    * member order for a sequence or set is **definition order** (§27.3.3 allows any);
    * set-of items are sorted by their own encoding (§30.3.3 allows any);
    * a component holding its DEFAULT is **omitted** — X.697 states no rule either way, so
      this follows X.690 §11.5 and COER rather than CXER §9.5, because JER earns its place
      in the candidate set on size;
    * hexadecimal digits are upper-case (§24.2.1/§25.3 permit either case);
    * no insignificant white-space anywhere;
    * only the escapes ECMA-404 requires, never a gratuitous `\\uXXXX` (§7.6.3 permits any).

    A `CANONICAL` decode is *stricter*, not different: it refuses the alternatives it would
    not emit, so what BCIR digests is what BCIR would have produced.
    """

    BASIC = 0
    CANONICAL = 1


#: §23.2 Table 2 — the four real values that are JSON strings rather than numbers.
_SPECIAL_REALS: dict[str, float] = {
    "-0": -0.0, "-INF": float("-inf"), "INF": float("inf"), "NaN": float("nan")}

#: §38.1 — the restricted character string types whose value IS a JSON string.
_TEXT_STRINGS = frozenset({
    Universal.IA5_STRING, Universal.VISIBLE_STRING, Universal.NUMERIC_STRING,
    Universal.PRINTABLE_STRING, Universal.BMP_STRING, Universal.UNIVERSAL_STRING,
    Universal.UTF8_STRING,
})

#: §38.2 — the remaining restricted string types, "encoded as if it were an octetstring
#: value consisting of the octets specified in Rec. ITU-T X.690, 8.23.5", i.e. as hex.
_OCTET_STRINGS = frozenset({
    Universal.TELETEX_STRING, Universal.VIDEOTEX_STRING, Universal.GRAPHIC_STRING,
    Universal.GENERAL_STRING,
})

#: §40 with §7.4.5 — the time types, and the useful types that X.680 clause 45 defines in
#: terms of VisibleString. All are JSON strings holding the value notation.
_TIME_STRINGS = frozenset({
    Universal.UTC_TIME, Universal.GENERALIZED_TIME, Universal.OBJECT_DESCRIPTOR,
    Universal.TIME, Universal.DATE, Universal.TIME_OF_DAY, Universal.DATE_TIME,
    Universal.DURATION, Universal.OID_IRI, Universal.RELATIVE_OID_IRI,
})


# --- the serializer ----------------------------------------------------------------------
#
# Written out rather than delegated to `json.dumps`, for the same reason `xer.py` writes its
# own reader: every encoder's option in this file is a decision the standard left open, and
# a general serializer makes those decisions somewhere this module cannot see them. Number
# formatting is the sharp case -- §21 forbids a fractional part, an exponent and a
# superfluous leading zero, and `repr` of a Python float supplies all three.

def _string(value: str) -> str:
    """A JSON string (ECMA-404 clause 9), with only the escapes it requires.

    §7.6.3 permits any of ECMA-404's escapes, so a conforming encoder may spell every
    character as `\\uXXXX`. Emitting the minimum is what makes the output stable, and is one
    of the options `JerRules.CANONICAL` pins.
    """
    out = ['"']
    for character in value:
        code = ord(character)
        if character == '"':
            out.append('\\"')
        elif character == "\\":
            out.append("\\\\")
        elif character == "\b":
            out.append("\\b")
        elif character == "\f":
            out.append("\\f")
        elif character == "\n":
            out.append("\\n")
        elif character == "\r":
            out.append("\\r")
        elif character == "\t":
            out.append("\\t")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        elif 0xD800 <= code <= 0xDFFF:
            # A lone surrogate is not a Unicode scalar and cannot be encoded in the UTF-8
            # §7.6.2 requires. Refusing beats emitting an escape that no decoder can turn
            # back into a character.
            raise Asn1Error(
                f"JER: U+{code:04X} is an unpaired surrogate and has no UTF-8 encoding "
                f"(7.6.2)")
        else:
            out.append(character)
    out.append('"')
    return "".join(out)


def _integer(value: int, where: str) -> str:
    """§21 — "a JSON number denoting the value, with no fractional part and no exponent"."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise Asn1Error(f"{where}: expected int, got {type(value).__name__}")
    # `str` of a Python int never produces a leading zero, a "+", or "-0", which is exactly
    # what §21's NOTE forbids.
    return str(value)


def _hex(octets: bytes) -> str:
    """§24.2.1/§25.3 — an even number of hexadecimal digits. Case is an encoder's option."""
    return octets.hex().upper()


# --- encoding ----------------------------------------------------------------------------

def _encode(kind: Asn1Type, value, rules: JerRules, context: dict | None = None) -> str:
    if isinstance(kind, Primitive):
        return _encode_primitive(kind, value, rules)
    if isinstance(kind, (Sequence, Set)):
        return _encode_components(kind, value, rules)
    if isinstance(kind, SequenceOf):
        return _encode_list(kind, value, rules, sort=False)
    if isinstance(kind, SetOf):
        # §30.2: "in any order". CANONICAL sorts so one abstract value has one encoding.
        return _encode_list(kind, value, rules, sort=rules is JerRules.CANONICAL)
    if isinstance(kind, Choice):
        return _encode_choice(kind, value, rules)
    if isinstance(kind, OpenType):
        return _encode_open_type(kind, value, rules, context)
    raise Asn1Error(f"JER: no encoding for schema type {type(kind).__name__}")


def _encode_primitive(kind: Primitive, value, rules: JerRules) -> str:
    universal = kind.universal

    if universal == Universal.BOOLEAN:                       # §20
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool, got {type(value).__name__}")
        return "true" if value else "false"

    if universal == Universal.INTEGER:                       # §21
        return _integer(value, kind.name)

    if universal == Universal.ENUMERATED:                    # §22.1: a JSON *string*
        return _string(_enumeration_identifier(kind, value))

    if universal == Universal.REAL:                          # §23
        return _encode_real(kind, value)

    if universal == Universal.NULL:                          # §26
        if value is not None:
            raise Asn1Error(f"{kind.name}: a NULL value is None, got {value!r}")
        return "null"

    if universal == Universal.BIT_STRING:                    # §24
        return _encode_bitstring(kind, value, rules)

    if universal == Universal.OCTET_STRING:                  # §25
        return _encode_octetstring(kind, value, rules)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        return _string(_oid_text(kind, value))               # §32, §33

    if universal in _TEXT_STRINGS:                           # §38.1
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        return _string(value)

    if universal in _OCTET_STRINGS:                          # §38.2
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
        return _string(_hex(bytes(value)))

    if universal in _TIME_STRINGS:                           # §40 with §7.4.5
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        return _string(value)

    raise Asn1Error(f"JER: no encoding for UNIVERSAL {int(universal)} in this rail")


def _enumeration_identifier(kind: Primitive, value) -> str:
    """§22.2 — the JSON string denotes "the identifier of the chosen enumeration item"."""
    if not kind.enumeration:
        raise Asn1Error(
            f"{kind.name}: ENUMERATED has no enumeration; JER encodes the identifier "
            f"(22.2), which -- unlike BER's value (X.690 8.4) -- cannot be derived from "
            f"the number alone")
    if isinstance(value, str):
        if value not in {name for name, _n in kind.enumeration}:
            raise Asn1Error(f"{kind.name}: {value!r} is not an enumeration identifier")
        return value
    for name, number in kind.enumeration:
        if number == value:
            return name
    raise Asn1Error(
        f"{kind.name}: {value} names no enumeration item, and 22.1 gives an enumerated "
        f"value no numeric spelling")


def _encode_real(kind: Primitive, value) -> str:
    """§23. A Python float is a base-2 value, which §23.1.2 sends straight to §23.3.

    §23.1.3-§23.1.5 route a base-10 value to §23.4's `{"base10Value": …}` object depending
    on the effective value constraint of the *base* — a constraint this rail cannot express,
    since `Primitive` carries no `WITH COMPONENTS { base (10) }` inner type constraint. The
    branch is therefore unreachable rather than wrong, and is documented instead of faked.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Asn1Error(f"{kind.name}: expected a number, got {type(value).__name__}")
    number = float(value)
    if number != number:                                     # §23.1.1 with Table 2
        return _string("NaN")
    if number == float("inf"):
        return _string("INF")
    if number == float("-inf"):
        return _string("-INF")
    if number == 0.0:
        # Table 2 spells minus zero as a string; plus zero is §23.1.2's "the real value
        # is 0", which is a JSON number.
        import math
        return _string("-0") if math.copysign(1.0, number) < 0 else "0"
    # §23.3: "a JSON number denoting the value". `repr` gives the shortest round-tripping
    # decimal; JSON has no `inf`/`nan` literals, both already handled above.
    return repr(number)


def _encode_bitstring(kind: Primitive, value, rules: JerRules) -> str:
    """§24. The one place a SIZE constraint reaches a JER encoder (§7.2.1 a))."""
    if kind.contains is not None and kind.encoded_by is None:  # §24.4 with §7.2.1 e)
        if not isinstance(value, BitString):
            raise Asn1Error(f"{kind.name}: expected BitString, got "
                            f"{type(value).__name__}")
        if value.unused:
            raise Asn1Error(
                f"{kind.name}: a CONTAINING bitstring's bits are a complete encoding "
                f"(X.682 11.4), so it cannot end {value.unused} bits into an octet")
        inner = kind.contains.decode(decode_one(bytes(value.octets)),
                                     strictness=Strictness.BER)
        return "{" + _string("containing") + ":" + _encode(kind.contains, inner, rules) + "}"
    if not isinstance(value, BitString):
        raise Asn1Error(f"{kind.name}: expected BitString, got {type(value).__name__}")
    low, high = _bitstring_size(kind)
    if low is not None and low == high:                      # §24.1 a) -> §24.2
        if value.bit_length != low:
            raise Asn1Error(
                f"{kind.name}: the effective size constraint fixes the length at {low} "
                f"bits, got {value.bit_length} (24.2)")
        return _string(_hex(bytes(value.octets)))
    # §24.3: a JSON object carrying the octets and the true bit length.
    return ("{" + _string("value") + ":" + _string(_hex(bytes(value.octets))) + ","
            + _string("length") + ":" + _integer(value.bit_length, kind.name) + "}")


def _bitstring_size(kind: Primitive) -> tuple[int | None, int | None]:
    """§7.2.8 with §7.2.1 a) — the effective size constraint, extensible ones excluded.

    §7.2.2 g) makes an extensible subtype constraint NOT JER-visible, so `BIT STRING
    (SIZE (10), ...)` is a variable-size bitstring for JER even though its root is a single
    size. That is the difference between Annex A.4's `MyBitString1` and `MyBitString2`.
    """
    from .constraints import root_size_bounds

    (low, high), extensible = root_size_bounds(kind.constraint)
    if extensible:
        return (None, None)
    return (low, high)


def _encode_octetstring(kind: Primitive, value, rules: JerRules) -> str:
    """§25. Note what is absent: no SIZE is consulted, because §7.2.2 h) excludes it."""
    if not isinstance(value, (bytes, bytearray)):
        raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
    if kind.contains is not None and kind.encoded_by is None:  # §25.4 with §7.2.1 e)
        inner = kind.contains.decode(decode_one(bytes(value)), strictness=Strictness.BER)
        return "{" + _string("containing") + ":" + _encode(kind.contains, inner, rules) + "}"
    return _string(_hex(bytes(value)))                       # §25.3


def _oid_text(kind: Primitive, value) -> str:
    """§32/§33 — the `XMLObjectIdentifierValue` production, i.e. dot-separated numbers."""
    if isinstance(value, str):
        arcs = tuple(int(part) for part in value.split(".") if part != "")
    elif isinstance(value, (tuple, list)):
        arcs = tuple(value)
    else:
        raise Asn1Error(f"{kind.name}: expected a tuple of arcs, got "
                        f"{type(value).__name__}")
    if not arcs:
        raise Asn1Error(f"{kind.name}: an object identifier has at least one arc")
    for arc in arcs:
        if isinstance(arc, bool) or not isinstance(arc, int) or arc < 0:
            raise Asn1Error(f"{kind.name}: {arc!r} is not an arc number")
    return ".".join(str(arc) for arc in arcs)


def _flatten(components: tuple[Component, ...]) -> tuple[Component, ...]:
    """X.680 §25.1 version brackets are transparent to JER: each member is its own JSON
    member, exactly as each is its own XML element."""
    out: list[Component] = []
    for comp in components:
        if comp.group is not None:
            out.extend(comp.group)
        else:
            out.append(comp)
    return tuple(out)


def _encode_components(kind, value, rules: JerRules) -> str:
    """§27.3 for a sequence, and §29 for a set — "encoded as if the type had been declared
    a sequence type"."""
    if not isinstance(value, dict):
        raise Asn1Error(f"{kind.name}: expected a dict, got {type(value).__name__}")
    components = _flatten(kind.components)
    known = {comp.name for comp in components}
    unknown = {n for n in value if n not in known and not n.endswith(".resolved")}
    if unknown:
        raise Asn1Error(f"{kind.name}: unknown component(s) {sorted(unknown)}")
    members: list[str] = []
    for comp in components:
        if comp.name in value:
            item = value[comp.name]
            # X.697 states no rule for a DEFAULT-valued component. The canonical profile
            # omits it -- see `JerRules` -- which is the X.690 §11.5 answer, not CXER's.
            if comp.has_default and item == comp.default \
                    and rules is JerRules.CANONICAL:
                continue
        elif comp.has_default or comp.optional:
            continue
        else:
            raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
        # §27.3.2 a): the member name is the component's identifier (a NAME encoding
        # instruction could change it; clause 16 is not implemented).
        members.append(_string(comp.name) + ":" + _encode(comp.type, item, rules, value))
    return "{" + ",".join(members) + "}"


def _encode_list(kind, value, rules: JerRules, *, sort: bool) -> str:
    """§28 for a sequence-of (order preserved) and §30.2 for a set-of (order free)."""
    if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
        raise Asn1Error(
            f"{kind.name}: expected a sequence of elements, got {type(value).__name__}")
    items = [_encode(kind.element, item, rules) for item in value]
    if sort:
        items.sort()
    return "[" + ",".join(items) + "]"


def _encode_choice(kind: Choice, value, rules: JerRules) -> str:
    """§31.3 — a JSON object with exactly one member, named for the chosen alternative."""
    if not (isinstance(value, tuple) and len(value) == 2):
        raise Asn1Error(
            f"{kind.name}: value must be an (alternative, value) pair, got "
            f"{type(value).__name__}")
    chosen, payload = value
    for alt in _flatten(kind.alternatives):
        if alt.name == chosen:
            return "{" + _string(alt.name) + ":" + _encode(alt.type, payload, rules) + "}"
    raise Asn1Error(f"{kind.name}: {chosen!r} is not an alternative")


def _governing_context(kind: OpenType, siblings: dict | None) -> dict:
    if not siblings:
        return {}
    return {path: siblings[path[-1]] for path in kind.governing if path[-1] in siblings}


def _encode_open_type(kind: OpenType, value, rules: JerRules,
                      context: dict | None) -> str:
    """§41 — "The encoding of an open type value shall be the encoding of the value of the
    contained type."

    JER offers no hexadecimal fallback the way XER's §8.5 does, so an open type is
    encodable exactly when its X.682 §10.19 table resolves it. Refusing otherwise is the
    only honest option: there is no spelling for "some octets whose type I do not know".
    """
    if not isinstance(value, (bytes, bytearray)):
        raise Asn1Error(
            f"{kind.name}: an open type value is the contained value's complete encoding, "
            f"so it must be bytes, not {type(value).__name__}")
    contained = kind.resolve(_governing_context(kind, context))
    if contained is None:
        raise Asn1Error(
            f"JER: 41 encodes an open type AS its contained type, and {kind.name} could "
            f"not be resolved by its table (X.682 10.19); JER has no hexadecimal "
            f"alternative to fall back to")
    inner = contained.decode(decode_one(bytes(value)), strictness=Strictness.BER)
    return _encode(contained, inner, rules)


def encode_jer(kind: Asn1Type, value, *, rules: JerRules = JerRules.CANONICAL) -> bytes:
    """Encode `value` as a complete JER encoding of `kind` (§7.6.2).

    The result is UTF-8 octets: §7.6.2 says the JSON tokens "shall be encoded in UTF-8 into
    an octet string, which is the complete encoding of the abstract value of the outermost
    type" — the encoding is the octets, not the character string.
    """
    return _encode(kind, value, rules).encode("utf-8")


# --- decoding ----------------------------------------------------------------------------

class _Raw:
    """A JSON number kept as its source lexeme.

    `json` hands back an `int` or a `float`, and both lose what §21 needs: whether the text
    had a fractional part or an exponent, which §21 forbids for an integer. Keeping the
    lexeme is what lets `1.0` be refused where `1` is accepted.

    Deliberately NOT a `str` subclass. It would be the convenient spelling, and it would
    silently make every "is this a JSON string?" check in this decoder accept a JSON number
    -- so `3.5` would satisfy §23.2's Table 2 lookup, §38.1's character string and §32's
    object identifier alike. The type distinction between a JSON number and a JSON string
    is load-bearing throughout clause 20-41, so it is kept in the Python types too.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


def _pairs(pairs):
    """ECMA-404 permits duplicate member names; a rail that digests what it decodes cannot.

    `json`'s default keeps the last occurrence silently, which would let a peer hide one
    value behind another in an encoding that hashes differently but decodes identically.
    """
    seen = set()
    for name, _value in pairs:
        if name in seen:
            raise Asn1Error(f"JER: member {name!r} appears more than once in one object")
        seen.add(name)
    return dict(pairs)


def _constant(name: str):
    """`json` accepts the non-standard `NaN`, `Infinity` and `-Infinity` literals.

    None of them is JSON (ECMA-404 clause 8 has no such token), and §23.2 spells those
    values as the *strings* "NaN", "INF" and "-INF". Accepting the literals would admit an
    encoding no conforming JER encoder can produce.
    """
    raise Asn1Error(
        f"JER: {name!r} is not a JSON token (ECMA-404 8); 23.2 spells the special real "
        f"values as JSON strings")


def _parse(text: str):
    try:
        return json.loads(text, parse_int=_Raw, parse_float=_Raw,
                          parse_constant=_constant, object_pairs_hook=_pairs)
    except json.JSONDecodeError as error:
        raise Asn1Error(f"JER: not a JSON text (ECMA-404): {error}") from None


def _decode(node, kind: Asn1Type, rules: JerRules, context: dict | None = None):
    if isinstance(kind, Primitive):
        return _decode_primitive(node, kind, rules)
    if isinstance(kind, (Sequence, Set)):
        return _decode_components(node, kind, rules)
    if isinstance(kind, (SequenceOf, SetOf)):
        return _decode_list(node, kind, rules)
    if isinstance(kind, Choice):
        return _decode_choice(node, kind, rules)
    if isinstance(kind, OpenType):
        return _decode_open_type(node, kind, rules, context)
    raise Asn1Error(f"JER: no decoding for schema type {type(kind).__name__}")


def _want(node, want: type, kind, what: str, clause: str):
    if not isinstance(node, want) or (want is not bool and isinstance(node, bool)):
        raise Asn1Error(
            f"{getattr(kind, 'name', kind)}: expected {what} ({clause}), got "
            f"{_describe(node)}")
    return node


def _describe(node) -> str:
    if node is None:
        return "the JSON token null"
    if isinstance(node, bool):
        return f"the JSON token {'true' if node else 'false'}"
    if isinstance(node, _Raw):
        return f"the JSON number {node.text}"
    if isinstance(node, str):
        return "a JSON string"
    if isinstance(node, list):
        return "a JSON array"
    return "a JSON object"


def _decode_integer(node, kind) -> int:
    """§21 — no fractional part, no exponent, no superfluous leading zeros."""
    text = _want(node, _Raw, kind, "a JSON number", "21").text
    if "." in text or "e" in text or "E" in text:
        raise Asn1Error(
            f"{kind.name}: 21 requires an integer to be a JSON number \"with no "
            f"fractional part and no exponent\"; got {text}")
    return int(text)


def _decode_primitive(node, kind: Primitive, rules: JerRules):
    universal = kind.universal

    if universal == Universal.BOOLEAN:                       # §20
        return _want(node, bool, kind, "the JSON token true or false", "20")

    if universal == Universal.INTEGER:                       # §21
        return _decode_integer(node, kind)

    if universal == Universal.ENUMERATED:                    # §22
        name = _want(node, str, kind, "a JSON string", "22.1")
        if not kind.enumeration:
            raise Asn1Error(
                f"{kind.name}: ENUMERATED has no enumeration, so {name!r} cannot be "
                f"mapped to a value (22.2)")
        for item, number in kind.enumeration:
            if item == name:
                return number
        raise Asn1Error(f"{kind.name}: {name!r} is not an enumeration identifier")

    if universal == Universal.REAL:                          # §23
        if isinstance(node, str):
            if node not in _SPECIAL_REALS:
                raise Asn1Error(
                    f"{kind.name}: {node!r} is not one of Table 2's special real values")
            return _SPECIAL_REALS[node]
        if isinstance(node, dict):                           # §23.4
            if list(node) != ["base10Value"]:
                raise Asn1Error(
                    f"{kind.name}: 23.4 gives a real object exactly one member named "
                    f"\"base10Value\"; got {sorted(node)}")
            return float(_want(node["base10Value"], _Raw, kind,
                               "a JSON number", "23.4").text)
        return float(_want(node, _Raw, kind, "a JSON number", "23.3").text)

    if universal == Universal.NULL:                          # §26
        if node is not None:
            raise Asn1Error(f"{kind.name}: expected the JSON token null (26), got "
                            f"{_describe(node)}")
        return None

    if universal == Universal.BIT_STRING:                    # §24
        return _decode_bitstring(node, kind, rules)

    if universal == Universal.OCTET_STRING:                  # §25
        if isinstance(node, dict) and kind.contains is not None:
            return _decode_containing(node, kind, rules)     # §25.4
        return _unhex(_want(node, str, kind, "a JSON string", "25.3"), kind)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        text = _want(node, str, kind, "a JSON string", "32")
        parts = text.split(".")
        if not text or any(not part.isdigit() for part in parts):
            raise Asn1Error(
                f"{kind.name}: {text!r} is not an XMLObjectIdentifierValue (32)")
        return tuple(int(part) for part in parts)

    if universal in _TEXT_STRINGS:                           # §38.1
        return _want(node, str, kind, "a JSON string", "38.1")

    if universal in _OCTET_STRINGS:                          # §38.2
        return _unhex(_want(node, str, kind, "a JSON string", "38.2"), kind)

    if universal in _TIME_STRINGS:                           # §40
        return _want(node, str, kind, "a JSON string", "40")

    raise Asn1Error(f"JER: no decoding for UNIVERSAL {int(universal)} in this rail")


def _unhex(text: str, kind) -> bytes:
    if len(text) % 2:
        raise Asn1Error(
            f"{kind.name}: a hexadecimal JSON string has an even number of digits "
            f"(25.3); got {len(text)}")
    try:
        return bytes.fromhex(text)
    except ValueError:
        raise Asn1Error(f"{kind.name}: {text!r} is not hexadecimal (25.3)") from None


def _decode_containing(node: dict, kind: Primitive, rules: JerRules) -> bytes:
    """§24.4/§25.4 — `{"containing": <JER of the contained value>}`.

    The model holds such a value as the contained value's octets, so decoding the JSON form
    means re-encoding what it denotes. Sound because those octets are canonical (DER) by
    construction, and the JSON -- not some earlier octet string -- is what the peer sent.
    """
    if list(node) != ["containing"]:
        raise Asn1Error(
            f"{kind.name}: 25.4 gives a contents-constrained value exactly one member "
            f"named \"containing\"; got {sorted(node)}")
    inner = _decode(node["containing"], kind.contains, rules)
    return encode_tlv(kind.contains.encode(inner))


def _decode_bitstring(node, kind: Primitive, rules: JerRules) -> BitString:
    if isinstance(node, dict) and kind.contains is not None and "containing" in node:
        return BitString(_decode_containing(node, kind, rules), 0)  # §24.4
    low, high = _bitstring_size(kind)
    if isinstance(node, dict):                               # §24.3
        if sorted(node) != ["length", "value"]:
            raise Asn1Error(
                f"{kind.name}: 24.3 gives a variable-size bitstring the members "
                f"\"value\" and \"length\"; got {sorted(node)}")
        octets = _unhex(_want(node["value"], str, kind, "a JSON string", "24.3"), kind)
        bits = _decode_integer(node["length"], kind)
        if not 0 <= bits <= len(octets) * 8 or (len(octets) * 8 - bits) >= 8:
            raise Asn1Error(
                f"{kind.name}: \"length\" of {bits} bits does not match "
                f"{len(octets)} octet(s) of \"value\" (24.3)")
        return BitString(octets, len(octets) * 8 - bits)
    text = _want(node, str, kind, "a JSON string", "24.2")   # §24.2
    if low is None or low != high:
        raise Asn1Error(
            f"{kind.name}: 24.1 c) gives a variable-size bitstring the 24.3 object form, "
            f"not a bare JSON string")
    octets = _unhex(text, kind)
    if not low <= len(octets) * 8 < low + 8:
        raise Asn1Error(
            f"{kind.name}: the effective size constraint fixes the length at {low} bits, "
            f"which {len(octets)} octet(s) cannot carry (24.2)")
    return BitString(octets, len(octets) * 8 - low)


def _decode_components(node, kind, rules: JerRules) -> dict:
    """§27.3 — "The components of the sequence value may be added to the encoding in any
    order", so the decoder matches by name and never by position."""
    members = _want(node, dict, kind, "a JSON object", "27.3.1")
    components = _flatten(kind.components)
    by_name = {comp.name: comp for comp in components}
    out: dict = {}
    for name, item in members.items():
        comp = by_name.get(name)
        if comp is None:
            if kind.extensible:
                # An addition from a newer version. JSON is self-delimiting, so an unknown
                # member is skippable in a way PER needs §19.9's open-type wrapper for.
                continue
            raise Asn1Error(
                f"{kind.name}: member {name!r} matches no component, and the type carries "
                f"no extension marker")
        out[comp.name] = _decode(item, comp.type, rules, out)
        _resolve_open_type(comp, out, Strictness.BER)
    if rules is JerRules.CANONICAL:
        order = [name for name in members if name in by_name]
        if order != [c.name for c in components if c.name in set(order)]:
            raise Asn1Error(
                f"{kind.name}: the BCIR canonical profile fixes member order at the "
                f"component order of the type; 27.3.3 leaves it free, so this is a "
                f"legal JER encoding that is not a canonical one")
    for comp in components:
        if comp.name in out:
            continue
        if comp.has_default:
            out[comp.name] = comp.default                    # X.680 §25.12
        elif not comp.optional:
            raise Asn1Error(
                f"{kind.name}: mandatory component {comp.name!r} is missing")
    return out


def _decode_list(node, kind, rules: JerRules) -> list:
    items = _want(node, list, kind, "a JSON array", "28")
    return [_decode(item, kind.element, rules) for item in items]


def _decode_choice(node, kind: Choice, rules: JerRules) -> tuple:
    """§31.3.1 — "a JSON object having exactly one member"."""
    members = _want(node, dict, kind, "a JSON object", "31.3.1")
    if len(members) != 1:
        raise Asn1Error(
            f"{kind.name}: 31.3.1 gives a choice value exactly one member; got "
            f"{len(members)}")
    (name, item), = members.items()
    for alt in _flatten(kind.alternatives):
        if alt.name == name:
            return (alt.name, _decode(item, alt.type, rules))
    raise Asn1Error(f"{kind.name}: {name!r} matches no alternative")


def _decode_open_type(node, kind: OpenType, rules: JerRules,
                      context: dict | None) -> bytes:
    contained = kind.resolve(_governing_context(kind, context))
    if contained is None:
        raise Asn1Error(
            f"JER: 41 encodes an open type AS its contained type, and {kind.name} could "
            f"not be resolved by its table (X.682 10.19)")
    return encode_tlv(contained.encode(_decode(node, contained, rules)))


def decode_jer(data: bytes | str, kind: Asn1Type, *,
               rules: JerRules = JerRules.CANONICAL) -> object:
    """Decode a complete JER encoding of `kind`.

    `rules` selects what the decoder *accepts*. `CANONICAL` is the stricter of the two: it
    refuses the encoder's options the BCIR profile does not emit, so what is digested is
    what BCIR would have produced. `BASIC` implements §6.3 — "Decoders that claim
    conformance to JER shall support all JER encoding alternatives".
    """
    if isinstance(data, str):
        text = data
    elif isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")               # §7.6.2
        except UnicodeDecodeError as error:
            raise Asn1Error(f"JER: the encoding is UTF-8 (7.6.2): {error}") from None
    else:
        raise Asn1Error("JER: expected bytes or str")
    return _decode(_parse(text), kind, rules)


__all__ = [
    "JER_OID", "JER_OID_DESCRIPTOR", "JerRules", "decode_jer", "encode_jer",
]
