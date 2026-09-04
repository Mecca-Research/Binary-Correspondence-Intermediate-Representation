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

THE ENCODING INSTRUCTIONS of clauses 14-19 — `ARRAY`, `BASE64`, `NAME`, `OBJECT`, `TEXT`
and `UNWRAPPED` — are built, and they are what makes JER a *schema* notation rather than a
fixed projection. Each one changes the shape of the JSON rather than its content, which is
exactly what an application needs when the JSON has to match a shape someone else already
published. They are carried in a `JerInstructions` side table rather than on the type model,
because five rails share `schema.py` and only this one has instructions.

`UNWRAPPED` is the interesting one. §31.2 drops the wrapping object, so nothing in the
encoding says which alternative was chosen and, as its NOTE puts it, the form "relies on the
decoder's ability to identify the alternative that was encoded by examining the JER encoding
of the alternative". §19.2.2-§19.2.4 are precisely the conditions that make that possible —
at most one alternative per JSON value kind, and object-producing alternatives separated by a
mandatory member name. So the restrictions and the decoder are two halves of one mechanism,
and `_json_kinds` computes both.

What is NOT built is the *assignment syntax*: the type prefixes of clause 10 and the JER
encoding control section of clause 11, which are X.680 surface (§31.3 and clause 54 there).
Instructions are assigned through `JerInstructions.assign`, which applies clause 13's
precedence — §13.2's negating removal and §13.3's one-per-category replacement — so the
semantics are the standard's even where the notation is not.
"""

from __future__ import annotations

import base64 as _base64
import json
from dataclasses import dataclass
from enum import Enum

from .codec import Strictness
from .schema import (
    Asn1Type,
    Choice,
    Component,
    OpenType,
    Primitive,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
    _resolve_open_type,
)
from .tags import Asn1Error, Universal
from .tlv import decode_one, encode_tlv
from .values import BitString, is_number_form

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


# --- clauses 14-19: the JER encoding instructions ----------------------------------------


class NameKeyword(Enum):
    """§16.1.1's `Keyword` — a case change applied to an identifier instead of a new name."""

    CAPITALIZED = "CAPITALIZED"
    UPPERCAMELCASED = "UPPERCAMELCASED"
    UPPERCASED = "UPPERCASED"
    LOWERCAMELCASED = "LOWERCAMELCASED"
    LOWERCASED = "LOWERCASED"


def apply_name_keyword(identifier: str, keyword: NameKeyword) -> str:
    """§16.1.5.1-§16.1.5.5 — the five case operations, exactly as written.

    Each clause is narrower than its name suggests and the differences matter.
    `UPPERCASED` touches "all characters of the identifier that are lower-case letters …
    Other characters are unchanged", so it leaves hyphens and digits alone, while
    `UPPERCAMELCASED` removes hyphens. `CAPITALIZED` changes exactly one character.
    """
    if keyword is NameKeyword.CAPITALIZED:  # §16.1.5.1
        return identifier[:1].upper() + identifier[1:]
    if keyword is NameKeyword.UPPERCASED:  # §16.1.5.2
        return "".join(c.upper() if c.islower() else c for c in identifier)
    if keyword is NameKeyword.LOWERCASED:  # §16.1.5.4
        return "".join(c.lower() if c.isupper() else c for c in identifier)
    # §16.1.5.3 and §16.1.5.5 differ only in whether the first character is raised.
    out: list[str] = []
    raise_next = keyword is NameKeyword.UPPERCAMELCASED
    for character in identifier:
        if character == "-":  # c) / b): hyphens removed
            raise_next = True
            continue
        out.append(character.upper() if raise_next and character.islower() else character)
        raise_next = False
    return "".join(out)


@dataclass(frozen=True)
class Array:
    """§14 — a sequence type encodes as a JSON array instead of a JSON object."""


@dataclass(frozen=True)
class Base64:
    """§15 — an octetstring encodes as Base64 (RFC 2045 §6.8) instead of hexadecimal."""


@dataclass(frozen=True)
class Name:
    """§16 — replaces the JSON member name of a sequence, set or choice component."""

    new: "str | NameKeyword"

    def applied(self, identifier: str) -> str:
        if isinstance(self.new, NameKeyword):
            return apply_name_keyword(identifier, self.new)
        return self.new


@dataclass(frozen=True)
class ObjectAs:
    """§17 — a set-of of two-component sequences encodes as a JSON object (a "map").

    Named `ObjectAs` rather than `Object` so nothing in this module shadows the builtin.
    """


@dataclass(frozen=True)
class Text:
    """§18 — replaces the JSON strings that identify an enumerated type's items."""

    changes: tuple[tuple[str, "str | NameKeyword"], ...]

    def applied(self, identifier: str) -> str:
        for name, replacement in self.changes:
            if name == identifier:
                return (
                    apply_name_keyword(identifier, replacement)
                    if isinstance(replacement, NameKeyword)
                    else replacement
                )
        for name, replacement in self.changes:
            # §18.1.5: ALL "applies to all the enumeration items whose identifiers do not
            # appear in this TEXT encoding instruction", so it is consulted second.
            if name == "ALL":
                return apply_name_keyword(identifier, replacement)
        return identifier


@dataclass(frozen=True)
class Unwrapped:
    """§19 — a choice encodes as its chosen alternative alone, with no wrapping object."""


@dataclass(frozen=True)
class Not:
    """§9.1's `NegatingInstruction`.

    §13.2's NOTE 2 is the reason this holds a *category* rather than an instruction: "When
    an `Instruction` occurs as part of a `NegatingInstruction`, the `Instruction` consists
    only of a keyword (for example, `NAME` rather than `NAME AS "a"`)."
    """

    category: type


#: §9.7 — "the category of each encoding instruction is denoted by the name of the
#: corresponding production", so the Python class *is* the category.
_CATEGORIES = (Array, Base64, Name, ObjectAs, Text, Unwrapped)


class JerInstructions:
    """The final encoding instructions of clauses 14-19, keyed by identity.

    Kept beside the type model rather than inside it: five rails share `schema.py` and only
    JER has encoding instructions, so a `jer_instructions` field on `Primitive` would be
    dead weight in four of them.

    Identity keying is not a shortcut, it models §9.9. A `typereference` inherits the final
    instructions of the type assigned to it, and the lowerer hands out one object per
    assigned name — so an instruction attached to that object reaches every reference,
    which is what inheritance means. The exception §9.9 carves out is `NAME`, which is *not*
    inherited; that is why `NAME` is assigned against the **`Component`** rather than
    against the component's type, so two components sharing one referenced type can carry
    different member names.
    """

    __slots__ = ("_by_id", "_keep")

    def __init__(self) -> None:
        self._by_id: dict[int, dict[type, object]] = {}
        self._keep: list[object] = []

    def assign(self, target, *instructions) -> "JerInstructions":
        """Clause 13 — assign one or more instructions to a type or component.

        §13.3.1/§13.3.2: a positive instruction is added if its category is absent, and
        otherwise *replaces* the one already there — which is how §9.8's "never more than
        one associated instruction of a given category" holds however many assignments
        happen. §13.2: a negating instruction removes the instruction of that category and
        never becomes part of the set itself.
        """
        current = self._by_id.setdefault(id(target), {})
        if not any(target is kept for kept in self._keep):
            self._keep.append(target)
        for instruction in instructions:
            if isinstance(instruction, Not):
                current.pop(instruction.category, None)  # §13.2
                continue
            category = type(instruction)
            if category not in _CATEGORIES:
                raise Asn1Error(
                    f"JER: {category.__name__} is not one of the Table 1 encoding instructions"
                )
            _check_restrictions(target, instruction)
            current[category] = instruction  # §13.3.1/§13.3.2
        return self

    def get(self, target, category: type):
        return self._by_id.get(id(target), {}).get(category)

    def has(self, target, category: type) -> bool:
        return self.get(target, category) is not None


def _instruction(opts: "_Opts", target, category: type):
    """The final instruction of `category` on `target`, or None when there is none."""
    if opts.instructions is None:
        return None
    return opts.instructions.get(target, category)


def _check_restrictions(target, instruction) -> None:
    """§14.2, §15.2, §16.2, §17.2, §18.2 and §19.2 — checked when the instruction is
    assigned, which is the earliest this rail can.

    §6.6 makes a specification that violates these "not in conformity … even if (without
    the encoding instructions) it would conform to all of the requirements of Rec. ITU-T
    X.680", so a violation is a defect in the schema rather than in any value.
    """
    if isinstance(instruction, Array):  # §14.2
        if not isinstance(target, Sequence):
            raise Asn1Error(
                f"JER: 14.2 restricts ARRAY to a sequence type; got {type(target).__name__}"
            )
        for comp in _flatten(target.components):
            loose = comp.optional or comp.has_default or comp.extension
            if loose and isinstance(comp.type, OpenType):
                raise Asn1Error(
                    f"JER: 14.2 forbids an open type as an optional or extension "
                    f"component of an ARRAY sequence; {comp.name!r} is one"
                )
            if loose and isinstance(comp.type, Primitive) and comp.type.universal == Universal.NULL:
                raise Asn1Error(
                    f"JER: 14.2 forbids a component that produces the JSON token null as "
                    f"an optional or extension component of an ARRAY sequence; "
                    f"{comp.name!r} is a NULL"
                )
    elif isinstance(instruction, Base64):  # §15.2
        if not (isinstance(target, Primitive) and target.universal == Universal.OCTET_STRING):
            raise Asn1Error("JER: 15.2 restricts BASE64 to an octetstring type")
    elif isinstance(instruction, ObjectAs):  # §17.2
        _check_object_restriction(target)
    elif isinstance(instruction, Text):  # §18.2
        _check_text_restriction(target, instruction)
    elif isinstance(instruction, Unwrapped):  # §19.2
        if not isinstance(target, Choice):
            raise Asn1Error(
                f"JER: 19.2.1 restricts UNWRAPPED to a choice type; got {type(target).__name__}"
            )


#: §17.2 — the types the "key" component of an OBJECT map may be.
_OBJECT_KEY_STRINGS = frozenset(
    {
        Universal.IA5_STRING,
        Universal.VISIBLE_STRING,
        Universal.NUMERIC_STRING,
        Universal.PRINTABLE_STRING,
        Universal.BMP_STRING,
        Universal.UNIVERSAL_STRING,
        Universal.UTF8_STRING,
        Universal.ENUMERATED,
    }
)


def _check_object_restriction(target) -> None:
    """§17.2 — every clause of it, because an OBJECT map is only unambiguous if all hold."""
    if not isinstance(target, SetOf):
        raise Asn1Error(f"JER: 17.2 restricts OBJECT to a set-of type; got {type(target).__name__}")
    element = target.element
    if not isinstance(element, Sequence):
        raise Asn1Error(
            "JER: 17.2 requires the component of an OBJECT set-of to be a sequence type"
        )
    if element.extensible:
        raise Asn1Error('JER: 17.2 requires that sequence type to be "without an extension marker"')
    components = _flatten(element.components)
    if len(components) != 2:
        raise Asn1Error(
            f"JER: 17.2 requires exactly two components (a key and a value); got {len(components)}"
        )
    if any(c.optional or c.has_default for c in components):
        raise Asn1Error("JER: 17.2 forbids either component being OPTIONAL or DEFAULT")
    key = components[0].type
    if not (isinstance(key, Primitive) and key.universal in _OBJECT_KEY_STRINGS):
        raise Asn1Error(
            "JER: 17.2 restricts the first component of an OBJECT map to one of the "
            "eight string types or an enumerated type -- it becomes a JSON member name, "
            "which ECMA-404 6 requires to be a JSON string"
        )


def _check_text_restriction(target, instruction: Text) -> None:
    """§18.2.1-§18.2.3."""
    if not (isinstance(target, Primitive) and target.universal == Universal.ENUMERATED):
        raise Asn1Error("JER: 18.2.1 restricts TEXT to an enumerated type")
    names = [name for name, _new in instruction.changes]
    duplicate = {name for name in names if names.count(name) > 1}
    if duplicate:  # §18.2.2
        raise Asn1Error(
            f"JER: 18.2.2 admits each enumeration identifier at most once in a TEXT "
            f"instruction, and ALL at most once; {sorted(duplicate)} repeats"
        )
    for name, new in instruction.changes:
        if name == "ALL" and not isinstance(new, NameKeyword):
            raise Asn1Error(
                "JER: 18.2.2 requires the NewTextOrKeyword to be a Keyword when the "
                "IdentifierOrAll is ALL"
            )
    known = {name for name, _n in (target.enumeration or ())}
    unknown = {name for name in names if name != "ALL" and name not in known}
    if unknown:
        raise Asn1Error(
            f"JER: a TEXT instruction names {sorted(unknown)}, which is not an "
            f"enumeration identifier of {target.name}"
        )
    if target.enumeration:  # §18.2.3
        produced = [instruction.applied(name) for name, _n in target.enumeration]
        clash = {name for name in produced if produced.count(name) > 1}
        if clash:
            raise Asn1Error(
                f"JER: 18.2.3 forbids two identical strings in the final set; "
                f"{sorted(clash)} appears twice"
            )


@dataclass(frozen=True)
class _Opts:
    """The rule profile and the encoding instructions, carried together down the walk."""

    rules: JerRules = JerRules.CANONICAL
    instructions: JerInstructions | None = None


#: §23.2 Table 2 — the four real values that are JSON strings rather than numbers.
_SPECIAL_REALS: dict[str, float] = {
    "-0": -0.0,
    "-INF": float("-inf"),
    "INF": float("inf"),
    "NaN": float("nan"),
}

#: §38.1 — the restricted character string types whose value IS a JSON string.
_TEXT_STRINGS = frozenset(
    {
        Universal.IA5_STRING,
        Universal.VISIBLE_STRING,
        Universal.NUMERIC_STRING,
        Universal.PRINTABLE_STRING,
        Universal.BMP_STRING,
        Universal.UNIVERSAL_STRING,
        Universal.UTF8_STRING,
    }
)

#: §38.2 — the remaining restricted string types, "encoded as if it were an octetstring
#: value consisting of the octets specified in Rec. ITU-T X.690, 8.23.5", i.e. as hex.
_OCTET_STRINGS = frozenset(
    {
        Universal.TELETEX_STRING,
        Universal.VIDEOTEX_STRING,
        Universal.GRAPHIC_STRING,
        Universal.GENERAL_STRING,
    }
)

#: §40 with §7.4.5 — the time types, and the useful types that X.680 clause 45 defines in
#: terms of VisibleString. All are JSON strings holding the value notation.
_TIME_STRINGS = frozenset(
    {
        Universal.UTC_TIME,
        Universal.GENERALIZED_TIME,
        Universal.OBJECT_DESCRIPTOR,
        Universal.TIME,
        Universal.DATE,
        Universal.TIME_OF_DAY,
        Universal.DATE_TIME,
        Universal.DURATION,
        Universal.OID_IRI,
        Universal.RELATIVE_OID_IRI,
    }
)


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
                f"JER: U+{code:04X} is an unpaired surrogate and has no UTF-8 encoding (7.6.2)"
            )
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


def _encode(kind: Asn1Type, value, opts: "_Opts", context: dict | None = None) -> str:
    if isinstance(kind, Primitive):
        return _encode_primitive(kind, value, opts)
    if isinstance(kind, (Sequence, Set)):
        return _encode_components(kind, value, opts)
    if isinstance(kind, SequenceOf):
        return _encode_list(kind, value, opts, sort=False)
    if isinstance(kind, SetOf):
        # §30.2: "in any order". CANONICAL sorts so one abstract value has one encoding.
        return _encode_list(kind, value, opts, sort=opts.rules is JerRules.CANONICAL)
    if isinstance(kind, Choice):
        return _encode_choice(kind, value, opts)
    if isinstance(kind, OpenType):
        return _encode_open_type(kind, value, opts, context)
    raise Asn1Error(f"JER: no encoding for schema type {type(kind).__name__}")


def _encode_primitive(kind: Primitive, value, opts: "_Opts") -> str:
    universal = kind.universal

    if universal == Universal.BOOLEAN:  # §20
        if not isinstance(value, bool):
            raise Asn1Error(f"{kind.name}: expected bool, got {type(value).__name__}")
        return "true" if value else "false"

    if universal == Universal.INTEGER:  # §21
        return _integer(value, kind.name)

    if universal == Universal.ENUMERATED:  # §22.1: a JSON *string*
        identifier = _enumeration_identifier(kind, value)
        text = _instruction(opts, kind, Text)  # §22.2 with clause 18
        return _string(text.applied(identifier) if text else identifier)

    if universal == Universal.REAL:  # §23
        return _encode_real(kind, value)

    if universal == Universal.NULL:  # §26
        if value is not None:
            raise Asn1Error(f"{kind.name}: a NULL value is None, got {value!r}")
        return "null"

    if universal == Universal.BIT_STRING:  # §24
        return _encode_bitstring(kind, value, opts)

    if universal == Universal.OCTET_STRING:  # §25
        return _encode_octetstring(kind, value, opts)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        return _string(_oid_text(kind, value))  # §32, §33

    if universal in _TEXT_STRINGS:  # §38.1
        if not isinstance(value, str):
            raise Asn1Error(f"{kind.name}: expected str, got {type(value).__name__}")
        return _string(value)

    if universal in _OCTET_STRINGS:  # §38.2
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, (bytes, bytearray)):
            raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
        return _string(_hex(bytes(value)))

    if universal in _TIME_STRINGS:  # §40 with §7.4.5
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
            f"the number alone"
        )
    if isinstance(value, str):
        if value not in {name for name, _n in kind.enumeration}:
            raise Asn1Error(f"{kind.name}: {value!r} is not an enumeration identifier")
        return value
    for name, number in kind.enumeration:
        if number == value:
            return name
    raise Asn1Error(
        f"{kind.name}: {value} names no enumeration item, and 22.1 gives an enumerated "
        f"value no numeric spelling"
    )


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
    if number != number:  # §23.1.1 with Table 2
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


def _encode_bitstring(kind: Primitive, value, opts: "_Opts") -> str:
    """§24. The one place a SIZE constraint reaches a JER encoder (§7.2.1 a))."""
    if kind.contains is not None and kind.encoded_by is None:  # §24.4 with §7.2.1 e)
        if not isinstance(value, BitString):
            raise Asn1Error(f"{kind.name}: expected BitString, got {type(value).__name__}")
        if value.unused:
            raise Asn1Error(
                f"{kind.name}: a CONTAINING bitstring's bits are a complete encoding "
                f"(X.682 11.4), so it cannot end {value.unused} bits into an octet"
            )
        inner = kind.contains.decode(decode_one(bytes(value.octets)), strictness=Strictness.BER)
        return "{" + _string("containing") + ":" + _encode(kind.contains, inner, opts) + "}"
    if not isinstance(value, BitString):
        raise Asn1Error(f"{kind.name}: expected BitString, got {type(value).__name__}")
    low, high = _bitstring_size(kind)
    if low is not None and low == high:  # §24.1 a) -> §24.2
        if value.bit_length != low:
            raise Asn1Error(
                f"{kind.name}: the effective size constraint fixes the length at {low} "
                f"bits, got {value.bit_length} (24.2)"
            )
        return _string(_hex(bytes(value.octets)))
    # §24.3: a JSON object carrying the octets and the true bit length.
    return (
        "{"
        + _string("value")
        + ":"
        + _string(_hex(bytes(value.octets)))
        + ","
        + _string("length")
        + ":"
        + _integer(value.bit_length, kind.name)
        + "}"
    )


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


def _encode_octetstring(kind: Primitive, value, opts: "_Opts") -> str:
    """§25. Note what is absent: no SIZE is consulted, because §7.2.2 h) excludes it."""
    if not isinstance(value, (bytes, bytearray)):
        raise Asn1Error(f"{kind.name}: expected bytes, got {type(value).__name__}")
    if kind.contains is not None and kind.encoded_by is None:  # §25.4 with §7.2.1 e)
        inner = kind.contains.decode(decode_one(bytes(value)), strictness=Strictness.BER)
        return "{" + _string("containing") + ":" + _encode(kind.contains, inner, opts) + "}"
    if _instruction(opts, kind, Base64) is not None:  # §25.1 a) -> §25.2
        # RFC 2045 §6.8, "except that the 76-character limit does not apply", which is why
        # `b64encode` is right and `encodebytes` (which folds at 76) is not.
        return _string(_base64.b64encode(bytes(value)).decode("ascii"))
    return _string(_hex(bytes(value)))  # §25.3


def _oid_text(kind: Primitive, value) -> str:
    """§32/§33 — the `XMLObjectIdentifierValue` production, i.e. dot-separated numbers."""
    if isinstance(value, str):
        arcs = tuple(int(part) for part in value.split(".") if part != "")
    elif isinstance(value, (tuple, list)):
        arcs = tuple(value)
    else:
        raise Asn1Error(f"{kind.name}: expected a tuple of arcs, got {type(value).__name__}")
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


def _encode_components(kind, value, opts: "_Opts") -> str:
    """§27.3 for a sequence, and §29 for a set — "encoded as if the type had been declared
    a sequence type".

    §29's NOTE is worth keeping in view: "The object-based encoding is always used for a set
    value because a set type is not allowed to have a final ARRAY encoding instruction."
    §14.2 restricts ARRAY to a sequence type, so the branch below cannot fire for a set.
    """
    if not isinstance(value, dict):
        raise Asn1Error(f"{kind.name}: expected a dict, got {type(value).__name__}")
    if _instruction(opts, kind, Array) is not None:
        return _encode_array(kind, value, opts)  # §27.1 -> §27.2
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
            if comp.has_default and item == comp.default and opts.rules is JerRules.CANONICAL:
                continue
        elif comp.has_default or comp.optional:
            continue
        else:
            raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
        members.append(
            _string(_member_name(opts, comp)) + ":" + _encode(comp.type, item, opts, value)
        )
    return "{" + ",".join(members) + "}"


def _member_name(opts: "_Opts", comp: Component) -> str:
    """§27.3.2 a) / §31.3.2 a) — the identifier, or the name a NAME instruction produces.

    Looked up against the COMPONENT, not against its type: §9.9 excludes NAME from the
    instructions a `typereference` inherits, so two components referencing one assigned type
    must be able to carry different member names.
    """
    instruction = _instruction(opts, comp, Name)
    return instruction.applied(comp.name) if instruction else comp.name


def _encode_array(kind, value, opts: "_Opts") -> str:
    """§27.2 — a JSON array with one element per component, positional rather than named.

    §27.2.1 puts the extension root first "in textual order", then each extension addition.
    A component absent from the value is the JSON token `null`, which is why §14.2 forbids
    an optional component that could *itself* produce `null`: the two would be
    indistinguishable, and the array has no names to tell them apart.
    """
    components = _flatten(kind.components)
    root = [comp for comp in components if not comp.extension]
    additions = [comp for comp in components if comp.extension]
    elements: list[str] = []
    for comp in root + additions:
        if comp.name in value:
            elements.append(_encode(comp.type, value[comp.name], opts, value))
        elif comp.optional or comp.has_default:
            elements.append("null")
        else:
            raise Asn1Error(f"{kind.name}: component {comp.name!r} is mandatory")
    if opts.rules is JerRules.CANONICAL:
        # §27.2.2: "Any number of instances of the JSON token null may be omitted from the
        # end of the JSON array, as a sender's option." The canonical profile omits them
        # all, since it omits a DEFAULT-valued component for the same reason.
        while elements and elements[-1] == "null":
            elements.pop()
    return "[" + ",".join(elements) + "]"


def _encode_list(kind, value, opts: "_Opts", *, sort: bool) -> str:
    """§28 for a sequence-of (order preserved) and §30.2 for a set-of (order free)."""
    if isinstance(value, (str, bytes, bytearray)) or not hasattr(value, "__iter__"):
        raise Asn1Error(f"{kind.name}: expected a sequence of elements, got {type(value).__name__}")
    if isinstance(kind, SetOf) and _instruction(opts, kind, ObjectAs) is not None:
        return _encode_object_map(kind, value, opts)  # §30.1 -> §30.3
    items = [_encode(kind.element, item, opts) for item in value]
    if sort:
        items.sort()
    return "[" + ",".join(items) + "]"


def _encode_object_map(kind: SetOf, value, opts: "_Opts") -> str:
    """§30.3 — a set-of of two-component sequences as a JSON object, i.e. a map.

    §30.3.2 a): the member NAME is the JER encoding of the first component's value, and
    §17.2 is what makes that legal — the first component is restricted to a string type or
    an enumerated type, both of which encode as a JSON string, and ECMA-404 clause 6 admits
    nothing else as a member name. So the quotation marks are already there and the encoded
    key is spliced in whole rather than re-quoted.
    """
    key_comp, value_comp = _flatten(kind.element.components)
    members: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            raise Asn1Error(f"{kind.name}: expected a dict per item, got {type(item).__name__}")
        for comp in (key_comp, value_comp):
            if comp.name not in item:
                raise Asn1Error(
                    f"{kind.name}: 17.2 forbids either component being OPTIONAL, so "
                    f"{comp.name!r} must be present in every item"
                )
        members.append(
            _encode(key_comp.type, item[key_comp.name], opts)
            + ":"
            + _encode(value_comp.type, item[value_comp.name], opts)
        )
    if opts.rules is JerRules.CANONICAL:
        members.sort()  # §30.3.3 leaves order free
    return "{" + ",".join(members) + "}"


def _encode_choice(kind: Choice, value, opts: "_Opts") -> str:
    """§31.3 — a JSON object with exactly one member, named for the chosen alternative."""
    if not (isinstance(value, tuple) and len(value) == 2):
        raise Asn1Error(
            f"{kind.name}: value must be an (alternative, value) pair, got {type(value).__name__}"
        )
    chosen, payload = value
    for alt in _flatten(kind.alternatives):
        if alt.name == chosen:
            inner = _encode(alt.type, payload, opts)
            if _instruction(opts, kind, Unwrapped) is not None:
                # §31.2: "the encoding of the chosen alternative", with the left brace, the
                # name, the colon and the right brace all omitted. Nothing in the octets
                # says which alternative it was -- see `_decode_choice`.
                return inner
            return "{" + _string(_member_name(opts, alt)) + ":" + inner + "}"
    raise Asn1Error(f"{kind.name}: {chosen!r} is not an alternative")


def _governing_context(kind: OpenType, siblings: dict | None) -> dict:
    if not siblings:
        return {}
    return {path: siblings[path[-1]] for path in kind.governing if path[-1] in siblings}


def _encode_open_type(kind: OpenType, value, opts: "_Opts", context: dict | None) -> str:
    """§41 — "The encoding of an open type value shall be the encoding of the value of the
    contained type."

    JER offers no hexadecimal fallback the way XER's §8.5 does, so an open type is
    encodable exactly when its X.682 §10.19 table resolves it. Refusing otherwise is the
    only honest option: there is no spelling for "some octets whose type I do not know".
    """
    if not isinstance(value, (bytes, bytearray)):
        raise Asn1Error(
            f"{kind.name}: an open type value is the contained value's complete encoding, "
            f"so it must be bytes, not {type(value).__name__}"
        )
    contained = kind.resolve(_governing_context(kind, context))
    if contained is None:
        raise Asn1Error(
            f"JER: 41 encodes an open type AS its contained type, and {kind.name} could "
            f"not be resolved by its table (X.682 10.19); JER has no hexadecimal "
            f"alternative to fall back to"
        )
    inner = contained.decode(decode_one(bytes(value)), strictness=Strictness.BER)
    return _encode(contained, inner, opts)


def encode_jer(
    kind: Asn1Type,
    value,
    *,
    rules: JerRules = JerRules.CANONICAL,
    instructions: "JerInstructions | None" = None,
) -> bytes:
    """Encode `value` as a complete JER encoding of `kind` (§7.6.2).

    The result is UTF-8 octets: §7.6.2 says the JSON tokens "shall be encoded in UTF-8 into
    an octet string, which is the complete encoding of the abstract value of the outermost
    type" — the encoding is the octets, not the character string.
    """
    return _encode(kind, value, _Opts(rules, instructions)).encode("utf-8")


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
        f"values as JSON strings"
    )


def _parse(text: str):
    try:
        return json.loads(
            text,
            parse_int=_Raw,
            parse_float=_Raw,
            parse_constant=_constant,
            object_pairs_hook=_pairs,
        )
    except json.JSONDecodeError as error:
        raise Asn1Error(f"JER: not a JSON text (ECMA-404): {error}") from None


def _decode(node, kind: Asn1Type, opts: "_Opts", context: dict | None = None):
    if isinstance(kind, Primitive):
        return _decode_primitive(node, kind, opts)
    if isinstance(kind, (Sequence, Set)):
        return _decode_components(node, kind, opts)
    if isinstance(kind, (SequenceOf, SetOf)):
        return _decode_list(node, kind, opts)
    if isinstance(kind, Choice):
        return _decode_choice(node, kind, opts)
    if isinstance(kind, OpenType):
        return _decode_open_type(node, kind, opts, context)
    raise Asn1Error(f"JER: no decoding for schema type {type(kind).__name__}")


def _want(node, want: type, kind, what: str, clause: str):
    if not isinstance(node, want) or (want is not bool and isinstance(node, bool)):
        raise Asn1Error(
            f"{getattr(kind, 'name', kind)}: expected {what} ({clause}), got {_describe(node)}"
        )
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
            f'{kind.name}: 21 requires an integer to be a JSON number "with no '
            f'fractional part and no exponent"; got {text}'
        )
    return int(text)


def _decode_primitive(node, kind: Primitive, opts: "_Opts"):
    universal = kind.universal

    if universal == Universal.BOOLEAN:  # §20
        return _want(node, bool, kind, "the JSON token true or false", "20")

    if universal == Universal.INTEGER:  # §21
        return _decode_integer(node, kind)

    if universal == Universal.ENUMERATED:  # §22
        name = _want(node, str, kind, "a JSON string", "22.1")
        if not kind.enumeration:
            raise Asn1Error(
                f"{kind.name}: ENUMERATED has no enumeration, so {name!r} cannot be "
                f"mapped to a value (22.2)"
            )
        text = _instruction(opts, kind, Text)  # §22.2
        for item, number in kind.enumeration:
            if (text.applied(item) if text else item) == name:
                return number
        raise Asn1Error(f"{kind.name}: {name!r} is not an enumeration identifier")

    if universal == Universal.REAL:  # §23
        if isinstance(node, str):
            if node not in _SPECIAL_REALS:
                raise Asn1Error(
                    f"{kind.name}: {node!r} is not one of Table 2's special real values"
                )
            return _SPECIAL_REALS[node]
        if isinstance(node, dict):  # §23.4
            if list(node) != ["base10Value"]:
                raise Asn1Error(
                    f"{kind.name}: 23.4 gives a real object exactly one member named "
                    f'"base10Value"; got {sorted(node)}'
                )
            return float(_want(node["base10Value"], _Raw, kind, "a JSON number", "23.4").text)
        return float(_want(node, _Raw, kind, "a JSON number", "23.3").text)

    if universal == Universal.NULL:  # §26
        if node is not None:
            raise Asn1Error(
                f"{kind.name}: expected the JSON token null (26), got {_describe(node)}"
            )
        return None

    if universal == Universal.BIT_STRING:  # §24
        return _decode_bitstring(node, kind, opts)

    if universal == Universal.OCTET_STRING:  # §25
        if isinstance(node, dict) and kind.contains is not None:
            return _decode_containing(node, kind, opts)  # §25.4
        text = _want(node, str, kind, "a JSON string", "25.3")
        if _instruction(opts, kind, Base64) is not None:  # §25.2
            try:
                return _base64.b64decode(text.encode("ascii"), validate=True)
            except Exception:
                raise Asn1Error(
                    f"{kind.name}: {text!r} is not a Base64 encoding (25.2, RFC 2045 6.8)"
                ) from None
        return _unhex(text, kind)

    if universal in (Universal.OBJECT_IDENTIFIER, Universal.RELATIVE_OID):
        text = _want(node, str, kind, "a JSON string", "32")
        parts = text.split(".")
        # §32 borrows X.680 §12.26's arc production: DIGIT ZERO..DIGIT NINE, and no leading
        # zero unless the arc is a single digit. `str.isdigit()` is Unicode-aware and knows
        # nothing about leading zeros, so `"1.2.\u0668\u0664\u0660"` and `"1.2.0840"` both
        # decoded to (1, 2, 840). A JER document is UTF-8 by design, so unlike the
        # octet-based rails nothing upstream had already restricted this to ASCII.
        if not text or any(not is_number_form(part) for part in parts):
            raise Asn1Error(
                f"{kind.name}: {text!r} is not an XMLObjectIdentifierValue; every arc is "
                f"DIGIT ZERO..DIGIT NINE with no leading zero (32, X.680 12.26)"
            )
        return tuple(int(part) for part in parts)

    if universal in _TEXT_STRINGS:  # §38.1
        return _want(node, str, kind, "a JSON string", "38.1")

    if universal in _OCTET_STRINGS:  # §38.2
        return _unhex(_want(node, str, kind, "a JSON string", "38.2"), kind)

    if universal in _TIME_STRINGS:  # §40
        return _want(node, str, kind, "a JSON string", "40")

    raise Asn1Error(f"JER: no decoding for UNIVERSAL {int(universal)} in this rail")


def _unhex(text: str, kind) -> bytes:
    if len(text) % 2:
        raise Asn1Error(
            f"{kind.name}: a hexadecimal JSON string has an even number of digits "
            f"(25.3); got {len(text)}"
        )
    try:
        return bytes.fromhex(text)
    except ValueError:
        raise Asn1Error(f"{kind.name}: {text!r} is not hexadecimal (25.3)") from None


def _decode_containing(node: dict, kind: Primitive, opts: "_Opts") -> bytes:
    """§24.4/§25.4 — `{"containing": <JER of the contained value>}`.

    The model holds such a value as the contained value's octets, so decoding the JSON form
    means re-encoding what it denotes. Sound because those octets are canonical (DER) by
    construction, and the JSON -- not some earlier octet string -- is what the peer sent.
    """
    if list(node) != ["containing"]:
        raise Asn1Error(
            f"{kind.name}: 25.4 gives a contents-constrained value exactly one member "
            f'named "containing"; got {sorted(node)}'
        )
    inner = _decode(node["containing"], kind.contains, opts)
    return encode_tlv(kind.contains.encode(inner))


def _decode_bitstring(node, kind: Primitive, opts: "_Opts") -> BitString:
    if isinstance(node, dict) and kind.contains is not None and "containing" in node:
        return BitString(_decode_containing(node, kind, opts), 0)  # §24.4
    low, high = _bitstring_size(kind)
    if isinstance(node, dict):  # §24.3
        if sorted(node) != ["length", "value"]:
            raise Asn1Error(
                f"{kind.name}: 24.3 gives a variable-size bitstring the members "
                f'"value" and "length"; got {sorted(node)}'
            )
        octets = _unhex(_want(node["value"], str, kind, "a JSON string", "24.3"), kind)
        bits = _decode_integer(node["length"], kind)
        if not 0 <= bits <= len(octets) * 8 or (len(octets) * 8 - bits) >= 8:
            raise Asn1Error(
                f'{kind.name}: "length" of {bits} bits does not match '
                f'{len(octets)} octet(s) of "value" (24.3)'
            )
        return BitString(octets, len(octets) * 8 - bits)
    text = _want(node, str, kind, "a JSON string", "24.2")  # §24.2
    if low is None or low != high:
        raise Asn1Error(
            f"{kind.name}: 24.1 c) gives a variable-size bitstring the 24.3 object form, "
            f"not a bare JSON string"
        )
    octets = _unhex(text, kind)
    if not low <= len(octets) * 8 < low + 8:
        raise Asn1Error(
            f"{kind.name}: the effective size constraint fixes the length at {low} bits, "
            f"which {len(octets)} octet(s) cannot carry (24.2)"
        )
    return BitString(octets, len(octets) * 8 - low)


def _decode_components(node, kind, opts: "_Opts") -> dict:
    """§27.3 — "The components of the sequence value may be added to the encoding in any
    order", so the decoder matches by name and never by position."""
    if _instruction(opts, kind, Array) is not None:
        return _decode_array(node, kind, opts)  # §27.2
    members = _want(node, dict, kind, "a JSON object", "27.3.1")
    components = _flatten(kind.components)
    by_name = {_member_name(opts, comp): comp for comp in components}
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
                f"no extension marker"
            )
        out[comp.name] = _decode(item, comp.type, opts, out)
        _resolve_open_type(comp, out, Strictness.BER)
    if opts.rules is JerRules.CANONICAL:
        order = [name for name in members if name in by_name]
        expected = [_member_name(opts, c) for c in components]
        if order != [name for name in expected if name in set(order)]:
            raise Asn1Error(
                f"{kind.name}: the BCIR canonical profile fixes member order at the "
                f"component order of the type; 27.3.3 leaves it free, so this is a "
                f"legal JER encoding that is not a canonical one"
            )
    for comp in components:
        if comp.name in out:
            continue
        if comp.has_default:
            out[comp.name] = comp.default  # X.680 §25.12
        elif not comp.optional:
            raise Asn1Error(f"{kind.name}: mandatory component {comp.name!r} is missing")
    return out


def _decode_array(node, kind, opts: "_Opts") -> dict:
    """§27.2 read back — positional, with `null` for an absent component.

    §27.2.2 lets a sender drop any number of trailing `null`s, so a short array is not a
    truncated one: the components it does not reach are simply absent.
    """
    elements = _want(node, list, kind, "a JSON array", "27.2.1")
    components = _flatten(kind.components)
    ordered = [c for c in components if not c.extension] + [c for c in components if c.extension]
    if len(elements) > len(ordered):
        raise Asn1Error(
            f"{kind.name}: the ARRAY encoding has {len(elements)} elements for "
            f"{len(ordered)} component(s) (27.2.1)"
        )
    out: dict = {}
    for comp, element in zip(ordered, elements):
        if element is None and (comp.optional or comp.has_default):
            continue  # §27.2.1: absent
        out[comp.name] = _decode(element, comp.type, opts, out)
        _resolve_open_type(comp, out, Strictness.BER)
    for comp in ordered:
        if comp.name in out:
            continue
        if comp.has_default:
            out[comp.name] = comp.default
        elif not comp.optional:
            raise Asn1Error(f"{kind.name}: mandatory component {comp.name!r} is missing (27.2.1)")
    return out


def _decode_list(node, kind, opts: "_Opts") -> list:
    if isinstance(kind, SetOf) and _instruction(opts, kind, ObjectAs) is not None:
        return _decode_object_map(node, kind, opts)  # §30.3
    items = _want(node, list, kind, "a JSON array", "28")
    return [_decode(item, kind.element, opts) for item in items]


def _decode_object_map(node, kind: SetOf, opts: "_Opts") -> list:
    """§30.3 read back — each member becomes one two-component sequence value."""
    members = _want(node, dict, kind, "a JSON object", "30.3.1")
    key_comp, value_comp = _flatten(kind.element.components)
    out: list = []
    for name, item in members.items():
        out.append(
            {
                key_comp.name: _decode(name, key_comp.type, opts),
                value_comp.name: _decode(item, value_comp.type, opts),
            }
        )
    return out


def _node_kind(node) -> str:
    """Which of §19.2.2's kinds of JSON value a parsed node is."""
    if node is None:
        return "null"
    if isinstance(node, bool):
        return "true" if node else "false"
    if isinstance(node, _Raw):
        return "number"
    if isinstance(node, str):
        return "string"
    if isinstance(node, list):
        return "array"
    return "object"


def _json_kinds(kind: Asn1Type, opts: "_Opts") -> frozenset:
    """Every kind of JSON value a type can produce — §19.2.2's discriminator.

    This is what makes the unwrapped encoding decodable at all, and it is why the clause
    lists exactly six kinds: an unwrapped choice carries no name, so the *shape* of the JSON
    has to name the alternative. Anything this function gets wrong is a value silently
    decoded as the wrong alternative, which is why it is derived from the same branches the
    encoder takes rather than guessed at.
    """
    if isinstance(kind, Primitive):
        universal = kind.universal
        if universal == Universal.BOOLEAN:
            return frozenset({"true", "false"})
        if universal == Universal.INTEGER:
            return frozenset({"number"})
        if universal == Universal.REAL:
            # §23.1.1 sends the four special values to a JSON string and §23.3 sends
            # everything this rail can build to a JSON number. §23.4's object form needs an
            # inner type constraint on the base that `Primitive` cannot express, so it is
            # not listed -- see `_encode_real`.
            return frozenset({"number", "string"})
        if universal == Universal.NULL:
            return frozenset({"null"})
        if universal == Universal.BIT_STRING:
            if kind.contains is not None and kind.encoded_by is None:
                return frozenset({"object"})  # §24.4
            low, high = _bitstring_size(kind)
            return (
                frozenset({"string"}) if low is not None and low == high else frozenset({"object"})
            )  # §24.2 / §24.3
        if universal == Universal.OCTET_STRING:
            if kind.contains is not None and kind.encoded_by is None:
                return frozenset({"object"})  # §25.4
            return frozenset({"string"})
        return frozenset({"string"})  # every remaining type
    if isinstance(kind, (Sequence, Set)):
        return (
            frozenset({"array"})
            if _instruction(opts, kind, Array) is not None
            else frozenset({"object"})
        )
    if isinstance(kind, SequenceOf):
        return frozenset({"array"})
    if isinstance(kind, SetOf):
        return (
            frozenset({"object"})
            if _instruction(opts, kind, ObjectAs) is not None
            else frozenset({"array"})
        )
    if isinstance(kind, Choice):
        if _instruction(opts, kind, Unwrapped) is None:
            return frozenset({"object"})  # §31.3
        kinds: set = set()
        for alt in _flatten(kind.alternatives):
            kinds |= _json_kinds(alt.type, opts)
        return frozenset(kinds)
    raise Asn1Error("JER: 19.2.4 forbids an open type as an alternative of an unwrapped choice")


def _mandatory_names(kind, opts: "_Opts") -> frozenset:
    """The member names §19.2.3 discriminates two object-producing alternatives by."""
    return frozenset(
        _member_name(opts, comp)
        for comp in _flatten(kind.components)
        if not (comp.optional or comp.has_default)
    )


def _decode_choice(node, kind: Choice, opts: "_Opts") -> tuple:
    """§31.3.1 — "a JSON object having exactly one member" — or §31.2 when UNWRAPPED."""
    if _instruction(opts, kind, Unwrapped) is not None:
        return _decode_unwrapped(node, kind, opts)
    members = _want(node, dict, kind, "a JSON object", "31.3.1")
    if len(members) != 1:
        raise Asn1Error(
            f"{kind.name}: 31.3.1 gives a choice value exactly one member; got {len(members)}"
        )
    ((name, item),) = members.items()
    for alt in _flatten(kind.alternatives):
        if _member_name(opts, alt) == name:
            return (alt.name, _decode(item, alt.type, opts))
    raise Asn1Error(f"{kind.name}: {name!r} matches no alternative")


def _decode_unwrapped(node, kind: Choice, opts: "_Opts") -> tuple:
    """§31.2 — identify the alternative from the shape of the JSON, because nothing names it.

    §31.2's NOTE says the form "relies on the decoder's ability to identify the alternative
    that was encoded by examining the JER encoding of the alternative", and §19.2.2-§19.2.4
    are the restrictions that make that ability possible. So this function IS the
    enforcement point for those restrictions: where they hold it succeeds, and where they
    are violated it reports the ambiguity against the clause rather than picking a winner.
    That is deliberate -- §6.6's NOTE says "It is the final encoding instructions that
    determine conformity", so a check at assignment time could be wrong in either direction
    once a later assignment lands.
    """
    shape = _node_kind(node)
    candidates = [
        alt for alt in _flatten(kind.alternatives) if shape in _json_kinds(alt.type, opts)
    ]
    if not candidates:
        raise Asn1Error(
            f"{kind.name}: no alternative of this unwrapped choice produces "
            f"{_describe(node)} (31.2)"
        )
    if len(candidates) > 1:
        if shape != "object":
            raise Asn1Error(
                f"{kind.name}: 19.2.2 admits at most one alternative producing "
                f"{_describe(node)}, but {[a.name for a in candidates]} all do; this "
                f"choice cannot carry a final UNWRAPPED encoding instruction"
            )
        # §19.2.3: the object-producing alternatives are separated by a mandatory member
        # name that the others do not have.
        present = set(node)
        matched = [
            alt
            for alt in candidates
            if isinstance(alt.type, (Sequence, Set)) and _mandatory_names(alt.type, opts) <= present
        ]
        if len(matched) != 1:
            raise Asn1Error(
                f"{kind.name}: 19.2.3 requires two object-producing alternatives to be "
                f"separated by a mandatory member name; {sorted(present)} matches "
                f"{[a.name for a in matched] or 'none'}"
            )
        candidates = matched
    alt = candidates[0]
    return (alt.name, _decode(node, alt.type, opts))


def _decode_open_type(node, kind: OpenType, opts: "_Opts", context: dict | None) -> bytes:
    contained = kind.resolve(_governing_context(kind, context))
    if contained is None:
        raise Asn1Error(
            f"JER: 41 encodes an open type AS its contained type, and {kind.name} could "
            f"not be resolved by its table (X.682 10.19)"
        )
    return encode_tlv(contained.encode(_decode(node, contained, opts)))


def decode_jer(
    data: bytes | str,
    kind: Asn1Type,
    *,
    rules: JerRules = JerRules.CANONICAL,
    instructions: "JerInstructions | None" = None,
) -> object:
    """Decode a complete JER encoding of `kind`.

    `opts` selects what the decoder *accepts*. `CANONICAL` is the stricter of the two: it
    refuses the encoder's options the BCIR profile does not emit, so what is digested is
    what BCIR would have produced. `BASIC` implements §6.3 — "Decoders that claim
    conformance to JER shall support all JER encoding alternatives".
    """
    if isinstance(data, str):
        text = data
    elif isinstance(data, (bytes, bytearray)):
        try:
            text = bytes(data).decode("utf-8")  # §7.6.2
        except UnicodeDecodeError as error:
            raise Asn1Error(f"JER: the encoding is UTF-8 (7.6.2): {error}") from None
    else:
        raise Asn1Error("JER: expected bytes or str")
    return _decode(_parse(text), kind, _Opts(rules, instructions))


__all__ = [
    "JER_OID",
    "JER_OID_DESCRIPTOR",
    "Array",
    "Base64",
    "JerInstructions",
    "JerRules",
    "Name",
    "NameKeyword",
    "Not",
    "ObjectAs",
    "Text",
    "Unwrapped",
    "apply_name_keyword",
    "decode_jer",
    "encode_jer",
]
