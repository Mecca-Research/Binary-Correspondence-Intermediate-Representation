"""ASN1-B: semantic bounds on canonical JER, derived at the point of application.

The 2026-10-06 audit (items 19 and 20) found no schema-derived memory bound for JER: the J2 plan
compiler bounds only BOOLEAN, NULL, ENUMERATED and a fixed-size BIT STRING (`jer_plan._bounded`),
because X.697 §7.2.2 makes a value constraint invisible to the ENCODING -- `INTEGER (0..255)` and
`INTEGER` produce the same JSON for the same value. True, and beside the point for a memory
guarantee: a value constraint restricts the VALUES, so under BCIR's canonical profile
(`JerRules.CANONICAL`: shortest spellings, minimal escapes, no whitespace) every VALID value of
`INTEGER (0..255)` encodes in at most 3 octets, and every valid `OCTET STRING (SIZE (4))` in
exactly 10. This module derives that bound for a type, and for a parameterized type at the point
it is applied -- after X.683 substitution, so `Bounded {9}` and `Bounded {99999}` get 1 and 5.

`max_octets(kind)` is the largest canonical JER encoding, in octets, of any value of `kind` that
satisfies its constraints, or None where no finite bound follows from the type. `witness(kind)`
is a valid value whose encoding is exactly that long: the bound is a maximum, never a guess,
and a test holds both to the encoder (`test_asn1_jer_bounds.py`).

**What bounds a value**, case by case (the encoder's spellings are in `jer.py`):

* BOOLEAN 5 (`false`), NULL 4; ENUMERATED its longest identifier, quoted (root and extension
  additions -- the encoder spells only known identifiers);
* INTEGER: the root value range, when it is not extensible and both ends are finite -- the
  longer of `str(lo)` and `str(hi)`, since an integer's spelling grows with its magnitude;
* OCTET STRING: the root SIZE's upper end, two hexadecimal digits an octet, quoted;
* BIT STRING: a fixed size is §24.2's hexadecimal string; any other bounded size is §24.3's
  `{"value":"…","length":n}` at the upper end;
* the character strings: the root SIZE's upper end times the longest JSON spelling of one
  character the type admits -- its X.680 repertoire (PrintableString and NumericString 1 octet;
  VisibleString 2, for `\\"` and `\\\\`; IA5String, BMPString, UniversalString and UTF8String 6,
  for a control character's `\\u00XX`), narrowed by a permitted alphabet;
* SEQUENCE and SET: every component present, named and comma-separated; SEQUENCE OF and SET
  OF: the root SIZE's upper end of elements; CHOICE: the longest alternative, wrapped.

**Unbounded** (None): an extensible constraint (its extension admits values the root does not),
an unbounded range or size, REAL, the object identifiers, the time types and §38.2's
hexadecimal strings, an open type, a CONTAINING string, a recursive type, and any type an
encoding instruction reshapes (instructions are not derived here).
"""

from __future__ import annotations

from .codec import NULL, BitString
from .constraints import root_alphabet, root_size_bounds, root_value_bounds
from .schema import (
    Choice,
    OpenType,
    Primitive,
    Reference,
    Sequence,
    SequenceOf,
    Set,
    SetOf,
)
from .tags import Universal

#: X.680 §41 Table 8/10: the characters each restricted string type admits (the ranges are the
#: type's repertoire; PermittedAlphabet narrows it).
_PRINTABLE = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '()+,-./:=?")
_NUMERIC = frozenset("0123456789 ")
_RANGES = {
    Universal.IA5_STRING: (0x00, 0x7F),
    Universal.VISIBLE_STRING: (0x20, 0x7E),
    Universal.BMP_STRING: (0x00, 0xFFFF),
    Universal.UNIVERSAL_STRING: (0x00, 0x10FFFF),
    Universal.UTF8_STRING: (0x00, 0x10FFFF),
}
_FIXED = {Universal.PRINTABLE_STRING: _PRINTABLE, Universal.NUMERIC_STRING: _NUMERIC}
_JSON_ESCAPES = {'"': 2, "\\": 2, "\b": 2, "\f": 2, "\n": 2, "\r": 2, "\t": 2}


def char_octets(character: str) -> int:
    """The octets `jer._string` spends on one character: its escape, or its UTF-8."""
    if character in _JSON_ESCAPES:
        return _JSON_ESCAPES[character]
    code = ord(character)
    if code < 0x20:
        return 6  # \u00XX
    return len(character.encode("utf-8"))


def _repertoire_max(universal: int) -> tuple[int, str] | None:
    """(the longest one-character spelling, a character that has it) over the type's whole
    repertoire, without enumerating a million code points: the extremes are a control
    character (6), an escaped quote (2), and the widest UTF-8 the range reaches."""
    if universal in _FIXED:
        best = max(sorted(_FIXED[universal]), key=char_octets)
        return char_octets(best), best
    span = _RANGES.get(universal)
    if span is None:
        return None
    low, high = span
    candidates = [chr(low), chr(high)]
    if low <= 0x1F:
        candidates.append("\x00")
    if low <= 0x22 <= high:
        candidates.append('"')
    for edge in (0x7F, 0x7FF, 0xFFFF, 0x10FFFF):
        if low <= edge <= high and not 0xD800 <= edge <= 0xDFFF:
            candidates.append(chr(edge))
    best = max(candidates, key=lambda c: (char_octets(c), -ord(c)))
    return char_octets(best), best


def _string_char(kind: Primitive) -> tuple[int, str] | None:
    """(the longest one-character spelling a value of `kind` can contain, that character)."""
    alphabet = root_alphabet(kind.constraint)
    if alphabet is not None:
        if kind.universal in _FIXED:
            alphabet = alphabet & _FIXED[kind.universal]
        elif kind.universal in _RANGES:
            low, high = _RANGES[kind.universal]
            alphabet = frozenset(c for c in alphabet if low <= ord(c) <= high)
        if not alphabet:
            return None
        best = max(sorted(alphabet), key=char_octets)
        return char_octets(best), best
    return _repertoire_max(kind.universal)


def _root_size(kind) -> tuple[int | None, int | None]:
    (low, high), extensible = root_size_bounds(getattr(kind, "constraint", None))
    if extensible:
        return None, None
    return low, high


def _bound(kind, stack: tuple) -> tuple[int, object] | None:
    """(max octets, a valid value that attains them), or None."""
    if isinstance(kind, Reference):
        target = kind.resolved()
        if any(target is seen for seen in stack):
            return None  # recursive: a value may nest without end
        return _bound(target, stack + (target,))
    if any(kind is seen for seen in stack[:-1]):
        return None
    stack = stack + (kind,)
    if isinstance(kind, Primitive):
        return _primitive(kind)
    if isinstance(kind, (Sequence, Set)):
        return _components(kind, stack)
    if isinstance(kind, (SequenceOf, SetOf)):
        _low, high = _root_size(kind)
        if high is None:
            return None
        element = _bound(kind.element, stack)
        if element is None:
            return None
        octets, value = element
        return 2 + high * octets + max(high - 1, 0), [value] * high
    if isinstance(kind, Choice):
        best = None
        for alt in _flatten(kind.alternatives):
            inner = _bound(alt.type, stack)
            if inner is None:
                return None  # any unbounded alternative unbounds the choice
            total = 2 + len(_quoted(alt.name)) + 1 + inner[0]
            if best is None or total > best[0]:
                best = (total, (alt.name, inner[1]))
        return best
    if isinstance(kind, OpenType):
        return None
    return None


def _flatten(components):
    from .jer import _flatten as flatten

    return flatten(components)


def _quoted(text: str) -> str:
    from .jer import _string

    return _string(text)


def _components(kind, stack) -> tuple[int, dict] | None:
    members, value = [], {}
    for comp in _flatten(kind.components):
        inner = _bound(comp.type, stack)
        if inner is None:
            return None
        octets, item = inner
        if comp.has_default and item == comp.default:
            return None  # the longest value is the default, which the profile omits
        members.append(len(_quoted(comp.name).encode("utf-8")) + 1 + octets)
        value[comp.name] = item
    return 2 + sum(members) + max(len(members) - 1, 0), value


def _primitive(kind: Primitive) -> tuple[int, object] | None:
    u = kind.universal
    if kind.contains is not None or kind.table_values is not None:
        return None
    if u == Universal.BOOLEAN:
        return 5, False
    if u == Universal.NULL:
        return 4, NULL
    if u == Universal.ENUMERATED:
        names = [n for n, _v in (kind.enumeration or ())] + [
            n for n, _v in (kind.enum_extension or ())
        ]
        if not names:
            return None
        best = max(names, key=len)
        return len(_quoted(best).encode("utf-8")), best
    if u == Universal.INTEGER:
        (low, high), extensible = root_value_bounds(kind.constraint)
        if extensible or low is None or high is None:
            return None
        best = max((low, high), key=lambda v: (len(str(v)), v == low))
        return len(str(best)), best
    if u == Universal.OCTET_STRING:
        _low, high = _root_size(kind)
        if high is None:
            return None
        return 2 + 2 * high, b"\xff" * high
    if u == Universal.BIT_STRING:
        low, high = _root_size(kind)
        if high is None:
            return None
        octets = (high + 7) // 8
        value = BitString(b"\xff" * octets, 8 * octets - high)
        if low == high:  # §24.2: the fixed-size hexadecimal string
            return 2 + 2 * octets, value
        # §24.3: {"value":"…","length":n}, at the upper end
        return len('{"value":"') + 2 * octets + len('","length":') + len(str(high)) + 1, value
    if u in _FIXED or u in _RANGES:
        _low, high = _root_size(kind)
        if high is None:
            return None
        char = _string_char(kind)
        if char is None:
            return None
        width, character = char
        return 2 + high * width, character * high
    return None  # REAL, the object identifiers, the time types, §38.2's strings


def max_octets(kind) -> int | None:
    """The largest canonical JER encoding, in octets, of any value of `kind` its constraints
    admit; None where none follows from the type (module docstring)."""
    found = _bound(kind, ())
    return None if found is None else found[0]


def witness(kind):
    """A value of `kind` its constraints admit whose canonical JER encoding is `max_octets`
    long. Raises ValueError where the type is unbounded."""
    found = _bound(kind, ())
    if found is None:
        raise ValueError(f"{getattr(kind, 'name', kind)!r} has no finite canonical JER bound")
    return found[1]


__all__ = ["char_octets", "max_octets", "witness"]
