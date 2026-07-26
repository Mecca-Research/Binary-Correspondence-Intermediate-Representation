"""Subtype constraints — Rec. ITU-T X.680 (02/2021) clauses 49–51.

A constraint restricts the *value set* of a type. In BER/DER that is invisible in the
octets, which is why `der.py` never needed it. In OER and PER it is the opposite: the
encoding **is chosen from the constraint**. `INTEGER (0..255)` is one octet with no
length determinant; unconstrained `INTEGER` is a length determinant plus a variable-size
signed number. Same abstract value, four times the octets.

That is also the BCIR payoff the roadmap points at: *a constrained ASN.1 type is a claim
geometry.* `INTEGER (0..255)` is an 8-bit lane, `SEQUENCE (SIZE(1..64)) OF` a bounded
extent — exactly the information a cost model needs to price a decode.

WHAT IS MODELLED. The set-arithmetic core of clause 51: single values, value ranges with
open endpoints and MIN/MAX, SIZE, FROM (permitted alphabet), and the UNION /
INTERSECTION / EXCEPT composition of clause 49. Table and component-relation constraints
(X.682 §10) are not here — they select a *type*, not a value set, and belong with the
open type in phase F.

THE ONE RULE THAT SURPRISES PEOPLE. An **extensible** constraint — one written
`(0..255, ...)` — is *not* OER-visible at all (X.696 §8.2.2 g). The extension marker
says the value set may grow in a later version of the protocol, so an encoder that sized
a field from today's bounds would produce octets tomorrow's peer cannot read. Such a type
encodes as though it had no bounds. `effective_value_constraint` therefore returns
`(None, None)` for it, and that is a correctness rule rather than a conservatism.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tags import Asn1Error

#: An effective constraint: (lower, upper), where None means "no finite bound".
Bounds = tuple[int | None, int | None]

#: The empty bound pair, meaning "unconstrained as far as the encoding is concerned".
UNBOUNDED: Bounds = (None, None)


class Constraint:
    """Base of the constraint model. Subclasses describe a set of permitted values."""

    def value_bounds(self) -> Bounds:              # pragma: no cover - abstract
        """The (lower, upper) integer bounds this constraint imposes on a value."""
        raise NotImplementedError

    def size_bounds(self) -> Bounds:
        """The (lower, upper) bounds this constraint imposes on a value's SIZE."""
        return UNBOUNDED

    def alphabet(self) -> frozenset[str] | None:
        """The permitted characters, or None when the constraint does not restrict them."""
        return None

    def permits(self, value) -> bool:              # pragma: no cover - abstract
        raise NotImplementedError


@dataclass(frozen=True)
class SingleValue(Constraint):
    """§51.2 `(3)` — exactly one value."""

    value: object

    def value_bounds(self) -> Bounds:
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            return UNBOUNDED
        return (self.value, self.value)

    def permits(self, value) -> bool:
        return value == self.value

    def __str__(self) -> str:
        return f"({_render(self.value)})"


@dataclass(frozen=True)
class ValueRange(Constraint):
    """§51.4 `(0..255)`, `(0<..255)`, `(MIN..MAX)`.

    `None` is MIN/MAX — an unspecified endpoint, which §51.4.4 says extends as far as the
    parent type allows. The `*_open` flags are the `<` of §51.4.3: `(0<..255)` excludes 0.
    They are normalized away by `value_bounds`, because for an integer an open endpoint is
    just the adjacent closed one, and the encoding only ever sees the closed form.
    """

    lower: int | None = None
    upper: int | None = None
    lower_open: bool = False
    upper_open: bool = False

    def value_bounds(self) -> Bounds:
        low = None if self.lower is None else self.lower + (1 if self.lower_open else 0)
        high = None if self.upper is None else self.upper - (1 if self.upper_open else 0)
        return (low, high)

    def permits(self, value) -> bool:
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        low, high = self.value_bounds()
        return (low is None or value >= low) and (high is None or value <= high)

    def __str__(self) -> str:
        low = "MIN" if self.lower is None else _render(self.lower)
        high = "MAX" if self.upper is None else _render(self.upper)
        return f"({low}{'<' if self.lower_open else ''}..{'<' if self.upper_open else ''}{high})"


@dataclass(frozen=True)
class Size(Constraint):
    """§51.5 `SIZE (1..64)` — a constraint on the value's length, not its value.

    §51.5.2 limits it to bit string, octet string, character string, SET OF and
    SEQUENCE OF types, which is why it reports through `size_bounds` and leaves
    `value_bounds` unconstrained: a SIZE says nothing about what the elements are.
    """

    inner: Constraint

    def value_bounds(self) -> Bounds:
        return UNBOUNDED

    def size_bounds(self) -> Bounds:
        return self.inner.value_bounds()

    def permits(self, value) -> bool:
        try:
            length = len(value)
        except TypeError:
            return False
        return self.inner.permits(length)

    def __str__(self) -> str:
        return f"(SIZE {self.inner})"


@dataclass(frozen=True)
class PermittedAlphabet(Constraint):
    """§51.7 `FROM ("0".."9")` — the sub-alphabet a string may draw from."""

    inner: Constraint

    def value_bounds(self) -> Bounds:
        return UNBOUNDED

    def alphabet(self) -> frozenset[str] | None:
        return _alphabet_of(self.inner)

    def permits(self, value) -> bool:
        allowed = self.alphabet()
        if allowed is None or not isinstance(value, str):
            return False
        return all(character in allowed for character in value)

    def __str__(self) -> str:
        return f"(FROM {self.inner})"


@dataclass(frozen=True)
class Union(Constraint):
    """§49.6 `A | B` — the union of two element sets.

    The effective bounds of a union are the SMALLEST RANGE THAT INCLUDES BOTH (X.696
    §8.2.7's "least permitted value" / "greatest permitted value"), not the union of the
    two ranges: an encoding is a single field width, so `(0..3 | 100..103)` is encoded as
    `0..103`. Holes in the value set are the verifier's business, not the encoder's.
    """

    parts: tuple[Constraint, ...]

    def value_bounds(self) -> Bounds:
        return _span(part.value_bounds() for part in self.parts)

    def size_bounds(self) -> Bounds:
        return _span(part.size_bounds() for part in self.parts)

    def alphabet(self) -> frozenset[str] | None:
        found = [part.alphabet() for part in self.parts]
        if any(item is None for item in found):
            return None
        return frozenset().union(*found)

    def permits(self, value) -> bool:
        return any(part.permits(value) for part in self.parts)

    def __str__(self) -> str:
        return "(" + " | ".join(str(p).strip("()") for p in self.parts) + ")"


@dataclass(frozen=True)
class Intersection(Constraint):
    """§49.7 `A ^ B` — the intersection, whose bounds are the tighter of each side."""

    parts: tuple[Constraint, ...]

    def value_bounds(self) -> Bounds:
        low, high = UNBOUNDED
        for part in self.parts:
            part_low, part_high = part.value_bounds()
            low = part_low if low is None else (
                low if part_low is None else max(low, part_low))
            high = part_high if high is None else (
                high if part_high is None else min(high, part_high))
        return (low, high)

    def size_bounds(self) -> Bounds:
        low, high = UNBOUNDED
        for part in self.parts:
            part_low, part_high = part.size_bounds()
            low = part_low if low is None else (
                low if part_low is None else max(low, part_low))
            high = part_high if high is None else (
                high if part_high is None else min(high, part_high))
        return (low, high)

    def alphabet(self) -> frozenset[str] | None:
        found = [part.alphabet() for part in self.parts if part.alphabet() is not None]
        if not found:
            return None
        result = found[0]
        for item in found[1:]:
            result &= item
        return result

    def permits(self, value) -> bool:
        return all(part.permits(value) for part in self.parts)

    def __str__(self) -> str:
        return "(" + " ^ ".join(str(p).strip("()") for p in self.parts) + ")"


@dataclass(frozen=True)
class Extensible(Constraint):
    """§49.4 `(0..255, ...)` — a constraint with an extension marker.

    This is the one that changes an encoding by *removing* information. X.696 §8.2.2 g)
    makes an extensible subtype constraint **not OER-visible**: the marker says the value
    set may grow in a later version, so an encoder that sized the field from today's
    bounds would emit octets a future peer cannot read. The type therefore encodes as if
    unbounded, and `value_bounds`/`size_bounds` say so.

    `root` is kept because it is still the value set a *verifier* should check today —
    only the ENCODER has to ignore it.
    """

    root: Constraint

    def value_bounds(self) -> Bounds:
        return UNBOUNDED                            # X.696 8.2.2 g)

    def size_bounds(self) -> Bounds:
        return UNBOUNDED                            # X.696 8.2.2 g)

    def alphabet(self) -> frozenset[str] | None:
        return None                                 # X.696 8.2.2 g)

    def permits(self, value) -> bool:
        # An extensible constraint admits its root today and anything a later version
        # adds, so it cannot refuse a value on the strength of the root alone.
        return True

    def __str__(self) -> str:
        return f"({str(self.root).strip('()')}, ...)"


def _root_bounds(constraint: Constraint, dimension: str) -> tuple[Bounds, bool]:
    """The EXTENSION ROOT's bounds along `dimension`, and whether a marker was found.

    `value_bounds`/`size_bounds` deliberately collapse an extensible constraint to
    UNBOUNDED, because X.696 §8.2.2 g) makes such a constraint invisible to OER: the marker
    says the value set may grow, so an encoder that sized a field from today's bounds would
    emit octets a future peer could not read.

    PER makes the opposite choice and needs both halves. X.691 §13.1/§17.3/§20.4/§30.4 all
    have the same shape: emit ONE bit saying whether the value falls inside the extension
    root, then encode it against the ROOT's bounds if it does and unconstrained if it does
    not. So a PER encoder needs the root bounds that OER discards, plus the flag -- and
    collapsing to UNBOUNDED would silently cost every constrained field its width.

    The walk descends through Union/Intersection because the marker is usually nested:
    `SIZE(8, ..., 9..20)` inside `FROM("0".."9") ^ SIZE(...)` is an extensible size sitting
    beside a non-extensible alphabet, and §10.3.21 keeps the intersection PER-visible.
    """
    if isinstance(constraint, Extensible):
        inner, _ = _root_bounds(constraint.root, dimension)
        return inner, True
    if isinstance(constraint, Size):
        # SIZE crosses dimensions: its SIZE bounds are its inner constraint's VALUE bounds
        # (`size_bounds` delegates exactly so). The extension marker in `SIZE(8, ..., 9..20)`
        # sits on that inner constraint, so a walk that does not cross here never sees the
        # `Extensible` and reports the type as non-extensible with unbounded size.
        if dimension != "size":
            return UNBOUNDED, False
        return _root_bounds(constraint.inner, "value")
    if isinstance(constraint, Intersection):
        low, high = UNBOUNDED
        extensible = False
        for part in constraint.parts:
            (part_low, part_high), part_ext = _root_bounds(part, dimension)
            extensible = extensible or part_ext
            low = part_low if low is None else (
                low if part_low is None else max(low, part_low))
            high = part_high if high is None else (
                high if part_high is None else min(high, part_high))
        return (low, high), extensible
    if isinstance(constraint, Union):
        parts = [_root_bounds(part, dimension) for part in constraint.parts]
        return _span(bounds for bounds, _ in parts), any(ext for _, ext in parts)
    getter = constraint.value_bounds if dimension == "value" else constraint.size_bounds
    return getter(), False


def without_extension(constraint: Constraint) -> Constraint:
    """Strip every extension marker from a constraint tree (X.680 §50.11).

    §50.11: when a subtype constraint is SERIALLY applied to a parent that is extensible,
    "the result of the second (serially applied) constraint is defined to be the same as if
    the constraint had been applied to the parent type without its extension marker and
    possible extension additions". So `NameString (SIZE(1))`, where NameString is
    `VisibleString (FROM(...) ^ SIZE(1..64, ...))`, is NOT extensible -- the outer SIZE(1)
    erases the inner marker.

    That is visible in the encoding and nowhere else: X.691 Annex A.3 encodes `initial`
    with no extension bit and no length determinant at all, while the sibling `givenName`
    of the same base type keeps both.
    """
    if isinstance(constraint, Extensible):
        return without_extension(constraint.root)
    if isinstance(constraint, Intersection):
        return Intersection(tuple(without_extension(p) for p in constraint.parts))
    if isinstance(constraint, Union):
        return Union(tuple(without_extension(p) for p in constraint.parts))
    if isinstance(constraint, Size):
        return Size(without_extension(constraint.inner))
    return constraint


def root_value_bounds(constraint: Constraint | None) -> tuple[Bounds, bool]:
    """`((low, high), extensible)` for the VALUE dimension (X.691 §13.1)."""
    if constraint is None:
        return UNBOUNDED, False
    return _root_bounds(constraint, "value")


def root_size_bounds(constraint: Constraint | None) -> tuple[Bounds, bool]:
    """`((low, high), extensible)` for the SIZE dimension (§17.3/§20.4/§30.4)."""
    if constraint is None:
        return UNBOUNDED, False
    return _root_bounds(constraint, "size")


def root_alphabet(constraint: Constraint | None) -> frozenset[str] | None:
    """The effective permitted alphabet, honouring §10.3.11.

    An EXTENSIBLE permitted-alphabet constraint is not PER-visible at all -- unlike a size
    or value constraint, there is no "encode against the root" fallback, the alphabet simply
    reverts to the base type's. `Extensible.alphabet()` already returns None for exactly
    that reason, so the ordinary accessor is correct here and this is a named alias that
    says so rather than a second implementation.
    """
    if constraint is None:
        return None
    return constraint.alphabet()


def _render(value) -> str:
    """Render an endpoint the way X.680 writes it.

    A permitted-alphabet endpoint is a CHARACTER STRING value (§51.7, and §51.4.4's NOTE
    requires it to be size 1), so it must keep its quotes: printing `FROM ("0".."9")` as
    `FROM (0..9)` re-parses as an integer range, and the alphabet silently becomes None.
    """
    if isinstance(value, str):
        return '"' + value.replace('"', '""') + '"'
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def _span(pairs) -> Bounds:
    """The smallest range containing every pair; None dominates (it means unbounded)."""
    low: int | None = None
    high: int | None = None
    first = True
    for pair_low, pair_high in pairs:
        if first:
            low, high, first = pair_low, pair_high, False
            continue
        low = None if (low is None or pair_low is None) else min(low, pair_low)
        high = None if (high is None or pair_high is None) else max(high, pair_high)
    return (low, high)


def _alphabet_of(constraint: Constraint) -> frozenset[str] | None:
    """The characters a constraint admits, when it is written as characters.

    A `FROM` constraint's inner element set is written in the *parent string's* value
    notation, so `FROM ("A".."Z")` is a ValueRange over one-character strings, not over
    integers — which is why this walks the constraint rather than reading bounds.
    """
    if isinstance(constraint, SingleValue):
        return frozenset(constraint.value) if isinstance(constraint.value, str) else None
    if isinstance(constraint, ValueRange):
        low, high = constraint.lower, constraint.upper
        if not (isinstance(low, str) and isinstance(high, str)
                and len(low) == 1 and len(high) == 1):
            return None                             # §51.4.4 NOTE requires size-1 endpoints
        start = ord(low) + (1 if constraint.lower_open else 0)
        stop = ord(high) - (1 if constraint.upper_open else 0)
        return frozenset(chr(code) for code in range(start, stop + 1))
    if isinstance(constraint, Union):
        found = [_alphabet_of(part) for part in constraint.parts]
        if any(item is None for item in found):
            return None
        return frozenset().union(*found)
    if isinstance(constraint, Intersection):
        found = [_alphabet_of(part) for part in constraint.parts]
        found = [item for item in found if item is not None]
        if not found:
            return None
        result = found[0]
        for item in found[1:]:
            result &= item
        return result
    return None


# --- the effective constraints the encoding rules ask for --------------------------------

def effective_value_constraint(constraint: Constraint | None) -> Bounds:
    """X.696 §8.2.7 — the smallest integer range including every permitted value.

    `(None, None)` when there is no OER-visible bound, which is the case both for an
    unconstrained type and for an extensible one.
    """
    return UNBOUNDED if constraint is None else constraint.value_bounds()


def effective_size_constraint(constraint: Constraint | None) -> Bounds:
    """X.696 §8.2.8 — the same, over the lengths of a string / SET OF / SEQUENCE OF."""
    return UNBOUNDED if constraint is None else constraint.size_bounds()


def is_unsatisfiable(constraint: Constraint | None) -> bool:
    """True when the constraint's value set is provably EMPTY.

    An empty value set is a static fault in the same family as two SET components sharing
    a tag: no value of the type can ever be written, so every use of it is dead. Verifier
    law R24 rejects it for that reason. Only the arithmetic case is decided here —
    `(10..1)` or an intersection with no overlap — because deciding emptiness in general
    would require evaluating every constraint form, and a false ACCEPT is much safer than
    a false REJECT for a rule that refuses a schema.
    """
    if constraint is None:
        return False
    if isinstance(constraint, Extensible):
        # The marker says later versions may add values, so the set is not empty even if
        # today's root admits nothing.
        return False
    for bounds in (constraint.value_bounds(), constraint.size_bounds()):
        low, high = bounds
        if low is not None and high is not None and low > high:
            return True
    if isinstance(constraint, Size):
        low, _ = constraint.size_bounds()
        if low is not None and low < 0:
            return True                             # a negative length is unsatisfiable
    alphabet = constraint.alphabet()
    if alphabet is not None and not alphabet:
        return True
    if isinstance(constraint, Intersection):
        return any(is_unsatisfiable(part) for part in constraint.parts)
    return False


def require_satisfiable(constraint: Constraint | None, where: str) -> None:
    if is_unsatisfiable(constraint):
        raise Asn1Error(
            f"{where}: constraint {constraint} permits no value at all (X.680 49); a "
            f"type with an empty value set can never be encoded")


__all__ = ["Bounds", "Constraint", "Extensible", "Intersection", "PermittedAlphabet",
           "Size", "SingleValue", "UNBOUNDED", "Union", "ValueRange",
           "effective_size_constraint", "effective_value_constraint", "is_unsatisfiable",
           "require_satisfiable"]
