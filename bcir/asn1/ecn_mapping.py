"""X.692 clause 19: mapping values from one encoding structure's fields to another's.

**This is the piece that makes ECN a metaprogramming layer rather than a wire-format
description.** Everything in clauses 21–25 answers "how do these bits come out"; clause 19
answers a different question — *which abstract value is being encoded at all*. §19.1.1: it
"specifies the syntax for mapping values (and tag numbers) to be encoded by the fields of one
encoding structure … to the fields of another encoding structure". So an ASN.1 `INTEGER` can be
encoded as the *concatenation* of two fields, or a four-alternative CHOICE as a compact
integer, and neither is a transform on bits.

§19.1.7 gives **six** mappings, not the five its own table of contents suggests —
`MappingIntToBits` is in the production and is numbered §19.7:

| clause | notation | what it maps |
|---|---|---|
| §19.2 | `VALUES {a TO b, …}` | named value pairs, one primitive class to another |
| §19.3 | `FIELDS` | components to like-named components, extra fields left to determinants |
| §19.4 | `TRANSFORMS {…}` | through clause 24's transforms |
| §19.5 | `ORDERED VALUES` | by position in each class's value ordering |
| §19.6 | `DISTRIBUTION {…}` | integer ranges, each to a different field |
| §19.7 | `TO BITS {…}` | integer values to bitstrings — Huffman's shape |

THREE READINGS THAT THE OBVIOUS IMPLEMENTATION GETS WRONG.

* **§19.5.5 orders `TRUE` before `FALSE`.** Every programming language orders booleans the
  other way — Python's `sorted([True, False])` is `[False, True]` — so an ordering built on the
  host language's comparison is exactly backwards, and silently: it produces well-formed
  encodings with the two values swapped. This is the single most dangerous sentence in the
  clause and `BooleanOrdering` exists to state it once.
* **§19.4.6 requires reversibility, where the value path does not.** Table 6 lets `modulo:n` be
  legal and lossy when a transform is *encoding* a value; here "It is an ECN specification or
  application error if any `Transform` … is not reversible for the abstract value being
  mapped". A mapping a decoder cannot undo is not a mapping. Same asymmetry §22.3.2.3 and
  §22.8.2.4 impose on determinants, for the same reason.
* **§19.5.11 and §19.5.12 are not symmetric.** A destination ordering *shorter* than the source
  "is not an error" — it means some abstract values cannot be encoded, which §19.5.11 asks the
  specifier to note in a comment. A destination *longer* than the source is also fine and means
  some encodings will never be generated. Neither is a fault, so neither is refused here; what
  *is* refused is asking to map a value that falls off the end.

WHAT IS NOT HERE. §19.2's Table 5 fixes the ASN.1 value notation per category — `bstring` or
`hstring` for bitstring, `SignedNumber` for integer, and so on. That is a *surface syntax*
concern and belongs with the parser; this module takes values already parsed. §19.3's
de-referencing rules (§19.3.3) resolve structure names until a constructor is reached, which
needs the ELM's view of generated structures — clause 12, the next slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ecn_transform import TransformChain
from .tags import Asn1Error

# --- §19.5's orderings: what "the n-th abstract value" means, per category ------------------
#
# §19.5.3 restricts this mapping to classes that de-reference to null, boolean, integer or
# real, or to a construction using the alternatives category, or to a concatenation "which has
# a single non-optional component". §19.5.4 then splits those into the finite and the infinite
# cases, and §19.5.10 states the invariant that makes the whole thing work: "the above rules
# ensure that there is a defined first value in each ordering, and a defined next value. There
# need not be a defined last value (either or both sets may be infinite)."


class Ordering:
    """An enumeration of a class's abstract values, in §19.5's order.

    `count()` returns `None` for §19.5.4.2's infinite orderings rather than a large number,
    because the difference is load-bearing: §19.5.11 and §19.5.12 both turn on comparing the
    two orderings' sizes, and an infinity that pretended to be finite would make a mapping
    that silently truncates look like one that fits.
    """

    def count(self) -> "int | None":
        raise NotImplementedError

    def index_of(self, value) -> int:
        raise NotImplementedError

    def value_at(self, index: int):
        raise NotImplementedError

    def _check_index(self, index: int) -> int:
        if index < 0:
            raise Asn1Error(f"ECN: §19.5.9 maps by position; {index} is not one")
        size = self.count()
        if size is not None and index >= size:
            raise Asn1Error(
                f"ECN: §19.5.9 — position {index} is past the end of an ordering with {size} "
                f"abstract values; §19.5.11 permits the destination to be shorter, so the "
                f"specification is legal and this particular value simply cannot be encoded")
        return index


@dataclass(frozen=True)
class NullOrdering(Ordering):
    """§19.5.5: "Classes in the null category have a single abstract value"."""

    def count(self) -> int:
        return 1

    def index_of(self, value) -> int:
        if value is not None:
            raise Asn1Error(
                f"ECN: a class in the null category has one abstract value; got {value!r}")
        return 0

    def value_at(self, index: int):
        self._check_index(index)
        return None


@dataclass(frozen=True)
class BooleanOrdering(Ordering):
    """§19.5.5: "Classes in the boolean category are defined to have `TRUE` before `FALSE`."

    **Read that twice.** Every programming language orders booleans the other way, so an
    ordering derived from the host's comparison is backwards — and backwards *silently*, since
    both directions produce well-formed encodings that differ only in which value the bits
    mean. It is written out here, rather than expressed as `sorted(...)`, precisely so that no
    future reader can "simplify" it into the wrong order without deleting a citation.
    """

    def count(self) -> int:
        return 2

    def index_of(self, value) -> int:
        return 0 if value else 1

    def value_at(self, index: int):
        return self._check_index(index) == 0


@dataclass(frozen=True)
class IntegerOrdering(Ordering):
    """§19.5.5: "higher integer values following lower integer values", within §19.5.6's bounds.

    §19.5.4.1 c) makes a *bounded* integer finite and §19.5.4.2 a) makes one "constrained to
    have a finite lower bound" infinite — so a lower bound is what an integer ordering needs to
    have a first value at all, and `low=None` is refused rather than started from zero.
    """

    low: "int | None" = 0
    high: "int | None" = None

    def __post_init__(self) -> None:
        if self.low is None:
            raise Asn1Error(
                "ECN: §19.5.4.2 a) gives an integer class an ordering only when it is "
                "\"constrained to have a finite lower bound\"; without one there is no first "
                "abstract value, and §19.5.10's \"defined first value\" is what the mapping "
                "counts from")
        if self.high is not None and self.high < self.low:
            raise Asn1Error(
                f"ECN: an integer ordering runs from {self.low} upwards; {self.high} is below "
                f"its lower bound")

    def count(self) -> "int | None":
        return None if self.high is None else self.high - self.low + 1

    def index_of(self, value) -> int:
        if value < self.low or (self.high is not None and value > self.high):
            raise Asn1Error(
                f"ECN: §19.5.6 — {value} is outside the bounds "
                f"{self.low}..{'MAX' if self.high is None else self.high} that define this "
                f"ordering")
        return value - self.low

    def value_at(self, index: int):
        return self.low + self._check_index(index)


@dataclass(frozen=True)
class AlternativesOrdering(Ordering):
    """§19.5.7: "the (ordered) abstract values from the textually first alternative, followed
    by those from the textually second alternative, and so on".

    §19.5.4.2 b) is the constraint that makes this well-defined when something is infinite:
    an alternatives ordering is infinite only if "all of the alternatives except the last are
    defined to have a finite set of ordered values, and the last alternative … infinite". An
    infinite alternative in the middle would put every later alternative at an unreachable
    position, so it is refused rather than allowed to produce an ordering nothing can index.
    """

    #: `(alternative name, that alternative's ordering)`, in textual order.
    alternatives: tuple = ()

    def __post_init__(self) -> None:
        if not self.alternatives:
            raise Asn1Error(
                "ECN: §19.5.3 admits a construction in the alternatives category; one with no "
                "alternatives has no abstract values to order")
        for position, (name, ordering) in enumerate(self.alternatives[:-1]):
            if ordering.count() is None:
                raise Asn1Error(
                    f"ECN: §19.5.4.2 b) — only the LAST alternative may have an infinite "
                    f"ordering; {name!r} is at position {position} and is infinite, which "
                    f"would put every later alternative past a position no index reaches")

    def count(self) -> "int | None":
        total = 0
        for _name, ordering in self.alternatives:
            size = ordering.count()
            if size is None:
                return None
            total += size
        return total

    def index_of(self, value) -> int:
        """`value` is the `(alternative name, value)` pair a CHOICE actually is."""
        name, chosen = value
        base = 0
        for candidate, ordering in self.alternatives:
            if candidate == name:
                return base + ordering.index_of(chosen)
            size = ordering.count()
            if size is None:  # pragma: no cover - __post_init__ forbids a non-final infinity
                raise Asn1Error(
                    f"ECN: the alternative {candidate!r} has an infinite ordering and is not "
                    f"the last")
            base += size
        raise Asn1Error(
            f"ECN: {name!r} is not one of this construction's alternatives "
            f"({', '.join(candidate for candidate, _o in self.alternatives)})")

    def value_at(self, index: int):
        self._check_index(index)
        remaining = index
        for name, ordering in self.alternatives:
            size = ordering.count()
            if size is None or remaining < size:
                return (name, ordering.value_at(remaining))
            remaining -= size
        raise Asn1Error(  # pragma: no cover - _check_index already bounded this
            f"ECN: position {index} is past the end of this alternatives ordering")


@dataclass(frozen=True)
class SingleComponentOrdering(Ordering):
    """§19.5.8: a concatenation "that has a single non-optional component" orders by that
    component's ordering.

    §19.5.3 admits exactly this shape and no wider one, which is worth stating because the
    obvious generalization — order a concatenation lexicographically by its components — is
    *not* what the clause says, and would give a different mapping for every structure with
    two fields.
    """

    name: str = ""
    component: "Ordering | None" = None

    def __post_init__(self) -> None:
        if self.component is None:
            raise Asn1Error(
                "ECN: §19.5.8 orders a concatenation by its single non-optional component, "
                "and none is given")

    def count(self) -> "int | None":
        return self.component.count()

    def index_of(self, value) -> int:
        if isinstance(value, dict):
            if set(value) != {self.name}:
                raise Asn1Error(
                    f"ECN: §19.5.3 admits a concatenation with a single non-optional "
                    f"component; this value carries {sorted(value)}")
            value = value[self.name]
        return self.component.index_of(value)

    def value_at(self, index: int):
        return {self.name: self.component.value_at(index)}


# --- clause 19's six mappings ----------------------------------------------------------------

class ValueMapping:
    """§19.1.7's `ValueMapping` CHOICE: six ways to say which value gets encoded.

    §19.1.6 is what they all mean: "The encodings specified for values mapped to the target
    encoding class become the encodings of those values in the source encoding class." So a
    mapping is applied *before* any clause 21–25 machinery runs, and what the encoding objects
    then see is the target's value.
    """

    def map(self, value):
        raise NotImplementedError


@dataclass(frozen=True)
class ExplicitValues(ValueMapping):
    """§19.2's `VALUES {v1 TO v2, …}`: named pairs between two primitive bit-field classes.

    §19.2.6 fixes which is which — "`MappedValue1` shall be value notation governed by the
    source governor and `MappedValue2` … by the target governor" — so the pairs read
    source-to-target and never the reverse.

    §19.1.6's NOTE 1 is why an unmapped value is a refusal rather than a pass-through: "If the
    total ECN specification maps only some of the values from an ASN.1 type into encodings,
    that is not an error. It is a **constraint imposed by ECN** on the values that can be used
    by the application." A constraint that silently let unlisted values through would not be
    one.
    """

    #: `(source value, target value)`, in the order written.
    pairs: tuple = ()

    def __post_init__(self) -> None:
        if not self.pairs:
            raise Asn1Error(
                "ECN: §19.2.5 gives `VALUES` a non-empty list; a mapping with no pairs maps "
                "nothing and would refuse every value")
        sources = [source for source, _target in self.pairs]
        # Keyed by (type, value), not by value. §19.2.6 governs each side by its own encoding
        # class, and Python makes `True == 1` — so a specification listing a boolean and an
        # integer would otherwise look like one value mapped twice, and `map` would answer
        # whichever came first.
        keyed = [(type(source).__name__, source) for source in sources]
        if len(set(keyed)) != len(keyed):
            raise Asn1Error(
                f"ECN: a source value is mapped twice in {sources}; §19.1.6's NOTE 2 makes two "
                f"values sharing one encoding a specification error, and one value with two "
                f"encodings is not a mapping at all")

    def map(self, value):
        for source, target in self.pairs:
            if source == value and isinstance(source, bool) == isinstance(value, bool):
                return target
        raise Asn1Error(
            f"ECN: §19.2 maps {[source for source, _t in self.pairs]}, and {value!r} is not "
            f"among them; §19.1.6's NOTE 1 makes a partial mapping a constraint ECN imposes "
            f"on the values the application may use, not a pass-through")


@dataclass(frozen=True)
class MatchingFields(ValueMapping):
    """§19.3's `FIELDS`: components map to like-named components.

    §19.3.1 gives the purpose exactly: "to enable the encoding of an ASN.1 type to be defined
    as the encoding of an encoding structure that has fields corresponding to the components
    of the type, **but also has added fields for determinants**". So this is the mapping that
    lets a length field exist in the encoding structure without existing in the ASN.1 type —
    which is precisely the `AUXILIARY` deviation this rail currently states, seen from the
    other side.

    §19.3.6's NOTE restricts the source fields to "the top-level fields of a concatenation",
    and this refuses to look deeper rather than matching by path.
    """

    #: Target field names that take no source value — §19.3.1's "added fields for
    #: determinants". Named rather than inferred, because a typo in a source field name would
    #: otherwise silently become a determinant field.
    added: tuple = ()
    #: Source field names with no target, when the target is deliberately narrower.
    dropped: tuple = ()

    def map(self, value):
        if not isinstance(value, dict):
            raise Asn1Error(
                f"ECN: §19.3.5 makes `FIELDS` a mapping between two concatenations (or two "
                f"repetitions of them); {type(value).__name__} has no fields to match")
        return {name: carried for name, carried in value.items()
                if name not in self.dropped}

    def target_fields(self, source_fields: tuple) -> tuple:
        """The target's top-level field names, given the source's. §19.3.13's shape.

        The added fields come last because §19.3.1 calls them "added"; where they actually sit
        is the encoding *structure's* business (§16.5's textual order), not the mapping's.
        """
        return tuple(name for name in source_fields if name not in self.dropped) + self.added


@dataclass(frozen=True)
class TransformMapping(ValueMapping):
    """§19.4's `TRANSFORMS {…}`: map through clause 24's transforms.

    **§19.4.6 requires reversibility, and the value path does not.** Table 6 lets `modulo:n` be
    legal and lossy when a transform is encoding a value; here "It is an ECN specification or
    application error if any `Transform` in the `OrderedTransformList` is not reversible for
    the abstract value being mapped". The reason is the same one §22.3.2.3 and §22.8.2.4 give
    for determinants: a decoder has to get back what the application put in, and a mapping it
    cannot undo loses the value rather than encoding it.

    §19.4.5 restricts both ends to "the bitstring, boolean, characterstring, integer, or
    octetstring category" — the five categories clause 24's transforms are defined over.
    """

    chain: "TransformChain | None" = None

    def __post_init__(self) -> None:
        if self.chain is None:
            raise Asn1Error(
                "ECN: §19.4.3 gives `TRANSFORMS` an OrderedTransformList; an empty mapping "
                "would be the identity written the long way")

    def map(self, value):
        if not self.chain.reversible(value):
            raise Asn1Error(
                f"ECN: §19.4.6 — this mapping's transforms are not reversible for {value!r}, "
                f"so a decoder could not recover it. Table 6 permits a lossy transform when it "
                f"is ENCODING a value; a mapping that cannot be undone loses the value instead")
        return self.chain.apply(value)

    def unmap(self, value):
        """The decoder's direction, which §19.4.6's reversibility is what guarantees exists."""
        return self.chain.inverse(value)


@dataclass(frozen=True)
class AbstractValueOrdering(ValueMapping):
    """§19.5's `ORDERED VALUES`: map by position in each class's value ordering.

    §19.5.1 gives the two directions this serves: distributing "abstract values associated with
    simple encoding classes … into the fields of complex encoding structures", and the reverse
    — mapping a complex structure "to simple encoding classes such as `#INT`". Plus the one
    that turns up most: "the compaction of integer values or enumerations into a contiguous set
    of integer values".

    §19.5.9 is the whole operation: "The mapping is defined from the abstract values in the
    first encoding class to the abstract values in the second encoding class **by their
    position** in the above ordering."
    """

    source: "Ordering | None" = None
    target: "Ordering | None" = None

    def __post_init__(self) -> None:
        if self.source is None or self.target is None:
            raise Asn1Error(
                "ECN: §19.5.9 maps between two orderings by position, and one is missing")

    def map(self, value):
        return self.target.value_at(self.source.index_of(value))

    def unmap(self, value):
        return self.source.value_at(self.target.index_of(value))

    def loses_values(self) -> bool:
        """§19.5.11: whether the destination is shorter, so some source values cannot encode.

        Reported rather than refused. The clause is explicit that this "is not an error" and
        asks only that it "should be identified by comment"; a rail that refused it would
        reject conforming specifications.
        """
        source, target = self.source.count(), self.target.count()
        if source is None:
            return target is not None
        return target is not None and target < source


@dataclass(frozen=True)
class DistributionEntry:
    """One `Distribution` of §19.6.6: a value, a range, or `REMAINDER`, and a target field."""

    field_name: str = ""
    #: A single §19.6.6 `SelectedValue`.
    value: "int | None" = None
    #: A `DistributionRange`, inclusive at both ends per §19.6.9.
    low: "int | None" = None
    high: "int | None" = None
    #: §19.6.10's `REMAINDER`.
    remainder: bool = False

    def __post_init__(self) -> None:
        if not self.field_name:
            raise Asn1Error("ECN: §19.6.6 maps SelectedValues TO an identifier; none is given")
        forms = [self.value is not None, self.low is not None or self.high is not None,
                 self.remainder]
        if sum(1 for form in forms if form) != 1:
            raise Asn1Error(
                "ECN: §19.6.6's `SelectedValues` is one of a SelectedValue, a "
                "DistributionRange or REMAINDER — exactly one")
        if self.low is not None or self.high is not None:
            if self.low is None or self.high is None:
                raise Asn1Error(
                    "ECN: §19.6.6's DistributionRange has both a lower and an upper value")
            if not self.low < self.high:
                raise Asn1Error(
                    f"ECN: §19.6.8 — DistributionRangeValue1 shall be less than "
                    f"DistributionRangeValue2; got {self.low}..{self.high}")

    def holds(self, value: int) -> bool:
        if self.remainder:
            return True                       # §19.6.10: everything not distributed earlier
        if self.value is not None:
            return value == self.value
        return self.low <= value <= self.high


@dataclass(frozen=True)
class ValueDistribution(ValueMapping):
    """§19.6's `DISTRIBUTION {…}`: integer ranges, each to a different field.

    §19.6.1: it "takes ranges of values from an encoding class in the integer category, mapping
    each range to a different integer field in a more complex encoding structure. **Fields
    which receive no abstract values shall have their values determined by the application of
    determinants.**" That last sentence is the one that connects this clause to §21.3 — the
    fields nobody's value lands in are the auxiliary ones.

    §19.6.13 is the constraint a naive implementation misses: mapping into a field "whose
    presence depends on optionality or choice of alternatives … is not an error, but the
    optionality and choice of alternatives in the target (when encoding such values) shall be
    such that the encoding of the target **includes the target field**". So the distribution
    decides which alternative is taken, rather than being checked against one.
    """

    entries: tuple = ()

    def __post_init__(self) -> None:
        if not self.entries:
            raise Asn1Error("ECN: §19.6.6 gives `DISTRIBUTION` a non-empty list")
        for position, entry in enumerate(self.entries):
            if entry.remainder and position != len(self.entries) - 1:
                raise Asn1Error(
                    f"ECN: §19.6.10 — REMAINDER "
                    f"\"shall only be used once for the last SelectedValues\"; it is at "
                    f"position {position} of {len(self.entries)}")
        if sum(1 for entry in self.entries if entry.remainder) > 1:  # pragma: no cover
            raise Asn1Error("ECN: §19.6.10 — REMAINDER shall only be used once")
        self._check_disjoint()

    def _check_disjoint(self) -> None:
        """§19.6.11: "A value shall not be mapped to more than one target field."

        Note what the clause permits in the same sentence — "several `SelectedValues` may have
        the same destination" — so overlapping entries that agree about the field are legal and
        only a disagreement is a fault. Checking the *field* rather than the *value* is what
        makes that distinction.
        """
        singles: dict = {}
        ranges: list = []
        for entry in self.entries:
            if entry.remainder:
                continue
            if entry.value is not None:
                seen = singles.get(entry.value)
                if seen is not None and seen != entry.field_name:
                    raise Asn1Error(
                        f"ECN: §19.6.11 — {entry.value} is mapped to both {seen!r} and "
                        f"{entry.field_name!r}; a value shall not be mapped to more than one "
                        f"target field")
                singles[entry.value] = entry.field_name
                continue
            for low, high, name in ranges:
                if entry.low <= high and low <= entry.high and name != entry.field_name:
                    raise Asn1Error(
                        f"ECN: §19.6.11 — {max(low, entry.low)}..{min(high, entry.high)} is "
                        f"mapped to both {name!r} and {entry.field_name!r}")
            ranges.append((entry.low, entry.high, entry.field_name))
        for value, name in singles.items():
            for low, high, other in ranges:
                if low <= value <= high and other != name:
                    raise Asn1Error(
                        f"ECN: §19.6.11 — {value} is mapped to both {name!r} and {other!r}")

    def map(self, value: int) -> dict:
        """§19.6.9, as the single-field dictionary the target structure receives.

        One key, not many: §19.6.1 maps "each range to a **different** integer field", so one
        abstract value lands in exactly one of them, and the rest are §19.6.1's fields "which
        receive no abstract values" and get theirs from determinants.
        """
        for entry in self.entries:
            if entry.holds(value):
                return {entry.field_name: value}
        raise Asn1Error(
            f"ECN: §19.6 distributes no field to {value}; add a REMAINDER entry if the "
            f"specification means to accept everything else")

    def fields(self) -> tuple:
        """Every target field this distribution can reach, in the order written."""
        out: list = []
        for entry in self.entries:
            if entry.field_name not in out:
                out.append(entry.field_name)
        return tuple(out)


@dataclass(frozen=True)
class IntToBitsEntry:
    """One `MappedIntToBits` of §19.7.5: a single pair, or a range mapped to a bit range."""

    #: §19.7.10's `SingleIntValMap`, or the low end of an `IntValRangeMap`.
    value: int = 0
    #: The bits, most significant first.
    bits: tuple = ()
    #: For a range map: the last integer, and the last bitstring.
    high: "int | None" = None
    high_bits: tuple = ()

    def __post_init__(self) -> None:
        for bit in tuple(self.bits) + tuple(self.high_bits):
            if bit not in (0, 1):
                raise Asn1Error(f"ECN: §19.7 maps to BIT STRING values; {bit!r} is not a bit")
        if not self.bits:
            raise Asn1Error("ECN: §19.7.10 maps an integer TO a bitstring value; none is given")
        if (self.high is None) != (not self.high_bits):
            raise Asn1Error(
                "ECN: §19.7.7's IntValRangeMap takes a range of integers AND a range of "
                "bitstrings; one of the two upper ends is missing")
        if self.high is None:
            return
        if self.high < self.value:
            raise Asn1Error(
                f"ECN: §19.7.7 maps \"contiguous and increasing integer values\"; "
                f"{self.value}..{self.high} is not increasing")
        # §19.7.8's definition of contiguous, in full: "a) They are all the same length in
        # bits. b) When interpreted as a positive integer value, the corresponding integer
        # values are contiguous and increasing." Both halves are checked, because a range whose
        # ends differ in length denotes no set of bitstrings at all.
        if len(self.high_bits) != len(self.bits):
            raise Asn1Error(
                f"ECN: §19.7.8 a) — a contiguous range of bitstrings is \"all the same length "
                f"in bits\"; this one runs from {len(self.bits)} bits to "
                f"{len(self.high_bits)}")
        span = _int_of(self.high_bits) - _int_of(self.bits)
        if span != self.high - self.value:
            raise Asn1Error(
                f"ECN: §19.7.8 b) — the bitstring range spans {span + 1} values and the "
                f"integer range spans {self.high - self.value + 1}; the two have to advance "
                f"together for the mapping to be one-to-one")

    def holds(self, value: int) -> bool:
        return value == self.value if self.high is None else self.value <= value <= self.high

    def bits_for(self, value: int) -> tuple:
        if self.high is None:
            return tuple(self.bits)
        return _bits_of(_int_of(self.bits) + (value - self.value), len(self.bits))


@dataclass(frozen=True)
class IntToBits(ValueMapping):
    """§19.7's `TO BITS {…}`: integer values to bitstrings, one at a time or in ranges.

    §19.7.1's NOTE says what it is for: "This mapping is intended to support self-delimiting
    encodings of integers, **such as Huffman encodings**." That is why the bitstrings need not
    be the same length across entries — a short code for a common value and a long one for a
    rare one is the point — while §19.7.8 requires them to be equal-length *within* a range,
    since a range is a contiguous block of codes.

    §19.7.9 makes the mapping total over what it lists and closed otherwise: "Only values
    specified in the mapping are encodable. Other abstract values of the source are not mapped
    and cannot be encoded … It is an ECN or application error if such values are presented to
    an encoder."
    """

    entries: tuple = ()

    def __post_init__(self) -> None:
        if not self.entries:
            raise Asn1Error("ECN: §19.7.5 gives `TO BITS` a non-empty list")
        for index, entry in enumerate(self.entries):
            for other in self.entries[index + 1:]:
                low, high = entry.value, entry.value if entry.high is None else entry.high
                other_low = other.value
                other_high = other.value if other.high is None else other.high
                if low <= other_high and other_low <= high:
                    raise Asn1Error(
                        f"ECN: §19.7 maps {max(low, other_low)}..{min(high, other_high)} "
                        f"twice; an integer with two codes is not a mapping")

    def map(self, value: int) -> tuple:
        for entry in self.entries:
            if entry.holds(value):
                return entry.bits_for(value)
        raise Asn1Error(
            f"ECN: §19.7.9 — {value} is not among the values this mapping lists, and \"other "
            f"abstract values of the source are not mapped and cannot be encoded\"")

    def unmap(self, bits: tuple) -> int:
        """The decoder's direction. A prefix code makes this unambiguous; §19.7 does not
        *require* the codes to form one, so this matches whole bitstrings only."""
        wanted = tuple(bits)
        for entry in self.entries:
            width = len(entry.bits)
            if len(wanted) != width:
                continue
            offset = _int_of(wanted) - _int_of(entry.bits)
            span = 0 if entry.high is None else entry.high - entry.value
            if 0 <= offset <= span:
                return entry.value + offset
        raise Asn1Error(
            f"ECN: no §19.7 entry produces the bitstring "
            f"'{''.join(str(bit) for bit in wanted)}'B")


def _int_of(bits: tuple) -> int:
    """A BIT STRING as the positive integer §19.7.8 b) interprets it as."""
    value = 0
    for bit in bits:
        value = (value << 1) | (1 if bit else 0)
    return value


def _bits_of(value: int, width: int) -> tuple:
    return tuple((value >> shift) & 1 for shift in range(width - 1, -1, -1))


__all__ = [
    "AbstractValueOrdering", "AlternativesOrdering", "BooleanOrdering", "DistributionEntry",
    "ExplicitValues", "IntToBits", "IntToBitsEntry", "IntegerOrdering", "MatchingFields",
    "NullOrdering", "Ordering", "SingleComponentOrdering", "TransformMapping", "ValueDistribution",
    "ValueMapping",
]
