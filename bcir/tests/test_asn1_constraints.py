"""X.680 clause 49–51 subtype constraints (roadmap phase B).

A constraint restricts a type's *value set*. DER never sees it — X.690 encodes a value the
same way whether or not a constraint admitted it — so until now the front-end consumed
constraints and threw them away. OER and PER are the opposite: **the encoding is chosen
from the constraint**, and `INTEGER (0..255)` is a quarter the octets of unconstrained
`INTEGER` for the same abstract value.

Three things are pinned here, in rough order of how easy they are to get wrong:

1. **The effective constraint** (X.696 §8.2.7/§8.2.8) — the smallest range including every
   permitted value, after the set arithmetic of clause 49 has been reduced.
2. **Extensibility removes bounds.** `(0..255, ...)` is *not* OER-visible (§8.2.2 g), so it
   encodes as though unbounded. This is the rule most likely to be implemented as an
   optimisation and thereby broken.
3. **DER does not move.** Adding a constraint to a type must not change one octet of its
   DER, or the whole "additive" premise of the ASN.1 rail fails.
"""

from __future__ import annotations

from bcir.asn1.codec import Strictness
from bcir.asn1.constraints import (
    Extensible,
    Intersection,
    PermittedAlphabet,
    SingleValue,
    Size,
    Union,
    ValueRange,
    effective_size_constraint,
    effective_value_constraint,
    is_unsatisfiable,
)
from bcir.asn1.oer import decode_oer, encode_oer
from bcir.asn1.schema import Primitive, SequenceOf
from bcir.asn1.tags import Asn1Error, Universal
from bcir.asn1.tlv import encode_tlv
from bcir.frontends.asn1 import Asn1SemanticError, compile_module, parse_module, print_module
from bcir.frontends.asn1.parser import Parser

_INT = Universal.INTEGER
_IA5 = Universal.IA5_STRING
_OCTETS = Universal.OCTET_STRING


def _parse_constraint(text: str):
    return Parser(text)._constraint()


# --- the effective constraint (X.696 8.2.7 / 8.2.8) -------------------------------------


def test_value_range_endpoints_including_min_max_and_the_open_marker():
    """§51.4. `None` is MIN/MAX; `<` excludes the endpoint (§51.4.3), which for an integer
    is just the adjacent closed bound — and the encoding only ever sees the closed form."""
    assert ValueRange(0, 255).value_bounds() == (0, 255)
    assert ValueRange(None, 255).value_bounds() == (None, 255)
    assert ValueRange(0, None).value_bounds() == (0, None)
    assert ValueRange(None, None).value_bounds() == (None, None)
    assert ValueRange(0, 256, lower_open=True, upper_open=True).value_bounds() == (1, 255)


def test_a_union_widens_to_the_smallest_enclosing_range_rather_than_keeping_holes():
    """X.696 §8.2.7 asks for the least and greatest PERMITTED value, so `(0..3 | 100..103)`
    is encoded over `0..103`. An encoding is one field width — a hole in the value set is
    the verifier's business, not the encoder's."""
    union = Union((ValueRange(0, 3), ValueRange(100, 103)))
    assert union.value_bounds() == (0, 103)
    assert union.permits(2) and union.permits(101)
    assert not union.permits(50)  # ...but the value set still has the hole


def test_an_intersection_tightens_to_the_narrower_bound_on_each_side():
    both = Intersection((ValueRange(0, 10), ValueRange(5, 20)))
    assert both.value_bounds() == (5, 10)
    assert Intersection((ValueRange(0, 10), ValueRange(None, None))).value_bounds() == (0, 10)


def test_size_reports_through_size_bounds_and_leaves_the_value_unconstrained():
    """§51.5.2 applies SIZE to strings and SET OF / SEQUENCE OF. A SIZE says nothing about
    what the elements ARE, which is why it must not leak into the value bounds."""
    size = Size(ValueRange(1, 64))
    assert size.size_bounds() == (1, 64)
    assert size.value_bounds() == (None, None)
    assert effective_size_constraint(size) == (1, 64)
    assert effective_value_constraint(size) == (None, None)


def test_a_permitted_alphabet_collects_its_characters():
    """§51.7 `FROM ("0".."9")`. The endpoints are CHARACTER STRING values of the parent
    type, not integers (§51.4.4 NOTE requires them to be size 1)."""
    digits = PermittedAlphabet(ValueRange("0", "9"))
    assert digits.alphabet() == frozenset("0123456789")
    assert digits.permits("2026") and not digits.permits("20x6")


# --- extensibility removes the bounds ---------------------------------------------------


def test_an_extensible_constraint_is_not_oer_visible():
    """X.696 §8.2.2 g) — THE rule to get right.

    The extension marker says the value set may grow in a later version of the protocol.
    An encoder that sized a field from today's root bounds would emit octets a future peer
    cannot read, so an extensible type encodes as though it had no bounds at all. Treating
    the root as usable would look like a harmless optimisation and would be a wire-format
    incompatibility.
    """
    extensible = Extensible(ValueRange(0, 255))
    assert extensible.value_bounds() == (None, None)
    assert effective_value_constraint(extensible) == (None, None)
    assert Extensible(Size(ValueRange(1, 4))).size_bounds() == (None, None)
    # ...and the encoding is the unbounded one, octet for octet.
    bounded = Primitive(_INT, "INTEGER", ValueRange(0, 255))
    marked = Primitive(_INT, "INTEGER", extensible)
    plain = Primitive(_INT, "INTEGER")
    assert encode_oer(bounded, 5) == b"\x05"
    assert encode_oer(marked, 5) == encode_oer(plain, 5) == b"\x01\x05"


# --- the encoding actually narrows (X.696 10, 14, 27) -----------------------------------


def test_an_integers_word_width_comes_from_its_effective_constraint():
    """§10.3 (a non-negative lower bound: unsigned) and §10.4 (otherwise: signed).

    The split is whether a lower bound EXISTS and is non-negative — not whether the
    bounds happen to be small. `(MIN..255)` has a tight upper bound and still takes the
    variable-size signed form, because nothing bounds it below.
    """
    cases = {
        None: b"\x01\x05",  # §10.4 e) length + variable signed
        ValueRange(0, 255): b"\x05",  # §10.3 a) one octet unsigned
        ValueRange(0, 65535): b"\x00\x05",  # §10.3 b) two octets
        ValueRange(0, 1 << 32): b"\x00\x00\x00\x00\x00\x00\x00\x05",  # §10.3 d) eight
        ValueRange(-128, 127): b"\x05",  # §10.4 a) one octet signed
        ValueRange(None, 255): b"\x01\x05",  # no lower bound -> §10.4 e)
        ValueRange(0, None): b"\x01\x05",  # no upper bound -> §10.3 e)
    }
    for constraint, expected in cases.items():
        kind = Primitive(_INT, "INTEGER", constraint)
        assert encode_oer(kind, 5) == expected, (constraint, encode_oer(kind, 5).hex())
        assert decode_oer(kind, expected) == 5, constraint


def test_a_signed_fixed_width_integer_round_trips_negative_values():
    kind = Primitive(_INT, "INTEGER", ValueRange(-128, 127))
    for value in (-128, -1, 0, 127):
        assert decode_oer(kind, encode_oer(kind, value)) == value


def test_a_value_outside_its_constraints_word_is_refused_rather_than_truncated():
    """A silent wrap here would produce octets that decode to a different value — the
    worst possible failure for a format whose whole purpose is byte-exact agreement."""
    kind = Primitive(_INT, "INTEGER", ValueRange(0, 255))
    try:
        encode_oer(kind, 256)
        raise AssertionError("256 was encoded into a one-octet word")
    except Asn1Error as exc:
        assert "does not fit" in str(exc), exc


def test_a_fixed_size_octet_string_drops_its_length_determinant():
    """§14.1: when the effective size constraint's bounds are IDENTICAL the length is
    implied. A range is not enough — only an exact length lets a decoder find the end."""
    fixed = Primitive(_OCTETS, "OCTET STRING", Size(SingleValue(4)))
    ranged = Primitive(_OCTETS, "OCTET STRING", Size(ValueRange(1, 4)))
    assert encode_oer(fixed, b"\x01\x02\x03\x04") == b"\x01\x02\x03\x04"
    assert encode_oer(ranged, b"\x01\x02\x03\x04") == b"\x04\x01\x02\x03\x04"
    assert decode_oer(fixed, b"\x01\x02\x03\x04") == b"\x01\x02\x03\x04"


def test_a_fixed_size_known_multiplier_string_drops_its_length_but_utf8_never_does():
    """§27.2 applies only to a KNOWN-MULTIPLIER type (§27.1). UTF8String is excluded
    because a character costs 1..4 octets there, so the character count never implies the
    octet count — the length determinant is not redundant and cannot be dropped."""
    ia5 = Primitive(_IA5, "IA5String", Size(SingleValue(3)))
    assert encode_oer(ia5, "abc") == b"abc"
    assert decode_oer(ia5, b"abc") == "abc"
    utf8 = Primitive(Universal.UTF8_STRING, "UTF8String", Size(SingleValue(3)))
    assert encode_oer(utf8, "abc") == b"\x03abc"


def test_a_size_constrained_sequence_of_still_carries_its_quantity_field():
    """§17.2 has no fixed-size shortcut: the quantity field is always present, because a
    SEQUENCE OF's occurrences are not fixed-width even when their number is."""
    kind = SequenceOf(
        Primitive(_INT, "INTEGER", ValueRange(0, 255)), "SEQUENCE OF INTEGER", Size(SingleValue(2))
    )
    assert encode_oer(kind, [1, 2]) == b"\x01\x02\x01\x02"
    assert decode_oer(kind, b"\x01\x02\x01\x02") == [1, 2]


# --- DER must not move ------------------------------------------------------------------


def test_adding_a_constraint_changes_no_der_octet():
    """The premise of the whole rail: a constraint restricts the value set, and X.690
    encodes a value the same way regardless. If this ever failed, every digest taken over
    a DER projection would depend on a schema detail that is invisible on the wire."""
    plain = Primitive(_INT, "INTEGER")
    for constraint in (
        ValueRange(0, 255),
        Extensible(ValueRange(0, 255)),
        Size(ValueRange(1, 4)),
        SingleValue(5),
        Union((ValueRange(0, 3), ValueRange(100, 103))),
    ):
        constrained = Primitive(_INT, "INTEGER", constraint)
        for value in (0, 5, 255):
            assert encode_tlv(constrained.encode(value)) == encode_tlv(plain.encode(value))


def test_the_streampack_projection_is_unchanged_by_the_constraint_machinery():
    from bcir.asn1.streampack import MODULE, pack_to_value
    from bcir.examples import PROGRAMS
    from bcir.gem import hydrate
    from bcir.kbcir import optimize
    from bcir.kbcir.cost import TargetProfile, Theta

    host, theta = TargetProfile.x86_avx512(), Theta.cool()
    # The module is SHIPPED inside bcir.asn1, so it is read from the package
    # rather than through a path relative to the working directory: the old
    # spelling resolved only when the test ran from the repository root, and
    # failed in the very wheel that ships the file it opens.
    from bcir.asn1 import STREAMPACK_MODULE, module_source

    compiled = compile_module(module_source(STREAMPACK_MODULE)).module
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        value = pack_to_value(hydrate(module, optimize(module, host, theta)))
        assert MODULE.encode("StreamPack", value) == compiled.encode("StreamPack", value), name


# --- the notation parses ----------------------------------------------------------------


def test_every_constraint_form_the_model_represents_parses_with_the_right_bounds():
    cases = {
        "(0..255)": ((0, 255), (None, None)),
        "(0<..<256)": ((1, 255), (None, None)),
        "(MIN..MAX)": ((None, None), (None, None)),
        "(5)": ((5, 5), (None, None)),
        "(1 | 2 | 3)": ((1, 3), (None, None)),
        "(0..10 ^ 5..20)": ((5, 10), (None, None)),
        "(SIZE (1..64))": ((None, None), (1, 64)),
        "(SIZE (4))": ((None, None), (4, 4)),
        "(0..255, ...)": ((None, None), (None, None)),
    }
    for text, (value_bounds, size_bounds) in cases.items():
        built = _parse_constraint(text)
        assert built is not None, text
        assert effective_value_constraint(built) == value_bounds, text
        assert effective_size_constraint(built) == size_bounds, text


def test_a_constraint_form_the_model_cannot_represent_leaves_the_type_unconstrained():
    """The SAFE direction. An unrepresented constraint must not narrow the encoding: the
    length-prefixed form carries every value the narrow form could, so falling back is
    lossless on the wire. Narrowing on a guess would not be."""
    for text in ("(WITH COMPONENTS { a PRESENT })", '(PATTERN "[0-9]#(3)")', "(0..255 EXCEPT 128)"):
        assert _parse_constraint(text) is None, text


def test_constraints_survive_the_round_trip_law():
    """Including the SIZE of a sequence-of, which must print between the keyword and OF
    (§51.5) -- printing it after the element type would re-parse as a constraint on the
    ELEMENT and silently change what the module means."""
    text = (
        "M DEFINITIONS ::= BEGIN\n"
        "  Small ::= INTEGER (0..255)\n"
        "  Ext ::= INTEGER (0..255, ...)\n"
        "  Names ::= SEQUENCE SIZE (1..64) OF UTF8String\n"
        '  Digits ::= IA5String (FROM ("0".."9"))\n'
        "  Fixed ::= OCTET STRING (SIZE (4))\n"
        "END\n"
    )
    node = parse_module(text)
    printed = print_module(node)
    assert parse_module(printed) == node, printed
    # The alphabet endpoints must keep their quotes: `FROM (0..9)` would re-parse as an
    # integer range and the alphabet would silently become None.
    assert 'FROM ("0".."9")' in printed, printed
    assert "SEQUENCE (SIZE (1..64)) OF" in printed, printed


def test_a_parsed_module_reaches_the_encoder_with_its_constraints_attached():
    compiled = compile_module(
        "M DEFINITIONS ::= BEGIN\n"
        "  Small ::= INTEGER (0..255)\n"
        "  Names ::= SEQUENCE SIZE (1..64) OF UTF8String\n"
        "END\n"
    ).module
    assert encode_oer(compiled.types["Small"], 5) == b"\x05"
    assert effective_size_constraint(compiled.types["Names"].constraint) == (1, 64)


# --- the R24 gate: an empty value set ---------------------------------------------------


def test_an_unsatisfiable_constraint_is_detected():
    """Verifier law R24 rejects these on the MLIR rail; the oracle refuses them at compile
    time. An empty value set means no value of the type can ever be encoded, so every use
    of it is dead -- a static fault in the same family as two components sharing a tag."""
    assert is_unsatisfiable(ValueRange(10, 1))
    assert is_unsatisfiable(Size(ValueRange(5, 2)))
    assert is_unsatisfiable(Size(ValueRange(-1, 4)))
    assert is_unsatisfiable(Intersection((ValueRange(0, 3), ValueRange(100, 103))))
    assert is_unsatisfiable(PermittedAlphabet(Union((SingleValue("a"), SingleValue("b"))))) is False
    # Not unsatisfiable: unconstrained, a normal range, or an EXTENSIBLE one -- the marker
    # says later versions may add values, so today's empty root is not the whole story.
    assert not is_unsatisfiable(None)
    assert not is_unsatisfiable(ValueRange(0, 255))
    assert not is_unsatisfiable(Extensible(ValueRange(10, 1)))


def test_the_front_end_refuses_a_module_with_an_empty_value_set():
    try:
        compile_module("M DEFINITIONS ::= BEGIN\n  T ::= INTEGER (10..1)\nEND\n")
        raise AssertionError("a type with an empty value set was accepted")
    except (Asn1Error, Asn1SemanticError) as exc:
        assert "permits no value" in str(exc), exc


def test_a_constrained_type_is_a_claim_geometry():
    """The BCIR-specific payoff the roadmap points at: the constraint IS the width.

    `INTEGER (0..255)` is an 8-bit lane and `INTEGER (0..65535)` a 16-bit one, and that is
    readable straight off the effective constraint rather than inferred from the octets
    after the fact. This is the information `realize.candidates_for` needs in order to
    price a decode, which is what makes phase B feed the optimizer instead of sitting
    beside it.
    """
    widths = {}
    for high in (255, 65535, 1 << 32):
        kind = Primitive(_INT, "INTEGER", ValueRange(0, high))
        widths[high] = len(encode_oer(kind, 1))
    assert widths == {255: 1, 65535: 2, 1 << 32: 8}, widths
    # ...and the bound is available without encoding anything at all.
    low, high = effective_value_constraint(ValueRange(0, 65535))
    assert (low, high) == (0, 65535)
    assert (high - low + 1).bit_length() - 1 == 16
