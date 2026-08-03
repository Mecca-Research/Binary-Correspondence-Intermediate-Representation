"""X.692 clause 24: the nineteen `#TRANSFORM` procedures.

A transform makes the transmitted value a declared function of the abstract one. That is the
mechanism the §6 reduction gate was reopened for, so getting each one exactly right matters
more than usual: a transform that is nearly correct produces a well-formed document of a
different value, which is the failure mode no schema check catches.

The tests below are organized by what can go wrong rather than by clause number. Three
patterns recur:

* **A rule with two directions**, where an implementation naturally checks one. §24.10.10.2
  requires distinct *characters*; §24.11.6.2 requires distinct characters **and** distinct
  bitstrings, and the asymmetry is deliberate.
* **A default that is not the obvious one.** §24.8.1 defaults `int-to-bits` to
  `twos-complement`, so an object relying on the default and holding a positive value still
  gets a sign bit.
* **A clause that says something surprising and means it.** §24.7.13 pre-fixes padding to a
  representation that already carries the sign, so `-7` in four characters is `00-7`.
"""

from bcir.asn1.ecn_props import IntForm, Pattern
from bcir.asn1.ecn_transform import (
    RESULT_SIZE_FIXED_TO_MAX, RESULT_SIZE_VARIABLE, BitsToBits, BitsToChar, BitsToCompositeBits,
    BitsToInt, BitToBits, BoolToBool, BoolToInt, CharsToCompositeChar, CharToBits,
    Composite, CompositeBitsToBits, CompositeBitsToOctets, CompositeCharToChars, IntOp,
    IntToBits, IntToBool, IntToChars, IntToInt, OctetsToCompositeBits, TransformChain,
)
from bcir.asn1.tags import Asn1Error


def _refuses(citation: str, build):
    """Run `build` and assert it refuses citing `citation`."""
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return str(error)
    raise AssertionError(f"expected a refusal citing {citation}")


# --- §24.8 int-to-bits: the four size arms and the sign-aware pad -------------------------

def test_int_to_bits_defaults_to_twos_complement_which_costs_a_sign_bit():
    """§24.8.1: `&int-to-bits-encoded-as ... DEFAULT twos-complement`.

    The default is the one that surprises: an object that says nothing about the form and
    carries only positive values still spends a bit on the sign, because §24.8.11's minimality
    rule for two's complement forbids "two successive leading zero bits".
    """
    assert IntToBits().apply(5) == (0, 1, 0, 1)
    assert IntToBits(encoded_as=IntForm.POSITIVE_INT).apply(5) == (1, 0, 1)


def test_the_three_size_arms_are_not_interchangeable():
    """§24.8.13, §24.8.14 and §24.8.15 give three different widths for one value."""
    assert IntToBits(encoded_as=IntForm.POSITIVE_INT,
                     size=RESULT_SIZE_VARIABLE).apply(5) == (1, 0, 1)
    assert IntToBits(encoded_as=IntForm.POSITIVE_INT, size=1, unit=8).apply(5) == (
        0, 0, 0, 0, 0, 1, 0, 1)
    # §24.8.15: the smallest MULTIPLE OF that holds any value of the class. 300 needs nine
    # bits, so two octets — not one, and not the nine bits the value alone would take.
    wide = IntToBits(encoded_as=IntForm.POSITIVE_INT, size=RESULT_SIZE_FIXED_TO_MAX, unit=8,
                     bounds=(0, 300))
    assert len(wide.apply(5)) == 16


def test_padding_a_negative_value_repeats_its_sign_bit_and_not_zero():
    """§24.8.17: a two's-complement encoding "shall have bits prefixed **equal in value to
    the original leading bit**".

    Zero-extending would change the sign, which is the whole point of the clause naming the
    fill rather than saying "padded". -5 is `1011`; widened to eight bits it must stay -5.
    """
    bits = IntToBits(size=1, unit=8).apply(-5)
    assert bits == (1, 1, 1, 1, 1, 0, 1, 1)
    assert IntToBits(size=1, unit=8).inverse(bits) == -5
    # And the positive form fills with zero, per the same clause.
    assert IntToBits(encoded_as=IntForm.POSITIVE_INT, size=1, unit=8).apply(5)[0] == 0


def test_a_value_too_wide_for_a_fixed_size_is_refused_rather_than_truncated():
    """§24.8.16: "encoders shall not encode such values"."""
    _refuses("24.8.16", lambda: IntToBits(encoded_as=IntForm.POSITIVE_INT, size=4).apply(255))


def test_positive_int_refuses_a_negative_value_by_the_clause_that_forbids_it():
    """§24.8.12, which is a specification error rather than a two's-complement fallback."""
    _refuses("24.8.12", lambda: IntToBits(encoded_as=IntForm.POSITIVE_INT).apply(-1))


def test_fixed_to_max_without_bounds_is_refused_at_construction():
    """§24.8.8 requires "the source class has both lower and upper bounds"; without them
    there is no widest encoding to size against, so the object is wrong when it is written."""
    _refuses("24.8.8", lambda: IntToBits(size=RESULT_SIZE_FIXED_TO_MAX))


# --- §24.4-§24.7 the boolean and character transforms --------------------------------------

def test_bool_to_bool_has_exactly_one_setting_and_is_its_own_inverse():
    """§24.4.5: "There is only one value for BOOL-TO-BOOL, AS logical:not"."""
    assert BoolToBool().apply(True) is False
    assert BoolToBool().apply(BoolToBool().apply(True)) is True
    assert BoolToBool().reversible(True)          # §24.4.6


def test_bool_to_int_maps_the_way_the_setting_says_and_not_the_way_c_does():
    """§24.5.5: `true-zero` produces 0 for TRUE. The default is `true-one`.

    Worth pinning because `true-zero` is the one that reads backwards to anyone who has
    written C, and a specification choosing it is describing hardware that asserts low.
    """
    assert BoolToInt().apply(True) == 1 and BoolToInt().apply(False) == 0
    assert BoolToInt(true_zero=True).apply(True) == 0
    assert BoolToInt(true_zero=True).apply(False) == 1


def test_int_to_bool_is_reversible_only_with_two_single_valued_lists():
    """§24.6.9: "reversible if and only if both TRUE-IS and FALSE-IS are set, and they each
    specify a single integer value".

    Every other setting maps many integers onto two booleans. The transform is still legal —
    it is a perfectly good encoder-side test — but a decoder cannot choose which integer it
    was, so `reversible()` reports False and `inverse` refuses.
    """
    exact = IntToBool(true_is=(7,), false_is=(0,))
    assert exact.apply(7) is True and exact.apply(0) is False
    assert exact.reversible(True) and exact.inverse(True) == 7

    for lossy in (IntToBool(), IntToBool(zero_true=True), IntToBool(true_is=(1, 2, 3)),
                  IntToBool(true_is=(1, 2), false_is=(0,))):
        assert not lossy.reversible(True), lossy
        _refuses("24.6.9", lambda spec=lossy: spec.inverse(True))


def test_the_two_int_to_bool_lists_must_be_disjoint_and_cover_what_is_encoded():
    """§24.6.8 states both halves: the lists "shall be disjoint", and a value in neither is
    an error for which "encoders shall not generate encodings"."""
    _refuses("24.6.8", lambda: IntToBool(true_is=(1, 2), false_is=(2, 3)))
    _refuses("24.6.8", lambda: IntToBool(true_is=(1,), false_is=(0,)).apply(9))


def test_int_to_chars_pads_in_front_of_the_sign_because_the_clause_says_pre_fixed():
    """§24.7.13 pre-fixes the pad to §24.7.9's representation, which already carries the sign.

    So `-7` in four characters with zero padding is `00-7`. That is not what any printf
    produces, and it is what the clause specifies — nothing in clause 24 says the pad goes
    after the sign, and saying so would have taken a sentence.
    """
    assert IntToChars(size=4).apply(-7) == "00-7"
    assert IntToChars(size=4, pad_with_spaces=True).apply(7) == "   7"
    assert IntToChars(size=4).apply(7) == "0007"
    # The decoder is liberal about which spelling it reads; the encoder is not.
    for spelling in ("00-7", "-007"):
        assert IntToChars(size=4).inverse(spelling) == -7


def test_int_to_chars_writes_a_plus_only_when_asked():
    """§24.7.9: "If, and only if, PLUS-SIGN is set to true"."""
    assert IntToChars().apply(7) == "7"
    assert IntToChars(plus_sign=True).apply(7) == "+7"
    assert IntToChars(plus_sign=True).apply(-7) == "-7"


def test_a_number_too_long_for_its_character_field_is_refused():
    """§24.7.12, the character-side twin of §24.8.16."""
    _refuses("24.7.12", lambda: IntToChars(size=2).apply(1234))


# --- §24.9-§24.13 the bit transforms -------------------------------------------------------

def test_bits_to_int_is_never_usable_where_reversibility_is_required():
    """§24.9.6 says it flatly — not "under conditions", never.

    §24.9.3 says why in passing: the result has no bounds, and the source width is not
    recoverable, so `0011` and `11` both decode to 3 with nothing to tell them apart.
    """
    assert BitsToInt().apply((0, 0, 1, 1)) == 3
    assert BitsToInt().apply((1, 1)) == -1          # the default is twos-complement
    assert BitsToInt(decoded_assuming=IntForm.POSITIVE_INT).apply((1, 1)) == 3
    assert not BitsToInt().reversible((1, 1))
    _refuses("24.9.6", lambda: BitsToInt().inverse(3))


def test_char_to_bits_compact_indexes_the_alphabet_and_refuses_without_one():
    """§24.10.12.1 orders the effective permitted alphabet by ISO/IEC 10646 value and indexes
    from zero; §24.10.12 makes the absence of that alphabet a specification error.

    This is how PER shrinks a constrained string, and the refusal matters: an index into an
    alphabet that does not exist is not a number, so defaulting to the code point would
    silently produce `iso10646` under a `compact` label.
    """
    compact = CharToBits(alphabet="dcba", size=1, unit=2)
    assert compact.apply("a") == (0, 0)             # canonical order, not source order
    assert compact.apply("d") == (1, 1)
    assert compact.inverse((1, 1)) == "d"
    _refuses("24.10.12", lambda: CharToBits(encoded_as="compact"))


def test_char_to_bits_iso10646_defers_to_int_to_bits_as_the_clause_builds_it():
    """§24.10.11.4 defines the transform *in terms of* `INT-TO-BITS AS positive-int` with this
    transform's own SIZE and MULTIPLE OF, so the implementation composes rather than restates."""
    wide = CharToBits(encoded_as="iso10646", size=2, unit=8)
    assert wide.apply("A") == (0,) * 9 + (1, 0, 0, 0, 0, 0, 1)
    assert wide.inverse(wide.apply("A")) == "A"


def test_the_mapped_lists_are_checked_in_the_directions_each_clause_names():
    """§24.10.10.2 requires distinct CHARACTERS; §24.11.6.2 requires both lists distinct.

    The asymmetry is deliberate, not an oversight. Char-to-bits with two characters sharing a
    bitstring is lossy but well defined; bits-to-char with two identical source bitstrings is
    not a function at all.
    """
    _refuses("24.10.10.2", lambda: CharToBits(
        encoded_as="mapped", chars=("a", "a"), bit_values=((0,), (1,))))
    # Duplicate BITS is legal for char-to-bits, and reports itself irreversible.
    lossy = CharToBits(encoded_as="mapped", chars=("a", "b"), bit_values=((0,), (0,)))
    assert not lossy.reversible("a")
    _refuses("24.11.6.2", lambda: BitsToChar(
        decoded_assuming="mapped", chars=("a", "b"), bit_values=((0,), (0,))))


def test_a_character_outside_the_mapping_is_an_error_and_not_a_fallback():
    """§24.10.10.4 and §24.11.6.4, the same rule from both ends."""
    mapped = CharToBits(encoded_as="mapped", chars=("a",), bit_values=((0, 1),))
    _refuses("24.10.10.4", lambda: mapped.apply("z"))
    back = BitsToChar(decoded_assuming="mapped", chars=("a",), bit_values=((0, 1),))
    _refuses("24.11.6.4", lambda: back.apply((1, 1)))


def test_bits_to_char_refuses_a_code_point_the_clause_bounds_out():
    """§24.11.5: "It is an ECN specification error if the integer value exceeds 32767"."""
    assert BitsToChar().apply((1, 0, 0, 0, 0, 0, 1)) == "A"
    _refuses("24.11.5", lambda: BitsToChar().apply((1,) * 16))


def test_bit_to_bits_refuses_patterns_one_of_which_prefixes_the_other():
    """§24.12.9 is stronger than "different": one being an initial sub-string of the other is
    also an error.

    Two patterns that merely differ can still be undecodable from a stream — reaching `011`
    with patterns `01` and `011`, a decoder cannot tell one pattern followed by something
    from the other. This is the check a "they're not equal" test would pass.
    """
    ok = BitToBits(zero_pattern=Pattern.from_bits("00"), one_pattern=Pattern.from_bits("11"))
    assert ok.apply(1) == (1, 1) and ok.apply(0) == (0, 0)
    assert ok.inverse((1, 1)) == 1
    _refuses("24.12.9", lambda: BitToBits(zero_pattern=Pattern.from_bits("01"),
                                          one_pattern=Pattern.from_bits("011")))
    _refuses("24.12.9", lambda: BitToBits(zero_pattern=Pattern.from_bits("1"),
                                          one_pattern=Pattern.from_bits("1")))
    _refuses("24.12.7", lambda: BitToBits(zero_pattern=Pattern.any_of_length(3)))


def test_bits_to_bits_needs_distinct_sources_to_be_a_function_and_distinct_results_to_invert():
    """§24.13.7 and §24.13.11 are two different properties of the same table.

    Duplicate sources make the transform ill-defined, so that is refused at construction.
    Duplicate results only make it irreversible, which is legal and reported.
    """
    table = BitsToBits(source_values=((0, 0), (1, 1)), result_values=((1, 0), (0, 1)))
    assert table.apply((1, 1)) == (0, 1) and table.inverse((0, 1)) == (1, 1)
    assert table.reversible((0, 0))
    _refuses("24.13.7", lambda: BitsToBits(source_values=((0,), (0,)),
                                           result_values=((1,), (0,))))
    lossy = BitsToBits(source_values=((0,), (1,)), result_values=((1,), (1,)))
    assert not lossy.reversible((0,))
    _refuses("24.13.10", lambda: table.apply((0, 1)))


# --- §24.14-§24.19 the composites -----------------------------------------------------------

def test_the_composite_constructors_and_collapsers_round_trip():
    """§24.14/§24.17, §24.15/§24.18 and §24.16/§24.19 are three matched pairs."""
    chars = CharsToCompositeChar().apply("hi")
    assert chars == Composite(("h", "i"), "char")
    assert CompositeCharToChars().apply(chars) == "hi"

    bits = BitsToCompositeBits(unit=4).apply((1, 0, 1, 0, 1, 1, 1, 1))
    assert bits.elements == ((1, 0, 1, 0), (1, 1, 1, 1)) and bits.unit == 4
    assert CompositeBitsToBits().apply(bits) == (1, 0, 1, 0, 1, 1, 1, 1)

    octets = OctetsToCompositeBits().apply(b"\xAB\xCD")
    assert octets.unit == 8 and len(octets.elements) == 2
    assert CompositeBitsToOctets().apply(octets) == b"\xAB\xCD"


def test_a_composite_carries_its_own_unit_which_is_why_the_round_trip_works():
    """§24.18.5's NOTE: the unit "is specified in the transform that produced the bitstring
    composite, and [is] associated with that composite".

    §24.19.1 then checks it rather than using it — a composite of 4-bit elements is not an
    octetstring, and turning it into one would silently repack the bits.
    """
    nibbles = BitsToCompositeBits(unit=4).apply((1, 0, 1, 0, 1, 1, 1, 1))
    _refuses("24.19.1", lambda: CompositeBitsToOctets().apply(nibbles))


def test_a_source_that_is_not_a_whole_number_of_units_is_refused():
    """§24.15.6 makes it an error rather than a short final element."""
    _refuses("24.15.6", lambda: BitsToCompositeBits(unit=4).apply((1, 0, 1)))


def test_a_value_transform_maps_over_a_composite_elementwise():
    """§24.4.4's rule, stated identically by every value transform and implemented once.

    This is the property that makes the composite family useful: build a composite, run an
    ordinary value transform over it, collapse it again. Checking it on one transform checks
    the shared implementation, which is the whole reason it is shared.
    """
    composite = BitsToCompositeBits(unit=4).apply((0, 0, 1, 1, 0, 1, 0, 1))
    ints = BitsToInt(decoded_assuming=IntForm.POSITIVE_INT).apply(composite)
    assert ints.elements == (3, 5)
    assert ints.kind == "int" and ints.unit == 4      # the unit survives the mapping

    flags = IntToBool(zero_true=True).apply(ints)
    assert flags.elements == (False, False) and flags.kind == "bool"


def test_a_chain_composes_a_constructor_a_value_transform_and_a_collapser():
    """§24.2.4.1: "the source of a following #TRANSFORM encoding object shall be the result of
    the preceding" — which is what makes the three groups compose into one pipeline.

    Here: split an octetstring into 8-bit elements, invert every bit through a substitution
    table, and put it back. That is a real thing a legacy format does, and no rule in the
    fixed candidate set can express it.
    """
    invert = BitsToBits(
        source_values=tuple(tuple((n >> s) & 1 for s in range(7, -1, -1)) for n in range(256)),
        result_values=tuple(
            tuple(((~n) & 0xFF) >> s & 1 for s in range(7, -1, -1)) for n in range(256)))
    chain = TransformChain((OctetsToCompositeBits(), invert, CompositeBitsToOctets()))
    assert chain.apply(b"\x00\xFF\xA5") == b"\xFF\x00\x5A"


def test_reversibility_of_a_composite_is_the_conjunction_over_its_elements():
    """Clause 24's conditions are about abstract values, so a composite is reversible exactly
    when every element of it is — `divide:4` over (40, 41) is not reversible because 41 is
    not a multiple of 4, and reporting the composite as reversible would lose that."""
    composite = Composite((40, 41), "int")
    divide = IntToInt(op=IntOp.DIVIDE, operand=4)
    assert divide.reversible(Composite((40, 80), "int"))
    assert not divide.reversible(composite)
