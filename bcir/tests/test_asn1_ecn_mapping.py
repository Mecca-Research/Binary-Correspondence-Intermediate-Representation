"""X.692 clause 19: which abstract value gets encoded, before any bits are chosen.

**Clause 19 answers a different question from every other clause in the ECN half.** Clauses
21–25 say how a value becomes bits; §19.1.1 says which value the fields of one encoding
structure hand to the fields of another. That is what lets an ASN.1 `INTEGER` be encoded as a
*concatenation* of fields, or a four-way CHOICE as a compact integer — neither of which is a
transform on bits.

Three of its sentences are traps, and this file exists mostly for them:

1. **§19.5.5 orders `TRUE` before `FALSE`.** Python's `sorted([True, False])` is `[False, True]`
   and every other language agrees with Python. An ordering built on the host's comparison is
   backwards, and backwards *silently* — both directions produce well-formed encodings that
   differ only in what the bits mean.
2. **§19.4.6 requires reversibility where the value path does not.** Table 6 lets `modulo:n` be
   legal and lossy when it is *encoding*; a mapping a decoder cannot undo loses the value.
3. **§19.5.11 and §19.5.12 are not symmetric, and neither is an error.** A short destination
   means some values cannot be encoded; a long one means some encodings never occur. A rail
   that refused either would reject conforming specifications.
"""

from bcir.asn1.ecn_mapping import (
    AbstractValueOrdering, AlternativesOrdering, BooleanOrdering, DistributionEntry,
    ExplicitValues, IntToBits, IntToBitsEntry, IntegerOrdering, MatchingFields, NullOrdering,
    SingleComponentOrdering, TransformMapping, ValueDistribution,
)
from bcir.asn1.ecn_transform import IntOp, IntToInt, TransformChain
from bcir.asn1.tags import Asn1Error


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


# --- §19.5's orderings ------------------------------------------------------------------------

def test_boolean_orders_true_before_false_which_no_language_does():
    """§19.5.5: "Classes in the boolean category are defined to have `TRUE` before `FALSE`."

    This is the single most dangerous sentence in clause 19. Every programming language orders
    booleans the other way, so an implementation that reached for the host's sort produces
    encodings whose two values are swapped — well-formed, decodable, and wrong. The test is
    written as an explicit pair rather than a property so that it fails loudly if anyone
    "simplifies" the ordering into `sorted(...)`.
    """
    ordering = BooleanOrdering()
    assert (ordering.value_at(0), ordering.value_at(1)) == (True, False)
    assert ordering.index_of(True) == 0
    assert ordering.index_of(False) == 1
    # And the direction Python would have given, stated so the contrast is on the record.
    assert sorted([True, False]) == [False, True]


def test_an_integer_ordering_needs_a_lower_bound_to_have_a_first_value():
    """§19.5.4.2 a) gives an integer class an ordering only when "constrained to have a finite
    lower bound", and §19.5.10 turns on there being "a defined first value in each ordering".
    An unbounded-below integer has none, so it is refused rather than started from zero."""
    _refuses("19.5.4.2", lambda: IntegerOrdering(low=None))
    assert IntegerOrdering(0, 255).count() == 256
    assert IntegerOrdering(low=0).count() is None       # §19.5.4.2 a)'s infinite case
    assert IntegerOrdering(-3, -1).value_at(0) == -3


def test_alternatives_concatenate_their_orderings_in_textual_order():
    """§19.5.7: "the (ordered) abstract values from the textually first alternative, followed by
    those from the textually second alternative, and so on"."""
    ordering = AlternativesOrdering(alternatives=(
        ("nothing", NullOrdering()), ("flag", BooleanOrdering()),
        ("small", IntegerOrdering(0, 1))))
    assert ordering.count() == 5
    assert [ordering.value_at(index) for index in range(5)] == [
        ("nothing", None), ("flag", True), ("flag", False), ("small", 0), ("small", 1)]
    assert ordering.index_of(("small", 1)) == 4


def test_only_the_last_alternative_may_be_infinite():
    """§19.5.4.2 b): an alternatives ordering is infinite only if "all of the alternatives
    except the last are defined to have a finite set of ordered values, and the last
    alternative is defined to have an infinite set". An infinite alternative in the middle puts
    every later one at a position no index reaches."""
    _refuses("19.5.4.2", lambda: AlternativesOrdering(alternatives=(
        ("unbounded", IntegerOrdering(low=0)), ("after", BooleanOrdering()))))
    tail = AlternativesOrdering(alternatives=(
        ("flag", BooleanOrdering()), ("unbounded", IntegerOrdering(low=0))))
    assert tail.count() is None
    assert tail.value_at(2) == ("unbounded", 0)


def test_a_concatenation_orders_by_its_single_non_optional_component():
    """§19.5.8, and §19.5.3 admits no wider shape. The obvious generalization — order a
    concatenation lexicographically over all its components — is not what the clause says and
    would give a different mapping for every two-field structure."""
    ordering = SingleComponentOrdering(name="v", component=IntegerOrdering(10, 12))
    assert ordering.count() == 3
    assert ordering.value_at(0) == {"v": 10}
    assert ordering.index_of({"v": 12}) == 2
    _refuses("19.5.3", lambda: ordering.index_of({"v": 10, "other": 1}))


# --- §19.5's mapping ---------------------------------------------------------------------------

def test_ordered_values_compacts_a_choice_into_a_contiguous_integer():
    """§19.5.1's headline use: "the compaction of integer values or enumerations into a
    contiguous set of integer values". §19.5.9 does it by position."""
    source = AlternativesOrdering(alternatives=(
        ("nothing", NullOrdering()), ("flag", BooleanOrdering())))
    mapping = AbstractValueOrdering(source=source, target=IntegerOrdering(0, 2))
    assert [mapping.map(value) for value in
            (("nothing", None), ("flag", True), ("flag", False))] == [0, 1, 2]
    assert mapping.unmap(2) == ("flag", False)


def test_unequal_orderings_are_reported_and_not_refused():
    """§19.5.11 and §19.5.12 are explicit that neither direction is an error.

    A short destination "is not an error. However, the ECN specification will be unable to
    encode some of the abstract values" — a fact to report, since the clause asks the specifier
    to note it in a comment. A long destination means "there may be some ECN-defined encodings
    that have no ASN.1 abstract value, and will never be generated". Refusing either would
    reject conforming specifications, so only *using* a position past the end fails.
    """
    narrow = AbstractValueOrdering(source=IntegerOrdering(0, 9), target=IntegerOrdering(0, 3))
    assert narrow.loses_values()
    assert narrow.map(3) == 3
    _refuses("19.5.11", lambda: narrow.map(4))

    wide = AbstractValueOrdering(source=IntegerOrdering(0, 3), target=IntegerOrdering(0, 9))
    assert not wide.loses_values()
    assert wide.map(3) == 3

    # An infinite source into a finite target loses values; the reverse does not.
    assert AbstractValueOrdering(source=IntegerOrdering(low=0),
                                 target=IntegerOrdering(0, 9)).loses_values()
    assert not AbstractValueOrdering(source=IntegerOrdering(0, 9),
                                     target=IntegerOrdering(low=0)).loses_values()


# --- §19.2, §19.3, §19.4 --------------------------------------------------------------------

def test_explicit_values_map_source_to_target_and_refuse_the_unlisted():
    """§19.2.6 fixes the direction — `MappedValue1` is the source, `MappedValue2` the target.

    §19.1.6's NOTE 1 is why an unlisted value is refused rather than passed through: a partial
    mapping "is not an error. It is a **constraint imposed by ECN** on the values that can be
    used by the application", and a constraint that let unlisted values through would not be
    one."""
    mapping = ExplicitValues(pairs=((1, 100), (2, 200)))
    assert mapping.map(1) == 100
    assert mapping.map(2) == 200
    _refuses("19.2", lambda: mapping.map(3))
    _refuses("19.2.5", lambda: ExplicitValues())
    _refuses("19.1.6", lambda: ExplicitValues(pairs=((1, 100), (1, 200))))


def test_explicit_values_does_not_confuse_a_boolean_with_one():
    """Python makes `True == 1`, so a mapping listing both would otherwise answer the first.
    Two categories, two values; §19.2.6 governs each side by its own class."""
    mapping = ExplicitValues(pairs=((True, "yes"), (1, "one")))
    assert mapping.map(True) == "yes"
    assert mapping.map(1) == "one"


def test_matching_fields_keeps_the_added_determinant_fields_out_of_the_value():
    """§19.3.1: the target "has fields corresponding to the components of the type, but also has
    added fields for determinants".

    This is the `AUXILIARY` deviation seen from the clause's own side — the structure has a
    length field the ASN.1 type does not, and §19.3 is where that difference is declared rather
    than inferred. The added fields are *named*, because inferring them from "the value did not
    carry it" would turn a typo in a field name into a silent determinant.
    """
    mapping = MatchingFields(added=("len",))
    assert mapping.map({"a": 1, "b": 2}) == {"a": 1, "b": 2}
    assert mapping.target_fields(("a", "b")) == ("a", "b", "len")
    narrowing = MatchingFields(added=("len",), dropped=("b",))
    assert narrowing.map({"a": 1, "b": 2}) == {"a": 1}
    assert narrowing.target_fields(("a", "b")) == ("a", "len")
    _refuses("19.3.5", lambda: mapping.map(7))


def test_a_transform_mapping_requires_reversibility_where_the_value_path_does_not():
    """§19.4.6: "It is an ECN specification or application error if any `Transform` … is not
    reversible for the abstract value being mapped."

    Table 6 permits a lossy transform when it is *encoding* a value — `modulo:n` is "Never
    reversible" and still legal there. The asymmetry is the same one §22.3.2.3 and §22.8.2.4
    impose on determinants: a decoder has to recover what the application supplied, and a
    mapping that cannot be undone loses the value rather than encoding it.
    """
    scaled = TransformMapping(chain=TransformChain((IntToInt(op=IntOp.DIVIDE, operand=4),)))
    assert scaled.map(40) == 10
    assert scaled.unmap(10) == 40
    # §24.3's divide is reversible exactly when the value is a multiple, per Table 6.
    _refuses("19.4.6", lambda: scaled.map(41))
    _refuses("19.4.3", lambda: TransformMapping())


# --- §19.6 and §19.7 --------------------------------------------------------------------------

def test_a_distribution_sends_each_range_to_its_own_field():
    """§19.6.1: it "takes ranges of values from an encoding class in the integer category,
    mapping each range to a different integer field". One value lands in one field; §19.6.1's
    remaining "fields which receive no abstract values shall have their values determined by
    the application of determinants"."""
    mapping = ValueDistribution(entries=(
        DistributionEntry(field_name="small", low=0, high=99),
        DistributionEntry(field_name="exact", value=1000),
        DistributionEntry(field_name="large", remainder=True)))
    assert mapping.map(5) == {"small": 5}
    assert mapping.map(1000) == {"exact": 1000}
    assert mapping.map(70000) == {"large": 70000}
    assert mapping.fields() == ("small", "exact", "large")


def test_remainder_is_once_and_last_and_a_value_reaches_one_field():
    """§19.6.10: REMAINDER "shall only be used once for the last `SelectedValues`". §19.6.11: "A
    value shall not be mapped to more than one target field" — while permitting, in the same
    sentence, that "several `SelectedValues` may have the same destination", so an overlap that
    agrees about the field is legal and only a disagreement is a fault."""
    _refuses("19.6.10", lambda: ValueDistribution(entries=(
        DistributionEntry(field_name="rest", remainder=True),
        DistributionEntry(field_name="small", low=0, high=9))))
    _refuses("19.6.11", lambda: ValueDistribution(entries=(
        DistributionEntry(field_name="a", low=0, high=9),
        DistributionEntry(field_name="b", low=5, high=20))))
    _refuses("19.6.11", lambda: ValueDistribution(entries=(
        DistributionEntry(field_name="a", low=0, high=9),
        DistributionEntry(field_name="b", value=5))))
    # Two SelectedValues with ONE destination: legal, and §19.6.11 says so outright.
    agreeing = ValueDistribution(entries=(
        DistributionEntry(field_name="a", low=0, high=9),
        DistributionEntry(field_name="a", value=50)))
    assert agreeing.map(50) == {"a": 50}
    _refuses("19.6.8", lambda: DistributionEntry(field_name="a", low=9, high=9))
    _refuses("19.6.6", lambda: DistributionEntry(field_name="a", value=1, remainder=True))
    _refuses("19.6", lambda: ValueDistribution(entries=(
        DistributionEntry(field_name="a", low=0, high=9),)).map(99))


def test_int_to_bits_is_huffmans_shape_and_ranges_must_advance_together():
    """§19.7.1's NOTE: "intended to support self-delimiting encodings of integers, **such as
    Huffman encodings**". So codes of different lengths across entries are the point — a short
    code for a common value, a long one for a rare one.

    §19.7.8 then defines "contiguous" for the bitstrings *within* a range, in two parts that are
    both load-bearing: "a) They are all the same length in bits. b) When interpreted as a
    positive integer value, the corresponding integer values are contiguous and increasing."
    """
    mapping = IntToBits(entries=(
        IntToBitsEntry(value=0, bits=(0,)),                                  # the common one
        IntToBitsEntry(value=1, bits=(1, 0, 0), high=4, high_bits=(1, 1, 1))))
    assert mapping.map(0) == (0,)
    assert mapping.map(1) == (1, 0, 0)
    assert mapping.map(3) == (1, 1, 0)
    assert mapping.map(4) == (1, 1, 1)
    assert mapping.unmap((1, 1, 0)) == 3
    # §19.7.9: everything else "cannot be encoded".
    _refuses("19.7.9", lambda: mapping.map(5))

    # §19.7.8 a) — the ends of a range have to be the same width.
    _refuses("19.7.8 a", lambda: IntToBitsEntry(value=0, bits=(0, 0), high=3,
                                                high_bits=(0, 1, 1)))
    # §19.7.8 b) — and the two ranges have to span the same number of values.
    _refuses("19.7.8 b", lambda: IntToBitsEntry(value=0, bits=(0, 0), high=3,
                                                high_bits=(1, 0)))
    _refuses("19.7.7", lambda: IntToBitsEntry(value=5, bits=(0,), high=1, high_bits=(1,)))
    _refuses("19.7", lambda: IntToBits(entries=(
        IntToBitsEntry(value=0, bits=(0,), high=3, high_bits=(1, 1)),
        IntToBitsEntry(value=2, bits=(1, 0, 1)))))
    _refuses("19.7.5", lambda: IntToBits())


def test_a_bit_that_is_not_a_bit_is_refused():
    _refuses("19.7", lambda: IntToBitsEntry(value=0, bits=(0, 2)))
    _refuses("19.7.10", lambda: IntToBitsEntry(value=0, bits=()))
