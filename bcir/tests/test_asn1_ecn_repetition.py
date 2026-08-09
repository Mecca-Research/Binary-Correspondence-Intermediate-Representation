"""X.692's repetition machinery, and the categories that turn out to depend on it.

**The finding that shaped this file.** §23.2's `#BITS`, §23.9's `#OCTETS` and §23.4's `#CHARS`
have **no `ENCODING-SPACE` group at all**. Their `WITH SYNTAX` gives pre-alignment, a start
pointer, `VALUE-REVERSAL`, `TRANSFORMS` and `REPETITION-ENCODING(S)` — so a string's size comes
from §22.7's repetition space, not from a stated width. The three string categories could not
be built before repetition was, which is why they arrive together.

§21.7.3 says the same thing from the other side: `RepetitionSpaceDetermination` "**replaces**
use of an encoding property of type `EncodingSpaceDetermination` in the encoding of
repetitions". Sibling, not subtype — which is why they are two enums with different members.
"""

from bcir.asn1.ecn_props import (
    UNIT_OCTET, UNIT_REPETITIONS, Comparison, Padding, Pattern,
    RepetitionSpaceDetermination, SizeBounds, SizeRangeCondition,
)
from bcir.asn1.ecn_user import (
    AuxIntSpec, BitFieldSpec, BitWriter, ConcatenationSpec, ConditionalRepetitionSpec, IntSpec, NullSpec,
    RepetitionSpace, RepetitionSpec, StringSpec, TagSpec, UserEncodingObject, encode_with_user,
)
from bcir.asn1.tags import Asn1Error

_OCTET = IntSpec(width=8)


def _concat(fields: dict, order: tuple, padding: tuple = ()) -> dict:
    return {"#C": UserEncodingObject("#C", ConcatenationSpec(
        fields=fields, order=order, padding=padding))}


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


# --- §21.13: the size sibling of §21.11, and where the two diverge -------------------------

def test_the_size_shapes_overlap_where_the_integer_shapes_partition():
    """§21.11.4's NOTE says "exactly one predicate will be satisfied". §21.13.4's says "Only
    the `fixed-size` case overlaps with other predicates" — the opposite claim.

    That is not a wording slip in either. A fixed size *is* an upper bound with a lower bound,
    so `SIZE(4)` genuinely satisfies both `ub-with-non-zero-lb` and `fixed-size`. An
    implementation that carried the integer sibling's exhaustiveness across would pick the
    wrong encoding for every fixed-size string, which is the common case rather than an edge.
    """
    assert SizeBounds(4, 4).shapes() == (
        SizeRangeCondition.UB_WITH_NON_ZERO_LB, SizeRangeCondition.FIXED_SIZE)
    assert SizeBounds(0, 0).shapes() == (
        SizeRangeCondition.UB_WITH_ZERO_LB, SizeRangeCondition.FIXED_SIZE)
    # And the non-fixed shapes still partition among themselves.
    for bounds in (SizeBounds(0, None), SizeBounds(0, 9), SizeBounds(2, None)):
        assert len(bounds.shapes()) == 1, bounds


def test_a_size_lower_bound_always_exists_so_the_shapes_test_zero_not_absence():
    """§21.13.4 a) turns on the lower bound being **zero**, where §21.11.4 a) turned on a
    lower bound *existing*. An X.680 size is `INTEGER (0..MAX)`-constrained, so there is
    always one — and the sibling's "no lower bound" case has no counterpart here."""
    assert SizeBounds(0, None).satisfies(SizeRangeCondition.NO_UB_WITH_ZERO_LB)
    assert SizeBounds(2, None).satisfies(SizeRangeCondition.NO_UB_WITH_NON_ZERO_LB)
    assert not SizeBounds(2, None).satisfies(SizeRangeCondition.NO_UB_WITH_ZERO_LB)


def test_the_comparison_rule_is_worded_identically_and_holds_both_ways():
    """§21.13.5 repeats §21.11.5 verbatim, so the same two-directional check applies."""
    bounds = SizeBounds(0, 9)
    assert bounds.satisfies(SizeRangeCondition.TEST_UPPER_BOUND, Comparison.EQUAL_TO, 9)
    _refuses("21.13.5", lambda: bounds.satisfies(SizeRangeCondition.TEST_RANGE))
    _refuses("21.13.5", lambda: bounds.satisfies(
        SizeRangeCondition.FIXED_SIZE, Comparison.EQUAL_TO, 4))


# --- §22.7: the repetition space ------------------------------------------------------------

def test_a_count_field_is_written_from_the_number_of_repetitions():
    """§21.7.4's `field-to-be-set`, over the auxiliary-field machinery §22.8.3.7 describes."""
    space = RepetitionSpace(reference="n", unit=UNIT_REPETITIONS)
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    objects = _concat({"n": AuxIntSpec(width=8),
                       "s": StringSpec(element=_OCTET, repetition=rep)}, ("n", "s"))
    assert encode_with_user(objects, "#C", {"s": [1, 2, 3]}) == bytes((3, 1, 2, 3))
    assert encode_with_user(objects, "#C", {"s": []}) == bytes((0,))


def test_the_count_unit_decides_whether_the_field_holds_elements_or_octets():
    """§21.1.5 admits `repetitions` as a `Unit` **only here**: "the associated count gives the
    number of repetitions in the encoding". Every other unit counts the space's size instead.

    A format carrying "how many items" and one carrying "how many octets" are both ordinary
    and are not the same number — so the element width has to differ from the unit for the
    distinction to show, which is what this uses 16-bit elements for.
    """
    wide = IntSpec(width=16)
    def encode(unit: int) -> bytes:
        space = RepetitionSpace(reference="n", unit=unit)
        rep = RepetitionSpec((ConditionalRepetitionSpec(element=wide, space=space),),
                             SizeBounds(0, None))
        objects = _concat({"n": AuxIntSpec(width=8),
                           "s": StringSpec(element=wide, repetition=rep)}, ("n", "s"))
        return encode_with_user(objects, "#C", {"s": [1, 2, 3]})

    assert encode(UNIT_REPETITIONS)[0] == 3      # three elements
    assert encode(UNIT_OCTET)[0] == 6            # six octets of them


def test_a_termination_pattern_ends_the_repetition_instead_of_a_count():
    """§22.7.1.1's `&termination-pattern` with §21.7.1's `pattern` determination.

    This is the NUL-terminated string, which is why the group carries a `Pattern` at all — and
    it needs no auxiliary field, so a format with no length prefix is expressible.
    """
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.PATTERN,
                            termination_pattern=Pattern.from_octets(b"\x00"))
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    objects = _concat({"s": StringSpec(element=_OCTET, repetition=rep)}, ("s",))
    assert encode_with_user(objects, "#C", {"s": [72, 73]}) == b"HI\x00"
    assert encode_with_user(objects, "#C", {"s": []}) == b"\x00"


def test_a_pattern_determination_with_no_pattern_is_refused():
    _refuses("22.7.1.1", lambda: RepetitionSpace(
        determination=RepetitionSpaceDetermination.PATTERN))


def test_the_two_unbuilt_determinations_each_name_what_they_would_need():
    """§21.7.1 gives eight; six are built. The other two are refused by name rather than
    approximated, and the message says what is missing — a continuation flag lives *inside*
    the repeated element (§21.7.6/§21.7.7), so it needs the element's own structure.

    Two determinations have left this list. `handle` went when §22.9's identification handles
    were built (§22.7.3.11 gives the encoder no action for it), and `container` went when
    §21.7.8's containment did — both are now covered by their own test files."""
    for determination, citation in (
        (RepetitionSpaceDetermination.FLAG_TO_BE_SET, "21.7.6"),
        (RepetitionSpaceDetermination.FLAG_TO_BE_USED, "21.7.7"),
    ):
        _refuses(citation, lambda d=determination: RepetitionSpace(determination=d))
    RepetitionSpace(determination=RepetitionSpaceDetermination.HANDLE)
    RepetitionSpace(determination=RepetitionSpaceDetermination.CONTAINER, reference="c")


def test_field_to_be_used_checks_the_applications_count():
    """§21.7.5, the repetition twin of §21.3.5: the encoder verifies rather than writes."""
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.FIELD_TO_BE_USED,
                            reference="n", unit=UNIT_REPETITIONS)
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    objects = _concat({"n": IntSpec(width=8),
                       "s": StringSpec(element=_OCTET, repetition=rep)}, ("n", "s"))
    assert encode_with_user(objects, "#C", {"n": 2, "s": [1, 2]}) == bytes((2, 1, 2))
    _refuses("21.7.5", lambda: encode_with_user(objects, "#C", {"n": 9, "s": [1, 2]}))


# --- §23.13: selecting an encoding by the size bounds ---------------------------------------

def test_a_fixed_size_needs_no_length_field_and_the_schema_decides_that():
    """§23.13.3.1 selects the first object whose conditions hold, over §21.13's size bounds.

    So one object set encodes `SIZE(2)` with no length prefix and an unbounded size with one —
    from the schema, with no value involved. That is the same schema-directed selection §23.6
    gives integers, and it is what makes an ECN object set reusable across types.
    """
    fixed = ConditionalRepetitionSpec(
        element=_OCTET,
        space=RepetitionSpace(RepetitionSpaceDetermination.NOT_NEEDED),
        conditions=((SizeRangeCondition.FIXED_SIZE, None, None),))
    counted = ConditionalRepetitionSpec(
        element=_OCTET, space=RepetitionSpace(reference="n", unit=UNIT_REPETITIONS))

    assert RepetitionSpec((fixed, counted), SizeBounds(2, 2)).select() is fixed
    assert RepetitionSpec((fixed, counted), SizeBounds(0, None)).select() is counted


def test_a_conditional_object_after_an_unconditional_one_is_dead_and_refused():
    """§23.13.2.3: "If an encoding object ... is defined using IF or IF-ALL, then all
    **preceding** encoding objects in that list shall be defined using IF or IF-ALL."

    An unconditional object matches everything, so anything after it can never be selected.
    The clause forbids writing the dead alternative rather than leaving it to be discovered by
    someone wondering why their second encoding never fires.
    """
    unconditional = ConditionalRepetitionSpec(
        element=_OCTET, space=RepetitionSpace(RepetitionSpaceDetermination.NOT_NEEDED))
    conditional = ConditionalRepetitionSpec(
        element=_OCTET, space=RepetitionSpace(RepetitionSpaceDetermination.NOT_NEEDED),
        conditions=((SizeRangeCondition.FIXED_SIZE, None, None),))
    assert RepetitionSpec((conditional, unconditional))          # this order is fine
    _refuses("23.13.2.3", lambda: RepetitionSpec((unconditional, conditional)))


def test_matching_no_conditional_encoding_is_a_refusal_not_a_default():
    """§23.13.3.1, worded exactly like §23.6.3.1 for integers."""
    only = ConditionalRepetitionSpec(
        element=_OCTET, space=RepetitionSpace(RepetitionSpaceDetermination.NOT_NEEDED),
        conditions=((SizeRangeCondition.FIXED_SIZE, None, None),))
    _refuses("23.13.3.1", lambda: RepetitionSpec((only,), SizeBounds(0, None)).select())


# --- §23.2/§23.4/§23.9: the string categories -----------------------------------------------

def test_a_string_category_has_no_encoding_space_and_says_so():
    """The finding this file is built around: §23.2.1's `#BITS` `WITH SYNTAX` has no
    `ENCODING-SPACE`. A string with no repetition encoding has no size at all."""
    _refuses("23.2.1", lambda: StringSpec(element=_OCTET))


def test_value_reversal_reverses_elements_and_is_not_bit_reversal():
    """§23.2.1's `&value-reversal BOOLEAN DEFAULT FALSE`, distinct from §22.12's group.

    One sends the string backwards; the other reverses bits within a unit. A format doing
    either is ordinary, a format doing both is possible, and conflating them would silently
    produce the wrong one — so ECN spells them with two properties and so does this.
    """
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.PATTERN,
                            termination_pattern=Pattern.from_octets(b"\x00"))
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    forward = _concat({"s": StringSpec(element=_OCTET, repetition=rep)}, ("s",))
    backward = _concat(
        {"s": StringSpec(element=_OCTET, repetition=rep, value_reversal=True)}, ("s",))
    assert encode_with_user(forward, "#C", {"s": [72, 73]}) == b"HI\x00"
    assert encode_with_user(backward, "#C", {"s": [72, 73]}) == b"IH\x00"


def test_a_transform_runs_before_the_repetition_splits_the_value():
    """§23.2.1 orders `TRANSFORMS` ahead of `REPETITION-ENCODINGS`, and the composite family
    is what makes the pair work: a transform turns the string into §24.2.1's composite, whose
    elements the repetition then writes one at a time."""
    from bcir.asn1.ecn_transform import OctetsToCompositeBits

    space = RepetitionSpace(determination=RepetitionSpaceDetermination.PATTERN,
                            termination_pattern=Pattern.from_bits("1" * 8))
    rep = RepetitionSpec((ConditionalRepetitionSpec(
        element=BitFieldSpec(width=8), space=space),), SizeBounds(0, None))
    # §24.16 splits b"\xAB" into one 8-bit composite element; the repetition writes it. The
    # element writer is a BIT FIELD rather than an IntSpec, because after a transform the
    # elements ARE bits — an IntSpec would be choosing bits for an integer it does not have.
    spec = StringSpec(element=BitFieldSpec(width=8), repetition=rep,
                      transform=OctetsToCompositeBits())
    writer = BitWriter()
    spec.write(b"\xAB", writer)
    assert writer.octets() == b"\xAB\xFF"


# --- §23.8 and §23.15: the two categories that needed nothing new ---------------------------

def test_a_null_encoding_is_all_padding_because_there_is_nothing_to_encode():
    """§23.8's `#NUL`. X.680's NULL carries no information, so every bit of its encoding
    space is padding — the one category where `VALUE-PADDING` *is* the value encoding."""
    writer = BitWriter()
    NullSpec(width=8).write(None, writer)
    assert writer.octets() == b"\x00"
    writer = BitWriter()
    NullSpec(width=8, padding=Padding.ONE).write(None, writer)
    assert writer.octets() == b"\xFF"
    # The commonest case: a NULL that occupies no bits at all.
    writer = BitWriter()
    NullSpec(width=0).write(None, writer)
    assert writer.bit_length == 0


def test_a_tag_is_a_prefix_that_composes_with_what_it_tags():
    """§20.2: a category's defined syntax may be used "preceded by one or more instances of a
    class in the tag category".

    So a tag is not a category of value — it is a prefix that composes, which is exactly how
    BER's identifier octet relates to its contents. Here `30 05` is the tag then the integer.
    """
    writer = BitWriter()
    TagSpec(width=8, number=0x30, tagged=IntSpec(width=8)).write(5, writer)
    assert writer.octets() == bytes((0x30, 0x05))
    # And a tag alone is legal, which §20.2's "one or more instances" implies.
    writer = BitWriter()
    TagSpec(width=8, number=0x30).write(None, writer)
    assert writer.octets() == bytes((0x30,))


# --- §21.7.6 and §21.7.7: the continuation flag inside the repeated element ----------------


def _flagged(determination, flag_spec):
    """A repetition whose end is announced by a flag that is a field OF each element."""
    element = ConcatenationSpec(fields={"more": flag_spec, "v": _OCTET}, order=("more", "v"))
    space = RepetitionSpace(determination=determination, reference="more")
    return StringSpec(
        element=element,
        repetition=RepetitionSpec((ConditionalRepetitionSpec(element=element, space=space),),
                                  SizeBounds(0, None)))


def test_flag_to_be_set_writes_a_continuation_flag_into_every_element():
    """§21.7.6, the last determination in clause 21.7 to be built.

    "A REFERENCE to a field **that is part of the repeated element**, and that will be set by
    the encoder to identify the last element of the repetition ... from a boolean value which
    is false if the element is the last in the repetition, and is true otherwise."

    So the flag says *continues*, not *ends*, and the last element is the one carrying zero.
    The octets are the assertion because that polarity is the whole content of the clause and
    an implementation that inverted it would still round-trip against itself.
    """
    writer = BitWriter()
    _flagged(RepetitionSpaceDetermination.FLAG_TO_BE_SET, AuxIntSpec(width=8)).write(
        [{"v": 7}, {"v": 8}, {"v": 9}], writer)
    assert writer.octets() == bytes((1, 7, 1, 8, 0, 9))

    # One element is the degenerate case and it is the last one, so its flag is zero.
    writer = BitWriter()
    _flagged(RepetitionSpaceDetermination.FLAG_TO_BE_SET, AuxIntSpec(width=8)).write(
        [{"v": 7}], writer)
    assert writer.octets() == bytes((0, 7))


def test_the_flag_is_resolved_per_element_and_not_across_the_repetition():
    """Why this determination needed a mechanism the other seven did not.

    Every other §21.7 determination writes its determinant **outside** the repeated elements,
    where `BitWriter`'s name-keyed patch is enough. §21.7.6's flag is inside them, so N
    elements carry N fields of one name — and a name-keyed patch cannot say which it means.

    Each element gets its own reference scope, which is §9.24.2's mechanism reused rather than
    a second one invented. The proof is that three elements each get their own flag rather
    than the first one being patched three times, which `patch` would refuse outright.
    """
    writer = BitWriter()
    _flagged(RepetitionSpaceDetermination.FLAG_TO_BE_SET, AuxIntSpec(width=8)).write(
        [{"v": 1}, {"v": 2}, {"v": 3}, {"v": 4}], writer)
    assert writer.octets() == bytes((1, 1, 1, 2, 1, 3, 0, 4))


def test_flag_to_be_used_checks_the_flag_rather_than_writing_it():
    """§21.7.7 splits from §21.7.6 exactly as §21.3.5 splits from §21.3.4.

    The field's value "may be set from the abstract syntax ... or may be set by some other
    encoder actions", so the encoder does not write it — it **checks** it: "a conforming
    encoder shall not produce encodings in which the decoder's transforms of this field do not
    correctly identify the last element of the repetition."
    """
    spec = _flagged(RepetitionSpaceDetermination.FLAG_TO_BE_USED, _OCTET)
    writer = BitWriter()
    spec.write([{"more": 1, "v": 7}, {"more": 0, "v": 8}], writer)
    assert writer.octets() == bytes((1, 7, 0, 8))

    # A flag that says "more follow" on the last element is the encoding §21.7.7 forbids.
    try:
        spec.write([{"more": 1, "v": 7}, {"more": 1, "v": 8}], BitWriter())
    except Asn1Error as error:
        assert "21.7.7" in str(error), str(error)
    else:
        raise AssertionError("a flag that misidentifies the last element was accepted")


def test_a_continuation_flag_needs_a_field_to_live_in():
    """§21.7.6 and §21.7.7 both "require the specification of a REFERENCE to a field that is
    part of the repeated element", so a flag determination with no reference names nothing."""
    for determination in (RepetitionSpaceDetermination.FLAG_TO_BE_SET,
                          RepetitionSpaceDetermination.FLAG_TO_BE_USED):
        try:
            RepetitionSpace(determination=determination)
        except Asn1Error as error:
            assert "21.7.6" in str(error), str(error)
        else:
            raise AssertionError(f"{determination.value} was accepted with no REFERENCE")


def test_every_repetition_space_determination_is_now_built():
    """§21.7.1 lists eight and all eight encode. The tuple is asserted whole rather than by
    membership, so a determination added to the enum without an encoder action fails here."""
    assert set(RepetitionSpace._BUILT) == set(RepetitionSpaceDetermination)
