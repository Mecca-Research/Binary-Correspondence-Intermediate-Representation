"""X.692 part two: user-defined encoding objects, and the gate evidence they required.

Part one (`test_asn1_ecn.py`) covers the class/object/object-set model and the built-in BER
and PER sets, which *name* encodings other rails already produce. This file covers the half
that defines octets of its own, and the reason it was allowed to exist.

**The gate.** The build-out roadmap's §6 reduction gate fired and was signed off: the fixed
DER/PER/OER/CJER candidate set already demonstrates cost-governed selection, so user-defined
ECN was closed, reopenable only with an approved measured workload *and* a proof that
ordinary BCIR lowering contracts cannot express it. Approval is the project owner's to give.
The proof is not, so it is executed here rather than asserted — and
`test_no_fixed_candidate_can_express_the_workload` **fails if any candidate ever produces the
target octets**, which is the day this module's justification disappears and the roadmap
should say so.
"""

from __future__ import annotations

from bcir.asn1.ecn import CONCATENATION
from bcir.asn1.ecn_user import (
    UNIT_BIT,
    UNIT_OCTET,
    AuxIntSpec,
    BitWriter,
    BoolSpec,
    Comparison,
    ConcatenationSpec,
    ConditionalIntSpec,
    EncodingSpaceDetermination,
    FIXED_CANDIDATES,
    HeadEndStructure,
    IntForm,
    IntOp,
    IntSelector,
    IntSpec,
    IntToBits,
    IntToInt,
    IntegerBounds,
    Justification,
    OuterSpec,
    PadSpec,
    Padding,
    Pattern,
    PreAlignment,
    RangeCondition,
    ReplaceAction,
    Replacement,
    ReplacementStructure,
    ReversalSpecification,
    SpaceDeterminant,
    StartPointer,
    TransformChain,
    UnusedBits,
    UnusedBitsDetermination,
    ValuePadding,
    UserEncodingObject,
    encode_with_user,
    legacy_frame_objects,
    legacy_frame_workload,
    refuted_by,
)
from bcir.asn1.tags import Asn1Error


def _encode_workload() -> bytes:
    _type, value, _expected = legacy_frame_workload()
    return encode_with_user(legacy_frame_objects(), CONCATENATION, value, outer=OuterSpec())


def test_the_user_defined_objects_produce_the_legacy_frames_octets():
    """The whole point: octets no rule in the repository could otherwise produce.

    `payloadOctets` is 40 and travels as `1010` because the field is scaled in 4-octet units;
    `urgent` is TRUE and travels as `0` because the flag is active low; `reserved` is two bits
    that belong to the octets and to no abstract component; and the length field is
    transmitted *before* the version, because the layout predates any opinion about
    declaration order.
    """
    _type, _value, expected = legacy_frame_workload()
    assert _encode_workload() == expected


def test_no_fixed_candidate_can_express_the_workload():
    """The gate's missing-expressiveness proof, run rather than claimed.

    Every candidate the §6 gate was signed off against is executed against the same abstract
    value and compared to the octets the ECN objects produce. A candidate that refuses is
    still not expressing it, so a refusal counts — but it is recorded, not swallowed, because
    "it threw" and "it produced different octets" are different facts about a rule.

    **This test is the gate.** If a candidate ever matches, the expressiveness argument that
    reopened clauses 19–25 is false and this file should fail loudly rather than let the
    justification rot in a docstring.
    """
    asn1_type, value, expected = legacy_frame_workload()
    results = refuted_by(asn1_type, value, expected)
    assert set(results) == set(FIXED_CANDIDATES), sorted(results)
    for candidate, (octets, note) in results.items():
        assert octets != expected, (
            f"{candidate} produced the target octets ({note}); the fixed candidate set CAN "
            f"express this workload, so the §6 gate's reopening condition is not met and "
            f"user-defined ECN no longer has a justification"
        )
    # And at least one must genuinely encode, or the "proof" would only be showing that a
    # malformed value breaks everything.
    assert any(octets is not None for octets, _ in results.values()), results


def test_the_gap_is_expressiveness_and_not_compactness():
    """The sharpest form of the refutation, and the one worth pinning.

    Canonical PER encodes this value in **exactly the same number of octets** as the ECN
    objects do — and different octets. So "user-defined ECN is only about saving space" is
    refuted by the measurement rather than by argument: at equal size the fixed set still
    cannot produce the layout, because the layout is not a size question.
    """
    asn1_type, value, expected = legacy_frame_workload()
    results = refuted_by(asn1_type, value, expected)
    for candidate in ("CANONICAL-PER-ALIGNED", "CANONICAL-PER-UNALIGNED"):
        octets, note = results[candidate]
        assert octets is not None, f"{candidate} did not encode: {note}"
        assert len(octets) == len(expected), (
            f"{candidate} produced {len(octets)} octets against ECN's {len(expected)}; this "
            f"test exists to show the two are the same size"
        )
        assert octets != expected


def test_table_6_decides_reversibility_per_operation_not_by_blanket_rule():
    """§24.3 Table 6, "Reversal of INT-TO-INT transforms", in force rather than paraphrased.

    The first version of this module refused any value a transform could not invert, on the
    reasoning that a lossy transform is not an encoding rule. Table 6 disagrees, and it is the
    authority: `modulo:n` is **Never reversible** and is a perfectly legal transform, while
    `divide:n` is reversible exactly when the "Value is a multiple of n".

    So the old refusal was the right rule for `divide` and the wrong rule for everything else,
    which is the sort of thing only reading the text settles.
    """
    scale4 = IntToInt(op=IntOp.DIVIDE, operand=4, name="OCTETS-TO-UNITS")
    assert scale4.apply(40) == 10
    assert scale4.inverse(10) == 40
    for value in range(0, 61, 4):
        assert scale4.reversible(value) and scale4.inverse(scale4.apply(value)) == value
    # Table 6's condition on divide, stated as a condition rather than as an exception.
    assert not scale4.reversible(41), "divide:4 is reversible only for multiples of 4"

    # Never reversible, and still legal to apply.
    modulo = IntToInt(op=IntOp.MODULO, operand=3)
    assert modulo.apply(10) == 1  # §24.3.8: i - ((i divide:n) multiply:n)
    assert not modulo.reversible(10)
    try:
        modulo.inverse(1)
    except Asn1Error as error:
        assert "Never reversible" in str(error), error
    else:
        raise AssertionError("modulo claimed an inverse Table 6 says it does not have")

    for always in (IntOp.INCREMENT, IntOp.DECREMENT, IntOp.MULTIPLY, IntOp.NEGATE):
        assert IntToInt(op=always, operand=2).reversible(7), always


def test_divide_truncates_toward_zero_as_the_clause_defines_not_as_python_floors():
    """§24.3.7, which is a real trap for an implementation written in Python.

    The clause: `divide:n` produces "the integer value that is closest to the mathematical
    result, but is no further from zero than that result... so a value of -1 with `divide:2`
    will give zero". Python's `//` floors, so `-1 // 2` is -1. An implementation that reached
    for the obvious operator would be wrong for every negative value.
    """
    half = IntToInt(op=IntOp.DIVIDE, operand=2)
    assert half.apply(-1) == 0, "§24.3.7 gives 0; Python's // would give -1"
    assert half.apply(-5) == -2 and -5 // 2 == -3
    assert half.apply(7) == 3


def test_one_object_specifies_precisely_one_operation():
    """§24.3.5, and the reason `TransformChain` exists at all.

    "any given encoding object to specify precisely one arithmetic operation. General
    arithmetic can, however, be defined by the use of an ordered list of transforms" — and
    §22.4.1.1 declares the property `&Encoder-transforms #TRANSFORM ORDERED OPTIONAL`, so the
    order is part of the specification rather than an artefact of storage.

    The `n - 1` idiom that lets a 4-bit field express 1..16 is `decrement:1`; a field both
    offset and scaled is a two-element chain.
    """
    minus_one = IntToInt(op=IntOp.DECREMENT, operand=1, name="LENGTH-MINUS-ONE")
    assert minus_one.apply(1) == 0 and minus_one.apply(16) == 15
    for value in range(1, 17):
        assert minus_one.inverse(minus_one.apply(value)) == value

    chain = TransformChain(
        (IntToInt(op=IntOp.SUBTRACT_LOWER_BOUND, operand=5), IntToInt(op=IntOp.DIVIDE, operand=4))
    )
    assert chain.apply(45) == 10
    assert chain.inverse(10) == 45
    # Reversibility is checked on the value each step actually sees: 45 reaches divide:4 as
    # 40, which is a multiple; 46 reaches it as 41, which is not.
    assert chain.reversible(45) and not chain.reversible(46)


def test_subtract_lower_bound_is_confined_to_the_first_position():
    """§24.3.9 — a statement about the LIST, so it is enforced on the list.

    "The transform for the value `subtract:lower-bound` shall only be used as the first of an
    ordered list of transforms". A single transform cannot know its own position, which is
    why this lives in `TransformChain` rather than in `IntToInt`.
    """
    first = TransformChain(
        (IntToInt(op=IntOp.SUBTRACT_LOWER_BOUND, operand=1), IntToInt(op=IntOp.MULTIPLY, operand=2))
    )
    assert first.apply(3) == 4
    try:
        TransformChain(
            (
                IntToInt(op=IntOp.MULTIPLY, operand=2),
                IntToInt(op=IntOp.SUBTRACT_LOWER_BOUND, operand=1),
            )
        )
    except Asn1Error as error:
        assert "24.3.9" in str(error), error
    else:
        raise AssertionError("subtract:lower-bound was accepted in a non-initial position")


def test_the_operand_ranges_are_the_clauses_own():
    """§24.3.1: increment/decrement take INTEGER (1..MAX), multiply/divide/modulo (2..MAX).

    A multiply by 1 is not a transform and the notation does not admit one, so the bound is a
    property of the specification rather than a convenience check.
    """
    for op in (IntOp.MULTIPLY, IntOp.DIVIDE, IntOp.MODULO):
        try:
            IntToInt(op=op, operand=1)
        except Asn1Error as error:
            assert "2..MAX" in str(error), error
        else:
            raise AssertionError(f"{op} accepted an operand below its §24.3.1 floor")
    for op in (IntOp.INCREMENT, IntOp.DECREMENT):
        assert IntToInt(op=op, operand=1).apply(5) in (4, 6)


def test_int_to_bits_round_trips():
    """§24.8's fixed-width `positive-int` form, which the frame header's neighbours use.

    The width is now `SIZE × MULTIPLE OF` (§24.8.14) rather than a single `width=`, because
    the clause gives three size arms and only one of them is a bit count — see
    `test_asn1_ecn_transform.py` for the other two.
    """
    transform = IntToBits(
        name="INT-TO-BITS", encoded_as=IntForm.POSITIVE_INT, size=4, unit=UNIT_BIT
    )
    assert transform.apply(10) == (1, 0, 1, 0)
    assert transform.inverse((1, 0, 1, 0)) == 10
    try:
        transform.apply(16)
    except Asn1Error:
        pass
    else:
        raise AssertionError("a value wider than the transform's width was accepted")


def test_transmission_order_is_the_objects_choice_not_the_types():
    """Reordering is something no candidate in the fixed set can be asked for.

    X.690 §8.9 and X.691 §19 both write SEQUENCE components in declaration order, and the
    canonical variants may only reorder a SET, by tag. Here the order is simply stated, and
    stating a different one produces different octets from the same abstract value.
    """
    fields = {
        "a": IntSpec(width=4),
        "b": IntSpec(width=4),
    }
    value = {"a": 1, "b": 2}
    forward = UserEncodingObject(CONCATENATION, ConcatenationSpec(fields=fields, order=("a", "b")))
    reverse = UserEncodingObject(CONCATENATION, ConcatenationSpec(fields=fields, order=("b", "a")))
    assert encode_with_user({CONCATENATION: forward}, CONCATENATION, value) == bytes((0x12,))
    assert encode_with_user({CONCATENATION: reverse}, CONCATENATION, value) == bytes((0x21,))


def test_a_stated_order_that_does_not_match_the_fields_is_refused():
    spec = ConcatenationSpec(fields={"a": IntSpec(width=4)}, order=("a", "b"))
    try:
        spec.transmission_order()
    except Asn1Error as error:
        assert "transmission order" in str(error), error
    else:
        raise AssertionError("an order naming an unknown field was accepted")


def test_a_value_wider_than_its_declared_space_is_a_specification_error():
    """A user-defined encoding states its width, so an overflow is not a wider field.

    This is the opposite of PER's posture, where the width is *derived* from the constraint
    and therefore always fits. Stating the width is what makes a legacy layout expressible,
    and it is also what makes an out-of-range value the specification's problem.
    """
    spec = IntSpec(width=3)
    out = BitWriter()
    try:
        spec.write(8, out)
    except Asn1Error as error:
        assert "encoding space" in str(error), error
    else:
        raise AssertionError("a value that overflows its declared space was written")


def test_twos_complement_and_positive_int_are_different_objects():
    signed = IntSpec(width=8, form=IntForm.TWOS_COMPLEMENT)
    unsigned = IntSpec(width=8, form=IntForm.POSITIVE_INT)
    out = BitWriter()
    signed.write(-2, out)
    assert out.octets() == bytes((0xFE,))
    out = BitWriter()
    unsigned.write(254, out)
    assert out.octets() == bytes((0xFE,))
    # The same octets from different abstract values is exactly why the form has to be part
    # of the object rather than inferred from the value.
    out = BitWriter()
    try:
        unsigned.write(-2, out)
    except Asn1Error as error:
        assert "negative" in str(error), error
    else:
        raise AssertionError("a negative value was written as positive-int")


def test_left_justification_moves_a_narrow_value_within_its_space():
    out = BitWriter()
    IntSpec(width=8, value_padding=ValuePadding(justification=Justification.left())).write(
        0b101, out
    )
    assert out.octets() == bytes((0b10100000,))
    out = BitWriter()
    IntSpec(width=8, value_padding=ValuePadding(justification=Justification.right())).write(
        0b101, out
    )
    assert out.octets() == bytes((0b00000101,))


def test_the_justification_offset_is_a_property_the_clause_gives_and_a_flag_cannot():
    """§21.8.1 is `CHOICE {left INTEGER(0..MAX), right INTEGER(0..MAX)}` — the offset is real.

    §22.8.3.4 splits the "b" padding bits as `n` before and `b-n` after for `left:n`, and
    §22.8.3.3 the other way round for `right:n`. A bare LEFT/RIGHT flag can only spell the
    two zero-offset cases, so a field sitting two bits in from the top of its space was
    unreachable before the offset was carried.
    """
    out = BitWriter()
    IntSpec(width=8, value_padding=ValuePadding(justification=Justification.left(2))).write(
        0b101, out
    )
    #  2 bits pre-padding, the 3-bit value, then the remaining 3 as post-padding.
    assert out.octets() == bytes((0b00101000,))
    out = BitWriter()
    IntSpec(width=8, value_padding=ValuePadding(justification=Justification.right(2))).write(
        0b101, out
    )
    assert out.octets() == bytes((0b00010100,))


def test_an_offset_wider_than_the_padding_is_refused_by_the_clause_that_bounds_it():
    """§22.8.2.1: the justification offset "shall be less than or equal to" the padding count.

    Checked against "b" rather than at construction, because "b" is not known until a value
    is encoded: `left:5` is fine for a 3-bit value in an 8-bit space and impossible for a
    7-bit one, and the object is the same object.
    """
    spec = IntSpec(width=8, value_padding=ValuePadding(justification=Justification.left(5)))
    out = BitWriter()
    spec.write(0b101, out)  # b = 5, so left:5 exactly fits.
    assert out.octets() == bytes((0b00000101,))
    try:
        spec.write(0b1111111, BitWriter())  # b = 1.
    except Asn1Error as error:
        assert "22.8.2.1" in str(error), error
    else:
        raise AssertionError("an offset wider than the available padding was accepted")


def test_the_padding_bits_come_from_the_clause_21_9_padding_value():
    """§21.9.4/§21.9.5/§21.9.6 give `zero`, `one` and `pattern` their meanings.

    §22.8.3.5 sets each side "with the leading bit of the pattern as the first inserted bit
    in each case", so the two sides start the pattern afresh rather than sharing a phase.
    """
    out = BitWriter()
    IntSpec(
        width=8,
        value_padding=ValuePadding(justification=Justification.left(), post_padding=Padding.ONE),
    ).write(0b101, out)
    assert out.octets() == bytes((0b10111111,))
    out = BitWriter()
    IntSpec(
        width=8,
        value_padding=ValuePadding(
            justification=Justification.left(4),
            pre_padding=Padding.PATTERN,
            pre_pattern=Pattern.from_bits("10"),
            post_padding=Padding.PATTERN,
            post_pattern=Pattern.from_bits("10"),
        ),
    ).write(0b1, out)
    #  Pre: 1010 (the pattern replicated). Value: 1. Post: 101 — the pattern from its start
    #  again, truncated, not continued from where the pre-padding left off.
    assert out.octets() == bytes((0b10101101,))


def test_a_pattern_shorter_than_the_run_is_replicated_from_its_leading_bit():
    """§22.2.3.3: "the pattern shall be re-used, most significant bit first"."""
    assert Pattern.from_bits("110").fill(8) == (1, 1, 0, 1, 1, 0, 1, 1)
    assert Pattern.from_bits("110").fill(2) == (1, 1)
    assert Pattern.from_octets(b"\xf0").fill(4) == (1, 1, 1, 1)


def test_an_encoder_option_pattern_denotes_a_length_and_not_a_bit_sequence():
    """§21.10.8 and §21.10.9 leave the value to the encoder, so there is nothing to return.

    Refused rather than defaulted: two conforming encoders may write different bits here and
    both be right, so a rail that picked one would be reporting a choice as a fact.
    """
    for pattern in (Pattern.any_of_length(4), Pattern.different_any()):
        try:
            pattern.bit_sequence()
        except Asn1Error as error:
            assert "encoder" in str(error), error
        else:
            raise AssertionError(f"{pattern} produced bits it does not determine")


def test_an_encoding_that_ends_mid_octet_needs_an_outer_object_to_complete_it():
    """Whether the last octet is padded, and with what, is the specification's decision.

    The writer refuses rather than padding silently, because a default here would put a
    choice §18.1.7 gives to `#OUTER` into the plumbing — and a silently padded encoding is
    indistinguishable from an intended one.
    """
    objects = {
        CONCATENATION: UserEncodingObject(
            CONCATENATION, ConcatenationSpec(fields={"a": IntSpec(width=3)})
        )
    }
    try:
        encode_with_user(objects, CONCATENATION, {"a": 5})
    except Asn1Error as error:
        assert "#OUTER" in str(error), error
    else:
        raise AssertionError("a 3-bit encoding was silently padded to an octet")
    assert encode_with_user(objects, CONCATENATION, {"a": 5}, outer=OuterSpec()) == bytes(
        (0b10100000,)
    )


def test_the_outer_object_controls_the_pad_value():
    objects = {
        CONCATENATION: UserEncodingObject(
            CONCATENATION, ConcatenationSpec(fields={"a": IntSpec(width=3)})
        )
    }
    zeros = encode_with_user(
        objects, CONCATENATION, {"a": 5}, outer=OuterSpec(padding=Padding.ZERO)
    )
    ones = encode_with_user(objects, CONCATENATION, {"a": 5}, outer=OuterSpec(padding=Padding.ONE))
    assert zeros == bytes((0b10100000,))
    assert ones == bytes((0b10111111,))


def test_pad_fields_take_no_value_from_the_abstract_type():
    """`#PAD` is a primitive class part one declares and no built-in set has an object for.

    None of BER or PER needs one, because neither ever emits a bit that carries no abstract
    value. A fixed-layout header does, and the value dict must not have to mention it.
    """
    spec = ConcatenationSpec(
        fields={"a": IntSpec(width=4), "rr": PadSpec(width=4, padding=Padding.ONE)},
        order=("a", "rr"),
        padding=("rr",),
    )
    objects = {CONCATENATION: UserEncodingObject(CONCATENATION, spec)}
    assert encode_with_user(objects, CONCATENATION, {"a": 5}) == bytes((0b01011111,))


def test_a_missing_component_is_refused_rather_than_defaulted():
    spec = ConcatenationSpec(fields={"a": IntSpec(width=4), "b": IntSpec(width=4)})
    objects = {CONCATENATION: UserEncodingObject(CONCATENATION, spec)}
    try:
        encode_with_user(objects, CONCATENATION, {"a": 1})
    except Asn1Error as error:
        assert "#OPTIONAL" in str(error), error
    else:
        raise AssertionError("a missing component was encoded anyway")


def test_a_class_with_no_object_in_the_set_is_refused():
    try:
        encode_with_user({}, CONCATENATION, {"a": 1})
    except Asn1Error as error:
        assert "9.5.1" in str(error), error
    else:
        raise AssertionError("a class with no encoding object encoded anyway")


def test_alignment_happens_where_the_object_says_and_nowhere_else():
    """Aligned PER aligns by its own rules; a user-defined object aligns where it is told."""
    spec = ConcatenationSpec(
        fields={
            "a": IntSpec(width=3),
            "b": IntSpec(width=8, pre_alignment=PreAlignment(unit=UNIT_OCTET)),
        },
        order=("a", "b"),
    )
    objects = {CONCATENATION: UserEncodingObject(CONCATENATION, spec)}
    assert encode_with_user(objects, CONCATENATION, {"a": 0b101, "b": 0xC3}) == bytes(
        (0b10100000, 0xC3)
    )


def test_a_boolean_object_may_be_active_low():
    """DER fixes TRUE at 0xFF and PER writes one bit set; hardware is not obliged to agree."""
    out = BitWriter()
    BoolSpec(true_value=0, false_value=1).write(True, out)
    BoolSpec(true_value=0, false_value=1).write(False, out)
    BoolSpec().write(True, out)
    BoolSpec().write(False, out)
    out.align()
    assert out.octets() == bytes((0b01100000,))


# --- the determinant groups: 21.3, 22.3 and 22.8's UNUSED BITS --------------------------


def _concat(fields: dict, order: tuple, padding: tuple = ()) -> dict:
    spec = ConcatenationSpec(fields=fields, order=order, padding=padding)
    return {"#C": UserEncodingObject("#C", spec)}


def test_a_length_field_can_precede_what_it_measures_because_the_writer_suspends_it():
    """§22.8.3.7's NOTE is the whole mechanism, stated by the specification itself.

    "The encoding of the USING reference in this case appears earlier in the encoding than
    the encoding of this field, and an encoder will need to **suspend** the encoding of that
    field until the value to be encoded has been determined by the encoding of this field."

    A single forward pass cannot do that and a re-running two-pass encoder would have to
    reproduce every encoder's-option choice the first pass made. Reserving and patching is
    what makes one pass sufficient.
    """
    objects = _concat(
        {
            "len": AuxIntSpec(width=8),
            "payload": IntSpec(
                width=16, space_determinant=SpaceDeterminant(reference="len", unit=8)
            ),
        },
        ("len", "payload"),
    )
    assert encode_with_user(objects, "#C", {"payload": 0xBEEF}) == bytes((2, 0xBE, 0xEF))


def test_a_determinant_that_nobody_sets_is_a_refusal_and_not_zeros():
    """An auxiliary field exists to carry a determinant. One with none is a length nobody
    computed, and transmitting zeros for it would be a well-formed document of a lie."""
    objects = _concat({"len": AuxIntSpec(width=8), "payload": IntSpec(width=8)}, ("len", "payload"))
    try:
        encode_with_user(objects, "#C", {"payload": 1})
    except Asn1Error as error:
        assert "reserved and never set" in str(error), error
    else:
        raise AssertionError("an unset determinant was transmitted as zeros")


def test_field_to_be_used_checks_the_applications_value_rather_than_writing_one():
    """§21.3.5 reads a field "whose value may be set from the abstract syntax", and
    §21.3.5's conforming-encoder rule makes a disagreement an error rather than a fix-up.

    The two determinations are opposite relationships to the same field: `field-to-be-set`
    computes, `field-to-be-used` verifies. An implementation that quietly wrote the right
    value here would be overriding the application.
    """
    spec = IntSpec(
        width=16,
        space_determinant=SpaceDeterminant(
            determination=EncodingSpaceDetermination.FIELD_TO_BE_USED, reference="declared", unit=8
        ),
    )
    objects = _concat({"declared": IntSpec(width=8), "payload": spec}, ("declared", "payload"))
    assert encode_with_user(objects, "#C", {"declared": 2, "payload": 1}) == bytes((2, 0, 1))
    try:
        encode_with_user(objects, "#C", {"declared": 3, "payload": 1})
    except Asn1Error as error:
        assert "21.3.5" in str(error), error
    else:
        raise AssertionError("a determinant disagreeing with the encoding was accepted")


def test_a_start_pointer_measures_from_its_own_field_and_not_from_the_encoding():
    """§22.3.3.1 fixes both ends, and neither is the obvious one.

    It counts "from the start of the encoding of the START-POINTER field (after any
    pre-alignment of that field) to the start of the encoding of the element with the
    start-pointer specification (after any pre-alignment of that element)". So two elements
    pointing at two different pointer fields get offsets measured from *different* origins,
    and pre-alignment padding is outside the span at both ends.
    """
    objects = _concat(
        {
            "ptr": AuxIntSpec(width=8),
            "gap": PadSpec(width=8),
            "body": IntSpec(width=8, start_pointer=StartPointer(reference="ptr", unit=8)),
        },
        ("ptr", "gap", "body"),
        padding=("gap",),
    )
    assert encode_with_user(objects, "#C", {"body": 0x7F}) == bytes((2, 0, 0x7F))


def test_unused_bits_records_the_padding_the_justification_actually_inserted():
    """§22.8.3.7: the encoder applies the transforms "to the value `b`" — the count of
    padding bits this encoding produced, not the space's width or the value's."""
    objects = _concat(
        {
            "unused": AuxIntSpec(width=4),
            "v": IntSpec(
                width=12,
                value_padding=ValuePadding(
                    justification=Justification.left(),
                    unused_bits=UnusedBits(UnusedBitsDetermination.FIELD_TO_BE_SET, "unused"),
                ),
            ),
        },
        ("unused", "v"),
    )
    #  0b101 is three bits in a twelve-bit space, so b = 9 and the field carries 9.
    assert encode_with_user(objects, "#C", {"v": 0b101}) == bytes((0x9A, 0x00))
    #  A wider value leaves less padding, and the determinant moves with it.
    assert encode_with_user(objects, "#C", {"v": 0b101010})[0] >> 4 == 6


def test_the_unused_bits_restrictions_reject_a_transform_list_that_would_never_run():
    """§22.8.2.2, §22.8.2.3 and §22.8.2.5 are three ways to write something meaningless.

    A transform list on the wrong determination is worse than an error, because it reads as
    though it ran. All three are checked on the group itself, so they hold however it was
    built — the surface parser is not the only way in.
    """
    chain = TransformChain((IntToInt(op=IntOp.MULTIPLY, operand=2),))
    for kwargs, citation in (
        ({"determination": UnusedBitsDetermination.NOT_NEEDED, "reference": "x"}, "22.8.2.2"),
        ({"determination": UnusedBitsDetermination.FIELD_TO_BE_SET}, "22.8.2.2"),
        (
            {
                "determination": UnusedBitsDetermination.FIELD_TO_BE_USED,
                "reference": "x",
                "encoder_transforms": chain,
            },
            "22.8.2.3",
        ),
        (
            {
                "determination": UnusedBitsDetermination.FIELD_TO_BE_SET,
                "reference": "x",
                "decoder_transforms": chain,
            },
            "22.8.2.5",
        ),
    ):
        try:
            UnusedBits(**kwargs)
        except Asn1Error as error:
            assert citation in str(error), (kwargs, str(error))
        else:
            raise AssertionError(f"{kwargs} was accepted")


def test_a_determinant_transform_must_be_reversible_where_a_value_transform_need_not_be():
    """§22.8.2.4 and §22.3.2.3 both say it, and it is stricter than the value path.

    Table 6 lets `modulo:n` be a legal, never-reversible transform on a *value*. On a
    *determinant* it is not: a length a decoder cannot invert is a length nobody can read.
    """
    spec = IntSpec(
        width=8,
        space_determinant=SpaceDeterminant(
            reference="len",
            unit=1,
            encoder_transforms=TransformChain((IntToInt(op=IntOp.MODULO, operand=3),)),
        ),
    )
    objects = _concat({"len": AuxIntSpec(width=8), "v": spec}, ("len", "v"))
    try:
        encode_with_user(objects, "#C", {"v": 1})
    except Asn1Error as error:
        assert "22.8.2.4" in str(error) or "22.3.2.3" in str(error), error
    else:
        raise AssertionError("an irreversible determinant transform was accepted")


# --- 21.11 range conditions ---------------------------------------------------------------


def test_an_integer_encoding_is_chosen_by_the_types_bounds_and_not_by_the_value():
    """§21.11.3 and §23.6.3.1 together are what makes ECN schema-directed.

    The conditions test "the bounds of the integer", and the encoder applies "the first
    #CONDITIONAL-INT encoding object whose conditions are satisfied". So the same object set
    and the same value produce different widths for differently-bounded types — which is the
    opposite of how every fixed rule in the candidate set behaves.
    """
    narrow = ConditionalIntSpec(
        IntSpec(width=8), ((RangeCondition.TEST_RANGE, Comparison.LESS_THAN_OR_EQUAL_TO, 256),)
    )
    wide = ConditionalIntSpec(IntSpec(width=16), ())
    assert IntSelector((narrow, wide), IntegerBounds(0, 255)).select().width == 8
    assert IntSelector((narrow, wide), IntegerBounds(0, 65535)).select().width == 16
    assert IntSelector((narrow, wide), IntegerBounds()).select().width == 16


def test_falling_off_the_end_of_the_encodings_is_a_refusal_not_a_default():
    """§23.6.3.1: "It is an ECN specification error if none of the conditional encodings have
    conditions that are satisfied." A default here would encode a type the specification
    never described."""
    only = ConditionalIntSpec(
        IntSpec(width=8), ((RangeCondition.BOUNDED_WITHOUT_NEGATIVES, None, None),)
    )
    try:
        IntSelector((only,), IntegerBounds(low=-5, high=5)).select()
    except Asn1Error as error:
        assert "23.6.3.1" in str(error), error
    else:
        raise AssertionError("an unmatched selection fell back on a default")


def test_the_five_shape_conditions_partition_as_the_clause_note_says():
    """§21.11.4's NOTE: "For any given set of bounds, exactly one predicate will be
    satisfied." The five are hand-written, so this asserts the property rather than trusting
    it — a partition that stopped covering would make selection silently pick nothing."""
    cases = {
        IntegerBounds(): RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND,
        IntegerBounds(high=9): RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND,
        IntegerBounds(low=-1): RangeCondition.SEMI_BOUNDED_WITH_NEGATIVES,
        IntegerBounds(low=-1, high=9): RangeCondition.BOUNDED_WITH_NEGATIVES,
        IntegerBounds(low=0): RangeCondition.SEMI_BOUNDED_WITHOUT_NEGATIVES,
        IntegerBounds(low=0, high=9): RangeCondition.BOUNDED_WITHOUT_NEGATIVES,
    }
    for bounds, shape in cases.items():
        assert bounds.exactly_one_shape() is shape, bounds


def test_a_missing_bound_fails_a_comparison_rather_than_standing_in_as_an_infinity():
    """§21.11.4's shapes are how a specification asks whether a bound exists at all.

    Treating an absent lower bound as negative infinity would make `test-lower-bound
    less-than 0` true for an unbounded INTEGER, which is exactly the reading
    `unbounded-or-no-lower-bound` exists to distinguish.
    """
    unbounded = IntegerBounds()
    assert not unbounded.satisfies(RangeCondition.TEST_LOWER_BOUND, Comparison.LESS_THAN, 0)
    assert not unbounded.satisfies(RangeCondition.TEST_LOWER_BOUND, Comparison.GREATER_THAN, 0)
    assert unbounded.satisfies(RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND)


def test_the_comparison_is_required_and_forbidden_by_the_same_clause():
    """§21.11.5 works in both directions, and the second is the one an implementation drops."""
    bounds = IntegerBounds(0, 9)
    try:
        bounds.satisfies(RangeCondition.TEST_RANGE)
    except Asn1Error as error:
        assert "requires" in str(error), error
    else:
        raise AssertionError("test-range was evaluated with nothing to compare against")
    try:
        bounds.satisfies(RangeCondition.BOUNDED_WITHOUT_NEGATIVES, Comparison.EQUAL_TO, 3)
    except Asn1Error as error:
        assert "does not admit" in str(error), error
    else:
        raise AssertionError("a bound shape was given a comparison it cannot use")


def test_subtract_lower_bound_needs_a_condition_that_guarantees_one():
    """§23.7.2.7 relates the transforms to the CONDITION, so neither alone can check it.

    Subtracting a bound that may not exist is not a narrower encoding; it is an undefined
    one. Four of §21.11.4's shapes guarantee a lower bound and `unbounded-or-no-lower-bound`
    is precisely the one that does not.
    """
    subtracting = IntSpec(
        width=8, transform=TransformChain((IntToInt(op=IntOp.SUBTRACT_LOWER_BOUND, operand=0),))
    )
    assert ConditionalIntSpec(
        subtracting, ((RangeCondition.BOUNDED_WITHOUT_NEGATIVES, None, None),)
    ).conditions
    try:
        ConditionalIntSpec(subtracting, ((RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND, None, None),))
    except Asn1Error as error:
        assert "23.7.2.7" in str(error), error
    else:
        raise AssertionError("subtract:lower-bound was allowed with no bound to subtract")


# --- 22.12 bit reversal --------------------------------------------------------------------


def test_the_four_reversals_do_what_their_names_say_and_not_what_21_14_6_lists():
    """§22.12.3.2 and the names agree; §21.14.6's a/b/c/d listing does not.

    §21.14.6 claims to describe the four "in the order of enumerations listed above" and then
    gives a different order. This pins the reading both other passages support, so that a
    future reader who finds §21.14.6 first cannot renumber one rail against the other.
    """
    source = (1, 1, 0, 1, 0, 0, 0, 0)
    assert ReversalSpecification.NO_REVERSAL.apply(source, 8) == source
    assert ReversalSpecification.REVERSE_BITS_IN_UNITS.apply(source, 8) == tuple(reversed(source))
    assert ReversalSpecification.REVERSE_HALF_UNITS.apply(source, 8) == (0, 0, 0, 0, 1, 1, 0, 1)
    assert ReversalSpecification.REVERSE_BITS_IN_HALF_UNITS.apply(source, 8) == (
        1,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
    )


def test_a_reversal_over_a_unit_that_cannot_be_halved_or_reversed_is_refused():
    """§22.12.2.3 (one bit is the identity) and §21.14.5 (an odd unit has no half).

    Both are checked when the object is BUILT rather than when it writes, because an object
    that could never encode anything is wrong the moment it is written down.
    """
    try:
        IntSpec(width=8, bit_reversal=ReversalSpecification.REVERSE_BITS_IN_UNITS)
    except Asn1Error as error:
        assert "22.12.2.3" in str(error), error
    else:
        raise AssertionError("a one-bit reversal was accepted")
    try:
        IntSpec(width=15, bit_reversal=ReversalSpecification.REVERSE_HALF_UNITS, reversal_unit=5)
    except Asn1Error as error:
        assert "21.14.5" in str(error), error
    else:
        raise AssertionError("a half of an odd unit was accepted")


def test_reversal_covers_the_value_padding_and_not_the_pre_alignment():
    """§22.12.1.4's NOTE 2 draws the line, and it is not where an implementation would guess.

    Reversal "applies to the contents of an encoding space or repetition space (including any
    value pre-padding or post-padding), but does not apply to any pre-alignment padding". So
    the alignment bits stay put while the space's contents turn over.
    """
    out = BitWriter()
    out.put_bits(0b1, 1)  # something to align away from
    IntSpec(
        width=8,
        pre_alignment=PreAlignment(unit=UNIT_OCTET, padding=Padding.ONE),
        bit_reversal=ReversalSpecification.REVERSE_BITS_IN_UNITS,
        reversal_unit=8,
    ).write(0b11010000, out)
    #  The 7 pre-alignment ones survive as written; only the value's octet is reversed.
    assert out.octets() == bytes((0b11111111, 0b00001011))


def test_the_outer_reversal_runs_over_the_whole_padded_encoding():
    """§22.12.3.1 gives `#OUTER` a different subject from a field's own reversal: "the entire
    encoding (after any PADDING has been applied) shall be divided into MULTIPLE OF units"."""
    objects = _concat({"a": IntSpec(width=4)}, ("a",))
    plain = encode_with_user(objects, "#C", {"a": 0b1011}, outer=OuterSpec())
    reversed_ = encode_with_user(
        objects,
        "#C",
        {"a": 0b1011},
        outer=OuterSpec(bit_reversal=ReversalSpecification.REVERSE_BITS_IN_UNITS, reversal_unit=8),
    )
    assert plain == bytes((0b10110000,))
    assert reversed_ == bytes((0b00001101,))


# --- 22.1 replacement ---------------------------------------------------------------------


def test_replacement_instantiates_a_structure_around_every_component():
    """§22.1.3.4 and §22.1.3.5: each component becomes an instantiation of the replacement
    structure, and the abstract value still reaches the field that took the parameter's place.

    This is the TLV shape, built out of ECN rather than assumed: a length field per component,
    set from the component's own encoding space.
    """
    tlv = ReplacementStructure(
        name="#Length-prefixed",
        order=("length", "value"),
        dummy="value",
        auxiliary={"length": AuxIntSpec(width=8)},
        determinant=SpaceDeterminant(reference="length", unit=8),
    )
    spec = ConcatenationSpec(
        fields={"a": IntSpec(width=16), "b": IntSpec(width=8)},
        order=("a", "b"),
        replacement=Replacement(ReplaceAction.ALL_COMPONENTS, tlv),
    )
    assert spec.transmission_order() == ("a$length", "a", "b$length", "b")
    objects = {"#C": UserEncodingObject("#C", spec)}
    assert encode_with_user(objects, "#C", {"a": 0xBEEF, "b": 0x2A}) == bytes(
        (2, 0xBE, 0xEF, 1, 0x2A)
    )


def test_head_end_insertions_go_to_the_front_as_a_block_in_component_order():
    """§22.1.3.6 puts them "before all components", "in the same textual order as the
    components being replaced" — not interleaved with the components they belong to.

    That is what makes them useful: a block of location determinants at a fixed offset is
    something a decoder can read before it knows any component's size.
    """
    head = HeadEndStructure(name="#Offsets", order=("at",), auxiliary={"at": AuxIntSpec(width=8)})
    spec = ConcatenationSpec(
        fields={
            "a": IntSpec(width=8, start_pointer=StartPointer(reference="a^at", unit=8)),
            "b": IntSpec(width=8, start_pointer=StartPointer(reference="b^at", unit=8)),
        },
        order=("a", "b"),
        replacement=Replacement(
            ReplaceAction.ALL_COMPONENTS,
            ReplacementStructure("#Bare", ("value",), "value"),
            head_end=head,
        ),
    )
    assert spec.transmission_order() == ("a^at", "b^at", "a", "b")
    objects = {"#C": UserEncodingObject("#C", spec)}
    #  Each offset is measured from its OWN pointer field (22.3.3.1), so both are 2.
    assert encode_with_user(objects, "#C", {"a": 0x11, "b": 0x22}) == bytes((2, 2, 0x11, 0x22))


def test_the_replacement_restrictions_are_the_ones_that_relate_two_clauses():
    """§22.1.2.8, about a combination rather than a single property.

    §22.1.1.7 c) and d) used to be refused here too, because sorting components by whether
    they are optional had nothing to sort by. §23.11's optionality category supplied it, so
    the two actions now select rather than refuse — `test_asn1_ecn_constructors.py` covers
    them, including §22.1.3.4's rule that a replaced optional component becomes mandatory.
    """
    bare = ReplacementStructure("#Bare", ("value",), "value")
    try:
        Replacement(ReplaceAction.STRUCTURE, bare)
    except Asn1Error as error:
        assert "22.1.2.8" in str(error), error
    else:
        raise AssertionError("REPLACE STRUCTURE with no ENCODED BY was accepted")
    for action in (ReplaceAction.OPTIONALS, ReplaceAction.NON_OPTIONALS):
        assert Replacement(action, bare).action is action


def test_a_replacement_will_not_add_a_second_determinant_to_a_field_that_has_one():
    """§21.2's NOTE calls a disagreement between two determinations an error, so two are
    refused rather than raced — the second one would silently win."""
    tlv = ReplacementStructure(
        name="#Length-prefixed",
        order=("length", "value"),
        dummy="value",
        auxiliary={"length": AuxIntSpec(width=8)},
        determinant=SpaceDeterminant(reference="length", unit=8),
    )
    already = IntSpec(width=8, space_determinant=SpaceDeterminant(reference="other", unit=8))
    spec = ConcatenationSpec(
        fields={"a": already},
        order=("a",),
        replacement=Replacement(ReplaceAction.ALL_COMPONENTS, tlv),
    )
    try:
        spec.transmission_order()
    except Asn1Error as error:
        assert "already carries" in str(error), error
    else:
        raise AssertionError("a field took two encoding-space determinants")
