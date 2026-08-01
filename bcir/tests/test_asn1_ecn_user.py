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
    BitWriter, BoolSpec, ConcatenationSpec, FIXED_CANDIDATES, IntForm, IntSpec, IntToBits,
    IntToInt, Justification, OuterSpec, PadSpec, UserEncodingObject, encode_with_user,
    legacy_frame_objects, legacy_frame_workload, refuted_by,
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
            f"user-defined ECN no longer has a justification")
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
            f"test exists to show the two are the same size")
        assert octets != expected


def test_an_int_to_int_transform_is_invertible_or_it_is_refused():
    """A transform that lost information would be a lossy channel, not an encoding rule.

    The scaled field can carry 40 because 40 is a whole number of 4-octet units. It cannot
    carry 41, and the refusal is the point: rounding would give 40 and 41 the same encoding,
    and a decoder could not tell which was sent.
    """
    scale4 = IntToInt(offset=0, scale=4, name="OCTETS-TO-UNITS")
    assert scale4.apply(40) == 10
    assert scale4.inverse(10) == 40
    for value in range(0, 61, 4):
        assert scale4.inverse(scale4.apply(value)) == value
    try:
        scale4.apply(41)
    except Asn1Error as error:
        assert "invertible" in str(error), error
    else:
        raise AssertionError("a non-representable value was silently rounded")


def test_an_offset_transform_round_trips_the_n_minus_one_idiom():
    """`n - 1` in a 4-bit field expresses 1..16, which is why so many headers do it."""
    minus_one = IntToInt(offset=1, scale=1, name="LENGTH-MINUS-ONE")
    assert minus_one.apply(1) == 0
    assert minus_one.apply(16) == 15
    for value in range(1, 17):
        assert minus_one.inverse(minus_one.apply(value)) == value


def test_int_to_bits_round_trips():
    transform = IntToBits(width=4, name="INT-TO-BITS")
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
    forward = UserEncodingObject(
        CONCATENATION, ConcatenationSpec(fields=fields, order=("a", "b")))
    reverse = UserEncodingObject(
        CONCATENATION, ConcatenationSpec(fields=fields, order=("b", "a")))
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
    IntSpec(width=8, justification=Justification.LEFT).write(0b101, out)
    assert out.octets() == bytes((0b10100000,))
    out = BitWriter()
    IntSpec(width=8, justification=Justification.RIGHT).write(0b101, out)
    assert out.octets() == bytes((0b00000101,))


def test_an_encoding_that_ends_mid_octet_needs_an_outer_object_to_complete_it():
    """Whether the last octet is padded, and with what, is the specification's decision.

    The writer refuses rather than padding silently, because a default here would put a
    choice §18.1.7 gives to `#OUTER` into the plumbing — and a silently padded encoding is
    indistinguishable from an intended one.
    """
    objects = {CONCATENATION: UserEncodingObject(
        CONCATENATION, ConcatenationSpec(fields={"a": IntSpec(width=3)}))}
    try:
        encode_with_user(objects, CONCATENATION, {"a": 5})
    except Asn1Error as error:
        assert "#OUTER" in str(error), error
    else:
        raise AssertionError("a 3-bit encoding was silently padded to an octet")
    assert encode_with_user(objects, CONCATENATION, {"a": 5},
                            outer=OuterSpec()) == bytes((0b10100000,))


def test_the_outer_object_controls_the_pad_value():
    objects = {CONCATENATION: UserEncodingObject(
        CONCATENATION, ConcatenationSpec(fields={"a": IntSpec(width=3)}))}
    zeros = encode_with_user(objects, CONCATENATION, {"a": 5}, outer=OuterSpec(pad_value=0))
    ones = encode_with_user(objects, CONCATENATION, {"a": 5}, outer=OuterSpec(pad_value=1))
    assert zeros == bytes((0b10100000,))
    assert ones == bytes((0b10111111,))


def test_pad_fields_take_no_value_from_the_abstract_type():
    """`#PAD` is a primitive class part one declares and no built-in set has an object for.

    None of BER or PER needs one, because neither ever emits a bit that carries no abstract
    value. A fixed-layout header does, and the value dict must not have to mention it.
    """
    spec = ConcatenationSpec(
        fields={"a": IntSpec(width=4), "rr": PadSpec(width=4, value=0b1111)},
        order=("a", "rr"), padding=("rr",))
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
        fields={"a": IntSpec(width=3), "b": IntSpec(width=8, align_before=True)},
        order=("a", "b"))
    objects = {CONCATENATION: UserEncodingObject(CONCATENATION, spec)}
    assert encode_with_user(objects, CONCATENATION, {"a": 0b101, "b": 0xC3}) == bytes(
        (0b10100000, 0xC3))


def test_a_boolean_object_may_be_active_low():
    """DER fixes TRUE at 0xFF and PER writes one bit set; hardware is not obliged to agree."""
    out = BitWriter()
    BoolSpec(true_value=0, false_value=1).write(True, out)
    BoolSpec(true_value=0, false_value=1).write(False, out)
    BoolSpec().write(True, out)
    BoolSpec().write(False, out)
    out.align()
    assert out.octets() == bytes((0b01100000,))
