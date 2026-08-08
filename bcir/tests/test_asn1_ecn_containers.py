"""X.692's containment, which is two relationships and not one.

**§22.11 and §21.3.6 point in opposite directions**, and reading them as one thing is the
mistake this file exists to prevent.

§22.11's `CONTENTS-ENCODING` is *"my contents are another type"*: an `OCTET STRING (CONTAINING
Inner)` whose contents are encoded by a **different encoding object set**. §22.11.2's whole
subject is which set that is, and the answer is a five-row table with an unobvious last row.

§21.3.6's, §21.5.6's and §21.7.8's `container` determination is *"my end is what bounds a
component"*: an element whose extent runs to the end of whatever holds it. Its encoder actions
are nil — §22.7.3.6 says "there is no further encoder action" in so many words — and what each
clause does give the encoder is a rule to **check**: the element must be the last encoding
placed in the container. §21.3.6's NOTE says why that is worth checking rather than trusting:
"It is an ECN encoder's error … if additional encodings are placed in the container", and the
symptom otherwise is a decoder reading one field's bits as another's.

**The notation has no `OUTER` keyword**, which is the reading that shaped the model here.
§21.3.6 says the second form is "a specification that the end of the PDU determines the end of
the encoding space (using `OUTER`)", but §22.4.1.2's syntax is `USING &encoding-space-reference`
in every case and §22.4.1.6 calls that reference one "to an auxiliary field or to a field
carrying abstract values, **or to a container**". So the PDU is just the outermost container
and clause 25's `#OUTER` names it — a reserved reference, not new syntax.
"""

from bcir.asn1.ecn_props import (
    UNIT_OCTET, UNIT_REPETITIONS, OptionalityDetermination, RepetitionSpaceDetermination,
    SizeBounds,
)
from bcir.asn1.ecn_user import (
    OUTER_CONTAINER, AuxIntSpec, ConcatenationSpec, ConditionalRepetitionSpec, ContainedType,
    ContainerSpec, EncodingSpaceDetermination, IntSpec, Optionality, OptionalSpec, OuterSpec,
    PadSpec, RepetitionSpace, RepetitionSpec, SpaceDeterminant, StringSpec,
    UserEncodingObject, encode_with_user,
)
from bcir.asn1.tags import Asn1Error

_OCTET = IntSpec(width=8)


def _objects(spec) -> dict:
    return {"#C": UserEncodingObject("#C", spec)}


def _concat(fields: dict, order: tuple = (), **kwargs) -> dict:
    return _objects(ConcatenationSpec(fields=fields, order=order, **kwargs))


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


# --- §22.11: which rules encode a contained type ---------------------------------------------

def test_the_combined_set_is_left_biased_and_never_the_other_way_round():
    """§9.23.2, which §22.11.1.4 defers to: the combined set is formed "by adding to the first
    set encoding objects for any encoding class for which the first set is **lacking** an
    encoding object and the second set contains one".

    So `COMPLETED BY` fills gaps and never overrides — which is what makes
    `COMPLETED BY PER-BASIC-UNALIGNED` (§9.23.2's own suggestion) a safe thing to write under
    a handful of specialized objects.
    """
    contents = ContainedType(primary={"#A": "mine"}, secondary={"#A": "theirs", "#B": "gap"})
    assert contents.combined() == {"#A": "mine", "#B": "gap"}
    assert ContainedType(primary={"#A": "mine"}).combined() == {"#A": "mine"}


def test_the_five_rows_of_22_11_2_and_the_one_the_text_contradicts_itself_about():
    """§22.11.2 and §13.2.10.6 are the same table written twice, and **they disagree about the
    last row**.

    §22.11.2.2's closing sentence: with `CONTENTS-ENCODING` set, an `ENCODED BY` present and
    `OVERRIDE` left FALSE, "the combined encoding set applied to the **containing type** shall
    be applied". §13.2.10.6 a) says the opposite for that exact case — an object that "specifies
    that it should not override an `ENCODED BY`" leaves it that "the `ENCODED BY` specification
    **shall be used**".

    §13.2.10.6 a) wins here on three counts, and the test asserts that reading so a future
    change back to the other one has to argue with all three:

    * §22.11.1.3 makes the group's purpose deciding "whether an ASN.1 `ENCODED BY` … shall be
      **overridden**" — declining to override should leave it standing, where §22.11.2.2's
      reading discards it.
    * §22.11.2.1 gives the parallel unset case to the `ENCODED BY`, and §13.2.10.6 a) folds
      both into one rule.
    * §13.2.10.6 is the application-point algorithm — what an encoder actually does.
    """
    mine = {"#A": "mine"}
    theirs = {"#A": "encoded-by"}
    outer = {"#A": "container"}

    # §22.11.2.1 / §13.2.10.6 d) and a) — the group is not set at all. These two clauses agree.
    unset = ContainerSpec(contained_class="#A", contents=None, encoded_by=None)
    assert unset.objects_for(outer) is outer
    named = ContainerSpec(contained_class="#A", contents=None, encoded_by=theirs)
    assert named.objects_for(outer) is theirs

    # §22.11.2.2 / §13.2.10.6 c) and b) — the group is set, and here too they agree.
    group = ContainedType(primary=mine)
    assert ContainerSpec(contained_class="#A", contents=group,
                         encoded_by=None).objects_for(outer) == mine
    overriding = ContainedType(primary=mine, override=True)
    assert ContainerSpec(contained_class="#A", contents=overriding,
                         encoded_by=theirs).objects_for(outer) == mine

    # THE CONTESTED ROW. §13.2.10.6 a)'s reading: the ENCODED BY stands.
    assert ContainerSpec(contained_class="#A", contents=group,
                         encoded_by=theirs).objects_for(outer) is theirs
    # And explicitly NOT §22.11.2.2's, which would have given the containing type's set.
    assert ContainerSpec(contained_class="#A", contents=group,
                         encoded_by=theirs).objects_for(outer) is not outer


def test_a_contained_type_is_encoded_by_its_own_set_and_placed_whole():
    """The container's octets are the contained type's encoding, and the contained type's
    encoding is chosen by §22.11.2 rather than inherited."""
    inner = ConcatenationSpec(fields={"a": IntSpec(width=8), "b": IntSpec(width=8)},
                              order=("a", "b"))
    contents = ContainedType(primary={"#Inner": UserEncodingObject("#Inner", inner)})
    spec = ContainerSpec(contained_class="#Inner", contents=contents, name="wrapper")
    objects = _concat({"w": spec}, ("w",))
    assert encode_with_user(objects, "#C", {"w": {"a": 1, "b": 2}}) == bytes((1, 2))


def test_a_container_declaring_no_object_for_its_contained_class_is_refused():
    """§9.5.1: applying an object set that has no object for the class is not a default, it is
    the absence of an encoding."""
    spec = ContainerSpec(contained_class="#Missing", contents=ContainedType(primary={}))
    _refuses("9.5.1", lambda: encode_with_user(_concat({"w": spec}, ("w",)), "#C", {"w": 1}))


def test_a_contained_types_determinants_resolve_inside_it():
    """§9.24.2 moves the application point into the contained type, so a determinant there is
    that encoding's business.

    The interesting half is the refusal: an auxiliary field the *contained* encoding reserved
    and never set is diagnosed against the contained type, not surfaced at the top level where
    nothing explains it. A start pointer reaching across the boundary would be measuring an
    offset in one encoding against a field in another, which no clause defines.
    """
    good = ConcatenationSpec(
        fields={"len": AuxIntSpec(width=8),
                "v": IntSpec(width=8,
                             space_determinant=SpaceDeterminant(reference="len",
                                                                unit=UNIT_OCTET))},
        order=("len", "v"))
    contents = ContainedType(primary={"#I": UserEncodingObject("#I", good)})
    spec = ContainerSpec(contained_class="#I", contents=contents)
    assert encode_with_user(_concat({"w": spec}, ("w",)), "#C", {"w": {"v": 9}}) == bytes((1, 9))

    orphan = ConcatenationSpec(fields={"len": AuxIntSpec(width=8), "v": IntSpec(width=8)},
                               order=("len", "v"))
    stranded = ContainerSpec(
        contained_class="#I",
        contents=ContainedType(primary={"#I": UserEncodingObject("#I", orphan)}))
    _refuses("9.24.2", lambda: encode_with_user(
        _concat({"w": stranded}, ("w",)), "#C", {"w": {"v": 9}}))


def test_a_contained_encoding_wider_than_its_container_is_refused():
    inner = ConcatenationSpec(fields={"a": IntSpec(width=8), "b": IntSpec(width=8)},
                              order=("a", "b"))
    spec = ContainerSpec(
        contained_class="#I", width=8,
        contents=ContainedType(primary={"#I": UserEncodingObject("#I", inner)}))
    _refuses("encoding space is 8", lambda: encode_with_user(
        _concat({"w": spec}, ("w",)), "#C", {"w": {"a": 1, "b": 2}}))


# --- §21.3.6: the other direction ------------------------------------------------------------

def test_a_container_determination_needs_no_transforms_because_it_reads_no_field():
    """§22.4.2.3/§22.4.2.4 confine the transform lists to the two field determinations, and the
    reason is structural rather than arbitrary: a container's end is a *position*, not a number
    carried through a field, so there is nothing for a transform to convert."""
    from bcir.asn1.ecn_transform import IntOp, IntToInt, TransformChain
    _refuses("22.4.2.3", lambda: SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER, reference="c",
        encoder_transforms=TransformChain((IntToInt(IntOp.INCREMENT, 1),))))
    # And every determination still needs a reference — §22.4.1.6 says the `container` one is
    # a reference too, just to a different kind of thing.
    _refuses("21.3.6", lambda: SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER))


def test_an_element_bounded_by_a_container_must_be_the_last_thing_in_it():
    """§21.3.6: "This specification can only be used if the encoding space of the element being
    encoded is the last encoding to be placed in the container." Its NOTE makes writing more
    afterwards an encoder's error, so this is checked rather than trusted."""
    tail = IntSpec(width=8, space_determinant=SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER, reference="box"))
    inner = ConcatenationSpec(fields={"v": tail}, order=("v",))
    contents = ContainedType(primary={"#I": UserEncodingObject("#I", inner)})
    ok = ContainerSpec(contained_class="#I", contents=contents, name="box")
    assert encode_with_user(_concat({"w": ok}, ("w",)), "#C", {"w": {"v": 7}}) == bytes((7,))

    # The same element, with two more octets written inside the container after it.
    late = ContainerSpec(contained_class="#I", contents=contents, name="box",
                         trailer=PadSpec(width=16))
    _refuses("21.3.6", lambda: encode_with_user(
        _concat({"w": late}, ("w",)), "#C", {"w": {"v": 7}}))


def test_a_container_reference_that_names_no_open_container_is_refused():
    """§21.3.6's REFERENCE is to a field "whose contents include this encoding space". A name
    that is not holding this element is not that field, whatever else it may be."""
    stray = IntSpec(width=8, space_determinant=SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER, reference="nowhere"))
    _refuses("21.3.6", lambda: encode_with_user(
        _concat({"v": stray}, ("v",)), "#C", {"v": 1}))


def test_the_pdu_is_the_outermost_container_and_outer_names_it():
    """§21.3.6's second form. There is no `OUTER` keyword in §22.4.1.2's syntax — the reference
    does the work in every case (§22.4.1.6) — so `#OUTER` is a reserved reference naming the
    outermost container, and the grammar needed nothing new to read it."""
    last = IntSpec(width=8, space_determinant=SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER, reference=OUTER_CONTAINER))
    objects = _concat({"a": IntSpec(width=8), "b": last}, ("a", "b"))
    assert encode_with_user(objects, "#C", {"a": 1, "b": 2}) == bytes((1, 2))

    # Anything after it makes the claim false.
    trailing = _concat({"a": last, "b": IntSpec(width=8)}, ("a", "b"))
    _refuses("21.3.6", lambda: encode_with_user(trailing, "#C", {"a": 1, "b": 2}))


def test_outer_padding_is_not_a_further_encoding_placed_in_the_pdu():
    """The `#OUTER` check runs before §25's post-padding, and that ordering is a reading rather
    than an accident: padding the last octet is a decision about the whole encoding (§21.9.3),
    not "an additional encoding placed in the container"."""
    last = IntSpec(width=4, space_determinant=SpaceDeterminant(
        determination=EncodingSpaceDetermination.CONTAINER, reference=OUTER_CONTAINER))
    objects = _concat({"a": IntSpec(width=4), "b": last}, ("a", "b"))
    assert encode_with_user(objects, "#C", {"a": 1, "b": 2},
                            outer=OuterSpec(boundary_bits=8)) == bytes((0x12,))
    # A 4-bit pair needs padding to reach an octet; the claim still holds.
    narrow = _concat({"b": last}, ("b",))
    assert encode_with_user(narrow, "#C", {"b": 2},
                            outer=OuterSpec(boundary_bits=8)) == bytes((0x20,))


def test_an_outer_determined_element_inside_a_container_is_refused():
    """The PDU's end and a container's end are different ends. §21.3.6's rule is about the
    container that immediately holds the element, so claiming the PDU's from inside one is a
    statement about the wrong boundary."""
    inner = ConcatenationSpec(
        fields={"v": IntSpec(width=8, space_determinant=SpaceDeterminant(
            determination=EncodingSpaceDetermination.CONTAINER,
            reference=OUTER_CONTAINER))},
        order=("v",))
    spec = ContainerSpec(
        contained_class="#I", name="box",
        contents=ContainedType(primary={"#I": UserEncodingObject("#I", inner)}))
    _refuses("21.3.6", lambda: encode_with_user(
        _concat({"w": spec}, ("w",)), "#C", {"w": {"v": 1}}))


# --- §21.5.6 and §21.7.8: the same relationship, two more clauses ---------------------------

def test_optionality_by_container_is_the_absence_of_anything_further():
    """§21.5.6: "If the container end is present when a decoder is looking for the start of
    this optional component, then the decoder shall determine that this optional component is
    absent." §22.5.3.5 gives the encoder no value to write and one thing to detect — that no
    further components follow inside the container."""
    optional = OptionalSpec(
        component=IntSpec(width=8),
        presence=Optionality(determination=OptionalityDetermination.CONTAINER,
                             reference=OUTER_CONTAINER))
    objects = _concat({"a": IntSpec(width=8), "o": optional}, ("a", "o"))
    assert encode_with_user(objects, "#C", {"a": 1, "o": 2}) == bytes((1, 2))
    # Absent: nothing is written, and the container simply ends.
    assert encode_with_user(objects, "#C", {"a": 1}) == bytes((1,))
    # Present, with a mandatory component after it: §21.5.6's NOTE calls this an error, because
    # a decoder that sees more bits cannot tell absence from presence.
    trailing = _concat({"o": optional, "z": IntSpec(width=8)}, ("o", "z"))
    _refuses("21.3.6", lambda: encode_with_user(trailing, "#C", {"o": 1, "z": 2}))


def test_a_repetition_can_end_where_its_container_does():
    """§21.7.8 and §22.7.3.6. The determination was refused for needing containment; it has it.

    §21.7.8's NOTE is the rule that survives — "This specification can only be used if the
    encoding of the (repetition category) class is the last encoding to be placed in the
    container" — and it is the same check the other two clauses get.
    """
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.CONTAINER,
                            reference=OUTER_CONTAINER, unit=UNIT_REPETITIONS)
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    objects = _concat({"s": StringSpec(element=_OCTET, repetition=rep)}, ("s",))
    assert encode_with_user(objects, "#C", {"s": [72, 73]}) == b"HI"
    assert encode_with_user(objects, "#C", {"s": []}) == b""

    trailing = _concat({"s": StringSpec(element=_OCTET, repetition=rep),
                        "z": IntSpec(width=8)}, ("s", "z"))
    _refuses("21.3.6", lambda: encode_with_user(trailing, "#C", {"s": [1], "z": 2}))


def test_a_container_determination_takes_no_count_reference():
    """§21.7.8 finds the end by containment, so there is no count field for a USING reference
    to name — the reference names the container itself."""
    _refuses("21.7.4", lambda: RepetitionSpace(
        determination=RepetitionSpaceDetermination.CONTAINER))


def test_a_nested_container_inherits_the_set_one_level_out_and_not_the_pdus():
    """§9.24.2 moves the *application point*, not merely name resolution.

    A container inside a contained type inherits **that type's** set as §22.11.2's "set applied
    to the containing type". Reading the PDU's would answer a question one level too far out —
    and this is the case that distinguishes the two, because the two sets disagree about what
    `#Leaf` encodes.
    """
    leaf_wide = ConcatenationSpec(fields={"n": IntSpec(width=16)}, order=("n",))
    leaf_narrow = ConcatenationSpec(fields={"n": IntSpec(width=8)}, order=("n",))

    # The middle layer's own set says a #Leaf is 16 bits; the PDU's says 8. The inner
    # container states no CONTENTS-ENCODING, so §22.11.2.1 sends it to its container's set.
    inner = ContainerSpec(contained_class="#Leaf", contents=None, name="inner")
    middle = ConcatenationSpec(fields={"x": inner}, order=("x",))
    outer_spec = ContainerSpec(
        contained_class="#Mid", name="outer",
        contents=ContainedType(primary={"#Mid": UserEncodingObject("#Mid", middle),
                                        "#Leaf": UserEncodingObject("#Leaf", leaf_wide)}))
    objects = {"#C": UserEncodingObject("#C", outer_spec),
               "#Leaf": UserEncodingObject("#Leaf", leaf_narrow)}
    # 16 bits, from the middle layer's set — not 8 from the PDU's.
    assert encode_with_user(objects, "#C", {"x": {"n": 0x0102}}) == bytes((1, 2))


def test_a_concatenation_is_a_container_a_reference_can_name():
    """§22.5.2.10: the `container` reference "shall be to a concatenation or to a repetition
    (or to a bitstring or octetstring with a contained type) in which the element being encoded
    is a component". A plain concatenation is one, so naming it is how a REFERENCE reaches it —
    without a name it is simply not referred to, which costs nothing.
    """
    optional = OptionalSpec(
        component=IntSpec(width=8),
        presence=Optionality(determination=OptionalityDetermination.CONTAINER,
                             reference="frame"))
    named = ConcatenationSpec(fields={"a": IntSpec(width=8), "o": optional},
                              order=("a", "o"), container_name="frame")
    assert encode_with_user(_objects(named), "#C", {"a": 1, "o": 2}) == bytes((1, 2))
    assert encode_with_user(_objects(named), "#C", {"a": 1}) == bytes((1,))

    # The same structure with a component after the optional one: §21.5.6's NOTE calls it an
    # error, since a decoder that finds more bits cannot tell absence from presence.
    trailing = ConcatenationSpec(fields={"o": optional, "z": IntSpec(width=8)},
                                 order=("o", "z"), container_name="frame")
    _refuses("21.3.6", lambda: encode_with_user(_objects(trailing), "#C", {"o": 1, "z": 2}))

    # And an unnamed concatenation is not reachable by that reference, which the message says.
    anonymous = ConcatenationSpec(fields={"o": optional}, order=("o",))
    _refuses("21.3.6", lambda: encode_with_user(_objects(anonymous), "#C", {"o": 1}))
