"""X.692's constructor categories: alternatives, optionality, and the handles they read.

**Why these three arrive together.** §22.9's identification handle is the mechanism ECN offers
*instead of* a discriminant field, and four separate clauses depend on it: §21.5.7 for
optionality, §21.6.6 for alternatives, §21.7.10 for repetition end, §22.10.2.1 for a randomly
ordered concatenation. Each of those was previously refused with the words "§22.9's
identification handles are not built". Building the handle turns all four on at once, which is
why one file covers the clause and its four consumers rather than four files repeating it.

**The thing a handle is.** Not a field. §22.9.1.4's three parts are a name, "the bit positions
that form the handle", and "the possible bit patterns (for the bit positions forming the
handle) occurring in the encodings produced by this encoding object". So it is a *declaration
about bits that are there anyway* — which is how BER's tag, IPv4's version nibble and a long
list of other formats actually discriminate. The encoder's only action (§22.9.3.1) is to check
its own output against the declaration, and this file pins that the check is real.
"""

from bcir.asn1.ecn_props import (
    UNIT_OCTET, AlternativeDetermination, ComponentOrder, ConcatenationAlignment,
    HandleValueKind, HandleValueSet, OptionalityDetermination, Pattern,
    RepetitionSpaceDetermination, SizeBounds,
)
from bcir.asn1.ecn_transform import BoolToInt, TransformChain
from bcir.asn1.ecn_user import (
    AlternativeSelection, AlternativesSpec, AuxIntSpec, BitWriter, Concatenation,
    ConcatenationSpec, ConditionalRepetitionSpec, HandleRegistry, IdentificationHandle,
    IntSpec, Optionality, OptionalSpec, OuterSpec, PadSpec, PreAlignment, ReplaceAction,
    Replacement, ReplacementStructure, RepetitionSpace, RepetitionSpec, SpaceDeterminant,
    StartPointer, StringSpec, TagSpec, UserEncodingObject, encode_with_user,
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


# --- §21.16: the handle value set, and the only question it has to answer fast --------------

def test_a_handle_value_set_reads_as_ranges_over_the_conceptual_field():
    """§22.9.1.7: "the bit in the conceptual handle field nearest to the zero position is the
    high-order bit", and a `number` "is right-justified within this field".

    Every alternative reduces to inclusive integer ranges, because the question the four
    consuming clauses ask is disjointness (§21.5.7, §21.6.6, §21.7.10, §22.10.2.1) and that is
    not answerable by enumerating 2**n patterns for a wide handle.
    """
    assert HandleValueSet.from_bits("1010").ranges_over(4) == ((10, 10),)
    assert HandleValueSet.from_octets(b"\xC0").ranges_over(8) == ((192, 192),)
    assert HandleValueSet.of_number(3).ranges_over(4) == ((3, 3),)
    assert HandleValueSet.of_range(4, 7).ranges_over(4) == ((4, 7),)
    # §21.16.7's `ranges` may overlap each other; coalescing them is what keeps the
    # cross-set disjointness test from having to reason about a set overlapping itself.
    assert HandleValueSet.of_ranges(((0, 2), (2, 5), (9, 9))).ranges_over(4) == (
        (0, 5), (9, 9))


def test_the_width_rules_are_checked_against_the_handle_and_not_the_set():
    """§22.9.1.8 and §21.16.4 are the same rule from two sides, and neither can be checked when
    the set is written: the width belongs to the *handle*'s `AT` positions."""
    _refuses("22.9.1.8", lambda: HandleValueSet.from_bits("101").ranges_over(4))
    _refuses("22.9.1.7", lambda: HandleValueSet.of_number(16).ranges_over(4))
    _refuses("21.16.4", lambda: HandleValueSet.of_range(0, 99).ranges_over(4))


def test_a_range_with_high_below_low_is_refused():
    """§21.16.6/§21.16.7: "with high greater than or equal to low"."""
    _refuses("21.16.6", lambda: HandleValueSet.of_range(7, 4))
    _refuses("21.16.1", lambda: HandleValueSet.of_ranges(()))


def test_tag_any_has_no_value_until_a_tag_number_determines_it():
    """§21.16.5 and §22.9.1.9. `tag:any` is the DEFAULT of `&handle-value-set`, which makes the
    refusal load-bearing: a handle nobody gave a set to would otherwise silently match nothing.

    §22.9.1.9's second sentence is the interesting one — a set that *is* stated and differs
    from the tag number "is an ECN specification error", so resolution checks rather than
    overrides.
    """
    _refuses("21.16.5", lambda: HandleValueSet.tag_any().ranges_over(4))
    assert HandleValueSet.tag_any().resolve_tag(5) == HandleValueSet.of_number(5)
    assert HandleValueSet.of_range(4, 7).resolve_tag(5).kind is HandleValueKind.RANGE
    _refuses("22.9.1.9", lambda: HandleValueSet.of_range(4, 7).resolve_tag(9))


def test_disjointness_is_the_property_four_clauses_ask_for():
    a = HandleValueSet.of_range(0, 3)
    b = HandleValueSet.of_range(4, 7)
    assert a.disjoint_from(b, 3) and b.disjoint_from(a, 3)
    assert not a.disjoint_from(HandleValueSet.of_number(2), 3)
    assert HandleValueSet.of_ranges(((0, 1), (6, 7))).disjoint_from(
        HandleValueSet.of_range(2, 5), 3)


# --- §22.9: the handle itself ---------------------------------------------------------------

def test_positions_are_a_set_ordered_from_zero_upwards():
    """§22.9.1.6: "a set of integer values (not necessarily contiguous, and not necessarily in
    ascending order in the ECN specification)", which "shall be ordered by encoders and
    decoders from the zero position ... upwards"."""
    handle = IdentificationHandle("h", (5, 0, 2), HandleValueSet.of_number(0))
    assert handle.ordered() == (0, 2, 5)
    assert handle.width == 3
    _refuses("22.9.1.6", lambda: IdentificationHandle("h", (1, 1), HandleValueSet.of_number(0)))
    _refuses("22.9.1.2", lambda: IdentificationHandle("h", (), HandleValueSet.of_number(0)))


def test_a_non_contiguous_handle_reads_its_bits_in_position_order():
    """The conceptual handle field is the *ascending-order* concatenation of the named bits,
    regardless of the order the specification wrote them in."""
    out = BitWriter()
    out.put_bits(0b1011_0001, 8)
    handle = IdentificationHandle("h", (7, 0, 4), HandleValueSet.of_number(0b101))
    # bits 0, 4 and 7 of 1011_0001 are 1, 0 and 1 → 0b101.
    assert handle.value_in(out, 0) == 0b101
    assert handle.check(out, 0) == 0b101


def test_the_encoder_checks_its_own_output_against_its_declared_set():
    """§22.9.3.1 is the one encoder action a handle has, and it is a real check: "the encoder
    shall check that the value of the identification handle occurring in the encoding produced
    is a member of the specified handle value set, and shall diagnose a specification or
    application error otherwise".

    An object that can leave its own declared set has made every determination reading the
    handle unsound, so this is not advisory.
    """
    top_nibble = IdentificationHandle("kind", (0, 1, 2, 3), HandleValueSet.of_range(4, 5))
    objects = _concat({"v": IntSpec(width=8, exhibits=top_nibble)}, ("v",))
    assert encode_with_user(objects, "#C", {"v": 0x41}) == b"\x41"
    _refuses("22.9.3.1", lambda: encode_with_user(objects, "#C", {"v": 0x91}))


def test_a_handle_position_past_the_encoding_is_refused():
    handle = IdentificationHandle("h", (0, 9), HandleValueSet.of_number(0))
    objects = _concat({"v": IntSpec(width=8, exhibits=handle)}, ("v",))
    _refuses("22.9.1.5", lambda: encode_with_user(objects, "#C", {"v": 0}))


def test_positions_are_taken_after_pre_alignment():
    """§22.9.1.5: the positions are "in the final encoding, **after any pre-alignment has been
    applied**". So position zero is where the encoding *space* starts, not where the padding
    that precedes it starts — a handle would otherwise shift with the field's alignment."""
    handle = IdentificationHandle("h", (0, 1), HandleValueSet.of_number(0b10))
    objects = _concat({
        "a": PadSpec(width=3),
        "v": IntSpec(width=8, exhibits=handle,
                     pre_alignment=PreAlignment(unit=UNIT_OCTET)),
    }, ("a", "v"), padding=("a",))
    assert encode_with_user(objects, "#C", {"v": 0b1000_0001}) == b"\x00\x81"


def test_tag_any_is_only_legal_for_a_tag_object():
    """§22.9.1.9: "shall not be specified as `tag:any` unless the specification is for an
    encoding object of the #TAG class". Every other category has to state its set."""
    tag_handle = IdentificationHandle("t", (0, 1, 2, 3))          # DEFAULT tag:any
    objects = _objects(ConcatenationSpec(
        fields={"t": TagSpec(width=4, number=6, exhibits=tag_handle)}, order=("t",),
        padding=("t",)))
    assert encode_with_user(objects, "#C", {}, outer=_pad_to_octet()) == b"\x60"
    bad = _concat({"v": IntSpec(width=4, exhibits=tag_handle)}, ("v",))
    _refuses("22.9.1.9", lambda: encode_with_user(bad, "#C", {"v": 1},
                                                  outer=_pad_to_octet()))


def _pad_to_octet() -> OuterSpec:
    return OuterSpec(boundary_bits=8)


def test_the_registry_enforces_the_two_specification_wide_rules():
    """§22.9.2.1 and §22.9.2.3 relate one `EXHIBITS HANDLE` clause to every other with the same
    name, so neither can be checked where a handle is written."""
    registry = HandleRegistry()
    registry.declare(IdentificationHandle("k", (0, 1), HandleValueSet.of_number(0)),
                     where="'a'", alignment_unit=UNIT_OCTET)
    _refuses("22.9.2.1", lambda: registry.declare(
        IdentificationHandle("k", (0, 2), HandleValueSet.of_number(1)), where="'b'",
        alignment_unit=UNIT_OCTET))
    _refuses("22.9.2.3", lambda: registry.declare(
        IdentificationHandle("k", (0, 1), HandleValueSet.of_number(1)), where="'c'",
        alignment_unit=4))
    # "No pre-alignment specification" and "align to bit" are one case: §22.2.1.1's default
    # unit is `bit`, and aligning to one bit inserts nothing.
    plain = HandleRegistry()
    plain.declare(IdentificationHandle("k", (0,), HandleValueSet.of_number(0)), where="'a'")
    plain.declare(IdentificationHandle("k", (0,), HandleValueSet.of_number(1)), where="'b'",
                  alignment_unit=1)


# --- §22.5 / §23.11: optionality --------------------------------------------------------------

def test_presence_is_mandatory_and_not_defaulted():
    """§22.5.1.6: the specification "is mandatory for it to be set in all places in the defined
    syntax where it is allowed. Defaulting all other parts of this defined syntax (e.g., use of
    `PRESENCE` alone) would not satisfy the above constraints"."""
    _refuses("22.5.1.6", lambda: OptionalSpec(component=_OCTET))
    _refuses("#OPTIONAL object wraps",
             lambda: OptionalSpec(presence=Optionality(reference="p")))


def test_field_to_be_set_writes_a_presence_bit_the_encoder_owns():
    """§22.5.3.2–§22.5.3.3: the conceptual boolean `element-is-present` becomes the value of an
    earlier field. §22.8.3.7's suspension applies — the bit precedes what it describes."""
    objects = _concat({
        "p": AuxIntSpec(width=1),
        "v": OptionalSpec(component=IntSpec(width=7),
                          presence=Optionality(reference="p")),
    }, ("p", "v"))
    assert encode_with_user(objects, "#C", {"v": 0x2A}) == bytes((0b1_0101010,))
    # Absent: the bit is zero and nothing else is written. Seven bits short of an octet, so
    # #OUTER completes it — which is exactly the point, absence costs no space of its own.
    assert encode_with_user(objects, "#C", {}, outer=_pad_to_octet()) == b"\x00"


def test_field_to_be_used_checks_the_applications_presence_flag():
    """§22.5.3.4: "It is an application error if this condition is not met, and encoding shall
    not proceed." The application owns the field; the encoder verifies rather than corrects."""
    objects = _concat({
        "p": IntSpec(width=8),
        "v": OptionalSpec(
            component=IntSpec(width=8),
            presence=Optionality(
                determination=OptionalityDetermination.FIELD_TO_BE_USED, reference="p")),
    }, ("p", "v"))
    assert encode_with_user(objects, "#C", {"p": 1, "v": 9}) == bytes((1, 9))
    assert encode_with_user(objects, "#C", {"p": 0}) == bytes((0,))
    _refuses("22.5.3.4", lambda: encode_with_user(objects, "#C", {"p": 0, "v": 9}))
    _refuses("22.5.3.4", lambda: encode_with_user(objects, "#C", {"p": 1}))


def test_an_encoder_transform_maps_the_boolean_onto_the_field():
    """§22.5.2.7: "The first transform shall have a source which is boolean." §24.5's
    `BOOL-TO-INT AS true-zero` is the active-low presence flag plenty of hardware asserts."""
    objects = _concat({
        "p": AuxIntSpec(width=1),
        "v": OptionalSpec(
            component=IntSpec(width=7),
            presence=Optionality(
                reference="p",
                encoder_transforms=TransformChain((BoolToInt(true_zero=True),)))),
    }, ("p", "v"))
    assert encode_with_user(objects, "#C", {"v": 0x2A}) == bytes((0b0_0101010,))
    assert encode_with_user(objects, "#C", {}, outer=_pad_to_octet()) == b"\x80"


def test_the_five_optionality_restrictions_are_each_enforced():
    """§22.5.2.2, §22.5.2.3, §22.5.2.6, §22.5.2.8 — four ways to write something that would
    never run, plus §22.5.2.4's requirement that `pointer` has a start pointer to read."""
    _refuses("22.5.2.2", lambda: Optionality(reference="p", handle_set=True))
    _refuses("22.5.2.3", lambda: Optionality(
        determination=OptionalityDetermination.HANDLE, reference="p"))
    _refuses("22.5.2.6", lambda: Optionality(
        determination=OptionalityDetermination.CONTAINER, reference="c",
        encoder_transforms=TransformChain((BoolToInt(),))))
    _refuses("22.5.2.8", lambda: Optionality(
        reference="p", decoder_transforms=TransformChain((BoolToInt(),))))
    _refuses("22.5.2.4", lambda: OptionalSpec(
        component=_OCTET,
        presence=Optionality(determination=OptionalityDetermination.POINTER)))


def test_a_pointer_determination_writes_zero_when_the_component_is_absent():
    """§21.5.9: "If that field is zero, then this component is absent." Zero works as a
    sentinel because the pointer field is encoded before the offset it measures, so a genuine
    offset can never be zero."""
    objects = _concat({
        "ptr": AuxIntSpec(width=8),
        "pad": PadSpec(width=8),
        "v": OptionalSpec(
            component=IntSpec(width=8),
            start_pointer=StartPointer(reference="ptr", unit=UNIT_OCTET),
            presence=Optionality(determination=OptionalityDetermination.POINTER)),
    }, ("ptr", "pad", "v"), padding=("pad",))
    assert encode_with_user(objects, "#C", {"v": 0x77}) == bytes((2, 0, 0x77))
    assert encode_with_user(objects, "#C", {}) == bytes((0, 0))


def test_optionality_by_handle_needs_no_encoder_action_beyond_the_handles_own():
    """§22.5.3.6: "If `DETERMINED BY` is `handle` there is no further action needed by the
    encoder." The bits are already in the component's own encoding."""
    handle = IdentificationHandle("lead", (0, 1), HandleValueSet.of_number(0b11))
    objects = _concat({
        "v": OptionalSpec(
            component=IntSpec(width=8, exhibits=handle),
            presence=Optionality(determination=OptionalityDetermination.HANDLE,
                                 handle_id="lead", handle_set=True)),
    }, ("v",))
    assert encode_with_user(objects, "#C", {"v": 0xC5}) == b"\xC5"
    assert encode_with_user(objects, "#C", {}) == b""
    _refuses("22.9.3.1", lambda: encode_with_user(objects, "#C", {"v": 0x05}))


def test_a_replacing_optional_object_is_refused_with_what_it_would_need():
    """§23.11.3.2 hands "the entire component (including any classes in the tag category, but
    excluding classes in the optionality category)" to a parameterized replacement structure.
    That is X.683 parameterization applied to an optional class, and it is named rather than
    approximated."""
    structure = _length_prefixed()
    _refuses("23.11.3.2", lambda: OptionalSpec(
        component=_OCTET, presence=Optionality(reference="p"),
        replacement=Replacement(action=ReplaceAction.STRUCTURE, structure=structure)))
    _refuses("23.11.1", lambda: OptionalSpec(
        component=_OCTET, presence=Optionality(reference="p"),
        replacement=Replacement(action=ReplaceAction.ALL_COMPONENTS, structure=structure)))


def _length_prefixed() -> "ReplacementStructure":
    """§22.1.2.2's example shape: `#Length-prefixed{#D} ::= #CONCATENATION {len #INT, v #D}`."""
    return ReplacementStructure(
        name="#Length-prefixed", order=("len", "v"), dummy="v",
        auxiliary={"len": AuxIntSpec(width=8)},
        determinant=SpaceDeterminant(reference="len", unit=UNIT_OCTET))


def test_replace_optionals_and_non_optionals_now_have_something_to_sort_by():
    """§22.1.1.7 c) and d) sort components by whether they are optional. Those two actions were
    refused with the words "the optionality category is not built on this rail"; §23.11 built
    it, so they are the fourth thing this slice turns on.

    A component the action does not select passes through untouched — and takes no head-end
    insertion either, since §22.1.3.6 orders insertions by "the components being replaced"."""
    optional = OptionalSpec(
        component=IntSpec(width=8),
        presence=Optionality(determination=OptionalityDetermination.FIELD_TO_BE_USED,
                             reference="p"))
    fields = {"p": IntSpec(width=8), "m": IntSpec(width=8), "o": optional}
    only_optional = ConcatenationSpec(
        fields=fields, order=("p", "m", "o"),
        replacement=Replacement(action=ReplaceAction.OPTIONALS,
                                structure=_length_prefixed()))
    assert only_optional.transmission_order() == ("p", "m", "o$len", "o")
    only_mandatory = ConcatenationSpec(
        fields=fields, order=("p", "m", "o"),
        replacement=Replacement(action=ReplaceAction.NON_OPTIONALS,
                                structure=_length_prefixed()))
    assert only_mandatory.transmission_order() == ("p$len", "p", "m$len", "m", "o")
    assert encode_with_user(_objects(only_optional), "#C",
                            {"p": 1, "m": 0x11, "o": 0x22}) == bytes((1, 0x11, 1, 0x22))


def test_a_replaced_optional_component_becomes_mandatory():
    """§22.1.3.4 is explicit and easy to miss: the component is replaced "with a **non-optional**
    instantiation of the replacement structure", and the actual parameter de-references to the
    original component "**except for any class in the optionality category**".

    So `REPLACE OPTIONALS` does not wrap optionality — it *removes* it, and the value has to be
    there. §22.1.3.3 says the same for an `#OPTIONAL` object's own `REPLACE STRUCTURE`.
    """
    optional = OptionalSpec(
        component=IntSpec(width=8),
        presence=Optionality(determination=OptionalityDetermination.FIELD_TO_BE_USED,
                             reference="p"))
    spec = ConcatenationSpec(
        fields={"p": IntSpec(width=8), "o": optional}, order=("p", "o"),
        replacement=Replacement(action=ReplaceAction.OPTIONALS,
                                structure=_length_prefixed()))
    # Without replacement the same value is legal and writes nothing for `o`.
    plain = ConcatenationSpec(fields={"p": IntSpec(width=8), "o": optional}, order=("p", "o"))
    assert encode_with_user(_objects(plain), "#C", {"p": 0}) == bytes((0,))
    _refuses("does not carry", lambda: encode_with_user(_objects(spec), "#C", {"p": 0}))


# --- §22.6 / §23.1: alternatives ---------------------------------------------------------------

def test_the_alternative_index_is_positional_and_the_order_property_fixes_it():
    """§22.6.3.3: "zero for the first alternative, one for the next, and so on, where the order
    of the alternatives is determined by `ORDER`". So renaming an alternative changes nothing
    and reordering the ECN structure changes the wire format."""
    spec = AlternativesSpec(
        alternatives={"a": TagSpec(width=8, number=9, tagged=_OCTET),
                      "b": TagSpec(width=8, number=2, tagged=_OCTET)},
        selection=AlternativeSelection(reference="k"))
    assert spec.ordering() == ("a", "b")
    assert spec.index_of("b") == 1
    by_tag = AlternativesSpec(
        alternatives=spec.alternatives,
        selection=AlternativeSelection(reference="k", ordering=ComponentOrder.TAG))
    # §22.6.3.4: "lowest tag number first" — which reverses the textual order here.
    assert by_tag.ordering() == ("b", "a")
    assert by_tag.index_of("a") == 1


def test_field_to_be_set_carries_the_alternative_index_in_an_earlier_field():
    """§22.6.3.5's NOTE describes the suspension exactly: the `USING` field "appears earlier in
    the encoding than the encoding of the alternative, and an encoder will need to suspend the
    encoding of that field until the alternative to be encoded has been determined"."""
    spec = AlternativesSpec(
        alternatives={"small": IntSpec(width=8), "flag": IntSpec(width=8)},
        selection=AlternativeSelection(reference="k"))
    objects = _concat({"k": AuxIntSpec(width=8), "c": spec}, ("k", "c"))
    assert encode_with_user(objects, "#C", {"c": ("small", 7)}) == bytes((0, 7))
    assert encode_with_user(objects, "#C", {"c": {"flag": 7}}) == bytes((1, 7))


def test_field_to_be_used_checks_the_applications_selector():
    """§22.6.3.6, the alternatives twin of §22.5.3.4."""
    spec = AlternativesSpec(
        alternatives={"a": IntSpec(width=8), "b": IntSpec(width=8)},
        selection=AlternativeSelection(
            determination=AlternativeDetermination.FIELD_TO_BE_USED, reference="k"))
    objects = _concat({"k": IntSpec(width=8), "c": spec}, ("k", "c"))
    assert encode_with_user(objects, "#C", {"k": 1, "c": ("b", 5)}) == bytes((1, 5))
    _refuses("22.6.3.6", lambda: encode_with_user(objects, "#C", {"k": 0, "c": ("b", 5)}))


def test_alternatives_by_handle_require_disjoint_sets_from_every_alternative():
    """§21.6.6: the handle "shall be exhibited by the encoding objects applied to each of the
    alternatives ... The handle value sets specified by those encoding objects shall all be
    disjoint." Without the second half, §22.6.4.4's decoder lookup has more than one answer."""
    low = IdentificationHandle("kind", (0, 1), HandleValueSet.of_range(0, 1))
    high = IdentificationHandle("kind", (0, 1), HandleValueSet.of_range(2, 3))
    spec = AlternativesSpec(
        alternatives={"a": IntSpec(width=8, exhibits=low),
                      "b": IntSpec(width=8, exhibits=high)},
        selection=AlternativeSelection(determination=AlternativeDetermination.HANDLE,
                                       handle_id="kind", handle_set=True))
    objects = _concat({"c": spec}, ("c",))
    assert encode_with_user(objects, "#C", {"c": ("a", 0x3F)}) == b"\x3F"
    assert encode_with_user(objects, "#C", {"c": ("b", 0xBF)}) == b"\xBF"
    _refuses("22.9.3.1", lambda: encode_with_user(objects, "#C", {"c": ("a", 0xBF)}))

    overlapping = IdentificationHandle("kind", (0, 1), HandleValueSet.of_range(1, 3))
    _refuses("21.6.6", lambda: AlternativesSpec(
        alternatives={"a": IntSpec(width=8, exhibits=low),
                      "b": IntSpec(width=8, exhibits=overlapping)},
        selection=AlternativeSelection(determination=AlternativeDetermination.HANDLE,
                                       handle_id="kind", handle_set=True)))
    _refuses("21.6.6", lambda: AlternativesSpec(
        alternatives={"a": IntSpec(width=8, exhibits=low), "b": IntSpec(width=8)},
        selection=AlternativeSelection(determination=AlternativeDetermination.HANDLE,
                                       handle_id="kind", handle_set=True)))


def test_alternative_ordering_has_two_values_where_concatenation_has_three():
    """§22.6.1.1 declares `ENUMERATED {textual, tag}`; §22.10.1.1 declares `{textual, tag,
    random}`. `random` is meaningless for a CHOICE — one alternative is encoded, so there is
    no order to permute — and the clause simply does not list it."""
    _refuses("22.6.1.1", lambda: AlternativeSelection(
        reference="k", ordering=ComponentOrder.RANDOM))


def test_tag_ordering_requires_distinct_tags_on_every_alternative():
    """§22.6.2.10 and §22.6.2.11: every alternative "shall start with an encoding class in the
    tag category", and "the component-tags of each alternative shall be distinct"."""
    _refuses("22.6.2.10", lambda: AlternativesSpec(
        alternatives={"a": TagSpec(width=8, number=1, tagged=_OCTET), "b": IntSpec(width=8)},
        selection=AlternativeSelection(reference="k", ordering=ComponentOrder.TAG)))
    _refuses("22.6.2.11", lambda: AlternativesSpec(
        alternatives={"a": TagSpec(width=8, number=1, tagged=_OCTET),
                      "b": TagSpec(width=8, number=1, tagged=_OCTET)},
        selection=AlternativeSelection(reference="k", ordering=ComponentOrder.TAG)))


def test_an_optional_component_is_transparent_to_the_component_tag():
    """§23.11.3.2 puts the tag classes *inside* the optionality wrapper — the replacement takes
    "the entire component (including any classes in the tag category, but excluding classes in
    the optionality category)" — so an `#OPTIONAL` wrapper does not hide the component-tag."""
    optional = OptionalSpec(component=TagSpec(width=8, number=3, tagged=_OCTET),
                            presence=Optionality(reference="p"))
    tagged_only = ConcatenationSpec(
        fields={"b": TagSpec(width=8, number=7, tagged=_OCTET), "a": optional},
        order=("b", "a"), concatenation=Concatenation(order=ComponentOrder.TAG))
    assert tagged_only.transmission_order() == ("a", "b")
    # A component with no tag at all still fails §22.10.2.4, and it fails when the object is
    # written rather than when it first encodes something.
    _refuses("22.10.2.4", lambda: ConcatenationSpec(
        fields={"p": AuxIntSpec(width=8), "a": optional}, order=("p", "a"),
        concatenation=Concatenation(order=ComponentOrder.TAG)))


def test_an_alternatives_object_does_not_inherit_its_components_handles():
    """§23.1.2.3: it "does not exhibit an identification handle unless `EXHIBITS HANDLE` is set
    (**even if the components of the defined construction exhibit an identification handle**)".
    The alternatives' handles are what `DETERMINED BY handle` reads; they are not inherited."""
    low = IdentificationHandle("kind", (0, 1), HandleValueSet.of_range(0, 1))
    spec = AlternativesSpec(
        alternatives={"a": IntSpec(width=8, exhibits=low)},
        selection=AlternativeSelection(determination=AlternativeDetermination.HANDLE,
                                       handle_id="kind", handle_set=True))
    assert spec.exhibits is None


def test_an_alternatives_object_needs_alternatives_and_a_selection():
    _refuses("alternatives category", lambda: AlternativesSpec(
        selection=AlternativeSelection(reference="k")))
    _refuses("22.6.2.9", lambda: AlternativesSpec(alternatives={"a": _OCTET}))
    _refuses("22.6.2.2", lambda: AlternativeSelection(reference="k", handle_set=True))
    _refuses("22.6.2.3", lambda: AlternativeSelection(
        determination=AlternativeDetermination.HANDLE, reference="k"))


# --- §22.10: what the concatenation group adds -----------------------------------------------

def test_tag_order_sorts_the_components_by_component_tag():
    """§22.10.3.2: "the order shall be that of the tag numbers in the component-tags (lowest
    tag number first)"."""
    spec = ConcatenationSpec(
        fields={"z": TagSpec(width=8, number=9, tagged=_OCTET),
                "a": TagSpec(width=8, number=2, tagged=_OCTET)},
        order=("z", "a"), concatenation=Concatenation(order=ComponentOrder.TAG))
    assert spec.transmission_order() == ("a", "z")
    objects = _objects(spec)
    assert encode_with_user(objects, "#C", {"z": 1, "a": 2}) == bytes((2, 2, 9, 1))


def test_random_order_requires_a_handle_on_every_component():
    """§22.10.2.1: `random` makes `HANDLE` "assume the default value of `default-handle` if not
    set", requires the objects applied to **all** components to exhibit it, and requires their
    value sets to be disjoint. §22.10.3.3 then lets the encoder choose freely — so, like
    §21.9.7's `encoder-option` padding, an encoding using it has no unique octets."""
    a = IdentificationHandle("default-handle", (0, 1), HandleValueSet.of_range(0, 1))
    b = IdentificationHandle("default-handle", (0, 1), HandleValueSet.of_range(2, 3))
    ok = ConcatenationSpec(
        fields={"x": IntSpec(width=8, exhibits=a), "y": IntSpec(width=8, exhibits=b)},
        order=("x", "y"), concatenation=Concatenation(order=ComponentOrder.RANDOM))
    assert encode_with_user(_objects(ok), "#C", {"x": 0x11, "y": 0x99}) == bytes((0x11, 0x99))
    _refuses("22.10.2.1", lambda: ConcatenationSpec(
        fields={"x": IntSpec(width=8, exhibits=a), "y": IntSpec(width=8)},
        order=("x", "y"), concatenation=Concatenation(order=ComponentOrder.RANDOM)))


def test_alignment_aligned_applies_the_class_pre_alignment_before_each_component():
    """§22.10.3.5: with `ALIGNMENT aligned` the concatenation's own pre-alignment specification
    runs "before encoding each component". §23.5.1 declares only one such group, so the class's
    own pre-alignment and the inter-component one are the same properties used twice."""
    fields = {"a": IntSpec(width=3), "b": IntSpec(width=3)}
    packed = ConcatenationSpec(
        fields=fields, order=("a", "b"),
        concatenation=Concatenation(alignment=ConcatenationAlignment.NONE),
        pre_alignment=PreAlignment(unit=UNIT_OCTET))
    assert encode_with_user(_objects(packed), "#C", {"a": 5, "b": 3},
                            outer=_pad_to_octet()) == bytes((0b101_011_00,))
    spread = ConcatenationSpec(
        fields=fields, order=("a", "b"),
        concatenation=Concatenation(alignment=ConcatenationAlignment.ALIGNED),
        pre_alignment=PreAlignment(unit=UNIT_OCTET))
    assert encode_with_user(_objects(spread), "#C", {"a": 5, "b": 3},
                            outer=_pad_to_octet()) == bytes((0b101_00000, 0b011_00000))


def test_an_absent_concatenation_group_means_every_default():
    """§22.10.2.6: "If it is not set then encoders and decoders act as if it was set with each
    encoding property taking its default value" — `textual`, `aligned`, `default-handle`. And
    since §22.2.1.1's default alignment unit is one bit, `aligned` inserts nothing."""
    fields = {"a": IntSpec(width=3), "b": IntSpec(width=3)}
    bare = ConcatenationSpec(fields=fields, order=("a", "b"))
    explicit = ConcatenationSpec(fields=fields, order=("a", "b"),
                                 concatenation=Concatenation())
    value = {"a": 5, "b": 3}
    assert (encode_with_user(_objects(bare), "#C", value, outer=_pad_to_octet())
            == encode_with_user(_objects(explicit), "#C", value, outer=_pad_to_octet()))


# --- §21.7.10: the fourth clause the handle turns on -------------------------------------------

def test_a_repetition_can_end_at_a_handle_now_that_handles_exist():
    """§21.7.10 and §22.7.3.11. The determination was refused with the words "§22.9's
    identification handles are not built"; they are, so it is not.

    §21.7.10 wants the handle from two sides — the repeated element **and** "each possible
    (taking account of optionality) following encoding class". Only the first is visible to a
    repetition object, and the refusal for a missing element handle says so.
    """
    element = IdentificationHandle("more", (0,), HandleValueSet.of_number(1))
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.HANDLE,
                            handle_id="more")
    ok = ConditionalRepetitionSpec(element=IntSpec(width=8, exhibits=element), space=space)
    rep = RepetitionSpec((ok,), SizeBounds(0, None))
    objects = _concat({"s": StringSpec(element=_OCTET, repetition=rep)}, ("s",))
    assert encode_with_user(objects, "#C", {"s": [0x81, 0xFF]}) == bytes((0x81, 0xFF))
    _refuses("22.9.3.1", lambda: encode_with_user(objects, "#C", {"s": [0x01]}))
    _refuses("21.7.10", lambda: ConditionalRepetitionSpec(element=_OCTET, space=space))
    _refuses("21.7.10", lambda: RepetitionSpace(
        determination=RepetitionSpaceDetermination.HANDLE, reference="n"))


def test_a_pattern_terminated_repetition_still_works_beside_the_handle_one():
    """A regression guard: enabling `handle` changed `record`'s early-return set, and the
    `pattern` determination shares it."""
    space = RepetitionSpace(determination=RepetitionSpaceDetermination.PATTERN,
                            termination_pattern=Pattern.from_octets(b"\x00"))
    rep = RepetitionSpec((ConditionalRepetitionSpec(element=_OCTET, space=space),),
                         SizeBounds(0, None))
    objects = _concat({"s": StringSpec(element=_OCTET, repetition=rep)}, ("s",))
    assert encode_with_user(objects, "#C", {"s": [72, 73]}) == b"HI\x00"
