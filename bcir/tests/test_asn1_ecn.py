"""X.692 ECN conformance — part one: the model and the built-in encoding object sets.

ECN is a notation for *defining* encoding rules, so most of what can be checked here is
structural rather than octet-level: does the class algebra of clause 9 behave as §9.6 says,
and do the two static laws (§9.5.2 and §12.2.5) refuse the constructions the standard calls
ambiguous?

The one place octets appear is the point of the whole chapter. §18.2.2's NOTE reads X.690
and X.691 as definitions of encoding objects, so applying a built-in set must produce
exactly what the corresponding rail already produces — and `test_naming_a_built_in_set_is_
the_whole_selection_mechanism` is the test that says so: one abstract value, two object
sets, two encodings, and nothing else varies.
"""

from __future__ import annotations

from bcir.asn1 import BER_OID, DER_OID
from bcir.asn1.codec import Asn1Error, Strictness
from bcir.asn1.constraints import Size, ValueRange
from bcir.asn1.ecn import (
    ABSENT_FROM_BUILTIN_SETS,
    BUILTIN_CLASSES,
    BUILTIN_SET_OID,
    CLASS_FOR_NOTATION,
    SHARED_CLASSES,
    BuiltinEncodingObjectSet,
    Category,
    CategoryGroup,
    EncodingApplication,
    EncodingClass,
    EncodingDefinitionModule,
    EncodingLinkModule,
    EncodingObject,
    EncodingObjectSet,
    builtin_object_set,
    category_group,
    encode_with,
)
from bcir.asn1.per import (
    BASIC_PER_ALIGNED_OID,
    BASIC_PER_UNALIGNED_OID,
    CANONICAL_PER_ALIGNED_OID,
    CANONICAL_PER_UNALIGNED_OID,
    PerRules,
    PerVariant,
    encode_per,
)
from bcir.asn1.schema import Component, Module, Primitive, Sequence
from bcir.asn1.tags import Universal


def _bounded_sequence() -> Sequence:
    return Sequence(
        (Component("v", Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 255))),),
        "S")


# --- clause 9: the class algebra ---------------------------------------------------------


def test_a_class_reference_begins_with_a_number_sign():
    """§9.2.1/§9.3.1 — the lexical rule that separates a class from a type."""
    try:
        EncodingClass("Sequence", Category.CONCATENATION)
    except Asn1Error as error:
        assert "9.2.1" in str(error)
    else:
        raise AssertionError("a class name without \"#\" must be refused")


def test_every_builtin_class_derives_from_a_primitive():
    """§9.6.3 — "All built-in encoding classes are derived from one of a small number of
    primitive encoding classes"."""
    primitives = {cls for cls in BUILTIN_CLASSES.values() if cls.derived_from is None}
    assert len(primitives) == 20, sorted(c.name for c in primitives)
    for cls in BUILTIN_CLASSES.values():
        assert cls.primitive() in primitives, cls.name


def test_table_2_maps_asn1_notation_to_its_encoding_class_and_primitive():
    """§11.2 Table 2, spot-checked where the mapping is not the obvious one."""
    assert CLASS_FOR_NOTATION["INTEGER"].name == "#INTEGER"
    assert CLASS_FOR_NOTATION["INTEGER"].primitive().name == "#INT"
    # ENUMERATED shares #INT with INTEGER -- the same fact PER's §14.1 index depends on.
    assert CLASS_FOR_NOTATION["ENUMERATED"].primitive().name == "#INT"
    # SET and SEQUENCE are both concatenation; SET OF and SEQUENCE OF both repetition.
    assert CLASS_FOR_NOTATION["SET"].primitive().name == "#CONCATENATION"
    assert CLASS_FOR_NOTATION["SEQUENCE OF"].primitive().name == "#REPETITION"
    assert CLASS_FOR_NOTATION["CHOICE"].primitive().name == "#ALTERNATIVES"
    # Every restricted character string type collapses to #CHARS, which is why one
    # encoding object can serve all of them.
    for notation in ("IA5String", "UTF8String", "VisibleString", "GeneralizedTime"):
        assert CLASS_FOR_NOTATION[notation].primitive().name == "#CHARS"
    # RELATIVE-OID shares the object identifier primitive with OBJECT IDENTIFIER.
    assert CLASS_FOR_NOTATION["RELATIVE-OID"].primitive().name == "#OBJECT-IDENTIFIER"


def test_a_class_assignment_keeps_the_category_of_what_it_assigns():
    """§9.6.1/§9.6.5 — `#My-Sequence ::= #SEQUENCE` "is still an encoding class concerned
    with the concatenation of components"."""
    edm = EncodingDefinitionModule("Test-EDM")
    derived = edm.assign_class("#My-Sequence", BUILTIN_CLASSES["#SEQUENCE"])
    assert derived.category is Category.CONCATENATION
    assert derived.primitive().name == "#CONCATENATION"
    assert derived.derives_from(BUILTIN_CLASSES["#SEQUENCE"])
    assert not BUILTIN_CLASSES["#SEQUENCE"].derives_from(derived)


def test_a_class_assignment_cannot_shadow_a_builtin_or_a_sibling():
    """§9.3.3 — a structure-based class "cannot have the same names as encoding classes
    that are imported into the module"."""
    edm = EncodingDefinitionModule("Test-EDM")
    edm.assign_class("#Mine", BUILTIN_CLASSES["#INTEGER"])
    for name in ("#Mine", "#INTEGER"):
        try:
            edm.assign_class(name, BUILTIN_CLASSES["#INTEGER"])
        except Asn1Error as error:
            assert "9.3.3" in str(error)
        else:
            raise AssertionError(f"{name} must not be assignable twice")


def test_an_encoding_procedure_class_cannot_be_renamed():
    """§9.6.7 — the four classes that "cannot be assigned new names"."""
    for name in ("#OUTER", "#TRANSFORM", "#CONDITIONAL-INT", "#CONDITIONAL-REPETITION"):
        cls = BUILTIN_CLASSES[name]
        assert cls.group is CategoryGroup.ENCODING_PROCEDURE
        try:
            cls.derive("#Nope")
        except Asn1Error as error:
            assert "9.6.7" in str(error)
        else:
            raise AssertionError(f"{name} must not be renameable")


def test_the_category_groups_are_the_three_clause_9_6_7_names_and_no_more():
    """§9.6.7, including the part that is easy to get wrong.

    Optionality and tag are categories (§9.6.6) but belong to NO group: §9.6.4 lists them
    alongside the groups rather than inside one. Inventing a fourth group, or folding them
    into the bit-field group, would put them somewhere the standard deliberately does not.
    """
    assert category_group(Category.OPTIONALITY) is None
    assert category_group(Category.TAG) is None
    assert category_group(Category.INTEGER) is CategoryGroup.BIT_FIELD
    assert category_group(Category.ENCODING_STRUCTURE) is CategoryGroup.BIT_FIELD
    assert category_group(Category.CONCATENATION) is CategoryGroup.ENCODING_CONSTRUCTOR
    assert category_group(Category.ALTERNATIVES) is CategoryGroup.ENCODING_CONSTRUCTOR
    assert category_group(Category.REPETITION) is CategoryGroup.ENCODING_CONSTRUCTOR


# --- §9.5.2 and §18.1.7: the law that makes application unambiguous ----------------------


def test_a_set_holds_at_most_one_object_per_class():
    """§9.5.2 — "Thus there is no ambiguity when an encoding object set is applied"."""
    sequence = BUILTIN_CLASSES["#SEQUENCE"]
    try:
        EncodingObjectSet((EncodingObject(sequence, "a"), EncodingObject(sequence, "b")))
    except Asn1Error as error:
        assert "9.5.2" in str(error)
    else:
        raise AssertionError("two objects for one class must be refused")


def test_the_one_object_rule_is_keyed_on_class_identity_not_on_category():
    """§9.6.2 — "encoding objects of both the old encoding class and the new encoding class
    can appear in an encoding object set".

    This is the subtle half of §9.5.2 and the reason the check cannot be written against
    the category or the primitive: `#SEQUENCE` and `#My-Sequence ::= #SEQUENCE` share both,
    and a set holding an object for each is explicitly legal.
    """
    edm = EncodingDefinitionModule("Test-EDM")
    mine = edm.assign_class("#My-Sequence", BUILTIN_CLASSES["#SEQUENCE"])
    both = EncodingObjectSet((EncodingObject(BUILTIN_CLASSES["#SEQUENCE"], "builtin"),
                              EncodingObject(mine, "mine")))
    assert both.object_for(mine).name == "mine"
    assert both.object_for(BUILTIN_CLASSES["#SEQUENCE"]).name == "builtin"


def test_a_set_admits_no_encoding_procedure_class_except_outer():
    """§18.1.7's second half."""
    for name in ("#TRANSFORM", "#CONDITIONAL-INT", "#CONDITIONAL-REPETITION"):
        try:
            EncodingObjectSet((EncodingObject(BUILTIN_CLASSES[name]),))
        except Asn1Error as error:
            assert "18.1.7" in str(error)
        else:
            raise AssertionError(f"{name} must not be admitted to a set")
    # #OUTER is the exception the clause names.
    EncodingObjectSet((EncodingObject(BUILTIN_CLASSES["#OUTER"]),))


def test_a_union_re_checks_the_one_object_law():
    """§18.1.5's `UnionMark` cannot be a way around §9.5.2."""
    one = EncodingObjectSet((EncodingObject(BUILTIN_CLASSES["#INTEGER"], "a"),))
    try:
        one.union(EncodingObjectSet((EncodingObject(BUILTIN_CLASSES["#INTEGER"], "b"),)))
    except Asn1Error as error:
        assert "9.5.2" in str(error)
    else:
        raise AssertionError("a union that duplicates a class must be refused")


def test_dereferencing_finds_the_base_class_object_and_a_specific_one_wins():
    """§18.2.3 and its NOTE 2 — "with appropriate de-referencing", and an object added for
    the specific class "will take precedence"."""
    edm = EncodingDefinitionModule("Test-EDM")
    mine = edm.assign_class("#My-Sequence", BUILTIN_CLASSES["#SEQUENCE"])
    builtin = builtin_object_set(BuiltinEncodingObjectSet.DER)
    assert builtin.object_for(mine).encoding_class == BUILTIN_CLASSES["#SEQUENCE"]
    specialized = builtin.union(EncodingObjectSet((EncodingObject(mine, "mine"),)))
    assert specialized.object_for(mine).name == "mine"


# --- clause 18.2: the built-in encoding object sets ---------------------------------------


def test_table_4_object_identifiers_agree_with_the_defining_clauses():
    """§18.2.2 Table 4, cross-checked against the rails that define those rules.

    Table 4 as printed carries a defect: the `PER-CANONICAL-UNALIGNED` row is given as
    `{joint-iso-itu-t(2) packed-encoding(3) canonical(1) unaligned(1)}` — four arcs, with
    `asn1(1)` missing, while its three siblings have five. X.691 §33.2 is the defining
    clause and gives `{joint-iso-itu-t asn1(1) packed-encoding(3) canonical(1)
    unaligned(1)}`. Transcribing Table 4 literally would put one built-in set under the
    wrong parent arc, which is exactly the kind of slip that survives a round trip and
    fails an interop test, so every value is pinned against the constant its own rail
    already carries.
    """
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.PER_BASIC_ALIGNED] \
        == BASIC_PER_ALIGNED_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED] \
        == BASIC_PER_UNALIGNED_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.PER_CANONICAL_ALIGNED] \
        == CANONICAL_PER_ALIGNED_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.PER_CANONICAL_UNALIGNED] \
        == CANONICAL_PER_UNALIGNED_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.PER_CANONICAL_UNALIGNED] \
        == (2, 1, 3, 1, 1), "the asn1(1) arc Table 4 drops"
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.BER] == BER_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.DER] == DER_OID
    assert BUILTIN_SET_OID[BuiltinEncodingObjectSet.CER] == (2, 1, 2, 0)
    assert len(BUILTIN_SET_OID) == 7, "18.2.1 reserves exactly seven names"
    assert len(set(BUILTIN_SET_OID.values())) == 7, "and they name seven distinct rules"


def test_no_builtin_set_claims_alternatives_repetition_or_pad():
    """§18.2.4 — "They do not contain encoding objects for #ALTERNATIVES, #REPETITION, and
    #PAD."

    Stated as a prohibition rather than an omission: a set that carried one would be
    claiming BER or PER defines something neither does, and the resulting encoding would be
    this rail's invention rather than the standard's.
    """
    for which in BuiltinEncodingObjectSet:
        objects = builtin_object_set(which)
        for cls in ABSENT_FROM_BUILTIN_SETS:
            assert objects.object_for(cls) is None, f"{which.value} claims {cls.name}"


def test_every_builtin_set_carries_the_shared_classes():
    """§18.2.4 — the classes every set holds "identical encoding objects" for."""
    for which in BuiltinEncodingObjectSet:
        objects = builtin_object_set(which)
        for cls in SHARED_CLASSES:
            assert objects.object_for(cls) is not None, f"{which.value} lacks {cls.name}"


def test_the_shared_object_design_errors_are_refused_at_application():
    """§18.2.5.1/§18.2.5.3/§18.2.5.4 — three "ECN design error" conditions.

    Each is a statement about the ECN *specification*, so refusing beats emitting: the
    standard defines no width for an unbounded #INT under these objects, and inventing one
    would produce octets no conforming peer could read back.
    """
    per = builtin_object_set(BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED)
    unbounded = Primitive(Universal.INTEGER, "INTEGER")
    try:
        encode_with(per, BUILTIN_CLASSES["#INT"], unbounded, 1)
    except Asn1Error as error:
        assert "18.2.5.1" in str(error)
    else:
        raise AssertionError("an unbounded #INT must be refused")

    loose = Primitive(Universal.OCTET_STRING, "OCTET STRING", Size(ValueRange(1, 4)))
    try:
        encode_with(per, BUILTIN_CLASSES["#OCTETS"], loose, b"ab")
    except Asn1Error as error:
        assert "18.2.5.3" in str(error)
    else:
        raise AssertionError("a multi-size #OCTETS must be refused")

    optional = Sequence(
        (Component("v", Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 3))),
         Component("w", Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 3)),
                   tag=0, optional=True)), "S")
    try:
        encode_with(per, BUILTIN_CLASSES["#CONCATENATION"], optional, {"v": 1})
    except Asn1Error as error:
        assert "18.2.5.4" in str(error)
    else:
        raise AssertionError("a #CONCATENATION with optional components must be refused")


def test_cer_is_named_but_has_no_realization_on_this_rail():
    """§18.2.1 reserves the name; X.690 §9.1 is why this repo does not implement it.

    The set still exists rather than being quietly dropped from the enum — refusing to name
    CER would misreport what ECN offers. Applying it says exactly why there are no octets.
    """
    cer = builtin_object_set(BuiltinEncodingObjectSet.CER)
    assert cer.object_for(BUILTIN_CLASSES["#SEQUENCE"]) is not None
    try:
        encode_with(cer, BUILTIN_CLASSES["#SEQUENCE"], _bounded_sequence(), {"v": 200})
    except Asn1Error as error:
        assert "does not implement" in str(error)
    else:
        raise AssertionError("CER must not silently produce octets")


# --- clause 13: applying a set is the selection mechanism ---------------------------------


def test_naming_a_built_in_set_is_the_whole_selection_mechanism():
    """§18.2.2's NOTE, made executable: one abstract value, six sets, six encodings.

    This is the test the chapter exists for. Nothing about the value or the type changes —
    only which encoding object set is named — and the octets that come out are byte-for-byte
    the ones the X.690 and X.691 rails already produce and already pin against each
    standard's own Annex A. ECN's contribution is the naming and the algebra, not new bits.
    """
    kind = _bounded_sequence()
    value = {"v": 200}
    sequence = BUILTIN_CLASSES["#SEQUENCE"]

    per_expected = {
        BuiltinEncodingObjectSet.PER_BASIC_ALIGNED: (PerRules.BASIC, PerVariant.ALIGNED),
        BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED:
            (PerRules.BASIC, PerVariant.UNALIGNED),
        BuiltinEncodingObjectSet.PER_CANONICAL_ALIGNED:
            (PerRules.CANONICAL, PerVariant.ALIGNED),
        BuiltinEncodingObjectSet.PER_CANONICAL_UNALIGNED:
            (PerRules.CANONICAL, PerVariant.UNALIGNED),
    }
    for which, (rules, variant) in per_expected.items():
        got = encode_with(builtin_object_set(which), sequence, kind, value)
        assert got == encode_per(kind, value, variant=variant, rules=rules), which.value

    module = Module("<ecn>", (), {"T": kind})
    for which in (BuiltinEncodingObjectSet.BER, BuiltinEncodingObjectSet.DER):
        got = encode_with(builtin_object_set(which), sequence, kind, value)
        assert got == module.encode("T", value), which.value
        assert module.decode("T", got, strictness=Strictness.DER) == value

    # And the point of the exercise: the same value, different lengths, chosen by name.
    packed = encode_with(builtin_object_set(
        BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED), sequence, kind, value)
    tagged = encode_with(builtin_object_set(BuiltinEncodingObjectSet.DER), sequence,
                         kind, value)
    assert len(packed) == 1 and len(tagged) == 6, (packed.hex(), tagged.hex())


def test_applying_a_set_that_holds_no_object_for_the_class_is_refused():
    """§9.5.1 — a set is what determines an encoding, so a gap is not a default."""
    empty = EncodingObjectSet((), "Empty")
    try:
        encode_with(empty, BUILTIN_CLASSES["#SEQUENCE"], _bounded_sequence(), {"v": 1})
    except Asn1Error as error:
        assert "9.5.1" in str(error)
    else:
        raise AssertionError("an empty set must not encode anything")


# --- clause 12: the Encoding Link Module -------------------------------------------------


def test_an_elm_applies_encodings_and_must_apply_at_least_one():
    """§12.1.9 — "the sole function of an ELM is to apply encodings"."""
    der = builtin_object_set(BuiltinEncodingObjectSet.DER)
    elm = EncodingLinkModule("Link", (EncodingApplication(("#Message",), der),))
    assert elm.encodings_for("#Message") is der
    assert elm.encodings_for("#Other") is None
    try:
        EncodingLinkModule("Empty", ())
    except Asn1Error as error:
        assert "12.1.9" in str(error)
    else:
        raise AssertionError("an ELM with no application must be refused")
    try:
        EncodingApplication((), der)
    except Asn1Error as error:
        assert "12.2.1" in str(error)
    else:
        raise AssertionError("an ENCODE naming no class must be refused")


def test_an_elm_never_encodes_the_same_type_twice():
    """§12.2.5 — the same ambiguity §9.5.2 removes, one level up.

    Two applications naming one type would leave that type with two encodings and no rule
    for choosing between them, so this is refused at construction rather than resolved by
    precedence.
    """
    der = builtin_object_set(BuiltinEncodingObjectSet.DER)
    per = builtin_object_set(BuiltinEncodingObjectSet.PER_BASIC_UNALIGNED)
    try:
        EncodingLinkModule("Link", (EncodingApplication(("#A", "#B"), der),
                                    EncodingApplication(("#B",), per)))
    except Asn1Error as error:
        assert "12.2.5" in str(error)
    else:
        raise AssertionError("applying encodings twice to one type must be refused")
    # Distinct types in separate applications are exactly what §12.2.1 is for.
    elm = EncodingLinkModule("Link", (EncodingApplication(("#A",), der),
                                      EncodingApplication(("#B",), per)))
    assert elm.encodings_for("#A") is der and elm.encodings_for("#B") is per
