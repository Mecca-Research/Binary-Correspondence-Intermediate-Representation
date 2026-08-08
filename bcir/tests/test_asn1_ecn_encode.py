"""X.692 §17.5: the `EncodeStructure`, and the three clauses that each demand `WITH`.

This production is what §22.1's `REPLACE` was waiting on, and the repository had the wrong
clause recorded until D.3.2.3 was read — §22.1.2.6 *classifies* a replacement structure's
auxiliary fields and never says how they are encoded. §17.5 does, component by component, and
`encoding_for` is the function that turns a field list into one object per field.
"""

from bcir.asn1.ecn_encode import (
    USE_SET, ComponentEncoding, EncodeStructure, GovernorCategory,
)
from bcir.asn1.tags import Asn1Error


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


def _structure(**changes) -> EncodeStructure:
    """D.3.2.3's shape: a concatenation of `determinant` and an optional `component`."""
    defaults = dict(
        governor=GovernorCategory.CONCATENATION,
        component_names=("determinant", "component"),
        optional_names=frozenset({"component"}),
        components=(
            ComponentEncoding(identifier="determinant", element="determinant-encoding"),
            ComponentEncoding(identifier="component", element=USE_SET,
                              optional_encoding="if-component-present-encoding",
                              actuals=("determinant",)),
        ),
        combined="Sequence2-combined-encoding-object-set")
    defaults.update(changes)
    return EncodeStructure(**defaults)


# --- the three independent reasons CombinedEncodings must be present ----------------------

def test_three_clauses_demand_the_object_set_for_three_different_reasons():
    """§17.5.3, §17.5.6 and §17.5.10 each require the trailing `WITH <object set>`, and an
    implementation that checks one accepts specifications the other two forbid.

    They are three different *repairs*, which is why they are three messages: add a
    `STRUCTURED WITH`, drop a `USE-SET`, or write the missing component in.
    """
    # §17.5.3: no StructureEncoding, so nothing else encodes the constructor itself. Its NOTE
    # is the whole argument — "a complete encoding has to be produced".
    _refuses("17.5.3", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a",),
        components=(ComponentEncoding(identifier="a", element="obj"),)))

    # §17.5.6: USE-SET *means* "apply the CombinedEncodings", so it is a dangling reference
    # without them. A StructureEncoding is present here, so §17.5.3 is satisfied and only
    # §17.5.6 can fire — which is what makes these genuinely independent.
    _refuses("17.5.6", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a",),
        components=(ComponentEncoding(identifier="a", element=USE_SET),),
        structure_encoding="concat-encoding"))
    _refuses("17.5.6", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, structure_encoding=USE_SET))

    # §17.5.10: a component nobody wrote an encoding for. Again with a StructureEncoding
    # present and no USE-SET anywhere, so this is the only clause left to fire.
    _refuses("17.5.10", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a", "b"),
        components=(ComponentEncoding(identifier="a", element="obj"),),
        structure_encoding="concat-encoding"))


def test_an_empty_component_list_is_a_specification_rather_than_a_degenerate_case():
    """§17.5.7's `ComponentEncodingList ::= ComponentEncoding "," *` is zero-or-more — unlike
    C.1's `"," +`, which is why the two productions cannot share a reader.

    Empty means "encode every component with the object set", which §17.5.10 makes complete and
    §17.5.4 makes the only shape where the constructor's own object may specify replacement
    actions.
    """
    empty = EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a", "b"),
        combined="per-basic-unaligned")
    assert empty.missing_components() == ("a", "b")
    assert empty.encoding_for("a") == "per-basic-unaligned"
    assert empty.replacement_actions_allowed() is True
    # §17.5.4: with any ComponentEncoding present, the constructor's object "shall not specify
    # any replacement actions". Reported rather than refused — the object is named here and
    # DEFINED elsewhere, so the fault belongs to whoever pairs the two.
    assert _structure().replacement_actions_allowed() is False


# --- §17.5.8: at most one per component, in the components' own order ---------------------

def test_component_encodings_are_a_subsequence_of_the_components_not_a_prefix():
    """§17.5.8: "There shall be at most one `ComponentEncoding` for each component ... The
    `ComponentEncoding`s shall be in the same textual order."

    A *subset* is legal — §17.5.10 covers whatever is left out — so the order test is a
    subsequence test. Equality or a prefix test would both reject the legal middle case, and
    the weaker check is the correct one rather than the lenient one.
    """
    skipping = EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a", "b", "c"),
        components=(ComponentEncoding(identifier="a", element="oa"),
                    ComponentEncoding(identifier="c", element="oc")),
        combined="set")
    assert skipping.missing_components() == ("b",)

    _refuses("17.5.8", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a", "b"),
        components=(ComponentEncoding(identifier="b", element="ob"),
                    ComponentEncoding(identifier="a", element="oa")),
        combined="set"))
    _refuses("17.5.8", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a",),
        components=(ComponentEncoding(identifier="a", element="oa"),
                    ComponentEncoding(identifier="a", element="oa2")),
        combined="set"))
    _refuses("17.5.11", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a",),
        components=(ComponentEncoding(identifier="nope", element="o"),),
        combined="set"))


# --- the two biconditionals ---------------------------------------------------------------

def test_the_optional_spec_is_used_if_and_only_if_the_component_is_optional():
    """§17.5.9, in the same "if and only if" shape §22.1.2.5 uses for `INSERT AT HEAD`. Both
    directions are faults: an optional component with no `OPTIONAL-ENCODING` leaves the class in
    the optionality category unencoded, and a mandatory one carrying the clause gives §17.5.14
    nothing to encode — "the class in the optionality category of the component", which is
    absent."""
    _structure()
    _refuses("17.5.9", lambda: _structure(components=(
        ComponentEncoding(identifier="determinant", element="determinant-encoding"),
        ComponentEncoding(identifier="component", element=USE_SET))))
    _refuses("17.5.9", lambda: _structure(components=(
        ComponentEncoding(identifier="determinant", element="determinant-encoding",
                          optional_encoding="oops"),
        ComponentEncoding(identifier="component", element=USE_SET,
                          optional_encoding="if-component-present-encoding"))))


def test_an_identifier_is_omitted_only_for_an_unnamed_repetition_element():
    """§17.5.11: the identifier "shall be omitted if and only if the governing encoding
    constructor is a class in the repetition category for which there is no identifier on the
    repeated element". Three ways to get it wrong, all refused."""
    EncodeStructure(governor=GovernorCategory.REPETITION, unnamed_element=True,
                    components=(ComponentEncoding(element="element-encoding"),),
                    combined="set")
    # A named component inside an unnamed-element repetition.
    _refuses("17.5.11", lambda: EncodeStructure(
        governor=GovernorCategory.REPETITION, unnamed_element=True,
        components=(ComponentEncoding(identifier="a", element="o"),), combined="set"))
    # An unnamed component where the governor does name its components.
    _refuses("17.5.11", lambda: EncodeStructure(
        governor=GovernorCategory.CONCATENATION, component_names=("a",),
        components=(ComponentEncoding(element="o"),), combined="set"))
    # §17.5.2 admits only three categories, and the unnamed-element escape is the repetition
    # one's alone.
    _refuses("17.5.11", lambda: EncodeStructure(
        governor=GovernorCategory.ALTERNATIVES, unnamed_element=True, combined="set"))


# --- what the replacement machinery wants out of it ---------------------------------------

def test_encoding_for_is_the_function_replace_was_waiting_on():
    """§22.1.3.5 says a replacement structure's other fields "shall be set according to the
    specification in the replacement structure encoding object", and this production is that
    specification. `encoding_for` turns a field list into one object per field — which is
    exactly `ecn_user.ReplacementStructure`'s `auxiliary` mapping.

    D.3.2.3's shape is the one modelled here: a named object for the auxiliary `determinant`,
    and `USE-SET` for the instantiated `component`, which resolves to the combined set.
    """
    structure = _structure()
    assert structure.encoding_for("determinant") == "determinant-encoding"
    assert structure.encoding_for("component") == "Sequence2-combined-encoding-object-set"
    assert structure.missing_components() == ()
    _refuses("is not a component", lambda: structure.encoding_for("nope"))


def test_use_set_is_a_keyword_and_not_a_name_a_module_could_take():
    """§17.5.1's `EncodingOrUseSet ::= EncodingObject | USE-SET`. A module is free to define an
    encoding object called `USE-SET`, so the sentinel is not that string — otherwise the two
    would collide and the collision would silently redirect an encoding to the object set."""
    assert USE_SET != "USE-SET"
    named = ComponentEncoding(identifier="a", element="USE-SET")
    assert named.uses_set() is False
    assert ComponentEncoding(identifier="a", element=USE_SET).uses_set() is True
    assert ComponentEncoding(identifier="a", element="o", tag=USE_SET).uses_set() is True
