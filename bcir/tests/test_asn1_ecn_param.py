"""X.692 Annex C: X.683's parameterization as ECN modifies it, and §22.1.2's use of it.

Two of these tests exist because the *obvious* implementation is wrong in a way nothing
catches:

* **The delimiters.** ECN's parameter list is `{< >}`, not X.683's `{ }` (C.1, C.4). Reusing
  an X.683 parser accepts the wrong spelling and rejects the only right one.
* **§17.5.17's scan is breadth-first.** The natural recursive walk is depth-first, and the two
  disagree only when an inner name shadows an outer one — at which point both still resolve to
  a real field, so the specification silently points somewhere else.
"""

from bcir.asn1.ecn_param import (
    ActualKind, ActualParameter, ActualParameterList, AssignmentKind, GovernorKind, Parameter,
    ParameterizedAssignment, ParameterizedReference, ParameterKind, ParameterList,
    ReplacementParameterization, bare_use, check_actuals, resolve_component,
)
from bcir.asn1.tags import Asn1Error


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


def _class_param(name: str = "#D") -> Parameter:
    return Parameter(name, ParameterKind.ENCODING_CLASS)


def _reference_param(name: str = "ref") -> Parameter:
    return Parameter(name, ParameterKind.IDENTIFIER, GovernorKind.REFERENCE)


def _object_set_param(name: str = "set") -> Parameter:
    return Parameter(name, ParameterKind.ENCODING_OBJECT_SET, GovernorKind.ENCODINGS)


# --- C.1: the governor each dummy kind requires ------------------------------------------

def test_a_dummy_carries_exactly_the_governor_its_kind_requires():
    """C.1 makes the `ParamGovernor` mandatory for four kinds and forbidden for the fifth, so
    "governor present" is a fact about the dummy rather than a style choice."""
    _class_param()                                                            # a) no governor
    Parameter("v", ParameterKind.VALUE, GovernorKind.ENCODING_CLASS_FIELD_TYPE, "#INT.&size")
    _reference_param()                                                        # governed by REFERENCE
    Parameter("obj", ParameterKind.ENCODING_OBJECT,
              GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS, "#INT")
    _object_set_param()                                                       # governed by #ENCODINGS

    _refuses("C.1 a)", lambda: Parameter(
        "#D", ParameterKind.ENCODING_CLASS, GovernorKind.REFERENCE))
    _refuses("shall be governed by REFERENCE", lambda: Parameter(
        "ref", ParameterKind.IDENTIFIER))
    _refuses("shall be governed by #ENCODINGS", lambda: Parameter(
        "set", ParameterKind.ENCODING_OBJECT_SET, GovernorKind.REFERENCE))
    _refuses("a Parameter needs a DummyReference", lambda: Parameter(
        "", ParameterKind.ENCODING_CLASS))


def test_a_keyword_governor_has_nothing_to_name_and_a_class_governor_must():
    """`REFERENCE` and `#ENCODINGS` are keywords; `EncodingClassFieldType` and
    `DefinedOrBuiltinEncodingClass` name something. Both directions are faults."""
    _refuses("names nothing", lambda: Parameter(
        "obj", ParameterKind.ENCODING_OBJECT, GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS))
    _refuses("keyword governor with nothing to name",
             lambda: Parameter("ref", ParameterKind.IDENTIFIER, GovernorKind.REFERENCE, "#INT"))


def test_dummy_governors_are_legal_asn1_and_refused_in_ecn():
    """C.1's NOTE: "DummyGovernors are not allowed in ECN". X.683 lets one dummy govern
    another; the identical text is an error here, which is why the check is on the list rather
    than on the parameter — a governor is only a *dummy* governor relative to its siblings."""
    # The same parameter is fine when `#Outer` is not a sibling dummy.
    ParameterList((Parameter("obj", ParameterKind.ENCODING_OBJECT,
                             GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS, "#Outer"),))
    _refuses("DummyGovernors are not allowed in ECN", lambda: ParameterList((
        _class_param("#Outer"),
        Parameter("obj", ParameterKind.ENCODING_OBJECT,
                  GovernorKind.DEFINED_OR_BUILTIN_ENCODING_CLASS, "#Outer"))))


def test_the_parameter_list_is_written_with_ecns_delimiters_and_is_never_empty():
    """C.1 is `"{<" Parameter "," + ">}"`. The delimiters are the modification to X.683 §8.3,
    and `"," +` makes an empty list not a `ParameterList` at all — which matters because C.3
    gives `{<>}` an opposite meaning as an empty ACTUAL list."""
    params = ParameterList((_class_param(), _reference_param()))
    assert params.render() == "{<#D, ref>}"
    assert params.names() == ("#D", "ref")
    assert len(params) == 2

    _refuses("one or more", lambda: ParameterList(()))
    _refuses("appears twice", lambda: ParameterList((_class_param(), _class_param())))


# --- C.4: which actual fits which dummy ---------------------------------------------------

def test_each_dummy_kind_takes_its_own_actual_alternative():
    """C.4 a)-g) pair one alternative with each dummy kind. The pairing is exact: an encoding
    object where an encoding class is wanted is a fault even though both are "an encoding
    something"."""
    params = ParameterList((_class_param(),))
    check_actuals(params, ActualParameterList(
        (ActualParameter(ActualKind.ENCODING_CLASS, "#INT"),)))
    _refuses("C.4", lambda: check_actuals(params, ActualParameterList(
        (ActualParameter(ActualKind.ENCODING_OBJECT, "int-object"),))))


def test_a_reference_dummy_has_four_spellings_and_one_of_them_lives_a_clause_away():
    """C.4 h) gives three — "the `identifier`, `STRUCTURE` or `OUTER` alternative". The
    production also lists `ComponentIdList`, which h) never mentions; §17.5.15 supplies the
    missing sentence, saying a `REFERENCE` actual "can either be supplied as a dummy parameter
    ... or it can be supplied as a `ComponentIdList`". All four are accepted here."""
    params = ParameterList((_reference_param(),))
    for actual in (ActualParameter(ActualKind.IDENTIFIER, "length"),
                   ActualParameter(ActualKind.COMPONENT_ID_LIST, "header.length"),
                   ActualParameter(ActualKind.STRUCTURE),
                   ActualParameter(ActualKind.OUTER)):
        check_actuals(params, ActualParameterList((actual,)))
    _refuses("C.4", lambda: check_actuals(params, ActualParameterList(
        (ActualParameter(ActualKind.VALUE, "3"),))))


def test_the_two_keyword_actuals_denote_themselves_and_the_others_do_not():
    """`STRUCTURE` and `OUTER` are keywords, so a name written beside one means nothing;
    every other alternative is a name and an empty one means nothing either."""
    assert ActualParameter(ActualKind.OUTER).text == ""
    _refuses("denotes itself", lambda: ActualParameter(ActualKind.STRUCTURE, "#Frame"))
    _refuses("needs a value", lambda: ActualParameter(ActualKind.ENCODING_CLASS))


def test_a_component_id_list_is_a_dotted_path_and_its_parts_are_identifiers():
    """§15.3.1: `ComponentIdList ::= identifier "." +`."""
    actual = ActualParameter(ActualKind.COMPONENT_ID_LIST, "outer.middle.inner")
    assert actual.components() == ("outer", "middle", "inner")
    assert ActualParameter(ActualKind.COMPONENT_ID_LIST, "solo").components() == ("solo",)
    _refuses("15.3.1", lambda: ActualParameter(ActualKind.COMPONENT_ID_LIST, "outer..inner"))
    _refuses("is not a ComponentIdList",
             lambda: ActualParameter(ActualKind.OUTER).components())


def test_the_wrong_count_and_the_wrong_kinds_are_named_as_two_different_faults():
    """X.683 §9.6 for the count, C.4 for the kinds. A specification with the right count and
    the wrong kinds is a different mistake, and only the second is fixable from the message."""
    params = ParameterList((_class_param(), _reference_param()))
    _refuses("9.6", lambda: check_actuals(params, ActualParameterList(
        (ActualParameter(ActualKind.ENCODING_CLASS, "#INT"),))))
    check_actuals(params, ActualParameterList((
        ActualParameter(ActualKind.ENCODING_CLASS, "#INT"),
        ActualParameter(ActualKind.OUTER))))


# --- C.3: the empty actual list is a reference, not an instantiation -----------------------

def test_an_empty_actual_list_is_a_legal_reference_and_an_empty_parameter_list_is_not():
    """C.3 modifies X.683 §9.1 to `ParameterizedReference ::= Reference | Reference "{<" ">}"`.
    So `Foo` and `Foo{<>}` denote the same thing while `{<>}` is not a `ParameterList` at all
    — the two productions share their delimiters and disagree about zero."""
    bare = ParameterizedReference("#Length-prefixed")
    empty = ParameterizedReference("#Length-prefixed", ActualParameterList(()))
    assert bare.is_bare and not empty.is_bare
    assert bare.render() == "#Length-prefixed"
    assert empty.render() == "#Length-prefixed{<>}"
    _refuses("C.1", lambda: ParameterList(()))


# --- C.2: the three parameterized assignments, and §8.4's ECN scope rule -------------------

def test_only_an_object_assignment_carries_a_governor_before_its_assignment_sign():
    """C.2 gives the object assignment a `DefinedOrBuiltinEncodingClass` between the parameter
    list and the `::=`; the class and object-set forms have no such slot."""
    params = ParameterList((_class_param(),))
    ParameterizedAssignment("obj", AssignmentKind.ENCODING_OBJECT, params, governor="#Wrapper")
    ParameterizedAssignment("#Wrapper", AssignmentKind.ENCODING_CLASS, params)
    _refuses("C.2", lambda: ParameterizedAssignment(
        "obj", AssignmentKind.ENCODING_OBJECT, params))
    _refuses("C.2", lambda: ParameterizedAssignment(
        "#Wrapper", AssignmentKind.ENCODING_CLASS, params, governor="#Other"))


def test_a_dummy_reaches_backwards_across_the_assignment_sign_only_for_an_object():
    """C.2 modifies X.683 §8.4 so that for a `ParameterizedEncodingObjectAssignment` "the
    scope extends to the `DefinedOrBuiltinEncodingClass` which **precedes** the `::=`", and
    its NOTE gives the shape that needs it::

        new-component-encoding {<#Any-class>} #New-component {<#Any-class>} ::= { ... }

    Under X.683's unmodified scope that line does not parse. The extension is granted to the
    object form alone, which is what makes the object-set form below a fault.
    """
    params = ParameterList((_class_param("#Any-class"),))
    borrowed = ActualParameterList((ActualParameter(ActualKind.ENCODING_CLASS, "#Any-class"),))
    ParameterizedAssignment("new-component-encoding", AssignmentKind.ENCODING_OBJECT,
                            params, governor="#New-component", governor_actuals=borrowed)
    _refuses("8.4", lambda: ParameterizedAssignment(
        "some-set", AssignmentKind.ENCODING_OBJECT_SET, params, governor_actuals=borrowed))


def test_instantiation_checks_the_correspondence_and_hands_back_bindings():
    """X.683 §9.7. Bindings rather than a substituted body: substituting into an ECN body
    needs that body's own vocabulary, which this layer deliberately does not have."""
    assignment = ParameterizedAssignment(
        "#Length-prefixed", AssignmentKind.ENCODING_CLASS, ParameterList((_class_param(),)))
    bindings = assignment.instantiate(ActualParameterList(
        (ActualParameter(ActualKind.ENCODING_CLASS, "#INT"),)))
    assert bindings["#D"].text == "#INT"
    _refuses("#Length-prefixed", lambda: assignment.instantiate(ActualParameterList(())))


# --- §22.1.2: what a replacement's parameterization has to look like ----------------------

def _replacement(**changes) -> ReplacementParameterization:
    defaults = dict(
        structure=ParameterList((_class_param(),)),
        encoded_by=ParameterList((_class_param(),)),
        governor_actuals=ActualParameterList(
            (ActualParameter(ActualKind.ENCODING_CLASS, "#D"),)))
    defaults.update(changes)
    return ReplacementParameterization(**defaults)


def test_a_replacement_structure_has_exactly_one_encoding_class_parameter():
    """§22.1.2.2: "The `WITH` replacement structures shall be parameterized encoding structures
    with a single encoding class parameter." Exactly one, and of exactly that kind — so a
    replacement parameterized over an object set is refused although C.1 admits such a dummy
    anywhere else."""
    _replacement()
    _refuses("22.1.2.2", lambda: _replacement(
        structure=ParameterList((_class_param(), _class_param("#E")))))
    _refuses("22.1.2.2", lambda: _replacement(
        structure=ParameterList((_object_set_param(),))))


def test_the_encoded_by_objects_governor_is_the_structure_instantiated_with_its_own_dummy():
    """§22.1.2.4: the objects "shall be defined in a parameterized encoding object assignment
    in which the governor is the corresponding `WITH` parameterized encoding structure,
    **instantiated with `#D`**".

    Not the structure — the structure applied to the object's own dummy. That is exactly the
    shape C.2's §8.4 modification exists to permit, so the two clauses are a matched pair.
    """
    _replacement()
    _refuses("22.1.2.4", lambda: _replacement(governor_actuals=None))
    _refuses("22.1.2.4", lambda: _replacement(governor_actuals=ActualParameterList(
        (ActualParameter(ActualKind.ENCODING_CLASS, "#Something-else"),))))
    _refuses("22.1.2.4", lambda: _replacement(
        encoded_by=ParameterList((_reference_param(),)), insert_at_head=True))


def test_the_reference_parameter_is_present_if_and_only_if_insert_at_head_is():
    """§22.1.2.5 makes it a biconditional, and both directions are real faults: a head-end
    insertion with no `REFERENCE` dummy has no way to reach the inserted structure (§22.1.2.7
    has the `ENCODED BY` object set its fields "through its `REFERENCE` parameter"), and a
    `REFERENCE` dummy with no head-end insertion would be instantiated with nothing."""
    with_both = _replacement(
        encoded_by=ParameterList((_class_param(), _reference_param())), insert_at_head=True)
    assert with_both.takes_object_set() is False

    _refuses("if and only if", lambda: _replacement(
        encoded_by=ParameterList((_class_param(), _reference_param()))))
    _refuses("if and only if", lambda: _replacement(insert_at_head=True))


def test_the_encoded_by_object_takes_at_most_one_object_set_and_one_reference():
    """§22.1.2.5's "(but only one)", twice, and nothing else at all. The object-set dummy's
    actual is "the current combined encoding object set" — the set `ecn_link`'s
    `EncodingApplication.combined` builds, which is why it is worth reporting."""
    both = _replacement(encoded_by=ParameterList(
        (_class_param(), _object_set_param(), _reference_param())), insert_at_head=True)
    assert both.takes_object_set() is True

    _refuses("only one", lambda: _replacement(encoded_by=ParameterList(
        (_class_param(), _object_set_param("s1"), _object_set_param("s2")))))
    _refuses("only one", lambda: _replacement(
        encoded_by=ParameterList((_class_param(), _reference_param("r1"),
                                  _reference_param("r2"))), insert_at_head=True))
    _refuses("is neither", lambda: _replacement(encoded_by=ParameterList(
        (_class_param(), Parameter("v", ParameterKind.VALUE,
                                   GovernorKind.ENCODING_CLASS_FIELD_TYPE, "#INT.&size")))))


def test_a_head_end_structure_has_no_dummy_parameters_at_all():
    """§22.1.2.7: "The `INSERT AT HEAD` encoding structures shall not have dummy parameters.
    All their fields are auxiliary fields." That is what distinguishes it from a replacement
    structure — it is the same shape minus the hole."""
    _refuses("22.1.2.7", lambda: _replacement(head_end=ParameterList((_class_param(),))))
    # And a head-end insertion with no ENCODED BY object has nothing to set its fields.
    _refuses("22.1.2.7", lambda: _replacement(
        encoded_by=None, governor_actuals=None, insert_at_head=True))
    # With no ENCODED BY and no insertion there is nothing left to check.
    _replacement(encoded_by=None, governor_actuals=None)


def test_inside_replace_the_name_is_bare_even_though_the_definition_is_parameterized():
    """§22.1.2.2 and §22.1.2.4 close with the same sentence — "only the ... name shall be
    given. They shall not have any parameter list in this use of the names."

    So one structure is written `#Length-prefixed{<#D>}` where it is defined and
    `#Length-prefixed` where it is used, and copying the definition's spelling is the mistake.
    C.3's `{<>}` is refused too: it is a legal `ParameterizedReference` elsewhere, and it is
    still a parameter list in this use of the name.
    """
    assert bare_use(ParameterizedReference("#Length-prefixed"),
                    clause="§22.1.2.2") == "#Length-prefixed"
    _refuses("22.1.2.2", lambda: bare_use(
        ParameterizedReference("#Length-prefixed", ActualParameterList(
            (ActualParameter(ActualKind.ENCODING_CLASS, "#D"),))), clause="§22.1.2.2"))
    _refuses("22.1.2.4", lambda: bare_use(
        ParameterizedReference("lp-object", ActualParameterList(())), clause="§22.1.2.4"))


# --- §17.5.16 to §17.5.18: resolving a ComponentIdList ------------------------------------

def _nested() -> dict:
    """A structure where one name appears at two depths, which is the only case that separates
    §17.5.17's scan from a recursive one."""
    return {
        "header": {"length": "leaf", "flags": {"length": "leaf-deep"}},
        "length": "leaf-outer",
    }


def test_the_first_identifier_is_found_breadth_first_and_not_by_a_recursive_walk():
    """§17.5.17: the match is "determined by the first match in a scan (in textual order) of
    the outer-level identifiers, then by a scan (in textual order) of the second level
    identifiers, then ... and so on".

    **That is breadth-first.** A recursive walk over `header` first would reach
    `header.length` before ever looking at the outer `length`, and §17.5.17 says the outer one
    wins. Both answers name a real field, so nothing fails at encode time — the encoding just
    points at a different component than the specification meant, which is why this is asserted
    against the depth-first answer explicitly rather than just against the right one.
    """
    assert resolve_component(_nested(), ("length",)) == ("length",)
    assert resolve_component(_nested(), ("length",)) != ("header", "length")
    # Second level beats third: `header.flags.length` is deeper than `header.length`.
    assert resolve_component({"header": {"flags": {"width": "d"}, "width": "s"}},
                             ("width",)) == ("header", "width")
    # With no shadowing the two orders agree, which is why the case above is the only witness.
    assert resolve_component({"header": {"crc": "leaf"}}, ("crc",)) == ("header", "crc")


def test_only_the_first_identifier_searches_and_the_rest_look_inside():
    """§17.5.16 gives the first identifier a search "at some level of nesting"; §17.5.18 makes
    each later one "an identifier in a `NamedType` of the structure identified by the previous
    part". So the path narrows monotonically — a later step is never searched for again."""
    assert resolve_component(_nested(), ("header", "length")) == ("header", "length")
    assert resolve_component(_nested(), ("header", "flags", "length")) == (
        "header", "flags", "length")
    # `flags` exists, but not inside what `length` named, and §17.5.18 does not go looking.
    _refuses("17.5.18", lambda: resolve_component(_nested(), ("length", "flags")))
    _refuses("17.5.16", lambda: resolve_component(_nested(), ("nope",)))
    _refuses("15.3.1", lambda: resolve_component(_nested(), ()))
