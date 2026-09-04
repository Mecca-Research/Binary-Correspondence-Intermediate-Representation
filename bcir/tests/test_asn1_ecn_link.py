"""X.692 clauses 12 and 13: the encoding link module, and the two deviations it retires.

**The ELM is clause 12.** Clause 12's own NOTE says so — "There are two top-level productions
in ECN, the `ELMDefinition` specified in this clause and the `EDMDefinition` specified in clause
14" — and this repository cited clause 14 for it twice before reading that sentence.

The reason this module matters more than its size suggests is that it retires two **stated
deviations**. `ecn_syntax.py` accepts an `AUXILIARY` keyword and a `BOUNDS` clause that X.692
gives no notation for, and both exist because the facts they carry live in the *link* between
an ASN.1 type and an encoding structure — which this rail did not have. With a link:

* **auxiliary** is §22.1.2.6's category *computed* from the two field lists, not declared;
* **bounds** are the ASN.1 component's own constraint, which is where §21.11.3 and §23.7.2.6's
  NOTE both say they live.

Both keywords stay accepted, because a specification written against this rail may still have
no ASN.1 type in hand — but a link supersedes them, and that is the difference between a
deviation and a fallback.
"""

from bcir.asn1.constraints import ValueRange
from bcir.asn1.ecn_link import ElmModule, EncodingApplication, LinkedStructure, resolve
from bcir.asn1.ecn_props import IntegerBounds, RangeCondition
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.tags import Asn1Error, Universal


def _refuses(citation: str, build):
    try:
        build()
    except Asn1Error as error:
        assert citation in str(error), (citation, str(error))
        return
    raise AssertionError(f"expected a refusal citing {citation}")


def _frame() -> Sequence:
    """A header type with one bounded component and one unbounded one."""
    return Sequence(
        (
            Component(
                "version", Primitive(Universal.INTEGER, "INTEGER", constraint=ValueRange(0, 7))
            ),
            Component("payloadOctets", Primitive(Universal.INTEGER, "INTEGER")),
        )
    )


# --- §13.2.2 / §13.2.3: the combined encoding object set --------------------------------------


def test_completed_by_fills_gaps_and_never_overrides():
    """§13.2.3 b): a secondary object is added "if (and only if) there is no encoding object
    already in the combined encoding object set that has the same encoding class".

    The same left-biased rule §9.23.2 states and §22.11.1.4 defers to — which is what makes
    `COMPLETED BY PER-BASIC-UNALIGNED` safe to write under a handful of specialized objects.
    """
    application = EncodingApplication(
        classes=("#Frame",), primary={"#A": "mine"}, secondary={"#A": "theirs", "#B": "gap"}
    )
    assert application.combined() == {"#A": "mine", "#B": "gap"}
    # §13.2.2: with no CompletionClause the primary set IS the combined set.
    assert EncodingApplication(classes=("#Frame",), primary={"#A": "mine"}).combined() == {
        "#A": "mine"
    }


# --- clause 12's ELM ---------------------------------------------------------------------------


def test_an_elm_applies_encodings_and_must_apply_at_least_one():
    """§12.1.9: the list "is required to contain at least one `EncodingApplication`, as the
    sole function of an ELM is to apply encodings"."""
    application = EncodingApplication(classes=("#Frame",), primary={"#A": 1})
    elm = ElmModule(name="Link", applications=(application,))
    assert elm.set_for("#Frame") == {"#A": 1}
    _refuses("12.1.9", lambda: ElmModule(name="Link"))
    _refuses("12.1.3", lambda: ElmModule(applications=(application,)))
    _refuses("12.2.2", lambda: elm.set_for("#Other"))


def test_no_type_is_encoded_twice_within_an_elm():
    """§12.2.5: "An ELM shall not apply encodings more than once to the same ASN.1 type."

    Checked across applications as well as within one, because §12.2.5 is about the ELM and
    §13.2.5 leans on it — "the rules of 12.2 ensure that applications are non-overlapping.
    They proceed independently." Two applications naming one class would make that false.
    """
    _refuses("12.2.5", lambda: EncodingApplication(classes=("#Frame", "#Frame"), primary={"#A": 1}))
    _refuses(
        "12.2.5",
        lambda: ElmModule(
            name="Link",
            applications=(
                EncodingApplication(classes=("#Frame",), primary={"#A": 1}),
                EncodingApplication(classes=("#Frame",), primary={"#B": 2}),
            ),
        ),
    )
    _refuses("12.2.1", lambda: EncodingApplication(primary={"#A": 1}))


def test_every_reference_must_be_imported_which_asn1_does_not_require():
    """§12.1.7: "All reference names used in the `ELMModuleBody` shall be imported into the
    ELM." Its NOTE calls this "a stronger requirement than that imposed for ASN.1 modules",
    where "external references can be used for types and values that have not been imported".

    An empty `imports` means the clause was not written, which is a different statement from
    importing nothing — only the second is checkable, so only it is checked.
    """
    application = EncodingApplication(classes=("#Frame",), primary={"#A": 1})
    ElmModule(name="Link", applications=(application,))  # not stated
    ElmModule(name="Link", applications=(application,), imports=("#Frame",))
    _refuses(
        "12.1.7", lambda: ElmModule(name="Link", applications=(application,), imports=("#Other",))
    )


# --- §13.2.10's application-point algorithm ----------------------------------------------------


def test_dereferencing_is_what_makes_a_class_assignment_mean_something():
    """§13.2.10.1 a) and §13.2.10.7: with no object of the class, the class "is de-referenced,
    and the procedures of 13.2.10 are recursively applied".

    This is what makes clause 11's `#Version ::= #INT` do any work — one object written for
    `#INT` encodes every class assigned from it, which is how a single object set covers a
    module full of assignments.
    """
    objects = {"#INT": "int-object"}
    assignments = {"#Version": "#Base", "#Base": "#INT"}
    assert resolve(objects, "#Version", assignments) == "int-object"
    assert resolve(objects, "#INT", assignments) == "int-object"
    # §9.5.2's one-object-per-class rule is what makes the first step unambiguous: a direct
    # object wins over the chain.
    assert (
        resolve({"#INT": "int-object", "#Version": "special"}, "#Version", assignments) == "special"
    )


def test_a_class_no_object_reaches_is_the_specification_error_the_clause_names():
    """§13.2.10.8: "Otherwise the ECN specification is in error." Named as that, with the chain
    that was tried — "no encoding object" and "no such class" are different faults, and a bare
    failure would not say which."""
    _refuses("13.2.10.8", lambda: resolve({"#INT": "o"}, "#Missing", {}))
    _refuses("13.2.10.8", lambda: resolve({"#INT": "o"}, "#A", {"#A": "#B"}))


def test_a_circular_assignment_chain_is_refused_rather_than_followed():
    """The same fault R25 rejects at parse time on the law rail: a class that is its own base
    names no encoding category, so no object could realize it."""
    _refuses("circular", lambda: resolve({}, "#A", {"#A": "#B", "#B": "#A"}))


# --- the two deviations, retired ---------------------------------------------------------------


def test_auxiliary_is_computed_from_the_link_and_not_declared():
    """§22.1.2.6's auxiliary fields are those "not part of the encoding class parameter", and
    §19.3.1 gives the same set from clause 19's side: a structure "has fields corresponding to
    the components of the type, **but also has added fields for determinants**".

    So which fields are auxiliary is a *computation* over the two field lists. `ecn_syntax.py`'s
    `AUXILIARY` keyword — a stated deviation, since X.692 has no such notation — is a fallback
    for when there is no ASN.1 type in hand, not the source of truth.
    """
    linked = LinkedStructure(asn1_type=_frame(), fields=("len", "version", "payloadOctets"))
    assert linked.auxiliary_fields() == ("len",)
    assert linked.missing_fields() == ()
    linked.check()

    # Order in the structure is the structure's business (§16.5), not the type's, so an
    # auxiliary field anywhere is found.
    trailing = LinkedStructure(asn1_type=_frame(), fields=("version", "payloadOctets", "crc"))
    assert trailing.auxiliary_fields() == ("crc",)


def test_a_component_with_no_field_has_nowhere_to_be_encoded():
    """§9.24.2 makes the structure's encodings the encodings of the type's abstract values, so
    the mapping has to be total in that direction. The other direction is expected to be
    partial — that is exactly what auxiliary fields are — so only this one is a fault."""
    linked = LinkedStructure(asn1_type=_frame(), fields=("version",))
    assert linked.missing_fields() == ("payloadOctets",)
    _refuses("9.24.2", linked.check)


def test_bounds_come_from_the_asn1_component_not_from_the_encoding_object():
    """§21.11.3 tests "the bounds on the integer values associated with an encoding class in
    the integer category", and §23.7.2.6's NOTE insists the condition is tested "on the bounds
    of the original value". Both put the bounds on the type.

    That retires `ecn_syntax.py`'s `BOUNDS` clause as the *source* of the bounds — it stays as
    a fallback for a specification with no ASN.1 type behind it.
    """
    linked = LinkedStructure(asn1_type=_frame(), fields=("len", "version", "payloadOctets"))
    assert linked.bounds_for("version") == IntegerBounds(0, 7)
    # An unconstrained INTEGER has no bounds, and "no bounds" is an ANSWER here rather than a
    # gap: §21.11.4 a)'s `unbounded-or-no-lower-bound` is the predicate for exactly this.
    unbounded = linked.bounds_for("payloadOctets")
    assert unbounded == IntegerBounds()
    assert unbounded.exactly_one_shape() is RangeCondition.UNBOUNDED_OR_NO_LOWER_BOUND
    # And the bounded one selects a different §21.11.4 shape, which is the whole point of
    # deriving them: the encoding chosen for `version` differs from `payloadOctets`'.
    assert linked.bounds_for("version").exactly_one_shape() is (
        RangeCondition.BOUNDED_WITHOUT_NEGATIVES
    )
    _refuses("is not a component", lambda: linked.bounds_for("nope"))


def test_a_linked_structure_needs_a_type_with_components():
    _refuses(
        "has none",
        lambda: LinkedStructure(
            asn1_type=Primitive(Universal.INTEGER, "INTEGER"), fields=("a",)
        ).auxiliary_fields(),
    )
