"""Cost-governed encoding selection — roadmap phase H's measurement half.

Two kinds of test live here, and the difference matters.

The **wire-size** facts are exact: the same value under the same rule is the same length on
every host, forever, so they are asserted as equalities and a change to any of them is a
change to an encoder. The **latency** facts are not: they are Python-oracle timings on
whatever runner happens to execute them, and the roadmap is explicit that phase H's real
selection uses the calibrated table in `kbcir/microbench.py`. So nothing here asserts a
timing, and the one test that touches ordering says out loud what it found and why it is
not a law.

That split is the two-truth law under test rather than merely described.
"""

from __future__ import annotations

from bcir.asn1.codec import Asn1Error
from bcir.asn1.constraints import ValueRange
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.selection import (
    ALL_CANDIDATES,
    SELECTABLE,
    Measurement,
    Objective,
    measure,
    measure_one,
    report,
    select,
)
from bcir.asn1.tags import Universal
from bcir.frontends.asn1.lower import compile_module

_A1_MODULE = """
AnnexA1 DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name, title [0] VisibleString, number EmployeeNumber,
      dateOfHire [1] Date, nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE OF ChildInformation DEFAULT {} }
  ChildInformation ::= SET { name Name, dateOfBirth [0] Date }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName VisibleString, initial VisibleString, familyName VisibleString }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER
  Date ::= [APPLICATION 3] IMPLICIT VisibleString
END
"""

_PERSONNEL_RECORD = {
    "name": {"givenName": "John", "initial": "P", "familyName": "Smith"},
    "title": "Director",
    "number": 51,
    "dateOfHire": "19710917",
    "nameOfSpouse": {"givenName": "Mary", "initial": "T", "familyName": "Smith"},
    "children": [
        {
            "name": {"givenName": "Ralph", "initial": "T", "familyName": "Smith"},
            "dateOfBirth": "19571111",
        },
        {
            "name": {"givenName": "Susan", "initial": "B", "familyName": "Jones"},
            "dateOfBirth": "19590717",
        },
    ],
}


def _annex_a1():
    return compile_module(_A1_MODULE, "<annex>").module.types["PersonnelRecord"]


def _sizes(kind, value) -> dict[str, int | None]:
    return {m.candidate: m.octets for m in measure(kind, value, repeats=1)}


# --- the record the roadmap's §6 gate asks for -------------------------------------------


def test_the_five_selectable_candidates_are_the_ones_section_6_names():
    """§6's reduction gate is decided over a *fixed* candidate set, so the set is pinned.

    The fifth is where the roadmap needs correcting: §6 calls it "CJER", but X.697 §42.2
    registers no canonical variant at all. The candidate is BCIR's own canonical JER
    profile, and it truthfully carries no object identifier.
    """
    assert [c.name for c in SELECTABLE] == [
        "DER",
        "CANONICAL-PER-UNALIGNED",
        "CANONICAL-PER-ALIGNED",
        "COER",
        "JER-BCIR-CANONICAL",
    ]
    jer = next(c for c in SELECTABLE if c.name == "JER-BCIR-CANONICAL")
    assert jer.oid is None, "X.697 42.2 registers no canonical variant to point at"
    for candidate in SELECTABLE:
        if candidate.name != "JER-BCIR-CANONICAL":
            assert candidate.oid is not None, candidate.name


def test_every_candidate_round_trips_the_annex_a_record():
    """Legality is a round trip, not "the encoder did not raise" — see `measure_one`."""
    kind = _annex_a1()
    for measurement in measure(kind, _PERSONNEL_RECORD, repeats=1):
        assert measurement.legal, f"{measurement.candidate}: {measurement.refusal}"
        assert measurement.octets is not None and measurement.octets > 0


def test_the_wire_sizes_of_the_annex_a_record():
    """Exact, and cross-checked against each standard's own Annex A where it states one.

    X.693 A.3 states 653 octets for BASIC-XER and, in the same sentence, 84 for UNALIGNED
    PER, 94 for ALIGNED PER and "a minimum of 136" for BER with the definite length form.
    Three of those four are candidates here and all three match, which is a stronger check
    than any of them alone: four independent encoders agreeing with one annex's arithmetic.
    """
    sizes = _sizes(_annex_a1(), _PERSONNEL_RECORD)
    assert sizes["DER"] == 136  # X.693 A.3, and X.690 Annex A
    assert sizes["CANONICAL-PER-UNALIGNED"] == 84  # X.691 A.1.4
    assert sizes["CANONICAL-PER-ALIGNED"] == 94  # X.691 A.1.3
    assert sizes["COER"] == 95
    assert sizes["JER-BCIR-CANONICAL"] == 385
    # The spread is what makes selection worth doing at all: 4.6x between the smallest and
    # the largest legal encoding of one value.
    assert max(sizes.values()) / min(sizes.values()) > 4


def test_the_report_records_the_measurement_the_gate_asks_to_be_recorded():
    """§6: "Record the decision with the measurement that justified it"."""
    text = report(_annex_a1(), _PERSONNEL_RECORD, label="Annex A.1", repeats=1)
    assert "Annex A.1" in text
    for candidate in ALL_CANDIDATES:
        assert candidate.name in text
    assert "% of DER" in text


# --- law 1: legality first ----------------------------------------------------------------


def test_an_unrepresentable_value_makes_a_candidate_not_a_candidate():
    """ "An encoding is a candidate only if the abstract value is representable in it,
    which is a verifier question, never a cost question."

    A bare ENUMERATED has no enumeration, so PER cannot find the §14.1 index and JER cannot
    find the §22.2 identifier — while DER encodes the number happily (X.690 §8.4). The
    illegal candidates must not appear in any selection, at any objective, however cheap
    their measurements look.
    """
    kind = Sequence((Component("e", Primitive(Universal.ENUMERATED, "ENUMERATED")),), "S")
    measurements = measure(kind, {"e": 1}, repeats=1)
    by_name = {m.candidate: m for m in measurements}
    assert by_name["DER"].legal
    assert not by_name["CANONICAL-PER-UNALIGNED"].legal
    assert not by_name["JER-BCIR-CANONICAL"].legal
    assert "enumeration" in by_name["JER-BCIR-CANONICAL"].refusal
    for objective in Objective:
        chosen = select(measurements, objective=objective, candidates=SELECTABLE)
        assert chosen is not None and chosen.legal
        assert chosen.candidate not in ("CANONICAL-PER-UNALIGNED", "JER-BCIR-CANONICAL")


def test_a_round_trip_that_returns_a_different_value_is_illegal_not_cheap():
    """The failure a size comparison would otherwise reward.

    An encoder that silently dropped a component would produce the shortest encoding in the
    set and win every wire-size selection. Making the round-trip comparison part of the
    *verdict* is what stops the optimizer preferring a lossy rail.
    """
    liar = Measurement(
        "LIAR", legal=False, octets=1, refusal="round trip returned a different value"
    )
    honest = Measurement("DER", legal=True, octets=1000)
    assert select((liar, honest), objective=Objective.WIRE_SIZE).candidate == "DER"


def test_selection_returns_nothing_rather_than_something_illegal():
    """With no legal canonical candidate there is no answer, and `select` says so."""
    assert (
        select((Measurement("DER", legal=False, refusal="nope"),), objective=Objective.WIRE_SIZE)
        is None
    )


# --- law 2: two-truth ---------------------------------------------------------------------


def test_cost_never_becomes_a_legality_verdict():
    """ "A measured encode/decode cost is graded truth and must not become a legality
    verdict."

    Expressed structurally: the verdict and the measurements are different fields, and a
    measurement carrying an absurd cost is still legal, while one carrying a zero cost is
    still illegal. Nothing in `select` can convert between them.
    """
    slow = Measurement("DER", legal=True, octets=10, encode_ns=10**12, decode_ns=10**12)
    fast_but_broken = Measurement(
        "COER", legal=False, octets=1, encode_ns=0, decode_ns=0, refusal="encode: no"
    )
    for objective in (Objective.WIRE_SIZE, Objective.ENCODE_LATENCY, Objective.DECODE_LATENCY):
        assert select((slow, fast_but_broken), objective=objective).candidate == "DER"


# --- law 3: canonical or excluded ---------------------------------------------------------


def test_a_rule_with_no_canonical_variant_is_decodable_but_never_selected():
    """ "A rule with no canonical variant may be decoded but never selected for emission,
    since a selected encoding is a digested artifact."

    BER and the BASIC-* variants are real decode targets and are measured — a peer may send
    them — but two conforming senders can produce two different encodings of one value under
    them, so a digest over the result would not be a digest over the value.
    """
    non_canonical = [c.name for c in ALL_CANDIDATES if not c.canonical]
    assert non_canonical == ["BER", "BASIC-PER-UNALIGNED", "BASIC-PER-ALIGNED", "BASIC-OER", "JER"]
    kind = _annex_a1()
    measurements = measure(kind, _PERSONNEL_RECORD, repeats=1)
    # They measure identically to their canonical siblings, so cost is not what excludes
    # them -- the property is.
    by_name = {m.candidate: m for m in measurements}
    assert by_name["BASIC-PER-UNALIGNED"].octets == by_name["CANONICAL-PER-UNALIGNED"].octets
    assert by_name["JER"].octets == by_name["JER-BCIR-CANONICAL"].octets
    for objective in Objective:
        chosen = select(measurements, objective=objective)
        assert chosen.candidate not in non_canonical, objective


# --- the gates the roadmap states for phase H ---------------------------------------------


def test_with_no_budget_selection_reproduces_todays_der_exactly():
    """The roadmap's degenerate case: "with no budget it reproduces today's DER exactly
    (the degenerate case, pinning that nothing regresses)"."""
    measurements = measure(_annex_a1(), _PERSONNEL_RECORD, repeats=1)
    chosen = select(measurements, objective=Objective.NONE, candidates=SELECTABLE)
    assert chosen.candidate == "DER"
    assert chosen.octets == 136


def test_on_a_bandwidth_capped_objective_selection_prefers_unaligned_per_over_der():
    """The roadmap's first gate: "on a bandwidth-capped Θ the optimizer selects UNALIGNED
    PER over DER".

    This one is decided by arithmetic rather than by measurement, which is why it is a
    genuine gate: 84 octets against 136 is a 38% reduction that holds on every host.
    """
    measurements = measure(_annex_a1(), _PERSONNEL_RECORD, repeats=1)
    chosen = select(measurements, objective=Objective.WIRE_SIZE, candidates=SELECTABLE)
    assert chosen.candidate == "CANONICAL-PER-UNALIGNED"
    assert chosen.octets == 84
    der = next(m for m in measurements if m.candidate == "DER")
    assert chosen.octets / der.octets < 0.65


def test_the_decode_latency_gate_does_not_hold_on_the_python_oracle_and_says_so():
    """The roadmap's second gate — "on a decode-latency-capped Θ it selects OER" — does
    NOT hold here, and this test records that rather than hiding it.

    The reason is an artifact of the *implementation*, not of the encoding rules: JER
    decodes through `json.loads`, which is C, while COER decodes through pure Python. So the
    measurement says JER and the rules say OER, and both are right about different things.

    This is exactly what the two-truth law is for, so nothing is asserted about the ordering
    — only that the harness reports a legal answer and that wire size, which IS exact, is
    unaffected. Phase H's real decision uses the calibrated cost table in
    `kbcir/microbench.py`; a Python-oracle timing is not that table and must not be mistaken
    for it.
    """
    measurements = measure(_annex_a1(), _PERSONNEL_RECORD, repeats=3)
    chosen = select(measurements, objective=Objective.DECODE_LATENCY, candidates=SELECTABLE)
    assert chosen is not None and chosen.legal
    # The exact half is untouched by whatever the clock said.
    assert next(m for m in measurements if m.candidate == "COER").octets == 95


def test_constraints_widen_the_gap_that_makes_selection_worth_governing():
    """X.691 Annex A.2 constrains the same value; PER shrinks and the others do not.

    §7.2.2 l) makes an integer's value constraint invisible to JER and X.690 encodes a value
    the same way regardless, so a constraint that costs PER nothing to exploit is free size
    for exactly one candidate. That asymmetry is the whole reason a fixed candidate set
    still needs an optimizer.
    """
    constrained = """
AnnexA2 DEFINITIONS ::= BEGIN
  PersonnelRecord ::= [APPLICATION 0] IMPLICIT SET {
      name Name, title [0] VisibleString, number EmployeeNumber,
      dateOfHire [1] Date, nameOfSpouse [2] Name,
      children [3] IMPLICIT SEQUENCE OF ChildInformation DEFAULT {} }
  ChildInformation ::= SET { name Name, dateOfBirth [0] Date }
  Name ::= [APPLICATION 1] IMPLICIT SEQUENCE {
      givenName NameString, initial NameString (SIZE(1)), familyName NameString }
  EmployeeNumber ::= [APPLICATION 2] IMPLICIT INTEGER
  Date ::= [APPLICATION 3] IMPLICIT VisibleString (FROM("0".."9") ^ SIZE(8))
  NameString ::= VisibleString (FROM("a".."z" | "A".."Z" | "-.") ^ SIZE(1..64))
END
"""
    kind = compile_module(constrained, "<annex>").module.types["PersonnelRecord"]
    tight = _sizes(kind, _PERSONNEL_RECORD)
    loose = _sizes(_annex_a1(), _PERSONNEL_RECORD)
    assert tight["CANONICAL-PER-UNALIGNED"] == 61  # X.691 A.2.4
    assert tight["CANONICAL-PER-ALIGNED"] == 74  # X.691 A.2.3
    assert tight["CANONICAL-PER-UNALIGNED"] < loose["CANONICAL-PER-UNALIGNED"]
    assert tight["DER"] == loose["DER"], "X.690 encodes a value the same way regardless"
    assert tight["JER-BCIR-CANONICAL"] == loose["JER-BCIR-CANONICAL"], "7.2.2 l)"


def test_an_integer_constraint_moves_only_the_candidates_that_can_read_it():
    """The same asymmetry at its smallest, where it is easiest to see."""
    plain = Sequence((Component("v", Primitive(Universal.INTEGER, "INTEGER")),), "S")
    bounded = Sequence(
        (Component("v", Primitive(Universal.INTEGER, "INTEGER", ValueRange(0, 255))),), "S"
    )
    before = _sizes(plain, {"v": 200})
    after = _sizes(bounded, {"v": 200})
    assert after["CANONICAL-PER-UNALIGNED"] < before["CANONICAL-PER-UNALIGNED"]
    assert after["COER"] < before["COER"]  # X.696 §10 sizes from it too
    assert after["DER"] == before["DER"]
    assert after["JER-BCIR-CANONICAL"] == before["JER-BCIR-CANONICAL"]


def test_measure_one_reports_a_refusal_rather_than_raising():
    """A candidate that cannot carry the value is data, not an exception — otherwise one
    unrepresentable value would abort the whole comparison."""
    kind = Primitive(Universal.ENUMERATED, "ENUMERATED")
    candidate = next(c for c in SELECTABLE if c.name == "JER-BCIR-CANONICAL")
    measurement = measure_one(candidate, kind, 1, repeats=1)
    assert not measurement.legal
    assert measurement.refusal and measurement.refusal.startswith("encode:")


def test_selection_is_deterministic_across_repeated_measurement():
    """Ties break on the declared candidate order, never on whichever run was fastest."""
    kind = _annex_a1()
    picks = {
        select(
            measure(kind, _PERSONNEL_RECORD, repeats=1),
            objective=Objective.WIRE_SIZE,
            candidates=SELECTABLE,
        ).candidate
        for _ in range(5)
    }
    assert picks == {"CANONICAL-PER-UNALIGNED"}


def test_the_round_trip_comparison_tolerates_what_a_rule_may_legitimately_change():
    """A canonical rule may reorder a SET OF (X.680 §28.3 NOTE 2) and a decode may add the
    X.682 §10.19 `.resolved` enrichment. Neither is a lost value, so neither is illegal."""
    from bcir.asn1.selection import _equivalent

    assert _equivalent([{"a": 2}, {"a": 1}], [{"a": 1}, {"a": 2}])
    assert _equivalent({"x": 1, "x.resolved": {"n": 7}}, {"x": 1})
    assert not _equivalent({"x": 1}, {"x": 2})
    assert not _equivalent([1, 2], [1, 2, 3])
