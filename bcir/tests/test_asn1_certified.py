"""J6 — the certified K_BCIR encoding choice.

The gate: *"Exact candidate sizes, controlled counters, repeatability, legality-first
refusal, and deterministic selection on at least two targets."*

The load-bearing test in this file is not any of the statistics — it is
`test_a_timing_objective_refuses_an_oracle_table`. §6.2 says production selection must
*"refuse an unmeasured required target instead of substituting Python timings"*, and §2
records why: on this rail `json.loads` is native C while COER decode is pure Python, so an
oracle timing orders candidates by **which implementation happens to be compiled**. Using
it would give a confident, reproducible, wrong answer — worse than no answer, because it
looks like evidence.

Everything else exists to make that refusal meaningful rather than absolute: a wire-size
decision needs no timing and must still work, or callers would learn to switch the guard
off by reflex and carry that habit into the decisions where it matters.
"""

from __future__ import annotations

from bcir.asn1.certified import (
    COST_TABLE_VERSION, MIN_SAMPLES, TIE_BREAK, Certificate, CostRow, EncodingCostTable,
    Infeasible, Interval, Stage, UnmeasuredTarget, build_table, interval_of,
    measure_repeatedly, select_budgeted, select_certified,
)
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.selection import ALL_CANDIDATES, Objective
from bcir.asn1.tags import Asn1Error, Universal

_INT = Primitive(Universal.INTEGER)
_RECORD = Sequence((
    Component("id", _INT),
    Component("name", Primitive(Universal.UTF8_STRING)),
), name="Record")
_VALUE = {"id": 42, "name": "a-name"}


def _only(*names: str):
    """The declared candidates restricted to `names`, in declaration order.

    A hand-built table lists a few rows, and `select_certified` refuses when a legal,
    canonical candidate has no row — correctly, because that IS the unmeasured case. So a
    test using a partial table must restrict the candidate set to match it, or it is
    testing the refusal rather than the thing it means to test.
    """
    chosen = [c for c in ALL_CANDIDATES if c.name in set(names)]
    assert len(chosen) == len(set(names)), f"unknown candidate in {names}"
    return chosen


def _oracle_table(target: str = "host", cal_gen: int = 1) -> EncodingCostTable:
    return build_table(_INT, 42, target=target, cal_gen=cal_gen)


def _as_measured(table: EncodingCostTable) -> EncodingCostTable:
    """Relabel an oracle table as measured — for testing the *selection* path only.

    A real `measured` table comes from a native harness on the part. This exists so the
    tests below can exercise the code that runs after the provenance gate without
    pretending the numbers are something they are not; every test that uses it says so.
    """
    return EncodingCostTable(target=table.target, cal_gen=table.cal_gen,
                             provenance="measured", rows=table.rows)


# --- §6.2's refusal ---------------------------------------------------------------------


def test_a_timing_objective_refuses_an_oracle_table():
    """The rule this whole module exists for.

    A decode-latency decision read from a Python-harness table would rank COER below JER
    because `json.loads` is C and the COER decoder is Python — a fact about this
    repository's implementation languages, not about the encodings.
    """
    table = _oracle_table()
    for objective in (Objective.ENCODE_LATENCY, Objective.DECODE_LATENCY):
        try:
            select_certified(_INT, 42, table, objective=objective)
        except UnmeasuredTarget as error:
            assert "oracle" in str(error) and "MEASURED" in str(error)
        else:
            raise AssertionError(f"{objective} was decided from an oracle table")


def test_a_wire_size_objective_is_decidable_without_any_timing():
    """And therefore without a measured table — which is why the guard is not absolute.

    Exact encoded size is arithmetic: the same value under the same rule is the same length
    on every host, forever. Refusing this decision because the table happens to be an oracle
    one would teach callers to pass `allow_oracle_table=True` everywhere, carrying the flag
    into the timing decisions where it is load-bearing.
    """
    certificate = select_certified(_INT, 42, _oracle_table(), objective=Objective.WIRE_SIZE)
    assert certificate.selected is not None
    # No interval was consulted, and the certificate says so rather than implying one.
    assert certificate.interval_rank_coverage_ppm == 0


def test_an_unmeasured_candidate_is_refused_even_in_a_measured_table():
    """A measured table is not a licence to guess about a row it does not have."""
    table = _as_measured(_oracle_table())
    thinned = EncodingCostTable(target=table.target, cal_gen=table.cal_gen,
                                provenance="measured",
                                rows=tuple(r for r in table.rows if r.candidate != "COER"))
    try:
        select_certified(_INT, 42, thinned, objective=Objective.DECODE_LATENCY)
    except UnmeasuredTarget as error:
        assert "COER" in str(error)
    else:
        raise AssertionError("a missing row was silently skipped")


def test_the_refusal_has_its_own_exception_class():
    """`UnmeasuredTarget` is not a generic error, because the right response is specific.

    A caller catching `Asn1Error` broadly could otherwise swallow the one condition that
    must never pass quietly. It stays an `Asn1Error` subclass so existing handlers still
    work, but it is nameable on its own.
    """
    assert issubclass(UnmeasuredTarget, Asn1Error)


# --- legality first ----------------------------------------------------------------------


def test_legality_is_settled_before_any_number_is_looked_at():
    """Law 1. A candidate that cannot carry the value is not an expensive candidate."""
    certificate = select_certified(_RECORD, _VALUE, _as_measured(
        build_table(_RECORD, _VALUE, target="host", cal_gen=1)),
        objective=Objective.WIRE_SIZE)
    admitted = set(certificate.admitted)
    refused = dict(certificate.refused)
    assert admitted and not (admitted & set(refused))
    # Every non-canonical candidate is refused with a reason, never merely ranked last.
    for candidate in ALL_CANDIDATES:
        if not candidate.canonical:
            assert candidate.name in refused, candidate.name
            assert "canonical" in refused[candidate.name]


def test_a_value_no_candidate_can_carry_produces_a_certificate_with_no_winner():
    """The honest outcome, rather than the least-bad encoding.

    A certificate that named a winner when nothing was legal would be asserting a
    representability claim the round trip refused.
    """
    table = EncodingCostTable(target="host", cal_gen=1, provenance="measured", rows=())
    certificate = select_certified(_INT, object(), table, objective=Objective.WIRE_SIZE)
    assert certificate.selected is None
    assert certificate.admitted == ()
    assert certificate.refused, "nothing was recorded as refused either"


# --- the intervals -----------------------------------------------------------------------


def test_the_interval_is_distribution_free_and_its_coverage_is_exact():
    """Order statistics, computed in integers.

    A coverage figure that varies with the host's floating-point rounding is not a coverage
    figure, and this one is written into a certificate.
    """
    samples = [10, 12, 11, 40, 13, 12, 11, 12, 300, 11, 12]
    interval = interval_of(samples)
    assert interval.low <= interval.median <= interval.high
    assert interval.low in samples and interval.high in samples, (
        "the bounds must be observed samples, not fitted parameters")
    assert 900_000 <= interval.coverage_ppm <= 1_000_000
    # The heavy tail is inside the interval's reach but does not drag the median, which is
    # the property a mean would not have.
    assert interval.median < 40


def test_too_few_samples_is_a_refusal_not_a_wider_interval():
    """A narrower table is not a more precise one."""
    try:
        interval_of([1, 2, 3])
    except Asn1Error as error:
        assert str(MIN_SAMPLES) in str(error)
    else:
        raise AssertionError("an interval was built from three samples")


def test_overlapping_intervals_are_reported_as_indistinguishable():
    """The behaviour that differs from a median comparison.

    Two candidates whose intervals overlap are the same as far as the measurement can say.
    A median comparison would still name a winner — and would name a different one on the
    next host.
    """
    a = Interval(low=10, high=20, median=15, samples=9, coverage_ppm=980_000)
    b = Interval(low=18, high=30, median=24, samples=9, coverage_ppm=980_000)
    c = Interval(low=40, high=50, median=45, samples=9, coverage_ppm=980_000)
    assert a.overlaps(b) and b.overlaps(a)
    assert not a.overlaps(c)


def test_indistinguishable_candidates_are_resolved_by_the_exact_measurement():
    """Two-truth, applied to the tie-break.

    When the timing cannot separate two candidates, the decision falls to a number with no
    distribution — exact encoded size — and then to the declared candidate order. That is
    what makes the selection reproducible on a different host rather than merely repeatable
    on this one.
    """
    wide = Interval(low=100, high=900, median=500, samples=9, coverage_ppm=980_000)
    rows = (
        CostRow(candidate="CANONICAL-PER-UNALIGNED", octets=9, encode=wide, decode=wide),
        CostRow(candidate="COER", octets=4, encode=wide, decode=wide),
        CostRow(candidate="DER", octets=7, encode=wide, decode=wide),
    )
    table = EncodingCostTable(target="host", cal_gen=1, provenance="measured", rows=rows)
    certificate = select_certified(
        _INT, 42, table, objective=Objective.DECODE_LATENCY,
        candidates=_only("CANONICAL-PER-UNALIGNED", "COER", "DER"))
    # Every interval overlaps every other, so the smallest exact encoding wins.
    assert certificate.selected == "COER"
    assert set(certificate.indistinguishable) == {"CANONICAL-PER-UNALIGNED", "DER"}
    assert "exact-octets" in certificate.tie_break


def test_the_certificate_records_which_candidates_it_could_not_separate():
    """"We chose A over B" and "A and B were the same" are different decisions.

    A certificate that hid the difference would overstate what the measurement showed.
    """
    tight = Interval(low=10, high=11, median=10, samples=9, coverage_ppm=980_000)
    loose = Interval(low=900, high=999, median=950, samples=9, coverage_ppm=980_000)
    rows = (CostRow(candidate="COER", octets=4, encode=tight, decode=tight),
            CostRow(candidate="DER", octets=7, encode=loose, decode=loose))
    table = EncodingCostTable(target="host", cal_gen=1, provenance="measured", rows=rows)
    certificate = select_certified(_INT, 42, table, objective=Objective.DECODE_LATENCY,
                                   candidates=_only("COER", "DER"))
    assert certificate.selected == "COER"
    assert certificate.indistinguishable == (), (
        "a clearly separated candidate was reported as indistinguishable")


# --- repeatability and the table -----------------------------------------------------------


def test_a_candidate_may_not_change_legality_or_size_between_runs():
    """Repeatability, as a refusal rather than a majority vote.

    Legality is a property of the value and the rule; a candidate that flipped between runs
    would be a codec defect, and averaging it away would hide exactly the bug worth finding.
    """
    verdict, encode_ns, decode_ns = measure_repeatedly(ALL_CANDIDATES[0], _INT, 42)
    assert verdict.legal
    assert len(encode_ns) == len(decode_ns) == MIN_SAMPLES


def test_a_frozen_table_is_content_addressed_and_names_no_candidate_twice():
    """The table is data with an identity, so a certificate can bind to it."""
    table = _oracle_table()
    assert table.version == COST_TABLE_VERSION
    assert table.digest() == EncodingCostTable(
        target=table.target, cal_gen=table.cal_gen, provenance=table.provenance,
        rows=table.rows).digest()
    duplicated = table.rows + (table.rows[0],)
    try:
        EncodingCostTable(target="host", cal_gen=1, provenance="measured",
                          rows=duplicated)
    except Asn1Error as error:
        assert "twice" in str(error)
    else:
        raise AssertionError("a table named a candidate twice")


def test_a_table_must_declare_an_honest_provenance():
    """`measured`, `modeled` and `oracle` are different claims and only one is evidence."""
    try:
        EncodingCostTable(target="host", cal_gen=1, provenance="pretty-sure", rows=())
    except Asn1Error as error:
        assert "measured" in str(error)
    else:
        raise AssertionError("an unrecognized provenance was accepted")


def test_build_table_defaults_to_oracle_provenance():
    """It runs under a Python codec on whatever host imported it, so `measured` would lie.

    Making the honest label the default means producing a `measured` table is a deliberate
    argument rather than something that happens by omission.
    """
    assert _oracle_table().provenance == "oracle"


# --- deterministic selection on more than one target ------------------------------------------


def test_the_same_table_selects_the_same_candidate_every_time():
    """Determinism on one target: no randomness, no host clock, in the decision."""
    table = _as_measured(_oracle_table())
    first = select_certified(_INT, 42, table, objective=Objective.DECODE_LATENCY)
    for _ in range(4):
        again = select_certified(_INT, 42, table, objective=Objective.DECODE_LATENCY)
        assert again.selected == first.selected
        assert again.digest() == first.digest()


def test_two_targets_with_different_tables_may_select_differently_and_say_why():
    """The gate's "deterministic selection on at least two targets".

    Two targets are two tables. The point is not that they agree — a different part may
    genuinely prefer a different encoding — but that each decision is reproducible from its
    own table and that the certificate names which table it read.
    """
    fast_binary = Interval(low=10, high=12, median=11, samples=9, coverage_ppm=980_000)
    slow_binary = Interval(low=900, high=950, median=920, samples=9, coverage_ppm=980_000)
    fast_text = Interval(low=20, high=22, median=21, samples=9, coverage_ppm=980_000)

    target_a = EncodingCostTable(
        target="alpha", cal_gen=3, provenance="measured",
        rows=(CostRow("COER", 4, fast_binary, fast_binary),
              CostRow("JER-BCIR-CANONICAL", 20, fast_text, fast_text)))
    target_b = EncodingCostTable(
        target="beta", cal_gen=5, provenance="measured",
        rows=(CostRow("COER", 4, slow_binary, slow_binary),
              CostRow("JER-BCIR-CANONICAL", 20, fast_text, fast_text)))

    pair = _only("COER", "JER-BCIR-CANONICAL")
    a = select_certified(_INT, 42, target_a, objective=Objective.DECODE_LATENCY,
                         candidates=pair)
    b = select_certified(_INT, 42, target_b, objective=Objective.DECODE_LATENCY,
                         candidates=pair)
    assert a.selected == "COER" and b.selected == "JER-BCIR-CANONICAL"
    # Each certificate binds to the table it read, so the two are distinguishable after
    # the fact rather than being two undated claims about "the" cost of an encoding.
    assert a.table_digest != b.table_digest
    assert (a.target, a.cal_gen) == ("alpha", 3)
    assert (b.target, b.cal_gen) == ("beta", 5)
    # And re-running either reproduces it exactly.
    assert select_certified(_INT, 42, target_b, objective=Objective.DECODE_LATENCY,
                            candidates=pair).digest() == b.digest()


# --- the certificate ---------------------------------------------------------------------------


def test_the_certificate_carries_the_6_2_fields_and_is_content_addressed():
    """§6.2's list, as fields a reader can check rather than claims to trust."""
    table = _as_measured(_oracle_table(target="alpha", cal_gen=9))
    certificate = select_certified(_INT, 42, table, objective=Objective.DECODE_LATENCY)
    assert isinstance(certificate, Certificate)
    assert certificate.schema_digest and certificate.value_digest
    assert certificate.table_digest == table.digest()
    assert certificate.target == "alpha" and certificate.cal_gen == 9
    assert certificate.provenance == "measured"
    assert certificate.tie_break == TIE_BREAK
    assert certificate.objective == Objective.DECODE_LATENCY.value
    # Two certificates that differ anywhere have different digests.
    other = select_certified(_INT, 43, table, objective=Objective.DECODE_LATENCY)
    assert other.digest() != certificate.digest()


def test_the_certificate_separates_the_verdict_from_the_costs():
    """Two-truth in the document: `admitted`/`refused` is the verdict and reads alone."""
    table = _as_measured(_oracle_table())
    certificate = select_certified(_INT, 42, table, objective=Objective.WIRE_SIZE)
    verdict_fields = {"admitted", "refused", "selected"}
    cost_fields = {"interval_rank_coverage_ppm", "indistinguishable"}
    for name in verdict_fields | cost_fields:
        assert hasattr(certificate, name), name
    # A refusal carries its reason, so the verdict is actionable without the costs.
    assert all(reason for _name, reason in certificate.refused)


# --- RCSP: the budgeted plan across several stages ------------------------------------------
#
# `select_certified` decides one encoding; a pipeline decides several under one global
# budget, and picking each stage's local best solves a different problem. These check the
# dynamic program against optima derived by hand, and check the part that is easy to get
# wrong for free: what summing intervals does to their coverage.


def _iv(low: int, median: int, high: int, coverage_ppm: int = 950_000) -> Interval:
    return Interval(low=low, high=high, median=median, samples=11,
                    coverage_ppm=coverage_ppm)


def _budget_table(coverage_ppm: int = 950_000) -> EncodingCostTable:
    """Three candidates spanning the trade-off: fast-and-fat, slow-and-thin, middling."""
    fixed = _iv(1, 1, 1, coverage_ppm)
    return EncodingCostTable(target="host", cal_gen=1, provenance="measured", rows=(
        CostRow("A", octets=10, encode=fixed, decode=_iv(90, 100, 110, coverage_ppm)),
        CostRow("B", octets=4, encode=fixed, decode=_iv(280, 300, 320, coverage_ppm)),
        CostRow("C", octets=6, encode=fixed, decode=_iv(190, 200, 210, coverage_ppm)),
    ))


_TWO = (Stage("s1", ("A", "B", "C")), Stage("s2", ("A", "B", "C")))


def test_the_budgeted_plan_reproduces_a_hand_derived_optimum():
    """Every budget worked out on paper first, including the one where the answer changes.

    The medians are 100/300/200 and the octets 10/4/6, so the optimum is not monotone in
    either axis alone — which is the whole reason this is a constrained problem and not a
    sort.
    """
    table = _budget_table()
    expected = {
        20: (["A", "A"], 200),   # both fast, exactly on budget
        16: (["A", "C"], 300),   # A+A no longer fits
        14: (["C", "C"], 400),   # ties A+B at 400 and spends four fewer octets
        10: (["B", "C"], 500),   # A cannot appear at all
        8: (["B", "B"], 600),    # the cheapest plan there is
    }
    for budget, (chosen, median) in expected.items():
        plan = select_budgeted(table, _TWO, budget=budget,
                               objective=Objective.DECODE_LATENCY)
        assert [name for _, name in plan.chosen] == chosen, budget
        assert plan.latency_median == median, budget
        assert plan.total_octets <= budget


def test_a_local_best_per_stage_is_not_the_budgeted_optimum():
    """The property that makes RCSP worth having rather than two independent selections."""
    table = _budget_table()
    plan = select_budgeted(table, _TWO, budget=16, objective=Objective.DECODE_LATENCY)
    # Greedy would take the fastest candidate at stage 1 and then find nothing affordable
    # at stage 2 that beats what the joint optimum reaches.
    assert [name for _, name in plan.chosen] == ["A", "C"]
    assert plan.latency_median == 300 < 100 + 300  # A then the only affordable rival, B


def test_an_unaffordable_budget_is_infeasible_and_says_what_the_floor_is():
    table = _budget_table()
    try:
        select_budgeted(table, _TWO, budget=7, objective=Objective.DECODE_LATENCY)
    except Infeasible as error:
        assert "cheapest legal plan costs 8" in str(error)
    else:
        raise AssertionError("7 octets cannot hold two stages whose floor is 4 each")


def test_summing_intervals_decays_their_coverage_and_the_plan_says_so():
    """The finding: a chain of twenty 95% statements certifies nothing.

    Each interval holds with some probability and the statement about the SUM holds only
    when all of them do, so the union bound gives 1 - n(1 - c). At ten stages that is
    exactly 50%, and at twenty it is zero — the DP still returns its optimum, and the plan
    reports that the optimum is no longer distinguishable from its rivals by evidence.

    Carrying the component coverage through unchanged is the obvious shortcut and it is
    wrong in the optimistic direction, which is the direction that gets believed.
    """
    table = _budget_table()
    seen = {}
    for count in (1, 2, 5, 10, 20):
        stages = tuple(Stage(f"s{i}", ("B",)) for i in range(count))
        plan = select_budgeted(table, stages, budget=4 * count,
                               objective=Objective.DECODE_LATENCY)
        seen[count] = (plan.coverage_ppm, plan.certified)
    assert seen[1] == (950_000, True)
    assert seen[2] == (900_000, True)
    assert seen[5] == (750_000, True)
    assert seen[10] == (500_000, True)     # exactly at the default floor
    assert seen[20] == (0, False)          # the answer survives; the evidence does not
    # Monotone, and never negative — a coverage that wrapped would read as a strong claim.
    values = [seen[n][0] for n in sorted(seen)]
    assert values == sorted(values, reverse=True) and min(values) >= 0


def test_the_coverage_floor_is_the_callers_and_is_reported_not_enforced():
    """A plan below the floor is returned and marked, not withheld.

    The optimum is still the optimum; what changed is whether the intervals separate it. A
    caller deciding a soft budget may act on it anyway, and a caller signing a certificate
    must not — that is a policy difference the selector should not make on their behalf.
    """
    stages = tuple(Stage(f"s{i}", ("B",)) for i in range(10))
    table = _budget_table()
    strict = select_budgeted(table, stages, budget=40, min_coverage_ppm=900_000,
                             objective=Objective.DECODE_LATENCY)
    lax = select_budgeted(table, stages, budget=40, min_coverage_ppm=100_000,
                          objective=Objective.DECODE_LATENCY)
    assert strict.chosen == lax.chosen and strict.latency_median == lax.latency_median
    assert strict.certified is False and lax.certified is True


def test_a_budgeted_timing_plan_refuses_an_oracle_table():
    """The same §6.2 refusal as the single selection, at the point a timing is consulted."""
    oracle = EncodingCostTable(target="host", cal_gen=1, provenance="oracle",
                               rows=_budget_table().rows)
    try:
        select_budgeted(oracle, _TWO, budget=20, objective=Objective.DECODE_LATENCY)
    except UnmeasuredTarget as error:
        assert "provenance is 'oracle'" in str(error)
    else:
        raise AssertionError("a budgeted timing plan must not read an oracle table")
    # And it is permitted when the caller records the experiment as one.
    plan = select_budgeted(oracle, _TWO, budget=20, objective=Objective.DECODE_LATENCY,
                           allow_oracle_table=True)
    assert plan.total_octets == 20


def test_a_stage_naming_an_unmeasured_candidate_refuses():
    table = _budget_table()
    stages = (Stage("s1", ("A",)), Stage("s2", ("A", "ZZZ")))
    try:
        select_budgeted(table, stages, budget=100, objective=Objective.DECODE_LATENCY)
    except UnmeasuredTarget as error:
        assert "ZZZ" in str(error)
    else:
        raise AssertionError("an unmeasured candidate must refuse, not be skipped")


def test_wire_size_is_the_resource_and_therefore_not_a_legal_objective():
    """Minimizing octets subject to an octet budget is a single selection with extra steps."""
    table = _budget_table()
    for objective in (Objective.WIRE_SIZE, Objective.NONE):
        try:
            select_budgeted(table, _TWO, budget=20, objective=objective)
        except Asn1Error as error:
            assert "the RESOURCE here" in str(error)
        else:
            raise AssertionError(f"{objective} must be refused as a budgeted objective")


def test_indistinguishable_plans_are_reported_as_a_lower_bound():
    """At budget 14 the optimum ties another plan four octets more expensive."""
    table = _budget_table()
    plan = select_budgeted(table, _TWO, budget=14, objective=Objective.DECODE_LATENCY)
    assert plan.latency_median == 400
    # C+C at 12 octets and A+B at 14 both total 400; the interval check finds the rival.
    assert plan.indistinguishable
    rival = [name for _, name in plan.indistinguishable[0]]
    assert sorted(rival) == ["A", "B"]
