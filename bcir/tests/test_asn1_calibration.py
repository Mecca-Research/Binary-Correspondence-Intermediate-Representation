"""J6's calibration records: what a frozen cost table is allowed to be built from.

`certified.py` will only let the planner use a timing from a frozen, generation-tagged table
measured on a real target — and `measured_table(target=...)` took the target on **trust**.
A table measured on a throttled shared runner and one measured on a pinned dedicated core
were the same type, carried the same `provenance="measured"`, and selected differently.

These tests pin the gate that closes it. Every refusal below is a reason the *numbers do not
mean what they appear to mean*, and the two that matter most are the ones `simd_hosts.py`
learned the hard way: a `dedicated` declaration contradicted by the host's own counters, and
a big.LITTLE run averaged across two different cores.
"""

from __future__ import annotations

import json
import os
import tempfile

from bcir.asn1.calibration import (
    DEDICATED, STORE, TIMING_METHOD, CalibrationRecord, CandidateRow, as_json, calibration_corpus,
    corpus_digest, load_records, render,
)
from bcir.asn1.certified import MIN_SAMPLES
from bcir.asn1.tags import Asn1Error

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _samples(base: int, n: int = MIN_SAMPLES) -> tuple[int, ...]:
    return tuple(base + (i % 3) for i in range(n))


def _rows() -> tuple[CandidateRow, ...]:
    return (
        CandidateRow("DER", 24, _samples(70), _samples(50)),
        CandidateRow("JER", 71, _samples(66), _samples(210)),
        # Encode-only, permanently: X.696 §6.2 denies OER a schema-free decode.
        CandidateRow("COER", 13, _samples(49), ()),
    )


def _record(**overrides) -> CalibrationRecord:
    fields = dict(target="Test Target", arch="aarch64", tenancy=DEDICATED, cal_gen=1,
                  rows=_rows(), cpus=(7,), steal_ticks=0, throttled_usec=0,
                  corpus=corpus_digest())
    fields.update(overrides)
    return CalibrationRecord(**fields)


def test_an_admissible_record_freezes_into_a_measured_table():
    record = _record()
    assert record.admissible(), record.refusals()
    table = record.table()
    assert table.provenance == "measured"
    assert table.target == "Test Target"
    assert table.cal_gen == 1
    # Only candidates with BOTH axes become rows: a `CostRow` carries encode and decode, and
    # COER can never have the second one.
    assert sorted(row.candidate for row in table.rows) == ["DER", "JER"]
    assert record.incomplete() == ("COER",)


def test_a_biglittle_run_is_refused_because_it_describes_no_core_that_exists():
    """The failure `simd_hosts.py` already caught once, on this exact SoC family.

    A Snapdragon 8 Gen 3 has a Cortex-X4, four A720s and three A520s. Rounds spread across
    clusters are two machines averaged — and a cost table, unlike a SIMD record, is then
    frozen and steers production selection.
    """
    problems = _record(cpus=(0, 7)).refusals()
    assert any("two machines averaged" in p for p in problems), problems


def test_an_unobserved_cpu_is_refused_rather_than_assumed_stable():
    problems = _record(cpus=(-1,)).refusals()
    assert any("unobserved rather than absent" in p for p in problems), problems


def test_a_dedicated_claim_is_refused_by_the_hosts_own_counters():
    """`dedicated` is a claim, and the machine keeps its own accounting of whether it held."""
    stolen = _record(steal_ticks=3).refusals()
    assert any("steal tick" in p for p in stolen), stolen
    throttled = _record(throttled_usec=1200).refusals()
    assert any("throttled" in p for p in throttled), throttled
    # A host that does not report the counters cannot be refused on them: the check exists to
    # catch a false declaration, not to invalidate an honest record.
    assert _record(steal_ticks=None, throttled_usec=None).admissible()


def test_a_shared_runner_cannot_freeze_a_table():
    problems = _record(tenancy="shared").refusals()
    assert any("shared runner" in p for p in problems), problems


def test_too_few_rounds_are_refused_on_either_axis():
    thin = (CandidateRow("DER", 24, _samples(70, MIN_SAMPLES - 1), _samples(50)),)
    problems = _record(rows=thin).refusals()
    assert any("below the" in p for p in problems), problems


def test_a_row_with_neither_axis_measures_nothing():
    problems = _record(rows=(CandidateRow("DER", 24, (), ()),)).refusals()
    assert any("measures nothing" in p for p in problems), problems


def test_generation_zero_names_no_calibration():
    problems = _record(cal_gen=0).refusals()
    assert any("generation 0" in p for p in problems), problems


def test_a_record_from_a_different_corpus_is_refused():
    """The silent failure the digest exists for.

    Two targets that measured different schemas produce two tables that look identical and
    compare cleanly. Nothing about the numbers reveals it, which is why the corpus is pinned
    by digest rather than by convention.
    """
    problems = _record(corpus="0000000000000000").refusals()
    assert any("different schema" in p for p in problems), problems
    # An absent digest is a record made before the field existed. It proves nothing either
    # way, so it is reported by its absence rather than refused.
    assert _record(corpus="").admissible()


def test_an_inadmissible_record_refuses_to_produce_a_table_at_all():
    """Not a table with a warning: the caller here is the planner, which is precisely the
    component that must not be the one weighing whether a measurement was sound."""
    try:
        _record(tenancy="shared").table()
    except Asn1Error as error:
        assert "cannot be frozen" in str(error), error
    else:
        raise AssertionError("an inadmissible record produced a cost table")


def test_the_corpus_is_fixed_and_its_digest_is_stable():
    """A corpus that varied per run would make two targets incomparable without saying so."""
    first, second = corpus_digest(), corpus_digest()
    assert first == second, "the corpus digest is not deterministic"
    assert len(first) == 16, first
    kind, value = calibration_corpus()
    assert kind.name == "Calibration"
    # Each component exists because some candidate prices it differently from the others.
    assert sorted(value) == ["flag", "label", "mode", "small", "wide"]


def test_a_record_round_trips_through_the_store_format():
    record = _record()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "store.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"targets": [json.loads(as_json(record))]}, handle)
        loaded = load_records(path)
    assert len(loaded) == 1
    assert loaded[0] == record, "a record did not survive the store format unchanged"


def test_the_loader_refuses_a_record_missing_a_declared_field():
    """A defaulted `tenancy` would read as a declaration nobody made."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "store.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"targets": [{"target": "T", "arch": "aarch64"}]}, handle)
        try:
            load_records(path)
        except Asn1Error as error:
            assert "cannot be defaulted" in str(error), error
        else:
            raise AssertionError("a record missing required fields loaded")


def test_the_report_states_refusals_as_loudly_as_admissions():
    text = render([_record(), _record(target="Shared Box", tenancy="shared")])
    assert "ADMITTED" in text and "REFUSED" in text, text
    assert "shared runner" in text, text
    # A one-axis row must be visibly excluded rather than quietly dropped.
    assert "one axis only" in text, text


def test_the_repository_store_loads_and_admits_only_what_it_should():
    """The store is evidence, so it is read here rather than trusted.

    An empty store is a legitimate state — it means no target has been calibrated yet, and
    `select_certified` correctly raises `UnmeasuredTarget` rather than selecting on a table
    nobody measured.
    """
    path = os.path.join(_ROOT, STORE)
    if not os.path.exists(path):
        return
    records = load_records(path)
    for record in records:
        # A record may be REFUSED and still belong here. The store holds evidence, and a
        # measurement superseded by a harness change is evidence about a machine that simply
        # cannot back a current table. What must never appear is a record refused for being
        # *unsound* — a shared runner, a big.LITTLE smear, an unpinned run — because nothing
        # about those improves by keeping them.
        superseded = ("timed by harness method", "measured against corpus")
        for problem in record.refusals():
            assert problem.startswith(superseded), (
                f"{record.target} is in the store and is refused for a reason other than "
                f"being superseded: {problem}")
        assert record.corpus in ("", corpus_digest()) or record.method != TIMING_METHOD, (
            f"{record.target} measured corpus {record.corpus} but this revision compiles "
            f"{corpus_digest()}")


def _ticked(ticks: int, quantum: int = 52, n: int = MIN_SAMPLES) -> tuple[int, ...]:
    """Samples from a clock that only resolves `quantum` ns: every read is the same tick."""
    return tuple([ticks * quantum] * n)


def test_a_coarse_clock_is_detected_from_the_samples_themselves():
    """The finding the first real aarch64 calibration produced.

    Every distinct value in that record — 104, 156, 208, 260, 417 — sat within 0.02 of an
    exact multiple of 52.083 ns, the period of the 19.2 MHz ARM architectural timer. Those
    are 2, 3, 4, 5 and 8 *ticks*. No field records the clock; it is inferred from the numbers,
    which is what makes the check work on a record taken before anyone thought to measure it.
    """
    rows = (CandidateRow("BER", 24, _ticked(2), _ticked(2)),
            CandidateRow("DER", 24, _ticked(3), _ticked(2)),
            CandidateRow("JER", 71, _ticked(4), _ticked(8)))
    assert 50 <= _record(rows=rows).observed_quantum() <= 54

    # A host with a fine clock must not be penalised: dense, non-multiple samples estimate
    # ~1 ns and nothing downstream is widened.
    fine = (CandidateRow("BER", 24, (204, 207, 211, 206, 209, 203, 212),
                         (201, 205, 208, 202, 210, 206, 213)),)
    assert _record(rows=fine).observed_quantum() == 1.0


def test_a_table_interval_is_never_narrower_than_the_clock_that_produced_it():
    """Forty-one identical samples are not a precise measurement on a 52 ns clock.

    `select_certified` decides significance by `Interval.overlaps`. Left alone, two candidates
    one tick apart would be "significantly different" on the strength of one unit of clock
    resolution — so the table widens, and the one-tick pair becomes correctly indistinguishable
    while a genuine four-tick gap stays disjoint.
    """
    rows = (CandidateRow("BER", 24, _ticked(2), _ticked(2)),
            CandidateRow("DER", 24, _ticked(3), _ticked(2)),
            CandidateRow("JER", 71, _ticked(4), _ticked(8)))
    table = {row.candidate: row for row in _record(rows=rows).table().rows}

    ber, der, jer = table["BER"].encode, table["DER"].encode, table["JER"].decode
    assert ber.high - ber.low >= 50, f"a degenerate interval survived: {ber}"
    assert ber.overlaps(der), (
        "one tick apart must read as indistinguishable, not as a significant difference")
    assert not ber.overlaps(jer), (
        f"a six-tick gap must survive widening: {ber} against {jer}")
    # The median is the measurement and must not move.
    assert table["BER"].encode.median == 104


def test_widening_leaves_an_already_honest_interval_alone():
    """This corrects a false precision; it does not inflate a real measurement."""
    spread = (CandidateRow("BER", 24, (100, 180, 260, 140, 220, 160, 240),
                           (100, 180, 260, 140, 220, 160, 240)),)
    record = _record(rows=spread)
    original = spread[0].encode_interval()
    widened = record.table().rows[0].encode
    assert (widened.low, widened.high) == (original.low, original.high), (
        f"an interval already wider than the clock was altered: {original} -> {widened}")


def test_the_counter_state_is_recorded_rather_than_acted_on():
    """An absent PMU is a fact about the machine, not a defect in the record.

    Cycles would resolve far below a 52 ns clock and would be frequency-invariant, which is
    why the harness asks. But a container commonly exposes no PMU and Android normally denies
    `perf_event_open` outright, so the wall-clock figures have to stand on their own — and
    the record says which happened, in the kernel's own words, so "this host has no counters"
    is never confused with "nobody asked".
    """
    assert _record(counters="No such file or directory").admissible()
    assert _record(counters="cycles").admissible()
    text = render([_record(counters="No such file or directory")])
    assert "counters: No such file or directory" in text, text


def test_the_native_harness_reports_its_counter_state_honestly():
    """Whatever this host offers, the harness must name it rather than imply cycles."""
    from bcir.asn1.native_bench import native_available, native_counters

    if not native_available():
        return
    state = native_counters()
    assert state and state != "not reported", (
        f"the harness did not report a counter state ({state!r}); a silent absence is how a "
        f"wall-clock figure ends up under a cycles heading")
    # Either a real PMU, or a reason. Never an empty claim.
    assert state == "cycles" or len(state) > 3, state
