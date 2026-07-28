"""The native microbench protocol — and the measurement that justifies J6's refusal.

J6 refuses to decide a timing objective from a Python-oracle table, on the argument in §2
that `json.loads` is native C while the other decoders are Python, so an oracle timing
orders candidates by which implementation happens to be compiled. That was an argument.
`test_the_oracle_ordering_is_inverted_by_the_native_measurement` turns it into evidence:
on this rail the Python harness reports JER decoding **faster** than DER, and the native
measurement reports DER decoding several times faster than JER. The ordering is not merely
noisy — it is backwards.

Everything else here is the protocol that makes such a number worth trusting: one corpus,
warmup discarded, interleaved rounds so drift cannot bias one candidate, per-round medians,
and a table that contains **only** what was natively measured.

Skips cleanly when no C compiler is visible, exactly as the other native tests do.
"""

from __future__ import annotations

from bcir.asn1.certified import MIN_SAMPLES, EncodingCostTable, UnmeasuredTarget, select_certified
from bcir.asn1.native_bench import (
    NATIVE_OPS, NativeOp, build_harness, measured_table, native_available, run_native_bench,
)
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.selection import ALL_CANDIDATES, Objective, measure_one
from bcir.asn1.tags import Asn1Error, Universal

_RECORD = Sequence((
    Component("id", Primitive(Universal.INTEGER)),
    Component("name", Primitive(Universal.UTF8_STRING)),
), name="Record")
_VALUE = {"id": 42, "name": "a-name"}


def _named(*names):
    return [c for c in ALL_CANDIDATES if c.name in set(names)]


# --- the finding ----------------------------------------------------------------------------


def test_the_oracle_ordering_is_inverted_by_the_native_measurement():
    """§2's warning, as evidence rather than as an argument.

    The Python harness times `json.loads` (C) against a Python DER decoder and concludes
    JER is the faster one. The native harness times two C decoders against each other and
    concludes the opposite, by a wide margin. Same values, same encodings, opposite answer
    — which is precisely why J6 refuses to let an oracle table decide a timing objective.

    If this test ever fails because the two rails AGREE, that is worth investigating rather
    than relaxing: it would most likely mean the Python DER decoder had been replaced by a
    native one, at which point §2's caveat needs rewriting rather than the test.
    """
    if not native_available():
        return
    oracle = {m.candidate: m for m in
              (measure_one(c, _RECORD, _VALUE)
               for c in _named("DER", "JER-BCIR-CANONICAL"))}
    assert oracle["JER-BCIR-CANONICAL"].decode_ns < oracle["DER"].decode_ns, (
        "the Python harness no longer reports JER as the faster decode; §2's caveat and "
        "this test both need revisiting")

    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1)
    rows = {row.candidate: row for row in table.rows}
    assert rows["DER"].decode.median < rows["JER-BCIR-CANONICAL"].decode.median, (
        "the native measurement agrees with the oracle; that would be a real change in "
        "the C rail, not a flaky benchmark")
    # And the two are separable, so this is not a coin flip between overlapping intervals.
    assert not rows["DER"].decode.overlaps(rows["JER-BCIR-CANONICAL"].decode)


def test_a_measured_table_lets_a_timing_objective_be_decided_at_last():
    """The point of the whole exercise: J6's guard stops being a permanent refusal.

    Before this module existed, every timing objective raised `UnmeasuredTarget` because no
    `measured` table could be produced. The same call now returns a certificate.
    """
    if not native_available():
        return
    only = _named("DER", "JER-BCIR-CANONICAL")
    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1, candidates=only)
    certificate = select_certified(_RECORD, _VALUE, table,
                                   objective=Objective.DECODE_LATENCY, candidates=only)
    assert certificate.selected == "DER"
    assert certificate.provenance == "measured"
    assert certificate.table_digest == table.digest()
    assert certificate.interval_rank_coverage_ppm > 900_000


# --- what is measured, and what is refused ---------------------------------------------------


def test_every_candidate_has_an_explicit_native_decision():
    """Adding a candidate must force a decision here rather than dropping it silently.

    A candidate absent from `NATIVE_OPS` would simply not appear in a measured table, and
    `select_certified` would then refuse objectives for a reason nobody wrote down.
    """
    assert {c.name for c in ALL_CANDIDATES} == set(NATIVE_OPS)
    for name, entry in NATIVE_OPS.items():
        assert isinstance(entry, NativeOp)
        if entry.op is None:
            assert entry.reason, f"{name} is unmeasured with no reason given"


def test_per_is_refused_permanently_and_oer_is_refused_for_now():
    """"Not yet" and "not ever" are different, and the map says which.

    X.691 §7.2 makes a PER encoding non-self-delimiting: without the type, the octets
    cannot be walked, so there is no schema-free structural pass to time and no comparable
    native number can exist. OER's absence is an ordinary gap that closes when somebody
    writes a C decoder. A consumer that treated them the same would either wait forever for
    a PER row or conclude the harness was broken.
    """
    for name in ("CANONICAL-PER-ALIGNED", "CANONICAL-PER-UNALIGNED",
                 "BASIC-PER-ALIGNED", "BASIC-PER-UNALIGNED"):
        entry = NATIVE_OPS[name]
        assert entry.op is None and entry.permanent, name
        assert "7.2" in entry.reason and "self-delimiting" in entry.reason, name
    for name in ("COER", "BASIC-OER"):
        entry = NATIVE_OPS[name]
        assert entry.op is None and not entry.permanent, name
        assert "yet" in entry.reason, name


def test_the_measured_table_contains_only_natively_measured_rows():
    """The refusal that keeps a `measured` label honest.

    A table row for a candidate with no native decoder could only come from a Python
    timing — the very substitution J6 exists to prevent, smuggled in one layer lower.
    """
    if not native_available():
        return
    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1)
    measured = {row.candidate for row in table.rows}
    expected = {name for name, entry in NATIVE_OPS.items() if entry.op is not None}
    assert measured == expected, sorted(measured ^ expected)
    assert table.provenance == "measured"


def test_an_unmeasurable_candidate_still_blocks_the_objective_that_needs_it():
    """The refusal chain, end to end.

    The table honestly omits COER; `select_certified` then refuses a decode-latency
    decision that would have needed it. Nothing is filled in at either step, and the error
    names what is missing.
    """
    if not native_available():
        return
    only = _named("DER", "COER")
    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1, candidates=only)
    assert {row.candidate for row in table.rows} == {"DER"}
    try:
        select_certified(_RECORD, _VALUE, table, objective=Objective.DECODE_LATENCY,
                         candidates=only)
    except UnmeasuredTarget as error:
        assert "COER" in str(error)
    else:
        raise AssertionError("a missing COER row was silently skipped")


def test_a_candidate_that_cannot_carry_the_value_is_skipped_with_its_refusal():
    """Legality still comes first, even in the measurement harness."""
    if not native_available():
        return
    _samples, skipped = run_native_bench(_RECORD, {"id": 1}, candidates=_named("DER"))
    assert "DER" in skipped and "not representable" in skipped["DER"]


# --- the protocol -----------------------------------------------------------------------------


def test_the_harness_builds_warning_clean():
    """Strict warnings, like every other C translation unit in the repository."""
    import tempfile
    if not native_available():
        return
    with tempfile.TemporaryDirectory() as tmp:
        assert build_harness(tmp) is not None


def test_enough_rounds_are_collected_for_an_order_statistic_interval():
    """A narrower table is not a more precise one — the floor is enforced end to end."""
    if not native_available():
        return
    samples, _skipped = run_native_bench(_RECORD, _VALUE, candidates=_named("DER"))
    assert samples and len(samples[0].decode_ns) >= MIN_SAMPLES


def test_too_few_rounds_is_refused_rather_than_averaged():
    """Asking for fewer rounds than the interval needs fails loudly."""
    if not native_available():
        return
    try:
        measured_table(_RECORD, _VALUE, target="host", cal_gen=1, rounds=3,
                       candidates=_named("DER"))
    except Asn1Error as error:
        assert str(MIN_SAMPLES) in str(error)
    else:
        raise AssertionError("a table was built from three rounds")


def test_the_same_octets_are_timed_every_round():
    """One corpus, identical bytes. A benchmark that regenerated its input would be timing
    the generator, and the octets a candidate produces ARE the candidate."""
    if not native_available():
        return
    first, _ = run_native_bench(_RECORD, _VALUE, candidates=_named("DER", "BER"))
    second, _ = run_native_bench(_RECORD, _VALUE, candidates=_named("DER", "BER"))
    assert {s.candidate: s.octets for s in first} == {s.candidate: s.octets for s in second}


def test_the_measurement_is_stable_enough_to_separate_what_it_claims_to_separate():
    """Repeatability, as the gate asks — checked by re-running, not asserted.

    Two independent runs must reach the same verdict about which candidate is faster. A
    protocol whose answer changed between runs would be reporting scheduler noise with a
    confidence interval attached, which is worse than reporting nothing.
    """
    if not native_available():
        return
    only = _named("DER", "JER-BCIR-CANONICAL")
    verdicts = []
    for _ in range(2):
        table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1, candidates=only)
        rows = {r.candidate: r for r in table.rows}
        verdicts.append(rows["DER"].decode.median < rows["JER-BCIR-CANONICAL"].decode.median)
    assert verdicts[0] == verdicts[1], "two runs disagreed about which decode is faster"


def test_a_measured_table_is_content_addressed_and_generation_tagged():
    """It is data with an identity, so a certificate can bind to it."""
    if not native_available():
        return
    table = measured_table(_RECORD, _VALUE, target="alpha", cal_gen=7,
                           candidates=_named("DER"))
    assert (table.target, table.cal_gen, table.provenance) == ("alpha", 7, "measured")
    rebuilt = EncodingCostTable(target="alpha", cal_gen=7, provenance="measured",
                                rows=table.rows)
    assert rebuilt.digest() == table.digest()
