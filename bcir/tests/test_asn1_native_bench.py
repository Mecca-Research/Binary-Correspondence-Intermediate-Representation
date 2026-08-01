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

import os

from bcir.asn1.certified import MIN_SAMPLES, EncodingCostTable, UnmeasuredTarget, select_certified
from bcir.asn1.codec import decode_one, encode_der, encode_tlv
from bcir.asn1.jer import encode_jer
from bcir.asn1.native_bench import (
    ENCODE_OPS, NATIVE_OPS, NativeOp, build_harness, measured_table, native_available,
    observed_encode_partition, run_native_bench, run_native_encode_bench,
)
from bcir.asn1.schema import Component, Primitive, Sequence
from bcir.asn1.selection import ALL_CANDIDATES, Objective, measure_one
from bcir.asn1.tags import Asn1Error, Universal

_ROOT_C = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "runtime", "c")

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


def test_per_and_oer_are_both_refused_by_law_not_by_a_missing_decoder():
    """The correction. This test asserted the opposite when it was written.

    OER's entry read "no C OER decoder exists yet", which called a law an ordinary gap.
    X.696 §6.2 states the same rule as X.691 §7.2 — *"without knowledge of the type of the
    value encoded, it is not possible to determine the structure of the encoding"* — so
    neither has a schema-free structural pass to time.

    `runtime/c/bcir_oer.c` now decodes OER natively, and writing it is what exposed the
    mislabel: the decoder is schema-**directed**, while every row in this table is a
    schema-free structural scan. Timing the two against each other would compare unlike
    work and report the difference as an encoding cost, which is the error §2 warns about
    one level up. So the decoder exists AND the row is still absent, for a stated reason.
    """
    for name in ("CANONICAL-PER-ALIGNED", "CANONICAL-PER-UNALIGNED",
                 "BASIC-PER-ALIGNED", "BASIC-PER-UNALIGNED"):
        entry = NATIVE_OPS[name]
        assert entry.op is None and entry.permanent, name
        assert "7.2" in entry.reason and "self-delimiting" in entry.reason, name
    for name in ("COER", "BASIC-OER"):
        entry = NATIVE_OPS[name]
        assert entry.op is None and entry.permanent, name
        assert "6.2" in entry.reason, name
        assert "yet" not in entry.reason, (
            f"{name} still describes a law as a temporary gap")
    # The native decoder really is there — the absence is about comparability, not capability.
    assert os.path.exists(os.path.join(_ROOT_C, "bcir_oer.c"))


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


# --- the encode column, and why it is not the decode table's mirror ------------------------


def test_the_recorded_encode_partition_matches_the_encoders_themselves():
    """`ENCODE_OPS` is derived-checkable, not a comment.

    The table above records which candidates have a schema-free encoder. That is a claim
    about the oracle's own code, so it is checked against the oracle's own code — a table
    that drifted from the encoders would be worse than no table, because it would be
    consulted.
    """
    observed = observed_encode_partition()
    assert {name: op.schema_free for name, op in ENCODE_OPS.items()} == observed
    assert set(ENCODE_OPS) == {c.name for c in ALL_CANDIDATES}, (
        "every candidate needs an encode verdict, or the column has a silent hole")


def test_only_the_x690_family_can_be_encoded_without_a_schema():
    """The finding, as a fact rather than as prose.

    X.690 is self-describing in *both* directions: a TLV tree carries its own tags and
    lengths, so `encode_der` takes a value and no type and a re-emit is the whole operation.
    Every other candidate needs the type — X.697 §22.2 puts member identifiers in a JER
    document and identifiers live only in the schema.
    """
    free = {name for name, op in ENCODE_OPS.items() if op.schema_free}
    assert free == {"DER", "BER"}
    # The complete operation, not a stand-in: octets in, tree, octets out, byte-identical.
    source = encode_der([1, 2, b"abc"])
    assert encode_tlv(decode_one(source)) == source
    # And the JER encoder refuses without a type rather than inventing identifiers.
    try:
        encode_jer(None, {"id": 1})
    except Asn1Error as error:
        assert "schema type" in str(error)
    else:
        raise AssertionError("JER must not encode without a schema")


def test_the_encode_and_decode_absences_are_different_absences():
    """The asymmetry that decides what an encode harness has to be.

    PER and OER are permanently absent from the *decode* table because X.691 §7.2 and
    X.696 §6.2 say the structure cannot be recovered without the type. Neither says anything
    about the write side, so both are perfectly encodable *given a plan* — while JER, which
    the decode table measures, is not encodable without one. A schema-free encode harness
    would therefore produce a two-row table with JER missing, which reads as an unfinished
    implementation rather than as the law it is.
    """
    decode_permanent = {name for name, op in NATIVE_OPS.items() if op.permanent}
    encode_directed = {name for name, op in ENCODE_OPS.items() if not op.schema_free}
    assert decode_permanent < encode_directed, "encode should be the stricter partition"
    # The two candidates that are measurable one way and not the other.
    assert {"JER", "JER-BCIR-CANONICAL"} <= encode_directed - decode_permanent
    # And the ones absent from decode are NOT absent for a write-side reason.
    for name in decode_permanent:
        assert "not self-delimiting" in NATIVE_OPS[name].reason or "structure" in \
            NATIVE_OPS[name].reason
        assert "self-delimiting" not in ENCODE_OPS[name].reason


# --- E2: the native encode column ------------------------------------------------------------


def test_the_encode_column_is_measured_natively_and_covers_oer_and_per():
    """The payoff of E1/E2: OER **and PER** encode numbers the DECODE table can never hold.

    X.696 §6.2 denies OER a schema-free decode forever, and X.691 §7.2 denies PER one for a
    different reason — a PER encoding is not self-delimiting at all. Both clauses are about
    *reading*. They say nothing about the write side, where an encoder is handed the type
    either way, so the encode column reaches two rows the decode column is permanently
    barred from.

    PER contributes **four** rows rather than one: ALIGNED/UNALIGNED is a real cost trade and
    CANONICAL/BASIC decides §19.5's DEFAULT rule, so one row for the pair would report one
    number for two encodings.
    """
    if not native_available():
        return
    samples, skipped = run_native_encode_bench(_RECORD, _VALUE)
    assert set(samples) >= {
        "DER", "BER", "JER", "COER",
        "CANONICAL-PER-ALIGNED", "CANONICAL-PER-UNALIGNED",
        "BASIC-PER-ALIGNED", "BASIC-PER-UNALIGNED",
    }, sorted(samples)
    for name, rounds in samples.items():
        assert len(rounds) >= MIN_SAMPLES, f"{name}: {len(rounds)} rounds"
        assert all(value >= 0 for value in rounds)
    # Whatever is still skipped is skipped for a STATED reason rather than missing silently.
    for name in skipped:
        assert ENCODE_OPS[name].reason, name


def test_ber_encodes_faster_than_der_because_the_standard_says_it_may():
    """A cost difference the standard predicts, not an artefact of this implementation.

    X.690 §10.1 forbids DER the indefinite length form, so a DER encoder must know each
    constructed length before writing its header — two passes, or a shift. §8.1.3.6 lets BER
    leave the length open and close with an EOC, so it needs one pass and no scratch. The
    gap between the two rows is therefore a property of the encodings.

    Asserted on the median only. The intervals are not required to separate: on a contended
    runner they will sometimes overlap, and widening the claim to non-overlap would make
    this test report the runner's load as an encoding fact.
    """
    if not native_available():
        return
    samples, _ = run_native_encode_bench(_RECORD, _VALUE, rounds=MIN_SAMPLES + 10)
    der = sorted(samples["DER"])[len(samples["DER"]) // 2]
    ber = sorted(samples["BER"])[len(samples["BER"]) // 2]
    assert ber <= der, f"BER {ber}ns did not beat DER {der}ns; §8.1.3.6 says it should"


def test_a_measured_table_now_carries_a_real_encode_interval():
    """It used to copy the decode figure, because the C rail had no encoder. It has one."""
    if not native_available():
        return
    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1,
                           candidates=_named("DER", "BER"))
    rows = {row.candidate: row for row in table.rows}
    assert set(rows) == {"DER", "BER"}
    # The two axes are now independent measurements, so demanding they differ would be
    # asserting on noise; what must hold is that both are real intervals over real samples.
    for row in rows.values():
        assert row.encode.samples >= MIN_SAMPLES and row.decode.samples >= MIN_SAMPLES
        assert row.encode.low <= row.encode.median <= row.encode.high


def test_oer_has_an_encode_row_but_still_no_two_axis_row():
    """`CostRow` needs both axes, and OER can never close the decode half.

    Refusing to fabricate the missing half is the same discipline §6.2 applies one level up:
    the number that exists is available, and the row that would need an invented number is
    not produced.
    """
    if not native_available():
        return
    samples, _ = run_native_encode_bench(_RECORD, _VALUE)
    assert "COER" in samples and len(samples["COER"]) >= MIN_SAMPLES
    table = measured_table(_RECORD, _VALUE, target="host", cal_gen=1)
    assert "COER" not in {row.candidate for row in table.rows}
    assert NATIVE_OPS["COER"].permanent and ENCODE_OPS["COER"].native_op == "coer"
