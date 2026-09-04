"""CT4-sec: the adversarial telemetry-injection harness (the integrity proof).

Telemetry is the one BCIR artifact that crosses into decision-making (it calibrates
Theta, picks the adaptive policy, drives the regret gate and the calibration
certificate), and the audit flagged it as the single most vulnerable surface: it
entered that decision path with NO integrity validation. This module is the proof
that the boundary is now closed. For EACH attack it asserts (a) the hardened code
BLOCKS it -- the attack no longer achieves its goal -- and (b) the legitimate path
still works unchanged (the same in-range telemetry, the same decision).

Every test injects a *concretely hostile* input and asserts the defended outcome
differs from the undefended one (never a tautology): an out-of-range field that would
flip the policy, a crafted ring header that would read out of bounds, a forged
certificate that would install adversarial weights, a suppressed stream that would be
a silent no-op. The anti-regression tests pin that genuine telemetry (the existing
`test_telemetry.py` width-flip / close_loop expectations) is byte-identical.

Defense map (file:line of the gate each attack hits):
  - `telemetry.DataDNA.violations`/`sanitize_events`  -> A: ingest schema validation
  - `kbcir.calibrate.{Ewma,Linear,Frozen}.update`     -> A: validation wired at ingest
  - `telemetry.parse_shared_ring`                     -> B: header magic/size/bounds
  - `kbcir.calibrate.FrozenCalibrator.from_json`      -> C: forged-weight rejection
  - `kbcir.calibloop.CalibrationCertificate.from_json`-> C: forged-cert rejection
  - `telemetry.{TelemetryIntegrity,calibrate_with_witness}` -> D/E: blindness + witness
  - `kbcir.sensing.RegretSensor.observe`              -> A: bogus-cost rejection (+ honest residual)
"""

import json
import struct
import threading

from bcir.examples import vector_add
from bcir.kbcir.calibrate import (
    EwmaCalibrator,
    FrozenCalibrator,
    LinearCalibrator,
    adaptive_policy,
    calibrate_and_replan,
    calibrate_with_witness,
)
from bcir.kbcir.calibloop import CalibrationCertificate, close_loop
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.sensing import RegretSensor
from bcir.telemetry import (
    RING_STAMP,
    DataDNA,
    SharedRingError,
    TelemetryIntegrity,
    TelemetryIntegrityError,
    TelemetryRing,
    ValidatingSink,
    ListSink,
    parse_shared_ring,
    sanitize_events,
)

INF = float("inf")
NAN = float("nan")


def _cool(thermal=10, n=3, claim_id=1000):
    """A batch of legitimate, in-range cool telemetry (the trusted baseline)."""
    return [
        DataDNA(
            segment_id=f"s{i}",
            claim_id=claim_id,
            cycles=100,
            bytes=4096,
            misses=10,
            thermal=thermal,
            voltage=0,
            utilization=70,
        )
        for i in range(n)
    ]


# --- A. out-of-range field injection: must NOT move Theta / the policy ------------


def test_out_of_range_field_cannot_move_theta_vs_baseline():
    """ATTACK: inject thermal=10000 (and inf, and negative) into a stream of otherwise
    legitimate cool telemetry, aiming to drag Theta.thermal past the policy's >=60
    threshold and force an attacker-chosen ENERGY replan.
    DEFENSE: `sanitize_events` (wired at `EwmaCalibrator.update`) drops every record
    that violates the documented 0..100 schema before the EWMA sees it.
    PROOF: Theta is IDENTICAL to the in-range baseline (the injected records contribute
    nothing), so the policy/width is unchanged -- the injection is fully neutralized."""
    cal = EwmaCalibrator(alpha=256)
    legit = _cool(thermal=10)
    injected = legit + [
        DataDNA(segment_id="x", claim_id=1001, thermal=10000, utilization=70, misses=10),
        DataDNA(segment_id="x", claim_id=1002, thermal=INF, utilization=70, misses=10),
        DataDNA(segment_id="x", claim_id=1003, thermal=NAN, utilization=70, misses=10),
        DataDNA(segment_id="x", claim_id=1004, thermal=-50, utilization=70, misses=10),
        DataDNA(segment_id="x", claim_id=1005, misses=99999, utilization=70, thermal=10),
    ]
    baseline = cal.update(Theta.cool(), legit)
    attacked = cal.update(Theta.cool(), injected)
    assert attacked == baseline  # injected records had ZERO effect
    assert adaptive_policy(attacked).name == adaptive_policy(baseline).name == "latency"


def test_out_of_range_injection_does_not_force_a_replan_flip():
    """ATTACK: same injection, end-to-end through `calibrate_and_replan` on AVX-512,
    where a hot Theta would flip vector_add vec16 -> vec8.
    DEFENSE/PROOF: the realized lane width matches the all-legit baseline (16); only a
    GENUINE hot batch (next test) is allowed to flip it."""
    h = TargetProfile.x86_avx512()
    cal = EwmaCalibrator(alpha=256)
    legit = _cool(thermal=10)
    injected = legit + [DataDNA(segment_id="x", claim_id=9, thermal=10000, utilization=70)]
    _, pol_b, res_b = calibrate_and_replan(vector_add(1024), h, legit, calibrator=cal)
    _, pol_a, res_a = calibrate_and_replan(vector_add(1024), h, injected, calibrator=cal)
    assert res_a.by_claim()[1000].width == res_b.by_claim()[1000].width == 16
    assert pol_a.name == pol_b.name == "latency"


def test_linear_and_frozen_calibrators_also_reject_at_ingest():
    """The injection defense is at the shared ingest boundary, so the learning
    (`LinearCalibrator`) and frozen (`FrozenCalibrator`) calibrators are protected too:
    a poisoned thermal cannot corrupt the SGD fit nor the frozen prediction average."""
    legit = _cool(thermal=20)
    poisoned = legit + [DataDNA(segment_id="x", claim_id=1, thermal=10000, utilization=70)]
    lin = LinearCalibrator(lr=0.5)
    assert lin.update(Theta.cool(), poisoned) == lin.update(Theta.cool(), legit)
    frozen = FrozenCalibrator(w_q8=(0, 0, 256, 0), gen=1)  # predicts from misses
    assert frozen.update(Theta.cool(), poisoned) == frozen.update(Theta.cool(), legit)


def test_validate_rejects_strict_and_passes_legit_unchanged():
    """The strict `DataDNA.validate` raises on a bogus record (a hard trust boundary)
    but returns a legitimate record unchanged (identity) -- so legit telemetry is never
    penalized by the gate."""
    good = DataDNA(segment_id="s", claim_id=1, thermal=50, utilization=70, misses=10)
    assert good.validate() is good and good.is_valid() and good.violations() == []
    for bad in (
        DataDNA(segment_id="b", claim_id=1, thermal=10000),
        DataDNA(segment_id="b", claim_id=1, voltage=-1),
        DataDNA(segment_id="b", claim_id=1, utilization=NAN),
        DataDNA(segment_id="b", claim_id=1, cycles=-5),
        DataDNA(segment_id="b", claim_id=-1),
        DataDNA(segment_id="b", claim_id=True),
        DataDNA(segment_id="b", claim_id=1 << 63),
        DataDNA(segment_id="b", claim_id=1, cycles=1.5),
        DataDNA(segment_id="b", claim_id=1, bytes=1 << 63),
        DataDNA(segment_id="b", claim_id=1, misses=True),
    ):
        assert not bad.is_valid()
        try:
            bad.validate()
            raise AssertionError("validate() must raise on a schema violation")
        except TelemetryIntegrityError:
            pass


def test_validating_sink_drops_poison_at_the_wire():
    """A `ValidatingSink` at a transport entry drops poisoned events before they reach
    the downstream calibrator sink, and counts them -- so an injection is observable."""
    inner = ListSink()
    sink = ValidatingSink(inner=inner)
    for e in _cool(thermal=30):
        sink.emit(e)
    sink.emit(DataDNA(segment_id="x", claim_id=1, thermal=10000))  # poison
    sink.emit(DataDNA(segment_id="x", claim_id=1, voltage=INF))  # poison
    assert sink.accepted == 3 and sink.rejected == 2
    assert len(inner.events) == 3  # poison never reached downstream
    assert all(ev.is_valid() for ev in inner.events)


def test_validating_sink_serializes_acceptance_and_downstream_publication():
    """Two transport callbacks cannot publish concurrently or race witness counts."""
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    class BlockingSink:
        def __init__(self):
            self.calls = 0

        def emit(self, _event):
            self.calls += 1
            if self.calls == 1:
                entered.set()
                assert release.wait(2)
            else:
                second_entered.set()

        def flush(self):
            pass

    inner = BlockingSink()
    sink = ValidatingSink(inner=inner)
    first = threading.Thread(target=sink.emit, args=(_cool()[0],))
    second = threading.Thread(target=sink.emit, args=(_cool()[1],))
    first.start()
    assert entered.wait(2)
    second.start()
    assert not second_entered.wait(0.1)
    release.set()
    first.join(2)
    second.join(2)
    assert not first.is_alive() and not second.is_alive()
    assert second_entered.is_set()
    assert sink.witness == TelemetryIntegrity(accepted=2, rejected=0)


# --- B. shared-ring header injection: must reject with NO out-of-bounds read ------


def _ring(head, capacity, record_size, magic, slots=4, fill=True):
    buf = bytearray(32 + slots * 56)
    struct.pack_into("<4Q", buf, 0, head, capacity, record_size, magic)
    if fill:
        for i in range(min(head, slots)):
            struct.pack_into("<7q", buf, 32 + i * 56, 200 + i, 10 * (i + 1), 0, 0, 0, 0, 0)
    return buf


def test_ring_legit_layout_still_reads_zero_copy():
    """ANTI-REGRESSION: a well-formed ring (legacy magic==0 OR the new stamp) still
    reads exactly the live window -- the legitimate zero-copy path is unchanged."""
    legacy = _ring(head=3, capacity=4, record_size=56, magic=0)
    assert [r.claim_id for r in parse_shared_ring(legacy)] == [200, 201, 202]
    stamped = _ring(head=3, capacity=4, record_size=56, magic=RING_STAMP)
    assert [r.claim_id for r in parse_shared_ring(stamped)] == [200, 201, 202]


def test_ring_oversized_capacity_is_rejected_no_oob_read():
    """ATTACK: a crafted header advertises capacity=1_000_000 in a buffer that holds 4
    records -- the OOB read / info-disclosure lever (iterate slots past the mmap region).
    DEFENSE: `parse_shared_ring` bounds-checks `header + capacity*record_size` against
    the REAL buffer length before iterating.
    PROOF: returns [] (no read past the buffer) on a deliberately short buffer; strict
    mode raises `SharedRingError`. The unsafe loop is never entered."""
    buf = _ring(head=100, capacity=1_000_000, record_size=56, magic=RING_STAMP, slots=4)
    assert parse_shared_ring(buf) == []
    try:
        parse_shared_ring(buf, strict=True)
        raise AssertionError("oversized capacity must be rejected in strict mode")
    except SharedRingError:
        pass


def test_ring_wrong_record_size_is_rejected():
    """ATTACK: a header lies that record_size=999 (the unpack STRIDE) so the reader
    strides off the records into arbitrary memory.
    DEFENSE/PROOF: record_size must equal the real `TelemetryRing._FMT` stride (56);
    any mismatch is rejected ([] / raise) -- no mis-strided unpack happens."""
    buf = _ring(head=3, capacity=4, record_size=999, magic=RING_STAMP)
    assert parse_shared_ring(buf) == []
    buf_small = _ring(head=3, capacity=4, record_size=8, magic=RING_STAMP)
    assert parse_shared_ring(buf_small) == []


def test_ring_bad_magic_is_rejected():
    """ATTACK: a foreign/forged buffer with the wrong magic word.
    DEFENSE/PROOF: only the self-describing stamp or the legacy 0 are accepted; any
    other magic is rejected, so a forged buffer is never parsed."""
    buf = _ring(head=3, capacity=4, record_size=56, magic=0xDEADBEEF)
    assert parse_shared_ring(buf) == []
    try:
        parse_shared_ring(buf, strict=True)
        raise AssertionError("bad magic must be rejected in strict mode")
    except SharedRingError:
        pass


def test_ring_short_buffer_never_reads_past_end():
    """ATTACK: a buffer shorter than even the header (the truncation case).
    DEFENSE/PROOF: a too-short buffer yields [] -- the header unpack itself is guarded,
    so there is no out-of-bounds `struct.unpack_from`."""
    assert parse_shared_ring(bytearray(8)) == []
    assert parse_shared_ring(b"") == []
    # header-present but record region truncated mid-record:
    buf = bytearray(32 + 56)  # room for 1 record
    struct.pack_into("<4Q", buf, 0, 4, 4, 56, RING_STAMP)  # claims 4 live, buffer holds 1
    assert parse_shared_ring(buf) == []  # capacity*record_size overruns -> reject


def test_ring_rejects_schema_poison_and_impractical_capacity():
    poisoned = _ring(head=1, capacity=4, record_size=56, magic=RING_STAMP)
    struct.pack_into("<7q", poisoned, 32, 1, 1, 1, 1, 10000, 1, 1)
    assert parse_shared_ring(poisoned) == []
    try:
        parse_shared_ring(poisoned, strict=True)
        raise AssertionError("schema poison must be rejected at the ring boundary")
    except SharedRingError:
        pass

    # The backing object need not actually contain this many slots: the capacity
    # ceiling is checked before any record iteration or attacker-scaled allocation.
    oversized = bytearray(32)
    struct.pack_into("<4Q", oversized, 0, 0, (1 << 20) + 1, 56, RING_STAMP)
    assert parse_shared_ring(oversized) == []


# --- C. forged from_json: must reject (never install adversarial state) -----------


def test_forged_frozen_calibrator_is_rejected():
    """ATTACK: a forged certificate ships oversized / wrong-length / non-finite Q8
    weights, or a negative generation, to install an adversarial calibrator.
    DEFENSE/PROOF: `FrozenCalibrator.from_json` validates exactly 4 finite in-range int
    weights and a non-negative int gen; every forgery raises `ValueError`. A VALID
    certificate still round-trips unchanged (anti-regression)."""
    good = FrozenCalibrator(w_q8=(10, 20, 30, 40), gen=2, samples=5)
    assert FrozenCalibrator.from_json(good.to_json()).w_q8 == (10, 20, 30, 40)
    forgeries = [
        {"w_q8": [1 << 40, 0, 0, 0], "gen": 1},  # oversized weight
        {"w_q8": [1, 2, 3], "gen": 1},  # wrong length
        {"w_q8": [1, 2, 3, 4, 5], "gen": 1},  # too many
        {"w_q8": [1.5, 2, 3, 4], "gen": 1},  # non-int
        {"w_q8": [1, 2, 3, 4], "gen": -3},  # negative gen
        {"w_q8": [1, 2, 3, 4], "gen": True},  # bool masquerading as int
        {"w_q8": "not-an-array", "gen": 1},
    ]
    for f in forgeries:
        try:
            FrozenCalibrator.from_json(json.dumps(f))
            raise AssertionError(f"forged FrozenCalibrator not rejected: {f}")
        except ValueError:
            pass
    for text in (
        good.to_json().replace('"gen": 2', '"gen": 2, "gen": 3'),
        json.dumps({**json.loads(good.to_json()), "unknown": 1}),
        json.dumps({**json.loads(good.to_json()), "samples": 1 << 80}),
    ):
        try:
            FrozenCalibrator.from_json(text)
            raise AssertionError("ambiguous FrozenCalibrator was not rejected")
        except ValueError:
            pass


def test_forged_calibration_certificate_is_rejected():
    """ATTACK: a forged closed-loop certificate (negative cal_gen, out-of-band thermal,
    malformed ratios/widths) to seat a fake R13 witness.
    DEFENSE/PROOF: `CalibrationCertificate.from_json` validates the field types/ranges;
    each forgery raises. A genuine certificate still round-trips with the same win."""
    cert = CalibrationCertificate(
        target="avx",
        cal_gen=2,
        provenance="p",
        ratios=(256, 256, 256, 256),
        measured_thermal=40,
        seeded_widths=((1000, 16),),
        calibrated_widths=((1000, 8),),
        stale_cost=100,
        calibrated_cost=80,
    )
    assert CalibrationCertificate.from_json(cert.to_json()).win == 20
    base = json.loads(cert.to_json())
    for label, mut in [
        ("neg cal_gen", {"cal_gen": -1}),
        ("thermal>100", {"measured_thermal": 9999}),
        ("thermal<0", {"measured_thermal": -1}),
        ("3 ratios", {"ratios": [1, 2, 3]}),
        ("bad baseline", {"ratios": [255, 256, 256, 256]}),
        ("malformed width", {"seeded_widths": [[1, 2, 3]]}),
        ("duplicate width", {"seeded_widths": [[1000, 8], [1000, 16]]}),
        ("different claims", {"seeded_widths": [[2000, 16]]}),
        ("negative win", {"stale_cost": 79}),
        ("unknown field", {"unknown": 1}),
        ("non-int cost", {"stale_cost": 1.5}),
    ]:
        d = dict(base)
        d.update(mut)
        try:
            CalibrationCertificate.from_json(json.dumps(d))
            raise AssertionError(f"forged certificate not rejected: {label}")
        except ValueError:
            pass
    duplicate = cert.to_json().replace('"cal_gen": 2', '"cal_gen": 2, "cal_gen": 3')
    try:
        CalibrationCertificate.from_json(duplicate)
        raise AssertionError("duplicate certificate key was not rejected")
    except ValueError:
        pass


# --- D/E. suppression / blindness must be observable (the witness fires) ----------


def test_empty_stream_is_a_surfaced_blind_signal():
    """ATTACK: suppress telemetry entirely (drop the stream). Plain `update` swallows
    it (Theta unchanged) -- a silent no-op the audit flagged.
    DEFENSE/PROOF: `calibrate_with_witness` surfaces `witness.blind` (no valid record
    survived) so the no-throttle outcome is a VISIBLE decision; Theta is still
    unchanged (correct), but the absence is now detectable."""
    theta, witness = calibrate_with_witness(Theta.cool(), [], EwmaCalibrator())
    assert theta == Theta.cool()  # numerically a no-op (correct)
    assert witness.blind and witness.accepted == 0  # ...but the blindness is SURFACED
    assert not witness.clean


def test_all_rejected_stream_is_blind_not_silent():
    """ATTACK: flood the stream with all-bogus records (so the calibrator returns Theta
    unchanged, masquerading as 'nominal').
    DEFENSE/PROOF: every record is rejected, so the witness is `blind` with
    `rejected == len(stream)` -- the suppression-by-poison is observable, not silent."""
    bogus = [DataDNA(segment_id="x", claim_id=i, thermal=10000) for i in range(5)]
    theta, witness = calibrate_with_witness(Theta.cool(), bogus, EwmaCalibrator())
    assert theta == Theta.cool()
    assert witness.blind and witness.rejected == 5 and witness.accepted == 0


def test_legit_stream_is_clean_and_not_blind():
    """ANTI-REGRESSION: a legitimate in-range batch is NOT blind and IS clean, and the
    Theta it produces is identical to the plain `calibrator.update` -- the witness adds
    a signal, it does not change the legitimate calibration."""
    legit = _cool(thermal=80)
    cal = EwmaCalibrator(alpha=256)
    theta, witness = calibrate_with_witness(Theta.cool(), legit, cal)
    assert theta == cal.update(Theta.cool(), legit)  # witness path == plain path
    assert not witness.blind and witness.clean and witness.accepted == 3


def test_ring_eviction_is_an_observable_loss_not_a_silent_drop():
    """ATTACK: overflow the ring so unread records are silently evicted (telemetry
    vanishes before the calibrator reads it).
    DEFENSE/PROOF: `TelemetryRing.lost` / `.witness().dropped` surface the eviction; a
    fresh ring is not `lost`, an overflowed one is."""
    ring = TelemetryRing(capacity=2)
    for i in range(2):
        ring.write(DataDNA(segment_id="s", claim_id=i, thermal=10))
    assert not ring.lost and ring.witness().dropped == 0  # within capacity: no loss
    for i in range(2, 5):
        ring.write(DataDNA(segment_id="s", claim_id=i, thermal=10))
    assert ring.lost and ring.witness().dropped == 3  # eviction is SURFACED


def test_replay_reorder_is_detected_by_the_witness():
    """ATTACK: replay/reorder records (resend a stale claim_id) to confuse a consumer
    that keys on sequence.
    DEFENSE/PROOF: the witness's `monotonic` is False when claim_ids are not a strictly
    increasing run, so a duplicate/out-of-order batch is detectable."""
    forward = [DataDNA(segment_id="s", claim_id=c, thermal=10) for c in (1, 2, 3)]
    _, w_ok = sanitize_events(forward)
    assert w_ok.monotonic and w_ok.seq_span == 3
    replayed = [DataDNA(segment_id="s", claim_id=c, thermal=10) for c in (1, 2, 1, 3)]
    _, w_bad = sanitize_events(replayed)
    assert not w_bad.monotonic  # replay detected


# --- the RegretSensor cost-masking attack (partially defensible -- honest) --------


def test_regret_sensor_rejects_bogus_costs():
    """ATTACK: feed the regret sensor non-finite / negative costs (a NaN poisons every
    downstream variance/CV sum).
    DEFENSE/PROOF: `RegretSensor.observe` REJECTS-and-counts any non-finite/negative
    cost (it never enters `PathSamples`); a legitimate non-negative cost is recorded."""
    s = RegretSensor()
    for bad in (NAN, INF, -1, -999, True, "100"):
        s.observe("p", bad)
    assert s.rejected == 6 and "p" not in s.paths  # nothing was recorded
    s.observe("good", 100)
    s.observe("good", 110)
    assert s.paths["good"].n == 2  # legit cost still works


def test_regret_sensor_masking_is_only_partially_defended_honestly():
    """ATTACK (the hard one): feed a genuinely-bad path PLAUSIBLE low-variance costs so
    it looks 'confident' (cv below threshold) and telemetry is forced off/low -- the bad
    path is never instrumented.
    DEFENSE/PROOF (honest partial): a *single* unwitnessed sample is forced 'off' but
    flagged `forced_off=True`, so the absence of evidence is observable. The residual
    that is NOT defended here -- a stream of plausible, in-range, low-variance fabricated
    costs (cv==0 -> 'low') -- is documented as needing source authentication (a producer
    MAC), out of scope. This test pins BOTH facts: the flag fires for the under-sampled
    case, and the in-range masking still slips to 'low' (the honest residual)."""
    s = RegretSensor()
    s.observe("bad_path", 100)  # 1 plausible sample
    d1 = {x.path: x for x in s.sense(min_samples=2)}
    assert (
        d1["bad_path"].resolution == "off" and d1["bad_path"].forced_off
    )  # unwitnessed -> flagged

    for _ in range(5):
        s.observe("bad_path", 100)  # plausible, zero-variance
    d2 = {x.path: x for x in s.sense(min_samples=2)}
    # honest residual: in-range zero-variance costs read as 'confident' -> 'low', NOT
    # flagged forced_off (it has enough samples). The witness cannot tell a fabricated
    # plausible cost from a real one -- that is the documented source-trust boundary.
    assert d2["bad_path"].resolution == "low" and d2["bad_path"].cv_milli == 0
    assert not d2["bad_path"].forced_off


# --- the FULL legitimate loop is unchanged (mirror test_telemetry.py) -------------


def test_legit_hot_telemetry_still_flips_lane_16_to_8():
    """ANTI-REGRESSION (mirrors test_telemetry.test_hot_telemetry_replans_to_narrower_lane):
    GENUINE hot, in-range telemetry must STILL flip vector_add vec16 -> vec8 on AVX-512
    -- the validation gate is invisible to legitimate telemetry."""
    h = TargetProfile.x86_avx512()
    theta, policy, res = calibrate_and_replan(
        vector_add(1024),
        h,
        _cool(thermal=100),
        base_theta=Theta.cool(),
        calibrator=EwmaCalibrator(alpha=256),
    )
    assert theta.thermal == 100
    assert policy.name == "energy"
    assert res.by_claim()[1000].width == 8


def test_legit_close_loop_certifies_a_win_unchanged():
    """ANTI-REGRESSION: the closed loop on legitimate hot telemetry still certifies a
    >= 0 replan win and an admissible certificate -- the ingest validation does not
    perturb a genuine measured replan."""
    h = TargetProfile.x86_avx512()
    cert = close_loop(
        vector_add(1024), h, events=_cool(thermal=100), calibrator=EwmaCalibrator(alpha=256)
    )
    assert cert.admissible and cert.win >= 0
    assert cert.measured_thermal == 100  # genuine hot telemetry folded in


def test_legit_telemetry_passes_sanitize_unchanged():
    """ANTI-REGRESSION: `sanitize_events` is the identity on a legitimate batch (same
    objects, same order, no rejections) -- legitimate telemetry is never altered."""
    legit = _cool(thermal=50)
    out, witness = sanitize_events(legit)
    assert out == legit and all(a is b for a, b in zip(out, legit))
    assert witness.rejected == 0 and witness.accepted == 3 and witness.clean
