"""Closing the calibration loop: measure -> freeze -> apply -> replan, certified.

The loop's value is the *measured* cost of not recalibrating (the replan win),
recorded in a generation-tagged certificate and witnessed by R13. A ratio-1 table
on a nominal machine wins nothing; hot telemetry flips vec16 -> vec8 and the win
is the heat/current the stale wide plan would have wasted.
"""

from dataclasses import replace

from bcir.examples import vector_add
from bcir.kbcir import (
    CalibrationCertificate,
    TARGETS,
    close_loop,
    measure_and_close,
    optimize,
    rescore_plan,
)
from bcir.kbcir.calibrate import EwmaCalibrator
from bcir.kbcir.cost import Theta
from bcir.kbcir.microbench import reference_table
from bcir.kbcir.weights import PERF
from bcir.telemetry import DataDNA
from bcir.verify import verify_calibration

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _laws(diags):
    return {d.law for d in diags}


def _hot_telemetry(n=4):
    # trust-recent telemetry (alpha 200/256) from a hot machine -> measured Theta
    # crosses the thermal-coupling threshold (>= 60), flipping wide SIMD.
    return [
        DataDNA(segment_id="s", claim_id=1000, thermal=100, utilization=90, voltage=70, misses=20)
        for _ in range(n)
    ]


# --- rescore is faithful (it reuses the optimizer's own primitives) --------------


def test_rescore_reproduces_the_optimizer_score():
    m = vector_add(1024)
    for theta in (COOL, Theta.hot()):
        r = optimize(m, AVX, theta, PERF)
        assert rescore_plan(m, r, AVX, theta, PERF) == r.score


# --- the closed loop -------------------------------------------------------------


def test_nominal_machine_wins_nothing():
    # ratio-1 reference table + no telemetry: recalibrating changes nothing, so the
    # win is correctly zero (no false value).
    cert = close_loop(vector_add(1024), AVX)
    assert not cert.replanned and cert.win == 0 and cert.admissible
    assert verify_calibration(cert) == []


def test_hot_telemetry_drives_a_measured_replan_win():
    cert = close_loop(
        vector_add(1024), AVX, events=_hot_telemetry(), calibrator=EwmaCalibrator(alpha=200)
    )
    assert cert.measured_thermal >= 60  # telemetry calibrated Theta hot
    assert cert.seeded_widths == ((1000, 16),)  # the stale plan: vec16
    assert cert.calibrated_widths == ((1000, 8),)  # the recalibrated plan: vec8
    assert cert.replanned and cert.win > 0  # a real, positive measured win
    assert verify_calibration(cert) == []


def test_win_is_the_stale_plan_overcost():
    # win == (stale vec16 rescored on the hot machine) - (vec8 optimum there).
    cert = close_loop(
        vector_add(1024), AVX, events=_hot_telemetry(), calibrator=EwmaCalibrator(alpha=200)
    )
    assert cert.win == cert.stale_cost - cert.calibrated_cost
    assert cert.stale_cost > cert.calibrated_cost


# --- on real hardware: measure this host, freeze, close the loop -----------------


def test_real_host_measurement_freezes_a_wellformed_table():
    # The "on real hardware" half: timings vary, so assert only host-independent
    # invariants of the frozen table (the deterministic artifact).
    cert, table = measure_and_close(vector_add(1024), AVX, n=1 << 14, repeats=3, cal_gen=2)
    assert table.cal_gen == 2 and table.samples >= 1
    assert table.stream_q8 == 256  # streaming baseline (R8)
    assert all(
        r >= 256 for r in (table.stream_q8, table.strided_q8, table.random_q8, table.compute_q8)
    )
    assert table.gather_penalty >= 1 and table.base_overhead >= 1
    assert cert.cal_gen == 2 and cert.admissible  # the loop closes soundly
    assert verify_calibration(cert) == []


# --- R13: the calibration-closure law --------------------------------------------


def test_untagged_table_is_R13():
    cert = close_loop(vector_add(1024), AVX)
    assert "R13" in _laws(verify_calibration(replace(cert, cal_gen=0)))


def test_a_regressed_loop_is_R13():
    # A certificate whose "recalibrated" plan is dearer than the stale one is a
    # broken/tampered loop (the optimum can never lose).
    cert = close_loop(vector_add(1024), AVX)
    bad = replace(cert, stale_cost=0, calibrated_cost=999)
    assert bad.win < 0 and "R13" in _laws(verify_calibration(bad))


# --- provenance: the frozen table chains into the manifest -----------------------


def test_certificate_json_round_trips():
    cert = close_loop(
        vector_add(1024), AVX, events=_hot_telemetry(), calibrator=EwmaCalibrator(alpha=200)
    )
    assert CalibrationCertificate.from_json(cert.to_json()) == cert


def test_reference_table_is_the_ratio1_anchor():
    # The checked-in reference reproduces the seeded constants exactly (gather
    # penalty 32, base overhead 4): closing the loop under it must not replan.
    t = reference_table()
    assert t.gather_penalty == 32 and t.base_overhead == 4
    assert not close_loop(vector_add(1024), AVX, table=t).replanned


# --- bare-metal calibration: real cache latency (closes the conservative half) ---


def test_native_microbench_measures_a_wellformed_table():
    from bcir.kbcir.microbench import calibrate_native, native_available

    if not native_available():
        return  # skip cleanly without a C compiler / the native source
    t = calibrate_native(AVX, n=1 << 20, repeats=3, cal_gen=2)
    assert t.cal_gen == 2 and t.name == AVX.name
    assert t.stream_q8 == 256  # streaming baseline (R8)
    assert all(r >= 256 for r in (t.stream_q8, t.strided_q8, t.random_q8, t.compute_q8))
    # bare metal sees real latency: gather is at least as dear as streaming.
    assert t.random_q8 >= t.stream_q8 and t.gather_penalty >= 1
    assert "bare-metal" in t.provenance


def test_native_loop_closes_admissibly():
    from bcir.kbcir.microbench import native_available

    if not native_available():
        return
    cert, table = measure_and_close(vector_add(1024), AVX, native=True, cal_gen=3)
    assert table.cal_gen == 3 and cert.admissible
    assert verify_calibration(cert) == []
