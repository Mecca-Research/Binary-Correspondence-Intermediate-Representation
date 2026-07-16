"""Physics-anchored calibration tests: measure -> quantize -> freeze -> tag."""

import json
import os
import shutil
import subprocess
import tempfile

from bcir.examples import histogram_gather, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.microbench import (
    CalibratedProfile,
    MicrobenchRaw,
    calibrate_from_raw,
    reference_table,
    run_microbench,
)
from bcir.verify import verify_plan

AVX = TargetProfile.x86_avx512()


def test_quantizer_is_pure_and_pinned():
    # Synthetic raw timings: random 32x streaming, strided 1x. The quantizer is
    # deterministic and reproduces the seeded constants at exactly these ratios.
    raw = MicrobenchRaw(stream_ns=100, strided_ns=100, random_ns=3200,
                        compute_ns=100, n=1 << 10, repeats=5)
    t = calibrate_from_raw(raw, AVX, cal_gen=3)
    assert (t.stream_q8, t.strided_q8, t.random_q8) == (256, 256, 8192)
    assert t.gather_penalty == 32 == AVX.gather_penalty
    assert t.base_overhead == 4 == AVX.base_overhead
    assert t.mem_unit == 1 == AVX.mem_unit
    assert t.cal_gen == 3


def test_ratios_floor_at_the_streaming_baseline():
    # No regime is ever cheaper than streaming: ratios clamp at Q8 1.0.
    raw = MicrobenchRaw(stream_ns=100, strided_ns=10, random_ns=10, compute_ns=10)
    t = calibrate_from_raw(raw, AVX)
    assert t.strided_q8 == t.random_q8 == t.compute_q8 == 256
    assert t.gather_penalty == 1 and t.base_overhead == 4


def test_reference_table_is_the_degenerate_case():
    # The checked-in ratio-1 table reproduces the seeded constants, so the
    # pinned worked example survives table application unchanged: vec16 @ 7808.
    t = reference_table()
    assert t.cal_gen == 1 and t.stream_q8 == 256
    h = t.apply(AVX)
    assert (h.gather_penalty, h.base_overhead, h.mem_unit) == (32, 4, 1)
    assert h.cal_gen == 1
    r = optimize(vector_add(1024), h, Theta.cool())
    assert r.score == 7808 and r.by_claim()[1000].width == 16
    assert verify_plan(vector_add(1024), r) == []


def test_apply_is_idempotent_and_data_only():
    t = reference_table()
    h1 = t.apply(AVX)
    h2 = t.apply(h1)
    assert h1 == h2                      # frozen data in, frozen data out
    assert AVX.cal_gen == 0              # the seeded profile is untouched


def test_table_json_round_trips_losslessly():
    t = reference_table()
    assert CalibratedProfile.from_json(t.to_json()) == t


def test_table_json_rejects_ambiguous_or_noncanonical_artifacts():
    table = reference_table()
    document = json.loads(table.to_json())
    bad_documents = []

    bad = dict(document)
    bad["stream_q8"] = 255
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["samples"] = True
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["random_q8"] = -1
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["unexpected"] = 1
    bad_documents.append(json.dumps(bad))
    bad_documents.append(table.to_json().replace('"cal_gen": 1',
                                                  '"cal_gen": 1, "cal_gen": 2'))
    bad_documents.append(json.dumps({**document, "provenance": "x" * ((1 << 20) + 1)}))

    for text in bad_documents:
        try:
            CalibratedProfile.from_json(text)
            assert False, "expected malformed calibrated profile to be rejected"
        except ValueError:
            pass


def test_calibrated_gather_penalty_reaches_the_planner():
    # A table measuring a 8x gather ratio lowers gather_penalty 32 -> 8: the
    # HAM-vs-flat decision in the planner sees the measured physics.
    raw = MicrobenchRaw(stream_ns=100, strided_ns=100, random_ns=800, compute_ns=100)
    h = calibrate_from_raw(raw, AVX).apply(AVX)
    assert h.gather_penalty == 8
    m = histogram_gather(1024)
    seeded = optimize(m, AVX, Theta.cool()).score
    measured = optimize(m, h, Theta.cool()).score
    assert measured < seeded             # cheaper gathers re-price the plan


def test_live_harness_smoke_invariants_only():
    # The live harness must run anywhere; we assert mechanism invariants, never
    # timing values (the frozen-table law keeps timing out of the test oracle).
    raw = run_microbench(n=1 << 12, repeats=3)
    assert min(raw.stream_ns, raw.strided_ns, raw.random_ns, raw.compute_ns) >= 1
    t = calibrate_from_raw(raw, AVX, cal_gen=2)
    assert t.stream_q8 == 256
    assert t.strided_q8 >= 256 and t.random_q8 >= 256 and t.compute_q8 >= 256
    assert t.gather_penalty >= 1 and t.cal_gen == 2
    assert isinstance(t.apply(AVX), TargetProfile)


def test_every_target_accepts_a_table():
    raw = MicrobenchRaw(stream_ns=100, strided_ns=200, random_ns=1600, compute_ns=100)
    for h in TARGETS.values():
        t = calibrate_from_raw(raw, h)
        hh = t.apply(h)
        assert hh.cal_gen == 1 and hh.gather_penalty == 16 and hh.base_overhead == 8


def test_native_microbench_rejects_overflowing_cli_sizes():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source = os.path.join(root, "runtime", "c", "bcir_microbench.c")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "microbench")
        build = subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", exe],
                               capture_output=True, text=True)
        assert build.returncode == 0, build.stderr
        for argv in (("18446744073709551615", "1", "1"), (str((1 << 24) + 1), "1", "1"),
                     ("16", "129", "1"), ("16", "1", "-1"), ("16", "-1", "1"),
                     ("16", "1", "1", "extra")):
            run = subprocess.run([exe, *argv], capture_output=True, text=True)
            assert run.returncode == 2, (argv, run.stdout, run.stderr)
