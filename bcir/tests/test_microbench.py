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
    TENANCIES,
    CalibratedProfile,
    MicrobenchRaw,
    NativeEvidence,
    calibrate_from_raw,
    calibrate_native,
    host_attestation,
    reference_table,
    run_microbench,
    strided_order,
)
from bcir.verify import verify_plan

AVX = TargetProfile.x86_avx512()


def test_quantizer_is_pure_and_pinned():
    # Synthetic raw timings: random 32x streaming, strided 1x. The quantizer is
    # deterministic and reproduces the seeded constants at exactly these ratios.
    raw = MicrobenchRaw(
        stream_ns=100, strided_ns=100, random_ns=3200, compute_ns=100, n=1 << 10, repeats=5
    )
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
    assert h1 == h2  # frozen data in, frozen data out
    assert AVX.cal_gen == 0  # the seeded profile is untouched


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
    bad_documents.append(table.to_json().replace('"cal_gen": 1', '"cal_gen": 1, "cal_gen": 2'))
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
    assert measured < seeded  # cheaper gathers re-price the plan


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


def test_strided_order_is_a_full_cycle_permutation():
    # G7 (S0-F): `(k * stride) % n` visits n / gcd(n, stride) elements and repeats that
    # cycle -- n/16 of a power-of-two buffer at the default stride. The coset walk
    # visits every element exactly once at the declared stride, for any (n, stride).
    from math import gcd

    for n, stride in ((1 << 12, 16), (1000, 16), (97, 16), (1 << 10, 1 << 10), (2, 16), (3, 16)):
        order = strided_order(n, stride)
        assert sorted(order) == list(range(n)), (n, stride)
        parent = {(k * stride) % n for k in range(n)}
        assert len(parent) == n // gcd(n, stride)  # the defect, pinned as the witness
    assert len({(k * 16) % (1 << 22) for k in range(1 << 22)}) == (1 << 22) // 16
    # the walk keeps the stride: consecutive visits within a coset differ by `stride` mod n
    # (n = 4096, stride 16: 16 cosets of 256 elements each)
    order = strided_order(1 << 12, 16)
    assert all(
        (order[k + 1] - order[k]) % (1 << 12) == 16
        for k in range(len(order) - 1)
        if (k + 1) % 256 != 0
    )
    assert order[:3] == [0, 16, 32] and order[256:258] == [1, 17]
    for bad in ((0, 16), (16, 0)):
        try:
            strided_order(*bad)
            assert False, "expected a refusal"
        except ValueError:
            pass


def _native_rig(d, n, repeats):
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return None
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source = os.path.join(root, "runtime", "c", "bcir_microbench.c")
    exe = os.path.join(d, "microbench")
    build = subprocess.run(
        [cc, "-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-O2", source, "-o", exe],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    run = subprocess.run([exe, str(n), str(repeats), "1"], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return run.stdout


def test_native_rig_reports_census_samples_and_an_attested_tenancy():
    """G7 (S0-F): the rig's table is the summary of evidence it prints -- a counted
    census (every regime touches all n elements), one raw sample per repeat with the
    statistic re-derived from them, and a tenancy decided by the host's own signals.
    "bare-metal" appears in the provenance only when the evidence proves it."""
    with tempfile.TemporaryDirectory() as d:
        out = _native_rig(d, 4096, 3)
        if out is None:
            return
    table = CalibratedProfile.from_json(out)  # re-derives the ratios from the evidence
    ev = table.evidence
    assert isinstance(ev, NativeEvidence)
    assert (ev.n, ev.repeats, ev.stride) == (4096, 3, 16)
    assert ev.unique_stream == ev.unique_strided == ev.unique_random == 4096
    assert ev.working_set_bytes == 4096 * 8
    for regime in ("stream", "strided", "random", "compute"):
        samples = getattr(ev, f"{regime}_ns")
        assert len(samples) == 3 and all(x >= 1 for x in samples)
        lo, med, hi, mad = getattr(ev, f"{regime}_stat")
        assert lo <= med <= hi and med == sorted(samples)[1]
    assert (table.strided_q8, table.random_q8, table.compute_q8) == ev.q8_ratios()
    assert table.samples == 3
    assert ev.tenancy in TENANCIES
    assert table.provenance.startswith(f"native microbench ({ev.tenancy}: {ev.signals})")
    assert ("bare-metal" in table.provenance) == ev.silicon
    assert "unique=4096/4096/4096" in table.provenance
    assert ev.timer_quantum_ns >= 0 and ev.os and ev.arch and ev.compiler
    # the attestation is ONE predicate on both rails: the C rig and `host_attestation`
    # read the same files by the same rules, field for field (the aarch64 CI runner is a
    # DMI-attested VM that also exposes a PMU; a reader with fewer signals called it
    # bare metal).
    att = host_attestation()
    assert (ev.tenancy, ev.signals, ev.hardware_pmu, ev.pmu_source) == (
        att["tenancy"],
        att["signals"],
        att["hardware_pmu"],
        att["pmu_source"],
    ), (ev.tenancy, ev.signals, att)
    assert (ev.dmi_vendor, ev.dmi_product) == (att["dmi_vendor"], att["dmi_product"])
    assert ev.silicon == (att["tenancy"] == "bare-metal")
    assert CalibratedProfile.from_json(table.to_json()) == table


def test_host_attestation_reserves_bare_metal_for_the_proved_case():
    """The Python twin of the rig's rule over synthetic hosts: a hypervisor, DMI or WSL
    signal is virtualized whatever the PMU says; a container alone is containerized; a
    PMU with no signal is bare metal; nothing is unproven."""
    from unittest import mock

    import bcir.kbcir.microbench as mb

    def host(files, readable):
        def contains(path, needle):
            return needle in files.get(path, "").lower()

        def first_line(path):
            return (
                files.get(path, "unavailable").splitlines()[0] if path in files else "unavailable"
            )

        with (
            mock.patch.object(mb, "_file_contains", contains),
            mock.patch.object(mb, "_first_line", first_line),
            mock.patch.object(mb, "_readable", lambda p: p in readable),
        ):
            return mb.host_attestation()

    pmu = {"/sys/bus/event_source/devices/armv8_pmuv3_0/type"}
    vm = host(
        {
            "/sys/class/dmi/id/sys_vendor": "Microsoft Corporation",
            "/sys/class/dmi/id/product_name": "Virtual Machine",
        },
        pmu,
    )
    assert (vm["tenancy"], vm["signals"], vm["hardware_pmu"]) == (
        "virtualized",
        "dmi-virtual",
        True,
    )
    flagged = host({"/proc/cpuinfo": "flags : fpu vme hypervisor sse"}, set())
    assert (flagged["tenancy"], flagged["signals"]) == ("virtualized", "hypervisor-flag")
    wsl = host({"/proc/version": "Linux version 5.15 (Microsoft@Microsoft.com)"}, pmu)
    assert (wsl["tenancy"], wsl["signals"]) == ("virtualized", "wsl")
    boxed = host({"/proc/1/cgroup": "0::/docker/abc"}, pmu)
    assert (boxed["tenancy"], boxed["signals"]) == ("containerized", "container")
    metal = host({}, pmu | {"/sys/class/powercap/intel-rapl/enabled"})
    assert (metal["tenancy"], metal["signals"], metal["pmu_source"]) == (
        "bare-metal",
        "pmu=armv8_pmuv3_0",
        "armv8_pmuv3_0",
    )
    nothing = host({}, set())
    assert (nothing["tenancy"], nothing["signals"], nothing["hardware_pmu"]) == (
        "unproven",
        "no PMU exposed",
        False,
    )


def test_native_table_summary_must_agree_with_its_evidence():
    """A table whose Q8 ratios, sample count, statistics or tenancy claim disagree with
    the evidence it carries is refused: the summary a v1 reader sees is derived from
    the samples, never a second source of truth."""
    with tempfile.TemporaryDirectory() as d:
        out = _native_rig(d, 4096, 3)
        if out is None:
            return
    document = json.loads(out)
    good = CalibratedProfile.from_json(json.dumps(document))
    assert good.evidence is not None
    bad_documents = []
    bad = json.loads(json.dumps(document))
    bad["random_q8"] = bad["random_q8"] + 1  # a ratio the medians do not imply
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["samples"] = 4  # more samples than the evidence carries
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["evidence"]["strided_stat"][1] += 1  # a statistic that is not the samples'
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["evidence"]["stream_ns"] = bad["evidence"]["stream_ns"][:-1]  # a dropped sample
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["evidence"]["tenancy"] = "silicon"  # outside the closed set
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["evidence"]["tenancy"] = "bare-metal"  # a claim without a PMU, or without the
    bad["evidence"]["hardware_pmu"] = True  # provenance saying so
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["provenance"] = "native microbench (bare-metal) n=4096"  # the old unconditional claim
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    bad["evidence"]["working_set_bytes"] = 8  # not the census
    bad_documents.append(bad)
    bad = json.loads(json.dumps(document))
    del bad["evidence"]["clocksource"]  # a missing field
    bad_documents.append(bad)
    for doc in bad_documents:
        try:
            CalibratedProfile.from_json(json.dumps(doc))
            assert False, "expected the disagreeing table to be refused"
        except ValueError:
            pass


def test_calibrate_native_refuses_an_unproved_bare_metal_claim():
    """The old rig said "bare-metal" under WSL and under a hypervisor alike, and the
    calibration loop froze that string into a certificate. `require_baremetal=True`
    now refuses every tenancy the evidence did not prove, naming the signals."""
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("ggc")
    if cc is None:
        return
    table = calibrate_native(AVX, n=4096, repeats=2, cal_gen=2)
    assert table.evidence is not None and table.name == AVX.name and table.cal_gen == 2
    if table.evidence.silicon:
        strict = calibrate_native(AVX, n=4096, repeats=2, cal_gen=2, require_baremetal=True)
        assert "bare-metal" in strict.provenance
    else:
        try:
            calibrate_native(AVX, n=4096, repeats=2, cal_gen=2, require_baremetal=True)
            assert False, "expected the unproved tenancy to be refused"
        except RuntimeError as exc:
            assert table.evidence.tenancy in str(exc) and "refusing the claim" in str(exc)
        assert "bare-metal" not in table.provenance


def test_native_microbench_rejects_overflowing_cli_sizes():
    cc = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        return
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    source = os.path.join(root, "runtime", "c", "bcir_microbench.c")
    with tempfile.TemporaryDirectory() as d:
        exe = os.path.join(d, "microbench")
        build = subprocess.run(
            [cc, "-std=c11", "-Wall", "-Wextra", "-Werror", source, "-o", exe],
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        for argv in (
            ("18446744073709551615", "1", "1"),
            (str((1 << 24) + 1), "1", "1"),
            ("16", "129", "1"),
            ("16", "1", "-1"),
            ("16", "-1", "1"),
            ("16", "1", "1", "extra"),
        ):
            run = subprocess.run([exe, *argv], capture_output=True, text=True)
            assert run.returncode == 2, (argv, run.stdout, run.stderr)
