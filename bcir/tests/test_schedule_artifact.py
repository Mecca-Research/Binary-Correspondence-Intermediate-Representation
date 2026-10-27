"""B1 measured schedule artifact and MLIR-export gates."""

from __future__ import annotations

import tempfile
from pathlib import Path

from bcir.kbcir.cost import TargetProfile
from bcir.kbcir.matmul import (
    TilePlan,
    cost_of,
    matmul_reference,
    matmul_tiled,
    plan_matmul,
    tile_origins,
)
from bcir.kbcir.schedule_artifact import (
    export_matmul_buffer_mlir,
    measure_schedule_candidates,
    read_schedule_artifact,
    target_fingerprint,
    write_schedule_artifact,
)


def _plan(M, N, K, tm, tn, tk, order, target):
    compute, memory, fits = cost_of(M, N, K, tm, tn, tk, target)
    return TilePlan(tm, tn, tk, order, compute, memory, max(compute, memory), fits)


def test_tiled_matmul_obeys_declared_outer_loop_order():
    ijk = TilePlan(2, 2, 2, "ijk", 1, 1, 1, True)
    ikj = TilePlan(2, 2, 2, "ikj", 1, 1, 1, True)
    assert list(tile_origins(4, 4, 4, ijk))[:3] == [(0, 0, 0), (0, 0, 2), (0, 2, 0)]
    assert list(tile_origins(4, 4, 4, ikj))[:3] == [(0, 0, 0), (0, 2, 0), (0, 0, 2)]
    a = list(range(16))
    b = [index % 5 - 2 for index in range(16)]
    reference = matmul_reference(a, b, 4, 4, 4)
    assert matmul_tiled(a, b, 4, 4, 4, ijk) == reference
    assert matmul_tiled(a, b, 4, 4, 4, ikj) == reference


def test_measured_schedule_records_real_counters_roundtrips_and_exports_mlir():
    M = N = K = 4
    target = TargetProfile.for_host()
    analytic = plan_matmul(M, N, K, target)
    candidates = {
        analytic,
        _plan(M, N, K, 1, 1, 1, "ijk", target),
        _plan(M, N, K, 2, 2, 2, "jik", target),
    }
    a = [float(index % 7) for index in range(M * K)]
    b = [float(index % 5 - 2) for index in range(K * N)]
    reference = matmul_reference(a, b, M, N, K)

    def work(plan):
        assert matmul_tiled(a, b, M, N, K, plan) == reference

    artifact = measure_schedule_candidates(
        M,
        N,
        K,
        target,
        sorted(
            candidates, key=lambda plan: (plan.loop_order, plan.tile_m, plan.tile_n, plan.tile_k)
        ),
        work,
        workload="gemm",
        source_commit="4a5ae9a",
        repeats=2,
    )
    assert artifact.analytic_plan == analytic
    assert artifact.measured_plan in candidates
    assert all(candidate.wall_ns > 0 and candidate.cpu_ns >= 0 for candidate in artifact.candidates)
    assert artifact.target_sha256 == target_fingerprint(target)
    assert artifact.analytic_regret_q20 >= 1 << 20
    mlir = export_matmul_buffer_mlir(artifact)
    selected = artifact.measured_plan
    assert f"tile({selected.tile_m}, {selected.tile_n}, {selected.tile_k})" in mlir
    assert f'order "{selected.loop_order}"' in mlir

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "schedule.json"
        write_schedule_artifact(path, artifact)
        assert read_schedule_artifact(path) == artifact
        raw = path.read_text("utf-8")
        path.write_text(raw.replace('"workload":"gemm"', '"workload":"unknown"'), "utf-8")
        try:
            read_schedule_artifact(path)
            assert False, "unknown workload was accepted"
        except ValueError:
            pass


def test_schedule_artifacts_keep_workload_families_distinct():
    target = TargetProfile.x86_avx2()
    M = N = K = 2
    analytic = plan_matmul(M, N, K, target)
    for workload in ("gemm", "fused", "attention"):
        artifact = measure_schedule_candidates(
            M,
            N,
            K,
            target,
            [analytic],
            lambda _plan: sum(range(64)),
            workload=workload,
            source_commit="4a5ae9a",
            repeats=1,
        )
        assert artifact.workload == workload and artifact.candidates[0].wall_ns > 0


def _refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def _measured(sample, *, repeats=1, also=()):
    """A 4x4x4 artifact of the host's analytic plan (and the tile plans `also` names, as
    (tile_m, tile_n, tile_k, order)) whose PMU sample is `sample`."""
    from unittest import mock

    from bcir.kbcir import schedule_artifact as artifacts

    target = TargetProfile.for_host()
    plans = [plan_matmul(4, 4, 4, target), *(_plan(4, 4, 4, *tile, target) for tile in also)]
    with mock.patch.object(artifacts, "read_hw_counters", lambda work, s=sample: s):
        return measure_schedule_candidates(
            4, 4, 4, target, plans, lambda plan: None,
            workload="gemm", source_commit="4a5ae9a", repeats=repeats,
        )  # fmt: skip


def test_a_schedule_candidate_records_a_pmu_sample_missing_a_count_as_null():
    """Each measured candidate carries the PMU's counts only when the sample read all three
    (`silicon.pmu_counts`); a sample missing one is three nulls (EVIDENCE-NULL), and the two
    constants the counts would give are null with them."""
    from bcir.silicon import HwSample

    for sample, want in (
        (HwSample(cycles=2_000, instructions=None, cache_misses=7), None),
        (HwSample(cycles=2_000, instructions=4_000, cache_misses=7), (2_000, 4_000, 7)),
    ):
        artifact = _measured(sample)
        (candidate,) = artifact.candidates
        assert artifact.schema == "bcir.schedule_artifact.v2"
        assert artifact.counts_of(candidate) == want
        assert (candidate.cycles, candidate.instructions, candidate.cache_misses) == (
            want or (None, None, None)
        )
        constants = artifact.constants
        pair = (constants.cycles_per_instruction_q20, constants.cache_misses_per_kmac_q20)
        assert pair == (((2_000 << 20) // 4_000, (7 * 1000 << 20) // 64) if want else (None, None))


def test_a_v2_artifacts_constants_are_its_winners_null_exactly_where_its_counts_are():
    """One derivation for the writer and the reader: a constant from the PMU's counts is null
    exactly when the winner has none (cycles per instruction also when it retired no
    instructions), and a 0 is a reading -- zero misses give 0. An artifact whose constants are
    not its winner's (v1's zeros where the counts are null, a number one off, a null where the
    counts are read) is refused, as is a plan measured twice or a group of counts with a gap."""
    from dataclasses import replace

    from bcir.kbcir.schedule_artifact import MeasuredCostConstants
    from bcir.silicon import HwSample

    retired_none = _measured(HwSample(cycles=2_000, instructions=0, cache_misses=0)).constants
    assert retired_none.cycles_per_instruction_q20 is None
    assert retired_none.cache_misses_per_kmac_q20 == 0
    nulls = _measured(None)
    ns = nulls.constants.ns_per_mac_q20
    assert nulls.constants == MeasuredCostConstants(ns, None, None)
    for forged in (
        MeasuredCostConstants(ns, 0, 0),
        MeasuredCostConstants(ns, None, 0),
        MeasuredCostConstants(ns + 1, None, None),
    ):
        assert _refused(replace, nulls, constants=forged), forged
    counted = _measured(HwSample(cycles=2_000, instructions=4_000, cache_misses=7))
    c = counted.constants
    cpi = c.cycles_per_instruction_q20
    assert _refused(replace, counted, constants=replace(c, cycles_per_instruction_q20=cpi + 1))
    assert _refused(replace, counted, constants=replace(c, cache_misses_per_kmac_q20=None))
    (candidate,) = counted.candidates
    assert _refused(replace, counted, candidates=(candidate, candidate))
    assert _refused(replace, candidate, instructions=None)
    gap = {"major_faults": 0, "voluntary_context_switches": 0, "involuntary_context_switches": 0}
    assert _refused(replace, candidate, minor_faults=None, **gap)
    # the shared bound holds for every candidate, not only the winner whose constants re-read it
    two = _measured(
        HwSample(cycles=2_000, instructions=4_000, cache_misses=7), also=[(1, 1, 1, "ijk")]
    )
    (loser,) = [c for c in two.candidates if c.plan != two.measured_plan]
    for past in ({"cycles": 1 << 63}, {"minor_faults": 1 << 63, **gap}):
        over = tuple(replace(c, **past) if c is loser else c for c in two.candidates)
        assert _refused(replace, two, candidates=over), past


def test_without_getrusage_a_candidate_records_four_nulls():
    """On a host without getrusage a candidate's summed faults and context switches are null,
    never 0, and the artifact round-trips them."""
    from bcir import silicon

    saved = silicon._resource
    silicon._resource = None
    try:
        artifact = _measured(None, repeats=2)
    finally:
        silicon._resource = saved
    (candidate,) = artifact.candidates
    assert candidate.rusage == (None, None, None, None) and candidate.wall_ns > 0
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "schedule.json"
        write_schedule_artifact(path, artifact)
        assert '"minor_faults":null' in path.read_text("utf-8")
        assert read_schedule_artifact(path) == artifact


# The bytes of a v1 artifact as the v1 writer wrote them (no PMU, so its zeros), and their digest.
_PLAN = (
    '{"bottleneck":4,"compute_cost":1,"fits_cache":true,"loop_order":"ijk","mem_cost":4,'
    '"tile_k":2,"tile_m":2,"tile_n":2}'
)
_V1_ARTIFACT = (
    '{"analytic_plan":' + _PLAN + ',"candidates":[{"cache_misses":0,"cpu_ns":900,"cycles":0,'
    '"instructions":0,"involuntary_context_switches":3,"major_faults":0,"minor_faults":1,'
    '"plan":' + _PLAN + ',"repeats":1,"voluntary_context_switches":2,"wall_ns":1000}],'
    '"constants":{"cache_misses_per_kmac_q20":0,"cycles_per_instruction_q20":0,'
    '"ns_per_mac_q20":117964800},"host_architecture":"x86_64","measured_plan":' + _PLAN + ","
    '"schema":"bcir.schedule_artifact.v1","shape":[2,2,2],"source_commit":"5b45fdca",'
    '"target_name":"x86-64-avx2",'
    '"target_sha256":"0674103a99a4c2ec90e669f9ec61ee624f2d051c73aea1f7323ff1a9ca2c1e59",'
    '"workload":"gemm"}'
)
_V1_ARTIFACT_DIGEST = "927a434cdb98049ab30be4a7c283bfead7a96c3faa3091e7ccb045f1bd6e5f6d"


def test_a_v1_artifact_reads_as_it_was_written_and_says_no_null():
    """v1 still reads, to its bytes and its digest; its zeros read as no counts; and it cannot
    say null -- not in a candidate's counts, its rusage or its constants."""
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "v1.json"
        path.write_text(_V1_ARTIFACT + "\n", "utf-8")
        artifact = read_schedule_artifact(path)
        assert (artifact.to_json(), artifact.digest) == (_V1_ARTIFACT, _V1_ARTIFACT_DIGEST)
        (candidate,) = artifact.candidates
        assert artifact.counts_of(candidate) is None
        assert _refused(artifact.counts_of, replace(candidate, repeats=2))  # not its candidate
        rusage_nulls = _V1_ARTIFACT
        for name, value in (
            ("minor_faults", 1),
            ("major_faults", 0),
            ("voluntary_context_switches", 2),
            ("involuntary_context_switches", 3),
        ):
            rusage_nulls = rusage_nulls.replace(f'"{name}":{value}', f'"{name}":null')
        pmu_nulls = _V1_ARTIFACT
        for name in ("cache_misses", "cycles", "instructions"):
            pmu_nulls = pmu_nulls.replace(f'"{name}":0,', f'"{name}":null,', 1)
        constant_null = _V1_ARTIFACT.replace(
            '"cycles_per_instruction_q20":0', '"cycles_per_instruction_q20":null'
        )
        for text in (rusage_nulls, pmu_nulls, constant_null):
            assert text != _V1_ARTIFACT
            path.write_text(text, "utf-8")
            assert _refused(read_schedule_artifact, path), text
            path.write_text(text.replace("schedule_artifact.v1", "schedule_artifact.v3"), "utf-8")
            assert _refused(read_schedule_artifact, path)
