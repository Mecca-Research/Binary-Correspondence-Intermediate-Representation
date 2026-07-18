"""B1 measured schedule artifact and MLIR-export gates."""
from __future__ import annotations

import tempfile
from pathlib import Path

from bcir.kbcir.cost import TargetProfile
from bcir.kbcir.matmul import (TilePlan, cost_of, matmul_reference, matmul_tiled, plan_matmul,
                               tile_origins)
from bcir.kbcir.schedule_artifact import (export_matmul_buffer_mlir,
                                          measure_schedule_candidates,
                                          read_schedule_artifact, target_fingerprint,
                                          write_schedule_artifact)


def _plan(M, N, K, tm, tn, tk, order, target):
    compute, memory, fits = cost_of(M, N, K, tm, tn, tk, target)
    return TilePlan(tm, tn, tk, order, compute, memory, max(compute, memory), fits)


def test_tiled_matmul_obeys_declared_outer_loop_order():
    ijk = TilePlan(2, 2, 2, "ijk", 1, 1, 1, True)
    ikj = TilePlan(2, 2, 2, "ikj", 1, 1, 1, True)
    assert list(tile_origins(4, 4, 4, ijk))[:3] == [
        (0, 0, 0), (0, 0, 2), (0, 2, 0)]
    assert list(tile_origins(4, 4, 4, ikj))[:3] == [
        (0, 0, 0), (0, 2, 0), (0, 0, 2)]
    a = list(range(16)); b = [index % 5 - 2 for index in range(16)]
    reference = matmul_reference(a, b, 4, 4, 4)
    assert matmul_tiled(a, b, 4, 4, 4, ijk) == reference
    assert matmul_tiled(a, b, 4, 4, 4, ikj) == reference


def test_measured_schedule_records_real_counters_roundtrips_and_exports_mlir():
    M = N = K = 4; target = TargetProfile.for_host()
    analytic = plan_matmul(M, N, K, target)
    candidates = {analytic,
                  _plan(M, N, K, 1, 1, 1, "ijk", target),
                  _plan(M, N, K, 2, 2, 2, "jik", target)}
    a = [float(index % 7) for index in range(M * K)]
    b = [float(index % 5 - 2) for index in range(K * N)]
    reference = matmul_reference(a, b, M, N, K)

    def work(plan):
        assert matmul_tiled(a, b, M, N, K, plan) == reference

    artifact = measure_schedule_candidates(
        M, N, K, target, sorted(candidates, key=lambda plan: (
            plan.loop_order, plan.tile_m, plan.tile_n, plan.tile_k)), work,
        workload="gemm", source_commit="4a5ae9a", repeats=2)
    assert artifact.analytic_plan == analytic
    assert artifact.measured_plan in candidates
    assert all(candidate.wall_ns > 0 and candidate.cpu_ns >= 0
               for candidate in artifact.candidates)
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
            read_schedule_artifact(path); assert False, "unknown workload was accepted"
        except ValueError:
            pass


def test_schedule_artifacts_keep_workload_families_distinct():
    target = TargetProfile.x86_avx2(); M = N = K = 2
    analytic = plan_matmul(M, N, K, target)
    for workload in ("gemm", "fused", "attention"):
        artifact = measure_schedule_candidates(
            M, N, K, target, [analytic], lambda _plan: sum(range(64)),
            workload=workload, source_commit="4a5ae9a", repeats=1)
        assert artifact.workload == workload and artifact.candidates[0].wall_ns > 0
