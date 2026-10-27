"""B1 measured schedule evidence and stable MLIR export.

An artifact records both the analytic K_BCIR choice and every bounded measured candidate.
Measurements may calibrate future cost constants, but they do not mutate legality or steer
an in-flight execution.  Activation requires a separately reviewed artifact generation.

Its counts are explicit (EVIDENCE-NULL). `bcir.schedule_artifact.v2`, the schema written, holds
for each candidate the PMU's three counts or three nulls, and getrusage's four sums or four nulls
on a host without getrusage; its constants are its measured winner's, derived by one function for
the writer and the reader, so a constant the PMU's counts would give is null exactly when they
are. `bcir.schedule_artifact.v1` had no null -- an unread count was 0, as was a constant from one
-- and still reads, to the same bytes and digest; `counts_of` reads its PMU counts as present
only when none is 0.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .._artifact_json import read_bounded_text, strict_json_loads
from ..silicon import CounterSampler, pmu_counts, read_hw_counters, recorded_counts, v1_pmu_counts
from .cost import TargetProfile
from .matmul import TilePlan, plan_matmul


SCHEMA = "bcir.schedule_artifact.v2"  # the schema written: counts or nulls
SCHEMA_V1 = "bcir.schedule_artifact.v1"  # read only: no null, an unread count recorded as 0
_WORKLOADS = frozenset({"gemm", "fused", "attention"})
_LOOP_ORDERS = frozenset({"ijk", "ikj", "jik"})
_RUSAGE_FIELDS = (
    "minor_faults",
    "major_faults",
    "voluntary_context_switches",
    "involuntary_context_switches",
)
_PMU_FIELDS = ("cycles", "instructions", "cache_misses")


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive(value, field: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _plan_dict(plan: TilePlan) -> dict:
    return asdict(plan)


def _plan_from_dict(value) -> TilePlan:
    if not isinstance(value, dict) or set(value) != {
        "tile_m",
        "tile_n",
        "tile_k",
        "loop_order",
        "compute_cost",
        "mem_cost",
        "bottleneck",
        "fits_cache",
    }:
        raise ValueError("schedule plan has missing or unknown fields")
    try:
        plan = TilePlan(**value)
    except TypeError as exc:
        raise ValueError(f"malformed schedule plan: {exc}") from exc
    for field in ("tile_m", "tile_n", "tile_k", "compute_cost", "mem_cost", "bottleneck"):
        _positive(
            getattr(plan, field),
            field,
            zero=field != "tile_m" and field != "tile_n" and field != "tile_k",
        )
    if (
        plan.loop_order not in _LOOP_ORDERS
        or type(plan.fits_cache) is not bool
        or plan.bottleneck != max(plan.compute_cost, plan.mem_cost)
    ):
        raise ValueError("schedule plan is internally inconsistent")
    return plan


def target_fingerprint(target: TargetProfile) -> str:
    if not isinstance(target, TargetProfile):
        raise ValueError("target must be a TargetProfile")
    memory = [
        {
            "name": tier.name,
            "latency_cyc": tier.latency_cyc,
            "bw_factor": tier.bw_factor,
            "lat_factor": tier.lat_factor,
            "capacity": tier.capacity,
        }
        for tier in target.mem.tiers
    ]
    value = {
        field: getattr(target, field)
        for field in (
            "name",
            "triple",
            "cacheline",
            "elem_bytes",
            "lane_widths",
            "warp",
            "scalable",
            "gather_penalty",
            "mem_unit",
            "base_overhead",
            "thermal_density",
            "power_density",
            "per_op_heat",
            "fma",
            "affinity_domains",
            "mem_channels",
            "cal_gen",
        )
    }
    value["lane_widths"] = list(value["lane_widths"])
    value["isa_features"] = sorted(target.isa_features)
    value["memory"] = memory
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MeasuredScheduleCandidate:
    """One tile plan's evidence: the median clocks and the summed getrusage counts over its
    repeats, and the PMU's counts for one more run. Each group of counts is all read or all
    None; whether its artifact's schema admits a None is the artifact's to check."""

    plan: TilePlan
    wall_ns: int
    cpu_ns: int
    minor_faults: int | None
    major_faults: int | None
    voluntary_context_switches: int | None
    involuntary_context_switches: int | None
    cycles: int | None
    instructions: int | None
    cache_misses: int | None
    repeats: int

    def __post_init__(self) -> None:
        _plan_from_dict(_plan_dict(self.plan))
        for field in ("wall_ns", "cpu_ns"):
            _positive(getattr(self, field), field, zero=True)
        for fields in (_RUSAGE_FIELDS, _PMU_FIELDS):
            values = tuple(getattr(self, field) for field in fields)
            if None in values:
                recorded_counts(*values)  # a group is all null or refused
            else:
                for field in fields:
                    _positive(getattr(self, field), field, zero=True)
        _positive(self.repeats, "repeats")
        if self.wall_ns == 0:
            raise ValueError("measured wall time must be positive")

    @property
    def rusage(self) -> tuple[int | None, ...]:
        return tuple(getattr(self, field) for field in _RUSAGE_FIELDS)

    @property
    def pmu(self) -> tuple[int | None, ...]:
        return tuple(getattr(self, field) for field in _PMU_FIELDS)

    def to_dict(self) -> dict:
        value = asdict(self)
        value["plan"] = _plan_dict(self.plan)
        return value


@dataclass(frozen=True)
class MeasuredCostConstants:
    """The cost constants of the measured winner, Q20. The two from the PMU's counts are None in
    a v2 artifact whose winner has none (cycles per instruction also when it retired no
    instructions); v1 recorded both as 0."""

    ns_per_mac_q20: int
    cycles_per_instruction_q20: int | None
    cache_misses_per_kmac_q20: int | None

    def __post_init__(self) -> None:
        _positive(self.ns_per_mac_q20, "ns_per_mac_q20", zero=True)
        for field in ("cycles_per_instruction_q20", "cache_misses_per_kmac_q20"):
            if getattr(self, field) is not None:
                _positive(getattr(self, field), field, zero=True)


def _constants(winner: MeasuredScheduleCandidate, macs: int) -> MeasuredCostConstants:
    """A v2 artifact's constants: its measured winner's, by the one derivation the writer and
    the reader share. Without the PMU's counts the two constants they would give are None, and
    cycles per instruction is None also when the winner retired no instructions; a count of 0 is
    a reading, so 0 cache misses give 0."""
    ns = (winner.cpu_ns << 20) // macs
    counts = recorded_counts(*winner.pmu)
    if counts is None:
        return MeasuredCostConstants(ns, None, None)
    cycles, instructions, cache_misses = counts
    return MeasuredCostConstants(
        ns,
        (cycles << 20) // instructions if instructions else None,
        (cache_misses * 1000 << 20) // macs,
    )


@dataclass(frozen=True)
class ScheduleArtifact:
    workload: str
    shape: tuple[int, int, int]
    target_name: str
    target_sha256: str
    host_architecture: str
    analytic_plan: TilePlan
    measured_plan: TilePlan
    candidates: tuple[MeasuredScheduleCandidate, ...]
    constants: MeasuredCostConstants
    source_commit: str
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in (SCHEMA, SCHEMA_V1) or self.workload not in _WORKLOADS:
            raise ValueError("unsupported schedule artifact schema or workload")
        if not isinstance(self.shape, (tuple, list)) or len(self.shape) != 3:
            raise ValueError("schedule shape must contain M, N, K")
        for name, value in zip(("M", "N", "K"), self.shape):
            _positive(value, name)
        object.__setattr__(self, "shape", tuple(self.shape))
        if (
            not isinstance(self.target_name, str)
            or not self.target_name
            or not isinstance(self.host_architecture, str)
            or not self.host_architecture
        ):
            raise ValueError("schedule target identity must be nonempty")
        _sha256(self.target_sha256, "target_sha256")
        _plan_from_dict(_plan_dict(self.analytic_plan))
        _plan_from_dict(_plan_dict(self.measured_plan))
        if (
            not isinstance(self.candidates, (tuple, list))
            or not self.candidates
            or any(
                not isinstance(candidate, MeasuredScheduleCandidate)
                for candidate in self.candidates
            )
        ):
            raise ValueError("schedule candidates must be nonempty measured records")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.measured_plan not in tuple(candidate.plan for candidate in self.candidates):
            raise ValueError("measured schedule winner is absent from candidate evidence")
        if not isinstance(self.constants, MeasuredCostConstants):
            raise ValueError("schedule constants have an invalid type")
        if (
            not isinstance(self.source_commit, str)
            or not 7 <= len(self.source_commit) <= 64
            or self.source_commit != self.source_commit.lower()
            or any(character not in "0123456789abcdef" for character in self.source_commit)
        ):
            raise ValueError("schedule source_commit must be a hexadecimal commit ID")
        if self.schema == SCHEMA_V1:
            nulls = [None in (*candidate.rusage, *candidate.pmu) for candidate in self.candidates]
            constants = (
                self.constants.cycles_per_instruction_q20,
                self.constants.cache_misses_per_kmac_q20,
            )
            if any(nulls) or None in constants:
                raise ValueError("a v1 schedule artifact has no null")
        else:
            self._check_v2()

    def _check_v2(self) -> None:
        """v2: each candidate's counts within the shared rule, each plan measured once, and the
        constants the measured winner's, derived as the writer derives them."""
        for candidate in self.candidates:
            recorded_counts(*candidate.rusage)
            recorded_counts(*candidate.pmu)
        plans = tuple(candidate.plan for candidate in self.candidates)
        if len(set(plans)) != len(plans):
            raise ValueError("a schedule artifact measures each candidate plan once")
        winner = self.candidates[plans.index(self.measured_plan)]
        M, N, K = self.shape
        if self.constants != _constants(winner, M * N * K):
            raise ValueError("schedule constants are not the measured winner's")

    def counts_of(self, candidate: MeasuredScheduleCandidate) -> tuple[int, int, int] | None:
        """A candidate's PMU counts as this artifact records them, or None when the PMU read
        none. A v1 artifact has no null and recorded an unread count as 0, so a v1 candidate's
        counts are the three only when none is 0 (`silicon.v1_pmu_counts`)."""
        if candidate not in self.candidates:
            raise ValueError("the candidate is not this artifact's")
        if self.schema == SCHEMA_V1:
            return v1_pmu_counts(*candidate.pmu)
        return recorded_counts(*candidate.pmu)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "workload": self.workload,
            "shape": list(self.shape),
            "target_name": self.target_name,
            "target_sha256": self.target_sha256,
            "host_architecture": self.host_architecture,
            "analytic_plan": _plan_dict(self.analytic_plan),
            "measured_plan": _plan_dict(self.measured_plan),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "constants": asdict(self.constants),
            "source_commit": self.source_commit,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @property
    def analytic_regret_q20(self) -> int:
        measured = {candidate.plan: candidate.wall_ns for candidate in self.candidates}
        return (
            (measured[self.analytic_plan] << 20) // measured[self.measured_plan]
            if self.analytic_plan in measured
            else 0
        )


def measure_schedule_candidates(
    M: int,
    N: int,
    K: int,
    target: TargetProfile,
    candidates,
    work,
    *,
    workload: str,
    source_commit: str,
    repeats: int = 3,
) -> ScheduleArtifact:
    if workload not in _WORKLOADS or not isinstance(target, TargetProfile) or not callable(work):
        raise ValueError("measurement workload, target, or work callback is invalid")
    _positive(repeats, "repeats")
    plans = tuple(candidates)
    if (
        not plans
        or any(not isinstance(plan, TilePlan) for plan in plans)
        or len(set(plans)) != len(plans)
    ):
        raise ValueError("measurement candidates must be unique TilePlan values")
    analytic = plan_matmul(M, N, K, target)
    if analytic not in plans:
        raise ValueError("measurement candidate set must include the analytic plan")
    evidence = []
    for plan in plans:
        work(plan)  # one bounded warmup, excluded from evidence
        laps = []
        for _ in range(repeats):
            sampler = CounterSampler()
            work(plan)
            laps.append(sampler.lap())
        hardware = read_hw_counters(functools.partial(work, plan))
        # The PMU's counts when it read all three, else three nulls; getrusage's four sums
        # when every lap read them, else four nulls.
        counts = pmu_counts(hardware)
        rusage = [(lap.minor_faults, lap.major_faults, lap.vol_ctx, lap.invol_ctx) for lap in laps]
        sums = (
            (None, None, None, None)
            if any(None in row for row in rusage)
            else tuple(sum(column) for column in zip(*rusage))
        )
        evidence.append(
            MeasuredScheduleCandidate(
                plan,
                int(statistics.median(lap.wall_ns for lap in laps)),
                int(statistics.median(lap.cpu_ns for lap in laps)),
                *sums,
                *(counts if counts is not None else (None, None, None)),
                repeats,
            )
        )
    winner = min(
        evidence,
        key=lambda item: (
            item.wall_ns,
            item.cpu_ns,
            item.plan.loop_order,
            -item.plan.tile_m,
            -item.plan.tile_n,
            -item.plan.tile_k,
        ),
    )
    constants = _constants(winner, M * N * K)
    return ScheduleArtifact(
        workload,
        (M, N, K),
        target.name,
        target_fingerprint(target),
        platform.machine() or "unknown",
        analytic,
        winner.plan,
        tuple(evidence),
        constants,
        source_commit,
    )


def write_schedule_artifact(path: os.PathLike | str, artifact: ScheduleArtifact) -> None:
    if not isinstance(artifact, ScheduleArtifact):
        raise ValueError("artifact must be a ScheduleArtifact")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(artifact.to_json().encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_schedule_artifact(path: os.PathLike | str) -> ScheduleArtifact:
    value = strict_json_loads(
        read_bounded_text(str(path), "schedule artifact"), "schedule artifact"
    )
    expected = {
        "schema",
        "workload",
        "shape",
        "target_name",
        "target_sha256",
        "host_architecture",
        "analytic_plan",
        "measured_plan",
        "candidates",
        "constants",
        "source_commit",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or not isinstance(value["candidates"], list)
    ):
        raise ValueError("schedule artifact has missing or unknown fields")
    try:
        candidates = tuple(
            MeasuredScheduleCandidate(plan=_plan_from_dict(row.pop("plan")), **row)
            for row in value["candidates"]
            if isinstance(row, dict)
        )
        if len(candidates) != len(value["candidates"]):
            raise ValueError("malformed measured candidate")
        return ScheduleArtifact(
            workload=value["workload"],
            shape=tuple(value["shape"]),
            target_name=value["target_name"],
            target_sha256=value["target_sha256"],
            host_architecture=value["host_architecture"],
            analytic_plan=_plan_from_dict(value["analytic_plan"]),
            measured_plan=_plan_from_dict(value["measured_plan"]),
            candidates=candidates,
            constants=MeasuredCostConstants(**value["constants"]),
            source_commit=value["source_commit"],
            schema=value["schema"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed schedule artifact: {exc}") from exc


def export_matmul_buffer_mlir(
    artifact: ScheduleArtifact,
    *,
    use_measured: bool = True,
    function_name: str = "scheduled_matmul",
) -> str:
    if (
        not isinstance(artifact, ScheduleArtifact)
        or not isinstance(function_name, str)
        or not function_name.replace("_", "").isalnum()
    ):
        raise ValueError("artifact or MLIR function name is invalid")
    M, N, K = artifact.shape
    plan = artifact.measured_plan if use_measured else artifact.analytic_plan
    return (
        f"func.func @{function_name}(%A: memref<{M}x{K}xf32>, "
        f"%B: memref<{K}x{N}xf32>, %C: memref<{M}x{N}xf32>) {{\n"
        f"  bcir.gem.matmul_buffer tile({plan.tile_m}, {plan.tile_n}, {plan.tile_k}) "
        f'order "{plan.loop_order}"\n'
        f"    %A, %B, %C : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>, "
        f"memref<{M}x{N}xf32>\n  return\n}}\n"
    )
