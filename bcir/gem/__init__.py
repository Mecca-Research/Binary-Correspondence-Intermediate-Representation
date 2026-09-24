"""GEM (BCIR-4): hydrate a selected K_BCIR plan into a StreamPack, then execute it.

Imports are **lazy** (PEP 562): hydrating a plan into a StreamPack (`streampack`)
does not eagerly load the execution / scheduling machinery (`execute`,
`concurrency`, `overlap`, `schedule`, `async_tokens`). The C-kernel emit path needs
only `hydrate`; the executor resolves on first access. `from bcir.gem import
hydrate` and `from bcir.gem import execute` both still work.
"""

from __future__ import annotations

import importlib
import importlib.util

_EXPORTS: dict[str, tuple[str, ...]] = {
    "streampack": (
        "Block",
        "LaneSegment",
        "Prefetch",
        "StreamPack",
        "Generation",
        "generation_vector",
        "TraceNote",
        "hydrate",
        "hydrate_pipelined",
    ),
    "execution_plan": (
        "COHERENCE_ACTIONS",
        "ExecutionPlan",
        "LIVENESS_DOMAINS",
        "Lifetime",
        "MOVE_KINDS",
        "MovementEdge",
        "PLAN_MODES",
        "PlanStep",
        "plan_from_realization",
        "realization_of",
        "schedule_of",
    ),
    "control": (
        "Activate",
        "Cancel",
        "ControlOutcome",
        "ControlPlane",
        "ControlRecord",
        "GenerationSwitch",
        "LeaseGrant",
        "Quiesce",
        "Rollback",
        "is_stale",
        "lease_key",
        "registry_digest",
    ),
    "ring": (
        "RingAccounting",
        "RingConsumer",
        "RingError",
        "RingGeometry",
        "RingOutcome",
        "RingProducer",
        "committed_accounting",
        "decode_geometry",
        "encode_geometry",
        "format_ring",
        "ring_fingerprint",
        "ring_state",
        "validate_geometry",
    ),
    "handoff": (
        "FreezeError",
        "GraphClaim",
        "GraphGeneration",
        "Handle",
        "HandoffOutcome",
        "PackTable",
        "View",
        "admit_manifest",
        "freeze_claims",
    ),
    "execute": ("ExecResult", "PhaseStat", "execute"),
    "concurrency": (
        "ConcurrentSchedule",
        "hazard_conflict",
        "hazard_predecessors",
        "is_fence",
        "schedule_concurrent",
    ),
    "overlap": ("ScheduledPrice", "optimize_scheduled", "price_scheduled", "price_waves_legacy"),
    "schedule": (
        "EftPlacer",
        "GemSchedule",
        "SCHEDULE_MODES",
        "Slot",
        "TAIL_STREAM",
        "bandwidth_knee",
        "durations_from",
        "stream_geometry",
        "execute_tokens",
        "phase_hazards",
        "schedule_eft",
        "schedule_plan",
        "PowerRail",
        "PowerRailDecision",
        "schedule_power_rail",
    ),
    "async_tokens": ("AsyncPlan", "async_plan"),
    "exact": (
        "DEFAULT_EXACT_BUDGET",
        "Bound",
        "ExactSchedule",
        "ExactSelection",
        "MeasuredCertificate",
        "PhaseSolution",
        "ScheduleCertificate",
        "SearchState",
        "SelectionCertificate",
        "SelectionSearchState",
        "certify_measured",
        "certify_schedule",
        "certify_selection",
        "exact_schedule",
        "exact_selection",
    ),
    "dispatch": (
        "BOUNDED_SIZE",
        "CERTIFICATE_CLASSES",
        "REGION_KINDS",
        "SOLVERS",
        "CensusError",
        "DispatchDecision",
        "DispatchRecord",
        "DispatchRequest",
        "policy_ranking",
        "ranked",
        "solve_measured",
        "solve_memory",
        "solve_path",
        "solve_schedule",
        "solve_selection",
    ),
    "diff": ("Move", "PlanDiff", "Relayout", "Repricing", "Reselection", "plan_diff"),
    "cim": ("CIMDecision", "annotate_cim", "cim_decision"),
    "dvfs": (
        "DVFSDecision",
        "DVFSPlan",
        "FreqTarget",
        "ActuationResult",
        "classify",
        "clock_for",
        "phase_totals",
        "plan_dvfs",
        "quantize_to_silicon",
        "actuate",
    ),
}

_NAME_TO_MOD: dict[str, str] = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = sorted(_NAME_TO_MOD)

# `execute` is both an exported function and a submodule name. Bind the function
# eagerly so the `execute` submodule does not shadow it on later import (Python
# won't overwrite an existing parent attribute, but an unbound name would be).
# execute.py only depends on the model, so this stays cheap.
from .execute import execute  # noqa: E402


def __getattr__(name: str):
    """Resolve an exported name (or a submodule) lazily on first access (PEP 562)."""
    mod = _NAME_TO_MOD.get(name)
    if mod is not None:
        value = getattr(importlib.import_module(f".{mod}", __name__), name)
        globals()[name] = value
        return value
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        submodule = importlib.import_module(f".{name}", __name__)
        globals()[name] = submodule
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
