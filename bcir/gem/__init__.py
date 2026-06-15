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
    "streampack": ("Block", "LaneSegment", "Prefetch", "StreamPack", "TraceNote",
                   "hydrate", "hydrate_pipelined"),
    "execute": ("ExecResult", "PhaseStat", "execute"),
    "concurrency": ("ConcurrentSchedule", "schedule_concurrent"),
    "overlap": ("ScheduledPrice", "optimize_scheduled", "price_scheduled"),
    "schedule": ("GemSchedule", "Slot", "TAIL_STREAM", "bandwidth_knee", "durations_from",
                 "execute_tokens", "schedule_eft"),
    "async_tokens": ("AsyncPlan", "async_plan"),
    "cim": ("CIMDecision", "annotate_cim", "cim_decision"),
    "dvfs": ("DVFSDecision", "DVFSPlan", "classify", "clock_for", "phase_totals",
             "plan_dvfs"),
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
