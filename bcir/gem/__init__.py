"""GEM (BCIR-4): hydrate a selected K_BCIR plan into a StreamPack, then execute it."""

from .async_tokens import AsyncPlan, async_plan
from .concurrency import ConcurrentSchedule, schedule_concurrent
from .execute import ExecResult, PhaseStat, execute
from .streampack import (
    Block,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    hydrate,
)

__all__ = [
    "Block", "LaneSegment", "Prefetch", "StreamPack", "TraceNote", "hydrate",
    "ExecResult", "PhaseStat", "execute",
    "ConcurrentSchedule", "schedule_concurrent",
    "AsyncPlan", "async_plan",
]
