"""GEM (BCIR-4): hydrate a selected K_BCIR plan into a StreamPack, then execute it."""

from .async_tokens import AsyncPlan, async_plan
from .concurrency import ConcurrentSchedule, schedule_concurrent
from .execute import ExecResult, PhaseStat, execute
from .overlap import ScheduledPrice, optimize_scheduled, price_scheduled
from .schedule import (
    GemSchedule,
    Slot,
    TAIL_STREAM,
    bandwidth_knee,
    durations_from,
    execute_tokens,
    schedule_eft,
)
from .streampack import (
    Block,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    hydrate,
    hydrate_pipelined,
)

__all__ = [
    "Block", "LaneSegment", "Prefetch", "StreamPack", "TraceNote",
    "hydrate", "hydrate_pipelined",
    "ExecResult", "PhaseStat", "execute",
    "ConcurrentSchedule", "schedule_concurrent",
    "ScheduledPrice", "optimize_scheduled", "price_scheduled",
    "GemSchedule", "Slot", "TAIL_STREAM", "bandwidth_knee", "durations_from",
    "execute_tokens", "schedule_eft",
    "AsyncPlan", "async_plan",
]
