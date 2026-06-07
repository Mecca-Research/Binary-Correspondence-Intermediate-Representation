"""GEM: hydrate a selected K_BCIR plan into a verified StreamPack (BCIR-4)."""

from .streampack import (
    Block,
    LaneSegment,
    Prefetch,
    StreamPack,
    TraceNote,
    hydrate,
)

__all__ = ["Block", "LaneSegment", "Prefetch", "StreamPack", "TraceNote", "hydrate"]
