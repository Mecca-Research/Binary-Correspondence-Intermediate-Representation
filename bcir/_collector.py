"""The cyclic collector, paused inside the oracle's pure hot paths (OR-GC).

The 2026-10-06 audit (section 5.1) measured CPython's cyclic collector at up to 68% of a hot
path's time at 32,768 claims -- the planner, StreamPack hydrate and decode, R8/R9 plan
verification, the schedulers and the plan's own path -- while each of those calls leaves no
cyclic garbage at all: reference counting already frees everything they allocate, so each
collection inside them walks the whole live heap and frees nothing. A full collection's cost
grows with that heap, which is why the share grows with the module.

`paused` is the decorator those entry points wear:

* it pauses only a collector that is running, and resumes it on every way out, an exception
  included; a caller that turned the collector off keeps it off, and a nested entry point
  (hydrate placing its schedule) leaves the pause to the outermost one;
* it changes no result -- the collector frees only cyclic garbage, which these paths do not
  make. `bcir/tests/test_collector.py` holds that per path, as the zero-cyclic-garbage witness
  the audit asked for, and the before/after audit holds every output byte-identical;
* the collector's state is the process's. A thread that enters a paused path while another
  thread is inside one runs paused too, and may run with the collector back on if the other
  leaves first. Either way only the timing of a collection moves, never a result.
"""

from __future__ import annotations

import functools
import gc
from typing import Callable, TypeVar

_F = TypeVar("_F", bound=Callable)


def paused(fn: _F) -> _F:
    """`fn` with the cyclic collector paused for the length of the call (module docstring)."""

    @functools.wraps(fn)
    def run(*args, **kwargs):
        if not gc.isenabled():
            return fn(*args, **kwargs)
        gc.disable()
        try:
            return fn(*args, **kwargs)
        finally:
            gc.enable()

    return run  # type: ignore[return-value]


__all__ = ["paused"]
