"""The calls a piece of Python makes: the one counter every call-count gate and harness uses.

`pstats` is the wrong instrument for a count. cProfile keeps one entry per code object, and
`pstats.Stats` re-keys the entries by their label -- (file, first line, name) -- so two functions
that share a label become one row, and the entry written second replaces the first. CPython 3.11
labels every dataclass-generated `__init__` ("<string>", 2, "__init__"), so
`pstats.Stats(profile).total_calls` kept one class's constructor calls and dropped every other
class's, and which class it kept depended on where the code objects happened to be allocated.
The 2026-10-06 before/after audit (AUDIT-0) found it as nondeterminism: `sched_eft@4` counted
34,368 calls in one process and 36,415 in the next, the difference one class's 2,048
constructions less the 1 that survived. `planner.calls` at scale 8 read 589,856 while the
planner made 655,393 -- the 65,537 calls lost were the very constructor calls its emission floor
is made of.

`total_calls` sums the entries cProfile kept, one per code object, before any label can merge
them. `profiled` is the whole measurement: the collector run first and paused across the call, so
a finalizer it happens to run inside the window is not counted as a call the code made.

Standard library only, and no `bcir` import: `tools/perf/ab_audit.py` loads this file by path
from its own checkout, so the before tree and the after tree of an A/B are counted by one
instrument -- the after tree's -- whatever the before tree's own counter was.
"""

from __future__ import annotations

import cProfile
import gc


def total_calls(profile: cProfile.Profile) -> int:
    """Every call `profile` recorded, counted per code object (builtins included) -- the number
    `pstats` reports as its total when no two functions share a label."""
    return sum(entry.callcount for entry in profile.getstats())


def profiled(fn, *args, **kwargs) -> tuple[int, object]:
    """(the calls one call of `fn` makes, its value). The collector is run first and paused for
    the call, and restored to the state it was found in."""
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    profile = cProfile.Profile()
    try:
        profile.enable()
        try:
            value = fn(*args, **kwargs)
        finally:
            profile.disable()
    finally:
        if was_enabled:
            gc.enable()
    return total_calls(profile), value
