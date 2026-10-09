"""Data-movement lower bounds: the red-blue pebble I/O bound and a hierarchical roofline (G25).

The 2026-10-06 audit (item 15, Partial) found the lower-bound stack complete for what the
schedule models -- critical path, work over capacity, the bandwidth knee, the tail's serial
work, the concurrent-live and region floors -- and empty for data movement: no communication
or I/O bound, no hierarchical roofline. This module is that member of the stack, at the scope
declared here and no wider.

**The game** (Hong and Kung 1981, the red-blue pebble game, at the granularity BCIR's movement
model uses: a resource's version is the whole resource). Two memory levels: a fast one of
`capacity` bytes and an unbounded slow one holding every resource at the start. A step (a
claim) runs only with every resource it reads or writes resident in fast memory. Bringing a
resource in costs its bytes (a load); a resource a step writes is dirty, and evicting a dirty
resource whose value is still needed -- read later, or live-out and not overwritten -- costs
its bytes (a store), as does storing every live-out dirty resource at the end. Q is the bytes
loaded plus the bytes stored, minimized over every eviction policy. `optimal_io` computes it
exactly by dynamic programming over the resident and dirty sets (small instances only: it is
the reference the bound is held to, not a planner).

**The bound** (`pebble_bound`), for an execution order, in three named parts:

* compulsory loads: every resource whose first access reads it holds a value only slow memory
  has, so it is loaded at least once;
* compulsory stores: every live-out resource the order writes produces its final value in fast
  memory, so it is stored at least once;
* capacity excess: after any step, the resources already materialized (loaded or written) whose
  next access reads them are live; fast memory holds at most `capacity` bytes of them, and each
  one it does not hold is loaded again later -- a load neither compulsory count includes (a
  value needed only to be stored at the end is not counted: evicting it costs a store, which
  may be the compulsory one). So
  Q >= compulsory + max over steps of (live - capacity), the largest excess at any one point
  (two points' excesses may name the same reload, so they are not summed).

A step whose own reads and writes exceed `capacity` cannot run at all; that is refused
(`IoError`), never priced.

**The hierarchical roofline** (`roofline_bound`): with levels of capacities C1, C2, ... (each
level holds what the faster ones do not -- exclusive), the traffic across the boundary below
level k is at least the pebble bound with capacity C1 + ... + Ck, and it moves at that
boundary's bandwidth; the compute moves at its peak rate. No execution in that order finishes
faster than the largest of the compute time and every boundary's traffic time. Exact
rationals throughout.

Not built, and declared: the communication cut between affinity domains, queue and network
calculus bounds, occupancy and energy at minimum work, and a bound over every execution order
(this one is for the order given -- the planner's).
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

#: The resources `optimal_io` enumerates subsets of; the reference is exponential in them.
OPTIMAL_IO_LIMIT = 10


class IoError(ValueError):
    """An order the game cannot run: a step whose working set exceeds the capacity, or a
    malformed size, capacity or step."""


@dataclass(frozen=True)
class IoBound:
    """The pebble bound for one order and capacity, its parts named, and the step after which
    the capacity excess is largest (-1 when there is none)."""

    compulsory_loads: int
    compulsory_stores: int
    capacity_excess: int
    at_step: int

    @property
    def total(self) -> int:
        return self.compulsory_loads + self.compulsory_stores + self.capacity_excess


def _check(steps, sizes, capacity, live_out):
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise IoError(f"the capacity is a non-negative integer, got {capacity!r}")
    for rid, size in sizes.items():
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise IoError(f"resource {rid} has no positive byte size")
    norm = []
    for i, step in enumerate(steps):
        reads, writes = (tuple(x) for x in step)
        for rid in (*reads, *writes):
            if rid not in sizes:
                raise IoError(f"step {i} names resource {rid}, which has no size")
        need = set(reads) | set(writes)
        if sum(sizes[r] for r in need) > capacity:
            raise IoError(
                f"step {i} needs {sum(sizes[r] for r in need)} bytes resident, over the capacity "
                f"{capacity}: it cannot run"
            )
        norm.append((frozenset(reads), frozenset(writes)))
    unknown = set(live_out) - set(sizes)
    if unknown:
        raise IoError(f"live-out resources {sorted(unknown)} have no size")
    return norm


def _needed(steps, live_out):
    """needed[i][r]: whether r's value after step i (i = -1: the initial value) is still
    needed -- its next access reads it, or it is live-out and nothing writes it again."""
    n = len(steps)
    resources = set(live_out)
    for reads, writes in steps:
        resources |= reads | writes
    needed = [dict() for _ in range(n + 1)]  # index i + 1 for "after step i"
    state = {r: r in live_out for r in resources}  # after the last step
    needed[n] = dict(state)
    for i in range(n - 1, -1, -1):
        reads, writes = steps[i]
        for r in writes:
            state[r] = False  # a value from before step i is overwritten here...
        for r in reads:
            state[r] = True  # ...unless step i reads it first
        needed[i] = dict(state)
    return needed  # needed[i]: after step i - 1, i.e. before step i


def _read_later(steps):
    """read_later[i]: the resources whose value after step i - 1 is next READ (not merely
    live-out): only those are loaded again if evicted -- a live-out value needed only at the
    end is stored, never reloaded, and its store may be the compulsory one."""
    state: dict = {}
    out = [dict() for _ in range(len(steps) + 1)]
    for i in range(len(steps) - 1, -1, -1):
        reads, writes = steps[i]
        for r in writes:
            state[r] = False
        for r in reads:
            state[r] = True
        out[i] = dict(state)
    return out


def pebble_bound(steps, sizes: dict, capacity: int, live_out=()) -> IoBound:
    """The red-blue pebble lower bound on Q for `steps` -- (reads, writes) per step, in order
    -- over a fast memory of `capacity` bytes (module docstring)."""
    live_out = frozenset(live_out)
    steps = _check(steps, sizes, capacity, live_out)
    read_later = _read_later(steps)
    touched: set = set()
    written: set = set()
    loads = 0
    for reads, writes in steps:
        for r in sorted(reads):
            if r not in touched and r not in written:
                loads += sizes[r]  # its first access reads the value slow memory holds
        touched |= reads
        written |= writes
    stores = sum(sizes[r] for r in sorted(live_out & written))
    excess, at = 0, -1
    materialized: set = set()
    for i, (reads, writes) in enumerate(steps):
        materialized |= reads | writes
        live = sum(sizes[r] for r in materialized if read_later[i + 1].get(r, False))
        if live - capacity > excess:
            excess, at = live - capacity, i
    return IoBound(loads, stores, excess, at)


def optimal_io(steps, sizes: dict, capacity: int, live_out=()) -> int:
    """The minimum Q over every eviction policy, by dynamic programming over (resident set,
    dirty set) after each step -- exact, and exponential: refused over OPTIMAL_IO_LIMIT
    resources. The reference `pebble_bound` is held to."""
    live_out = frozenset(live_out)
    steps = _check(steps, sizes, capacity, live_out)
    resources = sorted(set(live_out).union(*(r | w for r, w in steps)) if steps else live_out)
    if len(resources) > OPTIMAL_IO_LIMIT:
        raise IoError(f"optimal_io enumerates at most {OPTIMAL_IO_LIMIT} resources")
    needed = _needed(steps, live_out)
    frontier = {(frozenset(), frozenset()): 0}
    for i, (reads, writes) in enumerate(steps):
        need = reads | writes
        later = needed[i]  # values needed before step i runs (after step i - 1)
        nxt: dict = {}
        for (resident, dirty), cost in frontier.items():
            keepable = sorted(resident - need)
            for mask in range(1 << len(keepable)):
                kept = {keepable[k] for k in range(len(keepable)) if mask >> k & 1}
                new_resident = frozenset(need | kept)
                if sum(sizes[r] for r in new_resident) > capacity:
                    continue
                evicted = resident - new_resident
                c = cost
                c += sum(sizes[r] for r in evicted if r in dirty and later.get(r, False))
                c += sum(sizes[r] for r in reads - resident)  # loads for what is read
                new_dirty = frozenset((dirty & new_resident) | writes)
                key = (new_resident, new_dirty)
                if key not in nxt or c < nxt[key]:
                    nxt[key] = c
        frontier = nxt
    final = needed[len(steps)]
    return min(
        cost + sum(sizes[r] for r in dirty if final.get(r, False))
        for (_resident, dirty), cost in frontier.items()
    )


def roofline_bound(steps, sizes: dict, work: int, peak, levels, live_out=()):
    """The hierarchical roofline for an order: `work` units of compute at `peak` units per tick,
    and `levels` -- (capacity bytes, bandwidth bytes per tick of the boundary below it), fastest
    first, exclusive. Returns (bound in ticks, per-term breakdown): the largest of work / peak
    and, per boundary, the pebble bound at the capacity above it over that boundary's
    bandwidth."""
    terms = [("compute", Fraction(work) / Fraction(peak))]
    above = 0
    for k, (capacity, bandwidth) in enumerate(levels):
        above += capacity
        traffic = pebble_bound(steps, sizes, above, live_out).total
        terms.append((f"level{k}", Fraction(traffic) / Fraction(bandwidth)))
    return max(t for _n, t in terms), tuple(terms)


__all__ = [
    "IoBound",
    "IoError",
    "OPTIMAL_IO_LIMIT",
    "optimal_io",
    "pebble_bound",
    "roofline_bound",
]
