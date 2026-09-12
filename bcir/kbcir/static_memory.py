"""Verified static tensor-address planning for BCIR claim graphs.

The planner assigns every touched resource a byte offset inside one declared hardware
bank. Resources with disjoint lifetimes may reuse the same address range;
simultaneously-live resources may not. This is a deterministic compiler artifact, not a
runtime allocator and not a learned legality decision.

Since G5 (staged plan S1-D) a plan names the LIVENESS DOMAIN its lifetimes live in:

* ``"phase"`` -- topological phase positions (`allocator.live_intervals`), the historical
  model, safe under the phase-barriered executor and unsafe under `execute_tokens`, which
  legally overlaps independent claims of later phases with earlier ones (the 2026-08-12
  report's section 6.5: two resources live in phases 0 and 1 share offset 0 and alias the
  moment the token schedule runs them concurrently);
* ``"schedule"`` -- the canonical schedule artifact's own time intervals
  (`schedule_intervals`: for each resource, from the start of the first slot that touches
  it to the finish of the last, half-open ticks), so a plan computed from a token
  placement cannot alias under it, and a plan is bound to the placement it was derived
  from by a digest.

The verifier holds either domain to its schedule: handed a schedule, it refuses a
phase-liveness plan the schedule does not refine (the two-phase alias fixture is
REJECTED), and it recomputes a schedule-liveness plan's intervals from the placement.

The LAYOUT is first-fit (the predictable fast path, byte-identical to the historical
planner) or the bounded exact solver (`exact_layout`: a complete branch-and-bound over
aligned offsets within the first-fit incumbent, budgeted in its own work units -- candidate
placements expanded -- never seconds). Every bank summary records the concurrent-live
lower bound, the achieved extent and therefore the gap, and the stop reason (`first-fit`,
`optimal`, `budget`); the verifier recomputes an exact layout under the plan's own budget
rather than trust its stop reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads
from ..model import Module
from .allocator import live_intervals
from .provenance import ModuleIdentity, digest_of, module_identity

_SCHEMA = "bcir.static_memory_plan.v2"
_MAX_JSON = 16 * 1024 * 1024
#: The liveness domains a plan's lifetimes can live in (see the module docstring).
LIVENESS_DOMAINS = ("phase", "schedule")
#: The layouts a plan can be computed with.
LAYOUTS = ("first-fit", "exact")
#: Why a bank's layout stopped where it did.
STOP_REASONS = ("first-fit", "optimal", "budget")
#: The exact solver's default budget, in its own work units (candidate placements
#: expanded). Host-portable by construction: two runs with equal budgets and inputs
#: produce identical layouts and stop reasons.
DEFAULT_EXACT_BUDGET = 200_000
# The model-plan lowerer admits at most 4096 layers.  A two-times margin covers
# its per-layer resources while keeping the deterministic reference allocator
# safely bounded on malformed or synthetic inputs.
_MAX_RESOURCES = 8192
_MAX_BANKS = 32
_MAX_U63 = (1 << 63) - 1


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _module_digest(module: Module, identity: "ModuleIdentity | None" = None) -> str:
    """The plan's module digest: SHA-256 over the R13 module hash. The planner reads the
    module's identity (computed once per revision, G3 / S1-B); a verifier handed that
    identity validates it against the module's content and otherwise recomputes."""
    return hashlib.sha256(str(digest_of(module, identity)).encode("ascii")).hexdigest()


def _integer(value, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_U63:
        raise ValueError(f"{field} must be an integer in [{minimum}, {_MAX_U63}]")
    return value


def _name(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{field} must be a bounded, control-free string")
    return value


def _digest(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _checked_add(left: int, right: int) -> int:
    if left < 0 or right < 0 or left > _MAX_U63 - right:
        raise ValueError("static memory accounting exceeds signed 64-bit range")
    return left + right


def _align(value: int, alignment: int) -> int:
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError("static memory alignment must be a positive power of two")
    return _checked_add(value, alignment - 1) & ~(alignment - 1)


def _release_interval(free: list[tuple[int, int]], start: int, end: int) -> None:
    """Insert and coalesce one half-open free address interval."""
    if end <= start:
        return
    import bisect

    index = bisect.bisect_left(free, (start, end))
    if index and free[index - 1][1] >= start:
        index -= 1
        start = min(start, free[index][0])
        end = max(end, free[index][1])
        free.pop(index)
    while index < len(free) and free[index][0] <= end:
        start = min(start, free[index][0])
        end = max(end, free[index][1])
        free.pop(index)
    free.insert(index, (start, end))


def _take_first_fit(free: list[tuple[int, int]], size: int, alignment: int) -> int | None:
    """Consume the lowest aligned free span that fits, preserving its fragments."""
    for index, (start, end) in enumerate(free):
        offset = _align(start, alignment)
        allocation_end = _checked_add(offset, size)
        if allocation_end > end:
            continue
        replacement: list[tuple[int, int]] = []
        if start < offset:
            replacement.append((start, offset))
        if allocation_end < end:
            replacement.append((allocation_end, end))
        free[index : index + 1] = replacement
        return offset
    return None


@dataclass(frozen=True)
class ResourceBankBinding:
    rid: int
    bank: str

    def __post_init__(self) -> None:
        _integer(self.rid, "binding rid", minimum=1)
        _name(self.bank, "binding bank")


@dataclass(frozen=True)
class StaticAllocation:
    """One resource's address and lifetime. `first_phase`/`last_phase` is the declared phase
    span (closed); `first_tick`/`last_tick` is the HALF-OPEN liveness interval in the plan's
    liveness domain -- the phase positions (`last_phase + 1`) under phase liveness, the
    schedule's ticks under schedule liveness -- the interval the alias law is judged on."""

    rid: int
    bank: str
    offset: int
    size_bytes: int
    alignment: int
    first_phase: int
    last_phase: int
    first_tick: int
    last_tick: int

    def __post_init__(self) -> None:
        _integer(self.rid, "allocation rid", minimum=1)
        _name(self.bank, "allocation bank")
        for field in (
            "offset",
            "size_bytes",
            "alignment",
            "first_phase",
            "last_phase",
            "first_tick",
            "last_tick",
        ):
            _integer(
                getattr(self, field),
                f"allocation {field}",
                minimum=1 if field in ("size_bytes", "alignment") else 0,
            )
        if self.alignment & (self.alignment - 1):
            raise ValueError("allocation alignment must be a power of two")
        if self.offset % self.alignment:
            raise ValueError("allocation offset violates alignment")
        if self.last_phase < self.first_phase:
            raise ValueError("allocation lifetime is reversed")
        if self.last_tick <= self.first_tick:
            raise ValueError("allocation liveness interval is empty or reversed")
        _checked_add(self.offset, self.size_bytes)

    @property
    def end(self) -> int:
        return self.offset + self.size_bytes

    def overlaps(self, other: "StaticAllocation") -> bool:
        """Live at the same time (half-open ticks) AND at overlapping addresses."""
        return (
            self.first_tick < other.last_tick
            and other.first_tick < self.last_tick
            and self.offset < other.end
            and other.offset < self.end
        )


@dataclass(frozen=True)
class BankMemoryPlan:
    bank: str
    extent_bytes: int
    naive_bytes: int  # aligned sequential layout without lifetime reuse
    capacity_bytes: int
    #: The concurrent-live lower bound: the largest sum of sizes of resources live at one
    #: tick (the interval graph's heaviest clique). No layout can go below it; the gap to the
    #: achieved extent is the TMSAO optimality gap of this bank.
    lower_bound_bytes: int
    #: `first-fit` (the fast path), `optimal` (the exact solver proved the extent within its
    #: budget) or `budget` (the exact solver stopped on its budget; the incumbent stands).
    stop_reason: str

    def __post_init__(self) -> None:
        _name(self.bank, "bank plan name")
        for field in ("extent_bytes", "naive_bytes", "capacity_bytes", "lower_bound_bytes"):
            _integer(getattr(self, field), f"bank plan {field}")
        if self.extent_bytes > self.capacity_bytes:
            raise ValueError(f"static address extent exceeds bank {self.bank!r} capacity")
        if self.extent_bytes > self.naive_bytes:
            raise ValueError("static address plan cannot exceed its naive footprint")
        if self.lower_bound_bytes > self.extent_bytes:
            raise ValueError("static address plan cannot go below its concurrent-live bound")
        if self.stop_reason not in STOP_REASONS:
            raise ValueError(f"bank plan stop reason must be one of {STOP_REASONS}")

    @property
    def saved_bytes(self) -> int:
        return self.naive_bytes - self.extent_bytes

    @property
    def gap_bytes(self) -> int:
        """The distance from the achieved extent to the proved lower bound."""
        return self.extent_bytes - self.lower_bound_bytes


@dataclass(frozen=True)
class StaticMemoryPlan:
    module_digest: str
    hardware_digest: str
    allocations: tuple[StaticAllocation, ...]
    banks: tuple[BankMemoryPlan, ...]
    #: The liveness domain the lifetimes were computed in (`LIVENESS_DOMAINS`).
    liveness: str = "phase"
    #: The layout the offsets were computed with (`LAYOUTS`).
    layout: str = "first-fit"
    #: The exact solver's budget in work units (0 for first-fit); the verifier re-runs the
    #: solver under it, so it is part of the plan, not of the host.
    budget: int = 0
    #: SHA-256 of the placement a schedule-liveness plan was derived from
    #: (`schedule_digest`); empty under phase liveness.
    schedule_digest: str = ""
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported static memory plan schema")
        _digest(self.module_digest, "static plan module_digest")
        _digest(self.hardware_digest, "static plan hardware_digest")
        if self.liveness not in LIVENESS_DOMAINS:
            raise ValueError(f"static plan liveness must be one of {LIVENESS_DOMAINS}")
        if self.layout not in LAYOUTS:
            raise ValueError(f"static plan layout must be one of {LAYOUTS}")
        _integer(self.budget, "static plan budget")
        if (self.layout == "exact") != (self.budget > 0):
            raise ValueError("an exact static plan carries a positive budget; first-fit none")
        if self.liveness == "schedule":
            _digest(self.schedule_digest, "static plan schedule_digest")
        elif self.schedule_digest != "":
            raise ValueError("a phase-liveness static plan binds no schedule")
        if (
            not isinstance(self.allocations, tuple)
            or not self.allocations
            or len(self.allocations) > _MAX_RESOURCES
            or any(not isinstance(row, StaticAllocation) for row in self.allocations)
        ):
            raise ValueError("static memory plan needs a bounded allocation tuple")
        if (
            not isinstance(self.banks, tuple)
            or not self.banks
            or len(self.banks) > _MAX_BANKS
            or any(not isinstance(row, BankMemoryPlan) for row in self.banks)
        ):
            raise ValueError("static memory plan needs typed bank summaries")
        if len({row.rid for row in self.allocations}) != len(self.allocations):
            raise ValueError("static memory plan has duplicate resource allocations")
        if len({row.bank for row in self.banks}) != len(self.banks):
            raise ValueError("static memory plan has duplicate bank summaries")
        if tuple(sorted(self.allocations, key=lambda row: row.rid)) != self.allocations:
            raise ValueError("static memory allocations must be sorted by rid")
        if tuple(sorted(self.banks, key=lambda row: row.bank)) != self.banks:
            raise ValueError("static memory bank summaries must be sorted by name")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "module_digest": self.module_digest,
            "hardware_digest": self.hardware_digest,
            "liveness": self.liveness,
            "layout": self.layout,
            "budget": self.budget,
            "schedule_digest": self.schedule_digest,
            "allocations": [asdict(row) for row in self.allocations],
            "banks": [asdict(row) for row in self.banks],
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StaticMemoryPlan":
        doc = strict_json_loads(text, "static memory plan", max_bytes=_MAX_JSON)
        expected = {
            "schema",
            "module_digest",
            "hardware_digest",
            "liveness",
            "layout",
            "budget",
            "schedule_digest",
            "allocations",
            "banks",
        }
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("static memory plan has missing or unknown fields")
        if (
            not isinstance(doc["allocations"], list)
            or not doc["allocations"]
            or len(doc["allocations"]) > _MAX_RESOURCES
            or not isinstance(doc["banks"], list)
            or not doc["banks"]
            or len(doc["banks"]) > _MAX_BANKS
        ):
            raise ValueError("static memory plan arrays are empty, malformed, or unbounded")
        try:
            allocations = tuple(StaticAllocation(**row) for row in doc.pop("allocations"))
            banks = tuple(BankMemoryPlan(**row) for row in doc.pop("banks"))
            return cls(**doc, allocations=allocations, banks=banks)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed static memory plan: {exc}") from exc


def _bindings(value) -> dict[int, str]:
    if isinstance(value, dict):
        rows = tuple(ResourceBankBinding(rid, bank) for rid, bank in value.items())
    elif isinstance(value, (tuple, list)):
        rows = tuple(value)
        if any(not isinstance(row, ResourceBankBinding) for row in rows):
            raise ValueError("resource bank bindings must be typed rows")
    else:
        raise ValueError("resource bank bindings must be a mapping or sequence")
    if len(rows) > _MAX_RESOURCES or len({row.rid for row in rows}) != len(rows):
        raise ValueError("resource bank bindings are duplicate or unbounded")
    return {row.rid: row.bank for row in rows}


def _has_live_alias(rows: list[StaticAllocation], ticks=None) -> bool:
    """Detect overlapping address/liveness rectangles exactly, in O(n log n) comparisons.

    A sweep over the ticks: rows enter in first-tick order and leave when their last tick
    is reached (half-open, so a row dying at tick t is not live with one born at t). Every
    row in the live set is live at the current tick, so -- while no alias has been found --
    their address intervals are pairwise disjoint and sit in one sorted list; the new row
    aliases iff its neighbours by offset overlap it. The liveness interval is the row's
    ticks, or `ticks[rid] = (first, last)` when a caller judges the rows under another
    domain (the verifier holding a phase plan to a schedule)."""
    if len(rows) < 2:
        return False
    import bisect
    import heapq

    def interval(row: StaticAllocation) -> tuple[int, int]:
        return ticks[row.rid] if ticks is not None else (row.first_tick, row.last_tick)

    order = sorted(rows, key=lambda value: (interval(value)[0], value.rid))
    dying: list[tuple[int, int, int]] = []  # (last tick, serial, offset) of the live rows
    offsets: list[int] = []  # the live rows' offsets, sorted -- their address intervals
    ends: list[int] = []  # are pairwise disjoint while no alias has been found
    for serial, row in enumerate(order):
        first, last = interval(row)
        if last <= first:
            return True  # an empty liveness interval has no tick to be disjoint on
        while dying and dying[0][0] <= first:
            _last, _serial, offset = heapq.heappop(dying)
            at = bisect.bisect_left(offsets, offset)
            del offsets[at], ends[at]
        at = bisect.bisect_left(offsets, row.offset)
        if at and ends[at - 1] > row.offset:
            return True  # the live row just below reaches into this one
        if at < len(offsets) and offsets[at] < row.end:
            return True  # the live row at or above starts inside this one
        offsets.insert(at, row.offset)
        ends.insert(at, row.end)
        heapq.heappush(dying, (last, serial, row.offset))
    return False


# --- liveness domains -------------------------------------------------------------------


def _slots_of(schedule) -> tuple[str, int, dict[int, tuple[int, int, int]]]:
    """(mode, makespan, claim -> (stream, start, finish)) from a `gem.schedule.GemSchedule`
    or a `gem.execution_plan.ExecutionPlan` (duck-typed; the verifier stays dependency-free)."""
    if hasattr(schedule, "slots"):
        rows = {
            int(s.claim_id): (int(s.domain), int(s.start), int(s.finish)) for s in schedule.slots
        }
    elif hasattr(schedule, "steps"):
        rows = {
            int(s.claim_id): (int(s.stream), int(s.start), int(s.start) + int(s.duration))
            for s in schedule.steps
        }
    else:
        raise ValueError("a schedule is a GemSchedule (slots) or an ExecutionPlan (steps)")
    if len(rows) != len(getattr(schedule, "slots", None) or schedule.steps):
        raise ValueError("schedule places a claim twice")
    return str(schedule.mode), int(schedule.makespan), rows


def schedule_digest(schedule) -> str:
    """SHA-256 of the placement a schedule-liveness plan is bound to: its mode, makespan and
    every (claim, stream, start, finish) slot in claim order."""
    mode, makespan, rows = _slots_of(schedule)
    return _sha({"mode": mode, "makespan": makespan, "slots": sorted(rows.items())})


def schedule_intervals(module: Module, schedule) -> dict[int, tuple[int, int]]:
    """Each touched resource's HALF-OPEN liveness interval in the schedule's own ticks: from
    the start of the first slot that touches it to the finish of the last (a zero-length slot
    holds its resources for one tick, so nothing is ever live for no time). Every claim the
    module declares must be placed, and the schedule must place nothing else."""
    _mode, _makespan, slots = _slots_of(schedule)
    span: dict[int, tuple[int, int]] = {}
    declared: set[int] = set()
    for phase in module.phases:
        for claim in phase.claims:
            declared.add(claim.id)
            if claim.id not in slots:
                raise ValueError(f"schedule places no slot for claim {claim.id}")
            _stream, start, finish = slots[claim.id]
            finish = max(finish, start + 1)
            for rid in claim.io_rids():
                lo, hi = span.get(rid, (start, finish))
                span[rid] = (min(lo, start), max(hi, finish))
    stray = sorted(set(slots) - declared)
    if stray:
        raise ValueError(f"schedule places claims the module does not declare: {stray[:8]}")
    return span


def _phase_ticks(intervals: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Phase liveness as half-open ticks: [first_phase, last_phase + 1)."""
    return {rid: (lo, hi + 1) for rid, (lo, hi) in intervals.items()}


def _liveness(module: Module, schedule):
    """(domain, phase span per rid, half-open ticks per rid, schedule digest or '')."""
    spans = live_intervals(module)
    if schedule is None:
        return "phase", spans, _phase_ticks(spans), ""
    ticks = schedule_intervals(module, schedule)
    if set(ticks) != set(spans):
        raise ValueError("schedule liveness and phase liveness touch different resources")
    return "schedule", spans, ticks, schedule_digest(schedule)


# --- layouts -----------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutItem:
    """One resource to lay out: size, alignment and its half-open liveness ticks."""

    rid: int
    size: int
    alignment: int
    first_tick: int
    last_tick: int

    def conflicts(self, other: "LayoutItem") -> bool:
        return self.first_tick < other.last_tick and other.first_tick < self.last_tick


@dataclass(frozen=True)
class LayoutResult:
    """A bank's layout: offsets, the achieved extent, the proved lower bound, why the solver
    stopped (`first-fit`, `optimal`, `budget`) and the work it spent (candidate placements)."""

    offsets: dict[int, int]
    extent: int
    lower_bound: int
    stop_reason: str
    expansions: int

    @property
    def gap(self) -> int:
        return self.extent - self.lower_bound


def layout_lower_bound(items) -> int:
    """The concurrent-live lower bound: the heaviest set of items live at one tick. Every
    layout needs disjoint addresses for them, so no extent goes below their size sum. A
    sweep over the (tick, delta) events -- O(n log n), so the fast path stays fast at the
    audit's 2,048 resources."""
    events: list[tuple[int, int]] = []
    for item in items:
        events.append((item.first_tick, item.size))
        events.append((item.last_tick, -item.size))
    if not events:
        return 0
    # at one tick, releases (last_tick, half-open) come before takes: deltas sort ascending
    events.sort()
    live = best = 0
    for _tick, delta in events:
        live += delta
        if live > best:
            best = live
    return best


def _conflict_lists(rows: list[LayoutItem]) -> list[list[int]]:
    """For each item (in `rows` order, sorted by first tick), the earlier items it is live
    with -- built by a sweep with an active set, so the work is n x the clique size rather
    than n^2, and the exact search checks only the items that can block a placement."""
    import heapq

    conflicts: list[list[int]] = [[] for _ in rows]
    active: list[tuple[int, int]] = []  # (last_tick, index) of earlier items that may be live
    for index, item in enumerate(rows):
        while active and active[0][0] <= item.first_tick:
            heapq.heappop(active)
        # everything left started no later and is still live at item.first_tick; the heap is
        # not ordered by index, so sort for a deterministic search
        conflicts[index] = sorted(k for _last, k in active)
        heapq.heappush(active, (item.last_tick, index))
    return conflicts


def first_fit_layout(items) -> LayoutResult:
    """The predictable fast path: items in (first tick, rid) order, each at the lowest aligned
    free span, storage released when its holder dies. Byte-identical to the historical
    planner under phase liveness."""
    import heapq

    rows = sorted(items, key=lambda item: (item.first_tick, item.rid))
    active: list[tuple[int, int, int, int]] = []  # (last tick, rid, offset, end)
    free: list[tuple[int, int]] = [(0, _MAX_U63)]
    offsets: dict[int, int] = {}
    extent = 0
    for item in rows:
        while active and active[0][0] <= item.first_tick:
            _last, _rid, prior_offset, prior_end = heapq.heappop(active)
            _release_interval(free, prior_offset, prior_end)
        offset = _take_first_fit(free, item.size, item.alignment)
        if offset is None:  # pragma: no cover - the free list is unbounded above
            raise ValueError("first-fit layout exhausted the address space")
        offsets[item.rid] = offset
        end = _checked_add(offset, item.size)
        extent = max(extent, end)
        heapq.heappush(active, (item.last_tick, item.rid, offset, end))
    return LayoutResult(offsets, extent, layout_lower_bound(rows), "first-fit", 0)


def exact_layout(items, budget: int = DEFAULT_EXACT_BUDGET) -> LayoutResult:
    """The bounded exact layout: a complete branch-and-bound over every aligned offset below
    the incumbent extent, in (first tick, size descending, rid) order, pruned by the incumbent
    and stopped the moment the concurrent-live bound is met. The first-fit layout is the
    starting incumbent, so the result is never worse than the fast path. `budget` counts
    candidate placements expanded -- the solver's own work units, never seconds -- and an
    exhausted budget returns the incumbent with `stop_reason="budget"`: a stated gap, not a
    claimed optimum. Deterministic: equal inputs and budgets give identical layouts. The
    search keeps its own stack (one frame per item), so a bank of thousands of resources
    stops on its budget rather than on the interpreter's recursion limit."""
    rows = sorted(items, key=lambda item: (item.first_tick, -item.size, item.rid))
    if not rows:
        return LayoutResult({}, 0, 0, "optimal", 0)
    _integer(budget, "exact layout budget", minimum=1)
    incumbent = first_fit_layout(rows)
    lower_bound = incumbent.lower_bound
    if incumbent.extent == lower_bound:
        return LayoutResult(dict(incumbent.offsets), incumbent.extent, lower_bound, "optimal", 0)
    n = len(rows)
    conflicts = _conflict_lists(rows)
    best_extent = incumbent.extent
    best_offsets = dict(incumbent.offsets)
    placed = [0] * n
    expansions = 0
    exhausted = False

    def first_offset(index: int) -> int:
        """Symmetry breaking: identical items (size, alignment, interval) are placed in
        non-decreasing offsets; a twin is live with the item, so it is among its blockers."""
        item = rows[index]
        offset = 0
        for k in conflicts[index]:
            twin = rows[k]
            if (
                twin.size == item.size
                and twin.alignment == item.alignment
                and twin.first_tick == item.first_tick
                and twin.last_tick == item.last_tick
            ):
                offset = max(offset, placed[k])
        return _align(offset, item.alignment)

    # frame: [index, next candidate offset, extent before this item]
    stack: list[list[int]] = [[0, first_offset(0), 0]]
    while stack and not exhausted and best_extent != lower_bound:
        frame = stack[-1]
        index, offset, extent = frame
        if index == n:
            best_extent = extent
            best_offsets = {rows[k].rid: placed[k] for k in range(n)}
            stack.pop()
            continue
        item = rows[index]
        ceiling = best_extent - item.size  # any higher offset cannot beat the incumbent
        found = -1
        while offset <= ceiling:
            blocked = 0
            for k in conflicts[index]:
                if placed[k] < offset + item.size and offset < placed[k] + rows[k].size:
                    blocked = max(blocked, placed[k] + rows[k].size)
            if blocked:
                offset = _align(blocked, item.alignment)
                continue
            found = offset
            break
        if found < 0:
            stack.pop()
            continue
        expansions += 1
        if expansions > budget:
            exhausted = True
            break
        placed[index] = found
        frame[1] = _align(found + item.alignment, item.alignment)  # this frame's next candidate
        stack.append(
            [
                index + 1,
                first_offset(index + 1) if index + 1 < n else 0,
                max(extent, found + item.size),
            ]
        )
    reason = "budget" if exhausted else "optimal"
    return LayoutResult(best_offsets, best_extent, lower_bound, reason, expansions)


def plan_static_memory(
    module: Module,
    resource_banks,
    hardware,
    *,
    schedule=None,
    layout: str = "first-fit",
    budget: int = DEFAULT_EXACT_BUDGET,
) -> StaticMemoryPlan:
    """Assign aligned per-bank offsets, reusing storage only after a resource dies.

    `schedule` (a `GemSchedule` or an `ExecutionPlan`) makes the lifetimes the placement's
    own ticks (schedule liveness) and binds the plan to that placement; without one the
    lifetimes are phase positions. `layout="exact"` runs the bounded exact solver behind the
    first-fit incumbent under `budget` work units."""
    if layout not in LAYOUTS:
        raise ValueError(f"static memory layout must be one of {LAYOUTS}")
    domain, spans, ticks, bound_digest = _liveness(module, schedule)
    if not ticks:
        raise ValueError("static memory planning requires a touched resource")
    if len(ticks) > _MAX_RESOURCES:
        raise ValueError("static memory planning exceeds the resource bound")
    bindings = _bindings(resource_banks)
    if set(bindings) != set(ticks):
        raise ValueError("resource bank bindings must cover every touched resource exactly")

    by_bank: dict[str, list[int]] = {}
    for rid, bank_name in bindings.items():
        try:
            bank = hardware.bank(bank_name)
        except KeyError as exc:
            raise ValueError(f"resource {rid} names unknown bank {bank_name!r}") from exc
        resource = module.resource(rid)
        if resource is None:
            raise ValueError(f"binding references unknown resource {rid}")
        if resource.domain.name != bank.domain:
            raise ValueError(f"resource {rid} domain does not match bank {bank_name!r}")
        by_bank.setdefault(bank_name, []).append(rid)

    allocations: list[StaticAllocation] = []
    summaries: list[BankMemoryPlan] = []
    for bank_name in sorted(by_bank):
        bank = hardware.bank(bank_name)
        items: list[LayoutItem] = []
        naive = 0
        for rid in sorted(by_bank[bank_name], key=lambda value: (ticks[value][0], value)):
            resource = module.resource(rid)
            size = resource.count * resource.elem_bytes
            if not 1 <= size <= _MAX_U63:
                raise ValueError(f"resource {rid} has an invalid static byte size")
            alignment = max(resource.align, bank.alignment)
            if alignment & (alignment - 1):
                raise ValueError(
                    f"resource {rid} or bank {bank_name!r} has non-power-of-two alignment"
                )
            naive = _checked_add(_align(naive, alignment), size)
            items.append(LayoutItem(rid, size, alignment, ticks[rid][0], ticks[rid][1]))
        result = exact_layout(items, budget) if layout == "exact" else first_fit_layout(items)
        for item in items:
            offset = result.offsets[item.rid]
            if _checked_add(offset, item.size) > bank.allocatable_bytes:
                raise ValueError(
                    f"static address plan exceeds allocatable bytes in bank {bank_name!r}"
                )
            lo, hi = spans[item.rid]
            allocations.append(
                StaticAllocation(
                    item.rid,
                    bank_name,
                    offset,
                    item.size,
                    item.alignment,
                    lo,
                    hi,
                    item.first_tick,
                    item.last_tick,
                )
            )
        summaries.append(
            BankMemoryPlan(
                bank_name,
                result.extent,
                naive,
                bank.allocatable_bytes,
                result.lower_bound,
                result.stop_reason,
            )
        )
    identity = module_identity(module)  # the one digest of this module (G3 / S1-B)
    plan = StaticMemoryPlan(
        _module_digest(module, identity),
        hardware.digest,
        tuple(sorted(allocations, key=lambda row: row.rid)),
        tuple(summaries),
        liveness=domain,
        layout=layout,
        budget=budget if layout == "exact" else 0,
        schedule_digest=bound_digest,
    )
    errors = verify_static_memory_plan(
        plan, module, resource_banks, hardware, identity=identity, schedule=schedule
    )
    if errors:
        raise ValueError("static memory plan failed verification: " + "; ".join(errors))
    return plan


def verify_static_memory_plan(
    plan: StaticMemoryPlan,
    module: Module,
    resource_banks,
    hardware,
    identity: "ModuleIdentity | None" = None,
    schedule=None,
) -> tuple[str, ...]:
    """Independently check identity, capacity, alignment, lifetime, and alias safety.

    `identity` is the identity-bound API (G3 / S1-B): a caller that holds the module's
    `ModuleIdentity` passes it and the verifier validates it against the module's content
    (the canonical stream, never the revision) instead of re-hashing; without one, or with
    one that no longer describes the module, the verifier recomputes -- its right at a
    trust boundary.

    `schedule` is the placement the plan is composed with (G5 / S1-D). A schedule-liveness
    plan needs it: its digest must be the plan's and its intervals are recomputed from it. A
    phase-liveness plan is additionally held to it: two resources that share addresses under
    disjoint phase lifetimes must also be disjoint in the schedule's ticks -- the schedule
    must refine the phase order the plan used -- or the plan is refused (the two-phase alias
    fixture composed with the token placement). An exact layout is recomputed under the
    plan's own budget: the stop reason and the offsets are never taken on trust."""
    errors: list[str] = []
    try:
        bindings = _bindings(resource_banks)
    except ValueError as exc:
        return (str(exc),)
    spans = live_intervals(module)
    if plan.liveness == "schedule":
        if schedule is None:
            return ("plan lifetimes are schedule-derived but no schedule was supplied",)
        try:
            if schedule_digest(schedule) != plan.schedule_digest:
                errors.append(
                    "schedule digest mismatch: the plan was derived from another placement"
                )
            ticks = schedule_intervals(module, schedule)
        except ValueError as exc:
            return (f"schedule does not place this module: {exc}",)
    else:
        ticks = _phase_ticks(spans)
    if plan.module_digest != _module_digest(module, identity):
        errors.append("module digest mismatch")
    if plan.hardware_digest != hardware.digest:
        errors.append("hardware digest mismatch")
    rows = {row.rid: row for row in plan.allocations}
    if set(rows) != set(spans) or set(bindings) != set(spans) or set(ticks) != set(spans):
        errors.append("allocation/binding census mismatch")
    for rid in sorted(set(rows) & set(spans) & set(bindings) & set(ticks)):
        row = rows[rid]
        resource = module.resource(rid)
        if resource is None:
            errors.append(f"resource {rid} is absent from the module registry")
            continue
        try:
            bank = hardware.bank(bindings[rid])
        except KeyError:
            errors.append(f"resource {rid} names an unknown bank")
            continue
        if resource.domain.name != bank.domain:
            errors.append(f"resource {rid} domain does not match its bank")
        expected_size = resource.count * resource.elem_bytes
        expected_alignment = max(resource.align, bank.alignment)
        if row.bank != bindings[rid]:
            errors.append(f"resource {rid} bank mismatch")
        if row.size_bytes != expected_size:
            errors.append(f"resource {rid} size mismatch")
        if row.alignment != expected_alignment or row.offset % expected_alignment:
            errors.append(f"resource {rid} alignment mismatch")
        if (row.first_phase, row.last_phase) != spans[rid]:
            errors.append(f"resource {rid} lifetime mismatch")
        if (row.first_tick, row.last_tick) != ticks[rid]:
            errors.append(f"resource {rid} liveness interval mismatch ({plan.liveness})")
        if row.end > bank.allocatable_bytes:
            errors.append(f"resource {rid} exceeds bank capacity")
    for bank_name in sorted({row.bank for row in rows.values()}):
        bank_rows = [row for row in rows.values() if row.bank == bank_name]
        if _has_live_alias(bank_rows):
            errors.append(f"live resources in bank {bank_name!r} alias")
        if plan.liveness == "phase" and schedule is not None:
            # The schedule must refine the phase order the plan used: resources that share
            # addresses under disjoint phase lifetimes must be disjoint in the placement too.
            try:
                under = schedule_intervals(module, schedule)
            except ValueError as exc:
                errors.append(f"schedule does not place this module: {exc}")
            else:
                if all(row.rid in under for row in bank_rows) and _has_live_alias(
                    bank_rows, ticks=under
                ):
                    errors.append(
                        f"bank {bank_name!r}: the phase-liveness plan aliases under the "
                        f"supplied schedule (it does not refine the phase order)"
                    )
    summary = {row.bank: row for row in plan.banks}
    for bank_name in sorted({row.bank for row in rows.values()}):
        bank_rows = [row for row in rows.values() if row.bank == bank_name]
        if bank_name not in summary:
            errors.append(f"bank {bank_name!r} lacks a summary")
            continue
        try:
            bank = hardware.bank(bank_name)
        except KeyError:
            errors.append(f"bank summary names unknown bank {bank_name!r}")
            continue
        row = summary[bank_name]
        if row.extent_bytes != max(item.end for item in bank_rows):
            errors.append(f"bank {bank_name!r} extent mismatch")
        items = [
            LayoutItem(item.rid, item.size_bytes, item.alignment, item.first_tick, item.last_tick)
            for item in bank_rows
        ]
        if row.lower_bound_bytes != layout_lower_bound(items):
            errors.append(f"bank {bank_name!r} lower bound mismatch")
        if plan.layout == "first-fit":
            if row.stop_reason != "first-fit":
                errors.append(f"bank {bank_name!r} stop reason is not the fast path's")
        else:
            # The right to recompute: the exact solver is deterministic under its budget, so
            # the offsets and the stop reason are reproduced, never trusted.
            again = exact_layout(items, plan.budget)
            if again.stop_reason != row.stop_reason:
                errors.append(f"bank {bank_name!r} stop reason {row.stop_reason!r} not reproduced")
            if again.extent != row.extent_bytes or any(
                again.offsets.get(item.rid) != item.offset for item in bank_rows
            ):
                errors.append(f"bank {bank_name!r} exact layout not reproduced")
        try:
            naive = 0
            for item in sorted(bank_rows, key=lambda value: (value.first_tick, value.rid)):
                naive = _checked_add(_align(naive, item.alignment), item.size_bytes)
            if row.naive_bytes != naive:
                errors.append(f"bank {bank_name!r} naive footprint mismatch")
        except ValueError:
            errors.append(f"bank {bank_name!r} naive footprint overflows")
        if row.capacity_bytes != bank.allocatable_bytes:
            errors.append(f"bank {bank_name!r} capacity mismatch")
    if set(summary) != {row.bank for row in rows.values()}:
        errors.append("bank summary census mismatch")
    return tuple(errors)


__all__ = [
    "DEFAULT_EXACT_BUDGET",
    "LAYOUTS",
    "LIVENESS_DOMAINS",
    "STOP_REASONS",
    "BankMemoryPlan",
    "LayoutItem",
    "LayoutResult",
    "ResourceBankBinding",
    "StaticAllocation",
    "StaticMemoryPlan",
    "exact_layout",
    "first_fit_layout",
    "layout_lower_bound",
    "plan_static_memory",
    "schedule_digest",
    "schedule_intervals",
    "verify_static_memory_plan",
]
