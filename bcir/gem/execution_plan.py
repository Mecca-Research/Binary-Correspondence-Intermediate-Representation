"""ExecutionPlanV1 -- the plan as bytes (G11, staged plan S1-C).

The canonical plan cannot be Python objects if the C twin, the MLIR rail and a resident
executor are to read it. This module is the abstract value the `ExecutionPlanV1` wire
format (`bcir/abi/execution_plan_abi.py`, `docs/kernel/BCIR_EXECUTION_PLAN_ABI.md`,
`runtime/c/bcir_execution_plan.h`) carries: the four record families the roadmap names,

* **steps** -- what G1 makes canonical: the plan's realization (claim, phase, candidate,
  lane, width, cost) and the one placement both executors run and both pricers read
  (stream, start, duration), one record per claim in the realization's own order;
* **lifetimes** -- the static-memory planner's addresses (bank, offset, size, alignment,
  first/last phase), the family G5 makes schedule-aware;
* **moves** -- the movement edges G8 will produce (source and destination bank, byte range,
  route, coherence action, generation, overlap window, kind); declared, carried and verified
  for well-formedness now so G8 lands as content, not as a wire change;
* **generations** -- the registry's per-resource generation vector the plan was placed under
  (R11, the StreamPack v4 record carried forward).

The pack stays the executable; the plan is what the pack was derived from and what every
reader prices. `plan_from_realization` mints one; `schedule_of` / `realization_of` are the
readers the pricer, the executors and the StreamPack lowering share; `verify.
verify_execution_plan` is the verifier (unknown, duplicate, out-of-order and missing claims,
the binding to the module/target, the generation vector against the registry, the pack it
was lowered into).

Everything below the model imports lazily: this module is on the hot path through
`bcir.abi`, and the schedulers it reads are cold organs (`tools/perf/import_graph.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Lane, Module
from .streampack import Generation, generation_vector

#: The two placements `gem.schedule.schedule_plan` produces (phase-barriered, token-pipelined).
PLAN_MODES = ("eft", "tokens")

#: The decoupled GGG/random tail's stream, as `gem.schedule.TAIL_STREAM` reports it (the
#: wire spells it 0xFFFFFFFF; the model keeps the scheduler's -1).
TAIL_STREAM = -1

#: The G8 movement-edge kinds (the roadmap's "direct, peer, staged, rematerialized,
#: compressed or evicted") and coherence actions, closed sets on the wire (u8 codes).
MOVE_KINDS = ("direct", "peer", "staged", "rematerialized", "compressed", "evicted")
COHERENCE_ACTIONS = ("none", "flush", "invalidate", "writeback")


@dataclass(frozen=True)
class PlanStep:
    """One claim of the plan: its realization and its placement."""

    claim_id: int
    phase_id: int
    candidate: str  # the candidate's name (scalar / vec8 / tile / ...)
    lane: Lane
    width: int
    cost: int  # the plan's own step cost (signed; the serial bound is their sum)
    stream: int  # affinity domain, or TAIL_STREAM
    start: int
    duration: int  # max(0, cost): the slot length the dispatch placed


@dataclass(frozen=True)
class Lifetime:
    """A resource's static address and live range (`kbcir.static_memory.StaticAllocation`)."""

    rid: int
    bank: str
    offset: int
    size_bytes: int
    alignment: int
    first_phase: int
    last_phase: int


@dataclass(frozen=True)
class MovementEdge:
    """A G8 movement edge: `size_bytes` of `rid` from `src_bank` to `dst_bank` by `route`,
    `kind` (direct/peer/staged/rematerialized/compressed/evicted) with a `coherence` action,
    tagged with the generation it moves and the overlap window it executes in (after
    `after_claim` finishes and before `before_claim` starts; 0 = unconstrained)."""

    rid: int
    src_bank: str
    dst_bank: str
    offset: int
    size_bytes: int
    route: str = ""
    kind: str = "direct"
    coherence: str = "none"
    map_gen: int = 0
    data_gen: int = 0
    after_claim: int = 0
    before_claim: int = 0


@dataclass
class ExecutionPlan:
    source_plan: str = "plan0"
    mode: str = "eft"
    streams: int = 1  # affinity domains the plan was placed on (the tail is extra)
    knee: int = 1  # the bandwidth knee the dispatch clamped to
    makespan: int = 0
    module_hash: int = 0  # R13 `hash_module` of the goal graph the plan realizes
    target_hash: int = 0  # R13 `hash_target` of the target it was placed for (0 = none)
    steps: list[PlanStep] = field(default_factory=list)
    lifetimes: list[Lifetime] = field(default_factory=list)
    moves: list[MovementEdge] = field(default_factory=list)
    generations: list[Generation] = field(default_factory=list)

    @property
    def serial(self) -> int:
        """The serial bound: the sum of the plan's own step costs (R9's `serial`)."""
        return sum(s.cost for s in self.steps)

    def step_of(self, claim_id: int) -> PlanStep:
        for s in self.steps:
            if s.claim_id == claim_id:
                return s
        raise KeyError(claim_id)

    def slot_map(self) -> dict[int, tuple[int, int, int]]:
        """claim id -> (stream, start, finish): the trace a reader compares."""
        return {s.claim_id: (s.stream, s.start, s.start + s.duration) for s in self.steps}


# --- producers --------------------------------------------------------------------------


def plan_from_realization(
    module: Module,
    result,
    target=None,
    mode: str = "eft",
    *,
    plan: str = "plan0",
    static_plan=None,
    identity=None,
    locality: bool = True,
) -> ExecutionPlan:
    """Mint the plan of a K_BCIR realization: its steps placed by the canonical schedule
    artifact (`gem.schedule.schedule_plan`, the same placement `price_scheduled` reads and
    the executors run), bound to the module (R13 digest, computed once via the S1-B
    identity API) and the target, carrying the registry's generation vector and, when a
    static memory plan is supplied, its lifetimes and addresses."""
    from ..kbcir.provenance import digest_of, hash_target
    from .schedule import TAIL_STREAM as _TAIL, schedule_plan, stream_geometry

    if mode not in PLAN_MODES:
        raise ValueError(f"unknown plan mode {mode!r}; expected one of {PLAN_MODES}")
    sched = schedule_plan(module, result, target, mode, locality)
    slot_of = {s.claim_id: s for s in sched.slots}
    streams, knee = stream_geometry(target)
    steps: list[PlanStep] = []
    seen: set[int] = set()
    for step in result.steps:
        if step.claim_id in seen:
            raise ValueError(f"realization has two steps for claim {step.claim_id}")
        seen.add(step.claim_id)
        slot = slot_of.get(step.claim_id)
        if slot is None:
            raise ValueError(f"the schedule artifact placed no slot for claim {step.claim_id}")
        cand = step.candidate
        steps.append(
            PlanStep(
                claim_id=step.claim_id,
                phase_id=step.phase_id,
                candidate=str(cand.name),
                lane=Lane(cand.lane),
                width=int(cand.width),
                cost=int(step.cost),
                stream=TAIL_STREAM if slot.domain == _TAIL else int(slot.domain),
                start=int(slot.start),
                duration=int(slot.finish - slot.start),
            )
        )
    if len(steps) != len(sched.slots):
        raise ValueError("the schedule artifact placed claims the realization does not step")
    lifetimes: list[Lifetime] = []
    if static_plan is not None:
        for row in sorted(static_plan.allocations, key=lambda a: a.rid):
            lifetimes.append(
                Lifetime(
                    rid=row.rid,
                    bank=row.bank,
                    offset=row.offset,
                    size_bytes=row.size_bytes,
                    alignment=row.alignment,
                    first_phase=row.first_phase,
                    last_phase=row.last_phase,
                )
            )
    return ExecutionPlan(
        source_plan=plan,
        mode=mode,
        streams=streams,
        knee=knee,
        makespan=int(sched.makespan),
        module_hash=digest_of(module, identity),
        target_hash=hash_target(target) if target is not None else 0,
        steps=steps,
        lifetimes=lifetimes,
        moves=[],
        generations=generation_vector(module),
    )


# --- readers ----------------------------------------------------------------------------


def schedule_of(plan: ExecutionPlan):
    """The plan's placement as a `gem.schedule.GemSchedule` (slots in (start, stream,
    claim) order -- the trace; `slot_of`, `affinity`, `makespan` and `knee` as the executors
    report them). A reader that holds the bytes does not re-run the dispatch."""
    from .schedule import TAIL_STREAM as _TAIL, GemSchedule, Slot

    sched = GemSchedule(mode=plan.mode, knee=plan.knee, makespan=plan.makespan)
    for s in sorted(plan.steps, key=lambda s: (s.start, s.stream, s.claim_id)):
        stream = _TAIL if s.stream == TAIL_STREAM else s.stream
        sched.slots.append(Slot(s.claim_id, stream, s.start, s.start + s.duration))
        sched.affinity[s.claim_id] = stream
    return sched


def realization_of(plan: ExecutionPlan):
    """The plan's realization as a `kbcir.realize.RealizationResult`: the steps the
    StreamPack lowering hydrates and the schedulers re-place (`durations_from` reads the
    step costs). The candidates carry the name, lane and width the plan chose; their base
    cost vectors are not part of the plan and read as zero."""
    from ..kbcir.cost import CostVector
    from ..kbcir.realize import Candidate, ChosenStep, RealizationResult

    zero = CostVector.zero()
    steps = [
        ChosenStep(
            s.claim_id,
            s.phase_id,
            Candidate(lane=s.lane, width=s.width, name=s.candidate, base=zero),
            s.cost,
        )
        for s in plan.steps
    ]
    return RealizationResult(steps=steps, score=plan.serial)


__all__ = [
    "COHERENCE_ACTIONS",
    "ExecutionPlan",
    "Lifetime",
    "MOVE_KINDS",
    "MovementEdge",
    "PLAN_MODES",
    "PlanStep",
    "TAIL_STREAM",
    "plan_from_realization",
    "realization_of",
    "schedule_of",
]
