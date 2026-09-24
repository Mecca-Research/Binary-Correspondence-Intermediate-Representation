"""The data-plane hand-off (G16, S3-C) -- the reference model of the C <-> C++ seam's contract.

The artifact crossing the seam is a frozen StreamPack. This module is the oracle the freestanding
C twin (`runtime/c/bcir_handoff.h`) and the C++ seam (`runtime/cpp/bcir_handoff.hpp`) are held to,
operation for operation:

* **Borrowed views with explicit lifetime.** A `PackTable` owns fixed-size slots of one arena. A
  producer reserves a slot, writes the artifact into it ONCE and commits it (frozen: no API writes
  it again); consumers borrow it through a handle (slot index, epoch). Releasing the owner advances
  the epoch, so every view of that incarnation is refused from then on (`BCIR_ERR_LIFETIME`) --
  while a borrow already in progress keeps its bytes (a pinned slot retires and is freed by its
  last return, never reused under a reader). Nothing is copied where a borrow suffices.
* **Generation gating at admit().** Admission asks the LIVE plane (G14's `ControlPlane`) -- its one
  predicate `admit_pack` against the installed registry -- and records the resident generation it
  admitted at. Dispatch re-checks that the admission is of the resident generation (a switch
  between admit and dispatch makes the pack stale), runs the walk as a phase of the plane (so a
  switch requested mid-dispatch is deferred to the boundary) and never runs an unadmitted pack.
* **The per-step freeze.** `freeze_claims` is the oracle of `bcir_hydrate_generations`: a
  dynamic graph's claims, planned by the scalar planner and hydrated as a v4 pack bound to the
  live registry vector -- the bytes the C++ graph builder writes straight into a reserved slot.
* **The manifest at the plane.** `admit_manifest` gates a shard manifest (bcir.abi.shard_manifest)
  against the installed registry before any shard is fetched.

Every operation returns a `HandoffOutcome`: a verdict, a refusal and the C twin's status name. The
traces of the rails -- outcome, handle, resident generation, and the table's and the plane's state
digests after every operation -- are held identical by the G16 rows
(`tools/perf/gemplus_baseline.py --group handoff`).
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field

from .control import ControlPlane, is_stale

HO_VERDICTS = ("applied", "refused")
HO_REFUSALS = (
    "none",
    "lifetime",  # the handle/view names no live slot at its epoch, or a return that was not a borrow
    "full",  # no free slot
    "capacity",  # a length outside [1, slot capacity]
    "stale",  # the plane refused the pack as stale, or the admission is older than the resident
    "unadmitted",  # dispatch of a pack with no admission in this incarnation
    "malformed",  # the plane refused the pack as malformed
    "draining",  # the plane refused the dispatch phase (draining)
    "exhausted",  # an epoch, pin or plane counter at its bound
    "walk",  # the segment walk failed
)
SLOT_STATES = ("free", "building", "frozen", "retired", "exhausted")
SLOTS_MAX = 65535
EPOCH_FIRST = 1
EPOCH_MAX = 0xFFFFFFFF
PINS_MAX = 0xFFFFFFFF
_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF
_STATE_TAG = b"BHOF/state/v0\x00"
ZERO32 = bytes(32)

_STATUS = {
    "none": "BCIR_OK",
    "lifetime": "BCIR_ERR_LIFETIME",
    "full": "BCIR_ERR_FULL",
    "capacity": "BCIR_ERR_NOSPACE",
    "stale": "BCIR_ERR_STALE",
    "unadmitted": "BCIR_ERR_STALE",
    "draining": "BCIR_ERR_CONTROL",
    "exhausted": "BCIR_ERR_OVERFLOW",
}


@dataclass(frozen=True)
class Handle:
    """A slot incarnation: the index and the epoch it was issued at."""

    index: int
    epoch: int


@dataclass(frozen=True)
class View:
    """A borrow in progress: the handle it pinned and the bytes (valid until it is returned)."""

    index: int
    epoch: int
    data: memoryview


@dataclass(frozen=True)
class HandoffOutcome:
    verdict: str
    refusal: str = "none"
    status: str = "BCIR_OK"
    generation: int = 0
    handle: Handle | None = None
    claims: tuple[int, ...] = ()

    @property
    def applied(self) -> bool:
        return self.verdict == "applied"


def _refused(refusal: str, status: str | None = None, generation: int = 0) -> HandoffOutcome:
    return HandoffOutcome("refused", refusal, status or _STATUS[refusal], generation)


@dataclass
class _Slot:
    epoch: int = EPOCH_FIRST
    state: str = "free"
    admitted: bool = False
    pins: int = 0
    admitted_generation: int = 0
    admitted_registry: bytes = ZERO32
    length: int = 0


class PackTable:
    """Fixed-size slots over one arena: the mirror of `bcir_ho_table` decision for decision."""

    def __init__(self, n_slots: int, slot_capacity: int) -> None:
        if not isinstance(n_slots, int) or not 1 <= n_slots <= SLOTS_MAX:
            raise ValueError(f"a table has 1..{SLOTS_MAX} slots, got {n_slots!r}")
        if not isinstance(slot_capacity, int) or not 1 <= slot_capacity <= _U32:
            raise ValueError(f"a slot holds 1..{_U32} bytes, got {slot_capacity!r}")
        self.n_slots = n_slots
        self.capacity = slot_capacity
        self.arena = bytearray(n_slots * slot_capacity)
        self.slots = [_Slot() for _ in range(n_slots)]

    # --- reading the state ---------------------------------------------------------------

    def _region(self, index: int) -> slice:
        return slice(index * self.capacity, (index + 1) * self.capacity)

    def _live(self, handle, states=("frozen",)) -> _Slot | None:
        if not isinstance(handle, (Handle, View)) or not 0 <= handle.index < self.n_slots:
            return None
        slot = self.slots[handle.index]
        return slot if slot.state in states and slot.epoch == handle.epoch else None

    def bytes_of(self, index: int) -> bytes:
        slot = self.slots[index]
        base = index * self.capacity
        return bytes(self.arena[base : base + slot.length])

    def state_bytes(self) -> bytes:
        """The canonical serialization of the table (what two rails compare after every op)."""
        parts = [_STATE_TAG, struct.pack("<IQ", self.n_slots, self.capacity)]
        for index, slot in enumerate(self.slots):
            held = slot.state in ("frozen", "retired") or (
                slot.state == "exhausted" and slot.pins > 0
            )
            parts.append(
                struct.pack(
                    "<IBBIIQ",
                    slot.epoch,
                    SLOT_STATES.index(slot.state),
                    int(slot.admitted),
                    slot.pins,
                    slot.admitted_generation,
                    slot.length if held else 0,
                )
            )
            parts.append(slot.admitted_registry)
            parts.append(hashlib.sha256(self.bytes_of(index)).digest() if held else ZERO32)
        return b"".join(parts)

    def state_digest(self) -> bytes:
        return hashlib.sha256(self.state_bytes()).digest()

    # --- the producer: reserve, write once, commit or abort --------------------------------

    def reserve(self) -> HandoffOutcome:
        """The lowest free slot, now building; FULL when none is free (building, frozen,
        retired while a borrow drains, or exhausted: a slot out of epochs leaves service)."""
        for index, slot in enumerate(self.slots):
            if slot.state == "free":
                slot.state = "building"
                slot.length = 0
                handle = Handle(index, slot.epoch)
                return HandoffOutcome("applied", handle=handle)
        return _refused("full")

    def write(self, handle: Handle, offset: int, data: bytes) -> HandoffOutcome:
        """The producer's write into its reservation (the C producer writes through the pointer
        `bcir_ho_reserved` returns): only a building slot, only inside its region."""
        if self._live(handle, ("building",)) is None:
            return _refused("lifetime")
        if offset < 0 or offset + len(data) > self.capacity:
            return _refused("capacity")
        base = handle.index * self.capacity + offset
        self.arena[base : base + len(data)] = data
        return HandoffOutcome("applied", handle=handle)

    def commit(self, handle: Handle, length: int) -> HandoffOutcome:
        """Freeze the first `length` bytes: from here on no API writes the slot."""
        slot = self._live(handle, ("building",))
        if slot is None:
            return _refused("lifetime")
        if not 1 <= length <= self.capacity:
            return _refused("capacity")
        slot.state = "frozen"
        slot.length = length
        slot.admitted = False
        return HandoffOutcome("applied", handle=handle)

    def abort(self, handle: Handle) -> HandoffOutcome:
        """End a reservation with nothing published: the slot is free, its epoch advanced."""
        slot = self._live(handle, ("building",))
        if slot is None:
            return _refused("lifetime")
        return self._end(slot)

    def _end(self, slot: _Slot) -> HandoffOutcome:
        slot.admitted = False
        slot.admitted_generation = 0
        slot.admitted_registry = ZERO32
        if slot.epoch == EPOCH_MAX:  # an epoch never wraps: out of epochs, out of service
            slot.state = "exhausted"
        else:
            slot.epoch += 1
            slot.state = "free" if slot.pins == 0 else "retired"
        if slot.pins == 0:
            slot.length = 0
        return HandoffOutcome("applied")

    # --- the consumers: borrow, return, release ----------------------------------------------

    def borrow(self, handle: Handle) -> tuple[HandoffOutcome, View | None]:
        slot = self._live(handle)
        if slot is None:
            return _refused("lifetime"), None
        if slot.pins == PINS_MAX:
            return _refused("exhausted"), None
        slot.pins += 1
        base = handle.index * self.capacity
        data = memoryview(self.arena)[base : base + slot.length].toreadonly()
        return HandoffOutcome("applied", handle=handle), View(handle.index, handle.epoch, data)

    def give_back(self, view) -> HandoffOutcome:
        """Return a borrow: accepted for a frozen slot of its epoch or a slot retired from it."""
        if (
            not isinstance(view, (View, Handle))
            or view.epoch == 0  # never issued: it must not match a slot retired from epoch 1
            or not 0 <= view.index < self.n_slots
        ):
            return _refused("lifetime")
        slot = self.slots[view.index]
        current = slot.state == "frozen" and slot.epoch == view.epoch
        retired = slot.state == "retired" and slot.epoch == view.epoch + 1
        exhausted = slot.state == "exhausted" and slot.epoch == view.epoch
        if slot.pins == 0 or not (current or retired or exhausted):
            return _refused("lifetime")
        slot.pins -= 1
        if slot.pins == 0 and slot.state in ("retired", "exhausted"):
            slot.state = "free" if slot.state == "retired" else "exhausted"
            slot.length = 0
        return HandoffOutcome("applied")

    def release(self, handle: Handle) -> HandoffOutcome:
        """The owner lets go: every view of this incarnation is refused from now on; a borrow
        in progress keeps its bytes until it is returned."""
        slot = self._live(handle)
        if slot is None:
            return _refused("lifetime")
        return self._end(slot)

    # --- the plane: admit, dispatch ----------------------------------------------------------

    def admit(self, handle: Handle, plane: ControlPlane) -> HandoffOutcome:
        """Admit the artifact against the LIVE registry (G14's one predicate) and record the
        resident generation it was admitted at. A refusal revokes any earlier admission."""
        out, view = self.borrow(handle)
        if view is None:
            return out
        verdict = plane.admit_pack(bytes(view.data))
        self.give_back(view)
        slot = self.slots[handle.index]
        if verdict.applied:
            slot.admitted = True
            slot.admitted_generation = plane.generation
            slot.admitted_registry = plane.registry[3]
            return HandoffOutcome("applied", generation=plane.generation, handle=handle)
        slot.admitted = False
        slot.admitted_generation = 0
        slot.admitted_registry = ZERO32
        if verdict.refusal == "stale":
            return _refused("stale", generation=plane.generation)
        return _refused("malformed", "-", plane.generation)

    def dispatch(self, handle: Handle, plane: ControlPlane) -> HandoffOutcome:
        """Run the admitted artifact's segments (claim ids in dispatch order) as a phase of the
        plane. Refused: a dead view; no admission in this incarnation; an admission of a
        generation the plane has left; the plane refusing the phase."""
        out, view = self.borrow(handle)
        if view is None:
            return out
        slot = self.slots[handle.index]
        gen = plane.generation
        if not slot.admitted:
            self.give_back(view)
            return _refused("unadmitted", generation=gen)
        registry = plane.registry[3] if plane.registry is not None else ZERO32
        if (
            is_stale(slot.admitted_generation, gen)
            or slot.admitted_generation != gen
            or slot.admitted_registry != registry
        ):  # an admission lives within one resident generation of one registry
            self.give_back(view)
            return _refused("stale", generation=gen)
        entered = plane.enter()
        if entered.verdict == "refused":
            self.give_back(view)
            refusal = "draining" if entered.refusal == "draining" else "exhausted"
            return _refused(refusal, generation=plane.generation)
        from ..abi import AbiError, decode

        try:
            claims = tuple(seg.claim_id for seg in decode(bytes(view.data)).segments)
            walked = True
        except AbiError:
            claims, walked = (), False
        plane.leave()  # the boundary: a switch requested mid-dispatch lands here, once
        self.give_back(view)
        if not walked:
            return _refused("walk", "-", plane.generation)
        return HandoffOutcome("applied", generation=plane.generation, handle=handle, claims=claims)

    # --- the producer as a graph builder -----------------------------------------------------

    def freeze(self, claims, gens, topo_gen: int) -> HandoffOutcome:
        """One step of the dynamic-graph builder: reserve a slot, freeze the step's claim graph
        into it (the `bcir_hydrate_generations` oracle) and commit -- or abort, publishing
        nothing, with the freeze's own status."""
        reserved = self.reserve()
        if not reserved.applied:
            return reserved
        handle = reserved.handle
        try:
            data = freeze_claims(claims, gens, topo_gen, self.capacity)
        except FreezeError as exc:
            self.abort(handle)
            return HandoffOutcome(
                "refused",
                "capacity" if exc.status == "BCIR_ERR_NOSPACE" else "malformed",
                exc.status,
            )
        self.write(handle, 0, data)
        return self.commit(handle, len(data))


# --- the per-step freeze (the oracle of bcir_plan_func + bcir_hydrate_generations) -----------

OPCODE_NOP, OPCODE_LOAD, OPCODE_STORE, OPCODE_MUL = 0, 1, 2, 5
OPCODE_ATOMICS = (6, 7, 8, 9)
OPCODE_GEM_DISPATCH = 16
DOMAIN_MMIO = 3
LANE_MAX = 5
CLAIM_MAX_RD, CLAIM_MAX_WR = 6, 2
LABEL_MAX = 31


class FreezeError(ValueError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class GraphClaim:
    """One node of a step's claim graph, as the C builder hands it to the rail (bcir_claim)."""

    id: int
    opcode: int = 3  # ADD
    lane: int = 0
    domain: int = 0
    count: int = 1
    reads: tuple[int, ...] = ()
    writes: tuple[int, ...] = ()
    label: str = "c.add"


@dataclass(frozen=True)
class GraphGeneration:
    rid: int
    map_gen: int = 0
    data_gen: int = 0


def base_cost(claim: GraphClaim) -> int:
    """`bcir_plan_base_cost`: memory and MMIO dear, multiply dearer than add, atomics dearest."""
    if claim.opcode in (OPCODE_LOAD, OPCODE_STORE):
        cost = 4
    elif claim.opcode == OPCODE_MUL:
        cost = 3
    elif claim.opcode in OPCODE_ATOMICS:
        cost = 8
    elif claim.opcode == OPCODE_GEM_DISPATCH:
        cost = 2
    elif claim.opcode == OPCODE_NOP:
        cost = 0
    else:
        cost = 1
    return cost + (4 if claim.domain == DOMAIN_MMIO else 0)


def plan_claims(claims) -> tuple[int, ...]:
    """`bcir_plan_func`: the scalar realization (width 1) with every cost preflighted --
    BCIR_ERR_OVERFLOW when a step's cost leaves u32 or the total leaves u64."""
    total = 0
    for claim in claims:
        cost = base_cost(claim) * (claim.count or 1)
        if cost > _U32 or total > _U64 - cost:
            raise FreezeError("BCIR_ERR_OVERFLOW", f"claim {claim.id}: cost overflows")
        total += cost
    return tuple(1 for _ in claims)


def _label_ok(label: str) -> bool:
    raw = label.encode("latin-1", "replace")
    return len(raw) <= LABEL_MAX and all(0x20 <= b < 0x7F for b in raw) and label.isascii()


def freeze_claims(claims, gens, topo_gen: int, capacity: int | None = None) -> bytes:
    """The v4 StreamPack `bcir_hydrate_generations` writes for a step's claims under the live
    registry vector `gens` and topology generation `topo_gen` -- one segment per non-NOP claim
    in claim order (scalar width, phase 0, the label as name and opcode), one trace record per
    segment, the vector appended and its maxima in the header. The laws, in the C order:
    the plan's overflow law; a vector (non-empty, RIDs strictly ascending); per claim its
    read/write bounds and label, ascending ids, the lane, and every RID it touches declared by
    the vector; then capacity."""
    from ..abi.streampack_abi import encode
    from ..model.lanes import Lane
    from .streampack import Generation, LaneSegment, StreamPack, TraceNote

    claims = list(claims)
    if len(claims) > _U32:
        raise FreezeError("BCIR_ERR_NOSPACE", "too many claims")
    widths = plan_claims(claims)
    gens = list(gens)
    if not gens:
        raise FreezeError("BCIR_ERR_GENERATION", "a frozen step binds to a registry vector")
    for prev, cur in zip(gens, gens[1:]):
        if cur.rid <= prev.rid:
            raise FreezeError("BCIR_ERR_GENERATION", "vector RIDs must be strictly ascending")
    declared = {g.rid for g in gens}
    previous = None
    segments, trace = [], []
    for claim, width in zip(claims, widths):
        if (
            len(claim.reads) > CLAIM_MAX_RD
            or len(claim.writes) > CLAIM_MAX_WR
            or not _label_ok(claim.label)
        ):
            raise FreezeError("BCIR_ERR_PROVENANCE", f"claim {claim.id}: bounds or label")
        if previous is not None and claim.id <= previous:
            raise FreezeError("BCIR_ERR_PROVENANCE", f"claim {claim.id}: ids must ascend")
        previous = claim.id
        if claim.opcode == OPCODE_NOP:
            continue
        if not 0 <= claim.lane <= LANE_MAX:
            raise FreezeError("BCIR_ERR_LANE", f"claim {claim.id}: lane {claim.lane}")
        if width == 0 or width & (width - 1):
            raise FreezeError("BCIR_ERR_WIDTH", f"claim {claim.id}: width {width}")
        if any(rid not in declared for rid in (*claim.reads, *claim.writes)):
            raise FreezeError("BCIR_ERR_PROVENANCE", f"claim {claim.id}: undeclared RID")
        segments.append(
            LaneSegment(
                name=claim.label,
                claim_id=claim.id,
                phase_id=0,
                lane=Lane(claim.lane),
                width=width,
                opcode=claim.label,
                reads=tuple(claim.reads),
                writes=tuple(claim.writes),
            )
        )
        trace.append(TraceNote(claim.id))
    pack = StreamPack(
        source_plan="",
        topo_gen=topo_gen,
        map_gen=max(g.map_gen for g in gens),
        data_gen=max(g.data_gen for g in gens),
        segments=segments,
        trace_notes=trace,
        generations=[Generation(g.rid, g.map_gen, g.data_gen) for g in gens],
    )
    data = encode(pack)
    if capacity is not None and len(data) > capacity:
        raise FreezeError("BCIR_ERR_NOSPACE", f"{len(data)} bytes exceed the slot's {capacity}")
    return data


# --- the manifest at the plane -------------------------------------------------------------


def admit_manifest(plane: ControlPlane, manifest: bytes) -> HandoffOutcome:
    """Gate a shard manifest against the installed registry before any shard is fetched: its
    tags and its vector digest must be the registry's (the same law `admit_pack` applies)."""
    from ..abi.shard_manifest import ShardError, decode_manifest

    try:
        m = decode_manifest(manifest)
    except ShardError as exc:
        return _refused("malformed", exc.status, plane.generation)
    reg = plane.registry
    if reg is None or (m.map_gen, m.data_gen, m.topo_gen, m.registry_digest) != reg:
        return _refused("stale", generation=plane.generation)
    return HandoffOutcome("applied", generation=plane.generation)


__all__ = [
    "EPOCH_FIRST",
    "EPOCH_MAX",
    "HO_REFUSALS",
    "HO_VERDICTS",
    "SLOTS_MAX",
    "SLOT_STATES",
    "FreezeError",
    "GraphClaim",
    "GraphGeneration",
    "Handle",
    "HandoffOutcome",
    "PackTable",
    "View",
    "admit_manifest",
    "base_cost",
    "freeze_claims",
    "plan_claims",
]
