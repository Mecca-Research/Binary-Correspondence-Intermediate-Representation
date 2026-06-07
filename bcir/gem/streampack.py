"""GEM StreamPack hydration (BCIR-4).

The StreamPack is the hot executable artifact; the BCIR graph is the dormant
semantic artifact. A StreamPack retains provenance back to BCIR claims and
carries generation tags (topo/map/data) so a stale pack can be rehydrated
(patch / repack / replan) instead of silently executing against drifted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Lane, Module
from ..kbcir.realize import RealizationResult


@dataclass(frozen=True)
class Prefetch:
    name: str
    distance: int
    targets: tuple[int, ...]   # operand RIDs (we prefetch operand data, not descriptors)
    hint: str = "T0"
    pattern: str = "linear"


@dataclass(frozen=True)
class Block:
    base: int
    count: int
    strides: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class LaneSegment:
    name: str
    claim_id: int
    phase_id: int
    lane: Lane
    width: int
    opcode: str
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    prefetch: str | None = None
    fence_before: tuple[str, ...] = ()
    fence_after: tuple[str, ...] = ()


@dataclass(frozen=True)
class TraceNote:
    claim_id: int
    src_hash: int = 0
    trace_hash: int = 0


@dataclass
class StreamPack:
    source_plan: str = "plan0"
    topo_gen: int = 1
    map_gen: int = 0
    data_gen: int = 0
    segments: list[LaneSegment] = field(default_factory=list)
    prefetches: list[Prefetch] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    trace_notes: list[TraceNote] = field(default_factory=list)

    def provenance_ok(self) -> bool:
        """R10: every segment maps back to >= 1 BCIR claim via a trace note."""
        traced = {t.claim_id for t in self.trace_notes}
        return all(seg.claim_id in traced for seg in self.segments)


def hydrate(module: Module, result: RealizationResult, plan: str = "plan0") -> StreamPack:
    """Lower a selected realization plan into a StreamPack with provenance + tags."""
    map_gen = max((r.map_gen for r in module.resources.values()), default=0)
    data_gen = max((r.data_gen for r in module.resources.values()), default=0)
    pack = StreamPack(source_plan=plan, topo_gen=1, map_gen=map_gen, data_gen=data_gen)

    claim_by_id = {c.id: c for ph in module.phases for c in ph.claims}
    for n, step in enumerate(result.steps):
        claim = claim_by_id.get(step.claim_id)
        if claim is None:
            continue
        cand = step.candidate
        pf_name = None
        if cand.width > 1 and claim.rd:
            pf = Prefetch(name=f"pf{n}", distance=4, targets=tuple(claim.rd))
            pack.prefetches.append(pf)
            pf_name = pf.name
        pack.blocks.append(Block(base=claim.offset, count=claim.count, strides=(claim.stride_k,)))
        pack.segments.append(LaneSegment(
            name=f"seg{n}", claim_id=claim.id, phase_id=step.phase_id, lane=cand.lane,
            width=cand.width, opcode=f"{claim.op or cand.name}", reads=tuple(claim.rd),
            writes=tuple(claim.wr), prefetch=pf_name,
        ))
        pack.trace_notes.append(TraceNote(claim_id=claim.id))
    return pack
