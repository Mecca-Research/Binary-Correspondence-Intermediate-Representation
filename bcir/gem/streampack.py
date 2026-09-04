"""GEM StreamPack hydration (BCIR-4).

The StreamPack is the hot executable artifact; the BCIR graph is the dormant
semantic artifact. A StreamPack retains provenance back to BCIR claims and
carries generation tags (topo/map/data) so a stale pack can be rehydrated
(patch / repack / replan) instead of silently executing against drifted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import Lane, Module, topological_phase_ids
from ..kbcir.realize import RealizationResult


@dataclass(frozen=True)
class Prefetch:
    name: str
    distance: int
    targets: tuple[int, ...]  # operand RIDs (we prefetch operand data, not descriptors)
    hint: str = "T0"
    pattern: str = "linear"
    buffers: int = 1  # v2 (append-only): 2 = double-buffer contract


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
    # CIM/PIM dispatch target (append-only). "core" = execute on the compute core
    # (default); "pim" = dispatch to the memory controller / processing-in-memory,
    # set by `gem.cim.annotate_cim` only when the data-movement saved outweighs the
    # in-memory compute surcharge (a large reduction). The lowering reads this.
    dispatch: str = "core"
    # Heterogeneous channel (append-only): the HardwareChannel that executes this segment in a
    # mixed tower -- "host" (the planning CPU) by default, or e.g. "nvidia_ptx" / "fpga_systolic" /
    # "hbm_pim" when `channels.orchestrate` placed the claim off the host. The executor dispatches
    # per segment, so one StreamPack -- the same unified binary graph -- runs across the whole tower.
    channel: str = "host"


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
    pipeline_depth: int = 1  # v2 (append-only): phases in flight (2 = double-buffered)
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

    claim_rows = [(ph.phase_id, claim) for ph in module.phases for claim in ph.claims]
    claim_by_id = {claim.id: (phase_id, claim) for phase_id, claim in claim_rows}
    if len(claim_by_id) != len(claim_rows):
        raise ValueError("cannot hydrate a module with duplicate claim ids")
    phase_position = {phase_id: index for index, phase_id in enumerate(_topo_order(module))}
    seen: set[int] = set()
    last_phase = -1
    for n, step in enumerate(result.steps):
        row = claim_by_id.get(step.claim_id)
        if row is None:
            raise ValueError(f"cannot hydrate unknown claim {step.claim_id}")
        if step.claim_id in seen:
            raise ValueError(f"cannot hydrate duplicate plan step for claim {step.claim_id}")
        actual_phase, claim = row
        if step.phase_id != actual_phase:
            raise ValueError(
                f"claim {step.claim_id} plan phase {step.phase_id} != module phase {actual_phase}"
            )
        position = phase_position[actual_phase]
        if position < last_phase:
            raise ValueError(f"claim {step.claim_id} is out of topological phase order")
        last_phase = position
        seen.add(step.claim_id)
        cand = step.candidate
        pf_name = None
        if cand.width > 1 and claim.rd:
            pf = Prefetch(name=f"pf{n}", distance=4, targets=tuple(claim.rd))
            pack.prefetches.append(pf)
            pf_name = pf.name
        pack.blocks.append(Block(base=claim.offset, count=claim.count, strides=(claim.stride_k,)))
        pack.segments.append(
            LaneSegment(
                name=f"seg{n}",
                claim_id=claim.id,
                phase_id=step.phase_id,
                lane=cand.lane,
                width=cand.width,
                opcode=f"{claim.op or cand.name}",
                reads=tuple(claim.rd),
                writes=tuple(claim.wr),
                prefetch=pf_name,
            )
        )
        pack.trace_notes.append(TraceNote(claim_id=claim.id))
    missing = sorted(set(claim_by_id) - seen)
    if missing:
        raise ValueError(f"cannot hydrate a partial plan; missing claims {missing[:8]}")
    return pack


def hydrate_pipelined(
    module: Module, result: RealizationResult, plan: str = "plan0", depth: int = 2
) -> StreamPack:
    """Hydrate with software-pipelined phases (StreamPack v2, append-only).

    On top of `hydrate`, each phase transition gains a **double-buffer prefetch
    contract**: the read operands of the *next* phase are prefetched (buffers=2,
    pattern="double_buffer") while the current phase computes, and the pack's
    `pipeline_depth` tells the runtime how many phases are in flight. Token-DAG
    execution (`schedule.execute_tokens`) provides the matching overlap.
    """
    pack = hydrate(module, result, plan)
    pack.pipeline_depth = max(1, depth)
    if pack.pipeline_depth == 1:
        return pack

    pmap = module.phase_map()
    order = _topo_order(module)
    for prev_pid, next_pid in zip(order, order[1:]):
        rids: list[int] = []
        seen_rids: set[int] = set()
        for c in pmap[next_pid].claims:
            for rid in c.rd:
                if rid not in seen_rids:
                    seen_rids.add(rid)
                    rids.append(rid)
        if rids:
            pack.prefetches.append(
                Prefetch(
                    name=f"dbpf{prev_pid}_{next_pid}",
                    distance=4,
                    targets=tuple(rids),
                    hint="T1",
                    pattern="double_buffer",
                    buffers=2,
                )
            )
    return pack


def _topo_order(module: Module) -> list[int]:
    return topological_phase_ids(module)
