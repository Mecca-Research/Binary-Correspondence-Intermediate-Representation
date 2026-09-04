"""Compute-In-Memory (CIM / PIM) dispatch awareness for the GEM StreamPack.

The StreamPack schedules where each op *runs*. For a large reduction the data is
big and the result is tiny, so streaming the whole input to the core just to add it
up wastes bandwidth: the reduction can be dispatched to the **memory controller**
(processing-in-memory) instead, where the data already lives -- only the scalar
result returns. `annotate_cim` rewrites the eligible reduction segments to
`dispatch="pim"`.

GAINS-ONLY: a segment is offloaded only when the modeled cost strictly drops --
the bytes-not-moved must outweigh PIM's in-memory compute surcharge (PIM compute is
slower per element than the core, so small reductions stay on the core). Non-
reduction ops are never offloaded (PIM does simple element-local work, not general
SIMD). Pure + deterministic: it annotates a plan, it does not execute anything.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..model import Module
from ..kbcir.cost import TargetProfile
from ..kbcir.realize import RealizationResult
from .streampack import StreamPack, hydrate

# PIM does element-local compute slower than the core: a Q8 surcharge per element
# (256 == x1.0). 384 == x1.5: PIM pays 50% more compute, but moves no operand bytes.
_PIM_COMPUTE_Q8 = 384
# Fixed cost of issuing a PIM operation (command + result fence), in the same units
# as operand bytes. This is what makes CIM a *large*-reduction win: the setup only
# amortizes once the bytes-not-moved dwarf it, so small reductions stay on the core.
_PIM_DISPATCH_OVERHEAD = 4096


def _is_reduction(opcode: str) -> bool:
    return opcode.startswith("reduce.") or opcode.startswith("reduce_")


@dataclass(frozen=True)
class CIMDecision:
    claim_id: int
    offload: bool
    core_cost: int
    pim_cost: int

    @property
    def saved(self) -> int:
        return self.core_cost - self.pim_cost if self.offload else 0


def cim_decision(opcode: str, count: int, elem_bytes: int, h: TargetProfile) -> CIMDecision:
    """Model core vs PIM cost for a reduction over `count` elements.

    core = move every operand byte to the core + reduce there.
    pim  = reduce in memory (compute surcharge, no operand traffic) + return the
           scalar result (one element).
    Offload iff it is a reduction and pim strictly beats core.
    """
    if not _is_reduction(opcode):
        # placeholder cost; non-reductions are never offloaded.
        return CIMDecision(claim_id=-1, offload=False, core_cost=0, pim_cost=0)
    traffic = count * elem_bytes  # operand bytes streamed to the core
    compute = count * h.mem_unit  # per-element reduce work
    core_cost = traffic + compute
    # surcharged in-memory compute + fixed dispatch setup + 1-element result return
    pim_cost = (compute * _PIM_COMPUTE_Q8 >> 8) + _PIM_DISPATCH_OVERHEAD + elem_bytes
    return CIMDecision(
        claim_id=-1, offload=pim_cost < core_cost, core_cost=core_cost, pim_cost=pim_cost
    )


def annotate_cim(
    module: Module, result: RealizationResult, h: TargetProfile, pack: StreamPack | None = None
) -> tuple[StreamPack, list[CIMDecision]]:
    """Return (annotated StreamPack, decisions): reduction segments whose modeled
    PIM cost beats the core get `dispatch="pim"`; everything else is unchanged."""
    pack = pack or hydrate(module, result)
    claim_by_id = {c.id: c for ph in module.phases for c in ph.claims}
    decisions: list[CIMDecision] = []
    new_segs = []
    for seg in pack.segments:
        claim = claim_by_id.get(seg.claim_id)
        seg_out = seg
        if claim is not None and _is_reduction(seg.opcode):
            res = module.resource(claim.rd[0]) if claim.rd else None
            eb = res.elem_bytes if res is not None else h.elem_bytes
            d = cim_decision(seg.opcode, max(1, claim.count), eb, h)
            d = CIMDecision(
                claim_id=seg.claim_id, offload=d.offload, core_cost=d.core_cost, pim_cost=d.pim_cost
            )
            decisions.append(d)
            if d.offload:
                seg_out = replace(seg, dispatch="pim")
        new_segs.append(seg_out)
    pack.segments = new_segs
    return pack, decisions
