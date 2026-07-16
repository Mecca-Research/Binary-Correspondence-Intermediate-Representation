"""A2 (driver roadmap Part VII): DMA PROGRAMMING FROM STRIDED VIEWS -- D-R4's other half.

The stride vector already carries everything a scatter-gather descriptor needs; this
module makes that literal. `dma_descriptors(src, dst, man)` compiles an admissible
`StridedView` PAIR into a descriptor ring: innermost stride-1 elements coalesce into
contiguous runs (one descriptor per run), adjacent runs merge when BOTH sides stay
contiguous (a fully-contiguous copy is ONE descriptor), and a non-unit innermost
stride degrades honestly to per-element descriptors (scatter is scatter). A
`mem.move.*` claim is the WHAT (D-R2's explicit cast); the descriptor program is the
HOW -- the lowering the exemption audit (B2) waits on.

`dma_cost` prices the transfer from the manifest (D-R3: `move_cost`, distance-priced,
never guessed) plus a per-descriptor setup charge -- a fragmented view is honestly
more expensive than a coalesced one of the same bytes (pinned in the tests).

`dma_transfer_module` composes the WHOLE seam into one planned module:
phase 0 programs the ring; phase 1 ARMS the completion line (EV2's explicit
irq.unmask) and rings the doorbell (`dma.kick:<engine>`, one atomic volatile MMIO
write); the engine's work manifests in a COMPLETION EVENT PHASE (source
`dma.done:<engine>` -- A1 is the completion story) carrying the explicit `mem.move.*`
claim and the done-flag write; the consumer reads the flag inside a masked window
(EV3/B1). D-R2 + D-R3 + D-R4 + A1 + B1, one module, R-law clean.

Refusals (D-R4 live): both views must `check_strided_view`-admit against the manifest;
shapes and element widths must MATCH (a truncated or re-typed transfer is a lie);
same-bank overlapping ranges refuse (in-place DMA is the corruption shape).
Cost-side module: imports no verifier (two-truth); the tests verify."""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from .device_manifest import (DeviceManifest, StridedView, check_strided_view,
                              move_claim, move_cost)
from .events import mask_claim, unmask_claim

_DESC_SETUP = 16                       # Q8-ish per-descriptor programming charge


@dataclass(frozen=True)
class Descriptor:
    """One scatter-gather entry: a contiguous byte run, source offset -> destination
    offset. What every real DMA engine's ring slot carries, minus the device wrapping."""

    src_off: int
    dst_off: int
    nbytes: int


def _runs(view: StridedView) -> list:
    """The view's contiguous byte runs in walk order: (offset, nbytes) pairs. A unit
    innermost stride coalesces the last dimension; anything else is per-element."""
    eb = view.elem_bytes
    unit = view.strides[-1] == 1
    run_elems = view.shape[-1] if unit else 1
    outer_shape = view.shape[:-1] if unit else view.shape
    outer_strides = view.strides[:-1] if unit else view.strides
    runs: list = []

    def walk(dim: int, off: int) -> None:
        if dim == len(outer_shape):
            runs.append((off, run_elems * eb))
            return
        for i in range(outer_shape[dim]):
            walk(dim + 1, off + i * outer_strides[dim] * eb)

    walk(0, view.offset_bytes)
    return runs


def _span(view: StridedView) -> tuple:
    last = view.offset_bytes + sum((d - 1) * s for d, s in
                                   zip(view.shape, view.strides)) * view.elem_bytes \
        + view.elem_bytes
    return view.offset_bytes, last


def dma_descriptors(src: StridedView, dst: StridedView, man: DeviceManifest) -> list:
    """Compile the view pair into the descriptor ring (refusing first -- D-R4 live).
    Runs pair positionally (identical shapes required), then adjacent descriptors merge
    while BOTH sides remain contiguous."""
    for name, v in (("src", src), ("dst", dst)):
        bad = check_strided_view(v, man)
        if bad:
            raise ValueError(f"dma {name} view refused: " + "; ".join(bad))
    if tuple(src.shape) != tuple(dst.shape):
        raise ValueError(f"dma shape mismatch: src {tuple(src.shape)} != dst "
                         f"{tuple(dst.shape)} (a truncated transfer is a lie)")
    if src.elem_bytes != dst.elem_bytes:
        raise ValueError(f"dma element width mismatch: src {src.elem_bytes} != dst "
                         f"{dst.elem_bytes} (a re-typed transfer is a lie)")
    if src.bank == dst.bank:
        (a0, a1), (b0, b1) = _span(src), _span(dst)
        if a0 < b1 and b0 < a1:
            raise ValueError(f"dma overlap on bank {src.bank}: src [{a0},{a1}) vs dst "
                             f"[{b0},{b1}) (in-place DMA is the corruption shape)")
    # The two-pointer byte walk: runs on the two sides need not align (a scattering
    # source against a coalescing destination), so each descriptor takes the SMALLER
    # remainder of the two current runs -- total bytes match by construction (same
    # shape x width), so both sides drain together.
    sruns, druns = _runs(src), _runs(dst)
    descs: list = []
    i = j = si = dj = 0                # si/dj: bytes consumed within the current run
    while i < len(sruns) and j < len(druns):
        (so, sn), (do, dn) = sruns[i], druns[j]
        n = min(sn - si, dn - dj)
        top = descs[-1] if descs else None
        if (top and top.src_off + top.nbytes == so + si
                and top.dst_off + top.nbytes == do + dj):
            descs[-1] = Descriptor(top.src_off, top.dst_off, top.nbytes + n)
        else:
            descs.append(Descriptor(so + si, do + dj, n))
        si += n
        dj += n
        if si == sn:
            i, si = i + 1, 0
        if dj == dn:
            j, dj = j + 1, 0
    if i != len(sruns) or j != len(druns) or si or dj:
        raise ValueError("dma run accounting mismatch; refusing a partial descriptor program")
    return descs


def dma_cost(man: DeviceManifest, src: StridedView, dst: StridedView, descs: list,
             *, desc_setup: int = _DESC_SETUP) -> int:
    """D-R3 pricing: the manifest's distance matrix prices the bytes; each descriptor
    adds a setup charge -- fragmentation costs, honestly."""
    if not isinstance(desc_setup, int) or isinstance(desc_setup, bool) or desc_setup < 0:
        raise ValueError(f"descriptor setup cost must be a non-negative integer; "
                         f"got {desc_setup!r}")
    for index, desc in enumerate(descs):
        if (not isinstance(desc, Descriptor)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in (desc.src_off, desc.dst_off, desc.nbytes))
                or desc.src_off < 0 or desc.dst_off < 0 or desc.nbytes < 1):
            raise ValueError(f"invalid DMA descriptor {index}: {desc!r}")
    canonical = dma_descriptors(src, dst, man)
    if list(descs) != canonical:
        raise ValueError("DMA descriptor program does not exactly lower the supplied views")
    total = sum(d.nbytes for d in descs)
    return move_cost(man, src.bank, dst.bank, total) + desc_setup * len(descs)


# RIDs of the transfer module (a closed universe, the train_graph convention).
_SRCB, _DSTB, _RING, _CTRL, _DONE = 1, 2, 3, 4, 5


def dma_transfer_module(man: DeviceManifest, src: StridedView, dst: StridedView,
                        *, engine: str = "dma0") -> Module:
    """One DMA transfer as a planned, law-clean module -- the composition proof:
    program the ring (D-R4 views compiled to descriptors), arm + kick (EV2's explicit
    unmask; the doorbell is one atomic volatile MMIO write), completion as an EVENT
    PHASE (A1) carrying the explicit `mem.move.*` claim (D-R2, D-R3-priced) and the
    done flag, and the consumer draining the flag under mask (EV3/B1)."""
    descs = dma_descriptors(src, dst, man)
    nbytes = sum(d.nbytes for d in descs)
    line = f"dma.done:{engine}"
    m = Module(name=f"dma_{src.bank}_to_{dst.bank}_{engine}")
    eb = src.elem_bytes
    for r in (Resource(rid=_SRCB, domain=man.bank(src.bank).domain,
                       shape=(max(1, nbytes // eb),), name="SRCB"),
              Resource(rid=_DSTB, domain=man.bank(dst.bank).domain,
                       shape=(max(1, nbytes // eb),), name="DSTB"),
              Resource(rid=_RING, domain=Domain.RAM, shape=(len(descs), 3), name="RING"),
              Resource(rid=_CTRL, domain=Domain.MMIO, shape=(1,), name="CTRL"),
              Resource(rid=_DONE, domain=Domain.RAM, shape=(1,), name="DONE")):
        m.add_resource(r)
    program = Claim(id=1, opcode=Opcode.STORE, lane=Lane.U, stride_class=StrideClass.UNIT,
                    count=max(1, len(descs)), wr=(_RING,),
                    op=f"dma.program:{src.bank}->{dst.bank}", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=0, deps=(), claims=[program]))
    kick = Claim(id=3, opcode=Opcode.STORE, lane=Lane.U, stride_class=StrideClass.SCALAR,
                 count=1, rd=(_RING,), wr=(_CTRL,), op=f"dma.kick:{engine}",
                 domain=Domain.MMIO, hazard="atomic", volatile=True)
    m.add_phase(Phase(phase_id=1, deps=(0,),
                      claims=[unmask_claim(line, _CTRL, cid=2), kick]))
    mv = move_claim(man, src.bank, dst.bank, _SRCB, _DSTB, nbytes, cid=100)
    flag = Claim(id=101, opcode=Opcode.STORE, lane=Lane.U,
                 stride_class=StrideClass.SCALAR, count=1, wr=(_DONE,),
                 op="dma.complete", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=10, deps=(), claims=[mv, flag], event=line))
    poll = Claim(id=5, opcode=Opcode.LOAD, lane=Lane.U, stride_class=StrideClass.SCALAR,
                 count=1, rd=(_DONE,), wr=(_RING,), op="dma.reap", domain=Domain.RAM)
    m.add_phase(Phase(phase_id=2, deps=(1,),
                      claims=[mask_claim(line, _CTRL, cid=4), poll,
                              unmask_claim(line, _CTRL, cid=6)]))
    return m
