"""BCIR verifier (a runnable subset of LangRef laws R1-R12).

LLVM verifies IR structure; BCIR verifies execution truth. This is the oracle's
structural+semantic checker. Deep laws (full hazard proofs, K_BCIR plan legality,
generation freshness) are sketched and grow alongside the dialect verifier.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Lane, Module, StrideClass


@dataclass(frozen=True)
class Diagnostic:
    law: str
    message: str


# Which lanes are legal for a declared access-pattern shape (LangRef R6).
_LEGAL_LANES = {
    StrideClass.SCALAR: {Lane.U, Lane.H},
    StrideClass.UNIT: {Lane.U},
    StrideClass.STRIDED: {Lane.U, Lane.GGG},
    StrideClass.CACHELINE: {Lane.UX, Lane.GGG},
    StrideClass.TILE: {Lane.T},
    StrideClass.RANDOM: {Lane.GGG, Lane.A},
}


def verify(module: Module) -> list[Diagnostic]:
    diags: list[Diagnostic] = []

    # R1: registry uniqueness (RID unique within the module's registry namespace).
    seen: set[int] = set()
    for rid in module.resources:
        if rid in seen:
            diags.append(Diagnostic("R1", f"duplicate RID {rid}"))
        seen.add(rid)

    # R2: registry resolution -- every claim resource reference resolves.
    for ph in module.phases:
        for claim in ph.claims:
            for rid in claim.io_rids():
                if module.resource(rid) is None:
                    diags.append(Diagnostic("R2", f"claim {claim.id} references undeclared RID {rid}"))

    # R4: phase DAG legality (acyclic).
    if _has_cycle(module):
        diags.append(Diagnostic("R4", "phase dependency graph contains a cycle"))

    # R6: lane legality -- lane type matches the declared access pattern.
    for ph in module.phases:
        for claim in ph.claims:
            legal = _LEGAL_LANES.get(claim.stride_class, set())
            if claim.lane not in legal:
                diags.append(Diagnostic(
                    "R6",
                    f"claim {claim.id}: lane {claim.lane.name} illegal for "
                    f"stride_class {claim.stride_class.name}",
                ))

    return diags


def is_legal(module: Module) -> bool:
    return not verify(module)


def _has_cycle(module: Module) -> bool:
    pmap = module.phase_map()
    color: dict[int, int] = {}

    def visit(pid: int) -> bool:
        color[pid] = 1
        for d in pmap.get(pid).deps if pid in pmap else ():
            if d not in pmap:
                continue
            c = color.get(d, 0)
            if c == 1:
                return True
            if c == 0 and visit(d):
                return True
        color[pid] = 2
        return False

    for p in module.phases:
        if color.get(p.phase_id, 0) == 0 and visit(p.phase_id):
            return True
    return False
