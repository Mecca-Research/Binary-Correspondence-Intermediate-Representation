"""BCIR verifier: the runnable LangRef laws R1-R12.

LLVM verifies IR structure; BCIR verifies execution truth. The laws attach to the
four artifacts of the correspondence chain, one entry point per artifact:

    verify(module)                       R1-R8  module / claim laws
    verify_plan(module, result)          R8-R9  K_BCIR plan laws
    verify_pack(module, pack)            R10-R11 GEM stream laws
    verify_lowering(module, result, ll)  R12    lowering-contract law
    verify_all(...)                      the whole chain

Mirrored by the MLIR `-bcir-verify` pass (docs/PARITY.md): the structurally
checkable form of each law runs on the dialect; this oracle is the conformance
reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..model import Domain, Lane, Module, Opcode, StrideClass


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

# Contract mnemonics (LangRef Sec. 5; mirror BCIRAttrs.td).
_HAZARDS = {"unique", "atomic", "barriered"}
_BOUNDS = {"strict", "masked", "assumed_safe"}
_VERIFY = {"none", "bounds", "exact", "hash"}
_COST_CLASSES = {"latency", "bandwidth", "compute"}

# Opcodes whose semantics are atomic read-modify-write (R5).
_ATOMIC_OPCODES = {Opcode.ATOMIC_ADD, Opcode.ATOMIC_SUB, Opcode.ATOMIC_XOR, Opcode.CMPXCHG}
# Control/provenance opcodes realized on the H lane (legal for any plan).
_CONTROL_OPCODES = {Opcode.NOP, Opcode.PHASE_ENTER, Opcode.PHASE_LEAVE, Opcode.PROV_NOTE,
                    Opcode.BARRIER}

# Access patterns whose touched index set is data-dependent (R7): a strict bounds
# contract cannot be discharged statically, so a runtime verify contract is required.
_DATA_DEPENDENT = {StrideClass.CACHELINE, StrideClass.RANDOM}


def verify(module: Module) -> list[Diagnostic]:
    """Module/claim laws R1-R8 (the static half of R8: cost-class completeness)."""
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

    # R3: domain legality -- claim domain contracts correspond to registry placement.
    for res in module.resources.values():
        if res.access == "ham" and res.domain == Domain.MMIO:
            diags.append(Diagnostic(
                "R3", f"resource {res.rid}: HAM access is illegal in the MMIO domain"))
    for ph in module.phases:
        for claim in ph.claims:
            touched = [module.resource(rid) for rid in claim.io_rids()]
            touched = [r for r in touched if r is not None]
            if touched and claim.domain not in {r.domain for r in touched}:
                diags.append(Diagnostic(
                    "R3",
                    f"claim {claim.id}: declares domain {claim.domain.name} but touches only "
                    f"{{{', '.join(sorted({r.domain.name for r in touched}))}}}",
                ))
            for rid in claim.wr:
                res = module.resource(rid)
                if res is not None and res.domain == Domain.MMIO and claim.hazard == "unique":
                    diags.append(Diagnostic(
                        "R3",
                        f"claim {claim.id}: MMIO write to RID {rid} requires an "
                        f"atomic/barriered hazard contract",
                    ))

    # R4: phase DAG legality (acyclic).
    if _has_cycle(module):
        diags.append(Diagnostic("R4", "phase dependency graph contains a cycle"))

    # R5: hazard legality -- the hazard contract is sufficient for the declared semantics.
    for ph in module.phases:
        for claim in ph.claims:
            if claim.hazard not in _HAZARDS:
                diags.append(Diagnostic(
                    "R5", f"claim {claim.id}: unknown hazard contract {claim.hazard!r}"))
                continue
            if claim.opcode in _ATOMIC_OPCODES and claim.hazard == "unique":
                diags.append(Diagnostic(
                    "R5",
                    f"claim {claim.id}: atomic opcode {claim.opcode.name} requires an "
                    f"atomic/barriered hazard contract",
                ))
            if claim.lane == Lane.A and claim.hazard == "unique":
                diags.append(Diagnostic(
                    "R5",
                    f"claim {claim.id}: atomic lane A requires an atomic/barriered "
                    f"hazard contract",
                ))
        # CT2 decoupling soundness: the GGG/random tail executes decoupled from the
        # wave order, so a same-phase conflict touching a sparse claim loses its
        # implicit serialization -- both ends must carry an ordered hazard contract.
        claims = ph.claims
        for i, a in enumerate(claims):
            for b in claims[i + 1:]:
                if not _conflict(a, b):
                    continue
                if not (_is_sparse(a) or _is_sparse(b)):
                    continue
                for c in (a, b):
                    if c.hazard == "unique":
                        diags.append(Diagnostic(
                            "R5",
                            f"claim {c.id}: conflicts across the decoupled GGG tail in "
                            f"phase {ph.phase_id} without an atomic/barriered hazard",
                        ))

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

    # R7: bounds legality -- strict bounds are discharged statically (affine
    # patterns) or guarded by a runtime verify contract (data-dependent patterns).
    for ph in module.phases:
        for claim in ph.claims:
            if claim.bounds not in _BOUNDS:
                diags.append(Diagnostic(
                    "R7", f"claim {claim.id}: unknown bounds mode {claim.bounds!r}"))
                continue
            if claim.verify not in _VERIFY:
                diags.append(Diagnostic(
                    "R7", f"claim {claim.id}: unknown verify contract {claim.verify!r}"))
                continue
            if claim.bounds != "strict":
                continue
            if claim.stride_class in _DATA_DEPENDENT:
                if claim.verify == "none":
                    diags.append(Diagnostic(
                        "R7",
                        f"claim {claim.id}: data-dependent {claim.stride_class.name} access "
                        f"with strict bounds requires a runtime verify contract",
                    ))
                continue
            # Affine pattern: the touched extent is statically known. The stride
            # applies to the streamed read source; writes land unit-stride (a
            # conservative under-approximation -- never a false positive).
            k = max(1, claim.stride_k)
            read_extent = claim.offset + (claim.count - 1) * k + 1 if claim.count > 0 else 0
            write_extent = claim.offset + claim.count
            for rid, extent, kind in (
                [(r, read_extent, "read") for r in claim.rd]
                + [(w, write_extent, "write") for w in claim.wr]
            ):
                res = module.resource(rid)
                if res is None or not res.shape:
                    continue
                if extent > res.count:
                    diags.append(Diagnostic(
                        "R7",
                        f"claim {claim.id}: {kind} of RID {rid} overruns the resource "
                        f"(extent {extent} > {res.count})",
                    ))

    # R8 (static half): cost completeness -- every claim names a known cost class.
    for ph in module.phases:
        for claim in ph.claims:
            if claim.cost_class not in _COST_CLASSES:
                diags.append(Diagnostic(
                    "R8", f"claim {claim.id}: unknown cost class {claim.cost_class!r}"))

    return diags


def verify_plan(module: Module, result) -> list[Diagnostic]:
    """K_BCIR plan laws R8 (cost completeness) and R9 (plan legality).

    `result` is a `kbcir.realize.RealizationResult` (duck-typed to keep the
    verifier dependency-free).
    """
    diags: list[Diagnostic] = []
    claims = {c.id: c for ph in module.phases for c in ph.claims}

    seen: set[int] = set()
    total = 0
    for step in result.steps:
        claim = claims.get(step.claim_id)
        if claim is None:
            diags.append(Diagnostic("R9", f"plan realizes unknown claim {step.claim_id}"))
            continue
        if step.claim_id in seen:
            diags.append(Diagnostic("R9", f"claim {step.claim_id} realized more than once"))
        seen.add(step.claim_id)

        cand = step.candidate
        # R8: every realized step carries a complete, non-negative scalarized cost.
        if len(cand.base.v) != 12:
            diags.append(Diagnostic(
                "R8", f"claim {step.claim_id}: candidate cost vector is not 12-d"))
        if step.cost < 0:
            diags.append(Diagnostic(
                "R8", f"claim {step.claim_id}: negative realized cost {step.cost}"))
        total += step.cost

        # R9: the chosen realization is legal for the claim's declared geometry.
        if cand.lane == Lane.H:
            if claim.opcode not in _CONTROL_OPCODES and claim.stride_class != StrideClass.SCALAR:
                diags.append(Diagnostic(
                    "R9",
                    f"claim {step.claim_id}: H-lane realization {cand.name!r} for a "
                    f"non-control claim",
                ))
        elif cand.lane not in _LEGAL_LANES.get(claim.stride_class, set()):
            diags.append(Diagnostic(
                "R9",
                f"claim {step.claim_id}: chosen lane {cand.lane.name} illegal for "
                f"stride_class {claim.stride_class.name}",
            ))

    # R9: total coverage -- a plan must realize every claim exactly once.
    for cid in claims:
        if cid not in seen:
            diags.append(Diagnostic("R9", f"plan does not realize claim {cid}"))

    # R9: the reported score is the sum of the realized step costs.
    if result.steps and total != result.score:
        diags.append(Diagnostic(
            "R9", f"plan score {result.score} != sum of step costs {total}"))

    # R9: steps follow the topological phase order.
    pos = {pid: i for i, pid in enumerate(_topo_phase_ids(module))}
    last = -1
    for step in result.steps:
        p = pos.get(step.phase_id, -1)
        if p < last:
            diags.append(Diagnostic(
                "R9", f"claim {step.claim_id}: realized out of phase order"))
            break
        last = max(last, p)

    return diags


def verify_pack(module: Module, pack) -> list[Diagnostic]:
    """GEM stream laws R10 (provenance) and R11 (generation validity).

    `pack` is a `gem.streampack.StreamPack` (duck-typed).
    """
    diags: list[Diagnostic] = []
    claims = {c.id for ph in module.phases for c in ph.claims}
    traced = {t.claim_id for t in pack.trace_notes}
    prefetches = {p.name for p in pack.prefetches}

    # R10: stream structure -- v2 pipeline/double-buffer contracts are well-formed.
    if getattr(pack, "pipeline_depth", 1) < 1:
        diags.append(Diagnostic(
            "R10", f"invalid pipeline_depth {pack.pipeline_depth} (must be >= 1)"))
    for pf in pack.prefetches:
        if getattr(pf, "buffers", 1) not in (1, 2):
            diags.append(Diagnostic(
                "R10", f"prefetch {pf.name}: invalid buffer count {pf.buffers} (1 or 2)"))

    # R10: stream provenance -- every segment maps back to a live BCIR claim.
    for seg in pack.segments:
        if seg.claim_id not in traced:
            diags.append(Diagnostic(
                "R10", f"segment {seg.name}: no trace note for claim {seg.claim_id}"))
        if seg.claim_id not in claims:
            diags.append(Diagnostic(
                "R10", f"segment {seg.name}: references unknown claim {seg.claim_id}"))
        for rid in tuple(seg.reads) + tuple(seg.writes):
            if module.resource(rid) is None:
                diags.append(Diagnostic(
                    "R10", f"segment {seg.name}: references undeclared RID {rid}"))
        if seg.prefetch is not None and seg.prefetch not in prefetches:
            diags.append(Diagnostic(
                "R10", f"segment {seg.name}: undeclared prefetch {seg.prefetch!r}"))

    # R11: generation validity -- the pack's tags match the live registry. A
    # mismatch is a stale pack: rehydrate (keep/patch/repack/replan,
    # kbcir.calibrate.rehydrate_decide), never execute silently.
    if pack.topo_gen < 1:
        diags.append(Diagnostic("R11", f"invalid topo_gen {pack.topo_gen} (must be >= 1)"))
    reg_map = max((r.map_gen for r in module.resources.values()), default=0)
    reg_data = max((r.data_gen for r in module.resources.values()), default=0)
    if pack.map_gen != reg_map:
        diags.append(Diagnostic(
            "R11",
            f"stale StreamPack: map_gen {pack.map_gen} != registry {reg_map} "
            f"(rehydrate: repack)",
        ))
    if pack.data_gen != reg_data:
        diags.append(Diagnostic(
            "R11",
            f"stale StreamPack: data_gen {pack.data_gen} != registry {reg_data} "
            f"(rehydrate: replan)",
        ))

    return diags


# The textual instruction surface emit_kernel_ll may legally produce (R12: no
# invented opcodes; the emitter is legal-IR-only).
_LEGAL_RESULT_OPS = {"phi", "getelementptr", "load", "icmp",
                     "add", "sub", "mul", "fadd", "fsub", "fmul"}
_LEGAL_STMT_OPS = {"store", "br", "ret", "fence"}
_RESULT_RE = re.compile(r"^%[\w.]+\s*=\s*(\w+)")
_GUARD_RE = re.compile(r"icmp\s+\w+\s+i64\s+%\w+,\s*%n\b")
_WIDTH_RE = re.compile(r"\bwidth=(\d+)\b")


def verify_lowering(module: Module, result, ll_text: str, elem: str = "f32",
                    width_override: int | None = None) -> list[Diagnostic]:
    """Lowering law R12: the emitted LLVM IR preserves the BCIR semantic (lane
    geometry, bounds, hazard, precision) or carries an explicit discharge note.

    Checks the textual kernel produced by `lower.llvm.emit_kernel_ll` against the
    K_BCIR-selected realization. The head comment is the discharge record: it must
    declare the realized width, and that width must be the selected one or the
    documented scalar legalization (count not divisible by the vector width).
    """
    from ..lower.llvm import _FOP, _IOP, find_elementwise
    from ..lower.memory_model import hazard_to_ordering

    diags: list[Diagnostic] = []
    try:
        claim, cand = find_elementwise(module, result)
    except NotImplementedError:
        return [Diagnostic("R12", "no lowerable elementwise claim selected in this plan")]

    # Discharge note: the head comment carries the realized lane geometry.
    head = ll_text.splitlines()[0] if ll_text else ""
    if not head.startswith("; BCIR -> LLVM IR"):
        return [Diagnostic("R12", "missing lowering discharge note (head comment)")]
    m = _WIDTH_RE.search(head)
    if m is None:
        return [Diagnostic("R12", "discharge note does not declare a width")]
    declared_w = int(m.group(1))

    # Lane geometry: the realized width is the K_BCIR-selected width, or the
    # documented scalar legalization when the trip count is not divisible.
    n = max(1, claim.count)
    base_w = width_override if width_override else cand.width
    expected_w = base_w if (base_w >= 1 and n % base_w == 0) else 1
    if declared_w != expected_w:
        diags.append(Diagnostic(
            "R12",
            f"lane geometry not preserved: realized width {declared_w} != "
            f"selected width {expected_w} (candidate {cand.name})",
        ))

    # Precision + geometry of the kernel op: the compute instruction operates on
    # the contracted element type at the realized width.
    ety = "i32" if elem == "i32" else "float"
    op_ll = _IOP[claim.opcode] if elem == "i32" else _FOP[claim.opcode][0]
    kernel_ty = f"<{declared_w} x {ety}>" if declared_w > 1 else ety
    if f"{op_ll} {kernel_ty}" not in ll_text:
        diags.append(Diagnostic(
            "R12", f"precision not preserved: kernel op '{op_ll} {kernel_ty}' not emitted"))
    if declared_w == 1 and f"x {ety}>" in ll_text:
        diags.append(Diagnostic(
            "R12", "lane geometry not preserved: vector types in a scalar lowering"))

    # No invented instructions: every emitted instruction is in the legal set.
    for raw in ll_text.splitlines():
        s = raw.strip()
        if (not s or s.startswith(";") or s.startswith("source_filename")
                or s.startswith("define") or s == "}" or s.endswith(":")):
            continue
        rm = _RESULT_RE.match(s)
        op = rm.group(1) if rm else s.split()[0]
        if op not in (_LEGAL_RESULT_OPS | _LEGAL_STMT_OPS):
            diags.append(Diagnostic(
                "R12", f"instruction {op!r} outside the legal lowering set"))

    # Bounds: a strict bounds contract is discharged by the trip-count guard.
    if claim.bounds == "strict" and not _GUARD_RE.search(ll_text):
        diags.append(Diagnostic(
            "R12", "strict bounds contract not discharged (no trip-count guard on %n)"))

    # Hazard: an ordered hazard contract must materialize as a fence.
    ordering = hazard_to_ordering(claim.hazard)
    if ordering in ("acq_rel", "seq_cst") and "fence" not in ll_text:
        diags.append(Diagnostic(
            "R12",
            f"hazard contract {claim.hazard!r} requires a fence (>= {ordering}) "
            f"in the lowered kernel",
        ))

    return diags


def verify_all(module: Module, result=None, pack=None, ll_text: str | None = None,
               elem: str = "f32", width_override: int | None = None) -> list[Diagnostic]:
    """Run the full R1-R12 chain over every artifact provided."""
    diags = verify(module)
    if result is not None:
        diags += verify_plan(module, result)
    if pack is not None:
        diags += verify_pack(module, pack)
    if ll_text is not None and result is not None:
        diags += verify_lowering(module, result, ll_text, elem, width_override)
    return diags


def is_legal(module: Module) -> bool:
    return not verify(module)


def _conflict(a, b) -> bool:
    """A read/write hazard between two claims (RAW / WAR / WAW)."""
    aw, ar = set(a.wr), set(a.rd)
    bw, br = set(b.wr), set(b.rd)
    return bool(aw & (br | bw)) or bool(bw & ar)


def _is_sparse(c) -> bool:
    return c.lane == Lane.GGG or c.stride_class == StrideClass.RANDOM


def _topo_phase_ids(module: Module) -> list[int]:
    pmap = module.phase_map()
    color: dict[int, int] = {}
    order: list[int] = []

    def visit(pid: int) -> None:
        color[pid] = 1
        for d in pmap[pid].deps:
            if d in pmap and color.get(d, 0) == 0:
                visit(d)
        color[pid] = 2
        order.append(pid)

    for p in module.phases:
        if color.get(p.phase_id, 0) == 0:
            visit(p.phase_id)
    return order


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
