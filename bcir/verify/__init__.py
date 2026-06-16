"""BCIR verifier: the runnable LangRef laws R1-R13.

LLVM verifies IR structure; BCIR verifies execution truth. The laws attach to the
artifacts of the correspondence chain, one entry point per artifact:

    verify(module)                       R1-R8  module / claim laws
    verify_plan(module, result)          R8-R9  K_BCIR plan laws
    verify_pack(module, pack)            R10-R11 GEM stream laws
    verify_lowering(module, result, ll)  R12    lowering-contract law
    verify_provenance(portfolio, ...)    R13    policy/table provenance law
    verify_smart_lowering(module, ...)   R14-16 CIM dispatch / DVFS clock / alloc tier
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
            # conservative under-approximation -- never a false positive). A
            # reduction (op "reduce.*") accumulates count reads into a single
            # location, so its write extent is one element, not count.
            k = max(1, claim.stride_k)
            read_extent = claim.offset + (claim.count - 1) * k + 1 if claim.count > 0 else 0
            is_reduction = claim.op.startswith("reduce.")
            write_extent = claim.offset + (1 if is_reduction else claim.count)
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


def verify_support_preservation(source, target, mapping=None) -> list[Diagnostic]:
    """Objective-support law R12 (refinement): a mapping function must preserve
    where the objective matters -- `f(Supp(J)) ⊆ Supp(J')`. A lowering may
    sharpen, rescale, or fuse a cost dimension, but it may not silently *drop* one
    that was nonzero (lose the thermal/security/accuracy/verification term) unless
    `mapping` explicitly discharges it (the same escape R12 grants bounds/hazard/
    precision). The objective's footprint is an invariant of legal lowering.

    `source`/`target` are `kbcir.cost.CostVector`s (J, J'); `mapping` a
    `kbcir.mapping.MappingFunction` (defaults to the identity dimension map with
    no discharges).
    """
    from ..kbcir.mapping import MappingFunction

    f = mapping or MappingFunction(name="identity")
    diags: list[Diagnostic] = []
    for d in sorted(f.dropped(source, target)):
        tgt = f.dim_map.get(d, d)
        diags.append(Diagnostic(
            "R12",
            f"objective support not preserved: source dimension {d!r} is nonzero "
            f"but maps to {tgt!r} which the target drops, with no discharge "
            f"(map {f.name!r})",
        ))
    return diags


def verify_commutativity(square, inputs, eq=None) -> list[Diagnostic]:
    """Commutativity / path-independence law R12 (parity): two conversion paths
    that reach the same target must agree -- `Λ ∘ Ψ = Φ`. The PARITY/manifest
    discipline generalized to any representation rail: a result may not depend on
    which legal path produced it. `square` is a `kbcir.mapping.CommutingSquare`.
    """
    diags: list[Diagnostic] = []
    for x in square.mismatches(inputs, eq):
        diags.append(Diagnostic(
            "R12",
            f"conversion paths disagree on {x!r}: lam(psi(x)) != phi(x) -- the "
            f"square {square.name!r} does not commute (Λ∘Ψ ≠ Φ)",
        ))
    return diags


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


def verify_c_lowering(module: Module, result, c_text: str, elem: str = "f32",
                      width_override: int | None = None,
                      hw_width: int | None = None) -> list[Diagnostic]:
    """Lowering law R12 for the portable C backend: the emitted C23 kernel
    (`lower.c_kernel.emit_kernel_c`) preserves the K_BCIR-selected realization --
    lane geometry, precision (the contracted element type), bounds (a trip-count
    guard so the kernel is safe for any n), the elementwise op, and the
    non-aliasing contract (`restrict` when the read and write resources are
    disjoint).

    Lane geometry is *width-aware* (`hw_width` = the target's widest lane,
    `HProfile.vector_width`): the lowering must **honor** the selected width, which
    means different things at the full lane vs a throttle. (a) `w == 1`: a scalar
    loop, no vector cap. (b) `w == hw_width` (go-fast): the idiomatic loop is
    correct -- the compiler realizes >= the lane -- so the kernel must NOT cap below
    the lane (a hidden sub-width cap would throttle the backend). (c) `1 < w <
    hw_width` (a deliberate thermal/power throttle): the kernel MUST cap at `w`
    (a fixed-trip width-`w` loop) so the compiler cannot widen past the sub-maximal
    lane. With no `hw_width` the check conservatively requires the cap for `w > 1`
    (the literal geometry-encoding form -- unchanged from before)."""
    from ..gem import hydrate
    from ..lower.c_kernel import C_OP, _ctype
    from ..lower.llvm import find_elementwise

    diags: list[Diagnostic] = []
    try:
        claim, cand = find_elementwise(module, result)
    except NotImplementedError:
        return [Diagnostic("R12", "no lowerable elementwise claim selected in this plan")]

    head = c_text.splitlines()[0] if c_text else ""
    if not head.startswith("/* BCIR -> portable C"):
        return [Diagnostic("R12", "missing C lowering discharge note (head comment)")]
    m = re.search(r"\bwidth=(\d+)\b", head)
    if m is None:
        return [Diagnostic("R12", "C discharge note does not declare a width")]
    declared_w = int(m.group(1))

    w = int(width_override) if width_override else int(cand.width)
    w = w if w >= 1 else 1
    if declared_w != w:
        diags.append(Diagnostic(
            "R12",
            f"lane geometry not preserved: declared width {declared_w} != selected "
            f"width {w} (candidate {cand.name})"))

    # Lane geometry in the body, width-aware (see the docstring).
    full_lane = hw_width is not None and w == int(hw_width)
    if w == 1:
        if re.search(r"<\s*\d+u", c_text):
            diags.append(Diagnostic(
                "R12", "lane geometry not preserved: a vector-width loop in a scalar kernel"))
    elif full_lane:
        # Go-fast: the idiomatic loop realizes >= the full lane. Geometry is the
        # recorded width (checked above) + a bounds-safe loop; a cap *below* the
        # lane would secretly throttle the backend, so reject it.
        sub_caps = [int(x) for x in re.findall(r"<\s*(\d+)u", c_text) if int(x) < w]
        if sub_caps:
            diags.append(Diagnostic(
                "R12", f"lane geometry not preserved: full-width kernel caps below the "
                       f"hardware lane (found width-{min(sub_caps)} cap, lane is {w})"))
    else:
        # Deliberate sub-maximal throttle: the cap MUST be physically honored, else
        # the compiler widens past the selected (thermal/power-limited) lane.
        if f"< {w}u" not in c_text:
            diags.append(Diagnostic(
                "R12", f"lane geometry not preserved: no width-{w} throttle cap emitted"))

    # Precision: the contracted element type on the kernel signature.
    ctype = _ctype(elem)
    if f"const {ctype} " not in c_text:
        diags.append(Diagnostic(
            "R12", f"precision not preserved: kernel does not operate on {ctype}"))

    # The elementwise op.
    op = C_OP.get(claim.opcode)
    if op and f"] {op} B" not in c_text and f"] {op} B[i]" not in c_text:
        diags.append(Diagnostic(
            "R12", f"op not preserved: elementwise '{op}' not emitted"))

    # Bounds: a trip-count guard on n (the strict bounds contract -> a loop over n).
    if "< n" not in c_text and "<= n" not in c_text:
        diags.append(Diagnostic(
            "R12", "bounds not preserved: no trip-count guard on n"))

    # Non-aliasing: restrict when the read/write resources are disjoint.
    pack = hydrate(module, result)
    seg = next((s for s in pack.segments if s.claim_id == claim.id), None)
    reads = tuple(seg.reads) if seg else tuple(claim.rd)
    writes = tuple(seg.writes) if seg else tuple(claim.wr)
    if not (set(reads) & set(writes)) and "restrict" not in c_text:
        diags.append(Diagnostic(
            "R12", "aliasing contract not preserved: disjoint operands but no "
            "restrict-qualified pointers"))

    return diags


def verify_manifest(manifest, module, h, theta, policy=None, artifacts=()) -> list[Diagnostic]:
    """Provenance-manifest law R13: a deployed plan's manifest must (a) hash to the
    inputs/artifacts it claims (tamper-evidence) and (b) reproduce the recorded
    optimal score and plan shape (determinism). A failure means the plan is not
    reproducible from its stated provenance -- the debugging/determinism violation.
    """
    from ..kbcir.provenance import build_manifest
    from ..kbcir.weights import PERF

    diags: list[Diagnostic] = []
    fresh = build_manifest(module, h, theta, policy or PERF, artifacts)
    if fresh.digest != manifest.digest:
        diags.append(Diagnostic(
            "R13",
            f"manifest digest {manifest.digest} != recomputed {fresh.digest} "
            f"(changed components: {fresh.diff(manifest)})",
        ))
        return diags
    if fresh.score != manifest.score:
        diags.append(Diagnostic(
            "R13",
            f"manifest is not reproducible: recorded score {manifest.score} != "
            f"replayed {fresh.score}",
        ))
    if fresh.widths != manifest.widths:
        diags.append(Diagnostic(
            "R13", "manifest is not reproducible: replayed plan shape differs"))
    return diags


def verify_memory(mm, generation=None, *, recheck=True) -> list[Diagnostic]:
    """Memory-module fixpoint law R13: an artifact may be frozen and
    generation-tagged as a *memory module* only when it is the fixpoint of
    resolution -- `a = Lim(Res(U))`, `saturated == True` -- never a budget cutoff
    `Res^k(U)`. Freezing a non-saturated extraction pins a non-canonical,
    still-improvable representative as memory and breaks the determinism the
    provenance manifest depends on (two runs that cut off at different `k` need
    not agree). The saturation witness IS the admitting certificate for entry
    into a generation.

    With `recheck` (default), the verifier does not merely trust the recorded
    `saturated` flag: it independently re-resolves the stored representative
    (`Res(Lim(Res(U)))`) and confirms it is a genuine, stable fixpoint -- the
    tamper-evidence analog of `verify_manifest` recomputing the digest.
    """
    from ..kbcir.memory import try_freeze, _costs_for

    diags: list[Diagnostic] = []

    # R13: the fixpoint witness. A budget cutoff is not Lim(Res(U)).
    if not mm.saturated:
        diags.append(Diagnostic(
            "R13",
            f"memory module is not a fixpoint: saturated == False after "
            f"{mm.iterations} round(s) -- a budget cutoff Res^k(U), not "
            f"Lim(Res(U)); not admissible as memory",
        ))

    # R13: a frozen artifact must be generation-tagged (immutable within a gen).
    if mm.generation < 1:
        diags.append(Diagnostic(
            "R13",
            f"memory module has no generation tag (generation {mm.generation}); "
            f"a frozen artifact is immutable within a generation and must carry one",
        ))
    elif generation is not None and mm.generation != generation:
        diags.append(Diagnostic(
            "R13",
            f"memory module generation {mm.generation} != expected {generation}",
        ))

    # R13: independent fixpoint recheck (tamper-evidence). Re-resolving the stored
    # representative must reproduce it, saturated -- idempotence Res(Lim) = Lim.
    if recheck and mm.saturated:
        again = try_freeze(mm.expr, generation=mm.generation, costs=_costs_for(mm.expr))
        if not again.saturated:
            diags.append(Diagnostic(
                "R13",
                "memory module fails the fixpoint recheck: its representative does "
                "not re-resolve to saturation (the recorded witness is unsound)",
            ))
        elif again.expr != mm.expr:
            diags.append(Diagnostic(
                "R13",
                "memory module is not idempotent: re-resolving its representative "
                "yields a different form, so it is not Lim(Res(U)) (tampered or "
                "frozen before saturation)",
            ))
        elif again.fingerprint != mm.fingerprint:
            diags.append(Diagnostic(
                "R13",
                f"memory module fingerprint {mm.fingerprint} != recomputed "
                f"{again.fingerprint} (content does not match its hash)",
            ))

    return diags


def verify_calibration(cert) -> list[Diagnostic]:
    """Calibration-closure law R13: a closed calibration loop is admissible only
    when its frozen table is generation-tagged (`cal_gen >= 1`) and recalibrating
    never regresses -- the win (the measured cost of *not* recalibrating) is
    `>= 0`. A negative win means the "recalibrated" plan is not the optimum under
    the measured model (a broken loop, a stale rescore, or a tampered
    certificate); a `cal_gen < 1` table is an untagged measurement that may not be
    deployed. `cert` is a `kbcir.calibloop.CalibrationCertificate` (duck-typed).
    """
    diags: list[Diagnostic] = []
    if cert.cal_gen < 1:
        diags.append(Diagnostic(
            "R13",
            f"calibration table is not generation-tagged (cal_gen {cert.cal_gen}); "
            f"a measured table must be frozen + tagged before it is deployed",
        ))
    if cert.win < 0:
        diags.append(Diagnostic(
            "R13",
            f"calibration loop regressed: win {cert.win} < 0 (stale cost "
            f"{cert.stale_cost} < recalibrated {cert.calibrated_cost}); the "
            f"recalibrated plan is not optimal under the measured model",
        ))
    return diags


# --- R14-R16: the smart-lowering laws (dual-rail with -bcir-lower-to-llvm) --------
#
# These three already exist as MLIR laws + oracle *gates* (gem.cim / gem.dvfs /
# kbcir.allocator); the functions below add them to the Python verifier so the laws
# are symmetric on both rails (docs/PARITY.md, R14-R16). Each mirrors exactly the
# structural check `mlir/lib/BCIRPasses.cpp` (LowerToLLVMPass) performs.

# R16 on-chip capacity caps -- the static SRAM budgets the MLIR law uses
# (BCIRPasses.cpp): a placement into L1 must fit 64 KiB, into L2 4 MiB. L3/DRAM/HBM
# carry no static cap.
_L1_CAP_BYTES = 64 * 1024
_L2_CAP_BYTES = 4 * 1024 * 1024
# R15 legal DVFS clock range (Q8; 256 == nominal x1.0): 0.25x .. 2x.
_CLOCK_MIN_Q8 = 64
_CLOCK_MAX_Q8 = 512
_CLOCK_NOMINAL_Q8 = 256


def verify_cim(pack) -> list[Diagnostic]:
    """R14 (CIM/PIM dispatch legality): a StreamPack segment dispatched to
    processing-in-memory must be a reduction -- PIM does element-local reduce work,
    not general SIMD. Mirrors the MLIR `-bcir-lower-to-llvm` R14 law
    (`dispatch="pim"` legal only on a `reduce.*` op) and the `gem.cim` gate.
    `pack` is a `gem.streampack.StreamPack` (duck-typed)."""
    diags: list[Diagnostic] = []
    for seg in pack.segments:
        if getattr(seg, "dispatch", "core") == "pim" and not seg.opcode.startswith("reduce."):
            diags.append(Diagnostic(
                "R14",
                f"segment {seg.name}: pim dispatch illegal for non-reduction op "
                f"{seg.opcode!r} (claim {seg.claim_id})",
            ))
    return diags


def verify_dvfs(plan) -> list[Diagnostic]:
    """R15 (DVFS clock legality): every per-phase clock is a legal Q8 step in
    [64, 512] (0.25x..2x), and a memory-bound phase must not overclock -- more core
    frequency cannot speed bandwidth-bound work. Mirrors the MLIR
    `-bcir-lower-to-llvm` R15 law (clock_q8 in [64,512]; a pim/memory-bound segment
    must not overclock) and the `gem.dvfs` gate. `plan` is a `gem.dvfs.DVFSPlan`
    (duck-typed: `.decisions` with `phase_id`, `clock_q8`, `klass`)."""
    diags: list[Diagnostic] = []
    for d in plan.decisions:
        if not (_CLOCK_MIN_Q8 <= d.clock_q8 <= _CLOCK_MAX_Q8):
            diags.append(Diagnostic(
                "R15",
                f"phase {d.phase_id}: clock_q8 {d.clock_q8} out of legal range "
                f"[{_CLOCK_MIN_Q8}, {_CLOCK_MAX_Q8}]",
            ))
        elif d.klass == "memory" and d.clock_q8 > _CLOCK_NOMINAL_Q8:
            diags.append(Diagnostic(
                "R15",
                f"phase {d.phase_id}: memory-bound phase must not overclock "
                f"(clock_q8 {d.clock_q8} > {_CLOCK_NOMINAL_Q8})",
            ))
    return diags


def verify_allocator(module: Module, placement) -> list[Diagnostic]:
    """R16 (allocator placement legality): an on-chip placement must fit -- a
    resource placed in L1 must be <= 64 KiB, in L2 <= 4 MiB (static size =
    count * elem_bytes). The planner may not promote a tensor into an SRAM tier it
    cannot fit. Mirrors the MLIR `-bcir-lower-to-llvm` R16 law and the
    `kbcir.allocator` capacity gate. `placement` is a `kbcir.allocator.Placement`
    (duck-typed: `.tiers` maps rid -> MemTier); L3/DRAM/HBM carry no cap."""
    from ..kbcir.cost import MemTier

    caps = {MemTier.L1: _L1_CAP_BYTES, MemTier.L2: _L2_CAP_BYTES}
    diags: list[Diagnostic] = []
    for rid, tier in placement.tiers.items():
        cap = caps.get(tier)
        if cap is None:
            continue
        res = module.resource(rid)
        if res is None or not res.shape:
            continue  # dynamic/unknown extent: not statically checkable
        # Mirror the MLIR law EXACTLY: BCIRPasses.cpp computes product(shape) * 4
        # (the ResourceOp carries no elem_bytes attr, so the law assumes a 4-byte
        # element). Using 4 here keeps the two rails in lock-step at the cap boundary.
        nbytes = res.count * 4
        if nbytes > cap:
            diags.append(Diagnostic(
                "R16",
                f"placement {tier.name} does not fit RID {rid} "
                f"({nbytes} B > {cap} B)",
            ))
    return diags


def verify_accuracy(module: Module) -> list[Diagnostic]:
    """R17 (accuracy-contract legality): a claim that declares an accuracy tolerance
    (`tolerance_ulp > 0`) must realize within it -- its static worst-case Q8-ULP error
    bound (`precision.accuracy_bound`, compensated iff `precision == "compensated"`) must
    not exceed the declared tolerance. A `reduce.*` over `count` terms drifts up to `count`
    ULP with the naive accumulator but only 1 ULP compensated, so a tight tolerance is the
    law that FORCES the compensated realization (`precision="compensated"`, lowered by
    `lower.c_kernel.emit_compensated_reduce_c`). Dual-rail with the MLIR `-bcir-verify` R17
    law. Claims with no declared tolerance (the default) are unconstrained -- a no-op."""
    from ..kbcir.precision import accuracy_bound

    diags: list[Diagnostic] = []
    for ph in module.phases:
        for claim in ph.claims:
            tol = getattr(claim, "tolerance_ulp", 0)
            if tol <= 0:
                continue
            compensated = getattr(claim, "precision", "") == "compensated"
            bound = accuracy_bound(claim, compensated=compensated)
            if bound > tol:
                diags.append(Diagnostic(
                    "R17",
                    f"claim {claim.id} accuracy bound {bound} ULP exceeds tolerance {tol} "
                    f"ULP (precision={claim.precision or 'naive'!r}; a compensated "
                    f"reduction would bound it at 1)",
                ))
    return diags


def verify_smart_lowering(module: Module, pack=None, dvfs_plan=None,
                          placement=None) -> list[Diagnostic]:
    """Run the smart-lowering laws R14-R17 over whichever artifacts are provided
    (CIM StreamPack, DVFS plan, allocator placement) plus the always-checkable
    accuracy contract (R17) -- the dual-rail counterpart to the MLIR
    `-bcir-lower-to-llvm` / `-bcir-verify` R14/R15/R16/R17 checks."""
    diags: list[Diagnostic] = []
    if pack is not None:
        diags += verify_cim(pack)
    if dvfs_plan is not None:
        diags += verify_dvfs(dvfs_plan)
    if placement is not None:
        diags += verify_allocator(module, placement)
    diags += verify_accuracy(module)
    return diags


def verify_quarantine(verdict_inputs=(), decisions=(), diagnostics=None) -> list[Diagnostic]:
    """Two-truth quarantine law R13 (MOPC): graded truth may inform a verdict but
    never *be* one. The verifier speaks classical truth (deterministic, binary);
    the learned/measured machinery speaks graded truth `(value, confidence)`. A
    graded proposition may cross into a legality verdict only as the classical
    value of a *recorded* `decide` -- never raw, never silently.

    Flags (all R13):
      * any `verdict_inputs` element that is a graded proposition (or a decision
        still wrapping one) -- graded truth reached a classical boundary uncollapsed;
      * any `decisions` collapse that is malformed (confidence/threshold out of the
        Q-milli range) -- a crossing that cannot be audited;
      * any `diagnostics` carrying a confidence -- a verdict is binary; a graded
        verdict is a category error (the verifier must never emit one).
    """
    from ..kbcir.twotruth import Decision, Graded, is_classical

    diags: list[Diagnostic] = []

    # R13: no graded proposition crosses uncollapsed into a verdict input.
    for x in verdict_inputs:
        if not is_classical(x):
            inner = x.value if isinstance(x, Decision) else x
            conf = getattr(inner, "confidence_milli", "?")
            diags.append(Diagnostic(
                "R13",
                f"graded proposition (confidence {conf}/1000, source "
                f"{getattr(inner, 'source', '?')!r}) crossed into a legality "
                f"verdict; collapse it with decide() first",
            ))

    # R13: every recorded collapse is well-formed (auditable).
    for d in decisions:
        if not (0 <= d.confidence_milli <= 1000 and 0 <= d.threshold_milli <= 1000):
            diags.append(Diagnostic(
                "R13",
                f"malformed two-truth collapse (confidence {d.confidence_milli}, "
                f"threshold {d.threshold_milli}); a crossing must be auditable",
            ))
        if not is_classical(d.value):
            diags.append(Diagnostic(
                "R13",
                "two-truth collapse still wraps a graded value; decide() must "
                "yield classical truth",
            ))

    # R13: a verdict is classical -- a Diagnostic may not carry a confidence.
    for diag in (diagnostics or ()):
        if isinstance(diag, Graded) or hasattr(diag, "confidence_milli"):
            diags.append(Diagnostic(
                "R13",
                "a legality verdict carries a confidence; classical truth is "
                "binary (graded verdicts are quarantined out of the laws)",
            ))

    return diags


def verify_enriched(operad, root=None) -> list[Diagnostic]:
    """Enriched-operad integrity law R13 (the higher memory interface): the
    interpretive layer over the memory modules must be self-consistent so it can
    be trusted to *inform* (never legislate). Witnesses, for a
    `kbcir.operad.EnrichedOperad`:

      * **label consistency** -- when labeling is active, every operation carries a
        non-empty hierarchical label;
      * **content-addressed index integrity** -- when indexing is active, every
        operation's index equals `f_index(name, label, children)` (so identical
        operations share an index: CSE / the liked-pair identity, and tamper is
        evident);
      * **mapping integrity** -- every child index resolves (`Trace` never dangles),
        and, if `root` is given, the whole reachable structure is well-formed.

    This is the analog of `verify_memory` for the enriched structure. The labels
    and indexes it guards are interpretive metadata, quarantined out of R1-R12
    (§14): this law checks the memory interface's own integrity, not legality.
    """
    diags: list[Diagnostic] = []
    for index, reason in operad.problems():
        diags.append(Diagnostic("R13", f"enriched operation {index}: {reason}"))
    if root is not None and operad.get(root) is None:
        diags.append(Diagnostic(
            "R13", f"enriched operad root {root} does not resolve"))
    return diags


def verify_provenance(portfolio, certificates=(), h=None, table=None,
                      verdicts=(), gate_certificates=(), accel_certificates=(),
                      amortization_certificates=(), memory_modules=()) -> list[Diagnostic]:
    """Policy/table provenance law R13: every decision rule in force carries a
    generation tag and an admitting certificate -- rule swaps and table
    applications are never silent -- and every boundary verdict carries an MDL
    justification consistent with the recommendation it makes.

    `portfolio` is a `kbcir.portfolio.PolicyPortfolio`, `certificates` an
    iterable of `ReplayCertificate`s, `h` a `TargetProfile`, `table` a
    `kbcir.microbench.CalibratedProfile`, `verdicts` an iterable of
    `kbcir.regret.BoundaryVerdict`, `gate_certificates` the
    `ReplayCertificate`s of learned MoE gates proposed for deployment, and
    `memory_modules` the `kbcir.memory.MemoryModule`s proposed for freezing into
    a generation (all duck-typed).
    """
    diags: list[Diagnostic] = []

    # R13: memory-module provenance -- an e-graph extraction may be frozen and
    # generation-tagged only when it is the resolution fixpoint a = Lim(Res(U))
    # (saturated), never a budget cutoff. The saturation witness is its admitting
    # certificate; this ties the e-graph engine (Phase 21) to the manifest (20).
    for mm in memory_modules:
        diags += verify_memory(mm)

    # R13: learned-gate provenance -- a MoE gate (kbcir.moegate) may deploy only
    # behind an admitting replay certificate (zero regressions vs the incumbent
    # classify router). The network proposes a route; the verifier disposes.
    for cert in gate_certificates:
        if not cert.admitted:
            diags.append(Diagnostic(
                "R13",
                f"learned gate {cert.candidate!r} cannot deploy: replay gate "
                f"not passed ({cert.regressions} regression(s) over "
                f"{cert.episodes} episode(s) vs {cert.incumbent!r})",
            ))

    # R13: search-accelerator provenance -- a learned candidate ordering
    # (kbcir.accel) speeds the exact search but must reproduce the exact optimum.
    # A deployed accelerator carries an equivalence certificate with zero
    # mismatches; ordering changes work, never the result.
    for cert in accel_certificates:
        if not cert.admitted:
            diags.append(Diagnostic(
                "R13",
                f"search accelerator {cert.order_name!r} changed the optimum: "
                f"{cert.mismatches} mismatch(es) over {cert.checked} case(s)",
            ))

    # R13: amortization provenance (the L1 cost throttle) -- a deployed learned
    # component must belong at its tier: the L0 prohibition (no inference on the
    # hot path), it pays for itself (gain >= inference_cost), and it is within the
    # tier's inference budget. Performance is throttled where it matters.
    for cert in amortization_certificates:
        if cert.tier == "L0" and cert.inference_cost != 0:
            diags.append(Diagnostic(
                "R13",
                f"component {cert.component!r} at L0 runs learned inference "
                f"(cost {cert.inference_cost}); the hot path carries decisions, "
                f"not models",
            ))
        elif not cert.admitted:
            diags.append(Diagnostic(
                "R13",
                f"component {cert.component!r} fails amortization at {cert.tier}: "
                f"gain {cert.gain} vs inference_cost {cert.inference_cost}, "
                f"budget {cert._budget()}",
            ))

    # R13: boundary-verdict provenance -- a retune recommendation must be backed
    # by description-length evidence (data_fit > complexity), and a keep must not
    # sit on regret that has already outgrown the model-complexity cost. The
    # dashboard cannot recommend a swap the MDL/evidence does not justify.
    for v in verdicts:
        justified = v.data_fit_nats > v.complexity_nats
        if v.verdict == "retune" and not justified:
            diags.append(Diagnostic(
                "R13",
                f"retune verdict for {v.rule!r} is unjustified: data_fit "
                f"{v.data_fit_nats:.3f} <= complexity {v.complexity_nats:.3f}",
            ))
        if v.verdict == "keep" and justified:
            diags.append(Diagnostic(
                "R13",
                f"keep verdict for {v.rule!r} ignores justified regret: data_fit "
                f"{v.data_fit_nats:.3f} > complexity {v.complexity_nats:.3f}",
            ))

    # R13: gain-schedule provenance -- a promoted (gen > 1) certified entry must
    # be backed by an admitting replay certificate covering exactly that swap.
    if portfolio is not None:
        for slot, entry in sorted(portfolio.entries.items()):
            if not entry.certified:
                continue
            if entry.gen < 1:
                diags.append(Diagnostic(
                    "R13", f"certified policy {slot!r} has no generation tag"))
            if entry.gen > 1:
                covering = [c for c in certificates
                            if c.admitted and c.candidate == entry.policy.name
                            and c.incumbent == slot]
                if not covering:
                    diags.append(Diagnostic(
                        "R13",
                        f"promoted policy {slot!r} (gen {entry.gen}) has no "
                        f"admitting replay certificate",
                    ))

    # R13: cost-table provenance -- a profile claiming calibration must present
    # the certified table it was calibrated under, with matching generation and
    # constants (no silent drift between the table and the rule in force).
    if h is not None and getattr(h, "cal_gen", 0) >= 1:
        if table is None:
            diags.append(Diagnostic(
                "R13",
                f"profile {h.name!r} claims cal_gen {h.cal_gen} but presents "
                f"no certified table",
            ))
        elif table.cal_gen != h.cal_gen:
            diags.append(Diagnostic(
                "R13",
                f"stale calibration: profile cal_gen {h.cal_gen} != table "
                f"cal_gen {table.cal_gen} (recalibrate or rehydrate)",
            ))
        elif (h.gather_penalty, h.base_overhead, h.mem_unit) != (
                table.gather_penalty, table.base_overhead, table.mem_unit):
            diags.append(Diagnostic(
                "R13",
                f"profile constants drifted from the certified table "
                f"(cal_gen {h.cal_gen})",
            ))

    # R13: conformal-guarantee provenance -- a Bayesian-calibrated table that
    # claims a coverage level must carry a valid coverage in (0,1) and a
    # non-negative +/- delta, and it may not certify a finite interval from too
    # few samples (a coverage guarantee from <= 1 observation is not a guarantee).
    if table is not None and getattr(table, "coverage_milli", 0):
        cov = table.coverage_milli
        delta = getattr(table, "random_delta_q8", 0)
        samples = getattr(getattr(table, "point", None), "samples", 0)
        if not (0 < cov < 1000):
            diags.append(Diagnostic(
                "R13", f"conformal coverage {cov}/1000 out of range (0,1000)"))
        if delta < 0:
            diags.append(Diagnostic("R13", "conformal delta must be >= 0"))
        if delta > 0 and samples <= 1:
            diags.append(Diagnostic(
                "R13",
                f"conformal interval (delta {delta}) certified from {samples} "
                f"sample(s): a coverage guarantee needs >= 2 observations",
            ))

    return diags


def verify_all(module: Module, result=None, pack=None, ll_text: str | None = None,
               elem: str = "f32", width_override: int | None = None) -> list[Diagnostic]:
    """Run the R1-R12 chain over every artifact provided (R13 attaches to the
    decision rules, not the module: see `verify_provenance`)."""
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
