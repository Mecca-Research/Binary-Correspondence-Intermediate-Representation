"""Emit BCIR-MLIR textual IR from a planned `Module` (the Python -> law bridge).

The Python package is the executable conformance **oracle**; the MLIR dialect is
the **law** (docs/PARITY.md). Until now the only cross-rail artifacts were *curated*
`.mlir` files hand-pinned to the oracle's constants (e.g. vector_add @ 7808). This
module turns the oracle's plan for *any* `Module` into the same GEM-pipeline IR the
`-bcir-classify-lanes -bcir-select-realization -bcir-batch -bcir-schedule
-bcir-lower-to-llvm` passes consume, so the law can recompute the min-plus argmin
and cross-check it. Paired with `bcir.kbcir.differential` it makes Python<->MLIR
parity *generated*, not curated.

The emitted candidate costs are the oracle's **context-resolved** effective costs:
each claim's candidates are priced under the chosen predecessor (the fusion /
thermal coupling f_i(pi) the oracle's shortest path used), so the law's per-claim
argmin reproduces the oracle's globally-coupled selection and per-claim score. The
`plan_view` structure is the shared, IR-level view both `to_mlir` (emit) and
`differential.law_select` (recompute) read -- one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model import Domain, Lane, Module, StrideClass
from ..kbcir.cost import DIMS, HProfile, Theta
from ..kbcir.realize import (
    Candidate,
    RealizationResult,
    _context_factor,
    _flatten,
    fused_candidates,
)
from ..kbcir.weights import PERF, Policy, weights

# Enum -> MLIR spelling (BCIRAttrs.td; docs/PARITY.md). Normative.
_LANE_SPELL = {Lane.U: "u", Lane.UX: "ux", Lane.T: "t", Lane.GGG: "ggg",
               Lane.A: "a", Lane.H: "h"}
_STRIDE_SPELL = {StrideClass.SCALAR: "scalar", StrideClass.UNIT: "unit",
                 StrideClass.STRIDED: "strided", StrideClass.CACHELINE: "cacheline",
                 StrideClass.TILE: "tile", StrideClass.RANDOM: "random"}
_DOMAIN_SPELL = {Domain.RAM: "ram", Domain.VRAM: "vram", Domain.NVM: "nvm",
                 Domain.MMIO: "mmio", Domain.CXL: "cxl", Domain.HBM: "hbm"}
_POLICY_MODE = {"latency": "latency", "throughput": "throughput",
                "energy": "energy", "safe": "safe"}


@dataclass(frozen=True)
class PathView:
    """One candidate realization path as the IR sees it: a name, a lane, and the
    context-resolved 12-d cost the law scalarizes."""

    name: str
    lane: Lane
    cost: tuple[int, ...]


@dataclass(frozen=True)
class ClaimView:
    claim_id: int
    phase_id: int
    op: str
    lane: Lane
    stride_class: StrideClass
    stride_k: int
    count: int
    reads: tuple[int, ...]
    writes: tuple[int, ...]
    domain: Domain
    hazard: str
    verify: str
    bounds: str
    paths: tuple[PathView, ...]
    selected: str          # the oracle's chosen candidate name
    score: int             # the oracle's per-claim cost (effective cost . weights)
    width: int             # the chosen lane geometry (segment width)


@dataclass(frozen=True)
class PlanView:
    """The IR-level view of a planned module: the policy weights and, in flattened
    (topological phase, declared-claim) order, every claim's candidate set with the
    oracle's selection and per-claim score. The single source `to_mlir` emits and
    `differential.law_select` recomputes from."""

    module_name: str
    policy_mode: str
    weights: tuple[int, ...]
    claims: tuple[ClaimView, ...]
    phase_ids: tuple[int, ...]
    phase_deps: dict[int, tuple[int, ...]]
    total_score: int


def plan_view(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF,
              result: RealizationResult | None = None) -> PlanView:
    """Reconstruct the IR-level candidate view of `module`'s plan under (H, Theta,
    policy). Each claim's candidates are priced under the *chosen predecessor*'s
    context (so the per-claim argmin reproduces the oracle's coupled plan); the
    per-claim score is the chosen candidate's effective cost . weights. Asserts the
    reconstruction sums to the oracle's score (a self-check on drift)."""
    from ..kbcir.realize import optimize  # local: avoid eager optimizer import

    if result is None:
        result = optimize(module, h, theta, policy)
    flat = _flatten(module)
    cand_map = fused_candidates(module, h)
    chosen = result.by_claim()

    # Weights are phase-independent in the current model; the law uses one policy
    # for all selects, so assert that holds (defensive against future drift).
    w0 = weights(h, theta, flat[0][0], policy) if flat else weights(h, theta, 0, policy)
    for phase_id, _ in flat:
        if weights(h, theta, phase_id, policy) != w0:
            raise ValueError("phase-dependent weights break single-policy MLIR parity")

    views: list[ClaimView] = []
    prev_cand: Candidate | None = None
    total = 0
    for phase_id, claim in flat:
        sel = chosen[claim.id]
        paths: list[PathView] = []
        for cand in cand_map[claim.id]:
            eff = cand.base.couple(_context_factor(prev_cand, cand, theta))
            paths.append(PathView(name=cand.name, lane=cand.lane, cost=tuple(eff.v)))
        eff_sel = sel.base.couple(_context_factor(prev_cand, sel, theta))
        score = eff_sel.dot(w0)
        total += score
        views.append(ClaimView(
            claim_id=claim.id, phase_id=phase_id, op=claim.op or claim.opcode.name,
            lane=claim.lane, stride_class=claim.stride_class, stride_k=claim.stride_k,
            count=claim.count, reads=tuple(claim.rd), writes=tuple(claim.wr),
            domain=claim.domain, hazard=claim.hazard, verify=claim.verify,
            bounds=claim.bounds, paths=tuple(paths), selected=sel.name,
            score=score, width=sel.width))
        prev_cand = sel

    if total != result.score:
        raise AssertionError(
            f"plan_view reconstruction {total} != oracle score {result.score} "
            f"({module.name}): the IR-level per-claim decomposition drifted")

    pmap = module.phase_map()
    order = list(dict.fromkeys(pid for pid, _ in flat))   # unique phase ids, topo order
    deps = {pid: tuple(pmap[pid].deps) for pid in order}
    return PlanView(module_name=module.name, policy_mode=_POLICY_MODE.get(policy.name, "latency"),
                    weights=tuple(w0), claims=tuple(views), phase_ids=tuple(order),
                    phase_deps=deps, total_score=result.score)


# --- MLIR text emission ---------------------------------------------------------

def _refs(rids: tuple[int, ...]) -> str:
    return "[" + ", ".join(f"@r{r}" for r in rids) + "]"


def _costvec(cost: tuple[int, ...]) -> str:
    body = ", ".join(f"{name} = {cost[i]}" for i, name in enumerate(DIMS))
    return f"#bcir.costvec<{body}>"


def _path_sym(claim_id: int, name: str) -> str:
    return f"path_{claim_id}_{name}"


def to_mlir(module: Module, h: HProfile, theta: Theta, policy: Policy = PERF, *,
            result: RealizationResult | None = None, module_suffix: str = "",
            run_pipeline: bool = True, filecheck: bool = False, budget=None) -> str:
    """Emit the GEM-pipeline BCIR-MLIR text for `module`'s plan. The result parses
    and runs through `-bcir-classify-lanes -bcir-select-realization -bcir-batch
    -bcir-schedule -bcir-lower-to-llvm` (the same surface as
    `mlir/test/passes/gem_passes.mlir`); with `filecheck=True` it embeds the
    `// RUN:` + `// CHECK-DAG:` lines so the committed corpus is self-checking under
    bcir-opt + FileCheck."""
    pv = plan_view(module, h, theta, policy, result)
    name = f"{module.name}{module_suffix}"
    L = []
    if filecheck:
        L.append("// RUN: bcir-opt -bcir-classify-lanes -bcir-select-realization "
                 "-bcir-batch -bcir-schedule -bcir-lower-to-llvm %s | FileCheck %s")
        L.append("//")
        L.append(f"// Generated from the oracle plan for `{module.name}` on {h.name} "
                 "(cool Theta, latency).")
        L.append("// Regenerate: python -m bcir.kbcir.differential --emit-corpus. The law's")
        L.append("// min-plus argmin must recompute the oracle's per-claim score (parity).")
        L.append("")
    L.append(f"bcir.module @{_ident(name)} {{")
    # target capability H (the seeded constants -bcir-cost-model / -bcir-plan /
    # -bcir-overlap / -bcir-rcsp-plan read to recompute costs from first principles --
    # the per-target descriptor that makes the six-target capability matrix work).
    # A full mirror of cost.py TargetProfile: every seed the law consumes is emitted,
    # so a non-x86 target (arm64 / RISC-V / GPU) plans from its own constants, not the
    # CPU defaults. `warp` / `mem_channels` carry the warp geometry + roofline knee
    # (mem_channels feeds -bcir-schedule's bandwidth-bound parallelism).
    isa = ", ".join(f"\"{x}\"" for x in sorted(h.isa_features))
    lw = ", ".join(str(x) for x in h.lane_widths)
    L.append(f"  bcir.target.capability @cpu {{ triple = \"{h.triple}\", "
             f"isa_features = [{isa}], lane_widths = array<i64: {lw}>, "
             f"warp = {h.warp} : i32, cacheline = {h.cacheline} : i32, "
             f"gather_penalty = {h.gather_penalty} : i32, "
             f"affinity_domains = {h.affinity_domains} : i32, "
             f"mem_channels = {h.mem_channels} : i32, "
             f"thermal_density = {h.thermal_density} : i32, power_density = {h.power_density} : i32, "
             f"mem_unit = {h.mem_unit} : i32, base_overhead = {h.base_overhead} : i32, "
             f"per_op_heat = {h.per_op_heat} : i32, elem_bytes = {h.elem_bytes} : i32 }}")
    # registry: the resources the claims reference (so the cost model resolves the
    # tier from each claim's read[0] domain, and HAM access).
    L.append("  bcir.registry @RES {")
    for rid in sorted(module.resources):
        r = module.resources[rid]
        shape = ", ".join(str(d) for d in (r.shape or (1,)))
        access = f", access = #bcir.access<{'ham' if r.access == 'ham' else 'flat'}>"
        L.append(f"    %r{rid} = bcir.resource @r{rid} {{ rid = {rid} : i32, "
                 f"domain_kind = #bcir.domain<{_DOMAIN_SPELL[r.domain]}>, "
                 f"shape = array<i64: {shape}>, layout = #bcir.layout<soa>{access} }} "
                 ": !bcir.resource")
    L.append("  }")
    # live runtime state Theta: the emitter folds it into the policy weights (for
    # scalarization) AND emits it raw, so -bcir-plan / -bcir-overlap can apply the
    # multiplicative thermal coupling (the AVX-512 downclock) the weights cannot encode.
    L.append(f"  bcir.kbcir.theta @theta {{ thermal = {theta.thermal} : i32, "
             f"power = {theta.power} : i32, mem_pressure = {theta.mem_pressure} : i32, "
             f"contention = {theta.contention} : i32 }}")
    # policy (theta-folded weights; mode drives -bcir-select-realization lookup).
    wlist = ", ".join(str(x) for x in pv.weights)
    L.append(f"  bcir.kbcir.policy @perf {{ mode = #bcir.policy_mode<{pv.policy_mode}>, "
             f"weights = array<i64: {wlist}> }}")
    # optional plan-wide budget B(H,Theta): hard caps on additive resource dims that
    # -bcir-rcsp-plan's accumulated-budget label DP must respect (mirrors rcsp.Budget).
    if budget is not None and budget.caps:
        from ..kbcir.cost import DIMS as _DIMS
        dims = ", ".join(f"\"{_DIMS[d]}\"" for d, _ in budget.caps)
        caps = ", ".join(str(cap) for _, cap in budget.caps)
        L.append(f"  bcir.kbcir.budget @cap {{ dims = [{dims}], "
                 f"caps = array<i64: {caps}> }}")
    # phases
    for pid in pv.phase_ids:
        deps = ", ".join(f"@p{d}" for d in pv.phase_deps.get(pid, ()))
        L.append(f"  bcir.phase @p{pid} {{ id = {pid} : i32, deps = [{deps}] }}")
    # claims (region carries an index_range, like gem_passes.mlir)
    for cv in pv.claims:
        L.append(
            f"  bcir.claim @c{cv.claim_id} attributes {{ claim_id = {cv.claim_id} : i32, "
            f"phase = @p{cv.phase_id}, op = \"{cv.op}\", reads = {_refs(cv.reads)}, "
            f"writes = {_refs(cv.writes)}, count = {cv.count} : i64, "
            f"lane = #bcir.lane<{_LANE_SPELL[cv.lane]}>, "
            f"stride_class = #bcir.stride_class<{_STRIDE_SPELL[cv.stride_class]}>, "
            f"stride_k = {cv.stride_k} : i32, domain = #bcir.domain<{_DOMAIN_SPELL[cv.domain]}>, "
            f"hazard = #bcir.hazard<{cv.hazard}>, verify = #bcir.verify<{cv.verify}>, "
            f"bounds = #bcir.bounds<{cv.bounds}> }} "
            f"{{ %i = bcir.index_range 0 to {cv.count} step 1 }}")
    # the K_BCIR plan: candidate paths + per-claim min-plus select.
    L.append("  bcir.kbcir.plan @plan0 {")
    for cv in pv.claims:
        for p in cv.paths:
            sym = _path_sym(cv.claim_id, p.name)
            L.append(
                f"    %{sym} = bcir.kbcir.path @{sym} {{ claim = @c{cv.claim_id}, "
                f"realization = \"{_LANE_SPELL[p.lane]}.{p.name}\", "
                f"lane = #bcir.lane<{_LANE_SPELL[p.lane]}>, layout = #bcir.layout<soa>, "
                f"cost = {_costvec(p.cost)} }} : !bcir.path")
        frm = ", ".join(f"@{_path_sym(cv.claim_id, p.name)}" for p in cv.paths)
        L.append(
            f"    %sel_{cv.claim_id} = bcir.kbcir.select @c{cv.claim_id} from [{frm}] {{ "
            f"policy = #bcir.policy_mode<{pv.policy_mode}>, semiring = #bcir.semiring<min_plus>, "
            f"selected = @{_path_sym(cv.claim_id, cv.selected)}, score = {cv.score} : i64 }} "
            ": !bcir.path")
    L.append("  }")
    # GEM lane segments (one per claim; preserve lane + resource set for R12 lower).
    for cv in pv.claims:
        L.append(
            f"  bcir.gem.lane_segment @seg_{cv.claim_id} {{ claim = @c{cv.claim_id}, "
            f"phase = @p{cv.phase_id}, lane = #bcir.lane<{_LANE_SPELL[cv.lane]}>, "
            f"stride_k = {cv.stride_k} : i32, width = {cv.width} : i32, "
            f"opcode = \"{cv.op}\", reads = {_refs(cv.reads)}, writes = {_refs(cv.writes)}, "
            f"fence_before = [], fence_after = [] }}")
    L.append("}")
    if filecheck:
        L.append("")
        L.append(f"// CHECK-LABEL: bcir.module @{_ident(name)}")
        for cv in pv.claims:
            L.append(f"// CHECK-DAG: kbcir.classified_lane = \"{_LANE_SPELL[_canonical_lane(cv.stride_class)]}\"")
        for cv in pv.claims:
            L.append(f"// CHECK-DAG: kbcir.computed_selected = @{_path_sym(cv.claim_id, cv.selected)}")
            L.append(f"// CHECK-DAG: kbcir.computed_score = {cv.score}")
        L.append("// CHECK-DAG: kbcir.lowered = true")
    return "\n".join(L) + "\n"


def _ident(name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "_$.-") else "_" for ch in name)


def _canonical_lane(sc: StrideClass) -> Lane:
    """The lane -bcir-classify-lanes assigns (mirrors BCIRPasses.cpp canonicalLane)."""
    return {StrideClass.SCALAR: Lane.U, StrideClass.UNIT: Lane.U,
            StrideClass.STRIDED: Lane.U, StrideClass.CACHELINE: Lane.UX,
            StrideClass.TILE: Lane.T, StrideClass.RANDOM: Lane.GGG}[sc]
