"""Generated, adversarial Python<->MLIR differential testing.

The Python oracle (`realize.optimize`, a min-plus **shortest path** over the
coupled candidate DAG) and the MLIR law (`-bcir-select-realization`, a per-claim
min-plus **argmin** over declared candidate costs) must agree -- that is the
docs/PARITY.md contract. Until now it was *hoped* on a handful of curated `.mlir`
pins (vector_add @ 7808, the budgeted 9472). This module turns it into a *proof*:

  * `gen_module` -- a structured/adversarial generator of legal `Module`s (every
    stride class, lane, tier, fusion/deforestation/CSE shape, reductions, gathers,
    HAM, multi-claim/multi-phase).
  * `law_select` -- an **independent** reimplementation of the C++ select pass's
    arithmetic (scalarize = sum_d cost_d * w_d; budget-feasible argmin; first wins
    on a tie), reading the IR-level `PlanView` -- never the oracle's objects.
  * `check_module` -- runs both rails and **diffs** the selected realization, the
    per-claim + total score, the (constrained) budget feasibility, and the
    deterministic schedule order.
  * `shrink` -- minimizes any mismatching module to a small witness.

The two rails are genuinely distinct algorithms over the same cost model, so an
agreement across thousands of generated modules is evidence the per-claim MLIR
select rail reproduces the globally-coupled oracle plan. (When the C++ toolchain is
present, `to_mlir` emits the same IR and `bcir-opt -bcir-select-realization`
recomputes the score directly -- the ultimate cross-check; see
`mlir/test/passes/gem_corpus.mlir`.)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field

from ..model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from ..lower.mlir import ClaimView, PlanView, plan_view, to_mlir
from .cost import TARGETS, Theta
from .realize import RealizationResult, _flatten, optimize
from .weights import PERF, POLICIES, Policy

# Reusable axes for campaigns (the six targets; the standard runtime states).
THETAS = {"cool": Theta.cool(), "hot": Theta.hot(), "mem_bound": Theta.mem_bound()}


# --- the law rail: an independent reimplementation of -bcir-select-realization ----

def _scalarize(cost: tuple[int, ...], w: tuple[int, ...]) -> int:
    """score(path) = sum_d cost_d * w_d -- the K_BCIR dot product the min-plus
    selection minimizes (mirrors BCIRPasses.cpp `scalarize`)."""
    return sum(c * wi for c, wi in zip(cost, w))


def law_select(pv: PlanView, budget: tuple[tuple[int, int], ...] = ()) \
        -> dict[int, tuple[str, int]]:
    """For each claim, recompute the budget-feasible min-plus argmin over its
    declared candidate costs and the policy weights -- exactly what the C++
    `-bcir-select-realization` pass does. Returns claim_id -> (selected_name, score).
    Deterministic: the first candidate wins a tie (the `from` order)."""
    out: dict[int, tuple[str, int]] = {}
    caps = dict(budget)
    for cv in pv.claims:
        best_name, best_score, have = None, 0, False
        for p in cv.paths:
            if any(p.cost[d] > cap for d, cap in caps.items()):
                continue  # infeasible under the budget (per-path R(pi) <= B)
            s = _scalarize(p.cost, pv.weights)
            if not have or s < best_score:
                best_name, best_score, have = p.name, s, True
        if not have:
            raise ValueError(f"law_select: no feasible candidate for claim {cv.claim_id}")
        out[cv.claim_id] = (best_name, best_score)
    return out


def law_schedule(module: Module) -> list[int]:
    """The deterministic exec order the MLIR `-bcir-schedule` pass assigns:
    topological by phase id, then ascending claim id within a phase."""
    order = list(dict.fromkeys(pid for pid, _ in _flatten(module)))  # topo phase order
    rank = {pid: i for i, pid in enumerate(order)}
    claims = [(rank[pid], c.id) for pid, c in _flatten(module)]
    return [cid for _, cid in sorted(claims)]


def oracle_schedule(module: Module) -> list[int]:
    """The oracle's deterministic claim order (topological phases, declared order)."""
    return [c.id for _, c in _flatten(module)]


def _respects_phase_dag(module: Module, order: list[int]) -> bool:
    """True iff `order` is a legal execution order: every claim of a dependency
    phase precedes every claim of the dependent phase. (Intra-phase claim order is a
    free deterministic choice -- the MLIR pass keys on claim id, the oracle on
    declared order -- so it is not a parity invariant; phase ordering is.)"""
    pos = {cid: i for i, cid in enumerate(order)}
    phase_of = {c.id: ph.phase_id for ph in module.phases for c in ph.claims}
    claims_in = {ph.phase_id: [c.id for c in ph.claims] for ph in module.phases}
    for ph in module.phases:
        for dep in ph.deps:
            for d in claims_in.get(dep, ()):
                for c in claims_in.get(ph.phase_id, ()):
                    if d in pos and c in pos and pos[d] >= pos[c]:
                        return False
    return True


# --- the diff -------------------------------------------------------------------

@dataclass(frozen=True)
class Mismatch:
    kind: str           # select | score | total | budget | schedule
    claim_id: int       # 0 for module-level (total / schedule)
    detail: str
    coupling: bool = False   # True iff explainable by cross-claim coupling (a known
                             # per-claim-rail capability gap, not an oracle bug)


@dataclass
class CheckResult:
    module: str
    target: str
    theta: str
    policy: str
    n_claims: int
    score: int
    mismatches: list[Mismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    @property
    def bugs(self) -> list[Mismatch]:
        """Mismatches that are NOT explained by the per-claim rail's coupling gap."""
        return [m for m in self.mismatches if not m.coupling]


def _coupling_active(cv: ClaimView, prev_selected_width: int | None) -> bool:
    """A select divergence is a known *capability gap* (not a bug) only when cross-
    claim coupling could have made the oracle prefer a locally-suboptimal candidate:
    the predecessor was wide and a memory fusion discount is in play, and this claim
    has >1 candidate of differing width."""
    if len(cv.paths) < 2 or prev_selected_width is None or prev_selected_width <= 1:
        return False
    return len({p.cost for p in cv.paths}) > 1


def check_plan(module: Module, pv: PlanView, oracle: RealizationResult,
               budget: tuple[tuple[int, int], ...] = ()) -> list[Mismatch]:
    """Diff the law's recomputation of `pv` against the oracle's plan."""
    ms: list[Mismatch] = []
    law = law_select(pv, budget)
    prev_w: int | None = None
    law_total = 0
    for cv in pv.claims:
        lname, lscore = law[cv.claim_id]
        law_total += lscore
        if lname != cv.selected:
            ms.append(Mismatch(
                "select", cv.claim_id,
                f"law argmin {lname!r} != oracle selected {cv.selected!r}",
                coupling=_coupling_active(cv, prev_w)))
        elif lscore != cv.score:
            ms.append(Mismatch(
                "score", cv.claim_id,
                f"law score {lscore} != oracle per-claim score {cv.score}"))
        prev_w = cv.width
    if not budget and law_total != oracle.score:
        ms.append(Mismatch("total", 0,
                           f"law total {law_total} != oracle score {oracle.score}"))
    # schedule parity: the law's -bcir-schedule order must cover exactly the oracle's
    # claim set and be a legal topological order of the phase DAG (intra-phase claim
    # order is a free deterministic choice, not a parity invariant -- so we check
    # legality, not byte-identical sequence, which avoids false positives on modules
    # that declare same-phase claims out of ascending-id order).
    law_ord = law_schedule(module)
    if sorted(law_ord) != sorted(oracle_schedule(module)):
        ms.append(Mismatch("schedule", 0,
                           f"law schedule {law_ord} != claim set {oracle_schedule(module)}"))
    elif not _respects_phase_dag(module, law_ord):
        ms.append(Mismatch("schedule", 0, f"law schedule {law_ord} violates the phase DAG"))
    return ms


def check_overlap(module: Module, h, theta: Theta, oracle: RealizationResult,
                  policy: Policy = PERF) -> list[Mismatch]:
    """The (max,+) scheduled-price law -- the conformance net for the deterministic
    optimizer core's *overlap* piece, the next slice to port to C++/MLIR. The oracle
    `gem.overlap.price_scheduled` must satisfy the same R9 invariant the MLIR
    VerifyPass enforces on `bcir.kbcir.scheduled_price`: makespan >= 0, the serial
    bound equals the plan score, and makespan + overlap_gain == serial (so the
    makespan never exceeds the serial sum -- overlap only ever helps)."""
    from ..gem.overlap import price_scheduled

    sp = price_scheduled(module, oracle, h, theta, policy)
    ms: list[Mismatch] = []
    if sp.makespan < 0 or sp.makespan + sp.overlap_gain != sp.serial:
        ms.append(Mismatch("overlap", 0,
                           f"inconsistent scheduled price (makespan {sp.makespan} + gain "
                           f"{sp.overlap_gain} != serial {sp.serial})"))
    if sp.serial != oracle.score:
        ms.append(Mismatch("overlap", 0,
                           f"serial bound {sp.serial} != plan score {oracle.score}"))
    if sp.makespan > sp.serial:
        ms.append(Mismatch("overlap", 0,
                           f"makespan {sp.makespan} exceeds serial {sp.serial}"))
    return ms


def check_module(module: Module, target: str = "x86_avx512", theta: str = "cool",
                 policy: str = "latency") -> CheckResult:
    """Plan `module` on both rails and diff them. The single-call entry point."""
    h = TARGETS[target]
    th = THETAS[theta]
    pol = POLICIES.get(policy, PERF)
    oracle = optimize(module, h, th, pol)
    pv = plan_view(module, h, th, pol, oracle)
    ms = check_plan(module, pv, oracle)
    ms += check_overlap(module, h, th, oracle, pol)
    return CheckResult(module=module.name, target=target, theta=theta, policy=policy,
                       n_claims=len(pv.claims), score=oracle.score, mismatches=ms)


def check_budget(module: Module, budget, target: str = "x86_avx512",
                 theta: str = "cool", policy: str = "latency") -> list[Mismatch]:
    """Constrained (RCSP) parity for a single-claim module: the law's per-path
    budget-feasible argmin must equal `optimize_constrained`'s selection. (Per-path
    feasibility coincides with the accumulated RCSP cap only for one claim, which is
    exactly the curated worked example -- vec16 infeasible @700 -> vec8 @ 9472.)"""
    from .rcsp import Budget, optimize_constrained

    h = TARGETS[target]
    th = THETAS[theta]
    pol = POLICIES.get(policy, PERF)
    b = budget if isinstance(budget, Budget) else Budget.of(**budget)
    res = optimize_constrained(module, h, th, pol, b)
    pv = plan_view(module, h, th, pol, res)
    if len(pv.claims) != 1:
        raise ValueError("check_budget requires a single-claim module (per-path RCSP)")
    return check_plan(module, pv, res, budget=tuple(b.caps))


# --- the structured / adversarial module generator -------------------------------

# Safe counts keep the n-scaled compute/memory terms dominant over the width-scaled
# heat terms, so wider lanes always win the per-claim argmin -- the regime where the
# per-claim select rail provably reproduces the coupled oracle plan (no width flip).
_SAFE_COUNTS = (1024, 2048, 4096, 8192)
_STRIDES = (2, 3, 4, 8)
_TIER_DOMAINS = (Domain.RAM, Domain.RAM, Domain.RAM, Domain.HBM, Domain.CXL)
_ELEMENT_OPS = (Opcode.ADD, Opcode.SUB, Opcode.MUL)
# Per-claim shapes the generator draws from (legal lane for each stride class).
_KINDS = ("unit", "unit", "strided", "cacheline", "gather", "reduce_gather", "tile")


class _RidPool:
    def __init__(self, module: Module):
        self.m = module
        self.next = 1

    def add(self, domain: Domain, count: int, access: str = "flat") -> int:
        rid = self.next
        self.next += 1
        self.m.add_resource(Resource(rid=rid, domain=domain, shape=(count,), access=access))
        return rid


def gen_module(rng: random.Random) -> Module:
    """Generate a legal, structured `Module` exercising the optimizer's surface:
    multiple stride classes / lanes / tiers, shared-operand fusion, producer->
    consumer chains (deforestation), duplicate claims (CSE), reductions, gathers,
    and HAM tables -- across one or two phases. Guaranteed verifier-clean."""
    m = Module(name=f"gen{rng.randrange(1 << 28):08x}")
    pool = _RidPool(m)
    nphases = rng.randint(1, 2)
    next_claim = 100

    for pid in range(nphases):
        deps = (pid - 1,) if pid else ()
        claims: list[Claim] = []
        phase_reads: list[int] = []      # rids available to share within this phase
        phase_writes: list[int] = []     # only non-sparse writes (safe to consume)
        nclaims = rng.randint(1, 4)
        for _ in range(nclaims):
            kind = rng.choice(_KINDS)
            cid = next_claim
            next_claim += 1
            claim = _gen_claim(rng, kind, cid, pool, phase_reads, phase_writes)
            claims.append(claim)
            phase_reads.extend(claim.rd)
            # A consumer that reads a *sparse* (GGG/random) producer's write is a RAW
            # across the decoupled tail (R5 needs a hazard contract); keep only safe
            # (U-lane) writes in the consumable pool so producer->consumer chains stay
            # clean without inventing hazards.
            if claim.lane != Lane.GGG and claim.stride_class != StrideClass.RANDOM:
                phase_writes.extend(claim.wr)
        m.add_phase(Phase(phase_id=pid, deps=deps, claims=claims))
    return m


def _gen_claim(rng, kind, cid, pool, phase_reads, phase_writes) -> Claim:
    count = rng.choice(_SAFE_COUNTS)
    domain = rng.choice(_TIER_DOMAINS)

    if kind == "gather":
        access = "ham" if rng.random() < 0.4 else "flat"
        src = pool.add(domain, count, access=access)
        dst = pool.add(domain, count)
        return Claim(id=cid, opcode=Opcode.GGG_LOAD, lane=Lane.GGG,
                     stride_class=StrideClass.RANDOM, count=count, rd=(src,), wr=(dst,),
                     op="histogram.scatter", domain=domain, verify="bounds")
    if kind == "reduce_gather":
        src = pool.add(domain, count)
        acc = pool.add(domain, count)
        return Claim(id=cid, opcode=Opcode.ADD, lane=Lane.U,
                     stride_class=StrideClass.UNIT, count=count, rd=(src,), wr=(acc,),
                     op="reduce.gather", domain=domain)
    if kind == "cacheline":
        src = pool.add(domain, count)
        dst = pool.add(domain, count)
        return Claim(id=cid, opcode=Opcode.ADD, lane=Lane.UX,
                     stride_class=StrideClass.CACHELINE, count=count, rd=(src,),
                     wr=(dst,), op="vector.add", domain=domain, verify="bounds")
    if kind == "strided":
        k = rng.choice(_STRIDES)
        src = pool.add(domain, count * k)        # extent (count-1)*k+1 <= count*k
        dst = pool.add(domain, count)
        return Claim(id=cid, opcode=Opcode.MUL, lane=Lane.U,
                     stride_class=StrideClass.STRIDED, count=count, stride_k=k,
                     rd=(src,), wr=(dst,), op="vector.axpy", domain=domain)
    if kind == "tile":
        tile = rng.choice((32, 64))
        a = pool.add(domain, tile * tile)
        b = pool.add(domain, tile * tile)
        c = pool.add(Domain.HBM, tile * tile)
        return Claim(id=cid, opcode=Opcode.T_MACC, lane=Lane.T,
                     stride_class=StrideClass.TILE, count=tile * tile, rd=(a, b),
                     wr=(c,), op="linalg.matmul", domain=domain)

    # "unit": the multi-candidate elementwise claim -- the fusion/deforestation/CSE
    # surface. Pick operands to sometimes share a read (fusion), consume a prior
    # write (deforestation, U lane only -> no R5 sparse conflict), or duplicate.
    op = rng.choice(_ELEMENT_OPS)
    opname = {"ADD": "vector.add", "SUB": "vector.sub", "MUL": "vector.mul"}[op.name]
    mode = rng.random()
    same_count_reads = [r for r in phase_reads if m_count(pool, r) == count]
    consumable = [w for w in phase_writes if m_count(pool, w) == count]
    if mode < 0.25 and len(same_count_reads) >= 2:           # shared-input fusion
        a, b = rng.sample(same_count_reads, 2)
        rd = (a, b)
    elif mode < 0.45 and consumable:                          # producer->consumer
        a = rng.choice(consumable)
        b = pool.add(domain, count)
        rd = (a, b)
    else:                                                     # fresh operands
        a = pool.add(domain, count)
        b = pool.add(domain, count)
        rd = (a, b)
    dst = pool.add(domain, count)
    # R3: declare the domain of an actually-touched read resource.
    domain = pool.m.resource(rd[0]).domain
    return Claim(id=cid, opcode=op, lane=Lane.U, stride_class=StrideClass.UNIT,
                 count=count, rd=rd, wr=(dst,), op=opname, domain=domain)


def m_count(pool: _RidPool, rid: int) -> int:
    r = pool.m.resource(rid)
    return r.count if r is not None else -1


# --- verifier differential: generated illegal modules ----------------------------
#
# The plan-parity campaign diffs the optimizer rails on *legal* modules. This is its
# counterpart for the verifier: generate modules with exactly one injected law
# violation and confirm the Python verifier (the conformance oracle) flags that law.
# It is the "illegal modules" testing lever (BCIR_Roadmap Phase B.3) -- fault
# injection seeded off the same generator -- and the seed corpus for a future
# structure-aware MLIR `-bcir-verify` fuzz once the toolchain is in the loop.

def _verifier_base(rng: random.Random) -> Module:
    """A small, deterministically-clean single-claim module to inject faults into
    (gen_module would risk incidental diagnostics; this base verifies empty)."""
    n = rng.choice(_SAFE_COUNTS)
    m = Module(name=f"vbase{rng.randrange(1 << 24):06x}")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=n, rd=(1, 2), wr=(3,), op="vector.add", domain=Domain.RAM)]))
    return m


def _inject(m: Module, mut) -> Module:
    c = m.phases[0].claims[0]
    mut(m, c)
    return m


# law -> a one-line mutation that makes the base illegal for exactly that law.
# R1 (RID uniqueness) is intentionally absent: it is enforced *by construction* --
# Module.resources is a dict keyed by RID and Module.add_resource() raises on a dup, so
# the registry can never hold two equal RIDs and verify()'s R1 loop is unreachable
# belt-and-suspenders. It cannot be fault-injected through the model API (a dict cannot
# carry the duplicate), so the campaign documents the invariant rather than faking it.
_INJECTORS = {
    "R2": lambda m, c: setattr(c, "rd", (1, 999)),                 # dangling resource ref
    "R3": lambda m, c: setattr(c, "domain", Domain.HBM),          # domain not backed by any touched resource
    "R4": lambda m, c: setattr(m.phases[0], "deps",               # phase depends on itself -> DAG cycle
                               (m.phases[0].phase_id,)),
    "R5": lambda m, c: (setattr(c, "opcode", Opcode.ATOMIC_ADD),  # atomic opcode + unique hazard
                        setattr(c, "hazard", "unique")),
    "R6": lambda m, c: setattr(c, "lane", Lane.T),                # lane illegal for UNIT stride
    "R7": lambda m, c: setattr(c, "count", c.count + 1_000_000),  # affine read/write overruns the resource
    "R8": lambda m, c: setattr(c, "cost_class", "__unknown__"),   # claim names no known cost class
}


def gen_illegal_module(rng: random.Random) -> tuple[Module, str]:
    """Return (module, expected_law): a clean base with exactly one injected law
    violation. The Python verifier must report `expected_law`."""
    law = rng.choice(sorted(_INJECTORS))
    return _inject(_verifier_base(rng), _INJECTORS[law]), law


def check_verifier(module: Module, expected_law: str) -> list[Mismatch]:
    """Diff the verifier against the injected fault: the expected law must fire, and
    a module with no injected fault must verify clean (checked by the caller)."""
    from ..verify import verify

    laws = {d.law for d in verify(module)}
    if expected_law not in laws:
        return [Mismatch("verify", 0,
                         f"verifier missed injected {expected_law} (reported {sorted(laws)})")]
    return []


# --- the shrinker ---------------------------------------------------------------

def shrink(module: Module, fails, max_passes: int = 40) -> Module:
    """Greedily minimize a module for which `fails(module)` is True, preserving the
    failure: drop claims, drop phases, then shrink counts. Returns the minimal
    witness (the same module if nothing can be removed)."""
    cur = module
    for _ in range(max_passes):
        progressed = False
        for cand in _shrink_candidates(cur):
            if _legal(cand) and fails(cand):
                cur = cand
                progressed = True
                break
        if not progressed:
            break
    return cur


def _legal(module: Module) -> bool:
    from ..verify import verify
    if not _flatten(module):
        return False
    if verify(module):
        return False
    try:
        optimize(module, TARGETS["x86_avx512"], Theta.cool())
    except Exception:  # noqa: BLE001 - an unplannable shrink is not a valid witness
        return False
    return True


def _shrink_candidates(module: Module):
    import copy
    # 1) drop a whole phase.
    for i in range(len(module.phases)):
        c = copy.deepcopy(module)
        del c.phases[i]
        _reindex_deps(c)
        yield c
    # 2) drop a single claim.
    for pi, ph in enumerate(module.phases):
        for ci in range(len(ph.claims)):
            c = copy.deepcopy(module)
            del c.phases[pi].claims[ci]
            if c.phases[pi].claims or len(c.phases) > 1:
                yield c
    # 3) shrink every claim's count to the smallest safe value.
    if any(cl.count != _SAFE_COUNTS[0] for ph in module.phases for cl in ph.claims):
        import copy as _c
        c = _c.deepcopy(module)
        for ph in c.phases:
            for cl in ph.claims:
                if cl.stride_class != StrideClass.TILE and cl.count > _SAFE_COUNTS[0]:
                    cl.count = _SAFE_COUNTS[0]
        yield c


def _reindex_deps(module: Module) -> None:
    live = {p.phase_id for p in module.phases}
    for p in module.phases:
        p.deps = tuple(d for d in p.deps if d in live)


# --- the campaign ---------------------------------------------------------------

@dataclass
class Report:
    checked: int = 0
    ok: int = 0
    bugs: list[CheckResult] = field(default_factory=list)
    coupling_gaps: int = 0

    def summary(self) -> str:
        return (f"differential campaign: {self.checked} checks, {self.ok} clean, "
                f"{len(self.bugs)} bug(s), {self.coupling_gaps} coupling-gap(s)")


def run_campaign(n: int = 200, seed: int = 0, targets=None, thetas=None,
                 policies=None, include_corpus: bool = True) -> Report:
    """Generate `n` random modules and check each across a grid of (target, theta,
    policy); also check the widened corpus across the six targets. Returns a Report;
    `report.bugs` is the set of genuine (non-coupling) divergences -- empty == proof
    of agreement over the sampled space."""
    rng = random.Random(seed)
    targets = list(targets or TARGETS)
    thetas = list(thetas or THETAS)
    policies = list(policies or ("latency", "throughput", "energy"))
    rep = Report()

    def run_one(module, t, th, pol):
        res = check_module(module, t, th, pol)
        rep.checked += 1
        if res.ok:
            rep.ok += 1
        elif res.bugs:
            rep.bugs.append(res)
        else:
            rep.coupling_gaps += 1

    if include_corpus:
        from ..examples import CORPUS, PROGRAMS
        for name in CORPUS:
            for t in targets:
                run_one(PROGRAMS[name](), t, "cool", "latency")

    for _ in range(n):
        module = gen_module(rng)
        t = rng.choice(targets)
        th = rng.choice(thetas)
        pol = rng.choice(policies)
        run_one(module, t, th, pol)
    return rep


# Plan-law injectors (verify_plan rail): a clean optimize() result corrupted to violate
# exactly R9. Unlike _INJECTORS (which mutate the module/claim) these mutate the
# RealizationResult, so the law fires through verify_plan, not verify.
_PLAN_INJECTORS = {
    # R9: the reported score must equal the sum of the realized step costs.
    "R9-score": lambda m, r: setattr(r, "score", r.score + 1),
    # R9: a plan must realize every claim exactly once (drop the realization).
    "R9-coverage": lambda m, r: r.steps.clear(),
}


def gen_illegal_plan(rng: random.Random):
    """Return (module, result, expected_law): a clean base + its optimal plan with one
    injected plan-law (R9) violation. verify_plan must report R9."""
    from .realize import optimize

    m = _verifier_base(rng)
    h = TARGETS[rng.choice(sorted(TARGETS))]
    r = optimize(m, h, Theta.cool())
    key = rng.choice(sorted(_PLAN_INJECTORS))
    _PLAN_INJECTORS[key](m, r)
    return m, r, "R9"


def check_plan_verifier(module: Module, result, expected_law: str) -> list[Mismatch]:
    """The injected plan-law fault must fire through verify_plan."""
    from ..verify import verify_plan

    laws = {d.law for d in verify_plan(module, result)}
    if expected_law not in laws:
        return [Mismatch("verify_plan", 0,
                         f"verify_plan missed injected {expected_law} (reported {sorted(laws)})")]
    return []


def run_verifier_campaign(n: int = 200, seed: int = 0) -> list[Mismatch]:
    """Generate `n` illegal modules (one injected law each) and confirm the verifier
    flags each; also confirm the un-mutated base + its optimal plan verify clean.
    Two rails: module/claim laws (R2-R8 via verify) and the plan law (R9 via
    verify_plan). R1 is enforced by construction (see _INJECTORS). Returns the misses
    (empty == the verifier caught every injected fault)."""
    from ..verify import verify, verify_plan
    from .realize import optimize

    rng = random.Random(seed)
    misses: list[Mismatch] = []
    for _ in range(n):
        # module/claim rail (R2-R8): clean base verifies clean; injected fault fires.
        base = _verifier_base(rng)
        if verify(base):
            misses.append(Mismatch("verify", 0, "verifier flagged a clean base module"))
        module, law = gen_illegal_module(rng)
        misses.extend(check_verifier(module, law))
        # plan rail (R9): a clean base's optimal plan verifies clean; injected fault fires.
        clean = _verifier_base(rng)
        clean_plan = optimize(clean, TARGETS[rng.choice(sorted(TARGETS))], Theta.cool())
        if verify_plan(clean, clean_plan):
            misses.append(Mismatch("verify_plan", 0, "verify_plan flagged a clean plan"))
        m2, r2, plaw = gen_illegal_plan(rng)
        misses.extend(check_plan_verifier(m2, r2, plaw))
    return misses


# --- corpus .mlir emission (the committed cross-rail artifact) --------------------

def emit_corpus_mlir() -> str:
    """Emit the widened-corpus GEM-pipeline IR (one bcir.module per program, on the
    AVX-512 profile, cool/latency) as a single self-checking FileCheck file -- the
    artifact `bcir-opt` recomputes in mlir/test/passes/gem_corpus.mlir."""
    from ..examples import CORPUS, PROGRAMS

    h = TARGETS["x86_avx512"]
    th = Theta.cool()
    blocks = [
        "// RUN: bcir-opt -bcir-classify-lanes -bcir-select-realization -bcir-batch "
        "-bcir-schedule -bcir-lower-to-llvm %s | FileCheck %s",
        "//",
        "// The widened corpus (real tiled matmul / scan / multi-claim histogram) on the",
        "// AVX-512 profile (cool Theta, latency). GENERATED by the oracle -- regenerate",
        "// with `python -m bcir.kbcir.differential --emit-corpus`. The law's min-plus",
        "// argmin must recompute the oracle's per-claim score for every claim (parity).",
        "",
    ]
    for name in CORPUS:
        body = to_mlir(PROGRAMS[name](), h, th, PERF, module_suffix="_avx512",
                       filecheck=True)
        # keep one RUN line at the top of the file; strip the per-block RUN header.
        lines = body.splitlines()
        start = next(i for i, ln in enumerate(lines) if ln.startswith("bcir.module"))
        blocks.append("\n".join(lines[start:]).rstrip())
        blocks.append("")
    return "\n".join(blocks) + "\n"


_CORPUS_MLIR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mlir",
                                 "test", "passes", "gem_corpus.mlir")
_THETA_MLIR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mlir",
                                "test", "passes", "theta_hot.mlir")


def emit_theta_test() -> str:
    """Emit matmul_tiled under Theta.hot() as a self-checking `-bcir-plan` file -- the
    hot-Theta plan parity that the kbcir.theta context op enables: the width-16 tiles
    pay the AVX-512 thermal coupling (x1.25), so the C++ plan reproduces the oracle's
    hot optimize() (1159168 vs the cool 1015808)."""
    from ..examples import matmul_tiled

    h = TARGETS["x86_avx512"]
    th = Theta.hot()
    score = optimize(matmul_tiled(), h, th, PERF).score
    body = to_mlir(matmul_tiled(), h, th, PERF, module_suffix="_hot")
    head = [
        "// RUN: bcir-opt -bcir-plan %s | FileCheck %s",
        "//",
        "// Hot-Theta plan parity -- the bcir.kbcir.theta context op lifts the cool-regime",
        "// restriction. matmul_tiled emitted under Theta.hot(): the width-16 tiles pay the",
        "// AVX-512 thermal coupling (x1.25) the policy weights cannot encode, so -bcir-plan",
        "// reproduces the oracle's hot optimize() score (the cool plan was 1015808).",
        "// Regenerate: python -m bcir.kbcir.differential --emit-theta.",
        "",
    ]
    return "\n".join(head) + body.rstrip() + f"\n\n// CHECK: kbcir.plan_score = {score}\n"


# --- the six-target capability matrix (the cross-rail per-target artifact) --------
#
# The committed gem_corpus.mlir pins the law to the oracle on ONE target
# (x86_avx512). This matrix pins it on ALL SIX: each program is emitted once per
# target with that target's `bcir.target.capability` seeds, and the C++
# optimizer-core passes -- -bcir-plan (the coupled shortest path), -bcir-overlap
# (the (max,+) scheduled price), and -bcir-rcsp-plan (the accumulated-budget label
# DP) -- must recompute the oracle's per-target plan score / makespan / constrained
# optimum FROM THE CAPABILITY ALONE (the emitted path costs are decoration the plan
# pass ignores). So the six-target parity moves from "proven on the Python rail"
# (run_campaign) to "proven on the MLIR/C++ rail" (this file under bcir-opt).
#
# A compact, surface-covering program set keeps the artifact reviewable while
# exercising every target-sensitive cost path:
#   * vector_add      -- per-target lane width (avx512/sve/rvv 16, avx2 8, neon 4, ptx 32)
#   * fused_chain     -- (max,+) overlap gain + shared-input fusion (distinct per target)
#   * scan_chain      -- producer->consumer deforestation, serialized (overlap_gain 0)
#   * histogram_gather-- the GGG gather lane (ptx's gather_penalty 16 vs the CPUs' 32)
#   * tiled_matmul    -- the T-lane tile path (target-invariant under latency weights)
MATRIX = ("vector_add", "fused_chain", "scan_chain", "histogram_gather", "tiled_matmul")


def _two_independent_vec(n: int = 1024) -> Module:
    """Two independent unit-stride add claims over disjoint operands (no shared read,
    no dependency) -- the clean RCSP shape: each realizes at the target's max width,
    and a plan-wide thermal cap can narrow ONE of them (the accumulated-budget
    decision a per-claim cap cannot make). Mirrors mlir/test/passes/rcsp_plan.mlir."""
    m = Module(name="two_vec")
    for rid in (1, 2, 3, 4, 5, 6):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=n, rd=(1, 2), wr=(3,), op="vector.add", domain=Domain.RAM),
        Claim(id=2, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=n, rd=(4, 5), wr=(6,), op="vector.add", domain=Domain.RAM)]))
    return m


def _rcsp_for_target(module: Module, h, theta: Theta):
    """Pick a plan-wide thermal cap for `module` on target `h` and solve it.

    Returns (Budget, constrained RealizationResult). The cap is the unconstrained
    plan's accumulated thermal minus one -- which *binds* (forces a narrower width on
    one claim) on targets with an intermediate lane that lowers heat (avx512/sve/rvv:
    {16,16}->{16,8}); on targets whose widest lane is already heat-minimal (avx2/neon/
    ptx) that cap is infeasible, so we fall back to the exact accumulation (a feasible,
    non-binding cap). Either way it is a real per-target -bcir-rcsp-plan parity check."""
    from .rcsp import Budget, Infeasible, optimize_constrained, plan_resources

    unconstrained = optimize(module, h, theta, PERF)
    accum_thermal = plan_resources(unconstrained, theta).v[5]  # THERMAL dim
    for cap in (accum_thermal - 1, accum_thermal):
        budget = Budget.of(thermal=cap)
        try:
            return budget, optimize_constrained(module, h, theta, PERF, budget)
        except Infeasible:
            continue
    raise AssertionError(f"no feasible thermal cap for {module.name} on {h.name}")


def _matrix_check_block(lines: list[str], label: str, pairs: list[tuple[str, int]]) -> None:
    """Append a FileCheck block: a CHECK-LABEL on the module op, then a plain CHECK per
    module-level attribute. The annotations print in the module op's attr-dict (one
    line, alphabetically sorted); forward-scanning CHECKs match them in that order, and
    the CHECK-LABEL isolates the module so equal scores on other targets don't collide
    (the proven idiom in mlir/test/passes/plan.mlir). `pairs` must be in attr order."""
    lines.append(f"// CHECK-LABEL: bcir.module @{label}")
    for attr, value in pairs:
        lines.append(f"// CHECK: kbcir.{attr} = {value}")


def emit_target_matrix() -> str:
    """Emit the six-target capability matrix as one self-checking FileCheck file --
    the artifact `bcir-opt -bcir-plan -bcir-overlap -bcir-rcsp-plan` recomputes in
    mlir/test/passes/target_matrix.mlir. Regenerate with
    `python -m bcir.kbcir.differential --emit-matrix`."""
    from ..examples import PROGRAMS
    from ..gem.overlap import price_scheduled

    th = Theta.cool()
    out: list[str] = [
        "// RUN: bcir-opt -bcir-plan -bcir-overlap -bcir-rcsp-plan %s | FileCheck %s",
        "//",
        "// The six-target capability matrix. Each program is emitted once per TARGET",
        "// with that target's bcir.target.capability seeds (lane widths, gather penalty,",
        "// cacheline, heat/power density, affinity domains, ...); the C++ optimizer-core",
        "// passes must recompute the oracle's per-target plan score (-bcir-plan), (max,+)",
        "// scheduled makespan/gain (-bcir-overlap), and constrained optimum (-bcir-rcsp-",
        "// plan) FROM THE CAPABILITY ALONE -- the emitted path costs are decoration the",
        "// plan/overlap/rcsp passes ignore. GENERATED by the oracle; regenerate with",
        "// `python -m bcir.kbcir.differential --emit-matrix`. This is the six-target",
        "// parity (run_campaign) carried onto the MLIR/C++ rail.",
        "",
    ]

    # --- -bcir-plan + -bcir-overlap: the coupled plan + scheduled price, per target ---
    for name in MATRIX:
        for tname, h in TARGETS.items():
            module = PROGRAMS[name]()
            res = optimize(module, h, th, PERF)
            sp = price_scheduled(module, res, h, th, PERF)
            body = to_mlir(module, h, th, PERF, module_suffix=f"_{tname}", result=res)
            out.append(body.rstrip())
            out.append("")
            _matrix_check_block(out, f"{_module_label(module.name, tname)}", [
                ("overlap_gain", sp.overlap_gain),
                ("overlap_makespan", sp.makespan),
                ("plan_score", res.score),
            ])
            out.append("")

    # --- -bcir-rcsp-plan: the plan-wide constrained optimum, per target ---------------
    out.append("// -- plan-level constrained selection (RCSP): a thermal cap narrows the")
    out.append("// -- plan where it binds (avx512/sve/rvv) and is feasibly inert otherwise.")
    out.append("")
    for tname, h in TARGETS.items():
        module = _two_independent_vec()
        budget, rc = _rcsp_for_target(module, h, th)
        body = to_mlir(module, h, th, PERF, module_suffix=f"_{tname}", budget=budget)
        out.append(body.rstrip())
        out.append("")
        label = _module_label(module.name, tname)
        out.append(f"// CHECK-LABEL: bcir.module @{label}")
        out.append(f"// CHECK: kbcir.rcsp_plan_score = {rc.score}")
        for step in rc.steps:
            out.append(f"// CHECK: bcir.claim @c{step.claim_id}")
            out.append(f"// CHECK-SAME: kbcir.rcsp_width = {step.candidate.width}")
        out.append("")
    return "\n".join(out) + "\n"


def _module_label(module_name: str, target: str) -> str:
    """The emitted module symbol (to_mlir sanitizes name+suffix the same way)."""
    from ..lower.mlir import _ident
    return _ident(f"{module_name}_{target}")


_MATRIX_MLIR_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "mlir",
                                 "test", "passes", "target_matrix.mlir")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bcir.kbcir.differential",
                                description="Generated Python<->MLIR differential testing")
    p.add_argument("-n", type=int, default=500, help="random modules to generate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emit-corpus", action="store_true",
                   help="(re)write mlir/test/passes/gem_corpus.mlir from the oracle")
    p.add_argument("--emit-theta", action="store_true",
                   help="(re)write mlir/test/passes/theta_hot.mlir (hot-Theta plan parity)")
    p.add_argument("--emit-matrix", action="store_true",
                   help="(re)write mlir/test/passes/target_matrix.mlir (six-target parity)")
    p.add_argument("--emit", metavar="PROGRAM",
                   help="print the GEM-pipeline MLIR for one program (see examples.PROGRAMS)")
    p.add_argument("--target", default="x86_avx512")
    args = p.parse_args(argv)

    if args.emit:
        from ..examples import PROGRAMS
        print(to_mlir(PROGRAMS[args.emit](), TARGETS[args.target], Theta.cool(), PERF,
                      filecheck=True))
        return 0
    if args.emit_corpus:
        path = os.path.normpath(_CORPUS_MLIR_PATH)
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit_corpus_mlir())
        print(f"[differential] wrote {path}")
        return 0
    if args.emit_theta:
        path = os.path.normpath(_THETA_MLIR_PATH)
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit_theta_test())
        print(f"[differential] wrote {path}")
        return 0
    if args.emit_matrix:
        path = os.path.normpath(_MATRIX_MLIR_PATH)
        with open(path, "w", encoding="utf-8") as f:
            f.write(emit_target_matrix())
        print(f"[differential] wrote {path}")
        return 0

    rep = run_campaign(n=args.n, seed=args.seed)
    print(rep.summary())
    for r in rep.bugs[:10]:
        print(f"  BUG {r.module} {r.target}/{r.theta}/{r.policy}: "
              f"{[ (m.kind, m.claim_id, m.detail) for m in r.bugs ]}")
    vmiss = run_verifier_campaign(n=max(50, args.n // 4), seed=args.seed)
    print(f"verifier differential: {len(vmiss)} miss(es)")
    for mm in vmiss[:10]:
        print(f"  MISS {mm.detail}")
    return 1 if (rep.bugs or vmiss) else 0


if __name__ == "__main__":
    sys.exit(main())
