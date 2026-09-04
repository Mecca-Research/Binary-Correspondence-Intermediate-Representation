"""Generated, adversarial Python<->MLIR differential testing (the parity *proof*).

Turns the docs/PARITY.md contract -- the Python oracle (min-plus shortest path)
and the MLIR law (per-claim min-plus argmin) must agree -- from a handful of
curated pins into a generated proof over thousands of random modules, plus the
widened corpus (real tiled matmul / scan / multi-claim histogram) carried across
the six targets. When the MLIR toolchain is present it also runs the emitted IR
through `bcir-opt` (the ultimate cross-check); otherwise the independent Python
re-derivation of the law's arithmetic stands in (no third-party deps required).
"""

import copy
import os
import shutil
import subprocess

from bcir.examples import CORPUS, PROGRAMS, matmul_tiled, multi_histogram, scan, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta, _INDEX
from bcir.kbcir.differential import (
    _CORPUS_MLIR_PATH,
    check_budget,
    check_module,
    check_overlap,
    check_verifier,
    emit_corpus_mlir,
    gen_illegal_module,
    gen_module,
    law_select,
    run_campaign,
    run_verifier_campaign,
    shrink,
)
from bcir.gem.overlap import price_scheduled
from bcir.kbcir.realize import optimize as _optimize
from bcir.kbcir.rcsp import Budget, feasible, optimize_constrained, plan_resources
from bcir.kbcir.weights import PERF
from bcir.lower.mlir import plan_view, to_mlir
from bcir.model import Module, Phase
import random
from dataclasses import replace


# --- the campaign: generated adversarial parity (the headline) -------------------

# In-suite campaign size: a fast smoke check by default; the DEEP proof is CI's separate
# `python -m bcir.kbcir.differential -n 8000` (x2 seeds) + the ARM job's -n 3000, which are
# the real parity coverage. BCIR_THOROUGH=1 restores the larger in-suite N. The parity
# assertions (no bugs, no coupling gaps, ok == checked) are unchanged at any N.
_CAMPAIGN_N = 1500 if os.environ.get("BCIR_THOROUGH") else 400


def test_generated_campaign_has_no_divergence():
    # Random modules across the six targets x {cool, hot, mem_bound} x {latency, throughput,
    # energy}: the per-claim MLIR select rail must reproduce the globally-coupled oracle plan
    # on every one (no bugs, no coupling gaps).
    rep = run_campaign(n=_CAMPAIGN_N, seed=20240601)
    assert rep.checked >= _CAMPAIGN_N
    assert not rep.bugs, rep.bugs[:3]
    assert rep.coupling_gaps == 0, "unexpected per-claim/shortest-path divergence"
    assert rep.ok == rep.checked


def test_campaign_is_deterministic():
    a = run_campaign(n=200, seed=7)
    b = run_campaign(n=200, seed=7)
    assert (a.checked, a.ok, len(a.bugs)) == (b.checked, b.ok, len(b.bugs))


# --- the widened corpus across the six targets -----------------------------------


def test_corpus_parity_across_six_targets():
    # every real corpus program reproduces, claim-for-claim, on all six targets.
    for name in CORPUS:
        for t in TARGETS:
            res = check_module(PROGRAMS[name](), target=t)
            assert res.ok, (name, t, res.mismatches)


def test_corpus_per_target_scores_are_pinned():
    # the worked-example matrix, widened past toy kernels and frozen (regression net).
    def scores(name):
        return {
            t: optimize(PROGRAMS[name](), TARGETS[t], Theta.cool()).score for t in sorted(TARGETS)
        }

    # The tile lane is the widest the hardware can issue (capped at 16): AVX-512/SVE/RVV/PTX
    # issue the full 16-wide tile, but NEON tops out at vec4 and AVX2 at vec8 -- so the tiled
    # matmul costs strictly more on those parts (a realizable plan, not the old unrealizable 16).
    assert scores("matmul_tiled") == {
        "arm64_neon": 1703936,
        "arm64_sve": 1015808,
        "nvidia_ptx": 1015808,
        "riscv_rvv": 1015808,
        "x86_avx2": 1245184,
        "x86_avx512": 1015808,
    }
    assert scores("scan") == {
        "arm64_neon": 167936,
        "arm64_sve": 101888,
        "nvidia_ptx": 90880,
        "riscv_rvv": 101888,
        "x86_avx2": 123904,
        "x86_avx512": 101888,
    }
    assert scores("multi_histogram") == {
        "arm64_neon": 1602048,
        "arm64_sve": 1595520,
        "nvidia_ptx": 808000,
        "riscv_rvv": 1595520,
        "x86_avx2": 1597696,
        "x86_avx512": 1595520,
    }


def test_real_corpus_is_genuinely_multi_claim():
    # not toy single-claim skeletons: the optimizer sees real geometry.
    assert sum(len(p.claims) for p in matmul_tiled().phases) == 8  # 2x2x2 tiles
    assert sum(len(p.claims) for p in scan().phases) == 4  # 4 scan stages
    mh = multi_histogram()
    assert len(mh.phases) == 2 and sum(len(p.claims) for p in mh.phases) == 4


# --- the curated worked examples, reproduced through the generated machinery ------


def test_vector_add_7808_reproduced_by_law_rail():
    res = check_module(vector_add(1024), target="x86_avx512", theta="cool")
    assert res.ok and res.score == 7808
    pv = plan_view(vector_add(1024), TARGETS["x86_avx512"], Theta.cool())
    assert law_select(pv)[1000] == ("vec16", 7808)


def test_budget_9472_reproduced_by_constrained_law_rail():
    # vec16 (thermal 1088) infeasible under a 700 cap; the law's per-path feasible
    # argmin is vec8 @ 9472 -- the curated RCSP pin, recomputed by the generated rail.
    ms = check_budget(vector_add(1024), {"thermal": 700, "power": 700})
    assert ms == []
    pv = plan_view(
        vector_add(1024),
        TARGETS["x86_avx512"],
        Theta.cool(),
        result=optimize_constrained(
            vector_add(1024),
            TARGETS["x86_avx512"],
            Theta.cool(),
            PERF,
            Budget.of(thermal=700, power=700),
        ),
    )
    cap = ((_INDEX["thermal"], 700), (_INDEX["power"], 700))
    assert law_select(pv, cap)[1000] == ("vec8", 9472)


# --- the law rail is an independent algorithm (not the oracle's code path) --------


def test_law_select_is_argmin_not_shortest_path():
    # law_select recomputes per-claim argmin from declared costs; the oracle runs a
    # shortest path. They agree, but a hand-built counter-example proves law_select
    # is doing its own arithmetic (first wins on a tie; budget filters).
    from bcir.lower.mlir import ClaimView, PathView, PlanView

    cv = ClaimView(
        claim_id=1,
        phase_id=0,
        op="x",
        lane=None,
        stride_class=None,
        stride_k=1,
        count=1,
        reads=(),
        writes=(),
        domain=None,
        hazard="unique",
        verify="none",
        bounds="strict",
        paths=(
            PathView("a", None, (10,) + (0,) * 11),
            PathView("b", None, (4,) + (0,) * 11),
            PathView("c", None, (4,) + (0,) * 11),
        ),
        selected="b",
        score=8,
        width=1,
    )
    pv = PlanView("t", "latency", (2,) + (0,) * 11, (cv,), (0,), {0: ()}, 8)
    assert law_select(pv)[1] == ("b", 8)  # argmin, first of the tied 4s
    # a budget that excludes only the expensive candidate still yields the tie's first.
    assert law_select(pv, ((0, 5),))[1] == ("b", 8)
    assert law_select(pv, ((0, 100),))[1] == ("b", 8)
    # a budget that excludes *every* candidate is infeasible (the law raises).
    try:
        law_select(pv, ((0, 3),))
        raise AssertionError("expected infeasible")
    except ValueError:
        pass


# --- metamorphic properties (from the BCIR_Roadmap Phase B.2) ---------------------


def _rename(module, rid_off=100000, cid_off=500000):
    """ID-renaming: shift every RID and claim id by a constant (a relabeling that
    must preserve the plan score -- the optimizer is structural, not name-dependent)."""
    m = Module(name=module.name + "_r")
    for r in module.resources.values():
        m.add_resource(replace(r, rid=r.rid + rid_off))
    for ph in module.phases:
        claims = []
        for c in ph.claims:
            c2 = copy.deepcopy(c)
            c2.id += cid_off
            c2.rd = tuple(x + rid_off for x in c.rd)
            c2.wr = tuple(x + rid_off for x in c.wr)
            if c2.primary_rid is not None:
                c2.primary_rid += rid_off
            claims.append(c2)
        m.add_phase(Phase(phase_id=ph.phase_id, deps=ph.deps, claims=claims))
    return m


def test_id_renaming_preserves_the_plan_score():
    rng = random.Random(99)
    for _ in range(60):
        m = gen_module(rng)
        base = optimize(m, TARGETS["x86_avx512"], Theta.cool()).score
        renamed = optimize(_rename(m), TARGETS["x86_avx512"], Theta.cool()).score
        assert base == renamed


def test_tightening_a_budget_never_raises_the_capped_dimension():
    # a hard cap is honored, and a tighter cap can only lower (never raise) the
    # capped dimension's realized use -- a correctness property a budget-unaware
    # compiler has no concept of.
    m = vector_add(1024)
    h, th = TARGETS["x86_avx512"], Theta.cool()
    prev = None
    for cap in (4000, 2000, 1100, 900, 700):
        b = Budget.of(thermal=cap)
        r = optimize_constrained(m, h, th, PERF, b)
        used = plan_resources(r, th).v[_INDEX["thermal"]]
        assert feasible(r, th, b) and used <= cap
        if prev is not None:
            assert used <= prev
        prev = used


def test_reparse_roundtrip_preserves_the_plan():
    # plan_view (the IR-level view) -> law_select must recover the oracle's exact
    # per-claim selection + score: emitting and re-reading the IR is lossless.
    rng = random.Random(123)
    for _ in range(80):
        m = gen_module(rng)
        t = rng.choice(list(TARGETS))
        res = check_module(m, target=t)
        assert res.ok, res.mismatches


# --- verifier differential (illegal modules) + the overlap law net ----------------


def test_verifier_catches_every_injected_law():
    # generated illegal modules (one injected law each): the verifier must flag it,
    # and the un-mutated base must verify clean. (CI runs the deep campaign; the law-set
    # coverage is separately pinned by test_each_injector_is_caught_and_isolated at n=200.)
    assert (
        run_verifier_campaign(n=2000 if os.environ.get("BCIR_THOROUGH") else 400, seed=20240601)
        == []
    )


def test_each_injector_is_caught_and_isolated():
    from bcir.verify import verify

    rng = random.Random(13)
    seen = set()
    for _ in range(200):
        m, law = gen_illegal_module(rng)
        laws = {d.law for d in verify(m)}
        assert law in laws, (law, laws)
        seen.add(law)
    # module/claim rail: R2-R8 each exercised (R1 is enforced by construction -- the
    # registry is a dict keyed by RID, so a dup cannot exist to fault-inject).
    assert seen == {"R2", "R3", "R4", "R5", "R6", "R7", "R8"}


def test_plan_injectors_fire_R9():
    # plan rail: a clean optimal plan corrupted to violate R9 must be flagged by
    # verify_plan; the clean plan must verify clean.
    from bcir.kbcir.differential import gen_illegal_plan, check_plan_verifier

    rng = random.Random(29)
    for _ in range(60):
        m, r, law = gen_illegal_plan(rng)
        assert law == "R9"
        assert check_plan_verifier(m, r, law) == []


def test_artifact_laws_R10_to_R18_all_fire():
    # the artifact rails (pack R10/R11, lowering R12, provenance R13, smart-lowering
    # R14-R16, accuracy R17, compose R18): each clean fixture verifies clean and each
    # injected fault is flagged (R18 via plan_composite raising). Completes the
    # generative coverage of all 18 laws.
    from bcir.kbcir.differential import _artifact_law_misses

    for seed in range(8):
        misses = _artifact_law_misses(random.Random(seed))
        assert misses == [], [m.detail for m in misses]


def test_check_verifier_flags_a_missed_law():
    # a base with no fault has no expected law to find -> check_verifier reports the miss.
    m = gen_module(random.Random(1))
    assert check_verifier(m, "R7")  # R7 not present -> a miss is reported


def test_overlap_law_holds_across_generated_modules():
    # the (max,+) scheduled price is R9-consistent (makespan + gain == serial,
    # 0 <= makespan <= serial == score) on every generated module -- the conformance
    # net for the C++ overlap port.
    rng = random.Random(77)
    for _ in range(150):
        m = gen_module(rng)
        h = TARGETS[rng.choice(list(TARGETS))]
        r = _optimize(m, h, Theta.cool(), PERF)
        assert check_overlap(m, h, Theta.cool(), r, PERF) == []
        sp = price_scheduled(m, r, h, Theta.cool(), PERF)
        assert sp.makespan + sp.overlap_gain == sp.serial == r.score
        assert 0 <= sp.makespan <= sp.serial


# --- the emitter: well-formed IR + committed-corpus drift gate --------------------


def test_emitted_mlir_is_structurally_well_formed():
    for name in CORPUS:
        txt = to_mlir(PROGRAMS[name](), TARGETS["x86_avx512"], Theta.cool(), PERF, filecheck=True)
        assert txt.count("{") == txt.count("}")
        assert "bcir.module @" in txt and "bcir.kbcir.select" in txt
        assert "// RUN: bcir-opt" in txt and "// CHECK-LABEL:" in txt
        # every select declares a candidate set and a min-plus semiring.
        assert txt.count("bcir.kbcir.select") == txt.count("semiring = #bcir.semiring<min_plus>")


def test_committed_corpus_mlir_matches_the_emitter():
    # the committed gem_corpus.mlir must equal a fresh emission -- a reproducibility
    # gate so the MLIR-rail artifact never silently drifts from the oracle.
    path = os.path.normpath(_CORPUS_MLIR_PATH)
    assert os.path.exists(path), "run `python -m bcir.kbcir.differential --emit-corpus`"
    with open(path, encoding="utf-8") as f:
        committed = f.read()
    assert committed == emit_corpus_mlir(), "gem_corpus.mlir drifted; regenerate with --emit-corpus"


def test_committed_theta_mlir_matches_the_emitter():
    # the hot-Theta plan-parity artifact must equal a fresh emission (drift gate).
    from bcir.kbcir.differential import _THETA_MLIR_PATH, emit_theta_test

    path = os.path.normpath(_THETA_MLIR_PATH)
    assert os.path.exists(path), "run `python -m bcir.kbcir.differential --emit-theta`"
    with open(path, encoding="utf-8") as f:
        assert f.read() == emit_theta_test(), "theta_hot.mlir drifted; regenerate with --emit-theta"


# --- the shrinker -----------------------------------------------------------------


def test_shrink_minimizes_a_failing_module():
    # a synthetic predicate (fails iff the module still has a phase with >= 2 claims):
    # the shrinker must reduce a fat module toward the boundary while staying legal.
    from bcir.verify import verify

    rng = random.Random(5)
    big = None
    for _ in range(200):  # bounded: gen draws 1..4 claims/phase
        m = gen_module(rng)
        if any(len(p.claims) >= 2 for p in m.phases):
            big = m
            break
    assert big is not None
    fails = lambda mod: any(len(p.claims) >= 2 for p in mod.phases)  # noqa: E731
    small = shrink(big, fails)
    assert fails(small)  # the failure is preserved
    assert not verify(small)  # the witness stays verifier-legal
    assert sum(len(p.claims) for p in small.phases) <= sum(len(p.claims) for p in big.phases)
    # the minimal witness is one phase with exactly two claims (the predicate boundary).
    assert sum(len(p.claims) for p in small.phases) == 2 and len(small.phases) == 1


# --- the ultimate cross-check: real bcir-opt, when the toolchain is present -------


def test_bcir_opt_recomputes_the_corpus_when_available():
    bo = _find_bcir_opt()
    fc = _find_filecheck()
    if not bo:
        return  # MLIR toolchain not built in this environment (the oracle CI job)
    path = os.path.normpath(_CORPUS_MLIR_PATH)
    passes = [
        "-bcir-classify-lanes",
        "-bcir-select-realization",
        "-bcir-batch",
        "-bcir-schedule",
        "-bcir-lower-to-llvm",
    ]
    proc = subprocess.run([bo, *passes, path], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    if fc:
        check = subprocess.run([fc, path], input=proc.stdout, capture_output=True, text=True)
        assert check.returncode == 0, check.stderr


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None


def _find_filecheck():
    for name in ("FileCheck-19", "FileCheck-18", "FileCheck-17", "FileCheck"):
        p = shutil.which(name)
        if p:
            return p
    return None
