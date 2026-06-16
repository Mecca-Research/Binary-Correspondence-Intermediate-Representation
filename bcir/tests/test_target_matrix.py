"""Six-target capability matrix: per-target Python<->MLIR parity pins + drift gate.

Six-target capability matrix (docs/BCIR_MASTER_ROADMAP.md §4): emit + test all six TARGETS so the
C++ optimizer-core passes -- ``-bcir-plan`` (the coupled shortest path), ``-bcir-
overlap`` (the (max,+) scheduled price), ``-bcir-rcsp-plan`` (the accumulated-budget
label DP) -- reproduce the oracle *per target*. The real bcir-opt cross-check runs in
CI (mlir-rail-validate, on ``mlir/test/passes/target_matrix.mlir``); these tests guard
the *generated* artifact and pin the oracle side:

  * the committed matrix file must equal a fresh oracle emission (the drift gate CI
    also enforces with ``--emit-matrix`` + ``git diff --exit-code``);
  * the ``bcir.target.capability`` op must carry every seed the law reads, per target,
    so a non-x86 target plans from its own constants, not the CPU defaults;
  * the headline per-target plans / overlap gains / constrained optima are pinned, so
    the six-target parity (today only argued via ``run_campaign``) is an explicit,
    committed regression on both the corpus and the matrix programs.
"""

import os

from bcir.examples import CORPUS, PROGRAMS
from bcir.gem.overlap import price_scheduled
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir import differential as diff
from bcir.kbcir.cost import Theta
from bcir.kbcir.rcsp import Budget, optimize_constrained
from bcir.kbcir.weights import PERF
from bcir.lower.mlir import to_mlir

COOL = Theta.cool()
ALL_TARGETS = ("x86_avx2", "x86_avx512", "arm64_neon", "arm64_sve", "nvidia_ptx", "riscv_rvv")


def test_targets_registry_is_exactly_the_six():
    assert set(TARGETS) == set(ALL_TARGETS)


def test_matrix_file_matches_a_fresh_emission():
    """The committed target_matrix.mlir is a deterministic function of the oracle --
    if the cost model moves, the artifact is stale and must be regenerated (this is
    the same drift gate CI runs with `--emit-matrix` + `git diff --exit-code`)."""
    path = os.path.normpath(diff._MATRIX_MLIR_PATH)
    with open(path, encoding="utf-8") as f:
        committed = f.read()
    assert committed == diff.emit_target_matrix(), (
        "mlir/test/passes/target_matrix.mlir is stale -- regenerate with "
        "`python -m bcir.kbcir.differential --emit-matrix`")


def test_emission_is_deterministic():
    assert diff.emit_target_matrix() == diff.emit_target_matrix()


def test_capability_op_carries_every_target_seed():
    """A faithful per-target descriptor: every TargetProfile seed the C++ Cap reads
    (plus warp / mem_channels / affinity_domains the scheduler/overlap read) is emitted
    with the target's own value -- not the defaulted CPU seeds."""
    for name, h in TARGETS.items():
        ir = to_mlir(PROGRAMS["vector_add"](), h, COOL, PERF)
        cap = next(ln for ln in ir.splitlines() if "bcir.target.capability" in ln)
        assert f'triple = "{h.triple}"' in cap, name
        widths = ", ".join(str(w) for w in h.lane_widths)
        assert f"lane_widths = array<i64: {widths}>" in cap, name
        for field, value in (("warp", h.warp), ("cacheline", h.cacheline),
                             ("gather_penalty", h.gather_penalty),
                             ("affinity_domains", h.affinity_domains),
                             ("mem_channels", h.mem_channels),
                             ("thermal_density", h.thermal_density),
                             ("power_density", h.power_density),
                             ("mem_unit", h.mem_unit), ("base_overhead", h.base_overhead),
                             ("per_op_heat", h.per_op_heat), ("elem_bytes", h.elem_bytes)):
            assert f"{field} = {value} : i32" in cap, f"{name}: {field}"


# --- the headline per-target plans (regression pins) -----------------------------

# vector_add: the same C = A + B graph realizes at the target's max lane width, so the
# plan score is the per-target signal (the cost model recomputes these from the
# capability seeds alone -- the claim of the matrix).
_VEC_ADD = {"x86_avx2": (8, 9472), "x86_avx512": (16, 7808), "arm64_neon": (4, 12800),
            "arm64_sve": (16, 7808), "nvidia_ptx": (32, 6976), "riscv_rvv": (16, 7808)}


def test_vector_add_plan_is_per_target():
    for t, (width, score) in _VEC_ADD.items():
        r = optimize(PROGRAMS["vector_add"](), TARGETS[t], COOL, PERF)
        assert r.by_claim()[1000].width == width, f"{t} width"
        assert r.score == score, f"{t} score"


# The widened corpus (matmul / scan / multi_histogram) carried across all six targets
# -- the committed gem_corpus.mlir pins only x86_avx512; this pins the rest (the
# "six-target parity proven on the Python rail" made an explicit regression).
_CORPUS_PLAN = {
    "matmul_tiled": {"x86_avx2": 1015808, "x86_avx512": 1015808, "arm64_neon": 1015808,
                     "arm64_sve": 1015808, "nvidia_ptx": 1015808, "riscv_rvv": 1015808},
    "scan": {"x86_avx2": 123904, "x86_avx512": 101888, "arm64_neon": 167936,
             "arm64_sve": 101888, "nvidia_ptx": 90880, "riscv_rvv": 101888},
    "multi_histogram": {"x86_avx2": 1597696, "x86_avx512": 1595520, "arm64_neon": 1602048,
                        "arm64_sve": 1595520, "nvidia_ptx": 808000, "riscv_rvv": 1595520},
}


def test_corpus_plan_is_per_target():
    for name in CORPUS:
        for t, score in _CORPUS_PLAN[name].items():
            assert optimize(PROGRAMS[name](), TARGETS[t], COOL, PERF).score == score, \
                f"{name} on {t}"


def test_nvidia_gather_is_cheaper_than_the_cpus():
    """The GPU's coalesced gather (gather_penalty 16 vs the CPUs' 32) must halve the
    random-access cost -- a per-target capability effect, not a tuning constant."""
    cpu = optimize(PROGRAMS["multi_histogram"](), TARGETS["x86_avx512"], COOL, PERF).score
    gpu = optimize(PROGRAMS["multi_histogram"](), TARGETS["nvidia_ptx"], COOL, PERF).score
    assert gpu < cpu // 1.9, f"GPU gather {gpu} should roughly halve CPU {cpu}"


# --- overlap (max,+) per target --------------------------------------------------

# fused_chain: two independent shared-input claims overlap (makespan < serial); the
# gain is per-target (it scales with the per-target plan), and every target overlaps.
_FUSED_GAIN = {"x86_avx2": 7168, "x86_avx512": 5888, "arm64_neon": 9728,
               "arm64_sve": 5888, "nvidia_ptx": 5248, "riscv_rvv": 5888}


def test_fused_chain_overlap_gain_is_per_target():
    for t, gain in _FUSED_GAIN.items():
        r = optimize(PROGRAMS["fused_chain"](), TARGETS[t], COOL, PERF)
        sp = price_scheduled(PROGRAMS["fused_chain"](), r, TARGETS[t], COOL, PERF)
        assert sp.overlap_gain == gain, f"{t}"
        assert sp.makespan + sp.overlap_gain == sp.serial == r.score, f"{t} R9"
        assert sp.overlap_gain > 0, f"{t}: independent claims must overlap"


def test_scan_chain_serializes_on_every_target():
    """The RAW dependency chain cannot overlap (gain 0) on any target -- the
    counterpart to fused_chain -- though it still fuses (deforestation lowers plan)."""
    for t in TARGETS:
        r = optimize(PROGRAMS["scan_chain"](), TARGETS[t], COOL, PERF)
        sp = price_scheduled(PROGRAMS["scan_chain"](), r, TARGETS[t], COOL, PERF)
        assert sp.overlap_gain == 0, f"{t}: serialized chain has no overlap"


# --- plan-level RCSP per target --------------------------------------------------

def test_rcsp_matrix_binds_where_an_intermediate_lane_lowers_heat():
    """The matrix RCSP section caps the plan's thermal at (accumulated - 1). On targets
    with a (1, 8, 16) ladder a width-8 lane is heat-cheaper than width-16, so the cap
    narrows ONE claim ({16,16} -> {16,8} @ 17280) -- the accumulated-budget decision.
    On targets whose widest lane is already heat-minimal (avx2 / neon / ptx) the cap
    is feasibly inert (the plan is unchanged). Both are valid per-target parity."""
    binds = {"x86_avx512", "arm64_sve", "riscv_rvv"}
    for t, h in TARGETS.items():
        module = diff._two_independent_vec()
        budget, rc = diff._rcsp_for_target(module, h, COOL)
        unconstrained = optimize(module, h, COOL, PERF)
        widths = sorted(s.candidate.width for s in rc.steps)
        if t in binds:
            assert rc.score == 17280, t
            assert widths == [8, 16], t
            assert rc.score > unconstrained.score, f"{t}: cap must bind"
        else:
            # feasible, non-binding: the constrained optimum equals the free plan.
            assert rc.score == unconstrained.score, t
        # the budget the matrix emits really does admit the constrained plan.
        assert all(s.candidate.width >= 1 for s in rc.steps)
        assert optimize_constrained(module, h, COOL, PERF, budget).score == rc.score, t


def test_matrix_covers_plan_overlap_and_rcsp_for_all_six():
    """Structural: the artifact's RUN line drives all three optimizer-core passes and
    the file anchors a CHECK block for every (program, target) and every RCSP target."""
    text = diff.emit_target_matrix()
    assert "-bcir-plan -bcir-overlap -bcir-rcsp-plan" in text
    for name in diff.MATRIX:
        for t in TARGETS:
            label = diff._module_label(PROGRAMS[name]().name, t)
            assert f"// CHECK-LABEL: bcir.module @{label}\n" in text, label
    for t in TARGETS:
        label = diff._module_label("two_vec", t)
        assert f"// CHECK-LABEL: bcir.module @{label}\n" in text, label
    # 5 programs x 6 targets + 6 RCSP modules.
    assert text.count("// CHECK-LABEL: bcir.module @") == len(diff.MATRIX) * 6 + 6
