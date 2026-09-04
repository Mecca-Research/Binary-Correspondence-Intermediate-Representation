"""B1 dual-rail parity: the MLIR law rail (gem.matmul + -bcir-gem-matmul-cost) reproduces the oracle's
bcir/kbcir/matmul.py ROOFLINE COST -- the analytic compute term (M*N*K MACs / (vector_width*fma)), the
memory-traffic term (the blocked-GEMM A/B/C reuse / bandwidth), and the max,+ bottleneck = max(compute,
mem), all INFORMS-ONLY (the recomputed roofline informs the plan's cost but NEVER gates a legality verdict
-- the two-truth quarantine; every matmul realization reproduces the same result, the cost is the free
choice). This is the dual-rail gate that matters: the law's roofline == matmul.cost_of for the same
shape/tiling (the discipline is prototype-in-oracle -> port-to-MLIR-law, parity-gated; mirrors how
gem.activation/conv/attention + the sibling cost passes -cache-contention / -layout-pivot were ported --
test_layout_pivot_law_parity.py is the cost-producer parity template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative shapes it asserts the oracle's plan_matmul /
    cost_of are internally consistent under TargetProfile.x86_avx512() (bottleneck == max(compute, mem); the
    chosen plan's tiles in [1, dim] with a known loop order; the analytic cost is deterministic) -- the
    contract the MLIR op + the cost pass mirror;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-gem-matmul-cost` over the oracle-pinned gem.matmul IR and asserts the law recomputes the
    SAME compute/mem/bottleneck the oracle's cost_of computed AND annotates the informs-only quarantine flag.
    The oracle target is PINNED to x86-avx512 so the priced roofline is host-independent (CI-agnostic),
    exactly as test_layout_pivot_law_parity / test_cache_contention_law_parity pin their frozen constants.
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir.cost import TargetProfile
from bcir.kbcir.matmul import cost_of, plan_matmul

# Pin the oracle target so the priced roofline is deterministic regardless of the CI host arch
# (TargetProfile.for_host() would price the roofline differently per arch). avx512 is the reference,
# exactly as the sibling cost-producer parity tests pin their frozen constants.
_T = TargetProfile.x86_avx512()

# (M, N, K, tile_m, tile_n, tile_k, loop_order) -- representative oracle-priced tilings:
#   * a SQUARE untiled matmul (the whole-dimension realization): compute-bound, does NOT fit the budget.
#   * the SAME shape tiled 32^3: less reuse (more traffic) but the working set fits the cache budget.
#   * a non-square tiling (a different reuse ratio).
_CASES = [
    (
        64,
        64,
        64,
        64,
        64,
        64,
        "ijk",
    ),  # untiled square: compute 8192, mem 4096, bottleneck 8192, does not fit
    (
        64,
        64,
        64,
        32,
        32,
        32,
        "ijk",
    ),  # tiled 32^3: compute 8192, mem 7168, bottleneck 8192, fits the budget
    (128, 64, 32, 64, 32, 16, "ikj"),  # non-square tiling (a different A/B reuse ratio)
]


def _oracle_cost(M, N, K, tm, tn, tk, lo):
    """The oracle's priced roofline for a (shape, tiling) under the pinned x86-avx512 target. Returns the
    dict of fields the law-rail op carries (so the two rails are compared field-by-field)."""
    compute, mem, fits = cost_of(M, N, K, tm, tn, tk, _T)
    return {
        "m": M,
        "n": N,
        "k": K,
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "loop_order": lo,
        "compute_cost": compute,
        "mem_cost": mem,
        "bottleneck": max(compute, mem),
        "fits_cache": fits,
    }


def _matmul_mlir(p) -> str:
    """Emit the gem.matmul carrier-op MLIR for an oracle cost dict (the law-rail op the oracle pins)."""
    return (
        "bcir.module @m {\n"
        f"  bcir.gem.matmul @mm {{ m = {p['m']} : i64, n = {p['n']} : i64, k = {p['k']} : i64, "
        f"tile_m = {p['tile_m']} : i64, tile_n = {p['tile_n']} : i64, tile_k = {p['tile_k']} : i64, "
        f'loop_order = "{p["loop_order"]}", '
        f"compute_cost = {p['compute_cost']} : i64, mem_cost = {p['mem_cost']} : i64, "
        f"bottleneck = {p['bottleneck']} : i64 }}\n"
        "}\n"
    )


# --- the pure-Python parity arm (always runs) -----------------------------------------------------


def test_oracle_matmul_roofline_is_self_consistent_and_priced_through_the_cost_model():
    """The oracle's matmul roofline is internally consistent: the bottleneck is the max,+ over the compute
    and memory terms; the analytic cost is deterministic (cost_of has no nondeterminism); and plan_matmul's
    chosen plan is well-formed (tiles in [1, dim], a known loop order, bottleneck == max, fits_cache booled).
    The cost is priced THROUGH THE EXISTING analytic roofline (the compute/traffic terms), not a bespoke
    discount. This is the contract the MLIR op + the -bcir-gem-matmul-cost pass mirror."""
    for M, N, K, tm, tn, tk, lo in _CASES:
        p = _oracle_cost(M, N, K, tm, tn, tk, lo)
        # the max,+ step: the bottleneck is the binding roofline resource.
        assert p["bottleneck"] == max(p["compute_cost"], p["mem_cost"]), (
            f"{(M, N, K)}: bottleneck != max"
        )
        assert p["compute_cost"] >= 0 and p["mem_cost"] >= 0, f"{(M, N, K)}: negative roofline term"
        # determinism: cost_of recomputes bit-identically (no float / no host state).
        again = cost_of(M, N, K, tm, tn, tk, _T)
        assert again == (p["compute_cost"], p["mem_cost"], p["fits_cache"]), (
            f"{(M, N, K)}: non-deterministic"
        )
    # plan_matmul's chosen plan is internally consistent (the dual-semiring search's invariants).
    for M, N, K in [(64, 64, 64), (128, 64, 32), (96, 48, 64)]:
        plan = plan_matmul(M, N, K, _T)
        assert 1 <= plan.tile_m <= M and 1 <= plan.tile_n <= N and 1 <= plan.tile_k <= K, (
            f"{(M, N, K)}: plan tile out of [1, dim]"
        )
        assert plan.loop_order in ("ijk", "ikj", "jik"), f"{(M, N, K)}: unknown loop order"
        assert plan.bottleneck == max(plan.compute_cost, plan.mem_cost), (
            f"{(M, N, K)}: plan bottleneck != max"
        )
        # the plan's declared roofline == cost_of at its chosen tile (the law op carries exactly this).
        c, m, fits = cost_of(M, N, K, plan.tile_m, plan.tile_n, plan.tile_k, _T)
        assert (plan.compute_cost, plan.mem_cost, plan.fits_cache) == (c, m, fits), (
            f"{(M, N, K)}: plan roofline != cost_of at its tile"
        )


def test_matmul_cost_never_gates_legality_verify_plan_ignores_it():
    """The two-truth quarantine: the recomputed matmul roofline informs the plan's cost but is a cost lever,
    NOT a legality verdict -- the legality verifier never reads the matmul cost. We assert the legality
    verifier source never branches on a matmul roofline term (the structural quarantine the MLIR -bcir-verify
    rail also preserves -- it never reads the kbcir.compute_cost/mem_cost/bottleneck annotations the cost
    pass writes; informs-only, exactly as -cache-contention / -layout-pivot)."""
    import inspect

    import bcir.verify as verify_mod

    src = inspect.getsource(verify_mod)
    assert "bottleneck" not in src, (
        "the legality verifier must NEVER read the matmul roofline bottleneck (the informs-only quarantine)"
    )


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built bcir-opt) -


def test_law_rail_reproduces_oracle_matmul_roofline_via_bcir_opt():
    """Drive `bcir-opt -bcir-gem-matmul-cost` over the oracle-pinned gem.matmul IR and assert the law
    recomputes the SAME compute/mem/bottleneck the oracle's cost_of computed (matmul.py), and annotates the
    INFORMS-ONLY quarantine record (kbcir.informs_only = true). The recomputed roofline is host-independent
    because the oracle target is pinned to avx512 (the C++ pass pins the same x86-avx512 reference constants)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_cost(*case)
        proc = subprocess.run(
            [bo, "-bcir-gem-matmul-cost"], input=_matmul_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            f"{case}: law rail rejected the oracle roofline:\n{proc.stderr}"
        )
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@mm " in l)
        assert f"kbcir.compute_cost = {p['compute_cost']} : i64" in line, (
            f"{case}: compute != oracle\n{line}"
        )
        assert f"kbcir.mem_cost = {p['mem_cost']} : i64" in line, f"{case}: mem != oracle\n{line}"
        assert f"kbcir.bottleneck = {p['bottleneck']} : i64" in line, (
            f"{case}: bottleneck != oracle\n{line}"
        )
        # the INFORMS-ONLY quarantine record: the recomputed roofline informs the plan, never gates legality.
        assert "kbcir.informs_only = true" in line, (
            f"{case}: missing the informs-only quarantine flag\n{line}"
        )
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.matmul" in out, (
            f"{case}: the matmul op must NOT be erased (it is a producer)"
        )


def test_law_rail_rejects_an_inconsistent_matmul_roofline():
    """The headline parity gate, end-to-end on the law rail: a gem.matmul whose declared compute_cost is NOT
    the recomputed roofline (the analytic cost the op verifier deliberately omits, but bottleneck == max keeps
    it op-level well-formed) is REJECTED by -bcir-gem-matmul-cost (the dual-rail R13 parity the oracle's
    cost_of enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    p = _oracle_cost(64, 64, 64, 64, 64, 64, "ijk")
    assert p["compute_cost"] == 8192  # sanity: the oracle's recomputed compute for this tiling
    bad = p.copy()
    bad["compute_cost"] = 9999  # a wrong compute term...
    bad["bottleneck"] = 9999  # ...with a matching bottleneck so the OP verifier still admits it
    proc = subprocess.run(
        [bo, "-bcir-gem-matmul-cost"], input=_matmul_mlir(bad), capture_output=True, text=True
    )
    assert proc.returncode != 0, "the law rail must reject a non-cost_of compute term"
    assert "compute_cost 9999 != the recomputed roofline compute 8192" in proc.stderr, (
        f"the rejection must cite the recomputed roofline compute, got:\n{proc.stderr}"
    )


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
