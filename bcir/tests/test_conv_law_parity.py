"""G7 dual-rail parity: the MLIR law rail (gem.conv + -bcir-lower-gem-conv) reproduces the oracle's
bcir/kbcir/conv.py verdict -- the chosen lowering STRATEGY (direct vs im2col+gemm), the inner im2col gemm
TILE, the roofline COST (compute/memory + the R17 accuracy bound), all PRICED THROUGH THE EXISTING matmul
roofline (a conv IS a structured matmul -- no bespoke roofline term). This is the dual-rail gate that
matters: the law's plan == the oracle's plan_conv for the same conv (the discipline is prototype-in-oracle
-> port-to-MLIR-law, parity-gated; mirrors how gem.matmul/gem.activation were ported -- test_activation_
law_parity.py is the bcir-opt driver template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative conv shapes it asserts the oracle's plan is
    internally consistent (bottleneck == max(compute, mem), the dual-semiring invariant the law verifier
    checks), that the conv is priced through matmul.cost_of (NOT a bespoke term), and that the strategy is
    direct|im2col -- the contract the MLIR op + verifier must mirror;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-lower-gem-conv` over that conv's oracle-pinned gem.conv IR and asserts the law lowers it
    clean, ERASES the plan, emits the equivalent im2col gemm's tile gem.block sequence, and annotates the
    SAME strategy/compute/mem/bottleneck/acc_bound the oracle computed. The oracle target is PINNED to
    x86-avx512 so the strategy/cost are host-independent (CI-agnostic), exactly like the activation parity.
"""

from __future__ import annotations

import os
import subprocess

from bcir.kbcir import matmul
from bcir.kbcir.conv import cost_of, cost_vector, plan_conv
from bcir.kbcir.cost import ACCURACY, TargetProfile

# Pin the oracle target so the chosen strategy / cost are deterministic regardless of the CI host arch
# (TargetProfile.for_host() would price the roofline differently per arch -- the parity claim must not
# depend on which runner executes it). avx512 is the reference, exactly as test_activation_law_parity.
_T = TargetProfile.x86_avx512()

# (in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype, quant_bits) -- a small dense conv and a strided/
# padded quantized one (the R17 bound). Both single-group 2-D convs, the G7 deliverable.
_CASES = [
    (2, 5, 5, 3, 3, 3, 1, 0, "f32", 0),  # a (C_in=2, 5x5) input, 3 filters, 3x3, valid -> out 3x3
    (3, 8, 8, 4, 3, 3, 1, 1, "f32", 8),  # padded same-conv, QUANTIZED -> the R17 Q8-bridge bound
]


def _oracle_plan(in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype, quant_bits):
    """The oracle's planned conv realization: the chosen strategy + the inner gemm tile + the cost_of
    roofline (priced through matmul.cost_of) + the cost_vector accuracy axis. Returns the dict of fields the
    law-rail op carries (so the two rails are compared field-by-field)."""
    spec = plan_conv(in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype, target=_T)
    m, n, k = spec.gemm_dims
    compute, mem, _ = cost_of(spec, spec.strategy, _T, spec.tile)
    cv = cost_vector(spec, quant_bits=quant_bits)
    # the inner gemm tile: the chosen matmul plan for im2col; the whole (untiled) gemm for direct.
    if spec.strategy == "direct":
        tm, tn, tk, lo = m, n, k, "ijk"
    else:
        tile = spec.tile or matmul.plan_matmul(m, n, k, _T)
        tm, tn, tk, lo = tile.tile_m, tile.tile_n, tile.tile_k, tile.loop_order
    return {
        "in_c": in_c,
        "in_h": in_h,
        "in_w": in_w,
        "out_c": out_c,
        "kh": kh,
        "kw": kw,
        "stride": stride,
        "pad": pad,
        "dtype": dtype,
        "out_h": spec.out_h,
        "out_w": spec.out_w,
        "gemm_m": m,
        "gemm_n": n,
        "gemm_k": k,
        "strategy": spec.strategy,
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "loop_order": lo,
        "compute": compute,
        "mem": mem,
        "bottleneck": max(compute, mem),
        "quant_bits": quant_bits,
        "acc_bound": cv.v[ACCURACY],
    }


def _conv_mlir(p) -> str:
    """Emit the gem.conv plan record MLIR for an oracle plan dict (the law-rail op the oracle pins)."""
    return (
        "bcir.module @m {\n"
        f"  bcir.gem.conv @cv {{ in_c = {p['in_c']} : i64, in_h = {p['in_h']} : i64, "
        f"in_w = {p['in_w']} : i64, out_c = {p['out_c']} : i64, kh = {p['kh']} : i64, "
        f"kw = {p['kw']} : i64, stride = {p['stride']} : i64, pad = {p['pad']} : i64, "
        f'dtype = "{p["dtype"]}", out_h = {p["out_h"]} : i64, out_w = {p["out_w"]} : i64, '
        f"gemm_m = {p['gemm_m']} : i64, gemm_n = {p['gemm_n']} : i64, gemm_k = {p['gemm_k']} : i64, "
        f'strategy = "{p["strategy"]}", tile_m = {p["tile_m"]} : i64, tile_n = {p["tile_n"]} : i64, '
        f'tile_k = {p["tile_k"]} : i64, loop_order = "{p["loop_order"]}", '
        f"compute_cost = {p['compute']} : i64, mem_cost = {p['mem']} : i64, "
        f"bottleneck = {p['bottleneck']} : i64, quant_bits = {p['quant_bits']} : i32, "
        f"acc_bound = {p['acc_bound']} : i64 }}\n"
        "}\n"
    )


def _ceil_div(a, b):
    return (a + b - 1) // b


# --- the pure-Python parity arm (always runs) -----------------------------------------------------


def test_oracle_conv_plan_is_self_consistent_and_priced_through_the_matmul_roofline():
    """The oracle's conv plan is internally consistent (bottleneck == max(compute, mem), the dual-semiring
    invariant the law verifier checks), its strategy is direct|im2col, and -- the load-bearing G7 property
    -- its (compute, mem) is EXACTLY matmul.cost_of on the equivalent im2col gemm dims (a conv IS a
    structured matmul; NO bespoke roofline term). This is the contract the MLIR op + verifier mirror."""
    for case in _CASES:
        p = _oracle_plan(*case)
        assert p["bottleneck"] == max(p["compute"], p["mem"]), f"{case}: bottleneck != max"
        assert p["strategy"] in ("direct", "im2col"), f"{case}: unknown strategy {p['strategy']}"
        # the derived out dims + the im2col gemm dims are internally consistent.
        in_h, in_w, kh, kw, stride, pad = (
            p["in_h"],
            p["in_w"],
            p["kh"],
            p["kw"],
            p["stride"],
            p["pad"],
        )
        assert p["out_h"] == (in_h + 2 * pad - kh) // stride + 1
        assert p["out_w"] == (in_w + 2 * pad - kw) // stride + 1
        assert (p["gemm_m"], p["gemm_n"], p["gemm_k"]) == (
            p["out_h"] * p["out_w"],
            p["out_c"],
            p["in_c"] * kh * kw,
        )
        # the conv is priced THROUGH the matmul roofline: cost_of == matmul.cost_of at the chosen tile.
        m, n, k = p["gemm_m"], p["gemm_n"], p["gemm_k"]
        mc, mm, _ = matmul.cost_of(m, n, k, p["tile_m"], p["tile_n"], p["tile_k"], _T)
        assert (p["compute"], p["mem"]) == (mc, mm), (
            f"{case}: cost != the matmul roofline (bespoke term?)"
        )
        # the R17 accuracy axis: the Q8-bridge 1-ULP bound iff quantized, else 0.
        assert p["acc_bound"] == (1 if p["quant_bits"] > 0 else 0), (
            f"{case}: R17 acc bound mismatch"
        )


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built tool) -----


def test_law_rail_reproduces_oracle_conv_plan_via_bcir_opt():
    """Drive `bcir-opt -bcir-lower-gem-conv` over the oracle-pinned gem.conv IR and assert the law
    reproduces the oracle's verdict: the plan is ERASED, the equivalent im2col gemm's tile gem.block
    sequence is emitted, and the recomputed kbcir.strategy/compute_cost/mem_cost/bottleneck/acc_bound +
    the inner gemm tile dims EQUAL the oracle's plan_conv/cost_of/cost_vector. The strategy/cost are
    host-independent because the oracle target is pinned to avx512."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_plan(*case)
        proc = subprocess.run(
            [bo, "-bcir-lower-gem-conv"], input=_conv_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, f"{case}: law rail rejected the oracle plan:\n{proc.stderr}"
        out = proc.stdout
        # genuine lowering: the plan record is consumed.
        assert "bcir.gem.conv" not in out, f"{case}: plan record not erased by the lowering"
        # the realization: the im2col gemm tiled -> ceil(M/tm)*ceil(N/tn)*ceil(K/tk) tile blocks @cv_t0.. .
        n_tiles = (
            _ceil_div(p["gemm_m"], p["tile_m"])
            * _ceil_div(p["gemm_n"], p["tile_n"])
            * _ceil_div(p["gemm_k"], p["tile_k"])
        )
        for t in range(n_tiles):
            assert f"@cv_t{t} " in out, f"{case}: missing tile @cv_t{t}"
        assert f"@cv_t{n_tiles} " not in out, f"{case}: emitted too many tiles (> {n_tiles})"
        # the dual-rail parity: the law's recomputed plan EQUALS the oracle's, field by field, on @cv_t0.
        t0 = next(l for l in out.splitlines() if "@cv_t0 " in l)
        assert f'kbcir.strategy = "{p["strategy"]}"' in t0, f"{case}: strategy != oracle\n{t0}"
        assert f"kbcir.gemm_m = {p['gemm_m']} : i64" in t0, f"{case}: gemm_m != oracle\n{t0}"
        assert f"kbcir.gemm_n = {p['gemm_n']} : i64" in t0, f"{case}: gemm_n != oracle\n{t0}"
        assert f"kbcir.gemm_k = {p['gemm_k']} : i64" in t0, f"{case}: gemm_k != oracle\n{t0}"
        assert f"kbcir.compute_cost = {p['compute']} : i64" in t0, (
            f"{case}: compute != oracle\n{t0}"
        )
        assert f"kbcir.mem_cost = {p['mem']} : i64" in t0, f"{case}: mem != oracle\n{t0}"
        assert f"kbcir.bottleneck = {p['bottleneck']} : i64" in t0, (
            f"{case}: bottleneck != oracle\n{t0}"
        )
        assert f"kbcir.acc_bound = {p['acc_bound']} : i64" in t0, f"{case}: R17 acc != oracle\n{t0}"
        assert f'kbcir.loop_order = "{p["loop_order"]}"' in t0, (
            f"{case}: loop_order != oracle\n{t0}"
        )


def test_law_rail_rejects_an_inconsistent_conv_plan():
    """The headline op-verifier gate, end-to-end on the law rail: a gem.conv whose declared im2col gemm
    dims do not match its conv metadata is REJECTED by the verifier (the structured-matmul consistency the
    oracle's check_conv enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    bad = (
        "bcir.module @m {\n"
        "  bcir.gem.conv @bad { in_c = 2 : i64, in_h = 5 : i64, in_w = 5 : i64, out_c = 3 : i64, "
        'kh = 3 : i64, kw = 3 : i64, stride = 1 : i64, pad = 0 : i64, dtype = "f32", '
        "out_h = 3 : i64, out_w = 3 : i64, gemm_m = 9 : i64, gemm_n = 3 : i64, gemm_k = 17 : i64, "
        'strategy = "im2col", tile_m = 9 : i64, tile_n = 3 : i64, tile_k = 17 : i64, '
        'loop_order = "ijk", compute_cost = 16 : i64, mem_cost = 68 : i64, bottleneck = 68 : i64, '
        "quant_bits = 0 : i32, acc_bound = 0 : i64 }\n}\n"
    )
    proc = subprocess.run([bo, "-bcir-lower-gem-conv"], input=bad, capture_output=True, text=True)
    assert proc.returncode != 0, "the law rail must reject an inconsistent im2col gemm dim"
    assert "declared im2col gemm dims" in proc.stderr, (
        f"the rejection must cite the structured-matmul consistency, got:\n{proc.stderr}"
    )


# --- the -bcir-gem-conv-cost cost-producer parity arm (mirrors test_activation_law_parity) --------
# The SIBLING cost pass to the lowering above: -bcir-gem-conv-cost RECOMPUTES the conv roofline (port
# of conv.py::cost_of -- a conv IS a structured matmul priced through matmul.cost_of on its im2col
# gemm dims, NO bespoke term) and parity-checks the declared compute/mem/bottleneck, annotating
# kbcir.* INFORMS-ONLY -- a cost PRODUCER (the op is NOT erased), not a lowering. This is the EXACT
# analog of -bcir-gem-matmul-cost / -bcir-gem-activation-cost (their *_law_parity.py); the three cost
# passes are a tight family. The same oracle-pinned IR (_conv_mlir / _oracle_plan) drives both rails.


def test_law_rail_reproduces_oracle_conv_roofline_via_bcir_opt():
    """Drive `bcir-opt -bcir-gem-conv-cost` over the oracle-pinned gem.conv IR and assert the law
    recomputes the SAME compute/mem/bottleneck the oracle's cost_of computed (conv.py, == matmul.cost_of
    on the im2col gemm dims), and annotates the INFORMS-ONLY quarantine record (kbcir.informs_only =
    true). The recomputed roofline is host-independent because the oracle target is pinned to avx512
    (the C++ pass pins the same x86-avx512 reference constants). A cost producer, NOT a lowering: the
    carrier op is NOT erased (cf. -bcir-lower-gem-conv above, which ERASES the plan into its tiles)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for case in _CASES:
        p = _oracle_plan(*case)
        proc = subprocess.run(
            [bo, "-bcir-gem-conv-cost"], input=_conv_mlir(p), capture_output=True, text=True
        )
        assert proc.returncode == 0, (
            f"{case}: cost rail rejected the oracle roofline:\n{proc.stderr}"
        )
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@cv " in l)
        assert f"kbcir.compute_cost = {p['compute']} : i64" in line, (
            f"{case}: compute != oracle\n{line}"
        )
        assert f"kbcir.mem_cost = {p['mem']} : i64" in line, f"{case}: mem != oracle\n{line}"
        assert f"kbcir.bottleneck = {p['bottleneck']} : i64" in line, (
            f"{case}: bottleneck != oracle\n{line}"
        )
        # the INFORMS-ONLY quarantine record: the recomputed roofline informs the plan, never gates legality.
        assert "kbcir.informs_only = true" in line, (
            f"{case}: missing the informs-only quarantine flag\n{line}"
        )
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.conv" in out, f"{case}: the conv op must NOT be erased (it is a producer)"


def test_cost_rail_rejects_an_inconsistent_conv_roofline():
    """The headline parity gate, end-to-end on the cost rail: a gem.conv whose declared compute_cost is
    NOT the recomputed roofline (the analytic cost the op verifier deliberately omits, but bottleneck ==
    max keeps it op-level well-formed) is REJECTED by -bcir-gem-conv-cost (the dual-rail R13 parity the
    oracle's cost_of enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    p = _oracle_plan(*_CASES[0])
    assert (
        p["compute"] == 16
    )  # sanity: the oracle's recomputed compute for the small conv (gemm 9x3x18)
    bad = dict(p)
    bad["compute"] = 9999  # a wrong compute term...
    bad["bottleneck"] = 9999  # ...with a matching bottleneck so the OP verifier still admits it
    proc = subprocess.run(
        [bo, "-bcir-gem-conv-cost"], input=_conv_mlir(bad), capture_output=True, text=True
    )
    assert proc.returncode != 0, "the cost rail must reject a non-cost_of compute term"
    assert "compute_cost 9999 != the recomputed roofline compute 16" in proc.stderr, (
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
