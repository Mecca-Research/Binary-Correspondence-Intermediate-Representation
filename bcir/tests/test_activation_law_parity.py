"""G1 dual-rail parity: the MLIR law rail (gem.activation + -bcir-lower-gem-activation) reproduces the
oracle's bcir/kbcir/activation.py verdict -- the chosen lane WIDTH, the roofline COST (compute/memory +
the R17 accuracy bound), and the QUARANTINE split (relu exact / no libm edge; the transcendentals route
expf/tanhf through the trusted c.call.libm: edge). This is the dual-rail gate that matters: the law's
plan == the oracle's plan for the same op (the discipline is prototype-in-oracle -> port-to-MLIR-law,
parity-gated; mirrors how gem.matmul was ported -- test_differential.py is the bcir-opt driver template).

Two arms:
  * a PURE-PYTHON parity arm (always runs): for representative ops it hand-builds the gem.activation MLIR
    from the oracle's plan_activation/cost_of/cost_vector output, so the IR the law sees IS the oracle's
    decision -- and asserts the oracle's own checks (the quarantine side, the cost wiring) agree;
  * the ULTIMATE cross-check (self-skips without the built bcir-opt): drives the real
    `bcir-opt -bcir-lower-gem-activation` over that IR and asserts the law lowers it clean, ERASES the
    plan, emits ceil(count/width) gem.block stripes, and annotates the SAME width/compute/mem/bottleneck/
    acc_bound + the quarantine verdict (kbcir.lane_kind exact|transcendental, kbcir.libm_edge) the oracle
    computed. The oracle target is PINNED to x86-avx512 so the width/cost are host-independent (CI-agnostic).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from bcir.kbcir.activation import (ActivationSpec, cost_of, cost_vector, is_exact, libm_edges,
                                   plan_activation)
from bcir.kbcir.cost import ACCURACY, COMPUTE, MEMORY, TargetProfile

# Pin the oracle target so the chosen width / cost are deterministic regardless of the CI host arch
# (TargetProfile.for_host() would pick width 4 on ARM, 8 on AVX2, 16 on AVX-512 -- the parity claim must
# not depend on which runner executes it). avx512's lane set {1,8,16} is the reference.
_T = TargetProfile.x86_avx512()

# (kind, shape, axis_len, dtype, quant_bits) -- relu exact, a transcendental with the R17 bound, softmax.
_CASES = [
    ("relu", (64,), 0, "f32", 0),       # the EXACT, 0-ULP integer/Q-fixed-clean activation
    ("relu", (32,), 0, "i32", 0),       # relu MAY be integer (the clean quarantine side)
    ("gelu", (64,), 0, "f32", 0),       # a transcendental (tanhf edge), the heaviest op_weight
    ("sigmoid", (32,), 0, "f32", 8),    # a transcendental (expf edge), QUANTIZED -> the R17 bound
    ("softmax", (4, 8), 8, "f32", 8),   # the last-axis reduce, QUANTIZED -> the R17 bound
]


def _oracle_plan(kind, shape, axis_len, dtype, quant_bits):
    """The oracle's planned realization: chosen width + the cost_of roofline + the cost_vector accuracy
    axis. Returns the dict of fields the law-rail op carries (so the two rails are compared field-by-field)."""
    spec = plan_activation(kind, shape, dtype=dtype, axis_len=axis_len, target=_T)
    compute, mem = cost_of(spec, spec.width, _T)
    cv = cost_vector(spec, quant_bits=quant_bits)
    return {
        "kind": kind, "shape": tuple(shape), "dtype": dtype, "axis_len": axis_len,
        "width": spec.width, "compute": compute, "mem": mem, "bottleneck": max(compute, mem),
        "quant_bits": quant_bits, "acc_bound": cv.v[ACCURACY],
        "lane_kind": "exact" if is_exact(kind) else "transcendental",
        "libm_edge": (libm_edges(kind)[0] if libm_edges(kind) else ""),
    }


def _activation_mlir(p) -> str:
    """Emit the gem.activation plan record MLIR for an oracle plan dict (the law-rail op the oracle pins)."""
    shape = ", ".join(str(d) for d in p["shape"])
    return (
        'bcir.module @m {\n'
        f'  bcir.gem.activation @a {{ kind = "{p["kind"]}", shape = array<i64: {shape}>, '
        f'dtype = "{p["dtype"]}", axis_len = {p["axis_len"]} : i64, width = {p["width"]} : i64, '
        f'compute_cost = {p["compute"]} : i64, mem_cost = {p["mem"]} : i64, '
        f'bottleneck = {p["bottleneck"]} : i64, quant_bits = {p["quant_bits"]} : i32, '
        f'acc_bound = {p["acc_bound"]} : i64 }}\n'
        '}\n'
    )


# --- the pure-Python parity arm (always runs) -----------------------------------------------------

def test_oracle_plan_is_self_consistent_and_quarantine_split_holds():
    """The oracle's plan is internally consistent (bottleneck == max(compute, mem), the dual-semiring
    invariant the law verifier checks) and honors the quarantine split: relu is exact (no libm edge, 0
    accuracy cost when dense); a transcendental carries a libm edge and -- when quantized -- the 1-ULP R17
    bound. This is the contract the MLIR op + verifier must mirror."""
    for kind, shape, axis_len, dtype, quant_bits in _CASES:
        p = _oracle_plan(kind, shape, axis_len, dtype, quant_bits)
        assert p["bottleneck"] == max(p["compute"], p["mem"]), f"{kind}: bottleneck != max"
        if kind == "relu":
            assert p["lane_kind"] == "exact" and p["libm_edge"] == "", "relu must be exact, no libm edge"
        else:
            assert p["lane_kind"] == "transcendental" and p["libm_edge"] in ("expf", "tanhf"), \
                f"{kind} must route a libm edge"
            assert dtype == "f32", "a transcendental needs f32 (the quarantine dtype rule)"
        # the R17 accuracy axis: the Q8-bridge 1-ULP bound iff quantized, else 0 (relu exact, libm trusted).
        assert p["acc_bound"] == (1 if quant_bits > 0 else 0), f"{kind}: R17 acc bound mismatch"


# --- the ultimate cross-check: the real bcir-opt law rail (self-skips without the built tool) -----

def test_law_rail_reproduces_oracle_plan_via_bcir_opt():
    """Drive `bcir-opt -bcir-lower-gem-activation` over the oracle-pinned gem.activation IR and assert the
    law reproduces the oracle's verdict: the plan is ERASED, ceil(count/width) gem.block stripes are
    emitted, and the recomputed kbcir.width/compute_cost/mem_cost/bottleneck/acc_bound + the quarantine
    verdict (kbcir.lane_kind / kbcir.libm_edge) EQUAL the oracle's plan_activation/cost_of/cost_vector.
    The width/cost are host-independent because the oracle target is pinned to avx512."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for kind, shape, axis_len, dtype, quant_bits in _CASES:
        p = _oracle_plan(kind, shape, axis_len, dtype, quant_bits)
        proc = subprocess.run([bo, "-bcir-lower-gem-activation"], input=_activation_mlir(p),
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{kind}: law rail rejected the oracle plan:\n{proc.stderr}"
        out = proc.stdout
        # genuine lowering: the plan record is consumed.
        assert "bcir.gem.activation" not in out, f"{kind}: plan record not erased by the lowering"
        # the realization: ceil(count/width) stripes named @a_s0.. .
        count = 1
        for d in shape:
            count *= d
        n_stripes = (count + p["width"] - 1) // p["width"]
        for s in range(n_stripes):
            assert f"@a_s{s} " in out, f"{kind}: missing stripe @a_s{s}"
        assert f"@a_s{n_stripes} " not in out, f"{kind}: emitted too many stripes (> {n_stripes})"
        # the dual-rail parity: the law's recomputed plan EQUALS the oracle's, field by field.
        s0 = next(l for l in out.splitlines() if "@a_s0 " in l)
        assert f'kbcir.width = {p["width"]} : i64' in s0, f"{kind}: width != oracle {p['width']}\n{s0}"
        assert f'kbcir.compute_cost = {p["compute"]} : i64' in s0, f"{kind}: compute != oracle\n{s0}"
        assert f'kbcir.mem_cost = {p["mem"]} : i64' in s0, f"{kind}: mem != oracle\n{s0}"
        assert f'kbcir.bottleneck = {p["bottleneck"]} : i64' in s0, f"{kind}: bottleneck != oracle\n{s0}"
        assert f'kbcir.acc_bound = {p["acc_bound"]} : i64' in s0, f"{kind}: R17 acc != oracle\n{s0}"
        assert f'kbcir.lane_kind = "{p["lane_kind"]}"' in s0, f"{kind}: quarantine side != oracle\n{s0}"
        assert f'kbcir.libm_edge = "{p["libm_edge"]}"' in s0, f"{kind}: libm edge != oracle\n{s0}"


def test_law_rail_rejects_the_quarantine_violation():
    """The headline quarantine gate, end-to-end on the law rail: a transcendental on an integer dtype is
    REJECTED by the gem.activation op verifier (its libm edge returns float). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    bad = ('bcir.module @m {\n'
           '  bcir.gem.activation @bad { kind = "sigmoid", shape = array<i64: 8>, dtype = "i32", '
           'axis_len = 0 : i64, width = 8 : i64, compute_cost = 8 : i64, mem_cost = 16 : i64, '
           'bottleneck = 16 : i64, quant_bits = 0 : i32, acc_bound = 0 : i64 }\n}\n')
    proc = subprocess.run([bo, "-bcir-lower-gem-activation"], input=bad, capture_output=True, text=True)
    assert proc.returncode != 0, "the law rail must reject a transcendental on an integer dtype"
    assert "transcendental activation needs f32" in proc.stderr, \
        f"the rejection must cite the quarantine rule, got:\n{proc.stderr}"


# --- the -bcir-gem-activation-cost cost-producer parity arm (mirrors test_matmul_law_parity) -----
# The SIBLING cost pass to the lowering above: -bcir-gem-activation-cost RECOMPUTES the activation
# roofline (port of activation.py::cost_of) and parity-checks the declared compute/mem/bottleneck,
# annotating kbcir.* INFORMS-ONLY -- a cost PRODUCER (the op is NOT erased), not a lowering. This is
# the EXACT analog of -bcir-gem-matmul-cost (test_matmul_law_parity.py); the two cost passes are a
# tight family. The same oracle-pinned IR (_activation_mlir / _oracle_plan) drives both rails.


def test_law_rail_reproduces_oracle_activation_roofline_via_bcir_opt():
    """Drive `bcir-opt -bcir-gem-activation-cost` over the oracle-pinned gem.activation IR and assert
    the law recomputes the SAME compute/mem/bottleneck the oracle's cost_of computed (activation.py),
    and annotates the INFORMS-ONLY quarantine record (kbcir.informs_only = true). The recomputed
    roofline is host-independent because the oracle target is pinned to avx512 (the C++ pass pins the
    same x86-avx512 reference constants). A cost producer, NOT a lowering: the carrier op is NOT erased
    (cf. -bcir-lower-gem-activation above, which ERASES the plan into its stripes)."""
    bo = _find_bcir_opt()
    if not bo:
        return  # the MLIR toolchain is not built in this environment (the oracle-only CI job)
    for kind, shape, axis_len, dtype, quant_bits in _CASES:
        p = _oracle_plan(kind, shape, axis_len, dtype, quant_bits)
        proc = subprocess.run([bo, "-bcir-gem-activation-cost"], input=_activation_mlir(p),
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{kind}: cost rail rejected the oracle roofline:\n{proc.stderr}"
        out = proc.stdout
        line = next(l for l in out.splitlines() if "@a " in l)
        assert f'kbcir.compute_cost = {p["compute"]} : i64' in line, f"{kind}: compute != oracle\n{line}"
        assert f'kbcir.mem_cost = {p["mem"]} : i64' in line, f"{kind}: mem != oracle\n{line}"
        assert f'kbcir.bottleneck = {p["bottleneck"]} : i64' in line, f"{kind}: bottleneck != oracle\n{line}"
        # the INFORMS-ONLY quarantine record: the recomputed roofline informs the plan, never gates legality.
        assert "kbcir.informs_only = true" in line, f"{kind}: missing the informs-only quarantine flag\n{line}"
        # a cost producer, NOT a lowering: the carrier op is NOT erased.
        assert "bcir.gem.activation" in out, f"{kind}: the activation op must NOT be erased (it is a producer)"


def test_cost_rail_rejects_an_inconsistent_activation_roofline():
    """The headline parity gate, end-to-end on the cost rail: a gem.activation whose declared
    compute_cost is NOT the recomputed roofline (the analytic cost the op verifier deliberately omits,
    but bottleneck == max keeps it op-level well-formed) is REJECTED by -bcir-gem-activation-cost (the
    dual-rail R13 parity the oracle's cost_of enforces). Self-skips without bcir-opt."""
    bo = _find_bcir_opt()
    if not bo:
        return
    p = _oracle_plan("relu", (64,), 0, "f32", 0)
    assert p["compute"] == 2  # sanity: the oracle's recomputed compute for relu over 64 elements
    bad = dict(p)
    bad["compute"] = 9999                  # a wrong compute term...
    bad["bottleneck"] = 9999               # ...with a matching bottleneck so the OP verifier still admits it
    proc = subprocess.run([bo, "-bcir-gem-activation-cost"], input=_activation_mlir(bad),
                          capture_output=True, text=True)
    assert proc.returncode != 0, "the cost rail must reject a non-cost_of compute term"
    assert "compute_cost 9999 != the recomputed roofline compute 2" in proc.stderr, \
        f"the rejection must cite the recomputed roofline compute, got:\n{proc.stderr}"


def _find_bcir_opt():
    env = os.environ.get("BCIR_OPT")
    if env and os.path.exists(env):
        return env
    root = os.path.join(os.path.dirname(__file__), "..", "..", "build", "mlir-build")
    for dirpath, _dirs, files in os.walk(os.path.normpath(root)) if os.path.isdir(root) else []:
        if "bcir-opt" in files:
            return os.path.join(dirpath, "bcir-opt")
    return None
