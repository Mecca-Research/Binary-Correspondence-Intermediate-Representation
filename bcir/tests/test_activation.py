"""G1 -- gem.activation as a planned claim family (relu/sigmoid/tanh/gelu/softmax): an elementwise tensor
op the dual-semiring cost search plans, modeled exactly like gem.matmul (test_matmul.py is the template).

Covers: each activation's realization reproduces its reference (EXACT/0-ULP for relu; the trusted
transcendentals match their math reference to float round-off); the Q8 bridge tracks the reference within
the input-quantization (R17) bound; the plan is deterministic + its cost wires into the production 12-d
algebra (needing BOTH semirings -- min,+ to pick the width, max,+ for the roofline bottleneck); the
shape/dtype well-formedness check passes valid claims + rejects malformed ones (the gem.matmul op-level
validation shape, not a new global R-law); the quarantine split is honored (relu exact, the rest route a
transcendental to the c.call.libm: edge); and -- when a C toolchain is visible -- the emitted C kernels
compile and reproduce the reference (relu bit-exact for integer-valued inputs; the libm transcendentals
within a tight float tolerance). All deterministic + pure-Python on the oracle rail."""

import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.kbcir.activation import (ACTIVATIONS, EXACT_ACTIVATIONS, TRANSCENDENTAL_ACTIVATIONS,
                                   ActivationSpec, activation_reference, activation_via_bridge,
                                   bottleneck, check_activation, cost_of, cost_vector, is_exact,
                                   libm_edges, plan_activation, relu_reference, sigmoid_reference,
                                   softmax_reference, tanh_reference, gelu_reference)
from bcir.kbcir.cost import ACCURACY, COMPUTE, MEMORY, CostVector, TargetProfile
from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
from bcir.lower.c_kernel import emit_activation_kernel_c, emit_relu_kernel_c
from bcir.model import Claim, Domain, Lane, Opcode, StrideClass

_HOST = TargetProfile.for_host()


# --- the reference: every activation computes its definition --------------------------------------

def test_relu_is_exact_max_zero():
    # relu is the integer/Q-fixed-clean activation: max(0,x), 0 ULP, no transcendental.
    x = [-3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 100.0]
    assert relu_reference(x) == [0.0, 0.0, 0.0, 0.0, 0.5, 2.0, 100.0]
    # exact even for integer-valued inputs (the clean-rail property): bit-identical, no rounding.
    rng = random.Random(0x6E10)
    xi = [float(rng.randint(-50, 50)) for _ in range(64)]
    assert relu_reference(xi) == [v if v > 0 else 0.0 for v in xi]


def test_transcendental_references_match_their_definitions():
    rng = random.Random(0xAC71)
    x = [rng.uniform(-4, 4) for _ in range(50)]
    assert all(abs(s - 1.0 / (1.0 + math.exp(-v))) < 1e-12 for s, v in zip(sigmoid_reference(x), x))
    assert all(abs(t - math.tanh(v)) < 1e-12 for t, v in zip(tanh_reference(x), x))
    c = math.sqrt(2.0 / math.pi)
    want_gelu = [0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v**3))) for v in x]
    assert all(abs(g - w) < 1e-12 for g, w in zip(gelu_reference(x), want_gelu))


def test_softmax_is_stable_reduce_max_exp_normalize():
    # softmax over the last axis: each row sums to 1, is shift-invariant (the reduce-max stability), and
    # is monotone in the input.
    rows = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    out = softmax_reference(rows, axis_len=3)
    for r in (0, 3):
        assert abs(sum(out[r:r + 3]) - 1.0) < 1e-12
    # shift invariance: adding a constant to a whole row leaves softmax unchanged (numerically stable).
    shifted = softmax_reference([v + 1000.0 for v in rows], axis_len=3)
    assert all(abs(a - b) < 1e-9 for a, b in zip(out, shifted))
    # a big-magnitude row does not overflow (the reduce-max shift): finite, sums to 1.
    big = softmax_reference([700.0, 800.0, 900.0], axis_len=3)
    assert all(math.isfinite(v) for v in big) and abs(sum(big) - 1.0) < 1e-12


# --- the Q8 bridge tracks the reference within the input-quantization (R17) bound -----------------

def test_activation_bridge_tracks_the_reference_within_quant_error():
    # The bridge applies the activation to the round-tripped (dequantized) inputs, so the only error vs
    # the true activation is the INPUT quantization. relu is 1-Lipschitz, sigmoid/tanh have slope <= 1,
    # and gelu's slope is bounded ~1.13, so the activation-output error is bounded by ~1.2 * the input
    # round-trip error (the R17-certified quant step) -- never amplified beyond the bridge bound.
    rng = random.Random(0x6109)
    for kind in ("relu", "sigmoid", "tanh", "gelu"):
        x = [rng.uniform(-3, 3) for _ in range(32)]
        from bcir.kbcir.quantize import max_abs_error
        q_err = max_abs_error(x, group_size=8, bits=8)        # the R17 input round-trip bound (real units)
        ref = activation_reference(kind, x)
        got = activation_via_bridge(kind, x, group_size=8, bits=8)
        # output error <= Lipschitz(<= ~1.2) * input error, with a tiny float slack.
        assert all(abs(r - g) <= 1.2 * q_err + 1e-6 for r, g in zip(ref, got)), kind


def test_relu_bridge_is_exactly_the_activation_of_the_roundtrip():
    # relu of the bridged inputs == relu of the dequantized inputs EXACTLY (relu adds 0 ULP -- the clean
    # rail), so the bridge is the sole error source, mirroring matmul.gemm_via_bridge.
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(13)
    x = [rng.uniform(-5, 5) for _ in range(40)]
    dq = dequantize(quantize_per_group(x, 8, 8))
    assert activation_via_bridge("relu", x, 8, 8) == relu_reference(dq)


# --- the plan is deterministic + wires into the production 12-d cost algebra -----------------------

def test_plan_is_deterministic():
    for kind in ACTIVATIONS:
        ax = 4 if kind == "softmax" else 0
        a = plan_activation(kind, (8, 4), "f32", ax, _HOST)
        b = plan_activation(kind, (8, 4), "f32", ax, _HOST)
        assert a == b
        assert 1 <= a.width and a.count == 32


def test_plan_selects_the_min_bottleneck_width():
    # The chosen width minimizes the roofline bottleneck = max(compute, mem). Show the planned width is
    # no worse than the scalar (width-1) realization (the max,+ step the planner runs).
    sp = plan_activation("tanh", (64,), "f32", 0, _HOST)
    pc, pm = cost_of(sp, sp.width, _HOST)
    sc, smv = cost_of(sp, 1, _HOST)
    assert max(pc, pm) <= max(sc, smv)


def test_cost_vector_wires_into_the_production_12d_algebra():
    sp = plan_activation("gelu", (4, 8), "f32", 0, _HOST)
    cv = cost_vector(sp)
    assert isinstance(cv, CostVector)
    c, m = cost_of(sp, sp.width, _HOST)
    assert cv.v[COMPUTE] == c and cv.v[MEMORY] == m
    # the max,+ roofline bottleneck is recoverable from the vector.
    assert bottleneck(cv) == max(cv.v[COMPUTE], cv.v[MEMORY])
    # it composes with the existing cost algebra: couple by an identity factor + scalarize by weights.
    w = tuple(1 if i in (COMPUTE, MEMORY) else 0 for i in range(len(cv.v)))
    assert cv.couple((256,) * len(cv.v)).dot(w) == c + m


def test_cost_sees_transcendentals_as_heavier_than_relu():
    # The cost model must SEE that gelu (cube + tanh) costs more compute per element than relu (a max),
    # so the planner ranks them honestly -- the activation analog of matmul's compute term.
    shape = (256,)
    relu_c, _ = cost_of(ActivationSpec("relu", shape), 1, _HOST)
    gelu_c, _ = cost_of(ActivationSpec("gelu", shape), 1, _HOST)
    assert gelu_c > relu_c


def test_cost_vector_carries_the_r17_bound_on_the_accuracy_axis_when_quantized():
    sp = plan_activation("sigmoid", (16,), "f32", 0, _HOST)
    assert cost_vector(sp, quant_bits=0).v[ACCURACY] == 0     # dense: no accuracy cost
    assert cost_vector(sp, quant_bits=8).v[ACCURACY] == 1     # quantized: the R17 bridge bound (1 ULP)


# --- the R17 accuracy bound on a quantized activation claim (the certification path) --------------

def test_r17_certifies_the_bridge_on_a_quantized_activation_call():
    # An activation claim that consumes quantized lanes carries the bridge's round-trip step in its R17
    # bound (the trusted libm transcendental / the exact relu itself add no certified error beyond the op
    # boundary); a dense call does not. Mirrors test_blas_gemm's R17 boundary check exactly.
    def call(qbits):
        return Claim(id=7100, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT, count=1,
                     rd=(60,), wr=(61,), op="gem.activation", domain=Domain.RAM, quantized_bits=qbits)
    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1   # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1                            # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1                         # the R17 grid bound the bridge is held to


# --- the quarantine split: relu exact, the rest route a transcendental to the libm edge -----------

def test_quarantine_split_is_correct():
    assert set(EXACT_ACTIVATIONS) == {"relu"}
    assert set(TRANSCENDENTAL_ACTIVATIONS) == {"sigmoid", "tanh", "gelu", "softmax"}
    assert is_exact("relu") and not any(is_exact(k) for k in TRANSCENDENTAL_ACTIVATIONS)
    assert libm_edges("relu") == ()                              # exact: no transcendental edge
    assert libm_edges("sigmoid") == ("expf",) and libm_edges("softmax") == ("expf",)
    assert libm_edges("tanh") == ("tanhf",) and libm_edges("gelu") == ("tanhf",)


# --- shape/dtype well-formedness: gem.matmul-style op-level validation (NOT a new global R-law) ----

def test_check_activation_accepts_well_formed_claims():
    assert check_activation(ActivationSpec("relu", (8,), "f32", 0), (8,), "f32", (8,), "f32") == []
    assert check_activation(ActivationSpec("relu", (4,), "i32", 0), (4,), "i32", (4,), "i32") == []
    assert check_activation(ActivationSpec("gelu", (2, 3), "f32", 0), (2, 3), "f32", (2, 3), "f32") == []
    assert check_activation(ActivationSpec("softmax", (2, 5), "f32", 5), (2, 5), "f32", (2, 5), "f32") == []


def test_check_activation_rejects_malformed_claims():
    # (1) not shape-preserving.
    assert check_activation(ActivationSpec("relu", (8,), "f32", 0), (8,), "f32", (4,), "f32")
    # (2) dtype not preserved.
    assert check_activation(ActivationSpec("relu", (8,), "f32", 0), (8,), "f32", (8,), "i32")
    # (3) spec shape disagrees with the input.
    assert check_activation(ActivationSpec("relu", (9,), "f32", 0), (8,), "f32", (8,), "f32")
    # (4) a transcendental on i32 (the quarantine dtype rule -- the libm edge returns float).
    errs = check_activation(ActivationSpec("sigmoid", (8,), "i32", 0), (8,), "i32", (8,), "i32")
    assert any("needs f32" in e for e in errs)
    # (5) softmax axis mismatch + a missing/zero elementwise axis.
    assert check_activation(ActivationSpec("softmax", (2, 5), "f32", 4), (2, 5), "f32", (2, 5), "f32")
    assert check_activation(ActivationSpec("relu", (8,), "f32", 3), (8,), "f32", (8,), "f32")
    # an unknown activation kind is rejected.
    assert check_activation(ActivationSpec("selu", (8,), "f32", 0), (8,), "f32", (8,), "f32")


# --- the emitted C kernels compile + reproduce the reference (self-skip when toolchain is hidden) --

def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _compile_run_activation(kind, x, axis_len=None):
    """Compile the emitted activation kernel + a tiny main that prints C[i], run it, return the floats."""
    cc = _cc()
    n = len(x)
    fn = "act"
    kernel = emit_activation_kernel_c(kind, n, fn, axis_len=axis_len)
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float A[{n}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float C[{n}];\n  {fn}(A, C, {n});\n"
            f'  for (int i=0;i<{n};++i) printf("%.7f ", C[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "a.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "a")
        # -lm: the trusted libm edge (expf/tanhf), exactly as the B5 wrap links cblas.
        bld = subprocess.run([cc, "-std=c11", "-O2", src, "-lm", "-o", exe],
                             capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_relu_c_kernel_is_bit_exact():
    cc = _cc()
    if not cc:
        return                                                  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x6E1C)
    x = [float(rng.randint(-9, 9)) for _ in range(23)]          # integer-valued -> relu is bit-exact
    got = _compile_run_activation("relu", x)
    assert got == relu_reference(x)                             # 0 ULP, the clean-rail guarantee


def test_transcendental_c_kernels_match_the_reference():
    cc = _cc()
    if not cc:
        return                                                  # self-skip without a compiler
    rng = random.Random(0x713C)
    x = [rng.uniform(-4, 4) for _ in range(24)]
    for kind in ("sigmoid", "tanh", "gelu"):
        got = _compile_run_activation(kind, x)
        ref = activation_reference(kind, x)
        # the C kernel calls the SAME libm; float32 vs the oracle's float64 differ by round-off only.
        assert all(abs(g - r) < 1e-4 for g, r in zip(got, ref)), (kind, got, ref)


def test_softmax_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x50F7)
    x = [rng.uniform(-3, 3) for _ in range(12)]
    got = _compile_run_activation("softmax", x, axis_len=4)
    ref = softmax_reference(x, axis_len=4)
    assert all(abs(g - r) < 1e-5 for g, r in zip(got, ref)), (got, ref)
    for r in (0, 4, 8):
        assert abs(sum(got[r:r + 4]) - 1.0) < 1e-4              # each row still normalizes


def test_relu_kernel_supports_i32_clean_rail():
    # relu emits an integer kernel too (the Q-fixed clean rail) -- no libm, no float.
    src = emit_relu_kernel_c(8, "r", elem="i32")
    assert "int32_t" in src and "math.h" not in src and "expf" not in src
    src_f = emit_activation_kernel_c("gelu", 8)
    assert "math.h" in src_f and "tanhf" in src_f               # the transcendental DOES pull in libm
