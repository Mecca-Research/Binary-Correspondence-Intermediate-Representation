"""G7 -- gem.conv as a planned claim: a 2-D single-group convolution, a structured matmul whose lowering
strategy (direct vs im2col+gemm) the dual-semiring cost search plans (test_matmul.py / test_activation.py
are the template).

Covers: the im2col+gemm realization reproduces the naive direct conv EXACTLY (the verification that makes
the lowering a free choice); the quantized bridge tracks the float reference within the input-quantization
(R17) bound; the plan is deterministic + its cost wires into the production 12-d algebra, PRICED THROUGH
THE EXISTING matmul roofline (a conv is a structured matmul); the shape/dtype well-formedness check accepts
valid claims + rejects malformed ones (the op-level validation shape, NOT a new global R-law); and -- when
a C toolchain is visible -- the emitted C kernel compiles and reproduces the reference (bit-exact for
integer/exact inputs). All deterministic + pure-Python on the oracle rail."""

import os
import random
import shutil
import subprocess
import tempfile

from bcir.kbcir import matmul
from bcir.kbcir.conv import (
    ConvSpec,
    bottleneck,
    check_conv,
    conv2d_im2col,
    conv2d_realize,
    conv2d_reference,
    conv_via_bridge,
    cost_of,
    cost_vector,
    im2col,
    plan_conv,
    weights_as_matrix,
)
from bcir.kbcir.cost import ACCURACY, COMPUTE, MEMORY, CostVector, TargetProfile
from bcir.lower.c_kernel import emit_conv2d_c

_HOST = TargetProfile.for_host()


def _ivec(rng, n, lo=-4, hi=4):
    return [float(rng.randint(lo, hi)) for _ in range(n)]  # integer-valued -> float sums are exact


def _spec(in_c, in_h, in_w, out_c, kh, kw, stride=1, pad=0, dtype="f32"):
    return ConvSpec(in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype)


# --- the verification: the im2col+gemm realization reproduces the naive direct conv ----------------


def test_im2col_gemm_equals_the_direct_reference_bit_exact():
    rng = random.Random(0xC04)
    for stride, pad in ((1, 0), (1, 1), (2, 0), (2, 1)):
        spec = _spec(2, 5, 6, 3, 3, 3, stride, pad)
        x = _ivec(rng, spec.in_count)
        w = _ivec(rng, spec.weight_count)
        ref = conv2d_reference(x, w, spec)
        # im2col reshaping + the matmul accumulate the IDENTICAL products -> bit-identical for integers.
        assert conv2d_im2col(x, w, spec) == ref, (stride, pad)


def test_im2col_matches_an_explicit_patches_at_weights_matmul():
    # the structured-matmul identity, spelled out: P @ Wm == the conv, with P/Wm the reshapings.
    rng = random.Random(7)
    spec = _spec(2, 4, 4, 2, 2, 2, stride=1, pad=0)
    x, w = _ivec(rng, spec.in_count), _ivec(rng, spec.weight_count)
    p = im2col(x, spec)
    wm = weights_as_matrix(w, spec)
    m, n, k = spec.gemm_dims
    pm = matmul.matmul_reference(p, wm, m, n, k)
    # scatter the pixel-major gemm output to the channel-major conv layout and compare.
    out = [0.0] * spec.out_count
    for row in range(m):
        for oc in range(n):
            out[oc * m + row] = pm[row * n + oc]
    assert out == conv2d_reference(x, w, spec)


def test_planned_realization_matches_the_reference():
    rng = random.Random(0xC1A)
    spec = plan_conv(3, 7, 7, 4, 3, 3, stride=1, pad=1, target=_HOST)
    x, w = _ivec(rng, spec.in_count), _ivec(rng, spec.weight_count)
    base = _spec(3, 7, 7, 4, 3, 3, 1, 1)  # the reference (untiled) spec
    assert conv2d_realize(x, w, spec) == conv2d_reference(x, w, base)


def test_fuzz_realization_invariance_over_random_shapes():
    rng = random.Random(0xC0FF)
    for _ in range(60):
        in_c = rng.randint(1, 3)
        out_c = rng.randint(1, 3)
        kh = rng.randint(1, 3)
        kw = rng.randint(1, 3)
        in_h = rng.randint(kh, kh + 4)
        in_w = rng.randint(kw, kw + 4)
        stride = rng.randint(1, 2)
        pad = rng.randint(0, 1)
        spec = plan_conv(in_c, in_h, in_w, out_c, kh, kw, stride, pad, target=_HOST)
        if spec.out_h < 1 or spec.out_w < 1:
            continue
        x, w = _ivec(rng, spec.in_count), _ivec(rng, spec.weight_count)
        ref = conv2d_reference(x, w, _spec(in_c, in_h, in_w, out_c, kh, kw, stride, pad))
        assert conv2d_realize(x, w, spec) == ref
        assert conv2d_im2col(x, w, spec, spec.tile) == ref


# --- the Q8 bridge tracks the reference within the input-quantization (R17) bound ------------------


def test_conv_bridge_tracks_the_reference_within_quant_error():
    rng = random.Random(0xC819)
    spec = _spec(2, 5, 5, 3, 3, 3, stride=1, pad=1)
    x = [rng.uniform(-2, 2) for _ in range(spec.in_count)]
    w = [rng.uniform(-2, 2) for _ in range(spec.weight_count)]
    ref = conv2d_reference(x, w, spec)
    got = conv_via_bridge(x, w, spec, group_size=spec.taps, bits=8)
    # each output is a sum of `taps` products of bridged inputs -> error grows like the dot's term count
    # (the same shape as test_matmul's quant bound), with a tiny float slack.
    assert all(abs(r - g) <= 0.5 * spec.taps + 1e-6 for r, g in zip(ref, got))


def test_conv_bridge_is_the_conv_of_the_roundtrip():
    # conv_via_bridge == direct conv of the dequantized (round-tripped) inputs -- the bridge is the SOLE
    # error source, mirroring matmul.gemm_via_bridge / activation_via_bridge.
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(0xB1D6E)
    spec = _spec(2, 4, 4, 2, 2, 2, stride=1, pad=0)
    x = [rng.uniform(-3, 3) for _ in range(spec.in_count)]
    w = [rng.uniform(-3, 3) for _ in range(spec.weight_count)]
    dx = dequantize(quantize_per_group(x, spec.taps, 8))
    dw = dequantize(quantize_per_group(w, spec.taps, 8))
    assert conv_via_bridge(x, w, spec, spec.taps, 8) == conv2d_reference(dx, dw, spec)


# --- the plan is deterministic + wires into the production 12-d algebra (priced via matmul.cost_of) -


def test_plan_is_deterministic():
    for args in [(3, 8, 8, 4, 3, 3, 1, 1), (1, 16, 16, 8, 5, 5, 2, 2), (2, 5, 5, 3, 3, 3, 1, 0)]:
        a = plan_conv(*args, target=_HOST)
        b = plan_conv(*args, target=_HOST)
        assert a == b
        assert a.strategy in ("direct", "im2col")


def test_plan_is_priced_through_the_existing_matmul_roofline():
    # a conv IS a structured matmul: its (compute, mem) must equal matmul.cost_of on the im2col gemm dims.
    spec = plan_conv(3, 8, 8, 8, 3, 3, stride=1, pad=1, target=_HOST)
    m, n, k = spec.gemm_dims
    compute, mem, _ = cost_of(spec, "im2col", _HOST, spec.tile)
    tile = spec.tile or matmul.plan_matmul(m, n, k, _HOST)
    mc, mm, _ = matmul.cost_of(m, n, k, tile.tile_m, tile.tile_n, tile.tile_k, _HOST)
    assert (compute, mem) == (mc, mm)  # no new roofline term -- the matmul's, reused
    # the direct strategy is the UNTILED gemm cost (same compute -- the conv's MACs are fixed).
    dc, dm, _ = cost_of(spec, "direct", _HOST)
    uc, um, _ = matmul.cost_of(m, n, k, m, n, k, _HOST)
    assert (dc, dm) == (uc, um) and dc == compute


def test_cost_vector_wires_into_the_production_12d_algebra():
    spec = plan_conv(3, 8, 8, 8, 3, 3, stride=1, pad=1, target=_HOST)
    cv = cost_vector(spec)
    assert isinstance(cv, CostVector)
    compute, mem, _ = cost_of(spec, spec.strategy, _HOST, spec.tile)
    assert cv.v[COMPUTE] == compute and cv.v[MEMORY] == mem
    assert bottleneck(cv) == max(cv.v[COMPUTE], cv.v[MEMORY])
    # composes with the existing cost algebra: couple by an identity factor + scalarize by weights.
    w = tuple(1 if i in (COMPUTE, MEMORY) else 0 for i in range(len(cv.v)))
    assert cv.couple((256,) * len(cv.v)).dot(w) == compute + mem


def test_cost_vector_carries_the_r17_bound_on_the_accuracy_axis_when_quantized():
    spec = plan_conv(2, 6, 6, 4, 3, 3, target=_HOST)
    assert cost_vector(spec, quant_bits=0).v[ACCURACY] == 0  # dense: no accuracy cost
    assert (
        cost_vector(spec, quant_bits=8).v[ACCURACY] == 1
    )  # quantized: the R17 bridge bound (1 ULP)


def test_r17_certifies_the_bridge_on_a_quantized_conv_call():
    # a conv claim that consumes quantized lanes carries the bridge's round-trip step in its R17 bound;
    # a dense call does not. Mirrors test_blas_gemm's R17 boundary check exactly.
    from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
    from bcir.model import Claim, Domain, Lane, Opcode, StrideClass

    def call(qbits):
        return Claim(
            id=7200,
            opcode=Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1,
            rd=(70, 71),
            wr=(72,),
            op="gem.conv",
            domain=Domain.RAM,
            quantized_bits=qbits,
        )

    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1  # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1  # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1


# --- shape/dtype well-formedness: op-level validation (NOT a new global R-law) ---------------------


def test_check_conv_accepts_well_formed_claims():
    s = _spec(3, 8, 8, 4, 3, 3, 1, 1)
    assert check_conv(s, (3, 8, 8), "f32", (4, 3, 3, 3), (4, s.out_h, s.out_w), "f32") == []
    si = _spec(1, 5, 5, 2, 3, 3, 2, 0, "i32")  # integer conv (exact rail) is well-formed
    assert check_conv(si, (1, 5, 5), "i32", (2, 1, 3, 3), (2, si.out_h, si.out_w), "i32") == []


def test_check_conv_rejects_malformed_claims():
    s = _spec(3, 8, 8, 4, 3, 3, 1, 1)
    # (1) wrong input shape.
    assert check_conv(s, (3, 8, 9), "f32", (4, 3, 3, 3), (4, s.out_h, s.out_w), "f32")
    # (2) wrong weight shape.
    assert check_conv(s, (3, 8, 8), "f32", (4, 3, 3, 5), (4, s.out_h, s.out_w), "f32")
    # (3) dtype not preserved.
    assert check_conv(s, (3, 8, 8), "i32", (4, 3, 3, 3), (4, s.out_h, s.out_w), "f32")
    # (4) wrong output shape.
    assert check_conv(s, (3, 8, 8), "f32", (4, 3, 3, 3), (4, 99, 99), "f32")
    # (5) a kernel larger than the padded input.
    big = _spec(1, 3, 3, 1, 5, 5, 1, 0)
    assert any(
        "does not fit" in e or "empty" in e
        for e in check_conv(big, (1, 3, 3), "f32", (1, 1, 5, 5), (1, big.out_h, big.out_w), "f32")
    )
    # (6) bad hyperparameters: stride 0 / negative pad.
    assert check_conv(
        _spec(1, 4, 4, 1, 2, 2, 0, 0), (1, 4, 4), "f32", (1, 1, 2, 2), (1, 3, 3), "f32"
    )


# --- the emitted C kernel compiles + reproduces the reference (self-skip when toolchain hidden) ----


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def test_conv_c_kernel_is_bit_exact():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xC0DE)
    spec = _spec(2, 6, 7, 3, 3, 3, stride=2, pad=1)
    x = [float(rng.randint(-9, 9)) for _ in range(spec.in_count)]  # integer-valued -> bit-exact
    w = [float(rng.randint(-9, 9)) for _ in range(spec.weight_count)]
    ref = conv2d_reference(x, w, spec)
    kernel = emit_conv2d_c(2, 6, 7, 3, 3, 3, 2, 1, "cv")
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float in[{spec.in_count}] = {{{', '.join(f'{v:.1f}f' for v in x)}}};\n"
        f"  float W[{spec.weight_count}] = {{{', '.join(f'{v:.1f}f' for v in w)}}};\n"
        f"  float out[{spec.out_count}];\n  cv(in, W, out);\n"
        f'  for (int i=0;i<{spec.out_count};++i) printf("%.1f ", out[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "cv.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(d, "cv")
        bld = subprocess.run(
            [cc, "-std=c11", "-O2", src, "-o", exe], capture_output=True, text=True
        )
        assert bld.returncode == 0, bld.stderr  # no transcendental -> no -lm needed
        out = subprocess.run([exe], capture_output=True, text=True)
        got = [float(t) for t in out.stdout.split()]
        assert got == [round(r, 1) for r in ref], (got, ref)


def test_conv_c_emit_validates_dims():
    for bad in [(0, 4, 4, 1, 2, 2), (1, 0, 4, 1, 2, 2), (1, 4, 4, 1, 5, 5)]:
        try:
            emit_conv2d_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass
    src = emit_conv2d_c(2, 5, 5, 3, 3, 3, 1, 1, "cv")
    assert "math.h" not in src and "expf" not in src  # the conv is a pure MAC -- no libm
