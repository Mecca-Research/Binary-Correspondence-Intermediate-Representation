"""G7 -- gem.attention as a planned claim: single-head scaled-dot-product attention, a COMPOSITION of the
existing ops (two matmuls + a scale + the EXISTING softmax) whose realization the dual-semiring cost search
plans (test_matmul.py / test_activation.py are the template).

Covers: the realization reproduces the naive attention reference (bit-exact on the matmul/scale, float
round-off through the softmax transcendental); attention DECOMPOSES into matmul + softmax + matmul (the
softmax is activation.softmax_reference, reused -- not reinvented); the quantized bridge tracks the float
reference within the input-quantization (R17) bound; the plan is the two underlying matmul plans + its cost
wires into the production 12-d algebra, PRICED THROUGH THE EXISTING matmul roofline; the shape/dtype
well-formedness check accepts valid claims + rejects malformed ones (the op-level validation shape, NOT a
new global R-law); the softmax quarantine is honored (the exp rides the c.call.libm: edge, i32 rejected);
and -- when a C toolchain is visible -- the emitted C kernel compiles and reproduces the reference. All
deterministic + pure-Python on the oracle rail."""

import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.kbcir import matmul
from bcir.kbcir.activation import softmax_reference
from bcir.kbcir.attention import (
    AttentionSpec,
    attention_realize,
    attention_reference,
    attention_via_bridge,
    bottleneck,
    check_attention,
    cost_of,
    cost_vector,
    plan_attention,
    scores_reference,
)
from bcir.kbcir.cost import ACCURACY, COMPUTE, MEMORY, CostVector, TargetProfile
from bcir.toolchain import host_link_args

_HOST = TargetProfile.for_host()


def _mat(rng, n):
    return [rng.uniform(-1.5, 1.5) for _ in range(n)]


def _imat(rng, n, lo=-3, hi=3):
    return [float(rng.randint(lo, hi)) for _ in range(n)]  # integer-valued -> the matmuls are exact


# --- the reference: a naive attention computed independently ---------------------------------------


def _naive_attention(q, k, v, seq, d):
    """A from-scratch single-head attention (no BCIR helpers) -- the independent truth the module's
    reference is checked against."""
    scale = 1.0 / math.sqrt(d)
    out = [0.0] * (seq * d)
    for i in range(seq):
        logits = []
        for j in range(seq):
            s = sum(q[i * d + t] * k[j * d + t] for t in range(d)) * scale
            logits.append(s)
        m = max(logits)
        ex = [math.exp(s - m) for s in logits]
        z = sum(ex) or 1.0
        a = [e / z for e in ex]
        for t in range(d):
            out[i * d + t] = sum(a[j] * v[j * d + t] for j in range(seq))
    return out


def test_reference_matches_an_independent_naive_attention():
    rng = random.Random(0xA77)
    seq, d = 5, 4
    spec = AttentionSpec(seq, d)
    q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
    got = attention_reference(q, k, v, spec)
    want = _naive_attention(q, k, v, seq, d)
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want))


def test_attention_decomposes_into_matmul_softmax_matmul():
    # attention IS matmul(Q,K^T)/sqrt(d) -> the EXISTING softmax -> matmul(.,V). Reconstruct it from those
    # pieces and confirm the module's reference equals the hand-composed pipeline (the softmax is reused).
    rng = random.Random(13)
    seq, d = 4, 3
    spec = AttentionSpec(seq, d)
    q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
    s = scores_reference(q, k, spec)  # (Q@K^T)/sqrt(d)
    a = softmax_reference(s, axis_len=seq)  # the EXISTING activation softmax (reused)
    ctx = matmul.matmul_reference(a, v, seq, d, seq)  # A @ V
    assert attention_reference(q, k, v, spec) == ctx
    # each softmax row is a valid distribution (sums to 1).
    for r in range(0, seq * seq, seq):
        assert abs(sum(a[r : r + seq]) - 1.0) < 1e-12


def test_planned_realization_matches_the_reference():
    rng = random.Random(0xA15)
    seq, d = 6, 8
    spec = plan_attention(seq, d, target=_HOST)
    q, k, v = _imat(rng, seq * d), _imat(rng, seq * d), _imat(rng, seq * d)
    base = AttentionSpec(seq, d)
    # the matmuls are exact for integer inputs; the softmax is identical (same libm) -> bit-identical here.
    assert attention_realize(q, k, v, spec) == attention_reference(q, k, v, base)


def test_fuzz_realization_tracks_the_reference():
    rng = random.Random(0xA77F)
    for _ in range(40):
        seq = rng.randint(1, 6)
        d = rng.randint(1, 6)
        spec = plan_attention(seq, d, target=_HOST)
        q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
        got = attention_realize(q, k, v, spec)
        want = attention_reference(q, k, v, AttentionSpec(seq, d))
        assert all(abs(g - w) < 1e-9 for g, w in zip(got, want))


# --- the Q8 bridge tracks the reference within the input-quantization (R17) bound ------------------


def test_attention_bridge_tracks_the_reference():
    # the bridge applies attention to the round-tripped Q/K/V, so the only error is the INPUT quantization.
    # attention is a bounded, smooth map (softmax is 1-Lipschitz in its logits, the convex-combination
    # output is a contraction of V), so the output error stays on the order of the input round-trip error.
    rng = random.Random(0xA819)
    seq, d = 5, 4
    spec = AttentionSpec(seq, d)
    from bcir.kbcir.quantize import max_abs_error

    q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
    q_err = max(max_abs_error(q, d, 8), max_abs_error(k, d, 8), max_abs_error(v, d, 8))
    ref = attention_reference(q, k, v, spec)
    got = attention_via_bridge(q, k, v, spec, group_size=d, bits=8)
    # generous, shape-honest bound: the output is a convex combination of the (bridged) V rows plus the
    # softmax/score perturbation; bounded by a small multiple of the per-element round-trip error.
    assert all(abs(r - g) <= 8.0 * q_err + 1e-6 for r, g in zip(ref, got))


def test_attention_bridge_is_attention_of_the_roundtrip():
    # attention_via_bridge == attention of the dequantized (round-tripped) Q/K/V -- the bridge is the SOLE
    # error source, mirroring matmul.gemm_via_bridge / conv.conv_via_bridge.
    from bcir.kbcir.quantize import dequantize, quantize_per_group

    rng = random.Random(0xB1D6F)
    seq, d = 4, 4
    spec = AttentionSpec(seq, d)
    q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
    dq = dequantize(quantize_per_group(q, d, 8))
    dk = dequantize(quantize_per_group(k, d, 8))
    dv = dequantize(quantize_per_group(v, d, 8))
    assert attention_via_bridge(q, k, v, spec, d, 8) == attention_reference(dq, dk, dv, spec)


# --- the plan is deterministic + wires into the production 12-d algebra (priced via matmul.cost_of) -


def test_plan_is_deterministic_and_carries_both_matmul_plans():
    for seq, d in [(8, 8), (16, 4), (5, 7)]:
        a = plan_attention(seq, d, target=_HOST)
        b = plan_attention(seq, d, target=_HOST)
        assert a == b
        assert a.scores_tile is not None and a.context_tile is not None
        # the carried plans ARE the matmul.plan_matmul choices for the two gemm shapes.
        assert a.scores_tile == matmul.plan_matmul(seq, seq, d, _HOST)
        assert a.context_tile == matmul.plan_matmul(seq, d, seq, _HOST)


def test_cost_is_the_sum_of_the_two_priced_matmuls():
    # attention is a composition of two gem.matmuls -> its cost is the SUM of the two priced matmul costs
    # (no new roofline term -- the existing matmul.cost_of, reused for both).
    spec = plan_attention(8, 8, target=_HOST)
    compute, mem, _ = cost_of(spec, _HOST, spec.scores_tile, spec.context_tile)
    sm, sn, sk = spec.scores_dims
    cm, cn, ck = spec.context_dims
    st, ct = spec.scores_tile, spec.context_tile
    c1, m1, _ = matmul.cost_of(sm, sn, sk, st.tile_m, st.tile_n, st.tile_k, _HOST)
    c2, m2, _ = matmul.cost_of(cm, cn, ck, ct.tile_m, ct.tile_n, ct.tile_k, _HOST)
    assert (compute, mem) == (c1 + c2, m1 + m2)


def test_cost_vector_wires_into_the_production_12d_algebra():
    spec = plan_attention(8, 8, target=_HOST)
    cv = cost_vector(spec)
    assert isinstance(cv, CostVector)
    compute, mem, _ = cost_of(spec, _HOST, spec.scores_tile, spec.context_tile)
    assert cv.v[COMPUTE] == compute and cv.v[MEMORY] == mem
    assert bottleneck(cv) == max(cv.v[COMPUTE], cv.v[MEMORY])
    w = tuple(1 if i in (COMPUTE, MEMORY) else 0 for i in range(len(cv.v)))
    assert cv.couple((256,) * len(cv.v)).dot(w) == compute + mem


def test_cost_vector_carries_the_r17_bound_on_the_accuracy_axis_when_quantized():
    spec = plan_attention(8, 8, target=_HOST)
    assert cost_vector(spec, quant_bits=0).v[ACCURACY] == 0  # dense: no accuracy cost
    assert (
        cost_vector(spec, quant_bits=8).v[ACCURACY] == 1
    )  # quantized: the R17 bridge bound (1 ULP)


def test_r17_certifies_the_bridge_on_a_quantized_attention_call():
    from bcir.kbcir.precision import accuracy_bound, quantization_error_bound
    from bcir.model import Claim, Domain, Lane, Opcode, StrideClass

    def call(qbits):
        return Claim(
            id=7300,
            opcode=Opcode.MUL,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1,
            rd=(80, 81, 82),
            wr=(83,),
            op="gem.attention",
            domain=Domain.RAM,
            quantized_bits=qbits,
        )

    assert accuracy_bound(call(8)) == accuracy_bound(call(0)) + 1  # the +1 ULP bridge step
    assert accuracy_bound(call(0)) == 1  # dense call: the op's own 1 ULP
    assert quantization_error_bound() == 1


# --- shape/dtype well-formedness: op-level validation (NOT a new global R-law) ---------------------


def test_check_attention_accepts_well_formed_claims():
    s = AttentionSpec(8, 16)
    assert check_attention(s, (8, 16), (8, 16), (8, 16), "f32", (8, 16), "f32") == []
    assert check_attention(AttentionSpec(4, 4), (4, 4), (4, 4), (4, 4), "f32", (4, 4), "f32") == []


def test_check_attention_rejects_malformed_claims():
    s = AttentionSpec(8, 16)
    # (1) wrong Q/K/V shape.
    assert check_attention(s, (8, 15), (8, 16), (8, 16), "f32", (8, 16), "f32")
    assert check_attention(s, (8, 16), (7, 16), (8, 16), "f32", (8, 16), "f32")
    # (2) wrong output shape.
    assert check_attention(s, (8, 16), (8, 16), (8, 16), "f32", (8, 8), "f32")
    # (3) dtype not preserved.
    assert check_attention(s, (8, 16), (8, 16), (8, 16), "f32", (8, 16), "i32")
    # (4) the quarantine dtype rule: attention's softmax needs f32 (the libm edge returns float).
    errs = check_attention(AttentionSpec(4, 4, "i32"), (4, 4), (4, 4), (4, 4), "i32", (4, 4), "i32")
    assert any("needs f32" in e for e in errs)


# --- the emitted C kernel compiles + reproduces the reference (self-skip when toolchain hidden) ----


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _compile_run_attention(q, k, v, seq, d):
    from bcir.lower.c_kernel import emit_attention_c

    cc = _cc()
    kernel = emit_attention_c(seq, d, "att")
    main = (
        f"\n#include <stdio.h>\nint main(void){{\n"
        f"  float Q[{seq * d}] = {{{', '.join(f'{x:.6f}f' for x in q)}}};\n"
        f"  float K[{seq * d}] = {{{', '.join(f'{x:.6f}f' for x in k)}}};\n"
        f"  float V[{seq * d}] = {{{', '.join(f'{x:.6f}f' for x in v)}}};\n"
        f"  float O[{seq * d}];\n  att(Q, K, V, O);\n"
        f'  for (int i=0;i<{seq * d};++i) printf("%.7f ", O[i]);\n  return 0;\n}}\n'
    )
    with tempfile.TemporaryDirectory() as dd:
        src = os.path.join(dd, "att.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(dd, "att")
        # -lm: the trusted libm edge (the softmax expf), exactly as the activation/gemm wraps link.
        bld = subprocess.run(
            host_link_args([cc, "-std=c11", "-O2", src, "-lm", "-o", exe]),
            capture_output=True,
            text=True,
        )
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_attention_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xA77C)
    seq, d = 5, 4
    q, k, v = _mat(rng, seq * d), _mat(rng, seq * d), _mat(rng, seq * d)
    got = _compile_run_attention(q, k, v, seq, d)
    ref = attention_reference(q, k, v, AttentionSpec(seq, d))
    # the C kernel calls the SAME libm expf; float32 vs the oracle's float64 differ by round-off only.
    assert all(abs(g - r) < 1e-4 for g, r in zip(got, ref)), (got, ref)


def test_attention_c_emit_routes_softmax_through_the_libm_edge():
    from bcir.lower.c_kernel import emit_attention_c

    src = emit_attention_c(4, 4, "att")
    assert "math.h" in src and "expf" in src  # the softmax transcendental rides the libm edge
    assert "sqrtf" in src  # the 1/sqrt(d_k) scale
    for bad in [(0, 4), (4, 0)]:
        try:
            emit_attention_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass
