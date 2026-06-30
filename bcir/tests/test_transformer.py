"""E3 (ML-breadth) -- the full TRANSFORMER ENCODER BLOCK: an ORACLE-SIDE COMPOSITION of the ops BCIR already
has (matmul + softmax + a new layernorm), the THIRD ML-breadth slice after E1 (OLS) and E2 (PCA). Unlike E1/E2
(which each WRAPPED one external LAPACK kernel), a transformer block is a COMPOSITION, so this mirrors the G7
single-head attention (test_attention.py) -- it REUSES attention.scores_reference + activation.softmax_reference
+ matmul.matmul_reference and adds the ONE new numeric primitive, layernorm.

Covers: layernorm normalizes per row (a gamma=1/beta=0 row has mean~0, var~1 by the INDEPENDENT layernorm_stats
verifier) and matches a from-scratch population-variance reference; the causal mask zeros the strictly-upper
(future) triangle of the post-softmax weights EXACTLY (the genuine independent check) and every softmax row sums
to 1; multi-head == concat-of-INDEPENDENTLY-computed-heads then W_o (the independent multihead_concat_residual);
the feed-forward is act(x@W1+b1)@W2+b2; the POST-LN block == a naive from-scratch composition to float round-off,
the PRE-LN variant matches its own composition, and the zero-weight block reduces to LN(LN(x)); the Q8 bridge
tracks the reference within the R17 input bound AND equals the block of the round-tripped tensors (clean bridge);
the shape/dtype check accepts valid claims + rejects malformed ones (NOT a new global R-law); the quarantine
dtype rule is honored (the softmax+layernorm transcendentals need f32, i32 rejected); the layernorm C kernel
compiles + reproduces the reference + routes the rsqrt through the libm sqrtf edge; the sqrtf->-lm link rule is
unchanged (NO linkflags change); and the module imports no verifier / emits no Diagnostic (the two-truth
quarantine, AST-inspected). All deterministic + pure-Python on the oracle rail.
"""

import ast
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir import attention, matmul
from bcir.kbcir.activation import softmax_reference
from bcir.kbcir.precision import quantization_error_bound
from bcir.kbcir.transformer import (TransformerBlockParams, TransformerBlockSpec,
                                    causal_mask, causal_weights_upper_triangle_max,
                                    check_transformer, feedforward_reference,
                                    layernorm_reference, layernorm_stats,
                                    multihead_attention_reference, multihead_concat_residual,
                                    softmax_rowsum_residual, transformer_block_reference,
                                    transformer_block_via_bridge)
from bcir.lower.c_kernel import emit_layernorm_c


def _vec(rng, n, lo=-0.5, hi=0.5):
    return [rng.uniform(lo, hi) for _ in range(n)]


def _params(rng, d_model, d_ff, *, identity_ln=True, zeros=False):
    """A full block param set. ``identity_ln`` uses gamma=1/beta=0 (pure normalization); ``zeros`` makes every
    projection/feed-forward weight 0 (so the block reduces to LN(LN(x)))."""
    def w(n):
        return [0.0] * n if zeros else _vec(rng, n)
    g = (lambda: [1.0] * d_model) if identity_ln else (lambda: _vec(rng, d_model, 0.5, 1.5))
    be = (lambda: [0.0] * d_model) if identity_ln else (lambda: _vec(rng, d_model, -0.3, 0.3))
    return TransformerBlockParams(
        w_q=w(d_model * d_model), w_k=w(d_model * d_model), w_v=w(d_model * d_model), w_o=w(d_model * d_model),
        W1=w(d_model * d_ff), b1=([0.0] * d_ff if zeros else _vec(rng, d_ff)),
        W2=w(d_ff * d_model), b2=([0.0] * d_model if zeros else _vec(rng, d_model)),
        gamma1=g(), beta1=be(), gamma2=g(), beta2=be())


# --- 1. LayerNorm: the one new numeric primitive (independent stats verifier + a from-scratch reference) ----

def _naive_layernorm(x, rows, dim, gamma, beta, eps):
    """A from-scratch layernorm (no BCIR helpers) -- the independent truth the reference is checked against."""
    out = [0.0] * (rows * dim)
    for r in range(rows):
        row = x[r * dim:(r + 1) * dim]
        mean = sum(row) / dim
        var = sum((v - mean) ** 2 for v in row) / dim          # population variance (/dim)
        inv = 1.0 / math.sqrt(var + eps)
        for c in range(dim):
            out[r * dim + c] = gamma[c] * (row[c] - mean) * inv + beta[c]
    return out


def test_layernorm_matches_an_independent_naive_reference():
    rng = random.Random(0xE301)
    rows, dim = 6, 5
    x = _vec(rng, rows * dim, -3, 3)
    gamma, beta = _vec(rng, dim, 0.5, 1.5), _vec(rng, dim, -0.5, 0.5)
    got = layernorm_reference(x, rows, dim, gamma, beta, eps=1e-5)
    want = _naive_layernorm(x, rows, dim, gamma, beta, 1e-5)
    assert all(abs(g - w) < 1e-12 for g, w in zip(got, want))


def test_layernorm_normalizes_rows_to_mean0_var1():
    # The independent layernorm_stats verifier: a gamma=1/beta=0 row has mean ~ 0 and var ~ 1 (the eps inside
    # the sqrt makes var slightly < 1 -- it is var/(var+eps), within ~eps of 1).
    rng = random.Random(0xE302)
    rows, dim = 8, 6
    x = _vec(rng, rows * dim, -4, 4)
    y = layernorm_reference(x, rows, dim, [1.0] * dim, [0.0] * dim, eps=1e-5)
    for mean, var in layernorm_stats(y, rows, dim):
        assert abs(mean) < 1e-9, mean                          # exactly centered
        assert abs(var - 1.0) < 1e-3, var                      # unit variance (up to the eps floor)


def test_layernorm_gamma_beta_affine_after_normalization():
    # With gamma/beta the row is (normalized * gamma + beta): the recomputed mean ~ mean(beta), and the
    # constant-gamma case scales the variance by gamma^2 -- a property of the affine, independent of the code.
    rng = random.Random(0xE303)
    rows, dim = 5, 4
    x = _vec(rng, rows * dim, -2, 2)
    gconst, bconst = 2.0, 0.5
    y = layernorm_reference(x, rows, dim, [gconst] * dim, [bconst] * dim, eps=1e-6)
    for mean, var in layernorm_stats(y, rows, dim):
        assert abs(mean - bconst) < 1e-6, mean                 # mean shifts to beta
        assert abs(var - gconst * gconst) < 1e-2, var          # variance scales by gamma^2 (var~1 -> gamma^2)


def test_layernorm_is_side_effect_free_and_validates():
    rng = random.Random(0xE304)
    rows, dim = 4, 3
    x = _vec(rng, rows * dim)
    x0 = list(x)
    layernorm_reference(x, rows, dim, [1.0] * dim, [0.0] * dim)
    assert x == x0                                             # input not mutated
    bad = [
        ((x, 0, dim, [1.0] * dim, [0.0] * dim), {}),           # rows < 1
        ((x, rows, 0, [], []), {}),                            # dim < 1
        ((x, rows, dim + 1, [1.0] * (dim + 1), [0.0] * (dim + 1)), {}),  # len(x) != rows*dim
        ((x, rows, dim, [1.0] * (dim + 1), [0.0] * dim), {}),  # bad gamma length
        ((x, rows, dim, [1.0] * dim, [0.0] * dim), {"eps": 0.0}),  # eps <= 0
    ]
    for args, kw in bad:
        try:
            layernorm_reference(*args, **kw)
            assert False, f"{args[1:3]} should raise"
        except ValueError:
            pass


# --- 2. Masked multi-head attention: causal mask zeros the future + multi-head == concat-of-heads ----------

def _spec_params(rng, d_model=8, n_heads=2, d_ff=16, seq=5, batch=1):
    spec = TransformerBlockSpec(d_model, n_heads, d_ff, seq, batch=batch)
    return spec, _params(rng, d_model, d_ff)


def test_causal_mask_zeros_the_upper_triangle_exactly():
    # The genuine INDEPENDENT check: the post-softmax attention weights for j > i are EXACTLY 0 -- the causal
    # mask truly prevents attending to the future (exp(-inf) == 0.0).
    rng = random.Random(0xE305)
    spec, params = _spec_params(rng)
    seq, d_k = spec.seq_len, spec.d_k
    mask = causal_mask(seq)
    from bcir.kbcir.transformer import _project, _split_head
    for h in range(spec.n_heads):
        qh = _split_head(_project([rng.uniform(-1, 1) for _ in range(seq * spec.d_model)], params.w_q, seq, spec.d_model), seq, h, spec.n_heads, d_k)
        kh = _split_head(_project([rng.uniform(-1, 1) for _ in range(seq * spec.d_model)], params.w_k, seq, spec.d_model), seq, h, spec.n_heads, d_k)
        assert causal_weights_upper_triangle_max(qh, kh, seq, d_k, mask) == 0.0


def test_softmax_rows_sum_to_one_masked_and_unmasked():
    rng = random.Random(0xE306)
    spec, _ = _spec_params(rng)
    seq, d_k = spec.seq_len, spec.d_k
    qh, kh = _vec(rng, seq * d_k, -1, 1), _vec(rng, seq * d_k, -1, 1)
    assert softmax_rowsum_residual(qh, kh, seq, d_k) < 1e-12                 # unmasked
    assert softmax_rowsum_residual(qh, kh, seq, d_k, causal_mask(seq)) < 1e-12   # causal-masked


def test_multihead_equals_concat_of_independent_heads():
    # multi-head attention == project -> split -> per-head single-head attention (recomputed independently) ->
    # concat -> W_o. The independent path never calls the module's _single_head, so it is a genuine check.
    rng = random.Random(0xE307)
    spec, params = _spec_params(rng, d_model=12, n_heads=3, seq=6)
    x = _vec(rng, spec.seq_len * spec.d_model, -1, 1)
    assert multihead_concat_residual(x, spec, params) < 1e-9
    assert multihead_concat_residual(x, spec, params, causal_mask(spec.seq_len)) < 1e-9


def test_multihead_reuses_the_existing_softmax_and_scores():
    # Confirm the multi-head path is built from the EXISTING attention.scores_reference + softmax_reference +
    # matmul.matmul_reference (the reuse constraint): reconstruct head 0 by hand from those pieces.
    rng = random.Random(0xE308)
    d_model, n_heads, seq = 8, 2, 4
    spec = TransformerBlockSpec(d_model, n_heads, 16, seq, batch=1)
    params = _params(rng, d_model, 16)
    x = _vec(rng, seq * d_model, -1, 1)
    from bcir.kbcir.transformer import _project, _split_head
    d_k = spec.d_k
    q0 = _split_head(_project(x, params.w_q, seq, d_model), seq, 0, n_heads, d_k)
    k0 = _split_head(_project(x, params.w_k, seq, d_model), seq, 0, n_heads, d_k)
    v0 = _split_head(_project(x, params.w_v, seq, d_model), seq, 0, n_heads, d_k)
    aspec = attention.AttentionSpec(seq, d_k)
    s = attention.scores_reference(q0, k0, aspec)             # the EXISTING scaled scores
    w = softmax_reference(s, axis_len=seq)                    # the EXISTING softmax
    ctx0 = matmul.matmul_reference(w, v0, seq, d_k, seq)      # A @ V via the EXISTING matmul
    # the module's multi-head head-0 block must equal this hand-composed context.
    full = multihead_attention_reference(x, TransformerBlockSpec(d_model, n_heads, 16, seq, 1),
                                         params.mha)
    # undo W_o is hard; instead re-run the concat path with W_o = identity to read the concatenated heads.
    ident = [1.0 if i == j else 0.0 for i in range(d_model) for j in range(d_model)]
    concat = multihead_attention_reference(
        x, TransformerBlockSpec(d_model, n_heads, 16, seq, 1),
        TransformerBlockParams(params.w_q, params.w_k, params.w_v, ident, params.W1, params.b1, params.W2,
                               params.b2, params.gamma1, params.beta1, params.gamma2, params.beta2).mha)
    for i in range(seq):
        for t in range(d_k):
            assert abs(concat[i * d_model + t] - ctx0[i * d_k + t]) < 1e-9
    assert len(full) == seq * d_model


def test_batch_is_an_independent_outer_loop():
    # A batch of B sequences == running the single-sequence path on each independently (the documented simple
    # outer loop). Build a batch=2 input and compare each sequence's block-output to the batch==1 result.
    rng = random.Random(0xE309)
    d_model, n_heads, d_ff, seq = 8, 2, 16, 4
    params = _params(rng, d_model, d_ff)
    x0 = _vec(rng, seq * d_model, -1, 1)
    x1 = _vec(rng, seq * d_model, -1, 1)
    batch_spec = TransformerBlockSpec(d_model, n_heads, d_ff, seq, batch=2)
    one_spec = TransformerBlockSpec(d_model, n_heads, d_ff, seq, batch=1)
    out = transformer_block_reference(x0 + x1, batch_spec, params)
    o0 = transformer_block_reference(x0, one_spec, params)
    o1 = transformer_block_reference(x1, one_spec, params)
    assert out[:seq * d_model] == o0 and out[seq * d_model:] == o1


# --- 3. Feed-forward: act(x@W1+b1)@W2+b2 ---------------------------------------------------------------------

def test_feedforward_matches_a_hand_composition():
    rng = random.Random(0xE30A)
    rows, d_model, d_ff = 4, 6, 10
    x = _vec(rng, rows * d_model, -1, 1)
    W1, b1 = _vec(rng, d_model * d_ff), _vec(rng, d_ff)
    W2, b2 = _vec(rng, d_ff * d_model), _vec(rng, d_model)
    got = feedforward_reference(x, rows, d_model, d_ff, W1, b1, W2, b2, "relu")
    # hand composition: h = relu(x@W1 + b1); out = h@W2 + b2.
    h = matmul.matmul_reference(x, W1, rows, d_ff, d_model)
    h = [max(0.0, h[i * d_ff + j] + b1[j]) for i in range(rows) for j in range(d_ff)]
    o = matmul.matmul_reference(h, W2, rows, d_model, d_ff)
    want = [o[i * d_model + j] + b2[j] for i in range(rows) for j in range(d_model)]
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want))
    # gelu is also accepted (rides the libm tanhf edge); just runs without error and is shape-correct.
    assert len(feedforward_reference(x, rows, d_model, d_ff, W1, b1, W2, b2, "gelu")) == rows * d_model


# --- 4. The block: POST-LN == naive composition; PRE-LN variant; zero-weight identity ----------------------

def _naive_block(x, spec, params, mask, activation, eps, pre_ln):
    """A from-scratch transformer block (using only the leaf references) -- the independent truth."""
    rows, d_model = spec.rows, spec.d_model

    def mha(z):
        return multihead_attention_reference(z, spec, params.mha, mask)

    def ff(z):
        return feedforward_reference(z, rows, d_model, spec.d_ff, params.W1, params.b1, params.W2, params.b2,
                                     activation)

    def ln(z, g, b):
        return _naive_layernorm(z, rows, d_model, g, b, eps)

    if pre_ln:
        x1 = [x[i] + mha(ln(x, params.gamma1, params.beta1))[i] for i in range(rows * d_model)]
        f = ff(ln(x1, params.gamma2, params.beta2))
        return [x1[i] + f[i] for i in range(rows * d_model)]
    a = mha(x)
    x1 = ln([x[i] + a[i] for i in range(rows * d_model)], params.gamma1, params.beta1)
    f = ff(x1)
    return ln([x1[i] + f[i] for i in range(rows * d_model)], params.gamma2, params.beta2)


def test_postln_block_matches_a_naive_composition():
    rng = random.Random(0xE30B)
    spec, params = _spec_params(rng, d_model=8, n_heads=2, d_ff=16, seq=5)
    params = _params(rng, 8, 16, identity_ln=False)          # non-trivial gamma/beta
    x = _vec(rng, spec.rows * spec.d_model, -1, 1)
    got = transformer_block_reference(x, spec, params, mask=causal_mask(spec.seq_len))
    want = _naive_block(x, spec, params, causal_mask(spec.seq_len), "relu", 1e-5, pre_ln=False)
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want))


def test_preln_variant_matches_its_naive_composition():
    rng = random.Random(0xE30C)
    spec = TransformerBlockSpec(8, 2, 16, 5, batch=1)
    params = _params(rng, 8, 16, identity_ln=False)
    x = _vec(rng, spec.rows * spec.d_model, -1, 1)
    got = transformer_block_reference(x, spec, params, pre_ln=True)
    want = _naive_block(x, spec, params, None, "relu", 1e-5, pre_ln=True)
    assert all(abs(g - w) < 1e-9 for g, w in zip(got, want))
    # pre-LN and post-LN differ (a real structural choice, not a relabeling).
    post = transformer_block_reference(x, spec, params, pre_ln=False)
    assert any(abs(p - q) > 1e-6 for p, q in zip(got, post))


def test_zero_weight_block_reduces_to_layernorm_of_layernorm():
    # With all sublayer weights 0 the attention output is 0 and the feed-forward output is 0, so POST-LN reduces
    # to out = LN(LN(x)) (the documented exact identity). gamma=1/beta=0 so the LNs are pure normalization.
    rng = random.Random(0xE30D)
    spec = TransformerBlockSpec(6, 2, 12, 4, batch=1)
    params = _params(rng, 6, 12, zeros=True)                 # zero sublayer weights, identity LN
    x = _vec(rng, spec.rows * spec.d_model, -2, 2)
    out = transformer_block_reference(x, spec, params)
    ln1 = layernorm_reference(x, spec.rows, spec.d_model, [1.0] * spec.d_model, [0.0] * spec.d_model)
    ln2 = layernorm_reference(ln1, spec.rows, spec.d_model, [1.0] * spec.d_model, [0.0] * spec.d_model)
    assert all(abs(o - l) < 1e-12 for o, l in zip(out, ln2))


def test_block_is_side_effect_free():
    rng = random.Random(0xE30E)
    spec = TransformerBlockSpec(8, 2, 16, 4, batch=1)
    params = _params(rng, 8, 16)
    x = _vec(rng, spec.rows * spec.d_model)
    x0 = list(x)
    transformer_block_reference(x, spec, params)
    assert x == x0


# --- 5. The Q8 bridge: tracks the reference within the R17 input bound + is a clean round-trip -------------

def test_block_bridge_tracks_the_reference_within_quant_error():
    # The bridge runs the block on the round-tripped input, so the error is on the order of the input round-trip
    # (a transformer block is a bounded, smooth map: layernorm rescales, softmax is 1-Lipschitz, residuals add).
    rng = random.Random(0xE30F)
    spec = TransformerBlockSpec(8, 2, 16, 4, batch=1)
    params = _params(rng, 8, 16)
    from bcir.kbcir.quantize import max_abs_error
    x = _vec(rng, spec.rows * spec.d_model, -1, 1)
    q_err = max_abs_error(x, group_size=spec.d_model, bits=8)
    ref = transformer_block_reference(x, spec, params)
    got = transformer_block_via_bridge(x, spec, params, group_size=spec.d_model, bits=8)
    # generous shape-honest bound: the final layernorm normalizes the perturbation, but the within-block matmuls
    # amplify it -- a conservative multiple of the per-element round-trip error covers it.
    assert all(abs(r - g) <= 200.0 * q_err + 1e-2 for r, g in zip(ref, got)), (q_err,)


def test_block_bridge_is_block_of_the_roundtrip():
    # transformer_block_via_bridge == the block of the dequantized (round-tripped) input -- the bridge is the
    # SOLE error source, mirroring attention_via_bridge / pca_via_bridge / ols_via_bridge.
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(0xE310)
    spec = TransformerBlockSpec(8, 2, 16, 4, batch=1)
    params = _params(rng, 8, 16)
    x = _vec(rng, spec.rows * spec.d_model, -1, 1)
    dx = dequantize(quantize_per_group(x, spec.d_model, 8))
    assert transformer_block_via_bridge(x, spec, params, spec.d_model, 8) == \
        transformer_block_reference(dx, spec, params)


def test_block_bridge_can_quantize_weights_too():
    # With quantize_weights=True the projection/feed-forward weights are round-tripped too -- the realistic
    # weights-also-quantized inference path. It equals the block of round-tripped input AND weights.
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(0xE311)
    spec = TransformerBlockSpec(8, 2, 16, 4, batch=1)
    params = _params(rng, 8, 16)
    x = _vec(rng, spec.rows * spec.d_model, -1, 1)

    def rt(v):
        return dequantize(quantize_per_group(v, 8, 8))
    pq = TransformerBlockParams(rt(params.w_q), rt(params.w_k), rt(params.w_v), rt(params.w_o),
                                rt(params.W1), list(params.b1), rt(params.W2), list(params.b2),
                                list(params.gamma1), list(params.beta1), list(params.gamma2), list(params.beta2))
    got = transformer_block_via_bridge(x, spec, params, 8, 8, quantize_weights=True)
    want = transformer_block_reference(rt(x), spec, pq)
    assert got == want


def test_r17_bound_is_the_input_roundtrip():
    # The certified boundary error of a quantized block is the bridge's round-trip step (the matmuls are exact,
    # softmax/sqrt are the trusted libm edge -- no certified error). The R17 grid bound is 1 ULP, as every wrap.
    assert quantization_error_bound() == 1


# --- 6. Shape/dtype well-formedness (op-level validation, NOT a new global R-law) ---------------------------

def test_check_transformer_accepts_well_formed_claims():
    spec = TransformerBlockSpec(8, 2, 16, 5, batch=1)
    assert check_transformer(spec, (5, 8), "f32", (5, 8), "f32") == []
    spec2 = TransformerBlockSpec(12, 3, 24, 4, batch=2)
    assert check_transformer(spec2, (8, 12), "f32", (8, 12), "f32") == []   # batch*seq = 8 rows


def test_check_transformer_rejects_malformed_claims():
    spec = TransformerBlockSpec(8, 2, 16, 5, batch=1)
    # (1) wrong x shape.
    assert check_transformer(spec, (5, 7), "f32", (5, 8), "f32")
    # (2) wrong out shape.
    assert check_transformer(spec, (5, 8), "f32", (4, 8), "f32")
    # (3) dtype not preserved.
    assert check_transformer(spec, (5, 8), "f32", (5, 8), "i32")
    # (4) d_model not divisible by n_heads.
    bad = TransformerBlockSpec(8, 3, 16, 5, batch=1)
    assert any("divisible" in e for e in check_transformer(bad, (5, 8), "f32", (5, 8), "f32"))
    # (5) the quarantine dtype rule: the softmax + layernorm need f32 (the libm edges return float).
    i32 = TransformerBlockSpec(8, 2, 16, 5, batch=1, dtype="i32")
    assert any("needs f32" in e for e in check_transformer(i32, (5, 8), "i32", (5, 8), "i32"))


# --- 7. The emitted layernorm C kernel compiles + reproduces the reference (self-skip when no toolchain) ----

def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _compile_run_layernorm(x, rows, dim, gamma, beta, eps):
    cc = _cc()
    kernel = emit_layernorm_c(rows, dim, "ln")
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float X[{rows * dim}] = {{{', '.join(f'{v:.6f}f' for v in x)}}};\n"
            f"  float G[{dim}] = {{{', '.join(f'{v:.6f}f' for v in gamma)}}};\n"
            f"  float B[{dim}] = {{{', '.join(f'{v:.6f}f' for v in beta)}}};\n"
            f"  float O[{rows * dim}];\n  ln(X, G, B, {eps:.8f}f, O);\n"
            f'  for (int i=0;i<{rows * dim};++i) printf("%.7f ", O[i]);\n  return 0;\n}}\n')
    with tempfile.TemporaryDirectory() as dd:
        src = os.path.join(dd, "ln.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(dd, "ln")
        # -lm: the trusted libm edge (the layernorm rsqrt sqrtf), exactly as the activation/attention wraps link.
        bld = subprocess.run([cc, "-std=c11", "-O2", src, "-lm", "-o", exe], capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_layernorm_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return                                                # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xE312)
    rows, dim = 4, 5
    x = _vec(rng, rows * dim, -3, 3)
    gamma, beta = _vec(rng, dim, 0.5, 1.5), _vec(rng, dim, -0.5, 0.5)
    got = _compile_run_layernorm(x, rows, dim, gamma, beta, 1e-5)
    ref = layernorm_reference(x, rows, dim, gamma, beta, 1e-5)
    # the C kernel calls the SAME libm sqrtf; float32 vs the oracle's float64 differ by round-off only.
    assert all(abs(g - r) < 1e-3 for g, r in zip(got, ref)), (got, ref)
    # the genuine independent check on the C output: a gamma=1/beta=0 row would be mean~0/var~1, but with the
    # affine here we verify the stats reproduce the reference's stats.
    cstats = layernorm_stats(got, rows, dim)
    rstats = layernorm_stats(ref, rows, dim)
    assert all(abs(cm - rm) < 1e-3 and abs(cv - rv) < 1e-3
               for (cm, cv), (rm, rv) in zip(cstats, rstats))


def test_layernorm_c_normalizes_rows_to_mean0_var1():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0xE313)
    rows, dim = 3, 6
    x = _vec(rng, rows * dim, -4, 4)
    got = _compile_run_layernorm(x, rows, dim, [1.0] * dim, [0.0] * dim, 1e-5)
    for mean, var in layernorm_stats(got, rows, dim):
        assert abs(mean) < 1e-3, mean                         # centered (float32 round-off)
        assert abs(var - 1.0) < 1e-2, var                     # unit variance (up to eps + round-off)


def test_layernorm_c_emit_routes_rsqrt_through_the_libm_edge():
    src = emit_layernorm_c(4, 4, "ln")
    assert "math.h" in src and "sqrtf" in src                 # the rsqrt transcendental rides the libm edge
    assert "c.call.libm: edge" in src or "c.call.libm:sqrtf" in src   # the edge label in the docstring comment
    assert "void ln(const float *restrict x, const float *restrict gamma," in src   # the baked-in signature
    for bad in [(0, 4), (4, 0)]:
        try:
            emit_layernorm_c(*bad)
            assert False, f"{bad} should raise"
        except ValueError:
            pass


# --- 8. The layernorm link-flag rule: sqrtf -> -lm (REUSES the existing libm rule; no linkflags change) -----

def test_layernorm_link_flag_rule_reuses_the_existing_libm_rule():
    # The layernorm rsqrt sqrtf MATCHES the existing libm rule, so NO linkflags change is needed -- verify it.
    assert library_for_callee("sqrtf") == "-lm"                # the layernorm callee (the rsqrt edge)
    assert library_for_callee("sqrt") == "-lm"                 # the double form too
    assert library_for_callee("expf") == "-lm"                 # the softmax edge (the attention/activation reuse)
    # no regression on the other library rules + libc + unknown.
    assert library_for_callee("LAPACKE_ssyev") == "-llapack"   # E2 PCA still maps
    assert library_for_callee("LAPACKE_sgels") == "-llapack"   # E1 OLS still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"      # B5 BLAS still maps
    assert library_for_callee("free") == NO_FLAG               # libc-implicit, known (not unknown)
    assert library_for_callee("totally_unknown_fn") is None    # unknown-callee policy unchanged


# --- the two-truth quarantine: the transformer module imports no verifier / emits no Diagnostic ------------

def test_transformer_touches_no_verifier_and_emits_no_diagnostic():
    # The transformer block is a cost-side / oracle quantity (off the legality path), never an R-law legality
    # verdict. Inspect the IMPORTED module object: no verify / Diagnostic / Graded names bound, and no verifier
    # module imported (AST-inspected, exactly as test_ols / test_pca / test_attention).
    import bcir.kbcir.transformer as tf_mod
    bound = set(vars(tf_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(tf_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain floats (no graded/confidence wrapper).
    rng = random.Random(0xE314)
    spec = TransformerBlockSpec(4, 2, 8, 3, batch=1)
    params = _params(rng, 4, 8)
    x = _vec(rng, spec.rows * spec.d_model)
    out = transformer_block_reference(x, spec, params)
    assert isinstance(out, list) and all(isinstance(v, float) for v in out)
    assert isinstance(layernorm_stats(out, spec.rows, spec.d_model)[0][0], float)


def test_transformer_block_training_drives_loss_down():
    # E3 CONVERGENCE (H4): turn the forward-fidelity demo into an end-to-end TRAINING demo. The transformer block
    # is the (fixed) FEATURE EXTRACTOR -- its transcendental core (softmax / layernorm) breaks the closed-set
    # autodiff closure, so it is frozen and computed OUTSIDE the Tape -- and the M3 training loop fits a linear
    # readout over its features. The MSE loss must genuinely DROP to ~0 (the target is exactly linear in the
    # features, so a correctly wired forward + grad + optimizer recovers it). Mirrors the Tier-A RNN convergence
    # test (test_recurrent.test_tier_a_end_to_end_training_drives_loss_down): only the closed-set readout trains.
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE3C0)
    spec, params = _spec_params(rng, d_model=4, n_heads=2, d_ff=8, seq=3)
    dm = spec.d_model
    coef = [0.5, -0.3, 0.8, 0.2]
    feats, ys = [], []
    for _ in range(24):
        x = _vec(rng, spec.rows * dm, -1.0, 1.0)
        out = transformer_block_reference(x, spec, params)
        f = out[0:dm]                                              # the first row's d_model features
        feats.append(tuple(f))
        ys.append(sum(coef[j] * f[j] for j in range(dm)) + 0.1)   # an exactly-linear target over the features

    def model(tape, names, X):
        r = [tape.var(nm) for nm in names]                        # r0..r{dm-1} + bias
        out = []
        for row in X:
            pred = r[dm]
            for j in range(dm):
                pred = tape.add(pred, tape.mul(r[j], tape.const(row[j])))
            out.append(pred)
        return out

    names = [f"r{j}" for j in range(dm)] + ["b"]
    res = train(model, [0.0] * (dm + 1), Dataset(tuple(feats), tuple(ys)), loss="mse", optimizer="adam",
                epochs=150, lr=0.05, seed=1, param_names=names)
    assert res.train_loss[-1] < res.train_loss[0], (res.train_loss[0], res.train_loss[-1])   # loss decreases
    assert res.train_loss[-1] < 1e-3, res.train_loss[-1]          # the linear readout is exactly learnable


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
