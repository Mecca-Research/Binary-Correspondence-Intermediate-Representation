"""E3 deepening: TRANSCENDENTAL ADJOINTS for the transformer block -- the E4 Tier-B treatment
(closed-form gradients on the libm edge, FD-verified) applied to the block's transcendental core,
so BLOCK WEIGHTS can train (not just a linear readout). Also the rung-3 decoder primitives the
open-weight ladder's reference decode needs: RMSNorm and RoPE, forward + backward.

Contents (all flat row-major lists, matching `transformer.py`):
  * `softmax_grad`      -- dz = s * (g - <g, s>) per row (the exact softmax JVP-transpose)
  * `layernorm_grad`    -- the closed-form dx/dgamma/dbeta for `layernorm_reference`
  * `rmsnorm_reference` / `rmsnorm_grad` -- the Gemma/Llama-family normalizer, fwd + bwd
  * `rope_reference` / `rope_grad`       -- rotary position embedding; the backward is the
                                            EXACT inverse rotation (orthogonality)
  * `ff_ln_tail_reference` / `ff_ln_tail_grads` -- the post-attention sub-block
    y = LN(h + FF_relu(h)): forward + closed-form weight gradients (W1/b1/W2/b2/gamma/beta),
    the BLOCK-WEIGHT training surface (the attention-projection backward is the follow-on).

Cost-side oracle module (the two-truth quarantine): no verifier import, plain floats out.
FD verification + the block-weight convergence gate live in `test_transformer_grads.py`."""

from __future__ import annotations

import math

from .activation import softmax_reference
from .transformer import feedforward_reference, layernorm_reference

_DEFAULT_EPS = 1e-5


def softmax_grad(s: list[float], gout: list[float], axis_len: int) -> list[float]:
    """dL/dz given the softmax OUTPUT rows `s` and the upstream gradient `gout`:
    dz_i = s_i * (g_i - sum_j g_j s_j), per row of `axis_len`."""
    if axis_len < 1 or len(s) % axis_len or len(s) != len(gout):
        raise ValueError("softmax_grad: bad shapes")
    out = [0.0] * len(s)
    for r in range(len(s) // axis_len):
        o = r * axis_len
        dot = sum(gout[o + j] * s[o + j] for j in range(axis_len))
        for j in range(axis_len):
            out[o + j] = s[o + j] * (gout[o + j] - dot)
    return out


def layernorm_grad(x: list[float], rows: int, dim: int, gamma: list[float],
                   gout: list[float], eps: float = _DEFAULT_EPS
                   ) -> tuple[list[float], list[float], list[float]]:
    """The closed-form backward of `layernorm_reference` (population variance /dim):
    returns (dx, dgamma, dbeta). Per row, with xhat the normalized input and gy = gout*gamma:
    dx = inv/dim * (dim*gy - sum(gy) - xhat * sum(gy*xhat))."""
    if rows < 1 or dim < 1 or len(x) != rows * dim or len(gout) != rows * dim:
        raise ValueError("layernorm_grad: bad shapes")
    dx = [0.0] * (rows * dim)
    dgamma = [0.0] * dim
    dbeta = [0.0] * dim
    for r in range(rows):
        o = r * dim
        row = x[o:o + dim]
        mu = sum(row) / dim
        var = sum((v - mu) ** 2 for v in row) / dim
        inv = 1.0 / math.sqrt(var + eps)
        xhat = [(v - mu) * inv for v in row]
        gy = [gout[o + c] * gamma[c] for c in range(dim)]
        s1 = sum(gy)
        s2 = sum(gy[c] * xhat[c] for c in range(dim))
        for c in range(dim):
            dx[o + c] = inv / dim * (dim * gy[c] - s1 - xhat[c] * s2)
            dgamma[c] += gout[o + c] * xhat[c]
            dbeta[c] += gout[o + c]
    return dx, dgamma, dbeta


def rmsnorm_reference(x: list[float], rows: int, dim: int, gamma: list[float],
                      eps: float = 1e-6) -> list[float]:
    """RMSNorm (the Gemma/Llama-family normalizer): y = x / rms(x) * gamma, per row, with
    rms = sqrt(mean(x^2) + eps). No mean subtraction, no beta -- the rung-3 decode primitive."""
    if rows < 1 or dim < 1 or len(x) != rows * dim or len(gamma) != dim or eps <= 0:
        raise ValueError("rmsnorm_reference: bad shapes")
    out = [0.0] * (rows * dim)
    for r in range(rows):
        o = r * dim
        rms = math.sqrt(sum(v * v for v in x[o:o + dim]) / dim + eps)
        for c in range(dim):
            out[o + c] = x[o + c] / rms * gamma[c]
    return out


def rmsnorm_grad(x: list[float], rows: int, dim: int, gamma: list[float], gout: list[float],
                 eps: float = 1e-6) -> tuple[list[float], list[float]]:
    """The closed-form backward of `rmsnorm_reference`: returns (dx, dgamma). Per row, with
    S = sum(gamma * gout * x): dx_c = gamma_c*gout_c/rms - x_c * S / (dim * rms^3)."""
    if len(x) != rows * dim or len(gout) != rows * dim:
        raise ValueError("rmsnorm_grad: bad shapes")
    dx = [0.0] * (rows * dim)
    dgamma = [0.0] * dim
    for r in range(rows):
        o = r * dim
        ms = sum(v * v for v in x[o:o + dim]) / dim + eps
        rms = math.sqrt(ms)
        s = sum(gamma[c] * gout[o + c] * x[o + c] for c in range(dim))
        for c in range(dim):
            dx[o + c] = gamma[c] * gout[o + c] / rms - x[o + c] * s / (dim * ms * rms)
            dgamma[c] += gout[o + c] * x[o + c] / rms
    return dx, dgamma


def _rope_angles(pos: int, dim: int, base: float) -> list[float]:
    return [pos * (base ** (-(2.0 * k) / dim)) for k in range(dim // 2)]


def rope_reference(x: list[float], rows: int, dim: int, base: float = 10000.0,
                   pos_offset: int = 0) -> list[float]:
    """Rotary position embedding: row r (position pos_offset+r) has each channel pair
    (2k, 2k+1) rotated by theta_k = pos * base^(-2k/dim). `dim` must be even. Position 0 is
    the identity; every rotation preserves the pair norm (orthogonality)."""
    if rows < 1 or dim < 2 or dim % 2 or len(x) != rows * dim:
        raise ValueError("rope_reference: bad shapes (dim must be even)")
    out = [0.0] * (rows * dim)
    for r in range(rows):
        o = r * dim
        for k, th in enumerate(_rope_angles(pos_offset + r, dim, base)):
            c, s = math.cos(th), math.sin(th)
            x0, x1 = x[o + 2 * k], x[o + 2 * k + 1]
            out[o + 2 * k] = x0 * c - x1 * s
            out[o + 2 * k + 1] = x0 * s + x1 * c
    return out


def rope_grad(gout: list[float], rows: int, dim: int, base: float = 10000.0,
              pos_offset: int = 0) -> list[float]:
    """The EXACT backward of `rope_reference`: a rotation is orthogonal, so the adjoint is the
    inverse rotation of the upstream gradient (rotate by -theta). Satisfies the adjoint
    identity <rope(x), g> == <x, rope_grad(g)> to float round-off."""
    if rows < 1 or dim < 2 or dim % 2 or len(gout) != rows * dim:
        raise ValueError("rope_grad: bad shapes (dim must be even)")
    out = [0.0] * (rows * dim)
    for r in range(rows):
        o = r * dim
        for k, th in enumerate(_rope_angles(pos_offset + r, dim, base)):
            c, s = math.cos(th), math.sin(th)
            g0, g1 = gout[o + 2 * k], gout[o + 2 * k + 1]
            out[o + 2 * k] = g0 * c + g1 * s
            out[o + 2 * k + 1] = -g0 * s + g1 * c
    return out


# --- the block-weight training surface: the post-attention FF + LayerNorm tail -----------------

def ff_ln_tail_reference(h: list[float], rows: int, d_model: int, d_ff: int,
                         W1: list[float], b1: list[float], W2: list[float], b2: list[float],
                         gamma: list[float], beta: list[float],
                         eps: float = _DEFAULT_EPS) -> list[float]:
    """The post-attention sub-block y = LayerNorm(h + FF_relu(h)) -- the POST-LN tail of the E3
    block, reusing `feedforward_reference` + `layernorm_reference`. `h` is the (frozen) attention
    output; W1/b1/W2/b2/gamma/beta are the trainable BLOCK weights."""
    ff = feedforward_reference(h, rows, d_model, d_ff, W1, b1, W2, b2, activation="relu")
    u = [h[i] + ff[i] for i in range(rows * d_model)]
    return layernorm_reference(u, rows, d_model, gamma, beta, eps)


def ff_ln_tail_grads(h: list[float], rows: int, d_model: int, d_ff: int,
                     W1: list[float], b1: list[float], W2: list[float], b2: list[float],
                     gamma: list[float], beta: list[float], gout: list[float],
                     eps: float = _DEFAULT_EPS) -> dict:
    """Closed-form gradients of the tail's BLOCK WEIGHTS: {"W1","b1","W2","b2","gamma","beta"}.
    Backward chain: layernorm_grad over u = h + ff, then the two-matmul FF backward with the
    relu mask (a = h@W1+b1, r = relu(a), ff = r@W2+b2). `h` is frozen (its gradient is not
    materialized -- the attention-projection backward is the follow-on)."""
    n = rows * d_model
    if len(h) != n or len(gout) != n:
        raise ValueError("ff_ln_tail_grads: bad shapes")
    # forward intermediates (recomputed -- the oracle favors clarity over caching)
    a = [0.0] * (rows * d_ff)
    for r in range(rows):
        for j in range(d_ff):
            a[r * d_ff + j] = b1[j] + sum(h[r * d_model + i] * W1[i * d_ff + j]
                                          for i in range(d_model))
    relu = [v if v > 0.0 else 0.0 for v in a]
    ff = [0.0] * n
    for r in range(rows):
        for c in range(d_model):
            ff[r * d_model + c] = b2[c] + sum(relu[r * d_ff + j] * W2[j * d_model + c]
                                              for j in range(d_ff))
    u = [h[i] + ff[i] for i in range(n)]
    du, dgamma, dbeta = layernorm_grad(u, rows, d_model, gamma, gout, eps)
    # FF backward (dff == du: the residual path adds identity, which only dh would consume)
    dW2 = [0.0] * (d_ff * d_model)
    db2 = [0.0] * d_model
    da = [0.0] * (rows * d_ff)
    for r in range(rows):
        for c in range(d_model):
            g = du[r * d_model + c]
            db2[c] += g
            for j in range(d_ff):
                dW2[j * d_model + c] += relu[r * d_ff + j] * g
        for j in range(d_ff):
            dr = sum(W2[j * d_model + c] * du[r * d_model + c] for c in range(d_model))
            da[r * d_ff + j] = dr if a[r * d_ff + j] > 0.0 else 0.0
    dW1 = [0.0] * (d_model * d_ff)
    db1 = [0.0] * d_ff
    for r in range(rows):
        for j in range(d_ff):
            g = da[r * d_ff + j]
            db1[j] += g
            for i in range(d_model):
                dW1[i * d_ff + j] += h[r * d_model + i] * g
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "gamma": dgamma, "beta": dbeta}


# --- the attention-projection backward: the FULL block's weights now receive gradients ---------

def _t(m: list[float], rows: int, cols: int) -> list[float]:
    """Row-major transpose (a pure reshaping, mirroring attention._transpose)."""
    out = [0.0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            out[j * rows + i] = m[i * cols + j]
    return out


def mha_projection_grads(x: list[float], seq: int, d_model: int, n_heads: int,
                         w_q: list[float], w_k: list[float], w_v: list[float],
                         w_o: list[float], gout: list[float],
                         mask: list[float] | None = None) -> dict:
    """Closed-form gradients of the masked multi-head attention block's PROJECTION weights
    {"w_q","w_k","w_v","w_o"} and the block input {"x"} -- the follow-on `ff_ln_tail_grads`
    names, closing the block: EVERY transformer weight can now receive a gradient. The
    forward is exactly `transformer.multihead_attention_reference` for one sequence
    (Q/K/V projections -> per-head scaled scores + additive mask -> softmax -> A@V ->
    concat -> W_o); the backward composes the EXISTING adjoints (softmax_grad; matmul
    adjoints are the two transposed products; the additive mask is a constant, so it
    contributes identity on the scores and nothing to any parameter). Intermediates are
    recomputed (the oracle favors clarity over caching)."""
    from .attention import AttentionSpec, scores_reference   # noqa: PLC0415 -- local: no cycle
    from .matmul import matmul_reference                     # noqa: PLC0415
    if d_model % n_heads:
        raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
    n = seq * d_model
    if len(x) != n or len(gout) != n:
        raise ValueError("mha_projection_grads: bad shapes")
    if mask is not None and len(mask) != seq * seq:
        raise ValueError("mha_projection_grads: mask must be seq*seq")
    dk = d_model // n_heads
    spec = AttentionSpec(seq, dk)
    sc = spec.scale

    def split(m, h):
        return [m[i * d_model + h * dk + t] for i in range(seq) for t in range(dk)]

    q = matmul_reference(x, w_q, seq, d_model, d_model)      # forward, recomputed
    k = matmul_reference(x, w_k, seq, d_model, d_model)
    v = matmul_reference(x, w_v, seq, d_model, d_model)
    concat = [0.0] * n
    heads = []                                               # per head: (Ah, Qh, Kh, Vh)
    for h in range(n_heads):
        qh, kh, vh = split(q, h), split(k, h), split(v, h)
        s = scores_reference(qh, kh, spec)
        if mask is not None:
            s = [s[i] + float(mask[i]) for i in range(seq * seq)]
        a = softmax_reference(s, axis_len=seq)
        ctx = matmul_reference(a, vh, seq, dk, seq)
        for i in range(seq):
            concat[i * d_model + h * dk:i * d_model + (h + 1) * dk] = ctx[i * dk:(i + 1) * dk]
        heads.append((a, qh, kh, vh))

    dWo = matmul_reference(_t(concat, seq, d_model), gout, d_model, d_model, seq)
    dC = matmul_reference(gout, _t(w_o, d_model, d_model), seq, d_model, d_model)
    dQ = [0.0] * n
    dK = [0.0] * n
    dV = [0.0] * n
    for h in range(n_heads):
        a, qh, kh, vh = heads[h]
        dch = split(dC, h)
        da = matmul_reference(dch, _t(vh, seq, dk), seq, seq, dk)      # dA = dC @ V^T
        dvh = matmul_reference(_t(a, seq, seq), dch, seq, dk, seq)     # dV = A^T @ dC
        ds = softmax_grad(a, da, axis_len=seq)                         # through the softmax rows
        dqh = [sc * g for g in matmul_reference(ds, kh, seq, dk, seq)]        # dQ = sc * dS @ K
        dkh = [sc * g for g in matmul_reference(_t(ds, seq, seq), qh, seq, dk, seq)]  # dK = sc*dS^T@Q
        for i in range(seq):
            for t in range(dk):
                dQ[i * d_model + h * dk + t] = dqh[i * dk + t]
                dK[i * d_model + h * dk + t] = dkh[i * dk + t]
                dV[i * d_model + h * dk + t] = dvh[i * dk + t]
    xt = _t(x, seq, d_model)
    dWq = matmul_reference(xt, dQ, d_model, d_model, seq)
    dWk = matmul_reference(xt, dK, d_model, d_model, seq)
    dWv = matmul_reference(xt, dV, d_model, d_model, seq)
    dx1 = matmul_reference(dQ, _t(w_q, d_model, d_model), seq, d_model, d_model)
    dx2 = matmul_reference(dK, _t(w_k, d_model, d_model), seq, d_model, d_model)
    dx3 = matmul_reference(dV, _t(w_v, d_model, d_model), seq, d_model, d_model)
    dx = [dx1[i] + dx2[i] + dx3[i] for i in range(n)]
    return {"w_q": dWq, "w_k": dWk, "w_v": dWv, "w_o": dWo, "x": dx}


__all__ = ["softmax_grad", "layernorm_grad", "rmsnorm_reference", "rmsnorm_grad",
           "rope_reference", "rope_grad", "ff_ln_tail_reference", "ff_ln_tail_grads",
           "mha_projection_grads", "softmax_reference"]
