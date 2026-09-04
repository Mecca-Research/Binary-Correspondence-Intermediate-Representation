"""E3 deepening + rung-3 primitives: the transcendental adjoints (`kbcir/transformer_grads.py`).

The E4 Tier-B treatment applied to the transformer's transcendental core: closed-form
softmax/layernorm/RMSNorm backward, verified against CENTRAL FINITE DIFFERENCES through a random
scalar objective J = <c, f(x)> (the standard adjoint check); RoPE's backward verified by the
EXACT adjoint identity <rope(x), g> == <x, rope_grad(g)> (rotations are orthogonal). The
headline: BLOCK-WEIGHT training -- a student post-attention tail (FF + LayerNorm) recovers a
teacher's function over frozen attention features via the closed-form weight gradients, gated by
the shared convergence gate. This closes the audit's "no transformer weight ever receives a
gradient" gap for the tail; the attention-projection backward is the stated follow-on."""

import ast
import math
import random

from bcir.kbcir.activation import softmax_reference
from bcir.kbcir.transformer_grads import (
    ff_ln_tail_grads,
    ff_ln_tail_reference,
    layernorm_grad,
    rmsnorm_grad,
    rmsnorm_reference,
    rope_grad,
    rope_reference,
    softmax_grad,
)
from bcir.tests._convergence import assert_converged


def _vec(rng, n, lo=-1.0, hi=1.0):
    return [rng.uniform(lo, hi) for _ in range(n)]


def _fd_grad(J, x, i, h=1e-6):
    xp, xm = list(x), list(x)
    xp[i] += h
    xm[i] -= h
    return (J(xp) - J(xm)) / (2 * h)


def test_softmax_grad_matches_finite_differences():
    rng = random.Random(0xE3A1)
    rows, d = 3, 5
    x = _vec(rng, rows * d, -2, 2)
    c = _vec(rng, rows * d)

    def J(v):
        s = softmax_reference(v, d)
        return sum(c[i] * s[i] for i in range(rows * d))

    s = softmax_reference(x, d)
    got = softmax_grad(s, c, d)
    for i in range(rows * d):
        assert abs(got[i] - _fd_grad(J, x, i)) < 1e-6, i


def test_layernorm_grad_matches_finite_differences():
    rng = random.Random(0xE3A2)
    rows, d = 3, 4
    x = _vec(rng, rows * d, -2, 2)
    gamma, beta = _vec(rng, d, 0.5, 1.5), _vec(rng, d, -0.3, 0.3)
    c = _vec(rng, rows * d)

    from bcir.kbcir.transformer import layernorm_reference

    def Jx(v):
        y = layernorm_reference(v, rows, d, gamma, beta)
        return sum(c[i] * y[i] for i in range(rows * d))

    dx, dgamma, dbeta = layernorm_grad(x, rows, d, gamma, c)
    for i in range(rows * d):
        assert abs(dx[i] - _fd_grad(Jx, x, i)) < 1e-5, ("dx", i)
    for k in range(d):

        def Jg(g, k=k):
            gg = list(gamma)
            gg[k] = g[0]
            y = layernorm_reference(x, rows, d, gg, beta)
            return sum(c[i] * y[i] for i in range(rows * d))

        assert abs(dgamma[k] - _fd_grad(Jg, [gamma[k]], 0)) < 1e-5, ("dgamma", k)

        def Jb(b, k=k):
            bb = list(beta)
            bb[k] = b[0]
            y = layernorm_reference(x, rows, d, gamma, bb)
            return sum(c[i] * y[i] for i in range(rows * d))

        assert abs(dbeta[k] - _fd_grad(Jb, [beta[k]], 0)) < 1e-5, ("dbeta", k)


def test_rmsnorm_forward_matches_a_naive_direct_computation():
    rng = random.Random(0xE3A3)
    rows, d = 4, 6
    x = _vec(rng, rows * d, -3, 3)
    gamma = _vec(rng, d, 0.5, 1.5)
    got = rmsnorm_reference(x, rows, d, gamma, eps=1e-6)
    for r in range(rows):
        rms = math.sqrt(sum(v * v for v in x[r * d : (r + 1) * d]) / d + 1e-6)
        for c in range(d):
            assert abs(got[r * d + c] - x[r * d + c] / rms * gamma[c]) < 1e-12


def test_rmsnorm_grad_matches_finite_differences():
    rng = random.Random(0xE3A4)
    rows, d = 3, 4
    x = _vec(rng, rows * d, -2, 2)
    gamma = _vec(rng, d, 0.5, 1.5)
    c = _vec(rng, rows * d)

    def Jx(v):
        y = rmsnorm_reference(v, rows, d, gamma)
        return sum(c[i] * y[i] for i in range(rows * d))

    dx, dgamma = rmsnorm_grad(x, rows, d, gamma, c)
    for i in range(rows * d):
        assert abs(dx[i] - _fd_grad(Jx, x, i)) < 1e-5, ("dx", i)
    for k in range(d):

        def Jg(g, k=k):
            gg = list(gamma)
            gg[k] = g[0]
            y = rmsnorm_reference(x, rows, d, gg)
            return sum(c[i] * y[i] for i in range(rows * d))

        assert abs(dgamma[k] - _fd_grad(Jg, [gamma[k]], 0)) < 1e-5, ("dgamma", k)


def test_rope_rotates_pairs_norm_preserving_and_position0_is_identity():
    rng = random.Random(0xE3A5)
    rows, d = 4, 8
    x = _vec(rng, rows * d, -2, 2)
    y = rope_reference(x, rows, d)
    assert all(abs(a - b) < 1e-12 for a, b in zip(y[:d], x[:d]))  # position 0: identity
    for r in range(rows):  # pair norms preserved
        for k in range(d // 2):
            nx = x[r * d + 2 * k] ** 2 + x[r * d + 2 * k + 1] ** 2
            ny = y[r * d + 2 * k] ** 2 + y[r * d + 2 * k + 1] ** 2
            assert abs(nx - ny) < 1e-9


def test_rope_grad_satisfies_the_exact_adjoint_identity():
    rng = random.Random(0xE3A6)
    rows, d = 3, 6
    x, g = _vec(rng, rows * d), _vec(rng, rows * d)
    lhs = sum(a * b for a, b in zip(rope_reference(x, rows, d, pos_offset=2), g))
    rhs = sum(a * b for a, b in zip(x, rope_grad(g, rows, d, pos_offset=2)))
    assert abs(lhs - rhs) < 1e-9, (lhs, rhs)


def test_tail_weight_grads_match_finite_differences():
    rng = random.Random(0xE3A7)
    rows, dm, dff = 2, 3, 4
    h = _vec(rng, rows * dm)
    W1, b1 = _vec(rng, dm * dff), _vec(rng, dff)
    W2, b2 = _vec(rng, dff * dm), _vec(rng, dm)
    gamma, beta = _vec(rng, dm, 0.5, 1.5), _vec(rng, dm, -0.2, 0.2)
    c = _vec(rng, rows * dm)
    g = ff_ln_tail_grads(h, rows, dm, dff, W1, b1, W2, b2, gamma, beta, c)
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "gamma": gamma, "beta": beta}
    for name, p in params.items():
        for k in range(len(p)):

            def J(v, name=name, k=k):
                q = {n: list(x) for n, x in params.items()}
                q[name][k] = v[0]
                y = ff_ln_tail_reference(
                    h, rows, dm, dff, q["W1"], q["b1"], q["W2"], q["b2"], q["gamma"], q["beta"]
                )
                return sum(c[i] * y[i] for i in range(rows * dm))

            assert abs(g[name][k] - _fd_grad(J, [p[k]], 0)) < 1e-5, (name, k)


def test_block_weights_train_end_to_end_through_the_transcendental_tail():
    # E3 BLOCK-WEIGHT training (the audit gap: "no transformer weight ever receives a
    # gradient"): a student FF+LayerNorm tail -- REAL block weights, through a relu FF AND the
    # layernorm transcendental -- recovers a teacher tail's function over frozen attention
    # features, by plain SGD on the closed-form gradients. Shared convergence gate.
    rng = random.Random(0xE3A8)
    rows, dm, dff, n = 4, 3, 5, 12
    teacher = {
        "W1": _vec(rng, dm * dff),
        "b1": _vec(rng, dff),
        "W2": _vec(rng, dff * dm),
        "b2": _vec(rng, dm),
        "gamma": _vec(rng, dm, 0.8, 1.2),
        "beta": _vec(rng, dm, -0.1, 0.1),
    }
    feats = [_vec(rng, rows * dm) for _ in range(n)]  # the frozen attention outputs
    targets = [
        ff_ln_tail_reference(
            f,
            rows,
            dm,
            dff,
            teacher["W1"],
            teacher["b1"],
            teacher["W2"],
            teacher["b2"],
            teacher["gamma"],
            teacher["beta"],
        )
        for f in feats
    ]
    student = {k: [v + rng.uniform(-0.3, 0.3) for v in p] for k, p in teacher.items()}
    p0 = {k: list(v) for k, v in student.items()}
    losses = []
    lr = 0.05
    for _epoch in range(400):
        total = 0.0
        for f, t in zip(feats, targets):
            y = ff_ln_tail_reference(
                f,
                rows,
                dm,
                dff,
                student["W1"],
                student["b1"],
                student["W2"],
                student["b2"],
                student["gamma"],
                student["beta"],
            )
            m = rows * dm
            total += sum((y[i] - t[i]) ** 2 for i in range(m)) / m
            gout = [2.0 * (y[i] - t[i]) / m for i in range(m)]
            g = ff_ln_tail_grads(
                f,
                rows,
                dm,
                dff,
                student["W1"],
                student["b1"],
                student["W2"],
                student["b2"],
                student["gamma"],
                student["beta"],
                gout,
            )
            for k in student:
                student[k] = [w - lr * gv for w, gv in zip(student[k], g[k])]
        losses.append(total / n)
    assert any(student[k] != p0[k] for k in student)  # the block weights moved
    assert_converged(losses, final_below=3e-4, max_ratio=0.01, name="E3 block-weight tail")


def test_attention_projection_grads_match_finite_differences():
    """The follow-on ff_ln_tail_grads named: closed-form gradients of the masked multi-head
    attention block's projections (w_q/w_k/w_v/w_o) AND the block input, checked against
    central finite differences through J = <c, mha(x)> where the forward is the INDEPENDENT
    transformer.multihead_attention_reference (cross-implementation check; measured worst
    FD gap ~4e-10, gate 1e-5)."""
    from bcir.kbcir.transformer import (
        TransformerBlockSpec,
        _MHAParams,
        causal_mask,
        multihead_attention_reference,
    )
    from bcir.kbcir.transformer_grads import mha_projection_grads

    rng = random.Random(0xA77)
    seq, dm, nh = 3, 4, 2
    spec = TransformerBlockSpec(d_model=dm, n_heads=nh, d_ff=4, seq_len=seq, batch=1)
    x = _vec(rng, seq * dm)
    mask = causal_mask(seq)
    params = {k: _vec(rng, dm * dm) for k in ("w_q", "w_k", "w_v", "w_o")}
    c = _vec(rng, seq * dm)

    def fwd(p, xv):
        return multihead_attention_reference(
            xv, spec, _MHAParams(p["w_q"], p["w_k"], p["w_v"], p["w_o"]), mask
        )

    g = mha_projection_grads(
        x, seq, dm, nh, params["w_q"], params["w_k"], params["w_v"], params["w_o"], c, mask
    )
    for name, p in params.items():
        for k in range(len(p)):

            def J(v, name=name, k=k):
                q = {n2: list(w) for n2, w in params.items()}
                q[name][k] = v[0]
                y = fwd(q, x)
                return sum(c[i] * y[i] for i in range(seq * dm))

            assert abs(g[name][k] - _fd_grad(J, [p[k]], 0)) < 1e-5, (name, k)
    for k in range(seq * dm):  # the block-input gradient too

        def Jx(v, k=k):
            x2 = list(x)
            x2[k] = v[0]
            y = fwd(params, x2)
            return sum(c[i] * y[i] for i in range(seq * dm))

        assert abs(g["x"][k] - _fd_grad(Jx, [x[k]], 0)) < 1e-5, k


def test_attention_projections_train_end_to_end():
    """THE HEADLINE: block-weight fine-tuning through the FULL attention block -- a student's
    four projections recover a teacher's masked-MHA function via the closed-form gradients,
    gated by the shared convergence gate (measured 0.31 -> 9.2e-7 over 400 epochs, lr 2.0)."""
    from bcir.kbcir.transformer import (
        TransformerBlockSpec,
        _MHAParams,
        causal_mask,
        multihead_attention_reference,
    )
    from bcir.kbcir.transformer_grads import mha_projection_grads

    rng = random.Random(0xA77)
    seq, dm, nh = 3, 4, 2
    n = seq * dm
    spec = TransformerBlockSpec(d_model=dm, n_heads=nh, d_ff=4, seq_len=seq, batch=1)
    x = _vec(rng, n)
    mask = causal_mask(seq)
    teacher = {k: _vec(rng, dm * dm) for k in ("w_q", "w_k", "w_v", "w_o")}

    def fwd(p):
        return multihead_attention_reference(
            x, spec, _MHAParams(p["w_q"], p["w_k"], p["w_v"], p["w_o"]), mask
        )

    y = fwd(teacher)
    student = {k: _vec(rng, dm * dm, -0.3, 0.3) for k in ("w_q", "w_k", "w_v", "w_o")}
    losses = []
    for _ in range(400):
        out = fwd(student)
        losses.append(sum((out[i] - y[i]) ** 2 for i in range(n)) / n)
        gout = [2.0 * (out[i] - y[i]) / n for i in range(n)]
        g = mha_projection_grads(
            x,
            seq,
            dm,
            nh,
            student["w_q"],
            student["w_k"],
            student["w_v"],
            student["w_o"],
            gout,
            mask,
        )
        for k in student:
            student[k] = [w - 2.0 * gg for w, gg in zip(student[k], g[k])]
    assert_converged(
        losses, final_below=1e-5, max_ratio=1e-4, name="attention-projection fine-tuning"
    )


def test_transformer_grads_touches_no_verifier():
    import bcir.kbcir.transformer_grads as tg

    tree = ast.parse(open(tg.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)


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
