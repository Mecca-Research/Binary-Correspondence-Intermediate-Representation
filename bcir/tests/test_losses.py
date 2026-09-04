"""M1 (ML Tier-1, Pillar 5b / B4) -- the loss-function library on the B3 autodiff: the correctness gate.

``bcir.kbcir.losses`` adds MSE, softmax cross-entropy, BCE-with-logits, and hinge as a reference oracle
that composes with the EXISTING ``Tape`` autodiff so a full training step -- forward -> loss -> backward ->
parameter gradients -- runs end-to-end. The two-path design it implements (see the module docstring):

  * MSE is built ENTIRELY into the Tape (sub + dot + scale), so the EXISTING ``autodiff.grad`` and
    ``emit_autodiff_kernel_c`` differentiate / lower it for free -- the closed-set path.
  * softmax-CE and BCE-with-logits have transcendental forward VALUES (log/exp) but CLOSED-FORM gradients
    that live in the closed set (``softmax - onehot`` ; ``sigmoid - target``); each returns
    ``(loss_value, grad_logits)`` where ``grad_logits`` SEEDS the model's backward pass.

Covers, with the autodiff finite-difference gradient as the HARD gate:
  * MSE closed-set: ``autodiff.grad`` on the MSE-rooted Tape MATCHES ``finite_difference_grad``;
    ``mse_value`` / ``mse_grad`` match an independent stdlib recompute + the closed-form anchor;
  * MSE lowers to C: ``emit_autodiff_kernel_c`` + ``compile_and_run_grad_c`` reproduce the oracle (self-skips
    with no C compiler);
  * softmax-CE: loss == stdlib ``-log(softmax[true])`` + the uniform-logits anchor ``log(K)``;
    ``grad_logits`` == a central finite-difference of the loss; the grad sums to ~0;
  * BCE-with-logits: loss == an independent stable recompute; ``grad`` == a finite-difference of the loss;
    the stable form agrees with the naive form on moderate z and does not overflow on large |z|;
  * COMPOSITION: a 1-layer logistic-regression-shaped check -- ``logits = dot(w, x) + b`` (Tape), the
    ``grad_logits`` from BCE SEEDS the backward to give ``dL/dw`` / ``dL/db``, verified against a
    finite-difference of the full ``loss(w, b)`` (the headline: M1 unlocks logistic regression);
  * the two-truth invariant: the loss module touches no verifier and emits no Diagnostic.

All deterministic, pure-Python. The compile-and-run test self-skips when the toolchain is hidden (quick
tier), exactly like the autodiff-kernel / inference tests."""

import math
import shutil

from bcir.kbcir.autodiff import (
    Tape,
    evaluate,
    finite_difference_grad,
    grad,
    gradients_match,
    max_grad_error,
)
from bcir.kbcir.losses import (
    binary_cross_entropy_with_logits,
    hinge,
    mse,
    mse_grad,
    mse_value,
    softmax_cross_entropy,
)
from bcir.lower.autodiff_kernel import compile_and_run_grad_c, emit_autodiff_kernel_c


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


# === Path 1: MSE is closed-set -- autodiff.grad on the MSE-rooted Tape == finite difference ===========


def test_mse_into_tape_grad_matches_finite_difference_on_a_var_model():
    # the simplest model: predictions ARE the parameters (a few vars). MSE built into the Tape; the EXISTING
    # autodiff.grad must match the central finite-difference (the hard gate).
    t = Tape()
    preds = (t.var("p0"), t.var("p1"), t.var("p2"))
    loss = mse(t, preds, targets=[1.0, -2.0, 0.5])
    env = {"p0": 0.3, "p1": -1.1, "p2": 2.0}
    g = grad(t, loss, env)
    fd = finite_difference_grad(t, loss, env)
    assert gradients_match(g.grads, fd), (g.grads, fd, max_grad_error(g.grads, fd))
    # the closed-form anchor: dL/dp_i = (2/n)(p_i - target_i).
    anchor = mse_grad([env["p0"], env["p1"], env["p2"]], [1.0, -2.0, 0.5])
    for i, name in enumerate(("p0", "p1", "p2")):
        assert abs(g.grads[name] - anchor[i]) < 1e-6, (name, g.grads[name], anchor[i])


def test_mse_into_tape_grad_matches_fd_on_a_linear_model():
    # a real model DAG: pred = dot(w, x) + b for a couple of points; MSE over them; grad w.r.t. w, b matches
    # the finite difference -- proving the closed-set loss composes with a dot-based model end-to-end.
    t = Tape()
    w0, w1, b = t.var("w0"), t.var("w1"), t.var("b")
    xs = [(1.0, 2.0), (-1.0, 0.5), (3.0, -2.0)]
    targets = [1.0, 0.0, -1.0]
    preds = tuple(t.add(t.dot((w0, w1), (t.const(x0), t.const(x1))), b) for (x0, x1) in xs)
    loss = mse(t, preds, targets)
    env = {"w0": 0.4, "w1": -0.2, "b": 0.1}
    g = grad(t, loss, env)
    fd = finite_difference_grad(t, loss, env)
    assert gradients_match(g.grads, fd), (g.grads, fd, max_grad_error(g.grads, fd))


def test_mse_value_and_grad_reference_match_independent_recompute():
    pred = [0.5, -1.0, 2.0, 0.0]
    target = [1.0, -2.0, 1.5, 0.25]
    n = len(pred)
    # independent stdlib recompute of the value.
    expect_v = sum((p - tt) ** 2 for p, tt in zip(pred, target)) / n
    assert abs(mse_value(pred, target) - expect_v) < 1e-12
    # closed-form anchor for the gradient.
    expect_g = [(2.0 / n) * (p - tt) for p, tt in zip(pred, target)]
    got_g = mse_grad(pred, target)
    assert all(abs(a - b) < 1e-12 for a, b in zip(got_g, expect_g)), (got_g, expect_g)


def test_mse_value_matches_the_tape_forward():
    # the Tape-built MSE evaluates to the same scalar as the mse_value reference (the loss-into-Tape and the
    # reference are the same number).
    t = Tape()
    preds = (t.var("a"), t.var("b"), t.var("c"))
    target = [1.0, 2.0, -0.5]
    loss = mse(t, preds, target)
    env = {"a": 0.2, "b": 1.7, "c": -1.0}
    tape_v = evaluate(t, loss, env)
    ref_v = mse_value([env["a"], env["b"], env["c"]], target)
    assert abs(tape_v - ref_v) < 1e-12, (tape_v, ref_v)


def test_mse_rejects_bad_lengths():
    t = Tape()
    p = (t.var("p0"),)
    try:
        mse(t, p, targets=[1.0, 2.0])
        assert False, "length mismatch must raise"
    except ValueError:
        pass


# === MSE lowers to C: emitted forward + grads match the oracle (self-skips with no compiler) ===========


def test_mse_into_tape_lowers_to_c_and_matches_the_oracle():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    # a linear model + MSE, all closed-set, so emit_autodiff_kernel_c lowers it with NO new machinery.
    t = Tape()
    w0, w1, b = t.var("w0"), t.var("w1"), t.var("b")
    xs = [(1.0, 2.0), (-1.0, 0.5)]
    targets = [1.0, 0.0]
    preds = tuple(t.add(t.dot((w0, w1), (t.const(x0), t.const(x1))), b) for (x0, x1) in xs)
    loss = mse(t, preds, targets)
    params = ("w0", "w1", "b")
    env = {"w0": 0.4, "w1": -0.2, "b": 0.1}
    ok, value, grads, out = compile_and_run_grad_c(t, loss, params, env)
    assert ok, out
    oref = grad(t, loss, env)
    fd = finite_difference_grad(t, loss, env)
    assert abs(value - oref.value) < 1e-4, (value, oref.value, out)
    cgrad = {name: grads[i] for i, name in enumerate(params)}
    assert max_grad_error(cgrad, oref.grads) < 1e-4, (
        cgrad,
        oref.grads,
        out,
    )  # C == oracle reverse-mode
    assert max_grad_error(cgrad, fd) < 1e-4, (cgrad, fd, out)  # C == finite difference
    # the emit is the closed-set kernel: no libm, the closed primitive set only.
    src = emit_autodiff_kernel_c(t, loss, params)
    assert "math.h" not in src and "neg/add/sub/mul/div/dot/select" in src


# === Path 2a: softmax cross-entropy -- transcendental value, closed-form grad seed ====================


def _fd_grad_of_loss(loss_fn, logits, eps=1e-6):
    """Central finite-difference of a scalar loss w.r.t. each logit (the hard gate for the closed-form
    grad_logits). ``loss_fn`` maps a logit vector -> the scalar loss value."""
    out = []
    for i in range(len(logits)):
        plus = list(logits)
        plus[i] += eps
        minus = list(logits)
        minus[i] -= eps
        out.append((loss_fn(plus) - loss_fn(minus)) / (2 * eps))
    return out


def test_softmax_ce_value_matches_neg_log_softmax_of_the_true_class():
    logits = [2.0, 0.5, -1.0, 1.5]
    true = 2  # the true class index
    onehot = [1.0 if k == true else 0.0 for k in range(len(logits))]
    loss, _ = softmax_cross_entropy(logits, onehot)
    # independent stdlib recompute: -log(softmax[true]), stable max-subtraction.
    m = max(logits)
    ex = [math.exp(z - m) for z in logits]
    p_true = ex[true] / sum(ex)
    assert abs(loss - (-math.log(p_true))) < 1e-12, (loss, -math.log(p_true))


def test_softmax_ce_uniform_logits_anchor_is_log_K():
    # the closed-form anchor: with uniform logits every class is equally likely (p = 1/K), so the loss for
    # any true class is -log(1/K) = log(K).
    for K in (2, 3, 5):
        logits = [0.7] * K  # uniform -> equal probabilities
        onehot = [1.0] + [0.0] * (K - 1)
        loss, grad_logits = softmax_cross_entropy(logits, onehot)
        assert abs(loss - math.log(K)) < 1e-12, (K, loss, math.log(K))
        # at uniform logits softmax is 1/K everywhere; grad = 1/K - onehot, which sums to 0.
        assert abs(sum(grad_logits)) < 1e-12, (K, grad_logits)


def test_softmax_ce_grad_logits_matches_finite_difference_and_sums_to_zero():
    logits = [1.2, -0.4, 0.9, 2.1, -1.3]
    true = 3
    onehot = [1.0 if k == true else 0.0 for k in range(len(logits))]
    loss, grad_logits = softmax_cross_entropy(logits, onehot)

    def loss_only(z):
        return softmax_cross_entropy(z, onehot)[0]

    fd = _fd_grad_of_loss(loss_only, logits)
    assert all(abs(a - b) < 1e-5 for a, b in zip(grad_logits, fd)), (grad_logits, fd)
    # softmax - onehot: softmax sums to 1 and onehot sums to 1, so the gradient sums to ~0.
    assert abs(sum(grad_logits)) < 1e-9, grad_logits


def test_softmax_ce_rejects_bad_lengths():
    try:
        softmax_cross_entropy([1.0, 2.0], [1.0])
        assert False, "length mismatch must raise"
    except ValueError:
        pass


# === Path 2b: BCE-with-logits -- stable transcendental value, sigmoid-target grad seed ================


def test_bce_with_logits_value_matches_an_independent_stable_recompute():
    logits = [0.5, -1.2, 2.0, -0.3]
    target = [1.0, 0.0, 1.0, 0.0]
    loss, _ = binary_cross_entropy_with_logits(logits, target)
    # independent stable recompute (same algebra, recomputed here): mean of max(z,0) - z*y + log1p(exp(-|z|)).
    expect = sum(
        max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z))) for z, y in zip(logits, target)
    ) / len(logits)
    assert abs(loss - expect) < 1e-12, (loss, expect)


def test_bce_stable_form_agrees_with_naive_form_on_moderate_z():
    # on moderate |z| the naive -[y*log(sigmoid) + (1-y)*log(1-sigmoid)] is well-defined; the stable form
    # must agree with it.
    logits = [0.8, -0.6, 1.5, -2.0, 0.0]
    target = [1.0, 0.0, 1.0, 0.0, 1.0]
    loss, _ = binary_cross_entropy_with_logits(logits, target)
    naive = 0.0
    for z, y in zip(logits, target):
        s = 1.0 / (1.0 + math.exp(-z))
        naive += -(y * math.log(s) + (1.0 - y) * math.log(1.0 - s))
    naive /= len(logits)
    assert abs(loss - naive) < 1e-10, (loss, naive)


def test_bce_stable_form_does_not_overflow_on_large_logits():
    # the whole point of the stable form: large |z| where the naive sigmoid-then-log would overflow/underflow
    # (log(0)). The stable form returns a finite, correct value.
    logits = [100.0, -100.0, 50.0, -50.0]
    target = [1.0, 0.0, 0.0, 1.0]  # two correct (loss ~0), two wrong (loss ~|z|)
    loss, grad_logits = binary_cross_entropy_with_logits(logits, target)
    assert math.isfinite(loss) and loss >= 0.0, loss
    assert all(math.isfinite(g) for g in grad_logits), grad_logits
    # z=100,y=1 and z=-100,y=0 are confident+correct (~0 loss); z=50,y=0 and z=-50,y=1 are confident+WRONG
    # (~|z| loss each). mean ~= (0 + 0 + 50 + 50)/4 = 25.0 -- finite, the naive form would have hit log(0).
    assert abs(loss - (50.0 + 50.0) / 4.0) < 1e-3, loss


def test_bce_grad_matches_finite_difference_of_the_loss():
    logits = [0.7, -1.1, 0.3, 1.8]
    target = [1.0, 0.0, 0.0, 1.0]
    loss, grad_logits = binary_cross_entropy_with_logits(logits, target)

    def loss_only(z):
        return binary_cross_entropy_with_logits(z, target)[0]

    fd = _fd_grad_of_loss(loss_only, logits)
    assert all(abs(a - b) < 1e-5 for a, b in zip(grad_logits, fd)), (grad_logits, fd)


def test_bce_rejects_bad_lengths():
    try:
        binary_cross_entropy_with_logits([1.0], [0.0, 1.0])
        assert False, "length mismatch must raise"
    except ValueError:
        pass


# === Hinge (the optional SVM-style loss): value + subgradient match a finite-difference a.e. ===========


def test_hinge_value_and_subgradient():
    scores = [0.5, 2.0, -0.3, 1.5]
    target = [1.0, 1.0, -1.0, -1.0]  # labels in {-1, +1}
    loss, g = hinge(scores, target)
    n = len(scores)
    margins = [1.0 - y * s for s, y in zip(scores, target)]
    expect_v = sum(m if m > 0 else 0.0 for m in margins) / n
    assert abs(loss - expect_v) < 1e-12, (loss, expect_v)

    # away from the kinks (margins != 0) the subgradient is the true gradient; match a finite difference.
    def loss_only(s):
        return hinge(s, target)[0]

    fd = _fd_grad_of_loss(loss_only, scores, eps=1e-6)
    for i, m in enumerate(margins):
        if abs(m) > 1e-3:  # skip the measure-zero kink (a.e. convention)
            assert abs(g[i] - fd[i]) < 1e-5, (i, g[i], fd[i], m)


# === COMPOSITION: the headline -- logistic regression, grad_logits seeds the model backward ============


def _logreg_full_loss(w, b, x, y):
    """The full scalar logistic-regression loss as a plain Python function of (w, b): logits = dot(w, x) + b
    over one example, BCE-with-logits against label y. Used to finite-difference dL/dw, dL/db."""
    z = sum(wi * xi for wi, xi in zip(w, x)) + b
    loss, _ = binary_cross_entropy_with_logits([z], [y])
    return loss


def test_logistic_regression_composition_grad_via_seed_matches_finite_difference():
    # The end-to-end M1 headline: a 1-layer logistic regression. logits = dot(w, x) + b is built in the Tape;
    # grad_logits = sigmoid(z) - y from BCE SEEDS the model's backward to give dL/dw, dL/db; this must match a
    # finite-difference of the FULL loss(w, b). This proves the seed path composes with the model autodiff.
    x = [1.5, -0.5, 2.0]
    y = 1.0
    w_val = [0.3, -0.7, 0.2]
    b_val = 0.1

    # --- build the model (logits) in the Tape; NO loss node (the loss is transcendental) ---
    t = Tape()
    wvars = tuple(t.var(f"w{i}") for i in range(len(x)))
    bvar = t.var("b")
    xconst = tuple(t.const(xi) for xi in x)
    logits_node = t.add(t.dot(wvars, xconst), bvar)
    env = {f"w{i}": w_val[i] for i in range(len(x))}
    env["b"] = b_val

    # --- forward the model to the logit value, then the closed-form grad seed from BCE ---
    z = evaluate(t, logits_node, env)
    loss_val, grad_logits = binary_cross_entropy_with_logits([z], [y])
    seed = grad_logits[0]  # dL/dz = sigmoid(z) - y, the closed-set seed

    # --- SEED the model's backward: dL/dparam = seed * d(logits)/d(param). The model logits IS a scalar
    #     output, so we scale the model's own grad (seed = 1 case) by the seed. Equivalently, differentiate
    #     (seed * logits) -- but seed is a constant, so just multiply the model gradient by it. ---
    model_grad = grad(t, logits_node, env).grads  # d(logits)/d(param), seed 1.0
    seeded = {name: seed * gv for name, gv in model_grad.items()}

    # --- the gate: finite-difference the FULL loss(w, b) and compare to the seeded gradient ---
    eps = 1e-6
    fd = {}
    for i in range(len(x)):
        wp = list(w_val)
        wp[i] += eps
        wm = list(w_val)
        wm[i] -= eps
        fd[f"w{i}"] = (_logreg_full_loss(wp, b_val, x, y) - _logreg_full_loss(wm, b_val, x, y)) / (
            2 * eps
        )
    fd["b"] = (
        _logreg_full_loss(w_val, b_val + eps, x, y) - _logreg_full_loss(w_val, b_val - eps, x, y)
    ) / (2 * eps)

    assert gradients_match(seeded, fd, rtol=1e-4, atol=1e-6), (
        seeded,
        fd,
        max_grad_error(seeded, fd),
    )
    # and the closed forms directly: dL/dw_i = (sigmoid(z) - y) * x_i, dL/db = sigmoid(z) - y.
    s = 1.0 / (1.0 + math.exp(-z))
    for i in range(len(x)):
        assert abs(seeded[f"w{i}"] - (s - y) * x[i]) < 1e-7, (i, seeded[f"w{i}"], (s - y) * x[i])
    assert abs(seeded["b"] - (s - y)) < 1e-7, (seeded["b"], s - y)
    assert math.isfinite(loss_val)


def test_softmax_ce_composition_seed_matches_finite_difference_on_a_linear_head():
    # the multiclass analog: a linear head logits_k = dot(W_k, x) for K classes; the softmax-CE grad seeds
    # each logit's backward; dL/dW matches a finite-difference of the full loss. Proves softmax-CE composes.
    x = [1.0, -2.0]
    true = 1
    K = 3
    W = [[0.1, 0.2], [-0.3, 0.4], [0.5, -0.1]]  # K rows, each len(x)
    onehot = [1.0 if k == true else 0.0 for k in range(K)]

    def logits_of(Wmat):
        return [sum(wij * xi for wij, xi in zip(Wmat[k], x)) for k in range(K)]

    def full_loss(Wmat):
        return softmax_cross_entropy(logits_of(Wmat), onehot)[0]

    logits = logits_of(W)
    loss_val, grad_logits = softmax_cross_entropy(logits, onehot)
    # seed each class-k logit's backward; logits_k = dot(W_k, x), so dL/dW_kj = grad_logits[k] * x[j].
    seeded = {(k, j): grad_logits[k] * x[j] for k in range(K) for j in range(len(x))}
    # finite difference of the full loss w.r.t. each W_kj.
    eps = 1e-6
    for k in range(K):
        for j in range(len(x)):
            Wp = [row[:] for row in W]
            Wp[k][j] += eps
            Wm = [row[:] for row in W]
            Wm[k][j] -= eps
            fd = (full_loss(Wp) - full_loss(Wm)) / (2 * eps)
            assert abs(seeded[(k, j)] - fd) < 1e-5, (k, j, seeded[(k, j)], fd)
    assert math.isfinite(loss_val)


# === The two-truth invariant: the loss module is cost/optimization-side, never a verdict ===============


def test_losses_touch_no_verifier_and_emit_no_diagnostic():
    # the loss module must not import or call the verifier rail (a loss is a cost-side quantity, never an
    # R-law legality verdict). Inspect the IMPORTED module object: no verify / Diagnostic / Graded names are
    # bound in its namespace, and no verifier module is among its imports. (The module docstring discusses
    # the quarantine in prose; we check the live code, not the prose, so strip the docstring first.)
    import ast
    import bcir.kbcir.losses as losses_mod

    # (1) no verifier / diagnostic / graded name is bound in the live module namespace.
    bound = set(vars(losses_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    # (2) no verifier module is imported (parse the AST so a docstring mention can't trip it).
    tree = ast.parse(open(losses_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # (3) the returned values are plain numeric types (no graded/confidence wrapper).
    loss, g = binary_cross_entropy_with_logits([0.5, -0.5], [1.0, 0.0])
    assert isinstance(loss, float) and isinstance(g, list) and all(isinstance(v, float) for v in g)
    lv = mse_value([1.0, 2.0], [0.0, 0.0])
    assert isinstance(lv, float)


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
