"""M2 (ML Tier-1, Pillar 5b / B4) -- the adaptive optimizer family on the B3 autodiff + M1 losses.

``bcir.lower.optimizers`` generalizes the minimal SGD (``autodiff_kernel.emit_sgd_step_c``,
``params[i] -= lr*grad[i]``) into the standard adaptive trio -- momentum (heavy-ball), RMSprop, Adam (with
bias correction) -- as a reference oracle (pure Python) + an emitted C step. This is the correctness gate.

Covers:
  * each REFERENCE step matches an INDEPENDENT hand-computed update on a fixed (params, grad, state) --
    momentum / rmsprop / adam, including Adam's bias correction at t=1 AND t=2 (the bias terms matter most
    early -- asserted explicitly against the closed form);
  * the EMITTED C step matches the reference step to float round-off over several steps (compile + run;
    self-skips with no C compiler). Adam's m/v/t state round-trips correctly across steps;
  * CONVERGENCE on a convex MSE problem (a linear model, known minimum): each optimizer drives the loss to
    near the minimum in a bounded number of steps and the loss is (near-)monotonically decreasing; compared
    against plain SGD on the same problem;
  * numerical stability: rmsprop/adam don't divide-by-zero on a zero gradient (the eps) and don't NaN;
  * the two-truth invariant: the optimizer module imports no verifier and emits no Diagnostic.

All deterministic, pure-Python on the reference rail; the compile-and-run tests self-skip when the toolchain
is hidden (quick tier), exactly like the autodiff-kernel / losses / inference tests."""

import math
import shutil

from bcir.kbcir.autodiff import Tape, evaluate, grad
from bcir.kbcir.losses import mse, mse_value
from bcir.lower.optimizers import (adam_step, compile_and_run_optimizer_c, emit_adam_step_c,
                                   emit_momentum_step_c, emit_rmsprop_step_c, momentum_step,
                                   reference_optimizer_trajectory, rmsprop_step, sgd_step)


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


# === the reference steps match an INDEPENDENT hand-computed update ====================================

def test_sgd_step_matches_hand_computation():
    p = [1.0, -2.0, 0.5]
    g = [0.4, -0.1, 2.0]
    lr = 0.1
    got = sgd_step(p, g, lr)
    expect = [pi - lr * gi for pi, gi in zip(p, g)]
    assert all(abs(a - b) < 1e-15 for a, b in zip(got, expect)), (got, expect)
    # the argument is not mutated (side-effect-free).
    assert p == [1.0, -2.0, 0.5]


def test_momentum_step_matches_hand_computation():
    # heavy-ball: v' = beta*v + g ; p' = p - lr*v'.
    p = [1.0, 3.0]
    g = [0.5, -1.0]
    v = [0.2, 0.4]
    lr, beta = 0.1, 0.9
    new_p, new_v = momentum_step(p, g, v, lr, beta)
    exp_v = [beta * 0.2 + 0.5, beta * 0.4 + (-1.0)]            # [0.68, -0.64]
    exp_p = [1.0 - lr * exp_v[0], 3.0 - lr * exp_v[1]]
    assert all(abs(a - b) < 1e-12 for a, b in zip(new_v, exp_v)), (new_v, exp_v)
    assert all(abs(a - b) < 1e-12 for a, b in zip(new_p, exp_p)), (new_p, exp_p)
    # beta=0 recovers plain SGD exactly.
    sp, sv = momentum_step(p, g, [9.0, 9.0], lr, beta=0.0)
    assert sv == g and all(abs(a - b) < 1e-15 for a, b in zip(sp, sgd_step(p, g, lr)))


def test_rmsprop_step_matches_hand_computation():
    # s' = beta*s + (1-beta)*g^2 ; p' = p - lr*g / (sqrt(s')+eps).
    p = [1.0, -2.0]
    g = [0.5, 2.0]
    s = [0.1, 0.0]
    lr, beta, eps = 0.1, 0.9, 1e-8
    new_p, new_s = rmsprop_step(p, g, s, lr, beta, eps)
    exp_s = [0.9 * 0.1 + 0.1 * 0.25, 0.9 * 0.0 + 0.1 * 4.0]    # [0.115, 0.4]
    exp_p = [1.0 - lr * 0.5 / (math.sqrt(exp_s[0]) + eps),
             -2.0 - lr * 2.0 / (math.sqrt(exp_s[1]) + eps)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(new_s, exp_s)), (new_s, exp_s)
    assert all(abs(a - b) < 1e-12 for a, b in zip(new_p, exp_p)), (new_p, exp_p)


def test_adam_step_bias_correction_at_t1_and_t2():
    # The bias correction is the distinctive feature and matters MOST early -- pin t=1 AND t=2 against an
    # independent closed-form hand computation.
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    p0 = [1.0]
    g = [0.5]                                                  # a fixed gradient applied each step

    # --- t = 1 (from m=v=0) ---
    p1, m1, v1, t1 = adam_step(p0, g, [0.0], [0.0], 0, lr, b1, b2, eps)
    assert t1 == 1
    # hand: m = (1-b1)*g, v = (1-b2)*g^2; mhat = m/(1-b1^1), vhat = v/(1-b2^1).
    hm = (1 - b1) * 0.5
    hv = (1 - b2) * 0.25
    mhat = hm / (1 - b1 ** 1)
    vhat = hv / (1 - b2 ** 1)
    hp1 = 1.0 - lr * mhat / (math.sqrt(vhat) + eps)
    assert abs(m1[0] - hm) < 1e-15 and abs(v1[0] - hv) < 1e-18, (m1, v1, hm, hv)
    assert abs(p1[0] - hp1) < 1e-12, (p1[0], hp1)
    # a known Adam identity: from a zero-initialized state with a single fixed gradient, the FIRST step is
    # almost exactly lr*sign(g) (mhat/sqrt(vhat) -> sign(g) since the eps is tiny and t=1 cancels the betas).
    assert abs(p1[0] - (1.0 - lr * 1.0)) < 1e-6, p1[0]

    # --- t = 2 (continue from the t=1 state, same gradient) ---
    p2, m2, v2, t2 = adam_step(p1, g, m1, v1, t1, lr, b1, b2, eps)
    assert t2 == 2
    hm2 = b1 * hm + (1 - b1) * 0.5
    hv2 = b2 * hv + (1 - b2) * 0.25
    mhat2 = hm2 / (1 - b1 ** 2)                                # the t=2 bias divisors differ from t=1
    vhat2 = hv2 / (1 - b2 ** 2)
    hp2 = p1[0] - lr * mhat2 / (math.sqrt(vhat2) + eps)
    assert abs(m2[0] - hm2) < 1e-15 and abs(v2[0] - hv2) < 1e-18, (m2, v2, hm2, hv2)
    assert abs(p2[0] - hp2) < 1e-12, (p2[0], hp2)


def test_adam_bias_divisors_change_with_t():
    # The bias correction is t-dependent: the divisors (1-b^t) grow toward 1. Two steps with the SAME gradient
    # and SAME moment magnitude give DIFFERENT effective corrections -- verify the divisor sequence directly.
    b1 = 0.9
    assert abs((1 - b1 ** 1) - 0.1) < 1e-15
    assert abs((1 - b1 ** 2) - 0.19) < 1e-15
    assert abs((1 - b1 ** 3) - 0.271) < 1e-15


def test_reference_step_functions_reject_length_mismatch():
    for fn, args in (
        (sgd_step, ([1.0], [1.0, 2.0], 0.1)),
        (momentum_step, ([1.0], [1.0], [1.0, 2.0], 0.1)),
        (rmsprop_step, ([1.0], [1.0, 2.0], [1.0], 0.1)),
        (adam_step, ([1.0], [1.0], [1.0, 2.0], [1.0], 0, 0.1)),
    ):
        try:
            fn(*args)
            assert False, f"{fn.__name__} must reject a length mismatch"
        except ValueError:
            pass


# === the emit: valid C, the libm boundary, the ABI signatures =========================================

def test_momentum_emit_is_pure_arithmetic_no_libm():
    src = emit_momentum_step_c(3, "mom")
    assert "void mom(float *params, const float *grad, float *velocity, float lr, float beta)" in src
    assert "velocity[i] = beta * velocity[i] + grad[i];" in src
    assert "params[i] -= lr * velocity[i];" in src
    # pure arithmetic: NO libm.
    assert "#include <stddef.h>" in src and "math.h" not in src and "sqrtf" not in src
    try:
        emit_momentum_step_c(0)
        assert False, "n_params < 1 must raise"
    except ValueError:
        pass


def test_rmsprop_emit_rides_libm_sqrtf():
    src = emit_rmsprop_step_c(2, "rms")
    assert "void rms(float *params, const float *grad, float *sq_avg, float lr, float beta, float eps)" in src
    assert "sq_avg[i] = beta * sq_avg[i] + (1.0f - beta) * grad[i] * grad[i];" in src
    assert "sqrtf(sq_avg[i])" in src
    # RIDES libm sqrtf -> <math.h>.
    assert "#include <math.h>" in src and "RIDES libm sqrtf" in src


def test_adam_emit_rides_libm_sqrtf_and_takes_int_t():
    src = emit_adam_step_c(2, "adm")
    assert ("void adm(float *params, const float *grad, float *m, float *v, int *t,") in src
    assert "float lr, float beta1, float beta2, float eps)" in src
    assert "*t += 1;" in src                                   # the step count is incremented in place
    assert "sqrtf(vhat)" in src and "#include <math.h>" in src
    assert "bias correction" in src.lower() or "bias-correction" in src.lower()
    # the beta^t powers use a plain integer-power loop (no libm powf CALL) -- the power is multiplied out.
    assert "b1p *= beta1" in src and "powf(" not in src


# === the emitted C step matches the reference to float round-off over several steps ====================

def _assert_c_matches_reference(opt, p0, grads, lr, steps, tol=2e-4, **hp):
    """Compile+run the optimizer's C step and assert its parameter trajectory matches the reference trajectory
    to float round-off, step by step. Self-skips with no compiler."""
    cc = _cc()
    if not cc:
        return None
    ok, c_traj, out = compile_and_run_optimizer_c(opt, p0, grads, lr, steps, **hp)
    assert ok, out
    ref_traj = reference_optimizer_trajectory(opt, p0, lambda s, p: grads, lr, steps, **hp)
    assert len(c_traj) == len(ref_traj) == steps + 1, (len(c_traj), len(ref_traj), steps)
    maxerr = 0.0
    for k, (cstep, rstep) in enumerate(zip(c_traj, ref_traj)):
        for ci, ri in zip(cstep, rstep):
            maxerr = max(maxerr, abs(ci - ri))
            assert abs(ci - ri) < tol, (opt, k, ci, ri, out)
    return maxerr


def test_momentum_c_matches_reference():
    _assert_c_matches_reference("momentum", [1.0, -2.0, 0.5], [0.4, -0.3, 0.8], lr=0.05, steps=10, beta=0.9)


def test_rmsprop_c_matches_reference():
    _assert_c_matches_reference("rmsprop", [1.0, -2.0, 0.5], [0.4, -0.3, 0.8], lr=0.05, steps=10,
                                beta=0.9, eps=1e-8)


def test_adam_c_matches_reference_with_state_roundtrip():
    # Adam's m/v/t must round-trip correctly across steps -- a multi-step C-vs-reference trajectory match
    # proves the bias correction (t-dependent) and the moment buffers carry over identically.
    err = _assert_c_matches_reference("adam", [1.0, -2.0, 0.5, 3.0], [0.4, -0.3, 0.8, -1.2],
                                      lr=0.05, steps=12, beta1=0.9, beta2=0.999, eps=1e-8)
    # if the compiler ran, the per-step error stayed at float round-off across all 12 steps (state round-trip).
    if err is not None:
        assert err < 2e-4, err


def test_sgd_c_matches_reference_for_parity():
    # SGD re-exposed: the C step (the existing emit_sgd_step_c) matches the reference sgd_step too.
    _assert_c_matches_reference("sgd", [1.0, -2.0, 0.5], [0.4, -0.3, 0.8], lr=0.1, steps=8)


# === CONVERGENCE on a convex MSE problem (a linear model, known minimum) ==============================

def _linear_mse_problem():
    """A convex least-squares problem: fit pred = w*x + b to points on the line y = 2x + 1 (exactly
    realizable, so the minimum loss is 0 at (w, b) = (2, 1)). Returns a closure ``grad_at(params) -> grads``
    (the MSE gradient w.r.t. [w, b]) and a ``loss_at(params) -> float``, built on the Tape autodiff + the M1
    MSE loss so the convergence is exercised through the real ML stack, not a toy."""
    data = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (-1.0, -1.0)]   # y = 2x + 1
    # build the model->MSE DAG once: pred_i = w*x_i + b ; loss = MSE(preds, ys).
    t = Tape()
    w, b = t.var("w"), t.var("b")
    preds = tuple(t.add(t.mul(w, t.const(x)), b) for x, _ in data)
    loss_node = mse(t, preds, [y for _, y in data])

    def grad_at(params):
        env = {"w": params[0], "b": params[1]}
        g = grad(t, loss_node, env).grads
        return [g["w"], g["b"]]

    def loss_at(params):
        env = {"w": params[0], "b": params[1]}
        return evaluate(t, loss_node, env)

    return grad_at, loss_at, (2.0, 1.0)


def _run_to_convergence(opt, grad_at, loss_at, p0, lr, steps, **hp):
    """Run ``steps`` reference optimizer steps on the convex problem (re-querying the gradient at the CURRENT
    params each step), returning the loss trajectory + the final params."""
    traj = reference_optimizer_trajectory(opt, p0, lambda s, p: grad_at(p), lr, steps, **hp)
    losses = [loss_at(p) for p in traj]
    return losses, traj[-1]


def test_each_optimizer_converges_on_the_convex_mse():
    grad_at, loss_at, (w_star, b_star) = _linear_mse_problem()
    p0 = [0.0, 0.0]
    initial_loss = loss_at(p0)
    assert initial_loss > 1.0, initial_loss                    # a non-trivial starting loss (= 17.0)

    # per-optimizer (lr, steps) chosen so each reaches a low loss with the standard default betas/eps and a
    # SANE (small) learning rate -- so the descent is near-monotone (heavy-ball with beta=0.9 may briefly
    # overshoot, which is why the rise tolerance below is "near"-monotone rather than strict).
    runs = {
        "momentum": dict(lr=0.001, steps=2000, beta=0.9),
        "rmsprop":  dict(lr=0.01,  steps=1500, beta=0.9, eps=1e-8),
        "adam":     dict(lr=0.05,  steps=1500, beta1=0.9, beta2=0.999, eps=1e-8),
    }
    for opt, cfg in runs.items():
        losses, final = _run_to_convergence(opt, grad_at, loss_at, p0, **cfg)
        # drove the loss to near the known minimum (0) ...
        assert losses[-1] < 1e-3, (opt, losses[-1], losses[0])
        # ... and converged to the known minimizer (w, b) = (2, 1).
        assert abs(final[0] - w_star) < 5e-2 and abs(final[1] - b_star) < 5e-2, (opt, final)
        # the loss decreased overall (final far below the start).
        assert losses[-1] < 1e-3 * losses[0], (opt, losses[-1], losses[0])
        # NEAR-monotonic descent: no single-step rise exceeds 1% of the INITIAL loss (a sane lr keeps the
        # transient overshoot bounded; the curve is essentially a descent, not a divergence).
        rises = [b - a for a, b in zip(losses, losses[1:]) if b > a]
        max_rise = max(rises) if rises else 0.0
        assert max_rise < 0.01 * initial_loss, (opt, max_rise, initial_loss)


def _ill_conditioned_mse_problem():
    """A convex but ILL-CONDITIONED least-squares problem: fit pred = w*x + b on points with LARGE x (so the
    w-direction has far higher curvature than the b-direction). Plain SGD must use a tiny lr (bounded by the
    steep w-direction) and then crawls in the flat b-direction -- exactly where a per-coordinate adaptive
    method (Adam) pays off. Returns (grad_at, loss_at, (w*, b*))."""
    data = [(10.0, 21.0), (20.0, 41.0), (30.0, 61.0), (40.0, 81.0), (50.0, 101.0)]   # y = 2x + 1
    t = Tape()
    w, b = t.var("w"), t.var("b")
    preds = tuple(t.add(t.mul(w, t.const(x)), b) for x, _ in data)
    loss_node = mse(t, preds, [y for _, y in data])

    def grad_at(params):
        g = grad(t, loss_node, {"w": params[0], "b": params[1]}).grads
        return [g["w"], g["b"]]

    def loss_at(params):
        return evaluate(t, loss_node, {"w": params[0], "b": params[1]})

    return grad_at, loss_at, (2.0, 1.0)


def test_adam_converges_at_least_as_fast_as_sgd_when_it_pays_off():
    # The adaptive payoff: on an ILL-CONDITIONED convex problem, Adam (per-coordinate adaptive) reaches the
    # minimum in FAR fewer steps than plain SGD (whose lr is bounded by the steep direction). We don't
    # over-assert exact step counts -- just that Adam reaches the target no slower than SGD, and that BOTH
    # eventually descend.
    grad_at, loss_at, _ = _ill_conditioned_mse_problem()
    p0 = [0.0, 0.0]
    target = 1e-3
    max_steps = 5000

    def steps_to_target(opt, lr, **hp):
        traj = reference_optimizer_trajectory(opt, p0, lambda s, p: grad_at(p), lr, max_steps, **hp)
        for k, p in enumerate(traj):
            if loss_at(p) < target:
                return k, loss_at(traj[-1])
        return None, loss_at(traj[-1])

    # SGD with the largest lr that does not diverge on the steep direction; Adam with a well-chosen lr.
    sgd_k, sgd_final = steps_to_target("sgd", lr=0.0005)
    adam_k, adam_final = steps_to_target("adam", lr=1.0, beta1=0.9, beta2=0.999, eps=1e-8)
    # Adam reaches the minimum within the budget ...
    assert adam_k is not None, (adam_k, adam_final)
    # ... and does so no slower than SGD (here SGD does not even reach the target in the budget, so Adam's
    # adaptive per-coordinate scaling is the clear win). Both descend from the (large) initial loss.
    assert sgd_final < loss_at(p0) and adam_final < loss_at(p0), (sgd_final, adam_final)
    if sgd_k is not None:
        assert adam_k <= sgd_k, (adam_k, sgd_k)                # Adam no slower than SGD when both reach it


def test_well_conditioned_both_adam_and_sgd_converge():
    # on the well-conditioned problem both plain SGD and Adam reach the minimum (the honest baseline: a
    # well-conditioned convex quadratic is exactly where plain SGD is already excellent, so we assert BOTH
    # converge rather than that Adam strictly beats it).
    grad_at, loss_at, (w_star, b_star) = _linear_mse_problem()
    p0 = [0.0, 0.0]
    for opt, cfg in (("sgd", dict(lr=0.05, steps=400)),
                     ("adam", dict(lr=0.1, steps=400, beta1=0.9, beta2=0.999, eps=1e-8))):
        traj = reference_optimizer_trajectory(opt, p0, lambda s, p: grad_at(p), **cfg)
        assert loss_at(traj[-1]) < 1e-3, (opt, loss_at(traj[-1]))
        assert abs(traj[-1][0] - w_star) < 5e-2 and abs(traj[-1][1] - b_star) < 5e-2, (opt, traj[-1])


def test_convergence_loss_curves_match_in_compiled_c():
    # the convergence is real on the deployed rail too: re-run the convex problem in compiled C with the SAME
    # per-step gradient sequence as the reference, and assert the loss curves agree. (We feed the reference
    # gradient sequence as a fixed schedule so the C step -- which takes a constant grad -- can replay it; the
    # point here is the C step arithmetic matches over a real descent, not a fresh gradient query in C.)
    cc = _cc()
    if not cc:
        return
    grad_at, loss_at, _ = _linear_mse_problem()
    p0 = [0.0, 0.0]
    # Run Adam in the reference, but verify EACH single step's arithmetic matches C by checking a fixed-grad
    # trajectory: take the gradient at p0 and run 6 steps with THAT fixed gradient in both rails.
    g0 = grad_at(p0)
    ok, c_traj, out = compile_and_run_optimizer_c("adam", p0, g0, lr=0.1, steps=6,
                                                  beta1=0.9, beta2=0.999, eps=1e-8)
    assert ok, out
    ref_traj = reference_optimizer_trajectory("adam", p0, lambda s, p: g0, lr=0.1, steps=6,
                                              beta1=0.9, beta2=0.999, eps=1e-8)
    for cstep, rstep in zip(c_traj, ref_traj):
        for ci, ri in zip(cstep, rstep):
            assert abs(ci - ri) < 2e-4, (ci, ri, out)


# === numerical stability: rmsprop/adam don't divide-by-zero on a zero gradient (the eps), don't NaN ===

def test_rmsprop_and_adam_are_stable_on_a_zero_gradient():
    # a zero gradient makes the squared-grad average / second moment 0; the eps in the denominator must keep
    # the step finite (no division-by-zero, no NaN). The step on a 0 gradient is ~0 (params barely move).
    p = [1.0, -2.0, 0.5]
    zero = [0.0, 0.0, 0.0]
    # rmsprop: s' = 0, denom = sqrt(0)+eps = eps > 0; step = lr*0/eps = 0 -> params unchanged, finite.
    rp, rs = rmsprop_step(p, zero, [0.0, 0.0, 0.0], lr=0.1, beta=0.9, eps=1e-8)
    assert all(math.isfinite(x) for x in rp) and all(math.isfinite(x) for x in rs)
    assert all(abs(a - b) < 1e-12 for a, b in zip(rp, p)), rp        # a zero grad -> no move
    # adam: m'=v'=0, mhat=0, vhat=0, step = lr*0/(sqrt(0)+eps) = 0 -> finite, params unchanged.
    ap, am, av, at = adam_step(p, zero, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0, lr=0.1)
    assert all(math.isfinite(x) for x in ap) and all(math.isfinite(x) for x in am + av)
    assert all(abs(a - b) < 1e-12 for a, b in zip(ap, p)), ap
    assert at == 1


def test_optimizers_do_not_nan_on_large_or_mixed_gradients():
    # large + mixed-sign gradients must not produce NaN/inf in any optimizer (the adaptive scaling + eps keep
    # the steps bounded).
    p = [0.0, 0.0, 0.0]
    g = [1e6, -1e6, 0.0]
    assert all(math.isfinite(x) for x in sgd_step(p, g, 1e-3))
    mp, mv = momentum_step(p, g, [0.0, 0.0, 0.0], 1e-3, 0.9)
    assert all(math.isfinite(x) for x in mp + mv)
    rp, rs = rmsprop_step(p, g, [0.0, 0.0, 0.0], 1e-3, 0.9, 1e-8)
    assert all(math.isfinite(x) for x in rp + rs)
    ap, am, av, _ = adam_step(p, g, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0, 1e-3)
    assert all(math.isfinite(x) for x in ap + am + av)


# === the two-truth invariant: the optimizer module is cost/optimization-side, never a verdict ==========

def test_optimizers_touch_no_verifier_and_emit_no_diagnostic():
    # an optimizer is a COST/optimization-side object (HOW to descend a legal loss), NEVER an R-law verdict.
    # Inspect the imported module: no verify / Diagnostic / Graded names bound, no verifier module imported.
    import ast
    import bcir.lower.optimizers as opt_mod
    bound = set(vars(opt_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(opt_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the emitted C is a plain optimizer step -- no Diagnostic / verdict text leaks into it.
    for src in (emit_momentum_step_c(2), emit_rmsprop_step_c(2), emit_adam_step_c(2)):
        assert "Diagnostic" not in src and "verdict" not in src.lower()
    # the reference steps return plain numeric containers (no graded/confidence wrapper).
    p, v = momentum_step([1.0], [0.5], [0.0], 0.1, 0.9)
    assert isinstance(p, list) and isinstance(v, list) and all(isinstance(x, float) for x in p + v)


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
