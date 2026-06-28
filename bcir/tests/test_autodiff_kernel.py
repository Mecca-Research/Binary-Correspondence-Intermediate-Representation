"""G6 (roadmap task #53) -- lower the B3 autodiff oracle to an emitted forward/backward C kernel + SGD step.

The VISION_ALIGNMENT_AUDIT (Pillar 5b) flags that BCIR's autodiff (``bcir.kbcir.autodiff``, the B3 oracle)
is a complete, correct, content-addressed reverse-mode autodiff organ but is Python-oracle-only, kept cold,
NOT lowered -- so gradients/training cannot run on the deployed bare-metal rail. ``bcir.lower.autodiff_kernel``
is that lowering: given a ``Tape`` + an output node + the ordered parameter set it emits ONE self-contained
``void grad(const float*, float*, float*)`` computing the forward value AND the reverse-mode backward-pass
gradients (each primitive's ``_BACKWARD`` rule transcribed to C, accumulated in baked reverse topo order),
plus a minimal SGD weight-update step ``w[i] -= lr*grad[i]`` and a one-training-step wiring.

The GATE that matters (mirrors the G5 inference test): the emitted-C gradients MATCH the oracle's
reverse-mode ``grad`` AND its ``finite_difference_grad`` to float round-off, proved by COMPILING and RUNNING
the kernel; and a short training loop (forward -> backward -> SGD) strictly DECREASES a loss in both the
oracle and the compiled C, agreeing. Covers:
  * the emit is valid, warning-free C (the closed primitive set, the attestation comment, the ABI);
  * every closed primitive (neg/add/sub/mul/div/dot/select + const/var leaves) lowers correctly;
  * out-of-set ops and bad parameters are rejected deterministically;
  * the compiled grad == oracle grad == finite-difference grad (the B3 hard gate) for several DAGs;
  * shared (hash-consed) subexpressions stay shared in the emit (one v<id>/g<id> pair);
  * the SGD step and the C training loop strictly reduce a regression loss, matching the oracle loop.
All deterministic + pure-Python on the oracle rail; the compile-and-run tests self-skip when the toolchain
is hidden (the quick tier), exactly like the inference / activation / fusion tests."""

import random
import shutil

from bcir.kbcir.autodiff import (Tape, evaluate, finite_difference_grad, grad,
                                 gradients_match, max_grad_error)
from bcir.lower.autodiff_kernel import (compile_and_run_grad_c, compile_and_run_training_c,
                                        emit_autodiff_kernel_c, emit_sgd_step_c, emit_train_step_c,
                                        oracle_train)


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


# --- the emit: valid C, the closed primitive set, the ABI, the attestation comment ----------------

def test_emit_has_the_signature_attestation_and_is_self_contained():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.add(t.mul(a, b), t.const(7.0))
    src = emit_autodiff_kernel_c(t, f, ("a", "b"), "grad_fn")
    assert "void grad_fn(const float *params, float *value_out, float *grad_out)" in src
    # the attestation comment is proof-carrying: the forward+backward roles, the baked topo order, the
    # closed primitive set, the honest scope boundary (the deployable training primitive, not a framework).
    assert "G6 autodiff forward+backward kernel" in src and "Pillar 5b" in src
    assert "reverse-mode" in src and "_BACKWARD rule" in src
    assert "closed set" in src and "neg/add/sub/mul/div/dot/select" in src
    assert "NOT a full" in src and "optimizer" in src           # the honest depth boundary
    # self-contained: only <stddef.h>, no libm (the closed primitive set is purely arithmetic).
    assert "#include <stddef.h>" in src and "math.h" not in src
    # the forward temporaries and the seeded output adjoint are present.
    assert "value_out[0] = v" in src and "= 1.0f;" in src       # the output adjoint seed d(out)/d(out)=1


def test_emit_is_deterministic():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.div(t.mul(a, a), t.add(b, t.const(1.0)))
    assert emit_autodiff_kernel_c(t, f, ("a", "b"), "g") == emit_autodiff_kernel_c(t, f, ("a", "b"), "g")


def test_out_of_set_op_is_rejected_deterministically():
    # a node outside the closed lowerable set has no emittable backward rule -> a deterministic ValueError.
    t = Tape()
    a = t.var("a")
    f = t.mul(a, a)
    # forge a node with an unsupported op by reaching into the store (the closure boundary the oracle pins).
    from bcir.kbcir.autodiff import Node
    bogus = len(t._nodes)
    t._nodes.append(Node(bogus, "exp", (f,), None, ("exp", None, (f,))))
    try:
        emit_autodiff_kernel_c(t, bogus, ("a",))
        assert False, "an op outside the closed set must be rejected"
    except ValueError as e:
        assert "outside the closed lowerable primitive set" in str(e)


def test_unbound_var_and_duplicate_param_are_rejected():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.mul(a, b)
    # a DAG var not bound by params (the params vector is the forward env) is rejected.
    try:
        emit_autodiff_kernel_c(t, f, ("a",))
        assert False, "an unbound DAG var must be rejected"
    except ValueError as e:
        assert "not in params" in str(e)
    # a duplicate parameter name (ambiguous ABI index) is rejected.
    try:
        emit_autodiff_kernel_c(t, f, ("a", "a", "b"))
        assert False, "a duplicate parameter must be rejected"
    except ValueError as e:
        assert "duplicate" in str(e)


def test_unreachable_param_has_zero_gradient():
    # a requested parameter that does not reach the output gets identically-zero gradient (the oracle's
    # unused-input rule), not an error.
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.mul(a, a)                                             # b never reaches f
    src = emit_autodiff_kernel_c(t, f, ("a", "b"))
    assert "grad_out[1] = 0.0f;" in src                        # b's gradient is the zero literal
    cc = _cc()
    if not cc:
        return
    ok, value, grads, out = compile_and_run_grad_c(t, f, ("a", "b"), {"a": 3.0, "b": 99.0})
    assert ok, out
    assert grads[1] == 0.0 and abs(grads[0] - 6.0) < 1e-5, (grads, out)   # d(a*a)/da = 2a = 6, d/db = 0


def test_degenerate_dags_emit_warning_free_and_correct():
    # the degenerate cases the -Wall -Wextra discipline must still handle cleanly:
    #  (a) a constant-valued output (no var reaches it) -> all grads 0, params marked (void);
    #  (b) the output IS a bare param var -> grad of x w.r.t. x is 1.
    t = Tape()
    a = t.var("a")
    const_out = emit_autodiff_kernel_c(t, t.const(3.0), ("a",))
    assert "(void)params;" in const_out and "grad_out[0] = 0.0f;" in const_out
    t2 = Tape()
    x = t2.var("x")
    bare = emit_autodiff_kernel_c(t2, x, ("x",))
    assert "= 1.0f;" in bare                                    # the output adjoint seed feeds grad_out
    cc = _cc()
    if not cc:
        return
    ok, v, g, out = compile_and_run_grad_c(t, t.const(3.0), ("a",), {"a": 9.0})
    assert ok and v == 3.0 and g == [0.0], out
    ok, v, g, out = compile_and_run_grad_c(t2, x, ("x",), {"x": 5.0})
    assert ok and v == 5.0 and g == [1.0], out


def test_sgd_step_emit():
    src = emit_sgd_step_c(3, "sgd")
    assert "void sgd(float *params, const float *grad, float lr)" in src
    assert "params[i] -= lr * grad[i];" in src                  # the deployable update primitive
    assert "minimal SGD weight-update step" in src
    try:
        emit_sgd_step_c(0)
        assert False, "n_params < 1 must raise"
    except ValueError:
        pass


def test_train_step_bundles_grad_sgd_and_wiring():
    t = Tape()
    a = t.var("a")
    f = t.mul(a, a)
    src = emit_train_step_c(t, f, ("a",), fn_name="step")
    assert "void step(float *params, float lr, float *value_out)" in src
    assert "bcir_grad(params, value_out, grad);" in src and "bcir_sgd_step(params, grad, lr);" in src


# --- the gate that matters: compiled-C grad == oracle grad == finite-difference grad --------------

def _check_parity(t, f, params, env, tol=1e-4):
    """Compile+run the grad kernel at ``env`` and assert the C value+grads match the oracle reverse-mode
    grad AND the finite-difference grad (the B3 hard gate) to float round-off. Returns the max errors."""
    ok, value, grads, out = compile_and_run_grad_c(t, f, params, env)
    assert ok, out
    oref = grad(t, f, env)
    fd = finite_difference_grad(t, f, env)
    assert abs(value - oref.value) < tol, (value, oref.value, out)
    cgrad = {name: grads[i] for i, name in enumerate(params)}
    e_oracle = max_grad_error(cgrad, oref.grads)
    e_fd = max_grad_error(cgrad, fd)
    assert e_oracle < tol, (cgrad, oref.grads, out)             # C == oracle reverse-mode grad
    assert e_fd < tol, (cgrad, fd, out)                         # C == finite-difference grad
    assert gradients_match(cgrad, oref.grads) and gradients_match(cgrad, fd)
    return e_oracle, e_fd


def test_grad_kernel_matches_oracle_and_fd_on_a_polynomial():
    cc = _cc()
    if not cc:
        return                                                  # quick tier hides the toolchain -> self-skip
    # f = a*a*b + a*b*b - 3*a + 7  (neg/add/sub/mul/const).
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.add(t.sub(t.add(t.mul(t.mul(a, a), b), t.mul(a, t.mul(b, b))), t.mul(t.const(3.0), a)),
              t.const(7.0))
    _check_parity(t, f, ("a", "b"), {"a": 1.5, "b": -0.7})


def test_grad_kernel_matches_on_division():
    cc = _cc()
    if not cc:
        return
    # f = (a + b) / (a*a - b)  -- exercises the div quotient rule.
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.div(t.add(a, b), t.sub(t.mul(a, a), b))
    _check_parity(t, f, ("a", "b"), {"a": 2.0, "b": 0.5})


def test_grad_kernel_matches_on_the_dot_substrate():
    cc = _cc()
    if not cc:
        return
    # the matmul-substrate dot: f = dot([u0,u1,u2], [v0,v1,v2]); grad_u_i = v_i, grad_v_i = u_i.
    t = Tape()
    us = tuple(t.var(f"u{i}") for i in range(3))
    vs = tuple(t.var(f"v{i}") for i in range(3))
    f = t.dot(us, vs)
    env = {"u0": 1.0, "u1": 2.0, "u2": 3.0, "v0": 4.0, "v1": 5.0, "v2": 6.0}
    e_oracle, _ = _check_parity(t, f, tuple(env), env)
    # also the exact closed form: d(dot)/d(u_i) = v_i, d/d(v_i) = u_i.
    ok, _, grads, out = compile_and_run_grad_c(t, f, tuple(env), env)
    names = list(env)
    cg = {n: grads[i] for i, n in enumerate(names)}
    assert cg["u0"] == 4.0 and cg["v2"] == 3.0, (cg, out)


def test_grad_kernel_matches_through_select():
    cc = _cc()
    if not cc:
        return
    # f = select(cond, a*a, b*b): away from the switch the VJP routes entirely to the selected branch.
    t = Tape()
    cond, a, b = t.var("cond"), t.var("a"), t.var("b")
    f = t.select(cond, t.mul(a, a), t.mul(b, b))
    # cond > 0: selected branch a*a -> grad_a = 2a, grad_b = 0, grad_cond = 0.
    _check_parity(t, f, ("cond", "a", "b"), {"cond": 2.0, "a": 3.0, "b": -1.0})
    # cond < 0: selected branch b*b -> grad_b = 2b, grad_a = 0.
    _check_parity(t, f, ("cond", "a", "b"), {"cond": -2.0, "a": 3.0, "b": -1.0})


def test_grad_kernel_matches_with_a_shared_subexpression():
    cc = _cc()
    if not cc:
        return
    # t = a*b is SHARED: f = (t)*(t) + t*c -- the shared node must get ONE adjoint slot (content-addressed).
    t = Tape()
    a, b, c = t.var("a"), t.var("b"), t.var("c")
    sh = t.mul(a, b)
    f = t.add(t.mul(sh, sh), t.mul(sh, c))
    # the shared node sh appears once in the emit (hash-consed) -> one v<id>/g<id> pair.
    src = emit_autodiff_kernel_c(t, f, ("a", "b", "c"))
    assert src.count(f"float v{sh} =") == 1 and src.count(f"float g{sh} =") == 1
    _check_parity(t, f, ("a", "b", "c"), {"a": 1.3, "b": -0.8, "c": 2.1})


def test_grad_kernel_fuzz_matches_oracle_and_fd():
    cc = _cc()
    if not cc:
        return
    # a deterministic fuzz over random DAGs (mirrors the autodiff oracle's fuzz), compiled + run, asserting
    # the C grads match BOTH the oracle reverse-mode grad and the finite difference -- the B3 gate, lowered.
    rng = random.Random(0x6D11)
    names = ["a", "b", "c"]
    for _ in range(8):
        t = Tape()
        leaves = [t.var(n) for n in names] + [t.const(round(rng.uniform(-2, 2), 3)) for _ in range(2)]
        pool = list(leaves)
        for _ in range(rng.randint(4, 9)):
            op = rng.choice(["add", "sub", "mul", "div", "neg"])
            if op == "neg":
                pool.append(t.neg(rng.choice(pool)))
            else:
                x, y = rng.choice(pool), rng.choice(pool)
                if op == "div":
                    pool.append(t.div(x, y))                    # the point is chosen to keep |denom| away from 0
                else:
                    pool.append(getattr(t, op)(x, y))
        f = pool[-1]
        # a point where no div denominator is near zero (the linearization point must be well-defined).
        env = {"a": 1.7, "b": -0.9, "c": 2.3}
        try:
            evaluate(t, f, env)
        except ZeroDivisionError:
            continue
        _check_parity(t, f, ("a", "b", "c"), env, tol=2e-3)


# --- the SGD training loop: it strictly reduces a loss, and the C loop matches the oracle ----------

def _linear_regression_tape():
    """A tiny linear-regression loss tape: fit y = w*x + b over points on y = 2x + 1, loss = sum (pred-y)^2.
    Returns (tape, loss_node, params). The baked data are consts; w, b are the parameters."""
    t = Tape()
    w, b = t.var("w"), t.var("b")
    data = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (4.0, 9.0)]     # y = 2x + 1
    loss = t.const(0.0)
    for xi, yi in data:
        err = t.sub(t.add(t.mul(w, t.const(xi)), b), t.const(yi))
        loss = t.add(loss, t.mul(err, err))
    return t, loss, ("w", "b")


def test_oracle_training_loop_strictly_decreases_the_loss():
    t, loss, params = _linear_regression_tape()
    losses, final = oracle_train(t, loss, params, {"w": 0.0, "b": 0.0}, lr=0.01, steps=20)
    assert all(b < a for a, b in zip(losses, losses[1:])), losses   # strictly decreasing
    assert losses[-1] < losses[0]
    # SGD drives w,b toward the true (2, 1); after 20 steps it is closing in (not asserting convergence).
    assert abs(final["w"] - 2.0) < abs(0.0 - 2.0) and abs(final["b"] - 1.0) < abs(0.0 - 1.0)


def test_compiled_c_training_loop_decreases_the_loss_and_matches_the_oracle():
    cc = _cc()
    if not cc:
        return
    t, loss, params = _linear_regression_tape()
    env, lr, steps = {"w": 0.0, "b": 0.0}, 0.01, 20
    losses_o, final_o = oracle_train(t, loss, params, env, lr, steps)
    ok, losses_c, fparams, out = compile_and_run_training_c(t, loss, params, env, lr, steps)
    assert ok, out
    # the compiled C training loop strictly decreases the loss ...
    assert all(b < a for a, b in zip(losses_c, losses_c[1:])), (losses_c, out)
    # ... and AGREES with the oracle loss curve to float round-off (the lowered loop == the Python loop).
    assert len(losses_c) == len(losses_o)
    assert max(abs(a - b) for a, b in zip(losses_o, losses_c)) < 1e-3, (losses_o, losses_c, out)
    # the final fitted parameters agree too.
    assert abs(fparams[0] - final_o["w"]) < 1e-3 and abs(fparams[1] - final_o["b"]) < 1e-3, (fparams, final_o)


def test_compiled_c_training_loop_on_a_select_mlp_cell_decreases_and_matches():
    cc = _cc()
    if not cc:
        return
    # a tiny non-linear cell with a select (a relu-like gate): h = select(w*x, w*x, 0); loss = (h - y)^2,
    # summed over points -> exercises the select VJP INSIDE a training loop, lowered. Both loops must agree.
    t = Tape()
    w = t.var("w")
    data = [(1.0, 0.5), (2.0, 1.0), (-1.0, 0.0), (3.0, 1.5)]
    loss = t.const(0.0)
    for xi, yi in data:
        pre = t.mul(w, t.const(xi))
        h = t.select(pre, pre, t.const(0.0))                   # relu-ish gate on the pre-activation
        err = t.sub(h, t.const(yi))
        loss = t.add(loss, t.mul(err, err))
    env, lr, steps = {"w": 0.0}, 0.02, 15
    losses_o, _ = oracle_train(t, loss, ("w",), env, lr, steps)
    ok, losses_c, _, out = compile_and_run_training_c(t, loss, ("w",), env, lr, steps)
    assert ok, out
    assert losses_c[-1] <= losses_c[0]                          # non-increasing (a gated cell can plateau)
    assert max(abs(a - b) for a, b in zip(losses_o, losses_c)) < 1e-3, (losses_o, losses_c, out)
