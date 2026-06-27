"""B3 -- reverse-mode AD as content-addressed local graph rewrites: the correctness gate.

Covers, with the finite-difference gradient as the HARD correctness gate:
  * forward evaluation of the differentiable DAG;
  * reverse-mode grad == central-difference finite-difference on several functions,
    including a SHARED-subexpression function and the matmul-substrate ``dot``;
  * reverse-mode grad == the EXACT symbolic gradient on closed-form cases;
  * content-addressing dedup is MEASURABLE (unique-node count < naive-tree count, and
    a forward-shared node yields a single shared adjoint);
  * confluence / determinism: two different valid adjoint-accumulation orders give
    identical gradients, and a re-run replays byte-identically;
  * the cheap-gradient principle (backward op count ~ a small constant x forward);
  * edges: a const has zero grad, an unused input has zero grad;
  * a deterministic FUZZ over random expression DAGs asserting grad always matches
    finite-difference within tolerance.

All deterministic, pure-Python. This module is a research organ (kept cold)."""

import random

from bcir.kbcir.autodiff import (
    GradResult,
    Tape,
    evaluate,
    finite_difference_grad,
    grad,
    grad_at,
    gradients_match,
    max_grad_error,
    naive_tree_node_count,
    reverse_orders,
)


# --- forward evaluation ----------------------------------------------------------

def test_forward_eval_matches_the_closed_form():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    expr = t.add(t.mul(a, b), t.const(7.0))           # a*b + 7
    env = {"a": 3.0, "b": 5.0}
    assert evaluate(t, expr, env) == 3.0 * 5.0 + 7.0


def test_forward_eval_of_dot_is_sum_of_products():
    t = Tape()
    u = tuple(t.var(f"u{i}") for i in range(3))
    v = tuple(t.var(f"v{i}") for i in range(3))
    dp = t.dot(u, v)
    env = {"u0": 1.0, "u1": 2.0, "u2": 3.0, "v0": 4.0, "v1": 5.0, "v2": 6.0}
    assert evaluate(t, dp, env) == 1 * 4 + 2 * 5 + 3 * 6


def test_unbound_var_is_a_clear_error():
    t = Tape()
    try:
        evaluate(t, t.var("x"), {})
    except KeyError:
        return
    assert False, "expected KeyError for an unbound var"


# --- the hard gate: reverse grad == finite difference ----------------------------

def _check_fd(t: Tape, out: int, env: dict, *, rtol=1e-4, atol=1e-5):
    """Assert analytic grad matches central-difference at ``env``; report the exact
    case on failure (never weaken the tolerance to force a pass)."""
    g = grad(t, out, env)
    fd = finite_difference_grad(t, out, env)
    assert gradients_match(g.grads, fd, rtol=rtol, atol=atol), (
        f"grad != finite-difference: analytic={g.grads} fd={fd} "
        f"max_abs_err={max_grad_error(g.grads, fd)} env={env}")
    return g


def test_grad_matches_fd_on_a_polynomial():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    # f = a*a*b - b + 3a  (mixes mul/sub/add/const)
    f = t.add(t.sub(t.mul(t.mul(a, a), b), b), t.mul(t.const(3.0), a))
    _check_fd(t, f, {"a": 1.7, "b": -2.3})


def test_grad_matches_fd_with_a_shared_subexpression():
    # t = a*b shared by two users -> ONE forward node, ONE shared adjoint.
    t = Tape()
    a, b, c, d = (t.var(n) for n in "abcd")
    tt = t.mul(a, b)
    out = t.add(t.mul(tt, c), t.mul(tt, d))            # (a*b)*c + (a*b)*d
    g = _check_fd(t, out, {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0})
    # closed form: out = a*b*(c+d); checks the shared node's adjoint summed correctly.
    assert abs(g.grads["a"] - 3.0 * (4.0 + 5.0)) < 1e-9
    assert abs(g.grads["b"] - 2.0 * (4.0 + 5.0)) < 1e-9


def test_grad_matches_fd_on_the_dot_substrate():
    t = Tape()
    u = tuple(t.var(f"u{i}") for i in range(4))
    v = tuple(t.var(f"v{i}") for i in range(4))
    dp = t.dot(u, v)
    env = {f"u{i}": float(i + 1) for i in range(4)}
    env.update({f"v{i}": float(2 * i - 1) for i in range(4)})
    g = _check_fd(t, dp, env)
    # exact: d(dot)/d(u_i) = v_i and d(dot)/d(v_i) = u_i.
    for i in range(4):
        assert g.grads[f"u{i}"] == env[f"v{i}"]
        assert g.grads[f"v{i}"] == env[f"u{i}"]


def test_grad_matches_fd_on_division():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    q = t.div(a, b)                                   # a / b
    _check_fd(t, q, {"a": 3.0, "b": 4.0})
    # nested: (a-b) / (a*b)
    expr = t.div(t.sub(a, b), t.mul(a, b))
    _check_fd(t, expr, {"a": 1.5, "b": 2.5})


def test_grad_matches_fd_on_neg_and_chains():
    t = Tape()
    a = t.var("a")
    # f = -(-(a*a)) - this exercises neg twice and a deep chain.
    f = t.neg(t.neg(t.mul(a, a)))
    _check_fd(t, f, {"a": 2.5})


# --- reverse grad == exact symbolic gradient -------------------------------------

def test_grad_equals_exact_symbolic_product_rule():
    # d/da (a*b) = b, d/db (a*b) = a -- exact, no tolerance needed.
    t = Tape()
    a, b = t.var("a"), t.var("b")
    g = grad(t, t.mul(a, b), {"a": 6.0, "b": -2.0})
    assert g.grads == {"a": -2.0, "b": 6.0}


def test_grad_equals_exact_symbolic_for_a_quadratic():
    # f = 3*a*a + 2*a ; f'(a) = 6a + 2. At a=4 -> 26 (exact for integer-valued floats).
    t = Tape()
    a = t.var("a")
    f = t.add(t.mul(t.const(3.0), t.mul(a, a)), t.mul(t.const(2.0), a))
    g = grad(t, f, {"a": 4.0})
    assert g.grads["a"] == 6.0 * 4.0 + 2.0
    assert g.value == 3.0 * 16.0 + 2.0 * 4.0


def test_value_returned_by_grad_equals_forward_eval():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.sub(t.mul(a, b), t.div(a, b))
    env = {"a": 5.0, "b": 2.0}
    g = grad(t, f, env)
    assert g.value == evaluate(t, f, env)


# --- content-addressing dedup is measurable --------------------------------------

def test_hashcons_shares_structurally_identical_nodes():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    n1 = t.mul(a, b)
    n2 = t.mul(a, b)                                  # same structure -> same node id
    assert n1 == n2
    # a different structure is a different node.
    assert t.mul(b, a) != n1


def test_dedup_makes_the_dag_smaller_than_a_naive_tree():
    t = Tape()
    a, b, c, d = (t.var(n) for n in "abcd")
    tt = t.mul(a, b)
    out = t.add(t.mul(tt, c), t.mul(tt, d))
    # naive tree re-expands the shared t=a*b; the hash-consed DAG holds it once.
    assert t.unique_node_count < naive_tree_node_count(t, out)
    # concretely: vars a,b,c,d (4) + t (1) + t*c, t*d (2) + add (1) = 8 unique nodes;
    # a naive tree would expand a*b twice -> 11 nodes.
    assert t.unique_node_count == 8
    assert naive_tree_node_count(t, out) == 11


def test_shared_node_yields_a_single_shared_adjoint():
    # The shared t=a*b must receive ONE accumulated adjoint and its rule fire once.
    # We verify via the op-count: 4 mul + 1 add = 5 forward rule-bearing ops, and the
    # backward pass fires each reachable-with-nonzero-adjoint rule exactly once.
    t = Tape()
    a, b, c, d = (t.var(n) for n in "abcd")
    tt = t.mul(a, b)
    out = t.add(t.mul(tt, c), t.mul(tt, d))
    g = grad(t, out, {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0})
    # forward rule-bearing ops: add(1) + mul(t*c) + mul(t*d) + mul(a*b)=t + ... t is shared
    # so it is ONE node. unique rule-bearing forward ops = add, mul(tc), mul(td), mul(ab) = 4.
    assert g.forward_ops == 4
    # backward fires once per node reached with a nonzero adjoint: add, mul(tc), mul(td),
    # and the SHARED mul(ab) exactly once (not twice) -> 4 firings.
    assert g.backward_ops == 4


# --- confluence / determinism + replay -------------------------------------------

def test_confluence_two_orders_give_identical_gradients():
    t = Tape()
    a, b, c, d = (t.var(n) for n in "abcd")
    tt = t.mul(a, b)
    out = t.add(t.mul(tt, c), t.sub(t.mul(tt, d), t.mul(a, c)))
    env = {"a": 1.3, "b": 2.1, "c": -0.7, "d": 3.3}
    order_a, order_b = reverse_orders(t, out)
    assert order_a != order_b                         # genuinely different interior order
    g_a = grad(t, out, env, order=order_a)
    g_b = grad(t, out, env, order=order_b)
    assert g_a.grads == g_b.grads                     # the diamond property: order-independent
    # and both match finite-difference (the gate still holds under either order).
    fd = finite_difference_grad(t, out, env)
    assert gradients_match(g_a.grads, fd)


def test_replay_is_byte_identical():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    out = t.div(t.mul(a, b), t.add(a, b))
    env = {"a": 3.0, "b": 7.0}
    r1 = grad(t, out, env).grads
    r2 = grad(t, out, env).grads
    assert repr(r1) == repr(r2)                       # byte-identical replay
    assert r1 == r2


def test_grad_at_is_an_alias_for_grad():
    t = Tape()
    a, b = t.var("a"), t.var("b")
    out = t.mul(a, b)
    env = {"a": 2.0, "b": 9.0}
    assert grad_at(t, out, env).grads == grad(t, out, env).grads


# --- the cheap-gradient principle ------------------------------------------------

def test_cheap_gradient_ratio_is_a_small_constant():
    # Baur-Strassen: a full gradient costs O(1)x the forward op count. Build a chain and
    # check the backward/forward ratio stays bounded (does not grow with depth).
    ratios = []
    for depth in (4, 8, 16, 32):
        t = Tape()
        x = t.var("x")
        node = x
        for _ in range(depth):
            node = t.mul(node, x)                     # x^(depth+1)
        g = grad(t, node, {"x": 1.1})
        ratios.append(g.cheap_gradient_ratio)
    # every gradient touches at most a small constant multiple of the forward ops.
    assert all(r <= 2.0 for r in ratios), ratios


# --- edges -----------------------------------------------------------------------

def test_const_has_zero_gradient():
    t = Tape()
    a = t.var("a")
    out = t.add(a, t.const(5.0))                      # d/d(const) is not a var; var a -> 1
    g = grad(t, out, {"a": 2.0})
    assert g.grads["a"] == 1.0
    # a pure constant output: gradient of any var is zero.
    c = t.const(42.0)
    gc = grad(t, c, {"a": 2.0})
    assert gc.grads["a"] == 0.0


def test_unused_input_has_zero_gradient():
    t = Tape()
    a, b, unused = t.var("a"), t.var("b"), t.var("unused")
    out = t.add(a, b)                                 # 'unused' never reaches the output
    g = grad(t, out, {"a": 1.0, "b": 2.0, "unused": 99.0})
    assert g.grads["unused"] == 0.0
    assert g.grads["a"] == 1.0 and g.grads["b"] == 1.0


# --- the deterministic fuzz: grad == finite difference over random DAGs -----------

def _random_dag(rng: random.Random, n_vars: int, n_ops: int):
    """Build a random differentiable DAG. Returns (tape, output, var_names). Operands
    are drawn from EXISTING nodes, so subexpressions get shared (exercising the
    content-addressed adjoint path). Division guards against near-zero denominators so
    the finite-difference comparison stays well-conditioned (the gate must be a fair
    test of the math, not of float catastrophe)."""
    t = Tape()
    names = [f"v{i}" for i in range(n_vars)]
    pool = [t.var(nm) for nm in names]
    pool.append(t.const(rng.uniform(-2.0, 2.0)))
    env = {nm: rng.uniform(-1.5, 1.5) for nm in names}
    for _ in range(n_ops):
        op = rng.choice(["add", "sub", "mul", "neg", "div"])
        if op == "neg":
            pool.append(t.neg(rng.choice(pool)))
        elif op == "div":
            num = rng.choice(pool)
            # pick a denominator whose value is comfortably away from zero at env.
            den = rng.choice(pool)
            if abs(evaluate(t, den, env)) < 0.4:
                den = t.const(rng.choice([-1.0, 1.0]) * rng.uniform(0.8, 2.0))
            pool.append(t.div(num, den))
        else:
            a, b = rng.choice(pool), rng.choice(pool)
            pool.append({"add": t.add, "sub": t.sub, "mul": t.mul}[op](a, b))
    # occasionally fold a dot in over the vars (the substrate primitive).
    if n_vars >= 2 and rng.random() < 0.5:
        half = n_vars // 2
        us = tuple(t.var(names[i]) for i in range(half))
        vs = tuple(t.var(names[i]) for i in range(half, 2 * half))
        pool.append(t.add(pool[-1], t.dot(us, vs)))
    return t, pool[-1], env


def test_fuzz_grad_matches_finite_difference_over_random_dags():
    rng = random.Random(0xB3_AD)
    for trial in range(400):
        n_vars = rng.randint(2, 5)
        n_ops = rng.randint(3, 14)
        t, out, env = _random_dag(rng, n_vars, n_ops)
        g = grad(t, out, env)
        fd = finite_difference_grad(t, out, env)
        # tolerance is generous enough for central-difference truncation but tight
        # enough to catch a wrong rule; report the exact failing trial if it trips.
        assert gradients_match(g.grads, fd, rtol=2e-3, atol=2e-4), (
            f"FUZZ trial {trial} FAILED: analytic={g.grads} fd={fd} "
            f"max_abs_err={max_grad_error(g.grads, fd)} env={env} n_ops={n_ops}")


def test_fuzz_is_deterministic_across_runs():
    # the whole fuzz, replayed, must produce byte-identical gradients (determinism).
    def run():
        rng = random.Random(0xC0FFEE)
        out = []
        for _ in range(50):
            t, root, env = _random_dag(rng, rng.randint(2, 4), rng.randint(2, 8))
            out.append(repr(grad(t, root, env).grads))
        return out
    assert run() == run()


def test_grad_result_shape():
    t = Tape()
    a = t.var("a")
    g = grad(t, t.mul(a, a), {"a": 3.0})
    assert isinstance(g, GradResult)
    assert g.value == 9.0 and g.grads == {"a": 6.0}
    assert g.forward_ops >= 1 and g.backward_ops >= 1
