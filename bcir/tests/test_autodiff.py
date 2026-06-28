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
    grad_graph,
    gradients_match,
    hessian,
    hessians_match,
    max_grad_error,
    max_hessian_error,
    naive_tree_node_count,
    reverse_orders,
    second_difference_hessian,
    unroll_scan,
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


# --- control flow: the differentiable conditional `select` -----------------------

def test_select_forward_picks_the_right_branch():
    # predicate is the SIGN TEST of cond's forward value (JAX lax.select / C ?:).
    t = Tape()
    cond, a, b = t.var("cond"), t.var("a"), t.var("b")
    s = t.select(cond, a, b)
    assert evaluate(t, s, {"cond": 1.0, "a": 5.0, "b": 9.0}) == 5.0     # cond > 0 -> a
    assert evaluate(t, s, {"cond": -1.0, "a": 5.0, "b": 9.0}) == 9.0    # cond <= 0 -> b
    assert evaluate(t, s, {"cond": 0.0, "a": 5.0, "b": 9.0}) == 9.0     # boundary: 0 is not > 0 -> b


def test_grad_through_select_matches_fd_away_from_boundary():
    # AWAY from the switch (cond value comfortably != 0) select is locally just one branch,
    # so its gradient is finite-difference-checkable like any smooth function.
    t = Tape()
    cond, a, b = t.var("cond"), t.var("a"), t.var("b")
    s = t.select(cond, t.mul(a, a), t.mul(b, b))     # a*a if cond>0 else b*b
    # cond = +2.0 (well away from 0): selected branch is a*a -> d/da = 2a, d/db = 0, d/dcond = 0.
    _check_fd(t, s, {"cond": 2.0, "a": 1.3, "b": -0.7})
    # cond = -2.0: selected branch is b*b -> d/db = 2b, d/da = 0.
    _check_fd(t, s, {"cond": -2.0, "a": 1.3, "b": -0.7})


def test_select_non_selected_branch_gets_zero_gradient():
    # The a.e. VJP routes the whole adjoint to the SELECTED branch and 0 to the other
    # branch and to cond (the predicate is piecewise-constant; the boundary delta is dropped).
    t = Tape()
    cond, a, b = t.var("cond"), t.var("a"), t.var("b")
    s = t.select(cond, a, b)
    g = grad(t, s, {"cond": 3.0, "a": 5.0, "b": 9.0})   # picks a
    assert g.grads["a"] == 1.0       # selected branch: full adjoint
    assert g.grads["b"] == 0.0       # other branch: zero
    assert g.grads["cond"] == 0.0    # predicate: zero (a.e.)
    g2 = grad(t, s, {"cond": -3.0, "a": 5.0, "b": 9.0})  # picks b
    assert g2.grads["b"] == 1.0 and g2.grads["a"] == 0.0 and g2.grads["cond"] == 0.0


def test_grad_when_differentiated_var_is_inside_the_selected_branch():
    # The differentiated var lives INSIDE the selected branch (a non-trivial expression),
    # so the chain rule must flow through the branch -- checked against finite difference.
    t = Tape()
    cond, x = t.var("cond"), t.var("x")
    # f = select(cond, (x*x*x), 0) ; cond>0 picks the cubic in x.
    cubic = t.mul(t.mul(x, x), x)
    s = t.select(cond, cubic, t.const(0.0))
    g = _check_fd(t, s, {"cond": 1.5, "x": 2.0})
    assert abs(g.grads["x"] - 3.0 * 2.0 * 2.0) < 1e-6   # d/dx x^3 = 3x^2 = 12


# --- bounded control flow: unroll_scan demonstrates BPTT -------------------------

def test_unroll_scan_recurrence_grad_matches_fd():
    # A bounded recurrence carry_{k+1} = carry_k * w + x_k over a few steps (a linear RNN
    # cell). Unrolled into plain primitives, the existing reverse pass differentiates it ->
    # backprop-through-time, checked against finite difference.
    t = Tape()
    w = t.var("w")
    xs = tuple(t.var(f"x{i}") for i in range(4))

    def step(tape, carry, x):
        return tape.add(tape.mul(carry, w), x)       # carry*w + x

    final = unroll_scan(t, step, t.const(0.0), xs)
    env = {"w": 0.6, "x0": 1.0, "x1": -2.0, "x2": 0.5, "x3": 3.0}
    g = _check_fd(t, final, env)
    # closed form: final = sum_k x_k * w^(n-1-k); d/dx_k = w^(n-1-k). Spot-check the last input.
    assert abs(g.grads["x3"] - 1.0) < 1e-6           # x3 has exponent 0 -> w^0 = 1


def test_unroll_scan_shares_structure_across_steps():
    # A step body whose SUBEXPRESSION is structurally identical across iterations is
    # hash-consed to one shared node (content-addressing across the unroll). Here the
    # constant gain and the var w are reused every step, so the unrolled DAG is far smaller
    # than n independent copies.
    t = Tape()
    w = t.var("w")
    xs = tuple(t.var(f"x{i}") for i in range(5))

    def step(tape, carry, x):
        # carry*w + x : 'w' (one var node) and the mul/add ops reuse interned operands.
        return tape.add(tape.mul(carry, w), x)

    final = unroll_scan(t, step, t.const(0.0), xs)
    # The var w is ONE node shared by all 5 steps (not 5 copies): measurable via dedup.
    assert t.unique_node_count < naive_tree_node_count(t, final)
    # exactly: w(1) + init const(1) + 5 x-vars(5) + 5 mul + 5 add = 17 unique nodes.
    assert t.unique_node_count == 17


def test_unroll_scan_nonlinear_cell_grad_matches_fd():
    # A nonlinear cell mixing mul/add/div (a rational recurrence) to exercise the full rule
    # set under unrolling, gated against finite difference.
    t = Tape()
    w, x = t.var("w"), t.var("x")
    xs = (x, x, x)

    def step(tape, carry, xk):
        # carry = (carry + xk) / (1 + w) : a contractive nonlinear-in-w update.
        return tape.div(tape.add(carry, xk), tape.add(tape.const(1.0), w))

    final = unroll_scan(t, step, t.const(0.5), xs)
    _check_fd(t, final, {"w": 0.3, "x": 1.7})


# --- higher-order: symbolic (graph-valued) gradient ------------------------------

def _eval_grad_graph(t: Tape, gnodes: dict, env: dict) -> dict:
    """Evaluate each gradient NODE (from grad_graph) at env into a {name: float} dict."""
    return {name: evaluate(t, nid, env) for name, nid in gnodes.items()}


def test_grad_graph_evaluated_equals_numeric_grad():
    # The symbolic rail (gradient as NODES) and the numeric rail (gradient as floats) must
    # agree at every point: grad_graph evaluated == grad. Checked on several functions.
    cases = []

    t1 = Tape()
    a, b = t1.var("a"), t1.var("b")
    cases.append((t1, t1.add(t1.mul(t1.mul(a, a), b), t1.mul(t1.const(3.0), a)),
                  {"a": 1.7, "b": -2.3}))                       # a*a*b + 3a

    t2 = Tape()
    c, d = t2.var("c"), t2.var("d")
    cases.append((t2, t2.div(t2.sub(c, d), t2.mul(c, d)), {"c": 1.5, "d": 2.5}))  # (c-d)/(c*d)

    t3 = Tape()
    u = tuple(t3.var(f"u{i}") for i in range(3))
    v = tuple(t3.var(f"v{i}") for i in range(3))
    env3 = {f"u{i}": float(i + 1) for i in range(3)}
    env3.update({f"v{i}": float(2 * i - 1) for i in range(3)})
    cases.append((t3, t3.dot(u, v), env3))                     # the dot substrate

    for t, out, env in cases:
        numeric = grad(t, out, env).grads
        symbolic = _eval_grad_graph(t, grad_graph(t, out, env), env)
        assert gradients_match(symbolic, numeric, rtol=1e-9, atol=1e-9), (
            f"symbolic != numeric: sym={symbolic} num={numeric}")


def test_grad_graph_is_a_valid_dag():
    # The returned gradient is an ordinary node in the same DAG -- evaluable, and (because
    # it is in the closed primitive set) itself differentiable. Both are exercised here.
    t = Tape()
    a, b = t.var("a"), t.var("b")
    out = t.mul(t.mul(a, a), b)                                 # a*a*b
    gnodes = grad_graph(t, out, ["a", "b"])
    env = {"a": 2.0, "b": 3.0}
    # d/da (a*a*b) = 2ab = 12 ; the gradient NODE evaluates to that.
    assert abs(evaluate(t, gnodes["a"], env) - 2.0 * 2.0 * 3.0) < 1e-9
    # and the gradient node can be differentiated again (closure): d/da (2ab) = 2b = 6.
    g2 = grad(t, gnodes["a"], env)
    assert abs(g2.grads["a"] - 2.0 * 3.0) < 1e-9


def test_grad_graph_unused_input_is_zero_node():
    t = Tape()
    a, unused = t.var("a"), t.var("unused")
    out = t.mul(a, a)
    gnodes = grad_graph(t, out, ["a", "unused"])
    env = {"a": 4.0, "unused": 99.0}
    assert evaluate(t, gnodes["unused"], env) == 0.0           # never reaches output
    assert abs(evaluate(t, gnodes["a"], env) - 8.0) < 1e-9     # d/da a*a = 2a


# --- higher-order: the Hessian, gated against the second difference --------------

def _check_hessian_fd(t: Tape, out: int, env: dict, *, eps=1e-4):
    """Assert the analytic Hessian matches the central SECOND difference at env."""
    H = hessian(t, out, env, env)
    sd = second_difference_hessian(t, out, env, eps=eps)
    assert hessians_match(H, sd), (
        f"hessian != second-difference: analytic={H} sd={sd} "
        f"max_abs_err={max_hessian_error(H, sd)} env={env}")
    return H


def test_hessian_matches_second_difference_on_a_quadratic_form():
    # f = a*a*b : H_aa = 2b, H_ab = H_ba = 2a, H_bb = 0 (a known closed form).
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.mul(t.mul(a, a), b)
    env = {"a": 1.5, "b": -2.0}
    H = _check_hessian_fd(t, f, env)
    assert abs(H[("a", "a")] - 2.0 * env["b"]) < 1e-2
    assert abs(H[("a", "b")] - 2.0 * env["a"]) < 1e-2
    assert abs(H[("b", "b")] - 0.0) < 1e-2


def test_hessian_matches_second_difference_on_a_polynomial():
    # f = a^3 + a*a*b - b*b : a deeper polynomial with nonzero mixed + diagonal entries.
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.sub(t.add(t.mul(t.mul(a, a), a), t.mul(t.mul(a, a), b)), t.mul(b, b))
    _check_hessian_fd(t, f, {"a": 1.1, "b": 0.7})


def test_hessian_matches_second_difference_on_a_division():
    # f = a / b : H_aa = 0, H_ab = -1/b^2, H_bb = 2a/b^3 (the quotient's second derivatives).
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.div(a, b)
    env = {"a": 1.3, "b": 2.1}
    H = _check_hessian_fd(t, f, env)
    assert abs(H[("a", "a")] - 0.0) < 1e-2
    assert abs(H[("a", "b")] - (-1.0 / env["b"] ** 2)) < 1e-2
    assert abs(H[("b", "b")] - (2.0 * env["a"] / env["b"] ** 3)) < 1e-2


def test_hessian_is_symmetric():
    # For a twice-continuously-differentiable f, H is symmetric (Clairaut/Schwarz). Check
    # H[i,j] == H[j,i] exactly (the symbolic rail is deterministic, so equality is exact).
    t = Tape()
    a, b = t.var("a"), t.var("b")
    f = t.add(t.mul(t.mul(a, a), b), t.div(a, b))              # a*a*b + a/b
    env = {"a": 1.4, "b": 2.2}
    H = hessian(t, f, env, env)
    assert H[("a", "b")] == H[("b", "a")]


def _random_dag_no_select(rng, n_vars, n_ops):
    """A random DAG over the SMOOTH primitive set only (no select) -- used for the Hessian
    fuzz, which must stay AWAY from the select boundary delta. (Reuses _random_dag, whose
    pool already excludes select.)"""
    return _random_dag(rng, n_vars, n_ops)


def test_hessian_fuzz_matches_second_difference():
    # A deterministic fuzz: random smooth DAGs (no select), analytic Hessian vs the central
    # second difference. The eps/tol is the looser second-difference gate -- still tight
    # enough to catch a wrong second-derivative rule. select is excluded to avoid the
    # boundary delta (see the honest-limitation test).
    rng = random.Random(0xB3_2D)
    for trial in range(60):
        n_vars = rng.randint(2, 3)
        n_ops = rng.randint(2, 7)
        t, out, env = _random_dag_no_select(rng, n_vars, n_ops)
        H = hessian(t, out, env, env)
        sd = second_difference_hessian(t, out, env, eps=1e-4)
        assert hessians_match(H, sd), (
            f"FUZZ trial {trial} FAILED: analytic={H} sd={sd} "
            f"max_abs_err={max_hessian_error(H, sd)} env={env} n_ops={n_ops}")
        # symmetry where the second difference is well-conditioned.
        for i in env:
            for j in env:
                assert abs(H[(i, j)] - H[(j, i)]) < 1e-9


# --- the honest limitation: closure + the select boundary caveat -----------------

def test_higher_order_closure_every_vjp_stays_in_the_primitive_set():
    # The reverse-over-reverse soundness argument: the gradient graph of any expression in
    # this primitive set is ITSELF an expression in the SAME set (+,-,*,/,neg,dot,select).
    # We pin it structurally: every node of every gradient expression has an op drawn from
    # the closed set -- so it can be differentiated again (which is why `hessian` works).
    allowed = {"const", "var", "neg", "add", "sub", "mul", "div", "dot", "select"}
    t = Tape()
    a, b = t.var("a"), t.var("b")
    out = t.div(t.mul(t.mul(a, a), b), t.add(a, b))            # mixes mul/div/add
    gnodes = grad_graph(t, out, ["a", "b"])

    def ops_reachable(root):
        seen, stack, ops = set(), [root], set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            n = t.node(nid)
            ops.add(n.op)
            stack.extend(n.args)
        return ops

    for name, gnid in gnodes.items():
        assert ops_reachable(gnid) <= allowed, (name, ops_reachable(gnid) - allowed)


def test_select_hessian_away_from_boundary_matches_but_boundary_caveat_holds():
    # (4) THE honest-limitation test, both halves.
    #
    # AWAY from the switch, hessian through a select is the second derivative of the
    # selected branch and matches the second difference.
    t = Tape()
    cond, x = t.var("cond"), t.var("x")
    f = t.select(cond, t.mul(t.mul(x, x), x), x)   # x^3 if cond>0 else x ; pick the cubic
    env_away = {"cond": 1.0, "x": 1.5}             # cond comfortably > 0
    H = hessian(t, f, ["cond", "x"], env_away)
    sd = second_difference_hessian(t, f, env_away, eps=1e-4)
    assert hessians_match(H, sd), (H, sd)
    assert abs(H[("x", "x")] - 6.0 * 1.5) < 1e-2   # d^2/dx^2 x^3 = 6x = 9

    # AT the boundary, the EXACT second derivative carries a distributional (delta) term at
    # the value jump that this a.e. convention DROPS. We do NOT claim the analytic Hessian is
    # exact there -- and we PIN that it disagrees with the second difference (which sees the
    # jump and blows up). This is the documented caveat, asserted rather than papered over.
    t2 = Tape()
    cb = t2.var("cb")
    # branches differ in VALUE at the switch -> a unit jump at cb == 0 -> a delta in d^2.
    jump = t2.select(cb, t2.add(cb, t2.const(1.0)), t2.sub(cb, t2.const(1.0)))
    env_boundary = {"cb": 0.0}
    H_b = hessian(t2, jump, ["cb"], env_boundary)
    sd_b = second_difference_hessian(t2, jump, env_boundary, eps=1e-3)
    # the a.e. convention reports a finite (here zero) second derivative; the numeric stencil
    # diverges because of the dropped delta -- they are NOT equal at the boundary.
    assert H_b[("cb", "cb")] == 0.0                # delta dropped -> finite a.e. value
    assert abs(sd_b[("cb", "cb")]) > 1e3           # the stencil sees the jump (blows up)
    assert not hessians_match(H_b, sd_b)           # honest: NOT exact at the boundary
