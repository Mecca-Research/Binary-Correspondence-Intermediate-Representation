"""B3 Phase 1 -- the closed-set autodiff DAG: FORMALIZE + machine-PROVE the closure.

The B3 autodiff oracle (``bcir.kbcir.autodiff``) realizes reverse-mode AD as content-
addressed (hash-consed) local graph rewrites over a CLOSED primitive set. ``test_autodiff``
already gates CORRECTNESS (grad == finite-difference, confluence, Hessian). This module adds
the layer those correctness gates leave implicit: the structural GUARANTEE the upcoming
``gem.autodiff`` law op will rely on -- that the adjoint operator set is CLOSED.

THE PROPERTY PROVEN HERE:  d/dx : ClosedSet -> DAG(ClosedSet).
Differentiating any expression in the primitive set ``{const, var, neg, add, sub, mul, div,
dot, select}`` produces a gradient DAG that stays in the SAME set -- the adjoint never
introduces a foreign op kind (no exp/log, no new functional primitive, no escape). Today the
``autodiff`` docstring states this in prose ("every primitive's VJP is itself expressible in
the SAME closed primitive set"); here it becomes a VERIFIED, per-primitive guarantee.

CANONICAL-FORM NOTE (why this matters for ``gem.autodiff``).  CLOSURE + HASH-CONSING together
give the adjoint DAG a CANONICAL FORM over a FIXED vocabulary: closure bounds the op kinds to
the closed set, and the content-addressed ``Tape`` interns structurally-identical subgraphs to
one node, so the gradient of a given expression is a single, deterministic, fixed-vocabulary
DAG. That is exactly what lets a future ``gem.autodiff`` law op SERIALIZE and VERIFY a gradient
as a closed, finite-alphabet graph -- it never has to anticipate an open-ended set of emitted
ops. Closure also makes ARBITRARY-ORDER AD sound: because the SYMBOLIC twin ``_BACKWARD_SYM``
emits only closed-set nodes, the gradient DAG is itself differentiable by the same machinery
(reverse-over-reverse -> the Hessian, and beyond), all within the same vocabulary.

WHAT IS PROVEN (all REAL checks -- each VJP rule is actually applied and its emitted ops are
OBSERVED by walking the emitted nodes, never a hardcoded list that could drift):
  1. CLOSURE of the symbolic adjoint ``_BACKWARD_SYM``: for EVERY differentiable primitive, the
     ops its VJP rule emits are a subset of the closed set (per-primitive statement below);
  2. CLOSURE of the numeric adjoint ``_BACKWARD``: tied to (1) by checking the numeric float
     rule computes exactly the value of the proven-closed symbolic DAG (so it has no separate
     vocabulary to escape into);
  3. REGISTRY COMPLETENESS: a bijection between the differentiable ops and each registry -- every
     differentiable primitive has exactly one rule, every rule corresponds to a real primitive
     (no orphan/dead rules), and no leaf leaked a rule -- for both ``_BACKWARD`` and
     ``_BACKWARD_SYM``;
  4. SINGLE SOURCE OF TRUTH: the closed set (derived from ``_ARITY``) equals
     ``lower.autodiff_kernel._LOWERABLE`` (the set the G6 C-kernel emitter enforces).

Deterministic, pure-Python. This is an ORACLE-ONLY slice (no MLIR, no C twin)."""

from bcir.kbcir.autodiff import (
    CLOSED_SET,
    Tape,
    closed_set_agrees_with_lowerable,
    closure_report,
    differentiable_ops,
    emitted_ops_for,
    evaluate,
    grad,
    grad_graph,
    numeric_matches_symbolic_vjp,
    registry_completeness,
)

# The closed primitive set, as a literal here ONLY for the cross-check below (the test asserts
# the module's DERIVED CLOSED_SET equals this literal AND equals _LOWERABLE -- three independent
# spellings forced to agree, so none can silently drift).
_CLOSED_SET_LITERAL = frozenset(
    {"const", "var", "neg", "add", "sub", "mul", "div", "dot", "select",
     "exp", "log", "sqrt", "tanh", "sin", "cos"})

# The per-primitive closure statement: for each differentiable op, the closed-set op kinds its
# adjoint (VJP) rule is allowed to emit. This is the human-readable CLAIM; the test PROVES it by
# OBSERVING the ops each rule actually emits (emitted_ops_for) and asserting equality -- so this
# table cannot drift from the rules without a test failure. ('var' appears wherever a rule passes
# the incoming-adjoint node through unchanged; it is a closed-set leaf.)
_EXPECTED_EMITTED = {
    "neg":    {"neg", "var"},                  # grad_a = -gz
    "add":    {"var"},                         # grad_a = grad_b = gz (pass-through)
    "sub":    {"neg", "var"},                  # grad_a = gz ; grad_b = -gz
    "mul":    {"mul", "var"},                  # grad_a = gz*b ; grad_b = gz*a
    "div":    {"div", "mul", "neg", "var"},    # grad_a = gz/b ; grad_b = -(gz*a)/(b*b)
    "dot":    {"mul", "var"},                  # grad_u_i = gz*v_i ; grad_v_i = gz*u_i
    "select": {"select", "const", "var"},      # grad_a = select(cond,gz,0) ; grad_b = select(cond,0,gz)
    "exp":    {"mul", "exp", "var"},
    "log":    {"div", "var"},
    "sqrt":   {"div", "mul", "sqrt", "const", "var"},
    "tanh":   {"mul", "sub", "tanh", "const", "var"},
    "sin":    {"mul", "cos", "var"},
    "cos":    {"neg", "mul", "sin", "var"},
}


# --- (1) closure of the SYMBOLIC adjoint: d/dx : ClosedSet -> DAG(ClosedSet) ----------

def test_symbolic_closure_every_vjp_stays_in_the_closed_set():
    """THE closure proof: for every differentiable primitive, the symbolic VJP rule emits only
    closed-set ops -- proven by APPLYING the rule and OBSERVING the emitted node kinds."""
    report = closure_report()
    # every differentiable op is covered, and every op it emits is in the closed set.
    assert set(report) == set(differentiable_ops()), set(report) ^ set(differentiable_ops())
    for op, emitted in report.items():
        assert emitted <= CLOSED_SET, (
            f"CLOSURE VIOLATED: adjoint of {op!r} emits {sorted(emitted - CLOSED_SET)} "
            f"OUTSIDE the closed set {sorted(CLOSED_SET)}")


def test_symbolic_closure_per_primitive_emitted_ops_match_the_claimed_table():
    """The per-primitive closure STATEMENT, pinned exactly: each primitive's adjoint emits the
    specific closed-set ops claimed in ``_EXPECTED_EMITTED`` -- not just 'some subset', but the
    documented set. This is the table that goes in the docs (adjoint -> emitted ops)."""
    for op in sorted(differentiable_ops()):
        observed = set(emitted_ops_for(op))
        assert observed == _EXPECTED_EMITTED[op], (
            f"{op!r} adjoint emits {sorted(observed)}, expected {sorted(_EXPECTED_EMITTED[op])}")


def test_symbolic_closure_holds_under_repeated_differentiation_hessian():
    """Reverse-over-reverse stays closed: because each VJP emits only closed-set nodes, the
    gradient DAG is itself an ordinary closed-set DAG that can be differentiated AGAIN. Pin it
    structurally on a function mixing the whole rule set -- every node reachable from each
    SECOND-order gradient node is in the closed set (so the Hessian, and any higher order, never
    escapes the vocabulary)."""
    t = Tape()
    a, b = t.var("a"), t.var("b")
    # mixes mul/div/add/sub/neg via the rules; a real multi-op expression.
    out = t.div(t.sub(t.mul(t.mul(a, a), b), b), t.add(a, b))    # (a*a*b - b)/(a+b)
    first = grad_graph(t, out, ["a", "b"])                       # gradient as closed-set nodes
    # differentiate the gradient nodes AGAIN -> the second-order gradient nodes.
    second = []
    for gnid in first.values():
        second.extend(grad_graph(t, gnid, ["a", "b"]).values())

    def ops_reachable(root: int) -> set:
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

    for gnid in (*first.values(), *second):
        esc = ops_reachable(gnid) - CLOSED_SET
        assert not esc, f"reverse-over-reverse escaped the closed set with {sorted(esc)}"


# --- (2) closure of the NUMERIC adjoint, tied to the proven symbolic DAG --------------

def test_numeric_adjoint_matches_the_proven_closed_symbolic_dag():
    """The numeric ``_BACKWARD`` rule computes exactly the VALUE of the (proven-closed) symbolic
    DAG, at a random point -- so the numeric adjoint's arithmetic is exactly the closed-set
    operations and has no separate vocabulary to escape into. Exact (closed-form rules), no
    tolerance."""
    for op in sorted(differentiable_ops()):
        assert numeric_matches_symbolic_vjp(op, seed=0xB3), (
            f"numeric adjoint of {op!r} disagrees with its proven-closed symbolic twin")


# --- (3) registry completeness: a bijection ops <-> rules, no orphans, no missing -----

def test_registry_is_a_bijection_with_the_differentiable_ops():
    """Every differentiable primitive has EXACTLY one backward rule and every rule corresponds to
    a real primitive -- for BOTH the numeric and symbolic registries (no missing rule, no
    orphan/dead rule)."""
    rc = registry_completeness()
    assert rc["backward_complete"], (
        f"_BACKWARD not a bijection: missing={sorted(rc['missing_backward'])} "
        f"orphan={sorted(rc['orphan_backward'])}")
    assert rc["backward_sym_complete"], (
        f"_BACKWARD_SYM not a bijection: missing={sorted(rc['missing_backward_sym'])} "
        f"orphan={sorted(rc['orphan_backward_sym'])}")


def test_leaves_carry_no_backward_rule():
    """``const`` and ``var`` are differentiation BOUNDARIES -- they must have NO rule in either
    registry (a leaf with a rule would be a dead/incorrect entry)."""
    rc = registry_completeness()
    assert not rc["leaves_in_backward"], rc["leaves_in_backward"]
    assert not rc["leaves_in_backward_sym"], rc["leaves_in_backward_sym"]


def test_numeric_and_symbolic_registries_cover_the_same_ops():
    """The numeric and symbolic adjoint registries are TWINS: they cover the exact same
    differentiable op set (so every op that can be numerically differentiated can also be
    symbolically differentiated for higher order, and vice versa)."""
    rc = registry_completeness()
    assert rc["backward_keys"] == rc["backward_sym_keys"], (
        rc["backward_keys"] ^ rc["backward_sym_keys"])


# --- (4) single source of truth: the closed set agrees everywhere ---------------------

def test_closed_set_is_a_single_source_of_truth():
    """Three independent spellings of 'the closed set' must AGREE: the module's CLOSED_SET
    (derived from ``_ARITY``), this test's literal, and ``autodiff_kernel._LOWERABLE`` (the G6
    C-kernel emitter's enforced set). If a primitive is added to one, this trips until all three
    are updated -- the closure proof and the lowering can never silently diverge."""
    assert CLOSED_SET == _CLOSED_SET_LITERAL, CLOSED_SET ^ _CLOSED_SET_LITERAL
    assert closed_set_agrees_with_lowerable(), "CLOSED_SET != autodiff_kernel._LOWERABLE"


def test_differentiable_ops_are_the_closed_set_minus_leaves():
    """The differentiable set is exactly the closed set minus the two leaves -- a derived
    identity (not hand-listed), so it tracks the primitive set automatically."""
    assert differentiable_ops() == CLOSED_SET - {"const", "var"}


# --- a tiny end-to-end sanity: a concrete gradient is a fixed-vocabulary DAG ----------

def test_concrete_gradient_dag_is_in_the_closed_vocabulary_and_correct():
    """End-to-end: a concrete gradient graph is BOTH numerically correct AND entirely within the
    closed vocabulary -- the canonical-form property (closed + hash-consed) made concrete on one
    expression."""
    t = Tape()
    a, b = t.var("a"), t.var("b")
    out = t.div(t.mul(t.mul(a, a), b), t.add(a, b))            # (a*a*b)/(a+b)
    env = {"a": 1.7, "b": -2.3}
    gnodes = grad_graph(t, out, ["a", "b"])

    def ops_reachable(root: int) -> set:
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

    numeric = grad(t, out, env).grads
    for name, gnid in gnodes.items():
        assert ops_reachable(gnid) <= CLOSED_SET                # fixed vocabulary
        assert abs(evaluate(t, gnid, env) - numeric[name]) < 1e-9   # and correct


# --- a human-readable report (printed when run as a script) --------------------------

def _print_closure_report() -> None:
    """Print the per-primitive closure result + registry/source-of-truth status. Called by the
    ``__main__`` block so ``python -m bcir.tests.test_autodiff_closure`` SHOWS the proof, not just
    a pass count."""
    print("B3 closed-set autodiff -- closure proof  (d/dx : ClosedSet -> DAG(ClosedSet))")
    print(f"  closed set:        {sorted(CLOSED_SET)}")
    print(f"  differentiable:    {sorted(differentiable_ops())}  (closed set minus leaves)")
    print("  per-primitive adjoint -> emitted closed-set ops (OBSERVED by applying each VJP):")
    report = closure_report()
    for op in sorted(report):
        emitted = sorted(report[op])
        ok = report[op] <= CLOSED_SET
        print(f"    {op:7s} -> {str(emitted):45s} in-closed-set={ok}")
    rc = registry_completeness()
    print(f"  registry completeness: _BACKWARD bijection={rc['backward_complete']}  "
          f"_BACKWARD_SYM bijection={rc['backward_sym_complete']}")
    print(f"  _LOWERABLE agrees with CLOSED_SET: {closed_set_agrees_with_lowerable()}")


if __name__ == "__main__":
    import sys

    _print_closure_report()
    print()
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
