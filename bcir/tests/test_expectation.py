"""G22: expected cost from branch probabilities -- exact, over loops, with its uncertainty named.

The parent had one expectation, `compose.Cond`'s: a declared probability that defaulted to one
half and was clamped when out of range, floored at every branch, and nothing for loops. These
witnesses hold the replacement to hand-derived means, to an independent derivation over
generated region trees, and to the registry's semiring laws -- and each names the parent's
behavior it refuses.
"""

from __future__ import annotations

import dataclasses
import random
from fractions import Fraction as F

from bcir.kbcir import compose
from bcir.kbcir.compose import PRED_COST, Call, Cond, Function, Leaf, Loop, Seq, plan_composite
from bcir.kbcir.expectation import (
    BranchProfile,
    MarkovProgram,
    NonTerminating,
    ProbabilityError,
    expected_cost,
    expected_visits,
    loop_trips,
)
from bcir.kbcir.objectives import (
    ObjectiveError,
    dag_best_path,
    dag_path_sum,
    objective,
    verify_objective,
)
from bcir.kbcir.scope import scope_for
from bcir.tests import expectation_fixtures as fx


def _refused(call, error=(ProbabilityError, NonTerminating)) -> str:
    try:
        call()
    except error as exc:
        return str(exc)
    raise AssertionError("not refused")


def test_the_registry_admits_the_expectation_semiring_only_with_its_laws():
    """A `sum` objective is held to the semiring's laws, not a select objective's: the
    expectation semiring passes them; a componentwise max (a choice wearing the label), an
    order, and a combine that adds costs without weighting them are each refused for the law
    they break."""
    entry = objective("expectation")
    assert entry.kind == "sum" and entry.better is None
    assert verify_objective(entry) == []
    as_max = dataclasses.replace(entry, select=lambda a, b: (max(a[0], b[0]), max(a[1], b[1])))
    problems = verify_objective(as_max)
    assert "combine does not distribute over select" in problems
    assert any("idempotent" in p for p in problems)
    ordered = dataclasses.replace(entry, better=lambda a, b: a[1] < b[1])
    assert any("no order" in p for p in verify_objective(ordered))
    unweighted = dataclasses.replace(entry, combine=lambda a, b: (a[0] * b[0], a[1] + b[1]))
    assert any("annihilate" in p for p in verify_objective(unweighted))
    for name in ("min_plus", "max_plus"):
        try:
            dag_path_sum(2, [[(1, 3)], []], objective(name))
        except ObjectiveError:
            continue
        raise AssertionError(f"dag_path_sum summed under the select objective {name}")
    try:
        dag_best_path(2, [[(1, (F(1), F(3)))], []], entry)
    except ObjectiveError:
        pass
    else:
        raise AssertionError("dag_best_path chose under a sum objective")


def test_every_hand_derived_mean_is_reproduced_exactly():
    """Each corpus program's mean, derived by hand in its entry, to the last digit: branches,
    a do-while, a test-first while, nested loops, a loop with two exits, a random walk, an
    unrolled bounded loop and a zero-probability branch."""
    for name, program, mean, why in fx.corpus():
        assert expected_cost(program) == mean, (name, expected_cost(program), mean, why)
    # a loop header's expected visits are the loop's expected trips
    header = MarkovProgram({("h", "h"): (F(2, 3), 5), ("h", "x"): (F(1, 3), 5)}, "h", "x")
    assert expected_visits(header) == {"h": loop_trips(F(2, 3))} == {"h": F(3)}
    assert loop_trips(F(1, 2), 3) == F(7, 4) and loop_trips(1, 4) == 4 and loop_trips(0) == 1


def test_on_an_acyclic_program_the_markov_solve_is_the_semiring_path_sum():
    """Two derivations of one mean: the expectation semiring's path sum over a DAG
    (`objectives.dag_path_sum`) and the absorbing-chain solve, over generated programs."""
    entry = objective("expectation")
    for seed in range(40):
        rng = random.Random(seed)
        n = rng.randint(2, 9)
        edges, adj = {}, [[] for _ in range(n)]
        for u in range(n - 1):
            targets = sorted(rng.sample(range(u + 1, n), rng.randint(1, min(3, n - 1 - u))))
            cuts = sorted(rng.randint(0, 12) for _ in range(len(targets) - 1))
            weights = [b - a for a, b in zip([0] + cuts, cuts + [12])]
            for v, w in zip(targets, weights):
                p, c = F(w, 12), rng.randint(0, 50)
                edges[(str(u), str(v))] = (p, c)
                adj[u].append((v, (p, p * c)))
        program = MarkovProgram(edges, "0", str(n - 1))
        reach = dag_path_sum(n, adj, entry)[n - 1]
        assert reach[0] == 1, seed  # every path ends at the exit
        assert reach[1] == expected_cost(program), seed


def test_compose_and_the_markov_solve_agree_on_every_generated_region():
    """compose's exact mean (a geometric series per loop, a weighted sum per branch) against
    the Markov solve of the region's UNROLLED control-flow graph -- independent arithmetic --
    over 60 generated region trees covering every shape. The differential is shown to fire:
    the parent's floor at every branch, and a loop bound ignored, each split it."""
    compared, disagreements = fx.differential()
    assert compared == 60 and disagreements == 0
    floored = compose._result
    compose._result = lambda w, e, l, r=0: floored(w, F(e.numerator // e.denominator), l, r)
    try:
        assert fx.differential()[1] > 0, "the differential cannot see the parent's floor"
    finally:
        compose._result = floored
    trips = compose.loop_trips
    compose.loop_trips = lambda q, n=None: trips(q, None) if q != 1 else F(n)
    try:
        assert fx.differential()[1] > 0, "the differential cannot see a loop's bound ignored"
    finally:
        compose.loop_trips = trips


def test_the_mean_is_exact_where_the_parent_floored_every_branch():
    """RED on the parent: a branch nested in a branch, priced at the two arms' exact means. The
    parent floored the inner branch's mean before weighting it, so its expected cost was not
    the mean of any outcome; the exact mean is floored once, at the end."""
    leaves, resources, target, theta = fx.planning_context()
    a, _b, c = leaves
    cost_a = plan_composite(a, {}, resources, target, theta).worst_cost
    cost_c = plan_composite(c, {}, resources, target, theta).worst_cost
    inner = Cond("i", a, c, 333)
    region = Cond("o", a, inner, 250)
    got = plan_composite(region, {}, resources, target, theta)
    inner_mean = PRED_COST + F(333, 1000) * cost_a + F(667, 1000) * cost_c
    want = PRED_COST + F(250, 1000) * cost_a + F(750, 1000) * inner_mean
    assert got.expected == want
    assert got.expected_cost == want.numerator // want.denominator
    parent_inner = PRED_COST + (333 * cost_a + 667 * cost_c) // 1000
    parent = PRED_COST + (250 * cost_a + 750 * parent_inner) // 1000
    assert parent != got.expected_cost, "the fixture no longer separates the two"


def test_a_probability_is_refused_rather_than_repaired():
    """The parent clamped `prob_then_milli=5000` to a certain branch. Every law is a refusal:
    a declared probability outside [0, 1000], a float, mass that does not sum to 1, a negative
    cost, an absorbing exit with an edge out, a never-observed branch, a loop with no trip."""
    leaf = Leaf(())
    for bad in (5000, -1, True, 500.0):
        _refused(lambda bad=bad: Cond("p", leaf, leaf, bad))
    _refused(lambda: Loop(leaf, "l", 500, 0))
    _refused(lambda: Loop(leaf, "l", 1001, 3))
    assert "not float" in _refused(
        lambda: MarkovProgram({("s", "x"): (0.5, 1), ("s", "y"): (F(1, 2), 1)}, "s", "x")
    )
    assert "not 1" in _refused(lambda: MarkovProgram({("s", "x"): (F(1, 2), 1)}, "s", "x"))
    assert "non-negative" in _refused(lambda: MarkovProgram({("s", "x"): (1, -3)}, "s", "x"))
    assert "absorbing" in _refused(lambda: MarkovProgram({("x", "s"): (1, 0)}, "s", "x"))
    assert "never observed" in _refused(lambda: BranchProfile({"p": (0, 0)}))
    assert "taken <= observed" in _refused(lambda: BranchProfile({"p": (5, 4)}))


def test_a_program_that_can_be_trapped_has_no_finite_mean_and_names_the_trap():
    """`NonTerminating` names exactly the states the entry reaches and the exit cannot: a
    two-state cycle with no edge out, and a loop that always continues with no bound."""
    program = MarkovProgram(
        {
            ("s", "a"): (F(1, 2), 1),
            ("s", "x"): (F(1, 2), 1),
            ("a", "b"): (1, 1),
            ("b", "a"): (1, 1),
        },
        "s",
        "x",
    )
    try:
        expected_cost(program)
    except NonTerminating as exc:
        assert exc.trapped == ("a", "b")
    else:
        raise AssertionError("a trapped program was given a mean")
    _refused(lambda: loop_trips(1), NonTerminating)
    assert (
        plan_composite(Loop(Leaf(()), "l", 1000, 4), {}, {}, *fx.planning_context()[2:]).expected
        == 4 * PRED_COST
    )  # a bounded loop that always continues runs its bound


def test_a_measured_profile_prices_the_branch_and_is_the_scopes_uncertainty():
    """A `BranchProfile`'s measured probability replaces the declared one for the predicate it
    names (and only that one); a call site whose callee holds a measured predicate is planned,
    not served a summary priced under the declared probability; and the profile is the scope's
    `U` component, so two plans priced under different measurements differ in their scope."""
    leaves, resources, target, theta = fx.planning_context()
    a, b, _c = leaves
    cost = {id(x): plan_composite(x, {}, resources, target, theta).worst_cost for x in (a, b)}
    region = Seq((Cond("hot", a, b, 500), Cond("cold", a, b, 500)))
    profile = BranchProfile({"hot": (9, 10)})
    got = plan_composite(region, {}, resources, target, theta, profile=profile).expected
    want = (PRED_COST + F(9, 10) * cost[id(a)] + F(1, 10) * cost[id(b)]) + (
        PRED_COST + F(1, 2) * cost[id(a)] + F(1, 2) * cost[id(b)]
    )
    assert got == want
    functions = {"f": Function("f", Cond("hot", a, b, 500))}
    summary = compose.summarize(functions["f"], functions, resources, target, theta)
    served = plan_composite(
        Call("f"), functions, resources, target, theta, summaries={"f": summary}
    )
    profiled = plan_composite(
        Call("f"), functions, resources, target, theta, summaries={"f": summary}, profile=profile
    )
    assert served.reused == 2 and profiled.reused == 0  # served: both of the callee's leaves
    assert profiled.expected == PRED_COST + F(9, 10) * cost[id(a)] + F(1, 10) * cost[id(b)]
    u1 = scope_for(uncertainty=profile)
    u2 = scope_for(uncertainty=BranchProfile({"hot": (8, 10)}))
    assert u1.U == profile.component() and u1.U["branches"] == [["hot", 9, 10]]
    assert u1.component_digest("U") != u2.component_digest("U")
    assert scope_for().component_digest("U") != u1.component_digest("U")
