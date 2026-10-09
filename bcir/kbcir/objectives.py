"""The typed objective registry (G6 / S2-D): semirings with their laws, verified.

The 2026-08-12 report (section 9): tropical min-plus is one exact kernel in the portfolio, not
a master algorithm -- alternatives that combine by minimum and serial costs that add. Parallel
completion combines by maximum, a bottleneck by min-max, legality by Boolean reachability,
non-substitutable objectives by lexicographic or Pareto order. The roadmap's G6 asks for a
registry of those objectives "each verifying closure, identities, comparison semantics and
overflow policy" -- and warns that "semiring must not become a label that admits arbitrary
operators without their laws".

So an `Objective` here is a value with its laws attached, and `verify_objective` PROVES them on
samples before the registry admits the entry: closure of `combine` and `select` over the
carrier under the declared overflow policy, the identities (`one` for `combine`, `zero` for
`select`), associativity, the commutativity and idempotence of `select`, distributivity where
claimed, and that `better` is the strict order `select` realizes. `dag_best_path` is the one
relaxation over a layered DAG for every scalar objective -- the min-plus path the planner runs
(`semiring.dag_shortest_path`, reproduced exactly) and the max-plus longest path the exact
scheduler's critical-path bound is (`gem.exact._critical_path`, reproduced exactly).

Overflow is a policy, not an accident: the `checked` policy refuses any value outside the
signed 64-bit range (the plan's costs are `i64` on every rail), `saturate` clamps to it and
says so. No entry is admitted with an undeclared policy.

Two KINDS of objective, each with its own law set (G22). A `select` objective chooses: its
`select` is idempotent and realizes a strict order `better` -- every entry above. A `sum`
objective accumulates: its `select` adds alternatives weighted by how likely they are, so it is
neither idempotent nor ordered, and an entry that claimed either would be lying. The
expectation semiring (Eisner 2002) is the one `sum` entry: a value is a pair (p, r) -- the
probability of the paths it stands for and their probability-weighted cost -- `combine` is
(p1 p2, p1 r2 + p2 r1) and `select` is (p1 + p2, r1 + r2). Its laws are the semiring's own:
both operations associative and commutative with their identities, `combine` distributing
over `select`, and `zero` annihilating. The carrier is exact non-negative rationals, so no
value overflows and the policy is declared but never reached; the expected cost a path sum
names is a rational, and `bcir.kbcir.expectation` owns how it is projected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

I64_MAX = (1 << 63) - 1
I64_MIN = -(1 << 63)
OVERFLOW_POLICIES = ("checked", "saturate")
KINDS = ("select", "sum")

INF = float("inf")


class ObjectiveError(ValueError):
    """A registry law failed, or a value left the carrier."""


@dataclass(frozen=True)
class Objective:
    """A semiring-shaped objective: `combine` composes along a path, `select` chooses across
    alternatives, `one` / `zero` are their identities, `better(a, b)` is the strict order
    `select` realizes (`select(a, b) is a` iff not `better(b, a)` for scalars), `laws` are the
    properties claimed beyond the semiring's own (`distributive`, `commutative_combine`),
    and `overflow` the declared policy for values leaving the i64 range. `kind` names the law
    set: `select` (the default) chooses and is ordered; `sum` accumulates, has no `better`
    (None), and is held to the semiring laws instead (module docstring)."""

    name: str
    carrier: str
    combine: Callable
    select: Callable
    one: object
    zero: object
    better: Callable
    laws: frozenset = field(default_factory=frozenset)
    overflow: str = "checked"
    describe: str = ""
    kind: str = "select"

    def __post_init__(self) -> None:
        if self.overflow not in OVERFLOW_POLICIES:
            raise ObjectiveError(f"overflow policy must be one of {OVERFLOW_POLICIES}")
        if self.kind not in KINDS:
            raise ObjectiveError(f"objective kind must be one of {KINDS}")


def _clamp(value, policy: str):
    """Apply the overflow policy to a scalar (int or +/-inf)."""
    if isinstance(value, float):
        if value in (INF, -INF):
            return value  # the identities live outside the range by design
        raise ObjectiveError("a non-integral value left the carrier")
    if I64_MIN <= value <= I64_MAX:
        return value
    if policy == "saturate":
        return I64_MAX if value > I64_MAX else I64_MIN
    raise ObjectiveError(f"value {value} overflows i64 under the checked policy")


def _in_range(value) -> bool:
    return value in (INF, -INF) or (isinstance(value, int) and I64_MIN <= value <= I64_MAX)


# --- the scalar objectives -----------------------------------------------------------------


def _min_plus(policy: str) -> Objective:
    return Objective(
        "min_plus",
        "int",
        lambda a, b: _clamp(a + b, policy),
        lambda a, b: a if a <= b else b,
        0,
        INF,
        lambda a, b: a < b,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "additive serial costs, the cheapest alternative wins (the planner's path)",
    )


def _max_plus(policy: str) -> Objective:
    return Objective(
        "max_plus",
        "int",
        lambda a, b: _clamp(a + b, policy),
        lambda a, b: a if a >= b else b,
        0,
        -INF,
        lambda a, b: a > b,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "additive path lengths, the longest wins (critical path, earliest finish)",
    )


def _min_max(policy: str) -> Objective:
    return Objective(
        "min_max",
        "int",
        lambda a, b: a if a >= b else b,
        lambda a, b: a if a <= b else b,
        -INF,
        INF,
        lambda a, b: a < b,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "a path costs its bottleneck, the least bottleneck wins",
    )


def _boolean(policy: str) -> Objective:
    return Objective(
        "boolean",
        "bool",
        lambda a, b: a and b,
        lambda a, b: a or b,
        True,
        False,
        lambda a, b: a and not b,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "reachability: a path is legal iff every step is, a node is reachable iff some path is",
    )


def _lexicographic(policy: str) -> Objective:
    def combine(a, b):
        if len(a) != len(b):
            raise ObjectiveError("lexicographic values must have one arity")
        return tuple(_clamp(x + y, policy) for x, y in zip(a, b))

    return Objective(
        "lexicographic",
        "tuple",
        combine,
        lambda a, b: a if a <= b else b,
        (),  # the identity is the all-zero tuple of the value's arity: see `identity_for`
        (),
        lambda a, b: a < b,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "tuples added elementwise, compared in priority order (non-substitutable objectives)",
    )


def identity_for(objective: Objective, value):
    """The identities of an arity-typed objective for a value of that arity: a tuple
    objective's are the all-zero and all-infinite tuples, a frontier objective's the frontier
    holding the zero vector (nothing costs less) and the empty frontier (nothing to choose)."""
    if objective.carrier == "tuple":
        return tuple(0 for _ in value), tuple(INF for _ in value)
    if objective.carrier == "frontier":
        arity = len(next(iter(value))) if value else 0
        return frozenset({tuple(0 for _ in range(arity))}), frozenset()
    return objective.one, objective.zero


# --- the Pareto objective (a set-valued select) ------------------------------------------


def dominates(a: tuple, b: tuple) -> bool:
    """`a` dominates `b`: no worse in every component, strictly better in one."""
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def frontier(points) -> frozenset:
    """The non-dominated subset."""
    members = set(points)
    return frozenset(p for p in members if not any(q != p and dominates(q, p) for q in members))


def _pareto(policy: str) -> Objective:
    def combine(a: frozenset, b: frozenset) -> frozenset:
        return frontier(tuple(_clamp(x + y, policy) for x, y in zip(p, q)) for p in a for q in b)

    def select(a: frozenset, b: frozenset) -> frozenset:
        return frontier(set(a) | set(b))

    def better(a: frozenset, b: frozenset) -> bool:
        """`a` is better iff it dominates `b` as a set: every point of `b` is dominated by or
        equal to a point of `a`, and the sets differ."""
        return a != b and all(any(p == q or dominates(p, q) for p in a) for q in b)

    return Objective(
        "pareto",
        "frontier",
        combine,
        select,
        frozenset({()}),  # arity-typed: see `identity_for`
        frozenset(),
        better,
        frozenset({"commutative_combine"}),
        policy,
        "vector costs kept as a non-dominated frontier; the order is dominance (partial)",
    )


# --- the expectation semiring (a sum objective, G22) ----------------------------------------


def _expectation(policy: str) -> Objective:
    """(p, r): the probability of a set of paths and their probability-weighted cost. A branch
    taken with probability q at cost c is (q, q c); a path multiplies probabilities and adds
    its weighted costs; alternatives add. The path sum from entry to exit is (P[reach], E[cost
    ; reach]) -- `bcir.kbcir.expectation` solves it over cycles too."""

    def combine(a, b):
        return (a[0] * b[0], a[0] * b[1] + b[0] * a[1])

    def select(a, b):
        return (a[0] + b[0], a[1] + b[1])

    return Objective(
        "expectation",
        "expectation",
        combine,
        select,
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0)),
        None,
        frozenset({"distributive", "commutative_combine"}),
        policy,
        "probability-weighted path costs; alternatives add (the expectation semiring, Eisner 2002)",
        "sum",
    )


_FACTORIES = {
    "min_plus": _min_plus,
    "max_plus": _max_plus,
    "min_max": _min_max,
    "boolean": _boolean,
    "lexicographic": _lexicographic,
    "pareto": _pareto,
    "expectation": _expectation,
}

#: The names the law rail shares (`BCIR_Semiring` in `mlir/include/BCIR/BCIRAttrs.td`): the
#: registry is a superset; a name outside this set has no `#bcir.semiring<...>` spelling yet.
LAW_RAIL_NAMES = ("min_plus", "max_plus")


def objective(name: str, overflow: str = "checked") -> Objective:
    """The registry lookup: a verified objective by name under an overflow policy."""
    if name not in _FACTORIES:
        raise ObjectiveError(f"unknown objective {name!r}; the registry has {sorted(_FACTORIES)}")
    entry = _FACTORIES[name](overflow)
    problems = verify_objective(entry)
    if problems:
        raise ObjectiveError(f"objective {name!r} fails its laws: {problems}")
    return entry


def registry(overflow: str = "checked") -> dict[str, Objective]:
    return {name: objective(name, overflow) for name in _FACTORIES}


# --- the laws --------------------------------------------------------------------------------


def _samples(entry: Objective, count: int = 24, seed: int = 7) -> list:
    """Deterministic carrier samples, edge values included."""
    import random

    rng = random.Random(seed)
    if entry.carrier == "int":
        values = [0, 1, 2, 3, 5, 7, 11, 1000, 12345, I64_MAX // 2]
        values += [rng.randrange(0, 1 << 40) for _ in range(count)]
        if entry.overflow == "saturate":
            values.append(I64_MAX - 1)
        return values
    if entry.carrier == "bool":
        return [True, False]
    if entry.carrier == "tuple":
        return [tuple(rng.randrange(0, 1 << 20) for _ in range(3)) for _ in range(count)] + [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        ]
    if entry.carrier == "expectation":
        # Specials first (the law loops run over the first six): a certain path with no cost,
        # an impossible path with cost, and pairs whose probability and cost disagree on which
        # is larger -- a componentwise max passes distributivity on any sample where p and r
        # rise together, so the samples must not let them.
        values = [
            (Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(5)),
            (Fraction(1), Fraction(1)),
            (Fraction(1, 2), Fraction(3)),
            (Fraction(2), Fraction(1, 3)),
            (Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(7)),
        ]
        for _ in range(count):
            values.append(
                (
                    Fraction(rng.randrange(0, 17), rng.randrange(1, 9)),
                    Fraction(rng.randrange(0, 1 << 20), rng.randrange(1, 9)),
                )
            )
        return values
    if entry.carrier == "frontier":
        points = [tuple(rng.randrange(0, 50) for _ in range(2)) for _ in range(6 * count)]
        return [frontier(points[i : i + 6]) for i in range(0, len(points), 6)] + [
            frozenset({(0, 0)}),
            frozenset({(1, 5), (5, 1)}),
        ]
    raise ObjectiveError(f"unknown carrier {entry.carrier!r}")


def _member(entry: Objective, value) -> bool:
    if entry.carrier == "int":
        return _in_range(value)
    if entry.carrier == "bool":
        return isinstance(value, bool)
    if entry.carrier == "tuple":
        return isinstance(value, tuple) and all(_in_range(v) for v in value)
    if entry.carrier == "expectation":
        return (
            isinstance(value, tuple)
            and len(value) == 2
            and all(isinstance(v, (int, Fraction)) and not isinstance(v, bool) for v in value)
            and value[0] >= 0
            and value[1] >= 0
        )
    if entry.carrier == "frontier":
        return (
            isinstance(value, frozenset)
            and all(isinstance(p, tuple) and all(_in_range(v) for v in p) for p in value)
            and value == frontier(value)
        )
    return False


def verify_objective(entry: Objective, samples=None) -> list[str]:
    """Every law the entry claims, checked on samples. Returns the failures (empty = admitted)."""
    xs = list(samples) if samples is not None else _samples(entry)
    if not xs:
        return ["no samples"]
    if entry.kind == "sum":
        return _verify_sum(entry, xs)
    if entry.better is None:
        return ["a select objective needs the order its select realizes"]
    problems: list[str] = []
    combine, select, better = entry.combine, entry.select, entry.better
    one, zero = identity_for(entry, xs[0])
    # closure under the overflow policy
    for a in xs:
        for b in xs[:8]:
            try:
                c, s = combine(a, b), select(a, b)
            except ObjectiveError as exc:
                if entry.overflow == "checked" and "overflows" in str(exc):
                    continue  # the checked policy refuses: closure holds by refusal
                problems.append(f"combine/select raised {exc}")
                continue
            if not _member(entry, c) or not _member(entry, s):
                problems.append(f"closure: combine({a!r},{b!r}) or select left the carrier")
    # identities
    for a in xs:
        if combine(a, one) != a or combine(one, a) != a:
            problems.append(f"combine identity fails on {a!r}")
        if select(a, zero) != a or select(zero, a) != a:
            problems.append(f"select identity fails on {a!r}")
    # associativity of both, commutativity and idempotence of select
    trio = xs[:6]
    for a in trio:
        for b in trio:
            for c in trio:
                try:
                    if combine(combine(a, b), c) != combine(a, combine(b, c)):
                        problems.append("combine is not associative")
                    if select(select(a, b), c) != select(a, select(b, c)):
                        problems.append("select is not associative")
                    if "distributive" in entry.laws and combine(a, select(b, c)) != select(
                        combine(a, b), combine(a, c)
                    ):
                        problems.append("combine does not distribute over select")
                except ObjectiveError as exc:
                    if not (entry.overflow == "checked" and "overflows" in str(exc)):
                        problems.append(f"law evaluation raised {exc}")
            if select(a, b) != select(b, a):
                problems.append("select is not commutative")
            if "commutative_combine" in entry.laws:
                try:
                    if combine(a, b) != combine(b, a):
                        problems.append("combine is not commutative")
                except ObjectiveError as exc:
                    if not (entry.overflow == "checked" and "overflows" in str(exc)):
                        problems.append(f"law evaluation raised {exc}")
        if select(a, a) != a:
            problems.append("select is not idempotent")
    # comparison semantics: `better` is strict, irreflexive, transitive, and select realizes it
    for a in xs[:10]:
        if better(a, a):
            problems.append("better is not irreflexive")
        for b in xs[:10]:
            if better(a, b) and better(b, a):
                problems.append("better is not antisymmetric")
            chosen = select(a, b)
            if better(b, a) and chosen != b:
                problems.append("select does not realize better")
            if better(a, b) and chosen != a:
                problems.append("select does not realize better")
            for c in xs[:10]:
                if better(a, b) and better(b, c) and not better(a, c):
                    problems.append("better is not transitive")
    return sorted(set(problems))


def _verify_sum(entry: Objective, xs: list) -> list[str]:
    """The semiring laws of a `sum` objective: closure, both identities, both operations
    associative and `select` commutative (`combine` too when claimed), `combine` distributing
    over `select` from both sides, and `zero` annihilating. No order: an entry that offers one
    claims to choose, which a sum does not."""
    problems: list[str] = []
    combine, select, one, zero = entry.combine, entry.select, entry.one, entry.zero
    if entry.better is not None:
        problems.append("a sum objective has no order: it accumulates alternatives, never chooses")
    if not _member(entry, one) or not _member(entry, zero):
        problems.append("an identity is outside the carrier")
    for a in xs:
        for b in xs[:8]:
            if not _member(entry, combine(a, b)) or not _member(entry, select(a, b)):
                problems.append(f"closure: combine({a!r},{b!r}) or select left the carrier")
        if combine(a, one) != a or combine(one, a) != a:
            problems.append(f"combine identity fails on {a!r}")
        if select(a, zero) != a or select(zero, a) != a:
            problems.append(f"select identity fails on {a!r}")
        if combine(a, zero) != zero or combine(zero, a) != zero:
            problems.append(f"zero does not annihilate {a!r}")
    trio = xs[:6]
    for a in trio:
        for b in trio:
            if select(a, b) != select(b, a):
                problems.append("select is not commutative")
            if "commutative_combine" in entry.laws and combine(a, b) != combine(b, a):
                problems.append("combine is not commutative")
            for c in trio:
                if combine(combine(a, b), c) != combine(a, combine(b, c)):
                    problems.append("combine is not associative")
                if select(select(a, b), c) != select(a, select(b, c)):
                    problems.append("select is not associative")
                if "distributive" in entry.laws and (
                    combine(a, select(b, c)) != select(combine(a, b), combine(a, c))
                    or combine(select(b, c), a) != select(combine(b, a), combine(c, a))
                ):
                    problems.append("combine does not distribute over select")
    if "distributive" not in entry.laws:
        problems.append("a sum objective is a semiring: it must claim distributivity")
    if all(select(a, a) == a for a in xs):
        problems.append("select is idempotent: an objective that chooses is a select objective")
    return sorted(set(problems))


# --- the one relaxation over a layered DAG ---------------------------------------------------


def dag_best_path(
    n: int, adj: list[list[tuple[int, object]]], entry: Objective, source: int = 0
) -> tuple[list, list[int]]:
    """`(best, pred)` over a DAG whose edges only go to higher node ids, under `entry`: the
    value of a path is the `combine` of its edge weights, a node's value the `select` over
    its paths. Under `min_plus` this is `semiring.dag_shortest_path` to the digit (the same
    first-wins tie-break); under `max_plus` the longest path."""
    if entry.kind != "select" or entry.carrier not in ("int", "bool", "tuple"):
        raise ObjectiveError("dag_best_path relaxes scalar select objectives (int, bool, tuple)")
    probe = next((w for edges in adj for _v, w in edges), None)
    one, zero = identity_for(entry, probe if probe is not None else entry.one)
    best: list = [zero] * n
    pred: list[int] = [-1] * n
    best[source] = one
    for u in range(source, n):
        if best[u] == zero:
            continue
        for v, w in adj[u]:
            candidate = entry.combine(best[u], w)
            if entry.better(candidate, best[v]):
                best[v] = candidate
                pred[v] = u
    return best, pred


def dag_path_sum(n: int, adj: list[list[tuple[int, object]]], entry: Objective, source: int = 0):
    """The path sum over a DAG whose edges only go to higher node ids, under a `sum` objective:
    a node's value is the `select` (sum) over its incoming paths of their `combine`. Under
    `expectation`, with an edge taken with probability q at cost c weighted as (q, q c), a
    node's value is (P[reach it], E[cost of the paths to it ; reach it]) -- the acyclic case of
    `bcir.kbcir.expectation.expected_cost`, which the tests hold the two to."""
    if entry.kind != "sum":
        raise ObjectiveError("dag_path_sum sums under a sum objective; a select one chooses")
    value: list = [entry.zero] * n
    value[source] = entry.one
    for u in range(source, n):
        if value[u] == entry.zero:
            continue
        for v, w in adj[u]:
            if v <= u:
                raise ObjectiveError("dag_path_sum needs edges to higher node ids (a DAG)")
            value[v] = entry.select(value[v], entry.combine(value[u], w))
    return value


__all__ = [
    "I64_MAX",
    "KINDS",
    "I64_MIN",
    "INF",
    "LAW_RAIL_NAMES",
    "OVERFLOW_POLICIES",
    "Objective",
    "ObjectiveError",
    "dag_best_path",
    "dag_path_sum",
    "dominates",
    "frontier",
    "identity_for",
    "objective",
    "registry",
    "verify_objective",
]
