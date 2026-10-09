"""Expected cost from branch probabilities, exact and over loops (GEM+ G22).

The 2026-10-06 audit (item 8, Missing) found one expectation in the oracle: `compose.Cond`
weighting two branches by a declared `prob_then_milli` that defaulted to one half, clamped an
out-of-range value without a word, and floored at every branch, so a nested expectation was
not the mean of anything. Loops had none at all, and `tropical.py` said why: "the cost model
does not carry branch probabilities". This module carries them.

**A program with branch probabilities is a Markov chain**, and its expected cost is an
absorbing-chain expectation, not a shortest path (`tropical`'s min-plus answers the best case,
max-plus the worst; neither is a mean). `MarkovProgram` is a control-flow graph whose every edge
carries the probability it is taken and the cost of taking it; `expected_cost` solves
`(I - Q) x = b` over the states the entry reaches, in exact rationals, so a mean is a number a
hand derivation can be held to, digit for digit -- loops, nested loops and loops with several
exits included. On an acyclic program it is the path sum of the registry's `expectation`
semiring (`objectives.dag_path_sum`), which the tests hold the two to.

The laws, each a refusal rather than a repair:

* a probability is an exact rational in [0, 1] -- never a float, whose meaning depends on the
  host's binary rounding, and never clamped;
* every state but the exit leaves with total probability exactly 1 -- mass is neither created
  nor lost;
* a cost is a non-negative integer, the planner's unit;
* every state the entry reaches can reach the exit. A program that can be trapped -- a loop
  whose exit is never taken, a cycle with no edge out -- has no finite expected cost, and
  `NonTerminating` names the states it can be trapped in, as `tropical.NegativeCycle` names
  a diverging cycle.

**Where probabilities come from.** A declared probability is a modelling assumption; a
measured one is evidence. `BranchProfile` holds measured branch outcomes as counts (taken,
observed), estimates each probability as taken/observed exactly, and refuses a branch nobody
observed instead of guessing one half. `BranchProfile.component()` is what the plan's
`ExecutionScopeV1` records as its `U` component (uncertainty model and coverage): the model's
name, the estimator, and the counts themselves -- the sufficient statistics, from which any
interval is derived exactly rather than stored as a float.

The model assumes what a Markov chain assumes: each branch is decided by its own probability,
independently of the path that reached it. A loop whose trip count is correlated with what its
body did is outside it, and is declared that way, not approximated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Mapping


class ProbabilityError(ValueError):
    """A probability, a cost or a branch distribution that breaks a law above."""


class NonTerminating(ValueError):
    """A program the entry can leave trapped: its expected cost is not finite.

    Carries the states that cannot reach the exit, because the useful response is to look at
    them -- a loop whose exit probability is zero, or a cycle with no edge out."""

    def __init__(self, trapped: tuple[str, ...], message: str | None = None) -> None:
        super().__init__(
            message
            or f"the states {', '.join(trapped)} are reachable from the entry and cannot reach "
            f"the exit, so the program terminates with probability below 1 and its expected "
            f"cost is not finite"
        )
        self.trapped = trapped


def probability(value) -> Fraction:
    """`value` as an exact probability, or a `ProbabilityError`: an int or a Fraction in
    [0, 1]. A float is refused -- 0.1 is not one tenth, and a mean computed from it would
    depend on the host's rounding rather than on the program."""
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ProbabilityError(
            f"a probability is an exact rational (int or Fraction), not {type(value).__name__}"
        )
    p = Fraction(value)
    if not 0 <= p <= 1:
        raise ProbabilityError(f"a probability lies in [0, 1], got {p}")
    return p


def _cost(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbabilityError(f"a cost is a non-negative integer, got {value!r}")
    return value


@dataclass(frozen=True)
class MarkovProgram:
    """A control-flow graph whose edges carry (probability, cost): `edges[(src, dst)]`.

    `entry` is where the program starts and `exit` where it ends; the exit has no outgoing
    edge, and every other state the entry reaches leaves with total probability exactly 1.
    Parallel edges are one edge: two ways from `a` to `b` at different costs are a branch, and
    the caller writes it as one with its mean cost or splits it through a state of its own."""

    edges: Mapping[tuple[str, str], tuple]
    entry: str
    exit: str
    _out: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        out: dict[str, list[tuple[str, Fraction, int]]] = {}
        for (src, dst), spec in sorted(self.edges.items()):
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise ProbabilityError(f"the edge {src} -> {dst} carries (probability, cost)")
            p, c = probability(spec[0]), _cost(spec[1])
            if src == self.exit:
                raise ProbabilityError(f"the exit {self.exit} is absorbing: it has no edge out")
            out.setdefault(src, []).append((dst, p, c))
        object.__setattr__(self, "_out", {s: tuple(e) for s, e in out.items()})
        for state in self.reachable():
            if state == self.exit:
                continue
            total = sum((p for _dst, p, _c in self._out.get(state, ())), Fraction(0))
            if total != 1:
                raise ProbabilityError(
                    f"the state {state} leaves with total probability {total}, not 1: "
                    f"a branch distribution neither creates nor loses mass"
                )

    def successors(self, state: str) -> tuple[tuple[str, Fraction, int], ...]:
        return self._out.get(state, ())

    def reachable(self) -> tuple[str, ...]:
        """The states the entry reaches through edges taken with positive probability."""
        seen, stack = {self.entry}, [self.entry]
        while stack:
            for dst, p, _c in self._out.get(stack.pop(), ()):
                if p and dst not in seen:
                    seen.add(dst)
                    stack.append(dst)
        return tuple(sorted(seen))

    def transient(self) -> tuple[str, ...]:
        """The reachable states other than the exit, in a fixed order; `NonTerminating` names
        those that cannot reach the exit."""
        reach = self.reachable()
        into: dict[str, set[str]] = {}
        for state in reach:
            for dst, p, _c in self._out.get(state, ()):
                if p:
                    into.setdefault(dst, set()).add(state)
        exits, stack = set(), [self.exit] if self.exit in reach else []
        exits.update(stack)
        while stack:
            for src in into.get(stack.pop(), ()):
                if src not in exits:
                    exits.add(src)
                    stack.append(src)
        trapped = tuple(s for s in reach if s not in exits)
        if trapped:
            raise NonTerminating(trapped)
        return tuple(s for s in reach if s != self.exit)


def _solve(matrix: list[list[Fraction]], rhs: list[Fraction]) -> list[Fraction]:
    """Gauss-Jordan elimination over exact rationals. `matrix` is I - Q for a chain that
    terminates from every state, which makes it non-singular; a zero pivot column therefore
    means the caller broke that precondition, and is said so rather than divided by."""
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)
        if pivot is None:
            raise ArithmeticError("I - Q is singular: the chain does not terminate")
        a[col], a[pivot] = a[pivot], a[col]
        inv = 1 / a[col][col]
        a[col] = [v * inv for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [v - factor * w for v, w in zip(a[r], a[col])]
    return [a[i][n] for i in range(n)]


def expected_visits(program: MarkovProgram) -> dict[str, Fraction]:
    """The expected number of times each transient state is entered, starting from the entry
    (the entry's own start counts once): the entry's row of the fundamental matrix
    N = (I - Q)^-1. A loop header's value is the loop's expected trip count."""
    states = program.transient()
    index = {s: i for i, s in enumerate(states)}
    n = len(states)
    # v = e_entry + Q^T v  <=>  (I - Q^T) v = e_entry
    matrix = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for s in states:
        for dst, p, _c in program.successors(s):
            if dst in index:
                matrix[index[dst]][index[s]] -= p
    rhs = [Fraction(int(s == program.entry)) for s in states]
    visits = _solve(matrix, rhs)
    return {s: visits[index[s]] for s in states}


def expected_cost(program: MarkovProgram) -> Fraction:
    """The exact expected cost from the entry to the exit: every transient state's expected
    visits times the mean cost of leaving it once."""
    visits = expected_visits(program)
    return sum(
        (
            visits[s] * sum((p * c for _dst, p, c in program.successors(s)), Fraction(0))
            for s in visits
        ),
        Fraction(0),
    )


def loop_trips(p_continue, max_trips: int | None = None) -> Fraction:
    """The expected trip count of a do-while loop: the body runs once, then again with
    probability `p_continue` each time, at most `max_trips` times when a bound is declared.
    Unbounded: 1 / (1 - q). Bounded at N: (1 - q^N) / (1 - q), or N when q = 1. A loop that
    always continues and declares no bound never terminates."""
    q = probability(p_continue)
    if max_trips is not None:
        if isinstance(max_trips, bool) or not isinstance(max_trips, int) or max_trips < 1:
            raise ProbabilityError(
                f"a loop runs its body at least once, got max_trips={max_trips!r}"
            )
        if q == 1:
            return Fraction(max_trips)
        return (1 - q**max_trips) / (1 - q)
    if q == 1:
        raise NonTerminating(
            (),
            "a loop that continues with probability 1 and declares no bound never exits: its "
            "expected trip count is not finite",
        )
    return 1 / (1 - q)


@dataclass(frozen=True)
class BranchProfile:
    """Measured branch outcomes: `counts[name] = (taken, observed)`.

    The estimate of a branch's probability is taken/observed, exactly (the maximum-likelihood
    estimate of a Bernoulli parameter). A branch with no observation has no estimate, and
    asking for one is refused -- the one-half a declared default would invent is exactly what
    this record exists to replace."""

    counts: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        clean = {}
        for name, pair in sorted(self.counts.items()):
            if not isinstance(name, str) or not name:
                raise ProbabilityError("a branch is named by a non-empty string")
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ProbabilityError(f"the branch {name} carries (taken, observed)")
            taken, observed = pair
            for v in pair:
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ProbabilityError(f"the branch {name}'s counts are integers")
            if not 0 <= taken <= observed:
                raise ProbabilityError(f"the branch {name}: 0 <= taken <= observed, got {pair}")
            if observed == 0:
                raise ProbabilityError(
                    f"the branch {name} was never observed: it has no measured probability"
                )
            clean[name] = (taken, observed)
        object.__setattr__(self, "counts", clean)

    def __contains__(self, name: str) -> bool:
        return name in self.counts

    def probability(self, name: str) -> Fraction:
        if name not in self.counts:
            raise ProbabilityError(f"the branch {name} has no measurement in this profile")
        taken, observed = self.counts[name]
        return Fraction(taken, observed)

    def component(self) -> dict:
        """The `U` component of an `ExecutionScopeV1`: what the expectation assumed, in a form
        the scope digests (no floats): the model, the estimator, and per branch the counts it
        was estimated from."""
        return {
            "model": "branch-frequencies",
            "estimator": "taken/observed (maximum likelihood), exact",
            "branches": [[name, t, n] for name, (t, n) in self.counts.items()],
            "coverage": "the counts are the sufficient statistics; an interval is derived from "
            "them, not stored",
        }


__all__ = [
    "BranchProfile",
    "MarkovProgram",
    "NonTerminating",
    "ProbabilityError",
    "expected_cost",
    "expected_visits",
    "loop_trips",
    "probability",
]
