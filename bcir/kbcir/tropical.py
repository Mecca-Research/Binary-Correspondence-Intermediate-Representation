"""P4 — control flow as a cost problem, in the min-plus semiring.

§5 of `docs/BCIR_JSON_PROGRAM_REPRESENTATION.md` calls this *"the technically strongest part
of the proposal"* and asks for it to be *"implemented as mathematics rather than as
heuristics"*. That is the whole design brief, and it has consequences that a heuristic
version would not have.

In the min-plus semiring `(R ∪ {+∞}, min, +)`:

* **straight-line composition** is semiring multiplication — costs add along a path;
* **a conditional** is semiring addition — `min` over the alternatives, which is why an
  `if/else` maps naturally rather than by force;
* **a loop is the Kleene closure** `a* = min_k a^k`. Over min-plus the closure of a
  non-negative cycle is the empty path (cost 0), and a **negative** cycle diverges to `-∞`;
* **optimal unrolling is a minimum mean cycle problem** — Karp (1978) — not a search.

**A negative cycle is a refusal, not an answer.** §5 puts it exactly right: divergence is
"the algebra telling you the cost model is wrong rather than that the loop is free". A pass
that returned `-∞`, or clamped it to zero, would be reporting that going round a loop makes
a program cheaper — so `close` and `shortest_paths` raise `NegativeCycle` naming the cycle.
That is the one place in this module where raising is correct: there is no cost to report.

**The semiring is a declared parameter, and §5's first limit is why.** `min` optimizes a
*path*; a program is a *distribution over paths*. Without branch probabilities `min` picks
the best case, which is the wrong answer for expected cost — so `Semiring.MIN_PLUS` is a
choice a caller makes, `Semiring.MAX_PLUS` is available for worst-case real-time work, and
neither is the default hidden inside a function. A module that hard-coded `min` would be
answering a question the caller did not ask.

**Legality first, unchanged.** §5's second limit: this is a legality-preserving optimization
only over a graph that was legality-checked first. Nothing here consults a verifier, and
nothing here may be read as making an illegal program legal — `unroll_factor` returns a
count, not a permission.

Integer arithmetic throughout. The costs are Q16 fixed point like the rest of K_BCIR, so a
mean cycle weight is an exact rational rather than a float whose comparison depends on the
host.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

#: The additive identity of min-plus — the cost of an impossible path. Represented rather
#: than approximated: `None` would force every caller to write the same three-line guard.
INFINITY = None


class Semiring(Enum):
    """Which algebra the caller is asking about. §5's limit 1, made a parameter.

    MIN_PLUS answers "what is the cheapest path" — the best case, correct for a program with
    one path or for a lower bound. MAX_PLUS answers "what is the most expensive path" — the
    worst case, which is what a real-time budget needs. Neither is the expected cost, and
    this module deliberately does not offer one: expectation needs branch probabilities, at
    which point it is a Markov chain rather than a shortest path, and pretending otherwise
    is exactly the overclaim §5 warns about.
    """

    MIN_PLUS = "min-plus"
    MAX_PLUS = "max-plus"

    def better(self, a, b) -> bool:
        """Whether `a` is preferred to `b`, with None as the identity (no path yet)."""
        if b is INFINITY:
            return a is not INFINITY
        if a is INFINITY:
            return False
        return a < b if self is Semiring.MIN_PLUS else a > b


class NegativeCycle(Exception):
    """A cycle whose weight improves without bound — the cost model is wrong.

    Its own class, and it carries the cycle, because the useful response is to look at those
    edges rather than to retry. Under MIN_PLUS this is a negative cycle; under MAX_PLUS it
    is a positive one, and the name is kept because the *condition* is the same: the closure
    does not converge.
    """

    def __init__(self, cycle: tuple[str, ...], weight: int) -> None:
        super().__init__(
            f"the cycle {' -> '.join(cycle)} has total weight {weight}, so its closure "
            f"diverges; a loop cannot make a program cheaper without bound, so this is the "
            f"cost model being wrong rather than the loop being free")
        self.cycle = cycle
        self.weight = weight


@dataclass(frozen=True)
class CostGraph:
    """A control-flow graph whose edges carry a scalar cost projected from the 12 axes.

    The projection onto a scalar happens BEFORE this type: §6.1 keeps the twelve axes and
    this module optimizes one objective at a time, so a caller who wants a different
    objective builds a different graph rather than passing a weight vector nobody can
    compare lexicographically without inventing a policy.
    """

    #: (source, target) -> cost. Parallel edges are not representable, and do not need to
    #: be: two paths between the same pair of blocks are a conditional, which is `min` of
    #: their costs, so the caller has already reduced them.
    edges: dict[tuple[str, str], int]
    nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        named = set(self.nodes)
        for source, target in self.edges:
            named.add(source)
            named.add(target)
        object.__setattr__(self, "nodes", tuple(sorted(named)))

    def successors(self, node: str) -> tuple[tuple[str, int], ...]:
        return tuple((t, c) for (s, t), c in sorted(self.edges.items()) if s == node)


def compose(*costs: int) -> int:
    """Semiring multiplication: straight-line composition adds."""
    return sum(costs)


def alternative(semiring: Semiring, *costs) -> int | None:
    """Semiring addition: a conditional is `min` (or `max`) over its arms."""
    best = INFINITY
    for cost in costs:
        if semiring.better(cost, best):
            best = cost
    return best


def shortest_paths(graph: CostGraph, source: str, *,
                   semiring: Semiring = Semiring.MIN_PLUS) -> dict[str, int]:
    """Bellman–Ford, whose negative-cycle detection IS §5's divergence check.

    Dijkstra would be faster and is only correct for non-negative weights; a cost model with
    a negative edge is exactly the case worth detecting, so the slower algorithm that can
    detect it is the right one. `V-1` relaxation rounds suffice for any simple path, so a
    `V`-th round that still improves proves a cycle.
    """
    if source not in graph.nodes:
        raise ValueError(f"{source!r} is not a node of this graph")
    best: dict[str, int | None] = {node: INFINITY for node in graph.nodes}
    best[source] = 0
    predecessor: dict[str, str] = {}
    for _round in range(max(len(graph.nodes) - 1, 0)):
        changed = False
        for (a, b), cost in sorted(graph.edges.items()):
            if best[a] is INFINITY:
                continue
            candidate = compose(best[a], cost)
            if semiring.better(candidate, best[b]):
                best[b] = candidate
                predecessor[b] = a
                changed = True
        if not changed:
            break
    # One more round: any further improvement is reachable only through a diverging cycle.
    for (a, b), cost in sorted(graph.edges.items()):
        if best[a] is INFINITY:
            continue
        if semiring.better(compose(best[a], cost), best[b]):
            raise NegativeCycle(*_recover_cycle(predecessor, a, b, graph))
    return {node: cost for node, cost in best.items() if cost is not INFINITY}


def _recover_cycle(predecessor: dict[str, str], a: str, b: str,
                   graph: CostGraph) -> tuple[tuple[str, ...], int]:
    """Walk predecessors back into the cycle so the exception can name it.

    Reporting "there is a negative cycle somewhere" would leave the caller to find it; the
    edges are right here.
    """
    seen: list[str] = []
    node = a
    for _ in range(len(graph.nodes) + 1):
        seen.append(node)
        if node not in predecessor:
            break
        node = predecessor[node]
        if node in seen:
            cycle = seen[seen.index(node):]
            cycle.reverse()
            cycle.append(cycle[0])
            weight = sum(graph.edges.get((cycle[i], cycle[i + 1]), 0)
                         for i in range(len(cycle) - 1))
            return tuple(cycle), weight
    return (a, b), graph.edges.get((a, b), 0)


def close(weight: int, *, semiring: Semiring = Semiring.MIN_PLUS) -> int:
    """The Kleene closure of a single self-loop: `a* = min_k a^k`.

    Over min-plus a non-negative weight closes to **0** — the empty path, going round zero
    times — which is the algebra's way of saying a loop you are not forced to enter costs
    nothing. A negative weight has no closure at all, and that is the refusal above.
    """
    if semiring is Semiring.MIN_PLUS:
        if weight < 0:
            raise NegativeCycle(("self", "self"), weight)
        return 0
    if weight > 0:
        raise NegativeCycle(("self", "self"), weight)
    return 0


def minimum_mean_cycle(graph: CostGraph) -> tuple[Fraction, tuple[str, ...]] | None:
    """Karp (1978): the minimum mean cycle weight, exactly.

    Returns `(mean, cycle)` or None when the graph is acyclic. The mean is a `Fraction`
    rather than a float because it is a ratio of integers and the comparison that picks the
    winner must not depend on the host's rounding — the same reason `certified.py` computes
    coverage in exact integers.

    Karp's theorem: over the DAG of `k`-edge walks from a source that reaches every node,
    the minimum mean cycle is `min over v of max over k of (D[n][v] - D[k][v]) / (n - k)`.
    The implementation is that formula, not an approximation of it.
    """
    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        return None
    index = {node: i for i, node in enumerate(nodes)}
    # A virtual source reaching every node with cost 0, so every cycle is reachable and the
    # theorem's precondition holds without the caller having to supply an entry block.
    big: list[list[int | None]] = [[INFINITY] * n for _ in range(n + 1)]
    parent: list[list[int | None]] = [[None] * n for _ in range(n + 1)]
    for i in range(n):
        big[0][i] = 0
    for k in range(1, n + 1):
        for (a, b), cost in sorted(graph.edges.items()):
            i, j = index[a], index[b]
            if big[k - 1][i] is INFINITY:
                continue
            candidate = big[k - 1][i] + cost
            if big[k][j] is INFINITY or candidate < big[k][j]:
                big[k][j] = candidate
                parent[k][j] = i

    best: tuple[Fraction, int] | None = None
    for v in range(n):
        if big[n][v] is INFINITY:
            continue
        worst: Fraction | None = None
        for k in range(n):
            if big[k][v] is INFINITY:
                continue
            ratio = Fraction(big[n][v] - big[k][v], n - k)
            if worst is None or ratio > worst:
                worst = ratio
        if worst is not None and (best is None or worst < best[0]):
            best = (worst, v)
    if best is None:
        return None

    # Walk the parent chain back to recover an actual cycle for the caller to look at.
    mean, v = best
    seen: dict[int, int] = {}
    k = n
    node = v
    while k >= 0 and node is not None and node not in seen:
        seen[node] = k
        node = parent[k][node]
        k -= 1
    if node is None:
        return mean, ()
    cycle = [node]
    walk, at = seen[node] - 1, parent[seen[node]][node]
    while at is not None and at != node and walk >= 0:
        cycle.append(at)
        at, walk = parent[walk][at], walk - 1
    cycle.append(node)
    cycle.reverse()
    return mean, tuple(nodes[i] for i in cycle)


def unroll_factor(graph: CostGraph, *, prologue: int, epilogue: int,
                  iterations: int, maximum: int = 64) -> int:
    """The unroll factor that minimizes total cost, derived rather than searched blindly.

    The steady-state cost per iteration is the minimum mean cycle weight; unrolling by `u`
    pays the prologue and epilogue once per `ceil(iterations / u)` chunks while the body
    cost per iteration is unchanged. So total cost is

        chunks(u) * (prologue + epilogue) + iterations * mean

    and the second term does not depend on `u` — which is the useful, slightly deflating
    result: **for a loop whose body cost is unaffected by unrolling, the optimum is bounded
    entirely by the overhead term.** A pass that reported a large speedup from unrolling a
    body it did not change would be reporting an artefact.

    Returns 1 for an acyclic graph: there is no loop to unroll, and returning 0 or raising
    would make the caller special-case a perfectly ordinary program.
    """
    if prologue < 0 or epilogue < 0:
        raise ValueError("prologue and epilogue costs cannot be negative")
    if iterations <= 0:
        return 1
    found = minimum_mean_cycle(graph)
    if found is None:
        return 1
    mean, cycle = found
    if mean < 0:
        raise NegativeCycle(cycle, int(mean * len(cycle)))
    overhead = prologue + epilogue
    if overhead == 0:
        # Nothing to amortize; unrolling buys exactly nothing and saying so is the honest
        # answer rather than picking the largest factor because it "cannot hurt".
        return 1
    # The SMALLEST factor achieving the minimum, and the strict `<` is what makes it so.
    # Several factors usually tie -- with 100 iterations and a 64 cap, everything from 50 to
    # 64 costs two chunks -- and a larger one grows code for no gain in the objective. Code
    # size is a dimension this scalar cost does not carry, so breaking the tie towards less
    # of it is the honest default; a caller who wants otherwise passes a different maximum.
    best_factor, best_cost = 1, None
    for factor in range(1, min(maximum, iterations) + 1):
        chunks = -(-iterations // factor)
        cost = chunks * overhead
        if best_cost is None or cost < best_cost:
            best_factor, best_cost = factor, cost
    return best_factor


__all__ = [
    "INFINITY", "CostGraph", "NegativeCycle", "Semiring", "alternative", "close",
    "compose", "minimum_mean_cycle", "shortest_paths", "unroll_factor",
]
