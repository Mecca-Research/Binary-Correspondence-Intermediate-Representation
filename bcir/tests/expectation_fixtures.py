"""Fixtures of G22 (expected cost), shared by the tests and the GEM+ harness.

`CORPUS` is every program shape the expectation has to get right, each with its mean derived by
hand -- the derivation is in the entry, so a reader checks the arithmetic, not the code.
`region_cfg` compiles a `compose` region tree to a `MarkovProgram` by a route that shares no
arithmetic with `compose.plan_composite` (it unrolls a bounded loop into its trips instead of
summing a geometric series), so the generated differential holds two independent derivations
of one mean to each other.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

from bcir.kbcir.compose import PRED_COST, Cond, Leaf, Loop, Seq
from bcir.kbcir.expectation import MarkovProgram

#: (name, edges, entry, exit, mean, derivation). Edges map (src, dst) -> (probability, cost).
CORPUS = (
    (
        "if-else",
        {
            ("s", "a"): (F(1, 4), 10),
            ("s", "b"): (F(3, 4), 20),
            ("a", "x"): (1, 0),
            ("b", "x"): (1, 0),
        },
        "s",
        "x",
        F(35, 2),
        "1/4 * 10 + 3/4 * 20",
    ),
    (
        "do-while",
        {("h", "h"): (F(2, 3), 5), ("h", "x"): (F(1, 3), 5)},
        "h",
        "x",
        F(15),
        "trips 1 / (1 - 2/3) = 3, each costing 5",
    ),
    (
        "while (test first)",
        {("t", "b"): (F(3, 4), 1), ("t", "x"): (F(1, 4), 1), ("b", "t"): (1, 8)},
        "t",
        "x",
        F(28),
        "the test runs 1 / (1 - 3/4) = 4 times at 1, the body 3 times at 8: 4 + 24",
    ),
    (
        "nested loops",
        {
            ("o", "i"): (1, 2),
            ("i", "i"): (F(1, 2), 3),
            ("i", "e"): (F(1, 2), 3),
            ("e", "o"): (F(2, 3), 1),
            ("e", "x"): (F(1, 3), 1),
        },
        "o",
        "x",
        F(27),
        "outer trips 3, each 2 + (inner trips 2) * 3 + 1 = 9",
    ),
    (
        "two exits (break)",
        {
            ("h", "h"): (F(1, 2), 4),
            ("h", "brk"): (F(1, 4), 4),
            ("h", "x"): (F(1, 4), 4),
            ("brk", "x"): (1, 10),
        },
        "h",
        "x",
        F(13),
        "trips 2 at 4; the loop leaves by the break with probability (1/4)/(1/2) = 1/2, at 10",
    ),
    (
        "random walk",
        {
            ("1", "2"): (F(1, 2), 1),
            ("1", "x"): (F(1, 2), 1),
            ("2", "1"): (F(1, 2), 1),
            ("2", "x"): (F(1, 2), 1),
        },
        "1",
        "x",
        F(2),
        "gambler's ruin on 0..3 from 1: k (n - k) = 1 * 2 steps, each costing 1",
    ),
    (
        "bounded do-while (unrolled, 3 trips)",
        {
            ("b1", "b2"): (F(1, 2), 4),
            ("b1", "x"): (F(1, 2), 4),
            ("b2", "b3"): (F(1, 2), 4),
            ("b2", "x"): (F(1, 2), 4),
            ("b3", "x"): (1, 4),
        },
        "b1",
        "x",
        F(7),
        "trips (1 - (1/2)^3) / (1 - 1/2) = 7/4, each costing 4",
    ),
    (
        "a branch never taken",
        {("s", "a"): (0, 1000), ("s", "b"): (1, 3), ("a", "x"): (1, 0), ("b", "x"): (1, 0)},
        "s",
        "x",
        F(3),
        "the zero-probability edge costs nothing in expectation, and its target is not reached",
    ),
)


def corpus():
    """(name, MarkovProgram, mean, derivation) for every entry of CORPUS."""
    return [(n, MarkovProgram(e, s, x), mean, why) for n, e, s, x, mean, why in CORPUS]


def region_cfg(region, leaf_cost) -> MarkovProgram:
    """The control-flow graph of a compose region tree as a Markov program: a Leaf is one edge
    at its planned cost (`leaf_cost(leaf)`), a Seq a chain, a Cond a branch charging
    PRED_COST on each arm, and a Loop its `max_trips` trips UNROLLED -- after each trip the
    predicate (PRED_COST) continues to the next with its probability or leaves, and the last
    trip leaves for certain."""
    edges: dict = {}
    counter = [0]

    def fresh() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def link(a: str, b: str, p, c: int) -> None:
        key = (a, b)
        if key in edges:  # a parallel edge: route it through a state of its own
            mid = fresh()
            edges[(a, mid)] = (p, c)
            edges[(mid, b)] = (1, 0)
        else:
            edges[key] = (p, c)

    def build(r, a: str, b: str) -> None:
        if isinstance(r, Leaf):
            link(a, b, 1, leaf_cost(r))
        elif isinstance(r, Seq):
            if not r.parts:
                link(a, b, 1, 0)
                return
            cur = a
            for i, part in enumerate(r.parts):
                nxt = b if i == len(r.parts) - 1 else fresh()
                build(part, cur, nxt)
                cur = nxt
        elif isinstance(r, Cond):
            p = F(r.prob_then_milli, 1000)
            t, e = fresh(), fresh()
            link(a, t, p, PRED_COST)
            link(a, e, 1 - p, PRED_COST)
            build(r.then_, t, b)
            build(r.else_, e, b)
        elif isinstance(r, Loop):
            q = F(r.prob_continue_milli, 1000)
            start = a
            for trip in range(r.max_trips):
                done = fresh()
                build(r.body, start, done)
                if trip == r.max_trips - 1:
                    link(done, b, 1, PRED_COST)
                else:
                    nxt = fresh()
                    link(done, nxt, q, PRED_COST)
                    link(done, b, 1 - q, PRED_COST)
                    start = nxt
        else:  # pragma: no cover - the generator makes only the four shapes
            raise TypeError(type(r).__name__)

    build(region, "entry", "exit")
    return MarkovProgram(edges, "entry", "exit")


def random_region(rng: random.Random, leaves: tuple, depth: int = 0):
    """A random region tree over `leaves` (Seq / Cond / Loop / Leaf), bounded in depth and in
    loop trips so its unrolled graph stays small."""
    if depth >= 3 or rng.random() < 0.3:
        return rng.choice(leaves)
    kind = rng.choice(("seq", "cond", "loop"))
    if kind == "seq":
        return Seq(tuple(random_region(rng, leaves, depth + 1) for _ in range(rng.randint(1, 3))))
    if kind == "cond":
        return Cond(
            f"c{depth}",
            random_region(rng, leaves, depth + 1),
            random_region(rng, leaves, depth + 1),
            rng.choice((0, 1, 250, 333, 500, 999, 1000, rng.randint(0, 1000))),
        )
    return Loop(
        random_region(rng, leaves, depth + 1),
        f"l{depth}",
        rng.choice((0, 500, 750, 1000, rng.randint(0, 1000))),
        rng.randint(1, 3),
    )


def planning_context():
    """(leaves, resources, target, theta): three distinct straight-line leaves on the AVX-512
    target, whose planned costs differ, and what `plan_composite` plans them against."""
    from bcir.kbcir import TARGETS
    from bcir.kbcir.cost import Theta
    from bcir.model import Claim, Domain, Lane, Opcode, Resource, StrideClass

    resources = {r: Resource(rid=r, domain=Domain.RAM, shape=(1024,)) for r in range(16)}

    def claim(i, count, stride):
        return Claim(
            id=i,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=stride,
            count=count,
            rd=(2 * i, 2 * i + 1),
            wr=(15 - i,),
            op="vector.add",
            domain=Domain.RAM,
        )

    leaves = (
        Leaf((claim(1, 1024, StrideClass.UNIT),)),
        Leaf((claim(2, 256, StrideClass.STRIDED),)),
        Leaf((claim(3, 4096, StrideClass.CACHELINE), claim(4, 256, StrideClass.UNIT))),
    )
    return leaves, resources, TARGETS["x86_avx512"], Theta.cool()


def differential(seeds: int = 60) -> tuple[int, int]:
    """(regions compared, disagreements): `compose`'s exact expectation against the Markov
    solve of the region's unrolled control-flow graph, over `seeds` generated region trees."""
    from bcir.kbcir.compose import plan_composite
    from bcir.kbcir.expectation import expected_cost

    leaves, resources, target, theta = planning_context()
    # keyed by identity: a Leaf holds claims, which do not hash, and the generator reuses
    # these three objects, so identity names them
    costs = {
        id(leaf): plan_composite(leaf, {}, resources, target, theta).worst_cost for leaf in leaves
    }
    disagreements = 0
    for seed in range(seeds):
        region = random_region(random.Random(seed), leaves)
        planned = plan_composite(region, {}, resources, target, theta).expected
        if planned != expected_cost(region_cfg(region, lambda leaf: costs[id(leaf)])):
            disagreements += 1
    return seeds, disagreements


def measure() -> dict[str, float]:
    """The G22 harness rows: hand-derived means the solver misstates, and generated region
    trees on which compose and the Markov solve disagree."""
    from bcir.kbcir.expectation import expected_cost

    misstated = sum(1 for _n, program, mean, _why in corpus() if expected_cost(program) != mean)
    _compared, disagreements = differential()
    return {
        "expectation.corpus.misstated": float(misstated),
        "expectation.compose.disagreements": float(disagreements),
    }
