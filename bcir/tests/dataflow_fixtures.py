"""Fixtures of G23 / G24 (SDF/CSDF and timed-event regions), shared by the tests and the GEM+
harness.

`CORPUS` holds every case's expected model derived by hand (the derivation is in the entry).
`random_sdf` builds consistent graphs from a chosen repetition vector; `random_timed` builds
strongly connected homogeneous graphs; `max_cycle_ratio` enumerates every simple cycle of a
timed graph -- an independent derivation of the cycle time Karp's theorem gives -- and
`self_timed` runs the max-plus recurrence itself, so the tests hold the cycle time to the
execution it bounds.
"""

from __future__ import annotations

import random
from fractions import Fraction as F
from math import gcd

from bcir.model import Claim, Module, Opcode, Phase, Resource, StreamRate, Timing


def actor(i, rd=(), wr=(), consume=(), produce=(), initial=(), latency=0) -> Claim:
    return Claim(
        id=i,
        opcode=Opcode.ADD,
        rd=tuple(rd),
        wr=tuple(wr),
        op="stream.actor",
        count=64,
        stream=StreamRate(tuple(consume), tuple(produce), tuple(initial)),
        timing=Timing(latency_cycles=latency) if latency else None,
    )


def module_of(*phases_claims, name="dataflow") -> Module:
    """A module whose phases hold the given claim lists, with every rid they touch declared."""
    m = Module(name=name)
    rids = sorted({r for claims in phases_claims for c in claims for r in (*c.rd, *c.wr)})
    for rid in rids:
        m.add_resource(Resource(rid=rid, shape=(1024,)))
    for pid, claims in enumerate(phases_claims):
        m.add_phase(Phase(phase_id=pid, deps=(pid - 1,) if pid else (), claims=list(claims)))
    return m


def _chain(rates, base=100):
    """A chain of actors 1..n+1 with rates [(produce, consume), ...] on FIFOs base.."""
    out = []
    n = len(rates)
    for i in range(n + 1):
        consume = ((base + i - 1, (rates[i - 1][1],)),) if i else ()
        produce = ((base + i, (rates[i][0],)),) if i < n else ()
        out.append(
            actor(
                i + 1,
                rd=(base + i - 1,) if i else (),
                wr=(base + i,) if i < n else (),
                consume=consume,
                produce=produce,
            )
        )
    return out


def _ring(latencies, tokens, base=200):
    """A ring 1 -> 2 -> ... -> n -> 1 of homogeneous actors; `tokens` initial tokens on the
    channel closing the ring."""
    n = len(latencies)
    out = []
    for i in range(n):
        into, outof = base + (i - 1) % n, base + i
        out.append(
            actor(
                i + 1,
                rd=(into,),
                wr=(outof,),
                consume=((into, (1,)),),
                produce=((outof, (1,)),),
                initial=((outof, tokens),) if i == n - 1 else (),
                latency=latencies[i],
            )
        )
    return out


#: (name, claims, expected). `expected` is the region's kind and, for sdf, the repetition
#: vector, the firings and the FIFO bounds of the round-robin schedule; for timed, the cycle
#: time; for a refusal, its name.
CORPUS = (
    (
        "multirate pair 2:3",
        lambda: [
            actor(1, wr=(10,), produce=((10, (2,)),)),
            actor(2, rd=(10,), consume=((10, (3,)),)),
        ],
        {"kind": "sdf", "repetition": {1: 3, 2: 2}, "buffers": {10: 4}},
        "q = (3, 2) balances 3 * 2 = 2 * 3; round robin: A 2, A 4 -> B 1, A 3 -> B 0: peak 4",
    ),
    (
        "CD to DAT",
        lambda: _chain([(1, 1), (2, 3), (2, 7), (8, 7), (5, 1)]),
        {"kind": "sdf", "repetition": {1: 147, 2: 147, 3: 98, 4: 28, 5: 32, 6: 160}},
        "Bhattacharyya's sample-rate converter: 147 * 2 = 98 * 3, 98 * 2 = 28 * 7, "
        "28 * 8 = 32 * 7, 32 * 5 = 160 * 1",
    ),
    (
        "cycle with one delay",
        lambda: [
            actor(1, rd=(21,), wr=(20,), consume=((21, (1,)),), produce=((20, (1,)),)),
            actor(
                2,
                rd=(20,),
                wr=(21,),
                consume=((20, (1,)),),
                produce=((21, (1,)),),
                initial=((21, 1),),
            ),
        ],
        {"kind": "sdf", "repetition": {1: 1, 2: 1}, "buffers": {20: 1, 21: 1}},
        "the delay lets A fire first, then B returns the token",
    ),
    (
        "cyclo-static downsampler",
        lambda: [
            actor(1, wr=(50,), produce=((50, (1,)),)),
            actor(2, rd=(50,), wr=(51,), consume=((50, (1, 1)),), produce=((51, (1, 0)),)),
            actor(3, rd=(51,), consume=((51, (1,)),)),
        ],
        {"kind": "sdf", "repetition": {1: 2, 2: 1, 3: 1}, "firings": {1: 2, 2: 2, 3: 1}},
        "D's cycle consumes 2 and produces 1: two A firings and one B per D cycle (two phases)",
    ),
    (
        "cycle without a delay",
        lambda: [
            actor(1, rd=(21,), wr=(20,), consume=((21, (1,)),), produce=((20, (1,)),)),
            actor(2, rd=(20,), wr=(21,), consume=((20, (1,)),), produce=((21, (1,)),)),
        ],
        {"kind": "opaque", "refusal": "deadlock"},
        "each actor waits on the other's first token",
    ),
    (
        "inconsistent triangle",
        lambda: [
            actor(1, wr=(30, 32), produce=((30, (1,)), (32, (2,)))),
            actor(2, rd=(30,), wr=(31,), consume=((30, (1,)),), produce=((31, (1,)),)),
            actor(3, rd=(31, 32), consume=((31, (1,)), (32, (1,)))),
        ],
        {"kind": "opaque", "refusal": "inconsistent-rates"},
        "q_A = q_B = q_C by the first two FIFOs, 2 q_A = q_C by the third",
    ),
    (
        "open stream",
        lambda: [actor(1, wr=(40,), produce=((40, (1,)),))],
        {"kind": "opaque", "refusal": "open-stream"},
        "the FIFO's consumer is not in the region",
    ),
    (
        "two-phase source, one-phase sink",
        lambda: [
            actor(1, wr=(45,), produce=((45, (1, 1)),)),
            actor(2, rd=(45,), consume=((45, (1,)),)),
        ],
        {"kind": "sdf", "repetition": {1: 1, 2: 2}, "firings": {1: 2, 2: 2}, "buffers": {45: 1}},
        "one source cycle (two phases) makes 2 tokens, the sink takes 1 per firing: q = (1, 2); "
        "round robin alternates them, so the FIFO never holds more than 1",
    ),
    (
        "an actor whose ports disagree on phases",
        lambda: [
            actor(1, wr=(47,), produce=((47, (1,)),)),
            actor(2, rd=(47,), wr=(48,), consume=((47, (1,)),), produce=((48, (1, 0)),)),
            actor(3, rd=(48,), consume=((48, (1,)),)),
        ],
        {"kind": "opaque", "refusal": "rate"},
        "actor 2 lists one phase on its input and two on its output: it has no phase count",
    ),
    (
        "timed pipeline with feedback",
        lambda: [
            actor(1, rd=(61,), wr=(60,), consume=((61, (1,)),), produce=((60, (1,)),), latency=3),
            actor(
                2,
                rd=(60,),
                wr=(61,),
                consume=((60, (1,)),),
                produce=((61, (1,)),),
                initial=((61, 2),),
                latency=5,
            ),
        ],
        {"kind": "timed", "cycle_time": F(5)},
        "cycles: A's self 3/1, B's self 5/1, A->B->A (3 + 5) / 2 = 4: the slowest is B alone",
    ),
    (
        "timed ring, one token",
        lambda: _ring((2, 4, 3), 1),
        {"kind": "timed", "cycle_time": F(9)},
        "the ring carries 2 + 4 + 3 = 9 cycles of latency per token",
    ),
    (
        "timed ring, two tokens",
        lambda: _ring((2, 4, 3), 2),
        {"kind": "timed", "cycle_time": F(9, 2)},
        "9 / 2 = 4.5 beats B's own 4",
    ),
    (
        "timed ring, three tokens",
        lambda: _ring((2, 4, 3), 3),
        {"kind": "timed", "cycle_time": F(4)},
        "9 / 3 = 3 is below B's own latency 4: an actor never overlaps itself",
    ),
    (
        "homogeneous but untimed",
        lambda: _ring((0, 0, 0), 1),
        {"kind": "sdf", "repetition": {1: 1, 2: 1, 3: 1}},
        "no latency declared: the region has no cycle time to state, so it is sdf",
    ),
)


def corpus():
    """(name, module, expected, derivation) for every entry; each case is one phase."""
    return [(n, module_of(build(), name=n), want, why) for n, build, want, why in CORPUS]


def judged(module, want) -> list[str]:
    """How the region graph of `module` (one region expected) differs from `want`."""
    from bcir.kbcir.regions import region_graph

    regions = region_graph(module).regions
    if len(regions) != 1:
        return [f"{len(regions)} regions, not 1"]
    region = regions[0]
    out = []
    if region.kind != want["kind"]:
        out.append(f"kind {region.kind} != {want['kind']}")
        return out
    if "refusal" in want and region.refusal != want["refusal"]:
        out.append(f"refusal {region.refusal!r} != {want['refusal']!r}")
    sdf = None if region.model is None else getattr(region.model, "sdf", region.model)
    for key in ("repetition", "firings", "buffers"):
        if key in want and dict(getattr(sdf, key)) != want[key]:
            out.append(f"{key} {dict(getattr(sdf, key))} != {want[key]}")
    if "cycle_time" in want and region.model.cycle_time != want["cycle_time"]:
        out.append(f"cycle time {region.model.cycle_time} != {want['cycle_time']}")
    return out


def random_sdf(rng: random.Random):
    """A connected, consistent SDF graph: a random spanning tree with rates chosen from a
    random repetition vector, plus back channels carrying enough tokens to stay live."""
    n = rng.randint(2, 6)
    q = [rng.randint(1, 6) for _ in range(n)]
    ports = {i: {"consume": [], "produce": [], "initial": [], "rd": [], "wr": []} for i in range(n)}
    rid = 300
    for v in range(1, n):
        u = rng.randrange(v)  # a tree edge u -> v
        g = gcd(q[u], q[v])
        k = rng.randint(1, 3)
        p, c = k * q[v] // g, k * q[u] // g  # q[u] p = q[v] c
        ports[u]["produce"].append((rid, (p,)))
        ports[u]["wr"].append(rid)
        ports[v]["consume"].append((rid, (c,)))
        ports[v]["rd"].append(rid)
        rid += 1
        if rng.random() < 0.4:  # a back channel v -> u with enough delay for one iteration
            ports[v]["produce"].append((rid, (c,)))
            ports[v]["wr"].append(rid)
            ports[u]["consume"].append((rid, (p,)))
            ports[u]["rd"].append(rid)
            ports[v]["initial"].append((rid, q[u] * p))
            rid += 1
    return [
        actor(
            i + 1,
            rd=ports[i]["rd"],
            wr=ports[i]["wr"],
            consume=ports[i]["consume"],
            produce=ports[i]["produce"],
            initial=ports[i]["initial"],
        )
        for i in range(n)
    ]


def random_timed(rng: random.Random):
    """A strongly connected homogeneous timed graph: a ring with at least one token, plus
    chords, every actor with a latency."""
    n = rng.randint(1, 5)
    latency = [rng.randint(1, 9) for _ in range(n)]
    edges = [(i, (i + 1) % n, 0) for i in range(n)]
    edges[-1] = (n - 1, 0, rng.randint(1, 3))
    for _ in range(rng.randint(0, 3)):
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.append((u, v, rng.randint(0 if u < v else 1, 2)))
    ports = {i: {"consume": [], "produce": [], "initial": [], "rd": [], "wr": []} for i in range(n)}
    for k, (u, v, d) in enumerate(edges):
        rid = 400 + k
        ports[u]["produce"].append((rid, (1,)))
        ports[u]["wr"].append(rid)
        if d:
            ports[u]["initial"].append((rid, d))
        ports[v]["consume"].append((rid, (1,)))
        ports[v]["rd"].append(rid)
    claims = [
        actor(
            i + 1,
            rd=ports[i]["rd"],
            wr=ports[i]["wr"],
            consume=ports[i]["consume"],
            produce=ports[i]["produce"],
            initial=ports[i]["initial"],
            latency=latency[i],
        )
        for i in range(n)
    ]
    return claims, [(u + 1, v + 1, d) for u, v, d in edges], {i + 1: latency[i] for i in range(n)}


def max_cycle_ratio(edges, latency) -> F | None:
    """The maximum over every simple cycle (self-cycles of one token per actor included) of
    latency over tokens, by enumeration; None when a cycle carries no token."""
    all_edges = list(edges) + [(a, a, 1) for a in latency]
    nodes = sorted(latency)
    best = None
    for start in nodes:
        stack = [(start, [], frozenset([start]))]
        while stack:
            node, path, seen = stack.pop()
            for k, (u, v, d) in enumerate(all_edges):
                if u != node:
                    continue
                if v == start:
                    cycle = path + [k]
                    tokens = sum(all_edges[i][2] for i in cycle)
                    work = sum(latency[all_edges[i][0]] for i in cycle)
                    if tokens == 0:
                        return None
                    ratio = F(work, tokens)
                    if best is None or ratio > best:
                        best = ratio
                elif v > start and v not in seen:
                    stack.append((v, path + [k], seen | {v}))
    return best


def self_timed(edges, latency, iterations: int) -> list[dict[int, int]]:
    """Start times x_v(k) of the self-timed execution: firing k of v starts when, for every
    channel u -> v with d tokens, firing k - d of u has finished (tokens initially present are
    available at time 0), and v's own firing k - 1 has finished."""
    all_edges = list(edges) + [(a, a, 1) for a in latency]
    zero = {v: [u for u, w, d in all_edges if w == v and d == 0] for v in latency}
    order, done = [], set()
    while len(order) < len(latency):  # a topological order of the zero-token edges
        for v in sorted(latency):
            if v not in done and all(u in done for u in zero[v]):
                order.append(v)
                done.add(v)
    x: list[dict[int, int]] = []
    for k in range(iterations):
        row: dict[int, int] = {}
        for v in order:
            start = 0
            for u, w, d in all_edges:
                if w != v or k - d < 0:
                    continue
                ready = (row[u] if d == 0 else x[k - d][u]) + latency[u]
                start = max(start, ready)
            row[v] = start
        x.append(row)
    return x


def measure() -> dict[str, float]:
    """The G23 / G24 harness rows: corpus cases whose region differs from the hand-derived
    model, and generated timed graphs whose cycle time differs from the enumeration."""
    from bcir.kbcir.dataflow import sdf_model, timed_model

    misjudged = sum(1 for _n, module, want, _w in corpus() if judged(module, want))
    disagreements = 0
    for seed in range(60):
        claims, edges, latency = random_timed(random.Random(seed))
        want = max_cycle_ratio(edges, latency)
        got = timed_model(sdf_model(claims), latency).cycle_time
        disagreements += got != want
    return {
        "regions.dataflow.misjudged": float(misjudged),
        "regions.timed.cycle_time.disagreements": float(disagreements),
    }
