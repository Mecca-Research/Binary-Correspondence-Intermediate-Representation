"""Synchronous dataflow and timed-event semantics for fixed-rate streams (GEM+ G23 / G24).

The 2026-10-06 audit (items 11 and 12, Missing) found no repetition vector, balance equation
or bounded FIFO anywhere in the code -- fixed-rate claims fell to the opaque region -- and no
timed-event region, no cycle-time bound. This module is the local mathematics of the two
region kinds `kbcir.regions` recognizes from claims that declare a `StreamRate`:

**SDF / CSDF** (Lee and Messerschmitt 1987; Bilsen et al. 1996 for the cyclo-static case).
A claim with a `StreamRate` is an actor; a FIFO resource one actor produces into and another
consumes from is a channel. A cyclo-static actor fires its phases in turn, so a channel's
rates per CYCLE are the sums of its per-phase rates. `sdf_model` builds:

* the **balance equations** `q[p] * P = q[c] * C` per channel, solved exactly per connected
  component for the smallest positive integer **repetition vector** (in actor cycles). No
  positive solution is a refusal, `inconsistent-rates`: the FIFOs would grow or drain without
  bound, whatever the schedule;
* **liveness** by symbolic execution of one iteration with the declared initial tokens -- a
  round-robin data-driven schedule whose every sweep fires, in declared order, each actor that
  can fire and has firings left, once. A sweep that fires nothing before every actor has fired
  its repetition count is a refusal, `deadlock` (a cycle without enough initial tokens);
* the **FIFO bounds** that schedule needs: each channel's peak occupancy. They bound the
  buffers of THIS schedule; another schedule may need less, and none is claimed minimal.

**Timed-event** (max-plus; Baccelli, Cohen, Olsder and Quadrat 1992). A homogeneous region
-- every rate 1 -- whose every actor declares a latency (`Timing.latency_cycles`) is a timed
marked graph. Firing `k` of an actor starts when, for each input channel with `d` initial
tokens, firing `k - d` of its producer has finished; an actor does not overlap its own
firings (a self-channel with one token is implied, the standard assumption). `timed_model`
expands every channel to at most one token, builds the max-plus recurrence
`x(k) = A0 x(k) (+) A1 x(k-1)`, refuses a cycle with no token (`zero-token-cycle`: the
recurrence has no solution, the graph deadlocks), and returns the **cycle time**: the maximum
cycle mean of `A0* A1`, by Karp's theorem (`tropical.minimum_mean_cycle` on negated weights),
as an exact rational. It equals the maximum over the graph's cycles of latency over tokens,
which the tests enumerate independently, and it is a LOWER BOUND on the period of every
schedule: no execution completes iterations faster on average.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd

#: Why a run of stream claims is not a dataflow region (the refusal conditions, named).
DATAFLOW_REFUSALS = (
    "rate",
    "open-stream",
    "inconsistent-rates",
    "deadlock",
    "iteration-budget",
)

#: The firings one symbolic iteration may take before it is refused: a bound on the work a
#: verifier does, not a property of the graph.
ITERATION_BUDGET = 1 << 20


class DataflowError(ValueError):
    """A dataflow law the claims break; `refusal` names it (DATAFLOW_REFUSALS)."""

    def __init__(self, refusal: str, message: str) -> None:
        super().__init__(f"{refusal}: {message}")
        self.refusal = refusal


@dataclass(frozen=True)
class Channel:
    """A FIFO resource between two actors: per-phase rates on each side and its delays."""

    rid: int
    producer: int
    consumer: int
    produce: tuple[int, ...]
    consume: tuple[int, ...]
    initial: int


@dataclass(frozen=True)
class SdfModel:
    """The local model of an SDF/CSDF region: the actors in declared order, their phase counts,
    the channels, the repetition vector (actor cycles per iteration), the firings it implies,
    one live iteration's schedule, and the FIFO bounds that schedule needs."""

    actors: tuple[int, ...]
    phases: tuple[tuple[int, int], ...]
    channels: tuple[Channel, ...]
    repetition: tuple[tuple[int, int], ...]
    firings: tuple[tuple[int, int], ...]
    schedule: tuple[int, ...]
    buffers: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TimedModel:
    """The local model of a timed-event region: the SDF model it refines, each actor's
    latency, and the cycle time -- the exact maximum cycle mean of the max-plus recurrence, a
    lower bound on the period of every schedule -- with one critical cycle."""

    sdf: SdfModel
    latencies: tuple[tuple[int, int], ...]
    cycle_time: Fraction
    critical: tuple[str, ...]


def _rates(actor: int, ports, side: str) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for port in ports:
        if not isinstance(port, tuple) or len(port) != 2:
            raise DataflowError("rate", f"actor {actor}: a {side} port is (rid, rates)")
        rid, rates = port
        if isinstance(rid, bool) or not isinstance(rid, int):
            raise DataflowError("rate", f"actor {actor}: a {side} port names an integer rid")
        if (
            not isinstance(rates, tuple)
            or not rates
            or any(isinstance(r, bool) or not isinstance(r, int) or r < 0 for r in rates)
        ):
            raise DataflowError(
                "rate", f"actor {actor}: the {side} rates on {rid} are non-negative integers"
            )
        if rid in out:
            raise DataflowError("rate", f"actor {actor} names {rid} twice as a {side} port")
        out[rid] = rates
    return out


def _channels(claims) -> tuple[tuple[int, ...], dict[int, int], tuple[Channel, ...]]:
    actors = tuple(c.id for c in claims)
    phases: dict[int, int] = {}
    producers: dict[int, tuple[int, tuple[int, ...]]] = {}
    consumers: dict[int, tuple[int, tuple[int, ...]]] = {}
    initial: dict[int, int] = {}
    for claim in claims:
        stream = claim.stream
        consume = _rates(claim.id, stream.consume, "consume")
        produce = _rates(claim.id, stream.produce, "produce")
        lengths = {len(r) for r in (*consume.values(), *produce.values())}
        if len(lengths) > 1:
            raise DataflowError(
                "rate", f"actor {claim.id}: every port lists the same number of phases"
            )
        if not lengths:
            raise DataflowError("rate", f"actor {claim.id} declares a stream with no port")
        phases[claim.id] = lengths.pop()
        for rid in consume:
            if rid not in claim.rd:
                raise DataflowError("rate", f"actor {claim.id} consumes {rid} without reading it")
            if rid in consumers:
                raise DataflowError("rate", f"the FIFO {rid} has two consumers")
            consumers[rid] = (claim.id, consume[rid])
        for rid in produce:
            if rid not in claim.wr:
                raise DataflowError("rate", f"actor {claim.id} produces {rid} without writing it")
            if rid in producers:
                raise DataflowError("rate", f"the FIFO {rid} has two producers")
            producers[rid] = (claim.id, produce[rid])
        for item in stream.initial:
            if not isinstance(item, tuple) or len(item) != 2:
                raise DataflowError("rate", f"actor {claim.id}: an initial entry is (rid, tokens)")
            rid, tokens = item
            if rid not in produce:
                raise DataflowError(
                    "rate",
                    f"actor {claim.id} places initial tokens on {rid}, which it does not produce",
                )
            if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
                raise DataflowError(
                    "rate", f"the initial tokens on {rid} are a non-negative integer"
                )
            if rid in initial:
                raise DataflowError("rate", f"the initial tokens on {rid} are declared twice")
            initial[rid] = tokens
    open_ends = sorted(set(producers) ^ set(consumers))
    if open_ends:
        raise DataflowError(
            "open-stream",
            f"the FIFOs {open_ends} have a producer or a consumer outside the region, not both",
        )
    channels = tuple(
        Channel(
            rid,
            producers[rid][0],
            consumers[rid][0],
            producers[rid][1],
            consumers[rid][1],
            initial.get(rid, 0),
        )
        for rid in sorted(producers)
    )
    for ch in channels:
        if sum(ch.produce) == 0 or sum(ch.consume) == 0:
            raise DataflowError(
                "rate", f"the FIFO {ch.rid} moves no token per cycle on one side: no balance exists"
            )
    return actors, phases, channels


def _repetition(actors, channels) -> dict[int, int]:
    """The smallest positive integer solution of the balance equations, per connected
    component, in actor cycles; `inconsistent-rates` when none exists."""
    adjacent: dict[int, list[tuple[int, Fraction]]] = {a: [] for a in actors}
    for ch in channels:
        ratio = Fraction(sum(ch.produce), sum(ch.consume))  # q[consumer] = q[producer] * ratio
        adjacent[ch.producer].append((ch.consumer, ratio))
        adjacent[ch.consumer].append((ch.producer, 1 / ratio))
    q: dict[int, Fraction] = {}
    out: dict[int, int] = {}
    for start in actors:
        if start in q:
            continue
        component, stack = [start], [start]
        q[start] = Fraction(1)
        while stack:
            a = stack.pop()
            for b, ratio in adjacent[a]:
                want = q[a] * ratio
                if b not in q:
                    q[b] = want
                    component.append(b)
                    stack.append(b)
                elif q[b] != want:
                    raise DataflowError(
                        "inconsistent-rates",
                        f"actors {a} and {b} need {want} and {q[b]} firings per iteration at once: "
                        f"no repetition vector balances every FIFO",
                    )
        scale = 1
        for a in component:
            scale = scale * q[a].denominator // gcd(scale, q[a].denominator)
        ints = {a: int(q[a] * scale) for a in component}
        common = 0
        for v in ints.values():
            common = gcd(common, v)
        out.update({a: v // common for a, v in ints.items()})
    return out


def sdf_model(claims) -> SdfModel:
    """The SDF/CSDF model of a run of stream claims, or a `DataflowError` naming the law it
    breaks (module docstring)."""
    actors, phases, channels = _channels(claims)
    repetition = _repetition(actors, channels)
    firings = {a: repetition[a] * phases[a] for a in actors}
    if sum(firings.values()) > ITERATION_BUDGET:
        raise DataflowError(
            "iteration-budget",
            f"one iteration fires {sum(firings.values())} times, over the budget",
        )
    inputs = {a: [ch for ch in channels if ch.consumer == a] for a in actors}
    outputs = {a: [ch for ch in channels if ch.producer == a] for a in actors}
    tokens = {ch.rid: ch.initial for ch in channels}
    peak = dict(tokens)
    fired = {a: 0 for a in actors}
    schedule: list[int] = []
    remaining = sum(firings.values())
    while remaining:
        progressed = False
        for a in actors:  # one firing per actor per sweep: the round-robin self-timed order
            if fired[a] < firings[a]:
                phase = fired[a] % phases[a]
                if any(tokens[ch.rid] < ch.consume[phase] for ch in inputs[a]):
                    continue
                for ch in inputs[a]:
                    tokens[ch.rid] -= ch.consume[phase]
                for ch in outputs[a]:
                    tokens[ch.rid] += ch.produce[phase]
                    peak[ch.rid] = max(peak[ch.rid], tokens[ch.rid])
                fired[a] += 1
                schedule.append(a)
                remaining -= 1
                progressed = True
        if not progressed:
            starved = [a for a in actors if fired[a] < firings[a]]
            raise DataflowError(
                "deadlock",
                f"actors {starved} cannot fire: a cycle holds too few initial tokens to complete "
                f"one iteration",
            )
    for ch in channels:  # balance returns every FIFO to its initial tokens
        if tokens[ch.rid] != ch.initial:  # pragma: no cover - the repetition vector guarantees it
            raise DataflowError(
                "inconsistent-rates", f"the FIFO {ch.rid} did not return to its delays"
            )
    return SdfModel(
        actors=actors,
        phases=tuple((a, phases[a]) for a in actors),
        channels=channels,
        repetition=tuple((a, repetition[a]) for a in actors),
        firings=tuple((a, firings[a]) for a in actors),
        schedule=tuple(schedule),
        buffers=tuple((ch.rid, peak[ch.rid]) for ch in channels),
    )


def homogeneous(model: SdfModel) -> bool:
    """Every rate 1 and every actor one phase: the graph is a marked graph."""
    return all(n == 1 for _a, n in model.phases) and all(
        ch.produce == (1,) and ch.consume == (1,) for ch in model.channels
    )


def timed_model(model: SdfModel, latencies: dict[int, int]) -> TimedModel:
    """The timed-event model of a homogeneous region whose every actor has a latency, or a
    `DataflowError`: `zero-token-cycle` when a cycle carries no token (module docstring)."""
    from .tropical import CostGraph, minimum_mean_cycle

    if not homogeneous(model):
        raise DataflowError("rate", "a timed-event region is homogeneous: every rate is 1")
    # Nodes: the actors, and a chain of zero-latency relays for every token past the first on
    # a channel, so every edge carries 0 or 1 token. Edge (u, v, d, w): v's firing k waits for
    # u's firing k - d, which finishes w = latency(u) after it starts.
    nodes: list[str] = [f"a{a}" for a in model.actors]
    latency = {f"a{a}": latencies[a] for a in model.actors}
    edges: list[tuple[str, str, int, int]] = []
    channels = list(model.channels) + [
        Channel(-1 - i, a, a, (1,), (1,), 1) for i, a in enumerate(model.actors)
    ]  # the implied self-channels: an actor does not overlap its own firings
    for ch in channels:
        src, dst = f"a{ch.producer}", f"a{ch.consumer}"
        if ch.initial <= 1:
            edges.append((src, dst, ch.initial, latency[src]))
            continue
        prev = src
        for k in range(ch.initial - 1):
            relay = f"r{ch.rid}.{k}"
            nodes.append(relay)
            latency[relay] = 0
            edges.append((prev, relay, 1, latency[prev]))
            prev = relay
        edges.append((prev, dst, 1, 0))
    index = {n: i for i, n in enumerate(nodes)}
    size = len(nodes)
    neg_inf = None
    a0 = [[neg_inf] * size for _ in range(size)]
    a1 = [[neg_inf] * size for _ in range(size)]
    for src, dst, d, w in edges:
        m = a0 if d == 0 else a1
        i, j = index[dst], index[src]
        if m[i][j] is None or w > m[i][j]:
            m[i][j] = w
    # A0* by longest paths over the zero-token edges, which must be acyclic.
    order: list[int] = []
    state = [0] * size  # 0 new, 1 on the stack, 2 done
    for root in range(size):
        if state[root]:
            continue
        stack = [(root, iter([j for j in range(size) if a0[j][root] is not None]))]
        state[root] = 1
        while stack:
            node, successors = stack[-1]
            nxt = next(successors, None)
            if nxt is None:
                state[node] = 2
                order.append(node)
                stack.pop()
            elif state[nxt] == 1:
                raise DataflowError(
                    "zero-token-cycle",
                    f"a cycle through {nodes[nxt]} carries no token: the region deadlocks",
                )
            elif state[nxt] == 0:
                state[nxt] = 1
                stack.append((nxt, iter([j for j in range(size) if a0[j][nxt] is not None])))
    order.reverse()  # a topological order of the zero-token edges (sources first)
    star = [[0 if i == j else neg_inf for j in range(size)] for i in range(size)]
    for j in range(size):  # longest zero-token path from j to every node
        for u in order:
            if star[u][j] is None:
                continue
            for v in range(size):
                w = a0[v][u]
                if w is not None and (star[v][j] is None or star[u][j] + w > star[v][j]):
                    star[v][j] = star[u][j] + w
    a = [[neg_inf] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            best = None
            for m in range(size):
                if star[i][m] is not None and a1[m][j] is not None:
                    value = star[i][m] + a1[m][j]
                    if best is None or value > best:
                        best = value
            a[i][j] = best
    graph = CostGraph(
        {
            (nodes[j], nodes[i]): -a[i][j]
            for i in range(size)
            for j in range(size)
            if a[i][j] is not None
        },
        tuple(nodes),
    )
    result = minimum_mean_cycle(graph)
    if result is None:  # pragma: no cover - the implied self-channels make a cycle at every actor
        raise DataflowError("zero-token-cycle", "the region has no cycle to time")
    mean, cycle = result
    return TimedModel(
        sdf=model,
        latencies=tuple((x, latencies[x]) for x in model.actors),
        cycle_time=-mean,
        critical=tuple(cycle),
    )


__all__ = [
    "DATAFLOW_REFUSALS",
    "Channel",
    "DataflowError",
    "ITERATION_BUDGET",
    "SdfModel",
    "TimedModel",
    "homogeneous",
    "sdf_model",
    "timed_model",
]
