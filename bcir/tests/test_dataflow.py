"""G23 / G24: SDF/CSDF and timed-event regions -- each kind verified, expanded to the identity,
floored and refused on its own corpus.

The parent had neither: a claim with fixed stream rates fell to the opaque region, nothing
computed a repetition vector, a FIFO bound or a cycle time (the audit's items 11 and 12).
Every witness here fails there -- `StreamRate` and the two kinds do not exist -- and each is
held to a derivation that shares no code with the model: hand-derived values, an enumeration
of every simple cycle, and the self-timed execution the cycle time bounds.
"""

from __future__ import annotations

import dataclasses
import random
from math import gcd

from bcir.asn1.program import graph_to_module, module_to_graph
from bcir.examples import PROGRAMS
from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.kbcir.dataflow import DataflowError, sdf_model, timed_model
from bcir.kbcir.provenance import hash_module
from bcir.kbcir.realize import _flatten, fused_candidates
from bcir.kbcir.regions import (
    REFUSALS,
    claim_floor,
    expand,
    iteration_floor,
    period_floor,
    region_graph,
    verify_region,
)
from bcir.kbcir.weights import PERF, weights
from bcir.model import StreamRate
from bcir.tests import dataflow_fixtures as fx


def test_every_corpus_case_is_its_hand_derived_region():
    """Repetition vectors (the CD-to-DAT converter's 147, 147, 98, 28, 32, 160 among them),
    the round-robin FIFO bounds, a cyclo-static actor's firings, the cycle times of four timed
    graphs, and each refusal by name: deadlock, inconsistent rates, an open stream, an actor
    whose ports disagree on phases."""
    for name, module, want, why in fx.corpus():
        assert fx.judged(module, want) == [], (name, fx.judged(module, want), why)


def test_a_dataflow_region_expands_to_its_claims_and_its_verifier_refuses_a_forged_model():
    """The expansion is the identity, and the model is re-derived from the claims, never
    trusted: a forged repetition vector, a forged cycle time, a timed kind on an untimed
    region, and a dataflow kind over a claim without a stream are each refused."""
    for name, module, want, _why in fx.corpus():
        graph = region_graph(module)
        flat = [(pid, c.id) for pid, c in _flatten(module)]
        assert [(pid, c.id) for pid, c in expand(graph, module)] == flat, name
        for region in graph.regions:
            assert verify_region(region, module) == [], name
    sdf_case = dict((n, m) for n, m, _w, _y in fx.corpus())["multirate pair 2:3"]
    region = region_graph(sdf_case).regions[0]
    forged = dataclasses.replace(
        region, model=dataclasses.replace(region.model, repetition=((1, 6), (2, 4)))
    )
    assert any("not the one the claims build" in p for p in verify_region(forged, sdf_case))
    timed_case = dict((n, m) for n, m, _w, _y in fx.corpus())["timed ring, one token"]
    region = region_graph(timed_case).regions[0]
    forged = dataclasses.replace(region, model=dataclasses.replace(region.model, cycle_time=1))
    assert any("not the one the claims build" in p for p in verify_region(forged, timed_case))
    untimed = dict((n, m) for n, m, _w, _y in fx.corpus())["homogeneous but untimed"]
    region = region_graph(untimed).regions[0]
    claimed = dataclasses.replace(region, kind="timed")
    assert any("timed region is homogeneous" in p for p in verify_region(claimed, untimed))
    as_sdf = dataclasses.replace(region_graph(timed_case).regions[0], kind="sdf")
    assert any("is a timed region" in p for p in verify_region(as_sdf, timed_case))


def test_a_refused_run_is_opaque_and_names_why():
    for name, module, want, _why in fx.corpus():
        if want["kind"] != "opaque":
            continue
        region = region_graph(module).regions[0]
        assert region.refusal == want["refusal"] and region.refusal in REFUSALS, name
        nameless = dataclasses.replace(region, refusal="")
        assert any("names no refusal" in p for p in verify_region(nameless, module)), name


def test_generated_graphs_balance_every_fifo_with_the_smallest_repetition_vector():
    """Over generated consistent graphs: every FIFO balances (q_p P = q_c C), the vector is the
    smallest (its entries share no factor in a connected graph), the schedule fires each actor
    its firings and never reads a token that is not there, every FIFO returns to its delays,
    and each bound is the schedule's peak -- recomputed here from the schedule alone."""
    for seed in range(80):
        claims = fx.random_sdf(random.Random(seed))
        model = sdf_model(claims)
        q = dict(model.repetition)
        phases = dict(model.phases)
        for ch in model.channels:
            assert q[ch.producer] * sum(ch.produce) == q[ch.consumer] * sum(ch.consume), seed
        common = 0
        for v in q.values():
            common = gcd(common, v)
        assert common == 1, seed
        assert {a: model.schedule.count(a) for a in model.actors} == dict(model.firings), seed
        tokens = {ch.rid: ch.initial for ch in model.channels}
        peak = dict(tokens)
        fired = {a: 0 for a in model.actors}
        for a in model.schedule:
            phase = fired[a] % phases[a]
            for ch in model.channels:
                if ch.consumer == a:
                    tokens[ch.rid] -= ch.consume[phase]
                    assert tokens[ch.rid] >= 0, (seed, "a firing read a token that was not there")
            for ch in model.channels:
                if ch.producer == a:
                    tokens[ch.rid] += ch.produce[phase]
                    peak[ch.rid] = max(peak[ch.rid], tokens[ch.rid])
            fired[a] += 1
        assert tokens == {ch.rid: ch.initial for ch in model.channels}, seed
        assert peak == dict(model.buffers), seed


def test_the_cycle_time_is_the_maximum_cycle_ratio_and_the_self_timed_period():
    """Karp's cycle time against an enumeration of every simple cycle (latency over tokens) on
    60 generated strongly connected timed graphs -- and against the self-timed execution it
    bounds: in the periodic regime every actor's start advances by exactly c * lambda every c
    firings. The differential is shown to fire: dropping the implied self-channels, or the
    zero-token closure, splits it."""
    import sys
    import types

    def disagreements(module) -> int:
        out = 0
        for seed in range(60):
            claims, edges, latency = fx.random_timed(random.Random(seed))
            try:
                got = module.timed_model(module.sdf_model(claims), latency).cycle_time
            except DataflowError:
                got = None
            out += got != fx.max_cycle_ratio(edges, latency)
        return out

    from bcir.kbcir import dataflow

    assert disagreements(dataflow) == 0
    source = open(dataflow.__file__, encoding="utf-8").read()
    for name, old, new in (
        (
            "no_self",
            "for i, a in enumerate(model.actors)\n    ]",
            "for i, a in enumerate(model.actors) if False\n    ]",
        ),
        (
            "no_closure",
            "if star[i][m] is not None and a1[m][j] is not None:",
            "if i == m and a1[m][j] is not None:",
        ),
    ):
        assert source.count(old) == 1, name
        mutant = types.ModuleType(f"bcir.kbcir._dataflow_{name}")
        mutant.__package__ = "bcir.kbcir"
        sys.modules[mutant.__name__] = mutant
        try:
            exec(compile(source.replace(old, new), mutant.__name__, "exec"), mutant.__dict__)
            assert disagreements(mutant) > 0, f"the differential cannot see {name}"
        finally:
            del sys.modules[mutant.__name__]

    for seed in range(60):
        claims, edges, latency = fx.random_timed(random.Random(seed))
        lam = timed_model(sdf_model(claims), latency).cycle_time
        x = fx.self_timed(edges, latency, 400)
        period = next(
            (
                c
                for c in range(1, 121)
                if (c * lam).denominator == 1
                and all(x[k + c][v] - x[k][v] == c * lam for k in range(250, 270) for v in latency)
            ),
            None,
        )
        assert period is not None, (seed, lam)


def test_the_floors_hold():
    """An sdf region's iteration floor is every actor's firings times its claim's floor; a
    timed region's period floor is its cycle time, and no self-timed execution of the corpus
    completes iterations faster."""
    h, theta = TARGETS["x86_avx2"], Theta.cool()
    for name, module, want, _why in fx.corpus():
        region = region_graph(module).regions[0]
        if region.kind not in ("sdf", "timed"):
            continue
        cand = fused_candidates(module, h)
        flat = _flatten(module)
        index = {c.id: i for i, (_p, c) in enumerate(flat)}
        w = weights(h, theta, region.phase_id, PERF)
        sdf = region.model.sdf if region.kind == "timed" else region.model
        want_floor = sum(
            n
            * claim_floor(
                flat[index[a]][1], flat[index[a] - 1][1] if index[a] else None, cand, h, theta, w
            )
            for a, n in sdf.firings
        )
        assert iteration_floor(region, module, h, theta) == want_floor, name
        if region.kind == "timed":
            lam = period_floor(region)
            edges = [(ch.producer, ch.consumer, ch.initial) for ch in sdf.channels]
            latency = dict(region.model.latencies)
            x = fx.self_timed(edges, latency, 200)
            assert all(x[199][v] >= 199 * lam - sum(latency.values()) for v in latency), name


def test_a_stream_changes_no_existing_region_and_no_content_address():
    """Non-disturbance: no corpus program declares a stream, so every region graph has only
    the kinds it had; and `stream` is digest-excluded like `timing`, so declaring one moves no
    R13 content address. The ASN.1 program projection carries it -- absent and empty apart."""
    for name, build in sorted(PROGRAMS.items()):
        kinds = set(region_graph(build()).kinds())
        assert kinds <= {"affine", "opaque"}, name
    module = dict((n, m) for n, m, _w, _y in fx.corpus())["CD to DAT"]
    bare = graph_to_module(module_to_graph(module))
    assert [c.stream for p in bare.phases for c in p.claims] == [
        c.stream for p in module.phases for c in p.claims
    ]
    stripped = dataclasses.replace(module.phases[0].claims[0], stream=None)
    empty = dataclasses.replace(module.phases[0].claims[0], stream=StreamRate())
    for claim in (stripped, empty):
        clone = fx.module_of([claim])
        assert graph_to_module(module_to_graph(clone)).phases[0].claims[0].stream == claim.stream
    plain = fx.module_of([dataclasses.replace(c, stream=None) for c in module.phases[0].claims])
    assert hash_module(plain) == hash_module(fx.module_of(module.phases[0].claims))
