"""G30: polyhedral depth two -- a declared 2-D nest's exact dependences and their legality.

The parent's polyhedral model was one-dimensional and a tile claim was opaque (the audit's
item 14). These witnesses hold the depth-two model to a reference that enumerates every pair
of iteration instances and decides legality by simulating each execution order (interchange,
five tilings, each loop run in parallel), pin the classic stencils by hand, show the
differential fires on wrong copies of the analysis, and show the `nest` region in the region
graph: recognized, re-verified, refused by name when malformed, carried by the ASN.1 program
and outside the R13 digest.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace

from bcir.asn1.program import jer_to_module, module_to_jer
from bcir.examples import PROGRAMS
from bcir.kbcir import polyhedral
from bcir.kbcir.polyhedral import NestError, nest_model
from bcir.kbcir.provenance import hash_module
from bcir.kbcir.regions import region_graph, verify_region
from bcir.model import LoopNest
from bcir.tests import nest_fixtures as fixtures
from bcir.tests.nest_fixtures import NESTS, graded, measure, nest_module, random_nest, reference

A, B = 10, 11


def _model(rows, cols, reads, writes):
    m = nest_module(rows, cols, reads, writes)
    return nest_model(m.phases[0].claims[0], m)


def _flags(model):
    return (model.interchange_legal, model.tiling_legal, model.outer_parallel, model.inner_parallel)


def test_hand_derived_stencils():
    # rows of 8 elements, the nest over 7 columns of them (the padding column keeps a row's
    # last access from aliasing the next row's first)
    # A[i+1][j+1] = A[i][j+1] + A[i+1][j]: distances (1, 0) and (0, 1) -- every
    # transformation legal, neither loop parallel
    m = _model(4, 7, [(A, 1, 8, 1), (A, 8, 8, 1)], [(A, 9, 8, 1)])
    assert m.distances == {(1, 0), (0, 1)}
    assert _flags(m) == (True, True, False, False)
    # A[i+1][j] = A[i][j+1]: distance (1, -1) -- interchange and tiling illegal, the inner
    # loop parallel (a vector width may run across j), the outer not
    m = _model(4, 7, [(A, 1, 8, 1)], [(A, 8, 8, 1)])
    assert m.distances == {(1, -1)} and _flags(m) == (False, False, False, True)
    # B[j][i] = A[i][j]: no dependence -- everything legal and parallel
    m = _model(4, 4, [(A, 0, 4, 1)], [(B, 0, 1, 4)])
    assert m.dependences == () and _flags(m) == (True, True, True, True)
    # a reduction into one cell: output and flow at every distance -- nothing parallel
    m = _model(3, 3, [(A, 0, 0, 0)], [(A, 0, 0, 0)])
    assert (0, 1) in m.distances and (1, -2) in m.distances
    assert _flags(m) == (False, False, False, False)


def test_the_model_is_the_reference_on_every_generated_nest():
    """160 generated nests -- aliasing, negative and row-sized strides, one or two resources:
    the dependences (resource, kind, distance) and the four legality verdicts are the
    reference's, which enumerates every instance pair and simulates every order. Every
    verdict pattern is exercised."""
    assert measure() == {"nest.dependence.disagreements": 0.0, "nest.legality.disagreements": 0.0}
    patterns = {reference(random_nest(s))[1:] for s in range(NESTS)}
    assert len(patterns) >= 5, patterns
    assert any(p[0] for p in patterns) and any(not p[0] for p in patterns)


def test_the_differential_fires_on_wrong_analyses():
    """Three wrong copies of the analysis, loaded from the module source (never by editing
    the tree): counting an iteration's own accesses as a dependence, dropping the
    divisibility test of the access equation, and calling the inner loop parallel whenever no
    distance moves j forward -- each disagrees with the reference somewhere."""
    source = open(polyhedral.__file__, encoding="utf-8").read()
    for name, old, new in (
        ("same_instance", "if (i2, j2) > (i, j):", "if (i2, j2) >= (i, j):"),
        (
            "no_divisibility",
            "elif r % tc == 0 and 0 <= r // tc < cols:",
            "elif 0 <= r // tc < cols:",
        ),
        (
            "inner",
            "inner = not any(di == 0 and dj > 0 for di, dj in distances)",
            "inner = not any(dj > 0 for di, dj in distances)",
        ),
    ):
        assert source.count(old) == 1, name
        mutant = types.ModuleType(f"bcir.kbcir._polyhedral_{name}")
        mutant.__package__ = "bcir.kbcir"
        sys.modules[mutant.__name__] = mutant
        try:
            exec(compile(source.replace(old, new), mutant.__name__, "exec"), mutant.__dict__)
            wrong = 0
            fixtures.nest_model = mutant.nest_model
            try:
                for seed in range(NESTS):
                    deps_ok, legal_ok = graded(random_nest(seed))
                    wrong += not (deps_ok and legal_ok)
            finally:
                fixtures.nest_model = nest_model
            assert wrong > 0, f"the differential cannot see {name}"
        finally:
            del sys.modules[mutant.__name__]


def test_malformed_nests_are_refused():
    good = nest_module(2, 2, [(A, 0, 2, 1)], [(B, 0, 2, 1)])
    claim = good.phases[0].claims[0]
    for bad in (
        LoopNest((0, 2), claim.nest.maps),
        LoopNest((2, 2), claim.nest.maps[:1]),
        LoopNest((2, 2), ((B, 0, 2, 1), (B, 0, 2, 1))),
        LoopNest((2, 2), ((A, 0, 2, 1), (B, 254, 2, 1))),  # reaches past the resource
        LoopNest((2, 2), ((A, -1, 2, 1), (B, 0, 2, 1))),
        LoopNest((65, 65), claim.nest.maps),
    ):
        try:
            nest_model(replace(claim, nest=bad), good)
        except NestError:
            continue
        raise AssertionError(f"the nest {bad!r} was analysed")


def test_the_nest_region_in_the_region_graph():
    """A declared nest is its own `nest` region, re-verified (a forged verdict is refused);
    a nest the model refuses is opaque with the refusal `nest`; no corpus program's region
    graph changes; the nest travels with the ASN.1 program and moves no digest."""
    m = nest_module(4, 8, [(A, 1, 8, 1)], [(A, 8, 8, 1)])
    (region,) = region_graph(m).regions
    assert region.kind == "nest" and region.model == nest_model(m.phases[0].claims[0], m)
    forged = replace(region, model=replace(region.model, interchange_legal=True))
    assert verify_region(forged, m) and not verify_region(region, m)
    claim = m.phases[0].claims[0]
    broken = replace(claim, nest=LoopNest((4, 8), ((A, 250, 8, 1), (A, 8, 8, 1))))
    m.phases[0].claims[0] = broken
    (region,) = region_graph(m).regions
    assert (region.kind, region.refusal) == ("opaque", "nest")
    for name, build in PROGRAMS.items():
        assert all(r.kind != "nest" for r in region_graph(build()).regions), name
    plain = nest_module(4, 8, [(A, 1, 8, 1)], [(A, 8, 8, 1)])
    back = jer_to_module(module_to_jer(plain))
    assert back.phases[0].claims[0].nest == plain.phases[0].claims[0].nest
    bare = nest_module(4, 8, [(A, 1, 8, 1)], [(A, 8, 8, 1)])
    bare.phases[0].claims[0] = replace(bare.phases[0].claims[0], nest=None)
    assert hash_module(bare) == hash_module(plain)
