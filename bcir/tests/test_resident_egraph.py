"""The persistent ("living") e-graph: build+saturate once, then pivot to an
equivalent cheaper sub-graph on a telemetry bottleneck WITHOUT re-parsing or
rebuilding. Deterministic; extraction always returns the min (gains-only)."""

from bcir.kbcir import ResidentEGraph
from bcir.kbcir.egraph import C, Expr, V


def _mul(a, b):
    return Expr("mul", None, (a, b))


def _add(a, b):
    return Expr("add", None, (a, b))


def test_build_saturates_once_and_extracts_the_optimum():
    # x*b + x*c  -> x*(b+c): the factor rewrite fires during the single saturation.
    e = _add(_mul(V("x"), V("b")), _mul(V("x"), V("c")))
    r = ResidentEGraph.build(e)
    assert r.stats.saturated
    assert r.current().optimized.op == "mul"          # the factored form x*(b+c)


def test_pivot_recosts_over_the_standing_graph_without_rebuild():
    r = ResidentEGraph.build(_mul(V("x"), V("y")))
    base = r.resident_size
    before = r.current().optimized_cost               # mul = 2
    after = r.pivot(bump={"mul": 5}).optimized_cost   # telemetry: mul stalls -> recost
    assert after == before + 5                         # re-extracted under new costs
    assert r.pivots == 1
    assert r.resident_size == base and not r.rebuilt_since_build   # no rebuild/re-parse


def test_pivot_switches_to_an_equivalent_cheaper_subgraph():
    # The standing graph proves x*2 == x+x (injected as saturation would discover it).
    r = ResidentEGraph.build(_mul(V("x"), C(2)))
    cls_xx = r.eg.add(_add(V("x"), V("x")))
    r.eg.union(r.root, cls_xx)
    r.eg.rebuild()
    size = r.resident_size
    # default costs (mul=2, add=1): the adder form is cheaper.
    assert r.current().optimized.op == "add"
    # telemetry reports the adder as the bottleneck -> pivot; the mul form now wins.
    res = r.pivot(bump={"add": 10})
    assert res.optimized.op == "mul"
    assert r.resident_size == size                     # the pivot did not rebuild the graph


def test_pivot_is_deterministic():
    e = _add(_mul(V("x"), V("b")), _mul(V("x"), V("c")))
    a = ResidentEGraph.build(e); a.pivot(bump={"mul": 3})
    b = ResidentEGraph.build(e); b.pivot(bump={"mul": 3})
    assert a.current().optimized == b.current().optimized
    assert a.current().optimized_cost == b.current().optimized_cost
