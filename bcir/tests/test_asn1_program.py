"""P3 and P4 — the lifetime/timing projection, and the min-plus cost graph.

**P3's gate**: *"R19/R20/R21 verdicts identical whether the input arrived as MLIR or as
JER."* The load-bearing word is *verdicts*. R20 tracks which clock domain last wrote each
resource and R21 walks the phase and claim order maintaining a freed set, so both are
**order-sensitive**: a projection that preserved every field but perturbed the order would
produce a module that verified differently while matching field by field. Comparing the
diagnostics — in order, unsorted — is what catches that.

**P4's gate**: *"Negative-cycle detection refuses rather than diverges; unroll decisions
reproduce a hand-derived optimum on a fixture set; legality-first preserved."*

Nothing here re-implements a law or an algorithm. `verify_timing`/`verify_lifetime` are
called on both sides of the projection, and Karp's mean is checked against means computed
by hand from the cycle weights.
"""

from __future__ import annotations

from fractions import Fraction

from bcir.asn1.program import (
    CLAIM,
    PHASE,
    PROGRAM,
    graph_to_module,
    jer_to_module,
    module_to_graph,
    module_to_jer,
    verdicts,
)
from bcir.asn1.graph import graph_to_jer, jer_to_graph, resolve
from bcir.asn1.tags import Asn1Error
from bcir.examples import PROGRAMS
from bcir.kbcir.tropical import (
    CostGraph,
    NegativeCycle,
    Semiring,
    alternative,
    close,
    compose,
    minimum_mean_cycle,
    shortest_paths,
    unroll_factor,
)
from bcir.model.graph import Claim, Lifetime, Module, Phase, Timing
from bcir.model import Lane, Opcode, StrideClass


# --- P3: the projection ---------------------------------------------------------------------


def test_every_corpus_program_survives_the_graph_round_trip():
    """The whole module, field for field, over the real corpus rather than fixtures."""
    count = 0
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        assert jer_to_module(module_to_jer(module)) == module, name
        count += 1
    assert count >= 10, f"the corpus collapsed to {count} programs"


def test_the_verdicts_are_identical_whichever_rail_the_module_arrived_on():
    """P3's gate, over the corpus.

    Stronger than field equality: it is the *laws'* answer that must match, and R19–R21
    read order and cross-claim state rather than one claim at a time.
    """
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        assert verdicts(module) == verdicts(jer_to_module(module_to_jer(module))), name


def _timed(claim_id: int, domain: str, sync: str, reads=(), writes=(), hazard="unique"):
    return Claim(
        id=claim_id,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=StrideClass.UNIT,
        count=1,
        rd=tuple(reads),
        wr=tuple(writes),
        hazard=hazard,
        timing=Timing(
            clock_domain=domain,
            sync_type=sync,
            clock_frequency_mhz=100 if sync == "synchronous" else 0,
            latency_cycles=4,
            setup_hold_margin=1,
        ),
    )


def test_an_r20_clock_domain_crossing_survives_as_a_crossing():
    """R20 is the order-sensitive one, so it is the projection's real test.

    The law reports a crossing against the domain that **last wrote** the resource. If the
    projection reordered claims, or dropped `clock_domain`, or lost the `hazard` that
    excuses the crossing, the rebuilt module would verify differently — and would often
    verify *clean*, which is the dangerous direction.
    """
    module = Module(
        name="crossing",
        cacheline=64,
        align=64,
        target="x86",
        resources={},
        phases=[
            Phase(
                phase_id=0,
                deps=(),
                claims=[
                    _timed(1, "fast", "synchronous", writes=(7,)),
                    _timed(2, "slow", "synchronous", reads=(7,)),
                ],
            )
        ],
    )
    original = verdicts(module)
    assert any(law == "R20" for law, _m in original), "the fixture does not trip R20"
    assert verdicts(jer_to_module(module_to_jer(module))) == original


def test_a_barriered_crossing_stays_excused():
    """The negative direction: a projection that dropped `hazard` would invent an R20.

    Testing only that violations survive would miss a projection that made every module
    *more* illegal, which is just as wrong and much easier to ship.
    """
    module = Module(
        name="excused",
        cacheline=64,
        align=64,
        target="x86",
        resources={},
        phases=[
            Phase(
                phase_id=0,
                deps=(),
                claims=[
                    _timed(1, "fast", "synchronous", writes=(7,)),
                    _timed(2, "slow", "synchronous", reads=(7,), hazard="barriered"),
                ],
            )
        ],
    )
    assert not any(law == "R20" for law, _m in verdicts(module))
    assert verdicts(jer_to_module(module_to_jer(module))) == verdicts(module)


def test_an_r21_use_after_free_survives_with_its_order():
    """R21 walks a freed set in claim order, so order is part of the verdict."""

    def claim(cid, event, rid):
        return Claim(
            id=cid,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=1,
            rd=(rid,) if event != "alloc" else (),
            wr=(rid,) if event == "alloc" else (),
            lifetime=Lifetime(event=event, epoch=0),
        )

    module = Module(
        name="uaf",
        cacheline=64,
        align=64,
        target="x86",
        resources={},
        phases=[
            Phase(
                phase_id=0,
                deps=(),
                claims=[
                    claim(1, "alloc", 3),
                    claim(2, "free", 3),
                    claim(3, "use", 3),
                ],
            )
        ],
    )
    original = verdicts(module)
    assert any(law == "R21" for law, _m in original), "the fixture does not trip R21"
    assert verdicts(jer_to_module(module_to_jer(module))) == original


def test_absent_timing_and_all_default_timing_stay_distinguishable():
    """R19 is vacuous when `timing is None` and constraining when it is a zero block.

    Collapsing the two — the obvious mistake for a projection that omits empty values —
    would silently subject every untimed claim in the repository to the timing laws.
    """
    base = dict(id=1, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT, count=1)
    without = Module(
        name="m",
        cacheline=64,
        align=64,
        target="x86",
        resources={},
        phases=[Phase(phase_id=0, deps=(), claims=[Claim(**base)])],
    )
    with_empty = Module(
        name="m",
        cacheline=64,
        align=64,
        target="x86",
        resources={},
        phases=[
            Phase(
                phase_id=0, deps=(), claims=[Claim(**base, timing=Timing(sync_type="synchronous"))]
            )
        ],
    )
    assert verdicts(without) == ()
    assert any(law == "R19" for law, _m in verdicts(with_empty)), (
        "a synchronous claim with no clock must trip R19"
    )
    for module in (without, with_empty):
        rebuilt = jer_to_module(module_to_jer(module))
        assert verdicts(rebuilt) == verdicts(module)
        assert (rebuilt.phases[0].claims[0].timing is None) == (
            module.phases[0].claims[0].timing is None
        )


def test_the_projection_carries_phase_and_claim_order_structurally():
    """Order lives in the ordered edge lists, not in a rank attribute somebody maintains."""
    module = PROGRAMS["fused_chain"]()
    graph = module_to_graph(module)
    root = graph.roots[0]
    phases = [e.target for e in graph.nodes[root].edges if e.label == "phase"]
    assert len(phases) == len(module.phases)
    for at, phase in zip(phases, module.phases):
        claims = [e.target for e in graph.nodes[at].edges if e.label == "claim"]
        assert [graph.nodes[c].label for c in claims] == [str(c.id) for c in phase.claims]


def test_a_program_graph_is_an_ordinary_node_graph():
    """P3 reuses P1's table rather than inventing a second graph format.

    A parallel format would need its own reader, its own bounded path and its own
    round-trip law — three places to drift instead of none.
    """
    module = PROGRAMS["matmul_tiled"]()
    graph = module_to_graph(module)
    assert jer_to_graph(graph_to_jer(graph)) == graph
    kinds = {node.kind for node in graph.nodes}
    assert kinds <= {PROGRAM, PHASE, CLAIM, "resource"}
    # And it resolves with the ordinary machinery, reporting the kinds no table knows.
    from bcir.asn1.graph import DIALECT_NODE_CLASS

    report = resolve(graph, DIALECT_NODE_CLASS)
    assert len(report.unresolved) == len(graph.nodes), (
        "the dialect table should not claim to type program nodes"
    )


def test_a_graph_with_no_program_root_is_refused():
    from bcir.asn1.graph import Graph, Node

    for graph in (Graph(), Graph(nodes=(Node("phase", "0"),), roots=(0,))):
        try:
            graph_to_module(graph)
        except Asn1Error as error:
            assert "exactly one program root" in str(error)
        else:
            raise AssertionError("a graph with no program root was accepted")


# --- P4: the min-plus semiring ------------------------------------------------------------------


def test_a_conditional_is_semiring_addition():
    """`if/else` is `min` over the arms — the mapping §5 calls natural rather than forced."""
    assert alternative(Semiring.MIN_PLUS, 5, 3, 9) == 3
    assert alternative(Semiring.MAX_PLUS, 5, 3, 9) == 9
    assert alternative(Semiring.MIN_PLUS) is None, "no alternatives is the identity, +inf"
    assert compose(2, 3, 4) == 9, "straight-line composition adds"


def test_the_shortest_path_takes_the_cheaper_arm_of_a_diamond():
    graph = CostGraph({("a", "b"): 3, ("a", "c"): 5, ("b", "d"): 4, ("c", "d"): 1})
    assert shortest_paths(graph, "a")["d"] == 6  # a -> c -> d
    assert shortest_paths(graph, "a", semiring=Semiring.MAX_PLUS)["d"] == 7


def test_an_unreachable_node_has_no_cost_rather_than_a_large_one():
    """+inf is absence, and absence is how it is spelled — not a sentinel a caller must know."""
    graph = CostGraph({("a", "b"): 1}, nodes=("a", "b", "island"))
    assert "island" not in shortest_paths(graph, "a")


def test_a_negative_cycle_refuses_and_names_itself():
    """§5's divergence check. The algebra says the cost model is wrong; so does the error.

    Returning -inf, or clamping to zero, would report that going round a loop makes a
    program cheaper — which is the one answer that must never be produced quietly.
    """
    graph = CostGraph({("x", "y"): 1, ("y", "x"): -5})
    try:
        shortest_paths(graph, "x")
    except NegativeCycle as error:
        assert error.weight < 0
        assert set(error.cycle) == {"x", "y"}
        assert "cost model being wrong" in str(error)
    else:
        raise AssertionError("a negative cycle produced a shortest path")


def test_the_closure_of_a_non_negative_loop_is_the_empty_path():
    """Over min-plus, `a* = 0` for `a >= 0`: a loop you need not enter costs nothing."""
    assert close(0) == 0
    assert close(7) == 0
    try:
        close(-1)
    except NegativeCycle:
        pass
    else:
        raise AssertionError("a negative self-loop was given a closure")


def test_karp_reproduces_a_hand_derived_mean():
    """P4's gate asks for a hand-derived optimum, so the means are computed by hand here.

    A two-cycle of weights 2 and 4 has mean 3; a three-cycle of 1, 1, 1 has mean 1; and a
    graph with both must report the smaller. `Fraction` because the mean is a ratio of
    integers and a float comparison would make the winner host-dependent.
    """
    two = CostGraph({("h", "b"): 2, ("b", "h"): 4})
    assert minimum_mean_cycle(two)[0] == Fraction(6, 2) == 3

    three = CostGraph({("p", "q"): 1, ("q", "r"): 1, ("r", "p"): 1})
    assert minimum_mean_cycle(three)[0] == Fraction(3, 3) == 1

    both = CostGraph(
        {("h", "b"): 2, ("b", "h"): 4, ("p", "q"): 1, ("q", "r"): 1, ("r", "p"): 1, ("h", "p"): 0}
    )
    assert minimum_mean_cycle(both)[0] == 1, "the smaller mean must win"


def test_a_mean_that_is_not_an_integer_stays_exact():
    """The reason for `Fraction`: a 3-cycle of total 4 has mean 4/3, not 1.333…"""
    graph = CostGraph({("a", "b"): 1, ("b", "c"): 1, ("c", "a"): 2})
    mean, cycle = minimum_mean_cycle(graph)
    assert mean == Fraction(4, 3)
    assert isinstance(mean, Fraction)
    assert len(cycle) >= 3


def test_an_acyclic_graph_has_no_mean_cycle_and_unrolls_to_one():
    graph = CostGraph({("a", "b"): 1, ("b", "c"): 1})
    assert minimum_mean_cycle(graph) is None
    assert unroll_factor(graph, prologue=10, epilogue=10, iterations=100) == 1


def test_the_unroll_factor_is_the_hand_derived_optimum():
    """The overhead term is the only one that depends on the factor, and that is the point.

    Unrolling by `u` pays prologue+epilogue once per `ceil(iterations / u)` chunks; the body
    cost per iteration is unchanged, so it cancels. With 100 iterations, overhead 20 and a
    cap of 64, the fewest reachable chunks is 2 and the SMALLEST factor achieving it is 50.
    A pass reporting a large win from unrolling a body it did not change would be reporting
    an artefact.
    """
    loop = CostGraph({("h", "b"): 2, ("b", "h"): 4})
    assert unroll_factor(loop, prologue=10, epilogue=10, iterations=100, maximum=64) == 50
    # No overhead to amortize: unrolling buys nothing and the honest answer says so.
    assert unroll_factor(loop, prologue=0, epilogue=0, iterations=100) == 1
    # One iteration cannot be unrolled.
    assert unroll_factor(loop, prologue=10, epilogue=10, iterations=1) == 1


def test_a_negative_mean_cycle_refuses_to_produce_an_unroll_factor():
    """Legality and sanity first: a diverging loop has no optimal unrolling."""
    graph = CostGraph({("h", "b"): 1, ("b", "h"): -4})
    try:
        unroll_factor(graph, prologue=10, epilogue=10, iterations=100)
    except NegativeCycle as error:
        assert error.weight < 0
    else:
        raise AssertionError("a diverging loop was given an unroll factor")


def test_the_semiring_is_a_parameter_and_not_a_hidden_default():
    """§5's limit 1, as an API property rather than a comment.

    `min` is the best case and `max` the worst; neither is the expected cost, which needs
    branch probabilities. A module that hard-coded one would be answering a question the
    caller did not ask — so both are reachable and the caller names which.
    """
    graph = CostGraph({("a", "b"): 1, ("a", "c"): 100, ("b", "d"): 1, ("c", "d"): 1})
    assert shortest_paths(graph, "a", semiring=Semiring.MIN_PLUS)["d"] == 2
    assert shortest_paths(graph, "a", semiring=Semiring.MAX_PLUS)["d"] == 101
    assert {s.value for s in Semiring} == {"min-plus", "max-plus"}


def test_the_pass_never_consults_a_verifier():
    """§5's limit 2, checked structurally: legality is not this module's business.

    `unroll_factor` returns a count, not a permission, and nothing here imports the
    verifier — so no future reader can mistake a cost decision for a legality one.
    """
    import bcir.kbcir.tropical as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("verify_", "from ..verify", "import verify"):
        assert forbidden not in source, (
            f"the cost pass references {forbidden!r}; legality is decided elsewhere"
        )
