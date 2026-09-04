"""P1 and P2 — the flat node-table graph and its commuting projection from the dialect.

P1's gate: *"Round-trip of a cyclic graph; every edge typed; unresolvable edge is a value,
not a fault; byte-identical re-emission."*

P2's gate: *"`MLIR -> JER -> MLIR` is the identity on the dialect; `JER -> MLIR -> JER` is
byte-identical"* — the same two laws J4 part 2 proved into a *tree*, now proved into the
*graph*, which is the shape a program actually needs.

The note calls P1 *"the single largest unsolved design question in the proposal"*, and the
answer it tests here is the one LLVM bitcode and MLIR bytecode both reached: a flat table
where an edge names a row by **index**. `test_a_mutually_recursive_pair_round_trips` is the
case that motivates the whole design — two functions that call each other are a cycle no
nesting can hold, and `jer_plan.py` refuses schema recursion past depth 64.

`test_an_unknown_node_kind_is_a_value_and_still_re_emits` is the other load-bearing one, and
it records a finding: §4.1 proposed carrying a node's payload as an X.681 OPEN TYPE, but
X.697 §41 gives JER **no** fallback for an unresolvable one, so that design would have made
an unknown node *unencodable* — the exact opposite of the gate. The object set stayed as the
typing authority and moved to being an enrichment.
"""

from __future__ import annotations

import json

from bcir.asn1.dialect import parse_mlir
from bcir.asn1.graph import (
    DIALECT_NODE_CLASS,
    GRAPH_VERSION,
    NODE_GRAPH,
    Edge,
    Graph,
    Node,
    content_address,
    dialect_to_graph,
    graph_to_dialect,
    graph_to_jer,
    graph_to_value,
    jer_to_graph,
    resolve,
    value_to_graph,
)
from bcir.asn1.jer import JerRules, decode_jer, encode_jer
from bcir.asn1.jer_bounded import STRICT_LIMITS, JerBoundedError, decode_bounded
from bcir.asn1.schema import ObjectSetTable, Primitive
from bcir.asn1.tags import Asn1Error, Universal
import os

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_FIXTURE = os.path.join(_ROOT, "mlir", "test", "passes", "verify_asn1.mlir")


def _corpus():
    return parse_mlir(open(_FIXTURE, encoding="utf-8").read())


# --- P1: cycles ---------------------------------------------------------------------------


def test_a_mutually_recursive_pair_round_trips():
    """The case the whole design exists for.

    Two functions that call each other are a cycle, and no nesting holds one: a tree
    encoder either loops forever or truncates. With index edges the document is two
    ordinary rows and its JSON depth is two, however tangled the graph gets.
    """
    graph = Graph(
        nodes=(
            Node("function", "even", edges=(Edge("calls", 1),)),
            Node("function", "odd", edges=(Edge("calls", 0),)),
        ),
        roots=(0,),
    )
    octets = graph_to_jer(graph)
    assert jer_to_graph(octets) == graph
    assert graph_to_jer(jer_to_graph(octets)) == octets, "re-emission was not byte-identical"
    # The depth really is bounded: the JSON nests document -> nodes -> node -> edges -> edge.
    document = json.loads(octets)
    assert document["nodes"][0]["edges"][0]["target"] == 1
    assert document["nodes"][1]["edges"][0]["target"] == 0


def test_a_self_loop_round_trips():
    """The degenerate cycle, which a "detect and break" scheme usually gets wrong."""
    graph = Graph(nodes=(Node("function", "loop", edges=(Edge("calls", 0),)),), roots=(0,))
    assert jer_to_graph(graph_to_jer(graph)) == graph


def test_a_deep_chain_stays_shallow_in_json():
    """A thousand-node chain is a thousand ROWS and a constant JSON depth.

    This is the concrete reason the table beats nesting: `jer_bounded`'s §4.3 depth ceiling
    is 64, so a nested representation of this program could not be read at all, while the
    flat one is nowhere near any limit.
    """
    nodes = tuple(
        Node("step", f"s{i}", edges=((Edge("next", i + 1),) if i < 999 else ()))
        for i in range(1000)
    )
    graph = Graph(nodes=nodes, roots=(0,))
    octets = graph_to_jer(graph)
    assert jer_to_graph(octets) == graph
    # And it passes the bounded reader at its ordinary depth limit, unmodified.
    assert value_to_graph(decode_bounded(octets, NODE_GRAPH)) == graph


# --- P1: unresolvable is a value ----------------------------------------------------------------


def test_an_unknown_node_kind_is_a_value_and_still_re_emits():
    """§4.1's design, corrected by X.697 §41 — and the correction is the finding.

    The note proposed a node's payload be an OPEN TYPE resolved through the object set, with
    X.681 §12.9's extensibility making an unknown node ordinary traffic. JER cannot do that:
    §41 says an open type's encoding IS the contained value's encoding, and unlike XER §8.5
    there is no hexadecimal fallback, so `jer.py` refuses an unresolvable open type. An
    unknown node would have been *unencodable*.

    So the object set types the graph as an ENRICHMENT: the node still decodes, still
    re-emits byte-identically, and `resolve` reports what it could not type.
    """
    graph = Graph(nodes=(Node("a-kind-from-2030", "future", attributes=(("x", "1"),)),), roots=(0,))
    octets = graph_to_jer(graph)
    assert jer_to_graph(octets) == graph
    assert graph_to_jer(jer_to_graph(octets)) == octets

    report = resolve(graph, DIALECT_NODE_CLASS)
    assert len(report.unresolved) == 1
    resolution = report.unresolved[0]
    assert resolution.index == 0 and resolution.kind == "a-kind-from-2030"
    assert "12.9" in resolution.reason, "an extensible set's miss must cite §12.9"
    assert not resolution.resolved and resolution.payload is None


def test_a_non_extensible_set_reports_a_violation_rather_than_a_versioned_peer():
    """§12.3's `...` changes what a miss MEANS, and the report says which.

    A peer using an object outside an extensible set is ordinary traffic; the same miss
    against a closed set is a schema violation. Reporting both as "unknown" would lose the
    distinction a loader needs to decide whether to proceed.
    """
    closed = ObjectSetTable("CLOSED", rows=({"kind": "known", "Payload": None},), extensible=False)
    graph = Graph(nodes=(Node("other", "x"),), roots=(0,))
    reason = resolve(graph, closed).unresolved[0].reason
    assert "NOT extensible" in reason and "schema violation" in reason


def test_a_dangling_edge_is_a_value_and_does_not_stop_the_decode():
    """An edge out of range is reported, never raised.

    A format that refused to decode a document with one bad edge would be unable to show a
    user what is wrong with it — and a loader that must refuse can still refuse, because
    the report tells it to. The policy belongs to the caller.
    """
    graph = Graph(nodes=(Node("component", "c", edges=(Edge("type", 99),)),), roots=(0,))
    assert jer_to_graph(graph_to_jer(graph)) == graph
    faults = resolve(graph, DIALECT_NODE_CLASS).edge_faults
    assert len(faults) == 1
    assert faults[0].target == 99 and "outside the node table" in faults[0].reason


def test_every_edge_is_typed_and_a_mistyped_one_is_named():
    """P1's "every edge typed", through X.682 §10.19's row selection.

    The table's `TargetKind` column says what a `component` points at. That makes a mistyped
    edge a *schema* fault with a resolution law behind it, rather than a convention some
    reader may or may not enforce — which is §4.1's argument against inventing `$ref`.
    """
    good = Graph(
        nodes=(Node("component", "c", edges=(Edge("type", 1),)), Node("type", "T")), roots=(0,)
    )
    assert resolve(good, DIALECT_NODE_CLASS).edge_faults == ()

    bad = Graph(
        nodes=(Node("component", "c", edges=(Edge("type", 1),)), Node("operation", "o")), roots=(0,)
    )
    faults = resolve(bad, DIALECT_NODE_CLASS).edge_faults
    assert len(faults) == 1 and "points at 'type'" in faults[0].reason


def test_resolution_never_raises_on_any_graph_content():
    """The property that makes "a value, not a fault" real rather than aspirational."""
    graphs = [
        Graph(),
        Graph(nodes=(Node("", ""),)),
        Graph(nodes=(Node("type", "T", edges=(Edge("", -1),)),)),
        Graph(nodes=(Node("component", "c", edges=(Edge("type", 0),)),), roots=(5,)),
        Graph(nodes=tuple(Node("x", str(i), edges=(Edge("e", i - 1),)) for i in range(20))),
    ]
    for graph in graphs:
        report = resolve(graph, DIALECT_NODE_CLASS)
        assert isinstance(report.resolutions, tuple)
        assert len(report.resolutions) == len(graph.nodes)


# --- P1: content addressing ------------------------------------------------------------------------


def test_a_content_address_is_a_function_of_shape_not_of_position():
    """§4.1's secondary identity mechanism, and the property that makes it worth having.

    The same subgraph in two different programs must hash identically, or the address
    cannot deduplicate across files. The walk therefore renumbers edges into traversal
    order before hashing, so no absolute index leaks in.
    """
    left = Graph(nodes=(Node("f", "a", edges=(Edge("calls", 1),)), Node("f", "b")), roots=(0,))
    # The same two nodes, preceded by unrelated ones, so their indices differ.
    right = Graph(
        nodes=(
            Node("noise", "n"),
            Node("noise", "m"),
            Node("f", "a", edges=(Edge("calls", 3),)),
            Node("f", "b"),
        ),
        roots=(2,),
    )
    assert content_address(left, 0) == content_address(right, 2)
    # And a different shape hashes differently.
    other = Graph(nodes=(Node("f", "a", edges=(Edge("calls", 1),)), Node("f", "c")), roots=(0,))
    assert content_address(left, 0) != content_address(other, 0)


def test_a_content_address_terminates_on_a_cycle():
    """Cycle-safety is not optional here: the node table exists because cycles do."""
    graph = Graph(
        nodes=(
            Node("f", "a", edges=(Edge("calls", 1),)),
            Node("f", "b", edges=(Edge("calls", 0),)),
        ),
        roots=(0,),
    )
    assert len(content_address(graph, 0)) == 64
    assert content_address(graph, 0) != content_address(graph, 1)


def test_an_out_of_range_root_is_refused_by_the_addresser():
    """A caller asking for the address of a node that is not there is a caller error, and
    the one place in this module where raising is right — nothing was decoded."""
    try:
        content_address(Graph(), 0)
    except Asn1Error as error:
        assert "outside the table" in str(error)
    else:
        raise AssertionError("an address was computed for a node that does not exist")


# --- P1: the schema ------------------------------------------------------------------------------------


def test_the_graph_carries_its_version_and_refuses_another():
    graph = Graph(nodes=(Node("x", "y"),))
    value = graph_to_value(graph)
    assert value["version"] == GRAPH_VERSION
    value["version"] = GRAPH_VERSION + 1
    try:
        value_to_graph(value)
    except Asn1Error as error:
        assert "refuses rather than inferring" in str(error)
    else:
        raise AssertionError("a future graph version was silently accepted")


def test_the_graph_goes_through_the_bounded_reader():
    """A program graph arriving over a wire is attacker-chosen input like any other."""
    graph = dialect_to_graph(_corpus()[0])
    octets = graph_to_jer(graph)
    assert value_to_graph(decode_bounded(octets, NODE_GRAPH)) == graph
    tight = STRICT_LIMITS.tightened(input_bytes=len(octets) - 1)
    try:
        decode_bounded(octets, NODE_GRAPH, limits=tight)
    except JerBoundedError as error:
        assert error.diagnostic.code.value == "input-too-large"
    else:
        raise AssertionError("the bounded reader ignored its ceiling")


def test_the_canonical_form_omits_an_empty_default():
    """A node with no edges, no attributes and no label carries none of them.

    DEFAULT rather than OPTIONAL throughout, so "absent" and "empty" are one fact with one
    spelling — the same reasoning J4 part 3 arrived at, and the reason a program graph does
    not pay for its own generality on every node.
    """
    document = json.loads(graph_to_jer(Graph(nodes=(Node("x"),))))
    assert set(document["nodes"][0]) == {"kind"}, document["nodes"][0]


# --- P2: the dialect projection ---------------------------------------------------------------------------


def test_dialect_to_graph_to_dialect_is_the_identity_over_the_law_corpus():
    """P2's first law, over the same 26 fixtures J4 part 2 used — legal and illegal alike."""
    modules = _corpus()
    assert len(modules) >= 20
    for module in modules:
        assert graph_to_dialect(dialect_to_graph(module)) == module, module.name


def test_graph_to_jer_to_graph_is_byte_identical_over_the_law_corpus():
    """P2's second law. Bytes, because canonical JER is where bytes are defined."""
    for module in _corpus():
        octets = graph_to_jer(dialect_to_graph(module))
        assert graph_to_jer(jer_to_graph(octets)) == octets, module.name


def test_the_projection_composes_with_the_mlir_text_rail():
    """`MLIR -> graph -> JER -> graph -> MLIR`, end to end.

    J4 part 2 proved the tree projection; this shows the graph slots into the same pipeline
    rather than being a parallel universe with its own reader.
    """
    from bcir.asn1.dialect import emit_mlir

    for module in _corpus():
        rebuilt = graph_to_dialect(jer_to_graph(graph_to_jer(dialect_to_graph(module))))
        assert parse_mlir(emit_mlir(rebuilt)) == (module,), module.name


def test_a_component_points_at_its_type_with_a_real_edge():
    """The difference from the tree, and the reason P2 is not just P1 with extra steps.

    In `dialect.py` a component names its type with a STRING the reader has to look up.
    Here it points at the type's row, so the reference is structural — which is what makes
    a mutually recursive pair of types representable rather than merely nameable.
    """
    module = next(m for m in _corpus() if any(t.components for t in m.types))
    graph = dialect_to_graph(module)
    components = [(i, n) for i, n in enumerate(graph.nodes) if n.kind == "component"]
    assert components, "the fixture has no components"
    typed = [n for _i, n in components if any(e.label == "type" for e in n.edges)]
    assert typed, "no component carried a structural edge to its type"
    for node in typed:
        target = next(e.target for e in node.edges if e.label == "type")
        assert graph.nodes[target].kind == "type"
        # And the edge agrees with the name the tree form carried, so nothing was invented.
        assert graph.nodes[target].label == dict(node.attributes)["type"]


def test_every_node_of_a_projected_dialect_graph_resolves():
    """The object set must actually type what the projection emits.

    A table that could not resolve its own producer's output would be decoration. This is
    the positive direction of the same check the unknown-kind test does negatively.
    """
    for module in _corpus():
        report = resolve(dialect_to_graph(module), DIALECT_NODE_CLASS)
        assert report.unresolved == (), (module.name, report.unresolved)
        assert report.edge_faults == (), (module.name, report.edge_faults)


def test_a_graph_with_no_module_root_is_refused_by_the_dialect_reader():
    """The one place `graph_to_dialect` raises: it was asked for a module and given none.

    Distinct from an unresolvable node — that is a graph the format can carry and this is a
    caller asking for something the graph does not contain.
    """
    for graph in (
        Graph(),
        Graph(nodes=(Node("type", "T"),), roots=(0,)),
        Graph(nodes=(Node("module", "a"), Node("module", "b")), roots=(0, 1)),
    ):
        try:
            graph_to_dialect(graph)
        except Asn1Error as error:
            assert "exactly one module root" in str(error)
        else:
            raise AssertionError("a graph with no single module root was accepted")


def test_the_projection_is_a_projection_and_not_a_filter():
    """Same discipline as J4 part 2: an R24-illegal module must survive unchanged.

    A module that lost what makes it illegal would come back legal, which is worse than a
    refusal because nothing downstream can tell.
    """
    by_name = {m.name: m for m in _corpus()}
    cer = by_name["cer_module"]
    assert graph_to_dialect(dialect_to_graph(cer)).rules == "cer"
    identity = by_name["transcode_identity"]
    operation = graph_to_dialect(dialect_to_graph(identity)).operations[0]
    assert operation.from_rules == operation.to_rules == "der"


def test_the_graph_is_the_same_type_under_another_transfer_syntax():
    """One schema, several syntaxes — the property roadmap §0 turns on."""
    graph = dialect_to_graph(_corpus()[0])
    value = graph_to_value(graph)
    basic = encode_jer(NODE_GRAPH, value, rules=JerRules.BASIC)
    assert value_to_graph(decode_jer(basic, NODE_GRAPH, rules=JerRules.BASIC)) == graph
