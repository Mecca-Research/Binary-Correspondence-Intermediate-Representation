"""P1 and P2 — the flat node-table graph, and its commuting projection from the dialect.

`docs/BCIR_JSON_PROGRAM_REPRESENTATION.md` calls P1 *"the single largest unsolved design
question in the proposal"*: JSON is a tree and programs are graphs, so a program
representation needs a **cycle-tolerant** descriptor. `jer_plan.py` refuses schema recursion
beyond depth 64, and a mutually recursive pair of functions is a cycle no nesting can hold.

**The answer is a flat node table with integer index edges** — the shape LLVM bitcode and
MLIR bytecode both use, and for the same reason. A node is a row; an edge names a row by
*index*, not by nesting. Cycles then cost nothing: `nodes[0] -> nodes[1] -> nodes[0]` is two
ordinary rows, and the depth of the JSON is two regardless of how tangled the graph is. Every
alternative — JSON Pointer, `@id`, or nesting with back-references — reintroduces either a
resolution convention or unbounded depth.

**A finding that changed the design, and it is worth reading before the schema below.** §4.1
of the note proposes that a node's payload be an X.681 **OPEN TYPE** resolved by a sibling's
identifier through an X.682 §10.19 table constraint, with §12.9's extensibility making
"I do not know this node" ordinary traffic. The first half works. The second does not, in
JER specifically: X.697 §41 says the encoding of an open type value *is* the encoding of the
contained value, and — unlike XER §8.5 — offers **no hexadecimal fallback**. `jer.py` refuses
an unresolvable open type outright, and it is right to: there is no JSON spelling for "some
octets whose type I do not know". So an open-type payload would make an unknown node
*unencodable*, which is the exact opposite of P1's exit gate.

The resolution keeps the mechanism and moves the layer. The object set remains the **typing
authority** — it says what a `kind`'s payload must be, and `resolve` checks it — but the
payload travels in ordinary declared components, and resolution is an **enrichment** that
reports what it found. That is the same posture the repository already takes elsewhere: the
resolved result is recorded *alongside* the octets, never in place of them. An unresolvable
node then decodes, re-encodes byte-identically, and is reported as unresolved, which is what
"unresolvable is a value, not a fault" has to mean if the document is to survive at all.

**Content addressing is the secondary identity mechanism**, exactly as §4.1 says: the table
gives typed intra-program edges, the SHA-256 of a canonical sub-encoding gives global
identity. `content_address` walks cycle-safely, because the whole point of the node table is
that cycles exist.

P2 is the second half: `dialect_to_graph` / `graph_to_dialect` project the `bcir.asn1.*`
dialect into this shape and back. J4 part 2 already proved a commuting projection into a
*tree*; the same two laws are proved here into the *graph*, which is what the P2 gate asks
for and what a program representation actually needs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .jer import JerRules, decode_jer, encode_jer
from .schema import Component, ObjectSetTable, Primitive, Sequence, SequenceOf
from .tags import Asn1Error, Universal

#: The module OID for the graph schema, in the same private-enterprise arc.
GRAPH_MODULE_OID = (1, 3, 6, 1, 4, 1, 62596, 34)

#: Bumped when the node table's shape changes in a way an older reader would misread.
GRAPH_VERSION = 1

_INT = Primitive(Universal.INTEGER, "INTEGER")
_UTF8 = Primitive(Universal.UTF8_STRING, "UTF8String")


# --- the schema -------------------------------------------------------------------------------

ATTRIBUTE_TYPE = Sequence(
    (
        Component("name", _UTF8),
        Component("value", _UTF8),
    ),
    name="Attribute",
)

#: An edge names its target by INDEX. That single choice is what makes the representation
#: cycle-tolerant, and it is why this is a node TABLE rather than a nested document.
#:
#: `label` is the edge's role in the source node (an operand position, a component name),
#: so a node with several outgoing edges keeps them distinguishable without relying on
#: order — though order is preserved too, because SEQUENCE OF is ordered and a program's
#: operand list is not a set.
EDGE_TYPE = Sequence(
    (
        Component("label", _UTF8),
        Component("target", _INT),
    ),
    name="Edge",
)

NODE_TYPE = Sequence(
    (
        #: The object-set row key. The table types the node; see the module docstring for why
        #: it is not an OPEN TYPE discriminator.
        Component("kind", _UTF8),
        Component("label", _UTF8, default=""),
        Component("attributes", SequenceOf(ATTRIBUTE_TYPE, "SEQUENCE OF Attribute"), default=()),
        Component("edges", SequenceOf(EDGE_TYPE, "SEQUENCE OF Edge"), default=()),
    ),
    name="Node",
)

NODE_GRAPH = Sequence(
    (
        Component("version", _INT),
        Component("nodes", SequenceOf(NODE_TYPE, "SEQUENCE OF Node")),
        #: Entry points, by index. A graph with no roots is legal and means "a library" —
        #: refusing it would make the schema unable to describe half of what it is for.
        Component("roots", SequenceOf(_INT, "SEQUENCE OF INTEGER"), default=()),
    ),
    name="NodeGraph",
)


# --- the in-memory form ---------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    label: str
    target: int


@dataclass(frozen=True)
class Node:
    kind: str
    label: str = ""
    attributes: tuple[tuple[str, str], ...] = ()
    edges: tuple[Edge, ...] = ()


@dataclass(frozen=True)
class Graph:
    nodes: tuple[Node, ...] = ()
    roots: tuple[int, ...] = ()

    def attribute(self, index: int, name: str, default: str | None = None) -> str | None:
        for key, value in self.nodes[index].attributes:
            if key == name:
                return value
        return default


def graph_to_value(graph: Graph) -> dict:
    return {
        "version": GRAPH_VERSION,
        "nodes": tuple(
            {
                "kind": node.kind,
                "label": node.label,
                "attributes": tuple({"name": k, "value": v} for k, v in node.attributes),
                "edges": tuple({"label": e.label, "target": e.target} for e in node.edges),
            }
            for node in graph.nodes
        ),
        "roots": tuple(graph.roots),
    }


def value_to_graph(value: dict) -> Graph:
    version = value.get("version", 0)
    if version != GRAPH_VERSION:
        raise Asn1Error(
            f"node-graph version {version} is not {GRAPH_VERSION}; this reader refuses "
            f"rather than inferring the shape from which members are present"
        )
    return Graph(
        nodes=tuple(
            Node(
                kind=n["kind"],
                label=n.get("label", ""),
                attributes=tuple((a["name"], a["value"]) for a in n.get("attributes", ())),
                edges=tuple(Edge(label=e["label"], target=e["target"]) for e in n.get("edges", ())),
            )
            for n in value.get("nodes", ())
        ),
        roots=tuple(value.get("roots", ())),
    )


def graph_to_jer(graph: Graph, *, rules: JerRules = JerRules.CANONICAL) -> bytes:
    return encode_jer(NODE_GRAPH, graph_to_value(graph), rules=rules)


def jer_to_graph(data: bytes, *, rules: JerRules = JerRules.CANONICAL) -> Graph:
    return value_to_graph(decode_jer(data, NODE_GRAPH, rules=rules))


# --- X.681/X.682 typing, as an enrichment ------------------------------------------------------


@dataclass(frozen=True)
class Resolution:
    """What the object set says about one node — including that it says nothing.

    `payload` is the type the table's type-field column names for this `kind`, or None when
    the row is absent. `reason` is empty exactly when the node resolved.

    This is a VALUE, and that is the whole point. X.681 §12.9 lets a conforming peer use an
    object outside an extensible set, so an unknown node is ordinary traffic for a versioned
    program format — and X.697 §41 gives JER no way to carry one as an open type, so it has
    to be carried as ordinary components and reported here instead.
    """

    index: int
    kind: str
    payload: object | None = None
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return not self.reason


@dataclass(frozen=True)
class EdgeFault:
    """An edge the graph itself cannot honour. Also a value, for the same reason."""

    source: int
    label: str
    target: int
    reason: str


@dataclass(frozen=True)
class GraphReport:
    """The enrichment: resolutions and faults, recorded beside the graph rather than raised.

    Nothing in this record can make a graph fail to decode or re-encode. A caller decides
    what an unresolved node or a dangling edge means for *its* purpose — a loader may refuse
    where a viewer displays a placeholder — which is a policy decision the format must not
    make on their behalf.
    """

    resolutions: tuple[Resolution, ...] = ()
    edge_faults: tuple[EdgeFault, ...] = ()

    @property
    def unresolved(self) -> tuple[Resolution, ...]:
        return tuple(r for r in self.resolutions if not r.resolved)


def resolve(
    graph: Graph,
    table: ObjectSetTable,
    *,
    kind_field: str = "kind",
    payload_field: str = "Payload",
    edge_field: str | None = "TargetKind",
) -> GraphReport:
    """Type every node and edge against the object set. Never raises on the graph's content.

    §10.19's row selection is used exactly as the standard defines it — match the referenced
    column against the node's `kind` — so this is a reference with a resolution *law* rather
    than a convention, which is §4.1's whole argument against inventing `$ref`.

    `edge_field` names the column that says what an edge's target must be. When the table
    carries it, a mistyped edge is reported; when it does not, edges are only checked for
    being in range. An absent column is not a fault: not every class describes its edges.
    """
    resolutions: list[Resolution] = []
    faults: list[EdgeFault] = []
    for index, node in enumerate(graph.nodes):
        rows = table.select({kind_field: node.kind})
        if not rows:
            reason = f"no row of {table.object_class} has {kind_field} = {node.kind!r}" + (
                "; the set is extensible, so this is a peer using an object outside it (X.681 12.9)"
                if table.extensible
                else "; the set is NOT extensible, so this is a schema violation"
            )
            resolutions.append(Resolution(index, node.kind, None, reason))
            continue
        if len(rows) > 1:
            resolutions.append(
                Resolution(
                    index,
                    node.kind,
                    None,
                    f"{len(rows)} rows match {kind_field} = {node.kind!r}; a table constraint "
                    f"must select at most one",
                )
            )
            continue
        resolutions.append(Resolution(index, node.kind, rows[0].get(payload_field)))

    for index, node in enumerate(graph.nodes):
        expected = None
        rows = table.select({kind_field: node.kind})
        if len(rows) == 1 and edge_field:
            expected = rows[0].get(edge_field)
        for edge in node.edges:
            if not 0 <= edge.target < len(graph.nodes):
                faults.append(
                    EdgeFault(
                        index,
                        edge.label,
                        edge.target,
                        f"target {edge.target} is outside the node table "
                        f"(0..{len(graph.nodes) - 1})",
                    )
                )
                continue
            if expected is not None and graph.nodes[edge.target].kind != expected:
                faults.append(
                    EdgeFault(
                        index,
                        edge.label,
                        edge.target,
                        f"the table says {node.kind!r} points at {expected!r}, but node "
                        f"{edge.target} is {graph.nodes[edge.target].kind!r}",
                    )
                )
    return GraphReport(tuple(resolutions), tuple(faults))


# --- content addressing ------------------------------------------------------------------------


def content_address(graph: Graph, index: int) -> str:
    """The SHA-256 of the canonical encoding of the subgraph reachable from `index`.

    Cycle-safe by construction: the walk visits each node once and an edge back into the
    visited set is written as its *position in the traversal*, not followed. That makes the
    address a function of the subgraph's shape rather than of its position in the enclosing
    table, so the same function hashes identically in two different programs — which is the
    property §4.1 wants from content addressing and the reason it is worth having beside the
    typed edges rather than instead of them.
    """
    if not 0 <= index < len(graph.nodes):
        raise Asn1Error(f"node {index} is outside the table (0..{len(graph.nodes) - 1})")
    order: dict[int, int] = {}
    collected: list[Node] = []

    def visit(at: int) -> int:
        if at in order:
            return order[at]
        position = len(order)
        order[at] = position
        node = graph.nodes[at]
        collected.append(node)
        for edge in node.edges:
            if 0 <= edge.target < len(graph.nodes):
                visit(edge.target)
        return position

    visit(index)
    # Renumber every edge into traversal order, so the encoding carries no absolute index.
    renumbered = tuple(
        Node(
            kind=node.kind,
            label=node.label,
            attributes=node.attributes,
            edges=tuple(
                Edge(edge.label, order[edge.target] if 0 <= edge.target < len(graph.nodes) else -1)
                for edge in node.edges
            ),
        )
        for node in collected
    )
    subgraph = Graph(nodes=renumbered, roots=(0,))
    return hashlib.sha256(graph_to_jer(subgraph)).hexdigest()


# --- P2: the dialect projection ------------------------------------------------------------------
#
# J4 part 2 projected the `bcir.asn1.*` dialect into a TREE and proved both round trips.
# This projects the same dialect into the node GRAPH and proves the same two laws, which is
# what P2's gate asks for. The difference is not cosmetic: in the tree, a type's components
# are nested inside it and a component's type is a NAME that the reader has to look up. Here
# the component points at its type with an ordinary edge, so the reference is structural and
# a mutually recursive pair of types is representable rather than merely nameable.

_MODULE, _TYPE, _COMPONENT, _OPERATION = "module", "type", "component", "operation"


def dialect_to_graph(module) -> Graph:
    """Project one `DialectModule` into the node table."""
    from .dialect import DialectModule

    if not isinstance(module, DialectModule):
        raise Asn1Error("dialect_to_graph takes a DialectModule")
    nodes: list[Node] = []
    type_index: dict[str, int] = {}

    def attributes(pairs) -> tuple[tuple[str, str], ...]:
        # Sorted, so the projection is a function of the record rather than of dict order,
        # and `graph -> jer` is therefore stable across Python versions.
        return tuple(sorted((k, v) for k, v in pairs if v is not None))

    root = len(nodes)
    nodes.append(
        Node(
            kind=_MODULE,
            label=module.name,
            attributes=attributes(
                (
                    ("oid", ",".join(str(a) for a in module.oid)),
                    ("rules", module.rules),
                    ("default_tagging", module.default_tagging),
                )
            ),
        )
    )

    for kind in module.types:
        type_index[kind.name] = len(nodes)
        nodes.append(
            Node(
                kind=_TYPE,
                label=kind.name,
                attributes=attributes(
                    (
                        ("kind", kind.kind),
                        ("universal", None if kind.universal is None else str(kind.universal)),
                        (
                            "constraint_low",
                            None if kind.constraint_low is None else str(kind.constraint_low),
                        ),
                        (
                            "constraint_high",
                            None if kind.constraint_high is None else str(kind.constraint_high),
                        ),
                        ("size_low", None if kind.size_low is None else str(kind.size_low)),
                        ("size_high", None if kind.size_high is None else str(kind.size_high)),
                    )
                ),
            )
        )

    module_edges: list[Edge] = []
    for kind in module.types:
        at = type_index[kind.name]
        module_edges.append(Edge("type", at))
        edges: list[Edge] = []
        if kind.element is not None:
            # A structural edge where the tree had a name. An element type defined AFTER
            # this one — or the type itself — is now an ordinary index.
            if kind.element in type_index:
                edges.append(Edge("element", type_index[kind.element]))
            else:
                nodes[at] = Node(
                    kind=_TYPE,
                    label=kind.name,
                    attributes=nodes[at].attributes + (("element", kind.element),),
                    edges=(),
                )
        for component in kind.components:
            component_at = len(nodes)
            component_edges = (
                [Edge("type", type_index[component.type])] if component.type in type_index else []
            )
            nodes.append(
                Node(
                    kind=_COMPONENT,
                    label=component.name,
                    attributes=attributes(
                        (
                            ("type", component.type),
                            ("tag", None if component.tag is None else str(component.tag)),
                            ("tagging", component.tagging),
                            ("optional", "1" if component.optional else None),
                            ("has_default", "1" if component.has_default else None),
                            ("default_value", component.default_value),
                        )
                    ),
                    edges=tuple(component_edges),
                )
            )
            edges.append(Edge("component", component_at))
        if edges:
            nodes[at] = Node(
                kind=_TYPE,
                label=nodes[at].label,
                attributes=nodes[at].attributes,
                edges=tuple(list(nodes[at].edges) + edges),
            )

    for operation in module.operations:
        at = len(nodes)
        nodes.append(
            Node(
                kind=_OPERATION,
                label=operation.name,
                attributes=attributes(
                    (
                        ("op", operation.op),
                        ("type", operation.type),
                        ("rules", operation.rules),
                        ("strict_der", "1" if operation.strict_der else None),
                        ("strict_canonical", "1" if operation.strict_canonical else None),
                        ("source", operation.source),
                        ("from", operation.from_rules),
                        ("to", operation.to_rules),
                        ("preserve_value", "1" if operation.preserve_value else None),
                        ("native", operation.native),
                        ("additive", "1" if operation.additive else None),
                    )
                ),
                edges=(Edge("type", type_index[operation.type]),)
                if operation.type in type_index
                else (),
            )
        )
        module_edges.append(Edge("operation", at))

    nodes[root] = Node(
        kind=_MODULE,
        label=nodes[root].label,
        attributes=nodes[root].attributes,
        edges=tuple(module_edges),
    )
    return Graph(nodes=tuple(nodes), roots=(root,))


def graph_to_dialect(graph: Graph):
    """The inverse. Reads the node table back into a `DialectModule`."""
    from .dialect import DialectComponent, DialectModule, DialectOperation, DialectType

    roots = [i for i in graph.roots if 0 <= i < len(graph.nodes) and graph.nodes[i].kind == _MODULE]
    if len(roots) != 1:
        raise Asn1Error(f"a dialect graph has exactly one module root, found {len(roots)}")
    root = roots[0]
    module_node = graph.nodes[root]

    def attr(index: int, name: str, default=None):
        return graph.attribute(index, name, default)

    def number(index: int, name: str) -> int | None:
        raw = attr(index, name)
        return None if raw is None else int(raw)

    types: list[DialectType] = []
    operations: list[DialectOperation] = []
    for edge in module_node.edges:
        at = edge.target
        node = graph.nodes[at]
        if edge.label == "type":
            components = []
            element = attr(at, "element")
            for inner in node.edges:
                if inner.label == "element":
                    element = graph.nodes[inner.target].label
                elif inner.label == "component":
                    c = inner.target
                    components.append(
                        DialectComponent(
                            name=graph.nodes[c].label,
                            type=attr(c, "type", ""),
                            tag=number(c, "tag"),
                            tagging=attr(c, "tagging"),
                            optional=attr(c, "optional") == "1",
                            has_default=attr(c, "has_default") == "1",
                            default_value=attr(c, "default_value"),
                        )
                    )
            types.append(
                DialectType(
                    name=node.label,
                    kind=attr(at, "kind", ""),
                    universal=number(at, "universal"),
                    element=element,
                    constraint_low=number(at, "constraint_low"),
                    constraint_high=number(at, "constraint_high"),
                    size_low=number(at, "size_low"),
                    size_high=number(at, "size_high"),
                    components=tuple(components),
                )
            )
        elif edge.label == "operation":
            operations.append(
                DialectOperation(
                    op=attr(at, "op", ""),
                    name=node.label,
                    type=attr(at, "type", ""),
                    rules=attr(at, "rules"),
                    strict_der=attr(at, "strict_der") == "1",
                    strict_canonical=attr(at, "strict_canonical") == "1",
                    source=attr(at, "source"),
                    from_rules=attr(at, "from"),
                    to_rules=attr(at, "to"),
                    preserve_value=attr(at, "preserve_value") == "1",
                    native=attr(at, "native"),
                    additive=attr(at, "additive") == "1",
                )
            )

    oid = attr(root, "oid", "")
    return DialectModule(
        name=module_node.label,
        oid=tuple(int(a) for a in oid.split(",")) if oid else (),
        rules=attr(root, "rules", ""),
        default_tagging=attr(root, "default_tagging", ""),
        types=tuple(types),
        operations=tuple(operations),
    )


#: The object set that types a dialect graph — X.681 §13's associated table, with a
#: `TargetKind` column so `resolve` can check that an edge points at what the class says.
#: Extensible, because a graph carrying a node kind this table does not know is exactly the
#: versioned-peer case §12.9 exists for.
DIALECT_NODE_CLASS = ObjectSetTable(
    "BCIR-DIALECT-NODE",
    rows=(
        {"kind": _MODULE, "Payload": _UTF8, "TargetKind": None},
        {"kind": _TYPE, "Payload": _UTF8, "TargetKind": None},
        {"kind": _COMPONENT, "Payload": _UTF8, "TargetKind": _TYPE},
        {"kind": _OPERATION, "Payload": _UTF8, "TargetKind": _TYPE},
    ),
    extensible=True,
)


__all__ = [
    "ATTRIBUTE_TYPE",
    "DIALECT_NODE_CLASS",
    "EDGE_TYPE",
    "GRAPH_MODULE_OID",
    "GRAPH_VERSION",
    "NODE_GRAPH",
    "NODE_TYPE",
    "Edge",
    "EdgeFault",
    "Graph",
    "GraphReport",
    "Node",
    "Resolution",
    "content_address",
    "dialect_to_graph",
    "graph_to_dialect",
    "graph_to_jer",
    "graph_to_value",
    "jer_to_graph",
    "resolve",
    "value_to_graph",
]
