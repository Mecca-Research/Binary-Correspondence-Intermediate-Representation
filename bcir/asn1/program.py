"""P3 — carrying scope, lifetime and timing through the node graph.

P3's gate: *"R19/R20/R21 verdicts identical whether the input arrived as MLIR or as JER."*

That sentence is doing more work than it looks. R19 and R20 are the synchronous-timing laws
and R21 is pointer lifetime; all three read **optional** metadata that most claims do not
carry, and all three are *order-sensitive* — R20 tracks which clock domain last wrote each
resource, and R21 walks the phase and claim order maintaining a freed set. A projection that
preserved every field but perturbed the order would produce a module that verified
differently while looking identical field by field. So the property under test is not "the
fields survive"; it is **"the verdict survives"**, which is strictly stronger and is what
§4.3 of the note means by saying BCIR already has these laws and JER must project them.

**Nothing here re-implements a law.** `verify_timing` and `verify_lifetime` are called on
both sides and their diagnostics compared. A second implementation of R19–R21 that agreed
with the first would prove nothing about the projection, and a second one that disagreed
would be a bug in the checker rather than a finding about the format.

**Order is carried structurally, not as an attribute.** Phases are edges from the program
node in order, and claims are edges from their phase in order; `SEQUENCE OF` is ordered, so
the encoding preserves it without a rank field anybody could get wrong. That is the same
reason the P1 node table uses ordered edge lists rather than a set with sort keys.
"""

from __future__ import annotations

from .graph import Edge, Graph, Node, graph_to_jer, jer_to_graph
from .tags import Asn1Error

#: Node kinds this projection emits. Named constants because `resolve` types graphs against
#: an object set keyed on exactly these strings.
PROGRAM, PHASE, CLAIM, RESOURCE = "program", "phase", "claim", "resource"

#: Every `Claim` field the projection carries, with the constructor keyword it feeds. The
#: list is explicit rather than derived from `dataclasses.fields` so that a NEW field on
#: `Claim` fails the round-trip test loudly instead of being silently dropped — a projection
#: that quietly forgot a field is exactly what the R19–R21 verdict comparison exists to
#: catch, and it should catch it by failing, not by being lenient.
#: Plain integers.
_CLAIM_INTS = ("id", "count", "stride_k", "offset", "tolerance_ulp", "quantized_bits")
#: Plain strings. `verify` is here rather than with the booleans: it LOOKS boolean and is
#: a string on every real claim, and treating it as a flag would silently rewrite every
#: value that is not exactly "" into "1" — a projection that changes what it carries.
_CLAIM_STRS = (
    "hazard",
    "op",
    "cost_class",
    "precision",
    "callee_sig",
    "bounds_provenance",
    "bounds",
    "verify",
)
_CLAIM_BOOLS = ("dynamic", "volatile")
#: Integer-or-None. `primary_rid` distinguishes "no primary resource" from resource 0, so
#: it cannot default to zero the way the plain integers can.
_CLAIM_OPT_INTS = ("primary_rid",)
#: Ordered integer tuples, carried as comma-joined text like `rd`/`wr`.
_CLAIM_TUPLES = ("imm",)
#: Enumerations, carried by NAME. The name is stable across a renumbering of the enum and
#: is what a human reading the JSON needs; the number is an implementation detail.
_CLAIM_ENUMS = (
    ("opcode", "Opcode"),
    ("lane", "Lane"),
    ("stride_class", "StrideClass"),
    ("domain", "Domain"),
)
_TIMING_INTS = ("latency_cycles", "min_throughput_q16", "clock_frequency_mhz", "setup_hold_margin")
_TIMING_STRS = ("clock_domain", "power_domain", "sync_type")

_RESOURCE_INTS = ("rid", "elem_bytes", "align", "priority", "map_gen", "data_gen")
_RESOURCE_STRS = ("layout", "access", "name")


def _text(value) -> str:
    return "" if value is None else str(value)


def _rids(values) -> str:
    return ",".join(str(v) for v in values)


def _unrids(text: str) -> tuple[int, ...]:
    return tuple(int(v) for v in text.split(",") if v)


def module_to_graph(module) -> Graph:
    """Project a `Module` into the P1 node table, preserving phase and claim ORDER.

    Resources become their own nodes so a claim's reads and writes could later be edges
    rather than identifier lists. They are carried as identifiers today because R20 and R21
    are defined over resource *ids* and inventing an indirection the laws do not use would
    be adding a mechanism to a projection whose whole job is to change nothing.
    """
    from ..model.graph import Module as _Module

    if not isinstance(module, _Module):
        raise Asn1Error("module_to_graph takes a bcir.model.graph.Module")
    nodes: list[Node] = []
    root = 0
    nodes.append(
        Node(
            kind=PROGRAM,
            label=module.name,
            attributes=tuple(
                sorted(
                    (
                        ("cacheline", str(module.cacheline)),
                        ("align", str(module.align)),
                        ("target", _text(module.target)),
                    )
                )
            ),
        )
    )

    root_edges: list[Edge] = []
    # `module.resources` is a MAPPING from rid to Resource, not a sequence. Carried in
    # sorted rid order so the projection is a function of the module rather than of the
    # dict's insertion history — the same reason attributes are sorted.
    for rid in sorted(module.resources):
        resource = module.resources[rid]
        at = len(nodes)
        nodes.append(
            Node(
                kind=RESOURCE,
                label=_text(resource.name),
                attributes=tuple(
                    sorted(
                        [(k, _text(getattr(resource, k))) for k in _RESOURCE_INTS]
                        + [(k, _text(getattr(resource, k))) for k in _RESOURCE_STRS]
                        + [("domain", resource.domain.name), ("shape", _rids(resource.shape))]
                    )
                ),
            )
        )
        root_edges.append(Edge("resource", at))

    for phase in module.phases:
        phase_at = len(nodes)
        nodes.append(
            Node(
                kind=PHASE,
                label=str(phase.phase_id),
                attributes=tuple(
                    sorted(
                        (
                            ("deps", _rids(phase.deps)),
                            ("event", _text(getattr(phase, "event", "") or "")),
                        )
                    )
                ),
            )
        )
        root_edges.append(Edge("phase", phase_at))

        claim_edges: list[Edge] = []
        for claim in phase.claims:
            at = len(nodes)
            attributes: list[tuple[str, str]] = [("rd", _rids(claim.rd)), ("wr", _rids(claim.wr))]
            attributes += [(k, _text(getattr(claim, k))) for k in _CLAIM_INTS]
            attributes += [(k, _text(getattr(claim, k))) for k in _CLAIM_STRS]
            attributes += [(k, "1" if getattr(claim, k) else "0") for k in _CLAIM_BOOLS]
            attributes += [(k, _rids(getattr(claim, k))) for k in _CLAIM_TUPLES]
            attributes += [(k, getattr(claim, k).name) for k, _cls in _CLAIM_ENUMS]
            # An absent optional integer carries no attribute at all, so "unset" and
            # "zero" stay distinguishable.
            attributes += [
                (k, str(getattr(claim, k)))
                for k in _CLAIM_OPT_INTS
                if getattr(claim, k) is not None
            ]
            timing = getattr(claim, "timing", None)
            if timing is not None:
                # A PRESENCE marker, separate from the fields. R19 is vacuous when
                # `claim.timing is None` and constraining when it is a zero-valued block, so
                # "absent" and "all defaults" must not collapse into one spelling.
                attributes.append(("timing", "1"))
                attributes += [
                    (f"timing.{k}", _text(getattr(timing, k))) for k in _TIMING_INTS + _TIMING_STRS
                ]
                attributes.append(("timing.critical_path", "1" if timing.critical_path else "0"))
            lifetime = getattr(claim, "lifetime", None)
            if lifetime is not None:
                attributes.append(("lifetime", "1"))
                attributes.append(("lifetime.event", _text(lifetime.event)))
                attributes.append(("lifetime.epoch", _text(lifetime.epoch)))
            nodes.append(
                Node(kind=CLAIM, label=str(claim.id), attributes=tuple(sorted(attributes)))
            )
            claim_edges.append(Edge("claim", at))
        nodes[phase_at] = Node(
            kind=PHASE,
            label=nodes[phase_at].label,
            attributes=nodes[phase_at].attributes,
            edges=tuple(claim_edges),
        )

    nodes[root] = Node(
        kind=PROGRAM,
        label=nodes[root].label,
        attributes=nodes[root].attributes,
        edges=tuple(root_edges),
    )
    return Graph(nodes=tuple(nodes), roots=(root,))


def graph_to_module(graph: Graph):
    """The inverse. Rebuilds the `Module` the laws are defined over."""
    from ..model.graph import Claim, Lifetime, Module, Phase, Resource, Timing
    from ..model import Domain, Lane, Opcode, StrideClass

    roots = [i for i in graph.roots if 0 <= i < len(graph.nodes) and graph.nodes[i].kind == PROGRAM]
    if len(roots) != 1:
        raise Asn1Error(f"a program graph has exactly one program root, found {len(roots)}")
    root = roots[0]

    def attr(index: int, name: str, default: str = "") -> str:
        return graph.attribute(index, name, default)

    phases = []
    for edge in graph.nodes[root].edges:
        if edge.label != "phase":
            continue
        at = edge.target
        claims = []
        for inner in graph.nodes[at].edges:
            if inner.label != "claim":
                continue
            c = inner.target
            timing = None
            if attr(c, "timing") == "1":
                timing = Timing(
                    **{k: int(attr(c, f"timing.{k}") or 0) for k in _TIMING_INTS},
                    **{k: attr(c, f"timing.{k}") for k in _TIMING_STRS},
                    critical_path=attr(c, "timing.critical_path") == "1",
                )
            lifetime = None
            if attr(c, "lifetime") == "1":
                lifetime = Lifetime(
                    event=attr(c, "lifetime.event"), epoch=int(attr(c, "lifetime.epoch") or 0)
                )
            enums = {"Opcode": Opcode, "Lane": Lane, "StrideClass": StrideClass, "Domain": Domain}
            optional_ints = {
                k: (None if graph.attribute(c, k, None) is None else int(attr(c, k)))
                for k in _CLAIM_OPT_INTS
            }
            claims.append(
                Claim(
                    rd=_unrids(attr(c, "rd")),
                    wr=_unrids(attr(c, "wr")),
                    timing=timing,
                    lifetime=lifetime,
                    **{k: enums[cls][attr(c, k)] for k, cls in _CLAIM_ENUMS},
                    **{k: _unrids(attr(c, k)) for k in _CLAIM_TUPLES},
                    **optional_ints,
                    **{k: int(attr(c, k) or 0) for k in _CLAIM_INTS},
                    **{k: attr(c, k) for k in _CLAIM_STRS},
                    **{k: attr(c, k) == "1" for k in _CLAIM_BOOLS},
                )
            )
        phases.append(
            Phase(
                phase_id=int(graph.nodes[at].label),
                deps=_unrids(attr(at, "deps")),
                claims=claims,
                event=attr(at, "event") or "",
            )
        )

    from ..model import Domain as _Domain

    resources = {}
    for edge in graph.nodes[root].edges:
        if edge.label != "resource":
            continue
        at = edge.target
        resource = Resource(
            domain=_Domain[attr(at, "domain")],
            shape=_unrids(attr(at, "shape")),
            **{k: int(attr(at, k) or 0) for k in _RESOURCE_INTS},
            **{k: attr(at, k) for k in _RESOURCE_STRS},
        )
        resources[resource.rid] = resource

    return Module(
        name=graph.nodes[root].label,
        cacheline=int(attr(root, "cacheline") or 0),
        align=int(attr(root, "align") or 0),
        target=attr(root, "target"),
        resources=resources,
        phases=phases,
    )


def module_to_jer(module, **kwargs) -> bytes:
    return graph_to_jer(module_to_graph(module), **kwargs)


def jer_to_module(data: bytes, **kwargs):
    return graph_to_module(jer_to_graph(data, **kwargs))


def verdicts(module) -> tuple[tuple[str, str], ...]:
    """R19, R20 and R21's diagnostics as comparable tuples, in order.

    ORDER IS PART OF THE VERDICT. R20 reports a crossing against the domain that last wrote
    the resource and R21 walks a freed set, so two modules with the same diagnostics in a
    different order did not verify the same way — they verified two different programs that
    happen to be equally illegal. Sorting here would hide exactly the projection bug most
    worth catching.
    """
    from ..verify import verify_lifetime, verify_timing

    return tuple(
        (d.law, d.message) for d in list(verify_timing(module)) + list(verify_lifetime(module))
    )


__all__ = [
    "CLAIM",
    "PHASE",
    "PROGRAM",
    "RESOURCE",
    "graph_to_module",
    "jer_to_module",
    "module_to_graph",
    "module_to_jer",
    "verdicts",
]
