"""BCIR semantic model (BCIR-0..2): lanes, opcodes, resources, claims, phases."""

from .graph import (
    Claim,
    Lifetime,
    Module,
    Phase,
    Resource,
    Timing,
    derived_claim_domain,
    phase_graph_has_cycle,
    topological_phase_ids,
)
from .lanes import ISOLATED_DOMAINS, Domain, Lane, StrideClass
from .opcodes import ATOMIC_OPCODES, Opcode

__all__ = [
    "Claim",
    "Domain",
    "ISOLATED_DOMAINS",
    "derived_claim_domain",
    "Lane",
    "Lifetime",
    "Module",
    "ATOMIC_OPCODES",
    "Opcode",
    "Phase",
    "Resource",
    "StrideClass",
    "Timing",
    "phase_graph_has_cycle",
    "topological_phase_ids",
]
