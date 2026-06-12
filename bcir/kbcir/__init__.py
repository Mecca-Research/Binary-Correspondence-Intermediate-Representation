"""K_BCIR: the IR-level cost algebra and min-plus realization optimizer (BCIR-3)."""

from .cost import (
    CostVector,
    HProfile,
    MemoryHierarchy,
    MemTier,
    TARGETS,
    TargetProfile,
    Theta,
    Tier,
)
from .realize import Candidate, ChosenStep, RealizationResult, candidates_for, optimize
from .semiring import dag_shortest_path
from .weights import PERF, POLICIES, Policy, weights

__all__ = [
    "CostVector",
    "HProfile",
    "MemoryHierarchy",
    "MemTier",
    "TARGETS",
    "TargetProfile",
    "Theta",
    "Tier",
    "Candidate",
    "ChosenStep",
    "RealizationResult",
    "candidates_for",
    "optimize",
    "dag_shortest_path",
    "PERF",
    "POLICIES",
    "Policy",
    "weights",
]
