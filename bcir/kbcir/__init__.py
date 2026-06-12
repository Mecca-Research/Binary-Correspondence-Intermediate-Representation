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
# NOTE: `kbcir.microbench` is deliberately NOT re-exported here -- it doubles
# as a CLI (`python -m bcir.kbcir.microbench`), and importing it at package
# scope would shadow that entry point. Import it directly:
#     from bcir.kbcir.microbench import CalibratedProfile, calibrate_profile, ...
from .portfolio import (
    PolicyPortfolio,
    PortfolioEntry,
    ReplayCertificate,
    classify,
    episodes_from,
    replay_gate,
)
from .realize import Candidate, ChosenStep, RealizationResult, candidates_for, optimize
from .rcsp import Budget, Infeasible, optimize_constrained, pareto_plans
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
    "Budget",
    "Infeasible",
    "optimize_constrained",
    "pareto_plans",
    "PolicyPortfolio",
    "PortfolioEntry",
    "ReplayCertificate",
    "classify",
    "episodes_from",
    "replay_gate",
    "dag_shortest_path",
    "PERF",
    "POLICIES",
    "Policy",
    "weights",
]
