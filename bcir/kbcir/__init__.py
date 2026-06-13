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
# NOTE: the calibration modules `kbcir.microbench` and `kbcir.bayescal` are
# deliberately NOT re-exported here -- microbench doubles as a CLI
# (`python -m bcir.kbcir.microbench`) and importing either at package scope would
# shadow that entry point. Import them directly:
#     from bcir.kbcir.microbench import CalibratedProfile, calibrate_profile, ...
#     from bcir.kbcir.bayescal import bayes_calibrate, abc_calibrate, ...
from .portfolio import (
    PolicyPortfolio,
    PortfolioEntry,
    ReplayCertificate,
    classify,
    episodes_from,
    replay_gate,
)
from .moegate import (
    FrozenGate,
    GNNGate,
    freeze,
    gate_replay_gate,
    ledger_labels,
    train_gate,
)
from .provenance import (
    ProvenanceManifest,
    ProvenanceMismatch,
    build_manifest,
    manifest_for,
    replay,
    reproduces,
)
from .accel import (
    AccelCertificate,
    FrozenRanker,
    LearnedRanker,
    SearchStats,
    accelerator_certificate,
    greedy_order,
    identity_order,
    optimize_ordered,
    ranker_samples,
    train_ranker,
    worst_order,
)
from .realize import Candidate, ChosenStep, RealizationResult, candidates_for, optimize
from .softdp import (
    SoftResult,
    free_energy,
    grad_free_energy_wrt_weights,
    softselect,
    temperature_sweep,
)
from .regret import (
    BoundaryVerdict,
    RegretLedger,
    RegretMeasurement,
    boundary_report,
    ledger_from_episodes,
    measure_regret,
)
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
    "FrozenGate",
    "GNNGate",
    "freeze",
    "gate_replay_gate",
    "ledger_labels",
    "train_gate",
    "ProvenanceManifest",
    "ProvenanceMismatch",
    "build_manifest",
    "manifest_for",
    "replay",
    "reproduces",
    "AccelCertificate",
    "FrozenRanker",
    "LearnedRanker",
    "SearchStats",
    "accelerator_certificate",
    "greedy_order",
    "identity_order",
    "optimize_ordered",
    "ranker_samples",
    "train_ranker",
    "worst_order",
    "BoundaryVerdict",
    "RegretLedger",
    "RegretMeasurement",
    "boundary_report",
    "ledger_from_episodes",
    "measure_regret",
    "SoftResult",
    "free_energy",
    "grad_free_energy_wrt_weights",
    "softselect",
    "temperature_sweep",
    "dag_shortest_path",
    "PERF",
    "POLICIES",
    "Policy",
    "weights",
]
