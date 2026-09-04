"""K_BCIR: the IR-level cost algebra and min-plus realization optimizer (BCIR-3).

Imports are **lazy** (PEP 562): the public names below resolve to their submodule
on first access, so the simple plan -> emit path (`cost` + `realize` + `weights` +
`rcsp` + `semiring` + `provenance`) does not eagerly drag in the learned/categorical
research stack (egraph / memory / twotruth / operad / calibloop / throttle / accel /
softdp / regret / portfolio / moegate / mapping). This keeps startup latency low for
the lowest, simplest use -- planning a kernel -- while the full API stays available
on demand. `from bcir.kbcir import optimize` and `bcir.kbcir.EGraph` both still work.

Deliberately NOT re-exported (import the submodule directly):
    from bcir.kbcir.microbench import calibrate_profile, ...   # doubles as a CLI
    from bcir.kbcir.bayescal import bayes_calibrate, ...
    from bcir.kbcir.memory import freeze                        # `freeze` here == moegate.freeze
"""

from __future__ import annotations

import importlib
import importlib.util

# name -> submodule. The grouping is the old eager `from .mod import (...)` block,
# now resolved on demand. (The single source of truth for what each module exports
# to the package namespace.)
_EXPORTS: dict[str, tuple[str, ...]] = {
    "cost": (
        "CostVector",
        "HProfile",
        "MemoryHierarchy",
        "MemTier",
        "TARGETS",
        "TargetProfile",
        "Theta",
        "Tier",
    ),
    "portfolio": (
        "PolicyPortfolio",
        "PortfolioEntry",
        "ReplayCertificate",
        "classify",
        "episodes_from",
        "replay_gate",
    ),
    "moegate": (
        "FrozenGate",
        "GNNGate",
        "freeze",
        "gate_replay_gate",
        "harden",
        "ledger_labels",
        "train_gate",
    ),
    "provenance": (
        "ProvenanceManifest",
        "ProvenanceMismatch",
        "build_manifest",
        "manifest_for",
        "replay",
        "reproduces",
    ),
    "egraph": (
        "EGraph",
        "EGraphResult",
        "Expr",
        "ResidentEGraph",
        "module_exprs",
        "optimize_expr",
        "saturate",
        "shared_blocks",
    ),
    "memory": (
        "DEFAULT_BUDGET",
        "MemoryModule",
        "UnsaturatedMemory",
        "fingerprint",
        "freeze_module",
        "from_result",
        "is_idempotent",
        "memory_artifacts",
        "reresolve",
        "try_freeze",
    ),
    "twotruth": (
        "Decision",
        "Graded",
        "GradedTruthLeak",
        "assert_classical",
        "decide",
        "from_conformal",
        "from_regret",
        "from_soft",
        "g_and",
        "g_not",
        "g_or",
        "is_classical",
    ),
    "mapping": ("CommutingSquare", "MappingFunction", "identity_dimmap", "support"),
    "operad": ("EnrichedOp", "EnrichedOperad", "TwoCell", "enrich_memory", "f_index", "f_label"),
    "calibloop": ("CalibrationCertificate", "close_loop", "measure_and_close", "rescore_plan"),
    "throttle": (
        "AmortizationCertificate",
        "ThrottleReport",
        "TIER_BUDGET",
        "certify",
        "measure_inference_cost",
    ),
    "accel": (
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
    ),
    "realize": ("Candidate", "ChosenStep", "RealizationResult", "candidates_for", "optimize"),
    "softdp": (
        "SoftResult",
        "free_energy",
        "grad_free_energy_wrt_weights",
        "softselect",
        "temperature_sweep",
    ),
    "regret": (
        "BoundaryVerdict",
        "RegretLedger",
        "RegretMeasurement",
        "boundary_report",
        "ledger_from_episodes",
        "measure_regret",
    ),
    "rcsp": (
        "Budget",
        "Infeasible",
        "feasible",
        "optimize_constrained",
        "pareto_plans",
        "plan_resources",
    ),
    "bundle": ("Bundle", "BundleCertificate", "find_bundles", "optimize_bundled"),
    "fusion": (
        "FUSIBLE_ACTIVATIONS",
        "NON_FUSIBLE_ACTIVATIONS",
        "FusionCertificate",
        "MatmulActivationEpilogue",
        "find_matmul_activation_epilogues",
        "is_fusible_activation",
        "optimize_fused",
    ),
    "layout": (
        "AOS",
        "SOA",
        "SINGLE_FIELD",
        "WHOLE_RECORD",
        "FieldAccess",
        "LayoutCertificate",
        "LayoutPlan",
        "analyze_resource_accesses",
        "certify_layout",
        "plan_layout",
        "pivot_resource_layout",
    ),
    "cache_predict": (
        "AccessShape",
        "CachePredictor",
        "ContentionPrediction",
        "predict",
        "rank_realizations",
    ),
    "compose": (
        "Call",
        "Cond",
        "CompositeResult",
        "Effect",
        "Function",
        "FunctionSummary",
        "Leaf",
        "Region",
        "Seq",
        "effect",
        "independent",
        "plan_composite",
        "plan_holds_for",
        "summarize",
        "worst_case_module",
    ),
    # NB: proof.{replay,reduce} are intentionally NOT re-exported here -- `replay` would
    # shadow provenance.replay. Reach them via `bcir.kbcir.proof.{replay,reduce}`.
    "proof": (
        "ClaimDecision",
        "DecisionRecord",
        "ReplayResult",
        "RewriteCertificate",
        "explain",
        "explain_text",
    ),
    "semiring": ("dag_shortest_path",),
    "differential": (
        "CheckResult",
        "Mismatch",
        "Report",
        "check_module",
        "check_budget",
        "check_verifier",
        "gen_module",
        "gen_illegal_module",
        "law_select",
        "run_campaign",
        "run_verifier_campaign",
        "shrink",
    ),
    "weights": ("PERF", "POLICIES", "Policy", "weights"),
    "allocator": (
        "ResourceProfile",
        "FrozenPlacer",
        "Placement",
        "DEFAULT_PLACER",
        "profile_resources",
        "place",
        "train_placer",
        "silicon_tier_capacity",
        "Pool",
        "PoolPlan",
        "live_intervals",
        "pool_plan",
    ),
    "static_memory": (
        "BankMemoryPlan",
        "ResourceBankBinding",
        "StaticAllocation",
        "StaticMemoryPlan",
        "plan_static_memory",
        "verify_static_memory_plan",
    ),
    "hardware_rl": (
        "HardwareEpisode",
        "HardwareOutcome",
        "HardwarePreference",
        "HardwarePromotionCertificate",
        "HardwareRewardPolicy",
        "HardwareSearchResult",
        "HardwareTopology",
        "TelemetryToken",
        "bounded_mcts",
        "certify_hardware_promotion",
        "encode_candidate_features",
        "encode_hardware_topology",
        "hardware_context_digest",
        "outcomes_to_preferences",
    ),
    "ham": (
        "HAMAccess",
        "HAMAction",
        "HAMBankPeak",
        "HAMExecutionPlan",
        "HAMResource",
        "HAMWorkload",
        "LoweredHAMExecution",
        "lower_ham_workload",
        "verify_ham_plan",
    ),
    "context_shard": (
        "ContextShardActivation",
        "ContextShardCatalog",
        "ContextShardEntry",
        "ContextShardManifest",
        "certify_context_activation",
        "read_context_shard_catalog",
        "read_context_shard_manifest",
        "verify_context_payload",
        "write_context_shard_catalog",
        "write_context_shard_manifest",
    ),
    "optimization_memory": (
        "OptimizationFact",
        "OptimizationMatch",
        "OptimizationMemory",
        "OptimizationPattern",
        "query_optimization_memory",
    ),
    "native_ai": ("NativeAIKernels", "NativeOptimizationIndex", "NativeQuantizedTensor"),
    "adaptive_transformer": (
        "AdaptiveCostReport",
        "AdaptiveLanguageSpec",
        "ExogenousAnchorSpec",
        "LoweredAdaptivePlan",
        "MultiPatchCostReport",
        "MultiPatchSpec",
        "ReferenceWindowSpec",
        "WidthScheduleSpec",
        "assess_adaptive_language",
        "assess_multi_patch",
        "fixed_residual_update",
        "fixed_residual_view",
        "lower_adaptive_language",
    ),
    "byte_latent": (
        "ByteDiffusionSpec",
        "ByteIngestDecision",
        "ByteIngestProfile",
        "ByteLatentCostReport",
        "ByteLatentSpec",
        "BytePatchSpec",
        "ByteSpeculationSpec",
        "ByteVocabularySpec",
        "ByteifiedTransplantPlan",
        "DraftVerification",
        "LearnedPatchPolicySpec",
        "LoweredByteLatentPlan",
        "MambaByteCostReport",
        "MambaByteSpec",
        "PatchLayout",
        "RawByteSequence",
        "TensorShape",
        "UnicodeByteReference",
        "assess_byte_latent",
        "assess_mambabyte",
        "choose_byte_ingest",
        "decode_raw_bytes",
        "discounted_batch_advantages",
        "encode_raw_sequence",
        "lower_byte_latent",
        "plan_byteified_transplant",
        "verify_byte_draft",
    ),
    "sequence_interfaces": (
        "AlignedTokenChunk",
        "CausalSeriesCodecSpec",
        "CausalSeriesEncoding",
        "ContinuedBPEExpansionPlan",
        "CrossTokenizerProjection",
        "DecodedToken",
        "ExpansionRow",
        "FiniteScalarQuantizerSpec",
        "FrozenTokenCodeSpec",
        "GrowthStage",
        "LoweredGrowthStage",
        "ProgressiveGrowthSpec",
        "SequenceInterfaceEvidence",
        "TokenizerExpansionCost",
        "UnigramPiece",
        "align_decoded_tokens",
        "assess_tokenizer_expansion",
        "canonical_substring_candidate",
        "decode_causal_series",
        "encode_causal_series",
        "lower_growth_stage",
        "pareto_sequence_interfaces",
        "plan_continued_bpe_expansion",
        "project_aligned_log_probabilities",
        "segment_unigram",
        "thunder_candidate_score",
    ),
    "sensing": (
        "PathSamples",
        "RegretSensor",
        "TelemetryDecision",
        "total_overhead",
        "full_resolution_overhead",
        "sense_by_ranker",
        "DEFAULT_MARGIN_THRESHOLD",
    ),
    "precision": (
        "Interval",
        "iadd",
        "isub",
        "imul_q8",
        "ulp_distance",
        "reduction_error_bound",
        "accuracy_bound",
        "meets_tolerance",
        "naive_reduce_q8",
        "compensated_reduce_q8",
        "exact_reduce_q8",
        "cancellation",
        "condition_milli",
        "Q8",
        "ULP",
    ),
}

_NAME_TO_MOD: dict[str, str] = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = sorted(_NAME_TO_MOD)

# `weights` is both an exported function and a submodule name. Bind the function
# eagerly so it is not shadowed: Python will not overwrite an existing parent
# attribute when the `weights` submodule is later imported, but an unbound name
# would be. (weights.py is tiny and on the core planning path anyway.)
from .weights import weights  # noqa: E402


def __getattr__(name: str):
    """Resolve an exported name (or a submodule) lazily on first access (PEP 562)."""
    mod = _NAME_TO_MOD.get(name)
    if mod is not None:
        value = getattr(importlib.import_module(f".{mod}", __name__), name)
        globals()[name] = value  # cache so subsequent access skips __getattr__
        return value
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        submodule = importlib.import_module(f".{name}", __name__)
        globals()[name] = submodule
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
