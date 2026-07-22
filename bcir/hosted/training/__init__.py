"""Offline-first contracts and lazy tensor-backed stages for BCIR model training."""
from __future__ import annotations

from .bpe import BytePairTokenizer, token_source_from_corpus
from .architecture_spec import SmallModelSpec
from .contracts import (PPOExample, PreferenceExample, ReasoningExample, SFTExample,
                        StageRunReport, StageTrainEvent, StageTrainSpec)
from .data import (DataPreparationReport, DataPreparationSpec, PreparedCorpus,
                   PreparedDocument, RawDocument, prepare_corpus, write_prepared_corpus)
from .providers import (ArtifactFile, OfflineComputeAdapter, RecordedTeacherProvider,
                        RemoteComputeProvider, RemoteTrainingBundle, RemoteTrainingResult,
                        TeacherProvider, TeacherRequest, TeacherResponse,
                        relational_embedding_targets)
from .pipeline import (PipelineStageRecord, TrainingPipelineLedger, read_pipeline_ledger,
                       write_pipeline_ledger)
from .reasoning import (ReasoningCandidate, SearchBudget, SearchResult, Verification,
                        bounded_reasoning_search)
from .hardware_policy_spec import (HardwarePolicyArtifact, HardwarePolicyEvent,
                                   HardwarePolicyRunReport, HardwarePolicySpec,
                                   HardwarePolicyTrainSpec)

_LAZY = {
    "HostedRewardModel": (".stages", "HostedRewardModel"),
    "HostedValueModel": (".stages", "HostedValueModel"),
    "HostedEmbeddingStudent": (".stages", "HostedEmbeddingStudent"),
    "HostedSmallModel": (".architectures", "HostedSmallModel"),
    "train_dpo": (".stages", "train_dpo"),
    "train_embedding_distillation": (".stages", "train_embedding_distillation"),
    "train_ppo": (".stages", "train_ppo"),
    "train_reasoning_sft": (".stages", "train_reasoning_sft"),
    "train_reward_model": (".stages", "train_reward_model"),
    "train_sft": (".stages", "train_sft"),
    "train_small_supervised": (".architectures", "train_small_supervised"),
    "HardwarePolicyNetwork": (".hardware_policy", "HardwarePolicyNetwork"),
    "load_hardware_policy": (".hardware_policy", "load_hardware_policy"),
    "hardware_policy_priors_q20": (".hardware_policy", "hardware_policy_priors_q20"),
    "train_hardware_policy": (".hardware_policy", "train_hardware_policy"),
    "HostedAdaptiveLanguageModel": (".adaptive_transformer", "HostedAdaptiveLanguageModel"),
    "HostedMultiPatchTransformer": (".adaptive_transformer", "HostedMultiPatchTransformer"),
    "train_adaptive_pretrain": (".adaptive_transformer", "train_adaptive_pretrain"),
    "train_multipatch_reconstruction": (
        ".adaptive_transformer", "train_multipatch_reconstruction"),
    "HostedByteLatentModel": (".byte_latent", "HostedByteLatentModel"),
    "HostedMambaByteModel": (".byte_latent", "HostedMambaByteModel"),
    "apply_byteified_transplant": (".byte_latent", "apply_byteified_transplant"),
    "byte_latent_tensor_shapes": (".byte_latent", "byte_latent_tensor_shapes"),
    "byteified_parameter_groups": (".byte_latent", "byteified_parameter_groups"),
    "diffusion_draft": (".byte_latent", "diffusion_draft"),
    "diffusion_generate": (".byte_latent", "diffusion_generate"),
    "diffusion_verify_generate": (".byte_latent", "diffusion_verify_generate"),
    "greedy_generate_byte_latent": (".byte_latent", "greedy_generate_byte_latent"),
    "greedy_generate_mambabyte": (".byte_latent", "greedy_generate_mambabyte"),
    "self_speculative_generate": (".byte_latent", "self_speculative_generate"),
    "train_byte_latent_pretrain": (".byte_latent", "train_byte_latent_pretrain"),
    "train_mambabyte_pretrain": (".byte_latent", "train_mambabyte_pretrain"),
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module
    module, attribute = _LAZY[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "ArtifactFile", "BytePairTokenizer", "DataPreparationReport", "DataPreparationSpec",
    "HardwarePolicyArtifact", "HardwarePolicyEvent", "HardwarePolicyNetwork",
    "HardwarePolicyRunReport", "HardwarePolicySpec", "HardwarePolicyTrainSpec",
    "HostedAdaptiveLanguageModel", "HostedByteLatentModel", "HostedEmbeddingStudent",
    "HostedMambaByteModel", "HostedMultiPatchTransformer", "HostedRewardModel",
    "HostedSmallModel", "HostedValueModel",
    "OfflineComputeAdapter", "PipelineStageRecord", "PPOExample",
    "PreferenceExample", "PreparedCorpus", "PreparedDocument", "RawDocument",
    "ReasoningCandidate", "ReasoningExample", "RecordedTeacherProvider",
    "RemoteComputeProvider", "RemoteTrainingBundle", "RemoteTrainingResult", "SFTExample",
    "SearchBudget", "SearchResult", "SmallModelSpec", "StageRunReport", "StageTrainEvent",
    "StageTrainSpec", "TeacherProvider", "TrainingPipelineLedger",
    "TeacherRequest", "TeacherResponse", "Verification", "bounded_reasoning_search",
    "apply_byteified_transplant", "byte_latent_tensor_shapes",
    "byteified_parameter_groups", "diffusion_draft", "diffusion_generate",
    "diffusion_verify_generate", "greedy_generate_byte_latent",
    "greedy_generate_mambabyte", "prepare_corpus", "read_pipeline_ledger",
    "relational_embedding_targets", "self_speculative_generate",
    "token_source_from_corpus", "train_adaptive_pretrain", "train_dpo",
    "train_embedding_distillation", "train_ppo", "train_reasoning_sft", "train_reward_model",
    "train_sft", "train_small_supervised", "train_multipatch_reconstruction",
    "train_byte_latent_pretrain", "train_mambabyte_pretrain",
    "train_hardware_policy", "load_hardware_policy",
    "hardware_policy_priors_q20",
    "write_pipeline_ledger", "write_prepared_corpus",
]
