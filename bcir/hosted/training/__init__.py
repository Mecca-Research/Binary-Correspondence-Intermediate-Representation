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
    "HostedEmbeddingStudent", "HostedRewardModel", "HostedSmallModel", "HostedValueModel",
    "OfflineComputeAdapter", "PipelineStageRecord", "PPOExample",
    "PreferenceExample", "PreparedCorpus", "PreparedDocument", "RawDocument",
    "ReasoningCandidate", "ReasoningExample", "RecordedTeacherProvider",
    "RemoteComputeProvider", "RemoteTrainingBundle", "RemoteTrainingResult", "SFTExample",
    "SearchBudget", "SearchResult", "SmallModelSpec", "StageRunReport", "StageTrainEvent",
    "StageTrainSpec", "TeacherProvider", "TrainingPipelineLedger",
    "TeacherRequest", "TeacherResponse", "Verification", "bounded_reasoning_search",
    "prepare_corpus", "read_pipeline_ledger", "relational_embedding_targets",
    "token_source_from_corpus", "train_dpo",
    "train_embedding_distillation", "train_ppo", "train_reasoning_sft", "train_reward_model",
    "train_sft", "train_small_supervised",
    "write_pipeline_ledger", "write_prepared_corpus",
]
