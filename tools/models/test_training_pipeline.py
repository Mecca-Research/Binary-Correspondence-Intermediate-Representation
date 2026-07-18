#!/usr/bin/env python3
"""Bounded offline gate for BCIR's hosted training and alignment stages.

The gate uses one CPU thread and tiny synthetic, provenance-tagged examples.  It proves
the machinery and objective boundaries; it does not train or download a language model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.checkpoint import load_safe_checkpoint
from bcir.hosted.models.model import HostedLlama
from bcir.hosted.models.spec import HostedTrainSpec
from bcir.hosted.models.train import train_hosted
from bcir.hosted.training import (
    BytePairTokenizer,
    DataPreparationSpec,
    HostedEmbeddingStudent,
    HostedRewardModel,
    HostedSmallModel,
    HostedValueModel,
    PipelineStageRecord,
    PPOExample,
    PreferenceExample,
    RawDocument,
    ReasoningExample,
    SFTExample,
    SmallModelSpec,
    StageTrainSpec,
    TrainingPipelineLedger,
    prepare_corpus,
    relational_embedding_targets,
    token_source_from_corpus,
    train_dpo,
    train_embedding_distillation,
    train_ppo,
    train_reasoning_sft,
    train_reward_model,
    train_sft,
    train_small_supervised,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _report_dict(report) -> dict:
    return json.loads(report.to_json())


def _stage_report_digest(report) -> str:
    return _sha(report.to_json())


def _run_stage(call):
    """Require stage-local determinism controls to leave process-global state untouched."""
    rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    report = call()
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    assert math.isfinite(report.initial_loss) and math.isfinite(report.final_loss)
    return report


def _pretrain(seed: int, directory: Path):
    documents = (
        RawDocument("micro-a", "abcde abcde abcde", "generated-fixture", "CC0-1.0",
                    _sha("micro-a-source")),
        RawDocument("micro-b", "reason verify answer", "generated-fixture", "CC0-1.0",
                    _sha("micro-b-source")),
    )
    corpus = prepare_corpus(
        documents, DataPreparationSpec(("CC0-1.0",), validation_permyriad=0))
    tokenizer = BytePairTokenizer.train(
        [document.text for document in corpus.split("train")], vocab_size=272,
        min_frequency=2)
    source = token_source_from_corpus(corpus, tokenizer)
    decoder = DecoderSpec(
        vocab_size=tokenizer.vocab_size, d_model=8, n_heads=2, n_layers=1, d_ff=16,
        activation="silu_gate", n_kv_heads=1, tied_embeddings=True, rms_norm_eps=1e-6)
    spec = HostedTrainSpec(
        decoder=decoder, context_length=8, batch_size=1, gradient_accumulation=1,
        steps=2, checkpoint_every=2, learning_rate=1e-2, min_learning_rate=0.0,
        warmup_steps=0, weight_decay=0.0, seed=seed)
    report = train_hosted(spec, source, directory, device="cpu")

    torch.manual_seed(seed + 1)
    model = HostedLlama(decoder, context_length=8)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.learning_rate, betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon, weight_decay=spec.weight_decay, amsgrad=False)
    loaded = load_safe_checkpoint(
        directory, model, optimizer, spec=spec, corpus=source.manifest,
        device=torch.device("cpu"))
    assert loaded.global_step == spec.steps
    assert report.model_sha256 == loaded.model_sha256
    return corpus, tokenizer, model, report


def _alignment_run(seed: int, scratch: Path) -> dict:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    corpus, tokenizer, policy, pretrain = _pretrain(seed, scratch / "pretrain")
    assert not torch.are_deterministic_algorithms_enabled()

    vocab = policy.spec.vocab_size
    token = lambda character: 4 + ord(character)  # byte-fallback base vocabulary
    assert all(token(character) < vocab for character in "abcdefghijklmn")
    sft_examples = (
        SFTExample((1, token("a")), (token("b"), token("c")), _sha("sft-a")),
        SFTExample((1, token("d")), (token("e"), token("f")), _sha("sft-b")),
    )
    events = []
    sft = _run_stage(lambda: train_sft(
        policy, sft_examples,
        StageTrainSpec("sft", 2, 1e-2, batch_size=2, seed=seed + 1), events.append))
    assert sft.final_loss < sft.initial_loss
    sft_policy = copy.deepcopy(policy)

    preferences = (
        PreferenceExample((1, token("a")), (token("b"), token("c")),
                          (token("g"), token("h")), _sha("preference-a")),
        PreferenceExample((1, token("d")), (token("e"), token("f")),
                          (token("i"), token("j")), _sha("preference-b")),
    )
    reward_model = HostedRewardModel(sft_policy)
    reward = _run_stage(lambda: train_reward_model(
        reward_model, preferences,
        StageTrainSpec("reward", 2, 1e-2, batch_size=2, seed=seed + 2), events.append))
    assert reward.final_loss < reward.initial_loss

    reference = copy.deepcopy(sft_policy)
    reference.requires_grad_(False)
    dpo = _run_stage(lambda: train_dpo(
        policy, reference, preferences,
        StageTrainSpec("dpo", 2, 1e-2, batch_size=2, seed=seed + 3), events.append))
    assert dpo.final_loss < dpo.initial_loss

    ppo_reference = copy.deepcopy(policy)
    ppo_reference.requires_grad_(False)
    value_model = HostedValueModel(policy)
    ppo_examples = (
        PPOExample((1, token("a")), (token("b"), token("c")), 1.0, _sha("ppo-a")),
        PPOExample((1, token("d")), (token("e"), token("f")), 0.5, _sha("ppo-b")),
    )
    ppo = _run_stage(lambda: train_ppo(
        policy, ppo_reference, value_model, ppo_examples,
        StageTrainSpec("ppo", 2, 1e-3, batch_size=2, seed=seed + 4), events.append))

    reasoning_examples = (
        ReasoningExample((1, token("a")), (token("k"), token("l")), (token("m"),),
                         _sha("reasoning-a"), "exact-fixture"),
    )
    reasoning = _run_stage(lambda: train_reasoning_sft(
        policy, reasoning_examples,
        StageTrainSpec("reasoning", 2, 1e-2, seed=seed + 5), events.append))
    assert reasoning.final_loss < reasoning.initial_loss

    embedding_model = HostedEmbeddingStudent(sft_policy, 4)
    sequences = ((1, token("a"), token("b")), (1, token("d"), token("e")),
                 (1, token("g"), token("h")))
    targets = relational_embedding_targets(((1.0, 0.0), (0.0, 1.0), (1.0, 1.0)))
    embedding = _run_stage(lambda: train_embedding_distillation(
        embedding_model, sequences, targets,
        StageTrainSpec("embedding", 2, 1e-2, seed=seed + 6), events.append))
    assert embedding.final_loss < embedding.initial_loss

    features = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]] * 3, [[0.0, 1.0, 0.0, 0.0]] * 3,
         [[1.0, 0.0, 0.0, 0.0]] * 3, [[0.0, 1.0, 0.0, 0.0]] * 3])
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    small_reports = {}
    for index, family in enumerate(("mlp", "gru", "transformer_encoder")):
        torch.manual_seed(seed + 20 + index)
        model = HostedSmallModel(SmallModelSpec(
            family, 4, 8, 2, heads=2, sequence_length=3))
        report = _run_stage(lambda model=model, index=index: train_small_supervised(
            model, features, labels,
            StageTrainSpec("supervised", 2, 1e-2, seed=seed + 30 + index),
            events.append))
        assert report.final_loss < report.initial_loss
        small_reports[family] = _report_dict(report)

    data_report_sha = _sha(_canonical(asdict(corpus.report)))
    ledger = TrainingPipelineLedger((
        PipelineStageRecord("data", (), corpus.digest, data_report_sha, "prepared-corpus"),
        PipelineStageRecord("tokenizer", (corpus.digest,), tokenizer.digest,
                            _sha(tokenizer.to_json()), "byte-bpe"),
        PipelineStageRecord("pretrain", (tokenizer.digest,), pretrain.model_sha256,
                            _sha(pretrain.to_json()), "safe-checkpoint"),
        PipelineStageRecord("sft", (pretrain.model_sha256,), sft.model_sha256,
                            _stage_report_digest(sft), "hosted-state"),
        PipelineStageRecord("reward", (sft.model_sha256,), reward.model_sha256,
                            _stage_report_digest(reward), "reward-state"),
        PipelineStageRecord("dpo", (sft.model_sha256,), dpo.model_sha256,
                            _stage_report_digest(dpo), "hosted-state"),
        PipelineStageRecord("ppo", (dpo.model_sha256, reward.model_sha256), ppo.model_sha256,
                            _stage_report_digest(ppo), "policy-value-state"),
        PipelineStageRecord("reasoning", (ppo.model_sha256,), reasoning.model_sha256,
                            _stage_report_digest(reasoning), "hosted-state"),
        PipelineStageRecord("embedding", (sft.model_sha256,), embedding.model_sha256,
                            _stage_report_digest(embedding), "embedding-state"),
    ))
    reports = {name: _report_dict(report) for name, report in (
        ("pretrain", pretrain), ("sft", sft), ("reward", reward), ("dpo", dpo),
        ("ppo", ppo), ("reasoning", reasoning), ("embedding", embedding))}
    return {
        "schema": "bcir.training_pipeline_gate.v1",
        "mode": "offline-provider-neutral",
        "seed": seed,
        "dependencies": {
            "numpy": importlib.metadata.version("numpy"),
            "safetensors": importlib.metadata.version("safetensors"),
            "torch": torch.__version__.split("+", 1)[0],
        },
        "corpus_sha256": corpus.digest,
        "tokenizer_sha256": tokenizer.digest,
        "pipeline_sha256": ledger.digest,
        "pipeline_records": [asdict(record) for record in ledger.records],
        "stages": reports,
        "small_architectures": small_reports,
        "telemetry": [event.to_dict() for event in events],
    }


def _write_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical(value).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    with tempfile.TemporaryDirectory(prefix="bcir-training-pipeline-") as temporary:
        root = Path(temporary)
        first = _alignment_run(1729, root / "first")
        second = _alignment_run(1729, root / "second")
    assert _canonical(first) == _canonical(second), "training pipeline replay diverged"
    _write_atomic(args.output_dir / "report.json", first)
    print(_canonical({
        "pipeline_sha256": first["pipeline_sha256"],
        "stages": sorted(first["stages"]),
        "small_architectures": sorted(first["small_architectures"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
