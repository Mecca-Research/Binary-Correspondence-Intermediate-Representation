"""Dependency-free contracts for the opt-in hosted model laboratory."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec, decoder_param_count
from bcir.hosted.models import (
    CorpusManifest,
    HostedRunReport,
    HostedTrainEvent,
    HostedTrainSpec,
    SequenceTokenSource,
)

_ROOT = Path(__file__).resolve().parents[2]
_HASH_A = hashlib.sha256(b"hosted-corpus").hexdigest()
_HASH_B = hashlib.sha256(b"hosted-tokenizer").hexdigest()


def _spec() -> HostedTrainSpec:
    return HostedTrainSpec(
        decoder=DecoderSpec(
            vocab_size=260,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=128,
            activation="silu_gate",
            n_kv_heads=2,
            tied_embeddings=True,
            rms_norm_eps=1e-6,
        ),
        context_length=16,
        batch_size=4,
        gradient_accumulation=1,
        steps=64,
        checkpoint_every=32,
        learning_rate=3e-3,
        min_learning_rate=0.0,
        warmup_steps=4,
        weight_decay=0.0,
    )


def _manifest(token_count: int = 8) -> CorpusManifest:
    return CorpusManifest(
        name="generated-byte-sequence",
        revision="v1",
        split="train",
        license="LicenseRef-BCIR-NC-1.0",
        source_sha256=_HASH_A,
        tokenizer_sha256=_HASH_B,
        token_count=token_count,
        vocab_size=260,
        bos_token_id=257,
        eos_token_id=258,
        pad_token_id=259,
    )


def _must_refuse(call, fragment: str = "") -> None:
    try:
        call()
        raise AssertionError("invalid hosted-model input was accepted")
    except ValueError as exc:
        assert fragment in str(exc), (fragment, str(exc))


def test_hosted_training_spec_roundtrip_is_strict_and_canonical():
    spec = _spec()
    assert HostedTrainSpec.from_json(spec.to_json()) == spec
    assert HostedTrainSpec.from_json(spec.to_json()).digest == spec.digest
    assert spec.to_json() == HostedTrainSpec.from_json(spec.to_json()).to_json()

    unknown = json.loads(spec.to_json())
    unknown["fallback_backend"] = "cuda"
    _must_refuse(lambda: HostedTrainSpec.from_json(json.dumps(unknown)), "unknown")
    nested = json.loads(spec.to_json())
    nested["decoder"]["dropout"] = 0.0
    _must_refuse(lambda: HostedTrainSpec.from_json(json.dumps(nested)), "decoder")
    duplicate = spec.to_json().replace("{", '{"schema":"bcir.hosted_train.v1",', 1)
    _must_refuse(lambda: HostedTrainSpec.from_json(duplicate), "duplicate")
    _must_refuse(lambda: HostedTrainSpec.from_json("[]"), "unknown")


def test_hosted_training_spec_refuses_fallbacks_and_invalid_modes():
    spec = _spec()
    cases = (
        (lambda: replace(spec, optimizer="sgd"), "optimizer"),
        (lambda: replace(spec, dtype="float64"), "dtype"),
        (lambda: replace(spec, context_length=True), "context_length"),
        (lambda: replace(spec, warmup_steps=65), "warmup_steps"),
        (lambda: replace(spec, min_learning_rate=4e-3), "min_learning_rate"),
        (lambda: replace(spec, activation_checkpointing=1), "activation_checkpointing"),
    )
    for call, fragment in cases:
        _must_refuse(call, fragment)
    _must_refuse(
        lambda: HostedTrainSpec(
            **{**spec.__dict__, "decoder": replace(spec.decoder, activation="gelu")}
        ),
        "silu_gate",
    )


def test_hosted_token_source_is_index_addressed_and_bounded():
    source = SequenceTokenSource(list(range(8)), _manifest())
    assert source.batch(0, 2, 3) == [[0, 1, 2, 3], [3, 4, 5, 6]]
    assert source.batch(1, 2, 3) == [[6, 7, 0, 1], [1, 2, 3, 4]]
    assert source.batch(1, 2, 3) == source.batch(1, 2, 3)
    _must_refuse(lambda: source.batch(-1, 2, 3), "batch_index")
    _must_refuse(lambda: source.batch(0, 0, 3), "batch_size")
    _must_refuse(lambda: SequenceTokenSource([0, 260], _manifest(2)), "token 1")


def test_hosted_manifest_and_reports_have_content_identity():
    manifest = _manifest()
    assert len(manifest.digest) == 64
    assert manifest.digest == _manifest().digest
    event = HostedTrainEvent(1, 5.5, 3e-3, 0.75, 64, 0)
    assert event.to_dict()["tokens_processed"] == 64
    _must_refuse(lambda: replace(event, loss=float("nan")), "telemetry loss")
    _must_refuse(lambda: replace(event, tokens_processed=-1), "tokens_processed")
    report = HostedRunReport(
        schema="bcir.hosted_run.v1",
        spec_sha256=_HASH_A,
        corpus_sha256=_HASH_A,
        tokenizer_sha256=_HASH_B,
        steps=1,
        batch_index=1,
        tokens_processed=64,
        initial_loss=5.5,
        final_loss=5.0,
        model_sha256=_HASH_A,
        optimizer_sha256=_HASH_B,
        checkpoint_sha256=_HASH_A,
    )
    assert json.loads(report.to_json())["schema"] == "bcir.hosted_run.v1"
    _must_refuse(lambda: replace(manifest, source_sha256="mutable"), "SHA-256")
    _must_refuse(lambda: replace(report, final_loss=float("nan")), "final_loss")


def test_hosted_namespace_does_not_import_tensor_framework():
    code = (
        "import sys; import bcir; import bcir.hosted.models as m; "
        "import bcir.hosted.training as training; "
        "assert 'torch' not in sys.modules; "
        "assert m.HostedTrainSpec.__module__ == 'bcir.hosted.models.spec'; "
        "assert training.HardwarePolicySpec.__module__.endswith('hardware_policy_spec')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT)
    result = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(_ROOT)!r}); {code}"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_tinystories_32m_profile_and_source_inventory_are_pinned():
    config_path = _ROOT / "tools" / "models" / "configs" / "bcir-tinystories-32m.json"
    pins_path = _ROOT / "tools" / "models" / "tinystories_pins.json"
    spec = HostedTrainSpec.from_json(config_path.read_text("utf-8"))
    assert decoder_param_count(spec.decoder) == 31_203_840
    effective_tokens = spec.context_length * spec.batch_size * spec.gradient_accumulation
    assert effective_tokens == 32_768
    assert effective_tokens * spec.steps == 640_024_576
    pins = json.loads(pins_path.read_text("utf-8"))
    assert set(pins) == {"dataset", "files", "license", "revision", "schema", "tokenizer_plan"}
    assert pins["schema"] == "bcir.tinystories_source.v1"
    assert pins["revision"] == "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    assert [row["split"] for row in pins["files"]] == [
        "train",
        "train",
        "train",
        "train",
        "validation",
    ]
    assert sum(row["bytes"] for row in pins["files"]) == 1_000_775_442
    assert all(
        len(row["sha256"]) == 64 and row["sha256"] == row["sha256"].lower() for row in pins["files"]
    )
    assert pins["tokenizer_plan"] == {
        "add_dummy_prefix": True,
        "bos_id": 1,
        "byte_fallback": True,
        "eos_id": 2,
        "hard_vocab_limit": True,
        "implementation": "sentencepiece",
        "model_type": "bpe",
        "normalization_rule_name": "identity",
        "pad_id": 3,
        "remove_extra_whitespaces": False,
        "shuffle_input_sentence": False,
        "unk_id": 0,
        "version": "0.2.2",
        "vocab_size": 16_384,
    }
