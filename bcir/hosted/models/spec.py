"""Dependency-free contracts for the optional hosted transformer laboratory.

This module intentionally imports no tensor framework.  It is safe to validate model-lab
configuration, corpus identity, and reports on every BCIR host; actual PyTorch execution
lives in sibling modules and is reached only by an explicit import.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Protocol

from ...frontends.models.decode import DecoderSpec


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value, field: str, *, maximum: int = 0x7FFFFFFF) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _finite(value, field: str, *, minimum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    out = float(value)
    if (
        not math.isfinite(out)
        or (positive and out <= 0.0)
        or (minimum is not None and out < minimum)
    ):
        raise ValueError(f"{field} has an invalid value {value!r}")
    return out


@dataclass(frozen=True)
class CorpusManifest:
    """Content identity of the token stream consumed by one hosted run."""

    name: str
    revision: str
    split: str
    license: str
    source_sha256: str
    tokenizer_sha256: str
    token_count: int
    vocab_size: int
    bos_token_id: int = -1
    eos_token_id: int = -1
    pad_token_id: int = -1

    def __post_init__(self) -> None:
        for field in ("name", "revision", "split", "license"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"corpus {field} must be a nonempty string")
        _sha256(self.source_sha256, "source_sha256")
        _sha256(self.tokenizer_sha256, "tokenizer_sha256")
        _positive_int(self.token_count, "token_count")
        _positive_int(self.vocab_size, "vocab_size")
        for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(self, field)
            if type(value) is not int or value != -1 and not 0 <= value < self.vocab_size:
                raise ValueError(f"{field} must be -1 or in the corpus vocabulary")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


class HostedTokenSource(Protocol):
    """Deterministic, index-addressed token batches.

    Each row contains ``context_length + 1`` IDs; training uses the first ``N`` as
    inputs and the shifted final ``N`` as targets.  Addressing by batch index makes an
    exact resume independent of mutable iterator state.
    """

    manifest: CorpusManifest

    def batch(self, batch_index: int, batch_size: int, context_length: int) -> list[list[int]]: ...


class SequenceTokenSource:
    """A bounded deterministic token source used by tests and small hosted runs."""

    def __init__(self, token_ids, manifest: CorpusManifest):
        if not isinstance(manifest, CorpusManifest):
            raise ValueError("manifest must be a CorpusManifest")
        if not isinstance(token_ids, (list, tuple)) or len(token_ids) != manifest.token_count:
            raise ValueError("token_ids length must equal manifest.token_count")
        ids = []
        for index, token in enumerate(token_ids):
            if type(token) is not int or not 0 <= token < manifest.vocab_size:
                raise ValueError(f"token {index} must be in [0, {manifest.vocab_size})")
            ids.append(token)
        if len(ids) < 2:
            raise ValueError("token source needs at least two tokens")
        self._ids = tuple(ids)
        self.manifest = manifest

    def batch(self, batch_index: int, batch_size: int, context_length: int) -> list[list[int]]:
        if type(batch_index) is not int or batch_index < 0:
            raise ValueError("batch_index must be an integer >= 0")
        _positive_int(batch_size, "batch_size")
        _positive_int(context_length, "context_length")
        width = context_length + 1
        n = len(self._ids)
        rows = []
        base = batch_index * batch_size * context_length
        for row in range(batch_size):
            start = base + row * context_length
            rows.append([self._ids[(start + col) % n] for col in range(width)])
        return rows


_DECODER_FIELDS = {
    "vocab_size",
    "d_model",
    "n_heads",
    "n_layers",
    "d_ff",
    "rope_base",
    "activation",
    "n_kv_heads",
    "tied_embeddings",
    "rms_norm_eps",
}
_TRAIN_FIELDS = {
    "schema",
    "decoder",
    "context_length",
    "batch_size",
    "gradient_accumulation",
    "steps",
    "checkpoint_every",
    "optimizer",
    "learning_rate",
    "min_learning_rate",
    "warmup_steps",
    "beta1",
    "beta2",
    "epsilon",
    "weight_decay",
    "grad_clip",
    "seed",
    "dtype",
    "activation_checkpointing",
}


@dataclass(frozen=True)
class HostedTrainSpec:
    """Strict, replayable specification for one hosted Llama training run."""

    decoder: DecoderSpec
    context_length: int
    batch_size: int
    gradient_accumulation: int
    steps: int
    checkpoint_every: int
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seed: int = 1729
    dtype: str = "float32"
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.decoder, DecoderSpec):
            raise ValueError("decoder must be a DecoderSpec")
        for field in (
            "context_length",
            "batch_size",
            "gradient_accumulation",
            "steps",
            "checkpoint_every",
        ):
            _positive_int(getattr(self, field), field)
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps <= self.steps:
            raise ValueError("warmup_steps must be an integer in [0, steps]")
        if type(self.seed) is not int or not 0 <= self.seed <= 0x7FFFFFFF:
            raise ValueError("seed must be an integer in [0, 2^31-1]")
        if self.optimizer != "adamw":
            raise ValueError("the first hosted rail supports optimizer='adamw' only")
        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("dtype must be float32|float16|bfloat16")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be bool")
        if self.decoder.activation != "silu_gate":
            raise ValueError("hosted Llama training requires activation='silu_gate'")
        lr = _finite(self.learning_rate, "learning_rate", positive=True)
        min_lr = _finite(self.min_learning_rate, "min_learning_rate", minimum=0.0)
        if min_lr > lr:
            raise ValueError("min_learning_rate cannot exceed learning_rate")
        b1, b2 = _finite(self.beta1, "beta1"), _finite(self.beta2, "beta2")
        if not 0.0 <= b1 < 1.0 or not 0.0 <= b2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        _finite(self.epsilon, "epsilon", positive=True)
        _finite(self.weight_decay, "weight_decay", minimum=0.0)
        _finite(self.grad_clip, "grad_clip", positive=True)

    def to_dict(self) -> dict:
        decoder = asdict(self.decoder)
        out = asdict(self)
        out["schema"] = "bcir.hosted_train.v1"
        out["decoder"] = decoder
        return out

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "HostedTrainSpec":
        if not isinstance(text, str) or len(text) > 1024 * 1024:
            raise ValueError("hosted training JSON must be a string no larger than 1 MiB")
        try:
            raw = json.loads(text, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ValueError, RecursionError) as exc:
            raise ValueError(f"malformed hosted training JSON: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != _TRAIN_FIELDS:
            raise ValueError("hosted training JSON has missing or unknown fields")
        if raw.pop("schema") != "bcir.hosted_train.v1":
            raise ValueError("unsupported hosted training schema")
        decoder = raw.pop("decoder")
        if not isinstance(decoder, dict) or set(decoder) != _DECODER_FIELDS:
            raise ValueError("hosted decoder JSON has missing or unknown fields")
        try:
            spec = DecoderSpec(**decoder)
            return cls(decoder=spec, **raw)
        except TypeError as exc:
            raise ValueError(f"invalid hosted training fields: {exc}") from exc


@dataclass(frozen=True)
class HostedTrainEvent:
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    tokens_processed: int
    peak_memory_bytes: int

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step < 1:
            raise ValueError("telemetry step must be an integer >= 1")
        for field in ("tokens_processed", "peak_memory_bytes"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"telemetry {field} must be an integer >= 0")
        _finite(self.loss, "telemetry loss")
        _finite(self.learning_rate, "telemetry learning_rate", minimum=0.0)
        _finite(self.grad_norm, "telemetry grad_norm", minimum=0.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HostedRunReport:
    schema: str
    spec_sha256: str
    corpus_sha256: str
    tokenizer_sha256: str
    steps: int
    batch_index: int
    tokens_processed: int
    initial_loss: float
    final_loss: float
    model_sha256: str
    optimizer_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if self.schema != "bcir.hosted_run.v1":
            raise ValueError("unsupported hosted run report schema")
        for field in (
            "spec_sha256",
            "corpus_sha256",
            "tokenizer_sha256",
            "model_sha256",
            "optimizer_sha256",
            "checkpoint_sha256",
        ):
            _sha256(getattr(self, field), field)
        for field in ("steps", "batch_index", "tokens_processed"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"{field} must be an integer >= 0")
        _finite(self.initial_loss, "initial_loss")
        _finite(self.final_loss, "final_loss")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return _canonical(self.to_dict())
