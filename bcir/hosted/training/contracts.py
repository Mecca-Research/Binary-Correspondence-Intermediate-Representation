"""Dependency-free contracts for BCIR's staged hosted-training pipeline.

The contracts in this module deliberately contain no tensor-framework imports.  They are
the replay and admission boundary shared by local micro-model tests and future remote
compute adapters.  A stage name describes an optimization objective; it never weakens
BCIR legality or lets telemetry steer an in-flight update.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ..._artifact_json import strict_json_loads


_STAGES = frozenset({"pretrain", "sft", "reward", "dpo", "ppo", "reasoning",
                     "embedding", "supervised"})


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def require_sha256(value, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower() \
            or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def require_int(value, field: str, *, minimum: int = 0,
                maximum: int = 0x7FFFFFFF) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def require_finite(value, field: str, *, minimum: float | None = None,
                   maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or minimum is not None and result < minimum \
            or maximum is not None and result > maximum:
        raise ValueError(f"{field} is outside its valid range")
    return result


def token_tuple(value, field: str, *, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not allow_empty and not value:
        suffix = "a sequence" if allow_empty else "a nonempty sequence"
        raise ValueError(f"{field} must be {suffix} of token IDs")
    result = tuple(value)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{field} must contain non-negative integer token IDs")
    return result


@dataclass(frozen=True)
class StageTrainSpec:
    """Bounded optimizer settings for one pipeline stage.

    AdamW is the only admitted optimizer in the first hosted rail.  PPO-specific fields
    are still encoded here so an exact run specification carries all policy/value update
    constants rather than relying on framework defaults.
    """

    stage: str
    steps: int
    learning_rate: float
    batch_size: int = 1
    gradient_accumulation: int = 1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 1729
    preference_beta: float = 0.1
    ppo_clip: float = 0.2
    value_clip: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    kl_coefficient: float = 0.02

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError(f"stage must be one of {sorted(_STAGES)}")
        for field in ("steps", "batch_size", "gradient_accumulation"):
            require_int(getattr(self, field), field, minimum=1)
        require_int(self.seed, "seed")
        require_finite(self.learning_rate, "learning_rate", minimum=0.0)
        if self.learning_rate == 0.0:
            raise ValueError("learning_rate must be positive")
        for field in ("beta1", "beta2"):
            require_finite(getattr(self, field), field, minimum=0.0, maximum=1.0)
            if getattr(self, field) == 1.0:
                raise ValueError(f"{field} must be less than 1")
        require_finite(self.epsilon, "epsilon", minimum=0.0)
        if self.epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        for field in ("weight_decay", "grad_clip", "preference_beta", "ppo_clip",
                      "value_clip", "gamma", "gae_lambda", "kl_coefficient"):
            require_finite(getattr(self, field), field, minimum=0.0)
        if self.grad_clip == 0.0 or self.preference_beta == 0.0:
            raise ValueError("grad_clip and preference_beta must be positive")
        if self.gamma > 1.0 or self.gae_lambda > 1.0:
            raise ValueError("gamma and gae_lambda must not exceed 1")

    def to_dict(self) -> dict:
        return {"schema": "bcir.hosted_stage.v1", **asdict(self)}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "StageTrainSpec":
        value = strict_json_loads(
            text, "hosted stage specification", max_bytes=1024 * 1024)
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, dict) or set(value) != fields | {"schema"} \
                or value["schema"] != "bcir.hosted_stage.v1":
            raise ValueError("hosted stage specification has missing or unknown fields")
        arguments = {name: value[name] for name in fields}
        try:
            return cls(**arguments)
        except TypeError as exc:
            raise ValueError(f"malformed hosted stage specification: {exc}") from exc

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())


@dataclass(frozen=True)
class SFTExample:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    provenance_sha256: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", token_tuple(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "response_ids", token_tuple(self.response_ids, "response_ids"))
        require_sha256(self.provenance_sha256, "provenance_sha256")
        require_finite(self.weight, "weight", minimum=0.0)
        if self.weight == 0.0:
            raise ValueError("example weight must be positive")


@dataclass(frozen=True)
class PreferenceExample:
    prompt_ids: tuple[int, ...]
    chosen_ids: tuple[int, ...]
    rejected_ids: tuple[int, ...]
    provenance_sha256: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", token_tuple(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "chosen_ids", token_tuple(self.chosen_ids, "chosen_ids"))
        object.__setattr__(self, "rejected_ids", token_tuple(self.rejected_ids, "rejected_ids"))
        if self.chosen_ids == self.rejected_ids:
            raise ValueError("chosen and rejected responses must differ")
        require_sha256(self.provenance_sha256, "provenance_sha256")
        require_finite(self.weight, "weight", minimum=0.0)
        if self.weight == 0.0:
            raise ValueError("preference weight must be positive")


@dataclass(frozen=True)
class ReasoningExample:
    prompt_ids: tuple[int, ...]
    rationale_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]
    provenance_sha256: str
    verifier: str
    verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", token_tuple(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "rationale_ids", token_tuple(
            self.rationale_ids, "rationale_ids", allow_empty=True))
        object.__setattr__(self, "answer_ids", token_tuple(self.answer_ids, "answer_ids"))
        require_sha256(self.provenance_sha256, "provenance_sha256")
        if not isinstance(self.verifier, str) or not self.verifier:
            raise ValueError("reasoning verifier must be a nonempty string")
        if type(self.verified) is not bool:
            raise ValueError("verified must be bool")

    def as_sft(self) -> SFTExample:
        if not self.verified:
            raise ValueError("unverified reasoning traces cannot enter supervised training")
        return SFTExample(self.prompt_ids, self.rationale_ids + self.answer_ids,
                          self.provenance_sha256)


@dataclass(frozen=True)
class PPOExample:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    reward: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_ids", token_tuple(self.prompt_ids, "prompt_ids"))
        object.__setattr__(self, "response_ids", token_tuple(self.response_ids, "response_ids"))
        require_finite(self.reward, "reward")
        require_sha256(self.provenance_sha256, "provenance_sha256")


@dataclass(frozen=True)
class StageRunReport:
    stage: str
    spec_sha256: str
    input_sha256: str
    initial_loss: float
    final_loss: float
    steps: int
    examples: int
    model_sha256: str
    auxiliary_sha256: str

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("unknown stage in run report")
        for field in ("spec_sha256", "input_sha256", "model_sha256", "auxiliary_sha256"):
            require_sha256(getattr(self, field), field)
        require_finite(self.initial_loss, "initial_loss")
        require_finite(self.final_loss, "final_loss")
        require_int(self.steps, "steps", minimum=0)
        require_int(self.examples, "examples", minimum=1)

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.hosted_stage_report.v1", **asdict(self)})


@dataclass(frozen=True)
class StageTrainEvent:
    stage: str
    step: int
    loss: float
    grad_norm: float
    examples_processed: int

    def __post_init__(self) -> None:
        if self.stage not in _STAGES:
            raise ValueError("unknown stage telemetry event")
        require_int(self.step, "event step", minimum=1)
        require_int(self.examples_processed, "examples_processed", minimum=1)
        require_finite(self.loss, "event loss")
        require_finite(self.grad_norm, "event grad_norm", minimum=0.0)

    def to_dict(self) -> dict:
        return asdict(self)
