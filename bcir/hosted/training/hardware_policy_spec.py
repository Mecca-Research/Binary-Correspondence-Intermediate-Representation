"""Dependency-free specifications and reports for the hosted hardware policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..._artifact_json import strict_json_loads
from .contracts import canonical_json, require_finite, require_int, require_sha256, sha256_text


@dataclass(frozen=True)
class HardwarePolicySpec:
    hidden_dim: int = 32
    heads: int = 4
    transformer_layers: int = 1
    graph_layers: int = 2
    max_telemetry_tokens: int = 32

    def __post_init__(self) -> None:
        for field in ("hidden_dim", "heads", "transformer_layers", "graph_layers",
                      "max_telemetry_tokens"):
            require_int(getattr(self, field), field, minimum=1, maximum=1024)
        if self.hidden_dim % self.heads:
            raise ValueError("hardware policy hidden_dim must be divisible by heads")
        if self.hidden_dim > 256 or self.heads > 16 or self.transformer_layers > 8 \
                or self.graph_layers > 8 or self.max_telemetry_tokens > 4096:
            raise ValueError("hardware policy exceeds the bounded reference-model envelope")

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.hardware_policy_spec.v1", **asdict(self)})

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())


@dataclass(frozen=True)
class HardwarePolicyTrainSpec:
    reward_steps: int = 16
    dpo_steps: int = 16
    ppo_steps: int = 16
    learning_rate: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    dpo_beta: float = 0.2
    ppo_clip: float = 0.2
    value_coefficient: float = 0.5
    seed: int = 1729

    def __post_init__(self) -> None:
        for field in ("reward_steps", "dpo_steps", "ppo_steps"):
            require_int(getattr(self, field), field, minimum=1, maximum=4096)
        require_int(self.seed, "seed")
        for field in ("learning_rate", "epsilon", "grad_clip", "dpo_beta",
                      "ppo_clip", "value_coefficient"):
            require_finite(getattr(self, field), field, minimum=0.0)
            if getattr(self, field) == 0.0:
                raise ValueError(f"{field} must be positive")
        require_finite(self.weight_decay, "weight_decay", minimum=0.0)
        for field in ("beta1", "beta2"):
            require_finite(getattr(self, field), field, minimum=0.0, maximum=1.0)
            if getattr(self, field) == 1.0:
                raise ValueError(f"{field} must be less than one")
        if self.ppo_clip > 1.0:
            raise ValueError("ppo_clip must not exceed one")

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.hardware_policy_train.v1", **asdict(self)})

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())


@dataclass(frozen=True)
class HardwarePolicyEvent:
    phase: str
    step: int
    loss: float
    grad_norm: float

    def __post_init__(self) -> None:
        if self.phase not in ("reward", "dpo", "ppo"):
            raise ValueError("unknown hardware-policy training phase")
        require_int(self.step, "hardware-policy event step", minimum=1)
        require_finite(self.loss, "hardware-policy event loss")
        require_finite(self.grad_norm, "hardware-policy event grad_norm", minimum=0.0)


@dataclass(frozen=True)
class HardwarePolicyRunReport:
    model_spec_sha256: str
    train_spec_sha256: str
    input_sha256: str
    initial_reward_loss: float
    final_reward_loss: float
    initial_top1_correct: int
    final_top1_correct: int
    episodes: int
    candidates: int
    total_steps: int
    model_sha256: str

    def __post_init__(self) -> None:
        for field in ("model_spec_sha256", "train_spec_sha256", "input_sha256",
                      "model_sha256"):
            require_sha256(getattr(self, field), field)
        require_finite(self.initial_reward_loss, "initial_reward_loss", minimum=0.0)
        require_finite(self.final_reward_loss, "final_reward_loss", minimum=0.0)
        for field in ("initial_top1_correct", "final_top1_correct", "episodes",
                      "candidates", "total_steps"):
            require_int(getattr(self, field), field, minimum=0)
        if self.episodes < 1 or self.candidates < 2 or self.total_steps < 1:
            raise ValueError("hardware policy report has an empty training inventory")
        if self.episodes > 1024 or self.candidates > 256 or self.total_steps > 12_288:
            raise ValueError("hardware policy report exceeds the bounded training envelope")
        if self.initial_top1_correct > self.episodes or self.final_top1_correct > self.episodes:
            raise ValueError("hardware policy top-1 count exceeds episode count")

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.hardware_policy_report.v1", **asdict(self)})

    @classmethod
    def from_json(cls, text: str) -> "HardwarePolicyRunReport":
        doc = strict_json_loads(text, "hardware policy report", max_bytes=1024 * 1024)
        fields = {"schema", "model_spec_sha256", "train_spec_sha256", "input_sha256",
                  "initial_reward_loss", "final_reward_loss", "initial_top1_correct",
                  "final_top1_correct", "episodes", "candidates", "total_steps",
                  "model_sha256"}
        if not isinstance(doc, dict) or set(doc) != fields \
                or doc.pop("schema") != "bcir.hardware_policy_report.v1":
            raise ValueError("hardware policy report has missing or unknown fields")
        try:
            return cls(**doc)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed hardware policy report: {exc}") from exc


@dataclass(frozen=True)
class HardwarePolicyArtifact:
    directory: str
    manifest_sha256: str
    model_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str) or not self.directory:
            raise ValueError("hardware policy artifact directory must be nonempty")
        require_sha256(self.manifest_sha256, "manifest_sha256")
        require_sha256(self.model_sha256, "model_sha256")


__all__ = ["HardwarePolicyArtifact", "HardwarePolicyEvent", "HardwarePolicyRunReport",
           "HardwarePolicySpec", "HardwarePolicyTrainSpec"]
