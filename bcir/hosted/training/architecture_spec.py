"""Dependency-free specifications for bounded non-LLM hosted model tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import canonical_json, require_int, sha256_text


@dataclass(frozen=True)
class SmallModelSpec:
    family: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    layers: int = 1
    heads: int = 1
    sequence_length: int = 8

    def __post_init__(self) -> None:
        if self.family not in ("mlp", "gru", "transformer_encoder"):
            raise ValueError("small model family must be mlp, gru, or transformer_encoder")
        for field in (
            "input_dim",
            "hidden_dim",
            "output_dim",
            "layers",
            "heads",
            "sequence_length",
        ):
            require_int(getattr(self, field), field, minimum=1, maximum=1 << 20)
        if self.family == "transformer_encoder" and self.hidden_dim % self.heads:
            raise ValueError("transformer hidden_dim must be divisible by heads")

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.small_model.v1", **asdict(self)})

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())
