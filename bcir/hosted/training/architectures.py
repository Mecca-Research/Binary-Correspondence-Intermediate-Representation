"""Bounded MLP, GRU, and transformer-encoder confirmation models."""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "hosted small architectures require PyTorch; install bcir[model-lab]") from exc

from .architecture_spec import SmallModelSpec
from .contracts import StageRunReport, StageTrainSpec, canonical_json, sha256_text
from .stages import _deterministic_stage, _emit, _module_digest, _optimizer


class HostedSmallModel(nn.Module):
    def __init__(self, spec: SmallModelSpec):
        super().__init__()
        if not isinstance(spec, SmallModelSpec):
            raise ValueError("spec must be a SmallModelSpec")
        self.spec = spec
        if spec.family == "mlp":
            layers = [nn.Linear(spec.input_dim, spec.hidden_dim), nn.GELU()]
            for _ in range(spec.layers - 1):
                layers.extend((nn.Linear(spec.hidden_dim, spec.hidden_dim), nn.GELU()))
            self.body = nn.Sequential(*layers)
        elif spec.family == "gru":
            self.body = nn.GRU(spec.input_dim, spec.hidden_dim, spec.layers, batch_first=True)
        else:
            self.input_projection = nn.Linear(spec.input_dim, spec.hidden_dim)
            self.position = nn.Parameter(torch.zeros(1, spec.sequence_length, spec.hidden_dim))
            layer = nn.TransformerEncoderLayer(
                spec.hidden_dim, spec.heads, dim_feedforward=spec.hidden_dim * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(
                layer, spec.layers, enable_nested_tensor=False)
        self.output = nn.Linear(spec.hidden_dim, spec.output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] < 1 or values.shape[2] != self.spec.input_dim \
                or values.shape[1] > self.spec.sequence_length or not values.is_floating_point() \
                or not torch.isfinite(values).all():
            raise ValueError("small-model inputs must be finite [batch, sequence, input_dim] floats")
        if self.spec.family == "mlp":
            hidden = self.body(values).mean(dim=1)
        elif self.spec.family == "gru":
            _states, final = self.body(values); hidden = final[-1]
        else:
            hidden = self.input_projection(values)
            hidden = self.body(hidden + self.position[:, :values.shape[1], :]).mean(dim=1)
        return self.output(hidden)


@torch.no_grad()
def _classification_loss(model, features, targets) -> float:
    was_training = model.training
    model.eval()
    try:
        value = float(F.cross_entropy(model(features).float(), targets).item())
    finally:
        model.train(was_training)
    if not torch.isfinite(torch.tensor(value)):
        raise RuntimeError("small supervised evaluation produced a non-finite loss")
    return value


def train_small_supervised(model: HostedSmallModel, features: torch.Tensor,
                           targets: torch.Tensor, spec: StageTrainSpec,
                           telemetry_sink=None) -> StageRunReport:
    if not isinstance(model, HostedSmallModel) or not isinstance(spec, StageTrainSpec) \
            or spec.stage != "supervised":
        raise ValueError("small supervised training requires its model and stage='supervised'")
    if not isinstance(features, torch.Tensor) or not isinstance(targets, torch.Tensor) \
            or targets.ndim != 1 or targets.dtype != torch.long \
            or features.shape[0] != targets.shape[0] or features.shape[0] < 1 \
            or torch.any(targets < 0) or torch.any(targets >= model.spec.output_dim):
        raise ValueError("small supervised features/targets are malformed")
    if features.device != next(model.parameters()).device or targets.device != features.device:
        raise ValueError("small supervised model and tensors must share a device")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    input_sha = sha256_text(canonical_json({"features": features.detach().cpu().tolist(),
                                            "targets": targets.detach().cpu().tolist()}))
    with _deterministic_stage(spec.seed):
        initial = _classification_loss(model, features, targets)
        optimizer = _optimizer(spec, model.parameters()); model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features).float(), targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("small supervised model produced a non-finite gradient norm")
            optimizer.step()
            _emit(telemetry_sink, "supervised", step + 1, float(loss.detach().item()),
                  float(norm.item()), (step + 1) * features.shape[0])
        final = _classification_loss(model, features, targets)
        return StageRunReport("supervised", spec.digest, input_sha, initial, final,
                              spec.steps, features.shape[0], _module_digest(model),
                              sha256_text(model.spec.to_json()))
