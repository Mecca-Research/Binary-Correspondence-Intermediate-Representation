"""Bounded PyTorch references for BCIR sequence-interface research contracts.

This module is opt-in and import-quarantined.  It does not implement a production
tokenizer, large progressive-training system, or accelerator kernel.  It provides enough
tensor-backed machinery to prove three semantics on tiny CPU fixtures:

* continued-BPE rows are copied or source-decomposition means, with copied rows frozen;
* cross-tokenizer projected advantages enter a clipped on-policy objective; and
* constructive growth trains only newly appended blocks plus the language-model head.
"""
from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "sequence-interface probes require PyTorch; install bcir[model-lab]") from exc

from ...kbcir.sequence_interfaces import (
    ContinuedBPEExpansionPlan,
    CrossTokenizerProjection,
    FrozenTokenCodeSpec,
    GrowthStage,
    ProgressiveGrowthSpec,
)
from .contracts import StageRunReport, StageTrainSpec, canonical_json, sha256_text
from .stages import _deterministic_stage, _emit, _module_digest, _optimizer

_MAX_ROW_ELEMENTS = 1 << 24
_MAX_HOSTED_PARAMETERS = 5_000_000
_MAX_HOSTED_CONTEXT = 4096
_MAX_HOSTED_TOKENS = 1 << 16
_MAX_HOSTED_STEPS = 10_000


def _validate_float_rows(rows: torch.Tensor, expected_rows: int, field: str) -> None:
    if not isinstance(rows, torch.Tensor) or rows.ndim != 2 \
            or rows.shape[0] != expected_rows or rows.shape[1] < 1 \
            or rows.numel() > _MAX_ROW_ELEMENTS or not rows.dtype.is_floating_point \
            or not torch.isfinite(rows).all():
        raise ValueError(f"{field} must be finite bounded floating rows with the source vocabulary")


@torch.no_grad()
def initialize_expanded_rows(source_rows: torch.Tensor,
                             plan: ContinuedBPEExpansionPlan) -> torch.Tensor:
    """Copy source rows and initialize each added row as its source-token mean."""
    if not isinstance(plan, ContinuedBPEExpansionPlan):
        raise ValueError("plan must be ContinuedBPEExpansionPlan")
    _validate_float_rows(source_rows, plan.source_vocab_size, "source_rows")
    if plan.target_vocab_size * source_rows.shape[1] > _MAX_ROW_ELEMENTS:
        raise ValueError("expanded row table exceeds the bounded hosted limit")
    source = source_rows.detach()
    added = []
    for row in plan.new_rows:
        indices = torch.tensor(row.source_token_ids, dtype=torch.long, device=source.device)
        added.append(source.index_select(0, indices).mean(dim=0))
    return torch.cat((source.clone(), torch.stack(added)), dim=0)


class ExpandedRowTable(nn.Module):
    """Source rows plus separately owned added rows for two-stage vocabulary adaptation."""

    def __init__(self, source_rows: torch.Tensor, plan: ContinuedBPEExpansionPlan):
        super().__init__()
        expanded = initialize_expanded_rows(source_rows, plan)
        self.plan = plan
        self.source_rows = nn.Parameter(expanded[:plan.source_vocab_size].clone(),
                                        requires_grad=False)
        self.added_rows = nn.Parameter(expanded[plan.source_vocab_size:].clone())
        self.stage = 1

    @property
    def weight(self) -> torch.Tensor:
        return torch.cat((self.source_rows, self.added_rows), dim=0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor) or token_ids.dtype != torch.long \
                or torch.any(token_ids < 0) \
                or torch.any(token_ids >= self.plan.target_vocab_size):
            raise ValueError("expanded-row token IDs must be in-vocabulary torch.long")
        return F.embedding(token_ids, self.weight)

    def begin_full_model_stage(self) -> None:
        """Enter stage two, where copied and added rows may both be optimized."""
        self.source_rows.requires_grad_(True)
        self.added_rows.requires_grad_(True)
        self.stage = 2


def clipped_cross_tokenizer_opd_loss(
        current_student_log_probabilities: torch.Tensor,
        old_student_log_probabilities: torch.Tensor,
        projection: CrossTokenizerProjection, *, clip_epsilon: float = 0.2) -> torch.Tensor:
    """Compute the clipped on-policy objective from exact projected token advantages."""
    if not isinstance(projection, CrossTokenizerProjection):
        raise ValueError("projection must be CrossTokenizerProjection")
    if not isinstance(current_student_log_probabilities, torch.Tensor) \
            or not isinstance(old_student_log_probabilities, torch.Tensor) \
            or current_student_log_probabilities.ndim != 1 \
            or current_student_log_probabilities.shape != old_student_log_probabilities.shape \
            or current_student_log_probabilities.numel() != len(projection.advantages) \
            or not current_student_log_probabilities.dtype.is_floating_point \
            or current_student_log_probabilities.dtype != old_student_log_probabilities.dtype \
            or current_student_log_probabilities.device != old_student_log_probabilities.device \
            or not torch.isfinite(current_student_log_probabilities).all() \
            or not torch.isfinite(old_student_log_probabilities).all() \
            or torch.any(current_student_log_probabilities > 0) \
            or torch.any(old_student_log_probabilities > 0):
        raise ValueError("cross-tokenizer OPD logits must be matching finite rank-one tensors")
    if isinstance(clip_epsilon, bool) or not isinstance(clip_epsilon, (int, float)) \
            or not math.isfinite(float(clip_epsilon)) or not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be finite and in (0, 1)")
    advantages = torch.tensor(
        projection.advantages, dtype=current_student_log_probabilities.dtype,
        device=current_student_log_probabilities.device)
    ratio = torch.exp(current_student_log_probabilities -
                      old_student_log_probabilities.detach())
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    loss = -objective.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("cross-tokenizer OPD objective became non-finite")
    return loss


class _GrowthBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width, eps=1.0e-6)
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=0.0, bias=False, batch_first=True)
        self.mlp_norm = nn.LayerNorm(width, eps=1.0e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, ff_width, bias=False), nn.GELU(approximate="tanh"),
            nn.Linear(ff_width, width, bias=False))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(values)
        attention = self.attention(
            normalized, normalized, normalized, attn_mask=mask, need_weights=False)[0]
        values = values + attention
        return values + self.mlp(self.mlp_norm(values))


class HostedProgressiveLanguageModel(nn.Module):
    """Tiny fixed-interface decoder whose dense blocks are activated constructively."""

    def __init__(self, interface: FrozenTokenCodeSpec, *, heads: int, ff_width: int,
                 max_layers: int, context_length: int):
        super().__init__()
        if not isinstance(interface, FrozenTokenCodeSpec):
            raise ValueError("interface must be FrozenTokenCodeSpec")
        if type(heads) is not int or heads < 1 or interface.model_width % heads:
            raise ValueError("heads must be positive and divide the model width")
        if type(ff_width) is not int or ff_width < interface.model_width:
            raise ValueError("ff_width must be an integer at least as large as model width")
        if ff_width > 1 << 20:
            raise ValueError("ff_width exceeds the bounded hosted limit")
        if type(max_layers) is not int or not 1 <= max_layers <= 64:
            raise ValueError("max_layers must be an integer in [1, 64]")
        if type(context_length) is not int or not 1 <= context_length <= _MAX_HOSTED_CONTEXT:
            raise ValueError("context_length exceeds the bounded hosted limit")
        interface_elements = interface.vocab_size * interface.model_width
        position_elements = context_length * interface.model_width
        block_elements = (4 * interface.model_width * interface.model_width
                          + 2 * interface.model_width * ff_width
                          + 4 * interface.model_width)
        head_elements = 2 * interface.model_width \
            + interface.vocab_size * interface.model_width
        parameter_elements = max_layers * block_elements + head_elements
        if interface_elements > _MAX_ROW_ELEMENTS \
                or position_elements > _MAX_ROW_ELEMENTS:
            raise ValueError("fixed token interface exceeds the bounded hosted limit")
        if parameter_elements > _MAX_HOSTED_PARAMETERS:
            raise ValueError("progressive model exceeds the bounded hosted parameter limit")
        self.interface = interface
        self.context_length = context_length
        codebook = torch.tensor(
            [interface.code(token) for token in range(interface.vocab_size)],
            dtype=torch.float32)
        self.register_buffer("fixed_codebook", codebook, persistent=True)
        positions = torch.arange(context_length, dtype=torch.float32)[:, None]
        dimensions = torch.arange(interface.model_width, dtype=torch.float32)[None, :]
        frequencies = torch.exp(-math.log(10000.0) *
                                (2 * torch.floor(dimensions / 2)) /
                                interface.model_width)
        angles = positions * frequencies
        position_codes = torch.where(
            (dimensions.to(torch.long) % 2) == 0, torch.sin(angles), torch.cos(angles))
        self.register_buffer("position_codes", position_codes, persistent=True)
        self.blocks = nn.ModuleList(
            _GrowthBlock(interface.model_width, heads, ff_width)
            for _ in range(max_layers))
        self.final_norm = nn.LayerNorm(interface.model_width, eps=1.0e-6)
        self.lm_head = nn.Linear(interface.model_width, interface.vocab_size, bias=False)
        self.current_layers = 1
        self.apply(self._initialize)
        if sum(parameter.numel() for parameter in self.parameters()) != parameter_elements:
            raise RuntimeError("progressive model parameter accounting diverged")

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            for parameter in module.parameters(recurse=False):
                if parameter.ndim >= 2:
                    nn.init.normal_(parameter, mean=0.0, std=0.02)
                else:
                    nn.init.zeros_(parameter)

    @property
    def block_parameter_elements(self) -> int:
        counts = [sum(parameter.numel() for parameter in block.parameters())
                  for block in self.blocks]
        if len(set(counts)) != 1:
            raise RuntimeError("progressive hosted blocks do not share one parameter shape")
        return counts[0]

    @property
    def head_parameter_elements(self) -> int:
        return sum(parameter.numel() for parameter in self.final_norm.parameters()) \
            + sum(parameter.numel() for parameter in self.lm_head.parameters())

    @property
    def fixed_interface_elements(self) -> int:
        return self.fixed_codebook.numel()

    def _validate_ids(self, input_ids: torch.Tensor) -> None:
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 \
                or input_ids.dtype != torch.long or input_ids.shape[0] < 1 \
                or input_ids.shape[1] < 1 or input_ids.shape[1] > self.context_length \
                or input_ids.numel() > _MAX_HOSTED_TOKENS:
            raise ValueError("input_ids must be nonempty rank-2 torch.long within context")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.interface.vocab_size):
            raise ValueError("progressive-model token ID is outside the vocabulary")

    def activate_stage(self, growth: ProgressiveGrowthSpec,
                       stage_index: int) -> GrowthStage:
        if not isinstance(growth, ProgressiveGrowthSpec) or type(stage_index) is not int \
                or not 0 <= stage_index < len(growth.stages()):
            raise ValueError("growth specification or stage index is invalid")
        if growth.final_layers > len(self.blocks) \
                or growth.block_parameter_elements != self.block_parameter_elements \
                or growth.head_parameter_elements != self.head_parameter_elements \
                or growth.fixed_interface_elements != self.fixed_interface_elements \
                or growth.interface_width != self.interface.model_width:
            raise ValueError("growth specification does not match the hosted model")
        stage = growth.stages()[stage_index]
        if stage.kind != "dense_growth":
            raise ValueError("hosted reference does not execute LoRA readjustment stages")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for layer in stage.trainable_blocks:
            for parameter in self.blocks[layer].parameters():
                parameter.requires_grad_(True)
        for module in (self.final_norm, self.lm_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        self.current_layers = stage.total_layers
        active = sum(parameter.numel() for parameter in self.parameters()
                     if parameter.requires_grad)
        if active != stage.active_parameter_elements:
            raise ValueError("hosted trainable parameters diverge from the growth contract")
        return stage

    def forward(self, input_ids: torch.Tensor,
                targets: torch.Tensor | None = None):
        self._validate_ids(input_ids)
        if targets is not None and (not isinstance(targets, torch.Tensor)
                                    or targets.shape != input_ids.shape
                                    or targets.dtype != torch.long
                                    or targets.device != input_ids.device
                                    or torch.any(targets < 0)
                                    or torch.any(targets >= self.interface.vocab_size)):
            raise ValueError("targets must be in-vocabulary torch.long with the input shape")
        values = F.embedding(input_ids, self.fixed_codebook)
        values = values + self.position_codes[:input_ids.shape[1]].to(values.dtype)
        mask = torch.triu(torch.ones(
            input_ids.shape[1], input_ids.shape[1], dtype=torch.bool,
            device=input_ids.device), diagonal=1)
        for block in self.blocks[:self.current_layers]:
            values = block(values, mask)
        logits = self.lm_head(self.final_norm(values))
        loss = None if targets is None else F.cross_entropy(
            logits.float().reshape(-1, self.interface.vocab_size), targets.reshape(-1))
        return logits, loss


@torch.no_grad()
def _growth_loss(model: HostedProgressiveLanguageModel,
                 input_ids: torch.Tensor, targets: torch.Tensor) -> float:
    was_training = model.training
    model.eval()
    try:
        _logits, loss = model(input_ids, targets)
        result = float(loss.item())
    finally:
        model.train(was_training)
    if not math.isfinite(result):
        raise RuntimeError("progressive language evaluation produced non-finite loss")
    return result


def train_progressive_growth_stage(
        model: HostedProgressiveLanguageModel, growth: ProgressiveGrowthSpec,
        stage_index: int, input_ids: torch.Tensor, targets: torch.Tensor,
        spec: StageTrainSpec, telemetry_sink=None) -> StageRunReport:
    """Run one bounded dense-growth stage and prove every frozen parameter is unchanged."""
    if not isinstance(model, HostedProgressiveLanguageModel) \
            or not isinstance(spec, StageTrainSpec) or spec.stage != "pretrain":
        raise ValueError("progressive training requires its model and stage='pretrain'")
    if spec.steps > _MAX_HOSTED_STEPS:
        raise ValueError("progressive training steps exceed the bounded hosted limit")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(targets, torch.Tensor) \
            or input_ids.shape != targets.shape or input_ids.ndim != 2 \
            or input_ids.device != targets.device \
            or input_ids.device != next(model.parameters()).device:
        raise ValueError("progressive training tensors are malformed or on the wrong device")
    model._validate_ids(input_ids)
    if targets.dtype != torch.long or torch.any(targets < 0) \
            or torch.any(targets >= model.interface.vocab_size):
        raise ValueError("progressive targets must be in-vocabulary torch.long")
    stage = model.activate_stage(growth, stage_index)
    frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    input_sha = sha256_text(canonical_json({
        "growth": growth.digest, "stage": stage_index,
        "inputs": input_ids.detach().cpu().tolist(),
        "targets": targets.detach().cpu().tolist()}))
    with _deterministic_stage(spec.seed):
        initial = _growth_loss(model, input_ids, targets)
        optimizer = _optimizer(spec, trainable)
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            _logits, loss = model(input_ids, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("progressive language model produced non-finite gradients")
            optimizer.step()
            _emit(telemetry_sink, "pretrain", step + 1, float(loss.detach().item()),
                  float(norm.item()), (step + 1) * input_ids.shape[0])
        for name, parameter in model.named_parameters():
            if name in frozen and not torch.equal(parameter.detach(), frozen[name]):
                raise RuntimeError(f"frozen progressive parameter changed: {name}")
        final = _growth_loss(model, input_ids, targets)
        return StageRunReport(
            "pretrain", spec.digest, input_sha, initial, final, spec.steps,
            input_ids.shape[0], _module_digest(model),
            sha256_text(canonical_json({
                "growth": growth.digest, "stage": stage_index,
                "active_parameter_elements": stage.active_parameter_elements})))


__all__ = [
    "ExpandedRowTable", "HostedProgressiveLanguageModel",
    "clipped_cross_tokenizer_opd_loss", "initialize_expanded_rows",
    "train_progressive_growth_stage",
]
