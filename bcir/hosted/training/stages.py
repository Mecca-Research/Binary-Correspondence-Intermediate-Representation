"""PyTorch-backed SFT, preference, PPO, and reasoning stages for BCIR micro models.

These are bounded reference implementations.  They establish objective semantics and
artifact parity for small tests; target-specific fused training kernels remain the job of
BCIR/GEM lowering and hosted tensor providers.
"""

from __future__ import annotations

import copy
import hashlib
import math
from contextlib import contextmanager
from dataclasses import asdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "hosted alignment stages require PyTorch; install bcir[model-lab]"
    ) from exc

from ..models.checkpoint import tensor_mapping_digest
from ..models.model import HostedLlama
from .contracts import (
    PPOExample,
    PreferenceExample,
    ReasoningExample,
    SFTExample,
    StageRunReport,
    StageTrainEvent,
    StageTrainSpec,
    canonical_json,
    sha256_text,
)


_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()


def _module_digest(model: nn.Module) -> str:
    return tensor_mapping_digest({name: value for name, value in model.state_dict().items()})


def _input_digest(examples) -> str:
    return sha256_text(canonical_json([asdict(example) for example in examples]))


def _device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover - defensive
        raise ValueError("training model has no parameters") from exc


def _optimizer(spec: StageTrainSpec, parameters):
    return torch.optim.AdamW(
        parameters,
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
        amsgrad=False,
    )


@contextmanager
def _deterministic_stage(seed: int):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


def _validate_model_and_spec(
    model: HostedLlama, spec: StageTrainSpec, stages: tuple[str, ...]
) -> None:
    if not isinstance(model, HostedLlama) or not isinstance(spec, StageTrainSpec):
        raise ValueError("model and stage specification have invalid types")
    if spec.stage not in stages:
        raise ValueError(f"stage {spec.stage!r} is not admitted by this trainer")


def _validate_examples(examples, cls):
    if (
        not isinstance(examples, (tuple, list))
        or not examples
        or any(not isinstance(example, cls) for example in examples)
    ):
        raise ValueError(f"examples must be a nonempty sequence of {cls.__name__}")
    return tuple(examples)


def _emit(sink, stage: str, step: int, loss: float, norm: float, examples_processed: int) -> None:
    if sink is None:
        return
    if not callable(sink):
        raise ValueError("telemetry_sink must be callable or None")
    sink(StageTrainEvent(stage, step, loss, norm, examples_processed))


def _sequence_tensors(prompt, response, device: torch.device):
    tokens = tuple(prompt) + tuple(response)
    if len(tokens) < 2:
        raise ValueError("a training sequence needs at least two tokens")
    inputs = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)
    start = max(0, len(prompt) - 1)
    mask = torch.zeros_like(inputs, dtype=torch.float32)
    mask[:, start:] = 1.0
    return inputs, targets, mask


def _ensure_context(model: HostedLlama, length: int) -> None:
    if model.context_length is not None and length > model.context_length:
        raise ValueError(f"sequence length {length} exceeds model context {model.context_length}")


def _sft_loss(model: HostedLlama, example: SFTExample) -> torch.Tensor:
    _ensure_context(model, len(example.prompt_ids) + len(example.response_ids) - 1)
    inputs, targets, mask = _sequence_tensors(
        example.prompt_ids, example.response_ids, _device(model)
    )
    _logits, loss = model(inputs, targets, mask)
    return loss * example.weight


@torch.no_grad()
def _mean_sft_loss(model: HostedLlama, examples) -> float:
    was_training = model.training
    model.eval()
    try:
        value = sum(float(_sft_loss(model, example).item()) for example in examples) / len(examples)
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise RuntimeError("SFT evaluation produced a non-finite loss")
    return value


def train_sft(
    model: HostedLlama, examples, spec: StageTrainSpec, telemetry_sink=None
) -> StageRunReport:
    _validate_model_and_spec(model, spec, ("sft", "reasoning"))
    examples = _validate_examples(examples, SFTExample)
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    with _deterministic_stage(spec.seed):
        initial = _mean_sft_loss(model, examples)
        optimizer = _optimizer(spec, model.parameters())
        cursor = 0
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            count = spec.batch_size * spec.gradient_accumulation
            for _ in range(count):
                loss = _sft_loss(model, examples[cursor % len(examples)]) / count
                cursor += 1
                running += float(loss.detach().item())
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("SFT produced a non-finite gradient norm")
            optimizer.step()
            _emit(telemetry_sink, spec.stage, step + 1, running, float(norm.item()), cursor)
        final = _mean_sft_loss(model, examples)
        return StageRunReport(
            spec.stage,
            spec.digest,
            _input_digest(examples),
            initial,
            final,
            spec.steps,
            len(examples),
            _module_digest(model),
            _EMPTY_DIGEST,
        )


class HostedRewardModel(nn.Module):
    """A trainable scalar preference head over a copied hosted Llama backbone."""

    def __init__(self, policy: HostedLlama):
        super().__init__()
        if not isinstance(policy, HostedLlama):
            raise ValueError("reward model requires a HostedLlama policy")
        self.policy = copy.deepcopy(policy)
        self.score = nn.Linear(policy.spec.d_model, 1, bias=False, device=_device(policy))
        nn.init.zeros_(self.score.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.policy.hidden_states(input_ids)
        return self.score(states[:, -1, :]).squeeze(-1)


class HostedValueModel(nn.Module):
    """Token-value head used by the bounded PPO reference."""

    def __init__(self, policy: HostedLlama):
        super().__init__()
        if not isinstance(policy, HostedLlama):
            raise ValueError("value model requires a HostedLlama policy")
        self.policy = copy.deepcopy(policy)
        self.value = nn.Linear(policy.spec.d_model, 1, bias=False, device=_device(policy))
        nn.init.zeros_(self.value.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.value(self.policy.hidden_states(input_ids)).squeeze(-1)


class HostedEmbeddingStudent(nn.Module):
    """Normalized pooled embedding head trained from frozen relational targets."""

    def __init__(self, policy: HostedLlama, output_dim: int):
        super().__init__()
        if not isinstance(policy, HostedLlama) or type(output_dim) is not int or output_dim < 1:
            raise ValueError("embedding student requires a HostedLlama and positive output_dim")
        self.policy = copy.deepcopy(policy)
        self.projection = nn.Linear(
            policy.spec.d_model, output_dim, bias=False, device=_device(policy)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        states = self.policy.hidden_states(input_ids).mean(dim=1)
        return F.normalize(self.projection(states).float(), dim=-1)


def _embedding_loss(model: HostedEmbeddingStudent, sequences, targets) -> torch.Tensor:
    embeddings = []
    for sequence in sequences:
        _ensure_context(model.policy, len(sequence))
        values = torch.tensor([sequence], dtype=torch.long, device=_device(model))
        embeddings.append(model(values)[0])
    matrix = torch.stack(embeddings) @ torch.stack(embeddings).T
    expected = torch.tensor(targets, dtype=torch.float32, device=matrix.device)
    return F.mse_loss(matrix, expected)


def train_embedding_distillation(
    model: HostedEmbeddingStudent,
    sequences,
    relational_targets,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    if (
        not isinstance(model, HostedEmbeddingStudent)
        or not isinstance(spec, StageTrainSpec)
        or spec.stage != "embedding"
    ):
        raise ValueError("embedding distillation requires its model and stage='embedding'")
    if not isinstance(sequences, (tuple, list)) or len(sequences) < 2:
        raise ValueError("embedding distillation requires at least two token sequences")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    sequences = tuple(tuple(sequence) for sequence in sequences)
    if any(
        not sequence
        or any(
            type(token) is not int or token < 0 or token >= model.policy.spec.vocab_size
            for token in sequence
        )
        for sequence in sequences
    ):
        raise ValueError("embedding sequences contain invalid token IDs")
    if (
        not isinstance(relational_targets, (tuple, list))
        or len(relational_targets) != len(sequences)
        or any(
            not isinstance(row, (tuple, list)) or len(row) != len(sequences)
            for row in relational_targets
        )
    ):
        raise ValueError("relational target matrix shape is invalid")
    targets = tuple(tuple(float(value) for value in row) for row in relational_targets)
    if any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0 for row in targets for value in row
    ):
        raise ValueError("relational target values must be finite cosine similarities")
    input_sha = sha256_text(canonical_json({"sequences": sequences, "targets": targets}))
    with _deterministic_stage(spec.seed):
        with torch.no_grad():
            initial = float(_embedding_loss(model, sequences, targets).item())
        if not math.isfinite(initial):
            raise RuntimeError("embedding distillation produced a non-finite initial loss")
        optimizer = _optimizer(spec, model.parameters())
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _embedding_loss(model, sequences, targets)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("embedding distillation produced a non-finite gradient norm")
            optimizer.step()
            _emit(
                telemetry_sink,
                "embedding",
                step + 1,
                float(loss.detach().item()),
                float(norm.item()),
                (step + 1) * len(sequences),
            )
        with torch.no_grad():
            final = float(_embedding_loss(model, sequences, targets).item())
        if not math.isfinite(final):
            raise RuntimeError("embedding distillation produced a non-finite final loss")
        return StageRunReport(
            "embedding",
            spec.digest,
            input_sha,
            initial,
            final,
            spec.steps,
            len(sequences),
            _module_digest(model),
            _EMPTY_DIGEST,
        )


def _reward(model: HostedRewardModel, prompt, response) -> torch.Tensor:
    tokens = tuple(prompt) + tuple(response)
    _ensure_context(model.policy, len(tokens))
    values = torch.tensor([tokens], dtype=torch.long, device=_device(model))
    return model(values)[0]


def _reward_loss(model: HostedRewardModel, example: PreferenceExample) -> torch.Tensor:
    chosen = _reward(model, example.prompt_ids, example.chosen_ids)
    rejected = _reward(model, example.prompt_ids, example.rejected_ids)
    return -F.logsigmoid(chosen - rejected) * example.weight


@torch.no_grad()
def _mean_reward_loss(model, examples) -> float:
    was_training = model.training
    model.eval()
    try:
        value = sum(float(_reward_loss(model, example).item()) for example in examples) / len(
            examples
        )
    finally:
        model.train(was_training)
    if not math.isfinite(value):
        raise RuntimeError("reward evaluation produced a non-finite loss")
    return value


def train_reward_model(
    model: HostedRewardModel, examples, spec: StageTrainSpec, telemetry_sink=None
) -> StageRunReport:
    if (
        not isinstance(model, HostedRewardModel)
        or not isinstance(spec, StageTrainSpec)
        or spec.stage != "reward"
    ):
        raise ValueError("reward training requires HostedRewardModel and stage='reward'")
    examples = _validate_examples(examples, PreferenceExample)
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    with _deterministic_stage(spec.seed):
        initial = _mean_reward_loss(model, examples)
        optimizer = _optimizer(spec, model.parameters())
        cursor = 0
        model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            count = spec.batch_size * spec.gradient_accumulation
            for _ in range(count):
                loss = _reward_loss(model, examples[cursor % len(examples)]) / count
                cursor += 1
                running += float(loss.detach().item())
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("reward training produced a non-finite gradient norm")
            optimizer.step()
            _emit(telemetry_sink, "reward", step + 1, running, float(norm.item()), cursor)
        final = _mean_reward_loss(model, examples)
        return StageRunReport(
            "reward",
            spec.digest,
            _input_digest(examples),
            initial,
            final,
            spec.steps,
            len(examples),
            _module_digest(model),
            _EMPTY_DIGEST,
        )


def _token_log_probs(model: HostedLlama, prompt, response) -> torch.Tensor:
    _ensure_context(model, len(prompt) + len(response) - 1)
    inputs, targets, mask = _sequence_tensors(prompt, response, _device(model))
    logits, _loss = model(inputs)
    token_log_probs = (
        F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    )
    return token_log_probs[0][mask[0].bool()]


def _dpo_loss(policy, reference, example, beta: float) -> torch.Tensor:
    policy_margin = (
        _token_log_probs(policy, example.prompt_ids, example.chosen_ids).sum()
        - _token_log_probs(policy, example.prompt_ids, example.rejected_ids).sum()
    )
    with torch.no_grad():
        reference_margin = (
            _token_log_probs(reference, example.prompt_ids, example.chosen_ids).sum()
            - _token_log_probs(reference, example.prompt_ids, example.rejected_ids).sum()
        )
    return -F.logsigmoid(beta * (policy_margin - reference_margin)) * example.weight


@torch.no_grad()
def _mean_dpo_loss(policy, reference, examples, beta) -> float:
    policy_training = policy.training
    reference_training = reference.training
    policy.eval()
    reference.eval()
    try:
        value = sum(
            float(_dpo_loss(policy, reference, example, beta).item()) for example in examples
        ) / len(examples)
    finally:
        policy.train(policy_training)
        reference.train(reference_training)
    if not math.isfinite(value):
        raise RuntimeError("DPO evaluation produced a non-finite loss")
    return value


def train_dpo(
    policy: HostedLlama, reference: HostedLlama, examples, spec: StageTrainSpec, telemetry_sink=None
) -> StageRunReport:
    _validate_model_and_spec(policy, spec, ("dpo",))
    if not isinstance(reference, HostedLlama) or reference.spec != policy.spec:
        raise ValueError("DPO reference must be a HostedLlama with the policy specification")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("DPO reference parameters must be frozen")
    examples = _validate_examples(examples, PreferenceExample)
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    with _deterministic_stage(spec.seed):
        initial = _mean_dpo_loss(policy, reference, examples, spec.preference_beta)
        optimizer = _optimizer(spec, policy.parameters())
        cursor = 0
        policy.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            count = spec.batch_size * spec.gradient_accumulation
            for _ in range(count):
                loss = (
                    _dpo_loss(
                        policy, reference, examples[cursor % len(examples)], spec.preference_beta
                    )
                    / count
                )
                cursor += 1
                running += float(loss.detach().item())
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("DPO produced a non-finite gradient norm")
            optimizer.step()
            _emit(telemetry_sink, "dpo", step + 1, running, float(norm.item()), cursor)
        final = _mean_dpo_loss(policy, reference, examples, spec.preference_beta)
        return StageRunReport(
            "dpo",
            spec.digest,
            _input_digest(examples),
            initial,
            final,
            spec.steps,
            len(examples),
            _module_digest(policy),
            _module_digest(reference),
        )


def generalized_advantage_estimate(
    rewards: torch.Tensor, values: torch.Tensor, gamma: float, gae_lambda: float
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 1 or values.shape != rewards.shape or rewards.numel() < 1:
        raise ValueError("GAE rewards and values must be equal nonempty vectors")
    advantages = torch.zeros_like(rewards)
    following_value = torch.zeros_like(values[0])
    following_advantage = torch.zeros_like(values[0])
    for index in range(rewards.numel() - 1, -1, -1):
        delta = rewards[index] + gamma * following_value - values[index]
        following_advantage = delta + gamma * gae_lambda * following_advantage
        advantages[index] = following_advantage
        following_value = values[index]
    return advantages, advantages + values


def _ppo_snapshot(policy, reference, value_model, example, spec):
    with torch.no_grad():
        old = _token_log_probs(policy, example.prompt_ids, example.response_ids).detach()
        ref = _token_log_probs(reference, example.prompt_ids, example.response_ids).detach()
        inputs, _targets, mask = _sequence_tensors(
            example.prompt_ids, example.response_ids, _device(value_model)
        )
        values = value_model(inputs)[0][mask[0].bool()].detach()
        rewards = torch.zeros_like(values)
        rewards[-1] = example.reward
        rewards -= spec.kl_coefficient * (old - ref)
        advantages, returns = generalized_advantage_estimate(
            rewards, values, spec.gamma, spec.gae_lambda
        )
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return old, ref, values, advantages, returns


def _ppo_loss(policy, value_model, example, snapshot, spec):
    old, reference, old_values, advantages, returns = snapshot
    current = _token_log_probs(policy, example.prompt_ids, example.response_ids)
    inputs, _targets, mask = _sequence_tensors(
        example.prompt_ids, example.response_ids, _device(value_model)
    )
    values = value_model(inputs)[0][mask[0].bool()]
    ratio = torch.exp(current - old)
    clipped_ratio = torch.clamp(ratio, 1.0 - spec.ppo_clip, 1.0 + spec.ppo_clip)
    policy_loss = -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()
    clipped_values = old_values + torch.clamp(
        values - old_values, -spec.value_clip, spec.value_clip
    )
    value_loss = (
        0.5 * torch.maximum((values - returns).square(), (clipped_values - returns).square()).mean()
    )
    kl_penalty = (current - reference).square().mean()
    return policy_loss + value_loss + spec.kl_coefficient * kl_penalty


def train_ppo(
    policy: HostedLlama,
    reference: HostedLlama,
    value_model: HostedValueModel,
    examples,
    spec: StageTrainSpec,
    telemetry_sink=None,
) -> StageRunReport:
    _validate_model_and_spec(policy, spec, ("ppo",))
    if (
        not isinstance(reference, HostedLlama)
        or reference.spec != policy.spec
        or not isinstance(value_model, HostedValueModel)
        or value_model.policy.spec != policy.spec
    ):
        raise ValueError("PPO requires matching policy/reference and a HostedValueModel")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("PPO reference parameters must be frozen")
    examples = _validate_examples(examples, PPOExample)
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    with _deterministic_stage(spec.seed):
        snapshots = [
            _ppo_snapshot(policy, reference, value_model, example, spec) for example in examples
        ]
        initial = sum(
            float(_ppo_loss(policy, value_model, example, snapshot, spec).item())
            for example, snapshot in zip(examples, snapshots)
        ) / len(examples)
        if not math.isfinite(initial):
            raise RuntimeError("PPO evaluation produced a non-finite initial loss")
        parameters = [*policy.parameters(), *value_model.parameters()]
        optimizer = _optimizer(spec, parameters)
        cursor = 0
        policy.train()
        value_model.train()
        for step in range(spec.steps):
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            count = spec.batch_size * spec.gradient_accumulation
            for _ in range(count):
                index = cursor % len(examples)
                cursor += 1
                loss = (
                    _ppo_loss(policy, value_model, examples[index], snapshots[index], spec) / count
                )
                running += float(loss.detach().item())
                loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, spec.grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("PPO produced a non-finite gradient norm")
            optimizer.step()
            _emit(telemetry_sink, "ppo", step + 1, running, float(norm.item()), cursor)
        final = sum(
            float(_ppo_loss(policy, value_model, example, snapshot, spec).item())
            for example, snapshot in zip(examples, snapshots)
        ) / len(examples)
        if not math.isfinite(final):
            raise RuntimeError("PPO evaluation produced a non-finite final loss")
        return StageRunReport(
            "ppo",
            spec.digest,
            _input_digest(examples),
            initial,
            final,
            spec.steps,
            len(examples),
            _module_digest(policy),
            _module_digest(value_model),
        )


def train_reasoning_sft(
    model: HostedLlama, examples, spec: StageTrainSpec, telemetry_sink=None
) -> StageRunReport:
    if not isinstance(spec, StageTrainSpec) or spec.stage != "reasoning":
        raise ValueError("reasoning SFT requires stage='reasoning'")
    reasoning = _validate_examples(examples, ReasoningExample)
    sft = tuple(example.as_sft() for example in reasoning)
    report = train_sft(model, sft, spec, telemetry_sink)
    return StageRunReport(
        "reasoning",
        report.spec_sha256,
        _input_digest(reasoning),
        report.initial_loss,
        report.final_loss,
        report.steps,
        len(reasoning),
        report.model_sha256,
        report.auxiliary_sha256,
    )
