"""Deterministic AdamW training for the optional hosted Llama reference."""

from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError("hosted training requires PyTorch; install bcir[model-lab]") from exc

from .checkpoint import load_safe_checkpoint, save_safe_checkpoint
from .model import HostedLlama
from .spec import (
    CorpusManifest,
    HostedRunReport,
    HostedTokenSource,
    HostedTrainEvent,
    HostedTrainSpec,
)


def _learning_rate(spec: HostedTrainSpec, step: int) -> float:
    if spec.warmup_steps and step < spec.warmup_steps:
        return spec.learning_rate * (step + 1) / spec.warmup_steps
    if spec.steps <= spec.warmup_steps + 1:
        return spec.min_learning_rate
    progress = (step - spec.warmup_steps) / (spec.steps - spec.warmup_steps - 1)
    progress = min(1.0, max(0.0, progress))
    return spec.min_learning_rate + 0.5 * (spec.learning_rate - spec.min_learning_rate) * (
        1.0 + math.cos(math.pi * progress)
    )


def _device_and_dtype(device_name: str, dtype_name: str):
    if device_name not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda' (no implicit auto-selection)")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device_name == "cpu" and dtype_name != "float32":
        raise ValueError("the deterministic CPU rail supports dtype='float32' only")
    if device_name == "cuda" and dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 was requested but the CUDA device does not support it")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        dtype_name
    ]
    return torch.device(device_name), dtype


def _batch(
    source: HostedTokenSource, batch_index: int, spec: HostedTrainSpec, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = source.batch(batch_index, spec.batch_size, spec.context_length)
    if (
        not isinstance(rows, list)
        or len(rows) != spec.batch_size
        or any(not isinstance(row, list) or len(row) != spec.context_length + 1 for row in rows)
    ):
        raise ValueError("token source returned a malformed batch")
    for row_index, row in enumerate(rows):
        for column_index, token in enumerate(row):
            if type(token) is not int or not 0 <= token < spec.decoder.vocab_size:
                raise ValueError(
                    f"token source returned an invalid ID at [{row_index}][{column_index}]"
                )
    values = torch.tensor(rows, dtype=torch.long, device=device)
    return values[:, :-1], values[:, 1:]


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.no_grad()
def _loss(
    model: HostedLlama,
    source: HostedTokenSource,
    spec: HostedTrainSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    was_training = model.training
    model.eval()
    inputs, targets = _batch(source, 0, spec, device)
    with _autocast(device, dtype):
        _logits, loss = model(inputs, targets)
    if was_training:
        model.train()
    value = float(loss.item())
    if not math.isfinite(value):
        raise RuntimeError("hosted evaluation produced a non-finite loss")
    return value


def _write_report(path: Path, report: HostedRunReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".run-report.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(report.to_json().encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _train_hosted_impl(
    spec: HostedTrainSpec,
    source: HostedTokenSource,
    output_dir: os.PathLike | str,
    *,
    device: str,
    resume_from: os.PathLike | str | None,
    telemetry_sink,
) -> HostedRunReport:
    if not isinstance(spec, HostedTrainSpec):
        raise ValueError("spec must be a HostedTrainSpec")
    corpus = getattr(source, "manifest", None)
    if not isinstance(corpus, CorpusManifest):
        raise ValueError("source must expose a CorpusManifest")
    if corpus.vocab_size != spec.decoder.vocab_size:
        raise ValueError("source and decoder vocabulary sizes differ")
    if telemetry_sink is not None and not callable(telemetry_sink):
        raise ValueError("telemetry_sink must be callable or None")
    target_device, compute_dtype = _device_and_dtype(device, spec.dtype)

    torch.manual_seed(spec.seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)
        torch.cuda.reset_peak_memory_stats(target_device)
    model = HostedLlama(
        spec.decoder,
        activation_checkpointing=spec.activation_checkpointing,
        context_length=spec.context_length,
    ).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(spec.beta1, spec.beta2),
        eps=spec.epsilon,
        weight_decay=spec.weight_decay,
        amsgrad=False,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(target_device.type == "cuda" and spec.dtype == "float16")
    )
    global_step = batch_index = 0
    initial_loss = _loss(model, source, spec, target_device, compute_dtype)
    if resume_from is not None:
        loaded = load_safe_checkpoint(
            resume_from, model, optimizer, spec=spec, corpus=corpus, device=target_device
        )
        global_step, batch_index, initial_loss = (
            loaded.global_step,
            loaded.batch_index,
            loaded.initial_loss,
        )
        torch.set_rng_state(loaded.cpu_rng.to("cpu"))
        if loaded.cuda_rng:
            if target_device.type != "cuda" or len(loaded.cuda_rng) != torch.cuda.device_count():
                raise ValueError("checkpoint CUDA RNG topology differs from this host")
            torch.cuda.set_rng_state_all([state.to("cpu") for state in loaded.cuda_rng])
        if loaded.scaler_state:
            scaler.load_state_dict(loaded.scaler_state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    last_saved = None
    model.train()
    while global_step < spec.steps:
        optimizer.zero_grad(set_to_none=True)
        lr = _learning_rate(spec, global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for _micro in range(spec.gradient_accumulation):
            inputs, targets = _batch(source, batch_index, spec, target_device)
            batch_index += 1
            with _autocast(target_device, compute_dtype):
                _logits, loss = model(inputs, targets)
                scaled_loss = loss / spec.gradient_accumulation
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {global_step}")
            loss_sum += float(loss.detach().item())
            scaler.scale(scaled_loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), spec.grad_clip)
        grad_norm = float(grad_norm_tensor.item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at step {global_step}")
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        tokens = global_step * spec.batch_size * spec.context_length * spec.gradient_accumulation
        peak = (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else 0
        )
        event = HostedTrainEvent(
            global_step, loss_sum / spec.gradient_accumulation, lr, grad_norm, tokens, peak
        )
        if telemetry_sink is not None:
            telemetry_sink(event)
        if global_step % spec.checkpoint_every == 0 or global_step == spec.steps:
            last_saved = save_safe_checkpoint(
                output,
                model,
                optimizer,
                spec=spec,
                corpus=corpus,
                global_step=global_step,
                batch_index=batch_index,
                initial_loss=initial_loss,
                scaler_state=scaler.state_dict(),
            )
    if last_saved is None:
        raise RuntimeError("hosted run did not publish a final checkpoint")
    final_loss = _loss(model, source, spec, target_device, compute_dtype)
    report = HostedRunReport(
        schema="bcir.hosted_run.v1",
        spec_sha256=spec.digest,
        corpus_sha256=corpus.digest,
        tokenizer_sha256=corpus.tokenizer_sha256,
        steps=global_step,
        batch_index=batch_index,
        tokens_processed=(
            global_step * spec.batch_size * spec.context_length * spec.gradient_accumulation
        ),
        initial_loss=initial_loss,
        final_loss=final_loss,
        model_sha256=last_saved.model_sha256,
        optimizer_sha256=last_saved.optimizer_sha256,
        checkpoint_sha256=last_saved.manifest_sha256,
    )
    _write_report(output / "run-report.json", report)
    return report


def train_hosted(
    spec: HostedTrainSpec,
    source: HostedTokenSource,
    output_dir: os.PathLike | str,
    *,
    device: str = "cpu",
    resume_from: os.PathLike | str | None = None,
    telemetry_sink=None,
) -> HostedRunReport:
    """Train through a deterministic boundary without leaking global mode changes."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        return _train_hosted_impl(
            spec,
            source,
            output_dir,
            device=device,
            resume_from=resume_from,
            telemetry_sink=telemetry_sink,
        )
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
