#!/usr/bin/env python3
"""Bounded CPU gate for BCIR adaptive-transformer research architectures.

No network, checkpoint, GPU, external repository, or large training run is used.  Two
identical one-thread runs must produce the same timestamp-free report on each host.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import torch

from bcir.hosted.training.adaptive_transformer import (
    HostedAdaptiveLanguageModel,
    HostedMultiPatchTransformer,
    train_adaptive_pretrain,
    train_multipatch_reconstruction,
)
from bcir.hosted.training.contracts import StageTrainSpec
from bcir.kbcir.adaptive_transformer import (
    AdaptiveLanguageSpec,
    ExogenousAnchorSpec,
    MultiPatchSpec,
    ReferenceWindowSpec,
    WidthScheduleSpec,
    assess_adaptive_language,
    assess_multi_patch,
    fixed_residual_update,
    fixed_residual_view,
    lower_adaptive_language,
)


UPSTREAM_RESEARCH = {
    "deeploop": {
        "commit": "9d86da3367214b1e760c4713dc8612d2ae518430",
        "license": "Apache-2.0",
        "adoption": "looped physical blocks plus LoopDeepNorm",
    },
    "variable_width_transformers": {
        "commit": "69dde8143d9a7912de353b57093d36d8788070d4",
        "license": "NOASSERTION",
        "adoption": "paper-derived fixed residual carry-forward, schedule, and accounting only",
    },
    "mpdit": {
        "commit": "258ebda7a2e15dc99f3f3948520e3098f348dd9f",
        "license": "NOASSERTION",
        "adoption": "independent bounded coarse-to-fine token reference only",
    },
    "unlimited_ocr": {
        "commit": "1ab6b46b989ebf26328a968d87ce583a9650ab90",
        "license": "MIT",
        "adoption": "reference-plus-sliding attention and bounded continuation cache",
    },
    "exoformer": {
        "paper": "arXiv:2601.08131v4",
        "license": "paper-derived-independent-implementation",
        "adoption": "normalized H0/H1 anchor projection mixing",
    },
}


def _expect_value_error(call, fragment: str) -> None:
    try:
        call()
    except ValueError as exc:
        if fragment not in str(exc):
            raise AssertionError(f"unexpected refusal: {exc}") from exc
    else:
        raise AssertionError("bounded hosted reference accepted an oversized request")


def _language_case(name: str, architecture: AdaptiveLanguageSpec, seed: int) -> dict:
    torch.manual_seed(seed)
    model = HostedAdaptiveLanguageModel(architecture)
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
    cost = assess_adaptive_language(
        architecture, prefill_tokens=inputs.shape[1], decode_context=inputs.shape[1]
    )
    if model.unique_parameter_count() != cost.parameter_elements:
        raise AssertionError(f"{name} parameter accounting diverged from the hosted model")
    events = []
    report = train_adaptive_pretrain(
        model,
        inputs,
        targets,
        StageTrainSpec("pretrain", steps=3, learning_rate=3.0e-3, weight_decay=0.0, seed=seed),
        events.append,
    )
    if not report.final_loss < report.initial_loss:
        raise AssertionError(f"{name} tiny training did not reduce its repeated-batch loss")
    if len(events) != 3 or [event.step for event in events] != [1, 2, 3]:
        raise AssertionError(f"{name} emitted malformed training telemetry")
    lowered = lower_adaptive_language(architecture, sequence_length=inputs.shape[1])
    if not lowered.streampack.provenance_ok():
        raise AssertionError(f"{name} produced an untraced StreamPack")
    return {
        "architecture_sha256": architecture.digest,
        "cost_sha256": cost.digest,
        "parameters": cost.parameter_elements,
        "physical_layers": cost.physical_layers,
        "effective_layers": cost.effective_layers,
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "model_sha256": report.model_sha256,
        "plan_sha256": lowered.artifact_sha256,
        "streampack_bytes": len(lowered.streampack_data),
        "prefill_flops_lower_bound": cost.prefill_flops,
        "kv_cache_bytes": cost.kv_cache_bytes,
    }


def _multipatch_case(seed: int) -> dict:
    spec = MultiPatchSpec(
        image_size=8,
        input_channels=2,
        hidden_width=16,
        heads=4,
        coarse_patch=4,
        fine_patch=2,
        coarse_layers=1,
        fine_layers=1,
        condition_tokens=2,
    )
    torch.manual_seed(seed)
    model = HostedMultiPatchTransformer(spec)
    images = torch.linspace(-1.0, 1.0, 128, dtype=torch.float32).view(1, 2, 8, 8)
    cost = assess_multi_patch(spec)
    if cost.parameter_elements != sum(parameter.numel() for parameter in model.parameters()):
        raise AssertionError("multi-patch parameter accounting diverged from the hosted model")
    report = train_multipatch_reconstruction(
        model,
        images,
        StageTrainSpec("supervised", steps=3, learning_rate=3.0e-3, weight_decay=0.0, seed=seed),
    )
    if not report.final_loss < report.initial_loss:
        raise AssertionError("multi-patch tiny reconstruction did not reduce loss")
    return {
        "architecture_sha256": spec.digest,
        "cost_sha256": cost.digest,
        "parameters": cost.parameter_elements,
        "coarse_tokens": cost.coarse_tokens,
        "fine_tokens": cost.fine_tokens,
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "model_sha256": report.model_sha256,
        "flops_lower_bound": cost.estimated_flops,
        "uniform_fine_flops_lower_bound": cost.baseline_flops,
    }


def _run_once() -> dict:
    variable = AdaptiveLanguageSpec(
        vocab_size=32,
        context_length=8,
        schedule=WidthScheduleSpec.symmetric_u(
            layers=3, outer_width=24, inner_width=16, head_dim=8, ff_multiplier=2, quantum=8
        ),
        repeats=2,
        residual_mode="loop_deepnorm",
    )
    exogenous = AdaptiveLanguageSpec(
        vocab_size=32,
        context_length=8,
        schedule=WidthScheduleSpec((16, 16), (2, 2), (32, 32)),
        gated_attention=True,
        reference_window=ReferenceWindowSpec(prefix_tokens=2, window_tokens=3),
        anchor=ExogenousAnchorSpec(mode="dynamic", granularity="elementwise", source_depth=1),
    )
    static_exogenous = AdaptiveLanguageSpec(
        vocab_size=32,
        context_length=8,
        schedule=WidthScheduleSpec((16, 16), (2, 2), (32, 32)),
        gated_attention=True,
        anchor=ExogenousAnchorSpec(mode="static", granularity="headwise", source_depth=0),
    )

    baseline_for_scaling = AdaptiveLanguageSpec(
        vocab_size=32, context_length=8, schedule=WidthScheduleSpec((16,), (2,), (32,))
    )
    loop_for_scaling = AdaptiveLanguageSpec(
        vocab_size=32,
        context_length=8,
        schedule=WidthScheduleSpec((16,), (2,), (32,)),
        repeats=2,
        residual_mode="loop_deepnorm",
    )

    _expect_value_error(
        lambda: HostedAdaptiveLanguageModel(
            AdaptiveLanguageSpec(
                vocab_size=32, context_length=4097, schedule=WidthScheduleSpec((16,), (2,), (32,))
            )
        ),
        "context",
    )
    _expect_value_error(
        lambda: HostedAdaptiveLanguageModel(
            AdaptiveLanguageSpec(
                vocab_size=1_000_000,
                context_length=8,
                schedule=WidthScheduleSpec((64,), (4,), (128,)),
            )
        ),
        "parameter",
    )
    _expect_value_error(
        lambda: HostedMultiPatchTransformer(
            MultiPatchSpec(
                image_size=64,
                input_channels=256,
                hidden_width=64,
                heads=4,
                coarse_patch=64,
                fine_patch=32,
                coarse_layers=1,
                fine_layers=1,
            )
        ),
        "parameter",
    )
    torch.manual_seed(990)
    baseline_model = HostedAdaptiveLanguageModel(baseline_for_scaling)
    torch.manual_seed(990)
    loop_model = HostedAdaptiveLanguageModel(loop_for_scaling)
    baseline_block, loop_block = baseline_model.blocks[0], loop_model.blocks[0]
    if not torch.equal(loop_block.attention.q_proj.weight, baseline_block.attention.q_proj.weight):
        raise AssertionError("LoopDeepNorm unexpectedly scaled the query projection")
    for baseline_weight, loop_weight in (
        (baseline_block.attention.v_proj.weight, loop_block.attention.v_proj.weight),
        (baseline_block.attention.o_proj.weight, loop_block.attention.o_proj.weight),
        (baseline_block.mlp.gate.weight, loop_block.mlp.gate.weight),
        (baseline_block.mlp.up.weight, loop_block.mlp.up.weight),
        (baseline_block.mlp.down.weight, loop_block.mlp.down.weight),
    ):
        if not torch.equal(loop_weight, baseline_weight * loop_for_scaling.loop_beta):
            raise AssertionError("LoopDeepNorm residual-branch beta initialization regressed")

    # The carry-forward invariant distinguishes the paper's fixed residual stream
    # from its weaker zero-padding ablation.
    residual = [float(index) for index in range(8)]
    narrowed = fixed_residual_view(residual, 1, 8, 4)
    restored = fixed_residual_update(residual, [10.0, 11.0, 12.0, 13.0], 1, 8, 4)
    if narrowed != [0.0, 1.0, 2.0, 3.0] or restored[4:] != residual[4:]:
        raise AssertionError("fixed residual carry-forward semantics regressed")

    # Prove the dynamic coefficient generator starts at sigmoid(0)=0.5, and exercise
    # the static variant's forward/backward without another optimization trajectory.
    torch.manual_seed(991)
    dynamic_model = HostedAdaptiveLanguageModel(exogenous)
    gamma = torch.sigmoid(
        dynamic_model.mixers[0].dynamic_out(torch.zeros(1, 1, exogenous.anchor.dynamic_hidden))
    )
    if not torch.equal(gamma, torch.full_like(gamma, 0.5)):
        raise AssertionError("dynamic ExoFormer coefficients lost their identity initialization")
    torch.manual_seed(992)
    static_model = HostedAdaptiveLanguageModel(static_exogenous)
    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    _logits, static_loss = static_model(ids, torch.tensor([[2, 3, 4, 5]]))
    static_loss.backward()
    if not all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in static_model.parameters()
    ):
        raise AssertionError("static ExoFormer produced a non-finite gradient")

    return {
        "schema": "bcir.adaptive_transformer_gate.v1",
        "upstream_research": UPSTREAM_RESEARCH,
        "looped_variable_width": _language_case("looped-variable-width", variable, 1729),
        "dynamic_exogenous_rswa": _language_case("dynamic-exogenous-rswa", exogenous, 1730),
        "static_exogenous_gradient_finite": True,
        "loopdeepnorm_beta_initialization_verified": True,
        "oversized_hosted_requests_refused_before_allocation": True,
        "multipatch": _multipatch_case(1731),
        "carry_forward_restored_suffix": restored[4:],
        "large_training_performed": False,
        "gpu_used": False,
        "external_source_or_weights_imported": False,
    }


def run_gate() -> dict:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous_rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        first = _run_once()
        second = _run_once()
        if first != second:
            raise AssertionError("adaptive transformer gate is not deterministic on this host")
        return first
    finally:
        torch.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(previous_threads)


def _atomic_report(output_dir: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "report.json"
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(prefix=".adaptive-report-", suffix=".part", dir=output_dir)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = run_gate()
    path = _atomic_report(Path(args.output_dir), report)
    print(json.dumps({"report": str(path), "schema": report["schema"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
