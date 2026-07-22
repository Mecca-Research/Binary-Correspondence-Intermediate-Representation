#!/usr/bin/env python3
"""Bounded CPU gate for BCIR byte-native model research.

No network, dataset, checkpoint, GPU, CUDA kernel, or large training run is used.  The
gate executes twice with one tensor thread and requires an identical timestamp-free
report on the current host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import torch

from bcir.hosted.models.checkpoint import tensor_mapping_digest
from bcir.hosted.training.byte_latent import (
    HostedByteLatentModel,
    HostedMambaByteModel,
    apply_byteified_transplant,
    byte_latent_tensor_shapes,
    byteified_parameter_groups,
    diffusion_draft,
    diffusion_generate,
    diffusion_verify_generate,
    greedy_generate_byte_latent,
    greedy_generate_mambabyte,
    self_speculative_generate,
    train_byte_latent_pretrain,
    train_mambabyte_pretrain,
)
from bcir.hosted.training.contracts import StageTrainSpec
from bcir.kbcir.byte_latent import (
    ByteDiffusionSpec,
    ByteIngestProfile,
    ByteLatentSpec,
    BytePatchSpec,
    ByteSpeculationSpec,
    LearnedPatchPolicySpec,
    MambaByteSpec,
    PatchLayout,
    TensorShape,
    assess_byte_latent,
    assess_mambabyte,
    choose_byte_ingest,
    encode_raw_sequence,
    lower_byte_latent,
    plan_byteified_transplant,
)
from bcir.kbcir.cost import TargetProfile


RESEARCH_PROVENANCE = {
    "byte_latent_transformer": {
        "paper": "arXiv:2412.09871v1",
        "pdf_sha256": "23e0cc90e9e2ef0c11291043ca7ad4d48d5384328f31007b8b40230f370756ff",
        "source_commit": "9774ed4fcc78313f9f218295f3d7e4decdadf2ae",
        "source_license": "CC-BY-NC-4.0",
    },
    "fast_byte_latent_transformer": {
        "paper": "arXiv:2605.08044v1",
        "pdf_sha256": "7957dd70eb7175a86bdcb1f8e8fca2f1444a348c0058dc924633f892d7efc8a2",
    },
    "mambabyte": {
        "paper": "arXiv:2401.13660v3",
        "pdf_sha256": "9d14c1a8ee92272ec7410547abb65d19847f74de600a254ac67359c1d63fdee5",
        "source_commit": "5e16f780bc331d1dc9430f4258dfb247d73aff44",
        "source_license": "Apache-2.0",
    },
    "learned_tokenization": {
        "paper": "arXiv:2602.13940v2",
        "pdf_sha256": "5a119799667018b9d8e3fe49edfc1b77c041cbc4aa856b70af77d7b57ff825ec",
        "source_commit": "8992678ece92f33eb572d6e5fd213ac7510a04c0",
        "source_license": "MIT",
    },
    "gputok": {
        "paper": "arXiv:2603.02597v1",
        "pdf_sha256": "2fe962a87a737e4f65cc22b2ee85f0b623c69fb9b2dda2961466e03ce5e20ac",
        "source_commit": "d08e0bf8135bd25553a5887b81176f2347013464",
        "source_license": "Apache-2.0",
        "scope": "measured BPE merge provider; UTF-8/chunk assembly remain host-side",
    },
}


def _latent_spec(*, mode: str = "autoregressive", patch_method: str = "stride",
                 backbone: str = "transformer", joint_diffusion: bool = False):
    patcher = BytePatchSpec(
        method=patch_method, stride=4, min_patch_bytes=1, max_patch_bytes=4,
        learned_threshold_q20=1 << 19, learned_window=1)
    needs_diffusion = mode in ("diffusion", "diffusion_verify") or joint_diffusion
    return ByteLatentSpec(
        context_bytes=32,
        patcher=patcher,
        local_width=16,
        global_width=16,
        local_heads=2,
        global_heads=2,
        local_encoder_layers=1,
        global_layers=1,
        local_decoder_layers=1,
        local_ff_width=32,
        global_ff_width=32,
        local_window_bytes=16,
        hash_ngram_sizes=(3,),
        hash_buckets=32,
        local_backbone=backbone,
        ssm_state_size=4,
        generation_mode=mode,
        diffusion=ByteDiffusionSpec(
            block_size=4, max_steps=2, unmask_strategy="confidence")
        if needs_diffusion else None,
        speculation=ByteSpeculationSpec(draft_bytes=3)
        if mode in ("self_speculative", "diffusion_verify") else None,
        learned_policy=LearnedPatchPolicySpec()
        if patch_method == "learned" else None,
    )


def _training_batch():
    inputs = torch.tensor([
        [97, 98, 99, 100, 101, 102, 103, 104],
        [104, 103, 102, 101, 100, 99, 98, 97],
    ], dtype=torch.long)
    targets = torch.tensor([
        [98, 99, 100, 101, 102, 103, 104, 105],
        [103, 102, 101, 100, 99, 98, 97, 96],
    ], dtype=torch.long)
    return inputs, targets


def _all_gradients_finite(model) -> bool:
    return all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
               for parameter in model.parameters())


def _parameter_parity(model, expected: int, label: str) -> None:
    actual = model.unique_parameter_count()
    if actual != expected:
        raise AssertionError(
            f"{label} parameter inventory diverged: hosted={actual}, exact={expected}")


def _must_refuse(call, fragment: str) -> None:
    try:
        call()
        raise AssertionError("malformed byte-model operation was accepted")
    except ValueError as exc:
        if fragment not in str(exc):
            raise


def _training_cases(inputs, targets) -> dict:
    torch.manual_seed(1729)
    latent_spec = _latent_spec()
    latent = HostedByteLatentModel(latent_spec)
    latent_cost = assess_byte_latent(latent_spec, byte_count=8, patch_count=2, batch=2)
    _parameter_parity(latent, latent_cost.parameter_elements, "BLT")
    latent_events = []
    latent_report = train_byte_latent_pretrain(
        latent, inputs, targets,
        StageTrainSpec("pretrain", steps=3, learning_rate=3.0e-3,
                       weight_decay=0.0, seed=1729), latent_events.append)
    if not latent_report.final_loss < latent_report.initial_loss:
        raise AssertionError("tiny BLT pretraining did not lower repeated-batch loss")
    if [event.step for event in latent_events] != [1, 2, 3]:
        raise AssertionError("BLT telemetry steps are malformed")

    mamba_spec = MambaByteSpec(
        context_bytes=32, width=16, layers=1, state_size=4,
        expansion=2, conv_width=4)
    torch.manual_seed(1730)
    mamba = HostedMambaByteModel(mamba_spec)
    mamba_cost = assess_mambabyte(mamba_spec, sequence_bytes=8)
    _parameter_parity(mamba, mamba_cost.parameter_elements, "MambaByte")
    mamba_report = train_mambabyte_pretrain(
        mamba, inputs, targets,
        StageTrainSpec("pretrain", steps=3, learning_rate=3.0e-3,
                       weight_decay=0.0, seed=1730))
    if not mamba_report.final_loss < mamba_report.initial_loss:
        raise AssertionError("tiny MambaByte pretraining did not lower repeated-batch loss")
    return {
        "blt": {
            "architecture_sha256": latent_spec.digest,
            "parameters": latent_cost.parameter_elements,
            "initial_loss": latent_report.initial_loss,
            "final_loss": latent_report.final_loss,
            "model_sha256": latent_report.model_sha256,
            "telemetry_events": len(latent_events),
        },
        "mambabyte": {
            "architecture_sha256": mamba_spec.digest,
            "parameters": mamba_cost.parameter_elements,
            "initial_loss": mamba_report.initial_loss,
            "final_loss": mamba_report.final_loss,
            "model_sha256": mamba_report.model_sha256,
            "recurrent_state_bytes": mamba_cost.recurrent_state_bytes,
        },
    }


def _objective_cases(inputs, targets) -> dict:
    torch.manual_seed(1731)
    diffusion_spec = _latent_spec(joint_diffusion=True)
    diffusion_model = HostedByteLatentModel(diffusion_spec)
    objective = diffusion_model.loss_components(inputs, targets, seed=1731)
    objective.total.backward()
    if not all(bool(torch.isfinite(value.detach())) for value in (
            objective.total, objective.autoregressive, objective.diffusion)) \
            or not _all_gradients_finite(diffusion_model) \
            or not float(objective.diffusion.detach()) > 0.0:
        raise AssertionError("joint AR/diffusion objective or gradients are invalid")

    torch.manual_seed(1732)
    learned_spec = _latent_spec(patch_method="learned")
    learned_model = HostedByteLatentModel(learned_spec)
    learned = learned_model.loss_components(inputs, targets, seed=1732)
    learned.total.backward()
    boundary_gradient = float(learned_model.boundary_weight.grad.abs().sum().item())
    if not _all_gradients_finite(learned_model) or boundary_gradient <= 0.0:
        raise AssertionError("score-function patch policy did not train its boundary weights")

    torch.manual_seed(1733)
    hybrid_spec = _latent_spec(backbone="selective_ssm")
    hybrid = HostedByteLatentModel(hybrid_spec)
    hybrid_cost = assess_byte_latent(hybrid_spec, byte_count=8, patch_count=2, batch=2)
    _parameter_parity(hybrid, hybrid_cost.parameter_elements, "SSM-local BLT")
    _logits, hybrid_loss = hybrid(inputs, targets)
    hybrid_loss.backward()
    if not _all_gradients_finite(hybrid):
        raise AssertionError("SSM-local BLT produced non-finite gradients")
    return {
        "joint_ar_diffusion": {
            "architecture_sha256": diffusion_spec.digest,
            "autoregressive_loss": float(objective.autoregressive.detach().item()),
            "diffusion_loss": float(objective.diffusion.detach().item()),
            "total_loss": float(objective.total.detach().item()),
        },
        "learned_patcher": {
            "architecture_sha256": learned_spec.digest,
            "policy_loss": float(learned.policy.detach().item()),
            "target_rate_loss": float(learned.target_rate.detach().item()),
            "boundary_gradient_l1": boundary_gradient,
            "patch_starts": [list(layout.starts) for layout in learned.layouts],
        },
        "ssm_local_blt": {
            "architecture_sha256": hybrid_spec.digest,
            "parameters": hybrid_cost.parameter_elements,
            "recurrent_state_bytes": hybrid_cost.recurrent_state_bytes,
            "loss": float(hybrid_loss.detach().item()),
        },
    }


def _causality_case() -> dict:
    first = torch.tensor([[97, 98, 99, 100, 101, 102, 103, 104]], dtype=torch.long)
    changed = torch.tensor([[97, 98, 99, 100, 201, 202, 203, 204]], dtype=torch.long)

    torch.manual_seed(1740)
    entropy = HostedByteLatentModel(_latent_spec(patch_method="entropy_global"))
    first_hidden = entropy.local_hidden(first)
    changed_hidden = entropy.local_hidden(changed)
    if not torch.equal(first_hidden[:, :4], changed_hidden[:, :4]):
        raise AssertionError("causal local byte encoder observed a future suffix")
    first_layout = entropy._entropy_layouts(first_hidden)[0]
    changed_layout = entropy._entropy_layouts(changed_hidden)[0]
    if tuple(start for start in first_layout.starts if start <= 4) != tuple(
            start for start in changed_layout.starts if start <= 4):
        raise AssertionError("entropy patch start observed its current/future byte")
    invalid_layout = PatchLayout(8, (0, 7), "entropy_global", "c" * 64)
    _must_refuse(lambda: entropy(first, layouts=(invalid_layout,)), "bounds")

    torch.manual_seed(1741)
    learned = HostedByteLatentModel(_latent_spec(patch_method="learned"))
    first_hidden = learned.local_hidden(first)
    changed_hidden = learned.local_hidden(changed)
    _layouts_a, logits_a, _samples_a, _mask_a = learned._learned_layouts(
        first_hidden, sample=False, seed=0)
    _layouts_b, logits_b, _samples_b, _mask_b = learned._learned_layouts(
        changed_hidden, sample=False, seed=0)
    if not torch.equal(logits_a[:, :5], logits_b[:, :5]):
        raise AssertionError("learned patch policy observed a byte at/beyond its boundary")

    torch.manual_seed(1742)
    mamba = HostedMambaByteModel(MambaByteSpec(
        context_bytes=32, width=16, layers=1, state_size=4))
    with torch.no_grad():
        mamba_a, _loss = mamba(first)
        mamba_b, _loss = mamba(changed)
    if not torch.equal(mamba_a[:, :4], mamba_b[:, :4]):
        raise AssertionError("MambaByte selective recurrence observed a future suffix")
    return {
        "local_prefix_hidden_exact": True,
        "entropy_boundary_prefix_exact": True,
        "learned_boundary_logits_prefix_exact": True,
        "mambabyte_prefix_logits_exact": True,
    }


def _generation_cases() -> dict:
    prompt = (97, 98, 99)
    torch.manual_seed(1734)
    speculative = HostedByteLatentModel(_latent_spec(mode="self_speculative"))
    _must_refuse(lambda: greedy_generate_byte_latent(
        speculative, prompt + (speculative.spec.vocabulary.mask_id,), 1), "sentinels")
    greedy = greedy_generate_byte_latent(speculative, prompt, 6)
    verified = self_speculative_generate(speculative, prompt, 6)
    if verified != greedy:
        raise AssertionError("BLT-S diverged from exact greedy target output")

    torch.manual_seed(1735)
    diffusion_verify = HostedByteLatentModel(_latent_spec(mode="diffusion_verify"))
    _must_refuse(lambda: self_speculative_generate(
        diffusion_verify, prompt, 1), "generation_mode")
    greedy_dv = greedy_generate_byte_latent(diffusion_verify, prompt, 6)
    verified_dv = diffusion_verify_generate(diffusion_verify, prompt, 6)
    if verified_dv != greedy_dv:
        raise AssertionError("BLT-DV diverged from exact greedy target output")

    torch.manual_seed(1736)
    diffusion = HostedByteLatentModel(_latent_spec(mode="diffusion"))
    direct_draft = diffusion_draft(diffusion, prompt, 2)
    if len(direct_draft) != 2 or diffusion.training is not True:
        raise AssertionError("direct diffusion draft lost length or caller model mode")
    generated = diffusion_generate(diffusion, prompt, 6)
    if len(generated) != len(prompt) + 6 \
            or diffusion.spec.vocabulary.mask_id in generated[len(prompt):]:
        raise AssertionError("BLT-D failed bounded mask-free generation")

    torch.manual_seed(1737)
    mamba = HostedMambaByteModel(MambaByteSpec(
        context_bytes=32, width=16, layers=1, state_size=4))
    mamba_generated = greedy_generate_mambabyte(mamba, prompt, 4)
    forbidden = {
        speculative.spec.vocabulary.bos_id,
        speculative.spec.vocabulary.pad_id,
        speculative.spec.vocabulary.mask_id,
    }
    if any(forbidden.intersection(values[len(prompt):]) for values in (
            verified, verified_dv, generated, mamba_generated)):
        raise AssertionError("byte generation emitted a non-output control sentinel")
    if speculative.training is not True or diffusion_verify.training is not True \
            or diffusion.training is not True or mamba.training is not True:
        raise AssertionError("byte generation did not restore the caller's model mode")
    return {
        "blt_s_exact_ids": list(verified),
        "blt_dv_exact_ids": list(verified_dv),
        "blt_d_ids": list(generated),
        "mambabyte_ids": list(mamba_generated),
    }


def _transplant_case() -> dict:
    torch.manual_seed(1738)
    model = HostedByteLatentModel(_latent_spec())
    targets = dict(model.state_dict())
    mapped_targets = sorted(
        name for name in targets if name.startswith("global_blocks.0."))
    source_state = {
        "source." + name: torch.full_like(targets[name], (index + 1) / 100.0)
        for index, name in enumerate(mapped_targets)
    }
    source_shapes = tuple(
        TensorShape(name, tuple(value.shape)) for name, value in source_state.items())
    target_shapes = byte_latent_tensor_shapes(model)
    if sum(shape.elements for shape in target_shapes) != model.unique_parameter_count():
        raise AssertionError("tied parameters were duplicated in transplant inventory")
    if len({shape.name for shape in target_shapes}) != len(target_shapes):
        raise AssertionError("transplant inventory contains duplicate parameter names")
    mappings = {name: "source." + name for name in mapped_targets}
    plan = plan_byteified_transplant(
        model.spec, source_sha256=tensor_mapping_digest(source_state),
        source_shapes=source_shapes, target_shapes=target_shapes, mappings=mappings)
    apply_byteified_transplant(source_state, model, plan)
    if any(not torch.equal(model.state_dict()[target], source_state[source])
           for target, source in plan.mappings):
        raise AssertionError("byteified global transplant did not copy exactly")
    groups = byteified_parameter_groups(model, plan, base_learning_rate=1.0e-3)
    rates = {group["name"]: group["lr"] for group in groups}
    if rates != {"byte-organs": 1.0e-3,
                 "transplanted-global": 1.0e-3 * plan.global_learning_rate_q20 / (1 << 20)}:
        raise AssertionError("byteified optimizer groups lost the reduced global rate")

    # Prove all tensors are validated before the first destination copy.
    bad_source = {name: value.clone() for name, value in source_state.items()}
    bad_source[sorted(bad_source)[-1]].view(-1)[0] = float("nan")
    bad_plan = plan_byteified_transplant(
        model.spec, source_sha256=tensor_mapping_digest(bad_source),
        source_shapes=source_shapes, target_shapes=target_shapes, mappings=mappings)
    before = tensor_mapping_digest(dict(model.state_dict()))
    try:
        apply_byteified_transplant(bad_source, model, bad_plan)
        raise AssertionError("non-finite transplant source was accepted")
    except ValueError as exc:
        if "non-finite" not in str(exc):
            raise
    if tensor_mapping_digest(dict(model.state_dict())) != before:
        raise AssertionError("failed byteified transplant partially mutated the model")

    tampered_plan = replace(plan, reused_elements=plan.reused_elements + 1)
    before = tensor_mapping_digest(dict(model.state_dict()))
    _must_refuse(lambda: apply_byteified_transplant(
        source_state, model, tampered_plan), "element counts")
    if tensor_mapping_digest(dict(model.state_dict())) != before:
        raise AssertionError("invalid transplant metadata mutated the model")

    freeze_plan = plan_byteified_transplant(
        model.spec, source_sha256=tensor_mapping_digest(source_state),
        source_shapes=source_shapes, target_shapes=target_shapes, mappings=mappings,
        freeze_global=True)
    frozen_groups = byteified_parameter_groups(
        model, freeze_plan, base_learning_rate=1.0e-3)
    if [group["name"] for group in frozen_groups] != ["byte-organs"]:
        raise AssertionError("freeze-global plan leaked global parameters to the optimizer")
    apply_byteified_transplant(source_state, model, freeze_plan)
    if any(parameter.requires_grad for name, parameter in model.named_parameters()
           if name.startswith("global_blocks.")):
        raise AssertionError("freeze-global transplant left a global parameter trainable")
    return {
        "plan_sha256": plan.digest,
        "mapped_tensors": len(plan.mappings),
        "reused_elements": plan.reused_elements,
        "global_learning_rate_q20": plan.global_learning_rate_q20,
        "failure_atomic": True,
        "metadata_revalidated": True,
        "freeze_global_enforced": True,
    }


def _lowering_and_ingest_case() -> dict:
    spec = _latent_spec(mode="diffusion_verify")
    layout = spec.patcher.layout(12)
    lowered = lower_byte_latent(
        spec, layout, target=TargetProfile.x86_avx2())
    repeated = lower_byte_latent(
        spec, layout, target=TargetProfile.x86_avx2())
    if lowered.artifact_sha256 != repeated.artifact_sha256 \
            or lowered.streampack_data != repeated.streampack_data \
            or not lowered.streampack.provenance_ok():
        raise AssertionError("byte-model lowering is non-deterministic or unproven")
    profile = ByteIngestProfile(
        hashlib.sha256(b"bounded-byte-ingest-profile").hexdigest(),
        host_ns_per_byte_q20=10 << 20,
        device_ns_per_byte_q20=1 << 20,
        device_launch_ns=100,
        device_pool_bytes=4096,
        max_chunk_bytes=1024,
        exact_roundtrip=True)
    small = choose_byte_ingest(profile, 10)
    large = choose_byte_ingest(profile, 1024)
    if small.provider != "host" or large.provider != "device":
        raise AssertionError("measured byte-ingest crossover selection regressed")
    return {
        "artifact_sha256": lowered.artifact_sha256,
        "streampack_bytes": len(lowered.streampack_data),
        "claims": len(lowered.realization.steps),
        "small_provider": small.provider,
        "large_provider": large.provider,
    }


def _run_once() -> dict:
    inputs, targets = _training_batch()
    unicode_sequence = encode_raw_sequence("naïve🙂e\u0301")
    return {
        "schema": "bcir.byte_native_model_gate.v1",
        "research_provenance": RESEARCH_PROVENANCE,
        "input_semantics": {
            "model_units": "raw-octets",
            "normalization": "none",
            "utf8_sha256": unicode_sequence.sha256,
            "unicode_scalar_references": len(unicode_sequence.references),
            "character_references_are_diagnostic_only": True,
        },
        "tiny_training": _training_cases(inputs, targets),
        "objectives": _objective_cases(inputs, targets),
        "causality": _causality_case(),
        "generation": _generation_cases(),
        "byteified_transplant": _transplant_case(),
        "lowering_and_ingest": _lowering_and_ingest_case(),
        "large_training_performed": False,
        "gpu_used": False,
        "external_source_weights_or_datasets_imported": False,
        "cuda_provider_implemented": False,
    }


def run_gate() -> dict:
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        first = _run_once()
        second = _run_once()
        if first != second:
            raise AssertionError("byte-native model gate is not deterministic on this host")
        return first
    finally:
        torch.set_rng_state(previous_rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(previous_threads)


def _atomic_report(output_dir: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "report.json"
    payload = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    handle, temporary = tempfile.mkstemp(
        prefix=".byte-model-report-", suffix=".part", dir=output_dir)
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
