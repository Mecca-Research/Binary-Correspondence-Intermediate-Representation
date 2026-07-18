#!/usr/bin/env python3
"""Deterministic tiny train -> search -> verified-plan gate for BCIR hardware RL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch

from bcir.frontends.models.assessment import (
    BankEnvelope,
    HardwareEnvelope,
    KernelBenchmark,
    LinkEnvelope,
    ModelWorkloadSpec,
    assess_model,
)
from bcir.frontends.models.execution_plan import lower_model_execution
from bcir.frontends.models.inventory import build_tensor_inventory
from bcir.hosted.training import HardwarePolicySpec, HardwarePolicyTrainSpec
from bcir.hosted.training.hardware_policy import (
    hardware_policy_priors_q20,
    load_hardware_policy,
    train_hardware_policy,
)
from bcir.kbcir.hardware_rl import (
    Q20,
    HardwareEpisode,
    HardwareOutcome,
    HardwareRewardPolicy,
    TelemetryToken,
    bounded_mcts,
    certify_hardware_promotion,
    encode_candidate_features,
    encode_hardware_topology,
    hardware_context_digest,
)
from bcir.kbcir.static_memory import plan_static_memory, verify_static_memory_plan
from bcir.telemetry import DataDNA


def _write_toy_model(directory: Path):
    config = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32,
              "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "max_position_embeddings": 64, "tie_word_embeddings": False}
    tensors = {"model.embed_tokens.weight": (32, 8), "model.norm.weight": (8,),
               "lm_head.weight": (32, 8)}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": (8,),
            prefix + "self_attn.q_proj.weight": (8, 8),
            prefix + "self_attn.k_proj.weight": (4, 8),
            prefix + "self_attn.v_proj.weight": (4, 8),
            prefix + "self_attn.o_proj.weight": (8, 8),
            prefix + "post_attention_layernorm.weight": (8,),
            prefix + "mlp.gate_proj.weight": (16, 8),
            prefix + "mlp.up_proj.weight": (16, 8),
            prefix + "mlp.down_proj.weight": (8, 16)})
    offset = 0; header = {}
    for name, shape in sorted(tensors.items()):
        count = 1
        for dimension in shape: count *= dimension
        nbytes = count * 4
        header[name] = {"dtype": "F32", "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = directory / "model.safetensors"
    with path.open("xb") as stream:
        stream.write(struct.pack("<Q", len(encoded))); stream.write(encoded)
        stream.write(b"\0" * offset)
    (directory / "config.json").write_text(
        json.dumps(config, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return path, config


def _malformed_report_artifact(source: Path, root: Path) -> Path:
    staging = root / "staging"
    root.mkdir()
    shutil.copytree(source, staging)
    report_path = staging / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["unknown"] = True
    report_path.write_text(json.dumps(report, sort_keys=True, separators=(",", ":")),
                           encoding="utf-8")
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_data = report_path.read_bytes()
    manifest["files"]["report.json"] = {
        "bytes": len(report_data), "sha256": hashlib.sha256(report_data).hexdigest()}
    manifest_data = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest_path.write_bytes(manifest_data)
    destination = root / hashlib.sha256(manifest_data).hexdigest()
    os.replace(staging, destination)
    return destination


def _hardware():
    banks = (
        BankEnvelope("ram", "RAM", "host", 1 << 28, 1 << 28,
                     20_000_000_000, 20_000_000_000),
        BankEnvelope("vram", "VRAM", "gpu", 1 << 28, 1 << 28,
                     100_000_000_000, 100_000_000_000,
                     alignment=256, native_tile=16))
    links = (LinkEnvelope("ram", "vram", 8_000_000_000, 5000),
             LinkEnvelope("vram", "ram", 8_000_000_000, 5000))
    benchmarks = []
    for channel, multiplier in (("host", 4), ("gpu", 1)):
        for operation, tokens, operations in (("prefill", 8, 100_000),
                                               ("decode", 1, 20_000)):
            median = 1000 * multiplier
            benchmarks.append(KernelBenchmark(
                operation, channel, "source", median - 100, median, median + 100,
                tokens, operations, 4096, 2048, 5))
    return HardwareEnvelope("tiny-simulated-tower", "x86_64", banks, links,
                            tuple(benchmarks))


def _workload():
    return ModelWorkloadSpec(
        mode="inference", batch_size=1, sessions=2, prompt_tokens=4,
        context_tokens=8, max_new_tokens=4, activation_dtype="f32", kv_dtype="bf16",
        gradient_dtype="f32", optimizer="none", formats=("source",))


def _episodes(report, hardware, workload, candidates, policy):
    scenarios = (
        (10, 10, 10, False), (15, 20, 20, False),
        (25, 80, 35, False), (80, 25, 65, True),
        (70, 30, 55, True), (75, 70, 60, True))
    episodes = []
    candidate_rows = {row.candidate_id: row for row in report.candidates}
    for episode_index, (thermal, bandwidth, register, throttled) in enumerate(scenarios):
        telemetry = []
        for position in range(4):
            event = DataDNA(
                f"episode-{episode_index}", position + 1,
                cycles=1000 + position * 100, bytes=4096 + position * 256,
                misses=min(100, bandwidth // 2 + position), thermal=thermal,
                utilization=60 + position,
                provenance=f"simulated-fixture-{episode_index}")
            telemetry.append(TelemetryToken.from_data_dna(
                event, sequence=position, register_pressure=min(100, register + position),
                bandwidth_pressure=bandwidth, throttled=throttled))
        context = hardware_context_digest(
            report.digest, hardware.digest, workload.digest, telemetry)
        outcomes = []
        for feature in candidates:
            candidate = candidate_rows[feature.candidate_id]
            primary_gpu = candidate.compute_banks[0] == "vram"
            stream = feature.kind == "layer-stream"
            split = feature.kind == "cpu-gpu-split"
            if feature.kind == "resident":
                base = 400 if primary_gpu else 700
            elif split:
                base = 930 if primary_gpu else 950
            else:
                base = 800 if primary_gpu else 1050
            thermal_penalty = thermal * (9 if primary_gpu else 2)
            bandwidth_penalty = bandwidth * (8 if stream else 1)
            latency = (base + thermal_penalty + bandwidth_penalty) * 1000
            energy = ((320 if primary_gpu else 450)
                      + thermal * (5 if primary_gpu else 1) + (120 if stream else 0))
            contention = min(100, register + (20 if primary_gpu else 5))
            miss = min(100, bandwidth + (15 if stream else 0))
            outcome_thermal = min(100, thermal + (10 if primary_gpu else 0))
            source = hashlib.sha256(
                f"{context}:{feature.candidate_id}:{latency}:{energy}".encode()).hexdigest()
            outcomes.append(HardwareOutcome(
                context, feature.candidate_id, latency, energy, contention, miss,
                bandwidth, outcome_thermal, throttled and primary_gpu, True,
                "simulated", 7, source))
        episodes.append(HardwareEpisode(
            report.digest, hardware.digest, workload.digest, tuple(telemetry),
            tuple(outcomes), policy.digest))
    return tuple(episodes)


def run(output_dir: Path) -> dict:
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try: torch.set_num_interop_threads(1)
        except RuntimeError: pass
    if output_dir.exists(): shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    model_source = output_dir / "synthetic-model"; model_source.mkdir()
    model_path, config = _write_toy_model(model_source)
    inventory = build_tensor_inventory([str(model_path)], config)
    hardware = _hardware(); workload = _workload()
    report = assess_model(inventory, hardware, workload)
    candidates = encode_candidate_features(report)
    topology = encode_hardware_topology(hardware)
    policy = HardwareRewardPolicy()
    episodes = _episodes(report, hardware, workload, candidates, policy)
    model_spec = HardwarePolicySpec(hidden_dim=32, heads=4, transformer_layers=1,
                                    graph_layers=2, max_telemetry_tokens=8)
    train_spec = HardwarePolicyTrainSpec(reward_steps=96, dpo_steps=96, ppo_steps=96)
    events = []
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_only_before = torch.is_deterministic_algorithms_warn_only_enabled()
    rng_before = torch.get_rng_state().clone()
    first = train_hardware_policy(
        model_spec, train_spec, topology, candidates, episodes, policy,
        output_dir / "artifacts-a", telemetry_sink=events.append)
    second = train_hardware_policy(
        model_spec, train_spec, topology, candidates, episodes, policy,
        output_dir / "artifacts-b")
    if (torch.are_deterministic_algorithms_enabled() != deterministic_before
            or torch.is_deterministic_algorithms_warn_only_enabled() != warn_only_before
            or not torch.equal(torch.get_rng_state(), rng_before)):
        raise AssertionError("hardware-policy training leaked caller global state")
    if first.report != second.report or first.artifact.manifest_sha256 != second.artifact.manifest_sha256:
        raise AssertionError("hardware-policy training/export is not deterministic on one host")
    if first.report.final_reward_loss >= first.report.initial_reward_loss:
        raise AssertionError("tiny hardware reward model did not reduce loss")
    if first.report.final_top1_correct < first.report.initial_top1_correct \
            or first.report.final_top1_correct < len(episodes) - 1:
        raise AssertionError("tiny hardware policy did not learn the metric preference")
    rng_before_load = torch.get_rng_state().clone()
    loaded, loaded_spec, loaded_artifact = load_hardware_policy(first.artifact.directory)
    if not torch.equal(torch.get_rng_state(), rng_before_load):
        raise AssertionError("hardware-policy loading leaked caller RNG state")
    if loaded_spec != model_spec or loaded_artifact.model_sha256 != first.report.model_sha256:
        raise AssertionError("hardware-policy safe artifact did not round-trip")
    corrupt = output_dir / "corrupt" / first.artifact.manifest_sha256
    corrupt.parent.mkdir(); shutil.copytree(first.artifact.directory, corrupt)
    with (corrupt / "hardware-policy.safetensors").open("r+b") as stream:
        stream.seek(-1, os.SEEK_END); value = stream.read(1)
        stream.seek(-1, os.SEEK_END); stream.write(bytes((value[0] ^ 1,)))
    try:
        load_hardware_policy(corrupt)
        raise AssertionError("corrupted hardware-policy tensors must fail integrity")
    except ValueError as exc:
        if "integrity" not in str(exc): raise
    malformed = _malformed_report_artifact(
        Path(first.artifact.directory), output_dir / "malformed-report")
    try:
        load_hardware_policy(malformed)
        raise AssertionError("re-hashed malformed hardware-policy reports must fail schema checks")
    except ValueError as exc:
        if "report" not in str(exc): raise

    episode = episodes[0]
    loaded.train()
    priors = hardware_policy_priors_q20(loaded, topology, candidates, episode.telemetry)
    if not loaded.training:
        raise AssertionError("hardware-policy inference changed caller model mode")
    loaded.eval()
    outcome_by_id = {row.candidate_id: row for row in episode.outcomes}
    scores = {candidate_id: policy.score(outcome) for candidate_id, outcome in outcome_by_id.items()}
    worst = max(scores.values())
    search = bounded_mcts(
        scores, priors, lambda candidate_id: (worst - scores[candidate_id]) * Q20,
        simulations=64)
    oracle = min(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id))
    if search.candidate_id != oracle:
        raise AssertionError("bounded search did not recover the exact metric oracle")
    selected = next(row for row in report.candidates if row.candidate_id == oracle)
    lowered = lower_model_execution(report, selected, inventory, hardware, workload)
    memory = plan_static_memory(lowered.module, lowered.resource_banks, hardware)
    if verify_static_memory_plan(memory, lowered.module, lowered.resource_banks, hardware):
        raise AssertionError("static memory plan did not independently verify")
    baseline_id = max(scores, key=lambda candidate_id: (scores[candidate_id], candidate_id))
    try:
        certify_hardware_promotion(
            lowered, memory, hardware, outcome_by_id[oracle], outcome_by_id[baseline_id],
            policy, episode=episode, generation=2, quiescent=True)
        raise AssertionError("simulated micro-profile must not certify live promotion")
    except ValueError as exc:
        if "simulated" not in str(exc): raise

    result = {
        "schema": "bcir.hardware_rl_gate.v1",
        "evidence": "simulated-bounded-fixture",
        "hardware_digest": hardware.digest,
        "inventory_digest": inventory.header_digest,
        "report_digest": report.digest,
        "model_spec_sha256": model_spec.digest,
        "train_spec_sha256": train_spec.digest,
        "input_sha256": first.report.input_sha256,
        "model_sha256": first.report.model_sha256,
        "manifest_sha256": first.artifact.manifest_sha256,
        "deterministic_repeat": True,
        "training": {
            "episodes": first.report.episodes, "candidates": first.report.candidates,
            "steps": first.report.total_steps,
            "initial_reward_loss": first.report.initial_reward_loss,
            "final_reward_loss": first.report.final_reward_loss,
            "initial_top1_correct": first.report.initial_top1_correct,
            "final_top1_correct": first.report.final_top1_correct,
            "events": len(events)},
        "search": {"simulations": search.simulations, "selected": search.candidate_id,
                   "exact_oracle": oracle,
                   "visits": {row.candidate_id: row.visits for row in search.visits}},
        "lowering": {"plan_sha256": lowered.artifact.digest,
                     "streampack_sha256": lowered.artifact.streampack_sha256,
                     "static_memory_sha256": memory.digest,
                     "static_extent_bytes": sum(row.extent_bytes for row in memory.banks),
                     "naive_aligned_bytes": sum(row.naive_bytes for row in memory.banks)},
        "promotion": "withheld:simulated-evidence",
        "large_training_performed": False,
        "live_hardware_profiling_performed": False}
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")),
                           encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="build/hardware-rl-gate")
    arguments = parser.parse_args(argv)
    result = run(Path(arguments.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
