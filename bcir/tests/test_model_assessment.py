"""Header-only model assessment, placement, and verified StreamPack regressions."""

import builtins
import contextlib
import io
import json
import os
import struct
import tempfile
from dataclasses import replace

from bcir.frontends.models.assess_tool import main as assess_main
from bcir.frontends.models.assessment import (
    BankEnvelope,
    HardwareEnvelope,
    KernelBenchmark,
    LinkEnvelope,
    ModelCostReport,
    ModelExecutionPlan,
    ModelWorkloadSpec,
    PredictionInterval,
    assess_model,
    format_sizes,
    select_execution_candidate,
)
from bcir.frontends.models.execution_plan import lower_model_execution
from bcir.frontends.models.inventory import TensorInventory, build_tensor_inventory
from bcir.frontends.models.model_microbench import run_bounded_model_microbench


def _write_shard(path, tensors, *, sparse=False):
    header = {}; offset = 0
    bits = {"F4": 4, "F6_E2M3": 6, "F6_E3M2": 6, "F32": 32,
            "BF16": 16, "F16": 16, "I8": 8, "C64": 64}
    for name, (dtype, shape) in sorted(tensors.items()):
        count = 1
        for dim in shape:
            count *= dim
        nbytes = (count * bits[dtype] + 7) // 8
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded))); stream.write(encoded)
        if offset:
            if sparse:
                stream.seek(offset - 1, os.SEEK_CUR); stream.write(b"\0")
            else:
                stream.write(b"\0" * offset)
    return 8 + len(encoded), offset


def _toy(directory):
    config = {"architectures": ["LlamaForCausalLM"], "vocab_size": 32,
              "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
              "num_attention_heads": 4, "num_key_value_heads": 2,
              "max_position_embeddings": 64, "tie_word_embeddings": False}
    tensors = {"model.embed_tokens.weight": ("F32", (32, 8)),
               "model.norm.weight": ("F32", (8,)),
               "lm_head.weight": ("F32", (32, 8))}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": ("F32", (8,)),
            prefix + "self_attn.q_proj.weight": ("F32", (8, 8)),
            prefix + "self_attn.k_proj.weight": ("F32", (4, 8)),
            prefix + "self_attn.v_proj.weight": ("F32", (4, 8)),
            prefix + "self_attn.o_proj.weight": ("F32", (8, 8)),
            prefix + "post_attention_layernorm.weight": ("F32", (8,)),
            prefix + "mlp.gate_proj.weight": ("F32", (16, 8)),
            prefix + "mlp.up_proj.weight": ("F32", (16, 8)),
            prefix + "mlp.down_proj.weight": ("F32", (8, 16)),
        })
    path = os.path.join(directory, "model.safetensors")
    _write_shard(path, tensors)
    with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as stream:
        json.dump(config, stream)
    return path, config


def _hardware(*, host_capacity=1 << 28, device_capacity=1 << 28):
    banks = (
        BankEnvelope("ram", "RAM", "host", host_capacity, host_capacity,
                     20_000_000_000, 20_000_000_000),
        BankEnvelope("vram", "VRAM", "gpu", device_capacity, device_capacity,
                     100_000_000_000, 100_000_000_000, alignment=256, native_tile=16),
    )
    links = (LinkEnvelope("ram", "vram", 8_000_000_000, 5000),
             LinkEnvelope("vram", "ram", 8_000_000_000, 5000))
    benchmarks = []
    for channel, multiplier in (("host", 4), ("gpu", 1)):
        for fmt, fmt_scale in (("source", 2), ("bcirq8-group32", 1)):
            for operation, tokens, operations in (("prefill", 8, 100_000),
                                                   ("decode", 1, 20_000)):
                median = 1000 * multiplier * fmt_scale
                benchmarks.append(KernelBenchmark(operation, channel, fmt,
                                                  median - 100, median, median + 100,
                                                  tokens, operations, 4096, 2048, 5))
    return HardwareEnvelope("synthetic-tower", "x86_64", banks, links, tuple(benchmarks))


def _workload(**changes):
    values = {"mode": "inference", "batch_size": 1, "sessions": 2,
              "prompt_tokens": 4, "context_tokens": 8, "max_new_tokens": 4,
              "activation_dtype": "f32", "kv_dtype": "bf16", "gradient_dtype": "f32",
              "optimizer": "none", "lora_rank": 0, "checkpoint_segments": 1,
              "formats": ("source", "bcirq8-group32")}
    values.update(changes)
    return ModelWorkloadSpec(**values)


def test_inventory_is_header_only_and_payload_independent():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory)
        with open(path, "rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
        limit = 8 + header_length
        original_open = builtins.open

        class Guard:
            def __init__(self, stream):
                self.stream = stream; self.read_bytes = 0

            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, size=-1):
                if size < 0 or self.read_bytes + size > limit:
                    raise AssertionError("weight payload read attempted")
                data = self.stream.read(size); self.read_bytes += len(data); return data

        def guarded_open(value, mode="r", *args, **kwargs):
            stream = original_open(value, mode, *args, **kwargs)
            return Guard(stream) if os.fspath(value) == path and mode == "rb" else stream

        builtins.open = guarded_open
        try:
            first = build_tensor_inventory([path], config)
        finally:
            builtins.open = original_open
        with open(path, "r+b") as stream:
            stream.seek(-1, os.SEEK_END); stream.write(b"\x7f")
        second = build_tensor_inventory([path], config)
        assert first == second and first.header_digest == second.header_digest
        assert first.parameter_count == 1704 and first.payload_bytes > 0
        assert TensorInventory.from_json(first.to_json()) == first


def test_synthetic_large_model_header_uses_constant_read_volume():
    # A virtual 1-TiB shard proves accounting is based on a bounded header read. No
    # giant sparse file is created (important on Windows/CI filesystems).
    from bcir.frontends.models import manifest as manifest_module
    payload_bytes = 1 << 40
    header = {"model.embed_tokens.weight": {
        "dtype": "I8", "shape": [payload_bytes], "data_offsets": [0, payload_bytes]}}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<Q", len(encoded)) + encoded

    class VirtualShard:
        def __init__(self): self.position = 0; self.bytes_read = 0
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def fileno(self): return 4242
        def read(self, size=-1):
            if size < 0:
                raise AssertionError("unbounded payload read attempted")
            data = prefix[self.position:self.position + size]
            self.position += len(data); self.bytes_read += len(data)
            return data

    original_open = getattr(manifest_module, "open", None)
    original_fstat = manifest_module.os.fstat
    shard = VirtualShard()
    manifest_module.open = lambda *_args, **_kwargs: shard
    manifest_module.os.fstat = lambda _fd: type(
        "Stat", (), {"st_size": len(prefix) + payload_bytes})()
    try:
        config = {"architectures": ["SyntheticLarge"], "vocab_size": 1 << 20,
                  "hidden_size": 1024, "intermediate_size": 4096,
                  "num_hidden_layers": 24, "num_attention_heads": 16,
                  "num_key_value_heads": 4, "max_position_embeddings": 4096,
                  "tie_word_embeddings": True}
        inventory = build_tensor_inventory(["synthetic-large.safetensors"], config)
    finally:
        if original_open is None:
            del manifest_module.open
        else:
            manifest_module.open = original_open
        manifest_module.os.fstat = original_fstat
    assert inventory.payload_bytes == payload_bytes
    assert inventory.header_bytes == len(prefix) and shard.bytes_read == len(prefix)


def test_pinned_safetensors_subbyte_inventory_is_exact_and_aligned():
    config = {"architectures": ["InventoryOnly"], "vocab_size": 1,
              "hidden_size": 1, "intermediate_size": 1, "num_hidden_layers": 1,
              "num_attention_heads": 1, "num_key_value_heads": 1,
              "max_position_embeddings": 1, "tie_word_embeddings": True}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "official.safetensors")
        _header, payload = _write_shard(path, {
            "f4": ("F4", (2,)), "f6_e2m3": ("F6_E2M3", (4,)),
            "f6_e3m2": ("F6_E3M2", (4,)), "complex": ("C64", (1,))})
        inventory = build_tensor_inventory([path], config)
        assert payload == inventory.payload_bytes == 15

        misaligned = os.path.join(directory, "misaligned.safetensors")
        _write_shard(misaligned, {"bad": ("F4", (1,))})
        try:
            build_tensor_inventory([misaligned], config)
            raise AssertionError("a half-byte Safetensors span was accepted")
        except ValueError as exc:
            assert "byte-aligned" in str(exc)


def test_synthetic_multi_billion_parameter_header_lowers_a_streaming_plan():
    from bcir.frontends.models import manifest as manifest_module
    d, f, vocab, layers, heads, kv_heads = 8192, 28672, 128256, 4, 64, 8
    kvd = d * kv_heads // heads
    tensors = {"model.embed_tokens.weight": ("BF16", (vocab, d)),
               "model.norm.weight": ("BF16", (d,))}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        tensors.update({
            prefix + "input_layernorm.weight": ("BF16", (d,)),
            prefix + "self_attn.q_proj.weight": ("BF16", (d, d)),
            prefix + "self_attn.k_proj.weight": ("BF16", (kvd, d)),
            prefix + "self_attn.v_proj.weight": ("BF16", (kvd, d)),
            prefix + "self_attn.o_proj.weight": ("BF16", (d, d)),
            prefix + "post_attention_layernorm.weight": ("BF16", (d,)),
            prefix + "mlp.gate_proj.weight": ("BF16", (f, d)),
            prefix + "mlp.up_proj.weight": ("BF16", (f, d)),
            prefix + "mlp.down_proj.weight": ("BF16", (d, f)),
        })
    offset = 0; header = {}
    for name, (dtype, shape) in sorted(tensors.items()):
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * 2
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    prefix = struct.pack("<Q", len(encoded)) + encoded

    class VirtualShard:
        def __init__(self): self.position = 0
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def fileno(self): return 4343
        def read(self, size=-1):
            if size < 0:
                raise AssertionError("unbounded payload read attempted")
            data = prefix[self.position:self.position + size]
            self.position += len(data); return data

    original_open = getattr(manifest_module, "open", None)
    original_fstat = manifest_module.os.fstat
    manifest_module.open = lambda *_args, **_kwargs: VirtualShard()
    manifest_module.os.fstat = lambda _fd: type(
        "Stat", (), {"st_size": len(prefix) + offset})()
    try:
        config = {"architectures": ["LlamaForCausalLM"], "vocab_size": vocab,
                  "hidden_size": d, "intermediate_size": f, "num_hidden_layers": layers,
                  "num_attention_heads": heads, "num_key_value_heads": kv_heads,
                  "max_position_embeddings": 4096, "tie_word_embeddings": True}
        inventory = build_tensor_inventory(["synthetic-4.5b.safetensors"], config)
    finally:
        if original_open is None:
            del manifest_module.open
        else:
            manifest_module.open = original_open
        manifest_module.os.fstat = original_fstat
    assert inventory.parameter_count > 4_000_000_000
    hardware = _hardware(host_capacity=16 << 30, device_capacity=6 << 30)
    workload = _workload(sessions=1, prompt_tokens=16, context_tokens=128,
                         max_new_tokens=16, formats=("source",))
    report = assess_model(inventory, hardware, workload)
    resident = next(row for row in report.candidates
                    if row.candidate_id == "source:resident:vram")
    streamed = next(row for row in report.candidates
                    if row.candidate_id == "source:stream:ram->vram")
    assert not resident.feasible and streamed.feasible
    lowered = lower_model_execution(report, streamed, inventory, hardware, workload)
    assert len(lowered.streampack_data) < 16 * 1024
    assert lowered.artifact.classification == "exact"


def test_exact_format_kv_training_and_per_bank_costs():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        formats = format_sizes(inventory, ("source", "bcirq8-group32"))
        source, q8 = formats
        assert source.file_bytes == os.path.getsize(path)
        assert source.payload_bytes + source.metadata_bytes == source.file_bytes
        assert q8.executable and q8.classification == "quantized"
        assert q8.payload_bytes + q8.metadata_bytes == q8.file_bytes < source.file_bytes

        inference = assess_model(inventory, _hardware(), _workload(formats=("source",)))
        expected_kv = 2 * 8 * 2 * 2 * 2 * 2 * 2  # sessions, context, layers, K/V, kv heads, head dim, BF16
        assert inference.kv_cache_bytes == expected_kv
        workload = _workload(mode="full-finetune", optimizer="adamw",
                             activation_dtype="f16", formats=("source", "bcirq8-group32"))
        report = assess_model(inventory, _hardware(), workload)
        assert report.kv_cache_bytes == 0
        # Fused training workspace: token scratch + all-token logits + saved
        # checkpoints. Inference's one-token logits contract must not leak here.
        assert report.activation_workspace_bytes == 1792
        state = report.training_state; params = inventory.parameter_count
        assert state.trainable_parameters == params
        assert state.trainable_tensor_count == len(inventory.tensors)
        assert state.gradient_bytes == 4 * params
        assert state.optimizer_step_bytes == 4 * len(inventory.tensors)
        assert state.optimizer_bytes == 8 * params + state.optimizer_step_bytes
        assert state.master_weight_bytes == 4 * params and state.adapter_weight_bytes == 0
        assert all(not row.feasible and "full-weight training" in " ".join(row.blockers)
                   for row in report.candidates if row.weight_format == "bcirq8-group32")
        assert all(peak.peak_bytes == peak.weights_bytes + peak.kv_bytes
                   + peak.activation_bytes + peak.training_bytes + peak.staging_bytes
                   for candidate in report.candidates for peak in candidate.peaks)


def test_lora_state_counts_only_target_matrices():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        report = assess_model(inventory, _hardware(), _workload(
            mode="lora", optimizer="adamw", lora_rank=2, formats=("source",)))
        expected = 0
        targets = {"attention.q", "attention.k", "attention.v", "attention.o",
                   "mlp.gate", "mlp.up", "mlp.down"}
        for tensor in inventory.tensors:
            if tensor.role in targets:
                expected += 2 * sum(tensor.shape)
        assert report.training_state.trainable_parameters == expected
        assert report.training_state.trainable_tensor_count == 2 * 7 * 2
        assert report.training_state.adapter_weight_bytes == expected * 4


def test_candidates_include_resident_streaming_split_and_intervals():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        report = assess_model(inventory, _hardware(), _workload())
        assert {row.kind for row in report.candidates} == {
            "resident", "layer-stream", "cpu-gpu-split"}
        assert any(row.weight_format == "source" and row.classification == "exact"
                   for row in report.candidates)
        assert any(row.weight_format == "bcirq8-group32" and row.classification == "quantized"
                   for row in report.candidates)
        for candidate in report.candidates:
            assert candidate.predictions and candidate.feasible
            for interval in candidate.predictions:
                assert interval.lower_ns <= interval.median_ns <= interval.upper_ns
                assert interval.tokens_per_second_lower_q16 \
                    <= interval.tokens_per_second_median_q16 \
                    <= interval.tokens_per_second_upper_q16
        resident = next(row for row in report.candidates
                        if row.candidate_id == "source:resident:vram")
        streamed = next(row for row in report.candidates
                        if row.candidate_id == "source:stream:ram->vram")
        assert next(row.median_ns for row in streamed.predictions if row.operation == "prefill") \
            > next(row.median_ns for row in resident.predictions if row.operation == "prefill")
        source_bytes = next(row.file_bytes for row in report.formats if row.name == "source")
        assert resident.peaks[0].weights_bytes == source_bytes
        for candidate in report.candidates:
            if candidate.kind == "cpu-gpu-split" and candidate.weight_format == "source":
                assert sum(peak.weights_bytes for peak in candidate.peaks) == source_bytes


def test_candidate_selection_minimizes_complete_workload_latency():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        report = assess_model(inventory, _hardware(), _workload(formats=("source",)))
        left = next(row for row in report.candidates
                    if row.candidate_id == "source:resident:ram")
        right = next(row for row in report.candidates
                     if row.candidate_id == "source:resident:vram")

        def interval(operation, latency):
            return PredictionInterval(operation, latency, latency, latency, 1,
                                      1, 1, 1, "0" * 64)

        balanced = replace(left, candidate_id="balanced-worst-stage",
                           predictions=(interval("prefill", 100), interval("decode", 100)))
        lower_total = replace(right, candidate_id="lower-complete-workload",
                              predictions=(interval("prefill", 1), interval("decode", 101)))
        selection_report = replace(report, candidates=(balanced, lower_total))
        assert select_execution_candidate(selection_report) == lower_total


def test_stream_lowering_emits_and_verifies_every_action_class_deterministically():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        hardware = _hardware(); workload = _workload(formats=("source",))
        report = assess_model(inventory, hardware, workload)
        assert ModelCostReport.from_json(report.to_json()) == report
        candidate = next(row for row in report.candidates
                         if row.kind == "layer-stream" and row.backing_bank == "ram")
        first = lower_model_execution(report, candidate, inventory, hardware, workload)
        second = lower_model_execution(report, candidate, inventory, hardware, workload)
        assert {row.kind for row in first.artifact.actions} == {
            "move", "prefetch", "compute", "barrier", "evict"}
        assert {row.stage for row in first.artifact.actions} == {"prefill", "decode"}
        assert sum(row.nbytes * row.repeats for row in first.artifact.actions
                   if row.kind == "move" and row.stage == "prefill") \
            == candidate.prefill_transfer_bytes
        assert sum(row.nbytes * row.repeats for row in first.artifact.actions
                   if row.kind == "move" and row.stage == "decode") \
            == candidate.decode_transfer_bytes
        assert first.artifact == second.artifact
        assert ModelExecutionPlan.from_json(first.artifact.to_json()) == first.artifact
        assert first.streampack_data == second.streampack_data
        assert first.artifact.digest == second.artifact.digest
        assert first.streampack.provenance_ok()
        stage_resources = sorted(resource.name for resource in first.module.resources.values()
                                 if resource.name.startswith("layer-stage-"))
        assert stage_resources == ["layer-stage-0", "layer-stage-1"]
        claims = [phase.claims[0] for phase in first.module.phases]
        for action, claim in zip(first.artifact.actions, claims):
            if action.kind == "compute":
                assert claim.count == action.operations
            elif action.kind in ("move", "prefetch"):
                assert claim.count == action.nbytes
        assert {row.repeats for row in first.artifact.actions if row.stage == "prefill"} == {1}
        assert {row.repeats for row in first.artifact.actions if row.stage == "decode"} \
            == {workload.max_new_tokens}


def test_cpu_gpu_split_lowering_accounts_each_boundary_stage_exactly():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        hardware = _hardware(); workload = _workload(formats=("source",))
        report = assess_model(inventory, hardware, workload)
        candidate = next(row for row in report.candidates
                         if row.candidate_id == "source:split:ram:1:vram")
        lowered = lower_model_execution(report, candidate, inventory, hardware, workload)
        moves = [row for row in lowered.artifact.actions if row.kind == "move"]
        assert [(row.stage, row.repeats) for row in moves] == [
            ("prefill", 1), ("decode", workload.max_new_tokens)]
        assert moves[0].nbytes == candidate.prefill_transfer_bytes
        assert moves[1].nbytes * moves[1].repeats == candidate.decode_transfer_bytes
        assert all(row.operations > 0 and row.repeats >= 1
                   for row in lowered.artifact.actions if row.kind == "compute")


def test_capacity_and_missing_evidence_fail_closed():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory); inventory = build_tensor_inventory([path], config)
        hardware = _hardware(host_capacity=64, device_capacity=64)
        hardware = HardwareEnvelope(hardware.name, hardware.architecture,
                                    hardware.banks, hardware.links, ())
        report = assess_model(inventory, hardware, _workload(formats=("source",)))
        assert not any(row.feasible for row in report.candidates)
        assert all(any("allocatable" in reason or "benchmark" in reason
                       for reason in row.blockers) for row in report.candidates)

        complete = _hardware()
        partial = HardwareEnvelope(
            complete.name, complete.architecture, complete.banks, complete.links,
            tuple(row for row in complete.benchmarks
                  if not (row.channel == "host" and row.operation == "decode")))
        partial_report = assess_model(inventory, partial, _workload(formats=("source",)))
        host_resident = next(row for row in partial_report.candidates
                             if row.candidate_id == "source:resident:ram")
        split = next(row for row in partial_report.candidates
                     if row.candidate_id == "source:split:ram:1:vram")
        assert not host_resident.feasible and not split.feasible
        assert {row.operation for row in host_resident.predictions} == {"prefill"}
        assert any("decode" in reason for reason in host_resident.blockers + split.blockers)


def test_bounded_microbench_has_hard_limits_and_valid_evidence():
    prefill, decode = run_bounded_model_microbench(dimension=4, repeats=3)
    assert (prefill.operation, decode.operation) == ("prefill", "decode")
    assert prefill.lower_ns <= prefill.median_ns <= prefill.upper_ns
    for kwargs in ({"dimension": 65}, {"repeats": 2}, {"weight_format": "q4"}):
        try:
            run_bounded_model_microbench(**kwargs)
            raise AssertionError("unbounded/unsupported benchmark request must fail")
        except ValueError:
            pass


def test_cli_writes_report_plan_and_pack_without_model_payload_outputs():
    with tempfile.TemporaryDirectory() as directory:
        _path, _config = _toy(directory)
        hardware_path = os.path.join(directory, "hardware.json")
        workload_path = os.path.join(directory, "workload.json")
        inventory_path = os.path.join(directory, "inventory.json")
        report_path = os.path.join(directory, "report.json")
        plan_path = os.path.join(directory, "plan.json")
        pack_path = os.path.join(directory, "plan.bspk")
        with open(hardware_path, "w", encoding="utf-8") as stream:
            stream.write(_hardware().to_json())
        with open(workload_path, "w", encoding="utf-8") as stream:
            stream.write(_workload(formats=("source",)).to_json())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = assess_main([directory, "--hardware", hardware_path, "--workload", workload_path,
                              "--inventory-out", inventory_path, "--report-out", report_path,
                              "--plan-out", plan_path, "--pack-out", pack_path])
        assert rc == 0
        summary = json.loads(output.getvalue())
        assert summary["payload_bytes_read"] == 0 and summary["selected_candidate"]
        assert all(os.path.getsize(path) > 0 for path in
                   (inventory_path, report_path, plan_path, pack_path))
        assert not any(name.endswith((".safetensors", ".bcirq8")) and name != "model.safetensors"
                       for name in os.listdir(directory))


def test_artifact_parsers_reject_ambiguous_state():
    hardware = _hardware(); workload = _workload()
    for text, parser in (
        (hardware.to_json().replace('{"architecture"', '{"name":"duplicate","architecture"'),
         HardwareEnvelope.from_json),
        (workload.to_json().replace('"batch_size":1', '"batch_size":true'),
         ModelWorkloadSpec.from_json),
    ):
        try:
            parser(text)
            raise AssertionError("ambiguous/malformed artifact must fail")
        except ValueError:
            pass


def test_inventory_and_hardware_refuse_forged_physical_or_measurement_state():
    with tempfile.TemporaryDirectory() as directory:
        path, config = _toy(directory)
        inventory = build_tensor_inventory([path], config)
        document = json.loads(inventory.to_json())
        document["tensors"][1]["data_offset"] += 1
        try:
            TensorInventory.from_json(json.dumps(document))
            raise AssertionError("inventory hole must be rejected")
        except ValueError:
            pass
    hardware = _hardware()
    try:
        HardwareEnvelope(hardware.name, hardware.architecture, hardware.banks, hardware.links,
                         hardware.benchmarks + (hardware.benchmarks[0],))
        raise AssertionError("duplicate active benchmark evidence must be rejected")
    except ValueError:
        pass

    # Equal element counts are insufficient: BCIRQ8's executable contract fixes
    # matrix orientation as well as physical compactness.
    document = json.loads(inventory.to_json())
    down = next(row for row in document["tensors"] if row["role"] == "mlp.down")
    down["shape"] = list(reversed(down["shape"]))
    forged_shape = TensorInventory.from_json(json.dumps(document))
    refused = format_sizes(forged_shape, ("source", "bcirq8-group32"))
    assert all(not row.executable and "shape" in row.reason for row in refused)
