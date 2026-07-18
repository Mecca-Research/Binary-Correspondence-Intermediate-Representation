#!/usr/bin/env python3
"""Bounded from-random-weights -> Safetensors -> BCIRQ8 -> standalone-C gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from bcir.frontends.models.decode import next_token_logits
from bcir.frontends.models.assessment import (
    BankEnvelope,
    HardwareEnvelope,
    KernelBenchmark,
    ModelWorkloadSpec,
    assess_model,
    select_execution_candidate,
)
from bcir.frontends.models.execution_plan import lower_model_execution
from bcir.frontends.models.hf_ingest import ingest_checkpoint_with_report
from bcir.frontends.models.inventory import build_tensor_inventory
from bcir.frontends.models.tokenizer import bytes_to_unicode, load_tokenizer
from bcir.frontends.models.weights_io import read_q8_decoder, write_q8_decoder
from bcir.hosted.models.checkpoint import load_safe_checkpoint
from bcir.hosted.models.export import export_hf_checkpoint, sha256_file
from bcir.hosted.models.model import HostedLlama
from bcir.hosted.models.spec import (CorpusManifest, HostedTrainSpec,
                                     SequenceTokenSource)
from bcir.hosted.models.train import train_hosted
from bcir.frontends.models.decode import DecoderSpec


def _argmax(values) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _write_byte_tokenizer(path: Path) -> None:
    alphabet = bytes_to_unicode()
    value = {
        "added_tokens": [
            {"content": "<unk>", "id": 256, "special": True},
            {"content": "<s>", "id": 257, "special": True},
            {"content": "</s>", "id": 258, "special": True},
            {"content": "<pad>", "id": 259, "special": True},
        ],
        "model": {"type": "BPE", "vocab": {alphabet[byte]: byte for byte in range(256)},
                  "merges": []},
    }
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def _spec() -> HostedTrainSpec:
    return HostedTrainSpec(
        decoder=DecoderSpec(vocab_size=260, d_model=64, n_heads=4, n_layers=2,
                            d_ff=128, activation="silu_gate", n_kv_heads=2,
                            tied_embeddings=True, rms_norm_eps=1e-6),
        context_length=16, batch_size=4, gradient_accumulation=1,
        steps=64, checkpoint_every=32, optimizer="adamw",
        learning_rate=3e-3, min_learning_rate=0.0, warmup_steps=4,
        beta1=0.9, beta2=0.95, epsilon=1e-8, weight_decay=0.0,
        grad_clip=1.0, seed=1729, dtype="float32",
        activation_checkpointing=False)


def _optimizer(model: HostedLlama, spec: HostedTrainSpec):
    return torch.optim.AdamW(model.parameters(), lr=spec.learning_rate,
                             betas=(spec.beta1, spec.beta2), eps=spec.epsilon,
                             weight_decay=spec.weight_decay, amsgrad=False)


def _host_link_args() -> list[str]:
    from bcir.toolchain import host_link_args
    return list(host_link_args(["-lm"]))


def _build_c_cli(output: Path) -> None:
    compiler = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc") \
        or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("hosted model gate requires a C11 compiler")
    runtime = ROOT / "runtime" / "c"
    command = [compiler, "-std=c11", "-O2", "-ffp-contract=off", "-Wall", "-Wextra",
               "-Werror", "-I", str(runtime), str(runtime / "bcir_q8_model.c"),
               str(runtime / "bcir_decode.c"), str(runtime / "bcir_llama.c"),
               str(runtime / "bcir_llama_cli.c"), "-o", str(output), *_host_link_args()]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"standalone C build failed:\n{result.stderr}")


def _same_export(left: Path, right: Path) -> None:
    left_files = sorted(path.name for path in left.iterdir())
    right_files = sorted(path.name for path in right.iterdir())
    if left_files != right_files:
        raise AssertionError("repeated inference exports have different inventories")
    for name in left_files:
        if (left / name).read_bytes() != (right / name).read_bytes():
            raise AssertionError(f"repeated inference export differs at {name}")


def run_gate(output_dir: os.PathLike | str) -> dict:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"hosted model gate output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tokenizer_path = output / "byte-tokenizer.json"
    _write_byte_tokenizer(tokenizer_path)
    tokenizer = load_tokenizer(str(tokenizer_path))
    pattern = "abcdefghijklmnopqrstuvwxyz\n"
    token_ids = tokenizer.encode(pattern * 1024)
    source_raw = bytes(token_ids)
    corpus = CorpusManifest(
        name="bcir-next-byte-micro-v1", revision="generated-v1", split="train",
        license="LicenseRef-BCIR-NC-1.0",
        source_sha256=hashlib.sha256(source_raw).hexdigest(),
        tokenizer_sha256=sha256_file(tokenizer_path), token_count=len(token_ids),
        vocab_size=260, bos_token_id=257, eos_token_id=258, pad_token_id=259)
    source = SequenceTokenSource(token_ids, corpus)
    spec = _spec()

    telemetry_path = output / "training-telemetry.jsonl"
    with telemetry_path.open("w", encoding="utf-8", newline="\n") as telemetry:
        def sink(event):
            telemetry.write(json.dumps(event.to_dict(), sort_keys=True,
                                       separators=(",", ":")) + "\n")
        full = train_hosted(spec, source, output / "training-full", telemetry_sink=sink)
    telemetry_rows = [json.loads(line) for line in telemetry_path.read_text("utf-8").splitlines()]
    if len(telemetry_rows) != spec.steps \
            or [row["step"] for row in telemetry_rows] != list(range(1, spec.steps + 1)) \
            or telemetry_rows[-1]["tokens_processed"] != full.tokens_processed \
            or any(not math.isfinite(row[field]) for row in telemetry_rows
                   for field in ("loss", "learning_rate", "grad_norm")):
        raise AssertionError("hosted telemetry is incomplete, unordered, or non-finite")
    step32 = sorted((output / "training-full" / "checkpoints").glob("step-00000032-*"))
    if len(step32) != 1:
        raise AssertionError("the micro run did not publish exactly one step-32 checkpoint")
    resumed = train_hosted(spec, source, output / "training-resumed", resume_from=step32[0])
    identity_fields = ("steps", "batch_index", "tokens_processed", "initial_loss", "final_loss",
                       "model_sha256", "optimizer_sha256", "checkpoint_sha256")
    if any(getattr(full, name) != getattr(resumed, name) for name in identity_fields):
        raise AssertionError("uninterrupted and resumed hosted runs are not identical")
    if not full.final_loss < full.initial_loss:
        raise AssertionError("micro training did not reduce held-out deterministic loss")

    torch.manual_seed(spec.seed)
    model = HostedLlama(spec.decoder, context_length=spec.context_length)
    optimizer = _optimizer(model, spec)
    load_safe_checkpoint(output / "training-full", model, optimizer, spec=spec,
                         corpus=corpus, device=torch.device("cpu"))
    export = export_hf_checkpoint(model, output / "inference", tokenizer_path=tokenizer_path,
                                  corpus_manifest=corpus)
    repeat = export_hf_checkpoint(model, output / "inference-repeat",
                                  tokenizer_path=tokenizer_path,
                                  corpus_manifest=corpus)
    _same_export(export.directory, repeat.directory)

    decoder_spec, weights, ingest_report = ingest_checkpoint_with_report(str(export.directory))
    prompt_ids = tokenizer.encode("abc")
    expected_id = tokenizer.encode("d")[0]
    with torch.no_grad():
        model.eval()
        torch_logits, _ = model(torch.tensor([prompt_ids], dtype=torch.long))
        hosted_logits = torch_logits[0, -1].double().tolist()
    float_logits = next_token_logits(prompt_ids, decoder_spec, weights)
    max_host_error = max(abs(left - right) for left, right in zip(hosted_logits, float_logits))
    float_id = _argmax(float_logits)
    if max_host_error > 1e-5:
        raise AssertionError(f"PyTorch/BCIR float logit error {max_host_error} > 1e-5")
    if _argmax(hosted_logits) != expected_id or float_id != expected_id:
        raise AssertionError("trained float model did not learn prompt 'abc' -> token 'd'")

    hashes = {"model": export.model_sha256, "config": export.config_sha256,
              "tokenizer": export.tokenizer_sha256}
    tokenizer_ids = {"bos": corpus.bos_token_id, "eos": corpus.eos_token_id,
                     "pad": corpus.pad_token_id, "context_length": spec.context_length}
    artifact = output / "bcir-micro.bcirq8"
    artifact_repeat = output / "bcir-micro-repeat.bcirq8"
    metadata = write_q8_decoder(artifact, decoder_spec, weights, group_size=32,
                                source_hashes=hashes, tokenizer_ids=tokenizer_ids)
    metadata_repeat = write_q8_decoder(artifact_repeat, decoder_spec, weights, group_size=32,
                                       source_hashes=hashes, tokenizer_ids=tokenizer_ids)
    if metadata.artifact_sha256 != metadata_repeat.artifact_sha256 \
            or artifact.read_bytes() != artifact_repeat.read_bytes():
        raise AssertionError("micro BCIRQ8 export is not deterministic")

    # Assess the trained asset from its bounded Safetensors header.  These fixed
    # calibration rows are a portable CI fixture, not measurements of the host.
    config = json.loads((export.directory / "config.json").read_text("utf-8"))
    inventory = build_tensor_inventory(
        [str(export.directory / "model.safetensors")], config)
    if inventory.parameter_count != ingest_report.decoder_element_count:
        raise AssertionError("header-only inventory disagrees with strict checkpoint ingestion")
    bank = BankEnvelope(
        "ram", "RAM", "host", 1 << 30, 1 << 30,
        20_000_000_000, 20_000_000_000)
    benchmarks = tuple(
        KernelBenchmark(operation, "host", weight_format,
                        median - 100, median, median + 100,
                        tokens, 1_000_000, max(1, inventory.payload_bytes),
                        1 << 20, 5)
        for weight_format, scale in (("source", 2), ("bcirq8-group32", 1))
        for operation, tokens, median in (("prefill", len(prompt_ids), 2000 * scale),
                                          ("decode", 1, 1000 * scale)))
    hardware = HardwareEnvelope(
        "hosted-model-gate-synthetic", "portable-ci-reference", (bank,), (), benchmarks)
    workload = ModelWorkloadSpec(
        mode="inference", batch_size=1, sessions=1,
        prompt_tokens=len(prompt_ids), context_tokens=spec.context_length,
        max_new_tokens=1, activation_dtype="f32", kv_dtype="f32",
        gradient_dtype="f32", optimizer="none", lora_rank=0,
        checkpoint_segments=1, formats=("source", "bcirq8-group32"))
    cost_report = assess_model(inventory, hardware, workload)
    selected = select_execution_candidate(cost_report)
    if selected.weight_format != "bcirq8-group32" or selected.classification != "quantized":
        raise AssertionError("synthetic model assessment did not select the Q8 resident plan")
    predicted_q8 = next(row for row in cost_report.formats
                        if row.name == "bcirq8-group32")
    if predicted_q8.file_bytes != artifact.stat().st_size:
        raise AssertionError("header-only BCIRQ8 size prediction disagrees with export")
    lowered = lower_model_execution(
        cost_report, selected, inventory, hardware, workload)
    if lowered.artifact.classification != "quantized" \
            or lowered.artifact.streampack_bytes != len(lowered.streampack_data):
        raise AssertionError("quantized model execution plan did not reconcile")

    q8_spec, q8_weights, read_metadata = read_q8_decoder(artifact)
    if read_metadata != metadata:
        raise AssertionError("micro BCIRQ8 metadata changed on readback")
    q8_logits = next_token_logits(prompt_ids, q8_spec, q8_weights)
    q8_id = _argmax(q8_logits)
    if q8_id != expected_id:
        raise AssertionError("micro Q8 model did not retain prompt 'abc' -> token 'd'")

    executable = output / ("bcir-llama.exe" if os.name == "nt" else "bcir-llama")
    logits_path = output / "c-logits.f64"
    _build_c_cli(executable)
    result = subprocess.run(
        [str(executable), "--model", str(artifact), "--prompt-ids",
         ",".join(map(str, prompt_ids)), "--max-new", "1", "--logits-out", str(logits_path)],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"standalone C inference failed:\n{result.stderr}")
    c_ids = [int(value) for value in json.loads(result.stdout)["generated_ids"]]
    raw_logits = logits_path.read_bytes()
    if len(raw_logits) != 8 * decoder_spec.vocab_size:
        raise AssertionError("standalone C logits file has the wrong size")
    c_logits = list(struct.unpack(f"<{decoder_spec.vocab_size}d", raw_logits))
    if not all(math.isfinite(value) for value in float_logits + q8_logits + c_logits):
        raise AssertionError("non-finite train-to-C logits")
    max_c_error = max(abs(left - right) for left, right in zip(q8_logits, c_logits))
    if c_ids != [expected_id] or max_c_error > 1e-9:
        raise AssertionError(f"C/Q8 parity failed: ids={c_ids}, error={max_c_error}")

    report = {
        "schema": "bcir.hosted_model_gate.v2",
        "assessment": {
            "classification": lowered.artifact.classification,
            "cost_report_sha256": cost_report.digest,
            "inventory_sha256": inventory.header_digest,
            "parameter_count": inventory.parameter_count,
            "payload_bytes_read": 0,
            "plan_sha256": lowered.artifact.digest,
            "predicted_q8_bytes": predicted_q8.file_bytes,
            "selected_candidate": selected.candidate_id,
            "streampack_bytes": len(lowered.streampack_data),
        },
        "artifact": {"bytes": metadata.file_size, "group_size": metadata.group_size,
                     "sha256": metadata.artifact_sha256},
        "corpus": corpus.to_dict(),
        "decode": {"expected_id": expected_id, "float_id": float_id,
                   "q8_python_id": q8_id, "q8_c_ids": c_ids,
                   "max_abs_logit_error_host_vs_bcir": max_host_error,
                   "max_abs_logit_error_c_vs_q8": max_c_error,
                   "max_abs_logit_drift_q8_vs_float": max(
                       abs(left - right) for left, right in zip(float_logits, q8_logits)),
                   "prompt": "abc", "prompt_ids": prompt_ids},
        "export": {"config_sha256": export.config_sha256,
                   "manifest_sha256": export.artifact_manifest_sha256,
                   "model_sha256": export.model_sha256,
                   "tokenizer_sha256": export.tokenizer_sha256},
        "model": {"decoder_elements": ingest_report.decoder_element_count,
                  "d_ff": decoder_spec.d_ff, "d_model": decoder_spec.d_model,
                  "n_heads": decoder_spec.n_heads, "n_kv_heads": decoder_spec.kv_heads,
                  "n_layers": decoder_spec.n_layers, "tied_embeddings": True,
                  "vocab_size": decoder_spec.vocab_size},
        "training": {"batch_index": full.batch_index, "final_loss": full.final_loss,
                     "initial_loss": full.initial_loss, "model_sha256": full.model_sha256,
                     "optimizer_sha256": full.optimizer_sha256,
                     "resume_exact": True, "steps": full.steps,
                     "telemetry_sha256": sha256_file(telemetry_path),
                     "tokens_processed": full.tokens_processed},
        "passed": True,
    }
    (output / "parity-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ROOT / "build" / "hosted-model-gate"))
    args = parser.parse_args()
    report = run_gate(args.output_dir)
    print(json.dumps({"passed": report["passed"],
                      "report": str(Path(args.output_dir) / "parity-report.json")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
