#!/usr/bin/env python3
"""Pinned TinyLlama -> BCIRQ8 -> standalone C inference parity gate."""

from __future__ import annotations

import argparse
import gc
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
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from fetch_pinned_model import fetch_model  # noqa: E402
from bcir.frontends.models.decode import next_token_logits  # noqa: E402
from bcir.frontends.models.hf_ingest import ingest_checkpoint  # noqa: E402
from bcir.frontends.models.spm import load_sentencepiece  # noqa: E402
from bcir.frontends.models.weights_io import read_q8_decoder, write_q8_decoder  # noqa: E402


def _argmax(values: list[float]) -> int:
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _nll(logits: list[float], token: int) -> float:
    maximum = max(logits)
    return maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[token]


def _ingest_with_counts(model_dir: Path):
    try:
        from bcir.frontends.models.hf_ingest import ingest_checkpoint_with_report
    except ImportError:
        spec, weights = ingest_checkpoint(str(model_dir))
        from bcir.frontends.models.decode import decoder_param_count
        from bcir.frontends.models.safetensors_io import load_tensors

        tensors = load_tensors(str(model_dir / "model.safetensors"))
        total = sum(len(flat) for _dtype, _shape, flat in tensors.values())
        decoder = decoder_param_count(spec)
        return spec, weights, decoder, total - decoder
    spec, weights, report = ingest_checkpoint_with_report(str(model_dir))
    return (spec, weights, int(report.decoder_element_count), int(report.auxiliary_element_count))


def _host_link_args() -> list[str]:
    try:
        from bcir.toolchain import host_link_args
    except ImportError:
        return [] if os.name == "nt" else ["-lm"]
    return list(host_link_args(["-lm"]))


def _build_c_cli(output: Path) -> None:
    compiler = (
        os.environ.get("CC") or shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")
    )
    if not compiler:
        raise RuntimeError("real-model gate requires Clang or a C11 compiler")
    runtime = ROOT / "runtime" / "c"
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-I",
        str(runtime),
        str(runtime / "bcir_q8_model.c"),
        str(runtime / "bcir_decode.c"),
        str(runtime / "bcir_ai_kernels.c"),
        str(runtime / "bcir_llama.c"),
        str(runtime / "bcir_llama_cli.c"),
        "-o",
        str(output),
        *_host_link_args(),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"standalone C runtime build failed:\n{result.stderr}")


def run_gate(
    *,
    cache_dir: str | None = None,
    output_dir: str | os.PathLike | None = None,
    offline: bool = False,
) -> dict:
    model_dir, pin = fetch_model("tinyllama-v0", cache_dir=cache_dir, offline=offline)
    output = Path(output_dir or ROOT / "build" / "model-gate")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    spec, weights, decoder_elements, auxiliary_elements = _ingest_with_counts(model_dir)
    if decoder_elements != int(pin["decoder_elements"]):
        raise AssertionError(
            f"decoder element census {decoder_elements} != {pin['decoder_elements']}"
        )
    if auxiliary_elements != int(pin["auxiliary_elements"]):
        raise AssertionError(
            f"auxiliary element census {auxiliary_elements} != {pin['auxiliary_elements']}"
        )

    tokenizer = load_sentencepiece(str(model_dir / "tokenizer.model"))
    prompt_ids = [int(config["bos_token_id"])] + tokenizer.encode(str(pin["prompt"]))
    if prompt_ids != pin["prompt_ids"]:
        raise AssertionError(f"tokenizer drift: {prompt_ids} != {pin['prompt_ids']}")
    float_logits = next_token_logits(prompt_ids, spec, weights)
    float_id = _argmax(float_logits)
    expected_id = int(pin["expected_first_token_id"])
    if float_id != expected_id:
        raise AssertionError(f"float decoder emitted {float_id}, expected {expected_id}")

    hashes = {
        "model": pin["files"]["model.safetensors"]["sha256"],
        "config": pin["files"]["config.json"]["sha256"],
        "tokenizer": pin["files"]["tokenizer.model"]["sha256"],
    }
    tokenizer_ids = {
        "bos": int(config["bos_token_id"]),
        "eos": int(config["eos_token_id"]),
        "pad": int(config.get("pad_token_id", -1)),
        "context_length": int(config["max_position_embeddings"]),
    }
    artifact = output / "tinyllama-v0.bcirq8"
    repeat = output / "tinyllama-v0.repeat.bcirq8"
    metadata = write_q8_decoder(
        artifact, spec, weights, group_size=32, source_hashes=hashes, tokenizer_ids=tokenizer_ids
    )
    repeat_metadata = write_q8_decoder(
        repeat, spec, weights, group_size=32, source_hashes=hashes, tokenizer_ids=tokenizer_ids
    )
    if (
        metadata.artifact_sha256 != repeat_metadata.artifact_sha256
        or artifact.read_bytes() != repeat.read_bytes()
    ):
        raise AssertionError("BCIRQ8 export is not deterministic")
    repeat.unlink()
    if metadata.file_size >= int(pin["files"]["model.safetensors"]["bytes"]) * 0.56:
        raise AssertionError("BCIRQ8 artifact is not compact enough (<56% of BF16 source required)")
    del weights
    gc.collect()

    q8_spec, q8_weights, read_metadata = read_q8_decoder(artifact)
    if read_metadata != metadata:
        raise AssertionError("BCIRQ8 metadata changed on readback")
    q8_logits = next_token_logits(prompt_ids, q8_spec, q8_weights)
    q8_id = _argmax(q8_logits)
    if q8_id != expected_id:
        raise AssertionError(f"Python Q8 decoder emitted {q8_id}, expected {expected_id}")

    executable = output / ("bcir-llama.exe" if os.name == "nt" else "bcir-llama")
    logits_path = output / "logits.f64"
    _build_c_cli(executable)
    command = [
        str(executable),
        "--model",
        str(artifact),
        "--prompt-ids",
        ",".join(str(value) for value in prompt_ids),
        "--max-new",
        "1",
        "--logits-out",
        str(logits_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"standalone C inference failed:\n{result.stderr}")
    c_output = json.loads(result.stdout)
    c_ids = [int(value) for value in c_output["generated_ids"]]
    raw_logits = logits_path.read_bytes()
    if len(raw_logits) != 8 * spec.vocab_size:
        raise AssertionError("standalone C runtime wrote a malformed logits file")
    c_logits = list(struct.unpack(f"<{spec.vocab_size}d", raw_logits))
    if not all(math.isfinite(value) for value in float_logits + q8_logits + c_logits):
        raise AssertionError("non-finite real-model logits")
    max_c_error = max(abs(left - right) for left, right in zip(q8_logits, c_logits))
    if c_ids != [q8_id] or c_ids != [expected_id]:
        raise AssertionError(f"generated-id parity failed: Python={q8_id}, C={c_ids}")
    if max_c_error > 1e-9:
        raise AssertionError(f"C/Python Q8 max logit error {max_c_error} > 1e-9")
    drift = max(abs(left - right) for left, right in zip(float_logits, q8_logits))
    nll_f32, nll_q8 = _nll(float_logits, expected_id), _nll(q8_logits, expected_id)
    report = {
        "schema": "bcir.real_model_parity.v1",
        "source": {
            "repo": pin["repository"],
            "revision": pin["revision"],
            "license": pin["license"],
            "files": {
                name: {"bytes": int(info["bytes"]), "sha256": info["sha256"]}
                for name, info in sorted(pin["files"].items())
            },
        },
        "model": {
            "decoder_elements": decoder_elements,
            "auxiliary_elements": auxiliary_elements,
            "spec": {
                "vocab_size": spec.vocab_size,
                "d_model": spec.d_model,
                "n_heads": spec.n_heads,
                "n_kv_heads": spec.kv_heads,
                "n_layers": spec.n_layers,
                "d_ff": spec.d_ff,
                "rms_norm_eps": float(getattr(spec, "rms_norm_eps", 1e-6)),
                "tied_embeddings": spec.tied_embeddings,
            },
        },
        "tokenizer": {"prompt": pin["prompt"], "prompt_ids": prompt_ids},
        "artifact": {
            "format": "BCIRQ8",
            "version": metadata.version,
            "group_size": metadata.group_size,
            "bytes": metadata.file_size,
            "sha256": metadata.artifact_sha256,
            "body_crc32": metadata.body_crc32,
        },
        "decode": {
            "float_ids": [float_id],
            "q8_python_ids": [q8_id],
            "q8_c_ids": c_ids,
            "max_abs_logit_error_c_vs_q8": max_c_error,
            "max_abs_logit_drift_q8_vs_float": drift,
            "nll_f32": nll_f32,
            "nll_q8": nll_q8,
            "nll_delta": nll_q8 - nll_f32,
        },
        "passed": True,
    }
    report_path = output / "parity-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = run_gate(cache_dir=args.cache_dir, output_dir=args.output_dir, offline=args.offline)
    print(
        json.dumps(
            {
                "artifact_sha256": report["artifact"]["sha256"],
                "generated_ids": report["decode"]["q8_c_ids"],
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
