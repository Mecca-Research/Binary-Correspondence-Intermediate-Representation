"""Deterministic hosted-model export through BCIR's strict Llama ingest boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ModuleNotFoundError as exc:  # pragma: no cover - dependency gate
    raise ModuleNotFoundError(
        "hosted model export requires torch and safetensors; install bcir[model-lab]"
    ) from exc

from ...frontends.models.hf_ingest import ingest_checkpoint_with_report
from ...frontends.models.manifest import ModelManifest, build_manifest
from .model import HostedLlama
from .spec import CorpusManifest


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    encoded = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _config(model: HostedLlama, context_length: int, corpus: CorpusManifest) -> dict:
    spec = model.spec
    config = {
        "architectures": ["BCIRHostedLlamaForCausalLM"],
        "hidden_act": "silu",
        "hidden_size": spec.d_model,
        "intermediate_size": spec.d_ff,
        "max_position_embeddings": context_length,
        "model_type": "llama",
        "num_attention_heads": spec.n_heads,
        "num_hidden_layers": spec.n_layers,
        "num_key_value_heads": spec.kv_heads,
        "rms_norm_eps": spec.rms_norm_eps,
        "rope_theta": spec.rope_base,
        "tie_word_embeddings": spec.tied_embeddings,
        "torch_dtype": "float32",
        "vocab_size": spec.vocab_size,
    }
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(corpus, name)
        if value >= 0:
            config[name] = value
    return config


@dataclass(frozen=True)
class ExportedCheckpoint:
    directory: Path
    model_manifest: ModelManifest
    model_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    artifact_manifest_sha256: str


def export_hf_checkpoint(
    model: HostedLlama,
    output_dir: os.PathLike | str,
    *,
    tokenizer_path: os.PathLike | str,
    corpus_manifest: CorpusManifest,
) -> ExportedCheckpoint:
    """Atomically export a strict one-shard Llama checkpoint.

    Shared tied-head storage is represented once (the LM-head key is omitted), tensor
    names are sorted, and the temporary checkpoint must successfully re-ingest before it
    becomes visible at ``output_dir``.
    """
    if not isinstance(model, HostedLlama):
        raise ValueError("model must be a HostedLlama")
    if not isinstance(corpus_manifest, CorpusManifest):
        raise ValueError("corpus_manifest must be a CorpusManifest")
    if corpus_manifest.vocab_size != model.spec.vocab_size:
        raise ValueError("corpus and model vocabulary sizes differ")
    context_length = model.context_length or 0
    if type(context_length) is not int or context_length < 1:
        raise ValueError("context_length must be a positive integer")
    tokenizer = Path(tokenizer_path)
    if not tokenizer.is_file() or tokenizer.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("tokenizer_path must name a file no larger than 256 MiB")
    tokenizer_digest = sha256_file(tokenizer)
    if tokenizer_digest != corpus_manifest.tokenizer_sha256:
        raise ValueError("tokenizer bytes do not match the corpus manifest")

    target = Path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to replace existing export directory {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        state = model.state_dict()
        tensors = {}
        for name in sorted(state):
            if model.spec.tied_embeddings and name == "lm_head.weight":
                continue
            value = state[name].detach().to(device="cpu", dtype=torch.float32).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f"model tensor {name!r} contains a non-finite value")
            tensors[name] = value
        save_file(tensors, str(temporary / "model.safetensors"))
        config = _config(model, context_length, corpus_manifest)
        _write_json(temporary / "config.json", config)
        tokenizer_name = (
            "tokenizer.json" if tokenizer.suffix.lower() == ".json" else "tokenizer.model"
        )
        shutil.copyfile(tokenizer, temporary / tokenizer_name)

        model_manifest = build_manifest(
            [str(temporary / "model.safetensors")],
            config,
            name="bcir-hosted-llama",
            tokenizer_ref=tokenizer_name,
            tokenizer_path=str(temporary / tokenizer_name),
            license="LicenseRef-BCIR-NC-1.0",
            source=f"bcir-hosted:{corpus_manifest.digest}",
        )
        spec, _weights, report = ingest_checkpoint_with_report(str(temporary))
        if (
            spec != model.spec
            or report.auxiliary_element_count != 0
            or report.decoder_element_count != model_manifest.param_count
        ):
            raise RuntimeError("strict BCIR ingest did not reproduce the hosted model census")

        files = {}
        for name in ("config.json", "model.safetensors", tokenizer_name):
            path = temporary / name
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        artifact_manifest = {
            "schema": "bcir.hosted_inference.v1",
            "corpus": corpus_manifest.to_dict(),
            "files": files,
            "model_manifest": json.loads(model_manifest.to_json()),
        }
        _write_json(temporary / "manifest.json", artifact_manifest)
        manifest_sha = sha256_file(temporary / "manifest.json")
        os.replace(temporary, target)
        return ExportedCheckpoint(
            directory=target,
            model_manifest=model_manifest,
            model_sha256=files["model.safetensors"]["sha256"],
            config_sha256=files["config.json"]["sha256"],
            tokenizer_sha256=files[tokenizer_name]["sha256"],
            artifact_manifest_sha256=manifest_sha,
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
