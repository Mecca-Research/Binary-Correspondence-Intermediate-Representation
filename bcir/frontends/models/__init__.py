"""Open-weight model ingestion (ML/AI roadmap §7.4) -- the staged ladder, manifest-first.

Rung 1 (this package): `ModelManifest`-only ingestion -- architecture, tokenizer reference,
weight-shard inventory + hashes, dtype histogram, parameter count, context length -- built
from the safetensors HEADERS + the model config, BEFORE any weight is ever loaded or
interpreted. Later rungs (tokenizer parity, reference decode, quantized artifacts, the
C/MLIR law rail, serving) build on this record. Dep-free (stdlib only), oracle-side."""

from .manifest import (
    ModelManifest,
    ShardRecord,
    build_manifest,
    manifest_from_json,
    parse_safetensors_header,
    parse_safetensors_layout,
    shard_digest,
)
from .inventory import ShardLayout, TensorInventory, TensorRecord, build_tensor_inventory
from .tokenizer import Tokenizer, bytes_to_unicode, load_tokenizer, render_chat

_LAZY = {
    "BankEnvelope": (".assessment", "BankEnvelope"),
    "HardwareEnvelope": (".assessment", "HardwareEnvelope"),
    "KernelBenchmark": (".assessment", "KernelBenchmark"),
    "LinkEnvelope": (".assessment", "LinkEnvelope"),
    "ModelCostReport": (".assessment", "ModelCostReport"),
    "ModelExecutionPlan": (".assessment", "ModelExecutionPlan"),
    "ModelWorkloadSpec": (".assessment", "ModelWorkloadSpec"),
    "assess_model": (".assessment", "assess_model"),
    "lower_model_execution": (".execution_plan", "lower_model_execution"),
}


def __getattr__(name):
    """Keep the assessment/compiler rail cold until explicitly requested."""
    if name not in _LAZY:
        raise AttributeError(name)
    from importlib import import_module

    module_name, symbol = _LAZY[name]
    value = getattr(import_module(module_name, __name__), symbol)
    globals()[name] = value
    return value


__all__ = [
    "ModelManifest",
    "ShardLayout",
    "ShardRecord",
    "TensorInventory",
    "TensorRecord",
    "Tokenizer",
    "build_manifest",
    "build_tensor_inventory",
    "bytes_to_unicode",
    "load_tokenizer",
    "manifest_from_json",
    "parse_safetensors_header",
    "parse_safetensors_layout",
    "render_chat",
    "shard_digest",
    *_LAZY,
]
