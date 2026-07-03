"""Open-weight model ingestion (ML/AI roadmap §7.4) -- the staged ladder, manifest-first.

Rung 1 (this package): `ModelManifest`-only ingestion -- architecture, tokenizer reference,
weight-shard inventory + hashes, dtype histogram, parameter count, context length -- built
from the safetensors HEADERS + the model config, BEFORE any weight is ever loaded or
interpreted. Later rungs (tokenizer parity, reference decode, quantized artifacts, the
C/MLIR law rail, serving) build on this record. Dep-free (stdlib only), oracle-side."""

from .manifest import (ModelManifest, ShardRecord, build_manifest, manifest_from_json,
                       parse_safetensors_header, shard_digest)
from .tokenizer import Tokenizer, bytes_to_unicode, load_tokenizer, render_chat

__all__ = ["ModelManifest", "ShardRecord", "Tokenizer", "build_manifest", "bytes_to_unicode",
           "load_tokenizer", "manifest_from_json", "parse_safetensors_header", "render_chat",
           "shard_digest"]
