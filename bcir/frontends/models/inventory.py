"""Header-only tensor inventory for model capacity and placement assessment.

This module never reads weight payload bytes.  It consumes the already-strict
Safetensors layout parser, records exact physical spans, and gives the planner a
content-independent *layout identity*.  That identity is intentionally named
``header_digest``: it is suitable for planning before download/load, but it is not a
substitute for the source-artifact hashes required at execution time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..._artifact_json import strict_json_loads
from .manifest import _SAFETENSORS_DTYPE_BITS, parse_safetensors_layout

_SCHEMA = "bcir.tensor_inventory.v1"
_MAX_JSON = 64 * 1024 * 1024
_MAX_U63 = (1 << 63) - 1

# Shared with the strict manifest/ingest parser so no second dtype rail can drift.
_DTYPE_BITS = _SAFETENSORS_DTYPE_BITS

_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)block\.(\d+)(?:\.|$)"),
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _string(value, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{field} must be {'a string' if empty else 'a nonempty string'}")
    if len(value.encode("utf-8")) > 4096 or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{field} exceeds 4096 UTF-8 bytes or contains control characters")
    return value


def _product(shape: tuple[int, ...], field: str) -> int:
    result = 1
    for index, dim in enumerate(shape):
        _integer(dim, f"{field}[{index}]")
        if dim and result > _MAX_U63 // dim:
            raise ValueError(f"{field} element count exceeds signed 64-bit accounting")
        result *= dim
    return result


def _wire_bytes(elements: int, dtype: str) -> int:
    try:
        bits = _DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f"unsupported Safetensors dtype {dtype!r} for exact accounting") from exc
    total_bits = elements * bits
    if total_bits % 8:
        raise ValueError(f"Safetensors dtype {dtype!r} shape is not byte-aligned")
    return total_bits // 8


def _layer_index(name: str) -> int:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return -1


def _role(name: str) -> str:
    """Normalize common HF names without pretending unknown tensors are irrelevant."""
    leaf = name.lower()
    rules = (
        (("embed_tokens", "tok_embeddings", "wte.weight"), "embedding"),
        (("q_proj", "wq.weight", "query.weight"), "attention.q"),
        (("k_proj", "wk.weight", "key.weight"), "attention.k"),
        (("v_proj", "wv.weight", "value.weight"), "attention.v"),
        (("o_proj", "wo.weight", "out_proj"), "attention.o"),
        (("gate_proj", "w1.weight"), "mlp.gate"),
        (("up_proj", "w3.weight"), "mlp.up"),
        (("down_proj", "w2.weight"), "mlp.down"),
        (("input_layernorm", "attention_norm"), "norm.attention"),
        (("post_attention_layernorm", "ffn_norm"), "norm.mlp"),
        (("lm_head", "output.weight"), "lm_head"),
        (("model.norm", "norm.weight", "ln_f.weight"), "norm.final"),
        (("rotary_emb.inv_freq",), "aux.rope"),
    )
    for needles, role in rules:
        if any(needle in leaf for needle in needles):
            return role
    return "other"


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shard: str
    dtype: str
    shape: tuple[int, ...]
    element_count: int
    data_offset: int
    nbytes: int
    role: str
    layer: int

    def __post_init__(self) -> None:
        _string(self.name, "tensor name"); _string(self.shard, "tensor shard")
        _string(self.dtype, "tensor dtype"); _string(self.role, "tensor role")
        if not isinstance(self.shape, (tuple, list)):
            raise ValueError("tensor shape must be a sequence")
        shape = tuple(self.shape)
        if len(shape) > 32:
            raise ValueError("tensor rank exceeds the bounded limit of 32")
        elements = _product(shape, f"tensor {self.name!r} shape")
        object.__setattr__(self, "shape", shape)
        if self.element_count != elements:
            raise ValueError(f"tensor {self.name!r} element count does not match shape")
        _integer(self.data_offset, "tensor data_offset")
        _integer(self.nbytes, "tensor nbytes")
        if self.nbytes != _wire_bytes(elements, self.dtype):
            raise ValueError(f"tensor {self.name!r} payload span does not match dtype/shape")
        _integer(self.layer, "tensor layer", minimum=-1, maximum=(1 << 31) - 1)

    def to_dict(self) -> dict:
        value = asdict(self); value["shape"] = list(self.shape); return value


@dataclass(frozen=True)
class ShardLayout:
    file: str
    file_bytes: int
    data_offset: int
    payload_bytes: int
    tensor_count: int

    def __post_init__(self) -> None:
        _string(self.file, "shard file")
        if self.file != os.path.basename(self.file):
            raise ValueError("shard file must be a basename")
        for field in ("file_bytes", "data_offset", "payload_bytes", "tensor_count"):
            _integer(getattr(self, field), f"shard {field}")
        if self.data_offset + self.payload_bytes != self.file_bytes:
            raise ValueError("shard header/payload sizes do not reconcile")


@dataclass(frozen=True)
class TensorInventory:
    architecture: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    tied_embeddings: bool
    tensors: tuple[TensorRecord, ...]
    shards: tuple[ShardLayout, ...]
    schema: str = _SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA:
            raise ValueError("unsupported tensor inventory schema")
        _string(self.architecture, "architecture")
        for field in ("vocab_size", "hidden_size", "intermediate_size",
                      "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                      "max_position_embeddings"):
            _integer(getattr(self, field), field)
        if type(self.tied_embeddings) is not bool:
            raise ValueError("tied_embeddings must be Boolean")
        if not isinstance(self.tensors, (tuple, list)) or not isinstance(self.shards, (tuple, list)):
            raise ValueError("inventory tensors and shards must be sequences")
        tensors, shards = tuple(self.tensors), tuple(self.shards)
        if any(not isinstance(row, TensorRecord) for row in tensors) \
                or any(not isinstance(row, ShardLayout) for row in shards):
            raise ValueError("inventory contains malformed tensor/shard rows")
        if tuple(sorted(tensors, key=lambda row: (row.shard, row.data_offset, row.name))) != tensors:
            raise ValueError("inventory tensors are not in canonical physical order")
        if tuple(sorted(shards, key=lambda row: row.file)) != shards:
            raise ValueError("inventory shards are not in canonical filename order")
        if len({row.name for row in tensors}) != len(tensors) \
                or len({row.file for row in shards}) != len(shards):
            raise ValueError("inventory has duplicate tensor names or shard files")
        census = {row.file: 0 for row in shards}
        for tensor in tensors:
            if tensor.shard not in census:
                raise ValueError(f"tensor {tensor.name!r} names an unknown shard")
            if tensor.layer >= self.num_hidden_layers:
                raise ValueError(f"tensor {tensor.name!r} layer exceeds configured layer count")
            census[tensor.shard] += 1
        if any(census[row.file] != row.tensor_count for row in shards):
            raise ValueError("inventory shard tensor counts do not reconcile")
        for shard in shards:
            cursor = 0
            physical = [row for row in tensors if row.shard == shard.file]
            for tensor in physical:
                if tensor.data_offset != cursor:
                    raise ValueError(f"inventory shard {shard.file!r} has a hole or overlap")
                cursor += tensor.nbytes
            if cursor != shard.payload_bytes:
                raise ValueError(f"inventory shard {shard.file!r} payload does not reconcile")
        _integer(sum(row.element_count for row in tensors), "inventory total element count")
        _integer(sum(row.file_bytes for row in shards), "inventory total file bytes")
        object.__setattr__(self, "tensors", tensors); object.__setattr__(self, "shards", shards)

    @property
    def parameter_count(self) -> int:
        return sum(row.element_count for row in self.tensors if not row.role.startswith("aux."))

    @property
    def payload_bytes(self) -> int:
        return sum(row.payload_bytes for row in self.shards)

    @property
    def source_file_bytes(self) -> int:
        return sum(row.file_bytes for row in self.shards)

    @property
    def header_bytes(self) -> int:
        return self.source_file_bytes - self.payload_bytes

    @property
    def header_digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    def layer_bytes(self) -> tuple[int, ...]:
        return tuple(sum(row.nbytes for row in self.tensors if row.layer == layer)
                     for layer in range(self.num_hidden_layers))

    def to_dict(self) -> dict:
        return {
            "schema": self.schema, "architecture": self.architecture,
            "vocab_size": self.vocab_size, "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "tied_embeddings": self.tied_embeddings,
            "tensors": [row.to_dict() for row in self.tensors],
            "shards": [asdict(row) for row in self.shards],
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "TensorInventory":
        doc = strict_json_loads(text, "tensor inventory", max_bytes=_MAX_JSON)
        if not isinstance(doc, dict) or set(doc) != {
                "schema", "architecture", "vocab_size", "hidden_size", "intermediate_size",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                "max_position_embeddings", "tied_embeddings", "tensors", "shards"}:
            raise ValueError("tensor inventory has missing or unknown fields")
        if not isinstance(doc["tensors"], list) or not isinstance(doc["shards"], list):
            raise ValueError("tensor inventory tensors/shards must be arrays")
        try:
            tensors = tuple(TensorRecord(**{**row, "shape": tuple(row["shape"])})
                            for row in doc.pop("tensors"))
            shards = tuple(ShardLayout(**row) for row in doc.pop("shards"))
            return cls(**doc, tensors=tensors, shards=shards)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed tensor inventory: {exc}") from exc


def _config_int(config: dict, name: str, aliases: tuple[str, ...], default: int = 0) -> int:
    present = [alias for alias in aliases if alias in config]
    if len(present) > 1 and any(config[key] != config[present[0]] for key in present[1:]):
        raise ValueError(f"config has conflicting aliases for {name}: {present}")
    return _integer(config[present[0]] if present else default, f"config {name}")


def build_tensor_inventory(shard_paths: list[str], config: dict) -> TensorInventory:
    """Build an exact physical inventory while reading only Safetensors headers."""
    if not isinstance(config, dict) or not shard_paths:
        raise ValueError("inventory requires a config object and at least one shard")
    architectures = config.get("architectures", [config.get("model_type", "unknown")])
    architecture = architectures[0] if isinstance(architectures, list) and architectures else architectures
    _string(architecture, "config architecture")
    tied = config.get("tie_word_embeddings", True)
    if type(tied) is not bool:
        raise ValueError("config tie_word_embeddings must be Boolean")

    rows: list[TensorRecord] = []
    shards: list[ShardLayout] = []
    seen_names: set[str] = set(); seen_files: set[str] = set()
    for raw_path in sorted(shard_paths, key=lambda value: os.path.basename(os.fspath(value))):
        path = os.fspath(raw_path); basename = os.path.basename(path)
        if not basename or basename in seen_files:
            raise ValueError(f"duplicate or invalid shard basename {basename!r}")
        seen_files.add(basename)
        tensors, _metadata, data_offset, file_size = parse_safetensors_layout(path)
        for name, spec in sorted(tensors.items(), key=lambda item: (item[1]["data_offsets"][0], item[0])):
            if name in seen_names:
                raise ValueError(f"tensor {name!r} appears in multiple shards")
            seen_names.add(name)
            lo, hi = spec["data_offsets"]
            shape = tuple(spec["shape"]); elements = _product(shape, f"tensor {name!r} shape")
            row = TensorRecord(name, basename, spec["dtype"], shape, elements, lo,
                               hi - lo, _role(name), _layer_index(name))
            rows.append(row)
        shards.append(ShardLayout(basename, file_size, data_offset,
                                  file_size - data_offset, len(tensors)))

    return TensorInventory(
        architecture=str(architecture),
        vocab_size=_config_int(config, "vocab_size", ("vocab_size",)),
        hidden_size=_config_int(config, "hidden_size", ("hidden_size", "n_embd", "d_model")),
        intermediate_size=_config_int(config, "intermediate_size",
                                      ("intermediate_size", "n_inner", "d_ff")),
        num_hidden_layers=_config_int(config, "num_hidden_layers",
                                      ("num_hidden_layers", "n_layer", "n_layers")),
        num_attention_heads=_config_int(config, "num_attention_heads",
                                        ("num_attention_heads", "n_head", "n_heads")),
        num_key_value_heads=_config_int(config, "num_key_value_heads",
                                        ("num_key_value_heads", "n_kv_heads"),
                                        _config_int(config, "num_attention_heads",
                                                    ("num_attention_heads", "n_head", "n_heads"))),
        max_position_embeddings=_config_int(config, "max_position_embeddings",
                                            ("max_position_embeddings", "n_positions", "context_length")),
        tied_embeddings=tied, tensors=tuple(rows), shards=tuple(shards))


__all__ = ["ShardLayout", "TensorInventory", "TensorRecord", "build_tensor_inventory"]
