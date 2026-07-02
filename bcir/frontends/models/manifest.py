"""Rung 1 of the open-weight ladder (ML/AI roadmap §7.4): ModelManifest-ONLY ingestion.

The contract, verbatim from the roadmap: the manifest is built BEFORE any weight loading --
architecture, license, tokenizer reference, weight shards + hashes, dtypes, parameter count,
context length. This module therefore reads exactly two things per shard: the 8-byte
little-endian header length + the JSON header (tensor names -> dtype/shape/data_offsets --
never the tensor bytes as numbers), and the raw file stream for the integrity sha256 (bytes
hashed, never interpreted). Dep-free by the oracle rule (stdlib only); deterministic: the
manifest's own digest is a sha256 over its canonical JSON, so two ingestions of the same
shards + config agree byte-for-byte (the R13-style identity later rungs pin against)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass

_MAX_HEADER = 128 * 1024 * 1024          # a safetensors header larger than 128 MiB is malformed


def parse_safetensors_header(path: str) -> tuple[dict, dict]:
    """The safetensors HEADER of one shard: `(tensors, metadata)` where `tensors` maps
    tensor name -> {"dtype": str, "shape": [int, ...]} and `metadata` is the optional
    `__metadata__` block. Reads the 8-byte LE length + the JSON header ONLY -- the weight
    bytes are never touched here (the rung-1 contract)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file (short header length)")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen == 0 or hlen > _MAX_HEADER or 8 + hlen > size:
            raise ValueError(f"{path}: implausible safetensors header length {hlen}")
        try:
            header = json.loads(f.read(hlen).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"{path}: malformed safetensors JSON header: {e}") from e
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not a JSON object")
    metadata = header.pop("__metadata__", {}) or {}
    tensors: dict = {}
    for name, spec in header.items():
        if not isinstance(spec, dict) or "dtype" not in spec or "shape" not in spec:
            raise ValueError(f"{path}: tensor {name!r} lacks dtype/shape")
        tensors[name] = {"dtype": str(spec["dtype"]), "shape": [int(d) for d in spec["shape"]]}
    return tensors, metadata


def shard_digest(path: str) -> str:
    """The shard file's sha256 (streamed; bytes hashed for INTEGRITY, never interpreted)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ShardRecord:
    """One weight shard's identity: file basename, integrity hash, tensor census."""

    file: str
    sha256: str
    n_tensors: int
    n_params: int


@dataclass(frozen=True)
class ModelManifest:
    """The rung-1 record: everything later rungs pin against, no weight ever loaded."""

    name: str
    architecture: str
    param_count: int
    tensor_count: int
    dtypes: tuple[tuple[str, int], ...]     # (dtype, tensor count), sorted by dtype
    shards: tuple[ShardRecord, ...]         # in ingestion (filename-sorted) order
    context_length: int = 0
    vocab_size: int = 0
    tokenizer_ref: str = ""                 # e.g. the tokenizer.json/model path or hub id
    license: str = ""
    source: str = ""                        # where the shards came from (path / hub id)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        """The manifest's own identity: sha256 over its canonical JSON (deterministic, so two
        ingestions of the same shards + config agree byte-for-byte)."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def manifest_from_json(text: str) -> ModelManifest:
    d = json.loads(text)
    d["dtypes"] = tuple((str(k), int(v)) for k, v in d["dtypes"])
    d["shards"] = tuple(ShardRecord(**s) for s in d["shards"])
    return ModelManifest(**d)


def build_manifest(shard_paths: list[str], config: dict, *, name: str = "",
                   tokenizer_ref: str = "", license: str = "",  # noqa: A002 -- the manifest field
                   source: str = "") -> ModelManifest:
    """Ingest a model MANIFEST-FIRST: parse every shard's header (dtype/shape census +
    parameter count), hash every shard's bytes, and take architecture/context/vocab from the
    model `config` dict (the parsed `config.json`). No weight is loaded or interpreted."""
    shards: list[ShardRecord] = []
    dtype_hist: dict[str, int] = {}
    total_params = total_tensors = 0
    for path in sorted(shard_paths, key=os.path.basename):
        tensors, _meta = parse_safetensors_header(path)
        n_params = 0
        for spec in tensors.values():
            n = 1
            for d in spec["shape"]:
                n *= d
            n_params += n
            dtype_hist[spec["dtype"]] = dtype_hist.get(spec["dtype"], 0) + 1
        shards.append(ShardRecord(file=os.path.basename(path), sha256=shard_digest(path),
                                  n_tensors=len(tensors), n_params=n_params))
        total_params += n_params
        total_tensors += len(tensors)
    arch = config.get("architectures", [config.get("model_type", "")])
    architecture = arch[0] if isinstance(arch, list) and arch else str(arch)
    return ModelManifest(
        name=name or str(config.get("_name_or_path", config.get("model_type", "model"))),
        architecture=architecture,
        param_count=total_params,
        tensor_count=total_tensors,
        dtypes=tuple(sorted(dtype_hist.items())),
        shards=tuple(shards),
        context_length=int(config.get("max_position_embeddings", 0) or 0),
        vocab_size=int(config.get("vocab_size", 0) or 0),
        tokenizer_ref=tokenizer_ref,
        license=license or str(config.get("license", "") or ""),
        source=source,
    )
