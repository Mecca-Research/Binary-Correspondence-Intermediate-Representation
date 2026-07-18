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

_MAX_HEADER = 100_000_000  # Safetensors v0.8.0's normative parser ceiling
_SAFETENSORS_DTYPE_BITS = {
    "F4": 4, "F6_E2M3": 6, "F6_E3M2": 6,
    "BOOL": 8, "U8": 8, "I8": 8,
    "F8_E4M3": 8, "F8_E4M3FNUZ": 8,
    "F8_E5M2": 8, "F8_E5M2FNUZ": 8, "F8_E8M0": 8,
    "U16": 16, "I16": 16, "F16": 16, "BF16": 16,
    "U32": 32, "I32": 32, "F32": 32, "C64": 64,
    "U64": 64, "I64": 64, "F64": 64,
}


def _unique_object(pairs):
    """JSON object hook that rejects duplicate keys instead of silently taking the last.

    Duplicate tensor names or duplicate ``data_offsets`` fields make two parsers see
    different artifacts, so a model header must have one unambiguous interpretation.
    """
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON object key {key!r}")
        out[key] = value
    return out


def _tensor_nbytes(dtype: str, shape: list[int], data_size: int,
                   path: str, name: str) -> int:
    try:
        bits = _SAFETENSORS_DTYPE_BITS[dtype]
    except KeyError as exc:
        raise ValueError(f"{path}: tensor {name!r} has unknown dtype {dtype!r}") from exc
    if any(dim == 0 for dim in shape):
        elements = 0
    else:
        elements = 1
        max_elements = (data_size * 8) // bits
        for dim in shape:
            if dim and elements > max_elements // dim:
                raise ValueError(f"{path}: tensor {name!r} shape exceeds the shard payload")
            elements *= dim
    total_bits = elements * bits
    if total_bits % 8:
        raise ValueError(f"{path}: tensor {name!r} is not byte-aligned")
    return total_bits // 8


def _read_safetensors_header(stream, path: str, size: int) -> tuple[dict, dict, int]:
    """Read and fully validate one header from an already-open shard.

    Returning the normalized tensor records lets the weight loader keep validation and
    payload reads on the same file descriptor, avoiding a validate/reopen TOCTOU gap.
    """
    raw = stream.read(8)
    if len(raw) != 8:
        raise ValueError(f"{path}: not a safetensors file (short header length)")
    (hlen,) = struct.unpack("<Q", raw)
    if hlen == 0 or hlen > _MAX_HEADER or hlen > size - 8:
        raise ValueError(f"{path}: implausible safetensors header length {hlen}")
    encoded = stream.read(hlen)
    if len(encoded) != hlen:
        raise ValueError(f"{path}: truncated safetensors JSON header")
    if encoded[:1] != b"{":
        raise ValueError(f"{path}: safetensors JSON header must begin with '{{'")
    try:
        text = encoded.decode("utf-8")
        header, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(text)
        if any(character != " " for character in text[end:]):
            raise ValueError("header may contain only ASCII-space trailing padding")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{path}: malformed safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not a JSON object")

    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()):
        raise ValueError(f"{path}: __metadata__ must be a string-to-string object")

    data_off = 8 + hlen
    data_size = size - data_off
    tensors: dict = {}
    spans: list[tuple[int, int, str]] = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: tensor {name!r} is not an object")
        required = {"dtype", "shape", "data_offsets"}
        if set(spec) != required:
            raise ValueError(f"{path}: tensor {name!r} fields must be exactly "
                             f"{sorted(required)}; got {sorted(spec)}")
        dtype, shape, offsets = spec["dtype"], spec["shape"], spec["data_offsets"]
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"{path}: tensor {name!r} has an invalid dtype")
        if not isinstance(shape, list) or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim < 0
                for dim in shape):
            raise ValueError(f"{path}: tensor {name!r} has an invalid shape {shape!r}")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(not isinstance(value, int) or isinstance(value, bool)
                       for value in offsets)):
            raise ValueError(f"{path}: tensor {name!r} has invalid data_offsets")
        lo, hi = offsets
        if not (0 <= lo <= hi <= data_size):
            raise ValueError(f"{path}: tensor {name!r} byte span [{lo}, {hi}) "
                             "escapes the data section")
        expected_bytes = _tensor_nbytes(dtype, shape, data_size, path, name)
        if hi - lo != expected_bytes:
            raise ValueError(f"{path}: tensor {name!r} byte span has {hi - lo} bytes; "
                             f"dtype/shape require {expected_bytes}")
        tensors[name] = {"dtype": dtype, "shape": list(shape),
                         "data_offsets": [lo, hi]}
        spans.append((lo, hi, name))

    # Safetensors requires the payload to be one complete, non-overlapping partition.
    # Besides catching aliases, this rejects unindexed suffixes/polyglot payloads.
    cursor = 0
    for lo, hi, name in sorted(spans):
        if lo < cursor:
            raise ValueError(f"{path}: tensor {name!r} overlaps a previous tensor payload")
        if lo > cursor:
            raise ValueError(f"{path}: unindexed data hole [{cursor}, {lo})")
        cursor = hi
    if cursor != data_size:
        raise ValueError(f"{path}: unindexed data suffix [{cursor}, {data_size})")
    return tensors, metadata, data_off


def parse_safetensors_header(path: str) -> tuple[dict, dict]:
    """The safetensors HEADER of one shard: `(tensors, metadata)` where `tensors` maps
    tensor name -> {"dtype": str, "shape": [int, ...]} and `metadata` is the optional
    `__metadata__` block. Reads the 8-byte LE length + the JSON header ONLY -- the weight
    bytes are never touched here (the rung-1 contract)."""
    with open(path, "rb") as f:
        tensors, metadata, _data_off = _read_safetensors_header(
            f, path, os.fstat(f.fileno()).st_size)
    return ({name: {"dtype": spec["dtype"], "shape": spec["shape"]}
             for name, spec in tensors.items()}, metadata)


def parse_safetensors_layout(path: str) -> tuple[dict, dict, int, int]:
    """Return one validated shard's header-only physical layout.

    The result is ``(tensors, metadata, data_offset, file_size)``.  Unlike
    :func:`parse_safetensors_header`, tensor rows retain their payload-relative
    ``data_offsets`` so a model assessor can account for bytes and placement without
    reading, hashing, mapping, or interpreting any weight payload.  This function
    deliberately shares the strict parser used by real ingestion; it is not a second
    permissive header rail.
    """
    with open(path, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        tensors, metadata, data_offset = _read_safetensors_header(
            stream, path, size)
    return tensors, metadata, data_offset, size


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
    tokenizer_digest: str = ""              # sha256 of the tokenizer file (rung-2 parity tie)
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
    if not isinstance(text, str) or len(text) > 16 * 1024 * 1024:
        raise ValueError("model manifest JSON must be a string no larger than 16 MiB")
    try:
        d = json.loads(text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"malformed model manifest JSON: {exc}") from exc
    if not isinstance(d, dict):
        raise ValueError("model manifest root must be an object")
    required = {"name", "architecture", "param_count", "tensor_count", "dtypes", "shards"}
    optional = {"context_length", "vocab_size", "tokenizer_ref", "tokenizer_digest",
                "license", "source"}
    if not required <= set(d) or set(d) - required - optional:
        raise ValueError("model manifest has missing or unknown fields")

    def nonnegative_int(value, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"model manifest {field} must be a non-negative integer")
        return value

    for field in ("name", "architecture", "tokenizer_ref", "tokenizer_digest",
                  "license", "source"):
        if field in d and not isinstance(d[field], str):
            raise ValueError(f"model manifest {field} must be a string")
    for field in ("param_count", "tensor_count", "context_length", "vocab_size"):
        if field in d:
            d[field] = nonnegative_int(d[field], field)
    digest = d.get("tokenizer_digest", "")
    if digest and (len(digest) != 64 or digest != digest.lower()
                   or any(ch not in "0123456789abcdef" for ch in digest)):
        raise ValueError("model manifest tokenizer_digest must be lowercase SHA-256 hex")

    if not isinstance(d["dtypes"], list):
        raise ValueError("model manifest dtypes must be an array")
    dtype_rows = []
    seen_dtypes: set[str] = set()
    for row in d["dtypes"]:
        if (not isinstance(row, list) or len(row) != 2 or not isinstance(row[0], str)
                or not row[0] or row[0] in seen_dtypes):
            raise ValueError("model manifest has a malformed or duplicate dtype row")
        count = nonnegative_int(row[1], f"dtype {row[0]!r} count")
        if count < 1:
            raise ValueError("model manifest dtype counts must be positive")
        seen_dtypes.add(row[0]); dtype_rows.append((row[0], count))
    if dtype_rows != sorted(dtype_rows):
        raise ValueError("model manifest dtypes must be in canonical sorted order")
    d["dtypes"] = tuple(dtype_rows)

    if not isinstance(d["shards"], list):
        raise ValueError("model manifest shards must be an array")
    shards = []
    seen_files: set[str] = set()
    for index, row in enumerate(d["shards"]):
        if not isinstance(row, dict) or set(row) != {"file", "sha256", "n_tensors", "n_params"}:
            raise ValueError(f"model manifest shard {index} has invalid fields")
        file = row["file"]; sha = row["sha256"]
        if (not isinstance(file, str) or not file or file != os.path.basename(file)
                or file in seen_files):
            raise ValueError(f"model manifest shard {index} has an invalid/duplicate filename")
        if (not isinstance(sha, str) or len(sha) != 64 or sha != sha.lower()
                or any(ch not in "0123456789abcdef" for ch in sha)):
            raise ValueError(f"model manifest shard {index} has an invalid SHA-256")
        seen_files.add(file)
        shards.append(ShardRecord(file=file, sha256=sha,
                                  n_tensors=nonnegative_int(row["n_tensors"], "n_tensors"),
                                  n_params=nonnegative_int(row["n_params"], "n_params")))
    d["shards"] = tuple(shards)
    if sum(row.n_tensors for row in shards) != d["tensor_count"] \
            or sum(row.n_params for row in shards) != d["param_count"] \
            or sum(count for _dtype, count in dtype_rows) != d["tensor_count"]:
        raise ValueError("model manifest aggregate counts do not reconcile")
    return ModelManifest(**d)


def build_manifest(shard_paths: list[str], config: dict, *, name: str = "",
                   tokenizer_ref: str = "", tokenizer_path: str | None = None,
                   license: str = "",  # noqa: A002 -- the manifest field
                   source: str = "") -> ModelManifest:
    """Ingest a model MANIFEST-FIRST: parse every shard's header (dtype/shape census +
    parameter count), hash every shard's bytes, and take architecture/context/vocab from the
    model `config` dict (the parsed `config.json`). No weight is loaded or interpreted."""
    shards: list[ShardRecord] = []
    dtype_hist: dict[str, int] = {}
    total_params = total_tensors = 0
    seen_files: set[str] = set()
    seen_tensors: set[str] = set()
    for path in sorted(shard_paths, key=os.path.basename):
        basename = os.path.basename(path)
        if basename in seen_files:
            raise ValueError(f"duplicate shard basename {basename!r}")
        seen_files.add(basename)
        tensors, _meta = parse_safetensors_header(path)
        duplicates = seen_tensors.intersection(tensors)
        if duplicates:
            raise ValueError(f"tensor names appear in multiple shards: {sorted(duplicates)}")
        seen_tensors.update(tensors)
        n_params = 0
        for spec in tensors.values():
            n = 1
            for d in spec["shape"]:
                n *= d
            n_params += n
            dtype_hist[spec["dtype"]] = dtype_hist.get(spec["dtype"], 0) + 1
        shards.append(ShardRecord(file=basename, sha256=shard_digest(path),
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
        tokenizer_digest=shard_digest(tokenizer_path) if tokenizer_path else "",
        license=license or str(config.get("license", "") or ""),
        source=source,
    )
