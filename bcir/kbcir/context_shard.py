"""Content-addressed context-shard manifests and quiescent activation.

The shard envelope references an existing BCIR/Safetensors artifact; it is not a new
tensor format and it never embeds opaque executable code.  Teacher providers and remote
training adapters may produce candidate artifacts, but only an exact catalog match plus
provenance/certificate binding can activate one, and activation is restricted to a
quiescent generation boundary with an explicit rollback identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .._artifact_json import read_bounded_text, strict_json_loads
from .ham import HAMResource

_MAX_JSON = 16 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 1 << 50
_MANIFEST_SCHEMA = "bcir.context_shard.v1"
_CATALOG_SCHEMA = "bcir.context_shard_catalog.v1"
_ACTIVATION_SCHEMA = "bcir.context_shard_activation.v1"
_KINDS = frozenset(("bcirq8", "q8-table", "lora", "weight-delta", "execution-plan"))
_FORMATS = frozenset(("bcirq8-v1", "bcir-q8-table", "safetensors", "bcir-streampack"))
_CLASSIFICATIONS = frozenset(("exact", "quantized", "approximate"))


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _digest(value, field: str, *, empty: bool = False) -> str:
    if empty and value == "":
        return value
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _name(value, field: str, *, choices=None) -> str:
    if (not isinstance(value, str) or not value or len(value.encode("utf-8")) > 4096
            or any(ord(character) < 0x20 for character in value)):
        raise ValueError(f"{field} must be a bounded, control-free string")
    if choices is not None and value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _integer(value, field: str, *, minimum: int = 0, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True)
class ContextShardManifest:
    shard_kind: str
    tensor_format: str
    classification: str
    payload_sha256: str
    payload_bytes: int
    base_model_sha256: str
    target_hardware_sha256: str
    selector_sha256: str
    provenance_sha256: str
    certificate_sha256: str
    source_revision: str
    generation: int
    lora_rank: int = 0
    schema: str = _MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _MANIFEST_SCHEMA:
            raise ValueError("unsupported context-shard schema")
        _name(self.shard_kind, "context shard kind", choices=_KINDS)
        _name(self.tensor_format, "context shard tensor_format", choices=_FORMATS)
        _name(self.classification, "context shard classification",
              choices=_CLASSIFICATIONS)
        for value, field in (
                (self.payload_sha256, "payload_sha256"),
                (self.base_model_sha256, "base_model_sha256"),
                (self.target_hardware_sha256, "target_hardware_sha256"),
                (self.selector_sha256, "selector_sha256"),
                (self.provenance_sha256, "provenance_sha256"),
                (self.certificate_sha256, "certificate_sha256")):
            _digest(value, f"context shard {field}")
        _integer(self.payload_bytes, "context shard payload_bytes", minimum=1,
                 maximum=_MAX_PAYLOAD_BYTES)
        _name(self.source_revision, "context shard source_revision")
        _integer(self.generation, "context shard generation", minimum=1)
        _integer(self.lora_rank, "context shard lora_rank", maximum=1 << 20)
        expected_format = {
            "bcirq8": "bcirq8-v1", "q8-table": "bcir-q8-table",
            "lora": "safetensors", "weight-delta": "safetensors",
            "execution-plan": "bcir-streampack"}[self.shard_kind]
        if self.tensor_format != expected_format:
            raise ValueError(f"context shard {self.shard_kind!r} requires "
                             f"tensor_format {expected_format!r}")
        if (self.shard_kind == "lora") != (self.lora_rank > 0):
            raise ValueError("only LoRA context shards carry a positive lora_rank")
        if self.shard_kind == "bcirq8" and self.classification != "quantized":
            raise ValueError("BCIRQ8 context shards must be classified as quantized")
        if self.shard_kind in {"q8-table", "execution-plan"} \
                and self.classification != "exact":
            raise ValueError("Q8-table and execution-plan context shards must be exact")

    def to_dict(self) -> dict:
        return {"schema": self.schema, **{key: value for key, value in asdict(self).items()
                                          if key != "schema"}}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    def to_ham_resource(self, resource_id: int, home_bank: str, *,
                        dependencies: tuple[int, ...] = (), priority: int = 0) -> HAMResource:
        kind = {"bcirq8": "model", "q8-table": "adapter", "lora": "adapter",
                "weight-delta": "adapter", "execution-plan": "program"}[self.shard_kind]
        return HAMResource(resource_id, f"context-shard:{self.digest[:16]}", kind,
                           self.payload_bytes, home_bank, self.payload_sha256,
                           dependencies, False, priority)

    @classmethod
    def from_json(cls, text: str) -> "ContextShardManifest":
        doc = strict_json_loads(text, "context-shard manifest", max_bytes=_MAX_JSON)
        fields = {"schema", "shard_kind", "tensor_format", "classification",
                  "payload_sha256", "payload_bytes", "base_model_sha256",
                  "target_hardware_sha256", "selector_sha256", "provenance_sha256",
                  "certificate_sha256", "source_revision", "generation", "lora_rank"}
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError("context-shard manifest has missing or unknown fields")
        try:
            return cls(**doc)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed context-shard manifest: {exc}") from exc


@dataclass(frozen=True)
class ContextShardEntry:
    selector_sha256: str
    shard_sha256: str
    priority: int = 0

    def __post_init__(self) -> None:
        _digest(self.selector_sha256, "context entry selector_sha256")
        _digest(self.shard_sha256, "context entry shard_sha256")
        _integer(self.priority, "context entry priority", maximum=255)


@dataclass(frozen=True)
class ContextShardCatalog:
    base_model_sha256: str
    target_hardware_sha256: str
    generation: int
    entries: tuple[ContextShardEntry, ...]
    schema: str = _CATALOG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _CATALOG_SCHEMA:
            raise ValueError("unsupported context-shard catalog schema")
        _digest(self.base_model_sha256, "context catalog base_model_sha256")
        _digest(self.target_hardware_sha256, "context catalog target_hardware_sha256")
        _integer(self.generation, "context catalog generation", minimum=1)
        if (not isinstance(self.entries, tuple) or not self.entries or len(self.entries) > 65536
                or any(not isinstance(row, ContextShardEntry) for row in self.entries)):
            raise ValueError("context catalog needs a bounded typed entry inventory")
        key = lambda row: (row.selector_sha256, -row.priority, row.shard_sha256)
        if tuple(sorted(self.entries, key=key)) != self.entries:
            raise ValueError("context catalog entries must use canonical selector/priority order")
        if len({(row.selector_sha256, row.shard_sha256) for row in self.entries}) \
                != len(self.entries):
            raise ValueError("context catalog contains a duplicate selector/shard entry")

    def resolve(self, selector_sha256: str) -> ContextShardEntry:
        _digest(selector_sha256, "context selector")
        for row in self.entries:
            if row.selector_sha256 == selector_sha256:
                return row
        raise LookupError(f"no context shard for selector {selector_sha256}")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "base_model_sha256": self.base_model_sha256,
                "target_hardware_sha256": self.target_hardware_sha256,
                "generation": self.generation,
                "entries": [asdict(row) for row in self.entries]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ContextShardCatalog":
        doc = strict_json_loads(text, "context-shard catalog", max_bytes=_MAX_JSON)
        fields = {"schema", "base_model_sha256", "target_hardware_sha256",
                  "generation", "entries"}
        if not isinstance(doc, dict) or set(doc) != fields \
                or not isinstance(doc["entries"], list) or not doc["entries"] \
                or len(doc["entries"]) > 65536:
            raise ValueError("context-shard catalog has missing, unknown, or unbounded fields")
        try:
            entries = tuple(ContextShardEntry(**row) for row in doc.pop("entries"))
            return cls(**doc, entries=entries)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed context-shard catalog: {exc}") from exc


@dataclass(frozen=True)
class ContextShardActivation:
    catalog_sha256: str
    selector_sha256: str
    shard_sha256: str
    previous_shard_sha256: str
    activation_generation: int
    rollback_token_sha256: str
    schema: str = _ACTIVATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _ACTIVATION_SCHEMA:
            raise ValueError("unsupported context-shard activation schema")
        _digest(self.catalog_sha256, "activation catalog_sha256")
        _digest(self.selector_sha256, "activation selector_sha256")
        _digest(self.shard_sha256, "activation shard_sha256")
        _digest(self.previous_shard_sha256, "activation previous_shard_sha256", empty=True)
        _integer(self.activation_generation, "activation generation", minimum=1)
        _digest(self.rollback_token_sha256, "activation rollback_token_sha256")

    def to_json(self) -> str:
        return _canonical({"schema": self.schema, **{key: value for key, value in
                                                      asdict(self).items()
                                                      if key != "schema"}})

    @property
    def digest(self) -> str:
        return _sha(json.loads(self.to_json()))

    @classmethod
    def from_json(cls, text: str) -> "ContextShardActivation":
        doc = strict_json_loads(text, "context-shard activation", max_bytes=_MAX_JSON)
        fields = {"schema", "catalog_sha256", "selector_sha256", "shard_sha256",
                  "previous_shard_sha256", "activation_generation",
                  "rollback_token_sha256"}
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError("context-shard activation has missing or unknown fields")
        try:
            return cls(**doc)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"malformed context-shard activation: {exc}") from exc


def certify_context_activation(catalog: ContextShardCatalog,
                               manifest: ContextShardManifest,
                               selector_sha256: str, *, current_shard_sha256: str = "",
                               activation_generation: int,
                               quiescent: bool) -> ContextShardActivation:
    """Bind one catalog decision at a quiescent boundary; preserve rollback identity."""
    if not isinstance(catalog, ContextShardCatalog) \
            or not isinstance(manifest, ContextShardManifest):
        raise ValueError("context activation requires typed catalog and manifest")
    if type(quiescent) is not bool or not quiescent:
        raise ValueError("context-shard activation requires a quiescent generation boundary")
    _digest(current_shard_sha256, "current context shard", empty=True)
    _integer(activation_generation, "activation generation", minimum=1)
    if activation_generation <= catalog.generation:
        raise ValueError("activation generation must advance beyond the catalog generation")
    if (manifest.base_model_sha256 != catalog.base_model_sha256
            or manifest.target_hardware_sha256 != catalog.target_hardware_sha256):
        raise ValueError("context shard is not bound to the catalog model/hardware")
    if manifest.generation > catalog.generation:
        raise ValueError("context shard was authored after the catalog generation")
    entry = catalog.resolve(selector_sha256)
    if entry.shard_sha256 != manifest.digest or entry.selector_sha256 != manifest.selector_sha256:
        raise ValueError("context catalog selection does not match the shard manifest")
    rollback = _sha({"catalog": catalog.digest, "selector": selector_sha256,
                     "previous": current_shard_sha256,
                     "next": manifest.digest, "generation": activation_generation})
    return ContextShardActivation(catalog.digest, selector_sha256, manifest.digest,
                                  current_shard_sha256, activation_generation, rollback)


def _atomic_write(path: str | os.PathLike, text: str) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_context_shard_manifest(path: str | os.PathLike,
                                 manifest: ContextShardManifest) -> None:
    if not isinstance(manifest, ContextShardManifest):
        raise ValueError("context-shard writer requires a manifest")
    _atomic_write(path, manifest.to_json())


def read_context_shard_manifest(path: str | os.PathLike) -> ContextShardManifest:
    return ContextShardManifest.from_json(read_bounded_text(
        os.fspath(path), "context-shard manifest", max_bytes=_MAX_JSON))


def write_context_shard_catalog(path: str | os.PathLike,
                                catalog: ContextShardCatalog) -> None:
    if not isinstance(catalog, ContextShardCatalog):
        raise ValueError("context-shard writer requires a catalog")
    _atomic_write(path, catalog.to_json())


def read_context_shard_catalog(path: str | os.PathLike) -> ContextShardCatalog:
    return ContextShardCatalog.from_json(read_bounded_text(
        os.fspath(path), "context-shard catalog", max_bytes=_MAX_JSON))


def verify_context_payload(manifest: ContextShardManifest,
                           path: str | os.PathLike) -> None:
    """Stream and verify an artifact only when a deployment explicitly requests it."""
    if not isinstance(manifest, ContextShardManifest):
        raise ValueError("payload verification requires a context-shard manifest")
    digest = hashlib.sha256(); count = 0
    try:
        with open(path, "rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > manifest.payload_bytes:
                    raise ValueError("context-shard payload is larger than its manifest")
                digest.update(block)
    except OSError as exc:
        raise ValueError(f"cannot read context-shard payload: {exc}") from exc
    if count != manifest.payload_bytes or digest.hexdigest() != manifest.payload_sha256:
        raise ValueError("context-shard payload size or SHA-256 does not match its manifest")


__all__ = ["ContextShardActivation", "ContextShardCatalog", "ContextShardEntry",
           "ContextShardManifest", "certify_context_activation",
           "read_context_shard_catalog", "read_context_shard_manifest",
           "verify_context_payload", "write_context_shard_catalog",
           "write_context_shard_manifest"]
