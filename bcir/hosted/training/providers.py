"""Provider-neutral, offline-first teacher and remote-compute contracts.

An inference/embedding provider is not a gradient source.  It may produce immutable
targets that a later local training stage consumes.  A remote-compute provider instead
runs BCIR-owned code and returns BCIR-owned artifacts.  Keeping these interfaces separate
prevents an opaque embedding API from being mistaken for outsourced backpropagation.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from ..._artifact_json import strict_json_loads
from .contracts import canonical_json, require_int, require_sha256, sha256_text


_OPERATIONS = frozenset({"embedding", "preference", "score", "label"})
_MAX_RECORDING_BYTES = 256 * 1024 * 1024
_MAX_RECORD_BYTES = 16 * 1024 * 1024


def _validate_payload(value, *, depth: int = 0) -> None:
    if depth > 16:
        raise ValueError("provider payload nesting exceeds 16")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if value.bit_length() > 4096:
            raise ValueError("provider payload integer exceeds 4096 bits")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("provider payload string exceeds 16 MiB")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("provider payload contains a non-finite float")
        return
    if isinstance(value, list):
        if len(value) > 1_000_000:
            raise ValueError("provider payload list is too large")
        for item in value:
            _validate_payload(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > 100_000 or any(not isinstance(key, str) for key in value):
            raise ValueError("provider payload object is too large or has a non-string key")
        for item in value.values():
            _validate_payload(item, depth=depth + 1)
        return
    raise ValueError(f"provider payload type {type(value).__name__} is unsupported")


@dataclass(frozen=True)
class TeacherRequest:
    operation: str
    input_sha256: str
    payload: dict
    schema: str = "bcir.teacher_request.v1"

    def __post_init__(self) -> None:
        if self.schema != "bcir.teacher_request.v1" or self.operation not in _OPERATIONS:
            raise ValueError("unsupported teacher request schema or operation")
        require_sha256(self.input_sha256, "input_sha256")
        if not isinstance(self.payload, dict):
            raise ValueError("teacher request payload must be an object")
        _validate_payload(self.payload)
        if len(canonical_json(self.payload).encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("teacher request payload exceeds 16 MiB")

    @property
    def request_id(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


@dataclass(frozen=True)
class TeacherResponse:
    request_id: str
    provider: str
    model_revision: str
    payload: dict
    source_sha256: str
    schema: str = "bcir.teacher_response.v1"

    def __post_init__(self) -> None:
        if self.schema != "bcir.teacher_response.v1":
            raise ValueError("unsupported teacher response schema")
        require_sha256(self.request_id, "request_id")
        require_sha256(self.source_sha256, "source_sha256")
        for field in ("provider", "model_revision"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"teacher response {field} must be nonempty")
        if not isinstance(self.payload, dict):
            raise ValueError("teacher response payload must be an object")
        _validate_payload(self.payload)
        if len(canonical_json(self.payload).encode("utf-8")) > 16 * 1024 * 1024:
            raise ValueError("teacher response payload exceeds 16 MiB")

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


class TeacherProvider(Protocol):
    name: str

    def complete(self, request: TeacherRequest) -> TeacherResponse: ...


class RecordedTeacherProvider:
    """Serve content-addressed responses from a frozen recording; never use a network."""

    name = "recorded"

    def __init__(self, responses):
        if not isinstance(responses, (tuple, list)):
            raise ValueError("responses must be a sequence")
        table = {}
        for index, response in enumerate(responses):
            if not isinstance(response, TeacherResponse):
                raise ValueError(f"response {index} is not a TeacherResponse")
            if response.request_id in table:
                raise ValueError(f"duplicate recorded request {response.request_id}")
            table[response.request_id] = response
        self._responses = table

    def complete(self, request: TeacherRequest) -> TeacherResponse:
        if not isinstance(request, TeacherRequest):
            raise ValueError("request must be a TeacherRequest")
        try:
            return self._responses[request.request_id]
        except KeyError as exc:
            raise LookupError(f"no frozen teacher response for {request.request_id}") from exc

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "RecordedTeacherProvider":
        source = Path(path)
        if not source.is_file():
            raise ValueError("teacher recording must be a regular file")
        responses = []
        try:
            with source.open("rb") as stream:
                total = 0; index = 0
                while True:
                    raw = stream.readline(_MAX_RECORD_BYTES + 1)
                    if not raw:
                        break
                    index += 1; total += len(raw)
                    if total > _MAX_RECORDING_BYTES:
                        raise ValueError("teacher recording exceeds 256 MiB")
                    if len(raw) > _MAX_RECORD_BYTES:
                        raise ValueError(f"teacher recording line {index} is too large")
                    try:
                        line = raw.decode("utf-8")
                        value = strict_json_loads(
                            line, "teacher recording line", max_bytes=_MAX_RECORD_BYTES)
                        responses.append(TeacherResponse(**value))
                    except (TypeError, UnicodeError, ValueError) as exc:
                        raise ValueError(
                            f"invalid teacher recording line {index}: {exc}") from exc
        except OSError as exc:
            raise ValueError(f"cannot read teacher recording: {exc}") from exc
        return cls(responses)


@dataclass(frozen=True)
class ArtifactFile:
    name: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or Path(self.name).name != self.name:
            raise ValueError("artifact file name must be a plain basename")
        require_int(self.bytes, "artifact bytes", maximum=1 << 50)
        require_sha256(self.sha256, "artifact sha256")


@dataclass(frozen=True)
class RemoteTrainingBundle:
    source_commit: str
    train_spec_sha256: str
    corpus_sha256: str
    tokenizer_sha256: str
    dependency_lock_sha256: str
    container_sha256: str
    seed: int
    inputs: tuple[ArtifactFile, ...] = ()
    schema: str = "bcir.remote_training_bundle.v1"

    def __post_init__(self) -> None:
        if self.schema != "bcir.remote_training_bundle.v1" \
                or not isinstance(self.source_commit, str) \
                or not 7 <= len(self.source_commit) <= 64 \
                or self.source_commit != self.source_commit.lower() \
                or any(character not in "0123456789abcdef" for character in self.source_commit):
            raise ValueError("remote bundle has an invalid schema or source commit")
        for field in ("train_spec_sha256", "corpus_sha256", "tokenizer_sha256",
                      "dependency_lock_sha256", "container_sha256"):
            require_sha256(getattr(self, field), field)
        require_int(self.seed, "seed")
        if not isinstance(self.inputs, (tuple, list)) \
                or any(not isinstance(item, ArtifactFile) for item in self.inputs):
            raise ValueError("remote bundle inputs must be ArtifactFile values")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError("remote bundle contains duplicate input names")

    @property
    def digest(self) -> str:
        value = asdict(self); value["inputs"] = [asdict(item) for item in self.inputs]
        return sha256_text(canonical_json(value))


@dataclass(frozen=True)
class RemoteTrainingResult:
    bundle_sha256: str
    run_report_sha256: str
    outputs: tuple[ArtifactFile, ...]
    executor: str
    schema: str = "bcir.remote_training_result.v1"

    def __post_init__(self) -> None:
        if self.schema != "bcir.remote_training_result.v1":
            raise ValueError("unsupported remote result schema")
        require_sha256(self.bundle_sha256, "bundle_sha256")
        require_sha256(self.run_report_sha256, "run_report_sha256")
        if not isinstance(self.executor, str) or not self.executor:
            raise ValueError("remote executor must be nonempty")
        if not isinstance(self.outputs, (tuple, list)) or not self.outputs \
                or any(not isinstance(item, ArtifactFile) for item in self.outputs):
            raise ValueError("remote result outputs must be nonempty ArtifactFile values")
        names = [item.name for item in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError("remote result contains duplicate output names")


class RemoteComputeProvider(Protocol):
    name: str

    def run(self, bundle: RemoteTrainingBundle) -> RemoteTrainingResult: ...


class OfflineComputeAdapter:
    """Synchronous test adapter for a BCIR-owned executor callback.

    It models the remote trust boundary without spawning processes or making network
    calls.  Future cloud adapters must return the same immutable result contract.
    """

    name = "offline"

    def __init__(self, executor: Callable[[RemoteTrainingBundle], RemoteTrainingResult]):
        if not callable(executor):
            raise ValueError("executor must be callable")
        self._executor = executor

    def run(self, bundle: RemoteTrainingBundle) -> RemoteTrainingResult:
        if not isinstance(bundle, RemoteTrainingBundle):
            raise ValueError("bundle must be a RemoteTrainingBundle")
        result = self._executor(bundle)
        if not isinstance(result, RemoteTrainingResult) or result.bundle_sha256 != bundle.digest:
            raise ValueError("offline executor returned an unattested result")
        return result


def relational_embedding_targets(vectors) -> tuple[tuple[float, ...], ...]:
    """Return a normalized cosine-similarity matrix for embedding distillation.

    The matrix is basis-independent, unlike direct coordinate regression against an
    opaque provider embedding space.  It is a frozen target, not a gradient channel.
    """
    if not isinstance(vectors, (tuple, list)) or not vectors:
        raise ValueError("vectors must be a nonempty sequence")
    rows = []
    width = None
    for row in vectors:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError("every embedding must be a nonempty sequence")
        values = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("embedding contains a non-finite value")
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError("embedding widths differ")
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise ValueError("zero embedding cannot define cosine targets")
        rows.append(tuple(value / norm for value in values))
    return tuple(tuple(sum(a * b for a, b in zip(left, right)) for right in rows)
                 for left in rows)
