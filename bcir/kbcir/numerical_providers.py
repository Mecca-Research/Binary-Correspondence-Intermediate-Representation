"""B5 workload-scoped numerical providers and measured evidence.

Provider availability/performance is planning evidence only.  A vendor library never
defines BCIR legality, and a provider is not probed or selected until a concrete workload
asks for one of its declared capabilities.
"""

from __future__ import annotations

import ctypes.util
import hashlib
import json
import math
import os
import platform
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .._artifact_json import read_bounded_text, strict_json_loads
from ..silicon import CounterSampler, read_hw_counters


_OPERATIONS = frozenset({"gemm", "fft", "linsolve", "special", "vecmath"})
_DTYPES = frozenset({"float32", "float64", "complex64", "complex128"})


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _positive(value, field: str, *, zero: bool = False) -> int:
    minimum = 0 if zero else 1
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class NumericalWorkload:
    operation: str
    shape: tuple[int, ...]
    dtype: str
    accuracy_ulp: int

    def __post_init__(self) -> None:
        if self.operation not in _OPERATIONS or self.dtype not in _DTYPES:
            raise ValueError("unknown numerical operation or dtype")
        if (
            not isinstance(self.shape, (tuple, list))
            or not self.shape
            or any(type(dim) is not int or dim < 1 for dim in self.shape)
        ):
            raise ValueError("numerical workload shape must contain positive integers")
        object.__setattr__(self, "shape", tuple(self.shape))
        _positive(self.accuracy_ulp, "accuracy_ulp", zero=True)

    @property
    def digest(self) -> str:
        return hashlib.sha256(_canonical(asdict(self)).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LibraryProviderDescriptor:
    name: str
    logical_libraries: tuple[str, ...]
    capabilities: tuple[str, ...]
    dtypes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("provider name must be nonempty")
        if not isinstance(self.logical_libraries, (tuple, list)) or any(
            not isinstance(item, str) or not item for item in self.logical_libraries
        ):
            raise ValueError("logical_libraries must contain names")
        if (
            not isinstance(self.capabilities, (tuple, list))
            or not self.capabilities
            or any(item not in _OPERATIONS for item in self.capabilities)
        ):
            raise ValueError("provider capabilities are invalid")
        if (
            not isinstance(self.dtypes, (tuple, list))
            or not self.dtypes
            or any(item not in _DTYPES for item in self.dtypes)
        ):
            raise ValueError("provider dtypes are invalid")

    def supports(self, workload: NumericalWorkload) -> bool:
        return workload.operation in self.capabilities and workload.dtype in self.dtypes


@dataclass(frozen=True)
class ProviderMeasurement:
    workload_sha256: str
    wall_ns: int
    cpu_ns: int
    cycles: int
    instructions: int
    cache_misses: int
    repeats: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workload_sha256, str)
            or len(self.workload_sha256) != 64
            or self.workload_sha256 != self.workload_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.workload_sha256)
        ):
            raise ValueError("workload_sha256 is invalid")
        for field in ("wall_ns", "cpu_ns", "cycles", "instructions", "cache_misses"):
            _positive(getattr(self, field), field, zero=True)
        _positive(self.repeats, "repeats")
        if self.wall_ns == 0:
            raise ValueError("provider measurement wall time must be positive")


@dataclass(frozen=True)
class ProviderEvidence:
    provider: str
    available: bool
    detail: str
    measurements: tuple[ProviderMeasurement, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider, str)
            or not self.provider
            or type(self.available) is not bool
            or not isinstance(self.detail, str)
        ):
            raise ValueError("provider evidence identity is malformed")
        if not isinstance(self.measurements, (tuple, list)) or any(
            not isinstance(item, ProviderMeasurement) for item in self.measurements
        ):
            raise ValueError("provider measurements are malformed")
        object.__setattr__(self, "measurements", tuple(self.measurements))
        if not self.available and self.measurements:
            raise ValueError("unavailable provider cannot carry measurements")


class ExecutableNumericalProvider(Protocol):
    descriptor: LibraryProviderDescriptor

    def run(self, workload: NumericalWorkload): ...


def default_provider_descriptors() -> tuple[LibraryProviderDescriptor, ...]:
    return (
        LibraryProviderDescriptor(
            "cblas", ("cblas", "openblas", "blas"), ("gemm",), ("float32", "float64")
        ),
        LibraryProviderDescriptor("fftw", ("fftw3",), ("fft",), ("float64", "complex128")),
        LibraryProviderDescriptor("lapack", ("lapack",), ("linsolve",), ("float32", "float64")),
        LibraryProviderDescriptor("gsl", ("gsl",), ("special",), ("float64",)),
        LibraryProviderDescriptor("sleef", ("sleef",), ("vecmath",), ("float32", "float64")),
        LibraryProviderDescriptor("libcerf", ("cerf",), ("special",), ("float64", "complex128")),
    )


def probe_linked_provider(descriptor: LibraryProviderDescriptor) -> ProviderEvidence:
    if not isinstance(descriptor, LibraryProviderDescriptor):
        raise ValueError("descriptor must be a LibraryProviderDescriptor")
    found = [(name, ctypes.util.find_library(name)) for name in descriptor.logical_libraries]
    available = next(((name, path) for name, path in found if path), None)
    return ProviderEvidence(
        descriptor.name,
        available is not None,
        f"{available[0]}={available[1]}" if available else "not found",
    )


def measure_provider(
    provider: ExecutableNumericalProvider, workload: NumericalWorkload, *, repeats: int = 3
) -> ProviderEvidence:
    descriptor = getattr(provider, "descriptor", None)
    if not isinstance(descriptor, LibraryProviderDescriptor) or not descriptor.supports(workload):
        raise ValueError("provider does not support the requested workload")
    _positive(repeats, "repeats")
    provider.run(workload)  # bounded warmup
    laps = []
    for _ in range(repeats):
        sampler = CounterSampler()
        provider.run(workload)
        laps.append(sampler.lap())
    hardware = read_hw_counters(lambda: provider.run(workload))
    measurement = ProviderMeasurement(
        workload.digest,
        int(statistics.median(lap.wall_ns for lap in laps)),
        int(statistics.median(lap.cpu_ns for lap in laps)),
        hardware.cycles if hardware else 0,
        hardware.instructions if hardware else 0,
        hardware.cache_misses if hardware else 0,
        repeats,
    )
    return ProviderEvidence(descriptor.name, True, "executed", (measurement,))


@dataclass(frozen=True)
class ProviderSelection:
    workload_sha256: str
    provider: str
    measured_wall_ns: int
    legality_independent: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.workload_sha256, str)
            or len(self.workload_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.workload_sha256)
        ):
            raise ValueError("provider selection workload digest is invalid")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("provider selection name must be nonempty")
        _positive(self.measured_wall_ns, "measured_wall_ns")
        if not self.legality_independent:
            raise ValueError("numerical provider selection may not define legality")


class NumericalProviderRegistry:
    def __init__(self):
        self._descriptors = {}

    def register(self, descriptor: LibraryProviderDescriptor) -> None:
        if not isinstance(descriptor, LibraryProviderDescriptor):
            raise ValueError("descriptor must be a LibraryProviderDescriptor")
        if descriptor.name in self._descriptors:
            raise ValueError(f"duplicate numerical provider {descriptor.name!r}")
        self._descriptors[descriptor.name] = descriptor

    def candidates(self, workload: NumericalWorkload) -> tuple[LibraryProviderDescriptor, ...]:
        if not isinstance(workload, NumericalWorkload):
            raise ValueError("workload must be a NumericalWorkload")
        return tuple(
            descriptor
            for _name, descriptor in sorted(self._descriptors.items())
            if descriptor.supports(workload)
        )

    def select(self, workload: NumericalWorkload, evidence) -> ProviderSelection:
        candidates = {descriptor.name for descriptor in self.candidates(workload)}
        rows = []
        for item in evidence:
            if not isinstance(item, ProviderEvidence):
                raise ValueError("evidence contains an invalid record")
            if item.provider not in candidates or not item.available:
                continue
            matching = [
                measurement
                for measurement in item.measurements
                if measurement.workload_sha256 == workload.digest
            ]
            if matching:
                rows.append((min(measurement.wall_ns for measurement in matching), item.provider))
        if not rows:
            raise LookupError("no measured available provider supports the workload")
        wall, provider = min(rows, key=lambda row: (row[0], row[1]))
        return ProviderSelection(workload.digest, provider, wall)


@dataclass(frozen=True)
class NumericalEvidenceArtifact:
    host_architecture: str
    evidence: tuple[ProviderEvidence, ...]
    schema: str = "bcir.numerical_provider_evidence.v1"

    def __post_init__(self) -> None:
        if (
            self.schema != "bcir.numerical_provider_evidence.v1"
            or not isinstance(self.host_architecture, str)
            or not self.host_architecture
            or not isinstance(self.evidence, (tuple, list))
            or any(not isinstance(item, ProviderEvidence) for item in self.evidence)
        ):
            raise ValueError("numerical evidence artifact is malformed")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_json(self) -> str:
        return _canonical(
            {
                "schema": self.schema,
                "host_architecture": self.host_architecture,
                "evidence": [asdict(item) for item in self.evidence],
            }
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


def write_numerical_evidence(path: os.PathLike | str, artifact: NumericalEvidenceArtifact) -> None:
    if not isinstance(artifact, NumericalEvidenceArtifact):
        raise ValueError("artifact must be NumericalEvidenceArtifact")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(artifact.to_json().encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_numerical_evidence(path: os.PathLike | str) -> NumericalEvidenceArtifact:
    value = strict_json_loads(
        read_bounded_text(str(path), "numerical evidence"), "numerical evidence"
    )
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "host_architecture", "evidence"}
        or not isinstance(value["evidence"], list)
    ):
        raise ValueError("numerical evidence has missing or unknown fields")
    rows = []
    try:
        for item in value["evidence"]:
            measurements = tuple(
                ProviderMeasurement(**measurement) for measurement in item.pop("measurements")
            )
            rows.append(ProviderEvidence(measurements=measurements, **item))
        return NumericalEvidenceArtifact(value["host_architecture"], tuple(rows), value["schema"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(f"malformed numerical evidence: {exc}") from exc


def probe_default_numerical_environment() -> NumericalEvidenceArtifact:
    return NumericalEvidenceArtifact(
        platform.machine() or "unknown",
        tuple(probe_linked_provider(descriptor) for descriptor in default_provider_descriptors()),
    )
