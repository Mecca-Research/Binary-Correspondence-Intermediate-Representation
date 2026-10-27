"""B5 workload-scoped provider and measured-evidence gates."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from bcir.kbcir.numerical_providers import (
    LibraryProviderDescriptor,
    NumericalEvidenceArtifact,
    NumericalProviderRegistry,
    NumericalWorkload,
    ProviderEvidence,
    ProviderMeasurement,
    default_provider_descriptors,
    measure_provider,
    probe_default_numerical_environment,
    read_numerical_evidence,
    write_numerical_evidence,
)


class _Provider:
    def __init__(self, name, work):
        self.descriptor = LibraryProviderDescriptor(name, (name,), ("gemm",), ("float32",))
        self.work = work

    def run(self, workload):
        value = 0
        for index in range(self.work):
            value += index * index
        return value + sum(workload.shape)


def test_provider_registry_selects_only_measured_workload_capabilities():
    workload = NumericalWorkload("gemm", (4, 4, 4), "float32", 2)
    fast = LibraryProviderDescriptor("fast", ("fast",), ("gemm",), ("float32",))
    slow = LibraryProviderDescriptor("slow", ("slow",), ("gemm",), ("float32",))
    fft = LibraryProviderDescriptor("fft", ("fft",), ("fft",), ("complex128",))
    registry = NumericalProviderRegistry()
    for descriptor in (fast, slow, fft):
        registry.register(descriptor)
    assert [item.name for item in registry.candidates(workload)] == ["fast", "slow"]
    evidence = (
        ProviderEvidence(
            "slow", True, "fixture", (ProviderMeasurement(workload.digest, 200, 190, 0, 0, 0, 2),)
        ),
        ProviderEvidence(
            "fast", True, "fixture", (ProviderMeasurement(workload.digest, 100, 90, 0, 0, 0, 2),)
        ),
        ProviderEvidence("fft", True, "irrelevant"),
    )
    selection = registry.select(workload, evidence)
    assert selection.provider == "fast" and selection.legality_independent


def test_provider_measurement_uses_real_bounded_counters():
    workload = NumericalWorkload("gemm", (2, 2, 2), "float32", 1)
    evidence = measure_provider(_Provider("fixture", 200), workload, repeats=2)
    assert evidence.available and evidence.detail == "executed"
    measurement = evidence.measurements[0]
    assert measurement.wall_ns > 0 and measurement.repeats == 2
    try:
        replace(measurement, workload_sha256="z" * 64)
        assert False, "non-hex workload digest was accepted"
    except ValueError:
        pass


def test_default_library_inventory_is_explicit_and_probeable():
    descriptors = default_provider_descriptors()
    assert {item.name for item in descriptors} == {
        "cblas",
        "fftw",
        "lapack",
        "gsl",
        "sleef",
        "libcerf",
    }
    artifact = probe_default_numerical_environment()
    assert len(artifact.evidence) == len(descriptors)
    assert all(item.detail for item in artifact.evidence)


def test_numerical_evidence_artifact_roundtrips_deterministically():
    workload = NumericalWorkload("gemm", (2, 2, 2), "float32", 1)
    row = ProviderEvidence(
        "fixture", True, "measured", (ProviderMeasurement(workload.digest, 100, 90, 10, 20, 1, 2),)
    )
    artifact = NumericalEvidenceArtifact("test-arch", (row,))
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "providers.json"
        write_numerical_evidence(path, artifact)
        assert read_numerical_evidence(path) == artifact
        assert read_numerical_evidence(path).digest == artifact.digest


def test_a_pmu_sample_missing_a_count_is_recorded_as_no_counts():
    """A provider measurement carries the PMU's counts only when the sample read all three
    (`silicon.pmu_counts`, the one test every evidence record makes); a sample missing one is
    recorded as the record's no-PMU zeros -- never two counts and a gap."""
    from unittest import mock

    from bcir.kbcir import numerical_providers as providers
    from bcir.silicon import HwSample

    workload = NumericalWorkload("gemm", (2, 2, 2), "float32", 1)
    for sample, want in (
        (HwSample(cycles=2_000, instructions=3_000, cache_misses=None), (0, 0, 0)),
        (HwSample(cycles=None, instructions=None, cache_misses=None), (0, 0, 0)),
        (HwSample(cycles=2_000, instructions=3_000, cache_misses=7), (2_000, 3_000, 7)),
    ):

        def counted(work, sample=sample):
            work()
            return sample

        with mock.patch.object(providers, "read_hw_counters", counted):
            measurement = measure_provider(
                _Provider("fixture", 50), workload, repeats=1
            ).measurements[0]
        assert (measurement.cycles, measurement.instructions, measurement.cache_misses) == want
