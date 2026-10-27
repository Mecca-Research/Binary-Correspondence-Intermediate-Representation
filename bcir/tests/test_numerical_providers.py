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


def test_a_provider_measurement_records_a_pmu_sample_missing_a_count_as_null():
    """A provider measurement carries the PMU's counts only when the sample read all three
    (`silicon.pmu_counts`, the one test every evidence record makes); a sample missing one is
    recorded as three nulls (EVIDENCE-NULL) -- never two counts and a gap, never a 0."""
    from unittest import mock

    from bcir.kbcir import numerical_providers as providers
    from bcir.silicon import HwSample

    workload = NumericalWorkload("gemm", (2, 2, 2), "float32", 1)
    for sample, want in (
        (HwSample(cycles=2_000, instructions=3_000, cache_misses=None), (None, None, None)),
        (HwSample(cycles=None, instructions=None, cache_misses=None), (None, None, None)),
        (HwSample(cycles=2_000, instructions=3_000, cache_misses=0), (2_000, 3_000, 0)),
    ):

        def counted(work, sample=sample):
            work()
            return sample

        with mock.patch.object(providers, "read_hw_counters", counted):
            measurement = measure_provider(
                _Provider("fixture", 50), workload, repeats=1
            ).measurements[0]
        assert (measurement.cycles, measurement.instructions, measurement.cache_misses) == want


def _refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


# The bytes of a v1 artifact as the v1 writer wrote them (no PMU, so its zeros), and their digest.
_V1_EVIDENCE = (
    '{"evidence":[{"available":true,"detail":"executed","measurements":[{"cache_misses":0,'
    '"cpu_ns":900,"cycles":0,"instructions":0,"repeats":1,"wall_ns":1000,'
    '"workload_sha256":"81542931d9f6af5703fbe813e0ccf74b20234773d1ac4b2aea71017d2b3bf6f4"}],'
    '"provider":"cblas"}],"host_architecture":"x86_64",'
    '"schema":"bcir.numerical_provider_evidence.v1"}'
)
_V1_EVIDENCE_DIGEST = "7aaf4c460ee740a33b97abca690f5756d58eef7704646b87d814ae810a1c0592"


def test_provider_evidence_says_null_and_a_v1_artifact_reads_as_it_was_written():
    """v2 says null: three nulls round-trip and differ in the digest from three zeros the PMU
    read, and a gap or a count past 2^63 is refused. v1 still reads, to its bytes and its
    digest, and cannot say null."""
    workload = NumericalWorkload("gemm", (2, 2, 2), "float32", 1)

    def artifact(*counts):
        measurement = ProviderMeasurement(workload.digest, 100, 90, *counts, 2)
        return NumericalEvidenceArtifact(
            "x86_64", (ProviderEvidence("cblas", True, "executed", (measurement,)),)
        )

    none, zeros = artifact(None, None, None), artifact(0, 0, 0)
    assert none.schema == "bcir.numerical_provider_evidence.v2"
    assert '"cache_misses":null' in none.to_json() and none.digest != zeros.digest
    assert _refused(ProviderMeasurement, workload.digest, 100, 90, 1, None, None, 2)
    assert _refused(ProviderMeasurement, workload.digest, 100, 90, None, 0, None, 2)
    assert _refused(artifact, 1 << 63, 1, 1)
    assert _refused(replace, none, schema="bcir.numerical_provider_evidence.v3")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "providers.json"
        write_numerical_evidence(path, none)
        assert read_numerical_evidence(path) == none
        path.write_text(_V1_EVIDENCE + "\n", "utf-8")
        v1 = read_numerical_evidence(path)
        assert (v1.to_json(), v1.digest) == (_V1_EVIDENCE, _V1_EVIDENCE_DIGEST)
        nulls = _V1_EVIDENCE
        for name in ("cache_misses", "cycles", "instructions"):
            nulls = nulls.replace(f'"{name}":0,', f'"{name}":null,')
        assert nulls.count("null") == 3
        path.write_text(nulls, "utf-8")
        assert _refused(read_numerical_evidence, path)
