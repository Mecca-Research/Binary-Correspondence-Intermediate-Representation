"""Bounded reinforcement-learning contracts for hardware execution-plan selection.

Learning is advisory: it ranks a finite set of already-assessed candidates. Candidate
legality, claim verification, static-address safety, and StreamPack integrity remain
deterministic BCIR gates. Hardware outcomes retain measured/simulated provenance so a
simulator can train a policy but can never certify a live plan promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from .._artifact_json import strict_json_loads
from ..telemetry import DataDNA
from ..verify import verify_all, verify_smart_lowering
from .cost import CostVector, N
from .device_manifest import check_bank_moves
from .static_memory import StaticMemoryPlan, verify_static_memory_plan

Q20 = 1 << 20
_MAX_U63 = (1 << 63) - 1
_MAX_JSON = 16 * 1024 * 1024
_MAX_TOKENS = 4096
_MAX_CANDIDATES = 256
_MAX_OUTCOMES = 256
_MAX_SIMULATIONS = 4096

MISS_AVAILABLE = 1 << 0
REGISTER_AVAILABLE = 1 << 1
BANDWIDTH_AVAILABLE = 1 << 2
THERMAL_AVAILABLE = 1 << 3
UTILIZATION_AVAILABLE = 1 << 4
THROTTLE_AVAILABLE = 1 << 5
ALL_AVAILABILITY = (1 << 6) - 1
_AVAILABILITY_BITS = (
    MISS_AVAILABLE,
    REGISTER_AVAILABLE,
    BANDWIDTH_AVAILABLE,
    THERMAL_AVAILABLE,
    UTILIZATION_AVAILABLE,
    THROTTLE_AVAILABLE,
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _int(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _str(value, field: str, *, choices=None) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 4096
        or any(ord(character) < 0x20 for character in value)
    ):
        raise ValueError(f"{field} must be a bounded, control-free string")
    if choices is not None and value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _digest(value, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _pressure(value, field: str, available: bool) -> int:
    _int(value, field, maximum=100)
    if not available and value:
        raise ValueError(f"unavailable telemetry field {field} must be encoded as zero")
    return value


def _log2(value: int) -> int:
    return max(0, value.bit_length() - 1)


def _strict_sequences(rows, field: str) -> None:
    sequences = tuple(row.sequence for row in rows)
    if any(right <= left for left, right in zip(sequences, sequences[1:])):
        raise ValueError(f"{field} sequence must be strictly increasing")


def _q20_ratio(value: int, reference: int, field: str) -> int:
    result = value * Q20 // reference
    if result > _MAX_U63:
        raise ValueError(f"{field} exceeds the bounded K_BCIR cost range")
    return result


@dataclass(frozen=True)
class TelemetryToken:
    """One temporal hardware token with honest per-field availability.

    A zero reading and an unavailable reading are different: the latter has its mask bit
    clear and must encode zero. This prevents absent register/HBM counters from becoming
    fabricated evidence.
    """

    sequence: int
    cycles: int
    bytes_moved: int
    cache_miss_pressure: int
    register_pressure: int
    bandwidth_pressure: int
    thermal_pressure: int
    voltage_pressure: int
    utilization: int
    throttled: int
    availability_mask: int
    provenance_sha256: str

    def __post_init__(self) -> None:
        _int(self.sequence, "telemetry sequence")
        _int(self.cycles, "telemetry cycles")
        _int(self.bytes_moved, "telemetry bytes_moved")
        _int(self.availability_mask, "telemetry availability_mask", maximum=ALL_AVAILABILITY)
        _pressure(
            self.cache_miss_pressure,
            "cache_miss_pressure",
            bool(self.availability_mask & MISS_AVAILABLE),
        )
        _pressure(
            self.register_pressure,
            "register_pressure",
            bool(self.availability_mask & REGISTER_AVAILABLE),
        )
        _pressure(
            self.bandwidth_pressure,
            "bandwidth_pressure",
            bool(self.availability_mask & BANDWIDTH_AVAILABLE),
        )
        _pressure(
            self.thermal_pressure,
            "thermal_pressure",
            bool(self.availability_mask & THERMAL_AVAILABLE),
        )
        _pressure(self.voltage_pressure, "voltage_pressure", True)
        _pressure(
            self.utilization, "utilization", bool(self.availability_mask & UTILIZATION_AVAILABLE)
        )
        _int(self.throttled, "throttled", maximum=1)
        if not self.availability_mask & THROTTLE_AVAILABLE and self.throttled:
            raise ValueError("unavailable telemetry field throttled must be encoded as zero")
        _digest(self.provenance_sha256, "telemetry provenance_sha256")

    @classmethod
    def from_data_dna(
        cls,
        event: DataDNA,
        *,
        sequence: int,
        register_pressure: int | None = None,
        bandwidth_pressure: int | None = None,
        throttled: bool | None = None,
    ) -> "TelemetryToken":
        if not isinstance(event, DataDNA):
            raise ValueError("telemetry conversion requires DataDNA")
        if throttled is not None and type(throttled) is not bool:
            raise ValueError("optional telemetry throttled state must be Boolean")
        event.validate()
        mask = MISS_AVAILABLE | THERMAL_AVAILABLE | UTILIZATION_AVAILABLE
        if register_pressure is not None:
            mask |= REGISTER_AVAILABLE
        if bandwidth_pressure is not None:
            mask |= BANDWIDTH_AVAILABLE
        if throttled is not None:
            mask |= THROTTLE_AVAILABLE
        provenance = _sha(
            {
                "event": event.to_dict(),
                "sequence": sequence,
                "register_pressure": register_pressure,
                "bandwidth_pressure": bandwidth_pressure,
                "throttled": throttled,
            }
        )
        return cls(
            sequence,
            event.cycles,
            event.bytes,
            event.misses,
            0 if register_pressure is None else register_pressure,
            0 if bandwidth_pressure is None else bandwidth_pressure,
            event.thermal,
            event.voltage,
            event.utilization,
            0 if throttled is None else int(throttled),
            mask,
            provenance,
        )

    @property
    def feature_vector(self) -> tuple[int, ...]:
        return (
            _log2(self.cycles + 1),
            _log2(self.bytes_moved + 1),
            self.cache_miss_pressure,
            self.register_pressure,
            self.bandwidth_pressure,
            self.thermal_pressure,
            self.voltage_pressure,
            self.utilization,
            self.throttled * 100,
            *(100 if self.availability_mask & bit else 0 for bit in _AVAILABILITY_BITS),
        )


@dataclass(frozen=True)
class BankTopologyFeature:
    name: str
    domain_id: int
    capacity_log2: int
    allocatable_ratio_q20: int
    read_bandwidth_log2: int
    write_bandwidth_log2: int
    alignment_log2: int
    native_tile_log2: int

    def __post_init__(self) -> None:
        _str(self.name, "topology bank name")
        for field in (
            "domain_id",
            "capacity_log2",
            "allocatable_ratio_q20",
            "read_bandwidth_log2",
            "write_bandwidth_log2",
            "alignment_log2",
            "native_tile_log2",
        ):
            _int(getattr(self, field), f"topology bank {field}")
        if (
            self.domain_id >= 32
            or self.allocatable_ratio_q20 > Q20
            or max(self.capacity_log2, self.read_bandwidth_log2, self.write_bandwidth_log2) > 63
            or max(self.alignment_log2, self.native_tile_log2) > 62
        ):
            raise ValueError("topology bank feature exceeds its encoding range")

    @property
    def feature_vector(self) -> tuple[int, ...]:
        return (
            self.domain_id,
            self.capacity_log2,
            self.allocatable_ratio_q20,
            self.read_bandwidth_log2,
            self.write_bandwidth_log2,
            self.alignment_log2,
            self.native_tile_log2,
        )


@dataclass(frozen=True)
class TopologyLinkFeature:
    source: int
    destination: int
    bandwidth_log2: int
    latency_log2: int

    def __post_init__(self) -> None:
        for field in ("source", "destination", "bandwidth_log2", "latency_log2"):
            _int(getattr(self, field), f"topology link {field}")
        if self.source == self.destination:
            raise ValueError("topology link endpoints must differ")
        if self.bandwidth_log2 > 63 or self.latency_log2 > 63:
            raise ValueError("topology link feature exceeds its encoding range")


@dataclass(frozen=True)
class HardwareTopology:
    hardware_digest: str
    nodes: tuple[BankTopologyFeature, ...]
    links: tuple[TopologyLinkFeature, ...]

    def __post_init__(self) -> None:
        _digest(self.hardware_digest, "topology hardware_digest")
        if not isinstance(self.nodes, tuple) or not self.nodes or len(self.nodes) > 32:
            raise ValueError("hardware topology needs a bounded node tuple")
        if any(not isinstance(row, BankTopologyFeature) for row in self.nodes):
            raise ValueError("hardware topology has malformed nodes")
        if (
            not isinstance(self.links, tuple)
            or len(self.links) > 992
            or any(not isinstance(row, TopologyLinkFeature) for row in self.links)
        ):
            raise ValueError("hardware topology has malformed links")
        if any(
            row.source >= len(self.nodes) or row.destination >= len(self.nodes)
            for row in self.links
        ):
            raise ValueError("hardware topology link index is out of bounds")
        if len({row.name for row in self.nodes}) != len(self.nodes):
            raise ValueError("hardware topology bank names must be unique")
        if tuple(sorted(self.nodes, key=lambda row: row.name)) != self.nodes:
            raise ValueError("hardware topology nodes must use canonical name order")
        if len({(row.source, row.destination) for row in self.links}) != len(self.links):
            raise ValueError("hardware topology links must be unique and directed")
        if tuple(sorted(self.links, key=lambda row: (row.source, row.destination))) != self.links:
            raise ValueError("hardware topology links must use canonical endpoint order")


def encode_hardware_topology(hardware) -> HardwareTopology:
    banks = tuple(sorted(hardware.banks, key=lambda row: row.name))
    names = {row.name: index for index, row in enumerate(banks)}
    domains = {name: index for index, name in enumerate(sorted({row.domain for row in banks}))}
    nodes = tuple(
        BankTopologyFeature(
            row.name,
            domains[row.domain],
            _log2(row.capacity_bytes),
            row.allocatable_bytes * Q20 // row.capacity_bytes,
            _log2(row.read_bandwidth_bps),
            _log2(row.write_bandwidth_bps),
            _log2(row.alignment),
            _log2(row.native_tile),
        )
        for row in banks
    )
    links = tuple(
        TopologyLinkFeature(
            names[row.source],
            names[row.destination],
            _log2(row.bandwidth_bps),
            _log2(row.latency_ns + 1),
        )
        for row in sorted(hardware.links, key=lambda value: (value.source, value.destination))
    )
    return HardwareTopology(hardware.digest, nodes, links)


@dataclass(frozen=True)
class CandidateFeature:
    candidate_id: str
    kind: str
    weight_format: str
    classification: str
    compute_bank_count: int
    peak_ratio_q20: int
    prefill_transfer_log2: int
    decode_transfer_log2: int
    prefill_latency_log2: int
    decode_latency_log2: int
    split_layer: int
    compute_bank0: int
    compute_bank1: int
    backing_bank: int

    def __post_init__(self) -> None:
        _str(self.candidate_id, "candidate feature id")
        _str(
            self.kind,
            "candidate feature kind",
            choices={"resident", "layer-stream", "cpu-gpu-split"},
        )
        _str(self.weight_format, "candidate feature format", choices={"source", "bcirq8-group32"})
        _str(
            self.classification,
            "candidate feature classification",
            choices={"exact", "quantized", "approximate"},
        )
        for field in (
            "compute_bank_count",
            "peak_ratio_q20",
            "prefill_transfer_log2",
            "decode_transfer_log2",
            "prefill_latency_log2",
            "decode_latency_log2",
            "split_layer",
            "compute_bank0",
            "compute_bank1",
            "backing_bank",
        ):
            _int(
                getattr(self, field),
                f"candidate feature {field}",
                minimum=1 if field == "compute_bank_count" else 0,
            )
        if (
            not 1 <= self.compute_bank_count <= 2
            or self.peak_ratio_q20 > Q20
            or max(
                self.prefill_transfer_log2,
                self.decode_transfer_log2,
                self.prefill_latency_log2,
                self.decode_latency_log2,
            )
            > 63
            or self.split_layer > 4096
            or max(self.compute_bank0, self.compute_bank1, self.backing_bank) > 32
        ):
            raise ValueError("candidate feature exceeds its encoding range")
        if (
            self.compute_bank0 == 0
            or self.backing_bank == 0
            or (self.compute_bank_count == 1) != (self.compute_bank1 == 0)
        ):
            raise ValueError("candidate feature has inconsistent bank identities")
        if self.compute_bank_count == 2 and self.compute_bank0 == self.compute_bank1:
            raise ValueError("split candidate must name two distinct compute banks")

    @property
    def feature_vector(self) -> tuple[int, ...]:
        kind = {"resident": 0, "layer-stream": 1, "cpu-gpu-split": 2}[self.kind]
        classification = {"exact": 0, "quantized": 1, "approximate": 2}[self.classification]
        return (
            kind,
            0 if self.weight_format == "source" else 1,
            classification,
            self.compute_bank_count,
            self.peak_ratio_q20,
            self.prefill_transfer_log2,
            self.decode_transfer_log2,
            self.prefill_latency_log2,
            self.decode_latency_log2,
            self.split_layer,
            self.compute_bank0,
            self.compute_bank1,
            self.backing_bank,
        )


def encode_candidate_features(report) -> tuple[CandidateFeature, ...]:
    rows = []
    bank_names = sorted({peak.bank for candidate in report.candidates for peak in candidate.peaks})
    bank_ids = {name: index + 1 for index, name in enumerate(bank_names)}
    for candidate in report.candidates:
        if not candidate.feasible:
            continue
        ratios = [peak.peak_bytes * Q20 // max(1, peak.capacity_bytes) for peak in candidate.peaks]
        predictions = {row.operation: row for row in candidate.predictions}
        rows.append(
            CandidateFeature(
                candidate.candidate_id,
                candidate.kind,
                candidate.weight_format,
                candidate.classification,
                len(candidate.compute_banks),
                max(ratios, default=0),
                _log2(candidate.prefill_transfer_bytes + 1),
                _log2(candidate.decode_transfer_bytes + 1),
                _log2(predictions["prefill"].median_ns + 1),
                _log2(predictions["decode"].median_ns + 1),
                max(0, candidate.split_layer),
                bank_ids[candidate.compute_banks[0]],
                bank_ids[candidate.compute_banks[1]] if len(candidate.compute_banks) > 1 else 0,
                bank_ids[candidate.backing_bank],
            )
        )
    if not rows or len(rows) > _MAX_CANDIDATES:
        raise ValueError("RL selection requires 1..256 feasible candidates")
    return tuple(sorted(rows, key=lambda row: row.candidate_id))


@dataclass(frozen=True)
class HardwareOutcome:
    context_digest: str
    candidate_id: str
    latency_ns: int
    energy_uj: int
    register_contention: int
    cache_miss_pressure: int
    bandwidth_pressure: int
    thermal_pressure: int
    throttled: bool
    correctness_passed: bool
    provenance: str
    samples: int
    source_sha256: str

    def __post_init__(self) -> None:
        _digest(self.context_digest, "outcome context_digest")
        _str(self.candidate_id, "outcome candidate_id")
        _int(self.latency_ns, "outcome latency_ns", minimum=1)
        _int(self.energy_uj, "outcome energy_uj", minimum=1)
        for field in (
            "register_contention",
            "cache_miss_pressure",
            "bandwidth_pressure",
            "thermal_pressure",
        ):
            _int(getattr(self, field), f"outcome {field}", maximum=100)
        if type(self.throttled) is not bool or type(self.correctness_passed) is not bool:
            raise ValueError("outcome Boolean fields are malformed")
        _str(self.provenance, "outcome provenance", choices={"measured", "simulated"})
        _int(self.samples, "outcome samples", minimum=1, maximum=1_000_000)
        _digest(self.source_sha256, "outcome source_sha256")

    @property
    def digest(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True)
class HardwareRewardPolicy:
    weights: tuple[int, ...] = (4, 2, 0, 0, 0, 2, 3, 2, 0, 0, 3, 1)
    latency_reference_ns: int = 1_000_000
    energy_reference_uj: int = 1_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.weights, tuple)
            or len(self.weights) != N
            or any(type(value) is not int or not 0 <= value <= 1_000_000 for value in self.weights)
        ):
            raise ValueError(f"hardware reward policy needs {N} bounded integer weights")
        _int(self.latency_reference_ns, "latency_reference_ns", minimum=1)
        _int(self.energy_reference_uj, "energy_reference_uj", minimum=1)

    @property
    def digest(self) -> str:
        return _sha(asdict(self))

    def cost_vector(self, outcome: HardwareOutcome) -> CostVector:
        if not outcome.correctness_passed:
            raise ValueError("incorrect hardware outcomes cannot be rewarded")
        return CostVector.of(
            compute=_q20_ratio(outcome.latency_ns, self.latency_reference_ns, "latency cost"),
            memory=max(outcome.cache_miss_pressure, outcome.bandwidth_pressure) * Q20 // 100,
            thermal=outcome.thermal_pressure * Q20 // 100,
            power=_q20_ratio(outcome.energy_uj, self.energy_reference_uj, "energy cost"),
            reliability=Q20 if outcome.throttled else 0,
            contention=outcome.register_contention * Q20 // 100,
            verification=Q20 // max(1, outcome.samples),
        )

    def score(self, outcome: HardwareOutcome) -> int:
        """Lower is better; negative score is the corresponding RL reward."""
        result = self.cost_vector(outcome).dot(self.weights)
        if result > _MAX_U63:
            raise ValueError("hardware reward score exceeds signed 64-bit range")
        return result


@dataclass(frozen=True)
class HardwarePreference:
    context_digest: str
    chosen_id: str
    rejected_id: str
    chosen_outcome_sha256: str
    rejected_outcome_sha256: str
    margin: int

    def __post_init__(self) -> None:
        _digest(self.context_digest, "preference context_digest")
        _str(self.chosen_id, "preference chosen_id")
        _str(self.rejected_id, "preference rejected_id")
        if self.chosen_id == self.rejected_id:
            raise ValueError("hardware preference candidates must differ")
        _digest(self.chosen_outcome_sha256, "preference chosen outcome")
        _digest(self.rejected_outcome_sha256, "preference rejected outcome")
        _int(self.margin, "preference margin", minimum=1)


def outcomes_to_preferences(
    outcomes, policy: HardwareRewardPolicy
) -> tuple[HardwarePreference, ...]:
    rows = tuple(outcomes)
    if (
        not rows
        or len(rows) > _MAX_OUTCOMES
        or any(not isinstance(row, HardwareOutcome) for row in rows)
    ):
        raise ValueError("hardware outcomes must be a bounded nonempty sequence")
    contexts = {row.context_digest for row in rows}
    ids = [row.candidate_id for row in rows]
    if len(contexts) != 1 or len(set(ids)) != len(ids):
        raise ValueError("preference outcomes need one context and unique candidates")
    if any(not row.correctness_passed for row in rows):
        raise ValueError("incorrect outcomes cannot enter preference training")
    preferences = []
    ordered = sorted(rows, key=lambda row: row.candidate_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            left_score, right_score = policy.score(left), policy.score(right)
            if left_score == right_score:
                continue
            chosen, rejected = (left, right) if left_score < right_score else (right, left)
            preferences.append(
                HardwarePreference(
                    chosen.context_digest,
                    chosen.candidate_id,
                    rejected.candidate_id,
                    chosen.digest,
                    rejected.digest,
                    abs(left_score - right_score),
                )
            )
    return tuple(preferences)


@dataclass(frozen=True)
class HardwareEpisode:
    report_digest: str
    hardware_digest: str
    workload_digest: str
    telemetry: tuple[TelemetryToken, ...]
    outcomes: tuple[HardwareOutcome, ...]
    reward_policy_digest: str
    schema: str = "bcir.hardware_episode.v1"

    def __post_init__(self) -> None:
        if self.schema != "bcir.hardware_episode.v1":
            raise ValueError("unsupported hardware episode schema")
        for value, field in (
            (self.report_digest, "report_digest"),
            (self.hardware_digest, "hardware_digest"),
            (self.workload_digest, "workload_digest"),
            (self.reward_policy_digest, "reward_policy_digest"),
        ):
            _digest(value, f"episode {field}")
        if (
            not isinstance(self.telemetry, tuple)
            or not self.telemetry
            or len(self.telemetry) > _MAX_TOKENS
            or any(not isinstance(row, TelemetryToken) for row in self.telemetry)
        ):
            raise ValueError("episode telemetry must be a bounded nonempty tuple")
        _strict_sequences(self.telemetry, "episode telemetry")
        if (
            not isinstance(self.outcomes, tuple)
            or not self.outcomes
            or len(self.outcomes) > _MAX_OUTCOMES
            or any(not isinstance(row, HardwareOutcome) for row in self.outcomes)
        ):
            raise ValueError("episode outcomes must be a bounded nonempty tuple")
        if len({row.candidate_id for row in self.outcomes}) != len(self.outcomes):
            raise ValueError("episode outcome candidates must be unique")
        if {row.context_digest for row in self.outcomes} != {self.context_digest}:
            raise ValueError("episode outcomes do not match the episode context")

    @property
    def context_digest(self) -> str:
        return hardware_context_digest(
            self.report_digest, self.hardware_digest, self.workload_digest, self.telemetry
        )

    @property
    def digest(self) -> str:
        return _sha(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "report_digest": self.report_digest,
            "hardware_digest": self.hardware_digest,
            "workload_digest": self.workload_digest,
            "telemetry": [asdict(row) for row in self.telemetry],
            "outcomes": [asdict(row) for row in self.outcomes],
            "reward_policy_digest": self.reward_policy_digest,
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "HardwareEpisode":
        doc = strict_json_loads(text, "hardware episode", max_bytes=_MAX_JSON)
        fields = {
            "schema",
            "report_digest",
            "hardware_digest",
            "workload_digest",
            "telemetry",
            "outcomes",
            "reward_policy_digest",
        }
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError("hardware episode has missing or unknown fields")
        if (
            not isinstance(doc["telemetry"], list)
            or not doc["telemetry"]
            or len(doc["telemetry"]) > _MAX_TOKENS
            or not isinstance(doc["outcomes"], list)
            or not doc["outcomes"]
            or len(doc["outcomes"]) > _MAX_OUTCOMES
        ):
            raise ValueError("hardware episode arrays are empty, malformed, or unbounded")
        try:
            telemetry = tuple(TelemetryToken(**row) for row in doc.pop("telemetry"))
            outcomes = tuple(HardwareOutcome(**row) for row in doc.pop("outcomes"))
            return cls(**doc, telemetry=telemetry, outcomes=outcomes)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed hardware episode: {exc}") from exc


def hardware_context_digest(
    report_digest: str, hardware_digest: str, workload_digest: str, telemetry
) -> str:
    for value, field in (
        (report_digest, "report_digest"),
        (hardware_digest, "hardware_digest"),
        (workload_digest, "workload_digest"),
    ):
        _digest(value, f"hardware context {field}")
    rows = tuple(telemetry)
    if (
        not rows
        or len(rows) > _MAX_TOKENS
        or any(not isinstance(row, TelemetryToken) for row in rows)
    ):
        raise ValueError("hardware context telemetry must be a bounded nonempty sequence")
    _strict_sequences(rows, "hardware context telemetry")
    return _sha(
        {
            "report": report_digest,
            "hardware": hardware_digest,
            "workload": workload_digest,
            "telemetry": [asdict(row) for row in rows],
        }
    )


@dataclass(frozen=True)
class SearchVisit:
    candidate_id: str
    visits: int
    mean_utility: int

    def __post_init__(self) -> None:
        _str(self.candidate_id, "search visit candidate_id")
        _int(self.visits, "search visit count", minimum=1, maximum=_MAX_SIMULATIONS)
        _int(self.mean_utility, "search visit mean utility", minimum=-_MAX_U63, maximum=_MAX_U63)


@dataclass(frozen=True)
class HardwareSearchResult:
    candidate_id: str
    simulations: int
    visits: tuple[SearchVisit, ...]

    def __post_init__(self) -> None:
        _str(self.candidate_id, "search result candidate_id")
        _int(self.simulations, "search result simulations", minimum=1, maximum=_MAX_SIMULATIONS)
        if (
            not isinstance(self.visits, tuple)
            or not self.visits
            or len(self.visits) > _MAX_CANDIDATES
            or any(not isinstance(row, SearchVisit) for row in self.visits)
            or len({row.candidate_id for row in self.visits}) != len(self.visits)
            or self.candidate_id not in {row.candidate_id for row in self.visits}
            or sum(row.visits for row in self.visits) != self.simulations
        ):
            raise ValueError("search result visit inventory is inconsistent")


def bounded_mcts(
    candidate_ids, priors_q20, evaluator, *, simulations: int = 64, exploration_q20: int = 2 * Q20
) -> HardwareSearchResult:
    """A deterministic root-PUCT micro-profiling search over a finite plan portfolio.

    The callback is the hardware/simulator boundary and returns Q20 integer utility
    (higher is better). The search never synthesizes instructions and never expands
    outside the supplied feasible candidate IDs.
    """
    try:
        candidates = tuple(candidate_ids)
    except TypeError as exc:
        raise ValueError("MCTS candidate IDs must be a bounded sequence") from exc
    if (
        not candidates
        or len(candidates) > _MAX_CANDIDATES
        or any(not isinstance(row, str) or not row for row in candidates)
    ):
        raise ValueError("MCTS needs 1..256 unique candidate IDs")
    for candidate in candidates:
        _str(candidate, "MCTS candidate ID")
    if len(set(candidates)) != len(candidates):
        raise ValueError("MCTS needs 1..256 unique candidate IDs")
    candidates = tuple(sorted(candidates))
    _int(simulations, "MCTS simulations", minimum=len(candidates), maximum=_MAX_SIMULATIONS)
    _int(exploration_q20, "MCTS exploration_q20", maximum=100 * Q20)
    if (
        not callable(evaluator)
        or not isinstance(priors_q20, dict)
        or set(priors_q20) != set(candidates)
    ):
        raise ValueError("MCTS requires an evaluator and one prior per candidate")
    if (
        any(type(value) is not int or not 0 <= value <= _MAX_U63 for value in priors_q20.values())
        or sum(priors_q20.values()) <= 0
    ):
        raise ValueError("MCTS priors must be non-negative integers with positive mass")
    total_prior = sum(priors_q20.values())
    visits = {candidate: 0 for candidate in candidates}
    utility = {candidate: 0 for candidate in candidates}
    for step in range(simulations):
        if step < len(candidates):
            selected = candidates[step]
        else:
            root = math.isqrt(step + 1)

            def puct(candidate, root=root):
                mean = utility[candidate] // visits[candidate]
                explore = (
                    exploration_q20
                    * priors_q20[candidate]
                    * root
                    // (total_prior * (visits[candidate] + 1))
                )
                return mean + explore

            selected = min(candidates, key=lambda candidate: (-puct(candidate), candidate))
        value = evaluator(selected)
        if type(value) is not int or not -_MAX_U63 <= value <= _MAX_U63:
            raise ValueError("MCTS evaluator must return a bounded integer utility")
        visits[selected] += 1
        utility[selected] += value
    selected = min(
        candidates,
        key=lambda candidate: (
            -visits[candidate],
            -(utility[candidate] // visits[candidate]),
            candidate,
        ),
    )
    rows = tuple(
        SearchVisit(candidate, visits[candidate], utility[candidate] // visits[candidate])
        for candidate in candidates
    )
    return HardwareSearchResult(selected, simulations, rows)


@dataclass(frozen=True)
class HardwarePromotionCertificate:
    plan_digest: str
    static_memory_digest: str
    selected_outcome_digest: str
    baseline_outcome_digest: str
    policy_digest: str
    episode_digest: str
    improvement_q20: int
    generation: int

    def __post_init__(self) -> None:
        for value, field in (
            (self.plan_digest, "plan_digest"),
            (self.static_memory_digest, "static_memory_digest"),
            (self.selected_outcome_digest, "selected_outcome_digest"),
            (self.baseline_outcome_digest, "baseline_outcome_digest"),
            (self.policy_digest, "policy_digest"),
            (self.episode_digest, "episode_digest"),
        ):
            _digest(value, f"promotion {field}")
        _int(self.improvement_q20, "promotion improvement_q20", minimum=1, maximum=Q20)
        _int(self.generation, "promotion generation", minimum=1)

    @property
    def digest(self) -> str:
        return _sha(asdict(self))

    def to_json(self) -> str:
        return _canonical({"schema": "bcir.hardware_promotion.v1", **asdict(self)})

    @classmethod
    def from_json(cls, text: str) -> "HardwarePromotionCertificate":
        doc = strict_json_loads(text, "hardware promotion certificate", max_bytes=1024 * 1024)
        fields = {
            "schema",
            "plan_digest",
            "static_memory_digest",
            "selected_outcome_digest",
            "baseline_outcome_digest",
            "policy_digest",
            "episode_digest",
            "improvement_q20",
            "generation",
        }
        if (
            not isinstance(doc, dict)
            or set(doc) != fields
            or doc.pop("schema") != "bcir.hardware_promotion.v1"
        ):
            raise ValueError("hardware promotion certificate has missing or unknown fields")
        try:
            return cls(**doc)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed hardware promotion certificate: {exc}") from exc


def certify_hardware_promotion(
    lowered,
    static_plan: StaticMemoryPlan,
    hardware,
    selected: HardwareOutcome,
    baseline: HardwareOutcome,
    policy: HardwareRewardPolicy,
    *,
    episode: HardwareEpisode,
    generation: int,
    quiescent: bool,
    minimum_improvement_q20: int = 1,
) -> HardwarePromotionCertificate:
    """Admit a measured improvement only at a quiescent generation boundary."""
    if type(quiescent) is not bool or not quiescent:
        raise ValueError("hardware-plan promotion requires a quiescent generation boundary")
    _int(generation, "promotion generation", minimum=1)
    _int(minimum_improvement_q20, "minimum_improvement_q20", minimum=1, maximum=Q20)
    if not isinstance(episode, HardwareEpisode):
        raise ValueError("hardware-plan promotion requires a typed evidence episode")
    if selected.provenance != "measured" or baseline.provenance != "measured":
        raise ValueError("simulated outcomes cannot certify hardware-plan promotion")
    if selected.context_digest != baseline.context_digest:
        raise ValueError("promotion outcomes must share one execution context")
    if selected.candidate_id == baseline.candidate_id:
        raise ValueError("promotion baseline and selected candidates must differ")
    if selected.candidate_id != lowered.artifact.candidate_id:
        raise ValueError("selected outcome does not name the lowered candidate")
    if (
        episode.report_digest != lowered.artifact.report_digest
        or episode.hardware_digest != hardware.digest
        or episode.reward_policy_digest != policy.digest
        or selected not in episode.outcomes
        or baseline not in episode.outcomes
    ):
        raise ValueError("promotion evidence is not bound to the plan, hardware, and policy")
    if not selected.correctness_passed or not baseline.correctness_passed:
        raise ValueError("promotion requires correctness-passing outcomes")
    base_score, selected_score = policy.score(baseline), policy.score(selected)
    if base_score <= 0 or selected_score >= base_score:
        raise ValueError("selected plan does not strictly improve the baseline")
    improvement = (base_score - selected_score) * Q20 // base_score
    if improvement < minimum_improvement_q20:
        raise ValueError("selected plan does not meet the promotion margin")
    diagnostics = verify_all(lowered.module, result=lowered.realization, pack=lowered.streampack)
    diagnostics += verify_smart_lowering(lowered.module, pack=lowered.streampack)
    errors = [f"{row.law}: {row.message}" for row in diagnostics]
    errors += check_bank_moves(lowered.module)
    errors += list(
        verify_static_memory_plan(static_plan, lowered.module, lowered.resource_banks, hardware)
    )
    if errors:
        raise ValueError("promotion artifact verification failed: " + "; ".join(errors))
    return HardwarePromotionCertificate(
        lowered.artifact.digest,
        static_plan.digest,
        selected.digest,
        baseline.digest,
        policy.digest,
        episode.digest,
        improvement,
        generation,
    )


__all__ = [
    "ALL_AVAILABILITY",
    "BANDWIDTH_AVAILABLE",
    "BankTopologyFeature",
    "CandidateFeature",
    "HardwareEpisode",
    "HardwareOutcome",
    "HardwarePreference",
    "HardwarePromotionCertificate",
    "HardwareRewardPolicy",
    "HardwareSearchResult",
    "HardwareTopology",
    "MISS_AVAILABLE",
    "Q20",
    "REGISTER_AVAILABLE",
    "SearchVisit",
    "THERMAL_AVAILABLE",
    "THROTTLE_AVAILABLE",
    "TelemetryToken",
    "TopologyLinkFeature",
    "UTILIZATION_AVAILABLE",
    "bounded_mcts",
    "certify_hardware_promotion",
    "encode_candidate_features",
    "encode_hardware_topology",
    "hardware_context_digest",
    "outcomes_to_preferences",
]
