"""Deterministic, payload-free model capacity and placement assessment.

The assessor reasons from :class:`TensorInventory` headers, a user-supplied hardware
envelope, and an explicit workload.  It never opens weight files.  Measurements are
bounded evidence records and may price candidates, but cannot change BCIR legality.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from ..._artifact_json import strict_json_loads
from ...model import Domain
from .inventory import TensorInventory, TensorRecord

_MAX_JSON = 64 * 1024 * 1024
_MAX_U63 = (1 << 63) - 1
_HARDWARE_SCHEMA = "bcir.hardware_envelope.v1"
_WORKLOAD_SCHEMA = "bcir.model_workload.v1"
_REPORT_SCHEMA = "bcir.model_cost_report.v1"
_PLAN_SCHEMA = "bcir.model_execution_plan.v1"
_FORMATS = frozenset(("source", "bcirq8-group32"))
_CLASSIFICATIONS = frozenset(("exact", "quantized", "approximate"))
_WORKLOAD_MODES = frozenset(("inference", "lora", "full-finetune", "pretrain"))
_DTYPE_BYTES = {"u8": 1, "i8": 1, "f16": 2, "bf16": 2, "f32": 4, "f64": 8}
_MAX_BANKS = 32
_MAX_PLAN_LAYERS = 4096
_MAX_CANDIDATES = 100_000


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _int(value, field: str, *, minimum: int = 0, maximum: int = _MAX_U63) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _str(value, field: str, *, choices=None, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise ValueError(f"{field} must be {'a string' if empty else 'a nonempty string'}")
    if len(value.encode("utf-8")) > 4096 or any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{field} exceeds 4096 UTF-8 bytes or contains control characters")
    if choices is not None and value not in choices:
        raise ValueError(f"{field} must be one of {sorted(choices)}")
    return value


def _tuple(value, field: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field} must be an array")
    return tuple(value)


def _checked_add(*values: int) -> int:
    result = 0
    for value in values:
        if value < 0 or result > _MAX_U63 - value:
            raise ValueError("model memory accounting exceeds signed 64-bit range")
        result += value
    return result


def _checked_mul(*values: int) -> int:
    result = 1
    for value in values:
        if value < 0 or (value and result > _MAX_U63 // value):
            raise ValueError("model memory accounting exceeds signed 64-bit range")
        result *= value
    return result


def _align8(value: int) -> int:
    if value > _MAX_U63 - 7:
        raise ValueError("model format accounting exceeds signed 64-bit range")
    return (value + 7) & ~7


@dataclass(frozen=True)
class BankEnvelope:
    name: str
    domain: str
    channel: str
    capacity_bytes: int
    allocatable_bytes: int
    read_bandwidth_bps: int
    write_bandwidth_bps: int
    alignment: int = 64
    native_tile: int = 1

    def __post_init__(self) -> None:
        _str(self.name, "bank name"); _str(self.channel, "bank channel")
        _str(self.domain, "bank domain", choices={item.name for item in Domain})
        for field in ("capacity_bytes", "allocatable_bytes", "read_bandwidth_bps",
                      "write_bandwidth_bps", "alignment", "native_tile"):
            _int(getattr(self, field), f"bank {self.name} {field}", minimum=1)
        if self.allocatable_bytes > self.capacity_bytes:
            raise ValueError(f"bank {self.name} allocatable bytes exceed capacity")
        if self.alignment & (self.alignment - 1):
            raise ValueError(f"bank {self.name} alignment must be a power of two")


@dataclass(frozen=True)
class LinkEnvelope:
    source: str
    destination: str
    bandwidth_bps: int
    latency_ns: int

    def __post_init__(self) -> None:
        _str(self.source, "link source"); _str(self.destination, "link destination")
        _int(self.bandwidth_bps, "link bandwidth_bps", minimum=1)
        _int(self.latency_ns, "link latency_ns")
        if self.source == self.destination:
            raise ValueError("hardware link endpoints must differ")


@dataclass(frozen=True)
class KernelBenchmark:
    operation: str
    channel: str
    weight_format: str
    lower_ns: int
    median_ns: int
    upper_ns: int
    tokens: int
    operations: int
    weight_bytes: int
    workspace_bytes: int
    samples: int

    def __post_init__(self) -> None:
        _str(self.operation, "benchmark operation", choices={"prefill", "decode"})
        _str(self.channel, "benchmark channel")
        _str(self.weight_format, "benchmark weight_format", choices=_FORMATS)
        for field in ("lower_ns", "median_ns", "upper_ns", "tokens", "operations",
                      "weight_bytes", "workspace_bytes", "samples"):
            _int(getattr(self, field), f"benchmark {field}", minimum=1)
        if not self.lower_ns <= self.median_ns <= self.upper_ns:
            raise ValueError("benchmark interval must satisfy lower <= median <= upper")


@dataclass(frozen=True)
class HardwareEnvelope:
    name: str
    architecture: str
    banks: tuple[BankEnvelope, ...]
    links: tuple[LinkEnvelope, ...] = ()
    benchmarks: tuple[KernelBenchmark, ...] = ()
    schema: str = _HARDWARE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _HARDWARE_SCHEMA:
            raise ValueError("unsupported hardware envelope schema")
        _str(self.name, "hardware name"); _str(self.architecture, "hardware architecture")
        banks = _tuple(self.banks, "hardware banks")
        links = _tuple(self.links, "hardware links")
        benchmarks = _tuple(self.benchmarks, "hardware benchmarks")
        if not banks or len(banks) > _MAX_BANKS \
                or any(not isinstance(row, BankEnvelope) for row in banks):
            raise ValueError("hardware envelope needs typed banks")
        if len(links) > _MAX_BANKS * (_MAX_BANKS - 1) or len(benchmarks) > 4096:
            raise ValueError("hardware envelope exceeds bounded link/benchmark inventory")
        if any(not isinstance(row, LinkEnvelope) for row in links) \
                or any(not isinstance(row, KernelBenchmark) for row in benchmarks):
            raise ValueError("hardware envelope has malformed links or benchmarks")
        names = {row.name for row in banks}
        if len(names) != len(banks):
            raise ValueError("hardware bank names must be unique")
        pairs = set()
        for link in links:
            if link.source not in names or link.destination not in names:
                raise ValueError("hardware link references an unknown bank")
            key = (link.source, link.destination)
            if key in pairs:
                raise ValueError("hardware links must be unique and directed")
            pairs.add(key)
        channels = {row.channel for row in banks}
        if any(row.channel not in channels for row in benchmarks):
            raise ValueError("kernel benchmark names an unknown channel")
        benchmark_keys = [(row.operation, row.channel, row.weight_format) for row in benchmarks]
        if len(set(benchmark_keys)) != len(benchmark_keys):
            raise ValueError("hardware envelope has duplicate active benchmark evidence")
        object.__setattr__(self, "banks", banks); object.__setattr__(self, "links", links)
        object.__setattr__(self, "benchmarks", benchmarks)

    def bank(self, name: str) -> BankEnvelope:
        for bank in self.banks:
            if bank.name == name:
                return bank
        raise KeyError(name)

    def link(self, source: str, destination: str) -> LinkEnvelope | None:
        for link in self.links:
            if link.source == source and link.destination == destination:
                return link
        return None

    def to_dict(self) -> dict:
        return {"schema": self.schema, "name": self.name, "architecture": self.architecture,
                "banks": [asdict(row) for row in self.banks],
                "links": [asdict(row) for row in self.links],
                "benchmarks": [asdict(row) for row in self.benchmarks]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "HardwareEnvelope":
        doc = strict_json_loads(text, "hardware envelope", max_bytes=_MAX_JSON)
        expected = {"schema", "name", "architecture", "banks", "links", "benchmarks"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("hardware envelope has missing or unknown fields")
        try:
            banks = tuple(BankEnvelope(**row) for row in doc.pop("banks"))
            links = tuple(LinkEnvelope(**row) for row in doc.pop("links"))
            benchmarks = tuple(KernelBenchmark(**row) for row in doc.pop("benchmarks"))
            return cls(**doc, banks=banks, links=links, benchmarks=benchmarks)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed hardware envelope: {exc}") from exc


@dataclass(frozen=True)
class ModelWorkloadSpec:
    mode: str = "inference"
    batch_size: int = 1
    sessions: int = 1
    prompt_tokens: int = 1
    context_tokens: int = 1
    max_new_tokens: int = 1
    activation_dtype: str = "f32"
    kv_dtype: str = "f32"
    gradient_dtype: str = "f32"
    optimizer: str = "none"
    lora_rank: int = 0
    checkpoint_segments: int = 1
    formats: tuple[str, ...] = ("source", "bcirq8-group32")
    schema: str = _WORKLOAD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _WORKLOAD_SCHEMA:
            raise ValueError("unsupported model workload schema")
        _str(self.mode, "workload mode", choices=_WORKLOAD_MODES)
        for field in ("batch_size", "sessions", "prompt_tokens", "context_tokens",
                      "max_new_tokens", "checkpoint_segments"):
            _int(getattr(self, field), f"workload {field}", minimum=1)
        for field in ("activation_dtype", "kv_dtype", "gradient_dtype"):
            _str(getattr(self, field), f"workload {field}", choices=set(_DTYPE_BYTES))
        _str(self.optimizer, "workload optimizer", choices={"none", "adamw", "sgd"})
        _int(self.lora_rank, "workload lora_rank")
        formats = _tuple(self.formats, "workload formats")
        if not formats or len(set(formats)) != len(formats) \
                or any(item not in _FORMATS for item in formats):
            raise ValueError(f"workload formats must be unique members of {sorted(_FORMATS)}")
        if self.context_tokens < self.prompt_tokens + self.max_new_tokens:
            raise ValueError("context_tokens must cover prompt_tokens + max_new_tokens")
        if self.mode == "inference":
            if self.optimizer != "none" or self.lora_rank:
                raise ValueError("inference cannot request optimizer or LoRA state")
        elif self.mode == "lora":
            if self.optimizer == "none" or self.lora_rank < 1:
                raise ValueError("LoRA requires a positive rank and optimizer")
        elif self.optimizer == "none" or self.lora_rank:
            raise ValueError("training requires an optimizer and only LoRA accepts lora_rank")
        object.__setattr__(self, "formats", formats)

    def to_dict(self) -> dict:
        value = asdict(self); value["formats"] = list(self.formats); return value

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ModelWorkloadSpec":
        doc = strict_json_loads(text, "model workload", max_bytes=1 << 20)
        expected = {"schema", "mode", "batch_size", "sessions", "prompt_tokens",
                    "context_tokens", "max_new_tokens", "activation_dtype", "kv_dtype",
                    "gradient_dtype", "optimizer", "lora_rank", "checkpoint_segments",
                    "formats"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("model workload has missing or unknown fields")
        try:
            doc["formats"] = tuple(doc["formats"])
            return cls(**doc)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed model workload: {exc}") from exc


@dataclass(frozen=True)
class FormatSize:
    name: str
    classification: str
    file_bytes: int
    payload_bytes: int
    metadata_bytes: int
    group_size: int
    executable: bool
    reason: str = ""

    def __post_init__(self) -> None:
        _str(self.name, "format name", choices=_FORMATS)
        _str(self.classification, "format classification", choices=_CLASSIFICATIONS)
        for field in ("file_bytes", "payload_bytes", "metadata_bytes", "group_size"):
            _int(getattr(self, field), f"format {field}")
        if type(self.executable) is not bool or not isinstance(self.reason, str):
            raise ValueError("format executable/reason fields are malformed")
        if self.payload_bytes + self.metadata_bytes != self.file_bytes:
            raise ValueError("format payload and metadata bytes do not reconcile")
        if self.name == "source" and (self.classification != "exact" or self.group_size != 0):
            raise ValueError("source format must be exact and ungrouped")
        if self.name == "bcirq8-group32" \
                and (self.classification != "quantized" or self.group_size != 32):
            raise ValueError("BCIRQ8 format must be quantized group-32")
        if self.executable != (not self.reason):
            raise ValueError("format executable state and refusal reason are inconsistent")
        if self.executable and self.file_bytes < 1:
            raise ValueError("an executable format cannot be empty")


@dataclass(frozen=True)
class TrainingState:
    trainable_parameters: int
    trainable_tensor_count: int
    gradient_bytes: int
    optimizer_bytes: int
    optimizer_step_bytes: int
    master_weight_bytes: int
    adapter_weight_bytes: int

    def __post_init__(self) -> None:
        for field in ("trainable_parameters", "trainable_tensor_count", "gradient_bytes",
                      "optimizer_bytes", "optimizer_step_bytes", "master_weight_bytes",
                      "adapter_weight_bytes"):
            _int(getattr(self, field), f"training state {field}")
        if self.optimizer_step_bytes > self.optimizer_bytes:
            raise ValueError("optimizer step bytes cannot exceed total optimizer bytes")

    @property
    def total_bytes(self) -> int:
        return _checked_add(self.gradient_bytes, self.optimizer_bytes,
                            self.master_weight_bytes, self.adapter_weight_bytes)


@dataclass(frozen=True)
class PredictionInterval:
    operation: str
    lower_ns: int
    median_ns: int
    upper_ns: int
    tokens: int
    tokens_per_second_lower_q16: int
    tokens_per_second_median_q16: int
    tokens_per_second_upper_q16: int
    evidence: str

    def __post_init__(self) -> None:
        _str(self.operation, "prediction operation", choices={"prefill", "decode"})
        for field in ("lower_ns", "median_ns", "upper_ns", "tokens"):
            _int(getattr(self, field), f"prediction {field}", minimum=1)
        for field in ("tokens_per_second_lower_q16", "tokens_per_second_median_q16",
                      "tokens_per_second_upper_q16"):
            _int(getattr(self, field), f"prediction {field}")
        if not self.lower_ns <= self.median_ns <= self.upper_ns:
            raise ValueError("prediction interval latency ordering is invalid")
        if not (self.tokens_per_second_lower_q16 <= self.tokens_per_second_median_q16
                <= self.tokens_per_second_upper_q16):
            raise ValueError("prediction interval throughput ordering is invalid")
        if not isinstance(self.evidence, str) or len(self.evidence) != 64 \
                or any(ch not in "0123456789abcdef" for ch in self.evidence):
            raise ValueError("prediction evidence must be lowercase SHA-256")


@dataclass(frozen=True)
class BankPeak:
    bank: str
    weights_bytes: int
    kv_bytes: int
    activation_bytes: int
    training_bytes: int
    staging_bytes: int
    peak_bytes: int
    capacity_bytes: int

    def __post_init__(self) -> None:
        _str(self.bank, "bank peak bank")
        for field in ("weights_bytes", "kv_bytes", "activation_bytes", "training_bytes",
                      "staging_bytes", "peak_bytes", "capacity_bytes"):
            _int(getattr(self, field), f"bank peak {field}")
        expected = _checked_add(self.weights_bytes, self.kv_bytes, self.activation_bytes,
                                self.training_bytes, self.staging_bytes)
        if expected != self.peak_bytes:
            raise ValueError("bank peak components do not reconcile")


@dataclass(frozen=True)
class PlacementCandidate:
    candidate_id: str
    kind: str
    weight_format: str
    classification: str
    compute_banks: tuple[str, ...]
    backing_bank: str
    split_layer: int
    double_buffer: bool
    peaks: tuple[BankPeak, ...]
    prefill_transfer_bytes: int
    decode_transfer_bytes: int
    predictions: tuple[PredictionInterval, ...]
    feasible: bool
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _str(self.candidate_id, "candidate_id"); _str(self.kind, "candidate kind")
        _str(self.weight_format, "candidate format", choices=_FORMATS)
        _str(self.classification, "candidate classification", choices=_CLASSIFICATIONS)
        object.__setattr__(self, "compute_banks", _tuple(self.compute_banks, "compute_banks"))
        object.__setattr__(self, "peaks", _tuple(self.peaks, "candidate peaks"))
        object.__setattr__(self, "predictions", _tuple(self.predictions, "candidate predictions"))
        object.__setattr__(self, "blockers", _tuple(self.blockers, "candidate blockers"))
        if type(self.double_buffer) is not bool or type(self.feasible) is not bool:
            raise ValueError("candidate Boolean fields are malformed")
        if not self.compute_banks or any(not isinstance(name, str) or not name
                                         for name in self.compute_banks):
            raise ValueError("candidate compute_banks must be nonempty bank names")
        _str(self.backing_bank, "candidate backing_bank")
        _int(self.split_layer, "candidate split_layer", minimum=-1, maximum=(1 << 31) - 1)
        _int(self.prefill_transfer_bytes, "candidate prefill_transfer_bytes")
        _int(self.decode_transfer_bytes, "candidate decode_transfer_bytes")
        if any(not isinstance(row, BankPeak) for row in self.peaks) \
                or any(not isinstance(row, PredictionInterval) for row in self.predictions) \
                or any(not isinstance(row, str) or not row for row in self.blockers):
            raise ValueError("candidate peaks, predictions, or blockers are malformed")
        if self.feasible == bool(self.blockers):
            raise ValueError("candidate feasible/blocker state is inconsistent")


@dataclass(frozen=True)
class ModelCostReport:
    inventory_digest: str
    hardware_digest: str
    workload_digest: str
    formats: tuple[FormatSize, ...]
    kv_cache_bytes: int
    activation_workspace_bytes: int
    training_state: TrainingState
    candidates: tuple[PlacementCandidate, ...]
    schema: str = _REPORT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _REPORT_SCHEMA:
            raise ValueError("unsupported model cost report schema")
        for value, field in ((self.inventory_digest, "inventory_digest"),
                             (self.hardware_digest, "hardware_digest"),
                             (self.workload_digest, "workload_digest")):
            if not isinstance(value, str) or len(value) != 64 \
                    or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{field} must be lowercase SHA-256")
        object.__setattr__(self, "formats", _tuple(self.formats, "report formats"))
        object.__setattr__(self, "candidates", _tuple(self.candidates, "report candidates"))
        if any(not isinstance(row, FormatSize) for row in self.formats) \
                or any(not isinstance(row, PlacementCandidate) for row in self.candidates) \
                or not isinstance(self.training_state, TrainingState):
            raise ValueError("model cost report has malformed typed rows")
        if len({row.name for row in self.formats}) != len(self.formats) \
                or len({row.candidate_id for row in self.candidates}) != len(self.candidates):
            raise ValueError("model cost report has duplicate formats or candidate ids")
        format_names = {row.name for row in self.formats}
        if any(row.weight_format not in format_names for row in self.candidates):
            raise ValueError("model cost report candidate references an absent format")
        _int(self.kv_cache_bytes, "report kv_cache_bytes")
        _int(self.activation_workspace_bytes, "report activation_workspace_bytes")

    def to_dict(self) -> dict:
        return {"schema": self.schema, "inventory_digest": self.inventory_digest,
                "hardware_digest": self.hardware_digest, "workload_digest": self.workload_digest,
                "formats": [asdict(row) for row in self.formats],
                "kv_cache_bytes": self.kv_cache_bytes,
                "activation_workspace_bytes": self.activation_workspace_bytes,
                "training_state": asdict(self.training_state),
                "candidates": [_candidate_dict(row) for row in self.candidates]}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ModelCostReport":
        doc = strict_json_loads(text, "model cost report", max_bytes=_MAX_JSON)
        expected = {"schema", "inventory_digest", "hardware_digest", "workload_digest",
                    "formats", "kv_cache_bytes", "activation_workspace_bytes",
                    "training_state", "candidates"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("model cost report has missing or unknown fields")
        try:
            formats = tuple(FormatSize(**row) for row in doc.pop("formats"))
            training = TrainingState(**doc.pop("training_state"))
            candidates = []
            for row in doc.pop("candidates"):
                if not isinstance(row, dict):
                    raise ValueError("model cost report candidate must be an object")
                values = dict(row)
                values["compute_banks"] = tuple(values["compute_banks"])
                values["peaks"] = tuple(BankPeak(**peak) for peak in values["peaks"])
                values["predictions"] = tuple(PredictionInterval(**prediction)
                                              for prediction in values["predictions"])
                values["blockers"] = tuple(values["blockers"])
                candidates.append(PlacementCandidate(**values))
            return cls(**doc, formats=formats, training_state=training,
                       candidates=tuple(candidates))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed model cost report: {exc}") from exc


@dataclass(frozen=True)
class PlanAction:
    phase: int
    kind: str
    stage: str
    layer: int
    source_bank: str
    destination_bank: str
    nbytes: int
    operations: int
    repeats: int
    channel: str

    def __post_init__(self) -> None:
        # Phase 0 is the builder's transient "not attached yet" value. A persisted
        # ModelExecutionPlan requires the final contiguous sequence 1..N below.
        _int(self.phase, "plan action phase")
        _str(self.kind, "plan action kind",
             choices={"move", "prefetch", "compute", "barrier", "evict"})
        _str(self.stage, "plan action stage", choices={"prefill", "decode"})
        _int(self.layer, "plan action layer", minimum=-1, maximum=(1 << 31) - 1)
        _str(self.source_bank, "plan action source_bank")
        _str(self.destination_bank, "plan action destination_bank")
        _int(self.nbytes, "plan action nbytes")
        _int(self.operations, "plan action operations")
        if (self.kind == "compute") != (self.operations > 0):
            raise ValueError("only compute actions may carry a positive operation count")
        _int(self.repeats, "plan action repeats", minimum=1)
        _checked_mul(self.nbytes, self.repeats)
        _checked_mul(self.operations, self.repeats)
        _str(self.channel, "plan action channel")


@dataclass(frozen=True)
class ModelExecutionPlan:
    report_digest: str
    candidate_id: str
    classification: str
    actions: tuple[PlanAction, ...]
    module_digest: str
    streampack_sha256: str
    streampack_bytes: int
    schema: str = _PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != _PLAN_SCHEMA:
            raise ValueError("unsupported model execution plan schema")
        _str(self.candidate_id, "plan candidate_id")
        _str(self.classification, "plan classification", choices=_CLASSIFICATIONS)
        for value, field in ((self.report_digest, "report_digest"),
                             (self.module_digest, "module_digest"),
                             (self.streampack_sha256, "streampack_sha256")):
            if not isinstance(value, str) or len(value) != 64 \
                    or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"plan {field} must be lowercase SHA-256")
        object.__setattr__(self, "actions", _tuple(self.actions, "plan actions"))
        if not self.actions or any(not isinstance(row, PlanAction) for row in self.actions):
            raise ValueError("plan actions must be a nonempty typed array")
        if tuple(row.phase for row in self.actions) != tuple(range(1, len(self.actions) + 1)):
            raise ValueError("plan action phases must be contiguous and canonical")
        stages = tuple(row.stage for row in self.actions)
        if "prefill" not in stages or "decode" not in stages \
                or stages != tuple(sorted(stages, key={"prefill": 0, "decode": 1}.get)):
            raise ValueError("plan actions must contain contiguous prefill then decode templates")
        for stage in ("prefill", "decode"):
            repeats = {row.repeats for row in self.actions if row.stage == stage}
            if len(repeats) != 1 or (stage == "prefill" and repeats != {1}):
                raise ValueError("plan repeats apply uniformly to each complete stage template")
        _int(self.streampack_bytes, "plan streampack_bytes", minimum=1)

    def to_dict(self) -> dict:
        return {"schema": self.schema, "report_digest": self.report_digest,
                "candidate_id": self.candidate_id, "classification": self.classification,
                "actions": [asdict(row) for row in self.actions],
                "module_digest": self.module_digest,
                "streampack_sha256": self.streampack_sha256,
                "streampack_bytes": self.streampack_bytes}

    def to_json(self) -> str:
        return _canonical(self.to_dict())

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> "ModelExecutionPlan":
        doc = strict_json_loads(text, "model execution plan", max_bytes=_MAX_JSON)
        expected = {"schema", "report_digest", "candidate_id", "classification", "actions",
                    "module_digest", "streampack_sha256", "streampack_bytes"}
        if not isinstance(doc, dict) or set(doc) != expected:
            raise ValueError("model execution plan has missing or unknown fields")
        try:
            actions = tuple(PlanAction(**row) for row in doc.pop("actions"))
            return cls(**doc, actions=actions)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed model execution plan: {exc}") from exc


def _candidate_dict(row: PlacementCandidate) -> dict:
    return {"candidate_id": row.candidate_id, "kind": row.kind,
            "weight_format": row.weight_format, "classification": row.classification,
            "compute_banks": list(row.compute_banks), "backing_bank": row.backing_bank,
            "split_layer": row.split_layer, "double_buffer": row.double_buffer,
            "peaks": [asdict(peak) for peak in row.peaks],
            "prefill_transfer_bytes": row.prefill_transfer_bytes,
            "decode_transfer_bytes": row.decode_transfer_bytes,
            "predictions": [asdict(prediction) for prediction in row.predictions],
            "feasible": row.feasible, "blockers": list(row.blockers)}


def _bcirq8_compatible(inventory: TensorInventory) -> tuple[bool, str]:
    rows = tuple(row for row in inventory.tensors if not row.role.startswith("aux."))
    expected = 9 * inventory.num_hidden_layers + 2 + (0 if inventory.tied_embeddings else 1)
    if not all((inventory.vocab_size, inventory.hidden_size, inventory.intermediate_size,
                inventory.num_hidden_layers, inventory.num_attention_heads,
                inventory.num_key_value_heads)):
        return False, "decoder dimensions are incomplete"
    if len(rows) != expected:
        return False, f"BCIRQ8 Llama tensor census needs {expected} tensors; found {len(rows)}"
    if inventory.hidden_size % inventory.num_attention_heads \
            or inventory.num_attention_heads % inventory.num_key_value_heads:
        return False, "attention geometry is not divisible for BCIRQ8 v1"
    d = inventory.hidden_size; f = inventory.intermediate_size
    kvd = (d // inventory.num_attention_heads) * inventory.num_key_value_heads
    expected_shapes = {
        "attention.q": (d, d), "attention.k": (kvd, d),
        "attention.v": (kvd, d), "attention.o": (d, d),
        "mlp.gate": (f, d), "mlp.up": (f, d), "mlp.down": (d, f),
        "norm.attention": (d,), "norm.mlp": (d,),
        "embedding": (inventory.vocab_size, d), "norm.final": (d,),
        "lm_head": (inventory.vocab_size, d),
    }
    per_layer = {layer: [row for row in rows if row.layer == layer]
                 for layer in range(inventory.num_hidden_layers)}
    if any(len(layer_rows) != 9 for layer_rows in per_layer.values()):
        return False, "BCIRQ8 requires exactly nine tensors per decoder layer"
    required = {"attention.q", "attention.k", "attention.v", "attention.o",
                "mlp.gate", "mlp.up", "mlp.down", "norm.attention", "norm.mlp"}
    if any({row.role for row in layer_rows} != required for layer_rows in per_layer.values()):
        return False, "decoder layer roles do not match BCIRQ8 v1"
    for row in rows:
        if row.shape != expected_shapes.get(row.role):
            return False, f"tensor role {row.role} has a shape incompatible with BCIRQ8 v1"
    globals_ = {row.role for row in rows if row.layer < 0}
    expected_globals = {"embedding", "norm.final"} | (set() if inventory.tied_embeddings else {"lm_head"})
    if globals_ != expected_globals:
        return False, "global tensor roles do not match BCIRQ8 v1"
    return True, ""


def _q8_tensor_wire_bytes(count: int, *, group_size: int = 32) -> tuple[int, int]:
    exponent_bytes = 2 * ((count + group_size - 1) // group_size)
    return exponent_bytes, count


def _bcirq8_tensors(inventory: TensorInventory) -> list[TensorRecord]:
    by_role_layer = {(row.role, row.layer): row for row in inventory.tensors
                     if not row.role.startswith("aux.")}
    order = [by_role_layer[("embedding", -1)]]
    layer_roles = ("norm.attention", "attention.q", "attention.k", "attention.v",
                   "attention.o", "norm.mlp", "mlp.gate", "mlp.up", "mlp.down")
    for layer in range(inventory.num_hidden_layers):
        order.extend(by_role_layer[(role, layer)] for role in layer_roles)
    order.append(by_role_layer[("norm.final", -1)])
    if not inventory.tied_embeddings:
        order.append(by_role_layer[("lm_head", -1)])
    return order


def format_sizes(inventory: TensorInventory, formats: tuple[str, ...]) -> tuple[FormatSize, ...]:
    out = []
    compatible, compatibility_reason = _bcirq8_compatible(inventory)
    unsupported_dtypes = sorted({row.dtype for row in inventory.tensors
                                 if row.dtype not in {"F64", "F32", "F16", "BF16"}})
    for name in formats:
        if name == "source":
            reasons = []
            if not compatible:
                reasons.append(compatibility_reason)
            if unsupported_dtypes:
                reasons.append("source decoder cannot load dtypes " + ", ".join(unsupported_dtypes))
            out.append(FormatSize(name, "exact", inventory.source_file_bytes,
                                  inventory.payload_bytes, inventory.header_bytes, 0,
                                  not reasons, "; ".join(reasons)))
            continue
        if not compatible:
            out.append(FormatSize(name, "quantized", 0, 0, 0, 32, False,
                                  compatibility_reason))
            continue
        if unsupported_dtypes:
            out.append(FormatSize(name, "quantized", 0, 0, 0, 32, False,
                                  "BCIRQ8 source ingest cannot load dtypes "
                                  + ", ".join(unsupported_dtypes)))
            continue
        tensors = _bcirq8_tensors(inventory)
        offset = _align8(224 + len(tensors) * 48)
        payload = 0
        for tensor in tensors:
            aligned = _align8(offset); payload += aligned - offset; offset = aligned
            exponents, codes = _q8_tensor_wire_bytes(tensor.element_count)
            payload += exponents; offset += exponents
            aligned = _align8(offset); payload += aligned - offset; offset = aligned
            payload += codes; offset += codes
        metadata = offset - payload
        out.append(FormatSize(name, "quantized", offset, payload, metadata, 32, True))
    return tuple(out)


def format_tensor_sizes(inventory: TensorInventory, fmt: str) -> tuple[dict[str, int], int]:
    """Return exact per-tensor occupied spans plus non-tensor format metadata."""
    if fmt == "source":
        return {row.name: row.nbytes for row in inventory.tensors}, inventory.header_bytes
    tensors = _bcirq8_tensors(inventory)
    offset = _align8(224 + len(tensors) * 48)
    metadata = offset
    sizes = {}
    for tensor in tensors:
        start = offset
        offset = _align8(offset)
        exponents, codes = _q8_tensor_wire_bytes(tensor.element_count)
        offset += exponents
        offset = _align8(offset)
        offset += codes
        sizes[tensor.name] = offset - start
    return sizes, metadata


def _kv_bytes(inventory: TensorInventory, workload: ModelWorkloadSpec) -> int:
    if workload.mode != "inference":
        return 0  # training attention state belongs to the activation workspace, not decode KV
    if not inventory.num_attention_heads or inventory.hidden_size % inventory.num_attention_heads:
        raise ValueError("KV accounting requires divisible hidden_size/attention_heads")
    head_dim = inventory.hidden_size // inventory.num_attention_heads
    return _checked_mul(workload.sessions, workload.context_tokens,
                        inventory.num_hidden_layers, 2, inventory.num_key_value_heads,
                        head_dim, _DTYPE_BYTES[workload.kv_dtype])


def _activation_bytes(inventory: TensorInventory, workload: ModelWorkloadSpec) -> int:
    """Exact allocation of the BCIR fused-decoder workspace contract, not framework RSS."""
    elem = _DTYPE_BYTES[workload.activation_dtype]
    concurrent = (max(workload.sessions, _checked_mul(workload.batch_size,
                                                      workload.prompt_tokens))
                  if workload.mode == "inference"
                  else _checked_mul(workload.batch_size, workload.context_tokens))
    scratch_per_token = _checked_add(_checked_mul(4, inventory.hidden_size),
                                     _checked_mul(2, inventory.intermediate_size))
    scratch = _checked_mul(concurrent, scratch_per_token, elem)
    if workload.mode == "inference":
        logits = _checked_mul(workload.sessions, inventory.vocab_size, elem)
        return _checked_add(scratch, logits)
    train_tokens = _checked_mul(workload.batch_size, workload.context_tokens)
    logits = _checked_mul(train_tokens, inventory.vocab_size, elem)
    checkpoints = min(inventory.num_hidden_layers + 1, workload.checkpoint_segments + 1)
    saved = _checked_mul(train_tokens, inventory.hidden_size, elem, checkpoints)
    return _checked_add(scratch, logits, saved)


def _training_state(inventory: TensorInventory, workload: ModelWorkloadSpec) -> TrainingState:
    if workload.mode == "inference":
        return TrainingState(0, 0, 0, 0, 0, 0, 0)
    if workload.mode == "lora":
        targets = {"attention.q", "attention.k", "attention.v", "attention.o",
                   "mlp.gate", "mlp.up", "mlp.down"}
        trainable = 0; trainable_tensors = 0
        for tensor in inventory.tensors:
            if tensor.role in targets and len(tensor.shape) == 2:
                trainable = _checked_add(trainable,
                                         _checked_mul(workload.lora_rank,
                                                      tensor.shape[0] + tensor.shape[1]))
                trainable_tensors += 2  # one A and one B adapter tensor
        adapter = _checked_mul(trainable, _DTYPE_BYTES[workload.activation_dtype])
    else:
        trainable = inventory.parameter_count
        trainable_tensors = sum(not row.role.startswith("aux.") for row in inventory.tensors)
        adapter = 0
    gradients = _checked_mul(trainable, _DTYPE_BYTES[workload.gradient_dtype])
    # The hosted AdamW contract stores one validated scalar F32 step per trainable tensor,
    # matching the safe-checkpoint rail; moments are two F32 values per parameter.
    optimizer_steps = _checked_mul(trainable_tensors, 4) if workload.optimizer == "adamw" else 0
    optimizer = _checked_add(_checked_mul(trainable, 8), optimizer_steps) \
        if workload.optimizer == "adamw" else 0
    master = _checked_mul(trainable, 4) if workload.activation_dtype != "f32" else 0
    return TrainingState(trainable, trainable_tensors, gradients, optimizer,
                         optimizer_steps, master, adapter)


def _training_bytes_by_layer(inventory: TensorInventory, workload: ModelWorkloadSpec
                             ) -> tuple[tuple[int, ...], int]:
    if workload.mode == "inference":
        return (0,) * inventory.num_hidden_layers, 0
    target_roles = {"attention.q", "attention.k", "attention.v", "attention.o",
                    "mlp.gate", "mlp.up", "mlp.down"}
    per_layer = [0] * inventory.num_hidden_layers
    global_bytes = 0
    for tensor in inventory.tensors:
        if tensor.role.startswith("aux."):
            continue
        if workload.mode == "lora":
            if tensor.role not in target_roles or len(tensor.shape) != 2:
                continue
            params = _checked_mul(workload.lora_rank, tensor.shape[0] + tensor.shape[1])
            trainable_tensors = 2
            value_bytes = _checked_mul(params, _DTYPE_BYTES[workload.activation_dtype])
        else:
            params = tensor.element_count; trainable_tensors = 1; value_bytes = 0
        state = _checked_add(
            value_bytes,
            _checked_mul(params, _DTYPE_BYTES[workload.gradient_dtype]),
            _checked_mul(params, 8 if workload.optimizer == "adamw" else 0),
            _checked_mul(trainable_tensors, 4) if workload.optimizer == "adamw" else 0,
            _checked_mul(params, 4) if workload.activation_dtype != "f32" else 0)
        if 0 <= tensor.layer < inventory.num_hidden_layers:
            per_layer[tensor.layer] = _checked_add(per_layer[tensor.layer], state)
        else:
            global_bytes = _checked_add(global_bytes, state)
    return tuple(per_layer), global_bytes


def _model_operation_parts(inventory: TensorInventory, workload: ModelWorkloadSpec,
                           operation: str) -> tuple[int, int, int]:
    d = inventory.hidden_size; f = inventory.intermediate_size
    if not inventory.num_attention_heads or d % inventory.num_attention_heads:
        raise ValueError("operation accounting requires valid attention geometry")
    kvd = (d // inventory.num_attention_heads) * inventory.num_key_value_heads
    projection_macs = _checked_add(_checked_mul(2, d, d), _checked_mul(2, d, kvd),
                                   _checked_mul(3, d, f))
    if operation == "prefill":
        tokens = _checked_mul(workload.batch_size, workload.prompt_tokens)
        attention_macs = _checked_mul(2, workload.batch_size,
                                      workload.prompt_tokens, workload.prompt_tokens, d)
    else:
        tokens = _checked_mul(workload.sessions, workload.max_new_tokens)
        attention_macs = _checked_mul(2, workload.sessions,
                                      workload.max_new_tokens, workload.context_tokens, d)
    layer_operations = _checked_mul(2, _checked_add(
        _checked_mul(tokens, projection_macs), attention_macs))
    head_macs = _checked_mul(tokens, inventory.vocab_size, d)
    return tokens, layer_operations, _checked_mul(2, head_macs)


def _prediction(hardware: HardwareEnvelope, compute: tuple[BankEnvelope, ...],
                fmt: FormatSize, kind: str, split: int, peaks: tuple[BankPeak, ...],
                inventory: TensorInventory, workload: ModelWorkloadSpec,
                transfer_ns: dict[str, int]) -> tuple[PredictionInterval, ...]:
    peak_by_bank = {row.bank: row for row in peaks}
    out = []
    for operation in ("prefill", "decode"):
        tokens, layer_operations, head_operations = _model_operation_parts(
            inventory, workload, operation)
        if kind == "cpu-gpu-split":
            assignments = (
                (compute[0], _checked_mul(layer_operations, split)),
                (compute[1], _checked_add(
                    _checked_mul(layer_operations, inventory.num_hidden_layers - split),
                    head_operations)),
            )
        else:
            assignments = ((compute[-1], _checked_add(
                _checked_mul(layer_operations, inventory.num_hidden_layers),
                head_operations)),)
        bounds = [0, 0, 0]
        used_evidence = []
        complete = True
        for bank, operations in assignments:
            evidence = [row for row in hardware.benchmarks
                        if row.channel == bank.channel and row.operation == operation
                        and row.weight_format == fmt.name]
            if not evidence:
                complete = False
                break
            bench = min(evidence, key=lambda row: (row.upper_ns, row.median_ns, row.samples))
            peak = peak_by_bank[bank.name]
            target_weight_bytes = (fmt.file_bytes if kind == "layer-stream"
                                   else peak.weights_bytes)
            target_workspace_bytes = _checked_add(
                peak.kv_bytes, peak.activation_bytes, peak.staging_bytes)
            scale_q32 = max(
                1 << 32,
                ((operations << 32) + bench.operations - 1) // bench.operations,
                ((target_weight_bytes << 32) + bench.weight_bytes - 1)
                // bench.weight_bytes,
                ((target_workspace_bytes << 32) + bench.workspace_bytes - 1)
                // bench.workspace_bytes,
            )
            for index, value in enumerate((bench.lower_ns, bench.median_ns, bench.upper_ns)):
                scaled = max(1, (value * scale_q32 + (1 << 32) - 1) >> 32)
                _int(scaled, "scaled kernel prediction", minimum=1)
                bounds[index] = _checked_add(bounds[index], scaled)
            used_evidence.append({
                "kernel": asdict(bench), "operations": operations,
                "weight_bytes": target_weight_bytes,
                "workspace_bytes": target_workspace_bytes,
            })
        if not complete:
            continue
        transfer = transfer_ns.get(operation, 0)
        lower, median, upper = tuple(_checked_add(value, transfer) for value in bounds)
        # Throughput interval reverses latency endpoints. q16 preserves deterministic precision.
        throughput_numerator = _checked_mul(tokens, 1_000_000_000, 1 << 16)
        tps_low = throughput_numerator // upper
        tps_mid = throughput_numerator // median
        tps_high = throughput_numerator // lower
        evidence_id = hashlib.sha256(_canonical(
            {"kernels": used_evidence, "transfer_ns": transfer}).encode("utf-8")).hexdigest()
        out.append(PredictionInterval(operation, lower, median, upper, tokens,
                                      tps_low, tps_mid, tps_high, evidence_id))
    return tuple(out)


def _peak(bank: BankEnvelope, *, weights=0, kv=0, activation=0, training=0,
          staging=0) -> BankPeak:
    total = _checked_add(weights, kv, activation, training, staging)
    return BankPeak(bank.name, weights, kv, activation, training, staging,
                    total, bank.allocatable_bytes)


def _candidate(candidate_id: str, kind: str, fmt: FormatSize, compute: tuple[BankEnvelope, ...],
               backing: BankEnvelope, split: int, double_buffer: bool, peaks: tuple[BankPeak, ...],
               prefill_transfer: int, decode_transfer: int, hardware: HardwareEnvelope,
               inventory: TensorInventory, workload: ModelWorkloadSpec,
               extra_blockers=()) -> PlacementCandidate:
    blockers = list(extra_blockers)
    for peak in peaks:
        if peak.peak_bytes > peak.capacity_bytes:
            blockers.append(f"bank {peak.bank} needs {peak.peak_bytes} bytes; "
                            f"allocatable is {peak.capacity_bytes}")
    if not fmt.executable:
        blockers.append(fmt.reason or f"format {fmt.name} is not executable")
    if any(bank.name != backing.name and hardware.link(backing.name, bank.name) is None
           for bank in compute):
        blockers.append("required directed backing-to-compute link is absent")
    transfer_ns = {"prefill": 0, "decode": 0}
    if compute[-1].name != backing.name:
        link = hardware.link(backing.name, compute[-1].name)
        if link is not None:
            pre_events = inventory.num_hidden_layers if kind == "layer-stream" else 1
            dec_events = (_checked_mul(inventory.num_hidden_layers, workload.max_new_tokens)
                          if kind == "layer-stream" else workload.max_new_tokens)
            transfer_ns["prefill"] = ((prefill_transfer * 1_000_000_000
                                        + link.bandwidth_bps - 1) // link.bandwidth_bps
                                       + pre_events * link.latency_ns)
            transfer_ns["decode"] = ((decode_transfer * 1_000_000_000
                                       + link.bandwidth_bps - 1) // link.bandwidth_bps
                                      + dec_events * link.latency_ns)
    if fmt.executable:
        predictions = _prediction(hardware, compute, fmt, kind, split, peaks,
                                  inventory, workload, transfer_ns)
        missing_evidence = {"prefill", "decode"} - {row.operation for row in predictions}
        if missing_evidence:
            blockers.append("no matching bounded benchmark evidence for "
                            + ", ".join(sorted(missing_evidence)))
    else:
        predictions = ()
    blockers = tuple(sorted(set(blockers)))
    return PlacementCandidate(candidate_id, kind, fmt.name, fmt.classification,
                              tuple(bank.name for bank in compute), backing.name, split,
                              double_buffer, peaks, prefill_transfer, decode_transfer,
                              predictions, not blockers, blockers)


def assess_model(inventory: TensorInventory, hardware: HardwareEnvelope,
                 workload: ModelWorkloadSpec) -> ModelCostReport:
    """Enumerate deterministic resident, layer-streaming, and two-bank split plans."""
    if not isinstance(inventory, TensorInventory) or not isinstance(hardware, HardwareEnvelope) \
            or not isinstance(workload, ModelWorkloadSpec):
        raise ValueError("assess_model requires typed inventory/hardware/workload artifacts")
    if workload.context_tokens > inventory.max_position_embeddings \
            and inventory.max_position_embeddings:
        raise ValueError("workload context exceeds the model configuration")
    formats = format_sizes(inventory, workload.formats)
    if inventory.num_hidden_layers > _MAX_PLAN_LAYERS:
        raise ValueError(f"model has more than {_MAX_PLAN_LAYERS} plannable layers")
    n_banks = len(hardware.banks)
    split_pairs = sum(
        (left.channel == "host") != (right.channel == "host")
        and hardware.link(left.name, right.name) is not None
        for left in hardware.banks for right in hardware.banks if left.name != right.name)
    executable_formats = sum(row.executable for row in formats)
    estimate = ((len(formats) - executable_formats)
                + executable_formats * (n_banks + n_banks * (n_banks - 1)
                                         + split_pairs * max(0, inventory.num_hidden_layers - 1)))
    if estimate > _MAX_CANDIDATES:
        raise ValueError(f"model placement space has {estimate} candidates; "
                         f"bounded limit is {_MAX_CANDIDATES}")
    kv = _kv_bytes(inventory, workload)
    activation = _activation_bytes(inventory, workload)
    training = _training_state(inventory, workload)
    if workload.mode == "lora" and training.trainable_parameters == 0:
        raise ValueError("LoRA workload has no supported rank-2 target tensors")
    candidates: list[PlacementCandidate] = []

    for fmt in formats:
        if not fmt.executable:
            # Keep one explicit rejected row instead of silently dropping a requested format.
            bank = hardware.banks[0]
            candidates.append(_candidate(f"{fmt.name}:unsupported", "unsupported", fmt,
                                         (bank,), bank, -1, False,
                                         (_peak(bank),), 0, 0, hardware, inventory, workload))
            continue
        tensor_bytes, format_metadata = format_tensor_sizes(inventory, fmt.name)
        layer_bytes = tuple(sum(tensor_bytes[row.name] for row in inventory.tensors
                                if row.layer == layer and row.name in tensor_bytes)
                            for layer in range(inventory.num_hidden_layers))
        global_bytes = _checked_add(format_metadata,
                                    sum(tensor_bytes[row.name] for row in inventory.tensors
                                        if row.layer < 0 and row.name in tensor_bytes))
        weights = fmt.file_bytes
        state_total = training.total_bytes
        layer_training, global_training = _training_bytes_by_layer(inventory, workload)
        if _checked_add(sum(layer_training), global_training) != state_total:
            raise AssertionError("training-state layer accounting drifted")

        training_blockers = (() if workload.mode == "inference"
                             else ("training memory is accounted, but training-plan lowering "
                                   "is not implemented",))
        if fmt.name != "source" and workload.mode in ("full-finetune", "pretrain"):
            training_blockers += ("full-weight training on BCIRQ8 is not implemented",)

        # Every bank is a legitimate resident candidate. A candidate remains in the report
        # when infeasible so the reason is reviewable rather than silently pruned.
        for bank in hardware.banks:
            peak = _peak(bank, weights=weights, kv=kv, activation=activation,
                         training=state_total)
            candidates.append(_candidate(
                f"{fmt.name}:resident:{bank.name}", "resident", fmt, (bank,), bank, -1,
                False, (peak,), 0, 0, hardware, inventory, workload,
                extra_blockers=training_blockers))

        # Exact layer streaming from each backing bank to each distinct compute bank.
        max_layer = max((_checked_add(weight, state) for weight, state
                         in zip(layer_bytes, layer_training)), default=0)
        for backing in hardware.banks:
            for compute in hardware.banks:
                if backing.name == compute.name:
                    continue
                staged = _checked_mul(2, max_layer)
                compute_peak = _peak(compute, weights=global_bytes, kv=kv,
                                     activation=activation, training=global_training,
                                     staging=staged)
                backing_peak = _peak(backing, weights=weights, training=state_total)
                transferred = sum(layer_bytes)
                candidates.append(_candidate(
                    f"{fmt.name}:stream:{backing.name}->{compute.name}", "layer-stream",
                    fmt, (compute,), backing, -1, True, (backing_peak, compute_peak),
                    transferred, _checked_mul(transferred, workload.max_new_tokens),
                    hardware, inventory, workload, extra_blockers=training_blockers))

        # Exact contiguous two-bank placements. Globals and final head live with the
        # suffix; each layer executes where its weights and proportional KV state live.
        if inventory.num_hidden_layers > 1:
            kv_per_layer = kv // inventory.num_hidden_layers
            for left in hardware.banks:
                for right in hardware.banks:
                    host_device_pair = (left.channel == "host") != (right.channel == "host")
                    if (left.name == right.name or not host_device_pair
                            or hardware.link(left.name, right.name) is None):
                        continue
                    for split in range(1, inventory.num_hidden_layers):
                        left_weights = sum(layer_bytes[:split])
                        right_weights = _checked_add(global_bytes, sum(layer_bytes[split:]))
                        left_train = sum(layer_training[:split])
                        right_train = _checked_add(global_training,
                                                   sum(layer_training[split:]))
                        left_peak = _peak(left, weights=left_weights,
                                          kv=_checked_mul(kv_per_layer, split),
                                          activation=activation, training=left_train)
                        right_peak = _peak(right, weights=right_weights,
                                           kv=kv - _checked_mul(kv_per_layer, split),
                                           activation=activation, training=right_train)
                        boundary_tokens = (workload.batch_size * workload.prompt_tokens
                                           if workload.mode == "inference"
                                           else workload.batch_size * workload.context_tokens)
                        boundary = _checked_mul(boundary_tokens,
                                                inventory.hidden_size,
                                                _DTYPE_BYTES[workload.activation_dtype])
                        candidates.append(_candidate(
                            f"{fmt.name}:split:{left.name}:{split}:{right.name}", "cpu-gpu-split",
                            fmt, (left, right), left, split, False, (left_peak, right_peak),
                            boundary, _checked_mul(workload.sessions, workload.max_new_tokens,
                                                   inventory.hidden_size,
                                                   _DTYPE_BYTES[workload.activation_dtype]),
                            hardware, inventory, workload,
                            extra_blockers=training_blockers))

    return ModelCostReport(inventory.header_digest, hardware.digest, workload.digest,
                           formats, kv, activation, training, tuple(candidates))


def select_execution_candidate(report: ModelCostReport) -> PlacementCandidate:
    feasible = [row for row in report.candidates if row.feasible]
    if not feasible:
        raise ValueError("model has no feasible candidate with benchmark evidence")
    rank = {"exact": 0, "quantized": 1, "approximate": 2}
    return min(feasible, key=lambda row: (
        sum(p.median_ns for p in row.predictions),
        rank[row.classification], sum(peak.peak_bytes for peak in row.peaks),
        row.candidate_id))


__all__ = [
    "BankEnvelope", "FormatSize", "HardwareEnvelope", "KernelBenchmark", "LinkEnvelope",
    "ModelCostReport", "ModelExecutionPlan", "ModelWorkloadSpec", "PlanAction",
    "PlacementCandidate", "PredictionInterval", "TrainingState", "assess_model",
    "format_sizes", "format_tensor_sizes", "select_execution_candidate",
]
