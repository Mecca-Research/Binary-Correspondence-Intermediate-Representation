"""Append-only, content-addressed orchestration ledger for hosted training stages."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..._artifact_json import read_bounded_text, strict_json_loads
from .contracts import canonical_json, require_sha256, sha256_text


_PIPELINE_STAGES = frozenset({
    "data", "tokenizer", "pretrain", "sft", "reward", "dpo", "ppo", "reasoning",
    "embedding", "evaluate", "export",
})
_TRAINED = frozenset({"pretrain", "sft", "dpo", "ppo", "reasoning", "embedding"})


@dataclass(frozen=True)
class PipelineStageRecord:
    stage: str
    parent_sha256: tuple[str, ...]
    output_sha256: str
    report_sha256: str
    artifact_kind: str

    def __post_init__(self) -> None:
        if self.stage not in _PIPELINE_STAGES:
            raise ValueError("unknown pipeline stage")
        if not isinstance(self.parent_sha256, (tuple, list)) \
                or any(require_sha256(value, "parent_sha256") != value
                       for value in self.parent_sha256):
            raise ValueError("pipeline parents must be SHA-256 digests")
        object.__setattr__(self, "parent_sha256", tuple(self.parent_sha256))
        if len(set(self.parent_sha256)) != len(self.parent_sha256):
            raise ValueError("pipeline record repeats a parent")
        require_sha256(self.output_sha256, "output_sha256")
        require_sha256(self.report_sha256, "report_sha256")
        if not isinstance(self.artifact_kind, str) or not self.artifact_kind:
            raise ValueError("artifact_kind must be nonempty")

    @property
    def digest(self) -> str:
        return sha256_text(canonical_json(asdict(self)))


class TrainingPipelineLedger:
    def __init__(self, records=()):
        self._records = []
        for record in records:
            self.append(record)

    @property
    def records(self) -> tuple[PipelineStageRecord, ...]:
        return tuple(self._records)

    @property
    def digest(self) -> str:
        return sha256_text(self.to_json())

    def _parent_stages(self, record) -> list[str]:
        by_output = {row.output_sha256: row.stage for row in self._records}
        missing = [parent for parent in record.parent_sha256 if parent not in by_output]
        if missing:
            raise ValueError(f"pipeline record references unknown parent {missing[0]}")
        return [by_output[parent] for parent in record.parent_sha256]

    def append(self, record: PipelineStageRecord) -> None:
        if not isinstance(record, PipelineStageRecord):
            raise ValueError("record must be a PipelineStageRecord")
        if any(row.output_sha256 == record.output_sha256 for row in self._records):
            raise ValueError("pipeline output digest already exists")
        parents = self._parent_stages(record)
        stage = record.stage
        valid = False
        if stage == "data": valid = not parents
        elif stage == "tokenizer": valid = parents == ["data"]
        elif stage == "pretrain": valid = parents == ["tokenizer"]
        elif stage == "sft": valid = parents == ["pretrain"]
        elif stage == "reward": valid = parents == ["sft"]
        elif stage == "dpo": valid = parents == ["sft"]
        elif stage == "ppo": valid = len(parents) == 2 and "reward" in parents \
            and any(parent in ("sft", "dpo") for parent in parents)
        elif stage == "reasoning": valid = len(parents) == 1 and parents[0] in ("sft", "dpo", "ppo")
        elif stage == "embedding": valid = len(parents) == 1 and parents[0] in ("pretrain", "sft")
        elif stage == "evaluate": valid = len(parents) >= 1 and all(parent in _TRAINED for parent in parents)
        elif stage == "export": valid = parents == ["evaluate"]
        if not valid:
            raise ValueError(f"pipeline stage {stage!r} has invalid parent stages {parents}")
        self._records.append(record)

    def to_json(self) -> str:
        return canonical_json({"schema": "bcir.training_pipeline_ledger.v1",
                               "records": [asdict(record) for record in self._records]})

    @classmethod
    def from_json(cls, text: str) -> "TrainingPipelineLedger":
        value = strict_json_loads(text, "training pipeline ledger", max_bytes=16 * 1024 * 1024)
        if not isinstance(value, dict) or set(value) != {"schema", "records"} \
                or value["schema"] != "bcir.training_pipeline_ledger.v1" \
                or not isinstance(value["records"], list):
            raise ValueError("training pipeline ledger has missing or unknown fields")
        if len(value["records"]) > 100_000:
            raise ValueError("training pipeline ledger contains too many records")
        records = []
        try:
            expected = {"stage", "parent_sha256", "output_sha256", "report_sha256",
                        "artifact_kind"}
            for index, row in enumerate(value["records"]):
                if not isinstance(row, dict) or set(row) != expected:
                    raise ValueError(
                        f"training pipeline record {index} has missing or unknown fields")
                if not isinstance(row["parent_sha256"], list):
                    raise ValueError(
                        f"training pipeline record {index} parents must be a JSON list")
                records.append(PipelineStageRecord(
                    stage=row["stage"], parent_sha256=tuple(row["parent_sha256"]),
                    output_sha256=row["output_sha256"], report_sha256=row["report_sha256"],
                    artifact_kind=row["artifact_kind"]))
            return cls(records)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed training pipeline ledger: {exc}") from exc


def write_pipeline_ledger(path: os.PathLike | str, ledger: TrainingPipelineLedger) -> None:
    if not isinstance(ledger, TrainingPipelineLedger):
        raise ValueError("ledger must be a TrainingPipelineLedger")
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(ledger.to_json().encode("utf-8") + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_pipeline_ledger(path: os.PathLike | str) -> TrainingPipelineLedger:
    return TrainingPipelineLedger.from_json(read_bounded_text(
        str(path), "training pipeline ledger", max_bytes=16 * 1024 * 1024))
