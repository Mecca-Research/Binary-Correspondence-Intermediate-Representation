"""CT4: the "data DNA" telemetry schema + sinks.

Per-segment execution telemetry emitted by the runtime (the BCIR-trace data-DNA):
cycles/bytes/misses + thermal/voltage/utilization + provenance. A `TelemetrySink`
transports events; the calibrator (`bcir.kbcir.calibrate`) folds them back into the
runtime state Theta and the policy, closing the AI-guided optimization loop.

`thermal`, `voltage`, `utilization`, `misses` are normalized 0..100 pressures, so
they map directly onto Theta. Kafka is the intended production transport; the
sinks here are a null/in-memory/file (JSONL) interface that a broker backend can
later implement.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class DataDNA:
    segment_id: str
    claim_id: int
    cycles: int = 0
    bytes: int = 0
    misses: int = 0          # 0..100 normalized miss pressure
    thermal: int = 0         # 0..100
    voltage: int = 0         # 0..100 (0 = nominal)
    utilization: int = 0     # 0..100
    provenance: str = ""     # back-reference (e.g. plan/claim hash)

    def to_dict(self) -> dict:
        return asdict(self)


class TelemetrySink(ABC):
    """Transport for data-DNA events (Kafka in production; backends below now)."""

    @abstractmethod
    def emit(self, event: DataDNA) -> None: ...

    def flush(self) -> None:  # pragma: no cover - default no-op
        pass


class NullSink(TelemetrySink):
    def emit(self, event: DataDNA) -> None:
        pass


@dataclass
class ListSink(TelemetrySink):
    """In-memory sink for tests / the rehydrate loop."""

    events: list[DataDNA] = field(default_factory=list)

    def emit(self, event: DataDNA) -> None:
        self.events.append(event)


class FileSink(TelemetrySink):
    """Append-only JSONL sink (one event per line)."""

    def __init__(self, path: str):
        self.path = path

    def emit(self, event: DataDNA) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
