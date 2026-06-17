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
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
        # Explicit flat dict: DataDNA is 9 scalar fields, so the recursive
        # dataclasses.asdict (which deepcopies every field) was pure overhead on the
        # hot telemetry-emit path (measured ~2M deepcopy calls in the suite).
        return {
            "segment_id": self.segment_id,
            "claim_id": self.claim_id,
            "cycles": self.cycles,
            "bytes": self.bytes,
            "misses": self.misses,
            "thermal": self.thermal,
            "voltage": self.voltage,
            "utilization": self.utilization,
            "provenance": self.provenance,
        }


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


@dataclass
class Broker(TelemetrySink):
    """A live pub/sub broker: a `TelemetrySink` that fans out every event to its
    subscribers. The runtime emits the data-DNA once; the broker delivers it to,
    e.g., a `ListSink` feeding the calibrator, a `FileSink` for audit, and a
    `KafkaSink` for production -- the live half of the calibration loop (the
    runtime publishes; the trained calibrator subscribes). Pure fan-out;
    deterministic delivery order."""

    subscribers: list = field(default_factory=list)

    def subscribe(self, sink: TelemetrySink) -> TelemetrySink:
        """Register a sink to receive every subsequent event; returns it."""
        self.subscribers.append(sink)
        return sink

    def emit(self, event: DataDNA) -> None:
        for sink in self.subscribers:
            sink.emit(event)

    def flush(self) -> None:
        for sink in self.subscribers:
            sink.flush()


@dataclass
class RingStats:
    written: int = 0
    read: int = 0
    dropped: int = 0          # records overwritten before they were read (ring full)


class TelemetryRing(TelemetrySink):
    """A zero-copy telemetry ring buffer between the kernel (writer) and the GNN /
    calibrator (reader).

    The buffer is a single **preallocated** ``bytearray`` -- the model of a fixed
    shared-memory region. The kernel packs the atomic numeric stats (claim_id,
    cycles, bytes, misses, thermal, voltage, utilization) directly into the buffer
    with ``struct.pack_into`` (no per-event allocation, no JSON, no syscall); the
    reader unpacks straight out of the *same* buffer with ``struct.unpack_from``.
    No serialization, no transport, non-blocking: an empty read returns ``None`` and
    a full ring overwrites its oldest unread record (counted as ``dropped``) rather
    than blocking. Fixed-width records mean head/tail are simple monotonic counters.

    It is also a `TelemetrySink`, so a `Broker` can fan events into it like any other
    backend -- but the hot path is `write`/`read_one`, which never leaves the buffer.
    """

    # claim_id, cycles, bytes, misses, thermal, voltage, utilization (7 x int64).
    _FMT = "<7q"

    def __init__(self, capacity: int = 1024):
        if capacity < 1:
            raise ValueError("ring capacity must be >= 1")
        self.capacity = capacity
        self.record_size = struct.calcsize(self._FMT)
        self.buf = bytearray(capacity * self.record_size)   # the fixed shared region
        self._head = 0          # total records written (monotonic)
        self._tail = 0          # total records read (monotonic)
        self.stats = RingStats()

    def write(self, event: DataDNA) -> None:
        """Pack one record into the shared buffer at the head slot (no allocation).
        If the ring is full, the oldest unread record is overwritten (dropped)."""
        slot = self._head % self.capacity
        struct.pack_into(self._FMT, self.buf, slot * self.record_size,
                         event.claim_id, event.cycles, event.bytes, event.misses,
                         event.thermal, event.voltage, event.utilization)
        self._head += 1
        self.stats.written += 1
        if self._head - self._tail > self.capacity:         # overwrote an unread slot
            self._tail = self._head - self.capacity
            self.stats.dropped += 1

    def emit(self, event: DataDNA) -> None:                 # TelemetrySink interface
        self.write(event)

    def read_one(self) -> DataDNA | None:
        """Unpack the oldest unread record straight from the buffer, or None if the
        ring is empty (non-blocking)."""
        if self._tail >= self._head:
            return None
        slot = self._tail % self.capacity
        cid, cyc, byt, mis, th, vo, ut = struct.unpack_from(
            self._FMT, self.buf, slot * self.record_size)
        self._tail += 1
        self.stats.read += 1
        return DataDNA(segment_id="", claim_id=cid, cycles=cyc, bytes=byt, misses=mis,
                       thermal=th, voltage=vo, utilization=ut)

    def drain(self) -> list[DataDNA]:
        out: list[DataDNA] = []
        while (e := self.read_one()) is not None:
            out.append(e)
        return out

    @property
    def pending(self) -> int:
        return self._head - self._tail


def parse_shared_ring(buf) -> list[DataDNA]:
    """Read records from a shared-memory ring written by the C producer
    (`lower.memory_model.emit_ring_header_c`): an mmap/bytes/bytearray with the
    32-byte header (head, capacity, record_size) + fixed records. Samples by pointer
    offset only -- no syscall, no serialization. Returns up to `min(head, capacity)`
    records, oldest-first (the live window the producer has published)."""
    head, capacity, record_size, _ = struct.unpack_from("<4Q", buf, 0)
    if capacity == 0 or record_size == 0:
        return []
    n = min(head, capacity)                 # records currently live in the ring
    base = 4 * 8
    out: list[DataDNA] = []
    # oldest live slot first (head has wrapped iff head > capacity).
    start = head - n
    for k in range(n):
        slot = (start + k) % capacity
        cid, cyc, byt, mis, th, vo, ut = struct.unpack_from(
            TelemetryRing._FMT, buf, base + slot * record_size)
        out.append(DataDNA(segment_id="", claim_id=cid, cycles=cyc, bytes=byt,
                           misses=mis, thermal=th, voltage=vo, utilization=ut))
    return out


class KafkaSink(TelemetrySink):
    """Kafka transport for the data-DNA loop (the production backend).

    Inject any producer that is duck-typed `.send(topic, value=bytes)` (+ optional
    `.flush()`), or use `KafkaSink.connect(...)` to build a kafka-python producer
    lazily. Events are serialized as JSON bytes onto `topic`.
    """

    def __init__(self, producer, topic: str = "bcir.data_dna"):
        self.producer = producer
        self.topic = topic

    def emit(self, event: DataDNA) -> None:
        self.producer.send(self.topic, value=json.dumps(event.to_dict()).encode("utf-8"))

    def flush(self) -> None:
        flush = getattr(self.producer, "flush", None)
        if callable(flush):
            flush()

    @classmethod
    def connect(cls, bootstrap_servers, topic: str = "bcir.data_dna") -> "KafkaSink":
        """Build a KafkaSink backed by a real kafka-python producer (lazy import)."""
        try:
            from kafka import KafkaProducer  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only with a broker
            raise ImportError(
                "KafkaSink.connect requires kafka-python (pip install kafka-python)"
            ) from exc
        return cls(KafkaProducer(bootstrap_servers=bootstrap_servers), topic)
