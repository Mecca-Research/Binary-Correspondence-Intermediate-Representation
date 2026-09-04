"""CT4: the "data DNA" telemetry schema + sinks.

Per-segment execution telemetry emitted by the runtime (the BCIR-trace data-DNA):
cycles/bytes/misses + thermal/voltage/utilization + provenance. A `TelemetrySink`
transports events; the calibrator (`bcir.kbcir.calibrate`) folds them back into the
runtime state Theta and the policy, closing the AI-guided optimization loop.

`thermal`, `voltage`, `utilization`, `misses` are normalized 0..100 pressures, so
they map directly onto Theta. Kafka is the intended production transport; the
sinks here are a null/in-memory/file (JSONL) interface that a broker backend can
later implement.

INTEGRITY (CT4-sec): telemetry is the one BCIR artifact that crosses into
decision-making, so a poisoned record skews Theta / the adaptive policy / the regret
gate. `DataDNA.validate` enforces the documented signed-i64 schema at ingest
(normalized fields are integers in ``[0, 100]`; IDs/counters are non-negative);
`sanitize_events` is the validating filter the calibrate/sense path runs first, and it
surfaces a `TelemetryIntegrity` witness (rejected count + monotonic sequence + an
``empty``/``blind`` flag) so a suppression or injection attack is *observable* rather
than a silent no-op. See `TelemetryIntegrity` for the explicit defends / does-NOT-defend
boundary (it bounds values + detects drop/replay/reorder; it does NOT authenticate a
lying-but-in-range PMU -- that residual trust is what `MeasuredReplanCertificate`'s
provenance-of-source gate is for).
"""

from __future__ import annotations

import json
import os
import stat
import struct
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# The documented normalized-pressure band (inclusive). A field outside it is bogus
# (an injected `thermal=10000`, a negative, a NaN/inf), not a real silicon reading.
NORM_MIN = 0
NORM_MAX = 100
_NORM_FIELDS = ("misses", "thermal", "voltage", "utilization")   # the 0..100 pressures
_COUNTER_FIELDS = ("cycles", "bytes")                            # non-negative counters
_I64_MAX = (1 << 63) - 1
_MAX_TEXT_FIELD = 4096


class TelemetryIntegrityError(ValueError):
    """A telemetry record violated the documented type/width/range schema. Raised by
    the strict-mode validators; the sink/stream path
    instead *drops and counts* the record (see `sanitize_events`)."""


class SharedRingError(ValueError):
    """A shared-memory ring header failed integrity validation (bad magic/version,
    wrong `record_size`, or a `capacity * record_size` that overruns the buffer).
    `parse_shared_ring` raises this instead of reading arbitrary offsets out of the
    mmap'd region (the OOB-read / disclosure surface)."""


# Self-describing header for the shared-memory ring (see `parse_shared_ring`). The
# header is 4 x u64; slot [3] (byte offset 24) was a `reserved` word == 0 in the
# original layout, so it is repurposed APPEND-ONLY as `magic<<16 | version`: a reader
# that sees 0 there falls back to the legacy validation (record_size must still match),
# and the C producer (`emit_ring_header_c`) now stamps the magic so the wire format is
# self-describing on both sides. ``BCIR`` packed big-end into the high 32 bits.
RING_MAGIC = 0x42434952            # "BCIR" (0x42 'B' 0x43 'C' 0x49 'I' 0x52 'R')
RING_VERSION = 1
RING_STAMP = (RING_MAGIC << 16) | RING_VERSION   # the value written into header slot [3]
_MAX_RING_CAPACITY = 1 << 20                     # 56 MiB of fixed-width records


def _is_i64(v) -> bool:
    """True iff ``v`` is an exact signed-64-bit integer (never a bool).

    DataDNA's host schema is intentionally no wider than its frozen ``<7q>`` wire
    projection.  Rejecting floats and oversized Python integers at ingest prevents a
    record from passing policy validation and then failing (or changing type) when it
    reaches the ring/frame codec."""
    return isinstance(v, int) and not isinstance(v, bool) and -(1 << 63) <= v <= _I64_MAX


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

    def violations(self) -> list[str]:
        """Return the list of schema violations (empty == a valid, in-range record).

        The documented contract, enforced here at the ingest boundary: the four
        normalized pressures (`misses`/`thermal`/`voltage`/`utilization`) are signed-
        64-bit ints in ``[0, 100]``; the `cycles`/`bytes` counters and `claim_id` are
        non-negative signed-64-bit ints. A float/bool, oversized Python int, NaN/inf,
        negative value, or pressure ``> 100`` is bogus and is reported so the
        ingest path can REJECT-and-count it before it reaches Theta or the policy."""
        bad: list[str] = []
        for f in ("segment_id", "provenance"):
            v = getattr(self, f)
            if (not isinstance(v, str) or len(v) > _MAX_TEXT_FIELD or
                    any(ord(ch) < 0x20 for ch in v)):
                bad.append(f"{f} not a bounded control-free string")
        if not _is_i64(self.claim_id):
            bad.append("claim_id not a signed-64-bit int")
        elif self.claim_id < 0:
            bad.append(f"claim_id negative ({self.claim_id})")
        for f in _COUNTER_FIELDS:
            v = getattr(self, f)
            if not _is_i64(v):
                bad.append(f"{f} not a signed-64-bit int")
            elif v < 0:
                bad.append(f"{f} negative ({v})")
        for f in _NORM_FIELDS:
            v = getattr(self, f)
            if not _is_i64(v):
                bad.append(f"{f} not a signed-64-bit int")
            elif v < NORM_MIN or v > NORM_MAX:
                bad.append(f"{f} out of [0,100] ({v})")
        return bad

    def is_valid(self) -> bool:
        """True iff the record satisfies the documented schema (no violations)."""
        return not self.violations()

    def validate(self) -> "DataDNA":
        """Strict gate: return `self` unchanged if valid, else raise. Use at a trust
        boundary where a bogus record must HARD-FAIL (a forged certificate's seed,
        a single record from an untrusted producer). The stream/sink path uses the
        soft `sanitize_events` (drop-and-count) instead, so one poisoned record in a
        batch cannot deny service to the whole telemetry loop."""
        bad = self.violations()
        if bad:
            raise TelemetryIntegrityError(
                f"DataDNA(segment_id={self.segment_id!r}, claim_id={self.claim_id!r}) "
                f"violates the documented 0..100 schema: {'; '.join(bad)}")
        return self

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


@dataclass(frozen=True)
class TelemetryIntegrity:
    """The per-stream integrity witness the calibrate/sense ingest path produces
    (the telemetry analogue of the plan/pack rail's R13 content digest).

    DEFENDS (what an attacker can no longer do once the calibrator runs through
    `sanitize_events`):
      - *value injection*: a record outside the documented type, signed-i64, or
        ``[0, 100]`` pressure contract is `rejected` (dropped + counted), so a
        forged ``thermal=10000`` never reaches the EWMA / the ``>=60`` policy thresholds.
      - *stream suppression / eviction*: an empty stream (or one wholly rejected, or one
        the ring dropped) is no longer a silent no-op -- `blind` is True and the caller
        (and the security harness) can assert the absence is surfaced.
      - *record reorder*: decreasing surviving claim IDs make `monotonic` False.
        Frame duplicate/reorder is tracked independently by the frame fields.

    Does NOT DEFEND (the honest residual-trust boundary, pinned like the autodiff
    limitation note): it CANNOT authenticate a *lying-but-in-range* PMU. A producer
    that fabricates plausible 0..100 values (e.g. the low-variance cost-masking attack
    in `kbcir.sensing`) passes value validation by construction -- a content MAC keyed
    to the producer would be required, which is out of scope here. The witness bounds
    the blast radius (no out-of-band value can move the policy, and a *missing* stream
    is visible); authenticating the *source* of an in-range value remains the job of
    `MeasuredReplanCertificate`'s provenance-of-source gate, which this complements
    rather than replaces."""

    accepted: int = 0          # records that passed the documented schema
    rejected: int = 0          # records dropped for a schema violation
    dropped: int = 0           # records evicted/lost upstream (ring overwrite, etc.)
    monotonic: bool = True     # surviving claim_ids never decrease
    seq_span: int = 0          # distinct strictly-increasing claim_ids observed
    frames_accepted: int = 0   # validated BTLM frames recovered from a frame transport
    frames_rejected: int = 0   # magic-aligned candidates failing flags/version/length/CRC
    frames_missing: int = 0    # sequence numbers skipped between recovered frames
    frames_reordered: int = 0  # recovered frames behind the current sequence watermark
    frames_duplicated: int = 0 # recovered frames repeating the current sequence number

    @property
    def frame_monotonic(self) -> bool:
        """True iff recovered frame sequence numbers had no duplicate/reorder."""
        return self.frames_reordered == 0 and self.frames_duplicated == 0

    @property
    def blind(self) -> bool:
        """True when no valid telemetry survived (empty / all-rejected / all-dropped):
        the suppression-attack signal. A calibrator that returns Theta unchanged must
        report `blind` so the no-throttle outcome is a visible decision, not silence."""
        return self.accepted == 0

    @property
    def clean(self) -> bool:
        """True iff every record/frame candidate was accepted with no loss or reorder."""
        return (self.rejected == 0 and self.dropped == 0 and self.monotonic
                and self.frames_rejected == 0 and self.frames_missing == 0
                and self.frame_monotonic and self.accepted > 0)


def sanitize_events(events, *, dropped: int = 0) -> tuple[list[DataDNA], TelemetryIntegrity]:
    """The ingest gate: filter a telemetry batch to the records that satisfy the
    documented schema and return them alongside a `TelemetryIntegrity` witness.

    REJECT-and-count policy (deliberate, not clamp): a clearly-bogus value (wrong
    type/width, negative, or pressure ``> 100``) is DROPPED, not clamped.
    Clamping a hostile ``thermal=10000`` to 100 would let the attacker *saturate* the
    policy (force ENERGY) at will -- a free lever -- whereas dropping it leaves the
    EWMA/average computed only over genuine in-range samples, so an injected record
    cannot move Theta at all. The cost is that a hostile batch can *shrink* (fewer
    samples), but that is surfaced as `rejected`/`blind`, never silent. Legitimate,
    fully in-range telemetry passes through UNCHANGED (same objects, same order)."""
    out: list[DataDNA] = []
    rejected = 0
    monotonic = True
    last_cid = None
    seq = 0                                       # count of distinct strictly-increasing claim_ids
    for e in events:
        if not isinstance(e, DataDNA) or not e.is_valid():
            rejected += 1
            continue
        out.append(e)
        # Non-decreasing claim_ids are legitimate (a batch of segments for one claim, or
        # an advancing publish counter). A *decrease* is a stale record resent after a
        # newer one -> reorder/replay, which flags `monotonic`. A strictly higher id
        # advances the sequence span.
        if last_cid is not None and e.claim_id < last_cid:
            monotonic = False
        elif last_cid is None or e.claim_id > last_cid:
            seq += 1
        last_cid = e.claim_id if last_cid is None else max(last_cid, e.claim_id)
    witness = TelemetryIntegrity(accepted=len(out), rejected=rejected, dropped=dropped,
                                 monotonic=monotonic, seq_span=seq)
    return out, witness


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


@dataclass
class ValidatingSink(TelemetrySink):
    """A boundary sink that enforces the documented schema on EVERY event before it
    reaches the wrapped (downstream) sink, counting rejections in a live witness.

    Place this at the *entry* of any untrusted telemetry transport (a `Broker`
    subscriber feeding the calibrator, a network/Kafka ingest tap) so a poisoned
    record is dropped + counted at the wire rather than silently folded into Theta.
    REJECT-and-count (never clamp), matching `sanitize_events`. The running
    `rejected`/`accepted`/`blind` are inspectable so suppression/injection are
    observable in the live stream, not just in a post-hoc batch."""

    inner: TelemetrySink = field(default_factory=lambda: NullSink())
    accepted: int = 0
    rejected: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False,
                                   repr=False, compare=False)

    def emit(self, event: DataDNA) -> None:
        # Keep validation, witness accounting, and downstream publication one
        # transition. Concurrent transport callbacks cannot lose a count or reorder
        # acceptance relative to the wrapped sink.
        with self._lock:
            if isinstance(event, DataDNA) and event.is_valid():
                self.accepted += 1
                self.inner.emit(event)
            else:
                self.rejected += 1        # dropped at the boundary; never reaches Theta

    def flush(self) -> None:
        with self._lock:
            self.inner.flush()

    @property
    def witness(self) -> "TelemetryIntegrity":
        with self._lock:
            return TelemetryIntegrity(accepted=self.accepted, rejected=self.rejected)

    @property
    def blind(self) -> bool:
        with self._lock:
            return self.accepted == 0


class FileSink(TelemetrySink):
    """Append-only JSONL sink (one event per line)."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._identity: tuple[int, int] | None = None
        self._lock = threading.Lock()

    def emit(self, event: DataDNA) -> None:
        event.validate()
        with self._lock:
            f, identity = _open_regular_text(self.path, exclusive=False)
            with f:
                if self._identity is not None and identity != self._identity:
                    raise TelemetryIntegrityError(
                        "telemetry sink path was replaced after publication began"
                    )
                f.write(json.dumps(event.to_dict()) + "\n")
            if self._identity is None:
                self._identity = identity


@dataclass
class Broker(TelemetrySink):
    """A live pub/sub broker: a `TelemetrySink` that fans out every event to its
    subscribers. The runtime emits the data-DNA once; the broker delivers it to,
    e.g., a `ListSink` feeding the calibrator, a `FileSink` for audit, and a
    `KafkaSink` for production -- the live half of the calibration loop (the
    runtime publishes; the trained calibrator subscribes). Pure fan-out;
    deterministic delivery order."""

    subscribers: list = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False,
                                   repr=False, compare=False)

    def subscribe(self, sink: TelemetrySink) -> TelemetrySink:
        """Register a sink to receive every subsequent event; returns it."""
        with self._lock:
            self.subscribers.append(sink)
        return sink

    def emit(self, event: DataDNA) -> None:
        # Serialize fan-out so two publishers cannot interleave sink order or
        # mutate a bounded ring between its slot write and head publication.
        with self._lock:
            for sink in self.subscribers:
                sink.emit(event)

    def flush(self) -> None:
        with self._lock:
            for sink in self.subscribers:
                sink.flush()


@dataclass
class RingStats:
    written: int = 0
    read: int = 0
    dropped: int = 0          # records overwritten before they were read (ring full)


class TelemetryRing(TelemetrySink):
    """An in-process zero-copy telemetry ring model between a producer and the GNN /
    calibrator reader.

    The buffer is a single **preallocated** ``bytearray`` -- the model of a fixed
    shared-memory region. The kernel packs the atomic numeric stats (claim_id,
    cycles, bytes, misses, thermal, voltage, utilization) directly into the buffer
    with ``struct.pack_into`` (no per-event allocation, no JSON, no syscall); the
    reader unpacks straight out of the *same* buffer with ``struct.unpack_from``.
    A lock makes each write/read/overwrite transition indivisible across Python
    threads.  This is not the future cross-process live ring: that ABI still needs
    acquire/release publication, per-slot generations, and peer-death handling.
    No serialization, no transport, non-blocking: an empty read returns ``None`` and
    a full ring overwrites its oldest unread record (counted as ``dropped``) rather
    than blocking. Fixed-width records mean head/tail are simple monotonic counters.

    It is also a `TelemetrySink`, so a `Broker` can fan events into it like any other
    backend -- but the hot path is `write`/`read_one`, which never leaves the buffer.
    """

    # claim_id, cycles, bytes, misses, thermal, voltage, utilization (7 x int64).
    _FMT = "<7q"

    def __init__(self, capacity: int = 1024):
        if type(capacity) is not int or not 1 <= capacity <= _MAX_RING_CAPACITY:
            raise ValueError(f"ring capacity must be an integer in [1, {_MAX_RING_CAPACITY}]")
        self.capacity = capacity
        self.record_size = struct.calcsize(self._FMT)
        self.buf = bytearray(capacity * self.record_size)   # the fixed shared region
        self._head = 0          # total records written (monotonic)
        self._tail = 0          # total records read (monotonic)
        self.stats = RingStats()
        self._lock = threading.Lock()

    def write(self, event: DataDNA) -> None:
        """Pack one record into the shared buffer at the head slot (no allocation).
        If the ring is full, the oldest unread record is overwritten (dropped)."""
        event.validate()
        with self._lock:
            slot = self._head % self.capacity
            struct.pack_into(self._FMT, self.buf, slot * self.record_size,
                             event.claim_id, event.cycles, event.bytes, event.misses,
                             event.thermal, event.voltage, event.utilization)
            self._head += 1
            self.stats.written += 1
            if self._head - self._tail > self.capacity:     # overwrote an unread slot
                self._tail = self._head - self.capacity
                self.stats.dropped += 1

    def emit(self, event: DataDNA) -> None:                 # TelemetrySink interface
        self.write(event)

    def read_one(self) -> DataDNA | None:
        """Unpack the oldest unread record straight from the buffer, or None if the
        ring is empty (non-blocking)."""
        with self._lock:
            if self._tail >= self._head:
                return None
            slot = self._tail % self.capacity
            cid, cyc, byt, mis, th, vo, ut = struct.unpack_from(
                self._FMT, self.buf, slot * self.record_size)
            event = DataDNA(segment_id="", claim_id=cid, cycles=cyc, bytes=byt,
                            misses=mis, thermal=th, voltage=vo, utilization=ut)
            event.validate()
            self._tail += 1
            self.stats.read += 1
            return event

    def drain(self) -> list[DataDNA]:
        out: list[DataDNA] = []
        while (e := self.read_one()) is not None:
            out.append(e)
        return out

    @property
    def pending(self) -> int:
        with self._lock:
            return self._head - self._tail

    @property
    def lost(self) -> bool:
        """True iff the ring has overwritten any unread record. A silent eviction is a
        telemetry-suppression vector (records vanish before the calibrator reads them);
        the caller can assert ``not ring.lost`` to make the loss a visible signal rather
        than a number buried in `stats.dropped`."""
        with self._lock:
            return self.stats.dropped > 0

    def witness(self) -> "TelemetryIntegrity":
        """Surface the ring's read/drop accounting as a `TelemetryIntegrity` witness so
        an eviction (`dropped > 0`) is observable on the same channel as the calibrate
        path's injection/suppression signals."""
        with self._lock:
            return TelemetryIntegrity(accepted=self.stats.read, rejected=0,
                                      dropped=self.stats.dropped)


_RING_HEADER_BYTES = 4 * 8              # 4 x u64 (head, capacity, record_size, magic/ver)


def parse_shared_ring(buf, *, strict: bool = False) -> list[DataDNA]:
    """Read records from a shared-memory ring written by the C producer
    (`lower.memory_model.emit_ring_header_c`): an mmap/bytes/bytearray with the
    32-byte header (head, capacity, record_size, magic) + fixed records. Samples by
    pointer offset only -- no syscall, no serialization. Returns up to
    `min(head, capacity)` records, oldest-first (the live window the producer published).

    INTEGRITY (CT4-sec): the header is no longer TRUSTED. A crafted header (a large
    `capacity` x a wrong `record_size`, or an out-of-range `head`) could otherwise make
    the loop `unpack_from` arbitrary offsets out of the mmap'd region (an OOB read /
    info disclosure). Before iterating this validates, against the *real* buffer length:

      - the buffer is at least one header long;
      - `magic` (header slot [3]) is `RING_STAMP` (self-describing) OR 0 (legacy
        producers; back-compatible). Any other value is a foreign/forged buffer;
      - `record_size` equals the real `TelemetryRing._FMT` stride (56) -- it is the
        unpack stride, so a lie here is the disclosure lever;
      - `header + capacity * record_size` fits within `len(buf)` (no slot is OOB);
      - `head` is an unsigned monotonic publish counter; any value is valid and slot
        selection is bounded by the validated capacity.

    On ANY mismatch it REJECTS the buffer: returns `[]` (or raises `SharedRingError`
    when `strict=True`) -- it never reads past the validated record region. A
    deliberately short buffer therefore yields `[]`, not an out-of-bounds unpack."""
    blen = len(buf)

    def _reject(msg: str) -> list[DataDNA]:
        if strict:
            raise SharedRingError(msg)
        return []

    if blen < _RING_HEADER_BYTES:
        return _reject(f"buffer too short for header ({blen} < {_RING_HEADER_BYTES})")
    head, capacity, record_size, magic = struct.unpack_from("<4Q", buf, 0)

    if magic not in (RING_STAMP, 0):
        return _reject(f"bad ring magic/version 0x{magic:x} (expected 0x{RING_STAMP:x})")
    if capacity == 0 or record_size == 0:
        return []                            # an empty/uninitialized ring (legitimate)
    if capacity > _MAX_RING_CAPACITY:
        return _reject(f"ring capacity {capacity} exceeds {_MAX_RING_CAPACITY}")

    real_stride = struct.calcsize(TelemetryRing._FMT)
    if record_size != real_stride:
        return _reject(f"record_size {record_size} != real stride {real_stride} "
                       "(record-format mismatch / OOB-read lever)")
    # bounds: every live slot index must address a full record inside the buffer.
    need = _RING_HEADER_BYTES + capacity * record_size
    if need > blen:
        return _reject(f"capacity*record_size overruns buffer "
                       f"({need} > {blen}); header is forged/corrupt")
    n = min(head, capacity)                  # records currently live in the ring
    base = _RING_HEADER_BYTES
    out: list[DataDNA] = []
    start = head - n                         # oldest live slot first (head wrapped iff > capacity)
    for k in range(n):
        slot = (start + k) % capacity
        off = base + slot * record_size
        if off + real_stride > blen:         # defensive: never unpack past the buffer
            return _reject(f"record at offset {off} would overrun buffer {blen}")
        cid, cyc, byt, mis, th, vo, ut = struct.unpack_from(TelemetryRing._FMT, buf, off)
        event = DataDNA(segment_id="", claim_id=cid, cycles=cyc, bytes=byt,
                        misses=mis, thermal=th, voltage=vo, utilization=ut)
        if event.violations():
            return _reject(f"record at slot {slot} violates the telemetry schema")
        out.append(event)
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
                "KafkaSink.connect requires kafka-python (pip install 'bcir[telemetry-kafka]')"
            ) from exc
        return cls(KafkaProducer(bootstrap_servers=bootstrap_servers), topic)


# --- durable telemetry (release rung 0.3): the schema-tagged persistent log ----------------------

TELEMETRY_LOG_KIND = "bcir.telemetry_log"
TELEMETRY_LOG_SCHEMA = 1
_SCHEMA_FIELDS = {1: ("segment_id", "claim_id", "cycles", "bytes", "misses",
                      "thermal", "voltage", "utilization", "provenance")}
_MAX_DURABLE_LOG_BYTES = 64 << 20
_MAX_DURABLE_LINE_BYTES = 1 << 20
_MAX_DURABLE_RECORDS = 1 << 20


def _open_regular_text(path: str, *, exclusive: bool):
    """Open a telemetry output without following or racing a replacement path.

    ``DurableLog`` creates a new file exclusively; append sinks may reuse a regular
    file but bind to its device/inode after their first write.  ``O_NOFOLLOW`` closes
    the Unix symlink case before open, while the post-open lstat/fstat comparison
    provides the same no-replacement check on hosts without that flag.
    """
    flags = os.O_WRONLY | os.O_CREAT
    flags |= os.O_EXCL if exclusive else os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(fd)
        named = os.lstat(path)
        if (not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(named.st_mode) or
                (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)):
            raise TelemetryIntegrityError(
                "telemetry output must be one stable, non-symlink regular file"
            )
        mode = "w" if exclusive else "a"
        return os.fdopen(fd, mode, encoding="utf-8", newline="\n"), (
            opened.st_dev, opened.st_ino
        )
    except BaseException:
        os.close(fd)
        raise


def _reject_json_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


class DurableLog(TelemetrySink):
    """The DURABLE half of the telemetry story (release rung 0.3): an append-only JSONL sink
    whose FIRST line is a self-describing header -- kind, schema version, and the exact field
    list that version carries -- so a stream written today decodes under any later build (the
    0.4b envelope discipline applied to telemetry). Every event line is a full DataDNA record;
    `load_durable_log` refuses an unknown kind or a NEWER schema loudly (a telemetry stream is
    never silently misread) and replays the records back as DataDNA."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        f, self._identity = _open_regular_text(self.path, exclusive=True)
        with f:                                        # a fresh log per run: header first
            f.write(json.dumps({"kind": TELEMETRY_LOG_KIND, "schema": TELEMETRY_LOG_SCHEMA,
                                "fields": list(_SCHEMA_FIELDS[TELEMETRY_LOG_SCHEMA])},
                               sort_keys=True) + "\n")

    def emit(self, event: DataDNA) -> None:
        event.validate()
        with self._lock:
            f, identity = _open_regular_text(self.path, exclusive=False)
            with f:
                if identity != self._identity:
                    raise TelemetryIntegrityError(
                        "durable telemetry path was replaced after log creation"
                    )
                f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")


def load_durable_log(path: str) -> list[DataDNA]:
    """Decode a durable telemetry log: validate the header (kind + a schema this build knows +
    the field list that schema documents), then replay every record. Refusals are LOUD."""
    lines = []
    total = 0
    try:
        with open(path, "rb") as f:
            for raw in f:
                total += len(raw)
                if total > _MAX_DURABLE_LOG_BYTES:
                    raise ValueError(f"durable telemetry log exceeds {_MAX_DURABLE_LOG_BYTES} bytes")
                if len(raw) > _MAX_DURABLE_LINE_BYTES:
                    raise ValueError(f"durable telemetry line exceeds {_MAX_DURABLE_LINE_BYTES} bytes")
                if raw.strip():
                    lines.append(raw.decode("utf-8"))
                    if len(lines) > _MAX_DURABLE_RECORDS + 1:
                        raise ValueError("durable telemetry log has too many records")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"invalid durable telemetry log {path!r}: {exc}") from exc
    if not lines:
        raise ValueError("durable telemetry log is empty (no header)")
    try:
        head = json.loads(lines[0], object_pairs_hook=_reject_json_duplicates)
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError(f"invalid durable telemetry header: {exc}") from exc
    if not isinstance(head, dict) or set(head) != {"kind", "schema", "fields"}:
        raise ValueError("durable telemetry header has unknown or missing fields")
    if head.get("kind") != TELEMETRY_LOG_KIND:
        raise ValueError(f"not a {TELEMETRY_LOG_KIND} stream (kind={head.get('kind')!r})")
    version = head.get("schema")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("telemetry-log schema must be an integer")
    if version > TELEMETRY_LOG_SCHEMA:
        raise ValueError(f"telemetry-log schema v{version} is newer than this build's "
                         f"v{TELEMETRY_LOG_SCHEMA}; upgrade BCIR to read this stream")
    want = _SCHEMA_FIELDS.get(version)
    if want is None or tuple(head.get("fields", ())) != want:
        raise ValueError(f"telemetry-log header lies about schema v{version}'s fields")
    out = []
    fields = set(want)
    for index, line in enumerate(lines[1:]):
        try:
            doc = json.loads(line, object_pairs_hook=_reject_json_duplicates)
        except (json.JSONDecodeError, TypeError, RecursionError) as exc:
            raise ValueError(f"invalid durable telemetry record {index}: {exc}") from exc
        if not isinstance(doc, dict) or set(doc) != fields:
            raise ValueError(f"durable telemetry record {index} has unknown or missing fields")
        try:
            out.append(DataDNA(**doc).validate())
        except (TypeError, TelemetryIntegrityError) as exc:
            raise ValueError(f"invalid durable telemetry record {index}: {exc}") from exc
    return out
