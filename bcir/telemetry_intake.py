"""The host intake for TelemetryEnvelopeV0 records (G15, S3-B): the evidence boundary a consumer
applies to every envelope it receives, from any transport (the live ring, a UART, a file).

Per record, in the specification's order (docs/kernel/TELEMETRY_ENVELOPE_ABI.md, "Intake"):

  1. the wire laws (`decode_envelope`) -- a malformed record is refused with its status;
  2. the stream: (source, session) names one sequence stream; a bounded table of
     `INTAKE_STREAMS` streams is kept, and a record of a new stream when the table is full is
     refused (`BCIR_ERR_NOSPACE`) rather than evicting a stream and forgetting its continuity;
  3. continuity: the stream's `SequenceTracker` -- the frame ABI's own u32 predicate -- classifies
     the sequence (first / next / gap / duplicate / reorder) BEFORE any semantic refusal, so a
     record refused below is counted once, as refused, and never also as missing; the record's
     `lost` field (the producer's own drops) is summed into `reported_lost`;
  4. the signal: a sample whose signal the generated table does not define is refused when the
     producer marked it REQUIRED and skipped (counted, not delivered) otherwise;
  5. the generation: a record bound to an artifact generation older than the live one is stale
     (`is_stale`, the plane's own predicate) and one newer is ahead -- both refused
     (`BCIR_ERR_STALE`); generation 0 is unbound (a host sensor) and never stale;
  6. accepted. Duplicates and reorders are delivered and reported, as the frame ABI delivers and
     reports its frames; `witness()` hands accepted DataDNA to the RT3 gate (`sanitize_events`).

Telemetry is evidence, never a verdict: nothing here reaches `bcir/verify` or the cost vector.
The C twin is `bcir_tev_intake` (runtime/c/bcir_telemetry_envelope.h); both rails decide every
record identically (the S3-B parity traces).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .abi.telemetry_envelope import TelemetryEnvelope, TelemetryError, datadna_of, decode_envelope
from .gem.control import is_stale
from .signal_table import admit_signal, table_ids
from .telemetry import SequenceTracker, TelemetryIntegrity, sanitize_events

INTAKE_STREAMS = 16
INTAKE_VERDICTS = ("accepted", "skipped", "refused")
INTAKE_REASONS = ("none", "malformed", "streams", "unknown", "stale", "ahead")
CLASSIFICATIONS = ("none", "first", "next", "gap", "duplicate", "reorder")


@dataclass(frozen=True)
class IntakeOutcome:
    verdict: str
    reason: str = "none"
    status: str = "BCIR_OK"
    classification: str = "none"
    envelope: TelemetryEnvelope | None = None


@dataclass
class IntakeReport:
    accepted: int = 0
    skipped: int = 0
    refused: int = 0
    malformed: int = 0
    unknown: int = 0
    stale: int = 0
    ahead: int = 0
    streams_full: int = 0
    reported_lost: int = 0
    missing: int = 0
    reordered: int = 0
    duplicated: int = 0
    streams: int = 0


@dataclass
class _Stream:
    source: int
    session: int
    tracker: SequenceTracker = field(default_factory=SequenceTracker)


class TelemetryIntake:
    """One consumer's intake state: the live generation, the stream table and the counters."""

    def __init__(self, live_generation: int = 0, known_signals: frozenset[int] | None = None):
        if type(live_generation) is not int or not 0 <= live_generation <= 0xFFFFFFFF:
            raise ValueError("the live generation is an unsigned 32-bit integer")
        self.live_generation = live_generation
        self.known = table_ids() if known_signals is None else frozenset(known_signals)
        self._streams: list[_Stream] = []
        self._counts = IntakeReport()
        self._accepted: list[TelemetryEnvelope] = []

    def set_live_generation(self, generation: int) -> None:
        """The plane moved: records bound to the old generation are stale from here on."""
        if type(generation) is not int or not 0 <= generation <= 0xFFFFFFFF:
            raise ValueError("the live generation is an unsigned 32-bit integer")
        self.live_generation = generation

    def _stream(self, env: TelemetryEnvelope) -> _Stream | None:
        for stream in self._streams:
            if stream.source == env.source and stream.session == env.session:
                return stream
        if len(self._streams) >= INTAKE_STREAMS:
            return None
        stream = _Stream(env.source, env.session)
        self._streams.append(stream)
        return stream

    def _refuse(self, reason: str, status: str, classification: str = "none", env=None):
        self._counts.refused += 1
        return IntakeOutcome("refused", reason, status, classification, env)

    def admit(self, data) -> IntakeOutcome:
        """Decide one record's bytes."""
        try:
            env = decode_envelope(data)
        except TelemetryError as exc:
            self._counts.malformed += 1
            return self._refuse("malformed", exc.status)
        stream = self._stream(env)
        if stream is None:
            self._counts.streams_full += 1
            return self._refuse("streams", "BCIR_ERR_NOSPACE", env=env)
        classification = stream.tracker.observe(env.seq)
        self._counts.reported_lost += env.lost
        if env.kind == "sample":
            admitted = admit_signal(env.signal, env.required, self.known)
            if admitted == "refused":
                self._counts.unknown += 1
                return self._refuse("unknown", "BCIR_ERR_TELEMETRY", classification, env)
            if admitted == "skipped":
                self._counts.skipped += 1
                return IntakeOutcome("skipped", "unknown", "BCIR_OK", classification, env)
        if env.generation != 0:
            if is_stale(env.generation, self.live_generation):
                self._counts.stale += 1
                return self._refuse("stale", "BCIR_ERR_STALE", classification, env)
            if env.generation > self.live_generation:
                self._counts.ahead += 1
                return self._refuse("ahead", "BCIR_ERR_STALE", classification, env)
        self._counts.accepted += 1
        self._accepted.append(env)
        return IntakeOutcome("accepted", "none", "BCIR_OK", classification, env)

    def report(self) -> IntakeReport:
        """The counters, with the streams' continuity summed."""
        missing = sum(s.tracker.missing for s in self._streams)
        reordered = sum(s.tracker.reordered for s in self._streams)
        duplicated = sum(s.tracker.duplicated for s in self._streams)
        return replace(
            self._counts,
            missing=missing,
            reordered=reordered,
            duplicated=duplicated,
            streams=len(self._streams),
        )

    def stream_anomalies(self, source: int, session: int) -> tuple[int, int, int] | None:
        for stream in self._streams:
            if stream.source == source and stream.session == session:
                return stream.tracker.anomalies()
        return None

    def accepted(self) -> list[TelemetryEnvelope]:
        return list(self._accepted)

    def witness(self, *, dropped: int = 0) -> tuple[list, TelemetryIntegrity]:
        """The accepted DataDNA through the RT3 gate. `dropped` adds transport losses the intake
        cannot see (a ring's overwrite count); the producer's reported drops are added here."""
        records = [datadna_of(env) for env in self._accepted if env.kind == "datadna"]
        report = self.report()
        accepted, integrity = sanitize_events(records, dropped=dropped + report.reported_lost)
        return accepted, replace(
            integrity,
            frames_missing=report.missing,
            frames_reordered=report.reordered,
            frames_duplicated=report.duplicated,
        )
