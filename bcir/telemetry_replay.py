"""Transport replay: telemetry evidence replayed with the transport's own witness.

`telemetry.DurableLog` replays DataDNA records -- values without their transport. The only order
it can hand the RT3 gate (`sanitize_events`) is the records' static claim ids, and nothing it
stores says what the live consumer saw lost, reordered or duplicated: replayed telemetry was
not transport evidence (the 2026-09-04 review, finding 16).

An `EnvelopeLog` stores what the transport delivered instead: every TelemetryEnvelopeV0 record's
bytes in arrival order, malformed ones included, and every change of the live generation. It
closes with a trailer that retains the live intake's report (the witness), the transport's
loss the intake cannot see (a ring's overwrite count), the number of events and a SHA-256 over
every line before the trailer. `replay_envelope_log` re-runs those events through a fresh
`TelemetryIntake` -- the one evidence boundary, the one the live consumer ran and the one the C
twin `bcir_tev_intake` decides identically -- and refuses the log unless the replayed report
equals the retained one. The replay's `TelemetryIntegrity` therefore carries the transport's
sequence continuity (`frames_missing` / `frames_reordered` / `frames_duplicated`, per (source,
session) stream), not the claim ids', and a log that lost, reordered, altered or appended to
the records it was captured with is refused rather than replayed.

`RecordingIntake` is the capture side: an intake whose every input is logged before it is
decided, so the log and the live decisions cannot drift.

DOES NOT DEFEND: the digest is not a MAC. It detects a corrupted, truncated or reordered log,
and the report comparison detects a trailer that disagrees with its records; a writer that
fabricates records AND recomputes the trailer is out of scope, as it is for every telemetry
path here (`TelemetryIntegrity`'s boundary: in-range values are not authenticated).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass

from .telemetry import (
    DataDNA,
    TelemetryIntegrity,
    _open_regular_text,
    _reject_json_duplicates,
)
from .signal_table import table_ids
from .telemetry_intake import IntakeReport, TelemetryIntake

ENVELOPE_LOG_KIND = "bcir.envelope_log"
ENVELOPE_LOG_SCHEMA = 1

#: A record is kept as the transport delivered it, malformed or not, so the bound is a ring
#: slot's worth rather than an envelope's: anything longer is not a record any transport here
#: hands over, and the log refuses it on both sides (the writer never writes a log the reader
#: refuses).
MAX_LOG_RECORD = 4096
MAX_LOG_EVENTS = 1 << 20
MAX_LOG_BYTES = 64 << 20
_MAX_LINE = 2 * MAX_LOG_RECORD + 64
_U32 = 0xFFFFFFFF
_U64 = 0xFFFFFFFFFFFFFFFF
_HEX = frozenset("0123456789abcdef")
_REPORT_FIELDS = tuple(IntakeReport.__dataclass_fields__)


class ReplayError(ValueError):
    """An envelope log that cannot be replayed as the evidence it claims to be."""


def signal_table_fingerprint(known) -> str:
    """The signal ids an intake decides against, as a SHA-256 of their sorted u32le forms: a log
    captured against one table and replayed against another decides unknown signals differently,
    and says so by this rather than by a report mismatch nobody can explain."""
    ids = sorted(int(i) for i in known)
    return hashlib.sha256(b"".join(struct.pack("<I", i) for i in ids)).hexdigest()


def _uint(name: str, value, limit: int) -> int:
    if type(value) is not int or not 0 <= value <= limit:
        raise ReplayError(f"{name} must be an unsigned integer no greater than {limit}")
    return value


def _line(doc: dict) -> bytes:
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


class EnvelopeLog:
    """The capture file: a header, one line per event, and a trailer written by `close`."""

    def __init__(self, path, *, live_generation: int = 0, known_signals=None) -> None:
        _uint("the live generation", live_generation, _U32)
        known = table_ids() if known_signals is None else frozenset(known_signals)
        self.path = os.fspath(path)
        self._file, _ = _open_regular_text(self.path, exclusive=True)
        self._digest = hashlib.sha256()
        self._events = 0
        self._bytes = 0
        self._closed = False
        self._write(
            {
                "kind": ENVELOPE_LOG_KIND,
                "schema": ENVELOPE_LOG_SCHEMA,
                "live_generation": live_generation,
                "signal_table": signal_table_fingerprint(known),
            }
        )

    def _write(self, doc: dict, *, hashed: bool = True) -> None:
        if self._closed:
            raise ReplayError("the envelope log is closed")
        line = _line(doc)
        if self._bytes + len(line) > MAX_LOG_BYTES:
            raise ReplayError(f"an envelope log is at most {MAX_LOG_BYTES} bytes")
        self._file.write(line.decode("ascii"))
        self._bytes += len(line)
        if hashed:
            self._digest.update(line)

    def _event(self, doc: dict) -> None:
        if self._events >= MAX_LOG_EVENTS:
            raise ReplayError(f"an envelope log holds at most {MAX_LOG_EVENTS} events")
        self._write(doc)
        self._events += 1

    def record(self, data) -> None:
        """One record exactly as the transport delivered it."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ReplayError("a logged record is the transport's bytes")
        data = bytes(data)
        if len(data) > MAX_LOG_RECORD:
            raise ReplayError(f"a logged record is at most {MAX_LOG_RECORD} bytes")
        self._event({"record": data.hex()})

    def live(self, generation: int) -> None:
        """The live generation moved (the intake's `set_live_generation`)."""
        self._event({"live": _uint("the live generation", generation, _U32)})

    def close(self, report: IntakeReport, *, transport_lost: int = 0) -> str:
        """Retain the live intake's report and the transport's own loss count; returns the
        digest the trailer carries."""
        if not isinstance(report, IntakeReport):
            raise ReplayError("the retained witness is the live intake's IntakeReport")
        digest = self._digest.hexdigest()
        self._write(
            {
                "end": {
                    "digest": digest,
                    "events": self._events,
                    "report": asdict(report),
                    "transport_lost": _uint("the transport's loss", transport_lost, _U64),
                }
            },
            hashed=False,
        )
        self._closed = True
        self._file.close()
        return digest


class RecordingIntake:
    """A `TelemetryIntake` whose every input is logged before it is decided."""

    def __init__(self, path, live_generation: int = 0, known_signals=None) -> None:
        self.intake = TelemetryIntake(live_generation, known_signals)
        self.log = EnvelopeLog(
            path, live_generation=live_generation, known_signals=self.intake.known
        )

    def admit(self, data):
        self.log.record(data)
        return self.intake.admit(data)

    def set_live_generation(self, generation: int) -> None:
        self.intake.set_live_generation(generation)
        self.log.live(generation)

    def close(self, *, transport_lost: int = 0) -> str:
        return self.log.close(self.intake.report(), transport_lost=transport_lost)


@dataclass(frozen=True)
class EnvelopeReplay:
    """A replayed log: the intake that re-decided it, the DataDNA and `TelemetryIntegrity` its
    witness hands the RT3 gate, and what the trailer retained."""

    intake: TelemetryIntake
    records: list[DataDNA]
    integrity: TelemetryIntegrity
    report: IntakeReport
    transport_lost: int
    events: int
    digest: str


def _doc(raw: bytes, what: str) -> dict:
    try:
        text = raw.decode("ascii")
        doc = json.loads(
            text,
            object_pairs_hook=_reject_json_duplicates,
            parse_constant=lambda name: _refuse_constant(name),
            parse_float=lambda text: _refuse_float(text),
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ReplayError(f"{what} is not a JSON object: {exc}") from exc
    if not isinstance(doc, dict):
        raise ReplayError(f"{what} is not a JSON object")
    return doc


def _refuse_constant(name: str):
    raise ValueError(f"{name} is not an integer")


def _refuse_float(text: str):
    raise ValueError(f"{text} is not an integer")


def _hex(text, index: int) -> bytes:
    if (
        not isinstance(text, str)
        or len(text) % 2
        or len(text) > 2 * MAX_LOG_RECORD
        or not set(text) <= _HEX
    ):
        raise ReplayError(f"event {index} is not a record's lowercase hex")
    return bytes.fromhex(text)


def replay_envelope_log(path, *, known_signals=None) -> EnvelopeReplay:
    """Re-decide a captured log and hand back its transport witness, or refuse it."""
    lines: list[bytes] = []
    total = 0
    try:
        with open(path, "rb") as fh:
            for raw in fh:
                total += len(raw)
                if total > MAX_LOG_BYTES:
                    raise ReplayError(f"an envelope log is at most {MAX_LOG_BYTES} bytes")
                if len(raw) > _MAX_LINE:
                    raise ReplayError("an envelope log line exceeds its bound")
                if len(lines) > MAX_LOG_EVENTS + 1:
                    raise ReplayError(f"an envelope log holds at most {MAX_LOG_EVENTS} events")
                lines.append(raw)
    except OSError as exc:
        raise ReplayError(f"cannot read the envelope log {path!r}: {exc}") from exc
    if len(lines) < 2:
        raise ReplayError("the envelope log has no trailer: the capture never closed")
    if any(not raw.endswith(b"\n") for raw in lines):
        raise ReplayError("the envelope log is truncated mid-line")

    head = _doc(lines[0], "the header")
    if set(head) != {"kind", "schema", "live_generation", "signal_table"}:
        raise ReplayError("the header has unknown or missing fields")
    if head["kind"] != ENVELOPE_LOG_KIND:
        raise ReplayError(f"not an envelope log (kind={head['kind']!r})")
    if head["schema"] != ENVELOPE_LOG_SCHEMA or type(head["schema"]) is not int:
        raise ReplayError(f"envelope-log schema {head['schema']!r} is not one this build reads")
    live = _uint("the header's live generation", head["live_generation"], _U32)

    end = _doc(lines[-1], "the trailer")
    if set(end) != {"end"} or not isinstance(end["end"], dict):
        raise ReplayError("the envelope log has no trailer: the capture never closed")
    trailer = end["end"]
    if set(trailer) != {"digest", "events", "report", "transport_lost"}:
        raise ReplayError("the trailer has unknown or missing fields")
    report = trailer["report"]
    if not isinstance(report, dict) or set(report) != set(_REPORT_FIELDS):
        raise ReplayError("the retained report has unknown or missing fields")
    retained = IntakeReport(**{k: _uint(f"report.{k}", report[k], _U64) for k in _REPORT_FIELDS})
    transport_lost = _uint("the transport's loss", trailer["transport_lost"], _U64)

    digest = hashlib.sha256()
    digest.update(lines[0])
    events: list[tuple[str, object]] = []
    for index, raw in enumerate(lines[1:-1]):
        digest.update(raw)
        doc = _doc(raw, f"event {index}")
        if set(doc) == {"record"}:
            events.append(("record", _hex(doc["record"], index)))
        elif set(doc) == {"live"}:
            events.append(("live", _uint(f"event {index}'s generation", doc["live"], _U32)))
        else:
            raise ReplayError(f"event {index} is neither a record nor a generation change")
    if _uint("the trailer's event count", trailer["events"], MAX_LOG_EVENTS) != len(events):
        raise ReplayError(
            f"the trailer counts {trailer['events']} events, the log holds {len(events)}"
        )
    if trailer["digest"] != digest.hexdigest():
        raise ReplayError("the log's bytes are not the ones its trailer sealed")

    intake = TelemetryIntake(live, known_signals)
    if head["signal_table"] != signal_table_fingerprint(intake.known):
        raise ReplayError("the log was captured against a different signal table")
    for kind, value in events:
        if kind == "record":
            intake.admit(value)
        else:
            intake.set_live_generation(value)
    replayed = intake.report()
    if replayed != retained:
        differ = [k for k in _REPORT_FIELDS if getattr(replayed, k) != getattr(retained, k)]
        raise ReplayError(
            "the replayed intake disagrees with the witness the capture retained: "
            + ", ".join(f"{k} {getattr(replayed, k)} != {getattr(retained, k)}" for k in differ)
        )
    records, integrity = intake.witness(dropped=transport_lost)
    return EnvelopeReplay(
        intake=intake,
        records=records,
        integrity=integrity,
        report=replayed,
        transport_lost=transport_lost,
        events=len(events),
        digest=trailer["digest"],
    )


__all__ = [
    "ENVELOPE_LOG_KIND",
    "ENVELOPE_LOG_SCHEMA",
    "MAX_LOG_EVENTS",
    "MAX_LOG_RECORD",
    "EnvelopeLog",
    "EnvelopeReplay",
    "RecordingIntake",
    "ReplayError",
    "replay_envelope_log",
    "signal_table_fingerprint",
]
