"""Transport replay (the 2026-09-04 review, finding 16): replayed telemetry carries the witness of
the transport it was captured from.

`telemetry.DurableLog` replays DataDNA values, and the RT3 gate can only order them by their
static claim ids: a replayed stream that lost records in transport replays as clean. An
`EnvelopeLog` keeps the envelopes as delivered, re-decides them through the live intake on
replay, and refuses a log whose re-decision is not the witness its capture retained. The tests
hold: the replay IS the capture (report, integrity, records); its continuity is the transport's
sequence, not the claim ids'; a ring capture replays with the ring's own loss; every way a log can
stop being its capture is refused; the writer never writes what the reader refuses; and the C
intake re-decides a log to the same report.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile

from bcir.abi.telemetry_envelope import TelemetryEnvelope, encode_envelope
from bcir.gem.ring import RingConsumer, RingGeometry, RingProducer, format_ring
from bcir.telemetry import DurableLog, load_durable_log, sanitize_events
from bcir.telemetry_replay import (
    MAX_LOG_RECORD,
    EnvelopeLog,
    RecordingIntake,
    ReplayError,
    replay_envelope_log,
)
from bcir.tests import ring_fixtures as rf

UNKNOWN_SIGNAL = 0x7FFFFFF0  # no generated signal has this id


def _stream():
    """Every intake law in one capture: two sessions; a gap, a duplicate and a reorder; a
    producer-reported drop; unknown required and optional signals; malformed records; a live
    generation change with a stale, an ahead and a current record after it."""
    events = [rf.dna(0), rf.dna(1), rf.dna(3), rf.dna(3), rf.dna(2), rf.dna(4, lost=2)]
    events += [rf.dna(0, session=2), rf.dna(1, session=2), rf.sample(3, 17, seq=0, session=3)]
    events += [rf.sample(UNKNOWN_SIGNAL, 1, seq=1, session=3, required=True)]
    events += [rf.sample(UNKNOWN_SIGNAL, 2, seq=2, session=3)]
    events += [bad for _name, bad, _status in rf.malformed_envelopes()[:5]]
    events += [
        ("live", 2),
        rf.dna(5, generation=1),
        rf.dna(6, generation=3),
        rf.dna(7, generation=2),
    ]
    return events


def _capture(tmp: str, events, *, live: int = 1, transport_lost: int = 0, name="capture.jsonl"):
    path = os.path.join(tmp, name)
    rec = RecordingIntake(path, live)
    for event in events:
        if isinstance(event, tuple):
            rec.set_live_generation(event[1])
        else:
            rec.admit(event)
    rec.close(transport_lost=transport_lost)
    return path, rec.intake


def test_a_replay_re_decides_the_capture_and_carries_its_witness():
    with tempfile.TemporaryDirectory() as tmp:
        path, live = _capture(tmp, _stream(), transport_lost=5)
        replay = replay_envelope_log(path)
        report = live.report()
        assert replay.report == report
        assert report.missing and report.reordered and report.duplicated  # the corpus reaches them
        assert report.refused and report.skipped and report.stale and report.ahead
        records, integrity = live.witness(dropped=5)
        assert replay.records == records and replay.integrity == integrity
        assert (
            replay.integrity.frames_missing,
            replay.integrity.frames_reordered,
            replay.integrity.frames_duplicated,
        ) == (report.missing, report.reordered, report.duplicated)
        assert replay.integrity.dropped == 5 + report.reported_lost
        assert replay.transport_lost == 5 and replay.events == len(_stream())


def test_replayed_continuity_is_the_transports_sequence_not_the_claim_ids():
    """The finding itself: claim ids 0, 1, 5, 6 rise, so a value replay orders them cleanly;
    the transport numbered them 0, 1, 5, 6 too, and three records never arrived."""
    envelopes = [rf.dna(seq) for seq in (0, 1, 5, 6)]
    with tempfile.TemporaryDirectory() as tmp:
        path, _live = _capture(tmp, envelopes)
        replay = replay_envelope_log(path)
        assert replay.integrity.frames_missing == 3 and not replay.integrity.clean

        # The value log of the very same records replays as a clean stream: its only sequence is
        # the claim ids, which rose (the defect the envelope log exists for).
        value_log = os.path.join(tmp, "values.jsonl")
        sink = DurableLog(value_log)
        for record in replay.records:
            sink.emit(record)
        _records, blind_witness = sanitize_events(load_durable_log(value_log))
        assert blind_witness.monotonic and blind_witness.frames_missing == 0
        assert blind_witness.clean


def test_a_ring_capture_replays_with_the_rings_own_loss():
    """An overwrite ring lapped by its producer: the consumer admits what it is handed through
    a recording intake and counts what the ring reports lost; the replay re-decides the same
    report, and the witness carries the ring's loss and the sequence gaps it left."""
    g = RingGeometry("overwrite", "telemetry", slot_size=192, slot_count=4, ring_id=0x7E01)
    region = bytearray(g.region_size)
    format_ring(region, g)
    producer, consumer = RingProducer(region), RingConsumer(region)
    assert producer.attach().verdict == "ok" and consumer.attach().verdict == "ok"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ring.jsonl")
        rec = RecordingIntake(path, 1)
        lost = seq = 0
        for burst in (3, 9, 2, 12, 1, 6):
            for _ in range(burst):
                assert producer.publish(rf.dna(seq)).verdict == "ok"
                seq += 1
            while True:
                out = consumer.consume()
                if out.verdict == "delivered":
                    rec.admit(out.payload)
                elif out.verdict == "lost":
                    lost += out.count
                elif out.verdict == "empty":
                    break
                else:
                    raise AssertionError(out)
        rec.close(transport_lost=lost)
        assert lost > 0 and consumer.accounting().lost == lost
        replay = replay_envelope_log(path)
        assert replay.report == rec.intake.report()
        assert replay.integrity.frames_missing == replay.report.missing > 0
        assert replay.integrity.dropped == lost
        # Every published sequence number was either delivered or counted missing (the first
        # record arrived, so no loss led the stream before the tracker's baseline).
        assert replay.report.accepted + replay.report.missing == seq
        assert replay.report.accepted == consumer.accounting().delivered


def _lines(path: str) -> list[bytes]:
    with open(path, "rb") as fh:
        return fh.readlines()


def _write(path: str, lines: list[bytes]) -> None:
    with open(path, "wb") as fh:
        fh.writelines(lines)


def _edit_trailer(lines: list[bytes], edit) -> list[bytes]:
    doc = json.loads(lines[-1])
    edit(doc["end"])
    return lines[:-1] + [(json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()]


def _reseal(lines: list[bytes]) -> list[bytes]:
    """Recompute the trailer's digest and event count over edited events, so a case breaks
    only the law it names (L11: a witness must hit the law it exists to test)."""
    digest = hashlib.sha256()
    for raw in lines[:-1]:
        digest.update(raw)
    return _edit_trailer(
        lines, lambda end: end.update(digest=digest.hexdigest(), events=len(lines) - 2)
    )


def _record_line(lines: list[bytes], envelope: bytes) -> int:
    want = b'{"record":"' + envelope.hex().encode() + b'"}\n'
    return lines.index(want)


def test_a_log_that_is_not_its_capture_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path, _live = _capture(tmp, _stream())
        good = _lines(path)
        first, second = _record_line(good, rf.dna(0)), _record_line(good, rf.dna(1))

        # The same stream, sequence and generation with other values: every decision (and so the
        # report) is unchanged, and only the digest says the evidence is not what was captured.
        forged = encode_envelope(
            TelemetryEnvelope(
                "datadna",
                source=1,
                session=1,
                seq=1,
                generation=1,
                clock="monotonic",
                unit="ns",
                timestamp=1001,
                record=(99, 1, 1, 1, 1, 1, 1),
            )
        )
        same_decisions = list(good)
        same_decisions[second] = b'{"record":"' + forged.hex().encode() + b'"}\n'

        def hexed(transform):
            lines = list(good)
            raw = lines[second]
            value = raw[len(b'{"record":"') : -len(b'"}\n')]
            lines[second] = b'{"record":"' + transform(value) + b'"}\n'
            return _reseal(lines)

        def lie(end):
            end["report"]["missing"] += 1

        swapped = list(good)
        swapped[first], swapped[second] = swapped[second], swapped[first]
        cases = {
            # the retained witness: the log re-decides to another report
            "a dropped record": _reseal(good[:second] + good[second + 1 :]),
            "swapped records": _reseal(swapped),
            "a witness that lies": _edit_trailer(good, lie),
            # the seal: the report is unchanged
            "other values, same decisions": same_decisions,
            "a count that lies": _edit_trailer(
                good, lambda end: end.update(events=end["events"] + 1)
            ),
            # the grammar: the bytes decode the same under a host parser's wider language
            "uppercase hex": hexed(lambda value: value.upper()),
            "spaced hex": hexed(lambda value: value[:2] + b" " + value[2:]),
            # the frame of the log
            "no trailer": good[:-1],
            "appended after the trailer": good + [good[1]],
            "a float count": _edit_trailer(
                good, lambda end: end.update(events=float(end["events"]))
            ),
            "an unknown event": good[:1] + [b'{"note":1}\n'] + good[1:],
            "a duplicate key": good[:1] + [b'{"live":2,"live":3}\n'] + good[1:],
            "a NaN generation": good[:1] + [b'{"live":NaN}\n'] + good[1:],
            "a boolean generation": good[:1] + [b'{"live":true}\n'] + good[1:],
            "an oversized line": good[:1]
            + [b'{"record":"' + b"00" * (MAX_LOG_RECORD + 1) + b'"}\n']
            + good[1:],
            "a torn last line": good[:-1] + [good[-1][:-1]],
            "another kind": [good[0].replace(b"bcir.envelope_log", b"bcir.telemetry_log")]
            + good[1:],
            "a newer schema": [good[0].replace(b'"schema":1', b'"schema":2')] + good[1:],
        }
        refused = []
        for name, lines in cases.items():
            bad = os.path.join(tmp, "bad.jsonl")
            _write(bad, lines)
            try:
                replay_envelope_log(bad)
            except ReplayError:
                refused.append(name)
        assert refused == list(cases), set(cases) - set(refused)

        # The same bytes against another signal table: refused, and said so.
        try:
            replay_envelope_log(path, known_signals=frozenset({1, 2}))
        except ReplayError as exc:
            assert "signal table" in str(exc)
        else:
            raise AssertionError("a log captured against another signal table replayed")
        assert replay_envelope_log(path).events == len(
            _stream()
        )  # the control: the capture replays


def test_the_writer_never_writes_a_log_the_reader_refuses():
    with tempfile.TemporaryDirectory() as tmp:
        log = EnvelopeLog(os.path.join(tmp, "w.jsonl"), live_generation=1)
        log.record(b"\x00" * MAX_LOG_RECORD)  # the bound itself is a record
        for bad in (b"\x00" * (MAX_LOG_RECORD + 1), "not bytes"):
            try:
                log.record(bad)
            except ReplayError:
                pass
            else:
                raise AssertionError(f"the writer logged {bad!r:.20}")
        for generation in (True, -1, 1 << 32):
            try:
                log.live(generation)
            except ReplayError:
                pass
            else:
                raise AssertionError(f"the writer logged generation {generation!r}")
        from bcir.telemetry_intake import TelemetryIntake

        intake = TelemetryIntake(1)
        intake.admit(b"\x00" * MAX_LOG_RECORD)
        log.close(intake.report())
        try:
            log.record(b"x")
        except ReplayError:
            pass
        else:
            raise AssertionError("a closed log took a record")
        assert replay_envelope_log(os.path.join(tmp, "w.jsonl")).report.malformed == 1


def test_the_c_intake_re_decides_a_log_to_the_same_report():
    """Two rails: the log's events through `bcir_tev_intake` (the ring harness's script mode)
    reach the report the Python replay reached. UNAVAILABLE without a C compiler."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        path, _live = _capture(tmp, _stream())
        replay = replay_envelope_log(path)
        ops = []
        for raw in _lines(path)[1:-1]:
            doc = json.loads(raw)
            if "live" in doc:
                ops.append(rf.Op(rf.LIVE, arg=doc["live"]))
            else:
                data = bytes.fromhex(doc["record"])
                assert data, "a zero-length INTAKE op means the last payload in the C script"
                ops.append(rf.Op(rf.INTAKE, data=data))
        ops.append(rf.Op(rf.CHECK))
        geometry = RingGeometry("backpressure", "telemetry", slot_size=192, slot_count=4, ring_id=9)
        scenario = rf.Scenario("replay", "continuity", geometry, 1, ops)
        trace = rf.run_c(exe, tmp, [scenario])[0]
        fields = trace[-1].split()
        assert fields[2] == "A", trace[-1]
        c_report = [int(v) for v in fields[-8:]]
        r = replay.report
        assert c_report == [
            r.accepted,
            r.skipped,
            r.refused,
            r.missing,
            r.reordered,
            r.duplicated,
            r.reported_lost,
            r.streams,
        ]
