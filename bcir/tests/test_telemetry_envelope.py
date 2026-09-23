"""TelemetryEnvelopeV0 and the host intake (GEM+ roadmap G15, staged plan S3-B).

BTLM v1 carries records with no source, session, generation, clock or loss identity, so a
consumer could not tell a restarted producer from a reordered one, a producer's drop from a
transport's, or a measurement of the live artifact from one of the artifact it replaced. The
envelope carries all of it; the intake decides every record by its bytes:

    the wire laws        in one order on both rails; the encoder refuses what the decoder refuses;
                         a decoded record has exactly one spelling
    continuity           per (source, session), the frame ABI's own predicate (SequenceTracker),
                         classified before any semantic refusal so nothing is counted twice
    signals              an unknown REQUIRED signal refused, an unknown optional one skipped
    generations          a record of a generation the plane has left is stale, one it has not
                         reached is ahead -- both refused; generation 0 is unbound
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import replace

from bcir.abi.telemetry_envelope import (
    CLOCK_UNITS,
    ENVELOPE_HEADER_SIZE,
    ENVELOPE_SIZES,
    TelemetryEnvelope,
    TelemetryError,
    datadna_of,
    decode_envelope,
    encode_envelope,
)
from bcir.telemetry import SequenceTracker
from bcir.telemetry_frame import decode_frames, encode_frame
from bcir.telemetry_intake import INTAKE_STREAMS, TelemetryIntake
from bcir.tests import ring_fixtures as rf


def _status(data) -> str:
    try:
        decode_envelope(data)
    except TelemetryError as exc:
        return exc.status
    return "BCIR_OK"


def _raises(status: str, callback) -> None:
    try:
        callback()
    except TelemetryError as exc:
        assert exc.status == status, (exc.status, status, exc)
        return
    raise AssertionError(f"expected {status}")


# --- the bytes ------------------------------------------------------------------------------------


def test_every_envelope_round_trips_at_its_fixed_size():
    kinds = set()
    for name, data in rf.envelope_corpus():
        env = decode_envelope(data)
        assert encode_envelope(env) == data, name
        assert len(data) == ENVELOPE_SIZES[env.kind] == env.size, name
        kinds.add(env.kind)
    assert kinds == {"sample", "datadna"}
    assert ENVELOPE_SIZES == {"sample": 76, "datadna": 124} and ENVELOPE_HEADER_SIZE == 64


def test_every_wire_law_refuses_its_variant_with_its_status():
    variants = rf.malformed_envelopes()
    assert len({name for name, _, _ in variants}) == len(variants)
    for name, data, want in variants:
        assert _status(data) == want, name


def test_the_encoder_refuses_what_the_decoder_refuses():
    base = decode_envelope(rf.dna(1))
    smp = decode_envelope(rf.sample(1, 5))
    bad = {
        "unknown kind": replace(base, kind="video"),
        "source 0": replace(base, source=0),
        "session 0": replace(base, session=0),
        "seq beyond u32": replace(base, seq=1 << 32),
        "lost negative": replace(base, lost=-1),
        "bool as an int": replace(base, generation=True),
        "clock unknown": replace(base, clock="sundial"),
        "clock/unit pairing": replace(base, unit="cycles"),
        "timestamp without a clock": replace(base, clock="none", unit="none", timestamp=5),
        "datadna with a signal": replace(base, signal=1),
        "datadna required": replace(base, required=True),
        "datadna unbound": replace(base, generation=0),
        "datadna with a value": replace(base, value=3),
        "datadna short record": replace(base, record=base.record[:6]),
        "datadna field beyond i64": replace(base, record=(1 << 63,) + base.record[1:]),
        "sample signal 0": replace(smp, signal=0),
        "sample signal reserved": replace(smp, signal=0xFFFFFFFF),
        "sample with a record": replace(smp, record=base.record),
        "sample value beyond i64": replace(smp, value=1 << 63),
    }
    for label, env in bad.items():
        _raises("BCIR_ERR_TELEMETRY", lambda env=env: encode_envelope(env))
    # every clock names exactly the units it counts in, and nothing else is accepted
    for clock, units in CLOCK_UNITS.items():
        for unit in ("none", "ns", "us", "cycles"):
            env = replace(smp, clock=clock, unit=unit, timestamp=0 if clock == "none" else 9)
            if unit in units:
                assert decode_envelope(encode_envelope(env)) == env
            else:
                _raises("BCIR_ERR_TELEMETRY", lambda env=env: encode_envelope(env))


def test_the_first_violated_law_is_the_one_named():
    good = rf.dna(2)
    # a bad magic outranks everything after it; a bad version outranks the CRC and the fields
    both = b"BTEX" + bytes([0xFF]) + good[5:]
    assert _status(both) == "BCIR_ERR_MAGIC"
    version_and_crc = good[:4] + b"\x07\x00" + good[6:-1] + b"\x00"
    assert _status(version_and_crc) == "BCIR_ERR_VERSION"
    # the size is checked before the length, the length before the CRC
    assert _status(good[:100]) == "BCIR_ERR_TRUNCATED"
    assert _status(good[:12] + b"\x4c\x00" + good[14:100]) == "BCIR_ERR_TELEMETRY"


def test_a_decoded_record_has_one_spelling():
    """Canonical (Class A): nothing a decoder accepts can be re-encoded differently -- the REQUIRED
    flag, the kind's size and every reserved byte have exactly one legal value."""
    rng = random.Random(15)
    for _name, data in rf.envelope_corpus():
        for _ in range(64):
            mutated = bytearray(data)
            mutated[rng.randrange(len(data) - 4)] ^= 1 << rng.randrange(8)
            mutated = rf._recrc(bytes(mutated))
            try:
                env = decode_envelope(mutated)
            except TelemetryError:
                continue
            assert encode_envelope(env) == mutated


def test_datadna_carries_the_frozen_record_and_rt3_still_applies():
    env = decode_envelope(rf.dna(7))
    dna = datadna_of(env)
    assert (dna.claim_id, dna.cycles, dna.utilization) == (7, 49, 7 % 83)
    assert dna.is_valid()
    hostile = replace(env, record=(1, 2, 3, 4, 10_000, 5, 6))
    assert not datadna_of(decode_envelope(encode_envelope(hostile))).is_valid()


# --- continuity: one predicate --------------------------------------------------------------------


def test_the_frame_decoder_and_the_intake_share_one_continuity_predicate():
    """For random sequences the BTLM stream decoder and the envelope intake report the same
    (missing, reordered, duplicated) -- because both are `SequenceTracker`."""
    rng = random.Random(3)
    for _ in range(40):
        start = rng.choice((0, 0xFFFFFFF0, rng.randrange(1 << 32)))
        seqs = []
        at = start
        for _ in range(rng.randrange(1, 25)):
            at = (
                at + rng.choice((0, 1, 1, 1, 2, 5, -1 & 0xFFFFFFFF, -3 & 0xFFFFFFFF))
            ) & 0xFFFFFFFF
            seqs.append(at)
        frames = decode_frames(b"".join(encode_frame([], seq=s) for s in seqs))
        intake = TelemetryIntake(live_generation=1)
        tracker = SequenceTracker()
        for s in seqs:
            intake.admit(rf.dna(s))
            tracker.observe(s)
        report = intake.report()
        want = (frames.missing_frames, frames.reordered_frames, frames.duplicate_frames)
        assert (report.missing, report.reordered, report.duplicated) == want == tracker.anomalies()


def test_the_tracker_classifies_across_the_u32_wrap():
    t = SequenceTracker()
    got = [t.observe(s) for s in (0xFFFFFFFE, 0xFFFFFFFF, 0, 0, 3, 1)]
    assert got == ["first", "next", "next", "duplicate", "gap", "reorder"]
    assert t.anomalies() == (2, 1, 1)
    for bad in (-1, 1 << 32, 1.0, True):
        try:
            t.observe(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


# --- the intake -----------------------------------------------------------------------------------


def test_the_intake_classifies_before_it_refuses():
    """A record refused for its generation or its signal is still seen by continuity, so it is
    counted once (refused), never also as a gap."""
    intake = TelemetryIntake(live_generation=5)
    assert intake.admit(rf.dna(0, generation=5)).verdict == "accepted"
    stale = intake.admit(rf.dna(1, generation=4))
    assert (stale.verdict, stale.reason, stale.status, stale.classification) == (
        "refused",
        "stale",
        "BCIR_ERR_STALE",
        "next",
    )
    assert intake.admit(rf.dna(2, generation=5)).classification == "next"
    r = intake.report()
    assert (r.accepted, r.stale, r.missing) == (2, 1, 0)


def test_generations_unknown_signals_and_the_stream_bound():
    intake = TelemetryIntake(live_generation=3)
    assert intake.admit(rf.dna(0, generation=4)).reason == "ahead"
    assert intake.admit(rf.sample(1, 1, seq=1, generation=0)).verdict == "accepted"
    intake.set_live_generation(4)
    assert intake.admit(rf.dna(2, generation=4)).verdict == "accepted"
    assert intake.admit(rf.dna(3, generation=3)).reason == "stale"
    optional = intake.admit(rf.sample(0x80000001, 1, seq=4))
    assert (optional.verdict, optional.reason, optional.status) == ("skipped", "unknown", "BCIR_OK")
    required = intake.admit(rf.sample(0x10001, 1, seq=5, required=True))
    assert (required.verdict, required.status) == ("refused", "BCIR_ERR_TELEMETRY")
    for session in range(2, INTAKE_STREAMS + 1):
        assert intake.admit(rf.dna(0, session=session, generation=4)).verdict == "accepted"
    full = intake.admit(rf.dna(0, session=99, generation=4))
    assert (full.reason, full.status) == ("streams", "BCIR_ERR_NOSPACE")
    r = intake.report()
    assert r.streams == INTAKE_STREAMS and r.streams_full == 1 and r.unknown == 1 and r.skipped == 1


def test_the_witness_counts_the_producer_s_reported_drops():
    intake = TelemetryIntake(live_generation=1)
    for seq, lost in ((0, 0), (3, 2), (4, 0)):
        intake.admit(rf.dna(seq, lost=lost))
    records, witness = intake.witness(dropped=5)
    assert len(records) == 3 and witness.accepted == 3
    assert witness.dropped == 5 + 2 and witness.frames_missing == 2
    assert not witness.clean


# --- the C twin -----------------------------------------------------------------------------------


def test_the_c_twin_decodes_the_corpus_and_both_rails_re_encode_it_byte_for_byte():
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        corpus = rf.envelope_corpus()
        lines = rf.c_envelopes(exe, tmp, [data for _, data in corpus])
        assert len(lines) == len(corpus)
        for (name, data), line in zip(corpus, lines):
            env, c_bytes, status = rf.parse_c_envelope(line)
            assert status == "BCIR_OK", (name, line)
            assert encode_envelope(env) == data and c_bytes == data, name


def test_the_c_twin_refuses_every_variant_with_the_same_status():
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        variants = rf.malformed_envelopes()
        lines = rf.c_envelopes(exe, tmp, [data for _, data, _ in variants])
        assert len(lines) == len(variants)
        for (name, _data, want), line in zip(variants, lines):
            assert line == f"refused status={want}", (name, line)


def test_envelope_equality_is_value_equality():
    a = TelemetryEnvelope("sample", source=1, session=2, seq=3, signal=4, value=5)
    assert a == decode_envelope(encode_envelope(a)) and hash(a) == hash(replace(a))
