"""T2: the UART telemetry frame (bcir/telemetry_frame.py <-> runtime/c/bcir_telemetry_frame.c).

A framed, CRC-sealed, resync-able telemetry transport -- the dual-rail (Python reference
<-> freestanding C twin) discipline mirroring StreamPack (test_c_encoder.py). These tests
pin: the Python round-trip (records + seq/timestamp preserved); resync-on-magic (a garbage
prefix / a corrupt frame between two good frames -> the decoder recovers the good frames +
counts the bad one, no crash); per-frame CRC (a flipped byte fails ONE frame, counted);
RT3 reuse (parse_uart_frames -> the same TelemetryIntegrity witness the ring path produces,
incl. an out-of-band record rejected by sanitize_events + a dropped/blind stream surfaced);
and the headline byte-identical dual-rail (toolchain-gated: compile the C twin C11+C23, feed
it a Python-encoded frame, assert the C decode round-trips AND the C re-encode is byte-
identical to encode_frame). A two-truth assertion pins that the frame path touches no
verifier / emits no Diagnostic.
"""

import os
import shutil
import subprocess
import tempfile
import zlib

from bcir.telemetry import DataDNA, TelemetryIntegrity, TelemetryRing, sanitize_events
from bcir.telemetry_frame import (
    CRC_SIZE,
    HEADER_SIZE,
    RECORD_STRIDE,
    TELEMETRY_FRAME_MAGIC,
    TELEMETRY_FRAME_VERSION,
    TelemetryFrameError,
    decode_frame,
    decode_frames,
    drain_ring_to_frame,
    encode_frame,
    frame_length,
    parse_uart_frames,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RUNTIME_C = os.path.join(_ROOT, "runtime", "c")


def _batch():
    """A fixed, fully RT3-VALID DataDNA batch (closed-form; the four pressures span the
    band extremes 0/100, counters non-negative -- so it passes sanitize_events unchanged
    and the witness is `clean`). Used for the RT3-reuse + round-trip assertions."""
    return [
        DataDNA(
            segment_id="",
            claim_id=1,
            cycles=100,
            bytes=200,
            misses=5,
            thermal=40,
            voltage=10,
            utilization=30,
        ),
        DataDNA(
            segment_id="",
            claim_id=2,
            cycles=999999,
            bytes=4096,
            misses=0,
            thermal=0,
            voltage=0,
            utilization=100,
        ),
        DataDNA(
            segment_id="",
            claim_id=3,
            cycles=12345,
            bytes=0,
            misses=100,
            thermal=99,
            voltage=50,
            utilization=0,
        ),
    ]


def _wire_batch():
    """A batch that exercises the WIRE encoding of two's-complement int64 -- a negative
    counter (the ring carries raw int64; the schema gate is a separate, host-side concern).
    Used for the round-trip + byte-identity rails (NOT the RT3-reuse path, which is value-
    gated and would legitimately reject the negative)."""
    return [
        DataDNA(
            segment_id="",
            claim_id=-7,
            cycles=-50,
            bytes=0,
            misses=0,
            thermal=99,
            voltage=0,
            utilization=100,
        ),
        DataDNA(
            segment_id="",
            claim_id=9223372036854775807,
            cycles=-1,
            bytes=200,
            misses=5,
            thermal=40,
            voltage=10,
            utilization=30,
        ),
        DataDNA(
            segment_id="",
            claim_id=-(1 << 63),
            cycles=-(1 << 63),
            bytes=0,
            misses=5,
            thermal=40,
            voltage=10,
            utilization=30,
        ),
    ]


# --- Python round-trip ----------------------------------------------------------


def test_frame_layout_constants():
    """The frozen frame ABI: magic "BTLM", v1, 22-byte header, 56-byte record, 4-byte CRC."""
    assert TELEMETRY_FRAME_MAGIC == b"BTLM"
    assert TELEMETRY_FRAME_VERSION == 1
    assert HEADER_SIZE == 22
    assert RECORD_STRIDE == 56  # reuses TelemetryRing._FMT ("<7q")
    assert CRC_SIZE == 4
    assert frame_length(3) == 22 + 56 * 3 + 4 == len(encode_frame(_batch()))
    for invalid in (-1, 1 << 16, True, 1.0):
        try:
            frame_length(invalid)
            assert False, "expected frame-length range rejection"
        except TelemetryFrameError:
            pass


def test_python_round_trip_preserves_records_and_meta():
    recs = _batch()
    frame = encode_frame(recs, seq=42, timestamp=0xDEADBEEF)
    assert frame[:4] == b"BTLM"
    out, meta = decode_frame(frame)
    assert out == recs  # every field round-trips
    assert meta.seq == 42 and meta.timestamp == 0xDEADBEEF
    assert meta.version == 1 and meta.n_records == len(recs)


def test_wire_round_trip_signed_int64():
    """The wire carries raw two's-complement int64 (a negative counter, INT64_MAX): the
    record bytes round-trip byte-exact independent of the value-schema gate."""
    recs = _wire_batch()
    out, meta = decode_frame(encode_frame(recs, seq=11, timestamp=22))
    assert out == recs and meta.seq == 11 and meta.timestamp == 22


def test_empty_frame_round_trips():
    frame = encode_frame([], seq=1, timestamp=7)
    assert len(frame) == HEADER_SIZE + CRC_SIZE  # an empty, well-formed frame
    out, meta = decode_frame(frame)
    assert out == [] and meta.seq == 1 and meta.timestamp == 7


def test_writer_rejects_values_it_cannot_represent_without_masking():
    """The v1 writer never silently wraps metadata or accepts a non-i64 body field."""

    class UnstableRecords:
        def __init__(self, declared, actual):
            self.declared = declared
            self.actual = actual

        def __len__(self):
            return self.declared

        def __iter__(self):
            return iter(self.actual)

    bad_calls = (
        lambda: encode_frame([], version=2),
        lambda: encode_frame([], version=1.0),
        lambda: encode_frame([], seq=-1),
        lambda: encode_frame([], seq=1 << 32),
        lambda: encode_frame([], timestamp=-1),
        lambda: encode_frame([], timestamp=1 << 64),
        lambda: encode_frame([DataDNA("", 1 << 63)]),
        lambda: encode_frame([DataDNA("", True)]),
        lambda: encode_frame(UnstableRecords(1, _batch()[:2])),
        lambda: encode_frame(UnstableRecords(2, _batch()[:1])),
    )
    for call in bad_calls:
        try:
            call()
            assert False, "expected representability rejection"
        except TelemetryFrameError:
            pass


def test_strict_decode_rejects_trailing_bytes():
    frame = encode_frame(_batch(), seq=3)
    for suffix in (b"x", encode_frame([], seq=4)):
        try:
            decode_frame(frame + suffix)
            assert False, "strict one-frame decode must reject trailing bytes"
        except TelemetryFrameError as exc:
            assert "trailing" in str(exc)


def test_drain_ring_to_frame_preserves_shared_record_layout():
    """The producer step drains the ring and preserves every value in the shared
    56-byte record schema/layout."""
    ring = TelemetryRing(capacity=8)
    recs = _batch()
    for r in recs:
        ring.write(r)
    frame = drain_ring_to_frame(ring, seq=3)
    out, meta = decode_frame(frame)
    assert [r.claim_id for r in out] == [1, 2, 3] and meta.seq == 3


# --- resync-on-magic ------------------------------------------------------------


def test_resync_recovers_after_garbage_prefix():
    """Joining a stream mid-flight: a garbage prefix -> the decoder scans to BTLM and
    recovers the frame (no crash, nothing rejected since the garbage held no magic-aligned
    frame)."""
    good = encode_frame(_batch(), seq=5)
    stream = decode_frames(b"\x00\x01rubbish\xff" + good)
    assert len(stream.frames) == 1
    assert stream.frames[0].records == _batch()
    assert stream.frames[0].meta.seq == 5


def test_resync_recovers_good_frames_around_a_corrupt_one():
    """A corrupt frame BETWEEN two good frames: the decoder recovers BOTH good frames and
    counts the bad one (the corrupt byte costs exactly one frame, not the stream)."""
    f1 = encode_frame(_batch(), seq=1)
    f2 = encode_frame(_batch(), seq=2)
    f3 = encode_frame(_batch(), seq=3)
    bad = bytearray(f2)
    bad[HEADER_SIZE + 3] ^= 0xFF  # flip a payload byte -> CRC fails
    stream = decode_frames(f1 + bytes(bad) + f3)
    seqs = [fr.meta.seq for fr in stream.frames]
    assert seqs == [1, 3]  # f2 dropped, f1 + f3 recovered
    assert stream.rejected == 1


def test_resync_handles_injected_garbage_between_frames():
    f1 = encode_frame(_batch(), seq=1)
    f2 = encode_frame(_batch(), seq=2)
    stream = decode_frames(f1 + b"\xde\xad\xbe\xef garbage \x00\x00" + f2)
    assert [fr.meta.seq for fr in stream.frames] == [1, 2]
    assert stream.rejected == 0  # the junk held no magic-aligned frame


# --- per-frame CRC --------------------------------------------------------------


def test_crc_mismatch_rejects_single_frame_not_crash():
    frame = bytearray(encode_frame(_batch(), seq=9))
    frame[HEADER_SIZE + 1] ^= 0x01  # corrupt one payload byte
    # strict single-frame path raises (never crashes); stream path rejects + counts
    try:
        decode_frame(bytes(frame))
        assert False, "expected a CRC failure"
    except TelemetryFrameError as exc:
        assert "CRC" in str(exc)
    stream = decode_frames(bytes(frame))
    assert stream.frames == [] and stream.rejected == 1


def test_bad_magic_and_version_rejected():
    frame = bytearray(encode_frame(_batch()))
    bad_magic = bytes(frame)
    bad_magic = b"XXXX" + bad_magic[4:]
    for exc_sub, buf in (("magic", bad_magic),):
        try:
            decode_frame(buf)
            assert False, "expected rejection"
        except TelemetryFrameError as e:
            assert exc_sub in str(e)
    # a future version is rejected by the v1 gate
    bumped = bytearray(encode_frame(_batch()))
    bumped[4] = 2  # version u16 low byte -> 2
    body = bytes(bumped[: HEADER_SIZE + len(_batch()) * RECORD_STRIDE])
    bumped[-4:] = (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(
        4, "little"
    )  # fix CRC so version is the only fault
    try:
        decode_frame(bytes(bumped))
        assert False, "expected a version rejection"
    except TelemetryFrameError as e:
        assert "version" in str(e)
    # v1 flags are reserved and must be zero on both rails.
    flagged = bytearray(encode_frame(_batch()))
    flagged[6:8] = (1).to_bytes(2, "little")
    body = bytes(flagged[:-CRC_SIZE])
    flagged[-CRC_SIZE:] = (zlib.crc32(body) & 0xFFFFFFFF).to_bytes(4, "little")
    try:
        decode_frame(bytes(flagged))
        assert False, "expected a reserved-flags rejection"
    except TelemetryFrameError as e:
        assert "flags" in str(e)


def test_truncated_frame_rejected_not_crashed():
    frame = encode_frame(_batch(), seq=1)
    for cut in (0, 4, HEADER_SIZE - 1, HEADER_SIZE, len(frame) - 1):
        stream = decode_frames(frame[:cut])  # never raises, never OOB
        assert isinstance(stream.frames, list)
    # a header claiming more records than the buffer holds is rejected
    short = frame[: HEADER_SIZE + RECORD_STRIDE]  # header says 3 records, body holds <1
    try:
        decode_frame(short)
        assert False, "expected truncation rejection"
    except TelemetryFrameError as e:
        assert "short" in str(e) or "truncated" in str(e)
    # A magic-aligned suffix is one truncated candidate, not silently ignored.
    assert decode_frames(b"BTLM").rejected == 1


def test_u32_frame_sequence_continuity_and_wraparound():
    def stream(*seqs):
        return decode_frames(b"".join(encode_frame([], seq=s) for s in seqs))

    gap = stream(10, 12)
    assert gap.missing_frames == 1 and gap.reordered_frames == 0
    dup = stream(10, 10)
    assert dup.duplicate_frames == 1 and dup.missing_frames == 0
    reorder = stream(10, 9, 11)
    assert reorder.reordered_frames == 1 and reorder.missing_frames == 0
    wrapped = stream(0xFFFFFFFF, 0)
    assert (wrapped.missing_frames, wrapped.reordered_frames, wrapped.duplicate_frames) == (0, 0, 0)


# --- RT3 reuse (the host decoder reuses sanitize_events) ------------------------


def test_parse_uart_frames_reuses_rt3_clean():
    """A clean batch reuses the ring path's RT3 record verdict and adds explicit
    evidence that one BTLM frame was validated."""
    recs = _batch()
    frame = encode_frame(recs, seq=1)
    out, witness = parse_uart_frames(frame)
    assert out == recs
    assert isinstance(witness, TelemetryIntegrity)
    ref_out, ref_witness = sanitize_events(recs)
    assert out == ref_out
    assert (
        witness.accepted,
        witness.rejected,
        witness.dropped,
        witness.monotonic,
        witness.seq_span,
    ) == (
        ref_witness.accepted,
        ref_witness.rejected,
        ref_witness.dropped,
        ref_witness.monotonic,
        ref_witness.seq_span,
    )
    assert witness.frames_accepted == 1 and witness.frames_rejected == 0
    assert witness.clean and not witness.blind


def test_parse_uart_frames_rejects_out_of_band_record():
    """An injected out-of-band DataDNA (thermal > 100) is REJECTED by sanitize_events on
    the host decode path -- exactly as on the ring path (RT3 reuse)."""
    good = _batch()
    poisoned = good + [DataDNA(segment_id="", claim_id=4, thermal=10000)]  # forged reading
    frame = encode_frame(poisoned, seq=1)
    out, witness = parse_uart_frames(frame)
    assert all(r.thermal <= 100 for r in out)  # the forged record never survives
    assert witness.rejected == 1 and witness.accepted == len(good)


def test_parse_uart_frames_surfaces_rejected_frames_without_guessing_record_loss():
    """A corrupt frame has an unknown record count, so it is a frame rejection rather
    than a guessed record-level ring drop."""
    f1 = encode_frame(_batch(), seq=1)
    bad = bytearray(encode_frame(_batch(), seq=2))
    bad[HEADER_SIZE] ^= 0xFF
    out, witness = parse_uart_frames(f1 + bytes(bad))
    assert witness.frames_accepted == 1 and witness.frames_rejected == 1
    assert witness.dropped == 0  # no invented record-loss count
    assert witness.accepted == len(_batch())


def test_parse_uart_frames_blind_on_empty_stream():
    """An empty / all-dropped stream is BLIND (the suppression signal), not a silent no-op."""
    out, witness = parse_uart_frames(b"")
    assert out == [] and witness.blind
    # a stream of only corrupt frames is also blind, with frame rejection counted
    bad = bytearray(encode_frame(_batch(), seq=1))
    bad[HEADER_SIZE] ^= 0xFF
    out2, w2 = parse_uart_frames(bytes(bad))
    assert (
        out2 == []
        and w2.blind
        and w2.frames_accepted == 0
        and w2.frames_rejected == 1
        and w2.dropped == 0
    )


def test_parse_uart_frames_surfaces_frame_sequence_anomalies():
    frames = b"".join(
        (
            encode_frame([], seq=4),
            encode_frame([], seq=6),
            encode_frame([], seq=6),
            encode_frame([], seq=5),
        )
    )
    _, witness = parse_uart_frames(frames)
    assert witness.frames_accepted == 4
    assert witness.frames_missing == 1
    assert witness.frames_duplicated == 1
    assert witness.frames_reordered == 1
    assert not witness.frame_monotonic and not witness.clean


def test_parse_uart_frames_reorder_surfaces_non_monotonic():
    """A reordered/replayed batch (a decreasing claim_id, a stale record resent after a
    newer one) makes the witness non-monotonic -- the seq/reorder signal RT3 provides."""
    reordered = [
        DataDNA(segment_id="", claim_id=5),
        DataDNA(segment_id="", claim_id=2),
    ]  # decrease == reorder/replay
    out, witness = parse_uart_frames(encode_frame(reordered, seq=1))
    assert not witness.monotonic


# --- two-truth ------------------------------------------------------------------


def test_two_truth_frame_path_touches_no_verifier():
    """The frame path produces records + an integrity witness, NEVER a Diagnostic and
    NEVER a verifier verdict. It returns DataDNA / TelemetryIntegrity only."""
    out, witness = parse_uart_frames(encode_frame(_batch()))
    assert all(isinstance(r, DataDNA) for r in out)
    assert isinstance(witness, TelemetryIntegrity)
    # the witness is a value record, not a verdict object: no 'diagnostic'/'verdict'/'legal' field
    fields = set(TelemetryIntegrity.__dataclass_fields__)
    assert not (fields & {"diagnostic", "verdict", "legal", "diagnostics"})


# --- byte-identical dual-rail (toolchain-gated, like test_c_encoder) ------------


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _build(tmp):
    cc = _cc()
    if cc is None:
        return None
    exe = os.path.join(tmp, "test_telemetry_frame")
    r = subprocess.run(
        [
            cc,
            "-std=c23",
            "-O2",
            "-Wall",
            "-Wextra",
            "-I",
            _RUNTIME_C,
            os.path.join(_RUNTIME_C, "bcir_telemetry_frame.c"),
            os.path.join(_RUNTIME_C, "bcir_runtime.c"),
            os.path.join(_RUNTIME_C, "test_telemetry_frame.c"),
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    return exe


def test_c_twin_builds_freestanding():
    """The C twin compiles -ffreestanding -nostdlib under both C11 and C23 (the embedded
    contract: no libc beyond what the codec uses; the frozen-record static_assert holds)."""
    cc = _cc()
    if cc is None:
        return
    for std in ("c11", "c23"):
        r = subprocess.run(
            [
                cc,
                "-ffreestanding",
                "-nostdlib",
                f"-std={std}",
                "-Wall",
                "-Wextra",
                "-I",
                _RUNTIME_C,
                "-c",
                os.path.join(_RUNTIME_C, "bcir_telemetry_frame.c"),
                "-o",
                os.devnull,
            ],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{std}: {r.stderr}"


def test_c_decode_round_trips_and_reencode_is_byte_identical():
    """The headline dual-rail: Python encode_frame -> the C twin decodes it (round-trips
    the header + records) AND re-encodes byte-identically to encode_frame. Proves the C
    bcir_crc32 == Python zlib.crc32 and the two wire writers agree byte-for-byte."""
    if _cc() is None:
        return
    # cover both the RT3-valid batch AND the signed-int64 wire batch (negative claim_id /
    # counter, INT64_MAX) so the C two's-complement byte writes are pinned identical.
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        assert exe is not None
        for recs in (_batch(), _wire_batch()):
            for seq, ts in ((0, 0), (7, 123456), (0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF)):
                blob = encode_frame(recs, seq=seq, timestamp=ts)
                ip = os.path.join(tmp, "frame.bin")
                op = os.path.join(tmp, "reenc.bin")
                with open(ip, "wb") as f:
                    f.write(blob)
                if os.path.exists(op):
                    os.unlink(op)
                r = subprocess.run([exe, ip, op], capture_output=True, text=True)
                assert r.returncode == 0, r.stdout + r.stderr
                # the C harness printed the decoded records -- confirm they match the input
                lines = r.stdout.strip().splitlines()
                assert lines[0] == f"seq={seq} ts={ts} n={len(recs)}", (lines[0], seq, ts)
                for i, rec in enumerate(recs):
                    want = (
                        f"rec {i}: {rec.claim_id} {rec.cycles} {rec.bytes} {rec.misses} "
                        f"{rec.thermal} {rec.voltage} {rec.utilization}"
                    )
                    assert lines[1 + i] == want, (lines[1 + i], want)
                with open(op, "rb") as f:
                    reenc = f.read()
                assert reenc == blob, f"seq={seq} ts={ts}: C re-encode not byte-identical"


def test_c_rejects_corrupt_frame():
    """A CRC-corrupt / truncated / bad-magic frame is rejected by the C decoder (a status,
    never a crash) -- the C trust boundary mirrors the Python rail's reject."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build(tmp)
        good = encode_frame(_batch(), seq=1)
        bad_crc = bytearray(good)
        bad_crc[HEADER_SIZE] ^= 0xFF
        bad_magic = b"XXXX" + good[4:]
        flagged = bytearray(good)
        flagged[6:8] = (1).to_bytes(2, "little")
        flagged[-CRC_SIZE:] = (zlib.crc32(flagged[:-CRC_SIZE]) & 0xFFFFFFFF).to_bytes(4, "little")
        for bad in (
            b"",
            b"BTLM",
            good[:HEADER_SIZE],
            good[:-1],
            bytes(bad_crc),
            bad_magic,
            bytes(flagged),
            good + b"trailing",
        ):
            ip = os.path.join(tmp, "bad.bin")
            with open(ip, "wb") as f:
                f.write(bad)
            r = subprocess.run([exe, ip], capture_output=True, text=True)
            assert r.returncode == 1, (bad[:8], r.returncode, r.stdout, r.stderr)
            assert "ERR" in r.stdout, r.stdout


def _run_all():
    import inspect

    g = dict(globals())
    fns = [(n, f) for n, f in g.items() if n.startswith("test_") and inspect.isfunction(f)]
    fns.sort(key=lambda nf: nf[0])
    for n, f in fns:
        f()
        print(f"PASS {n}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
