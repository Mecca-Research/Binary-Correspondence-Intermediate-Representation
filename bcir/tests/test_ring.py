"""Zero-copy telemetry ring: the producer packs stats into a preallocated shared
buffer; the consumer reads them straight back -- no serialization, non-blocking,
wrap-around. Deterministic."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import bcir.telemetry as telemetry
from bcir.telemetry import Broker, DataDNA, TelemetryRing


def _dna(cid, cycles=0, misses=0):
    return DataDNA(segment_id="s", claim_id=cid, cycles=cycles, misses=misses)


def test_write_then_read_roundtrips_the_atomic_stats():
    ring = TelemetryRing(capacity=8)
    ring.write(DataDNA(segment_id="s", claim_id=7, cycles=123, bytes=456, misses=5,
                       thermal=60, voltage=40, utilization=90))
    out = ring.read_one()
    assert out.claim_id == 7 and out.cycles == 123 and out.bytes == 456
    assert out.misses == 5 and out.thermal == 60 and out.voltage == 40 and out.utilization == 90


def test_empty_read_is_nonblocking_and_returns_none():
    assert TelemetryRing(capacity=4).read_one() is None


def test_buffer_is_preallocated_and_shared_zero_copy():
    ring = TelemetryRing(capacity=16)
    before = len(ring.buf)
    buf_id = id(ring.buf)
    for i in range(10):
        ring.write(_dna(i, cycles=i * 10))
    # no growth, same object: writer and reader share one fixed region (zero-copy).
    assert len(ring.buf) == before and id(ring.buf) == buf_id
    assert [d.cycles for d in ring.drain()] == [i * 10 for i in range(10)]


def test_wraparound_drops_oldest_when_full():
    ring = TelemetryRing(capacity=4)
    for i in range(6):                                   # 2 more than capacity
        ring.write(_dna(i))
    drained = ring.drain()
    assert [d.claim_id for d in drained] == [2, 3, 4, 5]  # oldest two overwritten
    assert ring.stats.dropped == 2 and ring.stats.written == 6


def test_ring_plugs_into_the_broker_as_a_sink():
    ring = TelemetryRing(capacity=8)
    b = Broker()
    b.subscribe(ring)
    b.emit(_dna(1, cycles=11))
    b.emit(_dna(2, cycles=22))
    assert ring.pending == 2
    assert [d.cycles for d in ring.drain()] == [11, 22]


def test_interleaved_read_write_tracks_head_and_tail():
    ring = TelemetryRing(capacity=4)
    ring.write(_dna(1)); ring.write(_dna(2))
    assert ring.read_one().claim_id == 1 and ring.pending == 1
    ring.write(_dna(3))
    assert [d.claim_id for d in ring.drain()] == [2, 3] and ring.pending == 0


def test_ring_is_deterministic():
    def run():
        r = TelemetryRing(capacity=4)
        for i in range(7):
            r.write(_dna(i, cycles=i))
        return [(d.claim_id, d.cycles) for d in r.drain()], r.stats.dropped
    assert run() == run()


def test_overwrite_and_read_are_one_serialized_transition():
    """A reader must not observe a replacement slot before its head publication.

    The interposed pack holds the writer precisely between the slot copy and head
    update.  Without the ring lock the reader consumes the replacement as the old
    record, after which the writer publishes it a second time (a temporal duplicate).
    """
    ring = TelemetryRing(capacity=1)
    ring.write(_dna(1))
    packed = threading.Event()
    release = threading.Event()
    reader_started = threading.Event()
    original = telemetry.struct.pack_into

    def blocked_pack(fmt, buf, offset, *values):
        original(fmt, buf, offset, *values)
        if values[0] == 2:
            packed.set()
            assert release.wait(timeout=5)

    def read_one():
        reader_started.set()
        return ring.read_one()

    telemetry.struct.pack_into = blocked_pack
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer = pool.submit(ring.write, _dna(2))
            assert packed.wait(timeout=5)
            reader = pool.submit(read_one)
            assert reader_started.wait(timeout=5)
            time.sleep(0.05)  # bounded proof that the reader is waiting on the transition
            assert not reader.done()
            release.set()
            writer.result(timeout=5)
            event = reader.result(timeout=5)
    finally:
        release.set()
        telemetry.struct.pack_into = original
    assert event.claim_id == 2
    assert ring.pending == 0
    assert ring.stats.written == 2 and ring.stats.read == 1 and ring.stats.dropped == 1
