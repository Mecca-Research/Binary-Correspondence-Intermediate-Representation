"""G15 fixtures (S3-B): the live SPSC ring's scripted scenarios, the telemetry envelope corpus and
its malformed variants, the generated signal table, the two-rail drivers and the G15 rows.

A scenario is a script of operations on one shared region -- attach, write (or open/fill/close),
read (or open/copy/close), drop an endpoint (a peer's death), poke raw bytes (a hostile or crashed
peer), hand a record to the telemetry intake, move the live generation, check the accounting --
run on BOTH rails: the Python oracle (`run_python`) and the C twin (`runtime/c/test_ring.c
--script`). Every operation prints one trace line, the rails must print identical lines (the
region's CRC after every step included), and each scenario declares what the specification says
must happen: the refusals it pins, the final accounting, the continuity report.

The rows are counts of failing (fixture, rail) pairs, so a rail that cannot run a fixture fails it
(L1: every path is a count). On the parent tree there was no live ring, no envelope, no generated
table and no C twin: every fixture fails by absence.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import zlib
from dataclasses import dataclass, field

from ..abi.control_abi import CONTROL_RECORD_MAX_BYTES
from ..abi.telemetry_envelope import (
    ENVELOPE_HEADER_SIZE,
    ENVELOPE_SIZES,
    TelemetryEnvelope,
    TelemetryError,
    decode_envelope,
    encode_envelope,
)
from ..gem.ring import (
    ATTACHED,
    ATTACHING,
    CONTROL_SLOT_MIN,
    DETACHED,
    HEADER_SIZE,
    OFF_ACCT,
    OFF_C_BEAT,
    OFF_C_OWNER,
    OFF_HEAD,
    OFF_P_BEAT,
    OFF_P_OWNER,
    OFF_REFUSED,
    SLOT_HEADER,
    RingConsumer,
    RingError,
    RingGeometry,
    RingOutcome,
    RingProducer,
    committed_accounting,
    decode_geometry,
    encode_geometry,
    format_ring,
)
from ..signal_table import builtin_definitions, emit_c_header, encode_row
from ..telemetry_intake import TelemetryIntake

M32 = 0xFFFFFFFF
M64 = 0xFFFFFFFFFFFFFFFF

# --- the script -----------------------------------------------------------------------------------

(
    P_ATTACH,
    P_DETACH,
    P_WRITE,
    P_OPEN,
    P_FILL,
    P_CLOSE,
    P_BEAT,
    P_DROP,
    C_ATTACH,
    C_DETACH,
    C_READ,
    C_OPEN,
    C_COPY,
    C_CLOSE,
    C_BEAT,
    C_DROP,
    POKE,
    INTAKE,
    LIVE,
    CHECK,
) = range(1, 21)
RING_OPS = frozenset(range(P_ATTACH, C_DROP + 1))
ACTORS = 4
FAMILIES = ("saturation", "wrap", "torn", "restart", "stale", "continuity", "malformed", "control")


@dataclass(frozen=True)
class Op:
    code: int
    actor: int = 0
    arg: int = 0
    data: bytes = b""


@dataclass
class Scenario:
    name: str
    family: str
    geometry: RingGeometry
    live: int
    ops: list[Op]
    expect_ops: dict[int, tuple] = field(default_factory=dict)
    expect_final: dict[str, int] = field(default_factory=dict)
    continuity: tuple[int, int, int] | None = None
    poked_publications: int = 0
    accounting: bool = True


class _Build:
    """A scenario under construction; every helper returns the op's index."""

    _ids = 0

    def __init__(self, name, family, *, policy="backpressure", payload="telemetry", slot_size=192,
                 slot_count=4, origin=0, live=1):  # fmt: skip
        assert family in FAMILIES, family
        _Build._ids += 1
        self.s = Scenario(
            name,
            family,
            RingGeometry(
                policy, payload, slot_size, slot_count, ring_id=0x5200 + _Build._ids, origin=origin
            ),
            live,
            [],
        )

    def add(self, code, actor=0, arg=0, data=b"", expect=None) -> int:
        self.s.ops.append(Op(code, actor, arg, bytes(data)))
        index = len(self.s.ops) - 1
        if expect is not None:
            self.s.expect_ops[index] = expect
        return index

    def attach(self, *, producers=(0,), consumers=(0,)):
        for a in producers:
            self.add(P_ATTACH, a, 0, expect=("R", "ok", "BCIR_OK"))
        for a in consumers:
            self.add(C_ATTACH, a, 0, expect=("R", "ok", "BCIR_OK"))

    def write(self, data, actor=0, expect=None):
        return self.add(P_WRITE, actor, 0, data, expect)

    def read(self, actor=0, expect=None):
        return self.add(C_READ, actor, 0, b"", expect)

    def intake(self, actor=0, data=b"", expect=None):
        return self.add(INTAKE, actor, 0, data, expect)

    def poke(self, offset, data):
        return self.add(POKE, 0, offset, data)

    def done(self, **final) -> Scenario:
        self.s.expect_final.update(final)
        self.add(CHECK)
        return self.s


def blob(tag: int, n: int = 24) -> bytes:
    """A payload that names the write it came from."""
    return bytes((tag * 131 + i * 17 + 7) & 0xFF for i in range(n))


def dna(seq, *, session=1, source=1, generation=1, lost=0) -> bytes:
    """A DataDNA envelope whose every field is a function of (source, session, seq)."""
    seq &= M32
    return encode_envelope(
        TelemetryEnvelope(
            "datadna",
            source=source,
            session=session,
            seq=seq,
            generation=generation,
            lost=lost,
            clock="monotonic",
            unit="ns",
            timestamp=1000 + seq,
            record=(seq, seq * 7, seq * 13, seq % 101, seq % 97, seq % 89, seq % 83),
        )
    )


def sample(signal, value, *, seq=0, session=1, source=1, required=False, generation=0) -> bytes:
    return encode_envelope(
        TelemetryEnvelope(
            "sample",
            source=source,
            session=session,
            seq=seq & M32,
            generation=generation,
            signal=signal,
            required=required,
            value=value,
            clock="boot",
            unit="us",
            timestamp=7 + seq,
        )
    )


def _slot_offset(g: RingGeometry, position: int) -> int:
    return HEADER_SIZE + (position & (g.slot_count - 1)) * g.slot_size


def forged_slot(seq: int, position: int, epoch: int, payload: bytes, length=None) -> bytes:
    """A slot's bytes as a peer might leave them (for POKE)."""
    n = len(payload) if length is None else length
    body = payload + bytes(-len(payload) % 8)
    return struct.pack("<QQQ", seq, position, epoch | (n << 32)) + body


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value & M64)


# --- the corpus -----------------------------------------------------------------------------------

OK = ("R", "ok", "BCIR_OK")
EMPTY = ("R", "empty", "BCIR_OK")
FULL = ("R", "refused", "BCIR_ERR_FULL")


def _delivered(count=1):
    return ("R", "delivered", "BCIR_OK", count)


def _lost(count):
    return ("R", "lost", "BCIR_OK", count)


def saturation_scenarios() -> list[Scenario]:
    out = []
    b = _Build("bp-fill-refuse-drain", "saturation")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.write(blob(4), expect=FULL)
    b.write(blob(5), expect=FULL)
    for _ in range(4):
        b.read(expect=_delivered())
    b.read(expect=EMPTY)
    b.write(blob(6), expect=OK)
    b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=5, lost=0, stale=0, refused=2, published=5))

    b = _Build("bp-lockstep-slot-wrap", "saturation", slot_count=2)
    b.attach()
    for i in range(12):
        b.write(blob(i, 40), expect=OK)
        b.read(expect=_delivered())
    out.append(b.done(delivered=12, lost=0, published=12))

    b = _Build("bp-two-phase-in-place", "saturation")
    b.attach()
    b.add(P_OPEN, 0, 40, expect=("R", "open", "BCIR_OK"))
    b.read(expect=EMPTY)  # not published until close
    b.add(P_FILL, 0, 0, blob(9, 24), expect=("R", "open", "BCIR_OK"))
    b.add(P_FILL, 0, 24, blob(10, 16), expect=("R", "open", "BCIR_OK"))
    b.read(expect=EMPTY)
    b.add(P_CLOSE, 0, expect=OK)
    b.read(expect=_delivered())
    out.append(b.done(delivered=1, published=1))

    b = _Build("bp-partial-drain-then-full", "saturation")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.read(expect=_delivered())
    b.write(blob(4), expect=OK)
    b.write(blob(5), expect=FULL)
    for _ in range(4):
        b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=5, refused=1, published=5))

    b = _Build("bp-writer-never-touches-an-unconsumed-slot", "saturation")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.add(C_OPEN, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_COPY, 0, expect=("R", "open", "BCIR_OK"))
    b.write(blob(4), expect=FULL)
    b.add(C_CLOSE, 0, expect=_delivered())
    b.write(blob(5), expect=OK)
    for _ in range(4):
        b.read(expect=_delivered())
    out.append(b.done(delivered=5, refused=1, published=5))

    b = _Build("bp-capacity-and-empty-record", "saturation", slot_size=64)
    b.attach()
    b.write(blob(1, 41), expect=("R", "refused", "BCIR_ERR_NOSPACE"))
    b.write(blob(2, 40), expect=OK)
    b.write(b"", expect=OK)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    out.append(b.done(delivered=2, published=2))

    b = _Build("ow-lap-counts-seven", "saturation", policy="overwrite")
    b.attach()
    for i in range(11):
        b.write(blob(i), expect=OK)
    b.read(expect=_lost(7))
    for _ in range(4):
        b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=4, lost=7, published=11))

    b = _Build("ow-lap-exact-multiple", "saturation", policy="overwrite")
    b.attach()
    for i in range(8):
        b.write(blob(i), expect=OK)
    b.read(expect=_lost(4))
    for _ in range(4):
        b.read(expect=_delivered())
    out.append(b.done(delivered=4, lost=4, published=8))

    b = _Build("ow-no-lap-no-loss", "saturation", policy="overwrite")
    b.attach()
    for i in range(3):
        b.write(blob(i), expect=OK)
    for _ in range(3):
        b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=3, lost=0, published=3))

    b = _Build("ow-repeated-laps", "saturation", policy="overwrite", slot_count=2)
    b.attach()
    for i in range(3):
        b.write(blob(i), expect=OK)
    b.read(expect=_lost(1))
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.read(expect=EMPTY)
    for i in range(3, 8):
        b.write(blob(i), expect=OK)
    b.read(expect=_lost(3))
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=4, lost=4, published=8))

    b = _Build(
        "control-ring-backpressure",
        "control",
        payload="control",
        slot_size=CONTROL_SLOT_MIN,
        slot_count=2,
    )
    b.attach()
    b.write(blob(1, CONTROL_RECORD_MAX_BYTES), expect=OK)  # the declared bound fits
    b.write(blob(2, 108), expect=OK)
    b.write(blob(3, 140), expect=FULL)
    b.read(expect=_delivered())
    b.write(blob(3, 140), expect=OK)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.write(
        blob(4, CONTROL_SLOT_MIN - SLOT_HEADER + 1), expect=("R", "refused", "BCIR_ERR_NOSPACE")
    )
    out.append(b.done(delivered=3, refused=1, published=3))
    return out


def wrap_scenarios() -> list[Scenario]:
    out = []
    b = _Build("bp-positions-wrap-u64", "wrap", origin=M64 - 2)
    b.attach()
    for i in range(8):
        b.write(blob(i), expect=OK)
        b.read(expect=_delivered())
    out.append(b.done(delivered=8, published=8))

    b = _Build("ow-lap-across-u64-wrap", "wrap", policy="overwrite", slot_count=2, origin=M64 - 1)
    b.attach()
    for i in range(5):
        b.write(blob(i), expect=OK)
    b.read(expect=_lost(3))
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.read(expect=EMPTY)
    out.append(b.done(delivered=2, lost=3, published=5))

    b = _Build("envelope-seq-wraps-u32", "wrap", slot_count=8)
    b.attach()
    for seq in (0xFFFFFFFE, 0xFFFFFFFF, 0, 1):
        b.write(dna(seq), expect=OK)
    for cls in ("first", "next", "next", "next"):
        b.read(expect=_delivered())
        b.intake(expect=("I", "accepted", "none", "BCIR_OK", cls))
    s = b.done(delivered=4, published=4)
    s.continuity = (0, 0, 0)
    out.append(s)
    return out


def torn_scenarios() -> list[Scenario]:
    out = []
    b = _Build("ow-overwritten-during-copy", "torn", policy="overwrite")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.add(C_OPEN, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_COPY, 0, expect=("R", "open", "BCIR_OK"))
    b.write(blob(4), expect=OK)
    b.add(C_CLOSE, 0, expect=_lost(1))
    for _ in range(4):
        b.read(expect=_delivered())
    out.append(b.done(delivered=4, lost=1, published=5))

    b = _Build("ow-slot-being-written", "torn", policy="overwrite")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.add(P_OPEN, 0, 24, expect=("R", "open", "BCIR_OK"))
    b.read(expect=_lost(1))
    b.add(P_FILL, 0, 0, blob(4), expect=("R", "open", "BCIR_OK"))
    b.add(P_CLOSE, 0, expect=OK)
    for _ in range(4):
        b.read(expect=_delivered())
    out.append(b.done(delivered=4, lost=1, published=5))

    b = _Build("ow-copy-straddles-two-laps", "torn", policy="overwrite", slot_count=2)
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(C_OPEN, 0, expect=("R", "open", "BCIR_OK"))
    for i in range(1, 4):
        b.write(blob(i), expect=OK)
    b.add(C_COPY, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_CLOSE, 0, expect=_lost(1))
    b.read(expect=_lost(1))
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    out.append(b.done(delivered=2, lost=2, published=4))
    return out


def restart_scenarios() -> list[Scenario]:
    out = []
    b = _Build("bp-producer-dies-mid-write", "restart")
    b.attach()
    b.write(blob(0), expect=OK)
    b.write(blob(1), expect=OK)
    b.add(P_OPEN, 0, 32, expect=("R", "open", "BCIR_OK"))
    b.add(P_FILL, 0, 0, blob(99, 16), expect=("R", "open", "BCIR_OK"))
    b.add(P_DROP, 0)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.read(expect=EMPTY)
    b.add(P_ATTACH, 1, 1, expect=OK)
    b.write(blob(2), actor=1, expect=OK)
    b.read(expect=("R", "delivered", "BCIR_OK", 1))
    b.read(expect=EMPTY)
    out.append(b.done(delivered=3, published=3))

    b = _Build("ow-producer-dies-mid-overwrite", "restart", policy="overwrite")
    b.attach()
    for i in range(4):
        b.write(blob(i), expect=OK)
    b.add(P_OPEN, 0, 24, expect=("R", "open", "BCIR_OK"))
    b.add(P_FILL, 0, 0, blob(98, 8), expect=("R", "open", "BCIR_OK"))
    b.add(P_DROP, 0)
    b.read(expect=_lost(1))
    b.add(P_ATTACH, 1, 1, expect=OK)
    for _ in range(3):
        b.read(expect=_delivered())
    b.read(expect=EMPTY)
    b.write(blob(4), actor=1, expect=OK)
    b.read(expect=_delivered())
    out.append(b.done(delivered=4, lost=1, published=5))

    b = _Build("producer-detach-reattach", "restart")
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(P_DETACH, 0, expect=OK)
    b.add(P_ATTACH, 1, 0, expect=OK)
    b.write(blob(1), actor=1, expect=OK)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    out.append(b.done(delivered=2, stale=0, published=2))

    b = _Build("consumer-dies-mid-read", "restart")
    b.attach()
    for i in range(3):
        b.write(blob(i), expect=OK)
    b.read(expect=_delivered())
    b.add(C_OPEN, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_COPY, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_DROP, 0)
    b.add(C_ATTACH, 1, 1, expect=OK)
    b.read(actor=1, expect=_delivered())
    b.read(actor=1, expect=_delivered())
    b.read(actor=1, expect=EMPTY)
    out.append(b.done(delivered=3, published=3))

    b = _Build("consumer-detach-reattach", "restart")
    b.attach()
    b.write(blob(0), expect=OK)
    b.write(blob(1), expect=OK)
    b.read(expect=_delivered())
    b.add(C_DETACH, 0, expect=OK)
    b.add(C_ATTACH, 1, 0, expect=OK)
    b.read(actor=1, expect=_delivered())
    out.append(b.done(delivered=2, published=2))

    b = _Build("attach-laws", "restart")
    b.add(P_ATTACH, 0, 0, expect=OK)
    b.add(P_ATTACH, 1, 0, expect=("R", "refused", "BCIR_ERR_BUSY"))
    b.add(P_ATTACH, 1, 7, expect=("R", "refused", "BCIR_ERR_STALE"))
    b.add(C_ATTACH, 0, 0, expect=OK)
    b.add(C_ATTACH, 1, 0, expect=("R", "refused", "BCIR_ERR_BUSY"))
    b.add(P_DETACH, 0, expect=OK)
    b.add(P_ATTACH, 1, 1, expect=("R", "refused", "BCIR_ERR_STALE"))
    b.add(P_ATTACH, 1, 0, expect=OK)
    b.add(P_DETACH, 1, expect=OK)
    b.add(P_DETACH, 1, expect=("R", "refused", "BCIR_ERR_STALE"))
    out.append(b.done(published=0))

    b = _Build("heartbeats-move", "restart")
    b.attach()
    for _ in range(3):
        b.add(P_BEAT, 0, expect=OK)
    b.write(blob(0), expect=OK)
    for _ in range(2):
        b.add(C_BEAT, 0, expect=OK)
    b.read(expect=_delivered())
    out.append(b.done(delivered=1, published=1, p_heartbeat=4, c_heartbeat=3))
    return out


def stale_scenarios() -> list[Scenario]:
    out = []
    stale = ("R", "refused", "BCIR_ERR_STALE")
    b = _Build("deposed-producer-refuses-itself", "stale")
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(P_ATTACH, 1, 1, expect=OK)
    b.write(blob(1), actor=0, expect=stale)
    b.add(P_OPEN, 0, 8, expect=stale)
    b.add(P_BEAT, 0, expect=stale)
    b.add(P_DETACH, 0, expect=stale)
    b.write(blob(2), actor=1, expect=OK)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    out.append(b.done(delivered=2, published=2))

    b = _Build("deposed-consumer-refuses-itself", "stale")
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(C_ATTACH, 1, 1, expect=OK)
    b.read(actor=0, expect=stale)
    b.add(C_BEAT, 0, expect=stale)
    b.read(actor=1, expect=_delivered())
    out.append(b.done(delivered=1, published=1))

    # A record stamped with a deposed epoch at or after the takeover position.
    b = _Build("stale-epoch-after-origin", "stale")
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(P_DROP, 0)
    b.add(P_ATTACH, 1, 1, expect=OK)
    g = b.s.geometry
    b.poke(_slot_offset(g, 1), forged_slot(2, 1, 1, blob(50, 16)))
    b.poke(OFF_HEAD, _u64(2))
    b.read(expect=_delivered())
    b.read(expect=("R", "stale", "BCIR_OK", 1))
    b.write(blob(2), actor=1, expect=OK)
    b.read(expect=_delivered())
    s = b.done(delivered=2, stale=1, published=3)
    s.poked_publications = 1
    out.append(s)

    # A record before the takeover position whose epoch goes backwards.
    b = _Build("stale-epoch-goes-backwards", "stale", slot_count=8)
    b.attach()
    b.write(blob(0), expect=OK)
    b.add(P_DETACH, 0, expect=OK)
    b.add(P_ATTACH, 1, 0, expect=OK)
    b.write(blob(1), actor=1, expect=OK)
    b.write(blob(2), actor=1, expect=OK)
    b.poke(_slot_offset(b.s.geometry, 2), forged_slot(2, 2, 1, blob(51, 16)))
    b.add(P_DETACH, 1, expect=OK)
    b.add(P_ATTACH, 2, 0, expect=OK)
    b.read(expect=_delivered())
    b.read(expect=_delivered())
    b.read(expect=("R", "stale", "BCIR_OK", 1))
    b.write(blob(3), actor=2, expect=OK)
    b.read(expect=_delivered())
    out.append(b.done(delivered=3, stale=1, published=4))

    b = _Build("telemetry-stale-generation", "stale", slot_count=8, live=5)
    b.attach()
    cases = [
        (dna(0, generation=5), ("I", "accepted", "none", "BCIR_OK", "first")),
        (dna(1, generation=4), ("I", "refused", "stale", "BCIR_ERR_STALE", "next")),
        (dna(2, generation=6), ("I", "refused", "ahead", "BCIR_ERR_STALE", "next")),
        (sample(1, 42, seq=3, generation=0), ("I", "accepted", "none", "BCIR_OK", "next")),
    ]
    for data, _ in cases:
        b.write(data, expect=OK)
    for _, want in cases:
        b.read(expect=_delivered())
        b.intake(expect=want)
    b.add(LIVE, 0, 6)
    later = [
        (dna(4, generation=5), ("I", "refused", "stale", "BCIR_ERR_STALE", "next")),
        (dna(5, generation=6), ("I", "accepted", "none", "BCIR_OK", "next")),
    ]
    for data, want in later:
        b.write(data, expect=OK)
        b.read(expect=_delivered())
        b.intake(expect=want)
    s = b.done(delivered=6, published=6)
    s.continuity = (0, 0, 0)
    out.append(s)
    return out


def continuity_scenarios() -> list[Scenario]:
    out = []
    accepted = ("I", "accepted", "none", "BCIR_OK")

    b = _Build("telemetry-clean-stream", "continuity", slot_count=8)
    b.attach()
    for seq in range(6):
        b.write(dna(seq), expect=OK)
    for seq in range(6):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, "first" if seq == 0 else "next"))
    s = b.done(delivered=6, published=6)
    s.continuity = (0, 0, 0)
    out.append(s)

    b = _Build("telemetry-overwrite-gap-is-the-loss", "continuity", policy="overwrite")
    b.attach()
    b.write(dna(0), expect=OK)
    b.read(expect=_delivered())
    b.intake(expect=(*accepted, "first"))
    for seq in range(1, 7):
        b.write(dna(seq), expect=OK)
    b.read(expect=_lost(2))
    b.read(expect=_delivered())
    b.intake(expect=(*accepted, "gap"))
    for _ in range(3):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, "next"))
    s = b.done(delivered=5, lost=2, published=7)
    s.continuity = (2, 0, 0)
    out.append(s)

    # The producer's refusals travel in the next record's `lost` field.
    b = _Build("telemetry-refusals-reported-by-the-producer", "continuity")
    b.attach()
    for seq in range(4):
        b.write(dna(seq), expect=OK)
    b.write(dna(4), expect=FULL)
    b.write(dna(5), expect=FULL)
    for seq in range(4):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, "first" if seq == 0 else "next"))
    b.write(dna(6, lost=2), expect=OK)
    b.read(expect=_delivered())
    b.intake(expect=(*accepted, "gap"))
    s = b.done(delivered=5, refused=2, published=5, reported_lost=2)
    s.continuity = (2, 0, 0)
    out.append(s)

    # A consumer that handed a record to the sink and died before committing: re-delivered.
    b = _Build("telemetry-redelivery-is-a-duplicate", "continuity", slot_count=8)
    b.attach()
    for seq in range(3):
        b.write(dna(seq), expect=OK)
    b.read(expect=_delivered())
    b.intake(expect=(*accepted, "first"))
    b.add(C_OPEN, 0, expect=("R", "open", "BCIR_OK"))
    b.add(C_COPY, 0, expect=("R", "open", "BCIR_OK"))
    b.intake(data=dna(1), expect=(*accepted, "next"))
    b.add(C_DROP, 0)
    b.add(C_ATTACH, 1, 1, expect=OK)
    b.read(actor=1, expect=_delivered())
    b.intake(actor=1, expect=(*accepted, "duplicate"))
    b.read(actor=1, expect=_delivered())
    b.intake(actor=1, expect=(*accepted, "next"))
    s = b.done(delivered=3, published=3)
    s.continuity = (0, 0, 1)
    out.append(s)

    b = _Build("telemetry-restart-reusing-a-session-reorders", "continuity", slot_count=8)
    b.attach()
    for seq in range(4):
        b.write(dna(seq), expect=OK)
    b.add(P_DETACH, 0, expect=OK)
    b.add(P_ATTACH, 1, 0, expect=OK)
    for seq in range(2):
        b.write(dna(seq), actor=1, expect=OK)
    for cls in ("first", "next", "next", "next", "reorder", "reorder"):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, cls))
    s = b.done(delivered=6, published=6)
    s.continuity = (0, 2, 0)
    out.append(s)

    b = _Build("telemetry-restart-with-a-new-session", "continuity", slot_count=8)
    b.attach()
    for seq in range(4):
        b.write(dna(seq), expect=OK)
    b.add(P_DETACH, 0, expect=OK)
    b.add(P_ATTACH, 1, 0, expect=OK)
    for seq in range(2):
        b.write(dna(seq, session=2), actor=1, expect=OK)
    for cls in ("first", "next", "next", "next", "first", "next"):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, cls))
    s = b.done(delivered=6, published=6)
    s.continuity = (0, 0, 0)
    out.append(s)

    b = _Build("telemetry-two-streams", "continuity", slot_count=8)
    b.attach()
    order = [(1, 0), (2, 0), (1, 1), (2, 1), (1, 3), (2, 2)]
    for source, seq in order:
        b.write(dna(seq, source=source), expect=OK)
    for cls in ("first", "first", "next", "next", "gap", "next"):
        b.read(expect=_delivered())
        b.intake(expect=(*accepted, cls))
    s = b.done(delivered=6, published=6)
    s.continuity = (1, 0, 0)
    out.append(s)

    b = _Build("telemetry-stream-table-is-bounded", "continuity", slot_count=32)
    b.attach()
    for session in range(1, 18):
        b.write(dna(0, session=session), expect=OK)
    for session in range(1, 18):
        b.read(expect=_delivered())
        if session <= 16:
            b.intake(expect=(*accepted, "first"))
        else:
            b.intake(expect=("I", "refused", "streams", "BCIR_ERR_NOSPACE", "none"))
    s = b.done(delivered=17, published=17)
    s.continuity = (0, 0, 0)
    out.append(s)
    return out


def _geometry_line(g: RingGeometry, **override) -> bytes:
    """A geometry line with fields overridden and the CRC recomputed (so the law under test, not
    the CRC, is what refuses it)."""
    fields = dict(
        magic=b"BRNG",
        version=0,
        policy=("backpressure", "overwrite").index(g.policy) + 1,
        payload=("telemetry", "control").index(g.payload) + 1,
        slot_size=g.slot_size,
        slot_count=g.slot_count,
        ring_id=g.ring_id,
        region_size=g.region_size,
        origin=g.origin,
        reserved=bytes(20),
    )
    fields.update(override)
    line = struct.pack(
        "<4sHBBIIQQQ",
        fields["magic"],
        fields["version"],
        fields["policy"],
        fields["payload"],
        fields["slot_size"],
        fields["slot_count"],
        fields["ring_id"],
        fields["region_size"],
        fields["origin"],
    )
    line += fields["reserved"]
    return line + struct.pack("<I", zlib.crc32(line) & M32)


def malformed_scenarios() -> list[Scenario]:
    out = []
    probe = RingGeometry("backpressure", "telemetry", 192, 4, ring_id=1)

    def geometry_case(name, line, status):
        b = _Build(name, "malformed")
        b.poke(0, line)
        want = ("R", "refused", status)
        b.add(P_ATTACH, 0, 0, expect=want)
        b.add(C_ATTACH, 0, 0, expect=want)
        s = b.done()
        s.accounting = False
        return s

    good = encode_geometry(RingGeometry("backpressure", "telemetry", 192, 4, ring_id=1))
    out.append(geometry_case("geometry-magic", b"BRNX" + good[4:], "BCIR_ERR_MAGIC"))
    out.append(
        geometry_case("geometry-version", _geometry_line(probe, version=1), "BCIR_ERR_VERSION")
    )
    out.append(geometry_case("geometry-crc", good[:16] + b"\x02" + good[17:], "BCIR_ERR_CRC"))
    out.append(
        geometry_case(
            "geometry-reserved",
            _geometry_line(probe, reserved=bytes(19) + b"\x01"),
            "BCIR_ERR_RESERVED",
        )
    )
    out.append(geometry_case("geometry-policy", _geometry_line(probe, policy=3), "BCIR_ERR_RING"))
    out.append(geometry_case("geometry-payload", _geometry_line(probe, payload=0), "BCIR_ERR_RING"))
    out.append(
        geometry_case("geometry-slot-size", _geometry_line(probe, slot_size=100), "BCIR_ERR_RING")
    )
    out.append(
        geometry_case("geometry-slot-count", _geometry_line(probe, slot_count=3), "BCIR_ERR_RING")
    )
    out.append(geometry_case("geometry-ring-id", _geometry_line(probe, ring_id=0), "BCIR_ERR_RING"))
    out.append(
        geometry_case(
            "geometry-region-size", _geometry_line(probe, region_size=12345), "BCIR_ERR_RING"
        )
    )
    out.append(
        geometry_case(
            "geometry-control-overwrite",
            _geometry_line(probe, policy=2, payload=2),
            "BCIR_ERR_RING",
        )
    )
    out.append(  # a 192-byte slot holds 168 payload bytes: short of the control record bound
        geometry_case("geometry-control-slot", _geometry_line(probe, payload=2), "BCIR_ERR_RING")
    )
    bigger = RingGeometry("backpressure", "telemetry", 192, 8, ring_id=1)
    out.append(
        geometry_case(
            "geometry-region-beyond-buffer",
            _geometry_line(bigger),
            "BCIR_ERR_TRUNCATED",
        )
    )

    def owner_case(name, offset, word, which):
        b = _Build(name, "malformed")
        b.poke(offset, _u64(word))
        b.add(which, 0, 0, expect=("R", "refused", "BCIR_ERR_RING"))
        return b.done()

    out.append(owner_case("owner-epoch-exhausted", OFF_P_OWNER, 0xFFFFFFFF << 32, P_ATTACH))
    out.append(owner_case("owner-state-unknown", OFF_P_OWNER, 3, P_ATTACH))
    out.append(owner_case("owner-producer-attaching", OFF_P_OWNER, ATTACHING, P_ATTACH))
    out.append(owner_case("owner-consumer-attaching", OFF_C_OWNER, ATTACHING, C_ATTACH))

    def corrupt_case(name, pokes, *, policy="backpressure", writes=1):
        b = _Build(name, "malformed", policy=policy)
        b.attach()
        for i in range(writes):
            b.write(blob(i), expect=OK)
        for offset, data in pokes(b.s.geometry):
            b.poke(offset, data)
        b.read(expect=("R", "refused", "BCIR_ERR_RING"))
        b.read(expect=("R", "refused", "BCIR_ERR_RING"))  # the ring stays refused: no advance
        s = b.done()
        s.accounting = False
        return s

    slot0 = HEADER_SIZE
    out.append(corrupt_case("bp-head-overruns-the-ring", lambda g: [(OFF_HEAD, _u64(6))]))
    out.append(
        corrupt_case("head-behind-the-tail", lambda g: [(OFF_HEAD, _u64(M64))], policy="overwrite")
    )
    out.append(
        corrupt_case(
            "slot-length-beyond-capacity",
            lambda g: [(slot0 + 16, struct.pack("<II", 1, g.capacity + 1))],
        )
    )
    out.append(corrupt_case("bp-slot-position-forged", lambda g: [(slot0 + 8, _u64(77))]))
    out.append(corrupt_case("slot-epoch-zero", lambda g: [(slot0 + 16, struct.pack("<II", 0, 24))]))
    out.append(
        corrupt_case("slot-epoch-future", lambda g: [(slot0 + 16, struct.pack("<II", 9, 24))])
    )
    out.append(corrupt_case("bp-slot-marked-mid-write", lambda g: [(slot0, _u64(3))]))
    out.append(
        corrupt_case(
            "ow-slot-position-not-a-lap",
            lambda g: [(slot0 + 8, _u64(5))],
            policy="overwrite",
        )
    )

    b = _Build("tail-ahead-of-head-at-attach", "malformed")
    b.poke(OFF_ACCT[0], _u64(9))
    b.add(C_ATTACH, 0, 0, expect=("R", "refused", "BCIR_ERR_RING"))
    s = b.done()
    s.accounting = False
    out.append(s)

    # Unknown signals: skipped when optional, refused when REQUIRED -- end to end through a ring.
    b = _Build("unknown-signal-law", "malformed", slot_count=8)
    b.attach()
    cases = [
        (sample(9999, 1, seq=0), ("I", "skipped", "unknown", "BCIR_OK", "first")),
        (
            sample(9999, 1, seq=1, required=True),
            ("I", "refused", "unknown", "BCIR_ERR_TELEMETRY", "next"),
        ),
        (sample(1, 55, seq=2, required=True), ("I", "accepted", "none", "BCIR_OK", "next")),
        (
            sample(0x10001, 1, seq=3, required=True),
            ("I", "refused", "unknown", "BCIR_ERR_TELEMETRY", "next"),
        ),
        (sample(0x80000001, 1, seq=4), ("I", "skipped", "unknown", "BCIR_OK", "next")),
    ]
    for data, _ in cases:
        b.write(data, expect=OK)
    for _, want in cases:
        b.read(expect=_delivered())
        b.intake(expect=want)
    s = b.done(delivered=5, published=5)
    s.continuity = (0, 0, 0)
    out.append(s)

    # A malformed envelope carried by the ring is refused by the intake, not by the ring.
    for name, data, status in malformed_envelopes()[:6]:
        b = _Build(f"ring-carries-{name}", "malformed")
        b.attach()
        b.write(data, expect=OK)
        b.read(expect=_delivered())
        b.intake(expect=("I", "refused", "malformed", status, "none"))
        out.append(b.done(delivered=1, published=1))
    return out


def all_scenarios() -> list[Scenario]:
    return (
        saturation_scenarios()
        + wrap_scenarios()
        + torn_scenarios()
        + restart_scenarios()
        + stale_scenarios()
        + continuity_scenarios()
        + malformed_scenarios()
    )


# --- envelopes ------------------------------------------------------------------------------------


def envelope_corpus() -> list[tuple[str, bytes]]:
    """Well-formed envelopes covering both kinds, every clock and unit, the flag and the edges."""
    i64 = (-(1 << 63), (1 << 63) - 1)
    out = [
        ("datadna-plain", dna(0)),
        ("datadna-lost-and-wrap", dna(M32, lost=M32, session=M64, source=M64, generation=M32)),
        (
            "datadna-extremes",
            encode_envelope(
                TelemetryEnvelope(
                    "datadna",
                    source=3,
                    session=4,
                    seq=5,
                    generation=6,
                    record=(i64[0], i64[1], -1, 0, 1, 2, 3),
                )
            ),
        ),
        ("sample-plain", sample(1, 0)),
        ("sample-required", sample(15, -1, required=True, seq=9)),
        ("sample-max-signal", sample(0xFFFFFFFE, i64[1])),
        ("sample-min-value", sample(2, i64[0], generation=M32)),
    ]
    for clock, unit in (
        ("none", "none"),
        ("monotonic", "ns"),
        ("monotonic", "us"),
        ("realtime", "ns"),
        ("boot", "us"),
        ("cycles", "cycles"),
    ):
        out.append(
            (
                f"clock-{clock}-{unit}",
                encode_envelope(
                    TelemetryEnvelope(
                        "sample",
                        source=1,
                        session=1,
                        seq=1,
                        signal=3,
                        value=77,
                        clock=clock,
                        unit=unit,
                        timestamp=0 if clock == "none" else M64,
                    )
                ),
            )
        )
    return out


def _recrc(data: bytes) -> bytes:
    body = data[:-4]
    return body + struct.pack("<I", zlib.crc32(body) & M32)


def _put(data: bytes, offset: int, fmt: str, value) -> bytes:
    out = bytearray(data)
    struct.pack_into(fmt, out, offset, value)
    return bytes(out)


def malformed_envelopes() -> list[tuple[str, bytes, str]]:
    """One variant per wire law, each with the status the specification declares. Variants of a
    law checked after the CRC carry a repaired CRC, so the law -- not the CRC -- refuses them."""
    good = dna(3)
    smp = sample(1, 5)
    return [
        ("truncated-header", good[:63], "BCIR_ERR_TRUNCATED"),
        ("magic", b"BTEX" + good[4:], "BCIR_ERR_MAGIC"),
        ("version", _put(good, 4, "<H", 1), "BCIR_ERR_VERSION"),
        ("reserved-flag", _put(good, 6, "<H", 2), "BCIR_ERR_RESERVED"),
        ("kind", _put(good, 8, "<B", 3), "BCIR_ERR_TELEMETRY"),
        ("size-field", _put(good, 12, "<H", 76), "BCIR_ERR_TELEMETRY"),
        ("truncated-record", good[:-1], "BCIR_ERR_TRUNCATED"),
        ("trailing", good + b"\x00", "BCIR_ERR_TRAILING"),
        ("crc", good[:-1] + bytes([good[-1] ^ 1]), "BCIR_ERR_CRC"),
        ("reserved-u16", _recrc(_put(good, 14, "<H", 1)), "BCIR_ERR_RESERVED"),
        ("reserved-u64", _recrc(_put(good, 56, "<Q", 1)), "BCIR_ERR_RESERVED"),
        ("schema", _recrc(_put(good, 9, "<B", 1)), "BCIR_ERR_TELEMETRY"),
        ("clock-code", _recrc(_put(good, 10, "<B", 9)), "BCIR_ERR_TELEMETRY"),
        ("unit-code", _recrc(_put(good, 11, "<B", 9)), "BCIR_ERR_TELEMETRY"),
        ("clock-unit-pairing", _recrc(_put(good, 11, "<B", 3)), "BCIR_ERR_TELEMETRY"),
        (
            "timestamp-without-clock",
            _recrc(_put(_put(good, 10, "<B", 0), 11, "<B", 0)),
            "BCIR_ERR_TELEMETRY",
        ),
        ("source-zero", _recrc(_put(good, 16, "<Q", 0)), "BCIR_ERR_TELEMETRY"),
        ("session-zero", _recrc(_put(good, 24, "<Q", 0)), "BCIR_ERR_TELEMETRY"),
        ("sample-signal-zero", _recrc(_put(smp, 36, "<I", 0)), "BCIR_ERR_TELEMETRY"),
        ("sample-signal-reserved", _recrc(_put(smp, 36, "<I", M32)), "BCIR_ERR_TELEMETRY"),
        ("datadna-signal", _recrc(_put(good, 36, "<I", 1)), "BCIR_ERR_TELEMETRY"),
        ("datadna-required", _recrc(_put(good, 6, "<H", 1)), "BCIR_ERR_TELEMETRY"),
        ("datadna-unbound", _recrc(_put(good, 32, "<I", 0)), "BCIR_ERR_TELEMETRY"),
    ]


# --- the Python rail ------------------------------------------------------------------------------


def _ring_line(index: int, op: int, o: RingOutcome, region, size: int) -> str:
    if o.verdict == "delivered":
        plen, pcrc = len(o.payload), zlib.crc32(o.payload) & M32
    else:
        plen, pcrc = 0, 0
    rcrc = zlib.crc32(bytes(region[:size])) & M32
    return (
        f"{index} {op} R {o.verdict} {o.status} {o.position} {o.count} {o.epoch} {plen} "
        f"{pcrc:08x} {rcrc:08x}"
    )


def _load(region, offset: int) -> int:
    return struct.unpack_from("<Q", region, offset)[0]


def _check_line(index: int, op: int, region, intake: TelemetryIntake) -> str:
    try:
        g = decode_geometry(region)
    except RingError as exc:
        return f"{index} {op} A ERR {exc.status}"
    a = committed_accounting(region, g)
    r = intake.report()
    words = (
        _load(region, OFF_HEAD),
        _load(region, OFF_REFUSED),
        _load(region, OFF_P_OWNER),
        _load(region, OFF_C_OWNER),
        _load(region, OFF_P_BEAT),
        _load(region, OFF_C_BEAT),
    )
    fields = (a.tail, a.delivered, a.lost, a.stale, a.last_epoch, *words)
    fields += (r.accepted, r.skipped, r.refused, r.missing, r.reordered, r.duplicated)
    fields += (r.reported_lost, r.streams)
    return f"{index} {op} A " + " ".join(str(v) for v in fields)


def run_python(scenario: Scenario, index: int = 0) -> list[str]:
    """Run one scenario on the oracle; one trace line per operation."""
    g = scenario.geometry
    size = g.region_size
    region = bytearray(size)
    format_ring(region, g)
    producers = [RingProducer(region) for _ in range(ACTORS)]
    consumers = [RingConsumer(region) for _ in range(ACTORS)]
    last = [b""] * ACTORS
    intake = TelemetryIntake(scenario.live)
    lines = []
    for i, op in enumerate(scenario.ops):
        a = op.actor
        p, c = producers[a], consumers[a]
        if op.code in RING_OPS:
            if op.code == P_ATTACH:
                o = p.attach(op.arg)
            elif op.code == P_DETACH:
                o = p.detach()
            elif op.code == P_WRITE:
                o = p.publish(op.data)
            elif op.code == P_OPEN:
                o = p.open(op.arg)
            elif op.code == P_FILL:
                o = p.fill(op.arg, op.data)
            elif op.code == P_CLOSE:
                o = p.close()
            elif op.code == P_BEAT:
                o = p.beat()
            elif op.code == P_DROP:
                producers[a] = RingProducer(region)
                o = RingOutcome("ok")
            elif op.code == C_ATTACH:
                o = c.attach(op.arg)
            elif op.code == C_DETACH:
                o = c.detach()
            elif op.code == C_READ:
                o = c.consume()
            elif op.code == C_OPEN:
                o = c.open()
            elif op.code == C_COPY:
                o = c.copy()
            elif op.code == C_CLOSE:
                o = c.close()
            elif op.code == C_BEAT:
                o = c.beat()
            else:
                consumers[a] = RingConsumer(region)
                o = RingOutcome("ok")
            if o.verdict == "delivered":
                last[a] = o.payload
            lines.append(_ring_line(index, i, o, region, size))
        elif op.code == POKE:
            region[op.arg : op.arg + len(op.data)] = op.data
            lines.append(f"{index} {i} P {zlib.crc32(bytes(region)) & M32:08x}")
        elif op.code == INTAKE:
            t = intake.admit(op.data if op.data else last[a])
            lines.append(f"{index} {i} I {t.verdict} {t.reason} {t.status} {t.classification}")
        elif op.code == LIVE:
            intake.set_live_generation(op.arg)
            lines.append(f"{index} {i} L {op.arg}")
        else:
            lines.append(_check_line(index, i, region, intake))
    return lines


# --- grading a rail's trace -----------------------------------------------------------------------


def _fields(line: str) -> list[str]:
    return line.split()[2:]


def _op_holds(line: str, want: tuple) -> bool:
    got = _fields(line)
    if not got or got[0] != want[0]:
        return False
    if want[0] == "R":
        verdict, status = got[1], got[2]
        if (verdict, status) != (want[1], want[2]):
            return False
        return len(want) < 4 or int(got[4]) == want[3]
    return tuple(got[1:5]) == tuple(want[1:5])


def expectations_hold(scenario: Scenario, lines: list[str]) -> bool:
    if len(lines) != len(scenario.ops):
        return False
    return all(_op_holds(lines[i], want) for i, want in scenario.expect_ops.items())


def _written(scenario: Scenario, lines: list[str]) -> dict[int, bytes] | None:
    """position -> the bytes a successful write published there (the rail's own report)."""
    truth: dict[int, bytes] = {}
    opened: dict[int, tuple[int, bytearray]] = {}
    for op, line in zip(scenario.ops, lines):
        got = _fields(line)
        if not got or got[0] != "R":
            continue
        verdict, position = got[1], int(got[3])
        if op.code == P_OPEN and verdict == "open":
            opened[op.actor] = (position, bytearray(op.arg))
        elif op.code == P_FILL and verdict == "open" and op.actor in opened:
            opened[op.actor][1][op.arg : op.arg + len(op.data)] = op.data
        elif op.code == P_WRITE and verdict == "ok":
            truth[position] = op.data
        elif op.code == P_CLOSE and verdict == "ok" and op.actor in opened:
            truth[position] = bytes(opened.pop(op.actor)[1])
        elif op.code == P_DROP:
            opened.pop(op.actor, None)
    return truth


def torn_count(scenario: Scenario, lines: list[str]) -> int:
    """1 if the rail delivered any record that is not byte-identical to what was published at its
    position (a torn, mixed or phantom record); 0 otherwise."""
    if len(lines) != len(scenario.ops):
        return 1
    truth = _written(scenario, lines)
    for op, line in zip(scenario.ops, lines):
        got = _fields(line)
        if not got or got[0] != "R" or got[1] != "delivered":
            continue
        position, plen, pcrc = int(got[3]), int(got[6]), int(got[7], 16)
        want = truth.get(position)
        if want is None:
            if scenario.poked_publications and op.code in (C_READ, C_CLOSE):
                continue  # a forged record the scenario refuses elsewhere (graded as stale)
            return 1
        if (plen, pcrc) != (len(want), zlib.crc32(want) & M32):
            return 1
    return 0


def accounting_count(scenario: Scenario, lines: list[str]) -> int:
    """1 if the rail's committed accounting is not exactly what its own verdicts reported, or not
    what the specification declares; 0 otherwise."""
    if len(lines) != len(scenario.ops):
        return 1
    if not scenario.accounting:
        return 0
    g = scenario.geometry
    written = delivered = lost = stale = full = 0
    for op, line in zip(scenario.ops, lines):
        got = _fields(line)
        if not got or got[0] != "R":
            continue
        verdict, status, count = got[1], got[2], int(got[4])
        if op.code in (P_WRITE, P_CLOSE) and verdict == "ok":
            written += 1
        elif verdict == "delivered":
            delivered += 1
        elif verdict == "lost":
            lost += count
        elif verdict == "stale":
            stale += 1
        if status == "BCIR_ERR_FULL":
            full += 1
    final = _fields(lines[-1])
    if len(final) != 20 or final[0] != "A":
        return 1
    tail, a_delivered, a_lost, a_stale, _last, head, refused = (int(v) for v in final[1:8])
    p_beat, c_beat = int(final[10]), int(final[11])
    reported_lost = int(final[18])
    published = (head - g.origin) & M64
    ok = (
        (a_delivered, a_lost, a_stale) == (delivered, lost, stale)
        and refused == full
        and published == written + scenario.poked_publications
        and (tail - g.origin) & M64 == delivered + lost + stale
    )
    declared = {
        "delivered": a_delivered,
        "lost": a_lost,
        "stale": a_stale,
        "refused": refused,
        "published": published,
        "p_heartbeat": p_beat,
        "c_heartbeat": c_beat,
        "reported_lost": reported_lost,
    }
    for key, want in scenario.expect_final.items():
        ok = ok and declared[key] == want
    return 0 if ok else 1


def continuity_count(scenario: Scenario, lines: list[str]) -> int:
    if scenario.continuity is None:
        return 0
    if len(lines) != len(scenario.ops):
        return 1
    final = _fields(lines[-1])
    if len(final) != 20 or final[0] != "A":
        return 1
    return 0 if tuple(int(v) for v in final[15:18]) == scenario.continuity else 1


# --- the C rail -----------------------------------------------------------------------------------

C_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "runtime", "c"))
C_SOURCES = ("bcir_ring.c", "bcir_telemetry_envelope.c", "bcir_runtime.c", "test_ring.c")


def compiler() -> str | None:
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def build_harness(tmp: str, *, extra_flags=()) -> str | None:
    """Compile runtime/c/test_ring.c against the freestanding ring and envelope. None without a
    compiler (the quick tier hides one on purpose) and in an installed package (the wheel does not
    ship runtime/c); in a source checkout a visible compiler must build it -- a missing source
    there is a failure, never a skip (L21)."""
    from .run_all import _is_source_checkout

    cc = compiler()
    if cc is None:
        return None
    if not os.path.isfile(os.path.join(C_DIR, "test_ring.c")):
        if _is_source_checkout():
            raise RuntimeError("runtime/c/test_ring.c is missing from the checkout")
        return None
    exe = os.path.join(tmp, "test_ring" + (".exe" if os.name == "nt" else ""))
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra"] + (["-pthread"] if os.name == "posix" else [])
    build = subprocess.run(
        [
            cc,
            *flags,
            *extra_flags,
            "-I",
            C_DIR,
            *[os.path.join(C_DIR, source) for source in C_SOURCES],
            "-o",
            exe,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if build.returncode != 0:
        raise RuntimeError(f"ring harness build failed: {build.stderr[-2000:]}")
    return exe


def _run(argv, *, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run a harness; a harness that cannot be launched or that hangs is a RuntimeError the
    measurement counts as failed fixtures (L1: every exit is a verdict), never a traceback."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{argv[0]}: {exc}") from exc


def encode_script(scenarios) -> bytes:
    out = bytearray(b"BRTS" + struct.pack("<I", len(scenarios)))
    for s in scenarios:
        g = s.geometry
        out += struct.pack(
            "<BBIIQQII",
            ("backpressure", "overwrite").index(g.policy) + 1,
            ("telemetry", "control").index(g.payload) + 1,
            g.slot_size,
            g.slot_count,
            g.ring_id,
            g.origin,
            s.live,
            len(s.ops),
        )
        for op in s.ops:
            out += struct.pack("<BBQI", op.code, op.actor, op.arg & M64, len(op.data)) + op.data
    return bytes(out)


def run_c(exe: str, tmp: str, scenarios) -> list[list[str]]:
    path = os.path.join(tmp, "ring_script.bin")
    with open(path, "wb") as fh:
        fh.write(encode_script(scenarios))
    run = _run([exe, "--script", path])
    if run.returncode != 0:
        raise RuntimeError(f"ring harness failed: {run.stderr[-2000:]}")
    traces: list[list[str]] = [[] for _ in scenarios]
    for line in run.stdout.splitlines():
        head = line.split(" ", 1)[0]
        traces[int(head)].append(line)
    return traces


def c_signals(exe: str) -> list[str]:
    run = _run([exe, "--signals"], timeout=60)
    if run.returncode != 0:
        raise RuntimeError(f"signal dump failed: {run.stderr[-2000:]}")
    return run.stdout.split()


def c_api(exe: str) -> tuple[int, str]:
    try:
        run = _run([exe, "--api"], timeout=60)
    except RuntimeError as exc:
        return 2, str(exc)
    return run.returncode, run.stdout + run.stderr


def c_envelopes(exe: str, tmp: str, blobs) -> list[str]:
    """Decode each envelope on the C rail: its fields and its C re-encoding, or its status."""
    path = os.path.join(tmp, "envelopes.bin")
    with open(path, "wb") as fh:
        fh.write(b"".join(struct.pack("<I", len(b)) + b for b in blobs))
    run = _run([exe, "--envelopes", path], timeout=60)
    if run.returncode != 0:
        raise RuntimeError(f"envelope dump failed: {run.stderr[-2000:]}")
    return run.stdout.splitlines()


def parse_c_envelope(line: str) -> tuple[TelemetryEnvelope | None, bytes, str]:
    """(the envelope a C dump line describes, the C rail's own re-encoding, status)."""
    parts = line.split()
    if not parts or parts[0] != "env":
        return None, b"", parts[1].removeprefix("status=") if len(parts) > 1 else "unparseable"
    kv = dict(part.split("=", 1) for part in parts[1:])
    kind = kv["kind"]
    record = tuple(int(v) for v in kv["record"].split(",")) if kind == "datadna" else ()
    env = TelemetryEnvelope(
        kind=kind,
        source=int(kv["source"]),
        session=int(kv["session"]),
        seq=int(kv["seq"]),
        generation=int(kv["generation"]),
        signal=int(kv["signal"]),
        lost=int(kv["lost"]),
        clock=kv["clock"],
        unit=kv["unit"],
        timestamp=int(kv["timestamp"]),
        required=kv["required"] == "1",
        value=int(kv["value"]) if kind == "sample" else 0,
        record=record,
    )
    return env, bytes.fromhex(kv["hex"]), "BCIR_OK"


STRESS_RUNS = (
    ("--stress", "bp"),
    ("--stress", "ow"),
    ("--procs", "kill-producer", "bp"),
    ("--procs", "kill-producer", "ow"),
    ("--procs", "kill-consumer", "bp"),
    ("--procs", "kill-consumer", "ow"),
)


def c_concurrent(
    exe: str, records: int = 60000, seed: int = 20260923
) -> list[tuple[str, int, str]]:
    """Every stress configuration: (name, violations, output line). Violations are the harness's
    count, or 1 when a run fails, crashes or prints nothing parseable (L1). An empty list when
    the host has no POSIX threads/processes (exit 3: UNAVAILABLE, measured by CI's Linux jobs)."""
    out = []
    for run_args in STRESS_RUNS:
        argv = [exe, *run_args, str(records), str(seed)]
        try:
            run = _run(argv, timeout=300)
        except RuntimeError as exc:
            out.append((" ".join(run_args), 1, str(exc)))
            continue
        if run.returncode == 3 and "UNAVAILABLE" in run.stdout:
            return []
        line = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        try:
            violations = int(fields["violations"])
        except (KeyError, ValueError):
            violations = 1
        if run.returncode not in (0, 1):
            violations = max(violations, 1)
        out.append((" ".join(run_args), violations, line))
    return out


def c_bench(exe: str, records: int = 300000, repeats: int = 5) -> dict[str, float] | None:
    """The throughput row: ring time / memcpy time for the same bytes (lower is better)."""
    try:
        run = _run([exe, "--bench", str(records), "192", "256", str(repeats)], timeout=300)
    except RuntimeError:
        return None
    if run.returncode != 0:
        return None
    fields = dict(part.split("=", 1) for part in run.stdout.split() if "=" in part)
    try:
        return {
            key: float(fields[key]) for key in ("ring_ns", "memcpy_ns", "ratio", "records_per_s")
        }
    except (KeyError, ValueError):
        return None


def control_ring_transport():
    """A live control ring for bcir.tests.control_fixtures.run_python(transport=...): every
    submitted record is written by a producer endpoint and read back by a consumer endpoint."""
    g = RingGeometry("backpressure", "control", CONTROL_SLOT_MIN, 4, ring_id=1)
    region = bytearray(g.region_size)
    format_ring(region, g)
    producer, consumer = RingProducer(region), RingConsumer(region)
    producer.attach()
    consumer.attach()

    def carry(data: bytes) -> bytes:
        written = producer.publish(data)
        read = consumer.consume()
        if (
            written.verdict != "ok"
            or read.verdict != "delivered"
            or read.position != written.position
        ):
            return b"RINGFAIL"
        return read.payload

    return carry


# --- the rows -------------------------------------------------------------------------------------

ROWS = (
    "ring.abi.mismatches",
    "ring.malformed.accepted",
    "ring.loss.unaccounted",
    "ring.torn.delivered",
    "ring.sequence.misreported",
    "ring.stale.accepted",
    "ring.traces.divergent",
    "ring.control.divergent",
    "ring.concurrent.violations",
)


def signal_table_drift() -> int:
    """1 when the checked-in C table differs from the generator's output, else 0."""
    path = os.path.join(C_DIR, "bcir_signal_table.h")
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            return int(fh.read() != emit_c_header())
    except OSError:
        return 1


def measure(exe: str, tmp: str) -> dict[str, float]:
    """The G15 rows over the declared corpora, on both rails (`exe`: the built harness).

    ring.abi.mismatches         signal-table rows whose C bytes differ from Python's, the
                                generated header's drift, and corpus envelopes whose Python encode
                                -> C decode -> Python re-encode (or C re-encode) is not identical
    ring.malformed.accepted     (malformed variant, rail) pairs not refused with the declared status:
                                envelope wire laws, geometry laws, owner and slot corruption,
                                unknown-signal law
    ring.loss.unaccounted       (scenario, rail) pairs whose committed accounting is not exactly
                                what the rail's verdicts reported, or not the declared numbers
    ring.torn.delivered         (scenario, rail) pairs that delivered a record not byte-identical to
                                what was published at its position
    ring.sequence.misreported   (continuity fixture, rail) pairs whose (missing, reordered,
                                duplicated) is not the declared one
    ring.stale.accepted         (stale fixture, rail) pairs not refused as declared
    ring.traces.divergent       scenarios whose two rails' traces differ in any line
    ring.control.divergent      (S3-A control scenario, rail) pairs whose plane trace through a
                                live control ring differs from the direct trace, plus the
                                scenarios whose two rails' ring traces differ
    ring.concurrent.violations  concurrent C runs (threads; processes with a peer SIGKILLed and
                                taken over) with a torn, unaccounted or discontinuous outcome --
                                omitted (NOT-MEASURED) on a host with no POSIX threads/processes

    Every path is a count: a rail that crashes or prints what cannot be parsed fails every
    fixture it was handed (L1)."""
    from . import control_fixtures as cf

    out: dict[str, float] = {}

    # --- abi: the signal table and the envelope corpus
    mismatches = signal_table_drift()
    try:
        rows_c = c_signals(exe)
    except RuntimeError:
        rows_c = []
    rows_py = [encode_row(d).hex() for d in builtin_definitions()]
    if len(rows_c) != len(rows_py):
        rows_c = [""] * len(rows_py)
    mismatches += sum(a != b for a, b in zip(rows_c, rows_py))
    corpus = envelope_corpus()
    try:
        lines = c_envelopes(exe, tmp, [blob for _, blob in corpus])
    except RuntimeError:
        lines = []
    if len(lines) != len(corpus):
        lines = [""] * len(corpus)
    for (_name, data), line in zip(corpus, lines):
        try:
            python_ok = encode_envelope(decode_envelope(data)) == data
        except Exception:  # noqa: BLE001 -- a rail that raises decided nothing (L1)
            python_ok = False
        try:
            env, c_bytes, _status = parse_c_envelope(line)
            c_ok = env is not None and encode_envelope(env) == data and c_bytes == data
        except Exception:  # noqa: BLE001 -- an unparseable or unencodable C line fails
            c_ok = False
        mismatches += not (python_ok and c_ok)
    out["ring.abi.mismatches"] = float(mismatches)

    # --- malformed envelopes (both rails' decoders)
    accepted = 0
    variants = malformed_envelopes()
    try:
        lines = c_envelopes(exe, tmp, [data for _, data, _ in variants])
    except RuntimeError:
        lines = []
    if len(lines) != len(variants):
        lines = [""] * len(variants)
    for (_name, data, want), line in zip(variants, lines):
        try:
            decode_envelope(data)
            python_status = "BCIR_OK"
        except TelemetryError as exc:
            python_status = exc.status
        except Exception:  # noqa: BLE001 -- a rail that raises refused nothing by its law
            python_status = "<raised>"
        accepted += (python_status != want) + (line != f"refused status={want}")

    # --- the scripted scenarios
    scenarios = all_scenarios()
    try:
        traces = run_c(exe, tmp, scenarios)
    except (RuntimeError, ValueError, IndexError):
        traces = [[] for _ in scenarios]
    rows = {"loss": 0, "torn": 0, "continuity": 0, "stale": 0, "divergent": 0}
    for index, (scenario, c_lines) in enumerate(zip(scenarios, traces)):
        try:
            py_lines = run_python(scenario, index)
        except Exception:  # noqa: BLE001 -- an oracle that raises decided nothing
            py_lines = []
        for lines in (py_lines, c_lines):
            held = expectations_hold(scenario, lines)
            unaccounted = accounting_count(scenario, lines)
            if scenario.family == "malformed":
                accepted += not held
            elif scenario.family == "stale":
                rows["stale"] += not held
            else:  # a pinned verdict of a saturation/wrap/torn/restart/continuity fixture
                unaccounted = unaccounted or not held
            rows["loss"] += int(bool(unaccounted))
            rows["torn"] += torn_count(scenario, lines)
            rows["continuity"] += continuity_count(scenario, lines)
        rows["divergent"] += py_lines != c_lines or not py_lines
    out["ring.malformed.accepted"] = float(accepted)
    out["ring.loss.unaccounted"] = float(rows["loss"])
    out["ring.torn.delivered"] = float(rows["torn"])
    out["ring.sequence.misreported"] = float(rows["continuity"])
    out["ring.stale.accepted"] = float(rows["stale"])
    out["ring.traces.divergent"] = float(rows["divergent"])

    # --- control records through a live control ring, both rails
    control = cf.all_scenarios()
    try:
        direct_c = cf.run_c(cf_exe(tmp), tmp, control)
        ring_c = cf.run_c(cf_exe(tmp), tmp, control, via_ring=True)
    except (RuntimeError, ValueError, IndexError):
        direct_c, ring_c = [[]] * len(control), [None] * len(control)
    divergent = 0
    for scenario, d_c, r_c in zip(control, direct_c, ring_c):
        try:
            _, _, direct_py = cf.run_python(scenario)
            _, _, ring_py = cf.run_python(scenario, transport=control_ring_transport)
        except Exception:  # noqa: BLE001
            direct_py, ring_py = [], None
        divergent += (ring_py != direct_py) + (r_c != d_c) + (ring_py != r_c)
    out["ring.control.divergent"] = float(divergent)

    # --- concurrency (C, POSIX)
    runs = c_concurrent(exe)
    if runs:
        out["ring.concurrent.violations"] = float(sum(1 for _, v, _ in runs if v))
    return out


_CF_EXE: dict[str, str] = {}


def cf_exe(tmp: str) -> str:
    """The control harness (built once per temporary directory)."""
    from . import control_fixtures as cf

    if tmp not in _CF_EXE:
        sub = os.path.join(tmp, "control")
        os.makedirs(sub, exist_ok=True)
        exe = cf.build_harness(sub)
        if exe is None:
            raise RuntimeError("no control harness")
        _CF_EXE[tmp] = exe
    return _CF_EXE[tmp]


def measure_throughput(exe: str) -> dict[str, float]:
    bench = c_bench(exe)
    return {} if bench is None else {"ring.throughput": bench["ratio"]}


__all__ = [
    "ACTORS",
    "DETACHED",
    "ATTACHED",
    "ENVELOPE_HEADER_SIZE",
    "ENVELOPE_SIZES",
    "SLOT_HEADER",
]
