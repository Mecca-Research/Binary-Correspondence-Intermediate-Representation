"""The live SPSC ring, version zero ("BRNG", G15, S3-B) -- the executable oracle.

A single-producer / single-consumer ring in a shared region: head and tail, acquire/release
publication, a per-slot sequence, a declared overwrite or backpressure policy and exact loss
accounting, carrying TelemetryEnvelopeV0 records first and ControlRecordV1 records second. The
normative specification is docs/kernel/BCIR_LIVE_RING_ABI.md; the production rail is the
freestanding C twin runtime/c/bcir_ring.{h,c} (C11 atomics). This module is the oracle: it
executes the same algorithm over the same bytes one operation at a time, so a scripted
interleaving -- a consumer mid-copy while the producer laps it, a producer that dies between two
stores, a takeover, a zombie -- leaves the region byte-identical on both rails after every step.

Region (little-endian; the words the peers share are read and written atomically by the C twin):

    line 0   geometry, immutable after `format_ring`, CRC-sealed:
             magic[4]="BRNG" version:u16=0 policy:u8 payload:u8 slot_size:u32 slot_count:u32
             ring_id:u64 region_size:u64 origin:u64 reserved[20] crc32:u32
    line 1   producer ownership (written at attach/detach only): p_owner:u64 = epoch<<32 | state,
             epoch_origin:u64
    line 2   producer progress (written per publish, read by the consumer): head:u64 p_heartbeat:u64
    line 3   producer counters (the producer's alone): refused:u64
    line 4   consumer progress (written per commit, read by the producer): tail:u64
    line 5   consumer (the consumer's alone): c_owner:u64 c_heartbeat:u64 c_commit:u64 reserved:u64
             acct[0]
    line 6   acct[1]
             acct = tail:u64 lost:u64 stale:u64 last_epoch:u32 reserved:u32; the valid record is
             acct[c_commit & 1] -- double-buffered, so a consumer that dies mid-update never
             leaves torn accounting behind. The progress word is published after each commit, so
             it never runs ahead of the accounting; a successor republishes it at attach.
    slots    at 448, stride slot_size: seq:u64 (odd while written) pos:u64 epoch:u32 length:u32
             payload[slot_size - 24]

Each line has one writer, and a peer's hot path reads exactly one line the other writes per
publication (head one way, tail the other) -- the words a side writes every operation never share
a cache line with the words the other side polls.

Policies: BACKPRESSURE (a full ring refuses the record and counts it in `refused`; the producer
never overwrites an unconsumed slot) and OVERWRITE (the producer never waits; a consumer that
falls behind is told how many records it lost, exactly, and never receives a torn one). A control
ring is always BACKPRESSURE: a control record is never overwritten.

Every position the producer publishes is committed by the consumer exactly once as delivered,
lost or stale, so at quiescence `published == delivered + lost + stale` and the producer's
`attempted == published + refused` -- the exact loss accounting the G15 gate counts.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

RING_MAGIC = b"BRNG"
RING_VERSION = 0
HEADER_SIZE = 448
SLOT_HEADER = 24
# A control ring carries every legal ControlRecordV1: a slot's payload holds the record ABI's
# declared bound (bcir.abi.control_abi.CONTROL_RECORD_MAX_BYTES, 192 bytes), so the slot is that
# plus the slot header, rounded up to a cache line. bcir/tests/test_live_ring.py holds this to
# the record ABI and to the C twin's BCIR_RING_CONTROL_SLOT_MIN.
CONTROL_SLOT_MIN = 256
GEOMETRY_SIZE = 64

POLICIES = ("backpressure", "overwrite")  # code = index + 1
PAYLOADS = ("telemetry", "control")  # code = index + 1
SLOT_SIZE_MIN = 64
SLOT_SIZE_MAX = 65536
SLOT_COUNT_MIN = 2
SLOT_COUNT_MAX = 1 << 20

DETACHED, ATTACHING, ATTACHED = 0, 1, 2
STATES = ("detached", "attaching", "attached")
EPOCH_MAX = 0xFFFFFFFF

VERDICTS = ("ok", "open", "delivered", "empty", "lost", "stale", "refused")

# word offsets
OFF_P_OWNER = 64
OFF_ORIGIN = 72
OFF_HEAD = 128
OFF_P_BEAT = 136
OFF_REFUSED = 192
OFF_TAIL = 256
OFF_C_OWNER = 320
OFF_C_BEAT = 328
OFF_C_COMMIT = 336
OFF_ACCT = (352, 384)
SLOT_SEQ, SLOT_POS, SLOT_WORD, SLOT_PAYLOAD = 0, 8, 16, 24

_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF
_HALF = 1 << 63

# magic version policy payload slot_size slot_count ring_id region_size origin
_GEOMETRY = struct.Struct("<4sHBBIIQQQ")
_GEOMETRY_RESERVED = (40, 60)
_Q = struct.Struct("<Q")


class RingError(ValueError):
    """A region, a geometry or an operation the ring's laws refuse. `status` is the C twin's
    `bcir_status` name (`BCIR_ERR_TRUNCATED`, `_MAGIC`, `_VERSION`, `_CRC`, `_RESERVED`,
    `_RING`, `_NOSPACE`)."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status


@dataclass(frozen=True)
class RingGeometry:
    policy: str
    payload: str
    slot_size: int
    slot_count: int
    ring_id: int
    origin: int = 0

    @property
    def region_size(self) -> int:
        return HEADER_SIZE + self.slot_count * self.slot_size

    @property
    def capacity(self) -> int:
        """The payload bytes one slot holds."""
        return self.slot_size - SLOT_HEADER


@dataclass(frozen=True)
class RingAccounting:
    tail: int
    delivered: int
    lost: int
    stale: int
    last_epoch: int


@dataclass(frozen=True)
class RingOutcome:
    """One operation's result. `verdict` is one of `VERDICTS`; `status` the C twin's name;
    `position` the ring position it concerns; `count` the positions it consumed (a LOST verdict
    after a lap consumes many); `epoch` the record's producer epoch; `payload` a DELIVERED
    record's bytes."""

    verdict: str
    status: str = "BCIR_OK"
    position: int = 0
    count: int = 0
    epoch: int = 0
    payload: bytes = b""


def _refused(status: str, position: int = 0) -> RingOutcome:
    return RingOutcome("refused", status, position)


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def validate_geometry(g: RingGeometry) -> None:
    """The geometry's value laws (all `BCIR_ERR_RING`), in the specification's order."""

    def fail(message: str) -> RingError:
        return RingError("BCIR_ERR_RING", message)

    if g.policy not in POLICIES:
        raise fail(f"unknown policy {g.policy!r}")
    if g.payload not in PAYLOADS:
        raise fail(f"unknown payload {g.payload!r}")
    if (
        type(g.slot_size) is not int
        or not SLOT_SIZE_MIN <= g.slot_size <= SLOT_SIZE_MAX
        or g.slot_size % 64
    ):
        raise fail(f"slot_size {g.slot_size!r} is not a multiple of 64 in [64, 65536]")
    if (
        type(g.slot_count) is not int
        or not SLOT_COUNT_MIN <= g.slot_count <= SLOT_COUNT_MAX
        or not _is_pow2(g.slot_count)
    ):
        raise fail(f"slot_count {g.slot_count!r} is not a power of two in [2, 2**20]")
    if type(g.ring_id) is not int or not 0 < g.ring_id <= _M64:
        raise fail("ring_id is a nonzero u64")
    if type(g.origin) is not int or not 0 <= g.origin <= _M64:
        raise fail("origin is a u64")
    if g.payload == "control" and g.policy != "backpressure":
        raise fail("a control ring is backpressure: a control record is never overwritten")
    if g.payload == "control" and g.slot_size < CONTROL_SLOT_MIN:
        raise fail(
            f"a control ring carries every legal control record: slot_size >= {CONTROL_SLOT_MIN}"
        )


def encode_geometry(g: RingGeometry) -> bytes:
    validate_geometry(g)
    line = _GEOMETRY.pack(
        RING_MAGIC,
        RING_VERSION,
        POLICIES.index(g.policy) + 1,
        PAYLOADS.index(g.payload) + 1,
        g.slot_size,
        g.slot_count,
        g.ring_id,
        g.region_size,
        g.origin,
    ) + bytes(20)
    return line + struct.pack("<I", zlib.crc32(line) & _M32)


def decode_geometry(region) -> RingGeometry:
    """The geometry line's laws, in the specification's order, against the region's length."""
    if len(region) < HEADER_SIZE:
        raise RingError("BCIR_ERR_TRUNCATED", f"{len(region)} bytes is shorter than a header")
    line = bytes(region[:GEOMETRY_SIZE])
    (magic, version, policy, payload, slot_size, slot_count, ring_id, region_size, origin) = (
        _GEOMETRY.unpack_from(line, 0)
    )
    if magic != RING_MAGIC:
        raise RingError("BCIR_ERR_MAGIC", f"magic {magic!r}")
    if version != RING_VERSION:
        raise RingError("BCIR_ERR_VERSION", f"version {version} (a v0 peer reads v0)")
    (crc,) = struct.unpack_from("<I", line, 60)
    if zlib.crc32(line[:60]) & _M32 != crc:
        raise RingError("BCIR_ERR_CRC", "the geometry line's CRC does not match")
    if any(line[_GEOMETRY_RESERVED[0] : _GEOMETRY_RESERVED[1]]):
        raise RingError("BCIR_ERR_RESERVED", "a reserved geometry byte is nonzero")
    if not 1 <= policy <= len(POLICIES):
        raise RingError("BCIR_ERR_RING", f"unknown policy code {policy}")
    if not 1 <= payload <= len(PAYLOADS):
        raise RingError("BCIR_ERR_RING", f"unknown payload code {payload}")
    g = RingGeometry(
        POLICIES[policy - 1], PAYLOADS[payload - 1], slot_size, slot_count, ring_id, origin
    )
    validate_geometry(g)
    if region_size != g.region_size:
        raise RingError("BCIR_ERR_RING", f"region_size {region_size} != {g.region_size}")
    if g.region_size > len(region):
        raise RingError("BCIR_ERR_TRUNCATED", f"the region needs {g.region_size} bytes")
    return g


def _load(buf, off: int) -> int:
    return _Q.unpack_from(buf, off)[0]


def _store(buf, off: int, value: int) -> None:
    _Q.pack_into(buf, off, value & _M64)


def _owner(epoch: int, state: int) -> int:
    return (epoch << 32) | state


def format_ring(region, g: RingGeometry) -> None:
    """Lay an empty ring into `region` (a writable buffer at least `g.region_size` long)."""
    validate_geometry(g)
    if len(region) < g.region_size:
        raise RingError("BCIR_ERR_NOSPACE", f"the ring needs {g.region_size} bytes")
    region[: g.region_size] = bytes(g.region_size)
    region[:GEOMETRY_SIZE] = encode_geometry(g)
    _store(region, OFF_HEAD, g.origin)
    _store(region, OFF_ORIGIN, g.origin)
    _store(region, OFF_TAIL, g.origin)
    _store(region, OFF_ACCT[0], g.origin)


def _snapshot_owner(region, off: int) -> tuple[int, int]:
    word = _load(region, off)
    return word >> 32, word & _M32


def committed_accounting(region, g: RingGeometry) -> RingAccounting:
    """The consumer's committed accounting, read as any peer or supervisor reads it."""
    commit = _load(region, OFF_C_COMMIT)
    base = OFF_ACCT[commit & 1]
    tail, lost, stale, word = (_load(region, base + 8 * i) for i in range(4))
    delivered = (tail - g.origin - lost - stale) & _M64
    return RingAccounting(tail, delivered, lost, stale, word & _M32)


def ring_state(region) -> dict:
    """Every shared word a supervisor reads to judge the peers (heartbeats, owners, progress)."""
    g = decode_geometry(region)
    p_epoch, p_state = _snapshot_owner(region, OFF_P_OWNER)
    c_epoch, c_state = _snapshot_owner(region, OFF_C_OWNER)
    return {
        "geometry": g,
        "producer": (p_epoch, p_state),
        "consumer": (c_epoch, c_state),
        "head": _load(region, OFF_HEAD),
        "epoch_origin": _load(region, OFF_ORIGIN),
        "p_heartbeat": _load(region, OFF_P_BEAT),
        "c_heartbeat": _load(region, OFF_C_BEAT),
        "refused": _load(region, OFF_REFUSED),
        "tail": _load(region, OFF_TAIL),
        "accounting": committed_accounting(region, g),
    }


class _Endpoint:
    def __init__(self, region) -> None:
        self.region = region
        self.geometry: RingGeometry | None = None
        self.epoch = 0

    def _slot(self, position: int) -> int:
        g = self.geometry
        return HEADER_SIZE + (position & (g.slot_count - 1)) * g.slot_size

    def _attach_checks(self, owner_off: int, takeover: int, allow_attaching: bool):
        """(geometry, epoch) or a refusal: the geometry laws, then the owner word's."""
        try:
            g = decode_geometry(self.region)
        except RingError as exc:
            return None, _refused(exc.status)
        epoch, state = _snapshot_owner(self.region, owner_off)
        if state > ATTACHED or (state == ATTACHING and not allow_attaching):
            return None, _refused("BCIR_ERR_RING")
        if takeover == 0:
            if state != DETACHED:
                return None, _refused("BCIR_ERR_BUSY")
        elif state != ATTACHED or epoch != takeover:
            return None, _refused("BCIR_ERR_STALE")
        if epoch >= EPOCH_MAX:
            return None, _refused("BCIR_ERR_RING")
        return (g, epoch), None


class RingProducer(_Endpoint):
    """The producer endpoint. `publish` writes one record; `open`/`fill`/`close` are its steps
    (a zero-copy producer writes in place between them)."""

    def __init__(self, region) -> None:
        super().__init__(region)
        self.cached_tail = 0
        self._open: tuple[int, int, int, int] | None = None

    def _held(self) -> bool:
        return self.epoch != 0 and _load(self.region, OFF_P_OWNER) == _owner(self.epoch, ATTACHED)

    def _snapshot_tail(self) -> int:
        return _load(self.region, OFF_TAIL)

    def attach(self, takeover: int = 0) -> RingOutcome:
        """Take the producer end: from DETACHED, or (`takeover` = the epoch the embedding proved
        dead) from that dead producer -- whose half-written slot at the head is retired."""
        checked, refusal = self._attach_checks(OFF_P_OWNER, takeover, allow_attaching=False)
        if refusal is not None:
            return refusal
        g, epoch = checked
        region = self.region
        self.geometry = g
        _store(region, OFF_P_OWNER, _owner(epoch, ATTACHING))
        head = _load(region, OFF_HEAD)
        if takeover:
            base = self._slot(head)
            seq = _load(region, base + SLOT_SEQ)
            if seq & 1:
                _store(region, base + SLOT_POS, head)
                _store(region, base + SLOT_WORD, 0)
                _store(region, base + SLOT_SEQ, seq + 1)
        _store(region, OFF_ORIGIN, head)
        _store(region, OFF_P_OWNER, _owner(epoch + 1, ATTACHED))
        self.epoch = epoch + 1
        self._open = None
        self.cached_tail = self._snapshot_tail()
        return RingOutcome("ok", position=head, epoch=self.epoch)

    def detach(self) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        if self._open is not None:
            return _refused("BCIR_ERR_RING")
        _store(self.region, OFF_P_OWNER, _owner(self.epoch, DETACHED))
        return RingOutcome("ok", epoch=self.epoch)

    def open(self, length: int) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        if self._open is not None:
            return _refused("BCIR_ERR_RING")
        g = self.geometry
        if type(length) is not int or not 0 <= length <= g.capacity:
            return _refused("BCIR_ERR_NOSPACE")
        region = self.region
        head = _load(region, OFF_HEAD)
        if g.policy == "backpressure":
            if (head - self.cached_tail) & _M64 >= g.slot_count:
                self.cached_tail = self._snapshot_tail()
            depth = (head - self.cached_tail) & _M64
            if depth >= _HALF or depth > g.slot_count:
                return _refused("BCIR_ERR_RING", head)
            if depth == g.slot_count:
                _store(region, OFF_REFUSED, _load(region, OFF_REFUSED) + 1)
                return _refused("BCIR_ERR_FULL", head)
        base = self._slot(head)
        seq = _load(region, base + SLOT_SEQ)
        if seq & 1:
            return _refused("BCIR_ERR_RING", head)
        _store(region, base + SLOT_SEQ, seq + 1)
        _store(region, base + SLOT_POS, head)
        _store(region, base + SLOT_WORD, self.epoch | (length << 32))
        self._open = (head, base, seq, length)
        return RingOutcome("open", position=head, epoch=self.epoch)

    def fill(self, offset: int, data: bytes) -> RingOutcome:
        if self._open is None:
            return _refused("BCIR_ERR_RING")
        head, base, _seq, length = self._open
        data = bytes(data)
        if offset % 8 or offset < 0 or offset + len(data) > length:
            return _refused("BCIR_ERR_NOSPACE", head)
        for i in range(0, len(data), 8):
            word = data[i : i + 8].ljust(8, b"\x00")
            self.region[base + SLOT_PAYLOAD + offset + i : base + SLOT_PAYLOAD + offset + i + 8] = (
                word
            )
        return RingOutcome("open", position=head, epoch=self.epoch)

    def close(self) -> RingOutcome:
        if self._open is None:
            return _refused("BCIR_ERR_RING")
        head, base, seq, _length = self._open
        region = self.region
        _store(region, base + SLOT_SEQ, seq + 2)
        _store(region, OFF_HEAD, head + 1)
        _store(region, OFF_P_BEAT, _load(region, OFF_P_BEAT) + 1)
        self._open = None
        return RingOutcome("ok", position=head, count=1, epoch=self.epoch)

    def publish(self, payload: bytes) -> RingOutcome:
        opened = self.open(len(payload))
        if opened.verdict != "open":
            return opened
        self.fill(0, payload)
        return self.close()

    def beat(self) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        _store(self.region, OFF_P_BEAT, _load(self.region, OFF_P_BEAT) + 1)
        return RingOutcome("ok", epoch=self.epoch)


class RingConsumer(_Endpoint):
    """The consumer endpoint. `consume` decides one position: DELIVERED, EMPTY, LOST (with the exact
    count a lap consumed) or STALE; `open`/`copy`/`close` are its steps."""

    def __init__(self, region) -> None:
        super().__init__(region)
        self.commit = 0
        self.tail = 0
        self.lost = 0
        self.stale = 0
        self.last_epoch = 0
        self.cached_head = 0
        self._cursor: list | None = None

    def _held(self) -> bool:
        return self.epoch != 0 and _load(self.region, OFF_C_OWNER) == _owner(self.epoch, ATTACHED)

    def attach(self, takeover: int = 0) -> RingOutcome:
        """Take the consumer end: from DETACHED, or from the dead consumer of epoch `takeover`,
        resuming at its last committed accounting."""
        checked, refusal = self._attach_checks(OFF_C_OWNER, takeover, allow_attaching=False)
        if refusal is not None:
            return refusal
        g, epoch = checked
        region = self.region
        commit = _load(region, OFF_C_COMMIT)
        base = OFF_ACCT[commit & 1]
        tail, lost, stale, word = (_load(region, base + 8 * i) for i in range(4))
        head = _load(region, OFF_HEAD)
        if (head - tail) & _M64 >= _HALF:
            return _refused("BCIR_ERR_RING", tail)
        self.geometry = g
        _store(region, OFF_C_OWNER, _owner(epoch + 1, ATTACHED))
        _store(region, OFF_TAIL, tail)
        self.epoch = epoch + 1
        self.commit = commit
        self.tail, self.lost, self.stale, self.last_epoch = tail, lost, stale, word & _M32
        self.cached_head = tail
        self._cursor = None
        return RingOutcome("ok", position=tail, epoch=self.epoch)

    def detach(self) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        if self._cursor is not None:
            return _refused("BCIR_ERR_RING")
        _store(self.region, OFF_C_OWNER, _owner(self.epoch, DETACHED))
        return RingOutcome("ok", epoch=self.epoch)

    def _commit(self, tail: int, lost: int, stale: int, last_epoch: int) -> None:
        region = self.region
        nxt = self.commit + 1
        base = OFF_ACCT[nxt & 1]
        _store(region, base, tail)
        _store(region, base + 8, lost)
        _store(region, base + 16, stale)
        _store(region, base + 24, last_epoch)
        _store(region, OFF_C_COMMIT, nxt)
        _store(region, OFF_C_BEAT, _load(region, OFF_C_BEAT) + 1)
        _store(region, OFF_TAIL, tail)
        self.commit = nxt & _M64
        self.tail, self.lost, self.stale, self.last_epoch = tail & _M64, lost, stale, last_epoch

    def _lose(self, position: int, count: int) -> RingOutcome:
        self._commit(position + count, self.lost + count, self.stale, self.last_epoch)
        return RingOutcome("lost", position=position, count=count)

    def _owner_snapshot(self) -> tuple[int, int, int]:
        """(epoch, state, epoch_origin) of the producer end, read consistently."""
        region = self.region
        while True:
            first = _load(region, OFF_P_OWNER)
            origin = _load(region, OFF_ORIGIN)
            if _load(region, OFF_P_OWNER) == first:
                return first >> 32, first & _M32, origin

    def open(self) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        if self._cursor is not None:
            return _refused("BCIR_ERR_RING")
        g = self.geometry
        region = self.region
        q = self.tail
        if g.policy == "overwrite" or q == self.cached_head:
            self.cached_head = _load(region, OFF_HEAD)
        head = self.cached_head
        depth = (head - q) & _M64
        if depth == 0:
            return RingOutcome("empty", position=q)
        if depth >= _HALF:
            return _refused("BCIR_ERR_RING", q)
        if depth > g.slot_count:
            if g.policy == "backpressure":
                return _refused("BCIR_ERR_RING", q)
            return self._lose(q, depth - g.slot_count)
        base = self._slot(q)
        seq = _load(region, base + SLOT_SEQ)
        if seq & 1:
            if g.policy == "backpressure":
                return _refused("BCIR_ERR_RING", q)
            return self._lose(q, 1)
        pos = _load(region, base + SLOT_POS)
        word = _load(region, base + SLOT_WORD)
        self._cursor = [q, base, seq, pos, word & _M32, word >> 32, None]
        return RingOutcome("open", position=q)

    def copy(self) -> RingOutcome:
        if self._cursor is None:
            return _refused("BCIR_ERR_RING")
        q, base, _seq, _pos, _epoch, length, _data = self._cursor
        n = min(length, self.geometry.capacity)
        words = (n + 7) // 8
        start = base + SLOT_PAYLOAD
        self._cursor[6] = bytes(self.region[start : start + 8 * words])[:n]
        return RingOutcome("open", position=q)

    def close(self) -> RingOutcome:
        if self._cursor is None:
            return _refused("BCIR_ERR_RING")
        if self._cursor[6] is None:
            self.copy()
        q, base, seq, pos, epoch, length, data = self._cursor
        self._cursor = None
        g = self.geometry
        region = self.region
        overwrite = g.policy == "overwrite"
        if _load(region, base + SLOT_SEQ) != seq:
            return self._lose(q, 1) if overwrite else _refused("BCIR_ERR_RING", q)
        if pos != q:
            ahead = (pos - q) & _M64
            if overwrite and ahead < _HALF and ahead % g.slot_count == 0:
                return self._lose(q, 1)
            return _refused("BCIR_ERR_RING", q)
        if length > g.capacity:
            return _refused("BCIR_ERR_RING", q)
        current, state, origin = self._owner_snapshot()
        if state > ATTACHED or epoch == 0 or epoch > current:
            return _refused("BCIR_ERR_RING", q)
        at_or_after_origin = (q - origin) & _M64 < _HALF
        if (at_or_after_origin and epoch != current) or (
            not at_or_after_origin and epoch < self.last_epoch
        ):
            self._commit(q + 1, self.lost, self.stale + 1, self.last_epoch)
            return RingOutcome("stale", position=q, count=1, epoch=epoch)
        self._commit(q + 1, self.lost, self.stale, epoch)
        return RingOutcome("delivered", position=q, count=1, epoch=epoch, payload=data)

    def consume(self) -> RingOutcome:
        opened = self.open()
        if opened.verdict != "open":
            return opened
        self.copy()
        return self.close()

    def beat(self) -> RingOutcome:
        if not self._held():
            return _refused("BCIR_ERR_STALE")
        _store(self.region, OFF_C_BEAT, _load(self.region, OFF_C_BEAT) + 1)
        return RingOutcome("ok", epoch=self.epoch)

    def accounting(self) -> RingAccounting:
        return committed_accounting(self.region, self.geometry)


def ring_fingerprint(region, g: RingGeometry | None = None) -> int:
    """CRC-32 of the ring's bytes -- the per-step region fingerprint the parity traces carry."""
    size = (g or decode_geometry(region)).region_size
    return zlib.crc32(bytes(region[:size])) & _M32
