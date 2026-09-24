"""The live SPSC ring, version zero (GEM+ roadmap G15, staged plan S3-B).

Before this slice the only shared telemetry ring was a quiescent snapshot: a producer bumped a head
and a reader copied whatever the slots held, with no tail, no publication protocol, no loss count
and no notion of a restarted producer. These tests pin the G15 gates and the laws under them:

    ring.loss.accounting   every published position is committed exactly once as delivered,
                           lost or stale; a consumer the producer laps is told how many it lost,
                           exactly, and is never handed a torn record
    sequence continuity    gaps, reorders and duplicates reported the way the frame ABI reports
                           them (one predicate, `SequenceTracker`)
    peer death             a producer or consumer that dies mid-operation is taken over by a
                           successor; a deposed peer refuses itself; nothing torn, nothing lost
                           without a count

The corpora are declared once, with the outcome the specification requires, in
`bcir/tests/ring_fixtures.py`. The C-rail tests build `runtime/c/test_ring.c` and skip without a
compiler (the quick tier hides one on purpose) or outside a source checkout (the wheel does not
ship runtime/c); the concurrent runs additionally need POSIX threads and processes.
"""

from __future__ import annotations

import os
import random
import re
import struct
import subprocess
import tempfile

from bcir.abi.control_abi import CONTROL_RECORD_MAX_BYTES
from bcir.gem.ring import (
    CONTROL_SLOT_MIN,
    GEOMETRY_SIZE,
    HEADER_SIZE,
    OFF_ACCT,
    OFF_C_BEAT,
    OFF_C_COMMIT,
    OFF_C_OWNER,
    OFF_HEAD,
    OFF_ORIGIN,
    OFF_P_BEAT,
    OFF_P_OWNER,
    OFF_REFUSED,
    OFF_TAIL,
    PAYLOADS,
    POLICIES,
    SLOT_HEADER,
    VERDICTS,
    RingConsumer,
    RingError,
    RingGeometry,
    RingProducer,
    committed_accounting,
    decode_geometry,
    encode_geometry,
    format_ring,
    ring_state,
)
from bcir.tests import control_fixtures as cf
from bcir.tests import ring_fixtures as rf

M64 = (1 << 64) - 1
_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ring(policy="backpressure", slot_count=4, slot_size=192, origin=0, payload="telemetry"):
    g = RingGeometry(policy, payload, slot_size, slot_count, ring_id=7, origin=origin)
    region = bytearray(g.region_size)
    format_ring(region, g)
    producer, consumer = RingProducer(region), RingConsumer(region)
    assert producer.attach().verdict == "ok" and consumer.attach().verdict == "ok"
    return g, region, producer, consumer


def _status(callback) -> str:
    try:
        callback()
    except RingError as exc:
        return exc.status
    return "BCIR_OK"


# --- the region -----------------------------------------------------------------------------------


def test_the_geometry_line_is_sealed_and_every_law_names_its_status():
    g = RingGeometry("overwrite", "telemetry", 128, 8, ring_id=9, origin=M64 - 3)
    region = bytearray(g.region_size)
    format_ring(region, g)
    assert decode_geometry(region) == g
    assert bytes(region[:64]) == encode_geometry(g)
    assert g.region_size == HEADER_SIZE + 8 * 128 and g.capacity == 128 - SLOT_HEADER
    for label, geometry in {
        "policy": RingGeometry("sometimes", "telemetry", 128, 8, 1),
        "payload": RingGeometry("overwrite", "video", 128, 8, 1),
        "slot size": RingGeometry("overwrite", "telemetry", 96, 8, 1),
        "slot size max": RingGeometry("overwrite", "telemetry", 65600, 8, 1),
        "slot count": RingGeometry("overwrite", "telemetry", 128, 6, 1),
        "slot count max": RingGeometry("overwrite", "telemetry", 128, 1 << 21, 1),
        "ring id": RingGeometry("overwrite", "telemetry", 128, 8, 0),
        "control is backpressure": RingGeometry("overwrite", "control", 256, 8, 1),
        "control slot holds the record bound": RingGeometry("backpressure", "control", 192, 8, 1),
    }.items():
        assert _status(lambda geometry=geometry: encode_geometry(geometry)) == "BCIR_ERR_RING", (
            label
        )
    # a buffer shorter than the ring is refused at format and at decode
    assert _status(lambda: format_ring(bytearray(g.region_size - 1), g)) == "BCIR_ERR_NOSPACE"
    assert _status(lambda: decode_geometry(region[: g.region_size - 1])) == "BCIR_ERR_TRUNCATED"
    assert _status(lambda: decode_geometry(region[: HEADER_SIZE - 1])) == "BCIR_ERR_TRUNCATED"


def test_each_shared_word_has_one_writer_and_its_own_line():
    """The layout's whole performance argument: what a side writes per operation never shares a
    cache line with what the other side polls (head one way, tail the other)."""

    def line(offset: int) -> int:
        return offset // 64

    producer_hot = {line(OFF_HEAD), line(OFF_P_BEAT)}  # written per publish
    consumer_hot = {line(OFF_TAIL)}  # polled by the producer
    consumer_private = {line(OFF_C_OWNER), line(OFF_C_BEAT), line(OFF_C_COMMIT)} | {
        line(o) for o in OFF_ACCT
    }
    assert producer_hot == {2} and consumer_hot == {4}
    assert line(OFF_REFUSED) == 3 and line(OFF_P_OWNER) == line(OFF_ORIGIN) == 1
    assert not consumer_private & (producer_hot | consumer_hot | {1, 3})
    assert HEADER_SIZE % 64 == 0 and HEADER_SIZE == 7 * 64


def test_the_header_constants_are_the_c_twin_s():
    """Read out of runtime/c/bcir_ring.h (L14), not mirrored."""
    path = os.path.join(rf.C_DIR, "bcir_ring.h")
    if not os.path.isfile(path):
        from bcir.tests.run_all import _is_source_checkout

        assert not _is_source_checkout(), "runtime/c/bcir_ring.h is missing from the checkout"
        return
    header = open(path, encoding="utf-8").read()
    offsets = dict(re.findall(r"#define BCIR_RING_OFF_([A-Z0-9_]+)\s+(\d+)u", header))
    assert {k: int(v) for k, v in offsets.items()} == {
        "P_OWNER": OFF_P_OWNER,
        "ORIGIN": OFF_ORIGIN,
        "HEAD": OFF_HEAD,
        "P_BEAT": OFF_P_BEAT,
        "REFUSED": OFF_REFUSED,
        "TAIL": OFF_TAIL,
        "C_OWNER": OFF_C_OWNER,
        "C_BEAT": OFF_C_BEAT,
        "C_COMMIT": OFF_C_COMMIT,
        "ACCT0": OFF_ACCT[0],
        "ACCT1": OFF_ACCT[1],
    }
    for name, value in (
        ("HEADER_SIZE", HEADER_SIZE),
        ("GEOMETRY_SIZE", GEOMETRY_SIZE),
        ("SLOT_HEADER", SLOT_HEADER),
        ("CONTROL_SLOT_MIN", CONTROL_SLOT_MIN),
    ):
        assert re.search(rf"^#define BCIR_RING_{name}\s+{value}u$", header, re.M), name
    verdicts = re.findall(r"BCIR_RING_([A-Z]+)\s*=\s*(\d+),?\s*/\*", header)
    codes = {n.lower(): int(c) for n, c in verdicts if n.lower() in VERDICTS}
    assert codes == {v: i for i, v in enumerate(VERDICTS)}
    policies = re.findall(r"BCIR_RING_(BACKPRESSURE|OVERWRITE)\s*=\s*(\d+)", header)
    assert [(n.lower(), int(c)) for n, c in policies] == [
        (p, i + 1) for i, p in enumerate(POLICIES)
    ]
    payloads = re.findall(r"BCIR_RING_(TELEMETRY|CONTROL)\s*=\s*(\d+)", header)
    assert [(n.lower(), int(c)) for n, c in payloads] == [
        (p, i + 1) for i, p in enumerate(PAYLOADS)
    ]
    statuses = open(os.path.join(rf.C_DIR, "bcir_runtime.h"), encoding="utf-8").read()
    for name, code in (("TELEMETRY", 19), ("RING", 20), ("FULL", 21), ("BUSY", 22)):
        assert re.search(rf"BCIR_ERR_{name} = {code}\b", statuses), name


def test_a_control_ring_carries_every_legal_control_record():
    """A control record is never dropped: a control ring is BACKPRESSURE (never overwritten) and its
    slot holds the record ABI's declared bound (never unsendable). CONTROL_SLOT_MIN is the smallest
    cache-line multiple that does -- held here to bcir.abi.control_abi, and to the C twin's own
    constant by the header test and by test_control_plane.c's _Static_assert."""
    assert CONTROL_SLOT_MIN % 64 == 0
    assert CONTROL_SLOT_MIN - SLOT_HEADER >= CONTROL_RECORD_MAX_BYTES
    assert CONTROL_SLOT_MIN - 64 - SLOT_HEADER < CONTROL_RECORD_MAX_BYTES
    g, region, p, c = _ring(payload="control", slot_size=CONTROL_SLOT_MIN, slot_count=2)
    record = bytes(range(CONTROL_RECORD_MAX_BYTES))
    assert p.publish(record).verdict == "ok"
    assert c.consume().payload == record
    assert _status(lambda: encode_geometry(RingGeometry("backpressure", "control", 192, 2, 1))) == (
        "BCIR_ERR_RING"
    )
    path = os.path.join(rf.C_DIR, "bcir_control_plane.h")
    if os.path.isfile(path):
        header = open(path, encoding="utf-8").read()
        bound = re.search(r"#define BCIR_CTL_RECORD_MAX\s+(\d+)u", header)
        assert bound and int(bound.group(1)) == CONTROL_RECORD_MAX_BYTES


# --- the laws -------------------------------------------------------------------------------------


def test_backpressure_refuses_and_counts_never_overwrites():
    g, region, p, c = _ring()
    for i in range(4):
        assert p.publish(bytes([i]) * 8).verdict == "ok"
    for _ in range(3):
        assert p.publish(b"x").status == "BCIR_ERR_FULL"
    assert ring_state(region)["refused"] == 3
    assert [c.consume().payload for _ in range(4)] == [bytes([i]) * 8 for i in range(4)]
    assert c.consume().verdict == "empty"
    a = committed_accounting(region, g)
    assert (a.delivered, a.lost, a.stale) == (4, 0, 0)


def test_overwrite_counts_what_it_lost_exactly_and_never_hands_over_a_torn_record():
    g, region, p, c = _ring("overwrite")
    for i in range(11):
        p.publish(bytes([i]) * 16)
    lost = c.consume()
    assert (lost.verdict, lost.position, lost.count) == ("lost", 0, 7)
    assert [c.consume().payload[0] for _ in range(4)] == [7, 8, 9, 10]
    # a copy the writer tears is LOST, never delivered
    p.publish(b"A" * 16)
    assert c.open().verdict == "open" and c.copy().verdict == "open"
    for i in range(4):
        p.publish(bytes([0x40 + i]) * 16)
    torn = c.close()
    assert (torn.verdict, torn.count) == ("lost", 1)
    a = committed_accounting(region, g)
    assert a.delivered + a.lost + a.stale == (a.tail - g.origin) & M64


def test_positions_wrap_past_two_to_the_sixty_four():
    g, region, p, c = _ring("backpressure", origin=M64 - 1)
    positions = []
    for i in range(6):
        positions.append(p.publish(bytes([i])).position)
        assert c.consume().payload == bytes([i])
    assert positions == [M64 - 1, M64, 0, 1, 2, 3]
    assert committed_accounting(region, g).delivered == 6


def test_a_takeover_needs_the_dead_epoch_and_retires_its_half_written_slot():
    g, region, p, c = _ring("overwrite")
    for i in range(4):
        p.publish(bytes([i]) * 8)
    assert p.open(8).verdict == "open"  # the producer dies here, its slot odd
    successor = RingProducer(region)
    assert successor.attach().status == "BCIR_ERR_BUSY"
    assert successor.attach(takeover=9).status == "BCIR_ERR_STALE"
    assert successor.attach(takeover=1).verdict == "ok"
    assert p.publish(b"late").status == "BCIR_ERR_STALE"  # the deposed producer refuses itself
    assert c.consume().verdict == "lost"  # the slot the dead producer tore
    assert [c.consume().payload for _ in range(3)] == [bytes([i]) * 8 for i in range(1, 4)]
    assert successor.publish(b"next").verdict == "ok"
    record = c.consume()
    assert (record.payload, record.epoch) == (b"next", 2)


def test_a_consumer_takeover_resumes_at_the_committed_tail():
    g, region, p, c = _ring()
    for i in range(3):
        p.publish(bytes([i]))
    assert c.consume().payload == b"\x00"
    assert c.open().verdict == "open"  # dies before committing position 1
    successor = RingConsumer(region)
    assert successor.attach(takeover=1).verdict == "ok"
    assert [successor.consume().payload for _ in range(2)] == [b"\x01", b"\x02"]
    assert c.consume().status == "BCIR_ERR_STALE"
    assert committed_accounting(region, g).delivered == 3


def test_the_accounting_is_double_buffered_so_a_death_never_tears_it():
    g, region, p, c = _ring()
    p.publish(b"a")
    commit_before = struct.unpack_from("<Q", region, OFF_C_COMMIT)[0]
    c.consume()
    commit_after = struct.unpack_from("<Q", region, OFF_C_COMMIT)[0]
    assert commit_after == commit_before + 1
    # scribble over the INACTIVE copy (a death mid-update): the committed accounting stands
    inactive = OFF_ACCT[(commit_after + 1) & 1]
    region[inactive : inactive + 32] = b"\xff" * 32
    a = committed_accounting(region, g)
    assert (a.tail, a.delivered) == (1, 1)
    assert struct.unpack_from("<Q", region, OFF_TAIL)[0] == 1


def test_random_scripts_keep_every_law_on_the_oracle():
    """A seeded exploration of the oracle against a shadow model: writes whose bytes name their
    position, reads, two-phase interleavings, deaths and takeovers. After every operation the
    accounting is exact and every delivered record is what was published at its position."""
    rng = random.Random(20260923)
    for trial in range(60):
        policy = rng.choice(("backpressure", "overwrite"))
        g, region, p, c = _ring(policy, slot_count=rng.choice((2, 4, 8)), slot_size=128,
                                origin=rng.choice((0, M64 - 5)))  # fmt: skip
        shadow: dict[int, bytes] = {}
        published = delivered = lost = 0
        for _step in range(120):
            op = rng.randrange(8)
            if op <= 2:
                head = struct.unpack_from("<Q", region, OFF_HEAD)[0]
                data = bytes([(head + trial) & 0xFF]) * rng.randrange(0, g.capacity + 1)
                o = p.publish(data)
                if o.verdict == "ok":
                    shadow[o.position], published = data, published + 1
            elif op <= 5:
                o = c.consume()
                if o.verdict == "delivered":
                    assert shadow[o.position] == o.payload
                    delivered += 1
                elif o.verdict == "lost":
                    assert policy == "overwrite"
                    lost += o.count
                else:
                    assert o.verdict == "empty", o
            elif op == 6:
                dead = p.epoch
                p = RingProducer(region)
                assert p.attach(takeover=dead).verdict == "ok"
            else:
                dead = c.epoch
                c = RingConsumer(region)
                assert c.attach(takeover=dead).verdict == "ok"
            a = committed_accounting(region, g)
            head = struct.unpack_from("<Q", region, OFF_HEAD)[0]
            assert (head - g.origin) & M64 == published
            assert (a.tail - g.origin) & M64 == a.delivered + a.lost + a.stale
            assert (a.delivered, a.lost, a.stale) == (delivered, lost, 0)
            if policy == "backpressure":
                assert (head - a.tail) & M64 <= g.slot_count


# --- the scenarios --------------------------------------------------------------------------------


def test_every_family_has_fixtures_and_every_verdict_and_status_a_witness():
    scenarios = rf.all_scenarios()
    assert {s.family for s in scenarios} == set(rf.FAMILIES) - {"control"} | {"control"}
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names))
    seen = set()
    for s in scenarios:
        for want in s.expect_ops.values():
            seen.add(want[:3])
    for verdict in ("ok", "open", "delivered", "empty", "lost", "stale", "refused"):
        assert any(w[0] == "R" and w[1] == verdict for w in seen), verdict
    for status in ("BCIR_ERR_FULL", "BCIR_ERR_BUSY", "BCIR_ERR_STALE", "BCIR_ERR_RING",
                   "BCIR_ERR_NOSPACE", "BCIR_ERR_MAGIC", "BCIR_ERR_VERSION", "BCIR_ERR_CRC",
                   "BCIR_ERR_RESERVED", "BCIR_ERR_TRUNCATED"):  # fmt: skip
        assert any(w[0] == "R" and w[2] == status for w in seen), status
    assert any(s.continuity is not None for s in scenarios)


def test_every_scenario_decides_as_the_specification_requires_on_the_oracle():
    for index, scenario in enumerate(rf.all_scenarios()):
        lines = rf.run_python(scenario, index)
        assert rf.expectations_hold(scenario, lines), scenario.name
        assert rf.accounting_count(scenario, lines) == 0, scenario.name
        assert rf.torn_count(scenario, lines) == 0, scenario.name
        assert rf.continuity_count(scenario, lines) == 0, scenario.name


def test_the_grading_fails_a_rail_that_lies():
    """A witness must hit the law it tests (L11): an altered trace fails the check that owns it."""
    scenario = next(s for s in rf.all_scenarios() if s.name == "ow-lap-counts-seven")
    lines = rf.run_python(scenario, 0)
    # one delivered payload altered -> torn
    # a trace line: index op R verdict status position count epoch length pcrc rcrc
    i = next(k for k, line in enumerate(lines) if " R delivered " in line)
    parts = lines[i].split()
    parts[9] = f"{int(parts[9], 16) ^ 1:08x}"
    torn = lines[:i] + [" ".join(parts)] + lines[i + 1 :]
    assert rf.torn_count(scenario, torn) == 1
    # an epoch altered is not torn -- it is a divergence the trace comparison owns
    parts = lines[i].split()
    parts[7] = "9"
    assert rf.torn_count(scenario, lines[:i] + [" ".join(parts)] + lines[i + 1 :]) == 0
    # the lap reported one short -> unaccounted
    j = next(k for k, line in enumerate(lines) if " R lost " in line)
    parts = lines[j].split()
    parts[6] = str(int(parts[6]) - 1)
    short = lines[:j] + [" ".join(parts)] + lines[j + 1 :]
    assert rf.accounting_count(scenario, short) == 1
    assert not rf.expectations_hold(scenario, short)
    # a truncated trace fails everything that reads it
    assert (
        rf.accounting_count(scenario, lines[:-1]) == 1 and rf.torn_count(scenario, lines[:-1]) == 1
    )


def test_control_records_decide_identically_through_a_live_control_ring():
    """The ring is a transport, never a decision: every G14 scenario's plane trace is unchanged
    when every submitted record first crosses a control ring (the Python rail)."""
    for scenario in cf.all_scenarios():
        _, _, direct = cf.run_python(scenario)
        _, _, carried = cf.run_python(scenario, transport=rf.control_ring_transport)
        assert carried == direct, scenario.name


# --- the C twin -----------------------------------------------------------------------------------


def test_the_two_rails_trace_every_scenario_identically():
    """Every verdict, status, position, count, epoch, payload and the region's CRC after every
    operation of every scenario, line for line -- and the C rail's decisions are the
    specification's."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        scenarios = rf.all_scenarios()
        traces = rf.run_c(exe, tmp, scenarios)
        for index, (scenario, trace) in enumerate(zip(scenarios, traces)):
            assert trace == rf.run_python(scenario, index), scenario.name
            assert rf.expectations_hold(scenario, trace), scenario.name


def test_the_c_api_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        code, out = rf.c_api(exe)
        assert code == 0 and out.strip() == "OK", out


_BOTH_RINGS_MAIN = r"""
static _Alignas(64) uint8_t live[BCIR_RING_HEADER_SIZE + 4u * 128u];
static _Alignas(64) int64_t v1[(BCIR_RING_HEADER + 4 * BCIR_RING_RECORD) / 8];

int main(void) {
  bcir_ring_geometry g = {0};
  g.policy = BCIR_RING_BACKPRESSURE;
  g.payload = BCIR_RING_TELEMETRY;
  g.slot_size = 128u;
  g.slot_count = 4u;
  g.ring_id = 1u;
  if (bcir_ring_format(live, sizeof live, &g) != BCIR_OK) return 1;
  bcir_ring_producer p;
  bcir_ring_consumer c;
  bcir_ring_producer_init(&p, live, sizeof live);
  bcir_ring_consumer_init(&c, live, sizeof live);
  if (bcir_ring_producer_attach(&p, 0).verdict != BCIR_RING_OK) return 2;
  if (bcir_ring_consumer_attach(&c, 0).verdict != BCIR_RING_OK) return 3;
  const uint8_t record[5] = {'l', 'i', 'v', 'e', '!'};
  uint8_t out[128];
  if (bcir_ring_publish(&p, record, sizeof record).verdict != BCIR_RING_OK) return 4;
  bcir_ring_outcome o = bcir_ring_consume(&c, out, sizeof out);
  if (o.verdict != BCIR_RING_DELIVERED || o.length != sizeof record) return 5;
  bcir_ring_init(v1, 4u);
  bcir_ring_write(v1, 7, 1, 2, 3, 4, 5, 6);
  return v1[0] == 1 && v1[BCIR_RING_HEADER / 8] == 7 ? 0 : 6;
}
"""


def test_the_live_ring_and_the_v1_ring_emitter_coexist_in_one_program():
    """The v1 shared-ring emitter (`lower.memory_model.emit_ring_header_c`, default prefix
    `bcir_ring`) emits `BCIR_RING_HEADER`, `BCIR_RING_RECORD`, `bcir_ring_init` and
    `bcir_ring_write`. The live ring's C API claims none of those names, so a generated kernel that
    emits v1 telemetry can include and link the live ring: one translation unit here includes
    both, links both, and runs both."""
    from bcir.lower.memory_model import emit_ring_header_c
    from bcir.tests.run_all import _is_source_checkout

    cc = rf.compiler()
    if cc is None or not _is_source_checkout():
        return  # the quick tier hides the compiler; the wheel ships no runtime/c
    with tempfile.TemporaryDirectory() as tmp:
        source = os.path.join(tmp, "both_rings.c")
        with open(source, "w", encoding="utf-8") as fh:
            fh.write('#include "bcir_ring.h"\n' + emit_ring_header_c() + _BOTH_RINGS_MAIN)
        exe = os.path.join(tmp, "both_rings" + (".exe" if os.name == "nt" else ""))
        sources = [os.path.join(rf.C_DIR, name) for name in ("bcir_ring.c", "bcir_runtime.c")]
        build = subprocess.run(
            [
                cc,
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-I",
                rf.C_DIR,
                source,
                *sources,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert build.returncode == 0, build.stderr[-2000:]
        run = subprocess.run([exe], capture_output=True, timeout=60)
        assert run.returncode == 0, run.returncode


def test_control_records_decide_identically_through_the_c_control_ring():
    with tempfile.TemporaryDirectory() as tmp:
        exe = cf.build_harness(tmp)
        if exe is None:
            return
        scenarios = cf.all_scenarios()
        assert cf.run_c(exe, tmp, scenarios, via_ring=True) == cf.run_c(exe, tmp, scenarios)


def test_concurrent_producers_and_consumers_keep_every_law():
    """Threads, and processes with a peer SIGKILLed mid-stream and taken over: nothing torn,
    nothing unaccounted, continuity equal to the ring's own loss count. UNAVAILABLE (and owned by
    CI's Linux jobs) on a host with no POSIX threads/processes."""
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        runs = rf.c_concurrent(exe, records=20000)
        if os.name == "posix":
            assert len(runs) == len(rf.STRESS_RUNS)
        for name, violations, line in runs:
            assert violations == 0, (name, line)


def test_every_g15_row_is_zero_on_both_rails():
    with tempfile.TemporaryDirectory() as tmp:
        exe = rf.build_harness(tmp)
        if exe is None:
            return
        rows = rf.measure(exe, tmp)
        expected = set(rf.ROWS) - (set() if os.name == "posix" else {"ring.concurrent.violations"})
        assert set(rows) == expected, rows
        assert all(value == 0 for value in rows.values()), rows


# --- the gates are wired to the sources they grade ------------------------------------------------


def test_the_gate_and_the_harnesses_link_the_same_sources():
    """check_runtime.sh's ring and control link lines and the Python fixtures' source lists are
    two lists each, and nothing but this test makes them agree (the #719 wiring trap)."""
    from bcir.tests.run_all import _is_source_checkout

    gate_path = os.path.join(_ROOT, "tools", "c", "check_runtime.sh")
    if not _is_source_checkout():
        return  # the wheel ships neither tools/ nor runtime/c
    gate = open(gate_path, encoding="utf-8").read()
    ring_line = re.search(r"ring_sources=\(([^)]*)\)", gate).group(1)
    ring_linked = set(re.findall(r"(\w+\.c)", ring_line)) | {"bcir_ring.c"}
    assert ring_linked == set(rf.C_SOURCES), (ring_linked, rf.C_SOURCES)
    control_line = re.search(r"ctl_sources=\(([^)]*)\)", gate).group(1)
    control_linked = set(re.findall(r"(\w+\.c)", control_line)) | {"bcir_control_plane.c"}
    assert control_linked == set(cf.C_SOURCES), (control_linked, cf.C_SOURCES)
    fuzz = open(os.path.join(_ROOT, "tools", "c", "fuzz_streampack.sh"), encoding="utf-8").read()
    target = re.search(r'add_target ring "[^"]*" "[^"]*" \\\n\s*(.*)', fuzz).group(1)
    assert set(re.findall(r"(\w+\.c)", target)) == {
        "fuzz_ring.c",
        "bcir_ring.c",
        "bcir_telemetry_envelope.c",
        "bcir_runtime.c",
    }
