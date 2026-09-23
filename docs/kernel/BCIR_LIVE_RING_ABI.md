# BCIR live SPSC ring ("BRNG") — version zero (experimental, normative for v0)

The live ring is **the transport under the telemetry and control planes** (GEM+ roadmap G15,
staged plan S3-B): one single-producer / single-consumer ring in a region two peers share — two
threads, two processes over a `MAP_SHARED` mapping, a driver and its host — with head and tail,
acquire/release publication, a per-slot sequence, a declared overwrite or backpressure policy and
exact loss accounting. It carries [TelemetryEnvelopeV0](TELEMETRY_ENVELOPE_ABI.md) records first
and [ControlRecordV1](BCIR_CONTROL_PLANE_ABI.md) records second.

Before it, the only shared telemetry ring was the v1 snapshot ring
(`bcir.telemetry.parse_shared_ring` and the producer `lower.memory_model.emit_ring_header_c`
emits): a producer bumps a head and a reader copies whatever the slots hold — no tail, no
publication protocol, no loss count, no producer identity. Measured on the parent tree
(83c6c015): eight records published into a four-slot ring hand a reader that is a lap behind the
four survivors and **no loss count**, and a slot read while the producer is three fields into a
rewrite comes back with the fields of **two different records and no error**. That ring stays
what it is — a frozen, bounds-checked quiescent snapshot — and this one is the live transport.

The executable oracle is [`bcir/gem/ring.py`](../../bcir/gem/ring.py): it executes the same
algorithm over the same bytes one operation at a time, so a scripted interleaving (a consumer
mid-copy while the producer laps it, a producer that dies between two stores, a takeover, a
zombie) leaves the region byte-identical on both rails after every step. The production rail is
the freestanding C twin [`runtime/c/bcir_ring.h`](../../runtime/c/bcir_ring.h) /
`bcir_ring.c` (C11 atomics, no heap, no libc). `runtime/c/test_ring.c` is the harness (scripts,
concurrency, processes, the benchmark) and `runtime/c/fuzz_ring.c` the libFuzzer target.

**Version zero.** The driver roadmap keeps experimental structures at version zero until the UART
and virtio-blk traces prove both device classes
([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) §7.1): this layout carries **no
compatibility promise**, and a change is a version bump, never a reinterpretation. It is not the
BCIR UAPI and not the native-IPC v1 submission/completion format.

## Conventions

- **Host:** little-endian (the C twin refuses to compile on a big-endian target) with lock-free
  64-bit atomics (`ATOMIC_LLONG_LOCK_FREE == 2`, a static assertion): the words the peers share
  are read and written with C11 atomics, so the region must be **8-byte aligned** (the C twin
  refuses a misaligned or NULL region with `BCIR_ERR_RING`).
- **Positions** are unbounded `u64` counters: slot index = `position & (slot_count - 1)`, and every
  comparison is modular — a difference of 2⁶³ or more means "behind", never "far ahead" (a ring
  formatted at `origin = 2⁶⁴ − 3` wraps through zero; a fixture proves it).
- **CRC:** CRC-32 (zlib; `bcir_crc32` on the C rail) over the geometry line's first 60 bytes.
- **Statuses** are the runtime's `bcir_status` (Python: `RingError.status`, the same names). S3-B
  appended `BCIR_ERR_RING` (20: a protocol violation or a malformed region), `BCIR_ERR_FULL` (21: a
  backpressure ring refused a record) and `BCIR_ERR_BUSY` (22: an owner word is held, or an
  attach is in flight), after `BCIR_ERR_TELEMETRY` (19, the envelope's).

## The region

Seven 64-byte header lines, then the slots. **Each line has one writer**, and a side's
per-operation writes never share a line with what the other side polls — the head goes one way,
the tail progress word the other (the layout test pins it).

| Line | Bytes | Writer | Words (`*` = shared, accessed atomically) |
|---|---|---|---|
| 0 geometry | 0–63 | `format` only (then immutable) | below; CRC-sealed |
| 1 producer ownership | 64–127 | producer attach/detach | `p_owner*` @64 = `epoch << 32 \| state`; `epoch_origin*` @72 |
| 2 producer progress | 128–191 | producer, per publish | `head*` @128; `p_heartbeat*` @136 |
| 3 producer counters | 192–255 | producer | `refused*` @192 |
| 4 consumer progress | 256–319 | consumer, per commit | `tail*` @256 — the word the producer polls |
| 5 consumer | 320–383 | consumer | `c_owner*` @320; `c_heartbeat*` @328; `c_commit*` @336; `acct[0]` @352 |
| 6 consumer | 384–447 | consumer | `acct[1]` @384 |
| slots | 448 + i·`slot_size` | producer (repair: a taking-over producer) | per slot below |

**Geometry line** (little-endian, sealed by `crc32`):

| Offset | Field | Law |
|---:|---|---|
| 0 | `magic[4]` = `"BRNG"` | |
| 4 | `version:u16` = 0 | |
| 6 | `policy:u8` | 1 BACKPRESSURE, 2 OVERWRITE |
| 7 | `payload:u8` | 1 TELEMETRY, 2 CONTROL |
| 8 | `slot_size:u32` | a multiple of 64 in [64, 65536] |
| 12 | `slot_count:u32` | a power of two in [2, 2²⁰] |
| 16 | `ring_id:u64` | nonzero |
| 24 | `region_size:u64` | = 448 + `slot_count` · `slot_size` |
| 32 | `origin:u64` | the position of the first slot ever published |
| 40 | `reserved[20]` | zero |
| 60 | `crc32:u32` | CRC-32 of bytes 0–59 |

**Owner words** (`p_owner`, `c_owner`): `epoch << 32 | state`, state 0 DETACHED, 1 ATTACHING (a
producer claiming the ring while it writes its origin), 2 ATTACHED. Epochs start at 0 and
increase by one per attach; epoch 0 is never live.

**Accounting record** (`acct[0]`, `acct[1]`, 32 bytes each): `tail:u64`, `lost:u64`, `stale:u64`,
`last_epoch:u32`, `reserved:u32`. The valid record is `acct[c_commit & 1]` — double-buffered, so a
consumer that dies mid-update never leaves torn accounting behind. `delivered` is derived:
`tail − origin − lost − stale`.

**Slot** (`slot_size` bytes): `seq*:u64` @0 — a per-slot sequence, **odd while being written**;
`pos*:u64` @8 — the position the slot holds; `epoch | length << 32*` @16 — the producer epoch that
wrote it and the payload length; payload @24 (`capacity = slot_size − 24` bytes, word-copied).

## Geometry laws (in this order, both rails)

`decode_geometry` / `bcir_ring_decode_geometry` apply these to a region of `len` bytes, and every
attach applies them first:

1. `len < 448` → `TRUNCATED`
2. magic → `MAGIC`; 3. version → `VERSION`; 4. the CRC → `CRC`; 5. nonzero reserved bytes →
   `RESERVED`
6. the value laws → `RING`: policy and payload codes; `slot_size`; `slot_count`; `ring_id ≠ 0`;
   **a CONTROL ring is BACKPRESSURE** (a control record is never overwritten); **a CONTROL ring's
   slot is at least 256 bytes** (`CONTROL_SLOT_MIN`: its payload holds the control ABI's declared
   bound, `CONTROL_RECORD_MAX_BYTES` = 192, so a legal control record is never unsendable)
7. `region_size` ≠ 448 + `slot_count` · `slot_size` → `RING`
8. `region_size > len` → `TRUNCATED`

`format_ring` / `bcir_ring_format` validate the geometry (`RING`), refuse a buffer shorter than
`region_size` (`NOSPACE`), zero the region, write the sealed geometry line and set `head`,
`epoch_origin`, `tail` and `acct[0].tail` to `origin`.

## Ownership: attach, detach, takeover

An endpoint is **held** while the owner word equals `(its epoch, ATTACHED)`. Every open — and
every beat and detach — re-reads the owner word (acquire) first, so a deposed endpoint refuses
itself with `STALE` at its next record. (A fill, copy or close continues an operation its open
already admitted.)

**Attach** (`attach(takeover)`), both ends, in order: the region (C: NULL or misaligned → `RING`);
the geometry laws; the owner word's state > 2, or ATTACHING → `RING`; `takeover == 0` requires
DETACHED (else `BUSY`); `takeover == E` requires `(E, ATTACHED)` (else `STALE`) — the embedding
names the epoch it has proved dead; `E ≥ 0xFFFFFFFF` → `RING` (epochs exhausted); a lost
compare-and-swap → `BUSY`.

- **Producer:** CAS the owner word to `(E, ATTACHING)`; release fence; load `head`; on a takeover,
  **retire the dead producer's half-written slot** at `head` (if its `seq` is odd: `pos = head`,
  `epoch | length = 0`, `seq + 1` with release — the slot is even and unpublished, and the
  successor's first record overwrites it); `epoch_origin = head`; release the owner word as
  `(E + 1, ATTACHED)`; cache the tail progress word.
- **Consumer:** read the committed accounting (the active copy) and `head`; a committed tail ahead
  of `head` → `RING`; CAS the owner word to `(E + 1, ATTACHED)`; **republish the tail progress
  word** (a predecessor may have died between its commit and its publication); resume at the
  committed tail.

**Detach:** a held endpoint with no open two-phase operation (else `RING`) releases
`(E, DETACHED)`. **Heartbeats** (`p_heartbeat`, `c_heartbeat`) are counters an embedding may watch;
the ring never infers death from them.

## Publishing (the producer)

`publish(bytes)` = `write_open(length)` + `write_fill(0, bytes)` + `write_close()`; the two-phase
steps let a producer write a record in pieces.

- **`write_open(length)`:** held (else `STALE`); already open → `RING`; `length > capacity` →
  `NOSPACE`. BACKPRESSURE: `depth = head − cached_tail`; when `depth ≥ slot_count` the cached tail
  is refreshed from the tail progress word (acquire); `depth ≥ 2⁶³` or `> slot_count` → `RING` (a
  tail that ran ahead of the head); `depth == slot_count` → `refused += 1` and **`FULL`** — the
  record is refused and counted, never overwrites an unconsumed slot. The slot's `seq` odd → `RING`
  (a second writer). Then `seq + 1` (relaxed), a **release fence**, `pos = head` and
  `epoch | length << 32` (relaxed) → `OPEN`. OVERWRITE never refuses for depth.
- **`write_fill(offset, bytes)`:** `offset % 8 == 0` and `offset + n ≤ length` (else `NOSPACE`);
  relaxed word stores, the last word zero-padded. Bytes of `[0, length)` a producer does not fill
  hold whatever the slot held; `publish` fills them all.
- **`write_close()`:** `seq + 2` (**release**), `head + 1` (**release**), `p_heartbeat + 1` → `OK`
  at the published position.

## Consuming (the consumer)

`consume()` = `read_open()` + `read_copy(out)` + `read_close()`, one position per call, O(1),
non-blocking.

- **`read_open()`:** held (else `STALE`); a cursor already open → `RING`. `q` = the committed tail.
  The head is reloaded (acquire) always under OVERWRITE, and under BACKPRESSURE only when the
  cached head is exhausted. `depth = head − q`: 0 → **`EMPTY`**; `≥ 2⁶³` → `RING`;
  `> slot_count`: BACKPRESSURE → `RING` (a producer that overran its tail), OVERWRITE →
  **`LOST(depth − slot_count)`**, committed at once. The slot's `seq` (acquire) odd: BACKPRESSURE →
  `RING`, OVERWRITE → `LOST(1)`. Else load `pos` and `epoch | length` → `OPEN`.
- **`read_copy(out, cap)`:** relaxed word loads of `min(length, capacity, cap)` bytes. The copy is
  meaningful only if `read_close` delivers it.
- **`read_close()`:** an **acquire fence** and a reload of `seq` — **the seqlock**: a changed
  sequence means a writer raced the copy: OVERWRITE → `LOST(1)`, BACKPRESSURE → `RING`. `pos ≠ q`:
  OVERWRITE with `pos − q` a nonzero multiple of `slot_count` below 2⁶³ (a later lap's record) →
  `LOST(1)`, anything else → `RING`. `length > capacity` → `RING`. Then the producer epoch law:
  the owner word, `epoch_origin` and the owner word again around an acquire fence (64 attempts,
  else `BUSY`: an attach is in flight, read again); state > 2, a record epoch of 0 or beyond the
  owner's → `RING`; **at or after the origin** a record must carry the owner's epoch, **before
  it** an epoch no older than the last one delivered — otherwise **`STALE`** (`stale + 1`,
  committed). C only: a caller buffer shorter than the record → `NOSPACE`, **not consumed** (read
  again with room). Otherwise the position is committed → **`DELIVERED`** (position, length,
  epoch), and `last_epoch` becomes the record's epoch.

**Committing** (every `LOST`, `STALE` and `DELIVERED`): a release fence; the inactive accounting
copy is written (relaxed); `c_commit + 1` (release); `c_heartbeat + 1`; the tail progress word
(release) — published **after** the commit, so it never runs ahead of the accounting. A peer reads
a consistent snapshot (`committed_accounting` / `bcir_ring_accounting_of`) the same way a seqlock
reader does: `c_commit` (acquire), the copy, an acquire fence, `c_commit` again (64 attempts, else
`BUSY`).

## The accounting law

Every position the producer publishes is committed by the consumer **exactly once**, as delivered,
lost or stale:

```
delivered     := tail − origin − lost − stale    (derived from the committed record)
head − origin == delivered + lost + stale        (at quiescence: every published position
                                                  committed once, in exactly one class)
FULL refusals == refused                         (BACKPRESSURE; the producer's own count)
```

The gate checks more than the arithmetic: each class's count must equal what the rail's own
verdicts reported, operation by operation, and the declared numbers of the fixture.

A BACKPRESSURE ring never loses a record (`lost` stays 0) and never holds more than `slot_count`;
an OVERWRITE consumer that falls behind is told **exactly** how many records it lost and is never
handed a torn one. The G15 gate `ring.loss.accounting` is this law, counted by
`ring.loss.unaccounted`.

## Memory ordering (the C11 argument)

- **Publication.** The producer marks the slot odd (relaxed), issues a release fence, writes `pos`,
  `epoch | length` and the payload (relaxed), marks the slot even with a release store and releases
  the head. A consumer that acquires the head acquires the slot; one that raced a later writer sees
  a different sequence after its acquire fence (Boehm's seqlock), so a torn record is never
  delivered.
- **Payload words** are copied with relaxed atomic loads and stores, so the overwrite race a
  seqlock tolerates is not a data race in the C11 sense — ThreadSanitizer, which models the
  atomics, reports none, and reports one when the stores are made plain (the gate injects exactly
  that).
- **Accounting** is double-buffered behind `c_commit` with the same fence pattern; **ownership** is
  a compare-and-swap (acq_rel) and a release of the new owner word, read by the consumer as an
  (owner, origin, owner) snapshot around an acquire fence.

## Peer death and the SPSC contract

The ring's contract is **one live producer and one live consumer**.

- A **takeover** (attach with the dead peer's epoch) requires the embedding to have **proved the
  peer dead** — a reaped process, a watchdog reset. The ring does not guess from heartbeats.
- A **deposed peer that is still running** refuses itself at its next open (`STALE`), and the
  consumer refuses a record stamped with a deposed epoch at or after the takeover position
  (`STALE`, counted).
- A **producer that died mid-write** leaves an odd slot; its successor retires it. In OVERWRITE
  the record whose slot the dead producer had begun to overwrite is reported to the consumer as
  `LOST(1)` — counted, never delivered torn.
- A **consumer that died mid-commit** leaves the active accounting copy intact; its successor
  resumes at the committed tail and republishes the progress word.
- **Two live producers are outside SPSC.** The ring then reports `BCIR_ERR_RING` wherever it can
  see the violation (an odd slot at open, a position or sequence it cannot vouch for) rather than
  deliver what it cannot vouch for; it does not claim to detect every interleaving of two writers.

## Payloads

- **TELEMETRY** rings carry TelemetryEnvelopeV0 records: a 192-byte slot (capacity 168) carries
  either kind (76 or 124 bytes). The ring never parses a payload; the intake does
  ([`TELEMETRY_ENVELOPE_ABI.md`](TELEMETRY_ENVELOPE_ABI.md), "Intake").
- **CONTROL** rings carry ControlRecordV1 records and are always BACKPRESSURE with slots of at
  least 256 bytes (above). The ring is a transport, never a decision: all 52 G14 control scenarios
  produce the identical plane trace through a live control ring as submitted directly, on both
  rails (`ring.control.divergent`; `test_control_plane --via-ring`).

## Gates

| Gate | Where |
|---|---|
| every scripted scenario decides as specified, identically on both rails — verdicts, statuses, positions, counts, epochs, payloads and the region's CRC after every operation | `ring.traces.divergent`; `bcir/tests/test_live_ring.py` |
| committed accounting exactly what the verdicts reported, and the declared numbers | `ring.loss.unaccounted` |
| nothing delivered that is not byte-identical to what was published at its position | `ring.torn.delivered` |
| one malformed variant per geometry law, owner-word and slot corruption, refused with the declared status | `ring.malformed.accepted` |
| deposed producers and consumers, and records stamped with deposed epochs, refused | `ring.stale.accepted` |
| the 52 G14 scenarios through a live control ring, identical to the direct plane | `ring.control.divergent` |
| threads, and processes with a peer SIGKILLed and taken over: nothing torn, nothing unaccounted, continuity equal to the ring's loss count | `ring.concurrent.violations`; `test_ring --stress`, `--procs` |
| no data race under ThreadSanitizer (both policies), and the race reported when atomics are made plain | `tools/c/check_runtime.sh` — required where CI installs the TSan runtime (`BCIR_REQUIRE_TSAN=1`, the x86 C runtime job); an explicit, reported skip on a runner without it (the aarch64 job, whose concurrent rows still run natively) |
| the gate fires: the seqlock re-check removed turns rows red | `tools/c/check_runtime.sh` |
| every law can fire: 22 injected defects (the ring's laws on both rails, the envelope's, continuity, the stale-generation and unknown-signal laws, the table's bytes, the control transport), each caught by its own row | `tools/testing/faults/ring.json`, run by `tools/testing/red_sweep.py` over `tools/c/check_ring.py` |
| answers independent of the optimiser (-O0 == -O3 == the oracle) | `tools/c/check_runtime.sh` |
| the API's fail-closed laws no script reaches (NULL, misaligned, short, unattached, double-open) | `test_ring --api` |
| a hostile region (the geometry valid, every other byte from the input) and an honest scripted region with its invariants asserted after every operation, under ASan/UBSan | `runtime/c/fuzz_ring.c` in `tools/c/fuzz_streampack.sh` |
| throughput against a `memcpy` floor of the same bytes | `ring.throughput` (ratio, `host_dependent`) |

The corpora and their expected outcomes live in one place,
[`bcir/tests/ring_fixtures.py`](../../bcir/tests/ring_fixtures.py) (saturation, wrap, torn,
restart, stale, continuity, malformed and control families), and the rows are measured by one
function the harness, the C gate and the tests share (`tools/perf/gemplus_baseline.py --group
ring`).

**Throughput.** `ring.throughput` is ring time per record over `memcpy` time per record, the same
124-byte envelopes, two threads against one (`test_ring --bench`). On the reference host of this
program (4 vCPU, virtualized, clang 18) the first measurement was **26×** (the baseline). On that
host the ratio is dominated by where the two threads land and when: unpinned runs read 24–44×,
and runs pinned to each vCPU pair read ~13–49× within one hour (82–313 ns per record, `memcpy`
steady at ~6.4 ns), while a minimal unchecked Lamport queue moving the same bytes read ~1.1–6×
under the same pinning. The ring does cost more than an unchecked queue — the seqlock's two
sequence writes and reads, the per-record accounting commit, the ownership checks — but this host
cannot attribute that cost: back-to-back unpinned A/Bs early in the slice favoured the
one-writer-per-line layout (~127–153 against ~190–217 ns per record), and every stripped variant
since (no heartbeat, no held test, no owner snapshot, no accounting commit, whole-word payload
copies) stayed inside the placement noise. The row is host-dependent and INDICATIVE; a faithful
A/B needs pinned threads on isolated cores of an attested host (the G7 discipline).

## Relation to the other surfaces

- **The v1 snapshot ring** (`parse_shared_ring`, `emit_ring_header_c`) is unchanged and frozen.
  The live ring's C API claims none of the names the v1 emitter emits by default
  (`BCIR_RING_HEADER`, `BCIR_RING_RECORD`, `bcir_ring_init`, `bcir_ring_write`), so a generated
  kernel that emits v1 telemetry can also include and link the live ring; a test compiles, links
  and runs both in one program.
- **BTLM v1** ([`TELEMETRY_FRAME_ABI.md`](TELEMETRY_FRAME_ABI.md)) stays the frozen UART frame.
  The envelope, not BTLM, is what this ring carries; both report continuity with the same
  predicate (`SequenceTracker`).
- **RuntimeChannel v1** is the direct execution hook table; this ring is the telemetry/control
  transport beside it, not a submission queue.

## Versioning

Version zero: the layout, the laws and the statuses above may change, and a change is a version
bump (a reader refuses a version it does not implement), never a reinterpretation of these bytes.
The field set is revised from the UART and virtio-blk traces before any v1 freeze.

## Not claimed

- **No MPSC, no MPMC.** One producer, one consumer; a second live producer is a contract
  violation the ring reports where it can see it.
- **No death detection.** Takeover needs the embedding's proof; heartbeats are for its watchdog.
- **No blocking or wake-up.** Every operation returns at once; waiting (spin, futex, interrupt) is
  the embedding's.
- **No authentication.** The ring guarantees bytes, order and accounting, not who wrote them: a
  TELEMETRY payload is evidence the intake checks, and a CONTROL payload carries its own MAC.
- **No cross-host or big-endian use**, and no promise beyond version zero.
- **No performance certificate.** The throughput row is host-dependent and measured on a
  virtualized host with no PMU.
