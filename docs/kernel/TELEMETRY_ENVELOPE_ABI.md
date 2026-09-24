# TelemetryEnvelopeV0, the generated signal table and the host intake — version zero

This is the record half of the **version-zero triple** the driver roadmap requires before any
driver enters D2 ([`BCIR_DRIVER_KERNEL_ROADMAP.md`](BCIR_DRIVER_KERNEL_ROADMAP.md) §4.3; GEM+
roadmap G15, staged plan S3-B): (1) the **generated fixed-width signal definition table** with an
ID-range policy, (2) the identity-carrying **telemetry envelope**, and — in
[`BCIR_LIVE_RING_ABI.md`](BCIR_LIVE_RING_ABI.md) — (3) the live SPSC ring that carries it.

[BTLM v1](TELEMETRY_FRAME_ABI.md) stays frozen byte for byte. It carries records with no source,
session, generation, clock or loss identity, so a consumer cannot tell a restarted producer from
a reordered one, a producer's drop from a transport's, or a measurement of the live artifact from
one of the artifact it replaced. The envelope is a **new** versioned frame that carries all of
it, not a reinterpretation of BTLM's reserved bytes.

| Piece | Oracle | C twin |
|---|---|---|
| the envelope codec | [`bcir/abi/telemetry_envelope.py`](../../bcir/abi/telemetry_envelope.py) | `bcir_tev_decode` / `bcir_tev_encode` ([`runtime/c/bcir_telemetry_envelope.h`](../../runtime/c/bcir_telemetry_envelope.h)) |
| the signal table | [`bcir/signal_table.py`](../../bcir/signal_table.py) (the generator) | [`runtime/c/bcir_signal_table.h`](../../runtime/c/bcir_signal_table.h) (**generated**), `bcir_signal_find` |
| the continuity predicate | `bcir.telemetry.SequenceTracker` | `bcir_seq_observe` |
| the intake | [`bcir/telemetry_intake.py`](../../bcir/telemetry_intake.py) | `bcir_tev_intake_admit` |

The rails agree record for record (Python encode → C decode → the same fields → the same bytes
re-encoded) and decision for decision (every intake step's verdict, reason, status and
classification in the ring scenarios' traces). All three pieces are **version zero**: no
compatibility promise until UART and virtio-blk traces revise them; a change is a version bump,
never a reinterpretation.

## The envelope

Little-endian, every field fixed-width, one CRC-sealed record:

```
TelemetryEnvelopeV0 := header(64) || payload(kind) || crc32(4)      crc32 = CRC-32(header || payload)
```

**Header** (64 bytes, one cache line; every `u64` at an 8-aligned offset):

| Offset | Field | Meaning |
|---:|---|---|
| 0 | `magic[4]` = `"BTEV"` | |
| 4 | `version:u16` = 0 | a v0 reader reads v0 |
| 6 | `flags:u16` | bit 0 **REQUIRED** (the consumer must understand `signal`); every other bit reserved |
| 8 | `kind:u8` | 1 `sample` (one reading of one signal), 2 `datadna` (the per-claim execution record) |
| 9 | `schema:u8` = 0 | the payload schema of the kind |
| 10 | `clock:u8` | 0 none, 1 monotonic, 2 realtime, 3 boot, 4 cycles |
| 11 | `unit:u8` | 0 none, 1 ns, 2 µs, 3 cycles |
| 12 | `size:u16` | the whole record's length — fixed per kind |
| 14 | `reserved:u16` = 0 | |
| 16 | `source:u64` | the producer instance (nonzero) |
| 24 | `session:u64` | changes at every producer (re)start (nonzero) — the restart boundary BTLM leaves out of band |
| 32 | `generation:u32` | the artifact generation the measurement was taken under; 0 = unbound (a host sensor) |
| 36 | `signal:u32` | a stable ID from the signal table (samples only) |
| 40 | `seq:u32` | the `(source, session)` stream's sequence, modulo 2³² |
| 44 | `lost:u32` | records of this stream the **producer** dropped immediately before this one |
| 48 | `timestamp:u64` | in `unit` of `clock`; 0 when `clock` is none |
| 56 | `reserved:u64` = 0 | |

**Payload** (fixed per kind): `sample` — `value:i64` (8 bytes; a 76-byte record); `datadna` — the
frozen 56-byte `<7q>` DataDNA record, BTLM's record verbatim (a 124-byte record).

`lost` is what lets a consumer tell a producer-side drop from a transport loss: the ring's own
accounting counts what the transport lost, and the envelope's `lost` what the producer never
sent.

## The wire laws (one order, both rails, encoder and decoder)

`decode_envelope` and `bcir_tev_decode` apply these in this order, so both rails name the same
**first** violation; `encode_envelope` refuses exactly what the decoder refuses (the field laws
are one predicate, `validate_envelope`, applied by both).

1. fewer than 64 bytes → `TRUNCATED`
2. magic → `MAGIC`; 3. version → `VERSION`; 4. a flag bit other than REQUIRED → `RESERVED`
5. an unknown kind → `TELEMETRY`; 6. `size` ≠ the kind's size → `TELEMETRY`
7. fewer bytes than `size` → `TRUNCATED`; more → `TRAILING`
8. the CRC → `CRC`
9. a nonzero reserved field → `RESERVED`
10. the field laws, all `TELEMETRY`: `schema ≠ 0`; an unknown clock or unit code; a clock that does
    not count in that unit (none ↔ none; monotonic, realtime, boot ↔ ns or µs; cycles ↔ cycles);
    a timestamp with no clock; `source = 0`; `session = 0`; a `sample` whose signal is 0 or the
    reserved `0xFFFFFFFF`; a `datadna` record that names a signal, is marked REQUIRED, is unbound
    (generation 0) or carries a sample value.

**One spelling (Class A).** Every reserved byte and bit has exactly one legal value, the size is
the kind's, and REQUIRED is one bit: a record any decoder accepts re-encodes to its own bytes (a
test flips random bits of every corpus record — 64 mutants each — re-seals the CRC, and asserts
it of every mutant a decoder accepts; the fuzz target asserts the C re-encode is identical to its
input).

## The signal definition table

`bcir.signal_registry` stays the normative taxonomy oracle (names, units, metric semantics,
stable IDs). The table is its **projection as bytes** — what a resident driver, which cannot
import Python, uses to name a signal by its `u32` ID.

**Row** (64 bytes, little-endian; one per built-in definition, in ID order — today IDs 1–15):

| Offset | Field | Codes |
|---:|---|---|
| 0 | `id:u32` | the registry's signal ID (BCIR range only) |
| 4 | `unit:u8` | 0 none, 1 percent, 2 millicelsius, 3 microjoule, 4 milliwatt, 5 microwatt, 6 kHz, 7 bytes, 8 bytes/s, 9 bitmask, 10 count, 11 ratio (milli-units) |
| 5 | `kind:u8` | 0 gauge, 1 counter |
| 6 | `temporality:u8` | 0 unspecified, 1 delta, 2 cumulative |
| 7 | `flags:u8` | bit 0 monotonic |
| 8 | `sampling:u8` | 0 polled, 1 streamed, 2 event-driven |
| 9 | `provenance:u8` | 0 measured, 1 modeled, 2 simulated |
| 10 | `cost_dim:u8` | the index of the K_BCIR cost dimension the signal feeds, `0xFF` none |
| 11 | `reserved[5]` = 0 | |
| 16 | `min_interval_ns:u64` | the sampling-interval hint |
| 24 | `name[40]` | ASCII, NUL-padded (at most 39 bytes) |

A definition the row cannot carry (an ID outside the BCIR range, a name too long or not ASCII, a
duplicate ID, a definition the registry itself refuses) is refused (`SignalTableError`), never
truncated.

**Generated, never hand-edited.** `python -m bcir.signal_table --emit` writes
`runtime/c/bcir_signal_table.h` from the same rows; `--check` (a test, and the C gate in
`tools/c/check_runtime.sh`) refuses a drifted copy, and the harness prints the C rows so the
tests compare them with Python's byte for byte.

**ID ranges** (`signal_range`):

| Range | Owner |
|---|---|
| `0` | unassigned / local — never on a wire |
| `0x00000001`–`0x0000FFFF` | BCIR (assigned by the registry) |
| `0x00010000`–`0x7FFFFFFF` | vendor-assigned |
| `0x80000000`–`0xFFFFFFFE` | device-local |
| `0xFFFFFFFF` | reserved |

**The unknown-required-signal law** (`admit_signal`): a record whose signal the consumer's table
does not define is **refused** when the producer marked it REQUIRED, and **skipped** (counted,
not delivered) otherwise — forward compatibility for an optional signal a newer producer adds,
fail-closed for one the consumer must understand.

## Continuity: one predicate

`SequenceTracker` is the telemetry continuity predicate, and it is the **only** one: the BTLM
stream decoder (`telemetry_frame._sequence_anomalies`) and the envelope intake both express it,
and `bcir_seq_observe` is its C twin. It keeps the newest forward sequence as a watermark: a
sequence equal to it is a **duplicate**; one ahead of it by `1 ≤ δ < 2³¹` (modulo 2³², so
`0xFFFFFFFF → 0` is continuous) advances it and counts `δ − 1` **missing**; anything else is a
**reorder** and leaves the watermark where it is, so an old arrival cannot manufacture a second gap
on the next good one. The refactor that made the frame decoder express it is behaviour-identical
to the parent's own predicate (20,000 random streams, the 2³¹ boundary included).

## Intake

`TelemetryIntake(live_generation)` / `bcir_tev_intake` is the evidence boundary a consumer applies
to every envelope it receives, from any transport (the live ring, a UART, a file). Per record, in
this order:

1. **the wire laws** — a malformed record is refused (`malformed`, with its status);
2. **the stream** — `(source, session)` names one sequence stream; a bounded table of **16**
   streams is kept, and a record of a new stream when the table is full is refused (`streams`,
   `NOSPACE`) rather than evicting a stream and forgetting its continuity;
3. **continuity** — the stream's `SequenceTracker` classifies the sequence (first, next, gap,
   duplicate, reorder) **before any semantic refusal**, so a record refused below is counted once,
   as refused, and never also as missing; the record's `lost` is summed into `reported_lost`;
4. **the signal** — a sample whose signal the table does not define: REQUIRED → refused
   (`unknown`, `TELEMETRY`), optional → skipped;
5. **the generation** — a record bound to a generation older than the live one is **stale**, one
   newer is **ahead**, both refused (`STALE`) by the control plane's own `is_stale` predicate;
   generation 0 is unbound and never stale;
6. **accepted.** Duplicates and reorders are delivered and reported, as the frame ABI delivers and
   reports frames. `witness()` hands the accepted DataDNA to the RT3 ingest gate
   (`sanitize_events`), adding the producers' reported drops to the witness's `dropped`.

The report counts accepted, skipped, refused (malformed, unknown, stale, ahead, streams),
`reported_lost`, missing, reordered, duplicated and the streams in use. Telemetry is evidence,
never a verdict: nothing here reaches `bcir/verify` or the cost vector.

## Gates

| Gate | Where |
|---|---|
| every table row and every corpus envelope identical on both rails; the generated header current | `ring.abi.mismatches`; `bcir/tests/test_signal_table.py`, `test_telemetry_envelope.py` |
| one malformed variant per wire law (and the unknown-required-signal law), refused on both rails with the declared status | `ring.malformed.accepted` |
| continuity fixtures report the declared (missing, reordered, duplicated) on both rails | `ring.sequence.misreported` |
| telemetry of a generation the plane has left refused | `ring.stale.accepted` |
| the decoder under ASan/UBSan; a decoded record re-encodes to its input | `runtime/c/fuzz_ring.c` (envelope mode) in `tools/c/fuzz_streampack.sh` |
| the Python decoder's declared rejections only, over CRC-repaired mutants | the `envelope` surface of `tools/security/run_decoder_campaign.py` |

## Not claimed

- **No transport of its own.** The live ring carries envelopes; UART, HTTP, OTLP and Redfish
  transports remain unbuilt.
- **No clock translation.** `clock` and `unit` name the timestamp's clock; comparing two
  producers' clocks still needs calibration the envelope does not carry.
- **No authentication.** The CRC detects corruption; nothing proves who produced a record.
- **No compatibility promise** beyond version zero, for the envelope, the row layout or the codes.
