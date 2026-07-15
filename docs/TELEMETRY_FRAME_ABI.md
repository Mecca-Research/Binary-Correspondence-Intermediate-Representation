# BCIR UART telemetry frame ABI — v1 (frozen, normative)

The telemetry frame (T2 of [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md) §6)
is BCIR's **DataDNA UART codec**: a self-delimiting, resync-able, CRC-sealed frame a
bare-metal producer drains from the telemetry ring (`TelemetryRing`) and emits over a
byte egress (UART). It is a narrow single-producer v1 record batch—not the future
multi-device driver telemetry envelope and not StreamPack itself. **MIPI SyS-T**
(compact framed trace records + optional integrity checksum over UART/USB/TCP) is the
interop reference.

This document freezes the v1 wire format. The reference encoder/decoder is
[`bcir/telemetry_frame.py`](../bcir/telemetry_frame.py); the freestanding C twin is
[`runtime/c/bcir_telemetry_frame.h`](../runtime/c/bcir_telemetry_frame.h) /
[`bcir_telemetry_frame.c`](../runtime/c/bcir_telemetry_frame.c). Both rails must agree
byte-for-byte — a differential test ([`bcir/tests/test_telemetry_frame.py`](../bcir/tests/test_telemetry_frame.py))
plus the `#telemetry-frame` probe in [`tools/c/check_runtime.sh`](../tools/c/check_runtime.sh)
pin the round-trip and the byte-identical re-encode.

## Conventions

- **Endianness:** little-endian throughout (header fields + record fields), written byte
  by byte on both rails (host-endian-independent; no struct padding on the wire).
- **Record-layout reuse:** the per-record body is the ring's frozen 56-byte `<7q>` layout
  (`TelemetryRing._FMT` / `parse_shared_ring`). The current Python helper materializes
  `DataDNA` values and serializes them again; the C writer serializes fields explicitly
  little-endian. This preserves one schema and byte layout across both rails without
  claiming a current direct-copy implementation.
- **CRC reuse:** the trailer CRC is **zlib-compatible CRC-32** (reflected, poly
  `0xEDB88320`). Python uses `zlib.crc32(body) & 0xFFFFFFFF`; the C twin reuses
  `bcir_crc32` from [`runtime/c/bcir_runtime.c`](../runtime/c/bcir_runtime.c) (declared
  `extern`, never reimplemented). The two agree byte-for-byte.
- **Exactness:** reserved v1 flags must be zero. The strict Python `decode_frame` and C
  `bcir_tf_decode_exact` reject bytes after the one frame; the C prefix decoder
  `bcir_tf_decode_frame` intentionally returns the consumed length for stream walkers.

## Frame

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BTLM"` (BCIR TeLeMetry) — the **resync anchor** |
| 4 | `version` | `u16` | `1`; a v1 reader rejects a newer version |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `seq` | `u32` | producer-local monotonic sequence, modulo 2³² |
| 12 | `timestamp` | `u64` | opaque producer tick; `0` if unavailable |
| 20 | `n_records` | `u16` | record count in the body |
| 22 | `records` | `record[n_records]` | each the 56-byte `<7q>` DataDNA record |
| 22 + 56·n | `crc32` | `u32` | zlib-compatible CRC-32 over every preceding byte |

The fixed header is **22 bytes**; the CRC trailer is **4 bytes**; a frame carrying `n`
records is `22 + 56·n + 4` bytes. An empty (`n == 0`) frame is the minimum well-formed
frame (26 bytes).

### Record (56 bytes, the ring's `<7q>` layout)

| Offset | Field | Type |
|---|---|---|
| 0 | `claim_id` | `i64` |
| 8 | `cycles` | `i64` |
| 16 | `bytes` | `i64` |
| 24 | `misses` | `i64` |
| 32 | `thermal` | `i64` |
| 40 | `voltage` | `i64` |
| 48 | `utilization` | `i64` |

Signed fields are the two's-complement bit pattern `struct.pack("<q", ...)` emits, so a
negative counter round-trips byte-identical across both rails. The C `bcir_tf_record`
struct carries a `BCIR_TF_STATIC_ASSERT(sizeof == 56)` — the frozen-record ABI lock.
Raw wire decoding preserves all signed-i64 bit patterns. The RT3 host ingest contract is
stricter: `claim_id`, `cycles`, and `bytes` must be non-negative signed-i64 integers and
the four normalized fields must be integer `0..100`; floats, bools, oversized Python
integers, and negative host values are rejected before calibration.

## Resync semantics

The decoder can **join a stream mid-flight** or **recover from a corrupted frame** by
scanning forward to the next `BTLM` magic:

- A garbage prefix (no magic-aligned frame) → scan to the first `BTLM`, decode from there.
- A magic-aligned candidate whose **flags/version/length/CRC** check fails—including a
  magic-aligned truncated suffix—is **rejected and counted**, and the decoder scans to
  the **next** magic and continues.
- **Per-frame CRC** prevents a corrupted candidate from being accepted without poisoning
  subsequent valid frames. The recovered frames + a rejected-candidate count are returned
  (`decode_frames` → `FrameStream`); the stream path never raises and never reads out of
  bounds. (`decode_frame`, the strict single-frame path, raises `TelemetryFrameError`
  instead — for a caller decoding one known, frame-aligned buffer.)

`FrameStream` also classifies recovered `seq` values as missing, reordered, or duplicate.
The comparison is modulo 2³², so `0xffffffff → 0` is continuous. A reordered frame does
not move the forward watermark backwards. Resynchronization is deliberately best effort:
`BTLM` is only a four-byte anchor and may occur in corrupt payload bytes; every candidate
still has to pass flags, bounds, version, and CRC before it is accepted, and false anchors
may make the reject count larger than the number of originating frames.

## Host decode reuses RT3 (two-truth)

A telemetry frame carries **data** (graded L2/L3 in the telemetry-source ladder), never a
legality verdict. `parse_uart_frames(buf)` decodes the stream and then delegates to the
RT3 ingest gate (`bcir.telemetry.sanitize_events`), so the UART host path produces the
same record-integrity schema as the ring path, plus explicit frame-integrity fields:

- `frames_accepted` distinguishes a validated frame path from a non-frame witness;
  `frames_rejected`, `frames_missing`, `frames_reordered`, and `frames_duplicated` record
  wire-candidate and sequence anomalies; `frame_monotonic` is false on duplicate/reorder;
- `dropped` remains a **record** loss count (for example ring overwrite). A corrupt frame
  has an unknown record count, so the decoder does not invent one;
- an out-of-band record (a forged `thermal > 100`, a negative counter, NaN/inf) is
  **rejected-and-counted** by `sanitize_events` — it never reaches Theta;
- decreasing record `claim_id` makes record-level `monotonic` false independently of
  frame sequence continuity;
- an empty / all-rejected / all-dropped stream sets `blind` (the suppression signal).

The frame path returns `DataDNA` + `TelemetryIntegrity` only — never a `Diagnostic`,
never a verdict; it touches neither `bcir/verify` nor the cost-vector DIMS. This is the
two-truth quarantine applied to the embedded telemetry tap.

## Frozen-v1 scope and pre-driver extension

BTLM v1 has no source ID, session/restart generation, clock identity/unit, declared
record schema/kind, or producer loss counters. Consequently:

- sequence and timestamps are meaningful only within one externally separated producer
  stream whose restart boundary is known out-of-band;
- timestamps cannot be compared across producers or converted to age without external
  clock metadata;
- concatenating streams after a producer restart is ambiguous; and
- the reused shared-ring v1 is a quiescent-snapshot baseline, not a concurrent driver IPC
  ring (it has no tail, generation, per-slot publish sequence, or backpressure contract).

The bytes above remain frozen. Before D2 resident drivers, the roadmap requires a new
versioned **driver telemetry envelope**—not an in-place reinterpretation of BTLM v1—with
source/session/generation, clock ID and unit, record kind/schema/size, stable numeric
signal IDs, producer loss accounting, and explicit backpressure. The live shared ring
must separately define SPSC head/tail, acquire/release publication, per-slot generation
or sequence, peer-death/restart behavior, and concurrent wrap tests.

## Egress over UART (documented adapter, not built here)

The producer's testable core is the **frame bytes**: `encode_frame` returns them;
`bcir_tf_encode_frame(recs, n, seq, ts, out, out_len)` writes them to a caller `out`
buffer (returning the byte count, `0` on overflow). A future resident-driver adapter can
wire `out` to a UART byte sink:

```c
size_t n = bcir_tf_encode_frame(recs, n_recs, seq, ts, out, sizeof out);
for (size_t i = 0; i < n; i++) uart_send(uart, out[i]);   /* eventual adapter */
```

No UART is touched in T2. The repository's
[`runtime/c/uart_regs.h`](../runtime/c/uart_regs.h) and
[`runtime/c/cfront_driver_uart.c`](../runtime/c/cfront_driver_uart.c) are a compiler
fixture containing a sample `uart_send`, not a channel-backed resident driver or a
public UART API. The example above specifies the eventual byte-sink boundary only.
