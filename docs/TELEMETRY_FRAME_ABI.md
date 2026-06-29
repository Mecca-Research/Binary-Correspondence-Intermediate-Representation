# BCIR UART telemetry frame ABI — v1 (frozen, normative)

The telemetry frame (T2 of [`TELEMETRY_PIPELINE_RESEARCH.md`](TELEMETRY_PIPELINE_RESEARCH.md) §6)
is BCIR's **embedded telemetry tap**: a self-delimiting, resync-able, CRC-sealed frame a
bare-metal producer drains from the telemetry ring (`TelemetryRing`) and emits over a
byte egress (UART). It is the StreamPack-over-UART wire the research scoped, with
**MIPI SyS-T** (compact framed trace records + optional integrity checksum over
UART/USB/TCP) as the interop reference.

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
- **Record reuse:** the per-record body is the ring's frozen 56-byte `<7q>` layout
  (`TelemetryRing._FMT` / `parse_shared_ring`), so a ring drain frames the records with
  **no body re-encoding**.
- **CRC reuse:** the trailer CRC is **zlib-compatible CRC-32** (reflected, poly
  `0xEDB88320`). Python uses `zlib.crc32(body) & 0xFFFFFFFF`; the C twin reuses
  `bcir_crc32` from [`runtime/c/bcir_runtime.c`](../runtime/c/bcir_runtime.c) (declared
  `extern`, never reimplemented). The two agree byte-for-byte.

## Frame

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BTLM"` (BCIR TeLeMetry) — the **resync anchor** |
| 4 | `version` | `u16` | `1`; a v1 reader rejects a newer version |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `seq` | `u32` | monotonic frame sequence (drop/reorder detection) |
| 12 | `timestamp` | `u64` | producer clock (SyS-T-style); `0` if unavailable |
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

## Resync semantics

The decoder can **join a stream mid-flight** or **recover from a corrupted frame** by
scanning forward to the next `BTLM` magic:

- A garbage prefix (no magic-aligned frame) → scan to the first `BTLM`, decode from there.
- A magic-aligned frame whose **version/length/CRC** check fails → that frame is
  **rejected and counted**, the decoder scans to the **next** magic and continues.
- **Per-frame CRC** bounds the blast radius: a single corrupted byte fails **one** frame,
  not the whole stream. The recovered frames + a reject count are returned
  (`decode_frames` → `FrameStream`); the stream path never raises and never reads out of
  bounds. (`decode_frame`, the strict single-frame path, raises `TelemetryFrameError`
  instead — for a caller decoding one known, frame-aligned buffer.)

## Host decode reuses RT3 (two-truth)

A telemetry frame carries **data** (graded L2/L3 in the telemetry-source ladder), never a
legality verdict. `parse_uart_frames(buf)` decodes the stream and then delegates to the
RT3 ingest gate (`bcir.telemetry.sanitize_events`), so the UART host path produces the
**same `TelemetryIntegrity` witness** as the ring path:

- frame-level losses (rejected/corrupt frames) are carried into the witness's `dropped`
  field (the eviction analogue the ring surfaces on an overwrite);
- an out-of-band record (a forged `thermal > 100`, a negative counter, NaN/inf) is
  **rejected-and-counted** by `sanitize_events` — it never reaches Theta;
- a seq gap / reordered (decreasing-`claim_id`) / replayed batch makes `monotonic` False;
- an empty / all-rejected / all-dropped stream sets `blind` (the suppression signal).

The frame path returns `DataDNA` + `TelemetryIntegrity` only — never a `Diagnostic`,
never a verdict; it touches neither `bcir/verify` nor the cost-vector DIMS. This is the
two-truth quarantine applied to the embedded telemetry tap.

## Egress over UART (documented adapter, not built here)

The producer's testable core is the **frame bytes**: `encode_frame` returns them;
`bcir_tf_encode_frame(recs, n, seq, ts, out, out_len)` writes them to a caller `out`
buffer (returning the byte count, `0` on overflow). Wiring `out` to hardware is a 1-line
adapter over the existing MMIO UART driver
([`runtime/c/uart_regs.h`](../runtime/c/uart_regs.h) — `uart_send(volatile uart_regs_t*, uint32_t)`):

```c
size_t n = bcir_tf_encode_frame(recs, n_recs, seq, ts, out, sizeof out);
for (size_t i = 0; i < n; i++) uart_send(uart, out[i]);   /* the hardware adapter */
```

No UART is touched in T2 — `uart_send` is the **eventual byte sink**, documented here as
the adapter, not a build/runtime dependency. This seeds the planned UART driver.
