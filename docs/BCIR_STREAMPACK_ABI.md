# BCIR StreamPack binary ABI — v1 (frozen, normative) + v2/v3 (append-only)

The StreamPack is BCIR's **portable artifact** (its WASM analog): a self-contained,
hot, executable representation of a selected K_BCIR plan. This document freezes the
v1 wire format. The reference encoder/decoder is
[`bcir/abi/streampack_abi.py`](../bcir/abi/streampack_abi.py); the C view is
[`runtime/c/bcir_streampack.h`](../runtime/c/bcir_streampack.h). All three must
agree (a parity test pins the round-trip).

## Conventions

- **Endianness:** little-endian.
- **`str`** := `u16` byte-length, then that many UTF-8 bytes.
- **`u32_array`** := `u16` count, then `count` × `u32`.
- **`u64_array`** := `u16` count, then `count` × `u64`.
- **`str_array`** := `u16` count, then `count` × `str`.

## Header (64 bytes, cache-line aligned)

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0 | `magic` | `u8[4]` | `"BSPK"` |
| 4 | `version` | `u16` | `1` |
| 6 | `flags` | `u16` | reserved (0) |
| 8 | `topo_gen` | `u32` | generation tags (law R11) |
| 12 | `map_gen` | `u32` | |
| 16 | `data_gen` | `u32` | |
| 20 | `n_segments` | `u32` | body record counts |
| 24 | `n_prefetches` | `u32` | |
| 28 | `n_blocks` | `u32` | |
| 32 | `n_trace` | `u32` | |
| 36 | `pipeline_depth` | `u16` | **v2** (append-only); reads as 1 on v1 buffers |
| 38 | `reserved` | `u8[26]` | pad to 64 |

## Body (sequential, length-prefixed)

1. `source_plan : str`
2. `segments[n_segments]`, each:
   `name:str  claim_id:u64  phase_id:u32  lane:u8  width:u32  stride_k:u32
   opcode:str  reads:u32_array  writes:u32_array  prefetch:str  (""=none)
   fence_before:str_array  fence_after:str_array  [v3: dispatch:u8  channel:str]`
3. `prefetches[n_prefetches]`, each:
   `name:str  distance:u32  targets:u32_array  hint:str  pattern:str`
4. `blocks[n_blocks]`, each: `base:u64  count:u64  strides:u64_array`
5. `trace[n_trace]`, each: `claim_id:u64  src_hash:u64  trace_hash:u64`

## Trailer

- `crc32 : u32` — CRC-32 (zlib) of **every preceding byte** (header + body). Decoders
  reject a mismatch.

## v2 (append-only): pipelined phases + double-buffer prefetch

v2 demonstrates the append-only evolution mechanism. It changes **no** v1 field
offsets:

- **Header** gains `pipeline_depth : u16` at offset **36** (carved from the v1
  reserved pad; reserved shrinks to `u8[22]`). Phases in flight; `2` =
  double-buffered software pipelining. Decoders treat v1 buffers as depth 1.
- **Prefetch records** append `buffers : u8` after `pattern` (`2` = a
  double-buffer contract feeding the next pipelined phase, typically
  `pattern="double_buffer"`). Segment/block/trace records are **unchanged**, so
  v1 segment walkers remain correct on a v2 pack.
- **Encoders emit the lowest carrying version**: a pack with no v2 features
  (depth 1, all `buffers == 1`) is byte-identical frozen v1.
- Emitted by `gem.streampack.hydrate_pipelined`; scheduled by
  `gem.schedule.execute_tokens` (the token DAG provides the matching overlap);
  decoded by the freestanding C runtime (`bcir_streampack_header.pipeline_depth`).

## v3 (append-only): on-wire segment dispatch + channel

v3 moves the **execution-routing** fields onto the wire (and so inside the CRC),
closing the gap where a `dispatch`/`channel` mutation was invisible to byte-identity:

- **Segment records** append `dispatch:u8` (a dispatch code — `0=core` default,
  `1=pim` for processing-in-memory) + `channel:str` (the heterogeneous
  `HardwareChannel`, `"host"` by default, e.g. `"nvidia_ptx"`). The header, prefetch,
  block, and trace records are **unchanged**; v3 implies v2 (it carries
  `pipeline_depth` + the prefetch `buffers` tail).
- **Encoders emit the lowest carrying version**: a pack with every segment at the
  defaults (`dispatch="core"`, `channel="host"`) is byte-identical v1/v2; a pack that
  uses a non-default dispatch/channel on any segment encodes as v3.
- A decoder **rejects** an unknown `dispatch` code (only `0`/`1` are legal) just as it
  rejects an out-of-range `lane` — the two rails (Python `decode` and the freestanding
  C `bcir_sp_for_each_segment`) raise/return `BCIR_ERR_*` identically.

## Semantic trust boundary (R10/R11 in C)

The CRC + bounds decode is memory-safe, but a CRC-valid pack can still be
**semantically** corrupt. The freestanding C runtime now enforces, *after* the CRC:

- **Range gate** — every segment's `lane` is one of the 6 lanes and `width` is a
  nonzero power of two (`BCIR_ERR_LANE` / `BCIR_ERR_WIDTH`); a v3 `dispatch` is legal
  (`BCIR_ERR_DISPATCH`). Mirrors the Python decoder's `Lane(bad)` raise.
- **R10 provenance** (`bcir_sp_verify_semantic`) — every segment's `claim_id` resolves
  to a decoded trace record, a declared `prefetch` resolves to a prefetch record *and*
  covers ≥ 1 of the segment's reads, and the v2 well-formedness holds (`BCIR_ERR_PROVENANCE`).
- **R11 generation** (`bcir_sp_check_generation`, and the checked executor
  `bcir_sp_execute_checked`) — `map_gen`/`data_gen` must match the caller's expected
  (live registry) generation, else the pack is STALE (`BCIR_ERR_STALE`) and is rehydrated,
  never executed. `bcir_sp_execute` rejects an R10/range-failing pack before running it.

These mirror `bcir/verify::verify_pack`, so the Python and C rails agree.

## Versioning (the freeze)

- v1 is **frozen**: the field layout above does not change.
- New fields are **append-only** across versions (v2/v3 above are the worked
  instances); a v1 reader of a v1 buffer is exact and lossless.
- A reader **rejects** a buffer whose `version` exceeds the maximum it supports
  (v3 today). `lane` values match `bcir/model/lanes.py` / `BCIRAttrs.td`
  (`U=0,UX=1,T=2,GGG=3,A=4,H=5`).

## Why a frozen ABI now

It is the linchpin for the next phases: the freestanding **C runtime** (Phase 8)
loads exactly these bytes with no libc; cross-language **drivers** consume them; and
it is the stable hand-off between the compiler (`mlir/`) and any executor. The
artifact is portable in the same sense as a `.wasm` module — data the standard
engine loads.
