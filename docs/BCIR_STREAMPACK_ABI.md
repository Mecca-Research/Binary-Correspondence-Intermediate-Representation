# BCIR StreamPack binary ABI — v1 (frozen, normative) + v2 (append-only)

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
   fence_before:str_array  fence_after:str_array`
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

## Versioning (the freeze)

- v1 is **frozen**: the field layout above does not change.
- New fields are **append-only** across versions (v2 above is the worked
  instance); a v1 reader of a v1 buffer is exact and lossless.
- A reader **rejects** a buffer whose `version` exceeds the maximum it supports
  (v2 today). `lane` values match `bcir/model/lanes.py` / `BCIRAttrs.td`
  (`U=0,UX=1,T=2,GGG=3,A=4,H=5`).

## Why a frozen ABI now

It is the linchpin for the next phases: the freestanding **C runtime** (Phase 8)
loads exactly these bytes with no libc; cross-language **drivers** consume them; and
it is the stable hand-off between the compiler (`mlir/`) and any executor. The
artifact is portable in the same sense as a `.wasm` module — data the standard
engine loads.
