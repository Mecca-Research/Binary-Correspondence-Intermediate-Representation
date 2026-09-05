# BCIR StreamPack binary ABI — v1 (frozen, normative) + v2/v3/v4 (append-only)

The StreamPack is BCIR's **portable artifact** (its WASM analog): a self-contained,
hot, executable representation of a selected K_BCIR plan. This document freezes the
v1 wire format. The reference encoder/decoder is
[`bcir/abi/streampack_abi.py`](../../bcir/abi/streampack_abi.py); the C view is
[`runtime/c/bcir_streampack.h`](../../runtime/c/bcir_streampack.h). All three must
agree (a parity test pins the round-trip).

StreamPack is intentionally not a fat binary. When one portable plan is distributed with
multiple standard target images, it is the `root_variant` payload of a separately versioned
[`BCIR Artifact Bundle`](BCIR_ARTIFACT_BUNDLE_ABI.md). BCAB selection does not change any
StreamPack byte, generation rule, or R10/R11 obligation.

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
| 38 | `reserved` | `u8[2]` | pad (0) |
| 40 | `n_gens` | `u32` | **v4** (append-only): generation-vector record count; reads as 0 on v1–v3 buffers |
| 44 | `reserved` | `u8[20]` | pad to 64 |

## Body (sequential, length-prefixed)

1. `source_plan : str`
2. `segments[n_segments]`, each:
   `name:str  claim_id:u64  phase_id:u32  lane:u8  width:u32  stride_k:u32
   opcode:str  reads:u32_array  writes:u32_array  prefetch:str  (""=none)
   fence_before:str_array  fence_after:str_array  [v3: dispatch:u8  channel:str]`
3. `prefetches[n_prefetches]`, each:
   `name:str  distance:u32  targets:u32_array  hint:str  pattern:str
   [v2: buffers:u8]`
4. `blocks[n_blocks]`, each: `base:u64  count:u64  strides:u64_array`
5. `trace[n_trace]`, each: `claim_id:u64  src_hash:u64  trace_hash:u64`
6. `[v4: generations[n_gens]]`, each: `rid:u32  map_gen:u32  data_gen:u32` (12 bytes;
   RIDs strictly ascending)

## Trailer

- `crc32 : u32` — CRC-32 (zlib) of **every preceding byte** (header + body). Decoders
  reject a mismatch.

## v2 (append-only): pipelined phases + double-buffer prefetch

v2 demonstrates the append-only evolution mechanism. It changes **no** v1 field
offsets:

- **Header** gains `pipeline_depth : u16` at offset **36** (carved from the v1
  reserved pad; reserved shrank to `u8[26]`, from which v4 later carved `n_gens`).
  Phases in flight; `2` = double-buffered software pipelining. Decoders treat v1
  buffers as depth 1.
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

## v4 (append-only): per-resource generation vectors (R11)

v4 closes the R11 blind spot of the header tags (S0-2): `map_gen`/`data_gen` are the
**maxima** over the registry, so a resource that moved while another still held the
maximum, or one declared after hydration, left a stale pack indistinguishable from a
fresh one. v4 carries the generation of **every** declared resource:

- **Header** gains `n_gens : u32` at offset **40** (carved from the pad; bytes 38–39
  and 44–63 stay reserved and must be zero). Decoders read 0 on v1–v3 buffers, and a
  v1–v3 buffer with nonzero bytes at 40–43 is refused as reserved (`BCIR_ERR_RESERVED`).
- **Body** appends `generations[n_gens]` after the trace records: `rid:u32 map_gen:u32
  data_gen:u32` (`bcir_generation_view`, `BCIR_GENERATION_WIRE_SIZE` = 12), one record
  per declared resource in **strictly ascending RID order**. Segment, prefetch, block, and
  trace records are **unchanged**; v4 implies v3 and v2 (the records carry their tails and
  the header its `pipeline_depth`). Walkers that stop at the trace stream stay correct.
- **The header tags are the vector's summary**: `map_gen`/`data_gen` MUST equal the
  vector's maxima. A pack whose header and vector disagree, whose RIDs repeat or are out
  of order, or whose `n_gens` promises records the body does not carry is malformed
  (`AbiError` / `BCIR_ERR_GENERATION`, `BCIR_ERR_TRUNCATED`) on both rails before any
  registry is consulted. `topo_gen` stays the constant `1`: topology identity is bound
  through the plan's provenance manifest (`m_module`, R13), not through a pack tag.
- **Encoders emit the lowest carrying version**: a pack with an empty vector is
  byte-identical v1/v2/v3; `gem.hydrate` always emits the vector (`generation_vector`),
  so every hydrated pack is v4, and a hand-built pack without one stays v1–v3.
- **R11 per resource is ONE predicate on every rail** (`verify_pack`, the C
  `bcir_sp_check_generation_vector` / `bcir_sp_execute_checked_vector`, and the law
  rail's `-bcir-verify` over `bcir.gem.stream_pack`'s `generations` triples): with a
  vector present, every entry must match the live registry exactly and every declared
  resource must have an entry — a resource that moved under the maxima, one declared
  after hydration, or an entry naming an undeclared RID is STALE (`BCIR_ERR_STALE`,
  "rehydrate: repack" for `map_gen`, "replan" for `data_gen`). A pack with **no** vector
  is stale against any registry that declares resources (a v1–v3 artifact must be
  rehydrated) and, over an empty registry, is judged by the maxima alone (which must be 0).
  The maxima-only API (`bcir_sp_check_generation`) remains for callers that hold only the
  maxima; it cannot see a resource that moved under them, and the adversarial gate
  keeps that RED witness (`stale_vector`, `missing_vector_entry`, `undeclared_vector_rid`).
- Projected by the ASN.1 module as `generations [10] SEQUENCE OF Generation`
  ([`BCIR_ASN1_X690_ABI.md`](../BCIR_ASN1_X690_ABI.md) §3, projection version 2); the
  DER → native fast path re-derives v4 from the component's presence.

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
- **R11 per resource** (v4: `bcir_sp_check_generation_vector`, and the vector-checked
  executor `bcir_sp_execute_checked_vector`) — the caller's live registry table
  (`bcir_generation_view[]`, any order) must match the pack's generation vector entry for
  entry and resource for resource; a pack without a vector is STALE against any registry
  that declares resources. The vector's well-formedness (ascending RIDs, header maxima)
  is enforced by `bcir_sp_verify_semantic` (`BCIR_ERR_GENERATION`) before the registry
  is consulted; `bcir_sp_for_each_generation` walks the records in RID order.
- **Exact body consumption** — after the declared segment/prefetch/block/trace (and v4
  generation) records, the next four bytes must be the CRC trailer and the trailer must
  end the artifact.
  CRC-valid bytes inserted between the declared body and a recomputed CRC are rejected
  (`BCIR_ERR_TRAILING` in C) on both rails; parsers never treat an undeclared tail as an
  extension point.

These mirror `bcir/verify::verify_pack`, so the Python and C rails agree.

## Versioning (the freeze)

- v1 is **frozen**: the field layout above does not change.
- New fields are **append-only** across versions (v2/v3/v4 above are the worked
  instances); a v1 reader of a v1 buffer is exact and lossless.
- A reader **rejects** a buffer whose `version` exceeds the maximum it supports
  (v4 today). `lane` values match `bcir/model/lanes.py` / `BCIRAttrs.td`
  (`U=0,UX=1,T=2,GGG=3,A=4,H=5`).
- The exact-consumption rule is decoder hardening, not a field-layout change, so it does
  **not** increment the wire version. Future records/fields require an explicit append-only
  version and corresponding counts/lengths; reserved or trailing bytes are not implicit ABI.

Writers reject values that cannot be represented exactly: integer fields never mask or wrap,
`pipeline_depth` is in `1..65535`, prefetch `buffers` is `1` or `2`, segment width is a
nonzero `u32` power of two, dispatch is one of the closed v3 codes, and a v4 generation
vector has strictly ascending `u32` RIDs with the header tags as its maxima. Decoders apply
the same depth/buffer/width/dispatch/vector constraints. This is validation of the existing
v1–v4 contract, not a wire-layout revision.

## Why a frozen ABI now

It is the linchpin for the next phases: the freestanding **C runtime**
loads exactly these bytes with no libc; cross-language **drivers** consume them; and
it is the stable hand-off between the compiler (`mlir/`) and any executor. The
artifact is portable in the same sense as a `.wasm` module — data the standard
engine loads.
