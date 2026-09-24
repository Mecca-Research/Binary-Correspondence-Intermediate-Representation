# BCIR manifest-of-shards ("BSHM") — version zero (experimental, normative for v0)

The manifest-of-shards is **how a StreamPack too large to ship as one artifact crosses the data
plane** (GEM+ roadmap G16, staged plan S3-C). The whole pack travels as **shards named by
digest**, and one small manifest binds them. Each shard is itself a StreamPack that a node admits
and runs on its own. A further pack, the **frame**, carries everything that is not a segment. The
manifest names the frame and every shard by SHA-256, and names the whole by its length and
SHA-256. So the shards reassemble to the whole pack's bytes, byte for byte, or they are refused.

Before it, the C++ seam's `DistributedOrchestrator::shard()` returned bare segment ranges over a
buffer that only its caller owned. There was no format a node could receive, and no way to prove
that the ranges a cluster ran were the ones the whole declared. The retired follow-up list in
[`CPP_HANDOFF_BOUNDARY.md`](../languages/CPP_HANDOFF_BOUNDARY.md) called this "a manifest
variant of the artifact".

The executable oracle is [`bcir/abi/shard_manifest.py`](../../bcir/abi/shard_manifest.py). The
production rail is the freestanding C twin
[`runtime/c/bcir_shard_manifest.h`](../../runtime/c/bcir_shard_manifest.h) /
`bcir_shard_manifest.c`, which has no heap and no libc. The C++ seam cuts and reassembles shard
sets through it (`cut_shards` and `reassemble_shards` in
[`runtime/cpp/bcir_handoff.hpp`](../../runtime/cpp/bcir_handoff.hpp)). The harness is
`runtime/c/test_handoff.c`, and the libFuzzer target is `runtime/c/fuzz_handoff.c`.

**Version zero.** The layout carries **no compatibility promise** until a cluster exercises it. A
change is a version bump, never a reinterpretation. It is not the BCIR UAPI.

## Conventions

- Little-endian; every field has a fixed width.
- **CRC:** CRC-32 (zlib; `bcir_crc32` on the C rail) over every byte before the trailer.
- **Digests:** SHA-256 (`bcir_sha256` on the C rail). `registry_digest` is G14's
  `registry_digest` of the whole's generation vector: the same bytes that
  `bcir_ctl_pack_registry_digest` computes from a pack, and that `bcir_ctl_admit_pack` compares
  with the installed registry. There is one predicate for all three
  ([`BCIR_CONTROL_PLANE_ABI.md`](BCIR_CONTROL_PLANE_ABI.md)).
- **Statuses** are the runtime's `bcir_status`; Python raises `ShardError.status` with the same
  names. S3-C appended `BCIR_ERR_LIFETIME` (23, the pack table's) and `BCIR_ERR_SHARD` (24).
  Every law of this format other than the framing laws refuses with `BCIR_ERR_SHARD`.

## What shards

Only a **v4 pack with a generation vector**, in the **hydrated layout**, may shard:

- **S1:** exactly one trace record per segment, in segment order (trace *i* names segment *i*'s
  claim).
- **S2:** the prefetch records that the segments name appear in segment order, and each is named
  at most once.

Every BCIR hydrator emits this layout: `gem.hydrate`, `hydrate_pipelined`, and the per-step
freeze `bcir_hydrate_generations`. It lets both rails cut a shard in one forward pass, with no
search and no scratch memory. `bcir_shm_check_whole` / `hydrated_layout` decide it.

Every shard carries the whole generation vector. A node therefore admits the manifest against its
live registry before it fetches a single shard (`admit_manifest` / `bcir_ho_admit_manifest`), and
admits each shard again on arrival, by the one R11 predicate.

## The frame and the shards

- **Shard *r*** is the canonical v4 sub-pack of the whole over segments `[begin, end)`. It holds
  those segment records verbatim, the prefetch records they name, and their trace records, each in
  the whole's order. It has no blocks. It carries the whole vector, under the whole's tags and
  pipeline depth.
- **The frame** is the whole with no segments: every prefetch, block and trace record, and the
  vector. A pack with no segments is its own frame.

Each is a well-formed StreamPack that `bcir_sp_verify_semantic` accepts.

**Partition.** The seam cuts `ceil(n / world)` segments per shard, contiguous from 0, so a cluster
may get fewer shards than ranks. An empty pack is one empty shard, and a world of 0 counts as 1.
The C twin is `bcir_shm_partition` and the oracle is `partition`. Every rail uses the one
partition, and the C++ `DistributedOrchestrator::shard()` calls the C twin.

## The manifest

```
header (64)   @0   magic "BSHM"            @4  version u16 = 0   @6  flags u16 = 0
              @8   n_shards u32            @12 n_segments u32
              @16  whole_length u64        @24 frame_length u64
              @32  map_gen u32  @36 data_gen u32  @40 topo_gen u32
              @44  pack_version u16 = 4    @46 reserved u16 = 0
              @48  n_gens u32              @52 reserved[12] = 0
digests (96)  @64  whole_sha256[32]  @96 frame_sha256[32]  @128 registry_digest[32]
shards        @160 n_shards x 48:  seg_begin u32 | seg_end u32 | length u64 | sha256[32]
trailer            crc32 u32 of every preceding byte
```

The size is exactly `164 + 48·n_shards`. `n_shards` is at most **4096**, the declared bound, so a
manifest is at most 196,772 bytes. Every length is at most `2³² − 1`, so it fits a 32-bit
target's `size_t`. The smallest v4 pack with `n_gens` vector entries is `70 + 12·n_gens` bytes: a
header, an empty source plan, the vector and the CRC. That size is the lower bound on the frame
and on every shard.

## Wire laws (in this order, both rails)

`decode_manifest` / `bcir_shm_decode` apply them, and `encode_manifest` / `bcir_shm_encode`
decode their own output before they return it, so the encoder refuses what the decoder refuses.
Both rails report the same first violation.

1. shorter than the fixed part → `BCIR_ERR_TRUNCATED`
2. magic ≠ `BSHM` → `BCIR_ERR_MAGIC`
3. version ≠ 0 → `BCIR_ERR_VERSION`
4. the CRC → `BCIR_ERR_CRC`
5. flags, the pad at 46, or the pads 52..63 are not zero → `BCIR_ERR_RESERVED`
6. `n_shards` outside 1..4096 → `BCIR_ERR_SHARD`
7. the size is not `164 + 48·n_shards` → `BCIR_ERR_TRUNCATED` (short) or `BCIR_ERR_TRAILING` (long)
8. `pack_version` ≠ 4, or `n_gens` = 0 → `BCIR_ERR_SHARD`
9. a length over `2³² − 1` or under the smallest v4 pack; a whole shorter than its frame; a pack
   with segments that is its own frame, or a pack without segments that is not → `BCIR_ERR_SHARD`
10. the ranges do not partition `[0, n_segments)` into contiguous, non-empty shards starting at 0
    (an empty pack is one empty shard) → `BCIR_ERR_SHARD`

## Reassembly: one total predicate

`reassemble(manifest, fetch)` / `bcir_shm_reassemble` build the whole from the manifest and a
content-addressed store: `fetch(sha256)` returns a blob or nothing. They return the whole or
refuse. There is **one** predicate, and every failure in it is `BCIR_ERR_SHARD`, except that the C
twin reports `BCIR_ERR_NOSPACE` when the caller's buffer is short. It reports that only after the
blobs have proved the declared length. The predicate holds when all of these are true:

- the manifest is valid (its own status otherwise);
- every blob — the frame and each shard — is present, has its declared length and digest, and is
  a well-formed pack;
- the frame carries no segments;
- the whole built from the frame and the shards' segment records has the declared length and
  SHA-256, verifies, is in the hydrated layout, and carries the manifest's tags, its segment
  count, its vector length and its **registry digest**;
- the frame, and every shard over its range, are **byte for byte the ones the unit cuts from that
  whole**.

So a node that ran shard *r* ran exactly segments `[begin, end)` of this whole. A shard spelled
another way is refused, even when it carries the right segments: extra prefetches, a missing
trace, a reordered vector. So is a set whose manifest lies about the tags or the registry. The C
twin writes the whole into the caller's buffer and zeroes that buffer on any refusal.

## Admission at the node

`admit_manifest(plane, manifest)` / `bcir_ho_admit_manifest` is the **data plane's gate before
any fetch**. It decodes the manifest (`malformed` with its status otherwise). It then compares the
manifest's tags and registry digest with the live plane's installed registry. This is the law
`bcir_ctl_admit_pack` applies to a whole pack, and a manifest of an older vector with unchanged
maxima is refused as stale. Each fetched shard is then an ordinary pack. It is admitted into the
node's pack table and dispatched there
([`BCIR_DATA_PLANE_HANDOFF.md`](BCIR_DATA_PLANE_HANDOFF.md)).

## Gates

- **Rows** (`tools/perf/gemplus_baseline.py --group handoff`, graded by
  `tools/c/check_handoff.py` → `bcir/tests/handoff_fixtures.py::measure`):
  - `handoff.shards.mismatches` 392 → 0: the manifest, frame and shards of every split case
    agree between the oracle, the C twin and the C++ seam's cut; every whole reassembles byte for
    byte on every rail; packs outside the hydrated layout are refused.
  - `handoff.shards.malformed.accepted` 68 → 0: one variant per wire law, and tampered shard
    sets (a missing, forged, foreign, non-canonical or resealed shard, a lying manifest), each
    refused with its declared status.
  - `handoff.reentry.divergent` 115 → 0: every shard, admitted by the live plane and run by
    itself, reproduces the whole's dispatch.
- **Fuzz:** `fuzz_handoff.c` mode 0 decodes hostile manifests. Mode 1 tampers with and reseals
  shard sets, and asserts that only the declared whole ever reassembles. It is registered in
  `tools/c/fuzz_streampack.sh` and the decoder campaign's `manifest` surface.
- **Faults:** `tools/testing/faults/handoff.json` injects defects into the codec, the partition,
  the sub-pack and frame spellings, the layout law and the reassembly predicate. Each defect is
  caught by its own row.

## Relation to the other surfaces

- [`BCIR_STREAMPACK_ABI.md`](BCIR_STREAMPACK_ABI.md): shards and frames are ordinary v4 packs,
  and nothing here changes a StreamPack byte.
- [`BCIR_CONTROL_PLANE_ABI.md`](BCIR_CONTROL_PLANE_ABI.md): the registry digest, and the one
  admission predicate.
- [`BCIR_DATA_PLANE_HANDOFF.md`](BCIR_DATA_PLANE_HANDOFF.md): where shards live once they arrive
  (the pack table) and how they run.

## Not claimed

No transport. Getting shards to nodes (MPI/NCCL, RDMA, a file system) is the distributed
backend's dispatch, which remains a declared stub that throws `HandoffError`. No reduction
across ranks. No compression. No partial reassembly: a set is the whole or it is refused.
