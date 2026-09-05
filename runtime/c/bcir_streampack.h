/*===- bcir_streampack.h - BCIR StreamPack binary ABI v1 (frozen) ----------===
 *
 * The normative C view of the BCIR StreamPack wire format (the portable
 * artifact). The Python reference encoder/decoder is bcir/abi/streampack_abi.py;
 * docs/kernel/BCIR_STREAMPACK_ABI.md is the prose spec. The freestanding C runtime
 * loads v1 through the append-only v2/v3/v4 forms with no libc dependency.
 *
 * Wire format (little-endian):
 *   Header (64 bytes, cache-line aligned)  -- bcir_streampack_header
 *   Body (sequential, length-prefixed records):
 *     source_plan : str
 *     segments[n_segments] / prefetches[n_prefetches] / blocks[n_blocks] / trace[n_trace]
 *     [v4: generations[n_gens]]  (the per-resource generation vector, law R11)
 *   Trailer: u32 CRC-32 of every preceding byte.
 *
 * Conventions:
 *   str        := u16 length, then `length` UTF-8 bytes.
 *   u32_array  := u16 count, then `count` * u32.
 *   u64_array  := u16 count, then `count` * u64.
 *   str_array  := u16 count, then `count` * str.
 *
 * The format is frozen at v1; fields are append-only across versions and a v1
 * reader rejects a newer major version.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_STREAMPACK_H
#define BCIR_STREAMPACK_H

#include <stdint.h>

/* C23 niceties, portably guarded so the freestanding header still builds under C11
 * and as C++ (the runtime targets C23; the macros degrade cleanly). */
#if defined(__cplusplus)
#  define BCIR_RESTRICT __restrict
#  define BCIR_NODISCARD [[nodiscard]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#  define BCIR_RESTRICT restrict
#  define BCIR_NODISCARD [[nodiscard]]
#else
#  define BCIR_RESTRICT restrict
#  define BCIR_NODISCARD
#endif

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define BCIR_STATIC_ASSERT(c, m) static_assert(c, m)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define BCIR_STATIC_ASSERT(c, m) _Static_assert(c, m)
#else
#  define BCIR_STATIC_ASSERT(c, m)
#endif

#define BCIR_STREAMPACK_MAGIC   "BSPK"   /* bytes 0..3 of the header */
#define BCIR_STREAMPACK_VERSION 1
#define BCIR_STREAMPACK_VERSION_MAX 4    /* v2: pipeline/double-buffer; v3: segment dispatch/channel;
                                          * v4: the per-resource generation vector (R11) */
#define BCIR_STREAMPACK_HEADER_SIZE 64

/* v3 segment dispatch (the execution-routing target). u8 enum on the wire, mirroring
 * bcir/abi/streampack_abi.py::_DISPATCH_WIRE. "core" is the default a v1/v2 pack carries
 * implicitly; "pim" dispatches to processing-in-memory. A code outside this set is rejected. */
typedef enum bcir_dispatch {
  BCIR_DISPATCH_CORE = 0,   /* execute on the compute core (default) */
  BCIR_DISPATCH_PIM  = 1    /* dispatch to the memory controller (processing-in-memory) */
} bcir_dispatch;
#define BCIR_DISPATCH_MAX 1  /* the largest legal dispatch code (range gate) */

#ifdef __cplusplus
extern "C" {
#endif

/* 64-byte, cache-line header. v2 and v4 append fields into the v1 reserved pad. */
typedef struct bcir_streampack_header {
  uint8_t  magic[4];        /* "BSPK" */
  uint16_t version;         /* 1..BCIR_STREAMPACK_VERSION_MAX */
  uint16_t flags;           /* reserved (0) */
  uint32_t topo_gen;        /* generation tags (R11); map_gen/data_gen are the vector's maxima on v4 */
  uint32_t map_gen;
  uint32_t data_gen;
  uint32_t n_segments;      /* record counts in the body */
  uint32_t n_prefetches;
  uint32_t n_blocks;
  uint32_t n_trace;
  uint16_t pipeline_depth;  /* v2 @36: phases in flight (decoders see 1 for v1) */
  uint8_t  reserved0[2];    /* 38..39: reserved (0) */
  uint32_t n_gens;          /* v4 @40: generation records after the trace stream (decoders see 0 before v4) */
  uint8_t  reserved[20];    /* pad to 64 bytes (44 used + 20 = 64) */
} bcir_streampack_header;

/* The 64-byte cache-line header is the frozen ABI -- lock its size at compile time. */
BCIR_STATIC_ASSERT(sizeof(bcir_streampack_header) == BCIR_STREAMPACK_HEADER_SIZE,
                   "BCIR StreamPack header must be exactly 64 bytes (frozen ABI v1/v2)");

/* Lane geometries (must match bcir/model/lanes.py and BCIRAttrs.td). */
typedef enum bcir_lane {
  BCIR_LANE_U = 0, BCIR_LANE_UX = 1, BCIR_LANE_T = 2,
  BCIR_LANE_GGG = 3, BCIR_LANE_A = 4, BCIR_LANE_H = 5
} bcir_lane;

/* v4 (append-only): one per-resource generation record (law R11) -- the registry's
 * (map_gen, data_gen) for `rid` when the pack was hydrated. The body carries n_gens of
 * them after the trace records, RIDs strictly ascending; the header map_gen/data_gen are
 * their maxima. 12 bytes each on the wire (three little-endian u32). Mirrors
 * bcir/gem/streampack.py::Generation. */
typedef struct bcir_generation_view {
  uint32_t rid;
  uint32_t map_gen;
  uint32_t data_gen;
} bcir_generation_view;
#define BCIR_GENERATION_WIRE_SIZE 12u

/*
 * Body records are variable-length (length-prefixed), so they are described here
 * as field sequences rather than fixed C structs. A reader walks them in order.
 *
 *   segment   := name:str  claim_id:u64  phase_id:u32  lane:u8  width:u32
 *                stride_k:u32  opcode:str  reads:u32_array  writes:u32_array
 *                prefetch:str  fence_before:str_array  fence_after:str_array
 *                [v3: dispatch:u8  channel:str]
 *   prefetch  := name:str  distance:u32  targets:u32_array  hint:str  pattern:str
 *                [v2: buffers:u8  (2 = double-buffer contract)]
 *   block     := base:u64  count:u64  strides:u64_array
 *   trace     := claim_id:u64  src_hash:u64  trace_hash:u64
 *   [v4: generation := rid:u32  map_gen:u32  data_gen:u32   -- after the trace stream]
 *
 * v2 (append-only): the header gains pipeline_depth (in the v1 pad) and the
 * prefetch record appends buffers:u8. Segment/block/trace records are unchanged,
 * so v1 walkers remain correct on the segment stream of a v2 pack.
 *
 * v3 (append-only): the SEGMENT record appends dispatch:u8 (a bcir_dispatch code) +
 * channel:str (the heterogeneous HardwareChannel, "host" by default), so the
 * execution-routing target is now ON the wire and inside the CRC -- a dispatch/channel
 * mutation is no longer invisible to byte-identity. v3 implies v2 (it carries
 * pipeline_depth + the prefetch buffers tail). A pack that uses neither (every segment
 * "core"/"host") encodes as the lowest carrying version, so v1/v2 packs are unchanged.
 *
 * v4 (append-only, S0-2 / law R11): the header gains n_gens:u32 at offset 40 (carved
 * from the pad; 38..39 stay reserved) and the body appends n_gens GENERATION records
 * after the trace stream -- the registry's per-resource (map_gen, data_gen) at
 * hydration, RIDs strictly ascending, the header map_gen/data_gen being their maxima
 * (a decoder refuses a vector that is unsorted or whose maxima the header does not
 * carry: BCIR_ERR_GENERATION). The maxima alone could not see a resource that moved
 * while another still held the maximum, nor one declared after hydration;
 * bcir_sp_check_generation_vector compares every entry with the caller's live registry.
 * v4 implies v3 and v2 (every tail is written). A pack with no vector encodes as the
 * lowest carrying version, so v1/v2/v3 artifacts are unchanged; every hydrated pack
 * carries its vector and is v4.
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_STREAMPACK_H */
