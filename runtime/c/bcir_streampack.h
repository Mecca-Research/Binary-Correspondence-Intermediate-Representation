/*===- bcir_streampack.h - BCIR StreamPack binary ABI v1 (frozen) ----------===
 *
 * The normative C view of the BCIR StreamPack wire format (the portable
 * artifact). The Python reference encoder/decoder is bcir/abi/streampack_abi.py;
 * docs/BCIR_STREAMPACK_ABI.md is the prose spec. The forthcoming freestanding C
 * runtime (Phase 8) loads this format with no libc dependency.
 *
 * Wire format (little-endian):
 *   Header (64 bytes, cache-line aligned)  -- bcir_streampack_header
 *   Body (sequential, length-prefixed records):
 *     source_plan : str
 *     segments[n_segments] / prefetches[n_prefetches] / blocks[n_blocks] / trace[n_trace]
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
#define BCIR_STREAMPACK_VERSION_MAX 3    /* v2: pipeline/double-buffer; v3: segment dispatch/channel */
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

/* 64-byte, cache-line header. v2 appends fields into the v1 reserved pad. */
typedef struct bcir_streampack_header {
  uint8_t  magic[4];        /* "BSPK" */
  uint16_t version;         /* 1..BCIR_STREAMPACK_VERSION_MAX */
  uint16_t flags;           /* reserved (0) */
  uint32_t topo_gen;        /* generation tags (R11) */
  uint32_t map_gen;
  uint32_t data_gen;
  uint32_t n_segments;      /* record counts in the body */
  uint32_t n_prefetches;
  uint32_t n_blocks;
  uint32_t n_trace;
  uint16_t pipeline_depth;  /* v2: phases in flight (decoders see 1 for v1) */
  uint8_t  reserved[26];    /* pad to 64 bytes (38 used + 26 = 64) */
} bcir_streampack_header;

/* The 64-byte cache-line header is the frozen ABI -- lock its size at compile time. */
BCIR_STATIC_ASSERT(sizeof(bcir_streampack_header) == BCIR_STREAMPACK_HEADER_SIZE,
                   "BCIR StreamPack header must be exactly 64 bytes (frozen ABI v1/v2)");

/* Lane geometries (must match bcir/model/lanes.py and BCIRAttrs.td). */
typedef enum bcir_lane {
  BCIR_LANE_U = 0, BCIR_LANE_UX = 1, BCIR_LANE_T = 2,
  BCIR_LANE_GGG = 3, BCIR_LANE_A = 4, BCIR_LANE_H = 5
} bcir_lane;

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
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_STREAMPACK_H */
