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

#define BCIR_STREAMPACK_MAGIC   "BSPK"   /* bytes 0..3 of the header */
#define BCIR_STREAMPACK_VERSION 1
#define BCIR_STREAMPACK_HEADER_SIZE 64

#ifdef __cplusplus
extern "C" {
#endif

/* 64-byte, cache-line header. */
typedef struct bcir_streampack_header {
  uint8_t  magic[4];        /* "BSPK" */
  uint16_t version;         /* = BCIR_STREAMPACK_VERSION */
  uint16_t flags;           /* reserved (0) */
  uint32_t topo_gen;        /* generation tags (R11) */
  uint32_t map_gen;
  uint32_t data_gen;
  uint32_t n_segments;      /* record counts in the body */
  uint32_t n_prefetches;
  uint32_t n_blocks;
  uint32_t n_trace;
  uint8_t  reserved[24];    /* pad to 64 bytes */
} bcir_streampack_header;

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
 *   prefetch  := name:str  distance:u32  targets:u32_array  hint:str  pattern:str
 *   block     := base:u64  count:u64  strides:u64_array
 *   trace     := claim_id:u64  src_hash:u64  trace_hash:u64
 */

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* BCIR_STREAMPACK_H */
