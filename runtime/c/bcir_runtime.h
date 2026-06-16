/*===- bcir_runtime.h - freestanding BCIR StreamPack runtime --------------===
 *
 * A no-libc loader/executor for the frozen StreamPack binary ABI v1
 * (bcir_streampack.h / docs/BCIR_STREAMPACK_ABI.md). Depends only on
 * <stddef.h> + <stdint.h> (both freestanding-safe), so it links into kernels,
 * drivers, and embedded/WASM hosts with no runtime dependency.
 *
 * The Python reference is bcir/abi/streampack_abi.py; a cross-language parity
 * test (Python encodes, this C decodes) gates the ABI freeze.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_RUNTIME_H
#define BCIR_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_streampack.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum bcir_status {
  BCIR_OK = 0,
  BCIR_ERR_TRUNCATED = 1,
  BCIR_ERR_MAGIC = 2,
  BCIR_ERR_VERSION = 3,
  BCIR_ERR_CRC = 4
} bcir_status;

/* zlib-compatible CRC-32 (reflected, poly 0xEDB88320). */
BCIR_NODISCARD uint32_t bcir_crc32(const uint8_t *BCIR_RESTRICT data, size_t len);

/* Validate magic + version + CRC and copy the header out. A trust boundary: every
 * field is bounds-checked, so any malformed/hostile buffer returns an error status
 * (never reads out of bounds) -- exercised by runtime/c/fuzz_streampack.c. */
BCIR_NODISCARD bcir_status bcir_sp_validate(const uint8_t *BCIR_RESTRICT data,
                                            size_t len,
                                            bcir_streampack_header *BCIR_RESTRICT hdr);

/* A zero-copy view of one lane segment (pointers into the buffer). */
typedef struct bcir_segment_view {
  const char *name; uint16_t name_len;
  uint64_t claim_id;
  uint32_t phase_id;
  uint8_t  lane;        /* bcir_lane */
  uint32_t width;
  uint32_t stride_k;
  const char *opcode; uint16_t opcode_len;
  uint16_t n_reads;  const uint8_t *reads;   /* little-endian u32[n_reads]  */
  uint16_t n_writes; const uint8_t *writes;  /* little-endian u32[n_writes] */
} bcir_segment_view;

/* Decode reads[i] / writes[i] from the raw little-endian pointers. */
uint32_t bcir_seg_read_rid(const bcir_segment_view *seg, uint16_t i);
uint32_t bcir_seg_write_rid(const bcir_segment_view *seg, uint16_t i);

/* Per-segment callback; return nonzero to stop the walk early. */
typedef int (*bcir_seg_fn)(const bcir_segment_view *seg, void *ctx);

/* Validate, then walk every lane segment in order, invoking `fn`. Bounds-checked end
 * to end: a hostile/truncated buffer returns BCIR_ERR_TRUNCATED, never an OOB read. */
BCIR_NODISCARD bcir_status bcir_sp_for_each_segment(const uint8_t *BCIR_RESTRICT data,
                                                    size_t len, bcir_seg_fn fn,
                                                    void *ctx);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_RUNTIME_H */
