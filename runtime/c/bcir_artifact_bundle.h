/*===- bcir_artifact_bundle.h - freestanding BCAB v1 reader --------*- C -*-===*
 *
 * Allocation-free, fail-closed reader/selector for the BCIR Artifact Bundle.
 * Input and selected payloads are BORROWED immutable spans; callers retain the
 * backing bytes for the lifetime of every view. No pointer crosses the wire.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_ARTIFACT_BUNDLE_H
#define BCIR_ARTIFACT_BUNDLE_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_streampack.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_AB_MAGIC "BCAB"
#define BCIR_AB_VERSION 1u
#define BCIR_AB_HEADER_SIZE 128u
#define BCIR_AB_ENTRY_SIZE 448u
#define BCIR_AB_MAX_ENTRIES 1024u
#define BCIR_AB_MAX_BYTES (UINT64_C(1) << 30)
#define BCIR_AB_NO_INDEX UINT32_MAX
#define BCIR_AB_VARIANT_ID_MAX 47u
#define BCIR_AB_TRIPLE_MAX 47u
#define BCIR_AB_TARGET_NAME_MAX 23u
#define BCIR_AB_FEATURE_CSV_MAX 63u

#define BCIR_AB_FLAG_R12_ATTESTED 0x1u
#define BCIR_AB_FLAG_EXECUTABLE   0x2u
#define BCIR_AB_FLAG_PORTABLE     0x4u
#define BCIR_AB_FLAG_DEBUG        0x8u

typedef enum bcir_ab_status {
  BCIR_AB_OK = 0,
  BCIR_AB_ERR_INVALID = 1,
  BCIR_AB_ERR_SIZE = 2,
  BCIR_AB_ERR_MAGIC = 3,
  BCIR_AB_ERR_VERSION = 4,
  BCIR_AB_ERR_LAYOUT = 5,
  BCIR_AB_ERR_METADATA = 6,
  BCIR_AB_ERR_CRC = 7,
  BCIR_AB_ERR_SHA256 = 8,
  BCIR_AB_ERR_PAYLOAD = 9,
  BCIR_AB_ERR_INCOMPATIBLE = 10,
  BCIR_AB_ERR_NOT_FOUND = 11
} bcir_ab_status;

typedef enum bcir_ab_kind {
  BCIR_AB_STREAM_PACK = 1, BCIR_AB_ELF_OBJECT = 2, BCIR_AB_ELF_SHARED = 3,
  BCIR_AB_COFF_OBJECT = 4, BCIR_AB_MACHO_OBJECT = 5, BCIR_AB_ARCHIVE = 6,
  BCIR_AB_WASM = 7, BCIR_AB_LLVM_BITCODE = 8, BCIR_AB_LLVM_IR = 9,
  BCIR_AB_PTX = 10, BCIR_AB_CUBIN = 11, BCIR_AB_SPIRV = 12,
  BCIR_AB_JVM_CLASS = 13, BCIR_AB_CIL = 14, BCIR_AB_C_SOURCE = 15,
  BCIR_AB_CPP_SOURCE = 16, BCIR_AB_SYCL_SOURCE = 17, BCIR_AB_ASSEMBLY = 18,
  BCIR_AB_ELF_EXECUTABLE = 19, BCIR_AB_PE_EXECUTABLE = 20,
  BCIR_AB_PE_SHARED = 21, BCIR_AB_MACHO_EXECUTABLE = 22,
  BCIR_AB_MACHO_SHARED = 23, BCIR_AB_RAW_BINARY = 24
} bcir_ab_kind;

typedef enum bcir_ab_format {
  BCIR_AB_FMT_NONE = 0, BCIR_AB_FMT_STREAM_PACK = 1, BCIR_AB_FMT_ELF = 2,
  BCIR_AB_FMT_COFF = 3, BCIR_AB_FMT_MACHO = 4, BCIR_AB_FMT_ARCHIVE = 5,
  BCIR_AB_FMT_WASM = 6, BCIR_AB_FMT_LLVM_BITCODE = 7, BCIR_AB_FMT_TEXT = 8,
  BCIR_AB_FMT_SPIRV = 9, BCIR_AB_FMT_JVM_CLASS = 10, BCIR_AB_FMT_PE = 11,
  BCIR_AB_FMT_RAW = 12
} bcir_ab_format;

typedef enum bcir_ab_endianness {
  BCIR_AB_ENDIAN_NEUTRAL = 0, BCIR_AB_ENDIAN_LITTLE = 1,
  BCIR_AB_ENDIAN_BIG = 2
} bcir_ab_endianness;

typedef struct bcir_ab_view {
  const uint8_t *data; /* BORROWED */
  size_t length;
  uint32_t count;
  uint32_t root_index;
  uint32_t default_index;
  uint64_t provenance_digest;
  uint64_t generation;
  uint32_t body_crc32;
  uint32_t header_crc32;
  uint8_t embedded_sha256[32];
} bcir_ab_view;

typedef struct bcir_ab_entry {
  uint16_t kind;
  uint16_t format;
  uint8_t endianness;
  uint8_t pointer_bits;
  uint16_t flags;
  uint32_t machine;
  int32_t priority;
  uint64_t payload_offset;
  uint64_t payload_size;
  uint64_t provenance_digest;
  uint64_t cal_gen;
  uint32_t payload_crc32;
  uint8_t payload_sha256[32];
  uint8_t target_manifest_sha256[32];
  char variant_id[48];
  char triple[48];
  char architecture[24];
  char os_abi[24];
  char channel[24];
  char entry_symbol[32];
  char required_features[64];
  char prohibited_features[64];
  const uint8_t *payload; /* BORROWED from bcir_ab_view.data */
} bcir_ab_entry;

/* Caller-owned selector policy. Every non-NULL string is NUL-terminated within
 * its matching *_MAX bound above; target_manifest_sha256, when non-NULL, names
 * a readable 32-byte array. Zero kind/format masks accept all known values.
 * An absent identity field does not satisfy an entry that constrains that field.
 * Feature CSV must be sorted and duplicate-free, exactly like the wire
 * representation. */
typedef struct bcir_ab_envelope {
  const char *triple;              /* BORROWED, optional */
  const char *architecture;        /* BORROWED, optional */
  const char *os_abi;              /* BORROWED, optional */
  const char *channel;             /* BORROWED, optional */
  const char *features;            /* BORROWED canonical CSV, optional */
  uint64_t accepted_kind_mask;     /* bit kind, zero accepts all */
  uint64_t accepted_format_mask;   /* bit format, zero accepts all */
  uint8_t endianness;
  uint8_t pointer_bits;
  uint32_t machine;
  const uint8_t *target_manifest_sha256; /* BORROWED 32 bytes, NULL = unspecified */
  uint64_t cal_gen;
  uint8_t has_cal_gen;
  uint8_t require_r12;
  uint8_t allow_debug;
} bcir_ab_envelope;

/* On failure, output structures are zeroed and remain safe to inspect again. */
BCIR_NODISCARD bcir_ab_status bcir_ab_open(const uint8_t *data, size_t length,
                                           bcir_ab_view *out);
BCIR_NODISCARD bcir_ab_status bcir_ab_get(const bcir_ab_view *view, uint32_t index,
                                          bcir_ab_entry *out);
BCIR_NODISCARD bcir_ab_status bcir_ab_select(const bcir_ab_view *view,
                                             const bcir_ab_envelope *envelope,
                                             const char *requested_id,
                                             uint32_t *selected_index);
const char *bcir_ab_status_string(bcir_ab_status status);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_ARTIFACT_BUNDLE_H */
