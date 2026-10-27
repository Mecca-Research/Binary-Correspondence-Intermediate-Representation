/*===- bcir_q8_model.h - portable BCIRQ8 v1 loader ----------------------===*/
#ifndef BCIR_Q8_MODEL_H
#define BCIR_Q8_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "bcir_host_alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_Q8_VERSION 1u
#define BCIR_Q8_HEADER_SIZE 224u
#define BCIR_Q8_DIRECTORY_ENTRY_SIZE 48u
#define BCIR_Q8_DEFAULT_MAX_FILE_SIZE (64ull * 1024ull * 1024ull)

#define BCIR_Q8_FLAG_TIED_EMBEDDINGS 1u
#define BCIR_Q8_MODEL_OWNER_TAG 0x51384d44u

#define BCIR_Q8_TENSOR_EMBEDDING 1u
#define BCIR_Q8_TENSOR_G_ATTN 10u
#define BCIR_Q8_TENSOR_W_Q 11u
#define BCIR_Q8_TENSOR_W_K 12u
#define BCIR_Q8_TENSOR_W_V 13u
#define BCIR_Q8_TENSOR_W_O 14u
#define BCIR_Q8_TENSOR_G_FF 15u
#define BCIR_Q8_TENSOR_W_GATE 16u
#define BCIR_Q8_TENSOR_W_UP 17u
#define BCIR_Q8_TENSOR_W_DOWN 18u
#define BCIR_Q8_TENSOR_G_FINAL 100u
#define BCIR_Q8_TENSOR_LM_HEAD 101u

typedef struct bcir_q8_tensor {
  uint16_t tensor_id;
  int16_t layer;
  uint8_t rank;
  uint32_t dim0;
  uint32_t dim1;
  uint32_t element_count;
  uint32_t group_count;
  uint32_t tensor_crc32;
  uint64_t exponent_offset;
  uint64_t code_offset;
  const unsigned char *exponents;
  const unsigned char *codes;
} bcir_q8_tensor;

typedef struct bcir_q8_model {
  uint32_t flags;
  uint16_t group_size;
  uint8_t bits;
  uint32_t vocab_size;
  uint32_t d_model;
  uint32_t n_heads;
  uint32_t n_kv_heads;
  uint32_t n_layers;
  uint32_t d_ff;
  uint32_t context_length;
  int32_t bos_token_id;
  int32_t eos_token_id;
  int32_t pad_token_id;
  double rope_base;
  double rms_norm_eps;
  uint32_t tensor_count;
  uint32_t body_crc32;
  unsigned char source_model_sha256[32];
  unsigned char source_config_sha256[32];
  unsigned char tokenizer_sha256[32];
  unsigned char *storage;
  size_t storage_size;
  bcir_q8_tensor *tensors;
  /* HEAD-PACK: the head's rows (the embedding's when tied, else the LM head's) packed once at load for the
   * vector row dots, bcir_ai_q8_rows_pack's layout (bcir_ai_kernels.h), head_pack_size bytes; NULL and 0 where
   * d_model is not a whole number of groups, which no pack holds. */
  unsigned char *head_pack;
  size_t head_pack_size;
  bcir_host_allocator _allocator;
  uint32_t _owner_tag;
} bcir_q8_model;

/* A successfully loaded model owns `storage`, `tensors` and `head_pack`; callers
 * borrow the complete object as immutable state until bcir_q8_model_free.
 * Mutating any of them invalidates the loader's CRC/value validation and is
 * unsupported. */

void bcir_q8_model_init(bcir_q8_model *model);

/* Load and fully validate a BCIRQ8 v1 file.  The loader checks the header/body/tensor
 * CRCs, every span and shape, canonical tensor inventory, and model geometry. */
int bcir_q8_model_load(const char *path, bcir_q8_model *out,
                       char *error, size_t error_capacity);

/* The bounded form lets applications opt into a different artifact ceiling without
 * weakening the safe default used by bcir_q8_model_load. */
int bcir_q8_model_load_limited(const char *path, bcir_q8_model *out,
                               char *error, size_t error_capacity,
                               uint64_t max_file_size);

/* Allocator-injected loader. The allocator is borrowed during the call and its
 * context must remain valid until bcir_q8_model_free releases the owned model.
 * `out` is zero-initialized before all reported failures.  Callers must free a
 * previously loaded model before reusing the same output object. */
int bcir_q8_model_load_with_allocator(const char *path, bcir_q8_model *out,
                                      char *error, size_t error_capacity,
                                      const bcir_host_allocator *allocator);

int bcir_q8_model_load_with_allocator_limited(
    const char *path, bcir_q8_model *out, char *error, size_t error_capacity,
    const bcir_host_allocator *allocator, uint64_t max_file_size);

void bcir_q8_model_free(bcir_q8_model *model);

const bcir_q8_tensor *bcir_q8_model_tensor(const bcir_q8_model *model,
                                           uint16_t tensor_id, int16_t layer);

/* Dequantize one element exactly as code * 2**group_exponent. `tensor` is a
 * borrowed pointer returned by bcir_q8_model_tensor for this live model.
 * Returns NaN for a foreign/stale tensor, malformed span, or bad index. */
double bcir_q8_tensor_value(const bcir_q8_model *model,
                            const bcir_q8_tensor *tensor, uint32_t index);

/* DECODE-PASSES: 2**exponent for an exponent the format admits ([-300, 300]), formed from its bits; NaN past it. A
 * code (an integer in [-127, 127]) times it is exact, so it is ldexp(code, exponent) for every code and exponent a
 * loaded tensor holds. */
double bcir_q8_group_scale(int exponent);

/* DECODE-PASSES: elements [start, start + count) of `tensor` into out[0 .. count), each bcir_q8_tensor_value's value
 * bit for bit (NaN where it gives NaN), a group's scale formed once (bcir_q8_group_scale) instead of an ldexp an
 * element. 0; -1 with `out` untouched for a tensor bcir_q8_tensor_value refuses whole (foreign, stale, malformed), a
 * range past its elements, or a NULL `out` with a nonzero count. A zero count writes nothing. */
int bcir_q8_tensor_values(const bcir_q8_model *model, const bcir_q8_tensor *tensor,
                          uint32_t start, uint32_t count, double *out);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_Q8_MODEL_H */
