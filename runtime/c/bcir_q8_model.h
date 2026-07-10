/*===- bcir_q8_model.h - portable BCIRQ8 v1 loader ----------------------===*/
#ifndef BCIR_Q8_MODEL_H
#define BCIR_Q8_MODEL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BCIR_Q8_VERSION 1u
#define BCIR_Q8_HEADER_SIZE 224u
#define BCIR_Q8_DIRECTORY_ENTRY_SIZE 48u

#define BCIR_Q8_FLAG_TIED_EMBEDDINGS 1u

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
} bcir_q8_model;

/* Load and fully validate a BCIRQ8 v1 file.  The loader checks the header/body/tensor
 * CRCs, every span and shape, canonical tensor inventory, and model geometry. */
int bcir_q8_model_load(const char *path, bcir_q8_model *out,
                       char *error, size_t error_capacity);

void bcir_q8_model_free(bcir_q8_model *model);

const bcir_q8_tensor *bcir_q8_model_tensor(const bcir_q8_model *model,
                                           uint16_t tensor_id, int16_t layer);

/* Dequantize one element exactly as code * 2**group_exponent. */
double bcir_q8_tensor_value(const bcir_q8_model *model,
                            const bcir_q8_tensor *tensor, uint32_t index);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_Q8_MODEL_H */
