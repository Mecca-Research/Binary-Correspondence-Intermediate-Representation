/*===- bcir_q8_model.c - portable BCIRQ8 v1 loader ----------------------===*/
#include "bcir_q8_model.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char k_magic[8] = {'B','C','I','R','Q','8',0,0};

#if defined(_WIN32)
typedef int64_t bcir_file_offset;
#define BCIR_FSEEK _fseeki64
#define BCIR_FTELL _ftelli64
#else
typedef long bcir_file_offset;
#define BCIR_FSEEK fseek
#define BCIR_FTELL ftell
#endif

static uint16_t rd16(const unsigned char *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static int16_t rdi16(const unsigned char *p) { return (int16_t)rd16(p); }

static uint32_t rd32(const unsigned char *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static int32_t rdi32(const unsigned char *p) { return (int32_t)rd32(p); }

static uint64_t rd64(const unsigned char *p) {
  return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

static double rdf64(const unsigned char *p) {
  uint64_t bits = rd64(p);
  double out;
  memcpy(&out, &bits, sizeof out);
  return out;
}

static void set_error(char *out, size_t cap, const char *fmt, ...) {
  va_list ap;
  if (!out || cap == 0) return;
  va_start(ap, fmt);
  (void)vsnprintf(out, cap, fmt, ap);
  va_end(ap);
  out[cap - 1] = '\0';
}

static uint32_t crc_update(uint32_t state, const unsigned char *data, size_t size) {
  size_t i;
  int bit;
  for (i = 0; i < size; ++i) {
    state ^= data[i];
    for (bit = 0; bit < 8; ++bit)
      state = (state >> 1) ^ (0xEDB88320u & (0u - (state & 1u)));
  }
  return state;
}

static uint32_t crc32_bytes(const unsigned char *data, size_t size) {
  return crc_update(0xFFFFFFFFu, data, size) ^ 0xFFFFFFFFu;
}

static uint32_t header_crc(const unsigned char *data) {
  uint32_t state = crc_update(0xFFFFFFFFu, data, 116);
  static const unsigned char zero[4] = {0,0,0,0};
  state = crc_update(state, zero, 4);
  state = crc_update(state, data + 120, BCIR_Q8_HEADER_SIZE - 120);
  return state ^ 0xFFFFFFFFu;
}

static int add_overflows_u64(uint64_t a, uint64_t b, uint64_t *out) {
  *out = a + b;
  return *out < a;
}

static int mul_overflows_size(size_t a, size_t b, size_t *out) {
  if (b && a > SIZE_MAX / b) return 1;
  *out = a * b;
  return 0;
}

static int span_ok(uint64_t off, uint64_t length, uint64_t lower, uint64_t size,
                   uint64_t *end) {
  return off >= lower && !add_overflows_u64(off, length, end) && *end <= size;
}

typedef struct bcir_q8_span {
  uint64_t begin;
  uint64_t end;
} bcir_q8_span;

static int compare_spans(const void *left, const void *right) {
  const bcir_q8_span *a = (const bcir_q8_span *)left;
  const bcir_q8_span *b = (const bcir_q8_span *)right;
  if (a->begin < b->begin) return -1;
  if (a->begin > b->begin) return 1;
  if (a->end < b->end) return -1;
  if (a->end > b->end) return 1;
  return 0;
}

static const bcir_q8_tensor *find_tensor(const bcir_q8_model *model,
                                         uint16_t id, int16_t layer) {
  uint32_t i;
  for (i = 0; i < model->tensor_count; ++i)
    if (model->tensors[i].tensor_id == id && model->tensors[i].layer == layer)
      return &model->tensors[i];
  return NULL;
}

const bcir_q8_tensor *bcir_q8_model_tensor(const bcir_q8_model *model,
                                           uint16_t tensor_id, int16_t layer) {
  if (!model || model->_owner_tag != BCIR_Q8_MODEL_OWNER_TAG ||
      !model->storage || !model->tensors) return NULL;
  return find_tensor(model, tensor_id, layer);
}

double bcir_q8_tensor_value(const bcir_q8_model *model,
                            const bcir_q8_tensor *tensor, uint32_t index) {
  uintptr_t base, address;
  size_t tensor_bytes, delta;
  uint64_t groups, exponent_bytes;
  int code;
  int exponent;
  if (!model || !tensor || model->_owner_tag != BCIR_Q8_MODEL_OWNER_TAG ||
      !model->storage || !model->tensors || !model->group_size || model->bits != 8u)
    return NAN;
  if (mul_overflows_size((size_t)model->tensor_count,
                         sizeof *model->tensors,&tensor_bytes)) return NAN;
  base = (uintptr_t)(const void *)model->tensors;
  address = (uintptr_t)(const void *)tensor;
  if (address < base) return NAN;
  delta = (size_t)(address - base);
  if (delta >= tensor_bytes || delta % sizeof *model->tensors) return NAN;
  if (index >= tensor->element_count || !tensor->element_count) return NAN;
  groups = ((uint64_t)tensor->element_count + model->group_size - 1u) /
           model->group_size;
  exponent_bytes = groups * 2u;
  if (groups != tensor->group_count ||
      tensor->exponent_offset > model->storage_size ||
      exponent_bytes > model->storage_size - tensor->exponent_offset ||
      tensor->code_offset > model->storage_size ||
      tensor->element_count > model->storage_size - tensor->code_offset ||
      tensor->exponents != model->storage + tensor->exponent_offset ||
      tensor->codes != model->storage + tensor->code_offset)
    return NAN;
  code = (int)tensor->codes[index];
  if (code >= 128) code -= 256;
  exponent = (int)rdi16(tensor->exponents + 2u * (index / model->group_size));
  if (code == -128 || exponent < -300 || exponent > 300) return NAN;
  return ldexp((double)code, exponent);
}

void bcir_q8_model_free(bcir_q8_model *model) {
  bcir_host_allocator allocator;
  if (!model) return;
  if(model->_owner_tag!=BCIR_Q8_MODEL_OWNER_TAG||
     !bcir_host_allocator_valid(&model->_allocator)){
    memset(model,0,sizeof *model);return;
  }
  allocator=model->_allocator;
  bcir_host_deallocate(&allocator,model->tensors);
  bcir_host_deallocate(&allocator,model->storage);
  memset(model, 0, sizeof *model);
}

void bcir_q8_model_init(bcir_q8_model *model) {
  if(model) memset(model,0,sizeof *model);
}

static int expect_tensor(const bcir_q8_model *m, uint16_t id, int16_t layer,
                         uint8_t rank, uint32_t d0, uint32_t d1,
                         char *error, size_t cap) {
  const bcir_q8_tensor *t = find_tensor(m, id, layer);
  if (!t) {
    set_error(error, cap, "missing tensor id=%u layer=%d", (unsigned)id, (int)layer);
    return -1;
  }
  if (t->rank != rank || t->dim0 != d0 || t->dim1 != d1) {
    set_error(error, cap, "bad shape for tensor id=%u layer=%d", (unsigned)id, (int)layer);
    return -1;
  }
  return 0;
}

static int validate_inventory(const bcir_q8_model *m, char *error, size_t cap) {
  uint32_t layer;
  uint32_t d = m->d_model, f = m->d_ff;
  uint32_t dk = d / m->n_heads, kvd = m->n_kv_heads * dk;
  uint32_t expected = 2u + 9u * m->n_layers +
                      ((m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS) ? 0u : 1u);
  if (m->tensor_count != expected) {
    set_error(error, cap, "tensor count %u != canonical %u", m->tensor_count, expected);
    return -1;
  }
  if (expect_tensor(m, BCIR_Q8_TENSOR_EMBEDDING, -1, 2, m->vocab_size, d, error, cap))
    return -1;
  for (layer = 0; layer < m->n_layers; ++layer) {
    int16_t li = (int16_t)layer;
    if (expect_tensor(m, BCIR_Q8_TENSOR_G_ATTN, li, 1, d, 1, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_Q, li, 2, d, d, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_K, li, 2, d, kvd, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_V, li, 2, d, kvd, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_O, li, 2, d, d, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_G_FF, li, 1, d, 1, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_GATE, li, 2, d, f, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_UP, li, 2, d, f, error, cap) ||
        expect_tensor(m, BCIR_Q8_TENSOR_W_DOWN, li, 2, f, d, error, cap))
      return -1;
  }
  if (expect_tensor(m, BCIR_Q8_TENSOR_G_FINAL, -1, 1, d, 1, error, cap)) return -1;
  if (!(m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS) &&
      expect_tensor(m, BCIR_Q8_TENSOR_LM_HEAD, -1, 2, m->vocab_size, d, error, cap))
    return -1;
  return 0;
}

static int canonical_tensor_at(const bcir_q8_model *m, uint32_t index,
                               uint16_t *id, int16_t *layer) {
  static const uint16_t layer_ids[9] = {
      BCIR_Q8_TENSOR_G_ATTN, BCIR_Q8_TENSOR_W_Q, BCIR_Q8_TENSOR_W_K,
      BCIR_Q8_TENSOR_W_V, BCIR_Q8_TENSOR_W_O, BCIR_Q8_TENSOR_G_FF,
      BCIR_Q8_TENSOR_W_GATE, BCIR_Q8_TENSOR_W_UP, BCIR_Q8_TENSOR_W_DOWN};
  if (index == 0) { *id = BCIR_Q8_TENSOR_EMBEDDING; *layer = -1; return 0; }
  --index;
  if (index < 9u * m->n_layers) {
    *id = layer_ids[index % 9u]; *layer = (int16_t)(index / 9u); return 0;
  }
  index -= 9u * m->n_layers;
  if (index == 0) { *id = BCIR_Q8_TENSOR_G_FINAL; *layer = -1; return 0; }
  if (index == 1 && !(m->flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS)) {
    *id = BCIR_Q8_TENSOR_LM_HEAD; *layer = -1; return 0;
  }
  return -1;
}

int bcir_q8_model_load_with_allocator_limited(
    const char *path, bcir_q8_model *out, char *error, size_t error_capacity,
    const bcir_host_allocator *allocator, uint64_t max_file_size) {
  FILE *file = NULL;
  bcir_file_offset signed_size;
  size_t got;
  unsigned char *p;
  uint16_t version, header_size, group_size;
  uint32_t endian, flags, tensor_count, entry_size, body_crc, recorded_header_crc;
  uint64_t directory_offset, directory_end, payload_offset, recorded_size;
  uint32_t i, j;
  uint64_t expected_tensor_count;
  size_t tensor_bytes, span_count, span_bytes;
  bcir_q8_span *spans = NULL;
  bcir_q8_model m;
  memset(&m, 0, sizeof m);
  m._allocator=bcir_host_allocator_or_default(allocator);
  m._owner_tag=BCIR_Q8_MODEL_OWNER_TAG;
  if (error && error_capacity) error[0] = '\0';
  if (out) memset(out, 0, sizeof *out);
  if (!path || !out) {
    set_error(error, error_capacity, "path and output model are required");
    return -1;
  }
  if (max_file_size < BCIR_Q8_HEADER_SIZE) {
    set_error(error, error_capacity, "BCIRQ8 file-size limit is below the fixed header");
    return -1;
  }
  file = fopen(path, "rb");
  if (!file) {
    set_error(error, error_capacity, "cannot open BCIRQ8 file");
    return -1;
  }
  if (BCIR_FSEEK(file, 0, SEEK_END) || (signed_size = BCIR_FTELL(file)) < 0 ||
      BCIR_FSEEK(file, 0, SEEK_SET)) {
    set_error(error, error_capacity, "cannot determine BCIRQ8 file size");
    fclose(file);
    return -1;
  }
  if ((uint64_t)signed_size < BCIR_Q8_HEADER_SIZE ||
      (uint64_t)signed_size > max_file_size ||
      (uint64_t)signed_size > (uint64_t)SIZE_MAX) {
    set_error(error, error_capacity, "BCIRQ8 file size is invalid or exceeds limit");
    fclose(file);
    return -1;
  }
  m.storage_size = (size_t)signed_size;
  m.storage = (unsigned char *)bcir_host_allocate(&m._allocator,m.storage_size);
  if (!m.storage) {
    set_error(error, error_capacity, "out of memory reading BCIRQ8 file");
    fclose(file);
    return -1;
  }
  got = fread(m.storage, 1, m.storage_size, file);
  {
    int close_error = fclose(file);
    file = NULL;
    if (got != m.storage_size || close_error) {
      set_error(error, error_capacity, "short read from BCIRQ8 file");
      bcir_q8_model_free(&m);
      return -1;
    }
  }
  p = m.storage;
  if (memcmp(p, k_magic, sizeof k_magic)) {
    set_error(error, error_capacity, "bad BCIRQ8 magic");
    goto fail;
  }
  version = rd16(p + 8); header_size = rd16(p + 10); endian = rd32(p + 12);
  flags = rd32(p + 16); group_size = rd16(p + 20);
  if (version != BCIR_Q8_VERSION || header_size != BCIR_Q8_HEADER_SIZE ||
      endian != 0x01020304u || p[22] != 8 || p[23] != 0 || !group_size ||
      (flags & ~BCIR_Q8_FLAG_TIED_EMBEDDINGS)) {
    set_error(error, error_capacity, "unsupported BCIRQ8 header fields");
    goto fail;
  }
  for (i = 216; i < BCIR_Q8_HEADER_SIZE; ++i) if (p[i] != 0) {
    set_error(error, error_capacity, "nonzero BCIRQ8 reserved header bytes");
    goto fail;
  }
  recorded_header_crc = rd32(p + 116);
  if (header_crc(p) != recorded_header_crc) {
    set_error(error, error_capacity, "BCIRQ8 header CRC mismatch");
    goto fail;
  }
  m.flags = flags; m.group_size = group_size; m.bits = 8;
  m.vocab_size = rd32(p + 24); m.d_model = rd32(p + 28);
  m.n_heads = rd32(p + 32); m.n_kv_heads = rd32(p + 36);
  m.n_layers = rd32(p + 40); m.d_ff = rd32(p + 44);
  m.context_length = rd32(p + 48); m.bos_token_id = rdi32(p + 52);
  m.eos_token_id = rdi32(p + 56); m.pad_token_id = rdi32(p + 60);
  m.rope_base = rdf64(p + 64); m.rms_norm_eps = rdf64(p + 72);
  tensor_count = rd32(p + 80); entry_size = rd32(p + 84);
  directory_offset = rd64(p + 88); payload_offset = rd64(p + 96);
  recorded_size = rd64(p + 104); body_crc = rd32(p + 112);
  if (!m.vocab_size || !m.d_model || !m.n_heads || !m.n_kv_heads || !m.n_layers ||
      !m.d_ff || m.n_layers > (uint32_t)INT16_MAX || m.d_model % m.n_heads ||
      m.n_heads % m.n_kv_heads ||
      (m.d_model / m.n_heads) % 2 || !isfinite(m.rope_base) || m.rope_base <= 0.0 ||
      !isfinite(m.rms_norm_eps) || m.rms_norm_eps <= 0.0) {
    set_error(error, error_capacity, "invalid BCIRQ8 model geometry");
    goto fail;
  }
  if ((m.bos_token_id != -1 &&
       (m.bos_token_id < 0 || (uint32_t)m.bos_token_id >= m.vocab_size)) ||
      (m.eos_token_id != -1 &&
       (m.eos_token_id < 0 || (uint32_t)m.eos_token_id >= m.vocab_size)) ||
      (m.pad_token_id != -1 &&
       (m.pad_token_id < 0 || (uint32_t)m.pad_token_id >= m.vocab_size))) {
    set_error(error, error_capacity, "BCIRQ8 tokenizer id is outside the vocabulary");
    goto fail;
  }
  expected_tensor_count = 2u + 9u * (uint64_t)m.n_layers +
      ((m.flags & BCIR_Q8_FLAG_TIED_EMBEDDINGS) ? 0u : 1u);
  if (expected_tensor_count > UINT32_MAX || tensor_count != expected_tensor_count) {
    set_error(error, error_capacity, "BCIRQ8 tensor count is not canonical");
    goto fail;
  }
  if (entry_size != BCIR_Q8_DIRECTORY_ENTRY_SIZE ||
      directory_offset != BCIR_Q8_HEADER_SIZE || recorded_size != m.storage_size ||
      add_overflows_u64(directory_offset, (uint64_t)tensor_count * entry_size,
                        &directory_end) || directory_end > payload_offset ||
      payload_offset > recorded_size || (payload_offset & 7u)) {
    set_error(error, error_capacity, "invalid BCIRQ8 directory/payload offsets");
    goto fail;
  }
  {
    uint64_t padding;
    for (padding = directory_end; padding < payload_offset; ++padding) if (p[padding] != 0) {
      set_error(error, error_capacity, "nonzero BCIRQ8 directory padding");
      goto fail;
    }
  }
  if (crc32_bytes(p + directory_offset, m.storage_size - (size_t)directory_offset) != body_crc) {
    set_error(error, error_capacity, "BCIRQ8 body CRC mismatch");
    goto fail;
  }
  m.body_crc32 = body_crc; m.tensor_count = tensor_count;
  memcpy(m.source_model_sha256, p + 120, 32);
  memcpy(m.source_config_sha256, p + 152, 32);
  memcpy(m.tokenizer_sha256, p + 184, 32);
  if (mul_overflows_size((size_t)tensor_count, sizeof *m.tensors, &tensor_bytes)) {
    set_error(error, error_capacity, "BCIRQ8 tensor directory is too large");
    goto fail;
  }
  m.tensors = (bcir_q8_tensor *)bcir_host_allocate(&m._allocator,tensor_bytes);
  if (!m.tensors) {
    set_error(error, error_capacity, "out of memory for BCIRQ8 tensor directory");
    goto fail;
  }
  memset(m.tensors,0,tensor_bytes);
  if (mul_overflows_size((size_t)tensor_count, 2u, &span_count) ||
      mul_overflows_size(span_count, sizeof *spans, &span_bytes)) {
    set_error(error, error_capacity, "BCIRQ8 span inventory is too large");
    goto fail;
  }
  spans = (bcir_q8_span *)bcir_host_allocate(&m._allocator, span_bytes);
  if (!spans) {
    set_error(error, error_capacity, "out of memory for BCIRQ8 span inventory");
    goto fail;
  }
  for (i = 0; i < tensor_count; ++i) {
    const unsigned char *e = p + directory_offset + (uint64_t)i * entry_size;
    bcir_q8_tensor *t = &m.tensors[i];
    uint8_t eflags = e[5];
    uint16_t reserved = rd16(e + 6);
    uint32_t reserved2 = rd32(e + 28);
    uint64_t exponent_end, code_end;
    uint64_t expected_count, expected_groups;
    uint32_t actual_crc;
    uint16_t expected_id;
    int16_t expected_layer;
    t->tensor_id = rd16(e); t->layer = rdi16(e + 2); t->rank = e[4];
    t->dim0 = rd32(e + 8); t->dim1 = rd32(e + 12);
    t->element_count = rd32(e + 16); t->group_count = rd32(e + 20);
    t->tensor_crc32 = rd32(e + 24);
    t->exponent_offset = rd64(e + 32); t->code_offset = rd64(e + 40);
    if (canonical_tensor_at(&m, i, &expected_id, &expected_layer) ||
        t->tensor_id != expected_id || t->layer != expected_layer) {
      set_error(error, error_capacity, "non-canonical BCIRQ8 tensor order at entry %u", i);
      goto fail;
    }
    if (eflags || reserved || reserved2 || (t->rank != 1 && t->rank != 2) ||
        !t->dim0 || !t->dim1) {
      set_error(error, error_capacity, "invalid BCIRQ8 tensor directory entry %u", i);
      goto fail;
    }
    expected_count = t->rank == 1 ? t->dim0 : (uint64_t)t->dim0 * t->dim1;
    expected_groups = (expected_count + group_size - 1u) / group_size;
    if (expected_count != t->element_count || expected_groups != t->group_count ||
        (t->exponent_offset & 7u) || (t->code_offset & 7u) ||
        !span_ok(t->exponent_offset, (uint64_t)t->group_count * 2u,
                 payload_offset, recorded_size, &exponent_end) ||
        !span_ok(t->code_offset, t->element_count, payload_offset,
                 recorded_size, &code_end)) {
      set_error(error, error_capacity, "bad BCIRQ8 tensor shape/span at entry %u", i);
      goto fail;
    }
    t->exponents = p + t->exponent_offset; t->codes = p + t->code_offset;
    for (j = 0; j < t->group_count; ++j) {
      int exponent = (int)rdi16(t->exponents + (size_t)j * 2u);
      if (exponent < -300 || exponent > 300) {
        set_error(error, error_capacity, "invalid BCIRQ8 exponent at entry %u", i);
        goto fail;
      }
    }
    for (j = 0; j < t->element_count; ++j) if (t->codes[j] == 128u) {
      set_error(error, error_capacity, "non-canonical BCIRQ8 -128 code at entry %u", i);
      goto fail;
    }
    actual_crc = crc_update(0xFFFFFFFFu, t->exponents, (size_t)t->group_count * 2u);
    actual_crc = crc_update(actual_crc, t->codes, t->element_count) ^ 0xFFFFFFFFu;
    if (actual_crc != t->tensor_crc32) {
      set_error(error, error_capacity, "BCIRQ8 tensor CRC mismatch at entry %u", i);
      goto fail;
    }
    spans[2u * i].begin = t->exponent_offset;
    spans[2u * i].end = exponent_end;
    spans[2u * i + 1u].begin = t->code_offset;
    spans[2u * i + 1u].end = code_end;
  }
  qsort(spans, span_count, sizeof *spans, compare_spans);
  for (i = 1; i < span_count; ++i) {
    if (spans[i - 1u].end > spans[i].begin) {
      set_error(error, error_capacity, "overlapping BCIRQ8 tensor payloads");
      goto fail;
    }
  }
  bcir_host_deallocate(&m._allocator, spans);
  spans = NULL;
  if (validate_inventory(&m, error, error_capacity)) goto fail;
  *out = m;
  return 0;

fail:
  bcir_host_deallocate(&m._allocator, spans);
  bcir_q8_model_free(&m);
  return -1;
}

int bcir_q8_model_load(const char *path, bcir_q8_model *out,
                       char *error, size_t error_capacity) {
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_q8_model_load_with_allocator_limited(
      path, out, error, error_capacity, &allocator, BCIR_Q8_DEFAULT_MAX_FILE_SIZE);
}

int bcir_q8_model_load_limited(const char *path, bcir_q8_model *out,
                               char *error, size_t error_capacity,
                               uint64_t max_file_size) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  return bcir_q8_model_load_with_allocator_limited(
      path, out, error, error_capacity, &allocator, max_file_size);
}

int bcir_q8_model_load_with_allocator(const char *path, bcir_q8_model *out,
                                      char *error, size_t error_capacity,
                                      const bcir_host_allocator *allocator) {
  return bcir_q8_model_load_with_allocator_limited(
      path, out, error, error_capacity, allocator, BCIR_Q8_DEFAULT_MAX_FILE_SIZE);
}
