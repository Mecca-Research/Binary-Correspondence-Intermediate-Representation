/*===- fuzz_q8_model.c - libFuzzer harness for the BCIRQ8 artifact loader ---===
 *
 * bcir_q8_model.c parses a BCIRQ8 v1 model file. That file is an EXTERNAL artifact --
 * downloaded, shipped alongside a binary, or produced by another tool -- so every field
 * it declares (header CRC, geometry, tensor count, directory offsets, per-tensor
 * offset/length spans, group counts, exponents) is attacker-controlled. It was the one
 * binary trust boundary in runtime/c/ with no fuzz harness.
 *
 * The contract under test: for ANY byte string, `bcir_q8_model_load_limited` either
 * returns 0 with a fully validated model, or returns -1 with every allocation released
 * -- and never reads or writes out of bounds, never leaks, and never leaves a partially
 * initialized model behind. On success, every tensor the directory declares must be
 * addressable, so this harness also walks `bcir_q8_tensor_value` over each tensor's
 * first/last/out-of-range indices, and probes the accessor with tensors it does not own.
 *
 * The loader takes a PATH (it owns the read), so the harness materializes each input in
 * a temp file. libFuzzer keeps this cheap: one reused path, truncated per iteration.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 \
 *       runtime/c/fuzz_q8_model.c runtime/c/bcir_q8_model.c -I runtime/c -o fuzz_q8_model
 *   ./fuzz_q8_model -runs=200000       (see tools/c/fuzz_streampack.sh)
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bcir_q8_model.h"

/* A generous ceiling so the fuzzer can reach the geometry/span checks, but bounded so a
 * hostile size field can never make the harness itself allocate without limit. */
#define FUZZ_Q8_MAX_FILE (8u * 1024u * 1024u)

/* Mirrors of the on-disk layout the harness must know to REPAIR checksums (below). Kept
 * local on purpose: if the format's offsets move, this harness must be revisited rather
 * than silently drift into fuzzing the wrong bytes. */
#define FUZZ_Q8_BODY_CRC_OFF   112u   /* u32: CRC over [header_size, file_end)          */
#define FUZZ_Q8_HDR_CRC_OFF    116u   /* u32: CRC over the header with this field zeroed */
#define FUZZ_Q8_TENSOR_CNT_OFF  80u
#define FUZZ_Q8_ENTRY_SIZE_OFF  84u
#define FUZZ_Q8_DIR_OFF_OFF     88u

static char g_path[64];

static const char *fuzz_path(void) {
  if (!g_path[0]) snprintf(g_path, sizeof g_path, "/tmp/bcir_fuzz_q8_%ld.bin", (long)getpid());
  return g_path;
}

static int write_input(const uint8_t *data, size_t size) {
  FILE *f = fopen(fuzz_path(), "wb");
  if (!f) return -1;
  int ok = (size == 0) || (fwrite(data, 1, size, f) == size);
  return (fclose(f) == 0 && ok) ? 0 : -1;
}

/* Exercise every accessor a consumer would touch on a loaded model, at the boundary
 * indices where an off-by-one would show up. */
static void walk_model(const bcir_q8_model *m) {
  uint32_t i;
  bcir_q8_tensor foreign;                          /* a tensor the model does not own */
  memset(&foreign, 0, sizeof foreign);
  for (i = 0; i < m->tensor_count; ++i) {
    const bcir_q8_tensor *t = &m->tensors[i];
    const bcir_q8_tensor *looked = bcir_q8_model_tensor(m, t->tensor_id, t->layer);
    if (!looked) continue;
    (void)bcir_q8_tensor_value(m, looked, 0u);
    if (t->element_count) {
      (void)bcir_q8_tensor_value(m, looked, t->element_count - 1u);
      (void)bcir_q8_tensor_value(m, looked, t->element_count / 2u);
    }
    (void)bcir_q8_tensor_value(m, looked, t->element_count);      /* one past the end */
    (void)bcir_q8_tensor_value(m, looked, UINT32_MAX);
  }
  /* A tensor pointer that is not an element of this model's directory must be rejected
   * (the delta/alignment gate), not dereferenced. */
  (void)bcir_q8_tensor_value(m, &foreign, 0u);
  (void)bcir_q8_tensor_value(m, NULL, 0u);
  (void)bcir_q8_model_tensor(m, UINT16_MAX, INT16_MIN);
  (void)bcir_q8_model_tensor(NULL, 0u, 0);
}

/* --- checksum repair: the only way to fuzz PAST the CRC gates ---------------------
 *
 * BCIRQ8 seals the header with a CRC and the whole body with a second one, and every
 * tensor carries a CRC over its own payload. Those are exactly the right thing for the
 * format and exactly the wrong thing for a coverage-guided fuzzer: a random mutation
 * invalidates a checksum long before it reaches the geometry, directory-offset, span-
 * overlap, exponent-range or canonical-inventory checks, so a naive campaign only ever
 * proves "the CRC check works" while the real parser stays unexplored.
 *
 * So each input is driven twice: once RAW (keeping the reject-on-bad-CRC path covered)
 * and once REPAIRED -- tensor CRCs, then the body CRC, then the header CRC, recomputed
 * in that order because each region contains the previous field. Every read here is
 * bounds-checked against `size`: the harness must never be the thing that overflows. */
static uint32_t crc32_of(const uint8_t *d, size_t n) {
  uint32_t s = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    s ^= d[i];
    for (int b = 0; b < 8; ++b) s = (s >> 1) ^ (0xEDB88320u & (0u - (s & 1u)));
  }
  return s ^ 0xFFFFFFFFu;
}

static uint32_t ld32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t ld64(const uint8_t *p) { return (uint64_t)ld32(p) | ((uint64_t)ld32(p + 4) << 32); }
static void st32(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24);
}

static void repair_checksums(uint8_t *b, size_t size) {
  if (size < BCIR_Q8_HEADER_SIZE) return;
  memcpy(b, "BCIRQ8\0\0", 8);                        /* keep the magic so we reach the CRCs */

  /* (1) per-tensor CRCs, for every directory entry that lies wholly inside the file and
   *     whose declared exponent/code spans do too. A hostile entry that does NOT fit is
   *     left alone on purpose -- rejecting it is the behaviour under test. */
  uint32_t n_tensors = ld32(b + FUZZ_Q8_TENSOR_CNT_OFF);
  uint32_t entry_size = ld32(b + FUZZ_Q8_ENTRY_SIZE_OFF);
  uint64_t dir_off = ld64(b + FUZZ_Q8_DIR_OFF_OFF);
  if (entry_size == BCIR_Q8_DIRECTORY_ENTRY_SIZE && dir_off == BCIR_Q8_HEADER_SIZE &&
      n_tensors <= (size - dir_off) / entry_size) {
    for (uint32_t i = 0; i < n_tensors; ++i) {
      uint8_t *e = b + dir_off + (uint64_t)i * entry_size;
      uint32_t elems = ld32(e + 16), groups = ld32(e + 20);
      uint64_t exp_off = ld64(e + 32), code_off = ld64(e + 40);
      uint64_t exp_len = (uint64_t)groups * 2u;
      if (exp_off > size || exp_len > size - exp_off) continue;
      if (code_off > size || elems > size - code_off) continue;
      uint32_t crc = 0xFFFFFFFFu;
      for (uint64_t k = 0; k < exp_len; ++k) {
        crc ^= b[exp_off + k];
        for (int t = 0; t < 8; ++t) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
      }
      for (uint64_t k = 0; k < elems; ++k) {
        crc ^= b[code_off + k];
        for (int t = 0; t < 8; ++t) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
      }
      st32(e + 24, crc ^ 0xFFFFFFFFu);
    }
  }

  /* (2) body CRC over [header_size, end) -- must precede the header CRC, because the
   *     body-CRC field itself lives inside the header-CRC region. */
  st32(b + FUZZ_Q8_BODY_CRC_OFF, crc32_of(b + BCIR_Q8_HEADER_SIZE, size - BCIR_Q8_HEADER_SIZE));

  /* (3) header CRC: computed over the header with its own field read as four zeroes. */
  uint32_t s = 0xFFFFFFFFu;
  for (size_t i = 0; i < FUZZ_Q8_HDR_CRC_OFF; ++i) {
    s ^= b[i];
    for (int t = 0; t < 8; ++t) s = (s >> 1) ^ (0xEDB88320u & (0u - (s & 1u)));
  }
  for (int z = 0; z < 4; ++z) {
    s ^= 0u;
    for (int t = 0; t < 8; ++t) s = (s >> 1) ^ (0xEDB88320u & (0u - (s & 1u)));
  }
  for (size_t i = FUZZ_Q8_HDR_CRC_OFF + 4u; i < BCIR_Q8_HEADER_SIZE; ++i) {
    s ^= b[i];
    for (int t = 0; t < 8; ++t) s = (s >> 1) ^ (0xEDB88320u & (0u - (s & 1u)));
  }
  st32(b + FUZZ_Q8_HDR_CRC_OFF, s ^ 0xFFFFFFFFu);
}

static void load_once(uint64_t limit) {
  bcir_q8_model model;
  char error[256];
  bcir_q8_model_init(&model);
  if (bcir_q8_model_load_limited(fuzz_path(), &model, error, sizeof error, limit) == 0)
    walk_model(&model);
  /* Either way the model must end up owning nothing: on the failure path the loader has
   * already released everything, so this free must be a safe no-op on a zeroed model. */
  bcir_q8_model_free(&model);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > FUZZ_Q8_MAX_FILE) return 0;

  /* (a) RAW: keeps the magic/version/CRC rejection path covered. */
  if (write_input(data, size)) return 0;
  load_once(FUZZ_Q8_MAX_FILE);
  /* The tiny-limit branch a memory-constrained driver would take must clean up too. */
  load_once(256u);

  /* (b) REPAIRED: drives the semantic validation behind the checksums. */
  if (size >= BCIR_Q8_HEADER_SIZE) {
    uint8_t *copy = (uint8_t *)malloc(size);
    if (!copy) return 0;
    memcpy(copy, data, size);
    repair_checksums(copy, size);
    if (!write_input(copy, size)) load_once(FUZZ_Q8_MAX_FILE);
    free(copy);
  }
  return 0;
}
