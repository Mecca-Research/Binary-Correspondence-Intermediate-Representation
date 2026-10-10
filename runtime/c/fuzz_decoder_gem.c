/*===- fuzz_decoder_gem.c - libFuzzer harness for the decoder program interpreter ===
 *
 * bcir_decoder_gem.c executes a StreamPack as a decoder program: every claim's op string,
 * RIDs, block offset and count come from the pack, so all of them are attacker-controlled.
 * The input is a model and a program -- `[u32 LE model length][BCIRQ8 model][StreamPack]` --
 * seeded from real pairs (tools/c/fuzz_streampack.sh), so a mutation lands on the pack about
 * as often as the program is longer than the model; the model is reloaded only when its
 * bytes change, through a file, because the loader owns the read (fuzz_q8_model.c's pattern).
 * The pack is driven twice, as the BCIRQ8 harness drives its model: RAW (the CRC refusal stays
 * covered) and RESEALED -- its CRC trailer recomputed on a copy -- which is what reaches the
 * semantic walk, the claim checks and the run instead of dying at the checksum.
 *
 * The contract under test, for ANY input:
 *   - a load either succeeds or refuses with a status, and never reads or writes out of bounds;
 *   - a pack refused as malformed (E_PACK) was refused before anything was allocated: no count
 *     its header declares is believed before the semantic walk has read the records;
 *   - an accepted program runs to the same tokens and the same logits, bit for bit, every time
 *     it runs -- a run depends on the program, the prompt and the model alone;
 *   - every allocation is released (the counting allocator returns to zero).
 * The run is bounded (claims, tape, vocabulary) so an iteration stays cheap.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 -ffp-contract=off -I . \
 *       fuzz_decoder_gem.c bcir_decoder_gem.c bcir_llama.c bcir_q8_model.c bcir_q4_kernel.c \
 *       bcir_ai_kernels.c bcir_decode.c bcir_exec.c bcir_runtime.c -lm -o fuzz_decoder_gem
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "bcir_decoder_gem.h"

#define FUZZ_MAX_MODEL (64u * 1024u)
#define FUZZ_MAX_PACK (64u * 1024u)
#define FUZZ_MAX_CLAIMS 4096u
#define FUZZ_MAX_TAPE 64u

typedef struct counting {
  size_t live;  /* allocations not yet released */
  size_t total; /* allocations made since the last reset */
} counting;

static void *count_allocate(void *ctx, size_t size) {
  counting *c = (counting *)ctx;
  void *p = malloc(size ? size : 1u);
  if (p) { c->live++; c->total++; }
  return p;
}
static void *count_reallocate(void *ctx, void *old, size_t size) {
  counting *c = (counting *)ctx;
  void *p = realloc(old, size ? size : 1u);
  if (p && !old) { c->live++; c->total++; }
  return p;
}
static void count_deallocate(void *ctx, void *p) {
  counting *c = (counting *)ctx;
  if (p) { c->live--; free(p); }
}

static unsigned char g_model_bytes[FUZZ_MAX_MODEL];
static size_t g_model_len;
static int g_model_ok;
static bcir_q8_model g_model;
static char g_path[64];

/* The model these bytes describe, loaded through a file once per distinct byte string. */
static const bcir_q8_model *model_for(const uint8_t *bytes, size_t len) {
  if (len == g_model_len && memcmp(bytes, g_model_bytes, len) == 0)
    return g_model_ok ? &g_model : NULL;
  if (g_model_ok) bcir_q8_model_free(&g_model);
  g_model_ok = 0;
  memcpy(g_model_bytes, bytes, len);
  g_model_len = len;
  if (!g_path[0]) snprintf(g_path, sizeof g_path, "/tmp/bcir_fuzz_dgem_%ld.bin", (long)getpid());
  {
    FILE *f = fopen(g_path, "wb");
    char error[128];
    if (!f) abort();
    if (len && fwrite(bytes, 1, len, f) != len) abort();
    fclose(f);
    g_model_ok =
        bcir_q8_model_load_limited(g_path, &g_model, error, sizeof error, FUZZ_MAX_MODEL) == 0;
  }
  return g_model_ok ? &g_model : NULL;
}

/* One run on a prompt drawn from the input: its tokens and final logits. */
static int run_once(bcir_dgem *g, const int32_t *prompt, int32_t *tokens, double *logits) {
  return bcir_dgem_run(g, prompt, bcir_dgem_prompt_len(g), tokens,
                       bcir_dgem_capacity(g) - bcir_dgem_prompt_len(g), logits);
}

static void drive(const bcir_q8_model *model, const uint8_t *pack, size_t pack_len,
                  const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static uint8_t sealed[FUZZ_MAX_PACK];
  uint32_t model_len;
  const bcir_q8_model *model;
  const uint8_t *pack;
  size_t pack_len;
  if (size < 4u) return 0;
  model_len = (uint32_t)data[0] | (uint32_t)data[1] << 8 | (uint32_t)data[2] << 16 |
              (uint32_t)data[3] << 24;
  if (model_len > FUZZ_MAX_MODEL || model_len > size - 4u) return 0;
  model = model_for(data + 4, model_len);
  if (!model) return 0;
  pack = data + 4 + model_len;
  pack_len = size - 4u - model_len;
  drive(model, pack, pack_len, data, size); /* raw */
  if (pack_len >= 8u && pack_len <= FUZZ_MAX_PACK) {
    uint32_t crc;
    memcpy(sealed, pack, pack_len);
    crc = bcir_crc32(sealed, pack_len - 4u);
    sealed[pack_len - 4u] = (uint8_t)crc;
    sealed[pack_len - 3u] = (uint8_t)(crc >> 8);
    sealed[pack_len - 2u] = (uint8_t)(crc >> 16);
    sealed[pack_len - 1u] = (uint8_t)(crc >> 24);
    drive(model, sealed, pack_len, data, size); /* resealed */
  }
  return 0;
}

static void drive(const bcir_q8_model *model, const uint8_t *pack, size_t pack_len,
                  const uint8_t *data, size_t size) {
  counting count = {0, 0};
  bcir_host_allocator allocator;
  bcir_dgem g;
  int rc;
  allocator.context = &count;
  allocator.allocate = count_allocate;
  allocator.reallocate = count_reallocate;
  allocator.deallocate = count_deallocate;

  rc = bcir_dgem_load(&g, model, pack, pack_len, &allocator);
  if (rc == BCIR_DGEM_E_PACK && count.total != 0u) abort(); /* sized before it was read */
  if (rc == 0 && g.n_claims <= FUZZ_MAX_CLAIMS && bcir_dgem_capacity(&g) <= FUZZ_MAX_TAPE &&
      model->vocab_size <= 4096u) {
    int32_t prompt[FUZZ_MAX_TAPE], tokens[2][FUZZ_MAX_TAPE];
    double *logits[2];
    size_t i, n = bcir_dgem_prompt_len(&g), vocab = model->vocab_size;
    int r0, r1;
    for (i = 0; i < n; ++i) prompt[i] = (int32_t)((data[i % size] * 7u + i) % vocab);
    logits[0] = (double *)malloc(vocab * sizeof(double));
    logits[1] = (double *)malloc(vocab * sizeof(double));
    if (!logits[0] || !logits[1]) abort();
    r0 = run_once(&g, prompt, tokens[0], logits[0]);
    r1 = run_once(&g, prompt, tokens[1], logits[1]);
    if (r0 != r1) abort(); /* the same program refused one run and not the other */
    if (r0 == 0 &&
        (memcmp(tokens[0], tokens[1], (bcir_dgem_capacity(&g) - n) * sizeof(int32_t)) ||
         memcmp(logits[0], logits[1], vocab * sizeof(double))))
      abort(); /* a run depended on what an earlier run left */
    free(logits[0]);
    free(logits[1]);
  }
  bcir_dgem_free(&g);
  if (count.live != 0u) abort(); /* an allocation outlived the interpreter */
}
