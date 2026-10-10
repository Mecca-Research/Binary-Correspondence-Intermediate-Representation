/*===- test_decoder_gem.c - the decoder program interpreter's allocation contract ===
 *
 * Loads MODEL and PACK through bcir_dgem_load with a counting bcir_host_allocator and prints
 * what the load allocated:
 *
 *   status=<rc> allocations=<n> live_after_free=<m>
 *
 * bcir/tests/test_decoder_gem.py holds the contract with it: a pack the semantic walk refuses
 * (a header counting records the body does not carry) is refused with nothing allocated -- no
 * count is believed before the walk has read it -- and whatever a load allocates, accepted or
 * refused, bcir_dgem_free releases. A host harness (libc).
 *
 *   cc -std=c11 -ffp-contract=off -I . test_decoder_gem.c bcir_decoder_gem.c bcir_llama.c \
 *      bcir_q8_model.c bcir_q4_kernel.c bcir_ai_kernels.c bcir_decode.c bcir_exec.c \
 *      bcir_runtime.c -lm -o t && ./t model.bcirq8 pack.bin
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>

#include "bcir_decoder_gem.h"

typedef struct counting {
  size_t live, total;
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

int main(int argc, char **argv) {
  bcir_q8_model model;
  char error[256];
  FILE *f;
  long size;
  uint8_t *pack;
  counting count = {0, 0};
  bcir_host_allocator allocator;
  bcir_dgem g;
  int rc;
  if (argc != 3) { fprintf(stderr, "usage: %s MODEL PACK\n", argv[0]); return 2; }
  if (bcir_q8_model_load(argv[1], &model, error, sizeof error)) {
    fprintf(stderr, "model: %s\n", error);
    return 2;
  }
  f = fopen(argv[2], "rb");
  if (!f || fseek(f, 0, SEEK_END) || (size = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET)) return 2;
  pack = (uint8_t *)malloc((size_t)size);
  if (!pack || fread(pack, 1, (size_t)size, f) != (size_t)size) return 2;
  fclose(f);
  allocator.context = &count;
  allocator.allocate = count_allocate;
  allocator.reallocate = count_reallocate;
  allocator.deallocate = count_deallocate;
  rc = bcir_dgem_load(&g, &model, pack, (size_t)size, &allocator);
  printf("status=%d allocations=%zu", rc, count.total);
  bcir_dgem_free(&g);
  printf(" live_after_free=%zu\n", count.live);
  free(pack);
  bcir_q8_model_free(&model);
  return 0;
}
