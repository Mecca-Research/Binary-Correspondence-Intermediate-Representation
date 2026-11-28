/*===- fuzz_cpp.c - libFuzzer target for the C preprocessor (CSUB-1) -----------------------===
 *
 * A translation unit is text anyone hands the compiler, and the preprocessor reads it first. Since
 * CSUB-1 its macro engine keeps buffers grown from the context's allocator and a table of interned
 * hide sets from one expansion to the next. Every input runs three times: twice on one context
 * that every input shares, and once on a context made for it. All three must give one verdict:
 *   - the same text, NUL-terminated at the length reported;
 *   - or the same refusal, with nothing returned.
 * So nothing an expansion keeps reaches the next: a count or a table the engine failed to reset
 * would part the shared runs from the fresh one the first time it mattered. Two runs on the shared
 * context alone would not do; a leak that made every later unit refuse would refuse both alike.
 * `__DATE__` and `__TIME__` are frozen (SOURCE_DATE_EPOCH) so that the runs can be compared.
 *
 * Seeds are the cfront_*.c fixtures, the preprocessor corpus the two rails' differentials compare.
 *===----------------------------------------------------------------------===*/
#define _POSIX_C_SOURCE 200809L   /* setenv */
#include "bcir_cpp.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *what) {
  fprintf(stderr, "fuzz_cpp: %s\n", what);
  abort();
}

int LLVMFuzzerInitialize(int *argc, char ***argv) {
  (void)argc; (void)argv;
  if (setenv("SOURCE_DATE_EPOCH", "0", 1) != 0) die("cannot freeze __DATE__");
  return 0;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static bcir_cpp_context shared;
  static int ready;
  bcir_cpp_context fresh, *ctx[3];
  char *src, *out[3] = {NULL, NULL, NULL}, err[3][256];
  size_t len[3] = {0, 0, 0};
  int rc[3];
  if (!ready) {
    if (bcir_cpp_context_init(&shared, NULL) != 0) die("cannot make a context");
    ready = 1;
  }
  if (bcir_cpp_context_init(&fresh, NULL) != 0) die("cannot make a context");
  ctx[0] = &shared; ctx[1] = &shared; ctx[2] = &fresh;
  src = (char *)malloc(size + 1u);
  if (!src) die("out of memory");
  memcpy(src, data, size);
  src[size] = 0;                                      /* a NUL inside ends the unit there, as a C string */
  for (int k = 0; k < 3; k++) {
    err[k][0] = 0;
    rc[k] = bcir_cpp_run_ex_alloc_context(ctx[k], src, "fuzz.c", NULL, 0, NULL, 0, &out[k], &len[k], err[k],
                                          sizeof err[k]);
    if (rc[k] == 0 && (!out[k] || strlen(out[k]) != len[k])) die("a text not NUL-terminated at its length");
    if (rc[k] != 0 && (out[k] || !err[k][0])) die("a refusal that returned text or gave no reason");
  }
  for (int k = 1; k < 3; k++) {
    if (rc[k] != rc[0]) die(k == 1 ? "two runs on one context gave two verdicts" : "a fresh context gave another verdict");
    if (rc[0] == 0 && (len[k] != len[0] || memcmp(out[k], out[0], len[0]) != 0)) die("the runs gave two texts");
    if (rc[0] != 0 && strcmp(err[k], err[0]) != 0) die("the runs refused for two reasons");
  }
  for (int k = 0; k < 3; k++) bcir_host_deallocate(&ctx[k]->allocator, out[k]);
  bcir_cpp_context_destroy(&fresh);
  free(src);
  return 0;
}
