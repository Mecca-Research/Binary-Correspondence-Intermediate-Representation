/*===- test_cfront.c - host harness for the BCIR C frontend ----------------===
 * Reads a C source file (argv[1]), lowers it through bcir_cfront, verifies it, and
 * prints the canonical structural summary followed by the faithful emitted C (after a
 * marker). bcir/tests/test_c_cfront.py lowers the same source through the Python oracle
 * and asserts (a) the summaries agree -- the Python<->C dual-rail parity gate -- and
 * (b) the emitted C is behaviour-equivalent to the source under Clang. Host harness.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bcir_cfront.h"
#include "bcir_cpp.h"

/* the directory of `path` (for #include/#embed resolution). */
static void dirof(const char *path, char *out, size_t cap) {
  const char *s = strrchr(path, '/');
  if (s) { size_t n = (size_t)(s - path); if (n >= cap) n = cap - 1; memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <c-source>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen"); return 2; }
  static char raw[1 << 16];
  size_t n = fread(raw, 1, sizeof raw - 1, fp); raw[n] = 0; fclose(fp);

  /* L7: preprocess (macros / conditionals / #include / #embed) before the frontend. */
  static char src[1 << 16], cpperr[256], base[1024];
  dirof(argv[1], base, sizeof base);
  if (bcir_cpp_run(raw, base, src, sizeof src, cpperr, sizeof cpperr)) { printf("CPP-ERR %s\n", cpperr); return 1; }

  static bcir_cfront_result r;
  if (bcir_cfront_compile(src, &r) != 0) { printf("PARSE-ERR %s\n", r.diag); return 1; }
  char sum[200]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
  printf("%s\n", sum);
  if (!r.ok) printf("diag: %s\n", r.diag);
  printf("----EMIT----\n%s", r.emitted);
  bcir_cfront_free(&r);
  return r.ok ? 0 : 1;
}
