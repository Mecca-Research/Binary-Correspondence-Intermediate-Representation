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
#include "bcir_verify.h"

/* R21 (§5.12): print one `R21 <func>: <kind>` line per use-after-free / double-free diagnostic. */
static void r21_print(const char *funcname, const char *kind, void *ctx) {
  (void)ctx; printf("R21 %s: %s\n", funcname, kind);
}

/* the directory of `path` (for #include/#embed resolution). */
static void dirof(const char *path, char *out, size_t cap) {
  const char *s = strrchr(path, '/');
  if (s) { size_t n = (size_t)(s - path); if (n >= cap) n = cap - 1; memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

int main(int argc, char **argv) {
  /* args: [--target <abi>] <c-source>. --target selects the data model (x86_64-linux default). */
  const char *path = NULL, *target = NULL;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--target")) { if (++i < argc) target = argv[i]; }
    else if (!strncmp(argv[i], "--target=", 9)) target = argv[i] + 9;
    else path = argv[i];
  }
  if (!path) { fprintf(stderr, "usage: %s [--target <abi>] <c-source>\n", argv[0]); return 2; }
  FILE *fp = fopen(path, "rb");
  if (!fp) { perror("fopen"); return 2; }
  static char raw[1 << 16];
  size_t n = fread(raw, 1, sizeof raw - 1, fp); raw[n] = 0; fclose(fp);

  /* L7: preprocess (macros / conditionals / #include / #embed) before the frontend. */
  static char src[1 << 16], cpperr[256], base[1024];
  dirof(path, base, sizeof base);
  if (bcir_cpp_run(raw, base, src, sizeof src, cpperr, sizeof cpperr)) { printf("CPP-ERR %s\n", cpperr); return 1; }

  static bcir_cfront_result r;
  /* free even on the compile-error path (not just success): the in-progress unit owns heap arrays, so an
   * error-path leak (Bug 3) is surfaced -- under the harness's LSan/detect_leaks=1 pass this orphaned
   * memory becomes a LeakSanitizer report, pinning the regression. */
  if (bcir_cfront_compile_target(src, target, &r) != 0) { printf("PARSE-ERR %s\n", r.diag); bcir_cfront_free(&r); return 1; }
  char sum[200]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
  printf("%s\n", sum);
  if (!r.ok) printf("diag: %s\n", r.diag);
  /* R21 advisory (§5.12): print use-after-free / double-free diagnostics before the emit marker. */
  bcir_verify_lifetime(&r.unit, r21_print, NULL);
  printf("----EMIT----\n%s", r.emitted);
  bcir_cfront_free(&r);
  return r.ok ? 0 : 1;
}
