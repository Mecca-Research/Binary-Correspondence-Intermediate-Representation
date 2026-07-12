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
  const char *s = strrchr(path, '/'), *bs = strrchr(path, '\\');
  if (!s || (bs && bs > s)) s = bs;
  if (s) { size_t n = (size_t)(s - path); if (n == 0) n = 1; else if (n == 2 && path[1] == ':') n++;
    if (n >= cap) n = cap - 1; memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

int main(int argc, char **argv) {
  /* args: [--target <abi>] [--canon] <c-source>. --target selects the data model (x86_64-linux
   * default); --canon prints the raw cross-rail canonical serialization (the byte-identity proof). */
  const char *path = NULL, *target = NULL; int canon = 0, emit_cpp = 0, emit_lf = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--target")) { if (++i < argc) target = argv[i]; }
    else if (!strncmp(argv[i], "--target=", 9)) target = argv[i] + 9;
    else if (!strcmp(argv[i], "--canon")) canon = 1;
    else if (!strcmp(argv[i], "--emit-cpp")) emit_cpp = 1;   /* dump the preprocessed text and exit (L7) */
    else if (!strcmp(argv[i], "--emit-link-flags")) emit_lf = 1;  /* B1: the derived linker flags, then exit */
    else path = argv[i];
  }
  if (!path) { fprintf(stderr, "usage: %s [--target <abi>] [--emit-cpp] <c-source>\n", argv[0]); return 2; }
  FILE *fp = fopen(path, "rb");
  if (!fp) { perror("fopen"); return 2; }
  static char raw[1 << 16];
  size_t n = fread(raw, 1, sizeof raw - 1, fp); raw[n] = 0; fclose(fp);

  /* L7: preprocess (macros / conditionals / #include / #embed) before the frontend. */
  static char src[1 << 16], cpperr[256], base[1024];
  dirof(path, base, sizeof base);
  if (bcir_cpp_run(raw, base, src, sizeof src, cpperr, sizeof cpperr)) { printf("CPP-ERR %s\n", cpperr); return 1; }
  /* --emit-cpp: the C-twin preprocessor's expansion verbatim, so the Python differential can compare the
   * C rail against the reference compiler (catching a Python<->C preprocessor divergence). No frontend. */
  if (emit_cpp) { fputs(src, stdout); return 0; }

  static bcir_cfront_result r;
  /* free even on the compile-error path (not just success): the in-progress unit owns heap arrays, so an
   * error-path leak (Bug 3) is surfaced -- under the harness's LSan/detect_leaks=1 pass this orphaned
   * memory becomes a LeakSanitizer report, pinning the regression. */
  if (bcir_cfront_compile_target(src, target, &r) != 0) { printf("PARSE-ERR %s\n", r.diag); bcir_cfront_free(&r); return 1; }
  /* --emit-link-flags (B1): the deduped, sorted linker flags the unit's external-call edges need, one
   * line (e.g. `-lm`; empty for a pure-integer unit). bcir/tests/test_c_cfront.py compares this against
   * the oracle's linkflags.derive_link_flags -- the dual-rail parity gate. */
  if (emit_lf) { static char lf[256]; bcir_cfront_link_flags(&r.unit, lf, sizeof lf);
    printf("%s\n", lf); bcir_cfront_free(&r); return 0; }
  if (canon) { static char cbuf[1 << 17]; bcir_cfront_canon(&r.unit, cbuf, sizeof cbuf);
    fputs(cbuf, stdout); bcir_cfront_free(&r); return 0; }
  if (!r.emitted_ok) { printf("EMIT-ERR emitted C exceeds result capacity\n"); bcir_cfront_free(&r); return 1; }
  char sum[256]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
  printf("%s\n", sum);
  if (!r.ok) printf("diag: %s\n", r.diag);
  /* R21 advisory (§5.12): print use-after-free / double-free diagnostics before the emit marker. */
  bcir_verify_lifetime(&r.unit, r21_print, NULL);
  printf("----EMIT----\n%s", r.emitted);
  int ok = r.ok;
  bcir_cfront_free(&r);
  return ok ? 0 : 1;
}
