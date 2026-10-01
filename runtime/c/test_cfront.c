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
    if (n >= cap) n = cap - 1;
    memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

/* The whole source, NUL-terminated, in a block the caller releases (CF-LIMITS): the driver kept the first 64 KiB of
 * a file and lowered that prefix without saying so. The block grows two-phase through the host allocator, each size
 * checked, to the bound `bcir-cc` holds a source to (as its `cc_read_file` does); a file that cannot be handed on
 * whole -- past that bound, unreadable, or holding a NUL a C string would stop at -- is refused, never cut. NULL on
 * success, else the reason. */
#define TEST_CFRONT_MAX_SOURCE ((size_t)64u << 20)
static const char *read_source(FILE *fp, const bcir_host_allocator *heap, char **out) {
  size_t cap = (size_t)1u << 16, n = 0;
  char *buf = (char *)bcir_host_allocate(heap, cap);
  *out = NULL;
  if (!buf) return "out of memory";
  for (;;) {
    if (n == cap - 1u) {                    /* full: past the bound one byte more is refused, else the block grows */
      size_t want, grown;
      if (cap >= TEST_CFRONT_MAX_SOURCE + 1u) {
        int extra = fgetc(fp);
        if (extra == EOF && !ferror(fp)) break;
        bcir_host_deallocate(heap, buf);
        return extra == EOF ? "cannot read the source" : "the source is larger than 64 MiB";
      }
      if (!bcir_size_add(cap, 1u, &want) || !bcir_host_grow_capacity(cap, want, 1u, &grown)) {
        bcir_host_deallocate(heap, buf); return "out of memory"; }
      if (grown > TEST_CFRONT_MAX_SOURCE + 1u) grown = TEST_CFRONT_MAX_SOURCE + 1u;
      if (!bcir_host_realloc_array(heap, (void **)&buf, cap, grown, 1u, 0)) {
        bcir_host_deallocate(heap, buf); return "out of memory"; }
      cap = grown;
    }
    size_t avail = cap - 1u - n, got = fread(buf + n, 1, avail, fp);
    n += got;
    if (got < avail) {
      if (ferror(fp) || (!feof(fp) && !got)) { bcir_host_deallocate(heap, buf); return "cannot read the source"; }
      if (feof(fp)) break;
    }
  }
  if (memchr(buf, 0, n)) { bcir_host_deallocate(heap, buf); return "the source holds a NUL byte"; }
  buf[n] = 0; *out = buf;
  return NULL;
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
  bcir_host_allocator heap = bcir_host_allocator_default();
  char *raw = NULL;
  const char *why = read_source(fp, &heap, &raw);
  if (fclose(fp) && !why) why = "cannot read the source";
  if (why) { bcir_host_deallocate(&heap, raw); printf("READ-ERR %s\n", why); return 1; }

  /* L7: preprocess (macros / conditionals / #include / #embed) before the frontend. The preprocessed text is held in
   * 64 KiB, as `bcir-cc` holds it; a unit that runs past it is refused (`preprocessed output too large`), never cut. */
  static char src[1 << 16], cpperr[256], base[1024];
  dirof(path, base, sizeof base);
  int cpp_rc = bcir_cpp_run(raw, base, src, sizeof src, cpperr, sizeof cpperr);
  bcir_host_deallocate(&heap, raw);
  if (cpp_rc) { printf("CPP-ERR %s\n", cpperr); return 1; }
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
  if (canon) {   /* the whole canon, however long (CF-LIMITS): measured, held, then printed -- never a cut prefix */
    size_t need = bcir_cfront_canon(&r.unit, NULL, 0), cap = 0, got = SIZE_MAX;
    char *cbuf = need != SIZE_MAX && bcir_size_add(need, 1u, &cap) ? (char *)bcir_host_allocate(&heap, cap) : NULL;
    if (cbuf) got = bcir_cfront_canon(&r.unit, cbuf, cap);
    if (!cbuf || got != need) {
      printf("CANON-ERR no whole canon\n"); bcir_host_deallocate(&heap, cbuf); bcir_cfront_free(&r); return 1; }
    fwrite(cbuf, 1, need, stdout); bcir_host_deallocate(&heap, cbuf); bcir_cfront_free(&r); return 0; }
  /* defensive: a successful compile always emits, whole -- the emit grows with no fixed capacity (CF-BUF) */
  if (!r.emitted_ok) { printf("EMIT-ERR no emitted C\n"); bcir_cfront_free(&r); return 1; }
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
