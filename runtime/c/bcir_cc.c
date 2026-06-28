/*===- bcir_cc.c - the BCIR C compiler driver (a cc-like front for the production C rail) ---===
 *
 * Wraps the full plug-in C pipeline behind a `cc`-style command, no Python:
 *   file.c -> bcir_cpp (preprocess: -I search paths, -D defines, #include/#embed)
 *          -> bcir_cfront (lex / parse / lower / R1-R18 verify / C.2 attest)
 *          -> [bcir_plan -> bcir_hydrate] StreamPack
 *
 * The source file's own directory is always on the include search path, so a multi-file driver
 * with sibling headers builds with a normal-ish compile command. Host tool (libc).
 *
 *   bcir-cc [options] file.c ...
 *     -I <dir>         add an #include search dir (repeatable; -I<dir> too)
 *     -D name[=val]    predefine a macro (val defaults to 1); -U name undefines it
 *     -std=<std>       c23 (default) / c17 / c11   (sets __STDC_VERSION__)
 *     -E               preprocess only -> the expanded translation unit
 *     -o <file>        write output to <file> (binary for --emit-pack) instead of stdout
 *     --emit-c         emit the verified C (the C.2 output seam); a unit with a bounds-promoted
 *                      (masked) access pulls in "bcir_quarantine.h" -- link runtime/c/bcir_quarantine.c
 *     --emit-claimgraph  emit the structural summary + the per-function claim graph
 *     --emit-pack      emit the entry function's hydrated StreamPack (binary; use -o)
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_cfront.h"
#include "bcir_cpp.h"
#include "bcir_hydrate.h"
#include "bcir_plan.h"
#include "bcir_verify.h"

/* R21 (§5.12): print one `R21 <func>: <kind>` advisory line per use-after-free / double-free (ctx is FILE*). */
static void cc_r21_print(const char *funcname, const char *kind, void *ctx) {
  fprintf((FILE *)ctx, "R21 %s: %s\n", funcname, kind);
}

/* R21 lifetime policy (§5.12): how a detected use-after-free / double-free gates the compile.
 *   advisory -- the default: surfaced (with --emit-claimgraph) but never affects the verdict;
 *   fallback -- route the unit to the LLVM backend (exit 2), the --fallback contract;
 *   reject   -- a hard verify error (exit 1). The detection (the freed-set walk in
 * bcir_verify_lifetime) is unchanged; only the verdict the driver draws from it changes. */
typedef enum { R21_ADVISORY, R21_FALLBACK, R21_REJECT } r21_policy;

/* Count R21 findings and keep the first (func, kind) for the message; does NOT print -- the driver
 * emits a single summary line under a non-advisory policy. */
typedef struct { int count; char kind[32]; char func[64]; } r21_count_ctx;
static void cc_r21_count(const char *funcname, const char *kind, void *ctx) {
  r21_count_ctx *c = (r21_count_ctx *)ctx;
  if (c->count == 0) {
    snprintf(c->kind, sizeof c->kind, "%s", kind);
    snprintf(c->func, sizeof c->func, "%s", funcname);
  }
  c->count++;
}

#define MAXD 64

static const char *USAGE =
  "usage: bcir-cc [-I dir] [-D name[=val]] [-U name] [-std=c23] [-E] [-o out]\n"
  "               [--target abi] [--fallback] [--emit-c] [--emit-claimgraph] [--emit-pack] file.c ...\n"
  "  --target abi   data model to lay out for: x86_64-linux (default), aarch64-linux,\n"
  "                 riscv64-linux, x86_64-windows, i386-linux\n"
  "  --fallback     total compile: a construct outside the supported subset exits 2 with\n"
  "                 'fallback to LLVM backend: <phase>: <reason>' instead of a hard error\n"
  "  --r21 <policy> how a detected use-after-free / double-free (R21, §5.12) gates the compile:\n"
  "                 advisory (default; surfaced, never gates) | fallback (route to LLVM, exit 2)\n"
  "                 | reject (hard verify error, exit 1)\n"
  "  --emit-effects per-function module-scope effect footprints + the commutation matrix\n";

static void dirof(const char *path, char *out, size_t cap) {
  const char *s = strrchr(path, '/');
  if (s) { size_t n = (size_t)(s - path); if (n >= cap) n = cap - 1; memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

/* a -D spec "NAME" / "NAME=val" -> the "name body" form bcir_cpp seeds ("NAME 1" / "NAME val"). */
static void define_spec(const char *spec, char *out, size_t cap) {
  const char *eq = strchr(spec, '=');
  if (eq) snprintf(out, cap, "%.*s %s", (int)(eq - spec), spec, eq + 1);
  else snprintf(out, cap, "%s 1", spec);
}

static const char *std_version(const char *std) {
  if (!strcmp(std, "c11")) return "201112L";
  if (!strcmp(std, "c17") || !strcmp(std, "c18")) return "201710L";
  return "202311L";                                  /* c23 / c2x / default */
}

int main(int argc, char **argv) {
  const char *incdirs[MAXD]; int ninc = 0;
  char defbuf[MAXD][256]; const char *defs[MAXD]; int ndef = 0;
  const char *undefs[MAXD]; int nundef = 0;
  const char *files[256]; int nfiles = 0;
  const char *std = "c23", *out_path = NULL, *target = NULL;
  int pp_only = 0, emit_c = 0, emit_cg = 0, emit_pack = 0, emit_fx = 0, fallback = 0;
  r21_policy r21 = R21_ADVISORY;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "-E")) pp_only = 1;
    else if (!strcmp(a, "--target")) { if (++i < argc) target = argv[i]; }
    else if (!strncmp(a, "--target=", 9)) target = a + 9;
    else if (!strcmp(a, "--fallback")) fallback = 1;
    else if (!strcmp(a, "--r21") || !strncmp(a, "--r21=", 6)) {
      const char *v = a[5] == '=' ? a + 6 : (++i < argc ? argv[i] : "");
      if (!strcmp(v, "advisory")) r21 = R21_ADVISORY;
      else if (!strcmp(v, "fallback")) r21 = R21_FALLBACK;
      else if (!strcmp(v, "reject")) r21 = R21_REJECT;
      else { fprintf(stderr, "bcir-cc: unknown --r21 policy '%s' (advisory|fallback|reject)\n", v); return 2; }
    }
    else if (!strcmp(a, "--emit-effects")) emit_fx = 1;
    else if (!strcmp(a, "--emit-c")) emit_c = 1;
    else if (!strcmp(a, "--emit-claimgraph")) emit_cg = 1;
    else if (!strcmp(a, "--emit-pack")) emit_pack = 1;
    else if (!strcmp(a, "-o")) { if (++i < argc) out_path = argv[i]; }
    else if (!strncmp(a, "-o", 2)) out_path = a + 2;
    else if (!strcmp(a, "-I")) { if (++i < argc && ninc < MAXD) incdirs[ninc++] = argv[i]; }
    else if (!strncmp(a, "-I", 2)) { if (ninc < MAXD) incdirs[ninc++] = a + 2; }
    else if (!strcmp(a, "-D")) { if (++i < argc && ndef < MAXD) { define_spec(argv[i], defbuf[ndef], 256); defs[ndef] = defbuf[ndef]; ndef++; } }
    else if (!strncmp(a, "-D", 2)) { if (ndef < MAXD) { define_spec(a + 2, defbuf[ndef], 256); defs[ndef] = defbuf[ndef]; ndef++; } }
    else if (!strcmp(a, "-U")) { if (++i < argc && nundef < MAXD) undefs[nundef++] = argv[i]; }
    else if (!strncmp(a, "-U", 2)) { if (nundef < MAXD) undefs[nundef++] = a + 2; }
    else if (!strncmp(a, "-std=", 5)) std = a + 5;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(USAGE, stdout); return 0; }
    else if (a[0] == '-') { fprintf(stderr, "bcir-cc: unknown option '%s'\n%s", a, USAGE); return 2; }
    else if (nfiles < 256) files[nfiles++] = a;
  }
  if (!nfiles) { fputs(USAGE, stderr); return 2; }

  /* -std seeds __STDC_VERSION__ (a -D can still override it); honour -U by dropping a -D. */
  char stdver[64]; snprintf(stdver, sizeof stdver, "__STDC_VERSION__ %s", std_version(std));
  for (int u = 0; u < nundef; u++)
    for (int d = 0; d < ndef; d++)
      if (!strncmp(defs[d], undefs[u], strlen(undefs[u])) && defs[d][strlen(undefs[u])] == ' ')
        defs[d] = "";                                /* "" -> define_macro ignores (no name) */

  FILE *outf = stdout;
  if (out_path && (outf = fopen(out_path, emit_pack ? "wb" : "w")) == NULL) {
    fprintf(stderr, "bcir-cc: cannot open output '%s'\n", out_path); return 2;
  }

  int rc = 0;
  for (int fi = 0; fi < nfiles; fi++) {
    const char *path = files[fi];
    static char raw[1 << 16]; FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "bcir-cc: cannot open '%s'\n", path); rc = 2; continue; }
    size_t n = fread(raw, 1, sizeof raw - 1, fp); raw[n] = 0; fclose(fp);

    char base[1024]; dirof(path, base, sizeof base);
    const char *dirs[MAXD + 1]; int ndirs = 0;
    dirs[ndirs++] = base;                            /* the source dir first (quoted includes) */
    for (int d = 0; d < ninc && ndirs <= MAXD; d++) dirs[ndirs++] = incdirs[d];
    const char *alldefs[MAXD + 1]; int nalldef = 0;
    alldefs[nalldef++] = stdver;
    for (int d = 0; d < ndef && nalldef <= MAXD; d++) alldefs[nalldef++] = defs[d];

    static char src[1 << 16], cpperr[256];
    if (bcir_cpp_run_ex(raw, path, dirs, ndirs, alldefs, nalldef, src, sizeof src, cpperr, sizeof cpperr)) {
      /* --fallback: a construct outside the supported subset routes to the LLVM backend (rc 2),
       * the C twin of pipeline.compile_with_fallback (which classifies the rejecting phase). */
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: preprocess: %s\n", path, cpperr); rc = 2; continue; }
      fprintf(stderr, "%s: preprocessor error: %s\n", path, cpperr); rc = 1; continue;
    }
    if (pp_only) { fputs(src, outf); continue; }

    static bcir_cfront_result r;
    if (bcir_cfront_compile_target(src, target, &r) != 0) {
      /* free the partial unit on the compile-error path too (the success path frees at the loop foot): `r`
       * is a reused static, so a skipped free here would leak the in-progress unit across files (the next
       * call's entry memset zeroes out->unit.funcs without freeing it). */
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: compile: %s\n", path, r.diag); rc = 2; bcir_cfront_free(&r); continue; }
      fprintf(stderr, "%s: parse error: %s\n", path, r.diag); rc = 1; bcir_cfront_free(&r); continue;
    }
    if (!r.ok) { fprintf(stderr, "%s: verify error: %s\n", path, r.diag); rc = 1; bcir_cfront_free(&r); continue; }

    /* R21 lifetime policy (§5.12): a detected use-after-free / double-free routes the unit to the
     * LLVM backend (fallback, rc 2) or hard-rejects it (rc 1) under a non-advisory policy. The
     * detection is the same freed-set walk surfaced advisory below; only the verdict changes.
     * Parity: bcir/frontends/cfront/__main__.py applies the identical policy + exit codes. */
    if (r21 != R21_ADVISORY) {
      r21_count_ctx lc = {0, "", ""};
      bcir_verify_lifetime(&r.unit, cc_r21_count, &lc);
      if (lc.count > 0) {
        if (r21 == R21_FALLBACK) { fprintf(stderr, "%s: fallback to LLVM backend: lifetime: R21 %s in %s\n", path, lc.kind, lc.func); rc = 2; }
        else { fprintf(stderr, "%s: lifetime error: R21 %s in %s\n", path, lc.kind, lc.func); rc = 1; }
        bcir_cfront_free(&r);
        continue;
      }
    }

    if (emit_c) {
      /* §5.12: a masked (bounds-promoted) access emits `a[BCIR_CHK(...)]`, which references the
       * bounds-quarantine runtime ABI. Make the driver's output a self-contained translation unit by
       * pulling in the runtime header -- the user links runtime/c/bcir_quarantine.c (or overrides the
       * weak handler). A unit with no masked access needs nothing, so the include is conditional. */
      if (strstr(r.emitted, "BCIR_CHK"))
        fputs("#include \"bcir_quarantine.h\"\n", outf);
      fputs(r.emitted, outf);
    } else if (emit_fx) {
      static char fx[8192]; bcir_cfront_effects(&r.unit, fx, sizeof fx);
      fputs(fx, outf);
    } else if (emit_cg) {
      char sum[256]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
      fprintf(outf, "%s\n", sum);
      for (int f = 0; f < r.unit.n_funcs; f++) { const bcir_func *fn = &r.unit.funcs[f];
        fprintf(outf, "function %s (%zu claims):\n", fn->name, fn->n_claims);
        for (size_t c = 0; c < fn->n_claims; c++)
          fprintf(outf, "  c%zu  %-12s lane=%u dom=%d\n", c, fn->claims[c].op,
                  fn->claims[c].lane, (int)fn->claims[c].domain);
      }
      bcir_verify_lifetime(&r.unit, cc_r21_print, outf);   /* R21 advisory (§5.12), additive to the call graph */
    } else if (emit_pack) {
      const bcir_func *f = &r.unit.funcs[r.unit.n_funcs - 1];   /* the entry function */
      static bcir_plan_step steps[8192]; bcir_plan plan;
      if (bcir_plan_func(f, steps, 8192, &plan) != BCIR_OK) { fprintf(stderr, "%s: plan error\n", path); rc = 1; }
      else { static uint8_t pack[1 << 20]; size_t plen = 0;
        if (bcir_hydrate(f, &plan, pack, sizeof pack, &plen) != BCIR_OK) { fprintf(stderr, "%s: hydrate error\n", path); rc = 1; }
        else fwrite(pack, 1, plen, outf); }
    } else {
      char sum[256]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
      fprintf(outf, "%s: %s\n", path, sum);
    }
    bcir_cfront_free(&r);
  }

  if (outf != stdout) fclose(outf);
  return rc;
}
