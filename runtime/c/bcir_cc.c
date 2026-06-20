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
 *     --emit-c         emit the verified C (the C.2 output seam)
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

#define MAXD 64

static const char *USAGE =
  "usage: bcir-cc [-I dir] [-D name[=val]] [-U name] [-std=c23] [-E] [-o out]\n"
  "               [--target abi] [--fallback] [--emit-c] [--emit-claimgraph] [--emit-pack] file.c ...\n"
  "  --target abi   data model to lay out for: x86_64-linux (default), aarch64-linux,\n"
  "                 riscv64-linux, x86_64-windows, i386-linux\n"
  "  --fallback     total compile: a construct outside the supported subset exits 2 with\n"
  "                 'fallback to LLVM backend: <phase>: <reason>' instead of a hard error\n";

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
  int pp_only = 0, emit_c = 0, emit_cg = 0, emit_pack = 0, fallback = 0;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "-E")) pp_only = 1;
    else if (!strcmp(a, "--target")) { if (++i < argc) target = argv[i]; }
    else if (!strncmp(a, "--target=", 9)) target = a + 9;
    else if (!strcmp(a, "--fallback")) fallback = 1;
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
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: compile: %s\n", path, r.diag); rc = 2; continue; }
      fprintf(stderr, "%s: parse error: %s\n", path, r.diag); rc = 1; continue;
    }
    if (!r.ok) { fprintf(stderr, "%s: verify error: %s\n", path, r.diag); rc = 1; bcir_cfront_free(&r); continue; }

    if (emit_c) {
      fputs(r.emitted, outf);
    } else if (emit_cg) {
      char sum[256]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
      fprintf(outf, "%s\n", sum);
      for (int f = 0; f < r.unit.n_funcs; f++) { const bcir_func *fn = &r.unit.funcs[f];
        fprintf(outf, "function %s (%zu claims):\n", fn->name, fn->n_claims);
        for (size_t c = 0; c < fn->n_claims; c++)
          fprintf(outf, "  c%zu  %-12s lane=%u dom=%d\n", c, fn->claims[c].op,
                  fn->claims[c].lane, (int)fn->claims[c].domain);
      }
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
