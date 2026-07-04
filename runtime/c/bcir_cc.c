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
 *     --emit-link-flags  emit just the linker flags the unit's external-call edges need (one line,
 *                      space-separated, e.g. `-lm`), for build-system consumption (B1)
 *     --project        print the per-PROJECT verdict line (CLEAN / PARTIAL-FALLBACK / DIRTY) over
 *                      the compiled file set; automatic for a multi-file invocation. The verdict
 *                      string, bucket rules and exit codes (a hard error 1 DOMINATES a fallback 2)
 *                      are byte/value-identical to the Python rail (bcir-cfront). The other half of
 *                      Phase 3 project mode -- compile_commands.json (-p) and -M dependency rules --
 *                      stays on the Python rail for now (build-system glue, not law).
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
  "               [--target abi] [--fallback] [--project] [--emit-c] [--emit-claimgraph]\n"
  "               [--emit-pack] [--emit-link-flags] file.c ...\n"
  "  --target abi   data model to lay out for: x86_64-linux (default), aarch64-linux,\n"
  "                 riscv64-linux, x86_64-windows, i386-linux\n"
  "  --fallback     total compile: a construct outside the supported subset exits 2 with\n"
  "                 'fallback to LLVM backend: <phase>: <reason>' instead of a hard error\n"
  "  --linkable     emit the LINKABLE artifact (Phase 3): external-linkage real-name\n"
  "                 definitions + derived #includes, so two emitted TUs link to each other\n"
  "  --project      print the per-PROJECT verdict line over the compiled file set -- CLEAN /\n"
  "                 PARTIAL-FALLBACK / DIRTY; printed automatically for a multi-file invocation\n"
  "                 (the compile-database half of project mode, -p, stays on the Python rail)\n"
  "  --r21 <policy> how a detected use-after-free / double-free (R21, §5.12) gates the compile:\n"
  "                 advisory (default; surfaced, never gates) | fallback (route to LLVM, exit 2)\n"
  "                 | reject (hard verify error, exit 1)\n"
  "  --emit-effects per-function module-scope effect footprints + the commutation matrix\n"
  "  --emit-link-flags  the linker flags the unit's external-call edges need (one space-separated\n"
  "                 line, e.g. -lm; empty for a pure-integer unit) -- for build-system consumption\n";

/* naive whole-text replace (pat -> rep, rep no longer than pat) into `out`. */
static void cc_replace_all(const char *in, const char *pat, const char *rep, char *out, size_t on) {
  size_t pl = strlen(pat), rl = strlen(rep), w = 0;
  while (*in) {
    if (!strncmp(in, pat, pl)) {
      if (w + rl < on) { memcpy(out + w, rep, rl); w += rl; }
      in += pl;
    } else if (w + 1 < on) out[w++] = *in++;
    else in++;
  }
  out[w] = 0;
}

/* Phase 3 linking: re-render the emitted unit under its REAL names with SOURCE linkage (the C
 * twin of the oracle's emit_linkable): a source-`static` definition KEEPS its `static` (internal
 * linkage honored -- two TUs may each carry a same-named static helper); every other definition's
 * unindented `static ` is stripped (body statics are indented and untouched); in-unit calls are
 * unprefixed, and the derived #includes prepended (stdint always; stdlib/stdio/math by the
 * unit's external-call edges; the quarantine header when a masked access needs it). Parity note:
 * functions only -- file-scope global DEFINITIONS stay oracle-side (per the roadmap split). */
static int cc_linkable_static_fn(const bcir_cfront_result *r, const char *line, size_t len) {
  /* Does this unindented line name a SOURCE-static function directly before some '('? The
   * definition line is `static RET NAME(` and the tudefs prelude line is `extern RET NAME(`
   * -- but a RET spelling may itself carry parens (`_BitInt(13)`), so try EVERY '(' and
   * match the identifier before it against the unit's static functions (a parameter NAME is
   * never followed by '(', so the scan cannot false-positive on parameter lists). */
  for (const char *par = memchr(line, '(', len); par;
       par = memchr(par + 1, '(', len - (size_t)(par + 1 - line))) {
    const char *b = par;
    while (b > line && ((b[-1] >= 'a' && b[-1] <= 'z') || (b[-1] >= 'A' && b[-1] <= 'Z') ||
                        (b[-1] >= '0' && b[-1] <= '9') || b[-1] == '_')) b--;
    if (b == par) continue;
    for (int f = 0; f < r->unit.n_funcs; f++)
      if (r->unit.funcs[f].static_fn &&
          strlen(r->unit.funcs[f].name) == (size_t)(par - b) &&
          !strncmp(b, r->unit.funcs[f].name, (size_t)(par - b))) return 1;
  }
  return 0;
}

static void cc_emit_linkable(const bcir_cfront_result *r, FILE *outf) {
  int need_math = 0, need_stdlib = 0, need_stdio = 0;
  for (int f = 0; f < r->unit.n_funcs; f++)
    for (size_t k = 0; k < r->unit.funcs[f].n_claims; k++) {
      const char *op = r->unit.funcs[f].claims[k].op;
      if (!strncmp(op, "c.call.libm:", 12) || !strncmp(op, "c.call.libm.void:", 17)) {
        const char *nm = strchr(op + 7, ':') + 1;
        if (!strcmp(nm, "malloc") || !strcmp(nm, "calloc") || !strcmp(nm, "realloc") ||
            !strcmp(nm, "free")) need_stdlib = 1;
        else need_math = 1;
      } else if (!strncmp(op, "c.call.extern:", 14)) need_stdio = 1;
    }
  fputs("#include <stdint.h>\n", outf);
  if (strstr(r->emitted, "BCIR_CHK")) fputs("#include \"bcir_quarantine.h\"\n", outf);
  if (need_stdlib) fputs("#include <stdlib.h>\n", outf);
  if (need_stdio) fputs("#include <stdio.h>\n", outf);
  if (need_math) fputs("#include <math.h>\n", outf);
  static char ba[sizeof ((bcir_cfront_result *)0)->emitted], bb[sizeof ba];
  const char *cur = r->emitted;
  char *nxt = ba;
  for (int f = 0; f < r->unit.n_funcs; f++) {        /* unprefix every in-unit call/def name */
    char pat[BCIR_CIR_NAME + 8], rep[BCIR_CIR_NAME + 2];
    snprintf(pat, sizeof pat, "bcir_%s(", r->unit.funcs[f].name);
    snprintf(rep, sizeof rep, "%s(", r->unit.funcs[f].name);
    cc_replace_all(cur, pat, rep, nxt, sizeof ba);
    cur = nxt;
    nxt = (nxt == ba) ? bb : ba;
  }
  const char *s = cur;                               /* strip the unindented definition `static ` --
                                                      * except a SOURCE-static function's (kept) */
  while (*s) {
    const char *nl = strchr(s, '\n');
    size_t len = nl ? (size_t)(nl - s) + 1 : strlen(s);
    if (!strncmp(s, "static ", 7) && !cc_linkable_static_fn(r, s, len))
      fwrite(s + 7, 1, len - 7, outf);
    else if (!strncmp(s, "extern ", 7) && cc_linkable_static_fn(r, s, len)) {
      /* the tudefs prelude declared this SOURCE-static function `extern` (the prototype was
       * parsed before its definition); the kept-static definition would then violate C11
       * 6.2.2p7 ("static declaration follows non-static") -- re-key the declaration to
       * `static` (same 7 bytes), the static-forward-declaration idiom the source used. */
      fputs("static ", outf);
      fwrite(s + 7, 1, len - 7, outf);
    } else fwrite(s, 1, len, outf);
    s += len;
  }
}

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
  int pp_only = 0, emit_c = 0, emit_cg = 0, emit_pack = 0, emit_fx = 0, emit_lf = 0, fallback = 0;
  int project = 0, linkable = 0;
  r21_policy r21 = R21_ADVISORY;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "-E")) pp_only = 1;
    else if (!strcmp(a, "--target")) { if (++i < argc) target = argv[i]; }
    else if (!strncmp(a, "--target=", 9)) target = a + 9;
    else if (!strcmp(a, "--fallback")) fallback = 1;
    else if (!strcmp(a, "--project")) project = 1;
    else if (!strcmp(a, "--r21") || !strncmp(a, "--r21=", 6)) {
      const char *v = a[5] == '=' ? a + 6 : (++i < argc ? argv[i] : "");
      if (!strcmp(v, "advisory")) r21 = R21_ADVISORY;
      else if (!strcmp(v, "fallback")) r21 = R21_FALLBACK;
      else if (!strcmp(v, "reject")) r21 = R21_REJECT;
      else { fprintf(stderr, "bcir-cc: unknown --r21 policy '%s' (advisory|fallback|reject)\n", v); return 2; }
    }
    else if (!strcmp(a, "--emit-effects")) emit_fx = 1;
    else if (!strcmp(a, "--emit-link-flags")) emit_lf = 1;
    else if (!strcmp(a, "--emit-c")) emit_c = 1;
    else if (!strcmp(a, "--linkable")) linkable = 1;
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

  /* Per-project outcome tally (Phase 3): every file lands in exactly one bucket, and the exit
   * code keeps the per-unit rules -- a hard error (1) DOMINATES a fallback (2), so a fallback
   * path never overwrites an earlier rc=1. Parity: bcir/frontends/cfront/__main__.py `outcomes`. */
  int rc = 0, n_clean = 0, n_fallback = 0, n_dirty = 0;
  for (int fi = 0; fi < nfiles; fi++) {
    const char *path = files[fi];
    static char raw[1 << 16]; FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "bcir-cc: cannot open '%s'\n", path); if (rc != 1) rc = 2; n_dirty++; continue; }
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
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: preprocess: %s\n", path, cpperr); if (rc != 1) rc = 2; n_fallback++; continue; }
      fprintf(stderr, "%s: preprocessor error: %s\n", path, cpperr); rc = 1; n_dirty++; continue;
    }
    if (pp_only) { fputs(src, outf); n_clean++; continue; }

    static bcir_cfront_result r;
    if (bcir_cfront_compile_target(src, target, &r) != 0) {
      /* free the partial unit on the compile-error path too (the success path frees at the loop foot): `r`
       * is a reused static, so a skipped free here would leak the in-progress unit across files (the next
       * call's entry memset zeroes out->unit.funcs without freeing it). */
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: compile: %s\n", path, r.diag); if (rc != 1) rc = 2; n_fallback++; bcir_cfront_free(&r); continue; }
      fprintf(stderr, "%s: parse error: %s\n", path, r.diag); rc = 1; n_dirty++; bcir_cfront_free(&r); continue;
    }
    if (!r.ok) { fprintf(stderr, "%s: verify error: %s\n", path, r.diag); rc = 1; n_dirty++; bcir_cfront_free(&r); continue; }

    /* R21 lifetime policy (§5.12): a detected use-after-free / double-free routes the unit to the
     * LLVM backend (fallback, rc 2) or hard-rejects it (rc 1) under a non-advisory policy. The
     * detection is the same freed-set walk surfaced advisory below; only the verdict changes.
     * Parity: bcir/frontends/cfront/__main__.py applies the identical policy + exit codes. */
    if (r21 != R21_ADVISORY) {
      r21_count_ctx lc = {0, "", ""};
      bcir_verify_lifetime(&r.unit, cc_r21_count, &lc);
      if (lc.count > 0) {
        if (r21 == R21_FALLBACK) { fprintf(stderr, "%s: fallback to LLVM backend: lifetime: R21 %s in %s\n", path, lc.kind, lc.func); if (rc != 1) rc = 2; n_fallback++; }
        else { fprintf(stderr, "%s: lifetime error: R21 %s in %s\n", path, lc.kind, lc.func); rc = 1; n_dirty++; }
        bcir_cfront_free(&r);
        continue;
      }
    }

    if (emit_lf) {
      /* B1: just the linker flags the unit's external-call edges need (one space-separated line, e.g.
       * `-lm`; empty for a pure-integer unit) -- for build-system consumption. Byte-identical to the
       * Python rail (bcir-cfront --emit-link-flags); parity gated in check_runtime.sh. */
      char lflags[256]; bcir_cfront_link_flags(&r.unit, lflags, sizeof lflags);
      fprintf(outf, "%s\n", lflags);
      n_clean++;
    } else if (linkable) {
      cc_emit_linkable(&r, outf);                    /* Phase 3: the externally-linkable artifact */
      n_clean++;
    } else if (emit_c) {
      /* §5.12: a masked (bounds-promoted) access emits `a[BCIR_CHK(...)]`, which references the
       * bounds-quarantine runtime ABI. Make the driver's output a self-contained translation unit by
       * pulling in the runtime header -- the user links runtime/c/bcir_quarantine.c (or overrides the
       * weak handler). A unit with no masked access needs nothing, so the include is conditional. */
      if (strstr(r.emitted, "BCIR_CHK"))
        fputs("#include \"bcir_quarantine.h\"\n", outf);
      fputs(r.emitted, outf);
      n_clean++;
    } else if (emit_fx) {
      static char fx[8192]; bcir_cfront_effects(&r.unit, fx, sizeof fx);
      fputs(fx, outf);
      n_clean++;
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
      n_clean++;
    } else if (emit_pack) {
      const bcir_func *f = &r.unit.funcs[r.unit.n_funcs - 1];   /* the entry function */
      static bcir_plan_step steps[8192]; bcir_plan plan;
      if (bcir_plan_func(f, steps, 8192, &plan) != BCIR_OK) { fprintf(stderr, "%s: plan error\n", path); rc = 1; n_dirty++; }
      else { static uint8_t pack[1 << 20]; size_t plen = 0;
        if (bcir_hydrate(f, &plan, pack, sizeof pack, &plen) != BCIR_OK) { fprintf(stderr, "%s: hydrate error\n", path); rc = 1; n_dirty++; }
        else { fwrite(pack, 1, plen, outf); n_clean++; } }
    } else {
      char sum[256]; bcir_cfront_summary(&r.unit, r.ok, sum, sizeof sum);
      fprintf(outf, "%s: %s\n", path, sum);
      n_clean++;
    }
    bcir_cfront_free(&r);
  }

  /* The per-PROJECT verdict (Phase 3): one line over the whole compiled file set. CLEAN = every
   * unit compiled clean; PARTIAL-FALLBACK = no failure but >=1 unit routed to LLVM (--fallback /
   * --r21=fallback); DIRTY = >=1 unit failed. Printed automatically for a multi-file invocation,
   * or on request (--project). BYTE-IDENTICAL to the Python rail (bcir-cfront, __main__.py);
   * parity gated in check_runtime.sh (#project). */
  if (project || nfiles > 1) {
    int n = n_clean + n_fallback + n_dirty;
    if (n_dirty) {
      if (n_fallback) fprintf(outf, "project: DIRTY (%d/%d failed, %d fell back)\n", n_dirty, n, n_fallback);
      else fprintf(outf, "project: DIRTY (%d/%d failed)\n", n_dirty, n);
    } else if (n_fallback) {
      fprintf(outf, "project: PARTIAL-FALLBACK (%d/%d routed to the LLVM backend)\n", n_fallback, n);
    } else {
      fprintf(outf, "project: CLEAN (%d file%s)\n", n, n == 1 ? "" : "s");
    }
  }

  if (outf != stdout) fclose(outf);
  return rc;
}
