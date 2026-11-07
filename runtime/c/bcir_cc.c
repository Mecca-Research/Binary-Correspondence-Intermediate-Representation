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
 *     --emit-effects   emit each function's read/write footprint + the commutation matrix (G10)
 *     --emit-escape    emit each function's named locals by escape verdict + its indirect calls'
 *                      narrowed target sets (G10); both byte-identical to bcir/frontends/cfront/escape.py
 *     --emit-uart      attest the UART RegMap contract a unit embeds (the UART blueprint's U0): its
 *                      `uart_regmap_contract` literal and `uart_target_baud`, as the twin's linkable emit
 *                      spells them, held to the generated registry (uart_placements.inc) with the oracle's
 *                      messages (bcir/frontends/devices/uart_laws.py, uart_unit_diagnostics' attestation)
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
#include "bcir_host_alloc.h"
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
  "               [--emit-pack] [--emit-link-flags] [--emit-effects] [--emit-escape] file.c ...\n"
  "  --target abi   data model to lay out for: x86_64-linux (default), aarch64-linux,\n"
  "                 riscv64-linux, x86_64-windows, i386-linux\n"
  "  --fallback     total compile: a construct outside the supported subset exits 2 with\n"
  "                 'fallback to LLVM backend: <phase>: <reason>' instead of a hard error\n"
  "  --linkable     emit the LINKABLE artifact (Phase 3): the unit's types, every function\n"
  "                 declared and defined under its real name with its source linkage, its\n"
  "                 globals defined with their initializers and the #includes its text names,\n"
  "                 so emitted TUs link to each other\n"
  "  --project      print the per-PROJECT verdict line over the compiled file set -- CLEAN /\n"
  "                 PARTIAL-FALLBACK / DIRTY; printed automatically for a multi-file invocation\n"
  "                 (the compile-database half of project mode, -p, stays on the Python rail)\n"
  "  --r21 <policy> how a detected use-after-free / double-free (R21, §5.12) gates the compile:\n"
  "                 advisory (default; surfaced, never gates) | fallback (route to LLVM, exit 2)\n"
  "                 | reject (hard verify error, exit 1)\n"
  "  --emit-effects per-function read/write footprints (globals, statics, `*` for unnamed memory)\n"
  "                 + the commutation matrix\n"
  "  --emit-escape  per-function named locals by escape verdict (nonescaping / lent / escaping)\n"
  "                 + each indirect call's narrowed target set (`*` = may reach unknown code)\n"
  "  --emit-link-flags  the linker flags the unit's external-call edges need (one space-separated\n"
  "                 line, e.g. -lm; empty for a pure-integer unit) -- for build-system consumption\n"
  "  --emit-uart    the unit's UART RegMap contract attested against the U0 registry: one line naming\n"
  "                 the placement (none / unattested), then each R3-uart message\n";

/* --- --emit-uart: the UART RegMap contract, attested by the twin ------------------------------------ */

/* The U0 registry (bcir/frontends/devices/uart_model.py generates uart_placements.inc; `--check` holds it
 * current), sorted by name: the order the unknown-placement message lists it in. */
typedef struct { const char *name; unsigned long long stride, base, fcr_reset, ref_clk_hz; } uart_place;
static const uart_place uart_places[] = {
#define UART_PLACEMENT(n, s, b, f, r) {n, s, b, f, r},
#include "uart_placements.inc"
#undef UART_PLACEMENT
};

/* The file-scope global `name`'s initializer as the linkable artifact spells it -- the twin's own rendering,
 * which is the oracle's `ginit` text (CF-LINKTWIN): its first byte, its length in *len; NULL when the artifact
 * defines no such global with an initializer. The artifact defines a global on one line at column 0,
 * `<declarator> = <initializer>;`, the declarator `[static |extern ][_Thread_local ]<type> <name>[<dims>]`
 * (a function pointer behind a typedef) and never holding a `{` -- which every aggregate or enum body
 * before an ` = ` does. The declarator's name is its first identifier not followed by another word or a
 * `(`: the tag of `struct uart_regmap_contract uart_regmap_contract` is followed by the name. Spans of the
 * artifact, never copies: no initializer is too long to attest. */
static int uart_is_ident(char ch) {
  return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '_';
}
static const char *uart_global_init(const char *text, const char *name, size_t *len) {
  size_t n = strlen(name);
  for (const char *line = text; line && *line;) {
    const char *end = strchr(line, '\n');
    const char *stop = end ? end : line + strlen(line);
    const char *eq = NULL;
    for (const char *p = line; p + 3 <= stop; p++)
      if (p[0] == ' ' && p[1] == '=' && p[2] == ' ') { eq = p; break; }
    if (eq && line[0] != ' ' && line[0] != '\t' && stop > eq + 3 && stop[-1] == ';' &&
        !memchr(line, '{', (size_t)(eq - line))) {
      const char *declared = NULL;
      size_t dlen = 0;
      for (const char *p = line; p < eq && !declared;) {
        if (!uart_is_ident(*p) || (*p >= '0' && *p <= '9')) { p++; continue; }
        const char *id = p;
        while (p < eq && uart_is_ident(*p)) p++;
        const char *q = p;
        while (q < eq && *q == ' ') q++;
        if (q == eq || *q == '[' || *q == ')' || *q == ';') { declared = id; dlen = (size_t)(p - id); }
      }
      if (declared && dlen == n && !strncmp(declared, name, n)) {
        *len = (size_t)(stop - 1 - (eq + 3));
        return eq + 3;
      }
    }
    line = end ? end + 1 : NULL;
  }
  return NULL;
}

/* The oracle's literal grammar (uart_laws._LITERAL) over the n bytes at s: {"<name>", <dec>, <dec>, <dec>,
 * <dec>}, the name a lower-case C identifier (*name, *name_len: a span of s). 0 when the text is not exactly
 * that, or a number passes 64 bits -- which no C constant the rendering re-spells does. */
static int uart_parse_literal(const char *s, size_t n, const char **name, size_t *name_len,
                              unsigned long long v[4]) {
  const char *e = s + n;
  if (e - s < 3 || *s++ != '{' || *s++ != '"' || !(*s >= 'a' && *s <= 'z')) return 0;
  *name = s;
  while (s < e && ((*s >= 'a' && *s <= 'z') || (*s >= '0' && *s <= '9') || *s == '_')) s++;
  *name_len = (size_t)(s - *name);
  if (s == e || *s++ != '"') return 0;
  for (int i = 0; i < 4; i++) {
    unsigned long long x = 0;
    if (e - s < 3 || *s++ != ',' || *s++ != ' ' || !(*s >= '0' && *s <= '9')) return 0;
    while (s < e && *s >= '0' && *s <= '9') {
      if (x > (~0ULL - 9) / 10) return 0;
      x = x * 10 + (unsigned)(*s++ - '0');
    }
    v[i] = x;
  }
  return s + 1 == e && *s == '}';
}

/* The n bytes at s as the attestation's messages spell them (uart_laws._spelled: repr() of the text's UTF-8
 * bytes): single quotes unless they hold one and no double quote, the quote and the backslash escaped,
 * \n \r \t, any other control byte and every byte past ASCII \xNN -- exact without a Unicode table. */
static void uart_repr(FILE *outf, const char *s, size_t n) {
  char q = (memchr(s, '\'', n) && !memchr(s, '"', n)) ? '"' : '\'';
  fputc(q, outf);
  for (const unsigned char *p = (const unsigned char *)s; p < (const unsigned char *)s + n; p++) {
    if (*p == '\\' || *p == (unsigned char)q) { fputc('\\', outf); fputc(*p, outf); }
    else if (*p == '\n') fputs("\\n", outf);
    else if (*p == '\r') fputs("\\r", outf);
    else if (*p == '\t') fputs("\\t", outf);
    else if (*p < 0x20 || *p >= 0x7f) fprintf(outf, "\\x%02x", *p);
    else fputc(*p, outf);
  }
  fputc(q, outf);
}

/* The twin of uart_laws.uart_contract_attestation, message for message and in its order: a unit with no
 * `uart_regmap_contract` is no UART unit (none, no message); a literal the grammar refuses, or a placement
 * the registry does not hold, is unattested; otherwise the stride and base offset (the RegMap law), the FCR
 * reset and the reference clock are held to the placement, and `uart_target_baud`, where the unit defines
 * one, must be a positive decimal constant. Advisory, as the oracle's R3-uart is: the report carries the
 * verdict and the exit code stays the compile's. The five U1 protocol laws are the oracle's alone (a later
 * slice: the blueprint's section 11). */
static void uart_attest(const char *path, const char *text, FILE *outf) {
  size_t n = 0, name_len = 0;
  const char *lit = uart_global_init(text, "uart_regmap_contract", &n), *name = NULL;
  unsigned long long v[4];
  const uart_place *hit = NULL;
  size_t n_places = sizeof uart_places / sizeof uart_places[0];
  if (!lit) {
    fprintf(outf, "%s: uart contract none\n", path);
    return;
  }
  if (!uart_parse_literal(lit, n, &name, &name_len, v)) {
    fprintf(outf, "%s: uart contract unattested\n  R3-uart: the uart_regmap_contract literal ", path);
    uart_repr(outf, lit, n);
    fputs(" is not a U0 header's\n", outf);
    return;
  }
  for (size_t i = 0; i < n_places; i++)
    if (strlen(uart_places[i].name) == name_len && !strncmp(uart_places[i].name, name, name_len))
      hit = &uart_places[i];
  if (!hit) {
    fprintf(outf, "%s: uart contract unattested\n  R3-uart: regmap contract: unknown placement '%.*s' (registry: ",
            path, (int)name_len, name);
    for (size_t i = 0; i < n_places; i++) fprintf(outf, "%s%s", i ? ", " : "", uart_places[i].name);
    fputs(")\n", outf);
    return;
  }
  fprintf(outf, "%s: uart contract %s\n", path, hit->name);
  if (v[0] != hit->stride)
    fprintf(outf, "  R3-uart: regmap contract: stride: contract 0x%llx != placement 0x%llx\n", v[0], hit->stride);
  if (v[1] != hit->base)
    fprintf(outf, "  R3-uart: regmap contract: base offset: contract 0x%llx != placement 0x%llx\n", v[1], hit->base);
  if (v[2] != hit->fcr_reset)
    fprintf(outf, "  R3-uart: regmap contract: fcr_reset 0x%02llx != placement 0x%02llx\n", v[2], hit->fcr_reset);
  if (v[3] != hit->ref_clk_hz)
    fprintf(outf, "  R3-uart: regmap contract: ref_clk_hz %llu != placement %llu\n", v[3], hit->ref_clk_hz);
  const char *target = uart_global_init(text, "uart_target_baud", &n);
  if (target) {
    int decimal = n != 0, positive = 0;
    for (size_t i = 0; i < n; i++) {
      if (target[i] < '0' || target[i] > '9') decimal = 0;
      else if (target[i] != '0') positive = 1;
    }
    if (!decimal || !positive) {
      fputs("  R3-uart: uart_target_baud ", outf);
      uart_repr(outf, target, n);
      fputs(" is not a positive integer constant\n", outf);
    }
  }
}

static void dirof(const char *path, char *out, size_t cap) {
  const char *s = strrchr(path, '/'), *bs = strrchr(path, '\\');
  if (!s || (bs && bs > s)) s = bs;                 /* native Windows paths use backslashes */
  if (s) { size_t n = (size_t)(s - path); if (n == 0) n = 1; else if (n == 2 && path[1] == ':') n++;
    if (n >= cap) n = cap - 1;
    memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

/* Read a whole translation unit through the C front's one source reader (`bcir_cpp_read_source`: CF-LIMITS,
 * CF-PPLIMITS), never a prefix. Returns 0 on success, 1 when the path cannot be opened, and 2 when the file cannot be
 * handed on whole -- past 64 MiB, unreadable, or holding a NUL. */
static int cc_read_file(const char *path, char **out) {
  FILE *fp=fopen(path,"rb");
  bcir_host_allocator allocator=bcir_host_allocator_default();
  const char *why;
  *out=NULL;if(!fp)return 1;
  why=bcir_cpp_read_source(fp,&allocator,out);
  if(fclose(fp)&&!why)why="cannot read the source";
  if(why){bcir_host_deallocate(&allocator,*out);*out=NULL;return 2;}
  return 0;
}

static void cc_release_file(char *contents) {
  bcir_host_allocator allocator=bcir_host_allocator_default();
  bcir_host_deallocate(&allocator,contents);
}

/* a -D spec "NAME" / "NAME=val" -> the "name body" form bcir_cpp seeds ("NAME 1" / "NAME val"). */
static int define_spec(const char *spec, char *out, size_t cap) {
  const char *eq = strchr(spec, '=');
  int n;
  if (!spec[0] || (eq && eq == spec)) return 1;
  if (eq) n = snprintf(out, cap, "%.*s %s", (int)(eq - spec), spec, eq + 1);
  else n = snprintf(out, cap, "%s 1", spec);
  return n < 0 || (size_t)n >= cap;
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
  int pp_only = 0, emit_c = 0, emit_cg = 0, emit_pack = 0, emit_fx = 0, emit_esc = 0, emit_lf = 0, fallback = 0;
  int emit_uart = 0;
  int project = 0, linkable = 0;
  r21_policy r21 = R21_ADVISORY;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (!strcmp(a, "-E")) pp_only = 1;
    else if (!strcmp(a, "--target")) { if (++i < argc) target = argv[i]; else { fputs("bcir-cc: --target needs an argument\n",stderr); return 2; } }
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
    else if (!strcmp(a, "--emit-escape")) emit_esc = 1;
    else if (!strcmp(a, "--emit-uart")) emit_uart = 1;
    else if (!strcmp(a, "--emit-link-flags")) emit_lf = 1;
    else if (!strcmp(a, "--emit-c")) emit_c = 1;
    else if (!strcmp(a, "--linkable")) linkable = 1;
    else if (!strcmp(a, "--emit-claimgraph")) emit_cg = 1;
    else if (!strcmp(a, "--emit-pack")) emit_pack = 1;
    else if (!strcmp(a, "-o")) { if (++i < argc) out_path = argv[i]; else { fputs("bcir-cc: -o needs an argument\n",stderr); return 2; } }
    else if (!strncmp(a, "-o", 2)) out_path = a + 2;
    else if (!strcmp(a, "-I")) { if (++i >= argc) { fputs("bcir-cc: -I needs an argument\n",stderr); return 2; } if(ninc>=MAXD){fputs("bcir-cc: too many -I options\n",stderr);return 2;} incdirs[ninc++] = argv[i]; }
    else if (!strncmp(a, "-I", 2)) { if(ninc>=MAXD){fputs("bcir-cc: too many -I options\n",stderr);return 2;} incdirs[ninc++] = a + 2; }
    else if (!strcmp(a, "-D")) { if (++i >= argc) { fputs("bcir-cc: -D needs an argument\n",stderr); return 2; } if(ndef>=MAXD){fputs("bcir-cc: too many -D options\n",stderr);return 2;} if(define_spec(argv[i],defbuf[ndef],sizeof defbuf[ndef])){fputs("bcir-cc: invalid or overlong -D definition\n",stderr);return 2;} defs[ndef] = defbuf[ndef]; ndef++; }
    else if (!strncmp(a, "-D", 2)) { if(ndef>=MAXD){fputs("bcir-cc: too many -D options\n",stderr);return 2;} if(define_spec(a+2,defbuf[ndef],sizeof defbuf[ndef])){fputs("bcir-cc: invalid or overlong -D definition\n",stderr);return 2;} defs[ndef] = defbuf[ndef]; ndef++; }
    else if (!strcmp(a, "-U")) { if (++i >= argc) { fputs("bcir-cc: -U needs an argument\n",stderr); return 2; } if(nundef>=MAXD){fputs("bcir-cc: too many -U options\n",stderr);return 2;} undefs[nundef++] = argv[i]; }
    else if (!strncmp(a, "-U", 2)) { if(nundef>=MAXD){fputs("bcir-cc: too many -U options\n",stderr);return 2;} undefs[nundef++] = a + 2; }
    else if (!strncmp(a, "-std=", 5)) std = a + 5;
    else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { fputs(USAGE, stdout); return 0; }
    else if (a[0] == '-') { fprintf(stderr, "bcir-cc: unknown option '%s'\n%s", a, USAGE); return 2; }
    else if (nfiles < 256) files[nfiles++] = a;
    else { fputs("bcir-cc: too many input files\n",stderr); return 2; }
  }
  if (!nfiles) { fputs(USAGE, stderr); return 2; }
  if(strcmp(std,"c11")&&strcmp(std,"c17")&&strcmp(std,"c18")&&strcmp(std,"c23")&&strcmp(std,"c2x")){
    fprintf(stderr,"bcir-cc: unsupported -std=%s\n",std);return 2;
  }
  if(pp_only+emit_c+emit_cg+emit_pack+emit_fx+emit_esc+emit_lf+linkable>1){
    fputs("bcir-cc: select only one output mode\n",stderr);return 2;
  }
  if(emit_pack&&(nfiles!=1||project)){
    fputs("bcir-cc: --emit-pack requires exactly one input and cannot be combined with --project\n",stderr);return 2;
  }

  /* -std seeds __STDC_VERSION__ (a -D can still override it); honour -U by dropping every -D of its name, wherever
   * it stands -- as the oracle's CLI pops it (`__main__.py`). A dropped -D is left out of the seeds: it had been
   * handed on as an empty definition, which `define_macro` refuses (`macro name must be an identifier`), so
   * `-DXQ=1 -UXQ` failed every compile (CF-PPLIMITS). */
  char stdver[64]; snprintf(stdver, sizeof stdver, "__STDC_VERSION__ %s", std_version(std));
  for (int u = 0; u < nundef; u++)
    for (int d = 0; d < ndef; d++)
      if (defs[d] && !strncmp(defs[d], undefs[u], strlen(undefs[u])) && defs[d][strlen(undefs[u])] == ' ')
        defs[d] = NULL;

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
    char *raw=NULL;int read_rc=cc_read_file(path,&raw);
    if (read_rc) { fprintf(stderr, "bcir-cc: cannot %s '%s'\n",read_rc==1?"open":"read",path); if (rc != 1) rc = read_rc==1?2:1; n_dirty++; continue; }

    char base[1024]; dirof(path, base, sizeof base);
    const char *dirs[MAXD + 1]; int ndirs = 0;
    dirs[ndirs++] = base;                            /* the source dir first (quoted includes) */
    for (int d = 0; d < ninc && ndirs <= MAXD; d++) dirs[ndirs++] = incdirs[d];
    const char *alldefs[MAXD + 1]; int nalldef = 0;
    alldefs[nalldef++] = stdver;
    for (int d = 0; d < ndef && nalldef <= MAXD; d++) if (defs[d]) alldefs[nalldef++] = defs[d];

    static char cpperr[256];
    char *src=NULL;              /* the preprocessed text, in a block grown as it needs (CF-PPLIMITS): it had been held
                                  * in 64 KiB, and a longer unit refused that the oracle lowered */
    { int cs=1, ws=4, wsg=1;   /* the target's character types, which a `#if` reads (CF-PPARITH) */
      (void)bcir_cfront_target_chars(target, &cs, &ws, &wsg); bcir_cpp_set_chars(cs, ws, wsg); }
    int cpp_rc=bcir_cpp_run_ex_alloc(raw, path, dirs, ndirs, alldefs, nalldef, &src, NULL, cpperr, sizeof cpperr);
    cc_release_file(raw);
    if (cpp_rc) {
      /* --fallback: a construct outside the supported subset routes to the LLVM backend (rc 2),
       * the C twin of pipeline.compile_with_fallback (which classifies the rejecting phase). */
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: preprocess: %s\n", path, cpperr); if (rc != 1) rc = 2; n_fallback++; continue; }
      fprintf(stderr, "%s: preprocessor error: %s\n", path, cpperr); rc = 1; n_dirty++; continue;
    }
    if (pp_only) { fputs(src, outf); cc_release_file(src); n_clean++; continue; }

    static bcir_cfront_result r;
    /* --linkable: the twin's linkable artifact (CF-LINKTWIN), the front end's own emit of it -- the types, the
     * declarations and the globals need the parse, so the compile renders it whole */
    int compile_rc = (linkable || emit_uart) ? bcir_cfront_compile_linkable(src, target, &r)
                                             : bcir_cfront_compile_target(src, target, &r);
    cc_release_file(src);        /* the result owns copies of everything it names */
    if (compile_rc != 0 && linkable && !strncmp(r.diag, "linkable emit: ", 15)) {
      /* a global the artifact cannot define: the oracle's CLI says it so (`bcir-cfront --linkable`) */
      fprintf(stderr, "%s: linkable emit error: %s\n", path, r.diag); rc = 1; n_dirty++; bcir_cfront_free(&r); continue;
    }
    if (compile_rc != 0) {
      /* free the partial unit on the compile-error path too (the success path frees at the loop foot): `r`
       * is a reused static, so a skipped free here would leak the in-progress unit across files (the next
       * call's entry memset zeroes out->unit.funcs without freeing it). */
      if (fallback) { fprintf(stderr, "%s: fallback to LLVM backend: compile: %s\n", path, r.diag); if (rc != 1) rc = 2; n_fallback++; bcir_cfront_free(&r); continue; }
      fprintf(stderr, "%s: parse error: %s\n", path, r.diag); rc = 1; n_dirty++; bcir_cfront_free(&r); continue;
    }
    if (!r.ok) { fprintf(stderr, "%s: verify error: %s\n", path, r.diag); rc = 1; n_dirty++; bcir_cfront_free(&r); continue; }
    if ((emit_c || linkable || emit_uart) && !r.emitted_ok) {   /* a successful compile always emits (CF-BUF): defensive */
      fprintf(stderr, "%s: no emitted C\n", path);
      rc = 1; n_dirty++; bcir_cfront_free(&r); continue;
    }

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

    if (emit_uart) {
      uart_attest(path, r.emitted, outf);            /* the RegMap contract, from the linkable rendering */
      n_clean++;
    } else if (emit_lf) {
      /* B1: just the linker flags the unit's external-call edges need (one space-separated line, e.g.
       * `-lm`; empty for a pure-integer unit) -- for build-system consumption. Byte-identical to the
       * Python rail (bcir-cfront --emit-link-flags); parity gated in check_runtime.sh. */
      char lflags[256]; bcir_cfront_link_flags(&r.unit, lflags, sizeof lflags);
      fprintf(outf, "%s\n", lflags);
      n_clean++;
    } else if (linkable) {
      fputs(r.emitted, outf);                        /* Phase 3: the externally-linkable artifact, whole */
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
    } else if (emit_fx || emit_esc) {
      /* G10: measure the complete report, then write it whole (never a truncated report) */
      bcir_host_allocator heap = bcir_host_allocator_default();
      size_t need = emit_fx ? bcir_cfront_effects(&r.unit, NULL, 0) : bcir_cfront_escape(&r.unit, NULL, 0);
      char *rep = (need == SIZE_MAX) ? NULL : (char *)bcir_host_allocate(&heap, need + 1u);
      size_t got = !rep ? SIZE_MAX : emit_fx ? bcir_cfront_effects(&r.unit, rep, need + 1u)
                                             : bcir_cfront_escape(&r.unit, rep, need + 1u);
      if (!rep || got != need) {
        fprintf(stderr, "%s: %s: out of memory\n", path, emit_fx ? "effects" : "escape");
        bcir_host_deallocate(&heap, rep); rc = 1; n_dirty++; bcir_cfront_free(&r); continue;
      }
      fputs(rep, outf); bcir_host_deallocate(&heap, rep);
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
      if(r.unit.n_funcs==0){fprintf(stderr,"%s: pack error: translation unit has no entry function\n",path);rc=1;n_dirty++;bcir_cfront_free(&r);continue;}
      const bcir_func *f = &r.unit.funcs[r.unit.n_funcs - 1];   /* the entry function */
      static bcir_plan_step steps[8192]; bcir_plan plan;
      if (bcir_plan_func(f, steps, 8192, &plan) != BCIR_OK) { fprintf(stderr, "%s: plan error\n", path); rc = 1; n_dirty++; }
      else { static uint8_t pack[1 << 20]; size_t plen = 0;
        if (bcir_hydrate(f, &plan, pack, sizeof pack, &plen) != BCIR_OK) { fprintf(stderr, "%s: hydrate error\n", path); rc = 1; n_dirty++; }
        else if(fwrite(pack, 1, plen, outf)!=plen){fprintf(stderr,"%s: output write error\n",path);rc=1;n_dirty++;}
        else n_clean++; }
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

  int output_error=ferror(outf);
  int finish_error=out_path?fclose(outf):fflush(outf);
  if (output_error || finish_error) {
    fputs("bcir-cc: output write/close error\n",stderr);rc=1;
  }
  return rc;
}
