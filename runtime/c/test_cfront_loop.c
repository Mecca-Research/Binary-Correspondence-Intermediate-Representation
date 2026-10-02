/*===- test_cfront_loop.c - the full C compile->execute loop (no Python) ---===
 * Closes the loop entirely in C: read a C source file (argv[1]), lower it through the
 * plug-in C frontend (bcir_cfront), plan it (bcir_plan), hydrate it into a StreamPack
 * (bcir_hydrate), and execute that pack through the deterministic runtime (bcir_exec) --
 * the same executor a driver runs. Prints a one-line summary the Python parity test
 * (bcir/tests/test_c_cfront.py) checks against gem.execute on the same source.
 *
 *   C source -> claim graph -> K_BCIR plan -> StreamPack -> bcir_sp_execute
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_cfront.h"
#include "bcir_cpp.h"
#include "bcir_host_alloc.h"
#include "bcir_exec.h"
#include "bcir_hydrate.h"
#include "bcir_plan.h"
#include "bcir_verify.h"

static void dirof(const char *path, char *out, size_t cap) {
  const char *s = strrchr(path, '/'), *bs = strrchr(path, '\\');
  if (!s || (bs && bs > s)) s = bs;
  if (s) { size_t n = (size_t)(s - path); if (n == 0) n = 1; else if (n == 2 && path[1] == ':') n++;
    if (n >= cap) n = cap - 1; memcpy(out, path, n); out[n] = 0; }
  else snprintf(out, cap, ".");
}

static uint64_t g_order[8192]; static size_t g_n;
static int record(const bcir_exec_item *it, void *ctx) {
  (void)ctx; if (g_n < sizeof g_order / sizeof g_order[0]) g_order[g_n++] = it->claim_id;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <c-source>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb"); if (!fp) { perror("fopen"); return 2; }
  /* the whole source and its whole preprocessed text, each in a block grown as it needs (CF-PPLIMITS): this driver
   * had read the first 64 KiB of a source and lowered that prefix, silently, and held the text in 64 KiB */
  bcir_host_allocator heap = bcir_host_allocator_default();
  char *raw = NULL, *src = NULL;
  const char *why = bcir_cpp_read_source(fp, &heap, &raw);
  if (fclose(fp) && !why) why = "cannot read the source";
  if (why) { bcir_host_deallocate(&heap, raw); printf("READ-ERR %s\n", why); return 1; }
  static char cpperr[256], base[1024]; dirof(argv[1], base, sizeof base);
  int cpp_rc = bcir_cpp_run_alloc(raw, base, &src, NULL, cpperr, sizeof cpperr);
  bcir_host_deallocate(&heap, raw);
  if (cpp_rc) { printf("CPP-ERR %s\n", cpperr); return 1; }

  static bcir_cfront_result r;
  int compile_rc = bcir_cfront_compile(src, &r);
  bcir_host_deallocate(&heap, src);   /* the result owns copies of everything it names */
  if (compile_rc != 0) { printf("PARSE-ERR %s\n", r.diag); bcir_cfront_free(&r); return 1; }
  if (!r.ok) { printf("VERIFY-ERR %s\n", r.diag); bcir_cfront_free(&r); return 1; }

  const bcir_func *f = &r.unit.funcs[r.unit.n_funcs - 1];   /* the entry function */
  static bcir_plan_step steps[8192]; bcir_plan plan;
  if (bcir_plan_func(f, steps, 8192, &plan) != BCIR_OK) { printf("PLAN-ERR\n"); return 1; }

  char vdiag[256];
  int r9 = bcir_verify_plan(f, &plan, vdiag, sizeof vdiag);   /* R9: plan legality */
  if (!r9) { printf("R9-ERR %s\n", vdiag); return 1; }

  size_t real = 0;                          /* realizable claims (control-flow markers excluded) */
  for (size_t i = 0; i < f->n_claims; i++) if (f->claims[i].opcode != BCIR_OP_NOP) real++;

  static uint8_t pack[1 << 20]; size_t plen = 0;
  if (bcir_hydrate(f, &plan, pack, sizeof pack, &plen) != BCIR_OK) { printf("HYDRATE-ERR\n"); return 1; }

  int r10 = bcir_verify_pack(pack, plen, (uint32_t)real, vdiag, sizeof vdiag);  /* R10-R11 */
  if (!r10) { printf("R10R11-ERR %s\n", vdiag); return 1; }

  static bcir_exec_item scratch[8192]; static bcir_phase_stat phases[256]; bcir_exec_result res;
  bcir_status st = bcir_sp_execute(pack, plen, scratch, 8192, phases, 256, record, NULL, &res);
  if (st != BCIR_OK) { printf("EXEC-ERR %d\n", (int)st); return 1; }

  printf("loop: claims=%zu plan_cost=%llu pack_bytes=%zu executed=%zu r9=%d r10r11=%d prov=%016llx order=",
         real, (unsigned long long)plan.total_cost, plen, res.executed, r9, r10,
         (unsigned long long)bcir_provenance_digest(f));
  for (size_t i = 0; i < g_n; i++) printf("%s%llu", i ? "," : "", (unsigned long long)g_order[i]);
  printf("\n");
  bcir_cfront_free(&r);
  return 0;
}
