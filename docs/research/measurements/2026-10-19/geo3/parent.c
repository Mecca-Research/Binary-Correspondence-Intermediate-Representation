/* GEO-3 records, the parent's figures (docs/research/measurements/2026-10-19/geo3): the bf16
 * workspace the parent commit's plan admitted (bcir_decoder_workspace_bytes, ABI 4) and what its
 * harness probe recorded over one step at that size, for the shapes geo3.c measures. Built against
 * the parent's runtime/c with -DBCIR_DT_WORKSPACE_PROBE (run.sh). */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t bcir_decoder_workspace_probe(void);

static void *al(size_t bytes) {
  size_t r = bytes ? (bytes + 63) / 64 * 64 : 64;
  void *p = aligned_alloc(64, r);
  if (!p) { perror("aligned_alloc"); exit(2); }
  memset(p, 0, r);
  return p;
}
typedef struct shape { const char *name; bcir_decoder_spec sp; } shape;
static const shape shapes[] = {
  {"codex-attn", {260, 128, 4, 4, 3, 256, 4, 256, 1, 10000.0, 1e-6}},
  {"codex-attn B2", {260, 128, 4, 4, 3, 256, 2, 256, 1, 10000.0, 1e-6}},
  {"9.84M", {260, 384, 8, 4, 6, 1024, 8, 128, 1, 10000.0, 1e-6}},
};
#define CORE (BCIR_DT_CONTRACT_EXP | BCIR_DT_CONTRACT_REDUCE)
static const struct { const char *name; uint32_t bits; } contracts[] = {
  {"bf16+exp+reduce+attn", BCIR_DT_CONTRACT_BF16 | CORE | BCIR_DT_CONTRACT_ATTN},
  {"bf16+exp+reduce+attnbf16", BCIR_DT_CONTRACT_BF16 | CORE | BCIR_DT_CONTRACT_ATTNBF16},
};

int main(void) {
  size_t si, ci, i;
  static const bcir_decoder_adamw opt = {1e-3, 0.9, 0.95, 1e-8, 0.1, 1.0};
  printf("the parent (ABI %u): the plan's bf16 workspace and the probe's record over one step\n",
      bcir_decoder_abi_version());
  for (si = 0; si < sizeof shapes / sizeof shapes[0]; si++)
    for (ci = 0; ci < sizeof contracts / sizeof contracts[0]; ci++) {
      bcir_decoder_plan plan;
      bcir_decoder_state s;
      bcir_decoder_event ev[1];
      uint32_t *tok, *tgt;
      size_t n, bound, seen, done = 0;
      int rc;
      if (bcir_decoder_make_plan(&shapes[si].sp, &plan)) return 2;
      n = (size_t)plan.spec.batch * plan.spec.time;
      bound = bcir_decoder_workspace_bytes(&plan, contracts[ci].bits);
      memset(&s, 0, sizeof s);
      s.plan = plan; s.contract = contracts[ci].bits;
      s.weights = al(plan.parameters * 4); s.gradients = al(plan.parameters * 4);
      s.moment1 = al(plan.parameters * 4); s.moment2 = al(plan.parameters * 4);
      s.activations = al(plan.activations * 4); s.scratch = al(plan.scratch * 4);
      s.weight_capacity = s.gradient_capacity = s.moment1_capacity = s.moment2_capacity = plan.parameters;
      s.activation_capacity = plan.activations; s.scratch_capacity = plan.scratch;
      s.workspace = al(bound); s.workspace_bytes = bound;
      s.beta1_power = s.beta2_power = 1.0;
      for (i = 0; i < plan.parameters; i++) s.weights[i] = (float)((i * 2654435761u) % 2001u) * 1e-5f - 0.01f;
      tok = al(n * 4); tgt = al(n * 4);
      for (i = 0; i < n; i++) { tok[i] = (uint32_t)((i * 7u + 3u) % plan.spec.vocab); tgt[i] = (uint32_t)((i * 7u + 10u) % plan.spec.vocab); }
      (void)bcir_decoder_workspace_probe();
      rc = bcir_decoder_run(&s, &opt, tok, tgt, n, NULL, 1, ev, 1, &done);
      seen = bcir_decoder_workspace_probe();
      printf("  %-14s %-26s plan %9zu, probe %9zu (rc %d, %zu step)\n", shapes[si].name,
          contracts[ci].name, bound, seen, rc, done);
      free(s.weights); free(s.gradients); free(s.moment1); free(s.moment2); free(s.activations);
      free(s.scratch); free(s.workspace); free(tok); free(tgt);
    }
  return 0;
}
