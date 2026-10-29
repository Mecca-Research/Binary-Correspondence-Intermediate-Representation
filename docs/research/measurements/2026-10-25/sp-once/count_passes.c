/* SP-ONCE's pass counts: one call of the entry point MODE names, on one tree's GEM units, for callgrind to count the
 * CRC-32 passes inside it (collection on inside that call alone). Built at -O0 -fno-inline, so every call is a call.
 * Modes: execute (bcir_sp_execute, a kernel doing nothing), load (bcir_dgem_load), run (bcir_dgem_run: the card's
 * prompt and its new tokens, the load before it outside the count).
 * usage: count_passes MODE MODEL.bcirq8 PACK PROMPT_IDS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decoder_gem.h"
#include "bcir_exec.h"
#include "bcir_q8_model.h"
#include "bcir_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint8_t *read_all(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  long n;
  uint8_t *p;
  if (!f || fseek(f, 0, SEEK_END) || (n = ftell(f)) <= 0 || fseek(f, 0, SEEK_SET)) exit(2);
  p = malloc((size_t)n);
  if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) exit(2);
  fclose(f);
  *len = (size_t)n;
  return p;
}

int main(int argc, char **argv) {
  bcir_q8_model model;
  char err[256];
  size_t pack_len, n_prompt = 0;
  int32_t prompt[256];
  if (argc != 5) {
    fprintf(stderr, "usage: count_passes execute|load|run MODEL.bcirq8 PACK PROMPT_IDS\n");
    return 2;
  }
  for (char *t = strtok(argv[4], ","); t && n_prompt < 256; t = strtok(NULL, ",")) prompt[n_prompt++] = atoi(t);
  uint8_t *pack = read_all(argv[3], &pack_len);
  bcir_q8_model_init(&model);
  if (bcir_q8_model_load(argv[2], &model, err, sizeof err)) {
    fprintf(stderr, "model: %s\n", err);
    return 1;
  }
  if (strcmp(argv[1], "execute") == 0) {
    bcir_streampack_header hdr;
    bcir_exec_result res;
    if (bcir_sp_validate(pack, pack_len, &hdr) != BCIR_OK) return 1;
    size_t n = hdr.n_segments;
    bcir_exec_item *scratch = malloc(n * sizeof *scratch);
    bcir_phase_stat *phases = malloc(n * sizeof *phases);
    if (!scratch || !phases) return 1;
    if (bcir_sp_execute(pack, pack_len, scratch, n, phases, n, NULL, NULL, &res) != BCIR_OK) return 1;
    printf("execute: %zu claims\n", res.executed);
    free(scratch);
    free(phases);
  } else if (strcmp(argv[1], "load") == 0) {
    bcir_dgem g;
    if (bcir_dgem_load(&g, &model, pack, pack_len, NULL)) return 1;
    printf("load: ok\n");
    bcir_dgem_free(&g);
  } else if (strcmp(argv[1], "run") == 0) {
    bcir_dgem g;
    int32_t gen[512];
    if (bcir_dgem_load(&g, &model, pack, pack_len, NULL)) return 1;
    size_t max_new = bcir_dgem_capacity(&g) - bcir_dgem_prompt_len(&g);
    double *logits = malloc(model.vocab_size * sizeof(double));
    if (!logits || max_new > 512 || bcir_dgem_prompt_len(&g) != n_prompt) return 1;
    if (bcir_dgem_run(&g, prompt, n_prompt, gen, max_new, logits)) return 1;
    printf("run: %zu new tokens\n", max_new);
    free(logits);
    bcir_dgem_free(&g);
  } else {
    return 2;
  }
  bcir_q8_model_free(&model);
  free(pack);
  return 0;
}
