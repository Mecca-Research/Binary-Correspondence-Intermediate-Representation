/*===- test_runtime.c - parity harness for the freestanding runtime -------===
 *
 * Reads a StreamPack file (produced by the Python encoder), decodes it with the
 * freestanding runtime, and prints the parsed fields so a cross-language parity
 * test can assert Python-encoded == C-decoded. This harness uses libc (it is a
 * test); bcir_runtime.c itself stays freestanding.
 *
 *   cc bcir_runtime.c test_runtime.c -o test_runtime && ./test_runtime pack.bin
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "bcir_runtime.h"

typedef struct {
  uint32_t count;
  bcir_segment_view first;
  int have_first;
} walk_ctx;

static int on_segment(const bcir_segment_view *seg, void *vctx) {
  walk_ctx *ctx = (walk_ctx *)vctx;
  if (!ctx->have_first) { ctx->first = *seg; ctx->have_first = 1; }
  ctx->count++;
  return 0;
}

typedef struct {
  uint32_t count;
  bcir_generation_view first;
  int have_first;
} gen_ctx;

static int on_generation(const bcir_generation_view *g, void *vctx) {
  gen_ctx *ctx = (gen_ctx *)vctx;
  if (!ctx->have_first) { ctx->first = *g; ctx->have_first = 1; }
  ctx->count++;
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <streampack-file>\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "rb");
  if (!f) { perror("fopen"); return 2; }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  uint8_t *buf = (uint8_t *)malloc((size_t)n);
  if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "read failed\n"); return 2; }
  fclose(f);

  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(buf, (size_t)n, &hdr);
  if (st != BCIR_OK) { printf("VALIDATE_FAIL status=%d\n", st); free(buf); return 1; }

  walk_ctx ctx = {0};
  st = bcir_sp_for_each_segment(buf, (size_t)n, on_segment, &ctx);
  if (st != BCIR_OK) { printf("WALK_FAIL status=%d\n", st); free(buf); return 1; }
  gen_ctx gctx = {0};
  st = bcir_sp_for_each_generation(buf, (size_t)n, on_generation, &gctx);
  if (st != BCIR_OK) { printf("GEN_WALK_FAIL status=%d\n", st); free(buf); return 1; }

  printf("version=%u\n", hdr.version);
  printf("pipeline_depth=%u\n", hdr.pipeline_depth);
  printf("topo_gen=%u\n", hdr.topo_gen);
  printf("map_gen=%u\n", hdr.map_gen);
  printf("data_gen=%u\n", hdr.data_gen);
  printf("n_segments=%u\n", hdr.n_segments);
  printf("walked=%u\n", ctx.count);
  if (ctx.have_first) {
    const bcir_segment_view *s = &ctx.first;
    printf("seg0.claim_id=%llu\n", (unsigned long long)s->claim_id);
    printf("seg0.phase_id=%u\n", s->phase_id);
    printf("seg0.lane=%u\n", s->lane);
    printf("seg0.width=%u\n", s->width);
    printf("seg0.opcode=%.*s\n", (int)s->opcode_len, s->opcode);
    printf("seg0.n_reads=%u\n", s->n_reads);
    if (s->n_reads > 0) printf("seg0.read0=%u\n", bcir_seg_read_rid(s, 0));
    if (s->n_writes > 0) printf("seg0.write0=%u\n", bcir_seg_write_rid(s, 0));
    printf("seg0.dispatch=%u\n", s->dispatch);
    printf("seg0.channel=%.*s\n", (int)s->channel_len, s->channel ? s->channel : "");
    printf("seg0.prefetch=%.*s\n", (int)s->prefetch_len, s->prefetch ? s->prefetch : "");
  }
  /* v4: the per-resource generation vector (0 records before v4). */
  printf("n_gens=%u\n", hdr.n_gens);
  printf("gens_walked=%u\n", gctx.count);
  if (gctx.have_first) {
    printf("gen0.rid=%u\n", gctx.first.rid);
    printf("gen0.map_gen=%u\n", gctx.first.map_gen);
    printf("gen0.data_gen=%u\n", gctx.first.data_gen);
  }
  printf("OK\n");
  free(buf);
  return 0;
}
