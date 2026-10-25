/*===- test_exec.c - Python<->C parity harness for the StreamPack executor ===
 *
 * Reads a StreamPack file (argv[1], produced by the Python encoder), runs the
 * deterministic executor, and prints the dispatch order + per-phase telemetry. The
 * Python side (bcir/tests/test_c_executor.py) runs bcir.gem.execute on the same module
 * and asserts the order + (scheduled, executed) agree -- the executor's parity gate (the
 * analog of test_runtime.c for the decoder). A counting kernel records the dispatch order
 * to prove the per-claim callback fires in the right sequence. Uses libc (a host harness).
 *
 *   clang -std=c23 test_exec.c bcir_exec.c bcir_runtime.c -I . -o t && ./t pack.bin
 *
 * `./t --time pack.bin REPS` sizes its buffers from the pack instead, runs the executor (R10
 * walk included) REPS times with no kernel, and prints the fastest run: the complexity
 * witness of test_c_executor.py times two packs of one shape at two sizes.
 *
 * `./t --checked pack.bin` (SP-ONCE) walks the bytes plainly and over one bcir_sp_check token:
 * the check's status and header must be bcir_sp_validate's; on a refused pack every checked walk
 * must refuse the token; on an accepted one the semantic walk (under four generation pairs), the
 * generation check, and the segment and block walks (to the end and stopped after three) must give
 * the plain walks' statuses and hand their callbacks the same views. Forged tokens (no buffer, or
 * one too short for a header and the trailer) must be refused. Prints `checked: equal ...` or
 * `checked: refused ...`; exits 1 naming the first walk that differs.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_exec.h"

static int time_mode(const char *path, long reps) {
  FILE *fp = fopen(path, "rb");
  if (!fp || reps < 1) { if (fp) fclose(fp); fprintf(stderr, "time: bad arguments\n"); return 2; }
  uint8_t *buf = NULL;
  long size = 0;
  if (fseek(fp, 0, SEEK_END) == 0 && (size = ftell(fp)) > 0 && fseek(fp, 0, SEEK_SET) == 0)
    buf = malloc((size_t)size);
  if (!buf || fread(buf, 1, (size_t)size, fp) != (size_t)size) {
    fclose(fp); free(buf); fprintf(stderr, "time: cannot read %s\n", path); return 2;
  }
  fclose(fp);
  bcir_streampack_header hdr;
  bcir_status st = bcir_sp_validate(buf, (size_t)size, &hdr);
  size_t n = st == BCIR_OK ? (size_t)hdr.n_segments : 0;
  bcir_exec_item *scratch = malloc((n ? n : 1) * sizeof *scratch);
  bcir_phase_stat *phases = malloc((n ? n : 1) * sizeof *phases);
  if (st != BCIR_OK || !scratch || !phases) {
    free(buf); free(scratch); free(phases); printf("ERR %d\n", (int)st); return 1;
  }
  double best = -1.0;
  bcir_exec_result res;
  for (long r = 0; r < reps; r++) {
    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    st = bcir_sp_execute(buf, (size_t)size, scratch, n, phases, n, NULL, NULL, &res);
    timespec_get(&t1, TIME_UTC);
    if (st != BCIR_OK) break;
    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 + (double)(t1.tv_nsec - t0.tv_nsec);
    if (best < 0.0 || ns < best) best = ns;
  }
  free(buf); free(scratch); free(phases);
  if (st != BCIR_OK) { printf("ERR %d\n", (int)st); return 1; }
  printf("segments %zu ns %.0f\n", res.n_segments, best);
  return 0;
}

static uint64_t g_order[4096];
static size_t g_norder;

static int record(const bcir_exec_item *item, void *ctx) {
  (void)ctx;
  if (g_norder < sizeof g_order / sizeof g_order[0])
    g_order[g_norder++] = item->claim_id;
  return 0;
}

/* SP-ONCE's differential: what every callback sees, folded (FNV-1a over each field and each byte). */
typedef struct { uint64_t h; unsigned calls, stop_after; } fold_ctx;
static void fold(fold_ctx *f, uint64_t v) { f->h = (f->h ^ v) * UINT64_C(0x100000001B3); }
static void fold_bytes(fold_ctx *f, const void *p, size_t n) {
  for (size_t i = 0; i < n; i++) fold(f, ((const uint8_t *)p)[i]);
}
static int fold_segment(const bcir_segment_view *s, void *ctx) {
  fold_ctx *f = (fold_ctx *)ctx;
  fold(f, s->claim_id); fold(f, s->phase_id); fold(f, s->lane); fold(f, s->width); fold(f, s->stride_k);
  fold(f, s->dispatch); fold(f, s->n_reads); fold(f, s->n_writes);
  for (uint16_t i = 0; i < s->n_reads; i++) fold(f, bcir_seg_read_rid(s, i));
  for (uint16_t i = 0; i < s->n_writes; i++) fold(f, bcir_seg_write_rid(s, i));
  fold(f, s->name_len); fold_bytes(f, s->name, s->name_len);
  fold(f, s->opcode_len); fold_bytes(f, s->opcode, s->opcode_len);
  fold(f, s->prefetch_len); fold_bytes(f, s->prefetch, s->prefetch_len);
  fold(f, s->channel_len); fold_bytes(f, s->channel, s->channel_len);
  return ++f->calls == f->stop_after;
}
static int fold_block(const bcir_block_view *b, void *ctx) {
  fold_ctx *f = (fold_ctx *)ctx;
  fold(f, b->base); fold(f, b->count); fold(f, b->n_strides);
  for (uint16_t i = 0; i < b->n_strides; i++) fold(f, bcir_block_stride(b, i));
  return ++f->calls == f->stop_after;
}
static int same_header(const bcir_streampack_header *a, const bcir_streampack_header *b) {
  if (memcmp(a->magic, b->magic, 4) || memcmp(a->reserved0, b->reserved0, sizeof a->reserved0) ||
      memcmp(a->reserved, b->reserved, sizeof a->reserved))
    return 0;
  return a->version == b->version && a->flags == b->flags && a->topo_gen == b->topo_gen &&
         a->map_gen == b->map_gen && a->data_gen == b->data_gen && a->n_segments == b->n_segments &&
         a->n_prefetches == b->n_prefetches && a->n_blocks == b->n_blocks && a->n_trace == b->n_trace &&
         a->pipeline_depth == b->pipeline_depth && a->n_gens == b->n_gens;
}
#define DIFFERS(what) do { printf("checked: %s differs\n", what); return 1; } while (0)

static int forged_tokens_refused(const uint8_t *buf, size_t len) {
  bcir_sp_checked forged[4];
  memset(forged, 0, sizeof forged);
  forged[1].data = buf;                       /* a buffer with no length */
  forged[2].data = buf;                       /* one too short for a header and the trailer */
  forged[2].len = len < (size_t)BCIR_STREAMPACK_HEADER_SIZE + 3u ? len : (size_t)BCIR_STREAMPACK_HEADER_SIZE + 3u;
  forged[3].len = (size_t)BCIR_STREAMPACK_HEADER_SIZE + 4u + len;  /* a length with no buffer */
  for (int i = 0; i < 4; i++) {
    fold_ctx f = {0, 0, 0};
    if (bcir_sp_checked_verify_semantic(&forged[i], 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_ERR_TRUNCATED ||
        bcir_sp_checked_generation(&forged[i], 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_ERR_TRUNCATED ||
        bcir_sp_checked_for_each_segment(&forged[i], fold_segment, &f) != BCIR_ERR_TRUNCATED ||
        bcir_sp_checked_for_each_block(&forged[i], fold_block, &f) != BCIR_ERR_TRUNCATED || f.calls)
      return 0;
  }
  return bcir_sp_checked_for_each_segment(NULL, fold_segment, NULL) == BCIR_ERR_TRUNCATED;
}

static int checked_mode(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) { perror("fopen"); return 2; }
  static uint8_t buf[1 << 20];
  size_t len = fread(buf, 1, sizeof buf, fp);
  fclose(fp);
  bcir_streampack_header plain;
  bcir_sp_checked pack;
  bcir_status sv = bcir_sp_validate(buf, len, &plain);
  if (bcir_sp_check(buf, len, &pack) != sv) DIFFERS("bcir_sp_check");
  if (bcir_sp_check(buf, len, NULL) != sv) DIFFERS("bcir_sp_check without a token");
  if (!forged_tokens_refused(buf, len)) DIFFERS("a forged token");
  if (sv != BCIR_OK) {
    fold_ctx f = {0, 0, 0};
    if (pack.data || pack.len) DIFFERS("a refused check's token");
    if (bcir_sp_checked_verify_semantic(&pack, 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_ERR_TRUNCATED ||
        bcir_sp_checked_for_each_segment(&pack, fold_segment, &f) != BCIR_ERR_TRUNCATED ||
        bcir_sp_checked_for_each_block(&pack, fold_block, &f) != BCIR_ERR_TRUNCATED || f.calls)
      DIFFERS("a refused token's walks");
    printf("checked: refused %d\n", (int)sv);
    return 0;
  }
  if (pack.data != buf || pack.len != len || !same_header(&plain, &pack.hdr)) DIFFERS("the token");
  const uint32_t gens[4][2] = {{0xFFFFFFFFu, 0xFFFFFFFFu}, {plain.map_gen, plain.data_gen},
                               {plain.map_gen + 1u, 0xFFFFFFFFu}, {0xFFFFFFFFu, plain.data_gen + 1u}};
  bcir_status semantic = BCIR_OK;
  for (int i = 0; i < 4; i++) {
    bcir_status a = bcir_sp_verify_semantic(buf, len, gens[i][0], gens[i][1]);
    if (bcir_sp_checked_verify_semantic(&pack, gens[i][0], gens[i][1]) != a) DIFFERS("the semantic walk");
    if (bcir_sp_checked_generation(&pack, gens[i][0], gens[i][1]) !=
        bcir_sp_check_generation(buf, len, gens[i][0], gens[i][1]))
      DIFFERS("the generation check");
    if (i == 0) semantic = a;
  }
  bcir_status segments = BCIR_OK, blocks = BCIR_OK;
  for (unsigned stop = 0; stop < 2; stop++) {
    fold_ctx p = {UINT64_C(0xCBF29CE484222325), 0, stop ? 3u : 0u}, q = p;
    bcir_status a = bcir_sp_for_each_segment(buf, len, fold_segment, &p);
    if (bcir_sp_checked_for_each_segment(&pack, fold_segment, &q) != a || p.h != q.h || p.calls != q.calls)
      DIFFERS("the segment walk");
    fold_ctx r = {UINT64_C(0xCBF29CE484222325), 0, stop ? 3u : 0u}, s = r;
    bcir_status b = bcir_sp_for_each_block(buf, len, fold_block, &r);
    if (bcir_sp_checked_for_each_block(&pack, fold_block, &s) != b || r.h != s.h || r.calls != s.calls)
      DIFFERS("the block walk");
    if (!stop) segments = a, blocks = b;
  }
  printf("checked: equal %d %d %d\n", (int)semantic, (int)segments, (int)blocks);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 4 && strcmp(argv[1], "--time") == 0) return time_mode(argv[2], strtol(argv[3], NULL, 10));
  if (argc == 3 && strcmp(argv[1], "--checked") == 0) return checked_mode(argv[2]);
  if (argc < 2) { fprintf(stderr, "usage: %s <pack-file> | --time <pack-file> <reps> | --checked <pack-file>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen"); return 2; }
  static uint8_t buf[1 << 20];
  size_t len = fread(buf, 1, sizeof buf, fp);
  fclose(fp);

  static bcir_exec_item scratch[4096];
  static bcir_phase_stat phases[256];
  bcir_exec_result res;
  bcir_status st = bcir_sp_execute(buf, len, scratch, 4096, phases, 256, record, NULL, &res);
  if (st != BCIR_OK) { printf("ERR %d\n", (int)st); return 1; }

  {
    bcir_exec_result failed = {(size_t)-1, (size_t)-1, (size_t)-1};
    bcir_phase_stat sentinel = {0xA5A5A5A5u, 0xA5A5A5A5u, 0xA5A5A5A5u};
    st = bcir_sp_execute(buf, len, scratch, 4096, &sentinel, 0, NULL, NULL, &failed);
    if (st != BCIR_ERR_NOSPACE || failed.executed || failed.n_phases ||
        failed.n_segments || sentinel.phase_id != 0xA5A5A5A5u ||
        sentinel.scheduled != 0xA5A5A5A5u || sentinel.executed != 0xA5A5A5A5u) {
      fprintf(stderr, "executor failure-state regression\n");
      return 3;
    }
  }

  /* The dispatch order, as recorded by the kernel callback (must equal gem.execute). */
  printf("order:");
  for (size_t i = 0; i < g_norder; i++) printf(" %llu", (unsigned long long)g_order[i]);
  printf("\n");
  /* Per-phase telemetry, in execution (first-appearance / topological) order. */
  for (size_t i = 0; i < res.n_phases; i++)
    printf("phase %u %u %u\n", phases[i].phase_id, phases[i].scheduled, phases[i].executed);
  printf("executed %zu\n", res.executed);
  return 0;
}
