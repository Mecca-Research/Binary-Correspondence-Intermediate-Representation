/* NEXT-6: GEM's trust boundary a part at a time on one pack -- bcir_crc32 over the pack's body, bcir_sp_validate,
 * bcir_sp_verify_semantic, bcir_sp_for_each_segment and bcir_sp_for_each_block (each re-validates), the four as
 * bcir_dgem's load calls them, and the slicing-by-8 prototype's CRC -- each the median of ROUNDS rounds of REPS calls,
 * in rotated order after a warm round; every CRC the stored one.  usage: trust_parts PACK ROUNDS REPS */
#define _POSIX_C_SOURCE 200809L
#include "bcir_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint32_t crc_slice8(const uint8_t *d, size_t n);
enum { PARTS = 7, MAXR = 101 };
static const char *part_name[PARTS] = {"crc32", "validate", "verify_semantic", "for_each_segment", "for_each_block",
                                       "gem_load_four", "crc_slice8"};
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int seg_cb(const bcir_segment_view *v, void *ctx) {
  (void)v;
  ++*(size_t *)ctx;
  return 0;
}
static int blk_cb(const bcir_block_view *b, void *ctx) {
  (void)b;
  ++*(size_t *)ctx;
  return 0;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static volatile uint32_t sink;
static int part(int k, const uint8_t *p, size_t n) {
  bcir_streampack_header h;
  size_t count = 0;
  switch (k) {
  case 0: sink = bcir_crc32(p, n - 4u); return sink == (uint32_t)(p[n - 4] | p[n - 3] << 8 | p[n - 2] << 16 | (uint32_t)p[n - 1] << 24);
  case 1: return bcir_sp_validate(p, n, &h) == BCIR_OK;
  case 2: return bcir_sp_verify_semantic(p, n, 0xFFFFFFFFu, 0xFFFFFFFFu) == BCIR_OK;
  case 3: return bcir_sp_for_each_segment(p, n, seg_cb, &count) == BCIR_OK && count;
  case 4: return bcir_sp_for_each_block(p, n, blk_cb, &count) == BCIR_OK && count;
  case 5:
    return bcir_sp_validate(p, n, &h) == BCIR_OK &&
           bcir_sp_verify_semantic(p, n, 0xFFFFFFFFu, 0xFFFFFFFFu) == BCIR_OK &&
           bcir_sp_for_each_segment(p, n, seg_cb, &count) == BCIR_OK &&
           bcir_sp_for_each_block(p, n, blk_cb, &count) == BCIR_OK;
  default: sink = crc_slice8(p, n - 4u); return sink == bcir_crc32(p, n - 4u);
  }
}
int main(int argc, char **argv) {
  FILE *f;
  long len;
  uint8_t *p;
  int rounds, reps, r, a, k;
  double ms[PARTS][MAXR];
  if (argc != 4 || !(f = fopen(argv[1], "rb"))) return 2;
  rounds = atoi(argv[2]);
  reps = atoi(argv[3]);
  if (rounds < 1 || rounds > MAXR || reps < 1) return 2;
  fseek(f, 0, SEEK_END);
  len = ftell(f);
  rewind(f);
  p = malloc((size_t)len);
  if (!p || fread(p, 1, (size_t)len, f) != (size_t)len) return 2;
  fclose(f);
  for (r = -1; r < rounds; r++)
    for (a = 0; a < PARTS; a++) {
      double t0;
      k = (a + (r < 0 ? 0 : r)) % PARTS;
      t0 = now();
      for (int i = 0; i < reps; i++)
        if (!part(k, p, (size_t)len)) {
          fprintf(stderr, "%s refused the pack\n", part_name[k]);
          return 1;
        }
      if (r >= 0) ms[k][r] = (now() - t0) * 1e3 / reps;
    }
  printf("{\"pack_octets\":%ld,\"rounds\":%d,\"reps\":%d", len, rounds, reps);
  for (k = 0; k < PARTS; k++) {
    qsort(ms[k], (size_t)rounds, sizeof(double), cmp);
    printf(",\"%s_ms\":%.4f", part_name[k], ms[k][rounds / 2]);
  }
  printf("}\n");
  return 0;
}
