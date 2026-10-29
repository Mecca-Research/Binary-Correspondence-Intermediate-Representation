/* CRC-SLICE's A/B, one process: zlib's CRC-32 as the tree before computed it (BEFORE: its bcir_runtime.c's
 * bcir_crc32, a byte-table step a byte, the object's globals renamed before_*) against the tree after's (slicing-by-8,
 * bcir_crc32.h) and zlib's own crc32 (the system's libz), and BEFORE again (the A/A control); and the BCIRQ8 loader
 * as the tree before had it (its bcir_q8_model.c, the bitwise loop, renamed before_*) against the tree after's, on a
 * model file. Per case a warm round, then ROUNDS rounds in rotated arm order, an arm's time the mean of its calls;
 * each arm's ratio over BEFORE's in the same round, the geometric mean and its 95% interval (Student t to 24 rounds,
 * then 1.96 + 2.4/(n-1)). Every CRC is zlib's and every load succeeds (the run exits otherwise).
 * usage: crc_ab ROUNDS MODEL.bcirq8 BYTES...   -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include "bcir_q8_model.h"
#include "bcir_runtime.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <zlib.h>

uint32_t before_bcir_crc32(const uint8_t *data, size_t len);
int before_bcir_q8_model_load(const char *path, bcir_q8_model *out, char *error, size_t error_capacity);
void before_bcir_q8_model_free(bcir_q8_model *model);
void before_bcir_q8_model_init(bcir_q8_model *model);

enum { MAXR = 200 };
static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static void interval(const double *r, int n, double out[3]) {
  static const double t95[] = {0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
                               2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069};
  double m = 0, sd = 0, h;
  int i;
  for (i = 0; i < n; i++) m += log(r[i]);
  m /= n;
  for (i = 0; i < n; i++) sd += (log(r[i]) - m) * (log(r[i]) - m);
  sd = sqrt(sd / (n - 1));
  h = (n <= 24 ? t95[n] : 1.96 + 2.4 / (n - 1)) * sd / sqrt((double)n);
  out[0] = exp(m);
  out[1] = exp(m - h);
  out[2] = exp(m + h);
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(const double *x, int n) {
  double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmp);
  return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}
static void report(const char *what, const char **names, int arms, double ms[][MAXR], int rounds) {
  int a, r;
  printf("{\"case\":\"%s\",\"rounds\":%d,\"before_ms\":%.5f", what, rounds, median(ms[0], rounds));
  for (a = 1; a < arms; a++) {
    double ratio[MAXR], ci[3];
    for (r = 0; r < rounds; r++) ratio[r] = ms[a][r] / ms[0][r];
    interval(ratio, rounds, ci);
    printf(",\"%s\":{\"ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", names[a], median(ms[a], rounds), ci[0],
           ci[1], ci[2]);
  }
  printf("}\n");
  fflush(stdout);
}
static uint64_t state = UINT64_C(0x9E3779B97F4A7C15);
static uint32_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (uint32_t)(state >> 11);
}
int main(int argc, char **argv) {
  int rounds, c, r, a;
  static double ms[4][MAXR];
  if (argc < 4) {
    fprintf(stderr, "usage: crc_ab ROUNDS MODEL.bcirq8 BYTES...\n");
    return 2;
  }
  rounds = atoi(argv[1]);
  if (rounds < 2 || rounds > MAXR) return 2;
  for (c = 3; c < argc; c++) {
    static const char *names[4] = {"before", "after", "zlib", "A/A"};
    size_t n = (size_t)strtoull(argv[c], NULL, 10), i;
    uint8_t *d = malloc(n ? n : 1);
    uint32_t want;
    int reps;
    char what[64];
    if (!d) return 1;
    for (i = 0; i < n; i++) d[i] = (uint8_t)next();
    want = (uint32_t)crc32(0L, d, (uInt)n);
    reps = n < 65536u ? 200 : n < 1048576u ? 20 : 3;
    for (r = -1; r < rounds; r++)
      for (a = 0; a < 4; a++) {
        int arm = (a + (r < 0 ? 0 : r)) % 4, k;
        uint32_t got = 0;
        double t0 = now();
        for (k = 0; k < reps; k++)
          got = arm == 1 ? bcir_crc32(d, n) : arm == 2 ? (uint32_t)crc32(0L, d, (uInt)n) : before_bcir_crc32(d, n);
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3 / reps;
        if (got != want) {
          fprintf(stderr, "%s's CRC of %zu bytes is not zlib's\n", names[arm], n);
          return 1;
        }
      }
    snprintf(what, sizeof what, "crc32 %zu bytes", n);
    report(what, names, 4, ms, rounds);
    free(d);
  }
  {
    static const char *names[3] = {"before", "after", "A/A"};
    for (r = -1; r < rounds; r++)
      for (a = 0; a < 3; a++) {
        int arm = (a + (r < 0 ? 0 : r)) % 3, rc;
        char error[256];
        bcir_q8_model m;
        double t0 = now();
        if (arm == 1) {
          bcir_q8_model_init(&m);
          rc = bcir_q8_model_load(argv[2], &m, error, sizeof error);
          if (!rc) bcir_q8_model_free(&m);
        } else {
          before_bcir_q8_model_init(&m);
          rc = before_bcir_q8_model_load(argv[2], &m, error, sizeof error);
          if (!rc) before_bcir_q8_model_free(&m);
        }
        if (r >= 0) ms[arm][r] = (now() - t0) * 1e3;
        if (rc) {
          fprintf(stderr, "%s refused the model: %s\n", names[arm], error);
          return 1;
        }
      }
    report("bcir_q8_model_load", names, 3, ms, rounds);
  }
  return 0;
}
