/* DECODE-THREADS' kernel A/B: three builds of runtime/c/bcir_ai_kernels.c in one binary, each unit's exported
 * symbols renamed (measure_serial.sh) -- the tree before DECODE-THREADS (b_), DECODE-THREADS as first committed,
 * its kernels' column range a variable in every call (f_), and the tree after, each vector kernel's whole call an
 * instance of its own (a_) -- timing whole calls of bcir_ai_q8_matvec_f64_instance on one instance: the 860K
 * decoder's matvec shapes, 512 and 2,048 square, and one call whose ranges Q8-BOUND does not prove. A round runs
 * every arm in rotated order -- before, first, after and before again (the A/A control) -- each arm REPS calls; each
 * arm's time a call over before's in the same round, the geometric mean and its 95% interval (Student t to 24
 * rounds, then 1.96 + 2.4/(n-1): decode_ab.py's), after a warm round. Every arm writes the same output buffer, 64-byte
 * aligned, so that the arms differ in their code alone (with a buffer each, the A/A control read up to x1.06: where
 * a buffer's stores split cache lines another's did not). Then each arm's outputs and status, in a buffer of its
 * own, are before's (the run exits otherwise).
 * usage: kernels_ab INSTANCE ROUNDS  -- one JSON line a shape */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DECL(p)                                                                                           \
  int p##_bcir_ai_q8_matvec_f64_instance(int, const double *, size_t, const uint8_t *, size_t, const uint8_t *, \
                                         size_t, size_t, size_t, double *);
DECL(b)
DECL(f)
DECL(a)
typedef int (*matvec_fn)(int, const double *, size_t, const uint8_t *, size_t, const uint8_t *, size_t, size_t,
                         size_t, double *);

static const double T95[] = {0,     0,     12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365,
                             2.306, 2.262, 2.228,  2.201, 2.179, 2.160, 2.145, 2.131, 2.120,
                             2.110, 2.101, 2.093,  2.086, 2.080, 2.074, 2.069};

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static uint64_t state = UINT64_C(0x9E3779B97F4A7C15);
static uint64_t next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}
static int cmp(const void *x, const void *y) {
  double p = *(const double *)x, q = *(const double *)y;
  return (p > q) - (p < q);
}
static double median(const double *v, int n, double *scratch) {
  memcpy(scratch, v, (size_t)n * sizeof *v);
  qsort(scratch, (size_t)n, sizeof *scratch, cmp);
  return n % 2 ? scratch[n / 2] : 0.5 * (scratch[n / 2 - 1] + scratch[n / 2]);
}
/* the geometric mean of the ratios and its 95% interval */
static void interval(const double *ratios, int n, double out[3]) {
  double m = 0.0, v = 0.0, h;
  int i;
  for (i = 0; i < n; ++i) m += log(ratios[i]);
  m /= n;
  for (i = 0; i < n; ++i) v += (log(ratios[i]) - m) * (log(ratios[i]) - m);
  v /= n - 1;
  h = (n <= 24 ? T95[n] : 1.96 + 2.4 / (n - 1)) * sqrt(v / n);
  out[0] = exp(m), out[1] = exp(m - h), out[2] = exp(m + h);
}

int main(int argc, char **argv) {
  static const struct {
    size_t in, out;
    int exact;
  } shapes[] = {{128, 128, 1}, {128, 64, 1}, {128, 344, 1}, {344, 128, 1}, {512, 512, 1}, {2048, 2048, 1},
                {128, 128, 0}};
  static const char *const names[4] = {"before", "first", "after", "A/A"};
  matvec_fn arms[4];
  int inst, rounds;
  size_t s;
  if (argc != 3 || (inst = atoi(argv[1])) < 0 || (rounds = atoi(argv[2])) < 3) {
    fprintf(stderr, "usage: kernels_ab INSTANCE ROUNDS\n");
    return 2;
  }
  arms[0] = b_bcir_ai_q8_matvec_f64_instance, arms[1] = f_bcir_ai_q8_matvec_f64_instance;
  arms[2] = a_bcir_ai_q8_matvec_f64_instance, arms[3] = b_bcir_ai_q8_matvec_f64_instance;
  for (s = 0; s < sizeof shapes / sizeof *shapes; ++s) {
    size_t in = shapes[s].in, out = shapes[s].out, n = in * out, groups = (n + 31u) / 32u, i, k;
    size_t reps = 1u + 4000000u / n;
    uint8_t *codes = malloc(n), *exps = malloc(2u * groups);
    size_t ybytes = (out * sizeof(double) + 63u) / 64u * 64u;
    double *x = malloc(in * sizeof *x), *y = aligned_alloc(64u, ybytes), *check[4], *t[4];
    double *ratio = malloc((size_t)rounds * sizeof *ratio);
    double *scratch = malloc((size_t)rounds * sizeof *scratch), iv[3][3];
    int arm, r, status[4];
    for (arm = 0; arm < 4; ++arm) {
      check[arm] = malloc(out * sizeof *check[arm]);
      t[arm] = malloc((size_t)rounds * sizeof *t[arm]);
      if (check[arm] == NULL || t[arm] == NULL) return 1;
    }
    if (codes == NULL || exps == NULL || x == NULL || y == NULL || ratio == NULL || scratch == NULL) return 1;
    for (i = 0; i < n; ++i) codes[i] = (uint8_t)(int8_t)((int)(next() % 255u) - 127);
    for (i = 0; i < groups; ++i) {
      int16_t e = (int16_t)(-9 - (int)(next() % 4u));
      if (!shapes[s].exact && i == groups / 2u) e = -1060; /* a subnormal scale: Q8-BOUND's ranges not proved */
      memcpy(exps + 2u * i, &e, 2u);
    }
    for (i = 0; i < in; ++i) x[i] = ((double)(next() % 2001u) - 1000.5) / 997.0;
    for (r = -1; r < rounds; ++r)
      for (k = 0; k < 4u; ++k) {
        double t0;
        size_t rep;
        arm = (int)((k + (size_t)(r > 0 ? r : 0)) % 4u);
        t0 = now();
        for (rep = 0; rep < reps; ++rep) (void)arms[arm](inst, x, in, codes, n, exps, groups, out, 32u, y);
        if (r >= 0) t[arm][r] = (now() - t0) / (double)reps * 1e6;
      }
    for (arm = 0; arm < 4; ++arm) status[arm] = arms[arm](inst, x, in, codes, n, exps, groups, out, 32u, check[arm]);
    for (arm = 1; arm < 4; ++arm) {
      if (status[arm] != status[0] || memcmp(check[arm], check[0], out * sizeof *check[0]) != 0) {
        fprintf(stderr, "%s's outputs differ from before's at %zu x %zu\n", names[arm], in, out);
        return 1;
      }
      for (r = 0; r < rounds; ++r) ratio[r] = t[arm][r] / t[0][r];
      interval(ratio, rounds, iv[arm - 1]);
    }
    printf("{\"instance\":%d,\"in\":%zu,\"out\":%zu,\"bound_proved\":%s,\"status\":%d,\"rounds\":%d,\"reps\":%zu,"
           "\"outputs_the_same\":true",
           inst, in, out, shapes[s].exact ? "true" : "false", status[0], rounds, reps);
    for (arm = 0; arm < 3; ++arm) printf(",\"%s_us_median\":%.4f", names[arm], median(t[arm], rounds, scratch));
    for (arm = 1; arm < 4; ++arm)
      printf(",\"%s_ratio\":[%.4f,%.4f,%.4f]", arm == 3 ? "a_a" : names[arm], iv[arm - 1][0], iv[arm - 1][1],
             iv[arm - 1][2]);
    printf("}\n");
    fflush(stdout);
    for (arm = 0; arm < 4; ++arm) free(check[arm]), free(t[arm]);
    free(codes), free(exps), free(x), free(y), free(ratio), free(scratch);
  }
  return 0;
}
