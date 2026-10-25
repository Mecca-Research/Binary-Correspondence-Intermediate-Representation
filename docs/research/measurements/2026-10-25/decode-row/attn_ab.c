/* DECODE-ROW's kernel A/B, one process: the attention row and the rotation as the tree before computed them
 * (BEFORE: its bcir_decode.c, every global renamed before_*) against the tree after's, and BEFORE again (the A/A
 * control). A case is a request's decode steps at a model's shapes: for every context t = 1..T and every layer, the
 * cached row of every head (bcir_gqa_attention_row, as bcir_llama_op_attention calls it), or the rotation of every
 * query and key head at position t - 1 (BEFORE: bcir_rope a head at a time, as bcir_llama_op_rope called it; AFTER:
 * bcir_rope_table once a position and bcir_rope_apply a head, as it calls them now). Per case a warm round, then
 * ROUNDS rounds in rotated arm order; each arm's ratio over BEFORE's in the same round, the geometric mean and its 95%
 * interval (Student t to 24 rounds, then 1.96 + 2.4/(n-1)). Before the timed rounds, an untimed pass of each arm
 * hashes every output of every call: AFTER's and the A/A's must be BEFORE's bytes (the run exits otherwise).
 * usage: attn_ab ROUNDS  -> one JSON line a case */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decode.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void before_bcir_rope(const double *x, int rows, int dim, double base, int pos_offset, double *out);
int before_bcir_gqa_attention_row(const double *q_row, const double *k_rows, const double *v_rows, int t_len,
                                  int n_heads, int n_kv_heads, int d_k, double *scores, double *out);

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
static uint64_t fold(uint64_t h, const double *v, size_t n) { /* FNV-1a over the bytes */
  const unsigned char *p = (const unsigned char *)v;
  size_t i;
  for (i = 0; i < n * sizeof(double); i++) h = (h ^ p[i]) * UINT64_C(0x100000001B3);
  return h;
}
static uint64_t state = UINT64_C(0x9E3779B97F4A7C15);
static double next(void) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return (double)(state >> 11) / 9007199254740992.0 * 2.0 - 1.0;
}

struct shape {
  const char *name;
  int heads, kv_heads, dk, layers;
  double base;
};

/* one arm's pass over a case (0 BEFORE, 1 AFTER): with `hash`, the hash of every output, else 0 (timed) */
static uint64_t pass(int after, int rope, int hash, const struct shape *m, int T, const double *q, const double *k,
                     const double *v, double *s, double *o, double *cs) {
  uint64_t h = UINT64_C(0xCBF29CE484222325);
  size_t kvd = (size_t)m->kv_heads * m->dk, d = (size_t)m->heads * m->dk;
  int t, l, i;
  for (t = 1; t <= T; t++) {
    if (rope && after) bcir_rope_table(m->dk, m->base, (double)(t - 1), cs);
    for (l = 0; l < m->layers; l++) {
      const double *ql = q + ((size_t)l * T + (size_t)(t - 1)) * (d + kvd);
      if (rope) {
        for (i = 0; i < m->heads + m->kv_heads; i++) {
          if (after) bcir_rope_apply(ql + (size_t)i * m->dk, m->dk, cs, o + (size_t)i * m->dk);
          else before_bcir_rope(ql + (size_t)i * m->dk, 1, m->dk, m->base, t - 1, o + (size_t)i * m->dk);
        }
        if (hash) h = fold(h, o, d + kvd);
      } else {
        const double *kl = k + (size_t)l * T * kvd, *vl = v + (size_t)l * T * kvd;
        if ((after ? bcir_gqa_attention_row : before_bcir_gqa_attention_row)(ql, kl, vl, t, m->heads, m->kv_heads,
                                                                              m->dk, s, o))
          exit(3);
        if (hash) h = fold(h, o, d);
      }
    }
  }
  return hash ? h : 0;
}

int main(int argc, char **argv) {
  static const struct shape shapes[] = {{"860K", 4, 2, 32, 4, 10000.0}, {"TinyLlama", 32, 4, 64, 1, 10000.0}};
  static const int lens[2][4] = {{8, 64, 205, 1024}, {64, 256, 1024, 0}};
  static const char *names[3] = {"before", "after", "A/A"};
  static double ms[3][MAXR];
  int rounds, si, li, rope, r, a;
  if (argc != 2 || (rounds = atoi(argv[1])) < 2 || rounds > MAXR) {
    fprintf(stderr, "usage: attn_ab ROUNDS\n");
    return 2;
  }
  for (rope = 0; rope < 2; rope++)
    for (si = 0; si < 2; si++)
      for (li = 0; li < 4 && lens[si][li]; li++) {
        const struct shape *m = &shapes[si];
        int T = lens[si][li];
        size_t kvd = (size_t)m->kv_heads * m->dk, d = (size_t)m->heads * m->dk, i;
        size_t qn = (size_t)m->layers * T * (d + kvd), kn = (size_t)m->layers * T * kvd;
        double *q = malloc(qn * sizeof *q), *k = malloc(kn * sizeof *k), *v = malloc(kn * sizeof *v);
        double *s = malloc((size_t)T * sizeof *s), *o = malloc((d + kvd) * sizeof *o);
        double *cs = malloc((size_t)m->dk * sizeof *cs);
        uint64_t want = 0;
        if (!q || !k || !v || !s || !o || !cs) return 1;
        for (i = 0; i < qn; i++) q[i] = next();
        for (i = 0; i < kn; i++) k[i] = next(), v[i] = next();
        want = pass(0, rope, 1, m, T, q, k, v, s, o, cs);
        for (a = 1; a < 3; a++)
          if (pass(a == 1, rope, 1, m, T, q, k, v, s, o, cs) != want) {
            fprintf(stderr, "%s's outputs differ from before's: %s T %d %s\n", names[a], m->name, T,
                    rope ? "rope" : "attention");
            return 1;
          }
        for (r = -1; r < rounds; r++)
          for (a = 0; a < 3; a++) {
            int arm = (a + (r < 0 ? 0 : r)) % 3;
            double t0 = now();
            pass(arm == 1, rope, 0, m, T, q, k, v, s, o, cs);
            if (r >= 0) ms[arm][r] = (now() - t0) * 1e3;
          }
        printf("{\"case\":\"%s %s, T %d\",\"heads\":%d,\"kv_heads\":%d,\"head_dim\":%d,\"layers\":%d,\"rounds\":%d,"
               "\"every_output_before_s_bytes\":true,\"before_ms\":%.5f",
               rope ? "rope" : "attention row", m->name, T, m->heads, m->kv_heads, m->dk, m->layers, rounds,
               median(ms[0], rounds));
        for (a = 1; a < 3; a++) {
          double ratio[MAXR], ci[3];
          for (r = 0; r < rounds; r++) ratio[r] = ms[a][r] / ms[0][r];
          interval(ratio, rounds, ci);
          printf(",\"%s\":{\"ms\":%.5f,\"ratio\":%.4f,\"interval\":[%.4f,%.4f]}", names[a], median(ms[a], rounds),
                 ci[0], ci[1], ci[2]);
        }
        printf("}\n");
        fflush(stdout);
        free(q), free(k), free(v), free(s), free(o), free(cs);
      }
  return 0;
}
