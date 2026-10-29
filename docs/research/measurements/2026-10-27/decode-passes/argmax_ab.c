/* DECODE-PASSES' argmax A/B: the greedy pick of the tree before (the scalar loop, each comparison waiting on the last
 * best) against the tree after's (four lanes, lane = v > lane ? v : lane -- MAXPD's rule, the second operand kept on
 * a tie or a NaN -- a tail, a reduction and a first-equal scan), each copied verbatim from its tree's
 * bcir_llama_op_argmax with the model's vocabulary size a parameter.
 *
 * First the two are held to the same pick over 200,000 vocabularies of 1 to 40 logits drawn from -0.0, +0.0, +inf,
 * -inf, NaN, small integers (ties) and random values (status 2 on any difference). Then each is timed on 1,024
 * logits (the 860K lab model's vocabulary) and 32,000 (TinyLlama-1.1B's): a round runs the scalar loop, the lanes
 * and the scalar loop again (the A/A control) in an order rotated round by round, each arm 200 picks; an arm's ratio
 * over the first scalar arm's in the same round is the geometric mean over the rounds with its 95% interval (Student
 * t to 24 rounds, then 1.96 + 2.4/(n-1)).
 * usage: argmax_ab ROUNDS  -> one JSON line per vocabulary */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAXR = 401 };

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
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
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(const double *x, int n) {
  static double s[MAXR];
  memcpy(s, x, (size_t)n * sizeof(double));
  qsort(s, (size_t)n, sizeof(double), cmpd);
  return n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/* the tree before's bcir_llama_op_argmax (0311c60), m->vocab_size as n */
__attribute__((noinline)) static int32_t argmax_before(uint32_t n, const double *logits) {
  uint32_t token, best = 0;
  for (token = 1; token < n; ++token)
    if (logits[token] > logits[best]) best = token;
  return (int32_t)best;
}
/* the tree after's bcir_llama_op_argmax (DECODE-PASSES), m->vocab_size as n */
__attribute__((noinline)) static int32_t argmax_after(uint32_t n, const double *logits) {
  uint32_t token;
  double top, a, b, c, e;
  if (n == 0u || logits[0] != logits[0]) return 0;
  top = a = b = c = e = logits[0];
  for (token = 1u; n - token >= 4u; token += 4u) {
    a = logits[token] > a ? logits[token] : a;
    b = logits[token + 1u] > b ? logits[token + 1u] : b;
    c = logits[token + 2u] > c ? logits[token + 2u] : c;
    e = logits[token + 3u] > e ? logits[token + 3u] : e;
  }
  for (; token < n; ++token) top = logits[token] > top ? logits[token] : top;
  top = a > top ? a : top;
  top = b > top ? b : top;
  top = c > top ? c : top;
  top = e > top ? e : top;
  for (token = 0u; token < n; ++token)
    if (logits[token] == top) return (int32_t)token;
  return 0; /* unreachable: top is one of the logits */
}

static uint64_t s = UINT64_C(88172645463325252);
static uint64_t next(void) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}
static double rnd(void) { return (double)(next() % 2000001u) / 1000.0 - 1000.0; }

int main(int argc, char **argv) {
  static double l[32000];
  static double t[3][MAXR], r[MAXR];
  static const uint32_t vocab[2] = {1024u, 32000u};
  int rounds, cases = 0;
  if (argc != 2 || (rounds = atoi(argv[1])) < 5 || rounds > MAXR - 1) {
    fprintf(stderr, "usage: argmax_ab ROUNDS\n");
    return 64;
  }
  for (int trial = 0; trial < 200000; ++trial) {
    uint32_t n = 1u + (uint32_t)(next() % 40u);
    for (uint32_t i = 0; i < n; ++i) {
      double v = rnd();
      uint64_t k = next() % 16u;
      l[i] = k == 0 ? -0.0 : k == 1 ? 0.0 : k == 2 ? INFINITY : k == 3 ? -INFINITY : k == 4 ? NAN
             : k < 8 ? (double)(int)(v / 400.0) : v;
    }
    cases++;
    if (argmax_before(n, l) != argmax_after(n, l)) {
      printf("{\"error\":\"trial %d (n %u): the two picks differ\"}\n", trial, n);
      return 2;
    }
  }
  printf("{\"picks\":\"the tree after's pick the tree before's on every case\",\"cases\":%d}\n", cases);
  for (int which = 0; which < 2; ++which) {
    uint32_t n = vocab[which];
    volatile int32_t sink = 0;
    double ci[3];
    for (uint32_t i = 0; i < n; ++i) l[i] = rnd();
    for (int k = -1; k < rounds; ++k)
      for (int a = 0; a < 3; ++a) {
        int arm = (a + (k > 0 ? k : 0)) % 3;
        double t0 = now();
        for (int rep = 0; rep < 200; ++rep) sink += arm == 1 ? argmax_after(n, l) : argmax_before(n, l);
        if (k >= 0) t[arm][k] = (now() - t0) / 200.0 * 1e9;
      }
    printf("{\"vocab\":%u,\"rounds\":%d,\"before_ns\":%.1f,\"after_ns\":%.1f,\"aa_ns\":%.1f", n, rounds,
           median(t[0], rounds), median(t[1], rounds), median(t[2], rounds));
    for (int k = 0; k < rounds; ++k) r[k] = t[1][k] / t[0][k];
    interval(r, rounds, ci);
    printf(",\"after_over_before\":%.4f,\"interval\":[%.4f,%.4f]", ci[0], ci[1], ci[2]);
    for (int k = 0; k < rounds; ++k) r[k] = t[2][k] / t[0][k];
    interval(r, rounds, ci);
    printf(",\"aa_over_before\":%.4f,\"aa_interval\":[%.4f,%.4f],\"sink\":%d}\n", ci[0], ci[1], ci[2], (int)(sink & 1));
  }
  return 0;
}
