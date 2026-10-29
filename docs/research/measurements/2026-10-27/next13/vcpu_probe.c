/* NEXT-13's vCPU probe: is one of this guest's CPUs slower than another, alone or with every CPU busy?
 *
 * Three fixed workloads, each run by a thread pinned to one CPU (sched_setaffinity):
 *   chain  -- a dependent chain of 2^22 multiply-adds in double (the core's latency; the clock it runs at)
 *   fma    -- 16 independent 512-bit FMA chains, 2^20 steps (the vector unit's throughput) where AVX-512 builds,
 *             else 8 scalar chains
 *   stream -- a sum over 64 MiB, read once a pass (the memory system's bandwidth to that CPU)
 * Alone: a round runs each CPU in turn, in an order rotated round by round, the others idle. Together: a round
 * starts every CPU's thread at a barrier and times each thread's own run. Per workload and mode, each CPU's median
 * time and its ratio over the fastest CPU's median, and each CPU's 95% interval of the per-round ratio over that
 * round's fastest (Student t to 24 rounds, then 1.96 + 2.4/(n-1)). Times are CLOCK_MONOTONIC; no counter is read.
 * usage: vcpu_probe ROUNDS -> one JSON line per workload and mode */
#define _GNU_SOURCE
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

enum { MAXCPU = 64, MAXR = 201, STREAM_DOUBLES = 64 * 1024 * 1024 / 8 };

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
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
static void interval(const double *r, int n, double out[3]) {
  static const double t95[] = {0, 0, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
                               2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069};
  double m = 0, sd = 0, h;
  int i;
  for (i = 0; i < n; i++) m += log(r[i]);
  m /= n;
  for (i = 0; i < n; i++) sd += (log(r[i]) - m) * (log(r[i]) - m);
  sd = n > 1 ? sqrt(sd / (n - 1)) : 0;
  h = (n <= 24 ? t95[n] : 1.96 + 2.4 / (n - 1)) * sd / sqrt((double)n);
  out[0] = exp(m);
  out[1] = exp(m - h);
  out[2] = exp(m + h);
}

static double *stream_buf;
static volatile double sink;

static double run_chain(void) {
  double x = 1.0, a = 0.999999, b = 1e-7;
  for (long i = 0; i < (1L << 22); ++i) x = x * a + b;
  return x;
}
static double run_fma(void) {
#ifdef __AVX512F__
  __m512d acc[16], m = _mm512_set1_pd(0.999999), c = _mm512_set1_pd(1e-7);
  for (int k = 0; k < 16; ++k) acc[k] = _mm512_set1_pd(1.0 + k);
  for (long i = 0; i < (1L << 20); ++i)
    for (int k = 0; k < 16; ++k) acc[k] = _mm512_fmadd_pd(acc[k], m, c);
  __m512d s = acc[0];
  for (int k = 1; k < 16; ++k) s = _mm512_add_pd(s, acc[k]);
  return _mm512_reduce_add_pd(s);
#else
  double acc[8];
  for (int k = 0; k < 8; ++k) acc[k] = 1.0 + k;
  for (long i = 0; i < (1L << 21); ++i)
    for (int k = 0; k < 8; ++k) acc[k] = acc[k] * 0.999999 + 1e-7;
  double s = 0;
  for (int k = 0; k < 8; ++k) s += acc[k];
  return s;
#endif
}
static double run_stream(const double *buf) {
  double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
  for (long i = 0; i < STREAM_DOUBLES; i += 4) {
    s0 += buf[i];
    s1 += buf[i + 1];
    s2 += buf[i + 2];
    s3 += buf[i + 3];
  }
  return s0 + s1 + s2 + s3;
}
static double run(int workload, const double *buf) {
  return workload == 0 ? run_chain() : workload == 1 ? run_fma() : run_stream(buf);
}
static int pin(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

typedef struct {
  int cpu, workload;
  double *buf;
  double seconds;
  pthread_barrier_t *start;
} together_arg;
static void *together_thread(void *p) {
  together_arg *a = (together_arg *)p;
  if (pin(a->cpu) != 0) {
    a->seconds = -1;
    return NULL;
  }
  (void)run(a->workload, a->buf);  /* warm the CPU and its share of the buffer */
  pthread_barrier_wait(a->start);
  double t0 = now();
  sink += run(a->workload, a->buf);
  a->seconds = now() - t0;
  return NULL;
}

int main(int argc, char **argv) {
  static double t[MAXCPU][MAXR], r[MAXR];
  static const char *names[3] = {"chain", "fma", "stream"};
  int rounds, ncpu = 0, cpus[MAXCPU];
  cpu_set_t avail;
  if (argc != 2 || (rounds = atoi(argv[1])) < 5 || rounds > MAXR - 1) {
    fprintf(stderr, "usage: vcpu_probe ROUNDS\n");
    return 64;
  }
  if (sched_getaffinity(0, sizeof avail, &avail) != 0) return 2;
  for (int c = 0; c < MAXCPU; ++c)
    if (CPU_ISSET(c, &avail)) cpus[ncpu++] = c;
  double *bufs[MAXCPU];
  for (int i = 0; i < ncpu; ++i) {
    bufs[i] = (double *)aligned_alloc(64, (size_t)STREAM_DOUBLES * sizeof(double));
    if (!bufs[i]) return 3;
    for (long k = 0; k < STREAM_DOUBLES; ++k) bufs[i][k] = (double)(k & 1023) * 1e-3;
  }
  stream_buf = bufs[0];
  for (int mode = 0; mode < 2; ++mode)
    for (int w = 0; w < 3; ++w) {
      for (int k = -1; k < rounds; ++k) {
        if (mode == 0) {
          for (int j = 0; j < ncpu; ++j) {
            int i = (j + (k > 0 ? k : 0)) % ncpu;
            if (pin(cpus[i]) != 0) return 4;
            (void)run(w, bufs[i]);  /* the CPU warm, its buffer in its caches as far as they hold it */
            double t0 = now();
            sink += run(w, bufs[i]);
            if (k >= 0) t[i][k] = now() - t0;
          }
        } else {
          pthread_t th[MAXCPU];
          together_arg args[MAXCPU];
          pthread_barrier_t start;
          pthread_barrier_init(&start, NULL, (unsigned)ncpu);
          for (int i = 0; i < ncpu; ++i) {
            args[i] = (together_arg){cpus[i], w, bufs[i], 0, &start};
            pthread_create(&th[i], NULL, together_thread, &args[i]);
          }
          for (int i = 0; i < ncpu; ++i) pthread_join(th[i], NULL);
          pthread_barrier_destroy(&start);
          for (int i = 0; i < ncpu; ++i) {
            if (args[i].seconds < 0) return 5;
            if (k >= 0) t[i][k] = args[i].seconds;
          }
        }
      }
      double med[MAXCPU], best = 1e300;
      for (int i = 0; i < ncpu; ++i) {
        med[i] = median(t[i], rounds);
        if (med[i] < best) best = med[i];
      }
      printf("{\"workload\":\"%s\",\"mode\":\"%s\",\"rounds\":%d,\"cpus\":[", names[w], mode ? "together" : "alone",
             rounds);
      for (int i = 0; i < ncpu; ++i) {
        double ci[3];
        for (int k = 0; k < rounds; ++k) {
          double fastest = 1e300;
          for (int j = 0; j < ncpu; ++j)
            if (t[j][k] < fastest) fastest = t[j][k];
          r[k] = t[i][k] / fastest;
        }
        interval(r, rounds, ci);
        printf("%s{\"cpu\":%d,\"median_ms\":%.3f,\"over_fastest_median\":%.4f,\"over_round_fastest\":%.4f,"
               "\"interval\":[%.4f,%.4f]}",
               i ? "," : "", cpus[i], med[i] * 1e3, med[i] / best, ci[0], ci[1], ci[2]);
      }
      printf("]}\n");
      fflush(stdout);
    }
  return 0;
}
