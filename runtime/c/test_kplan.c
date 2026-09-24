/*===- test_kplan.c - the native K_BCIR planner's harness (G17, S4-A) -------===
 *
 * Hosted (libc) driver for bcir_kplan.{h,c}; the grader is bcir/tests/planner_fixtures.py.
 *
 *   test_kplan --plan IN OUT         decode + plan one BKPI record; OUT gets the BKPR record;
 *                                    prints the first refusal's status name, or "OK <bytes>"
 *   test_kplan --batch IN OUT        IN: a sequence of (u32 length, BKPI bytes); OUT: for each,
 *                                    (u32 status, u32 length, BKPR bytes) -- one process for a
 *                                    whole corpus
 *   test_kplan --decode-realization IN   prints the BKPR decoder's status name
 *   test_kplan --api                 the fail-closed API laws (NULL, short scratch, short output,
 *                                    zeroing on failure); prints "API OK <checks>"
 *   test_kplan --bench IN REPS ROUNDS    the median over ROUNDS of REPS plans, in ns per plan
 *===----------------------------------------------------------------------===*/
#define _POSIX_C_SOURCE 200809L /* clock_gettime under a strict -std */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bcir_kplan.h"

static const char *status_name(bcir_status s) {
  switch (s) {
    case BCIR_OK:              return "BCIR_OK";
    case BCIR_ERR_TRUNCATED:   return "BCIR_ERR_TRUNCATED";
    case BCIR_ERR_MAGIC:       return "BCIR_ERR_MAGIC";
    case BCIR_ERR_VERSION:     return "BCIR_ERR_VERSION";
    case BCIR_ERR_CRC:         return "BCIR_ERR_CRC";
    case BCIR_ERR_NOSPACE:     return "BCIR_ERR_NOSPACE";
    case BCIR_ERR_LANE:        return "BCIR_ERR_LANE";
    case BCIR_ERR_WIDTH:       return "BCIR_ERR_WIDTH";
    case BCIR_ERR_DISPATCH:    return "BCIR_ERR_DISPATCH";
    case BCIR_ERR_PROVENANCE:  return "BCIR_ERR_PROVENANCE";
    case BCIR_ERR_STALE:       return "BCIR_ERR_STALE";
    case BCIR_ERR_OVERFLOW:    return "BCIR_ERR_OVERFLOW";
    case BCIR_ERR_TRAILING:    return "BCIR_ERR_TRAILING";
    case BCIR_ERR_RESERVED:    return "BCIR_ERR_RESERVED";
    case BCIR_ERR_UTF8:        return "BCIR_ERR_UTF8";
    case BCIR_ERR_GENERATION:  return "BCIR_ERR_GENERATION";
    case BCIR_ERR_PLAN:        return "BCIR_ERR_PLAN";
    case BCIR_ERR_CONTROL:     return "BCIR_ERR_CONTROL";
    case BCIR_ERR_MAC:         return "BCIR_ERR_MAC";
    case BCIR_ERR_TELEMETRY:   return "BCIR_ERR_TELEMETRY";
    case BCIR_ERR_RING:        return "BCIR_ERR_RING";
    case BCIR_ERR_FULL:        return "BCIR_ERR_FULL";
    case BCIR_ERR_BUSY:        return "BCIR_ERR_BUSY";
    case BCIR_ERR_LIFETIME:    return "BCIR_ERR_LIFETIME";
    case BCIR_ERR_SHARD:       return "BCIR_ERR_SHARD";
    case BCIR_ERR_PLANNER:     return "BCIR_ERR_PLANNER";
  }
  return "BCIR_ERR_UNKNOWN";
}

static uint8_t *read_file(const char *path, size_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  size_t cap = 1u << 16, n = 0;
  uint8_t *buf = (uint8_t *)malloc(cap);
  for (;;) {
    if (!buf) break;
    size_t got = fread(buf + n, 1, cap - n, f);
    n += got;
    if (n < cap) break;
    uint8_t *grown = (uint8_t *)realloc(buf, cap * 2u);
    if (!grown) {
      free(buf);
      buf = NULL;
      break;
    }
    buf = grown;
    cap *= 2u;
  }
  fclose(f);
  *len = n;
  return buf;
}

static int write_file(const char *path, const uint8_t *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return 0;
  size_t put = len ? fwrite(data, 1, len, f) : 0;
  return fclose(f) == 0 && put == len;
}

/* Decode + plan one record into a fresh buffer; the first refusal's status. */
static bcir_status plan_one(const uint8_t *data, size_t len, uint8_t **out, size_t *out_len) {
  bcir_kp_input in;
  *out = NULL;
  *out_len = 0;
  bcir_status st = bcir_kp_decode_input(data, len, &in);
  if (st != BCIR_OK) return st;
  size_t scratch_len = 0, cap = 0;
  st = bcir_kp_scratch_size(&in, &scratch_len);
  if (st == BCIR_OK) st = bcir_kp_realization_size(&in, &cap);
  if (st != BCIR_OK) return st;
  void *scratch = malloc(scratch_len ? scratch_len : 1u);
  uint8_t *buf = (uint8_t *)malloc(cap);
  if (!scratch || !buf) {
    free(scratch);
    free(buf);
    return BCIR_ERR_NOSPACE;
  }
  st = bcir_kp_plan(&in, scratch, scratch_len, buf, cap, out_len);
  free(scratch);
  if (st != BCIR_OK) {
    free(buf);
    *out_len = 0;
    return st;
  }
  /* the planner's own output must decode */
  bcir_kp_realization r;
  bcir_status back = bcir_kp_decode_realization(buf, *out_len, &r);
  if (back != BCIR_OK || r.n_steps != in.n_claims) {
    fprintf(stderr, "the planner wrote a realization its decoder refuses (%s)\n",
            status_name(back));
    exit(3);
  }
  *out = buf;
  return BCIR_OK;
}

static uint32_t rd32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static void put32(FILE *f, uint32_t v) {
  uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
  fwrite(b, 1, 4, f);
}

static int run_batch(const char *in_path, const char *out_path) {
  size_t len = 0;
  uint8_t *all = read_file(in_path, &len);
  if (!all) return 2;
  FILE *out = fopen(out_path, "wb");
  if (!out) {
    free(all);
    return 2;
  }
  size_t pos = 0, cases = 0;
  while (pos + 4u <= len) {
    size_t n = rd32(all + pos);
    pos += 4u;
    if (n > len - pos) {
      fprintf(stderr, "batch: a record runs past the end\n");
      fclose(out);
      free(all);
      return 2;
    }
    uint8_t *plan = NULL;
    size_t plan_len = 0;
    bcir_status st = plan_one(all + pos, n, &plan, &plan_len);
    put32(out, (uint32_t)st);
    put32(out, (uint32_t)plan_len);
    if (plan_len) fwrite(plan, 1, plan_len, out);
    free(plan);
    pos += n;
    cases++;
  }
  int ok = pos == len;
  if (fclose(out) != 0) ok = 0;
  free(all);
  printf("BATCH %zu\n", cases);
  return ok ? 0 : 2;
}

static int checks = 0, failures = 0;
#define CHECK(cond, what)                     \
  do {                                        \
    checks++;                                 \
    if (!(cond)) {                            \
      failures++;                             \
      fprintf(stderr, "API FAIL: %s\n", what); \
    }                                         \
  } while (0)

static int all_zero(const uint8_t *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (p[i]) return 0;
  return 1;
}

/* The fail-closed API laws, over one valid record (argv) and its plan. */
static int run_api(const char *path) {
  size_t len = 0;
  uint8_t *data = read_file(path, &len);
  if (!data) return 2;
  bcir_kp_input in, zero_in;
  memset(&zero_in, 0, sizeof zero_in);
  CHECK(bcir_kp_decode_input(data, len, NULL) == BCIR_ERR_NOSPACE, "decode with no out");
  memset(&in, 0x5A, sizeof in);
  CHECK(bcir_kp_decode_input(NULL, len, &in) == BCIR_ERR_TRUNCATED, "decode NULL data");
  CHECK(all_zero((const uint8_t *)&in, sizeof in), "a refused decode zeroes its view");
  CHECK(bcir_kp_decode_input(data, 0, &in) == BCIR_ERR_TRUNCATED, "decode of no bytes");
  CHECK(bcir_kp_decode_input(data, len, &in) == BCIR_OK, "decode of the valid record");
  size_t scratch_len = 0, cap = 0, out_len = 7;
  CHECK(bcir_kp_scratch_size(NULL, &scratch_len) == BCIR_ERR_NOSPACE, "scratch_size NULL");
  CHECK(bcir_kp_scratch_size(&zero_in, &scratch_len) == BCIR_ERR_NOSPACE,
        "scratch_size of an undecoded view");
  CHECK(bcir_kp_scratch_size(&in, NULL) == BCIR_ERR_NOSPACE, "scratch_size with no out");
  CHECK(bcir_kp_scratch_size(&in, &scratch_len) == BCIR_OK, "scratch_size");
  CHECK(bcir_kp_realization_size(&in, &cap) == BCIR_OK, "realization_size");
  CHECK(cap == BCIR_KP_REALIZATION_HEADER_SIZE + BCIR_KP_STEP_SIZE * (size_t)in.n_claims +
                   BCIR_KP_CRC_SIZE,
        "realization_size is 32 + 120n + 4");
  uint8_t *scratch = (uint8_t *)malloc(scratch_len + 1u);
  uint8_t *out = (uint8_t *)malloc(cap + 16u);
  if (!scratch || !out) {
    free(scratch);
    free(out);
    free(data);
    return 2;
  }
  memset(out, 0xA5, cap + 16u);
  CHECK(bcir_kp_plan(&in, scratch, scratch_len, out, cap, NULL) == BCIR_ERR_NOSPACE,
        "plan with no out_len");
  CHECK(all_zero(out, cap), "a refused plan zeroes the output");
  memset(out, 0xA5, cap + 16u);
  out_len = 7;
  if (scratch_len) {
    CHECK(bcir_kp_plan(&in, scratch, scratch_len - 1u, out, cap, &out_len) == BCIR_ERR_NOSPACE,
          "plan with a short scratch");
    CHECK(out_len == 0 && all_zero(out, cap), "a short scratch zeroes the output");
  }
  memset(out, 0xA5, cap + 16u);
  out_len = 7;
  CHECK(bcir_kp_plan(&in, scratch, scratch_len, out, cap - 1u, &out_len) == BCIR_ERR_NOSPACE,
        "plan with a short output");
  CHECK(out_len == 0 && all_zero(out, cap - 1u), "a short output is zeroed");
  CHECK(bcir_kp_plan(&in, NULL, scratch_len, out, cap, &out_len) == BCIR_ERR_NOSPACE ||
            scratch_len == 0,
        "plan with no scratch");
  CHECK(bcir_kp_plan(&zero_in, scratch, scratch_len, out, cap, &out_len) == BCIR_ERR_NOSPACE,
        "plan of an undecoded view");
  /* a hand-built view whose width count the geometry cannot index by: refused, never read */
  bcir_kp_input bad = in;
  size_t bad_len = 1;
  bad.n_widths = 0;
  CHECK(bcir_kp_scratch_size(&bad, &bad_len) == BCIR_ERR_PLANNER && bad_len == 0,
        "scratch_size of a view with no lane widths");
  bad.n_widths = BCIR_KP_WIDTHS_MAX + 1u;
  memset(out, 0xA5, cap + 16u);
  out_len = 7;
  CHECK(bcir_kp_plan(&bad, scratch, scratch_len, out, cap, &out_len) == BCIR_ERR_PLANNER &&
            out_len == 0 && all_zero(out, cap),
        "plan of a view with too many lane widths");
  /* the plan, twice: same bytes (no state survives a call) and misaligned scratch is fine */
  CHECK(bcir_kp_plan(&in, scratch, scratch_len, out, cap, &out_len) == BCIR_OK && out_len == cap,
        "plan");
  uint8_t *again = (uint8_t *)malloc(cap);
  size_t again_len = 0;
  if (!again) {
    free(scratch);
    free(out);
    free(data);
    return 2;
  }
  memset(scratch, 0xFF, scratch_len + 1u);
  CHECK(bcir_kp_plan(&in, scratch + 1, scratch_len, again, cap, &again_len) == BCIR_OK &&
            again_len == cap && memcmp(out, again, cap) == 0,
        "a replan over dirty, misaligned scratch writes the same bytes");
  bcir_kp_realization r;
  CHECK(bcir_kp_decode_realization(out, out_len, &r) == BCIR_OK && r.n_steps == in.n_claims,
        "the plan decodes");
  CHECK(bcir_kp_decode_realization(out, out_len, NULL) == BCIR_ERR_NOSPACE,
        "decode_realization with no out");
  memset(&r, 0x5A, sizeof r);
  CHECK(bcir_kp_decode_realization(out, out_len - 1u, &r) != BCIR_OK &&
            all_zero((const uint8_t *)&r, sizeof r),
        "a refused realization zeroes its view");
  free(again);
  free(scratch);
  free(out);
  free(data);
  if (failures) return 1;
  printf("API OK %d\n", checks);
  return 0;
}

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return x < y ? -1 : x > y;
}

static int run_bench(const char *path, long reps, long rounds) {
  size_t len = 0;
  uint8_t *data = read_file(path, &len);
  bcir_kp_input in;
  if (!data || reps < 1 || rounds < 1 || rounds > 64) return 2;
  if (bcir_kp_decode_input(data, len, &in) != BCIR_OK) return 2;
  size_t scratch_len = 0, cap = 0, out_len = 0;
  if (bcir_kp_scratch_size(&in, &scratch_len) != BCIR_OK ||
      bcir_kp_realization_size(&in, &cap) != BCIR_OK)
    return 2;
  void *scratch = malloc(scratch_len ? scratch_len : 1u);
  uint8_t *out = (uint8_t *)malloc(cap);
  if (!scratch || !out) {
    free(scratch);
    free(out);
    free(data);
    return 2;
  }
  uint64_t per[64];
  for (long r = 0; r < rounds; r++) {
    uint64_t t0 = now_ns();
    for (long i = 0; i < reps; i++)
      if (bcir_kp_plan(&in, scratch, scratch_len, out, cap, &out_len) != BCIR_OK) return 2;
    per[r] = (now_ns() - t0) / (uint64_t)reps;
  }
  qsort(per, (size_t)rounds, sizeof per[0], cmp_u64);
  printf("BENCH %llu\n", (unsigned long long)per[rounds / 2]);
  free(scratch);
  free(out);
  free(data);
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 4 && strcmp(argv[1], "--plan") == 0) {
    size_t len = 0;
    uint8_t *data = read_file(argv[2], &len);
    if (!data) return 2;
    uint8_t *plan = NULL;
    size_t plan_len = 0;
    bcir_status st = plan_one(data, len, &plan, &plan_len);
    free(data);
    if (st != BCIR_OK) {
      printf("%s\n", status_name(st));
      return 0;
    }
    int ok = write_file(argv[3], plan, plan_len);
    free(plan);
    if (!ok) return 2;
    printf("OK %zu\n", plan_len);
    return 0;
  }
  if (argc == 4 && strcmp(argv[1], "--batch") == 0) return run_batch(argv[2], argv[3]);
  if (argc == 3 && strcmp(argv[1], "--decode-realization") == 0) {
    size_t len = 0;
    uint8_t *data = read_file(argv[2], &len);
    if (!data) return 2;
    bcir_kp_realization r;
    printf("%s\n", status_name(bcir_kp_decode_realization(data, len, &r)));
    free(data);
    return 0;
  }
  if (argc == 3 && strcmp(argv[1], "--api") == 0) return run_api(argv[2]);
  if (argc == 5 && strcmp(argv[1], "--bench") == 0)
    return run_bench(argv[2], strtol(argv[3], NULL, 10), strtol(argv[4], NULL, 10));
  fprintf(stderr,
          "usage: test_kplan --plan IN OUT | --batch IN OUT | --decode-realization IN |"
          " --api IN | --bench IN REPS ROUNDS\n");
  return 2;
}
