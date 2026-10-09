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
 *   test_kplan --bench-floor IN REPS ROUNDS   the same median for writing the realization's bytes
 *                                    once -- the floor of `planner.native.scale4`; prints
 *                                    "FLOOR <ns> <bytes>"
 *   test_kplan --hydrate IN PLAN BIND OUT   the native hydrate (CXX4): BKPI + BKPR + BKPB ->
 *                                    the StreamPack in OUT; prints the first refusal's status
 *                                    name, or "OK <bytes>"
 *   test_kplan --hydrate-batch IN OUT  IN: a sequence of (u32 length, BKPI, u32 length, BKPR,
 *                                    u32 length, BKPB); OUT: for each, (u32 status, u32 length,
 *                                    StreamPack bytes)
 *   test_kplan --api-hydrate IN      the hydrate's fail-closed API laws over the seed's own plan
 *                                    and a binding built here; prints "API OK <checks>"
 *   test_kplan --bench-hydrate IN PLAN BIND REPS ROUNDS   the median hydrate, in ns
 *   test_kplan --bench-hydrate-floor IN PLAN BIND REPS ROUNDS   the same median for writing the
 *                                    pack's bytes once -- the floor of `hydrate.native.scale4`;
 *                                    prints "FLOOR <ns> <bytes>"
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

/* The floor of the native row (tools/perf/gemplus_baseline.py, `planner.native.scale4`): writing
 * the realization once. BKPR is fixed-size -- 32 + 120 x claims + 4 bytes, every one of them
 * written by a plan (the steps, the score, the CRC over them) -- so storing its bytes once is
 * work no planner avoids, measured beside `--bench` under the same conditions: the same record,
 * a warm buffer of the same size reused across repetitions, the same clock and median. The store
 * goes through a volatile function pointer so the compiler cannot drop a buffer nothing reads. */
static void *(*volatile floor_memset)(void *, int, size_t) = memset;

static int run_bench_floor(const char *path, long reps, long rounds) {
  size_t len = 0;
  uint8_t *data = read_file(path, &len);
  bcir_kp_input in;
  size_t cap = 0;
  if (!data || reps < 1 || rounds < 1 || rounds > 64 ||
      bcir_kp_decode_input(data, len, &in) != BCIR_OK ||
      bcir_kp_realization_size(&in, &cap) != BCIR_OK) {
    free(data);
    return 2;
  }
  uint8_t *out = (uint8_t *)malloc(cap);
  if (!out) {
    free(data);
    return 2;
  }
  size_t wrote = cap; /* the whole realization, and the size the grader holds to 32 + 120n + 4 */
  floor_memset(out, 0, wrote); /* the bench's buffer is warm after its first plan: so is this one */
  uint64_t per[64];
  for (long r = 0; r < rounds; r++) {
    uint64_t t0 = now_ns();
    for (long i = 0; i < reps; i++) floor_memset(out, (int)(i & 0xFF), wrote);
    per[r] = (now_ns() - t0) / (uint64_t)reps;
  }
  qsort(per, (size_t)rounds, sizeof per[0], cmp_u64);
  printf("FLOOR %llu %zu\n", (unsigned long long)per[rounds / 2], wrote);
  free(out);
  free(data);
  return 0;
}

/* ---- the native hydrate (CXX4) --------------------------------------------------------- */

static void put32m(uint8_t *p, uint32_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8);
  p[2] = (uint8_t)(v >> 16);
  p[3] = (uint8_t)(v >> 24);
}

/* A binding for a decoded input with RID i + 1 at index i, a generation for each declared
 * resource and the plan "plan0" (what the oracle writes for a module whose RIDs are those). */
static uint8_t *make_binding(const bcir_kp_input *in, size_t *len) {
  uint32_t n_gens = 0;
  for (uint32_t r = 0; r < in->n_resources; r++)
    if (in->data[in->off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)r]) n_gens++;
  size_t n = BCIR_KP_BINDING_HEADER_SIZE + 4u * (size_t)in->n_resources + 12u * (size_t)n_gens +
             5u + BCIR_KP_CRC_SIZE;
  uint8_t *b = (uint8_t *)calloc(n, 1);
  if (!b) return NULL;
  memcpy(b, "BKPB", 4);
  put32m(b + 8, in->n_resources);
  put32m(b + 12, n_gens);
  put32m(b + 16, 1u);
  put32m(b + 20, 5u);
  size_t at = BCIR_KP_BINDING_HEADER_SIZE;
  for (uint32_t r = 0; r < in->n_resources; r++, at += 4u) put32m(b + at, r + 1u);
  for (uint32_t r = 0; r < in->n_resources; r++) {
    if (!in->data[in->off_resources + BCIR_KP_RESOURCE_SIZE * (size_t)r]) continue;
    put32m(b + at, r + 1u);
    put32m(b + at + 4u, r % 3u);
    put32m(b + at + 8u, r % 5u);
    at += 12u;
  }
  memcpy(b + at, "plan0", 5);
  at += 5u;
  put32m(b + at, bcir_crc32(b, at));
  *len = n;
  return b;
}

/* Decode the three records and hydrate; the first refusal's status. A pack the native hydrate
 * writes must be one the runtime's own reader accepts in full (bcir_sp_verify_semantic). */
static bcir_status hydrate_one(const uint8_t *ki, size_t ki_len, const uint8_t *kr, size_t kr_len,
                               const uint8_t *kb, size_t kb_len, uint8_t **out, size_t *out_len) {
  bcir_kp_input in;
  bcir_kp_realization r;
  bcir_kp_binding b;
  *out = NULL;
  *out_len = 0;
  bcir_status st = bcir_kp_decode_input(ki, ki_len, &in);
  if (st == BCIR_OK) st = bcir_kp_decode_realization(kr, kr_len, &r);
  if (st == BCIR_OK) st = bcir_kp_decode_binding(kb, kb_len, &b);
  size_t scratch_len = 0, cap = 0;
  if (st == BCIR_OK) st = bcir_kp_hydrate_scratch_size(&in, &b, &scratch_len);
  if (st != BCIR_OK) return st;
  void *scratch = malloc(scratch_len ? scratch_len : 1u);
  if (!scratch) return BCIR_ERR_NOSPACE;
  st = bcir_kp_hydrate_size(&in, &r, &b, scratch, scratch_len, &cap);
  uint8_t *buf = st == BCIR_OK ? (uint8_t *)malloc(cap) : NULL;
  if (st == BCIR_OK && !buf) st = BCIR_ERR_NOSPACE;
  if (st == BCIR_OK) st = bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len, buf, cap, out_len);
  free(scratch);
  if (st != BCIR_OK) {
    free(buf);
    *out_len = 0;
    return st;
  }
  bcir_streampack_header hdr;
  bcir_status back = bcir_sp_validate(buf, *out_len, &hdr);
  if (back == BCIR_OK) back = bcir_sp_verify_semantic(buf, *out_len, hdr.map_gen, hdr.data_gen);
  if (back != BCIR_OK || *out_len != cap) {
    fprintf(stderr, "the native hydrate wrote a StreamPack the runtime refuses (%s)\n",
            status_name(back));
    exit(3);
  }
  *out = buf;
  return BCIR_OK;
}

static int run_hydrate_batch(const char *in_path, const char *out_path) {
  size_t len = 0;
  uint8_t *all = read_file(in_path, &len);
  if (!all) return 2;
  FILE *out = fopen(out_path, "wb");
  if (!out) {
    free(all);
    return 2;
  }
  size_t pos = 0, cases = 0;
  int ok = 1;
  while (ok && pos < len) {
    const uint8_t *rec[3];
    size_t rec_len[3];
    for (int k = 0; k < 3; k++) {
      if (len - pos < 4u || rd32(all + pos) > len - pos - 4u) {
        ok = 0;
        break;
      }
      rec_len[k] = rd32(all + pos);
      rec[k] = all + pos + 4u;
      pos += 4u + rec_len[k];
    }
    if (!ok) {
      fprintf(stderr, "hydrate-batch: a record runs past the end\n");
      break;
    }
    uint8_t *pack = NULL;
    size_t pack_len = 0;
    bcir_status st =
        hydrate_one(rec[0], rec_len[0], rec[1], rec_len[1], rec[2], rec_len[2], &pack, &pack_len);
    put32(out, (uint32_t)st);
    put32(out, (uint32_t)pack_len);
    if (pack_len) fwrite(pack, 1, pack_len, out);
    free(pack);
    cases++;
  }
  if (fclose(out) != 0) ok = 0;
  free(all);
  printf("BATCH %zu\n", cases);
  return ok ? 0 : 2;
}

/* The hydrate's fail-closed API laws, over the seed record, its own plan and a built binding. */
static int run_api_hydrate(const char *path) {
  size_t len = 0, plan_len = 0, bind_len = 0;
  uint8_t *data = read_file(path, &len), *plan = NULL;
  bcir_kp_input in;
  if (!data || bcir_kp_decode_input(data, len, &in) != BCIR_OK ||
      plan_one(data, len, &plan, &plan_len) != BCIR_OK) {
    free(data);
    return 2;
  }
  uint8_t *bind = make_binding(&in, &bind_len);
  bcir_kp_realization r;
  bcir_kp_binding b, zero_b;
  memset(&zero_b, 0, sizeof zero_b);
  if (!bind || bcir_kp_decode_realization(plan, plan_len, &r) != BCIR_OK) {
    free(data);
    free(plan);
    free(bind);
    return 2;
  }
  CHECK(bcir_kp_decode_binding(bind, bind_len, NULL) == BCIR_ERR_NOSPACE, "binding decode, no out");
  memset(&b, 0x5A, sizeof b);
  CHECK(bcir_kp_decode_binding(bind, bind_len - 1u, &b) != BCIR_OK &&
            all_zero((const uint8_t *)&b, sizeof b),
        "a refused binding zeroes its view");
  CHECK(bcir_kp_decode_binding(bind, bind_len, &b) == BCIR_OK, "binding decode");
  size_t scratch_len = 0, cap = 0, out_len = 7;
  CHECK(bcir_kp_hydrate_scratch_size(&in, &zero_b, &scratch_len) == BCIR_ERR_NOSPACE &&
            scratch_len == 0,
        "hydrate_scratch_size of an undecoded binding");
  CHECK(bcir_kp_hydrate_scratch_size(&in, &b, &scratch_len) == BCIR_OK, "hydrate_scratch_size");
  uint8_t *scratch = (uint8_t *)malloc(scratch_len + 1u);
  if (!scratch) return 2;
  CHECK(bcir_kp_hydrate_size(&in, &r, &b, scratch, scratch_len, &cap) == BCIR_OK && cap,
        "hydrate_size");
  uint8_t *out = (uint8_t *)malloc(cap + 16u), *again = (uint8_t *)malloc(cap);
  if (!out || !again) return 2;
  memset(out, 0xA5, cap + 16u);
  CHECK(bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len, out, cap, NULL) == BCIR_ERR_NOSPACE &&
            all_zero(out, cap),
        "hydrate with no out_len zeroes the output");
  memset(out, 0xA5, cap + 16u);
  if (scratch_len) {
    CHECK(bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len - 1u, out, cap, &out_len) ==
                  BCIR_ERR_NOSPACE &&
              out_len == 0 && all_zero(out, cap),
          "a short scratch zeroes the output");
  }
  memset(out, 0xA5, cap + 16u);
  out_len = 7;
  CHECK(bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len, out, cap - 1u, &out_len) ==
                BCIR_ERR_NOSPACE &&
            out_len == 0 && all_zero(out, cap - 1u),
        "a short output is zeroed");
  CHECK(bcir_kp_hydrate(&in, &r, &b, NULL, scratch_len, out, cap, &out_len) == BCIR_ERR_NOSPACE,
        "hydrate with no scratch");
  CHECK(bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len, out, cap, &out_len) == BCIR_OK &&
            out_len == cap,
        "hydrate");
  memset(scratch, 0xFF, scratch_len + 1u);
  size_t again_len = 0;
  CHECK(bcir_kp_hydrate(&in, &r, &b, scratch + 1, scratch_len, again, cap, &again_len) ==
                BCIR_OK &&
            again_len == cap && memcmp(out, again, cap) == 0,
        "a rehydrate over dirty, misaligned scratch writes the same bytes");
  bcir_streampack_header hdr;
  CHECK(bcir_sp_validate(out, out_len, &hdr) == BCIR_OK &&
            bcir_sp_verify_semantic(out, out_len, hdr.map_gen, hdr.data_gen) == BCIR_OK,
        "the runtime's reader accepts the pack");
  /* a binding for another input: one resource fewer (when there is one to drop) */
  if (in.n_resources) {
    bcir_kp_binding short_b = b;
    short_b.n_resources--;
    size_t bad = 1;
    CHECK(bcir_kp_hydrate_size(&in, &r, &short_b, scratch, scratch_len, &bad) ==
                  BCIR_ERR_PLANNER &&
              bad == 0,
          "a binding of another resource count");
  }
  free(again);
  free(out);
  free(scratch);
  free(bind);
  free(plan);
  free(data);
  if (failures) return 1;
  printf("API OK %d\n", checks);
  return 0;
}

static int run_bench_hydrate(char **argv, long reps, long rounds) {
  size_t lens[3];
  uint8_t *rec[3];
  for (int k = 0; k < 3; k++) rec[k] = read_file(argv[k], &lens[k]);
  bcir_kp_input in;
  bcir_kp_realization r;
  bcir_kp_binding b;
  if (!rec[0] || !rec[1] || !rec[2] || reps < 1 || rounds < 1 || rounds > 64 ||
      bcir_kp_decode_input(rec[0], lens[0], &in) != BCIR_OK ||
      bcir_kp_decode_realization(rec[1], lens[1], &r) != BCIR_OK ||
      bcir_kp_decode_binding(rec[2], lens[2], &b) != BCIR_OK)
    return 2;
  size_t scratch_len = 0, cap = 0, out_len = 0;
  if (bcir_kp_hydrate_scratch_size(&in, &b, &scratch_len) != BCIR_OK) return 2;
  void *scratch = malloc(scratch_len ? scratch_len : 1u);
  if (!scratch || bcir_kp_hydrate_size(&in, &r, &b, scratch, scratch_len, &cap) != BCIR_OK)
    return 2;
  uint8_t *out = (uint8_t *)malloc(cap);
  if (!out) return 2;
  uint64_t per[64];
  for (long k = 0; k < rounds; k++) {
    uint64_t t0 = now_ns();
    for (long i = 0; i < reps; i++)
      if (bcir_kp_hydrate(&in, &r, &b, scratch, scratch_len, out, cap, &out_len) != BCIR_OK)
        return 2;
    per[k] = (now_ns() - t0) / (uint64_t)reps;
  }
  qsort(per, (size_t)rounds, sizeof per[0], cmp_u64);
  printf("BENCH %llu %zu\n", (unsigned long long)per[rounds / 2], out_len);
  free(out);
  free(scratch);
  for (int k = 0; k < 3; k++) free(rec[k]);
  return 0;
}

/* The floor of `hydrate.native.scale4` (tools/perf/gemplus_baseline.py): writing the pack once.
 * The hydrate writes every byte of the StreamPack (records, blocks, notes, the CRC over them), so
 * storing that many bytes into a warm buffer is work no hydrate avoids -- `--bench-floor`'s
 * convention, under the bench's conditions: the same three records (one hydrate fixes the pack's
 * length), the same clock and median. */
static int run_bench_hydrate_floor(char **argv, long reps, long rounds) {
  size_t lens[3];
  uint8_t *rec[3];
  for (int k = 0; k < 3; k++) rec[k] = read_file(argv[k], &lens[k]);
  uint8_t *pack = NULL;
  size_t wrote = 0;
  int ok = rec[0] && rec[1] && rec[2] && reps >= 1 && rounds >= 1 && rounds <= 64 &&
           hydrate_one(rec[0], lens[0], rec[1], lens[1], rec[2], lens[2], &pack, &wrote) ==
               BCIR_OK;
  for (int k = 0; k < 3; k++) free(rec[k]);
  if (!ok) {
    free(pack);
    return 2;
  }
  floor_memset(pack, 0, wrote); /* warm, as the bench's output buffer is after its first pack */
  uint64_t per[64];
  for (long r = 0; r < rounds; r++) {
    uint64_t t0 = now_ns();
    for (long i = 0; i < reps; i++) floor_memset(pack, (int)(i & 0xFF), wrote);
    per[r] = (now_ns() - t0) / (uint64_t)reps;
  }
  qsort(per, (size_t)rounds, sizeof per[0], cmp_u64);
  printf("FLOOR %llu %zu\n", (unsigned long long)per[rounds / 2], wrote);
  free(pack);
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
  if (argc == 5 && strcmp(argv[1], "--bench-floor") == 0)
    return run_bench_floor(argv[2], strtol(argv[3], NULL, 10), strtol(argv[4], NULL, 10));
  if (argc == 6 && strcmp(argv[1], "--hydrate") == 0) {
    size_t lens[3];
    uint8_t *rec[3];
    for (int k = 0; k < 3; k++) rec[k] = read_file(argv[2 + k], &lens[k]);
    if (!rec[0] || !rec[1] || !rec[2]) return 2;
    uint8_t *pack = NULL;
    size_t pack_len = 0;
    bcir_status st =
        hydrate_one(rec[0], lens[0], rec[1], lens[1], rec[2], lens[2], &pack, &pack_len);
    for (int k = 0; k < 3; k++) free(rec[k]);
    if (st != BCIR_OK) {
      printf("%s\n", status_name(st));
      return 0;
    }
    int ok = write_file(argv[5], pack, pack_len);
    free(pack);
    if (!ok) return 2;
    printf("OK %zu\n", pack_len);
    return 0;
  }
  if (argc == 4 && strcmp(argv[1], "--hydrate-batch") == 0)
    return run_hydrate_batch(argv[2], argv[3]);
  if (argc == 3 && strcmp(argv[1], "--api-hydrate") == 0) return run_api_hydrate(argv[2]);
  if (argc == 7 && strcmp(argv[1], "--bench-hydrate") == 0)
    return run_bench_hydrate(argv + 2, strtol(argv[5], NULL, 10), strtol(argv[6], NULL, 10));
  if (argc == 7 && strcmp(argv[1], "--bench-hydrate-floor") == 0)
    return run_bench_hydrate_floor(argv + 2, strtol(argv[5], NULL, 10),
                                   strtol(argv[6], NULL, 10));
  fprintf(stderr,
          "usage: test_kplan --plan IN OUT | --batch IN OUT | --decode-realization IN |"
          " --api IN | --bench IN REPS ROUNDS | --bench-floor IN REPS ROUNDS |"
          " --hydrate IN PLAN BIND OUT | --hydrate-batch IN OUT | --api-hydrate IN |"
          " --bench-hydrate IN PLAN BIND REPS ROUNDS |"
          " --bench-hydrate-floor IN PLAN BIND REPS ROUNDS\n");
  return 2;
}
