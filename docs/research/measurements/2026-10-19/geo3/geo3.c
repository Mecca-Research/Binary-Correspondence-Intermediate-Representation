/* GEO-3 records driver (docs/research/measurements/2026-10-19/geo3): per-layer recompute (REMAT)
 * coupled with the bf16 workspace's tiers (WS-PLAN), on the trainer as this commit builds it --
 * runtime/c's library, its public ABI only (the identity mode also with the build's probes).
 *
 *   geo3 inventory
 *       every shape x contract: the parameters, the arenas, one layer's cache, the workspace's
 *       need / panels / floor (one worker, four, any pool), and the state's bytes under each
 *       recompute set the memory planner walks, at each workspace tier
 *   geo3 identity THREADS
 *       a run from one initial state under every (recompute set, workspace tier) ends in the
 *       all-kept state at the need, bit for bit (weights, both moments, every event); a byte below
 *       the floor is refused with BCIR_DT_CAPACITY before anything is written; with the probes
 *       (-DBCIR_DT_WORKSPACE_PROBE -DBCIR_DT_PANELS_PROBE) the step's GEMM groups packed the need
 *       (more without the panels) and the SwiGLU panels ran exactly where the tier admits them
 *   geo3 frontier SHAPE CONTRACT THREADS ROUNDS STEPS
 *       ABBA rounds of the all-kept state at the need against each candidate (recompute set,
 *       workspace tier): the run's ratio (geometric mean, 95% percentile bootstrap), the bytes the
 *       candidate frees and its cost per MiB freed
 *   geo3 scenario SHAPE CONTRACT THREADS ROUNDS UPDATES BUDGET
 *       B all kept at the need; the plan the memory planner's order picks inside BUDGET bytes
 *       (bcir/hosted/models/native.py memory_plan: the workspace at the need, the panels' need,
 *       the floor, and within each the recompute sets 0, then (1<<r)-1 for r = 2..L, the first
 *       that fits); B/2 in two accumulated micro-batches all kept at its need (the same tokens an
 *       update). Memory, the update's time and its ratio to the first arm, the last loss, digests.
 *
 * Every figure of memory is the state's arenas as the planner counts them: four parameter-sized
 * arenas (weights, gradients, both moments), the activation arena at bcir_decoder_activation_count
 * floats, the scratch arena and the lent workspace. */
#define _POSIX_C_SOURCE 200809L
#include "bcir_decoder_train.h"
#include "bcir_tensor.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef BCIR_DT_WORKSPACE_PROBE
size_t bcir_decoder_workspace_probe(void);
#endif
#ifdef BCIR_DT_PANELS_PROBE
unsigned bcir_decoder_panels_probe(void);
#endif

typedef struct shape { const char *name; bcir_decoder_spec sp; size_t identity_steps; } shape;
/* vocab, width, heads, kvheads, layers, ff, batch, time, tied, rope base, rms eps */
static const shape shapes[] = {
  /* Codex's attention baseline (the PR818 geometric-head report): width 128, three layers, four
   * heads, FF 256, byte vocabulary, four sequences of 256 */
  {"codex-attn", {260, 128, 4, 4, 3, 256, 4, 256, 1, 10000.0, 1e-6}, 3},
  /* its micro-batch of two, which the report's attention32 accumulates twice an update */
  {"codex-attn-b2", {260, 128, 4, 4, 3, 256, 2, 256, 1, 10000.0, 1e-6}, 3},
  /* the 9.84M decoder of the LM comparisons, eight sequences of 128 */
  {"9.84M", {260, 384, 8, 4, 6, 1024, 8, 128, 1, 10000.0, 1e-6}, 2},
};
#define NSHAPES (sizeof shapes / sizeof shapes[0])
typedef struct named { const char *name; uint32_t bits; } named;
#define CORE (BCIR_DT_CONTRACT_EXP | BCIR_DT_CONTRACT_REDUCE)
static const named contracts[] = {
  {"kblock+exp+reduce+attn", BCIR_DT_CONTRACT_KBLOCK | CORE | BCIR_DT_CONTRACT_ATTN},
  {"bf16+exp+reduce+attn", BCIR_DT_CONTRACT_BF16 | CORE | BCIR_DT_CONTRACT_ATTN},
  {"bf16+exp+reduce+attnbf16", BCIR_DT_CONTRACT_BF16 | CORE | BCIR_DT_CONTRACT_ATTNBF16},
};
#define NCONTRACTS (sizeof contracts / sizeof contracts[0])
static const bcir_decoder_adamw opt = {1e-3, 0.9, 0.95, 1e-8, 0.1, 1.0};

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}
static void *al(size_t bytes) {
  size_t r = bytes ? (bytes + 63) / 64 * 64 : 64;
  void *p = aligned_alloc(64, r);
  if (!p) { perror("aligned_alloc"); exit(2); }
  memset(p, 0, r);
  return p;
}
static uint64_t fnv(uint64_t h, const void *p, size_t n) {
  const unsigned char *b = (const unsigned char *)p;
  size_t i;
  for (i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
  return h;
}
static unsigned lcg(unsigned *x) { *x = *x * 1103515245u + 12345u; return *x >> 8; }
static float unit(unsigned *x) { return (float)(lcg(x) & 0xffffff) / 16777216.0f * 2.0f - 1.0f; }
static bcir_tensor_pool *make_pool(unsigned threads) {
  size_t bytes = bcir_tensor_pool_size(threads), r;
  void *m;
  if (threads < 2) return NULL;
  if (!bytes) { fprintf(stderr, "a pool of %u refused\n", threads); exit(2); }
  r = (bytes + 63) / 64 * 64;
  m = aligned_alloc(64, r);
  if (!m || !bcir_tensor_pool_init(m, r, threads)) { fprintf(stderr, "pool init failed\n"); exit(2); }
  return (bcir_tensor_pool *)m;
}
static int find(const char *what, const char *name, int count, const char *(*at)(int)) {
  int i;
  for (i = 0; i < count; i++) if (!strcmp(at(i), name)) return i;
  fprintf(stderr, "unknown %s %s\n", what, name);
  exit(2);
}
static const char *shape_at(int i) { return shapes[i].name; }
static const char *contract_at(int i) { return contracts[i].name; }

typedef struct bench {
  bcir_decoder_spec sp;
  uint32_t contract;
  bcir_decoder_plan plan;
  bcir_decoder_state s;
  float *w0;
  uint32_t *tok, *tgt;
  bcir_decoder_event *ev;
  unsigned char *ws;
  size_t steps, micro, n, need, panels, floor, wsmax;
  unsigned threads;
} bench;

static void bench_init(bench *b, bcir_decoder_spec sp, uint32_t contract, size_t steps,
    bcir_tensor_pool *pool, size_t micro) {
  size_t i, l, d, k, f, v, lp, off, tokens;
  unsigned seed = 20261019u;
  memset(b, 0, sizeof *b);
  b->sp = sp; b->contract = contract; b->steps = steps; b->micro = micro;
  if (bcir_decoder_make_plan(&sp, &b->plan)) { fprintf(stderr, "plan refused\n"); exit(2); }
  b->n = (size_t)sp.batch * sp.time;
  b->threads = bcir_tensor_pool_threads(pool);
  b->s.plan = b->plan; b->s.contract = contract; b->s.pool = pool;
  b->s.weights = al(b->plan.parameters * 4); b->s.gradients = al(b->plan.parameters * 4);
  b->s.moment1 = al(b->plan.parameters * 4); b->s.moment2 = al(b->plan.parameters * 4);
  b->s.activations = al(b->plan.activations * 4); b->s.scratch = al(b->plan.scratch * 4);
  b->s.weight_capacity = b->s.gradient_capacity = b->s.moment1_capacity =
      b->s.moment2_capacity = b->plan.parameters;
  b->s.activation_capacity = b->plan.activations; b->s.scratch_capacity = b->plan.scratch;
  /* the initial weights: unit norm gammas, the projections uniform over +-1/sqrt(fan-in) */
  b->w0 = al(b->plan.parameters * 4);
  d = sp.width; k = (size_t)sp.kvheads * (sp.width / sp.heads); f = sp.ff; v = sp.vocab;
  lp = 2 * d + 2 * d * d + 2 * d * k + 3 * d * f;
  for (i = 0; i < v * d; i++) b->w0[i] = 0.02f * unit(&seed);
  for (l = 0; l < sp.layers; l++) {
    float *x = b->w0 + v * d + l * lp;
    for (i = 0; i < d; i++) x[i] = 1.0f;
    off = d;
    for (i = 0; i < d * d + 2 * k * d + d * d; i++) x[off + i] = unit(&seed) / sqrtf((float)d);
    off += d * d + 2 * k * d + d * d;
    for (i = 0; i < d; i++) x[off + i] = 1.0f;
    off += d;
    for (i = 0; i < 2 * f * d; i++) x[off + i] = unit(&seed) / sqrtf((float)d);
    off += 2 * f * d;
    for (i = 0; i < d * f; i++) x[off + i] = unit(&seed) / sqrtf((float)f);
  }
  for (i = 0; i < d; i++) b->w0[v * d + sp.layers * lp + i] = 1.0f;
  /* the tokens: a skewed byte stream, each update's targets the next tokens */
  tokens = micro * steps * b->n;
  b->tok = al(tokens * 4); b->tgt = al(tokens * 4);
  for (i = 0; i < tokens + 1; i++) {
    uint32_t t = (uint32_t)(lcg(&seed) % 64u + (i % 7u) * 16u) % sp.vocab;
    if (i < tokens) b->tok[i] = t;
    if (i) b->tgt[i - 1] = t;
  }
  b->ev = al(steps * sizeof *b->ev);
  b->need = bcir_decoder_workspace_need(&b->plan, contract, b->threads);
  b->panels = bcir_decoder_workspace_panels(&b->plan, contract);
  b->floor = bcir_decoder_workspace_floor(&b->plan, contract, b->threads);
  b->wsmax = bcir_decoder_workspace_bytes(&b->plan, contract);  /* the need on any pool */
  if (b->need > b->wsmax) b->wsmax = b->need;
  b->ws = al(b->wsmax);
}
static void bench_free(bench *b) {
  free(b->s.weights); free(b->s.gradients); free(b->s.moment1); free(b->s.moment2);
  free(b->s.activations); free(b->s.scratch); free(b->w0); free(b->tok); free(b->tgt);
  free(b->ev); free(b->ws);
}
static size_t count_of(const bench *b, uint64_t r) { return bcir_decoder_activation_count(&b->plan, r); }
/* the state's bytes as the memory planner counts them */
static size_t memory_of(const bench *b, uint64_t r, size_t ws) {
  return 16 * b->plan.parameters + 4 * count_of(b, r) + 4 * b->plan.scratch + ws;
}
static uint64_t digest_of(const bench *b, size_t done) {
  uint64_t h = 1469598103934665603ull;
  size_t i;
  h = fnv(h, b->s.weights, b->plan.parameters * 4);
  h = fnv(h, b->s.moment1, b->plan.parameters * 4);
  h = fnv(h, b->s.moment2, b->plan.parameters * 4);
  for (i = 0; i < done; i++) h = fnv(h, &b->ev[i], sizeof b->ev[i]);
  return h;
}
/* One run of b->steps updates from the initial state under (r, ws); the arena held to the set's
 * count exactly. Returns the trainer's code, 99 for a run that stopped short. */
static int bench_run(bench *b, uint64_t r, size_t ws, uint64_t *digest, double *secs, size_t *done) {
  size_t got = 99;
  int rc;
  double t0;
  memcpy(b->s.weights, b->w0, b->plan.parameters * 4);
  memset(b->s.moment1, 0, b->plan.parameters * 4);
  memset(b->s.moment2, 0, b->plan.parameters * 4);
  b->s.step = 0; b->s.beta1_power = b->s.beta2_power = 1.0;
  b->s.recompute = r;
  b->s.activation_capacity = count_of(b, r);
  b->s.workspace = ws ? b->ws : NULL; b->s.workspace_bytes = ws;
  t0 = now();
  if (b->micro > 1)
    rc = bcir_decoder_run_accum(&b->s, &opt, b->tok, b->tgt, b->micro * b->steps * b->n, NULL,
        b->steps, b->micro, b->ev, b->steps, &got);
  else
    rc = bcir_decoder_run(&b->s, &opt, b->tok, b->tgt, b->steps * b->n, NULL, b->steps, b->ev,
        b->steps, &got);
  if (secs) *secs = now() - t0;
  if (done) *done = got;
  if (digest) *digest = digest_of(b, rc ? 0 : got);
  return rc ? rc : (got == b->steps ? 0 : 99);
}

/* The recompute sets: the planner's walk (0, then the lowest r layers for r = 2..L) and the sets
 * no walk reaches -- one layer alone, the first, every other, the highest two. */
static size_t sets_of(unsigned layers, uint64_t *set, const char **name, char names[][24]) {
  size_t n = 0;
  unsigned r;
  set[n] = 0; name[n++] = "kept";
  for (r = 2; r <= layers && r <= 64; r++) {
    set[n] = r == 64 ? ~0ull : (1ull << r) - 1;
    snprintf(names[n], 24, "lowest %u", r); name[n] = names[n]; n++;
  }
  set[n] = 1ull << (layers - 1); name[n++] = "the top alone";
  set[n] = 1; name[n++] = "the first alone";
  set[n] = 0; for (r = 0; r < layers; r += 2) set[n] |= 1ull << r;
  name[n++] = "every other";
  if (layers >= 3) { set[n] = 3ull << (layers - 2); name[n++] = "the top two"; }
  return n;
}

static int inventory(void) {
  size_t si, ci;
  unsigned t;
  printf("GEO-3 inventory: bytes; memory = 16 x parameters + 4 x (activation count + scratch) + workspace\n");
  for (si = 0; si < NSHAPES; si++) for (ci = 0; ci < NCONTRACTS; ci++) {
    bench b;
    uint64_t set[80];
    const char *name[80];
    char names[80][24];
    size_t nsets, i, la, tiers[3], nt = 1, j;
    const unsigned threads[] = {1, 4};
    bench_init(&b, shapes[si].sp, contracts[ci].bits, 1, NULL, 1);
    la = count_of(&b, 0) - count_of(&b, 3);  /* one slot: two recomputed layers share one */
    printf("\n%s (vocab %u, width %u, heads %u/%u, layers %u, ff %u, batch %u x %u), %s\n",
        shapes[si].name, b.sp.vocab, b.sp.width, b.sp.heads, b.sp.kvheads, b.sp.layers, b.sp.ff,
        b.sp.batch, b.sp.time, contracts[ci].name);
    printf("  parameters %zu (x16 = %zu), activations %zu floats (%zu), scratch %zu floats (%zu), "
        "a layer's cache %zu floats (%zu)\n", b.plan.parameters, 16 * b.plan.parameters,
        b.plan.activations, 4 * b.plan.activations, b.plan.scratch, 4 * b.plan.scratch, la, 4 * la);
    for (j = 0; j < 2; j++) {
      unsigned th = threads[j];
      printf("  workspace at %u worker%s: need %zu, panels %zu, floor %zu\n", th, th > 1 ? "s" : "",
          bcir_decoder_workspace_need(&b.plan, b.contract, th),
          bcir_decoder_workspace_panels(&b.plan, b.contract),
          bcir_decoder_workspace_floor(&b.plan, b.contract, th));
    }
    printf("  workspace on any pool (bcir_decoder_workspace_bytes): %zu\n",
        bcir_decoder_workspace_bytes(&b.plan, b.contract));
    /* the tiers at four workers, as the planner builds them */
    t = 4;
    tiers[0] = bcir_decoder_workspace_need(&b.plan, b.contract, t);
    {
      size_t panels = bcir_decoder_workspace_panels(&b.plan, b.contract);
      size_t floor = bcir_decoder_workspace_floor(&b.plan, b.contract, t);
      if (floor <= panels && panels < tiers[0]) tiers[nt++] = panels;
      if (floor < tiers[nt - 1]) tiers[nt++] = floor;
    }
    nsets = sets_of(b.sp.layers, set, name, names);
    printf("  %-16s %6s %12s", "recompute set", "re-run", "activations");
    for (j = 0; j < nt; j++) {  /* right-aligned over its column, as the counts are */
      char head[32];
      snprintf(head, sizeof head, "memory @ws %zu", tiers[j]);
      printf("  %20s", head);
    }
    printf("\n");
    for (i = 0; i < nsets; i++) {
      unsigned named_count = 0, l;
      for (l = 0; l < b.sp.layers && l < 64; l++) named_count += (unsigned)((set[i] >> l) & 1);
      printf("  %-16s %6u %12zu", name[i], named_count ? named_count - 1 : 0, 4 * count_of(&b, set[i]));
      for (j = 0; j < nt; j++) printf("  %20zu", memory_of(&b, set[i], tiers[j]));
      printf("\n");
    }
    bench_free(&b);
  }
  return 0;
}

static int identity(int argc, char **argv) {
  unsigned threads;
  size_t si, ci;
  int fails = 0, runs = 0;
  bcir_tensor_pool *pool;
  if (argc < 3) { fprintf(stderr, "identity THREADS\n"); return 2; }
  threads = (unsigned)atoi(argv[2]);
  pool = make_pool(threads);
  printf("GEO-3 identity at %u thread%s%s\n", threads, threads > 1 ? "s" : "",
#if defined(BCIR_DT_WORKSPACE_PROBE) && defined(BCIR_DT_PANELS_PROBE)
      ", with the workspace and panels probes"
#else
      ""
#endif
      );
  for (si = 0; si < NSHAPES; si++) for (ci = 0; ci < NCONTRACTS; ci++) {
    bench b;
    uint64_t base, h, set[80];
    const char *name[80];
    char names[80][24];
    size_t nsets, i, k, ws[6], nws = 0, done;
    const char *wsn[6];
    int rc, bf16;
    bench_init(&b, shapes[si].sp, contracts[ci].bits, shapes[si].identity_steps, pool, 1);
    bf16 = (b.contract & BCIR_DT_CONTRACT_BF16) != 0;
    rc = bench_run(&b, 0, b.need, &base, NULL, &done);
    printf("%s %s, %zu steps: need %zu, panels %zu, floor %zu; all kept at the need rc %d, loss %.6f, state %016llx\n",
        shapes[si].name, contracts[ci].name, b.steps, b.need, b.panels, b.floor, rc,
        rc ? 0.0 : b.ev[b.steps - 1].loss, (unsigned long long)base);
    if (rc) { fails++; bench_free(&b); continue; }
    if (bf16) {
#define TIER(v, nm) do { size_t v_ = (v); for (k = 0; k < nws && ws[k] != v_; k++) {} \
      if (k == nws && v_ >= b.floor && v_ <= b.need) { ws[nws] = v_; wsn[nws] = (nm); nws++; } } while (0)
      TIER(b.need, "need"); TIER(b.need - 1, "need-1");
      if (b.panels) { TIER(b.panels, "panels"); TIER(b.panels - 1, "panels-1"); }
      TIER(b.floor, "floor");
#undef TIER
    } else { ws[0] = 0; wsn[0] = "none"; nws = 1; }
    nsets = sets_of(b.sp.layers, set, name, names);
    for (i = 0; i < nsets; i++) for (k = 0; k < nws; k++) {
      unsigned pan = 0;
      size_t seen = 0;
      int ok;
#ifdef BCIR_DT_PANELS_PROBE
      (void)bcir_decoder_panels_probe();
#endif
#ifdef BCIR_DT_WORKSPACE_PROBE
      (void)bcir_decoder_workspace_probe();
#endif
      rc = bench_run(&b, set[i], ws[k], &h, NULL, &done);
#ifdef BCIR_DT_PANELS_PROBE
      pan = bcir_decoder_panels_probe();
#endif
#ifdef BCIR_DT_WORKSPACE_PROBE
      seen = bcir_decoder_workspace_probe();
#endif
      ok = rc == 0 && h == base;
#if defined(BCIR_DT_WORKSPACE_PROBE) && defined(BCIR_DT_PANELS_PROBE)
      if (bf16) {
        /* the GEMM groups' need, the bf16 attention's regions aside; without the panels, more */
        size_t gemms = bcir_decoder_workspace_need(&b.plan,
            b.contract & ~(uint32_t)BCIR_DT_CONTRACT_ATTNBF16, b.threads);
        int with = b.panels && ws[k] >= b.panels;
        ok = ok && pan == (with ? 7u : 0u) && (with || !b.panels ? seen == gemms : seen > gemms);
      }
#endif
      runs++;
      fails += !ok;
      printf("  %-16s ws %-8s %9zu: rc %d memory %10zu, panels %u, probe %9zu, %s\n", name[i], wsn[k],
          ws[k], rc, memory_of(&b, set[i], ws[k]), pan, seen, ok ? "the same state" : "DIFFERENT");
    }
    if (bf16) {  /* a byte below the floor: refused before anything is written, the count included */
      uint64_t start = digest_of(&b, 0);
      done = 99;
      b.s.workspace = b.ws; b.s.workspace_bytes = b.floor - 1; b.s.recompute = 0;
      b.s.activation_capacity = count_of(&b, 0);
      rc = bcir_decoder_run(&b.s, &opt, b.tok, b.tgt, b.steps * b.n, NULL, b.steps, b.ev, b.steps, &done);
      runs++;
      if (rc != BCIR_DT_CAPACITY || done != 99 || digest_of(&b, 0) != start) fails++;
      printf("  %-16s ws %-8s %9zu: rc %d (CAPACITY), nothing written: %s\n", "kept", "floor-1",
          b.floor - 1, rc, rc == BCIR_DT_CAPACITY && done == 99 && digest_of(&b, 0) == start ? "yes" : "NO");
    }
    {  /* a recompute bit past the last layer: refused, nothing written */
      uint64_t start;
      memcpy(b.s.weights, b.w0, b.plan.parameters * 4);
      start = digest_of(&b, 0);
      done = 99;
      b.s.workspace = bf16 ? b.ws : NULL; b.s.workspace_bytes = bf16 ? b.need : 0;
      b.s.recompute = b.sp.layers < 64 ? 1ull << b.sp.layers : 0;
      b.s.activation_capacity = b.plan.activations;
      rc = bcir_decoder_run(&b.s, &opt, b.tok, b.tgt, b.steps * b.n, NULL, b.steps, b.ev, b.steps, &done);
      runs++;
      if (rc != BCIR_DT_INVALID || done != 99 || digest_of(&b, 0) != start) fails++;
      printf("  %-16s: rc %d (INVALID), nothing written: %s\n", "bit L set", rc,
          rc == BCIR_DT_INVALID && done == 99 && digest_of(&b, 0) == start ? "yes" : "NO");
    }
    bench_free(&b);
  }
  if (pool) bcir_tensor_pool_stop(pool);
  free(pool);
  printf("identity at %u thread%s: %s (%d runs, %d failures)\n", threads, threads > 1 ? "s" : "",
      fails ? "FAIL" : "PASS", runs, fails);
  return fails != 0;
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
/* the geometric mean of the ratios and a 95% percentile bootstrap (2000 resamples, fixed seed) */
static void ratio_ci(const double *r, size_t n, double *mid, double *lo, double *hi) {
  double *m = malloc(2000 * sizeof *m), s = 0;
  unsigned seed = 7u;
  size_t i, j;
  if (!m) exit(2);
  for (i = 0; i < n; i++) s += log(r[i]);
  *mid = exp(s / (double)n);
  for (j = 0; j < 2000; j++) {
    double t = 0;
    for (i = 0; i < n; i++) t += log(r[lcg(&seed) % n]);
    m[j] = exp(t / (double)n);
  }
  qsort(m, 2000, sizeof *m, cmp_double);
  *lo = m[49]; *hi = m[1949];
  free(m);
}

static int frontier(int argc, char **argv) {
  int si, ci, rounds, r, fails = 0;
  unsigned threads;
  size_t steps, c, ncand = 0, L;
  bench b;
  bcir_tensor_pool *pool;
  uint64_t hb, hc, every;
  struct cand { uint64_t set; size_t ws; char name[48]; } cands[12];
  double *ratios;
  if (argc < 7) { fprintf(stderr, "frontier SHAPE CONTRACT THREADS ROUNDS STEPS\n"); return 2; }
  si = find("shape", argv[2], (int)NSHAPES, shape_at);
  ci = find("contract", argv[3], (int)NCONTRACTS, contract_at);
  threads = (unsigned)atoi(argv[4]); rounds = atoi(argv[5]); steps = (size_t)atoi(argv[6]);
  if (rounds < 2 || steps < 1) { fprintf(stderr, "rounds >= 2, steps >= 1\n"); return 2; }
  pool = make_pool(threads);
  bench_init(&b, shapes[si].sp, contracts[ci].bits, steps, pool, 1);
  L = b.sp.layers;
  every = L >= 64 ? ~0ull : (1ull << L) - 1;
#define CAND(s_, w_, ...) do { cands[ncand].set = (s_); cands[ncand].ws = (w_); \
    snprintf(cands[ncand].name, sizeof cands[ncand].name, __VA_ARGS__); ncand++; } while (0)
  CAND(3, b.need, "lowest 2 (1 re-run), need");
  if (L > 3) CAND((1ull << (L / 2 + 1)) - 1, b.need, "lowest %zu (%zu re-run), need", L / 2 + 1, L / 2);
  CAND(every, b.need, "all %zu (%zu re-run), need", L, L - 1);
  if (b.contract & BCIR_DT_CONTRACT_BF16) {
    if (b.panels && b.panels < b.need && b.panels >= b.floor) CAND(0, b.panels, "kept, panels");
    if (b.floor < b.need) CAND(0, b.floor, "kept, floor");
    if (b.floor < b.need) CAND(every, b.floor, "all %zu (%zu re-run), floor", L, L - 1);
  }
#undef CAND
  printf("GEO-3 frontier: %s %s, %u thread%s, %d ABBA rounds of %zu steps; need %zu, panels %zu, floor %zu\n",
      shapes[si].name, contracts[ci].name, threads, threads > 1 ? "s" : "", rounds, steps, b.need,
      b.panels, b.floor);
  printf("  the reference: every layer kept, the workspace at the need, memory %zu\n", memory_of(&b, 0, b.need));
  ratios = malloc((size_t)rounds * sizeof *ratios);
  if (!ratios) return 2;
  if (bench_run(&b, 0, b.need, &hb, NULL, NULL)) fails++;  /* warm, and the reference state */
  for (c = 0; c < ncand; c++) {
    double mid, lo, hi, tb_sum = 0, tc_sum = 0, freed;
    for (r = 0; r < rounds; r++) {
      double tb, tc, tb2, tc2;
      uint64_t h1, h2, h3, h4;
      fails += bench_run(&b, 0, b.need, &h1, &tb, NULL) != 0;
      fails += bench_run(&b, cands[c].set, cands[c].ws, &h2, &tc, NULL) != 0;
      fails += bench_run(&b, cands[c].set, cands[c].ws, &h3, &tc2, NULL) != 0;
      fails += bench_run(&b, 0, b.need, &h4, &tb2, NULL) != 0;
      fails += h1 != hb || h2 != hb || h3 != hb || h4 != hb;
      ratios[r] = (tc + tc2) / (tb + tb2);
      tb_sum += tb + tb2; tc_sum += tc + tc2;
    }
    (void)hc;
    ratio_ci(ratios, (size_t)rounds, &mid, &lo, &hi);
    freed = (double)(memory_of(&b, 0, b.need) - memory_of(&b, cands[c].set, cands[c].ws)) / 1048576.0;
    printf("  %-30s memory %10zu (frees %7.2f MiB): step %7.2f ms vs %7.2f ms, x%.4f (%.4f-%.4f), "
        "%+.3f%% a MiB freed%s\n", cands[c].name, memory_of(&b, cands[c].set, cands[c].ws), freed,
        1e3 * tc_sum / (2.0 * rounds * (double)steps), 1e3 * tb_sum / (2.0 * rounds * (double)steps),
        mid, lo, hi, freed > 0 ? 100.0 * (mid - 1.0) / freed : 0.0, fails ? " FAIL" : "");
    fflush(stdout);
  }
  printf("every run the reference's state: %s\n", fails ? "NO" : "yes");
  free(ratios);
  bench_free(&b);
  if (pool) bcir_tensor_pool_stop(pool);
  free(pool);
  return fails != 0;
}

/* the memory planner's order (bcir/hosted/models/native.py memory_plan), the first fit */
static int plan_within(const bench *b, size_t budget, uint64_t *set, size_t *ws, size_t *least) {
  /* memory_plan's order: each set at the need and then at the panels' need, the floor after them */
  size_t tiers[2], nt = 1, t, pass;
  unsigned r;
  tiers[0] = b->need;
  if (b->floor <= b->panels && b->panels < b->need) tiers[nt++] = b->panels;
  *least = SIZE_MAX;
  for (pass = 0; pass < 2; pass++) {
    if (pass == 1 && !(b->floor < tiers[nt - 1])) break;
    for (r = 0; r <= b->sp.layers && r <= 64; r++) {
      uint64_t s;
      if (r == 1) continue;  /* one recomputed layer is free: the sets are 0, then 2..L */
      s = r == 0 ? 0 : r == 64 ? ~0ull : (1ull << r) - 1;
      for (t = 0; t < (pass ? 1 : nt); t++) {
        size_t w = pass ? b->floor : tiers[t], m = memory_of(b, s, w);
        if (m <= budget) { *set = s; *ws = w; return 1; }
        *least = m;
      }
    }
  }
  return 0;
}

static int scenario(int argc, char **argv) {
  int si, ci, rounds, r, k, fails = 0, fits;
  unsigned threads;
  size_t updates, budget, least, pws = 0;
  uint64_t pset = 0;
  bench full, half;
  bcir_tensor_pool *pool;
  bcir_decoder_spec sp2;
  struct arm { bench *b; uint64_t set; size_t ws; char name[64]; } arms[3];
  double times[3][64], sum[3] = {0, 0, 0}, loss[3];
  uint64_t h[3];
  if (argc < 8) { fprintf(stderr, "scenario SHAPE CONTRACT THREADS ROUNDS UPDATES BUDGET\n"); return 2; }
  si = find("shape", argv[2], (int)NSHAPES, shape_at);
  ci = find("contract", argv[3], (int)NCONTRACTS, contract_at);
  threads = (unsigned)atoi(argv[4]); rounds = atoi(argv[5]); updates = (size_t)atoi(argv[6]);
  budget = (size_t)strtoull(argv[7], NULL, 10);
  if (rounds < 2 || rounds > 64 || updates < 1) { fprintf(stderr, "2 <= rounds <= 64, updates >= 1\n"); return 2; }
  pool = make_pool(threads);
  bench_init(&full, shapes[si].sp, contracts[ci].bits, updates, pool, 1);
  sp2 = shapes[si].sp; sp2.batch /= 2;
  bench_init(&half, sp2, contracts[ci].bits, updates, pool, 2);
  fits = plan_within(&full, budget, &pset, &pws, &least);
  arms[0].b = &full; arms[0].set = 0; arms[0].ws = full.need;
  snprintf(arms[0].name, sizeof arms[0].name, "B=%u, every layer kept", full.sp.batch);
  arms[1].b = &full; arms[1].set = pset; arms[1].ws = pws;
  snprintf(arms[1].name, sizeof arms[1].name, "B=%u, planned (recompute %#llx, ws %zu)", full.sp.batch,
      (unsigned long long)pset, pws);
  arms[2].b = &half; arms[2].set = 0; arms[2].ws = half.need;
  snprintf(arms[2].name, sizeof arms[2].name, "B=%u x2 accumulated, every layer kept", half.sp.batch);
  printf("GEO-3 scenario: %s %s, %u thread%s, %d rounds of %zu update%s, budget %zu bytes (%.2f MiB)\n",
      shapes[si].name, contracts[ci].name, threads, threads > 1 ? "s" : "", rounds, updates,
      updates > 1 ? "s" : "", budget, (double)budget / 1048576.0);
  if (!fits) { printf("  no plan fits the budget: the least is %zu\n", least); fails++; }
  for (k = 0; k < 3; k++) {
    if (k == 1 && !fits) continue;
    if (bench_run(arms[k].b, arms[k].set, arms[k].ws, &h[k], NULL, NULL)) fails++;
    loss[k] = arms[k].b->ev[updates - 1].loss;
  }
  for (r = 0; r < rounds; r++) {
    int order[3] = {0, 1, 2};
    if (r % 2) { order[0] = 2; order[1] = 1; order[2] = 0; }
    for (k = 0; k < 3; k++) {
      int a = order[k];
      uint64_t hh;
      if (a == 1 && !fits) { times[a][r] = 0; continue; }
      if (bench_run(arms[a].b, arms[a].set, arms[a].ws, &hh, &times[a][r], NULL)) fails++;
      if (hh != h[a]) fails++;
      sum[a] += times[a][r];
    }
  }
  for (k = 0; k < 3; k++) {
    double ratios[64], mid, lo, hi;
    size_t mem = memory_of(arms[k].b, arms[k].set, arms[k].ws);
    if (k == 1 && !fits) continue;
    for (r = 0; r < rounds; r++) ratios[r] = times[k][r] / times[0][r];
    ratio_ci(ratios, (size_t)rounds, &mid, &lo, &hi);
    printf("  %-48s memory %10zu (%6.2f MiB, %s): update %7.2f ms, x%.4f (%.4f-%.4f); last loss %.6f, state %016llx%s\n",
        arms[k].name, mem, (double)mem / 1048576.0, mem <= budget ? "fits" : "over",
        1e3 * sum[k] / ((double)rounds * (double)updates), mid, lo, hi, loss[k], (unsigned long long)h[k],
        k == 1 ? (h[1] == h[0] ? " = the first arm's, bit for bit" : " DIFFERS from the first arm's") :
        k == 2 ? (h[2] == h[0] ? " = the first arm's" : " (another state: the sum's order)") : "");
  }
  if (fits && h[1] != h[0]) fails++;
  printf("scenario: %s\n", fails ? "FAIL" : "PASS");
  bench_free(&full); bench_free(&half);
  if (pool) bcir_tensor_pool_stop(pool);
  free(pool);
  return fails != 0;
}

int main(int argc, char **argv) {
  if (bcir_decoder_abi_version() != 5) { fprintf(stderr, "ABI %u, not 5\n", bcir_decoder_abi_version()); return 2; }
  if (argc > 1 && !strcmp(argv[1], "inventory")) return inventory();
  if (argc > 1 && !strcmp(argv[1], "identity")) return identity(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "frontier")) return frontier(argc, argv);
  if (argc > 1 && !strcmp(argv[1], "scenario")) return scenario(argc, argv);
  fprintf(stderr, "usage: geo3 inventory | identity THREADS | frontier SHAPE CONTRACT THREADS ROUNDS STEPS"
      " | scenario SHAPE CONTRACT THREADS ROUNDS UPDATES BUDGET\n");
  return 2;
}
