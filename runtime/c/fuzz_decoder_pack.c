/*===- fuzz_decoder_pack.c - libFuzzer harness for the native training-pack lowering ===
 *
 * bcir_decoder_lower_pack (PX-2b, bcir_decoder_pack.h) takes a StreamPack and an ExecutionPlanV1
 * from the caller and lowers each claim GEM's executor dispatches to a launch record: every byte of
 * both is attacker-controlled. The input is `[configuration byte][shape byte][pack length, u32
 * little-endian][pack][plan]`: the configuration names the state's contract (one of eight) and
 * whether its recompute set is every layer, the shape the run's steps and accumulation (one or two
 * each). Seeded from real packs (tools/c/fuzz_streampack.sh: the Python planner's, for the
 * configurations and shapes the seeds name, and near misses of the lowering's refusals), so a
 * campaign starts inside the two wire formats and the lowering's rules rather than rediscovering
 * them. Both artifacts end in a CRC, at which a random mutation dies long before the laws and the
 * lowering behind it, so each input is driven RAW (the reject-on-bad-CRC path stays covered) and
 * RESEALED -- both CRC trailers recomputed on copies -- which is what reaches them (the decoder
 * program harness's discipline, fuzz_decoder_gem.c).
 *
 * The contract under test, for ANY input:
 *   - the lowering reads and writes nothing out of bounds (ASan/UBSan), whatever the bytes say;
 *   - a refused pack writes nothing: the program buffer is as it was, no record counted;
 *   - the lowering is a function of its input: the same bytes lower to the same records;
 *   - what a pack lowers to runs through bcir_decoder_run_program to a status the API names, at
 *     most `steps` steps completed (the runner's own contract, held here on what packs lower to).
 * The state is the training harness's (fuzz_decoder_train.c: a two-layer decoder of width 8 over a
 * vocabulary of 16, four tokens a micro-batch, one thread), reset before each run. A pack whose
 * lowering needs more records or work than the harness lends is skipped before anything is lent
 * (the bound where the resource commits).
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 -I . fuzz_decoder_pack.c \
 *       bcir_decoder_pack.c bcir_decoder_train.c bcir_tensor.c bcir_runtime.c bcir_exec.c \
 *       -lm -o fuzz_decoder_pack
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_decoder_pack.h"
#include "bcir_runtime.h"

#define FUZZ_MAX_RECORDS 512u     /* the largest seeded pack, two steps of two, lowers to 385 */
#define FUZZ_WORK_WORDS (1u << 17) /* 1 MB of lowering work: every seed's needs a fraction */
#define FUZZ_WORKSPACE (1u << 20)  /* above every configuration's floor for these shapes */
#define FUZZ_MAX_STEPS 2u
#define FUZZ_MAX_BATCHES 4u        /* steps x accumulation */
#define FUZZ_SENTINEL 0xA5A5A5A5A5A5A5A5ull
#define FUZZ_MAX_INPUT (1u << 18)  /* the campaign's -max_len: the resealed copies' room */

/* the contracts a configuration names, as the training harness names them */
static const uint32_t g_contracts[8] = {0u, 3u, 4u, 14u, 16u, 82u, 32u, 36u};

typedef struct fuzz_arena { float *p; size_t n; } fuzz_arena;
static fuzz_arena g_arena[6];
static unsigned char *g_workspace;
static bcir_decoder_plan g_plan;
static int g_ready;
static uint64_t g_program[FUZZ_MAX_RECORDS * BCIR_DT_RECORD_WORDS];
static uint64_t g_again[FUZZ_MAX_RECORDS * BCIR_DT_RECORD_WORDS];
static uint64_t g_work[FUZZ_WORK_WORDS];
static uint32_t g_tokens[FUZZ_MAX_BATCHES * 4u], g_targets[FUZZ_MAX_BATCHES * 4u];
static uint8_t g_sealed[FUZZ_MAX_INPUT];

static void setup(void) {
  bcir_decoder_spec spec;
  size_t i;
  memset(&spec, 0, sizeof spec);
  spec.vocab = 16; spec.width = 8; spec.heads = 2; spec.kvheads = 1; spec.layers = 2;
  spec.ff = 16; spec.batch = 1; spec.time = 4; spec.tied = 1;
  spec.rope_base = 10000.0; spec.rms_eps = 1e-6;
  if (bcir_decoder_make_plan(&spec, &g_plan) != BCIR_DT_OK) abort();
  g_arena[0].n = g_arena[1].n = g_arena[2].n = g_arena[3].n = g_plan.parameters;
  g_arena[4].n = g_plan.activations;
  g_arena[5].n = g_plan.scratch;
  for (i = 0; i < 6; i++) {
    g_arena[i].p = (float *)malloc(g_arena[i].n * sizeof(float));
    if (!g_arena[i].p) abort();
  }
  g_workspace = (unsigned char *)malloc(FUZZ_WORKSPACE);
  if (!g_workspace) abort();
  for (i = 0; i < FUZZ_MAX_BATCHES * 4u; i++) {
    g_tokens[i] = (uint32_t)((7u * i + 3u) % 16u);
    g_targets[i] = (uint32_t)((5u * i + 1u) % 16u);
  }
  g_ready = 1;
}

/* the state every run starts from: seeded weights, everything else zero, step 0 */
static void reset(bcir_decoder_state *s, uint32_t contract, uint64_t recompute) {
  size_t i, j;
  for (i = 0; i < 6; i++) memset(g_arena[i].p, 0, g_arena[i].n * sizeof(float));
  for (j = 0; j < g_arena[0].n; j++)
    g_arena[0].p[j] = (float)((int)((j * 2654435761u) % 2001u) - 1000) * 2e-5f;
  memset(g_workspace, 0, FUZZ_WORKSPACE);
  memset(s, 0, sizeof *s);
  s->plan = g_plan;
  s->weights = g_arena[0].p; s->gradients = g_arena[1].p;
  s->moment1 = g_arena[2].p; s->moment2 = g_arena[3].p;
  s->activations = g_arena[4].p; s->scratch = g_arena[5].p;
  s->weight_capacity = s->gradient_capacity = s->moment1_capacity = s->moment2_capacity =
      g_plan.parameters;
  s->activation_capacity = g_plan.activations;
  s->scratch_capacity = g_plan.scratch;
  s->beta1_power = s->beta2_power = 1.0;
  s->contract = contract;
  s->workspace = (contract & BCIR_DT_CONTRACT_BF16) ? g_workspace : NULL;
  s->workspace_bytes = (contract & BCIR_DT_CONTRACT_BF16) ? FUZZ_WORKSPACE : 0u;
  s->recompute = recompute;
}

static int untouched(const uint64_t *words, size_t n) {
  size_t i;
  for (i = 0; i < n; i++)
    if (words[i] != FUZZ_SENTINEL) return 0;
  return 1;
}

/* one pack and plan, under the configuration's state: the four properties of the header comment */
static void drive(uint32_t contract, uint64_t recompute, size_t steps, size_t accumulation,
                  const uint8_t *pack, size_t pack_len, const uint8_t *plan, size_t plan_len) {
  bcir_decoder_state state;
  bcir_decoder_adamw opt = {1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0};
  bcir_decoder_event events[FUZZ_MAX_STEPS];
  size_t records = 0, again = 0, need_records = 0, need_work = 0, done = 77u;
  const size_t words = FUZZ_MAX_RECORDS * BCIR_DT_RECORD_WORDS;
  int rc, why = 0, why2 = 0;
  /* the lowering's own figures, before anything is lent: a pack past the harness's buffers is
   * skipped, never lowered into a short one */
  if (bcir_decoder_pack_sizes(pack, pack_len, plan, plan_len, &need_records, &need_work, &why) ==
          BCIR_DT_OK &&
      (need_records > FUZZ_MAX_RECORDS || need_work > sizeof g_work))
    return;
  reset(&state, contract, recompute);
  memset(g_program, 0xA5, sizeof g_program);
  rc = bcir_decoder_lower_pack(&state, pack, pack_len, plan, plan_len, g_work, sizeof g_work,
                               g_program, FUZZ_MAX_RECORDS, &records, &why);
  if (rc != BCIR_DT_OK) {
    /* refused: nothing written, no record counted, a status and a reason the header names */
    if (rc != BCIR_DT_INVALID && rc != BCIR_DT_CAPACITY) abort();
    if (records != 0u || !untouched(g_program, words)) abort();
    if (rc == BCIR_DT_INVALID && why == 0) abort();
    return;
  }
  if (records != need_records || records > FUZZ_MAX_RECORDS) abort();
  /* a function of its input: the same bytes, the same records */
  memset(g_again, 0xA5, sizeof g_again);
  rc = bcir_decoder_lower_pack(&state, pack, pack_len, plan, plan_len, g_work, sizeof g_work,
                               g_again, FUZZ_MAX_RECORDS, &again, &why2);
  if (rc != BCIR_DT_OK || again != records ||
      memcmp(g_again, g_program, records * BCIR_DT_RECORD_WORDS * sizeof(uint64_t)) != 0)
    abort();
  /* what it lowered to runs under the runner's contract */
  rc = bcir_decoder_run_program(&state, &opt, g_program, records, g_tokens, g_targets,
                                steps * accumulation * 4u, NULL, steps, accumulation, events,
                                steps, &done);
  if (rc < BCIR_DT_OK || rc > BCIR_DT_NUMERIC) abort();
  if (rc == BCIR_DT_INVALID ? done != 77u : done > steps) abort();
}

static void seal(uint8_t *bytes, size_t len) {
  uint32_t crc;
  if (len < 4u) return;
  crc = bcir_crc32(bytes, len - 4u);
  bytes[len - 4u] = (uint8_t)crc;
  bytes[len - 3u] = (uint8_t)(crc >> 8);
  bytes[len - 2u] = (uint8_t)(crc >> 16);
  bytes[len - 1u] = (uint8_t)(crc >> 24);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  uint32_t contract, pack_len;
  uint64_t recompute;
  size_t steps, accumulation, plan_len;
  if (!g_ready) setup();
  if (size < 6) return 0;
  contract = g_contracts[data[0] & 7u];
  recompute = (data[0] & 8u) ? 3u : 0u;
  steps = 1u + (data[1] & 1u);
  accumulation = 1u + ((data[1] >> 1) & 1u);
  pack_len = (uint32_t)data[2] | (uint32_t)data[3] << 8 | (uint32_t)data[4] << 16 |
             (uint32_t)data[5] << 24;
  if (pack_len > size - 6u) return 0;
  plan_len = size - 6u - pack_len;
  drive(contract, recompute, steps, accumulation, data + 6, pack_len, data + 6 + pack_len,
        plan_len); /* raw */
  if (size - 6u <= sizeof g_sealed) { /* resealed: both CRCs recomputed on a copy */
    memcpy(g_sealed, data + 6, size - 6u);
    seal(g_sealed, pack_len);
    seal(g_sealed + pack_len, plan_len);
    drive(contract, recompute, steps, accumulation, g_sealed, pack_len, g_sealed + pack_len,
          plan_len);
  }
  return 0;
}
