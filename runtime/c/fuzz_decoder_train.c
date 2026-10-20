/*===- fuzz_decoder_train.c - libFuzzer harness for the native training-program runner ===
 *
 * bcir_decoder_run_program (PX-2, bcir_decoder_train.h) runs a launch program on a training
 * state: every record's kind, flags, dimensions and operand places come from the caller, so all of
 * them are attacker-controlled. The input is `[configuration byte][shape byte][program words,
 * little-endian]`: the configuration names the state's contract (one of eight) and whether its
 * recompute set is every layer, and (PX-4) whether the two transient arenas are lent as the trainer
 * lends them (the recompute set's activations and the plan's scratch) or at the bytes the program
 * itself reaches in each -- a placement's arena, which holds the scratch values too and so may
 * reach past the plan's activations, up to the plan's whole transient span -- and then whether
 * one float short; the shape the run's steps and accumulation (one or two each); the rest
 * is the program, whole records of BCIR_DT_RECORD_WORDS words (a partial record's bytes are
 * ignored). Every transient arena is lent as the tail of its allocation, so a byte past what is
 * lent lies past the allocation, where ASan sees it. Seeded from real programs
 * (tools/c/fuzz_streampack.sh: the Python planner's, for the configurations and shapes the seeds
 * name, placed by lifetime too, and near misses of the check's refusals), so a campaign starts
 * inside the format and its refusals rather than rediscovering the record grammar.
 *
 * The contract under test, for ANY input:
 *   - the run reads and writes nothing out of bounds (ASan/UBSan), whatever the program says;
 *   - a refused program (BCIR_DT_INVALID) wrote nothing: the arenas, the workspace, the step, the
 *     bias powers, the events and the completed count lent are byte for byte as they were;
 *   - a run is a function of its input: the same program from the same state runs to the same
 *     status, completed count, events and state, bit for bit, every time it runs.
 * The state is small (a two-layer decoder of width 8 over a vocabulary of 16, four tokens a
 * micro-batch, one thread) and reset before each run, so an iteration stays cheap.
 *
 *   clang -fsanitize=fuzzer,address,undefined -std=c23 -I . fuzz_decoder_train.c \
 *       bcir_decoder_train.c bcir_tensor.c -lm -o fuzz_decoder_train
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_decoder_train.h"

#define FUZZ_RECORD_BYTES (BCIR_DT_RECORD_WORDS * 8u)
#define FUZZ_MAX_RECORDS 400u      /* the largest seeded program, two steps of two, is 333 */
#define FUZZ_WORKSPACE (1u << 20)  /* above every configuration's floor for these shapes */
#define FUZZ_MAX_STEPS 2u
#define FUZZ_MAX_BATCHES 4u        /* steps x accumulation */

/* the contracts a configuration names: v1, fused+exp, kblock, kblock+exp+reduce, bf16,
 * bf16+exp+attnbf16, attn, kblock+attn -- every GEMM entry, both softmaxes, the reduce pass, the
 * bf16 workspace and both attention contracts */
static const uint32_t g_contracts[8] = {0u, 3u, 4u, 14u, 16u, 82u, 32u, 36u};

typedef struct fuzz_arena { float *p; size_t n; } fuzz_arena;
static fuzz_arena g_arena[6];   /* weights, gradients, moment1, moment2, activations, scratch */
static unsigned char *g_workspace;
static bcir_decoder_plan g_plan;
static int g_ready;
static uint64_t g_program[FUZZ_MAX_RECORDS * BCIR_DT_RECORD_WORDS];
static uint32_t g_tokens[FUZZ_MAX_BATCHES * 4u], g_targets[FUZZ_MAX_BATCHES * 4u];
/* the state's bytes before a run and after the first of a pair, for the two properties */
static unsigned char *g_before, *g_first;
static size_t g_bytes;

#ifdef BCIR_FUZZ_REPLAY
/* A replay build (bcir/tests/test_native_place.py: a driver in libFuzzer's place) reads each
 * input's verdict here: its first run's status, -1 for an input that reached no run. */
extern int bcir_fuzz_replay[2];
int bcir_fuzz_replay[2];
#define FUZZ_REPLAY(slot, rc) ((void)(bcir_fuzz_replay[slot] = (int)(rc)))
#else
#define FUZZ_REPLAY(slot, rc) ((void)0)
#endif

static void setup(void) {
  bcir_decoder_spec spec;
  size_t i;
  memset(&spec, 0, sizeof spec);
  spec.vocab = 16; spec.width = 8; spec.heads = 2; spec.kvheads = 1; spec.layers = 2;
  spec.ff = 16; spec.batch = 1; spec.time = 4; spec.tied = 1;
  spec.rope_base = 10000.0; spec.rms_eps = 1e-6;
  if (bcir_decoder_make_plan(&spec, &g_plan) != BCIR_DT_OK) abort();
  g_arena[0].n = g_arena[1].n = g_arena[2].n = g_arena[3].n = g_plan.parameters;
  /* room for every lending: the plan's whole transient span (a placement of the activation and
   * scratch values both, and every recompute set's arena, fits) */
  g_arena[4].n = g_plan.activations + g_plan.scratch;
  g_arena[5].n = g_plan.scratch;
  g_bytes = FUZZ_WORKSPACE;
  for (i = 0; i < 6; i++) {
    g_arena[i].p = (float *)malloc(g_arena[i].n * sizeof(float));
    if (!g_arena[i].p) abort();
    g_bytes += g_arena[i].n * sizeof(float);
  }
  g_workspace = (unsigned char *)malloc(FUZZ_WORKSPACE);
  g_before = (unsigned char *)malloc(g_bytes);
  g_first = (unsigned char *)malloc(g_bytes);
  if (!g_workspace || !g_before || !g_first) abort();
  for (i = 0; i < FUZZ_MAX_BATCHES * 4u; i++) {
    g_tokens[i] = (uint32_t)((7u * i + 3u) % 16u);
    g_targets[i] = (uint32_t)((5u * i + 1u) % 16u);
  }
  g_ready = 1;
}

/* PX-4: the floats of space `space` (5 activations, 6 scratch) the program's operands reach --
 * each one's offset and bytes there, rounded up to a float -- counting only operands inside the
 * arena's `most` floats (the check refuses the others whatever is lent) */
static size_t reach(size_t records, unsigned space, size_t most) {
  size_t i, q, floats = 0;
  for (i = 0; i < records; i++)
    for (q = 0; q < 10u; q++) {
      const uint64_t *r = g_program + i * BCIR_DT_RECORD_WORDS + 11u + 3u * q;
      if (r[0] == space && r[1] <= (uint64_t)most * 4u && r[2] <= (uint64_t)most * 4u - r[1] &&
          (size_t)((r[1] + r[2] + 3u) / 4u) > floats)
        floats = (size_t)((r[1] + r[2] + 3u) / 4u);
    }
  return floats;
}

/* the state every run starts from: seeded weights, everything else zero, step 0; the transient
 * arenas lent at `acts` and `scratch` floats, each the tail of its allocation */
static void reset(bcir_decoder_state *s, uint32_t contract, uint64_t recompute, size_t acts,
                  size_t scratch) {
  size_t i, j;
  for (i = 0; i < 6; i++) memset(g_arena[i].p, 0, g_arena[i].n * sizeof(float));
  for (j = 0; j < g_arena[0].n; j++)
    g_arena[0].p[j] = (float)((int)((j * 2654435761u) % 2001u) - 1000) * 2e-5f;
  memset(g_workspace, 0, FUZZ_WORKSPACE);
  memset(s, 0, sizeof *s);
  s->plan = g_plan;
  s->weights = g_arena[0].p; s->gradients = g_arena[1].p;
  s->moment1 = g_arena[2].p; s->moment2 = g_arena[3].p;
  s->activations = g_arena[4].p + (g_arena[4].n - acts);
  s->scratch = g_arena[5].p + (g_arena[5].n - scratch);
  s->weight_capacity = s->gradient_capacity = s->moment1_capacity = s->moment2_capacity =
      g_plan.parameters;
  s->activation_capacity = acts;
  s->scratch_capacity = scratch;
  s->beta1_power = s->beta2_power = 1.0;
  s->contract = contract;
  s->workspace = (contract & BCIR_DT_CONTRACT_BF16) ? g_workspace : NULL;
  s->workspace_bytes = (contract & BCIR_DT_CONTRACT_BF16) ? FUZZ_WORKSPACE : 0u;
  s->recompute = recompute;
}

/* the state's bytes, in one buffer: the arenas, the workspace (all of it, written or not) */
static void capture(unsigned char *out) {
  size_t i, at = 0;
  for (i = 0; i < 6; i++) {
    memcpy(out + at, g_arena[i].p, g_arena[i].n * sizeof(float));
    at += g_arena[i].n * sizeof(float);
  }
  memcpy(out + at, g_workspace, FUZZ_WORKSPACE);
}

typedef struct fuzz_run {
  int rc; size_t completed; bcir_decoder_event events[FUZZ_MAX_STEPS];
  uint64_t step; double powers[2];
} fuzz_run;

static void run(bcir_decoder_state *s, size_t records, size_t steps, size_t accumulation,
                fuzz_run *out) {
  bcir_decoder_adamw opt = {1e-3, 0.9, 0.999, 1e-8, 0.01, 1.0};
  size_t count = steps * accumulation * 4u;
  memset(out, 0, sizeof *out);
  out->completed = 777u;  /* lent: a refusal must leave it */
  memset(out->events, 0xA5, sizeof out->events);
  out->rc = bcir_decoder_run_program(s, &opt, g_program, records, g_tokens, g_targets, count,
                                     NULL, steps, accumulation, out->events, steps,
                                     &out->completed);
  out->step = s->step; out->powers[0] = s->beta1_power; out->powers[1] = s->beta2_power;
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  bcir_decoder_state state;
  fuzz_run first, second;
  uint32_t contract;
  uint64_t recompute;
  size_t steps, accumulation, records, i, acts, scratch;
  unsigned char lent[sizeof first.events];
  FUZZ_REPLAY(0, -1);
  FUZZ_REPLAY(1, -1);
  if (!g_ready) setup();
  if (size < 2) return 0;
  contract = g_contracts[data[0] & 7u];
  recompute = (data[0] & 8u) ? 3u : 0u;  /* both layers: one shared cache slot */
  steps = 1u + (data[1] & 1u);
  accumulation = 1u + ((data[1] >> 1) & 1u);
  records = (size - 2u) / FUZZ_RECORD_BYTES;
  if (records > FUZZ_MAX_RECORDS) records = FUZZ_MAX_RECORDS;
  for (i = 0; i < records * BCIR_DT_RECORD_WORDS; i++) {  /* little-endian, at any alignment */
    uint64_t w = 0;
    unsigned b;
    for (b = 0; b < 8u; b++) w |= (uint64_t)data[2u + 8u * i + b] << (8u * b);
    g_program[i] = w;
  }
  /* the trainer's own arenas (NativeDecoder): the recompute set's activations, the plan's scratch */
  acts = bcir_decoder_activation_count(&g_plan, recompute);
  scratch = g_arena[5].n;
  if (acts > g_arena[4].n) abort();  /* a set of these two layers always has its count */
  if (data[0] & 16u) {  /* PX-4: lent at what the program reaches, one float short with bit 5 */
    acts = reach(records, 5u, g_arena[4].n);
    scratch = reach(records, 6u, g_arena[5].n);
    if (data[0] & 32u) {
      if (acts) acts--;
      else if (scratch) scratch--;
    }
  }

  reset(&state, contract, recompute, acts, scratch);
  capture(g_before);
  memset(lent, 0xA5, sizeof lent);
  run(&state, records, steps, accumulation, &first);
  FUZZ_REPLAY(0, first.rc);
  if (first.rc == BCIR_DT_INVALID) {
    /* refused before anything ran: no byte of the state, the events or the count written */
    capture(g_first);
    if (memcmp(g_first, g_before, g_bytes) != 0) abort();
    if (first.completed != 777u || first.step != 0u || first.powers[0] != 1.0 ||
        first.powers[1] != 1.0 || memcmp(first.events, lent, sizeof lent) != 0)
      abort();
    return 0;
  }
  if (first.rc < BCIR_DT_OK || first.rc > BCIR_DT_NUMERIC) abort();  /* a status the API names */
  if (first.completed > steps) abort();
  capture(g_first);
  /* the same program from the same state: the same status, count, events and state */
  reset(&state, contract, recompute, acts, scratch);
  run(&state, records, steps, accumulation, &second);
  if (second.rc != first.rc || second.completed != first.completed ||
      second.step != first.step || memcmp(second.powers, first.powers, sizeof first.powers) ||
      memcmp(second.events, first.events, first.completed * sizeof first.events[0]))
    abort();
  capture(g_before);
  if (memcmp(g_before, g_first, g_bytes) != 0) abort();
  return 0;
}
