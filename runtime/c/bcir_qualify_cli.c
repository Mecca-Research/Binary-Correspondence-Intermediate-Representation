/*===- bcir_qualify_cli.c - one decoder program, GEM against monolithic ---===
 *
 *   bcir-qualify MODEL.bcirq8 PACK (--der | --native) --prompt ID,ID,... [--reps N]
 *                [--logits-out FILE]
 *
 * Loads a BCIRQ8 decoder and a decoder program's StreamPack -- the ASN.1 DER projection
 * (converted to the native pack here, in C) or the native pack itself -- and runs the program
 * through GEM (bcir_decoder_gem) and the same request through the monolithic runner
 * (bcir_llama_generate_greedy), interleaved, N times each. One JSON object on stdout: the
 * tokens and final logits of both (and whether they agree bit for bit), wall-clock medians
 * and minima, time to first token and per-token latency of an instrumented GEM run, and the
 * octets each operation's kernels read and wrote. `--logits-out` writes the GEM run's final
 * logits as little-endian float64, for a comparison with another rail. `gem_runs_identical`
 * reports whether every GEM run -- each repetition and the instrumented one -- produced the same
 * tokens and logits. The process exits 0 only when both rails produced identical tokens and
 * bit-identical logits.
 *
 * Built with -DBCIR_QUALIFY_CALLGRIND (valgrind's <valgrind/callgrind.h>), the first repetition
 * of each rail is bracketed by callgrind client requests: the profile is zeroed before it and
 * dumped after it as a part named for the rail ("monolithic", "gem"), so a run under callgrind
 * measures one run of each rail on every architecture valgrind supports, without relying on its
 * call-graph tracking. Outside valgrind a client request does nothing; without the define the
 * brackets are empty.
 *===----------------------------------------------------------------------===*/
#define _POSIX_C_SOURCE 200809L
#include "bcir_asn1_streampack.h"
#include "bcir_decoder_gem.h"
#include "bcir_llama.h"
#include "bcir_sha256.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef BCIR_QUALIFY_CALLGRIND
#include <valgrind/callgrind.h>
static void rail_begin(void) { CALLGRIND_ZERO_STATS; }
static void rail_end(const char *rail) { CALLGRIND_DUMP_STATS_AT(rail); }
#else
static void rail_begin(void) {}
static void rail_end(const char *rail) { (void)rail; }
#endif

#define MAX_PROMPT 4096
#define MAX_REPS 1000
#define MAX_PACK_BYTES (256u * 1024u * 1024u)

static uint64_t now_ns(void *ctx) {
  struct timespec ts;
  (void)ctx;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

typedef struct token_clock {
  uint64_t start;
  uint64_t *at; /* per generated token: ns since the run started */
  size_t prompt_len, n;
} token_clock;

static void on_token(void *ctx, size_t position, int32_t token) {
  token_clock *tc = (token_clock *)ctx;
  (void)token;
  if (position >= tc->prompt_len && position - tc->prompt_len < tc->n)
    tc->at[position - tc->prompt_len] = now_ns(NULL) - tc->start;
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

static uint64_t median(uint64_t *v, size_t n) {
  qsort(v, n, sizeof *v, cmp_u64);
  return n % 2u ? v[n / 2u] : (v[n / 2u - 1u] + v[n / 2u]) / 2u;
}

static void hex_sha256(const void *data, size_t len, char out[65]) {
  bcir_sha256 s;
  uint8_t digest[32];
  size_t i;
  bcir_sha256_init(&s);
  bcir_sha256_update(&s, (const uint8_t *)data, len);
  bcir_sha256_final(&s, digest);
  for (i = 0; i < 32u; ++i) snprintf(out + 2u * i, 3u, "%02x", digest[i]);
}

/* SHA-256 over a run's generated ids and final logits: one run's outcome, to compare runs. */
static void run_sha256(const int32_t *ids, size_t n, const double *logits, size_t vocab,
                       uint8_t out[32]) {
  bcir_sha256 s;
  bcir_sha256_init(&s);
  bcir_sha256_update(&s, (const uint8_t *)ids, n * sizeof *ids);
  bcir_sha256_update(&s, (const uint8_t *)logits, vocab * sizeof *logits);
  bcir_sha256_final(&s, out);
}

static uint8_t *read_file(const char *path, size_t *len, bcir_host_allocator *a) {
  FILE *f = fopen(path, "rb");
  uint8_t *buf = NULL;
  long size;
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) == 0 && (size = ftell(f)) > 0 && (unsigned long)size <= MAX_PACK_BYTES &&
      fseek(f, 0, SEEK_SET) == 0) {
    buf = (uint8_t *)bcir_host_allocate(a, (size_t)size);
    if (buf && fread(buf, 1, (size_t)size, f) != (size_t)size) {
      bcir_host_deallocate(a, buf);
      buf = NULL;
    }
    if (buf) *len = (size_t)size;
  }
  fclose(f);
  return buf;
}

static size_t parse_ids(const char *text, int32_t *out, size_t cap) {
  size_t n = 0;
  while (*text && n < cap) {
    char *end;
    long v = strtol(text, &end, 10);
    if (end == text || v < 0 || v > INT32_MAX) return 0;
    out[n++] = (int32_t)v;
    text = end;
    if (*text == ',') ++text;
    else if (*text) return 0;
  }
  return *text ? 0 : n;
}

static void print_ids(const char *key, const int32_t *ids, size_t n) {
  size_t i;
  printf("\"%s\":[", key);
  for (i = 0; i < n; ++i) printf("%s%d", i ? "," : "", (int)ids[i]);
  printf("]");
}

int main(int argc, char **argv) {
  bcir_host_allocator a = bcir_host_allocator_default();
  bcir_q8_model model;
  bcir_dgem g;
  char error[256], sha_gem[65], sha_mono[65], sha_native[65];
  const char *model_path, *pack_path, *prompt_text = NULL, *logits_out = NULL;
  int der = -1, rc, i, ok;
  long reps = 5;
  int32_t prompt[MAX_PROMPT];
  size_t prompt_len, max_new, file_len = 0, pack_len = 0, k, vocab;
  uint8_t *file = NULL, *pack = NULL;
  uint64_t t0, convert_ns = 0, load_ns, *gem_ns, *mono_ns, *token_at, *trust_ns;
  int32_t *gen_gem, *gen_mono;
  double *logits_gem, *logits_mono, max_diff = 0.0;
  token_clock tc;
  uint64_t read_total = 0, written_total = 0, run_ns;
  uint8_t run_digest[32], first_digest[32];
  int runs_identical = 1;

  if (argc < 5) {
    fprintf(stderr, "usage: bcir-qualify MODEL.bcirq8 PACK (--der|--native) --prompt IDS [--reps N]"
                    " [--logits-out FILE]\n");
    return 2;
  }
  model_path = argv[1];
  pack_path = argv[2];
  for (i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--der")) der = 1;
    else if (!strcmp(argv[i], "--native")) der = 0;
    else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt_text = argv[++i];
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = strtol(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--logits-out") && i + 1 < argc) logits_out = argv[++i];
    else {
      fprintf(stderr, "bcir-qualify: unknown argument %s\n", argv[i]);
      return 2;
    }
  }
  if (der < 0 || !prompt_text || reps < 1 || reps > MAX_REPS ||
      !(prompt_len = parse_ids(prompt_text, prompt, MAX_PROMPT))) {
    fprintf(stderr, "bcir-qualify: need --der or --native, a --prompt, and 1..%d reps\n", MAX_REPS);
    return 2;
  }
  if (bcir_q8_model_load(model_path, &model, error, sizeof error)) {
    fprintf(stderr, "bcir-qualify: model: %s\n", error);
    return 1;
  }
  file = read_file(pack_path, &file_len, &a);
  if (!file) {
    fprintf(stderr, "bcir-qualify: cannot read %s\n", pack_path);
    bcir_q8_model_free(&model);
    return 1;
  }
  if (der) { /* the ASN.1 artifact, converted to the native pack in C */
    size_t cap = bcir_asn1_streampack_bound(file_len);
    pack = (uint8_t *)bcir_host_allocate(&a, cap ? cap : 1u);
    t0 = now_ns(NULL);
    if (!pack || bcir_asn1_to_streampack(file, file_len, pack, cap, &pack_len) != BCIR_OK) {
      fprintf(stderr, "bcir-qualify: the DER pack does not convert\n");
      bcir_host_deallocate(&a, pack);
      bcir_host_deallocate(&a, file);
      bcir_q8_model_free(&model);
      return 1;
    }
    convert_ns = now_ns(NULL) - t0;
  } else {
    pack = file;
    pack_len = file_len;
    file = NULL;
  }
  t0 = now_ns(NULL);
  rc = bcir_dgem_load(&g, &model, pack, pack_len, &a);
  load_ns = now_ns(NULL) - t0;
  if (rc) {
    fprintf(stderr, "bcir-qualify: the pack is not a decoder program for this model (%d)\n", rc);
    bcir_host_deallocate(&a, pack);
    bcir_host_deallocate(&a, file);
    bcir_q8_model_free(&model);
    return 1;
  }
  if (prompt_len != bcir_dgem_prompt_len(&g)) {
    fprintf(stderr, "bcir-qualify: the program's prompt has %zu ids, --prompt %zu\n",
            bcir_dgem_prompt_len(&g), prompt_len);
    bcir_dgem_free(&g);
    bcir_host_deallocate(&a, pack);
    bcir_host_deallocate(&a, file);
    bcir_q8_model_free(&model);
    return 2;
  }
  max_new = bcir_dgem_capacity(&g) - prompt_len;
  vocab = model.vocab_size;
  gen_gem = (int32_t *)bcir_host_allocate(&a, max_new * sizeof(int32_t));
  gen_mono = (int32_t *)bcir_host_allocate(&a, max_new * sizeof(int32_t));
  logits_gem = (double *)bcir_host_allocate(&a, vocab * sizeof(double));
  logits_mono = (double *)bcir_host_allocate(&a, vocab * sizeof(double));
  gem_ns = (uint64_t *)bcir_host_allocate(&a, (size_t)reps * sizeof(uint64_t));
  mono_ns = (uint64_t *)bcir_host_allocate(&a, (size_t)reps * sizeof(uint64_t));
  token_at = (uint64_t *)bcir_host_allocate(&a, max_new * sizeof(uint64_t));
  trust_ns = (uint64_t *)bcir_host_allocate(&a, (size_t)reps * sizeof(uint64_t));
  if (!gen_gem || !gen_mono || !logits_gem || !logits_mono || !gem_ns || !mono_ns || !token_at ||
      !trust_ns) {
    fprintf(stderr, "bcir-qualify: out of memory\n");
    return 1;
  }

  /* Interleaved: the same request on both rails, alternately, `reps` times each. */
  for (i = 0; i < reps; ++i) {
    t0 = now_ns(NULL);
    if (!i) rail_begin();
    rc = bcir_llama_generate_greedy(&model, prompt, prompt_len, max_new, gen_mono, logits_mono);
    if (!i) rail_end("monolithic");
    mono_ns[i] = now_ns(NULL) - t0;
    if (rc) {
      fprintf(stderr, "bcir-qualify: the monolithic runner refused (%d)\n", rc);
      return 1;
    }
    t0 = now_ns(NULL);
    if (!i) rail_begin();
    rc = bcir_dgem_run(&g, prompt, prompt_len, gen_gem, max_new, logits_gem);
    if (!i) rail_end("gem");
    gem_ns[i] = now_ns(NULL) - t0;
    if (rc) {
      fprintf(stderr, "bcir-qualify: GEM refused the program (%d, pack status %d)\n", rc,
              (int)g.pack_status);
      return 1;
    }
    run_sha256(gen_gem, max_new, logits_gem, vocab, i ? run_digest : first_digest);
    if (i && memcmp(run_digest, first_digest, sizeof run_digest)) runs_identical = 0;
  }
  /* The executor's trust boundary, alone: every bcir_sp_execute validates the pack (magic,
   * version, CRC) and walks R10 before it dispatches anything -- on every run, by design. */
  for (i = 0; i < reps; ++i) {
    bcir_streampack_header hdr;
    t0 = now_ns(NULL);
    if (bcir_sp_validate(pack, pack_len, &hdr) != BCIR_OK ||
        bcir_sp_verify_semantic(pack, pack_len, 0xFFFFFFFFu, 0xFFFFFFFFu) != BCIR_OK)
      return 1;
    trust_ns[i] = now_ns(NULL) - t0;
  }
  /* One instrumented GEM run: per-operation time and per-token latency. */
  bcir_dgem_reset_stats(&g);
  g.clock = now_ns;
  tc.prompt_len = prompt_len;
  tc.n = max_new;
  tc.at = token_at;
  g.on_token = on_token;
  g.token_ctx = &tc;
  tc.start = now_ns(NULL);
  rc = bcir_dgem_run(&g, prompt, prompt_len, gen_gem, max_new, logits_gem);
  run_ns = now_ns(NULL) - tc.start;
  if (rc) return 1;
  run_sha256(gen_gem, max_new, logits_gem, vocab, run_digest);
  if (memcmp(run_digest, first_digest, sizeof run_digest)) runs_identical = 0;

  ok = !memcmp(gen_gem, gen_mono, max_new * sizeof(int32_t)) &&
       !memcmp(logits_gem, logits_mono, vocab * sizeof(double));
  for (k = 0; k < vocab; ++k) {
    double diff = fabs(logits_gem[k] - logits_mono[k]);
    if (diff > max_diff) max_diff = diff;
  }
  hex_sha256(logits_gem, vocab * sizeof(double), sha_gem);
  hex_sha256(logits_mono, vocab * sizeof(double), sha_mono);
  hex_sha256(pack, pack_len, sha_native);
  if (logits_out) { /* little-endian float64, as bcir-llama --logits-out writes them */
    FILE *f = fopen(logits_out, "wb");
    int wrote = 0;
    if (f) {
      for (k = 0; k < vocab; ++k) {
        uint64_t bits;
        unsigned char le[8];
        int b;
        memcpy(&bits, &logits_gem[k], sizeof bits);
        for (b = 0; b < 8; ++b) le[b] = (unsigned char)(bits >> (8 * b));
        if (fwrite(le, 1, 8, f) != 8) break;
      }
      wrote = k == vocab;
      if (fclose(f)) wrote = 0;
    }
    if (!wrote) {
      fprintf(stderr, "bcir-qualify: cannot write %s\n", logits_out);
      return 1;
    }
  }

  printf("{\"status\":\"%s\",", ok ? "PASS" : "FAIL");
  printf("\"model\":{\"vocab\":%u,\"d_model\":%u,\"layers\":%u,\"heads\":%u,\"kv_heads\":%u,"
         "\"d_ff\":%u,\"group_size\":%u},",
         model.vocab_size, model.d_model, model.n_layers, model.n_heads, model.n_kv_heads,
         model.d_ff, model.group_size);
  printf("\"program\":{\"claims\":%zu,\"capacity\":%zu,\"prompt_len\":%zu,\"max_new\":%zu,"
         "\"pack_format\":\"%s\",\"pack_bytes\":%zu,\"native_bytes\":%zu,"
         "\"native_sha256\":\"%s\"},",
         g.n_claims, bcir_dgem_capacity(&g), prompt_len, max_new, der ? "der" : "native",
         der ? file_len : pack_len, pack_len, sha_native);
  printf("\"tokens\":{");
  print_ids("gem", gen_gem, max_new);
  printf(",");
  print_ids("monolithic", gen_mono, max_new);
  printf("},");
  printf("\"logits\":{\"bitwise_identical\":%s,\"max_abs_diff\":%.17g,\"sha256_gem\":\"%s\","
         "\"sha256_monolithic\":\"%s\",\"gem_runs_identical\":%s},",
         memcmp(logits_gem, logits_mono, vocab * sizeof(double)) ? "false" : "true", max_diff,
         sha_gem, sha_mono, runs_identical ? "true" : "false");
  {
    uint64_t gem_min = gem_ns[0], mono_min = mono_ns[0];
    for (i = 1; i < reps; ++i) {
      if (gem_ns[i] < gem_min) gem_min = gem_ns[i];
      if (mono_ns[i] < mono_min) mono_min = mono_ns[i];
    }
    printf("\"timing\":{\"reps\":%ld,\"der_to_native_ns\":%llu,\"load_ns\":%llu,"
           "\"gem_min_ns\":%llu,\"monolithic_min_ns\":%llu,",
           reps, (unsigned long long)convert_ns, (unsigned long long)load_ns,
           (unsigned long long)gem_min, (unsigned long long)mono_min);
    printf("\"gem_median_ns\":%llu,\"monolithic_median_ns\":%llu,\"trust_boundary_median_ns\":%llu,",
           (unsigned long long)median(gem_ns, (size_t)reps),
           (unsigned long long)median(mono_ns, (size_t)reps),
           (unsigned long long)median(trust_ns, (size_t)reps));
  }
  printf("\"instrumented_run_ns\":%llu,\"time_to_first_token_ns\":%llu,\"token_ns\":[",
         (unsigned long long)run_ns, (unsigned long long)(max_new ? token_at[0] : 0u));
  for (k = 0; k < max_new; ++k)
    printf("%s%llu", k ? "," : "", (unsigned long long)token_at[k]);
  printf("]},\"operations\":{");
  for (k = 0; k < (size_t)BCIR_DGEM_N_OPS; ++k) {
    printf("%s\"%s\":{\"claims\":%llu,\"bytes_read\":%llu,\"bytes_written\":%llu,\"ns\":%llu}",
           k ? "," : "", bcir_dgem_op_name((bcir_dgem_op)k),
           (unsigned long long)g.stats.claims[k], (unsigned long long)g.stats.bytes_read[k],
           (unsigned long long)g.stats.bytes_written[k], (unsigned long long)g.stats.ns[k]);
    read_total += g.stats.bytes_read[k];
    written_total += g.stats.bytes_written[k];
  }
  printf("},\"traffic\":{\"bytes_read\":%llu,\"bytes_written\":%llu}}\n",
         (unsigned long long)read_total, (unsigned long long)written_total);

  bcir_host_deallocate(&a, gen_gem);
  bcir_host_deallocate(&a, gen_mono);
  bcir_host_deallocate(&a, logits_gem);
  bcir_host_deallocate(&a, logits_mono);
  bcir_host_deallocate(&a, gem_ns);
  bcir_host_deallocate(&a, mono_ns);
  bcir_host_deallocate(&a, token_at);
  bcir_host_deallocate(&a, trust_ns);
  bcir_dgem_free(&g);
  bcir_host_deallocate(&a, pack);
  bcir_host_deallocate(&a, file);
  bcir_q8_model_free(&model);
  return ok ? 0 : 3;
}
