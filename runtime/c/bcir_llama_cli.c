/*===- bcir_llama_cli.c - host CLI for standalone BCIRQ8 inference ------===*/
#include "bcir_llama.h"
#include "bcir_tensor.h" /* DECODE-THREADS: the deterministic pool --threads lends the decoder */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BCIR_LLAMA_CLI_MAX_NEW_TOKENS (1u << 16)

static void *cli_allocate(size_t size) {
  bcir_host_allocator allocator=bcir_host_allocator_default();
  return bcir_host_allocate(&allocator,size);
}

static void cli_deallocate(void *allocation) {
  bcir_host_allocator allocator=bcir_host_allocator_default();
  bcir_host_deallocate(&allocator,allocation);
}

static FILE *cli_open_file(const char *path, const char *mode) {
#if defined(_WIN32)
  FILE *file = NULL;
  return fopen_s(&file, path, mode) == 0 ? file : NULL;
#else
  return fopen(path, mode);
#endif
}

static void usage(const char *argv0) {
  fprintf(stderr, "usage: %s --model FILE --prompt-ids I,J,... --max-new N "
                  "[--logits-out FILE] [--threads T] [--split-min W]\n", argv0);
}

/* DECODE-THREADS: the pool runs a decoder operation's parts; every token and logit is the same at every --threads */
static void cli_pool_run(void *self, bcir_llama_task_fn task, void *ctx, size_t tasks) {
  bcir_tensor_pool_run((bcir_tensor_pool *)self, task, ctx, tasks);
}

static int parse_size(const char *text, size_t *out) {
  char *end;
  unsigned long long value;
  if (!text || *text < '0' || *text > '9') return -1;
  errno = 0;
  value = strtoull(text, &end, 10);
  if (errno || *end || value > SIZE_MAX) return -1;
  *out = (size_t)value;
  return 0;
}

static int parse_prompt(const char *text, int32_t **out, size_t *count) {
  char *copy, *p, *end;
  size_t text_size, n = 1, i = 0;
  long value;
  if (!text || !*text) return -1;
  for (p = (char *)text; *p; ++p) if (*p == ',') {
    if (n == SIZE_MAX) return -1;
    ++n;
  }
  text_size = strlen(text);
  if (n > SIZE_MAX / sizeof **out || text_size == SIZE_MAX) return -1;
  *out = (int32_t *)cli_allocate(n * sizeof **out);
  copy = (char *)cli_allocate(text_size + 1);
  if (!*out || !copy) { cli_deallocate(*out); cli_deallocate(copy); return -1; }
  memcpy(copy, text, text_size + 1); p = copy;
  while (*p) {
    errno = 0; value = strtol(p, &end, 10);
    if (errno || end == p || value < INT32_MIN || value > INT32_MAX ||
        (*end && *end != ',')) { cli_deallocate(copy); cli_deallocate(*out); *out = NULL; return -1; }
    (*out)[i++] = (int32_t)value;
    if (!*end) break;
    p = end + 1;
    if (!*p) { cli_deallocate(copy); cli_deallocate(*out); *out = NULL; return -1; }
  }
  cli_deallocate(copy); *count = i; return 0;
}

static int write_f64_le(const char *path, const double *values, size_t count) {
  FILE *f = cli_open_file(path, "wb");
  size_t i;
  if (!f) return -1;
  for (i = 0; i < count; ++i) {
    uint64_t bits; unsigned char b[8]; int j;
    memcpy(&bits, &values[i], sizeof bits);
    for (j = 0; j < 8; ++j) b[j] = (unsigned char)(bits >> (8 * j));
    if (fwrite(b, 1, 8, f) != 8) { fclose(f); return -1; }
  }
  return fclose(f) ? -1 : 0;
}

int main(int argc, char **argv) {
  const char *model_path = NULL, *prompt_text = NULL, *logits_path = NULL;
  size_t max_new = 0, prompt_count = 0, i, generated_bytes, logits_bytes;
  size_t threads = 1, split_min = SIZE_MAX, pool_bytes = 0;
  int32_t *prompt = NULL, *generated = NULL;
  double *logits = NULL;
  void *pool_memory = NULL;
  bcir_tensor_pool *pool = NULL;
  bcir_llama_runner runner;
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_q8_model model;
  char error[256];
  int rc;
  for (i = 1; i < (size_t)argc; ++i) {
    if (!strcmp(argv[i], "--model") && i + 1 < (size_t)argc) model_path = argv[++i];
    else if (!strcmp(argv[i], "--prompt-ids") && i + 1 < (size_t)argc) prompt_text = argv[++i];
    else if (!strcmp(argv[i], "--max-new") && i + 1 < (size_t)argc) {
      char *end; unsigned long long value;
      errno = 0; value = strtoull(argv[++i], &end, 10);
      if (errno || *end || value > SIZE_MAX) { usage(argv[0]); return 2; }
      max_new = (size_t)value;
    } else if (!strcmp(argv[i], "--logits-out") && i + 1 < (size_t)argc) {
      logits_path = argv[++i];
    } else if (!strcmp(argv[i], "--threads") && i + 1 < (size_t)argc) {
      if (parse_size(argv[++i], &threads) || threads < 1u || threads > BCIR_LLAMA_MAX_PARTS) {
        usage(argv[0]); return 2;
      }
    } else if (!strcmp(argv[i], "--split-min") && i + 1 < (size_t)argc) {
      if (parse_size(argv[++i], &split_min) || split_min == SIZE_MAX) { usage(argv[0]); return 2; }
    } else { usage(argv[0]); return 2; }
  }
  if (!model_path || !prompt_text || !max_new || parse_prompt(prompt_text, &prompt, &prompt_count)) {
    usage(argv[0]); return 2;
  }
  if (bcir_q8_model_load(model_path, &model, error, sizeof error)) {
    fprintf(stderr, "bcir-llama: %s\n", error); cli_deallocate(prompt); return 1;
  }
  if (max_new > BCIR_LLAMA_CLI_MAX_NEW_TOKENS ||
      prompt_count > SIZE_MAX - max_new ||
      (model.context_length && prompt_count + max_new - 1u > model.context_length)) {
    fprintf(stderr, "bcir-llama: request exceeds the model/CLI context limit\n");
    rc = 1; goto done;
  }
  if (!bcir_size_mul(max_new,sizeof *generated,&generated_bytes) ||
      !bcir_size_mul((size_t)model.vocab_size,sizeof *logits,&logits_bytes)) {
    fprintf(stderr, "bcir-llama: requested output is too large\n");
    rc = 1; goto done;
  }
  generated = (int32_t *)cli_allocate(generated_bytes);
  logits = (double *)cli_allocate(logits_bytes);
  if (!generated || !logits) { fprintf(stderr, "bcir-llama: out of memory\n"); rc = 1; goto done; }
  /* DECODE-THREADS: a pool of `threads` lent as the decoder's runner (the caller's thread its first), its
   * thresholds the decoder's or --split-min's for both */
  memset(&runner, 0, sizeof runner);
  if (threads > 1u) {
    pool_bytes = bcir_tensor_pool_size((unsigned)threads);
    pool_memory = pool_bytes ? cli_allocate(pool_bytes) : NULL;
    pool = pool_memory ? bcir_tensor_pool_init(pool_memory, pool_bytes, (unsigned)threads) : NULL;
    if (!pool) { fprintf(stderr, "bcir-llama: cannot start %zu threads\n", threads); rc = 1; goto done; }
    runner.run = cli_pool_run;
    runner.self = pool;
    runner.workers = (unsigned)threads;
    runner.split_min = split_min == SIZE_MAX ? BCIR_LLAMA_SPLIT_MIN : split_min;
    runner.head_split_min = split_min == SIZE_MAX ? BCIR_LLAMA_HEAD_SPLIT_MIN : split_min;
  }
  rc = bcir_llama_generate_greedy_with_runner(&model, prompt, prompt_count, max_new, generated, logits,
                                              &allocator, pool ? &runner : NULL);
  if (rc) { fprintf(stderr, "bcir-llama: inference failed (%d)\n", rc); rc = 1; goto done; }
  fputs("{\"generated_ids\":[", stdout);
  for (i = 0; i < max_new; ++i) printf("%s%d", i ? "," : "", generated[i]);
  fputs("]}\n", stdout);
  if (logits_path && write_f64_le(logits_path, logits, model.vocab_size)) {
    fprintf(stderr, "bcir-llama: cannot write logits\n"); rc = 1; goto done;
  }
  rc = 0;
done:
  bcir_tensor_pool_stop(pool); cli_deallocate(pool_memory);
  cli_deallocate(logits); cli_deallocate(generated); cli_deallocate(prompt);
  bcir_q8_model_free(&model); return rc;
}
