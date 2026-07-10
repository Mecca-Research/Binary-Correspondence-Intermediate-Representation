/*===- bcir_llama_cli.c - host CLI for standalone BCIRQ8 inference ------===*/
#include "bcir_llama.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *argv0) {
  fprintf(stderr, "usage: %s --model FILE --prompt-ids I,J,... --max-new N "
                  "[--logits-out FILE]\n", argv0);
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
  *out = (int32_t *)malloc(n * sizeof **out);
  copy = (char *)malloc(text_size + 1);
  if (!*out || !copy) { free(*out); free(copy); return -1; }
  strcpy(copy, text); p = copy;
  while (*p) {
    errno = 0; value = strtol(p, &end, 10);
    if (errno || end == p || value < INT32_MIN || value > INT32_MAX ||
        (*end && *end != ',')) { free(copy); free(*out); *out = NULL; return -1; }
    (*out)[i++] = (int32_t)value;
    if (!*end) break;
    p = end + 1;
    if (!*p) { free(copy); free(*out); *out = NULL; return -1; }
  }
  free(copy); *count = i; return 0;
}

static int write_f64_le(const char *path, const double *values, size_t count) {
  FILE *f = fopen(path, "wb");
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
  size_t max_new = 0, prompt_count = 0, i;
  int32_t *prompt = NULL, *generated = NULL;
  double *logits = NULL;
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
    } else { usage(argv[0]); return 2; }
  }
  if (!model_path || !prompt_text || !max_new || parse_prompt(prompt_text, &prompt, &prompt_count)) {
    usage(argv[0]); return 2;
  }
  if (bcir_q8_model_load(model_path, &model, error, sizeof error)) {
    fprintf(stderr, "bcir-llama: %s\n", error); free(prompt); return 1;
  }
  if (max_new > SIZE_MAX / sizeof *generated ||
      (size_t)model.vocab_size > SIZE_MAX / sizeof *logits) {
    fprintf(stderr, "bcir-llama: requested output is too large\n");
    rc = 1; goto done;
  }
  generated = (int32_t *)malloc(max_new * sizeof *generated);
  logits = (double *)malloc((size_t)model.vocab_size * sizeof *logits);
  if (!generated || !logits) { fprintf(stderr, "bcir-llama: out of memory\n"); rc = 1; goto done; }
  rc = bcir_llama_generate_greedy(&model, prompt, prompt_count, max_new, generated, logits);
  if (rc) { fprintf(stderr, "bcir-llama: inference failed (%d)\n", rc); rc = 1; goto done; }
  fputs("{\"generated_ids\":[", stdout);
  for (i = 0; i < max_new; ++i) printf("%s%d", i ? "," : "", generated[i]);
  fputs("]}\n", stdout);
  if (logits_path && write_f64_le(logits_path, logits, model.vocab_size)) {
    fprintf(stderr, "bcir-llama: cannot write logits\n"); rc = 1; goto done;
  }
  rc = 0;
done:
  free(logits); free(generated); free(prompt); bcir_q8_model_free(&model); return rc;
}
