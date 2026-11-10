/*===- bcir_tokenize_cli.c - bcir-tokenize: byte-level BPE encode / decode / info -===
 *
 * The command line of the C tokenizer twin (PORT-2, bcir_bpe.h), held to the oracle's
 * bcir/frontends/models/tokenizer.py by bcir/tests/test_c_bpe_tokenizer.py:
 *
 *   bcir-tokenize [--max-bytes N] info TOKENIZER          sha256, vocab, merges and specials counts
 *   bcir-tokenize [--max-bytes N] encode TOKENIZER [TEXT]  UTF-8 text -> ids, one line, space-separated
 *   bcir-tokenize [--max-bytes N] decode TOKENIZER [IDS]   ids -> the decoded octets
 *   bcir-tokenize limits                                  the reader's limits profile, a field a line
 *
 * TEXT and IDS are files, standard input when absent or "-"; IDS holds ASCII decimal ids separated
 * by ASCII white space (space, tab, LF, CR). --max-bytes bounds the tokenizer file as
 * load_tokenizer's max_bytes does, from 1 to the reader's own input limit (268435456, the default);
 * the text and the ids are bounded by that limit too. Output is octets on every host ("\n", never
 * "\r\n"). A refusal prints `bcir-tokenize: <reason>` on standard error, nothing on standard output,
 * and exits 2. Arguments are read in their full spellings, numbers as ASCII decimal digits.
 *===----------------------------------------------------------------------===*/
#include "bcir_bpe.h"
#include "bcir_host_alloc.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#define TOKENIZE_MAX_BYTES ((size_t)256u << 20)

static const char USAGE[] =
    "usage: bcir-tokenize [--max-bytes N] {info,encode,decode} TOKENIZER [INPUT]\n"
    "       bcir-tokenize limits\n";

static int refuse(const char *reason, const char *detail) {
  fprintf(stderr, "bcir-tokenize: %s%s\n", reason, detail ? detail : "");
  return 2;
}

static int usage_error(const char *reason) {
  fputs(USAGE, stderr);
  return refuse(reason, NULL);
}

/* ASCII decimal digits text[0..len), at least one. 1 parsed, 0 not digits, -1 past `max`. */
static int parse_count(const char *text, size_t len, uint64_t max, uint64_t *out) {
  uint64_t value = 0;
  int over = 0;
  if (!len) return 0;
  for (size_t i = 0; i < len; i++) {
    if (text[i] < '0' || text[i] > '9') return 0;
    uint64_t digit = (uint64_t)(text[i] - '0');
    if (over || value > (max - digit) / 10u) over = 1;
    else value = value * 10u + digit;
  }
  if (over) return -1;
  *out = value;
  return 1;
}

/* `--name VALUE` or `--name=VALUE` at argv[*i]: 1 matched, 0 not this option, -1 no value. */
static int option(int argc, char **argv, int *i, const char *name, const char **value) {
  size_t n = strlen(name);
  const char *arg = argv[*i];
  if (strncmp(arg, name, n) != 0) return 0;
  if (arg[n] == '=') {
    *value = arg + n + 1;
    (*i)++;
    return 1;
  }
  if (arg[n] != '\0') return 0;
  if (*i + 1 >= argc) return -1;
  *value = argv[*i + 1];
  *i += 2;
  return 1;
}

static const char *law_text(bcir_bpe_law law) {
  switch (law) {
  case BCIR_BPE_LAW_ROOT: return "tokenizer.json root must be an object";
  case BCIR_BPE_LAW_MODEL: return "the tokenizer model must be an object of type \"BPE\"";
  case BCIR_BPE_LAW_VOCAB: return "vocab must be an object of nonempty byte-alphabet tokens";
  case BCIR_BPE_LAW_ID: return "a token id must be an integer in [0, 2^31-1] naming one token";
  case BCIR_BPE_LAW_BASE_BYTES: return "byte-level BPE vocab is missing base byte tokens";
  case BCIR_BPE_LAW_MERGES: return "merges must be distinct, well-formed pairs of vocab tokens";
  case BCIR_BPE_LAW_ADDED: return "added_tokens must be distinct, consistent tokens with content and id";
  case BCIR_BPE_LAW_DUPLICATE_KEY: return "a member name repeats within one object";
  case BCIR_BPE_LAW_NONE:
  default: return "malformed tokenizer.json";
  }
}

static int refuse_load(const bcir_bpe_diag *d) {
  char detail[128];
  if (d->status == BCIR_BPE_ERR_NOMEM) return refuse("out of memory loading the tokenizer", NULL);
  if (d->status == BCIR_BPE_ERR_JSON) {
    if (d->offset == BCIR_JER_NO_OFFSET) snprintf(detail, sizeof detail, "JER status %d", (int)d->jer.status);
    else snprintf(detail, sizeof detail, "JER status %d at octet %zu", (int)d->jer.status, d->offset);
    return refuse("malformed tokenizer.json: ", detail);
  }
  if (d->offset == BCIR_JER_NO_OFFSET) return refuse(law_text(d->law), NULL);
  snprintf(detail, sizeof detail, " (at octet %zu)", d->offset);
  return refuse(law_text(d->law), detail);
}

static FILE *open_file(const char *path) {
#if defined(_WIN32)
  FILE *file = NULL;
  return fopen_s(&file, path, "rb") == 0 ? file : NULL;
#else
  return fopen(path, "rb");
#endif
}

/* The whole of `file`, at most `maximum` octets: *data owned by the caller. 0 read, 2 refused. */
static int read_stream(FILE *file, const char *name, size_t maximum, uint8_t **data, size_t *len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t limit = maximum + 1u; /* one past the bound tells */
  size_t cap = limit < ((size_t)1u << 16) ? limit : ((size_t)1u << 16);
  size_t have = 0;
  uint8_t *buf = (uint8_t *)bcir_host_allocate(&allocator, cap);
  if (!buf) return refuse("out of memory reading ", name);
  for (;;) {
    if (have == cap) {
      if (cap == limit) break;
      size_t grown = cap > limit / 2u ? limit : cap * 2u;
      void *next = buf;
      if (!bcir_host_realloc_array(&allocator, &next, cap, grown, 1u, 0)) {
        bcir_host_deallocate(&allocator, buf);
        return refuse("out of memory reading ", name);
      }
      buf = (uint8_t *)next;
      cap = grown;
    }
    size_t want = cap - have;
    size_t got = fread(buf + have, 1u, want, file);
    have += got;
    if (got < want) break; /* end of file or an error (ferror below says which): read no further */
  }
  if (ferror(file)) {
    bcir_host_deallocate(&allocator, buf);
    return refuse("cannot read ", name);
  }
  if (have > maximum) {
    bcir_host_deallocate(&allocator, buf);
    fprintf(stderr, "bcir-tokenize: %s exceeds %zu octets\n", name, maximum);
    return 2;
  }
  *data = buf;
  *len = have;
  return 0;
}

static int read_path(const char *path, size_t maximum, uint8_t **data, size_t *len) {
  if (!path || strcmp(path, "-") == 0) return read_stream(stdin, "standard input", maximum, data, len);
  FILE *file = open_file(path);
  if (!file) return refuse("cannot open ", path);
  int rc = read_stream(file, path, maximum, data, len);
  fclose(file);
  return rc;
}

static void print_limits(void) {
  bcir_jer_limits l;
  bcir_bpe_limits(&l);
  printf("input_bytes %llu\ndepth %u\nnodes %llu\nmembers %llu\nelements %llu\nstring_bytes %llu\n"
         "number_bytes %llu\ninteger_digits %u\nexponent_magnitude %llu\nwork %llu\n",
         (unsigned long long)l.input_bytes, (unsigned)l.depth, (unsigned long long)l.nodes,
         (unsigned long long)l.members, (unsigned long long)l.elements, (unsigned long long)l.string_bytes,
         (unsigned long long)l.number_bytes, (unsigned)l.integer_digits, (unsigned long long)l.exponent_magnitude,
         (unsigned long long)l.work);
}

static void print_info(const bcir_bpe *t) {
  static const char hex[] = "0123456789abcdef";
  uint8_t digest[32];
  char text[65];
  bcir_bpe_digest(t, digest);
  for (unsigned i = 0; i < 32u; i++) {
    text[2u * i] = hex[digest[i] >> 4];
    text[2u * i + 1u] = hex[digest[i] & 15u];
  }
  text[64] = '\0';
  printf("sha256 %s\nvocab %zu\nmerges %zu\nspecials %zu\n", text, bcir_bpe_vocab_size(t), bcir_bpe_merge_count(t),
         bcir_bpe_special_count(t));
}

static int encode(const bcir_bpe *t, const uint8_t *text, size_t len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t count = 0, bytes;
  uint32_t *ids;
  if (!bcir_size_mul(len ? len : 1u, sizeof(uint32_t), &bytes) ||
      !(ids = (uint32_t *)bcir_host_allocate(&allocator, bytes)))
    return refuse("out of memory encoding the text", NULL);
  bcir_bpe_status st = bcir_bpe_encode(t, text, len, ids, len, &count); /* len ids always suffice */
  if (st == BCIR_BPE_OK) {
    for (size_t i = 0; i < count; i++) printf(i ? " %u" : "%u", (unsigned)ids[i]);
    putchar('\n');
  }
  bcir_host_deallocate(&allocator, ids);
  switch (st) {
  case BCIR_BPE_OK: return 0;
  case BCIR_BPE_ERR_NOT_UTF8: return refuse("the text is not UTF-8", NULL);
  case BCIR_BPE_ERR_NOMEM: return refuse("out of memory encoding the text", NULL);
  default: return refuse("cannot encode the text", NULL);
  }
}

static int is_space(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static int decode(const bcir_bpe *t, const uint8_t *in, size_t len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t n = 0, bytes, written = 0;
  uint32_t *ids;
  uint8_t *out;
  /* at most one id per two input octets, plus one */
  if (!bcir_size_mul(len / 2u + 1u, sizeof(uint32_t), &bytes) ||
      !(ids = (uint32_t *)bcir_host_allocate(&allocator, bytes)))
    return refuse("out of memory decoding", NULL);
  for (size_t i = 0; i < len;) {
    size_t j = i;
    uint64_t id = 0;
    if (is_space(in[i])) {
      i++;
      continue;
    }
    while (j < len && !is_space(in[j])) j++;
    int parsed = parse_count((const char *)in + i, j - i, UINT32_MAX, &id);
    if (parsed <= 0) {
      bcir_host_deallocate(&allocator, ids);
      return parsed < 0 ? refuse("unknown tokenizer id", NULL) : refuse("ids are ASCII decimal numbers", NULL);
    }
    ids[n++] = (uint32_t)id;
    i = j;
  }
  /* measure, then decode into a buffer of exactly that size */
  bcir_bpe_status st = bcir_bpe_decode(t, ids, n, NULL, 0, &written);
  if (st == BCIR_BPE_ERR_NOSPACE) {
    if (!(out = (uint8_t *)bcir_host_allocate(&allocator, written))) {
      bcir_host_deallocate(&allocator, ids);
      return refuse("out of memory decoding", NULL);
    }
    st = bcir_bpe_decode(t, ids, n, out, written, &written);
    if (st == BCIR_BPE_OK && fwrite(out, 1u, written, stdout) != written) st = BCIR_BPE_ERR_INVALID;
    bcir_host_deallocate(&allocator, out);
  }
  bcir_host_deallocate(&allocator, ids);
  switch (st) {
  case BCIR_BPE_OK: return 0;
  case BCIR_BPE_ERR_UNKNOWN_ID: return refuse("unknown tokenizer id", NULL);
  case BCIR_BPE_ERR_NOT_UTF8: return refuse("the decoded bytes are not UTF-8", NULL);
  case BCIR_BPE_ERR_INVALID: return refuse("cannot write standard output", NULL);
  default: return refuse("cannot decode the ids", NULL);
  }
}

int main(int argc, char **argv) {
  uint64_t max_bytes = TOKENIZE_MAX_BYTES;
  const char *value = NULL, *path = NULL, *input = NULL;
  int i = 1, m, rc;

  for (int k = 1; k < argc; k++) {
    if (strcmp(argv[k], "-h") == 0 || strcmp(argv[k], "--help") == 0) {
      fputs(USAGE, stdout);
      return 0;
    }
  }
#if defined(_WIN32)
  (void)_setmode(_fileno(stdin), _O_BINARY);
  (void)_setmode(_fileno(stdout), _O_BINARY); /* "\n", never "\r\n" */
#endif
  while (i < argc && (m = option(argc, argv, &i, "--max-bytes", &value)) != 0) {
    if (m < 0 || parse_count(value, strlen(value), UINT64_MAX, &max_bytes) <= 0)
      return usage_error("--max-bytes takes a number");
  }
  if (i >= argc) return usage_error("a command is required");
  const char *command = argv[i++];
  int info = !strcmp(command, "info"), enc = !strcmp(command, "encode"), dec = !strcmp(command, "decode");
  if (!strcmp(command, "limits")) {
    if (i < argc) return usage_error("limits takes no arguments");
    print_limits();
    return fflush(stdout) == 0 ? 0 : refuse("cannot write standard output", NULL);
  }
  if (!(info || enc || dec)) return usage_error("the command is info, encode, decode or limits");
  for (; i < argc; i++) {
    if (argv[i][0] == '-' && argv[i][1] != '\0') return usage_error("unknown option");
    if (!path) path = argv[i];
    else if (!info && !input) input = argv[i];
    else return usage_error("too many arguments");
  }
  if (!path) return usage_error("a TOKENIZER is required");
  if (max_bytes < 1u || max_bytes > TOKENIZE_MAX_BYTES) return refuse("--max-bytes must be in [1, 268435456]", NULL);

  uint8_t *json = NULL, *data = NULL;
  size_t json_len = 0, data_len = 0;
  if ((rc = read_path(path, (size_t)max_bytes, &json, &json_len)) != 0) return rc;
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_bpe *t = NULL;
  bcir_bpe_diag diag;
  bcir_bpe_status st = bcir_bpe_load(json, json_len, NULL, &t, &diag);
  bcir_host_deallocate(&allocator, json);
  if (st != BCIR_BPE_OK) return refuse_load(&diag);
  if (info) {
    print_info(t);
    rc = 0;
  } else if ((rc = read_path(input, TOKENIZE_MAX_BYTES, &data, &data_len)) == 0) {
    rc = enc ? encode(t, data, data_len) : decode(t, data, data_len);
    bcir_host_deallocate(&allocator, data);
  }
  bcir_bpe_free(t);
  if (rc) return rc;
  return fflush(stdout) == 0 ? 0 : refuse("cannot write standard output", NULL);
}
