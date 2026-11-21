/*===- bcir_tokenize_cli.c - bcir-tokenize: BPE and SentencePiece encode / decode / info -===
 *
 * The command line of the C tokenizer twins (PORT-2): the byte-level BPE of bcir_bpe.h, held to
 * bcir/frontends/models/tokenizer.py by bcir/tests/test_c_bpe_tokenizer.py, and the
 * SentencePiece-BPE of bcir_spm.h, held to bcir/frontends/models/spm.py by
 * bcir/tests/test_c_spm_tokenizer.py:
 *
 *   bcir-tokenize [--max-bytes N] [--format bpe|spm] info TOKENIZER     the model's digest and counts
 *   bcir-tokenize [--max-bytes N] [--format bpe|spm] encode TOKENIZER [TEXT] [--no-dummy-prefix]
 *                                                     UTF-8 text -> ids, one line, space-separated
 *   bcir-tokenize [--max-bytes N] [--format bpe|spm] decode TOKENIZER [IDS]  ids -> the decoded octets
 *   bcir-tokenize limits                              the BPE reader's limits profile, a field a line
 *
 * --format bpe (the default) reads a tokenizer.json, --format spm a tokenizer.model;
 * --no-dummy-prefix is SentencePiece's encode with add_dummy_prefix=False. TEXT and IDS are files,
 * standard input when absent or "-"; IDS holds ASCII decimal ids separated by ASCII white space
 * (space, tab, LF, CR). --max-bytes bounds the tokenizer file as the loaders' max_bytes does, from 1
 * to 268435456 (the default); the text and the ids are bounded by that limit too. Output is octets
 * on every host ("\n", never "\r\n"). A refusal prints `bcir-tokenize: <reason>` on standard
 * error, nothing on standard output, and exits 2. Arguments are read in their full spellings,
 * numbers as ASCII decimal digits.
 *===----------------------------------------------------------------------===*/
#include "bcir_bpe.h"
#include "bcir_host_alloc.h"
#include "bcir_spm.h"

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
    "usage: bcir-tokenize [--max-bytes N] [--format bpe|spm] {info,encode,decode} TOKENIZER [INPUT]\n"
    "                     [--no-dummy-prefix]\n"
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

/* One tokenizer of either format. */
typedef struct {
  int spm;
  bcir_bpe *bpe;
  bcir_spm *sp;
} tokenizer_t;

static const char *spm_law_text(bcir_spm_law law) {
  switch (law) {
  case BCIR_SPM_LAW_VARINT: return "malformed tokenizer.model: a protobuf varint is truncated or exceeds uint64";
  case BCIR_SPM_LAW_FIELD: return "malformed tokenizer.model: a protobuf field is malformed or truncated";
  case BCIR_SPM_LAW_PIECE_FIELDS: return "malformed tokenizer.model: a piece lacks its text or repeats a field";
  case BCIR_SPM_LAW_PIECE_TEXT: return "malformed tokenizer.model: a piece text is not UTF-8";
  case BCIR_SPM_LAW_SCORE: return "malformed tokenizer.model: a piece has a non-finite score";
  case BCIR_SPM_LAW_TYPE: return "malformed tokenizer.model: a piece has an unknown type";
  case BCIR_SPM_LAW_NO_PIECES: return "no SentencePiece pieces (not a tokenizer.model?)";
  case BCIR_SPM_LAW_DUPLICATE_TEXT: return "two SentencePiece pieces have one text";
  case BCIR_SPM_LAW_BYTE_PIECE: return "a byte piece is not <0xHH>, or names a byte twice";
  case BCIR_SPM_LAW_UNKNOWN_PIECE: return "tokenizer.model needs exactly one unknown piece";
  case BCIR_SPM_LAW_BYTE_ALPHABET: return "partial byte fallback alphabet";
  case BCIR_SPM_LAW_NONE:
  default: return "malformed tokenizer.model";
  }
}

static int load(tokenizer_t *t, const uint8_t *data, size_t len) {
  if (!t->spm) {
    bcir_bpe_diag diag;
    return bcir_bpe_load(data, len, NULL, &t->bpe, &diag) == BCIR_BPE_OK ? 0 : refuse_load(&diag);
  }
  bcir_spm_diag diag;
  if (bcir_spm_load(data, len, NULL, &t->sp, &diag) == BCIR_SPM_OK) return 0;
  if (diag.status == BCIR_SPM_ERR_NOMEM) return refuse("out of memory loading the tokenizer", NULL);
  if (diag.offset == BCIR_SPM_NO_OFFSET) return refuse(spm_law_text(diag.law), NULL);
  char detail[64];
  snprintf(detail, sizeof detail, " (at octet %zu)", diag.offset);
  return refuse(spm_law_text(diag.law), detail);
}

static void release(tokenizer_t *t) {
  bcir_bpe_free(t->bpe);
  bcir_spm_free(t->sp);
}

static void print_info(const tokenizer_t *t) {
  static const char hex[] = "0123456789abcdef";
  uint8_t digest[32];
  char text[65];
  if (t->spm) bcir_spm_digest(t->sp, digest);
  else bcir_bpe_digest(t->bpe, digest);
  for (unsigned i = 0; i < 32u; i++) {
    text[2u * i] = hex[digest[i] >> 4];
    text[2u * i + 1u] = hex[digest[i] & 15u];
  }
  text[64] = '\0';
  if (t->spm)
    printf("sha256 %s\npieces %zu\nunk %u\nbytes %u\nspecials %zu\n", text, bcir_spm_piece_count(t->sp),
           (unsigned)bcir_spm_unk_id(t->sp), bcir_spm_has_byte_fallback(t->sp) ? 256u : 0u,
           bcir_spm_special_count(t->sp));
  else
    printf("sha256 %s\nvocab %zu\nmerges %zu\nspecials %zu\n", text, bcir_bpe_vocab_size(t->bpe),
           bcir_bpe_merge_count(t->bpe), bcir_bpe_special_count(t->bpe));
}

/* 0 encoded, 1 not UTF-8, 2 out of memory, 3 anything else. */
static int encode_ids(const tokenizer_t *t, const uint8_t *text, size_t len, int prefix, uint32_t *ids, size_t cap,
                      size_t *count) {
  if (!t->spm) {
    switch (bcir_bpe_encode(t->bpe, text, len, ids, cap, count)) {
    case BCIR_BPE_OK: return 0;
    case BCIR_BPE_ERR_NOT_UTF8: return 1;
    case BCIR_BPE_ERR_NOMEM: return 2;
    default: return 3;
    }
  }
  switch (bcir_spm_encode(t->sp, text, len, prefix, ids, cap, count)) {
  case BCIR_SPM_OK: return 0;
  case BCIR_SPM_ERR_NOT_UTF8: return 1;
  case BCIR_SPM_ERR_NOMEM: return 2;
  default: return 3;
  }
}

static int encode(const tokenizer_t *t, const uint8_t *text, size_t len, int prefix) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t count = 0, cap = 0, bytes;
  uint32_t *ids;
  /* a BPE text of len octets yields at most len ids, a SentencePiece one at most 3 * len + 3 */
  if (t->spm ? !bcir_size_mul(len, 3u, &cap) || !bcir_size_add(cap, 3u, &cap) : !bcir_size_add(len, 0u, &cap))
    return refuse("the text is too long", NULL);
  if (!bcir_size_mul(cap ? cap : 1u, sizeof(uint32_t), &bytes) ||
      !(ids = (uint32_t *)bcir_host_allocate(&allocator, bytes)))
    return refuse("out of memory encoding the text", NULL);
  int rc = encode_ids(t, text, len, prefix, ids, cap, &count);
  if (rc == 0) {
    for (size_t i = 0; i < count; i++) printf(i ? " %u" : "%u", (unsigned)ids[i]);
    putchar('\n');
  }
  bcir_host_deallocate(&allocator, ids);
  switch (rc) {
  case 0: return 0;
  case 1: return refuse("the text is not UTF-8", NULL);
  case 2: return refuse("out of memory encoding the text", NULL);
  default: return refuse("cannot encode the text", NULL);
  }
}

/* 0 decoded (or measured: NOSPACE is 4), 1 unknown id, 2 not UTF-8, 3 anything else. */
static int decode_ids(const tokenizer_t *t, const uint32_t *ids, size_t n, uint8_t *out, size_t cap,
                      size_t *written) {
  if (!t->spm) {
    switch (bcir_bpe_decode(t->bpe, ids, n, out, cap, written)) {
    case BCIR_BPE_OK: return 0;
    case BCIR_BPE_ERR_NOSPACE: return 4;
    case BCIR_BPE_ERR_UNKNOWN_ID: return 1;
    case BCIR_BPE_ERR_NOT_UTF8: return 2;
    default: return 3;
    }
  }
  switch (bcir_spm_decode(t->sp, ids, n, out, cap, written)) {
  case BCIR_SPM_OK: return 0;
  case BCIR_SPM_ERR_NOSPACE: return 4;
  case BCIR_SPM_ERR_UNKNOWN_ID: return 1;
  default: return 3;
  }
}

static int is_space(uint8_t c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static int decode(const tokenizer_t *t, const uint8_t *in, size_t len) {
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
  int rc = decode_ids(t, ids, n, NULL, 0, &written);
  if (rc == 4) {
    if (!(out = (uint8_t *)bcir_host_allocate(&allocator, written))) {
      bcir_host_deallocate(&allocator, ids);
      return refuse("out of memory decoding", NULL);
    }
    rc = decode_ids(t, ids, n, out, written, &written);
    if (rc == 0 && fwrite(out, 1u, written, stdout) != written) rc = 5;
    bcir_host_deallocate(&allocator, out);
  }
  bcir_host_deallocate(&allocator, ids);
  switch (rc) {
  case 0: return 0;
  case 1: return refuse("unknown tokenizer id", NULL);
  case 2: return refuse("the decoded bytes are not UTF-8", NULL);
  case 5: return refuse("cannot write standard output", NULL);
  default: return refuse("cannot decode the ids", NULL);
  }
}

int main(int argc, char **argv) {
  uint64_t max_bytes = TOKENIZE_MAX_BYTES;
  const char *value = NULL, *path = NULL, *input = NULL;
  int i = 1, m, rc, prefix = 1;
  tokenizer_t t;
  memset(&t, 0, sizeof t);

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
  while (i < argc) {
    if ((m = option(argc, argv, &i, "--max-bytes", &value)) != 0) {
      if (m < 0 || parse_count(value, strlen(value), UINT64_MAX, &max_bytes) <= 0)
        return usage_error("--max-bytes takes a number");
    } else if ((m = option(argc, argv, &i, "--format", &value)) != 0) {
      if (m < 0 || (strcmp(value, "bpe") != 0 && strcmp(value, "spm") != 0))
        return usage_error("--format is bpe or spm");
      t.spm = strcmp(value, "spm") == 0;
    } else {
      break;
    }
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
    if (enc && t.spm && strcmp(argv[i], "--no-dummy-prefix") == 0) {
      prefix = 0;
      continue;
    }
    if (argv[i][0] == '-' && argv[i][1] != '\0') return usage_error("unknown option");
    if (!path) path = argv[i];
    else if (!info && !input) input = argv[i];
    else return usage_error("too many arguments");
  }
  if (!path) return usage_error("a TOKENIZER is required");
  if (max_bytes < 1u || max_bytes > TOKENIZE_MAX_BYTES) return refuse("--max-bytes must be in [1, 268435456]", NULL);

  uint8_t *model = NULL, *data = NULL;
  size_t model_len = 0, data_len = 0;
  if ((rc = read_path(path, (size_t)max_bytes, &model, &model_len)) != 0) return rc;
  bcir_host_allocator allocator = bcir_host_allocator_default();
  rc = load(&t, model, model_len);
  bcir_host_deallocate(&allocator, model);
  if (rc) return rc;
  if (info) {
    print_info(&t);
  } else if ((rc = read_path(input, TOKENIZE_MAX_BYTES, &data, &data_len)) == 0) {
    rc = enc ? encode(&t, data, data_len, prefix) : decode(&t, data, data_len);
    bcir_host_deallocate(&allocator, data);
  }
  release(&t);
  if (rc) return rc;
  return fflush(stdout) == 0 ? 0 : refuse("cannot write standard output", NULL);
}
