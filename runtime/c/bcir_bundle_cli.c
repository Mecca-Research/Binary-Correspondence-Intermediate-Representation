/*===- bcir_bundle_cli.c - bcir-bundle: validated BCAB listing and selection ---===
 *
 * The C twin of bcir/abi/artifact_tool.py's command line (PORT-1, the BCAB half):
 *
 *   bcir-bundle [--max-bytes N] list FILE
 *   bcir-bundle [--max-bytes N] hexdump FILE [--width N]
 *   bcir-bundle [--max-bytes N] select FILE [SELECTOR...]
 *   bcir-bundle [--max-bytes N] extract FILE OUTPUT [SELECTOR...]
 *   bcir-bundle [--max-bytes N] dis FILE [SELECTOR...]
 *   bcir-bundle [--max-bytes N] mnemonics FILE [SELECTOR...]
 *
 *   SELECTOR: --id ID | --triple T | --architecture A | --os-abi O | --channel C | --feature F...
 *             | --kind K... | --format F... | --endianness neutral|little|big | --pointer-bits N
 *             | --machine N | --manifest-sha256 HEX | --cal-gen N | --allow-unattested | --allow-debug
 *
 * Standard output is the oracle's text as UTF-8 octets, "\n" line ends on every host; a refusal prints
 * `bcir-bundle: <reason>` on standard error, nothing on standard output, and exits 2. The bundle is
 * validated whole before anything is printed or written (bcir_ab_open), and selection is the reader's
 * own (bcir_ab_select) over the envelope the options name, which the oracle builds and checks the same
 * way: with --id, the envelope options are not consulted at all.
 *
 * Not in the twin, each refused with exit 2: `dis` and `mnemonics` of a payload the oracle hands to a
 * resident disassembler (objdump, llvm-dis, spirv-dis, javap -- every format but StreamPack and text),
 * and the ASN.1 projections (to-der, from-der, to-oer, from-oer). Arguments are read in their full
 * spellings, numbers as ASCII decimal digits and names as ASCII; argparse's abbreviations and int()'s
 * and str.upper()'s wider languages are not taken.
 *===----------------------------------------------------------------------===*/
#if !defined(_WIN32)
#define _XOPEN_SOURCE 700 /* mkstemp, fdopen, fileno and fsync under a strict -std */
#endif
#include "bcir_ab_listing.h"
#include "bcir_host_alloc.h"
#include "bcir_sha256.h"
#include "bcir_sp_listing.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif

#define BUNDLE_MAX_BYTES ((size_t)BCIR_AB_MAX_BYTES)
#define MAX_FEATURES 32u

static const char USAGE[] =
    "usage: bcir-bundle [--max-bytes N] {list,hexdump,select,extract,dis,mnemonics} FILE "
    "[OUTPUT] [--width N] [selector options]\n";

static int refuse(const char *reason, const char *detail) {
  fprintf(stderr, "bcir-bundle: %s%s\n", reason, detail ? detail : "");
  return 2;
}

static int usage_error(const char *reason) {
  fputs(USAGE, stderr);
  return refuse(reason, NULL);
}

/* ASCII decimal digits only, at least one, no overflow past `max`. */
static int parse_count(const char *text, uint64_t max, uint64_t *out) {
  uint64_t value = 0;
  if (!text || !*text) return 0;
  for (const char *p = text; *p; p++) {
    if (*p < '0' || *p > '9') return 0;
    uint64_t digit = (uint64_t)(*p - '0');
    if (value > (max - digit) / 10u) return 0;
    value = value * 10u + digit;
  }
  *out = value;
  return 1;
}

/* `--name VALUE` or `--name=VALUE` at argv[*i]: 1 matched, 0 not this option, -1 no value. */
static int option(int argc, char **argv, int *i, const char *name, const char **value) {
  size_t n = strlen(name);
  const char *arg = argv[*i];
  if (strncmp(arg, name, n) != 0) return 0;
  if (arg[n] == '=') { *value = arg + n + 1; (*i)++; return 1; }
  if (arg[n] != '\0') return 0;
  if (*i + 1 >= argc) return -1;
  *value = argv[*i + 1];
  *i += 2;
  return 1;
}

static const char *status_text(bcir_ab_status st) {
  switch (st) {
  case BCIR_AB_ERR_INVALID: return "invalid request";
  case BCIR_AB_ERR_SIZE: return "bundle size out of range";
  case BCIR_AB_ERR_MAGIC: return "bad magic (expected 'BCAB')";
  case BCIR_AB_ERR_VERSION: return "unsupported BCAB version";
  case BCIR_AB_ERR_LAYOUT: return "non-canonical bundle layout";
  case BCIR_AB_ERR_METADATA: return "malformed directory metadata";
  case BCIR_AB_ERR_CRC: return "bundle CRC mismatch";
  case BCIR_AB_ERR_SHA256: return "bundle SHA-256 mismatch";
  case BCIR_AB_ERR_PAYLOAD: return "malformed payload";
  case BCIR_AB_ERR_INCOMPATIBLE: return "no compatible artifact variant";
  case BCIR_AB_ERR_NOT_FOUND: return "no such variant";
  case BCIR_AB_ERR_OUTPUT: return "output refused (write error)";
  default: return "malformed bundle";
  }
}

/* --- the selector options: the oracle's CompatibilityEnvelope, field for field --------------- */

typedef struct {
  const char *id;
  const char *triple, *architecture, *os_abi, *channel, *manifest;
  const char *features[MAX_FEATURES];
  unsigned n_features;
  uint64_t kinds, formats, pointer_bits, machine, cal_gen;
  int has_cal_gen, endianness, allow_unattested, allow_debug, any_kind_name_bad, any_format_name_bad;
} selector_t;

static const char *const KIND_NAME[26] = {
    "", "STREAM_PACK", "ELF_OBJECT", "ELF_SHARED", "COFF_OBJECT", "MACHO_OBJECT", "ARCHIVE",
    "WASM", "LLVM_BITCODE", "LLVM_IR", "PTX", "CUBIN", "SPIRV", "JVM_CLASS", "CIL", "C_SOURCE",
    "CPP_SOURCE", "SYCL_SOURCE", "ASSEMBLY", "ELF_EXECUTABLE", "PE_EXECUTABLE", "PE_SHARED",
    "MACHO_EXECUTABLE", "MACHO_SHARED", "RAW_BINARY", "EXECUTION_PLAN"};
static const char *const FORMAT_NAME[14] = {
    "NONE", "STREAM_PACK", "ELF", "COFF", "MACHO", "ARCHIVE", "WASM", "LLVM_BITCODE", "TEXT",
    "SPIRV", "JVM_CLASS", "PE", "RAW", "EXECUTION_PLAN"};

/* `_enum_set`: value.upper().replace("-", "_") names a member, or the name is refused. */
static int enum_bit(const char *const *names, unsigned count, unsigned first, const char *value,
                    uint64_t *mask) {
  char name[32];
  size_t n = 0;
  for (; value[n]; n++) {
    char c = value[n];
    if (n + 1u >= sizeof name || (unsigned char)c >= 0x80u) return 0;
    name[n] = c == '-' ? '_' : (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
  }
  name[n] = '\0';
  for (unsigned v = first; v < count; v++)
    if (strcmp(names[v], name) == 0) { *mask |= UINT64_C(1) << v; return 1; }
  return 0;
}

static int feature_name(const char *f) {
  if (!*f) return 0;
  for (const char *p = f; *p; p++) {
    char c = *p;
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
          c == '_' || c == '.' || c == '+' || c == ':' || c == '-'))
      return 0;
  }
  return 1;
}

static int by_name(const void *a, const void *b) {
  return strcmp(*(const char *const *)a, *(const char *const *)b);
}

/* The envelope the options name, checked as CompatibilityEnvelope checks it; 0 refused. */
static int build_envelope(selector_t *s, bcir_ab_envelope *e, char csv[BCIR_AB_FEATURE_CSV_MAX + 1],
                          uint8_t manifest[32], const char **why) {
  memset(e, 0, sizeof *e);
  *why = "malformed compatibility envelope";
  /* The strings' own law (canonical printable ASCII, no edge space, within the field) is the
   * reader's: bcir_ab_select refuses an envelope that breaks it. What is checked here is what the
   * reader cannot see -- a feature name with a comma in it, an empty one, the set's canonical order,
   * and values wider than the fields they are narrowed into. */
  if (s->any_kind_name_bad || s->any_format_name_bad) { *why = "unknown kind or format"; return 0; }
  /* frozenset(features), sorted: each a feature name, joined within the field */
  qsort(s->features, s->n_features, sizeof s->features[0], by_name);
  size_t n = 0;
  for (unsigned i = 0; i < s->n_features; i++) {
    if (!feature_name(s->features[i])) return 0;
    if (i && strcmp(s->features[i], s->features[i - 1]) == 0) continue;
    size_t len = strlen(s->features[i]);
    if (n + (n ? 1u : 0u) + len > BCIR_AB_FEATURE_CSV_MAX) return 0;
    if (n) csv[n++] = ',';
    memcpy(csv + n, s->features[i], len);
    n += len;
  }
  csv[n] = '\0';
  if (s->pointer_bits != 0 && s->pointer_bits != 32 && s->pointer_bits != 64) return 0;
  if (s->machine > UINT32_MAX) return 0;
  int digest = 0;
  if (*s->manifest) {
    int zero = 1;
    if (strlen(s->manifest) != 64u) return 0;
    for (unsigned i = 0; i < 64u; i++) {
      char c = s->manifest[i];
      unsigned v = (c >= '0' && c <= '9') ? (unsigned)(c - '0')
                   : (c >= 'a' && c <= 'f') ? (unsigned)(c - 'a' + 10) : 16u;
      if (v > 15u) return 0;
      if (i % 2u == 0u) manifest[i / 2u] = (uint8_t)(v << 4); else manifest[i / 2u] |= (uint8_t)v;
      if (v) zero = 0;
    }
    if (zero) return 0;  /* the all-zero absent-value sentinel */
    digest = 1;
  }
  e->triple = s->triple; e->architecture = s->architecture; e->os_abi = s->os_abi;
  e->channel = s->channel; e->features = csv;
  e->accepted_kind_mask = s->kinds; e->accepted_format_mask = s->formats;
  e->endianness = (uint8_t)s->endianness; e->pointer_bits = (uint8_t)s->pointer_bits;
  e->machine = (uint32_t)s->machine; e->target_manifest_sha256 = digest ? manifest : NULL;
  e->cal_gen = s->cal_gen; e->has_cal_gen = (uint8_t)s->has_cal_gen;
  e->require_r12 = (uint8_t)!s->allow_unattested; e->allow_debug = (uint8_t)s->allow_debug;
  return 1;
}

/* One selector option at argv[*i]: 1 taken, 0 not a selector option, -1 malformed. */
static int selector_option(int argc, char **argv, int *i, selector_t *s) {
  static const struct { const char *name; size_t offset; } text[] = {
      {"--id", offsetof(selector_t, id)}, {"--triple", offsetof(selector_t, triple)},
      {"--architecture", offsetof(selector_t, architecture)},
      {"--os-abi", offsetof(selector_t, os_abi)}, {"--channel", offsetof(selector_t, channel)},
      {"--manifest-sha256", offsetof(selector_t, manifest)}};
  const char *value;
  int m;
  for (unsigned k = 0; k < sizeof text / sizeof text[0]; k++) {
    if ((m = option(argc, argv, i, text[k].name, &value)) != 0) {
      if (m < 0) return -1;
      *(const char **)((char *)s + text[k].offset) = value;
      return 1;
    }
  }
  if ((m = option(argc, argv, i, "--feature", &value)) != 0) {
    if (m < 0 || s->n_features >= MAX_FEATURES) return -1;
    s->features[s->n_features++] = value;
    return 1;
  }
  if ((m = option(argc, argv, i, "--kind", &value)) != 0) {
    if (m < 0) return -1;
    if (!enum_bit(KIND_NAME, 26u, 1u, value, &s->kinds)) s->any_kind_name_bad = 1;
    return 1;
  }
  if ((m = option(argc, argv, i, "--format", &value)) != 0) {
    if (m < 0) return -1;
    if (!enum_bit(FORMAT_NAME, 14u, 0u, value, &s->formats)) s->any_format_name_bad = 1;
    return 1;
  }
  if ((m = option(argc, argv, i, "--endianness", &value)) != 0) {
    if (m < 0) return -1;
    if (strcmp(value, "neutral") == 0) s->endianness = BCIR_AB_ENDIAN_NEUTRAL;
    else if (strcmp(value, "little") == 0) s->endianness = BCIR_AB_ENDIAN_LITTLE;
    else if (strcmp(value, "big") == 0) s->endianness = BCIR_AB_ENDIAN_BIG;
    else return -1;  /* argparse `choices` */
    return 1;
  }
  if ((m = option(argc, argv, i, "--pointer-bits", &value)) != 0)
    return m < 0 || !parse_count(value, UINT64_MAX, &s->pointer_bits) ? -1 : 1;
  if ((m = option(argc, argv, i, "--machine", &value)) != 0)
    return m < 0 || !parse_count(value, UINT64_MAX, &s->machine) ? -1 : 1;
  if ((m = option(argc, argv, i, "--cal-gen", &value)) != 0) {
    if (m < 0 || !parse_count(value, UINT64_MAX, &s->cal_gen)) return -1;
    s->has_cal_gen = 1;
    return 1;
  }
  if (strcmp(argv[*i], "--allow-unattested") == 0) { s->allow_unattested = 1; (*i)++; return 1; }
  if (strcmp(argv[*i], "--allow-debug") == 0) { s->allow_debug = 1; (*i)++; return 1; }
  return 0;
}

/* --- input and output ------------------------------------------------------------------------ */

static FILE *open_file(const char *path, const char *mode) {
#if defined(_WIN32)
  FILE *file = NULL;
  return fopen_s(&file, path, mode) == 0 ? file : NULL;
#else
  return fopen(path, mode);
#endif
}

/* The whole input, at most `maximum` octets: *data owned by the caller. 0 read, 2 refused. */
static int read_bounded(const char *path, size_t maximum, uint8_t **data, size_t *len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t limit = maximum < SIZE_MAX ? maximum + 1u : SIZE_MAX;  /* one past the bound tells */
  size_t cap = limit < ((size_t)1u << 16) ? limit : ((size_t)1u << 16);
  size_t have = 0;
  FILE *file = open_file(path, "rb");
  if (!file) return refuse("cannot open ", path);
  uint8_t *buf = (uint8_t *)bcir_host_allocate(&allocator, cap);
  if (!buf) { fclose(file); return refuse("out of memory reading ", path); }
  for (;;) {
    if (have == cap) {
      if (cap == limit) break;
      size_t grown = cap > limit / 2u ? limit : cap * 2u;
      void *next = buf;
      if (!bcir_host_realloc_array(&allocator, &next, cap, grown, 1u, 0)) {
        bcir_host_deallocate(&allocator, buf);
        fclose(file);
        return refuse("out of memory reading ", path);
      }
      buf = (uint8_t *)next;
      cap = grown;
    }
    size_t want = cap - have;
    size_t got = fread(buf + have, 1u, want, file);
    have += got;
    if (got < want) break;  /* end of file or an error (ferror below says which): read no further */
  }
  int failed = ferror(file);
  fclose(file);
  if (failed) {
    bcir_host_deallocate(&allocator, buf);
    return refuse("cannot read ", path);
  }
  if (have > maximum) {
    char detail[64];
    snprintf(detail, sizeof detail, "%zu", maximum);
    bcir_host_deallocate(&allocator, buf);
    return refuse("input exceeds --max-bytes=", detail);
  }
  *data = buf;
  *len = have;
  return 0;
}

static int to_stdout(void *ctx, const char *text, size_t len) {
  (void)ctx;
  return fwrite(text, 1u, len, stdout) != len;
}

/* `_atomic_write`: the payload in a temporary beside `path`, flushed, then renamed over it. */
static int atomic_write(const char *path, const uint8_t *payload, size_t len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t n = strlen(path), bytes;
  if (!bcir_size_add(n, 16u, &bytes)) return 0;
  char *temporary = (char *)bcir_host_allocate(&allocator, bytes);
  if (!temporary) return 0;
  memcpy(temporary, path, n);
#if defined(_WIN32)
  memcpy(temporary + n, ".tmp", 5);
  FILE *file = open_file(temporary, "wb");
#else
  memcpy(temporary + n, ".XXXXXX", 8);
  int fd = mkstemp(temporary);
  FILE *file = fd >= 0 ? fdopen(fd, "wb") : NULL;
  if (fd >= 0 && !file) close(fd);
#endif
  int ok = file != NULL;
  if (ok) ok = fwrite(payload, 1u, len, file) == len && fflush(file) == 0;
#if !defined(_WIN32)
  if (ok) ok = fsync(fileno(file)) == 0;
#endif
  if (file && fclose(file) != 0) ok = 0;
#if defined(_WIN32)
  if (ok) (void)remove(path);
#endif
  if (ok) ok = rename(temporary, path) == 0;
  if (!ok && file) (void)remove(temporary);
  bcir_host_deallocate(&allocator, temporary);
  return ok;
}

static void sha256_hex(const uint8_t *data, size_t len, char out[65]) {
  static const char hex[] = "0123456789abcdef";
  uint8_t digest[32];
  bcir_sha256 sha;
  bcir_sha256_init(&sha);
  bcir_sha256_update(&sha, data, len);
  bcir_sha256_final(&sha, digest);
  for (unsigned i = 0; i < 32u; i++) {
    out[2u * i] = hex[digest[i] >> 4];
    out[2u * i + 1u] = hex[digest[i] & 0xFu];
  }
  out[64] = '\0';
}

/* dis / mnemonics of one validated payload: StreamPack and text only (see the file comment). */
static int payload_view(const bcir_ab_entry *e, int mnemonics) {
  bcir_ab_status st;
  if (e->kind == BCIR_AB_STREAM_PACK) {
    if (mnemonics) {
      st = bcir_ab_pack_mnemonics(e->payload, (size_t)e->payload_size, to_stdout, NULL);
    } else {
      size_t need = 0;
      if (bcir_sp_listing_scratch_bytes(e->payload, (size_t)e->payload_size, &need) != BCIR_OK)
        return refuse("malformed StreamPack payload", NULL);
      bcir_host_allocator allocator = bcir_host_allocator_default();
      uint8_t *scratch = need ? (uint8_t *)bcir_host_allocate(&allocator, need) : NULL;
      if (need && !scratch) return refuse("out of memory for the trace index", NULL);
      bcir_status sp = bcir_sp_format_listing(e->payload, (size_t)e->payload_size, scratch, need,
                                              to_stdout, NULL);
      bcir_host_deallocate(&allocator, scratch);
      st = sp == BCIR_OK ? BCIR_AB_OK : sp == BCIR_ERR_NOSPACE ? BCIR_AB_ERR_OUTPUT
                                                                 : BCIR_AB_ERR_PAYLOAD;
    }
  } else if (e->format == BCIR_AB_FMT_TEXT) {
    st = mnemonics ? bcir_ab_text_mnemonics(e->payload, (size_t)e->payload_size, to_stdout, NULL)
                   : bcir_ab_text_listing(e->payload, (size_t)e->payload_size, to_stdout, NULL);
  } else {
    return refuse("this payload's view delegates to a resident disassembler, which the C twin "
                  "does not run (use the Python bcir-bundle)", NULL);
  }
  return st == BCIR_AB_OK ? 0 : refuse(status_text(st), NULL);
}

int main(int argc, char **argv) {
  uint64_t max_bytes = BUNDLE_MAX_BYTES, width = 16u;
  const char *value = NULL, *path = NULL, *output = NULL;
  int i = 1, m;
  selector_t s;
  memset(&s, 0, sizeof s);
  s.id = s.triple = s.architecture = s.os_abi = s.channel = s.manifest = "";

  for (int k = 1; k < argc; k++) {
    if (strcmp(argv[k], "-h") == 0 || strcmp(argv[k], "--help") == 0) {
      fputs(USAGE, stdout);
      return 0;
    }
  }
  while (i < argc && (m = option(argc, argv, &i, "--max-bytes", &value)) != 0) {
    if (m < 0 || !parse_count(value, UINT64_MAX, &max_bytes))
      return usage_error("--max-bytes takes a number");
  }
  if (i >= argc) return usage_error("a command is required");
  const char *command = argv[i++];
  int list = !strcmp(command, "list"), dump = !strcmp(command, "hexdump");
  int select = !strcmp(command, "select"), extract = !strcmp(command, "extract");
  int dis = !strcmp(command, "dis"), mnemonics = !strcmp(command, "mnemonics");
  if (!strcmp(command, "to-der") || !strcmp(command, "from-der") || !strcmp(command, "to-oer") ||
      !strcmp(command, "from-oer"))
    return refuse("the ASN.1 projections are the Python bcir-bundle's, not the C twin's", NULL);
  if (!(list || dump || select || extract || dis || mnemonics))
    return usage_error("the command is list, hexdump, select, extract, dis or mnemonics");
  int selecting = select || extract || dis || mnemonics;
  while (i < argc) {
    if (dump && (m = option(argc, argv, &i, "--width", &value)) != 0) {
      if (m < 0 || !parse_count(value, UINT64_MAX, &width)) return usage_error("--width takes a number");
      continue;
    }
    if (selecting && (m = selector_option(argc, argv, &i, &s)) != 0) {
      if (m < 0) return usage_error("a selector option takes a well-formed value");
      continue;
    }
    if (argv[i][0] == '-' && argv[i][1] != '\0') return usage_error("unknown option");
    if (!path) path = argv[i++];
    else if (extract && !output) output = argv[i++];
    else return usage_error("too many arguments");
  }
  if (!path) return usage_error("a FILE is required");
  if (extract && !output) return usage_error("extract needs an OUTPUT");
  if (max_bytes < BCIR_AB_HEADER_SIZE || max_bytes > BUNDLE_MAX_BYTES)
    return refuse("--max-bytes must be in [128, 1073741824]", NULL);

  uint8_t *data = NULL;
  size_t len = 0;
  int rc = read_bounded(path, (size_t)max_bytes, &data, &len);
  if (rc) return rc;
#if defined(_WIN32)
  (void)_setmode(_fileno(stdout), _O_BINARY);  /* "\n", never "\r\n" */
#endif
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_ab_status st = BCIR_AB_OK;
  rc = 0;
  if (list) {
    st = bcir_ab_format_listing(data, len, to_stdout, NULL);
  } else if (dump) {
    st = width >= 1u && width <= 64u ? bcir_ab_format_hexdump(data, len, (uint32_t)width, to_stdout, NULL)
                                     : BCIR_AB_ERR_INVALID;
    if (st == BCIR_AB_ERR_INVALID && !(width >= 1u && width <= 64u)) {
      bcir_host_deallocate(&allocator, data);
      return refuse("hexdump width must be in [1, 64]", NULL);
    }
  } else {
    bcir_ab_view view;
    st = bcir_ab_open(data, len, &view);
    if (st == BCIR_AB_OK) {
      bcir_ab_envelope envelope;
      char csv[BCIR_AB_FEATURE_CSV_MAX + 1];
      uint8_t manifest[32];
      const char *why = NULL;
      uint32_t index = BCIR_AB_NO_INDEX;
      int by_id = s.id[0] != '\0';
      if (!by_id && !build_envelope(&s, &envelope, csv, manifest, &why)) {
        bcir_host_deallocate(&allocator, data);
        return refuse(why, NULL);
      }
      st = bcir_ab_select(&view, by_id ? NULL : &envelope, by_id ? s.id : NULL, &index);
      bcir_ab_entry e;
      if (st == BCIR_AB_OK) st = bcir_ab_get(&view, index, &e);
      if (st == BCIR_AB_OK) {
        if (select) {
          printf("%s\n", e.variant_id);
        } else if (extract) {
          if (!atomic_write(output, e.payload, (size_t)e.payload_size)) {
            bcir_host_deallocate(&allocator, data);
            return refuse("cannot write ", output);
          }
          char hex[65];
          sha256_hex(e.payload, (size_t)e.payload_size, hex);
          printf("%s %llu bytes %s\n", e.variant_id, (unsigned long long)e.payload_size, hex);
        } else {
          rc = payload_view(&e, mnemonics);
        }
      }
    }
  }
  bcir_host_deallocate(&allocator, data);
  if (rc) return rc;
  if (fflush(stdout) != 0 && st == BCIR_AB_OK) st = BCIR_AB_ERR_OUTPUT;
  if (st != BCIR_AB_OK) return refuse(status_text(st), NULL);
  return 0;
}
