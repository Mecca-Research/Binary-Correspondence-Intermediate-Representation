/*===- bcir_pack_cli.c - bcir-pack: validated StreamPack listing and dump ----===
 *
 * The C twin of bcir/abi/streampack_tool.py's command line (PORT-1):
 *
 *   bcir-pack [--max-bytes N] dis FILE
 *   bcir-pack [--max-bytes N] hexdump FILE [--width N]
 *
 * Standard output is the oracle's format_listing / format_hexdump text as UTF-8 octets, line ends
 * "\n" on every host; a refusal prints `bcir-pack: <reason>` on standard error, nothing on standard
 * output, and exits 2, as the Python tool does. The input is read to at most --max-bytes + 1 octets
 * (default 64 MiB, at least 68) and refused past --max-bytes: the read is the bound, so a file that
 * grows while it is read is refused as one that was large to begin with.
 *
 * Arguments are read in their full spellings, as ASCII decimal numbers (`--max-bytes=N` and
 * `--width=N` too); argparse's abbreviations and int()'s wider language (signs, underscores,
 * non-ASCII digits) are not taken, so an argument the Python tool reads more loosely is a usage
 * error here.
 *===----------------------------------------------------------------------===*/
#include "bcir_host_alloc.h"
#include "bcir_sp_listing.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#endif

#define PACK_DEFAULT_MAX_BYTES ((size_t)64u * 1024u * 1024u)
#define PACK_MIN_BYTES 68u /* a header and a CRC trailer */

static const char USAGE[] =
    "usage: bcir-pack [--max-bytes N] {dis,hexdump} FILE [--width N]\n";

static int refuse(const char *reason, const char *detail) {
  fprintf(stderr, "bcir-pack: %s%s\n", reason, detail ? detail : "");
  return 2;
}

static int usage_error(const char *reason) {
  fputs(USAGE, stderr);
  return refuse(reason, NULL);
}

/* ASCII decimal digits only, at least one, no overflow past `max`. */
static int parse_count(const char *text, size_t max, size_t *out) {
  size_t value = 0;
  if (!text || !*text) return 0;
  for (const char *p = text; *p; p++) {
    if (*p < '0' || *p > '9') return 0;
    size_t digit = (size_t)(*p - '0');
    if (value > (max - digit) / 10u) return 0;
    value = value * 10u + digit;
  }
  *out = value;
  return 1;
}

/* `--name VALUE` or `--name=VALUE` at argv[*i]: 1 matched (and *value set, *i advanced past it),
 * 0 not this option, -1 the option without a value. */
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

static const char *status_text(bcir_status st) {
  switch (st) {
  case BCIR_ERR_TRUNCATED: return "truncated StreamPack";
  case BCIR_ERR_MAGIC: return "bad magic (expected 'BSPK')";
  case BCIR_ERR_VERSION: return "unsupported ABI version";
  case BCIR_ERR_CRC: return "CRC mismatch (corrupt StreamPack)";
  case BCIR_ERR_NOSPACE: return "output refused (write error)";
  case BCIR_ERR_LANE: return "unknown segment lane code";
  case BCIR_ERR_WIDTH: return "segment width must be a nonzero power of two";
  case BCIR_ERR_DISPATCH: return "unknown segment dispatch code";
  case BCIR_ERR_PROVENANCE: return "pipeline_depth or prefetch buffers out of range";
  case BCIR_ERR_OVERFLOW: return "hexdump width must be in [1, 64]";
  case BCIR_ERR_TRAILING: return "unexpected trailing body bytes";
  case BCIR_ERR_RESERVED: return "reserved StreamPack field must be zero";
  case BCIR_ERR_UTF8: return "StreamPack string is not valid UTF-8";
  case BCIR_ERR_GENERATION: return "malformed generation vector";
  default: return "malformed StreamPack";
  }
}

static FILE *open_input(const char *path) {
#if defined(_WIN32)
  FILE *file = NULL;
  return fopen_s(&file, path, "rb") == 0 ? file : NULL;
#else
  return fopen(path, "rb");
#endif
}

/* The whole input, at most `maximum` octets: *data owned by the caller. 0 read, 2 refused. */
static int read_bounded(const char *path, size_t maximum, uint8_t **data, size_t *len) {
  bcir_host_allocator allocator = bcir_host_allocator_default();
  size_t limit = maximum < SIZE_MAX ? maximum + 1u : SIZE_MAX;  /* one past the bound tells */
  size_t cap = limit < ((size_t)1u << 16) ? limit : ((size_t)1u << 16);
  size_t have = 0;
  FILE *file = open_input(path);
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

int main(int argc, char **argv) {
  size_t max_bytes = PACK_DEFAULT_MAX_BYTES;
  size_t width = 16u;
  const char *command = NULL, *path = NULL, *value = NULL;
  int i = 1, m;

  for (int k = 1; k < argc; k++) {
    if (strcmp(argv[k], "-h") == 0 || strcmp(argv[k], "--help") == 0) {
      fputs(USAGE, stdout);
      return 0;
    }
  }
  while (i < argc && (m = option(argc, argv, &i, "--max-bytes", &value)) != 0) {
    if (m < 0) return usage_error("--max-bytes takes a number");
    if (!parse_count(value, SIZE_MAX, &max_bytes)) return usage_error("--max-bytes takes a number");
  }
  if (i >= argc) return usage_error("a command is required (dis or hexdump)");
  command = argv[i++];
  int dump = strcmp(command, "hexdump") == 0;
  if (!dump && strcmp(command, "dis") != 0) return usage_error("the command is dis or hexdump");
  while (i < argc) {
    if (dump && (m = option(argc, argv, &i, "--width", &value)) != 0) {
      if (m < 0 || !parse_count(value, SIZE_MAX, &width))
        return usage_error("--width takes a number");
      continue;
    }
    if (argv[i][0] == '-' && argv[i][1] != '\0') return usage_error("unknown option");
    if (path) return usage_error("one FILE");
    path = argv[i++];
  }
  if (!path) return usage_error("a FILE is required");
  if (max_bytes < PACK_MIN_BYTES) return refuse("--max-bytes must be at least 68", NULL);
  if (dump && (width < 1u || width > BCIR_HEXDUMP_WIDTH_MAX))
    return refuse(status_text(BCIR_ERR_OVERFLOW), NULL);

  uint8_t *data = NULL;
  size_t len = 0;
  int rc = read_bounded(path, max_bytes, &data, &len);
  if (rc) return rc;

#if defined(_WIN32)
  (void)_setmode(_fileno(stdout), _O_BINARY);  /* "\n", never "\r\n": the listing's octets */
#endif
  bcir_host_allocator allocator = bcir_host_allocator_default();
  bcir_status st;
  if (dump) {
    st = bcir_sp_format_hexdump(data, len, (uint32_t)width, to_stdout, NULL);
  } else {
    size_t need = 0;
    st = bcir_sp_listing_scratch_bytes(data, len, &need);
    uint8_t *scratch = NULL;
    if (st == BCIR_OK && need) {
      scratch = (uint8_t *)bcir_host_allocate(&allocator, need);
      if (!scratch) {
        bcir_host_deallocate(&allocator, data);
        return refuse("out of memory for the trace index", NULL);
      }
    }
    if (st == BCIR_OK) st = bcir_sp_format_listing(data, len, scratch, need, to_stdout, NULL);
    bcir_host_deallocate(&allocator, scratch);
  }
  bcir_host_deallocate(&allocator, data);
  if (fflush(stdout) != 0 && st == BCIR_OK) st = BCIR_ERR_NOSPACE;
  if (st != BCIR_OK) return refuse(status_text(st), NULL);
  return 0;
}
