/*===- bcir_tokload_tool.c - bcir-tokload: token shards from files, and the batches over them ------------===
 *
 *   bcir-tokload pack VOCAB WIDTH IN OUT
 *       the decimal token ids of IN (whitespace between) as the BTKS shard OUT, its digest on the hardware path
 *       when the CPU has one. Prints `ok COUNT DIGEST`.
 *   bcir-tokload info SHARD...
 *       per shard, opened whole: `ok VOCAB WIDTH COUNT DIGEST`, or `ERR CODE` (bcir_tokload.h's codes).
 *   bcir-tokload batch K0 K1 CONTEXT SIZE SHUFFLE FIRST COUNT SHARD...
 *       batches FIRST .. FIRST + COUNT - 1 of the loader over the shards, one line each:
 *       `b INDEX I,I,...;I,... | T,T,...;T,...` (rows ;-separated, inputs then targets), or `ERR CODE`.
 *
 * Shards are mapped read-only where the host has mmap and read through the hosted allocator where it has not,
 * or where the map is refused (an empty file); either way the loader reads them in place. A malformed command
 * line is the tool's (`TOOL ...`, exit 2); a refusal is a verdict (`ERR CODE`, exit 1). bcir/tests/test_c_tokload.py
 * holds every line to the oracle (bcir/frontends/models/token_shards.py).
 *===----------------------------------------------------------------------===*/
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define BCIR_TOKLOAD_MMAP 1
#endif

#include "bcir_digest_hw.h"
#include "bcir_host_alloc.h"
#include "bcir_tokload.h"

/* libc's allocator through the hosted interface, spelled as its fields: each function that allocates holds one */
#define BCIR_TOKLOAD_HOST {NULL, bcir_host_libc_allocate, bcir_host_libc_reallocate, bcir_host_libc_deallocate}

static int tool_fail(const char *why) {
  printf("TOOL %s\n", why);
  return 2;
}

/* a decimal, digits only, no more than `max`: the overflow checked before each step */
static int number(const char *s, uint64_t max, uint64_t *out) {
  if (!s || !*s) return 0;
  uint64_t v = 0;
  for (; *s; s++) {
    if (*s < '0' || *s > '9') return 0;
    unsigned d = (unsigned)(*s - '0');
    if (v > (UINT64_MAX - d) / 10u) return 0;
    v = v * 10u + d;
  }
  if (v > max) return 0;
  *out = v;
  return 1;
}

typedef struct mapped {
  uint8_t *bytes;
  size_t length;
  int is_mapped;
} mapped;

/* A file's octets: mapped read-only where the host maps, else (or where the map is refused: an empty file) read
 * through the hosted allocator. One function, so the reader's frames stay few. */
static int map_file(const char *path, mapped *m) {
  bcir_host_allocator host = BCIR_TOKLOAD_HOST;
#ifdef BCIR_TOKLOAD_MMAP
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 0;
  struct stat st;
  if (fstat(fd, &st) == 0 && st.st_size > 0 && (uintmax_t)st.st_size <= SIZE_MAX) {
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p != MAP_FAILED) {
      close(fd);
      m->bytes = (uint8_t *)p;
      m->length = (size_t)st.st_size;
      m->is_mapped = 1;
      return 1;
    }
  }
  close(fd);
#endif
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  size_t cap = 4096, len = 0;
  void *b = bcir_host_allocate(&host, cap);
  for (; b;) {
    size_t r = fread((char *)b + len, 1, cap - len, f);
    len += r;
    if (len < cap) break;
    if (cap > SIZE_MAX / 2 || !bcir_host_realloc_array(&host, &b, cap, cap * 2, 1, 0)) {
      bcir_host_deallocate(&host, b);
      b = NULL;
      break;
    }
    cap *= 2;
  }
  int failed = ferror(f);
  fclose(f);
  if (b && failed) {
    bcir_host_deallocate(&host, b);
    b = NULL;
  }
  m->bytes = (uint8_t *)b;
  m->length = len;
  m->is_mapped = 0;
  return b != NULL;
}

static void unmap_file(mapped *m) {
  bcir_host_allocator host = BCIR_TOKLOAD_HOST;
#ifdef BCIR_TOKLOAD_MMAP
  if (m->is_mapped) {
    munmap(m->bytes, m->length);
    return;
  }
#endif
  bcir_host_deallocate(&host, m->bytes);
}

static void put_hex(const uint8_t *d, size_t n) {
  for (size_t i = 0; i < n; i++) printf("%02x", d[i]);
}

static void put_row(const uint32_t *v, uint32_t rows, uint32_t width) {
  for (uint32_t r = 0; r < rows; r++)
    for (uint32_t k = 0; k < width; k++)
      printf("%s%u", k ? "," : (r ? ";" : ""), (unsigned)v[(size_t)r * width + k]);
}

static int pack(int argc, char **argv) {
  bcir_host_allocator host = BCIR_TOKLOAD_HOST;
  uint64_t vocab, width;
  if (argc != 6 || !number(argv[2], UINT32_MAX, &vocab) || !number(argv[3], 4, &width))
    return tool_fail("usage: pack VOCAB WIDTH IN OUT");
  if ((width != 2 && width != 4) || vocab == 0 || (width == 2 && vocab > 65536u))
    return tool_fail("a vocabulary its width cannot hold");
  mapped in;
  if (!map_file(argv[4], &in)) return tool_fail("cannot read IN");
  size_t cap = 1024, n = 0;
  uint8_t *out = (uint8_t *)bcir_host_allocate(&host, BCIR_TOKSHARD_HEADER + cap * width);
  int rc = 0;
  for (size_t i = 0; out && i < in.length && !rc;) {
    while (i < in.length && (in.bytes[i] == ' ' || in.bytes[i] == '\n' || in.bytes[i] == '\r' || in.bytes[i] == '\t'))
      i++;
    if (i == in.length) break;
    uint64_t v = 0;
    size_t digits = 0;
    while (i < in.length && in.bytes[i] >= '0' && in.bytes[i] <= '9' && digits < 11) {
      v = v * 10u + (uint64_t)(in.bytes[i++] - '0');
      digits++;
    }
    if (!digits || v >= vocab || (i < in.length && in.bytes[i] > ' ')) { rc = 1; break; }
    if (n == cap) {
      void *grow = out;
      if (cap > SIZE_MAX / 2 / width - BCIR_TOKSHARD_HEADER ||
          !bcir_host_realloc_array(&host, &grow, BCIR_TOKSHARD_HEADER + cap * width,
                                   BCIR_TOKSHARD_HEADER + cap * 2 * width, 1, 0)) { rc = 2; break; }
      out = (uint8_t *)grow;
      cap *= 2;
    }
    uint8_t *p = out + BCIR_TOKSHARD_HEADER + n * width;
    for (uint64_t b = 0; b < width; b++) p[b] = (uint8_t)(v >> (8 * b));
    n++;
  }
  unmap_file(&in);
  if (!out || rc == 2) { bcir_host_deallocate(&host, out); return tool_fail("out of memory"); }
  if (rc) { bcir_host_deallocate(&host, out); return tool_fail("IN is not decimal token ids below VOCAB"); }
  memset(out, 0, BCIR_TOKSHARD_HEADER);
  memcpy(out, "BTKS", 4);
  out[6] = (uint8_t)width;
  for (int b = 0; b < 4; b++) out[8 + b] = (uint8_t)(vocab >> (8 * b));
  for (int b = 0; b < 8; b++) out[16 + b] = (uint8_t)((uint64_t)n >> (8 * b));
  bcir_sha256_digest_hw(out + BCIR_TOKSHARD_HEADER, n * width, out + 32, bcir_digest_probe());
  FILE *f = fopen(argv[5], "wb");
  size_t total = BCIR_TOKSHARD_HEADER + n * width;
  int wrote = f && fwrite(out, 1, total, f) == total;
  if (f && fclose(f) != 0) wrote = 0;
  if (wrote) {
    printf("ok %zu ", n);
    put_hex(out + 32, 32);
    printf("\n");
  }
  bcir_host_deallocate(&host, out);
  return wrote ? 0 : tool_fail("cannot write OUT");
}

static int info(int argc, char **argv) {
  if (argc < 3) return tool_fail("usage: info SHARD...");
  unsigned features = bcir_digest_probe();
  int worst = 0;
  for (int k = 2; k < argc; k++) {
    mapped m;
    if (!map_file(argv[k], &m)) return tool_fail("cannot read a shard");
    bcir_tokshard s;
    int rc = bcir_tokshard_open(&s, m.bytes, m.length, features);
    if (rc) {
      printf("ERR %d\n", rc);
      worst = 1;
    } else {
      printf("ok %u %u %llu ", (unsigned)s.vocab, (unsigned)s.width, (unsigned long long)s.count);
      put_hex(s.digest, 32);
      printf("\n");
    }
    unmap_file(&m);
  }
  return worst;
}

static int batch(int argc, char **argv) {
  bcir_host_allocator host = BCIR_TOKLOAD_HOST;
  uint64_t k0, k1, context, size, shuffle, first, count;
  if (argc < 10 || !number(argv[2], UINT32_MAX, &k0) || !number(argv[3], UINT32_MAX, &k1) ||
      !number(argv[4], UINT32_MAX, &context) || !number(argv[5], UINT32_MAX, &size) ||
      !number(argv[6], 1, &shuffle) || !number(argv[7], UINT64_MAX, &first) || !number(argv[8], 4096, &count))
    return tool_fail("usage: batch K0 K1 CONTEXT SIZE SHUFFLE FIRST COUNT SHARD...");
  size_t n = (size_t)(argc - 9);
  unsigned features = bcir_digest_probe();
  mapped *maps = (mapped *)bcir_host_allocate_zeroed(&host, n * sizeof *maps);
  bcir_tokshard *shards = (bcir_tokshard *)bcir_host_allocate_zeroed(&host, n * sizeof *shards);
  uint32_t *perm = NULL, *in = NULL, *tg = NULL;
  int rc = 0, tool = 0;
  size_t opened = 0;
  if (!maps || !shards) tool = 1;
  for (; !tool && !rc && opened < n; opened++) {
    if (!map_file(argv[9 + opened], &maps[opened])) { tool = 2; break; }
    rc = bcir_tokshard_open(&shards[opened], maps[opened].bytes, maps[opened].length, features);
  }
  bcir_tokload l;
  const uint32_t key[2] = {(uint32_t)k0, (uint32_t)k1};
  if (!tool && !rc) {
    uint64_t windows = 0;
    for (size_t s = 0; s < n; s++) windows += bcir_tokshard_windows(shards[s].count, (uint32_t)context);
    /* the permutation's workspace: what init will ask, capped so a refusal stays the loader's */
    size_t words = windows > 0 && windows <= UINT32_MAX ? (size_t)windows : 1u;
    perm = shuffle ? (uint32_t *)bcir_host_allocate(&host, words * sizeof *perm) : NULL;
    if (shuffle && !perm) tool = 1;
    else rc = bcir_tokload_init(&l, shards, n, (uint32_t)context, key, (int)shuffle, perm, words);
  }
  size_t cells = 0;
  if (!tool && !rc && count) {
    if (!bcir_size_mul((size_t)size, (size_t)context, &cells) || cells > ((size_t)1 << 26)) tool = 3;
    else {
      in = (uint32_t *)bcir_host_allocate(&host, cells * sizeof *in);
      tg = (uint32_t *)bcir_host_allocate(&host, cells * sizeof *tg);
      if (!in || !tg) tool = 1;
    }
  }
  for (uint64_t b = 0; !tool && !rc && b < count; b++) {
    if (first > UINT64_MAX - b) { rc = BCIR_TOKLOAD_ERR_ARG; break; }
    rc = bcir_tokload_batch(&l, first + b, (uint32_t)size, in, tg);
    if (rc) break;
    printf("b %llu ", (unsigned long long)(first + b));
    put_row(in, (uint32_t)size, (uint32_t)context);
    printf(" | ");
    put_row(tg, (uint32_t)size, (uint32_t)context);
    printf("\n");
  }
  if (!tool && rc) printf("ERR %d\n", rc);
  bcir_host_deallocate(&host, in);
  bcir_host_deallocate(&host, tg);
  bcir_host_deallocate(&host, perm);
  for (size_t s = 0; maps && s < opened && s < n; s++)
    if (maps[s].bytes) unmap_file(&maps[s]);
  bcir_host_deallocate(&host, shards);
  bcir_host_deallocate(&host, maps);
  if (tool) return tool_fail(tool == 1 ? "out of memory" : tool == 2 ? "cannot read a shard" : "a batch too large");
  return rc ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "pack")) return pack(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "info")) return info(argc, argv);
  if (argc >= 2 && !strcmp(argv[1], "batch")) return batch(argc, argv);
  return tool_fail("usage: bcir-tokload pack|info|batch ...");
}
