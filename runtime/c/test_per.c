/*===- test_per.c - differential driver for the X.691 PER C twin ------------===
 *
 * Reads one operation per line from stdin and prints the C twin's answer, so
 * bcir/tests/test_c_per.py can drive the SAME campaign through both rails and compare.
 * The driver owns the line parsing and stdio; bcir_per.c stays freestanding.
 *
 *   constrained <lb> <ub> <variant> <hex>
 *   semi <lb> <variant> <hex>
 *   unconstrained <variant> <hex>
 *   small <variant> <hex>
 *   smalllen <variant> <hex>
 *   length <lb> <ub> <has_ub> <variant> <hex>
 *   bits <width> <variant> <hex>
 *
 * <variant> is 0 for UNALIGNED, 1 for ALIGNED. Output is "OK <value> <bitpos>" or
 * "ERR <status>", plus " MORE" when a length determinant named a fragment.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "bcir_per.h"

#define MAX_LINE (1 << 16)
#define MAX_BYTES (MAX_LINE / 2)

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* Decode `text` into `out`; returns the byte count, or -1 on a malformed literal. */
static long unhex(const char *text, unsigned char *out, size_t cap) {
  size_t n = 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    int hi = hex_nibble((unsigned char)text[0]);
    int lo = text[1] ? hex_nibble((unsigned char)text[1]) : -1;
    if (hi < 0 || lo < 0 || n >= cap) return -1;
    out[n++] = (unsigned char)((hi << 4) | lo);
    text += 2;
  }
  return (long)n;
}

static void emit(bcir_per_status st, unsigned long long value, size_t pos, int more) {
  if (st != BCIR_PER_OK) {
    printf("ERR %d\n", (int)st);
    return;
  }
  printf("OK %llu %lu%s\n", value, (unsigned long)pos, more ? " MORE" : "");
}

int main(void) {
  static char line[MAX_LINE];
  static unsigned char bytes[MAX_BYTES];

  while (fgets(line, (int)sizeof line, stdin) != NULL) {
    char op[32];
    long long a = 0, b = 0, c = 0;
    int variant = 0;
    char hex[MAX_LINE];
    bcir_per_reader r;
    bcir_per_status st;
    long len;
    int matched;

    hex[0] = '\0';
    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "constrained") == 0) {
      matched = sscanf(line, "%31s %lld %lld %d %65000s", op, &a, &b, &variant, hex);
      if (matched < 4) { printf("ERR 4\n"); continue; }
    } else if (strcmp(op, "semi") == 0 || strcmp(op, "bits") == 0) {
      matched = sscanf(line, "%31s %lld %d %65000s", op, &a, &variant, hex);
      if (matched < 3) { printf("ERR 4\n"); continue; }
    } else if (strcmp(op, "length") == 0) {
      matched = sscanf(line, "%31s %lld %lld %lld %d %65000s", op, &a, &b, &c, &variant, hex);
      if (matched < 5) { printf("ERR 4\n"); continue; }
    } else {
      matched = sscanf(line, "%31s %d %65000s", op, &variant, hex);
      if (matched < 2) { printf("ERR 4\n"); continue; }
    }

    len = unhex(hex, bytes, sizeof bytes);
    if (len < 0) { printf("ERR 4\n"); continue; }

    st = bcir_per_reader_init(&r, bytes, (size_t)len,
                              variant ? BCIR_PER_ALIGNED : BCIR_PER_UNALIGNED);
    if (st != BCIR_PER_OK) { printf("ERR %d\n", (int)st); continue; }

    if (strcmp(op, "constrained") == 0) {
      int64_t out = 0;
      st = bcir_per_constrained(&r, (int64_t)a, (int64_t)b, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else if (strcmp(op, "semi") == 0) {
      int64_t out = 0;
      st = bcir_per_semi_constrained(&r, (int64_t)a, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else if (strcmp(op, "unconstrained") == 0) {
      int64_t out = 0;
      st = bcir_per_unconstrained(&r, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else if (strcmp(op, "small") == 0) {
      uint64_t out = 0;
      st = bcir_per_normally_small(&r, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else if (strcmp(op, "smalllen") == 0) {
      uint64_t out = 0;
      st = bcir_per_normally_small_length(&r, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else if (strcmp(op, "length") == 0) {
      uint64_t out = 0;
      int more = 0;
      st = bcir_per_length(&r, (int64_t)a, (int64_t)b, (int)c, &out, &more);
      emit(st, (unsigned long long)out, r.pos, more);
    } else if (strcmp(op, "bits") == 0) {
      uint64_t out = 0;
      st = bcir_per_get_bits(&r, (unsigned)a, &out);
      emit(st, (unsigned long long)out, r.pos, 0);
    } else {
      printf("ERR 4\n");
    }
    fflush(stdout);
  }
  return 0;
}
