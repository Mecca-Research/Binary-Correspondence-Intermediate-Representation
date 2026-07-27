/*===- test_xer.c - differential driver for the X.693 XER lexical C twin ----===
 *
 * Reads one operation per line from stdin and prints the C twin's answer, so
 * bcir/tests/test_c_xer.py can drive the SAME campaign through both rails and compare.
 * The driver owns the line parsing and stdio; bcir_xer.c stays freestanding.
 *
 * Every buffer argument is passed as hex, so a case may hold any octet sequence at all --
 * including the invalid UTF-8 and truncated escapes that are the point of the exercise.
 *
 *   tag <hex> <pos>                 scan the tag at <pos>
 *   space <hex> <pos>               skip 8.1.4 white-space from <pos>
 *   utf8 <hex> <pos>                decode one UTF-8 scalar at <pos>
 *   escape <hex>                    xmlcstring-escape the UTF-8 in <hex>
 *   unescape <allow_numeric> <hex>  the inverse
 *
 * Output is "OK ..." or "ERR <status>"; a scan that finds an excluded construct prints
 * "ERR 3 <reason>" so the excluded-construct classification is compared too, not just the
 * fact of a refusal.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bcir_xer.h"

#define MAX_LINE (1 << 16)
#define MAX_BYTES (MAX_LINE / 2)

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* Decode `text` into `out`; returns the byte count, or -1 on a malformed literal. The
 * empty string is a legal input and decodes to zero bytes, which is how a case exercises
 * the empty-buffer edge. */
static long unhex(const char *text, unsigned char *out, size_t cap) {
  size_t n = 0;
  if (text[0] == '-' && (text[1] == '\0' || text[1] == '\n' || text[1] == '\r')) return 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    int hi = hex_nibble((unsigned char)text[0]);
    int lo = text[1] ? hex_nibble((unsigned char)text[1]) : -1;
    if (hi < 0 || lo < 0 || n >= cap) return -1;
    out[n++] = (unsigned char)((hi << 4) | lo);
    text += 2;
  }
  return (long)n;
}

static void print_hex(const unsigned char *data, size_t len) {
  size_t at;
  if (len == 0) {
    printf("-");
    return;
  }
  for (at = 0; at < len; at++) printf("%02x", data[at]);
}

int main(void) {
  static char line[MAX_LINE];
  static unsigned char input[MAX_BYTES];
  static unsigned char output[MAX_BYTES * 8];

  while (fgets(line, (int)sizeof(line), stdin) != NULL) {
    char op[32];
    char hex[MAX_LINE];
    long extra = 0;
    long len;

    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "tag") == 0 || strcmp(op, "space") == 0 || strcmp(op, "utf8") == 0) {
      if (sscanf(line, "%31s %s %ld", op, hex, &extra) != 3) {
        printf("ERR 6\n");
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0 || extra < 0) {
        printf("ERR 6\n");
        continue;
      }
      if (strcmp(op, "space") == 0) {
        printf("OK %lu\n",
               (unsigned long)bcir_xer_skip_space((const char *)input, (size_t)len,
                                                  (size_t)extra));
      } else if (strcmp(op, "utf8") == 0) {
        uint32_t code = 0;
        size_t width = 0;
        bcir_xer_status st = bcir_xer_utf8_next(input, (size_t)len, (size_t)extra, &code,
                                                &width);
        if (st != BCIR_XER_OK) {
          printf("ERR %d\n", (int)st);
        } else {
          printf("OK %lu %lu\n", (unsigned long)code, (unsigned long)width);
        }
      } else {
        bcir_xer_tag tag;
        bcir_xer_status st = bcir_xer_scan_tag((const char *)input, (size_t)len,
                                               (size_t)extra, &tag);
        if (st == BCIR_XER_EXCLUDED) {
          printf("ERR 3 %d\n", (int)tag.excluded);
        } else if (st != BCIR_XER_OK) {
          printf("ERR %d\n", (int)st);
        } else {
          printf("OK %d %lu %lu %lu\n", (int)tag.kind, (unsigned long)tag.name_off,
                 (unsigned long)tag.name_len, (unsigned long)tag.end);
        }
      }
      continue;
    }

    if (strcmp(op, "escape") == 0) {
      size_t written = 0;
      bcir_xer_status st;
      if (sscanf(line, "%31s %s", op, hex) != 2) {
        printf("ERR 6\n");
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR 6\n");
        continue;
      }
      st = bcir_xer_escape(input, (size_t)len, (char *)output, sizeof(output), &written);
      if (st != BCIR_XER_OK) {
        printf("ERR %d\n", (int)st);
        continue;
      }
      printf("OK ");
      print_hex(output, written);
      printf("\n");
      continue;
    }

    if (strcmp(op, "unescape") == 0) {
      size_t written = 0;
      bcir_xer_status st;
      long allow = 0;
      if (sscanf(line, "%31s %ld %s", op, &allow, hex) != 3) {
        printf("ERR 6\n");
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR 6\n");
        continue;
      }
      st = bcir_xer_unescape((const char *)input, (size_t)len, allow != 0, output,
                             sizeof(output), &written);
      if (st != BCIR_XER_OK) {
        printf("ERR %d\n", (int)st);
        continue;
      }
      printf("OK ");
      print_hex(output, written);
      printf("\n");
      continue;
    }

    printf("ERR 6\n");
  }
  return 0;
}
