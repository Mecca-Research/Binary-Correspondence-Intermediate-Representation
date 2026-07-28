/*===- test_oer.c - differential driver for the X.696 OER C twin ------------===
 *
 * Reads one operation per line from stdin and prints the C twin's answer, so
 * bcir/tests/test_c_oer.py can drive the SAME campaign through both rails and compare.
 * The driver owns the line parsing and stdio; bcir_oer.c stays freestanding.
 *
 *   length <hex> <pos>                     8.6 length determinant
 *   integer <hex> <pos> <width> <signed>   10.3 / 10.4
 *   preamble <hex> <pos> <optional_count>  16.2
 *   sequence <hex> <plan>                  the plan-driven decode
 *
 * `<plan>` is a comma-separated field list, each `kind:width:signed:optional:fixed`, so a
 * test can build any schema the decoder claims to support without a second encoder here.
 *
 * Output is "OK ..." or "ERR <status> <offset> <needed>", with -1 for an unset offset.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_oer.h"

#define MAX_LINE (1 << 16)
#define MAX_BYTES (MAX_LINE / 2)
#define MAX_FIELDS 64

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

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

static void print_err(const bcir_oer_diag *diag) {
  long offset = diag->offset == BCIR_OER_NO_OFFSET ? -1 : (long)diag->offset;
  printf("ERR %d %ld %llu\n", (int)diag->status, offset,
         (unsigned long long)diag->needed);
}

/* `kind:width:signed:optional:fixed,...` -> a field table. Returns the count, or -1. */
static long parse_plan(const char *text, bcir_oer_field *fields, size_t cap) {
  size_t n = 0;
  while (*text != '\0' && *text != '\n' && *text != '\r') {
    unsigned kind = 0, width = 0, is_signed = 0, optional = 0, fixed = 0;
    int consumed = 0;
    if (n >= cap) return -1;
    if (sscanf(text, "%u:%u:%u:%u:%u%n", &kind, &width, &is_signed, &optional, &fixed,
               &consumed) != 5)
      return -1;
    fields[n].kind = (bcir_oer_kind)kind;
    fields[n].width = (uint8_t)width;
    fields[n].is_signed = (uint8_t)is_signed;
    fields[n].optional = (uint8_t)optional;
    fields[n].fixed_len = fixed;
    n++;
    text += consumed;
    if (*text == ',') text++;
    else break;
  }
  return (long)n;
}

int main(void) {
  static char line[MAX_LINE];
  static char hex[MAX_LINE];
  static char plan_text[MAX_LINE];
  static unsigned char input[MAX_BYTES];
  static bcir_oer_field fields[MAX_FIELDS];
  static bcir_oer_value values[MAX_FIELDS];

  while (fgets(line, (int)sizeof(line), stdin) != NULL) {
    char op[32];
    bcir_oer_diag diag;
    long len;
    long pos = 0, extra = 0, more = 0;

    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "length") == 0 || strcmp(op, "preamble") == 0) {
      if (sscanf(line, "%31s %s %ld %ld", op, hex, &pos, &extra) < 3) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0 || pos < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      if (strcmp(op, "length") == 0) {
        uint64_t value = 0;
        size_t end = 0;
        int canonical = 1;
        if (bcir_oer_length(input, (size_t)len, (size_t)pos, &value, &end, &canonical,
                            &diag) != BCIR_OER_OK) {
          print_err(&diag);
        } else {
          printf("OK %llu %lu %d\n", (unsigned long long)value, (unsigned long)end,
                 canonical);
        }
      } else {
        uint64_t present = 0;
        size_t end = 0;
        int canonical = 1;
        if (bcir_oer_preamble(input, (size_t)len, (size_t)pos, (unsigned)extra, &present,
                              &end, &canonical, &diag) != BCIR_OER_OK) {
          print_err(&diag);
        } else {
          printf("OK %llu %lu %d\n", (unsigned long long)present, (unsigned long)end,
                 canonical);
        }
      }
      continue;
    }

    if (strcmp(op, "integer") == 0) {
      int64_t value = 0;
      size_t end = 0;
      if (sscanf(line, "%31s %s %ld %ld %ld", op, hex, &pos, &extra, &more) != 5) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0 || pos < 0 || extra < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      if (bcir_oer_integer(input, (size_t)len, (size_t)pos, (unsigned)extra, more != 0,
                           &value, &end, &diag) != BCIR_OER_OK) {
        print_err(&diag);
      } else {
        printf("OK %lld %lu\n", (long long)value, (unsigned long)end);
      }
      continue;
    }

    if (strcmp(op, "sequence") == 0) {
      long count;
      size_t end = 0;
      int canonical = 1;
      if (sscanf(line, "%31s %s %s", op, hex, plan_text) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      count = parse_plan(plan_text, fields, MAX_FIELDS);
      if (len < 0 || count < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
        continue;
      }
      if (bcir_oer_decode_sequence(input, (size_t)len, 0, fields, (size_t)count, values,
                                   &end, &canonical, &diag) != BCIR_OER_OK) {
        print_err(&diag);
        continue;
      }
      printf("OK %lu %d", (unsigned long)end, canonical);
      for (long at = 0; at < count; at++) {
        if (!values[at].present) {
          printf(" -");
          continue;
        }
        if (fields[at].kind == BCIR_OER_FIXED_OCTETS ||
            fields[at].kind == BCIR_OER_VAR_OCTETS) {
          printf(" s");
          if (values[at].length == 0) printf("-");
          for (size_t b = 0; b < values[at].length; b++)
            printf("%02x", input[values[at].offset + b]);
        } else {
          printf(" i%lld", (long long)values[at].integer);
        }
      }
      printf("\n");
      continue;
    }

    printf("ERR %d -1 0\n", (int)BCIR_OER_INVALID);
  }
  return 0;
}
