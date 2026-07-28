/*===- test_jer.c - differential driver for the bounded JER C twin ----------===
 *
 * Reads one operation per line from stdin and prints the C twin's answer, so
 * bcir/tests/test_c_jer.py can drive the SAME campaign through both rails and compare.
 * The driver owns the line parsing and stdio; bcir_jer.c stays freestanding.
 *
 * Every buffer argument is passed as hex, so a case may hold any octet sequence at all --
 * including the invalid UTF-8, lone surrogates and truncated escapes that are the point.
 *
 *   scan <strict> <hex>        stage 1: the 4.3 limits, in one octet pass
 *   utf8doc <hex>              stage 2: 7.6.2's encoding, over the whole document
 *   parse <strict> <hex>       stage 3: the grammar, printing the event trace
 *   unescape <cap> <hex>       decode one string literal's CONTENTS (cap 0 measures)
 *   utf8 <hex> <pos>           decode one UTF-8 scalar
 *   unframe <hex>              3.3: verify a frame and report its fields
 *   tighten <field> <value>    4.3's "tightened, never expanded" direction check
 *
 * `<strict>` selects the limit profile: 0 is `JerLimits()`, 1 is `STRICT_LIMITS`.
 *
 * Output is "OK ..." or "ERR <status> <offset> <needed>", where <offset> is -1 for
 * BCIR_JER_NO_OFFSET. Refusals are compared by CODE, OFFSET and NEEDED -- not merely by the
 * fact of a refusal -- because a twin that refuses the right documents for the wrong reason
 * gives a peer a diagnostic it cannot act on.
 *
 * The parse trace prints one event per line inside a "TRACE ... END" block. A number event
 * prints its RAW token, never a parsed double: see bcir_jer.h's no-floating-point note.
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <string.h>

#include "bcir_jer.h"

#define MAX_LINE (1 << 18)
#define MAX_BYTES (MAX_LINE / 2)
#define MAX_DEPTH 64
#define SCRATCH (1 << 16)

static int hex_nibble(int c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* Decode `text` into `out`; returns the byte count, or -1 on a malformed literal. "-" is
 * the empty string, which is a legal input and exercises the empty-buffer edge. */
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

static void print_err(const bcir_jer_diag *diag) {
  long offset = diag->offset == BCIR_JER_NO_OFFSET ? -1 : (long)diag->offset;
  printf("ERR %d %ld %llu\n", (int)diag->status, offset,
         (unsigned long long)diag->needed);
}

static void profile(long strict, bcir_jer_limits *out) {
  if (strict) bcir_jer_strict_limits(out);
  else bcir_jer_default_limits(out);
}

/* The two stages that must run BEFORE bcir_jer_parse, in 4.2's order.
 *
 * `bcir_jer_parse` is deliberately not a whole decode: it enforces the grammar and leaves
 * the encoding to stage 2, because a reader that answered the UTF-8 question in two places
 * would report two different offsets for one fault depending on which found it. That makes
 * calling it alone a mis-use, and the `parse` op composes the stages the way `decode_bounded`
 * does so the driver measures what a real caller gets rather than one layer in isolation.
 *
 * It matters concretely: a raw 0x80 inside a string literal is well-formed JSON *structure*,
 * so the parser copies it through untouched -- and `json.loads` refuses the same document,
 * because decoding UTF-8 is part of what it does. Only the composed pipeline is comparable
 * to it. Returns nonzero when a stage refused. */
static int prestages(const unsigned char *input, size_t len, const bcir_jer_limits *limits,
                     bcir_jer_level *stack, size_t entries, bcir_jer_diag *diag) {
  uint64_t nodes = 0;
  if (bcir_jer_scan(input, len, limits, stack, entries, &nodes, diag) != BCIR_JER_OK)
    return 1;
  return bcir_jer_validate_utf8(input, len, diag) != BCIR_JER_OK;
}

/* --- the event sink -------------------------------------------------------------------- */

static const char *const kEventName[10] = {
  "{", "}", "[", "]", "key", "str", "num", "true", "false", "null"
};

typedef struct trace_ctx {
  long refuse_at;    /* the 0-based event index at which the sink returns nonzero, or -1 */
  long seen;
} trace_ctx;

static int trace_sink(void *ctx, bcir_jer_event event, size_t offset,
                      const uint8_t *text, size_t len) {
  trace_ctx *state = (trace_ctx *)ctx;
  if (state->refuse_at >= 0 && state->seen == state->refuse_at) {
    state->seen++;
    return 7;                                     /* an arbitrary nonzero sink code */
  }
  state->seen++;
  printf("%s %lu", kEventName[(int)event], (unsigned long)offset);
  if (event == BCIR_JER_EV_MEMBER_NAME || event == BCIR_JER_EV_STRING ||
      event == BCIR_JER_EV_NUMBER) {
    printf(" ");
    print_hex(text, len);
  }
  printf("\n");
  return 0;
}

/* --- 4.3's tightening direction ---------------------------------------------------------- */

static bcir_jer_status tighten(const char *field, unsigned long long value,
                               bcir_jer_limits *out) {
  bcir_jer_limits base;
  bcir_jer_limits want;
  bcir_jer_default_limits(&base);
  want = base;
  if (strcmp(field, "input_bytes") == 0) want.input_bytes = (uint64_t)value;
  else if (strcmp(field, "depth") == 0) want.depth = (uint32_t)value;
  else if (strcmp(field, "nodes") == 0) want.nodes = (uint64_t)value;
  else if (strcmp(field, "members") == 0) want.members = (uint64_t)value;
  else if (strcmp(field, "elements") == 0) want.elements = (uint64_t)value;
  else if (strcmp(field, "string_bytes") == 0) want.string_bytes = (uint64_t)value;
  else if (strcmp(field, "number_bytes") == 0) want.number_bytes = (uint64_t)value;
  else if (strcmp(field, "integer_digits") == 0) want.integer_digits = (uint32_t)value;
  else if (strcmp(field, "exponent_magnitude") == 0)
    want.exponent_magnitude = (uint64_t)value;
  else if (strcmp(field, "work") == 0) want.work = (uint64_t)value;
  else return BCIR_JER_INVALID;
  return bcir_jer_limits_tightened(&base, &want, out);
}

int main(void) {
  static char line[MAX_LINE];
  static unsigned char input[MAX_BYTES];
  static unsigned char scratch[SCRATCH];
  static bcir_jer_level stack[MAX_DEPTH];

  while (fgets(line, (int)sizeof(line), stdin) != NULL) {
    char op[32];
    static char hex[MAX_LINE];
    bcir_jer_diag diag;
    bcir_jer_limits limits;
    long extra = 0;
    long len;

    if (sscanf(line, "%31s", op) != 1) continue;

    if (strcmp(op, "scan") == 0 || strcmp(op, "parse") == 0) {
      long strict = 0;
      uint64_t nodes = 0;
      bcir_jer_status st;
      if (sscanf(line, "%31s %ld %s", op, &strict, hex) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      profile(strict, &limits);
      if (strcmp(op, "scan") == 0) {
        st = bcir_jer_scan(input, (size_t)len, &limits, stack, MAX_DEPTH, &nodes, &diag);
        if (st != BCIR_JER_OK) print_err(&diag);
        else printf("OK %llu\n", (unsigned long long)nodes);
      } else {
        trace_ctx state;
        state.refuse_at = -1;
        state.seen = 0;
        printf("TRACE\n");
        if (prestages(input, (size_t)len, &limits, stack, MAX_DEPTH, &diag)) {
          print_err(&diag);
          printf("END\n");
          continue;
        }
        st = bcir_jer_parse(input, (size_t)len, &limits, stack, MAX_DEPTH, scratch,
                            sizeof(scratch), trace_sink, &state, &diag);
        if (st != BCIR_JER_OK) print_err(&diag);
        else printf("OK\n");
        printf("END\n");
      }
      continue;
    }

    if (strcmp(op, "refuse") == 0) {
      /* Same as `parse`, but the sink refuses at event N -- the path that proves a schema
       * layer can stop the walk and still get 4.2's structured diagnostic out. */
      trace_ctx state;
      bcir_jer_status st;
      if (sscanf(line, "%31s %ld %s", op, &extra, hex) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      profile(0, &limits);
      state.refuse_at = extra;
      state.seen = 0;
      printf("TRACE\n");
      if (prestages(input, (size_t)len, &limits, stack, MAX_DEPTH, &diag)) {
        print_err(&diag);
        printf("END\n");
        continue;
      }
      st = bcir_jer_parse(input, (size_t)len, &limits, stack, MAX_DEPTH, scratch,
                          sizeof(scratch), trace_sink, &state, &diag);
      if (st != BCIR_JER_OK) printf("ERR %d %ld %d\n", (int)diag.status,
                                    diag.offset == BCIR_JER_NO_OFFSET ? -1L
                                                                      : (long)diag.offset,
                                    diag.sink_code);
      else printf("OK\n");
      printf("END\n");
      continue;
    }

    if (strcmp(op, "utf8doc") == 0) {
      if (sscanf(line, "%31s %s", op, hex) != 2) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      if (bcir_jer_validate_utf8(input, (size_t)len, &diag) != BCIR_JER_OK) print_err(&diag);
      else printf("OK\n");
      continue;
    }

    if (strcmp(op, "unescape") == 0) {
      size_t written = 0;
      bcir_jer_status st;
      if (sscanf(line, "%31s %ld %s", op, &extra, hex) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0 || extra < 0 || (size_t)extra > sizeof(scratch)) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      st = bcir_jer_unescape(input, (size_t)len, extra == 0 ? 0 : scratch, (size_t)extra,
                             &written, &diag);
      if (st != BCIR_JER_OK) {
        print_err(&diag);
        continue;
      }
      printf("OK ");
      print_hex(scratch, written);
      printf("\n");
      continue;
    }

    if (strcmp(op, "utf8") == 0) {
      uint32_t code = 0;
      size_t width = 0;
      bcir_jer_status st;
      if (sscanf(line, "%31s %s %ld", op, hex, &extra) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0 || extra < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      st = bcir_jer_utf8_next(input, (size_t)len, (size_t)extra, &code, &width);
      if (st != BCIR_JER_OK) printf("ERR %d -1 0\n", (int)st);
      else printf("OK %lu %lu\n", (unsigned long)code, (unsigned long)width);
      continue;
    }

    if (strcmp(op, "unframe") == 0) {
      bcir_jer_frame frame;
      if (sscanf(line, "%31s %s", op, hex) != 2) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      len = unhex(hex, input, sizeof(input));
      if (len < 0) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      if (bcir_jer_unframe(input, (size_t)len, &frame, &diag) != BCIR_JER_OK) {
        print_err(&diag);
        continue;
      }
      printf("OK %lu %llu %llu ", (unsigned long)frame.version,
             (unsigned long long)frame.sequence, (unsigned long long)frame.generation);
      print_hex(frame.payload, frame.payload_len);
      printf("\n");
      continue;
    }

    if (strcmp(op, "tighten") == 0) {
      char field[64];
      unsigned long long value = 0;
      bcir_jer_status st;
      if (sscanf(line, "%31s %63s %llu", op, field, &value) != 3) {
        printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
        continue;
      }
      st = tighten(field, value, &limits);
      if (st != BCIR_JER_OK) printf("ERR %d -1 0\n", (int)st);
      else printf("OK\n");
      continue;
    }

    printf("ERR %d -1 0\n", (int)BCIR_JER_INVALID);
  }
  return 0;
}
