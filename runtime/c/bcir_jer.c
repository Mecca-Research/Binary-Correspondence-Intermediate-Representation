/*===- bcir_jer.c - freestanding bounded X.697 JER reader -------------------===
 *
 * See bcir_jer.h for the contract. Every function here is total over its inputs and touches
 * no memory outside the buffers the caller passed.
 *
 * The three stages share their lexical primitives -- `scan_string`, `scan_number`, the
 * UTF-8 decoder -- so a document cannot be bounded by one set of rules and then parsed by
 * another. That sharing is the point: the classic parser-differential bug is a validator
 * and a consumer that disagree about where a token ends.
 *===----------------------------------------------------------------------===*/
#include "bcir_jer.h"

/* --- diagnostics ------------------------------------------------------------------------- */

static bcir_jer_status fail(bcir_jer_diag *diag, bcir_jer_status status, size_t offset,
                            uint64_t needed) {
  if (diag != 0) {
    diag->status = status;
    diag->offset = offset;
    diag->needed = needed;
    diag->sink_code = 0;
  }
  return status;
}

static void clear(bcir_jer_diag *diag) {
  if (diag != 0) {
    diag->status = BCIR_JER_OK;
    diag->offset = BCIR_JER_NO_OFFSET;
    diag->needed = 0;
    diag->sink_code = 0;
  }
}

/* --- limits -------------------------------------------------------------------------------
 *
 * The defaults are deliberately modest rather than generous: a limit that no realistic
 * input reaches is a limit nobody notices is missing. */

void bcir_jer_default_limits(bcir_jer_limits *out) {
  if (out == 0) return;
  out->input_bytes = (uint64_t)1 << 20;
  out->depth = 64;
  out->nodes = 100000;
  out->members = 10000;
  out->elements = 100000;
  out->string_bytes = (uint64_t)1 << 16;
  out->number_bytes = 128;
  out->integer_digits = 64;
  out->exponent_magnitude = 4096;
  out->work = (uint64_t)1 << 24;
}

void bcir_jer_strict_limits(bcir_jer_limits *out) {
  if (out == 0) return;
  out->input_bytes = 8192;
  out->depth = 16;
  out->nodes = 512;
  out->members = 128;
  out->elements = 512;
  out->string_bytes = 1024;
  out->number_bytes = 40;
  out->integer_digits = 20;
  out->exponent_magnitude = 308;
  out->work = (uint64_t)1 << 18;
}

bcir_jer_status bcir_jer_limits_tightened(const bcir_jer_limits *base,
                                          const bcir_jer_limits *from,
                                          bcir_jer_limits *out) {
  if (base == 0 || from == 0 || out == 0) return BCIR_JER_INVALID;
  /* 4.3: limits "may be tightened by a caller, never silently expanded". A struct
   * assignment cannot say that, so the direction is checked field by field. */
  if (from->input_bytes > base->input_bytes) return BCIR_JER_MALFORMED;
  if (from->depth > base->depth) return BCIR_JER_MALFORMED;
  if (from->nodes > base->nodes) return BCIR_JER_MALFORMED;
  if (from->members > base->members) return BCIR_JER_MALFORMED;
  if (from->elements > base->elements) return BCIR_JER_MALFORMED;
  if (from->string_bytes > base->string_bytes) return BCIR_JER_MALFORMED;
  if (from->number_bytes > base->number_bytes) return BCIR_JER_MALFORMED;
  if (from->integer_digits > base->integer_digits) return BCIR_JER_MALFORMED;
  if (from->exponent_magnitude > base->exponent_magnitude) return BCIR_JER_MALFORMED;
  if (from->work > base->work) return BCIR_JER_MALFORMED;
  *out = *from;
  return BCIR_JER_OK;
}

size_t bcir_jer_stack_bytes(const bcir_jer_limits *limits) {
  if (limits == 0) return 0;
  return (size_t)limits->depth * sizeof(bcir_jer_level);
}

/* --- character classes ------------------------------------------------------------------- */

static int is_space(uint8_t c) {
  /* ECMA-404 clause 4, exactly: SPACE, HORIZONTAL TABULATION, LINE FEED, CARRIAGE RETURN.
   * Nothing else -- and in particular not FORM FEED or VERTICAL TABULATION, which some
   * JSON readers admit and which would let two rails disagree about a document's shape. */
  return c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D;
}

static int is_digit(uint8_t c) { return c >= '0' && c <= '9'; }

static int hex_value(uint8_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

/* --- UTF-8 --------------------------------------------------------------------------------- */

bcir_jer_status bcir_jer_utf8_next(const uint8_t *data, size_t len, size_t pos,
                                   uint32_t *code, size_t *width) {
  uint8_t lead;
  size_t need;
  uint32_t value;
  size_t at;

  if (code != 0) *code = 0;
  if (width != 0) *width = 0;
  if (data == 0 && len != 0) return BCIR_JER_INVALID;
  if (pos > len) return BCIR_JER_INVALID;
  if (pos == len) return BCIR_JER_MALFORMED;

  lead = data[pos];
  if (lead < 0x80) {
    if (code != 0) *code = lead;
    if (width != 0) *width = 1;
    return BCIR_JER_OK;
  }
  if (lead < 0xC2) return BCIR_JER_NOT_UTF8;     /* a bare continuation, or an overlong C0/C1 */
  if (lead < 0xE0) { need = 2; value = (uint32_t)(lead & 0x1F); }
  else if (lead < 0xF0) { need = 3; value = (uint32_t)(lead & 0x0F); }
  else if (lead < 0xF5) { need = 4; value = (uint32_t)(lead & 0x07); }
  else return BCIR_JER_NOT_UTF8;                 /* F5..FF encodes nothing below U+110000 */

  if (len - pos < need) return BCIR_JER_NOT_UTF8;
  for (at = 1; at < need; at++) {
    uint8_t cont = data[pos + at];
    if ((cont & 0xC0) != 0x80) return BCIR_JER_NOT_UTF8;
    value = (value << 6) | (uint32_t)(cont & 0x3F);
  }
  /* The three refusals a permissive decoder skips, and each is a real attack: an overlong
   * form smuggles an ASCII character past a byte-level filter, a surrogate denotes no
   * character at all, and anything above U+10FFFF is outside Unicode. */
  if (need == 2 && value < 0x80) return BCIR_JER_NOT_UTF8;
  if (need == 3 && value < 0x800) return BCIR_JER_NOT_UTF8;
  if (need == 4 && value < 0x10000) return BCIR_JER_NOT_UTF8;
  if (value >= 0xD800 && value <= 0xDFFF) return BCIR_JER_NOT_UTF8;
  if (value > 0x10FFFF) return BCIR_JER_NOT_UTF8;

  if (code != 0) *code = value;
  if (width != 0) *width = need;
  return BCIR_JER_OK;
}

bcir_jer_status bcir_jer_validate_utf8(const uint8_t *data, size_t len,
                                       bcir_jer_diag *diag) {
  size_t pos = 0;
  clear(diag);
  if (data == 0 && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  while (pos < len) {
    uint32_t code = 0;
    size_t width = 0;
    bcir_jer_status st = bcir_jer_utf8_next(data, len, pos, &code, &width);
    if (st != BCIR_JER_OK) {
      /* The offset of the FIRST octet of the invalid sequence, which is the octet
       * `UnicodeDecodeError.start` names on the Python rail. */
      return fail(diag, BCIR_JER_NOT_UTF8, pos, 0);
    }
    pos += width;
  }
  return BCIR_JER_OK;
}

/* Write one scalar as UTF-8 into `out` if there is room; always count it in `*written`.
 * Measuring (out == NULL, cap == 0) is the same code path, which is what keeps the measured
 * length and the written length from ever disagreeing. */
static bcir_jer_status emit_utf8(uint32_t code, uint8_t *out, size_t cap, size_t *written) {
  uint8_t buf[4];
  size_t n;
  size_t at;

  if (code < 0x80) {
    n = 1;
    buf[0] = (uint8_t)code;
  } else if (code < 0x800) {
    n = 2;
    buf[0] = (uint8_t)(0xC0 | (code >> 6));
    buf[1] = (uint8_t)(0x80 | (code & 0x3F));
  } else if (code < 0x10000) {
    n = 3;
    buf[0] = (uint8_t)(0xE0 | (code >> 12));
    buf[1] = (uint8_t)(0x80 | ((code >> 6) & 0x3F));
    buf[2] = (uint8_t)(0x80 | (code & 0x3F));
  } else {
    n = 4;
    buf[0] = (uint8_t)(0xF0 | (code >> 18));
    buf[1] = (uint8_t)(0x80 | ((code >> 12) & 0x3F));
    buf[2] = (uint8_t)(0x80 | ((code >> 6) & 0x3F));
    buf[3] = (uint8_t)(0x80 | (code & 0x3F));
  }
  if (*written > (size_t)-1 - n) return BCIR_JER_OVERFLOW;
  /* Written the way it is so the measuring call (out == NULL, cap == 0) and the undersized
   * call take the SAME arithmetic path: a measure that used different arithmetic from the
   * write is the classic place for the two to disagree by one. */
  if (out != 0 && n <= cap && *written <= cap - n) {
    for (at = 0; at < n; at++) out[*written + at] = buf[at];
    *written += n;
    return BCIR_JER_OK;
  }
  *written += n;
  return BCIR_JER_OVERFLOW;
}

/* --- string escapes ------------------------------------------------------------------------ */

/* Read the four hex digits of a `\uXXXX` at `pos` (which addresses the backslash). */
static bcir_jer_status read_u16(const uint8_t *data, size_t len, size_t pos, uint32_t *out,
                                bcir_jer_diag *diag) {
  int a, b, c, d;
  if (len - pos < 6) return fail(diag, BCIR_JER_MALFORMED, pos, 6 - (len - pos));
  a = hex_value(data[pos + 2]);
  b = hex_value(data[pos + 3]);
  c = hex_value(data[pos + 4]);
  d = hex_value(data[pos + 5]);
  if (a < 0 || b < 0 || c < 0 || d < 0)
    return fail(diag, BCIR_JER_MALFORMED, pos + 2, 0);
  *out = (uint32_t)((a << 12) | (b << 8) | (c << 4) | d);
  return BCIR_JER_OK;
}

bcir_jer_status bcir_jer_unescape(const uint8_t *data, size_t len,
                                  uint8_t *out, size_t cap, size_t *written,
                                  bcir_jer_diag *diag) {
  size_t pos = 0;
  size_t produced = 0;
  int overflowed = 0;

  clear(diag);
  if (written != 0) *written = 0;
  if (written == 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (data == 0 && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (out == 0 && cap != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);

  while (pos < len) {
    uint8_t byte = data[pos];
    if (byte == '"') {
      /* A quote inside the contents means the caller mis-bounded the literal. */
      return fail(diag, BCIR_JER_MALFORMED, pos, 0);
    }
    if (byte < 0x20) {
      /* ECMA-404 clause 9: a control character may not appear literally in a string. */
      return fail(diag, BCIR_JER_MALFORMED, pos, 0);
    }
    if (byte != '\\') {
      /* Copy the octet through unchanged. Validity as UTF-8 is stage 2's question, and
       * answering it again here would give one document two different offsets for the same
       * fault depending on which entry point found it. */
      if (out != 0 && produced < cap) out[produced] = byte;
      else overflowed = 1;
      produced++;
      pos++;
      continue;
    }
    if (pos + 1 >= len) return fail(diag, BCIR_JER_MALFORMED, pos, 2 - (len - pos));
    {
      uint8_t what = data[pos + 1];
      uint32_t code = 0;
      bcir_jer_status st;
      switch (what) {
        case '"': code = 0x22; break;
        case '\\': code = 0x5C; break;
        case '/': code = 0x2F; break;
        case 'b': code = 0x08; break;
        case 'f': code = 0x0C; break;
        case 'n': code = 0x0A; break;
        case 'r': code = 0x0D; break;
        case 't': code = 0x09; break;
        case 'u': break;
        default:
          /* ECMA-404 clause 9 lists exactly nine escapes and no others. */
          return fail(diag, BCIR_JER_MALFORMED, pos + 1, 0);
      }
      if (what != 'u') {
        st = emit_utf8(code, out, cap, &produced);
        if (st == BCIR_JER_OVERFLOW) overflowed = 1;
        pos += 2;
        continue;
      }
      st = read_u16(data, len, pos, &code, diag);
      if (st != BCIR_JER_OK) return st;
      if (code >= 0xD800 && code <= 0xDBFF) {
        /* A high surrogate. 7.6.2 makes the document UTF-8, and a lone surrogate has no
         * UTF-8 encoding at all -- which is exactly why the ENCODER refuses to emit one.
         * A decoder without the matching refusal can produce a value it could never
         * re-encode, so this is NOT_UTF8 rather than MALFORMED: the JSON is well formed
         * and denotes no text. */
        uint32_t low = 0;
        if (pos + 7 >= len || data[pos + 6] != '\\' || data[pos + 7] != 'u')
          return fail(diag, BCIR_JER_NOT_UTF8, pos, 0);
        st = read_u16(data, len, pos + 6, &low, diag);
        if (st != BCIR_JER_OK) return st;
        if (low < 0xDC00 || low > 0xDFFF)
          return fail(diag, BCIR_JER_NOT_UTF8, pos, 0);
        code = 0x10000u + ((code - 0xD800u) << 10) + (low - 0xDC00u);
        st = emit_utf8(code, out, cap, &produced);
        if (st == BCIR_JER_OVERFLOW) overflowed = 1;
        pos += 12;
        continue;
      }
      if (code >= 0xDC00 && code <= 0xDFFF)
        return fail(diag, BCIR_JER_NOT_UTF8, pos, 0);   /* an unpaired low surrogate */
      st = emit_utf8(code, out, cap, &produced);
      if (st == BCIR_JER_OVERFLOW) overflowed = 1;
      pos += 6;
    }
  }
  *written = produced;
  /* A capacity fault, not a position fault: there is no octet of the input to point at, so
   * the offset stays unset and `needed` carries the whole answer. */
  if (overflowed)
    return fail(diag, BCIR_JER_OVERFLOW, BCIR_JER_NO_OFFSET, (uint64_t)produced);
  return BCIR_JER_OK;
}

/* --- the shared token scanners --------------------------------------------------------------
 *
 * Both the bounding pass and the parser use these, so the two cannot disagree about where a
 * token ends. `work` is charged here as well as in the callers, which is why it is threaded
 * through rather than accumulated by the caller afterwards: 4.3 asks for a ceiling on TOTAL
 * work, and a budget only checked between tokens is no ceiling on a single enormous one. */

typedef struct scan_ctx {
  const bcir_jer_limits *limits;
  uint64_t work;
  bcir_jer_diag *diag;
} scan_ctx;

static bcir_jer_status spend(scan_ctx *ctx, uint64_t amount, size_t offset) {
  ctx->work += amount;
  if (ctx->work > ctx->limits->work)
    return fail(ctx->diag, BCIR_JER_WORK_EXCEEDED, offset, ctx->work);
  return BCIR_JER_OK;
}

/* From the opening quote to just past the closing one, counting DECODED octets so the
 * `string_bytes` ceiling bounds what a consumer must store rather than what the sender
 * typed. `*end` receives the offset just past the closing quote. */
static bcir_jer_status scan_string(const uint8_t *data, size_t len, size_t pos,
                                   scan_ctx *ctx, size_t *end) {
  size_t start = pos;
  uint64_t decoded = 0;
  bcir_jer_status st;

  pos++;
  while (pos < len) {
    uint8_t byte = data[pos];
    st = spend(ctx, 1, pos);
    if (st != BCIR_JER_OK) return st;
    if (byte == '"') {
      *end = pos + 1;
      return BCIR_JER_OK;
    }
    if (byte == '\\') {
      if (pos + 1 >= len)
        return fail(ctx->diag, BCIR_JER_MALFORMED, pos, 2 - (len - pos));
      if (data[pos + 1] == 'u') {
        uint32_t code = 0;
        st = read_u16(data, len, pos, &code, ctx->diag);
        if (st != BCIR_JER_OK) return st;
        st = spend(ctx, 6, pos);
        if (st != BCIR_JER_OK) return st;
        if (code >= 0xD800 && code <= 0xDBFF) {
          uint32_t low = 0;
          if (pos + 7 >= len || data[pos + 6] != '\\' || data[pos + 7] != 'u')
            return fail(ctx->diag, BCIR_JER_NOT_UTF8, pos, 0);
          st = read_u16(data, len, pos + 6, &low, ctx->diag);
          if (st != BCIR_JER_OK) return st;
          if (low < 0xDC00 || low > 0xDFFF)
            return fail(ctx->diag, BCIR_JER_NOT_UTF8, pos, 0);
          st = spend(ctx, 6, pos + 6);
          if (st != BCIR_JER_OK) return st;
          pos += 12;
          decoded += 4;
        } else if (code >= 0xDC00 && code <= 0xDFFF) {
          return fail(ctx->diag, BCIR_JER_NOT_UTF8, pos, 0);
        } else {
          pos += 6;
          decoded += code < 0x80 ? 1u : (code < 0x800 ? 2u : 3u);
        }
      } else {
        pos += 2;
        decoded += 1;
      }
    } else if (byte < 0x20) {
      return fail(ctx->diag, BCIR_JER_MALFORMED, pos, 0);
    } else {
      pos++;
      decoded += 1;
    }
    if (decoded > ctx->limits->string_bytes)
      return fail(ctx->diag, BCIR_JER_STRING_TOO_LONG, start, decoded);
  }
  return fail(ctx->diag, BCIR_JER_MALFORMED, start, 0);   /* an unterminated string */
}

/* Bound the number token, its integer digits and its exponent magnitude.
 *
 * The digit and exponent ceilings are separate from the token-length ceiling on purpose:
 * `1e999999999` is a short token denoting a number no ASN.1 real can hold, and a thousand
 * digits is a long token denoting a perfectly ordinary integer. 4.3 asks for both, and
 * neither implies the other. */
static bcir_jer_status scan_number(const uint8_t *data, size_t len, size_t pos,
                                   scan_ctx *ctx, size_t *end) {
  size_t start = pos;
  uint64_t digits = 0;
  bcir_jer_status st;

  if (pos < len && data[pos] == '-') pos++;
  {
    /* ECMA-404's `int` production is `0` or a nonzero digit followed by more: `01` is not a
     * number, it is two tokens. Refusing it HERE rather than downstream matters because a
     * reader that admits a leading zero has a grammar the encoder does not, so a document
     * exists that this rail accepts and can never reproduce. */
    size_t first = pos;
    if (first < len && data[first] == '0' && first + 1 < len && is_digit(data[first + 1]))
      return fail(ctx->diag, BCIR_JER_MALFORMED, first + 1, 0);
  }
  while (pos < len && is_digit(data[pos])) {
    digits++;
    st = spend(ctx, 1, pos);
    if (st != BCIR_JER_OK) return st;
    pos++;
  }
  if (digits == 0) return fail(ctx->diag, BCIR_JER_MALFORMED, start, 0);
  if (digits > ctx->limits->integer_digits)
    return fail(ctx->diag, BCIR_JER_DIGITS_EXCEEDED, start, digits);

  if (pos < len && data[pos] == '.') {
    uint64_t fraction = 0;
    pos++;
    while (pos < len && is_digit(data[pos])) {
      fraction++;
      st = spend(ctx, 1, pos);
      if (st != BCIR_JER_OK) return st;
      pos++;
    }
    if (fraction == 0) return fail(ctx->diag, BCIR_JER_MALFORMED, pos, 0);
  }
  if (pos < len && (data[pos] == 'e' || data[pos] == 'E')) {
    size_t exponent_start;
    uint64_t magnitude = 0;
    int saturated = 0;
    pos++;
    if (pos < len && (data[pos] == '+' || data[pos] == '-')) pos++;
    exponent_start = pos;
    while (pos < len && is_digit(data[pos])) {
      /* Accumulate with a saturating guard rather than wrapping: an exponent of a hundred
       * digits must be refused for being too large, not admitted because it wrapped to a
       * small number. */
      if (magnitude > ((uint64_t)-1 - 9) / 10) saturated = 1;
      else magnitude = magnitude * 10 + (uint64_t)(data[pos] - '0');
      st = spend(ctx, 1, pos);
      if (st != BCIR_JER_OK) return st;
      pos++;
    }
    if (pos == exponent_start) return fail(ctx->diag, BCIR_JER_MALFORMED, pos, 0);
    if (saturated || magnitude > ctx->limits->exponent_magnitude)
      return fail(ctx->diag, BCIR_JER_EXPONENT_EXCEEDED, start,
                  saturated ? (uint64_t)-1 : magnitude);
  }
  if ((uint64_t)(pos - start) > ctx->limits->number_bytes)
    return fail(ctx->diag, BCIR_JER_NUMBER_TOO_LONG, start, (uint64_t)(pos - start));
  *end = pos;
  return BCIR_JER_OK;
}

/* Match one of the three literals ECMA-404 clause 8 defines. Returns its length, or 0. */
static size_t match_literal(const uint8_t *data, size_t len, size_t pos,
                            bcir_jer_event *event) {
  static const char *const names[3] = {"true", "false", "null"};
  static const size_t sizes[3] = {4, 5, 4};
  static const bcir_jer_event events[3] = {
    BCIR_JER_EV_TRUE, BCIR_JER_EV_FALSE, BCIR_JER_EV_NULL
  };
  size_t which;
  for (which = 0; which < 3; which++) {
    size_t at;
    if (len - pos < sizes[which]) continue;
    for (at = 0; at < sizes[which]; at++)
      if (data[pos + at] != (uint8_t)names[which][at]) break;
    if (at == sizes[which]) {
      if (event != 0) *event = events[which];
      return sizes[which];
    }
  }
  return 0;
}

/* --- stage 1: the bounding pass ------------------------------------------------------------- */

bcir_jer_status bcir_jer_scan(const uint8_t *data, size_t len,
                              const bcir_jer_limits *limits,
                              bcir_jer_level *stack, size_t stack_entries,
                              uint64_t *nodes, bcir_jer_diag *diag) {
  scan_ctx ctx;
  size_t pos = 0;
  uint32_t depth = 0;
  uint64_t counted = 0;
  bcir_jer_status st;

  clear(diag);
  if (nodes != 0) *nodes = 0;
  if (limits == 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (data == 0 && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (stack == 0 && limits->depth != 0) return fail(diag, BCIR_JER_INVALID,
                                                    BCIR_JER_NO_OFFSET, 0);
  if (stack_entries < limits->depth)
    return fail(diag, BCIR_JER_OVERFLOW, BCIR_JER_NO_OFFSET, (uint64_t)limits->depth);

  if ((uint64_t)len > limits->input_bytes)
    return fail(diag, BCIR_JER_INPUT_TOO_LARGE, 0, (uint64_t)len);

  ctx.limits = limits;
  ctx.work = 0;
  ctx.diag = diag;

  while (pos < len) {
    uint8_t byte = data[pos];
    st = spend(&ctx, 1, pos);
    if (st != BCIR_JER_OK) return st;
    if (is_space(byte)) {
      pos++;
      continue;
    }
    if (byte == '{' || byte == '[') {
      depth++;
      if (depth > limits->depth)
        return fail(diag, BCIR_JER_DEPTH_EXCEEDED, pos, depth);
      stack[depth - 1].count = 0;
      stack[depth - 1].is_object = (uint8_t)(byte == '{');
      stack[depth - 1].state = 0;
      counted++;
      if (counted > limits->nodes)
        return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      pos++;
      continue;
    }
    if (byte == '}' || byte == ']') {
      if (depth == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      if (stack[depth - 1].is_object != (uint8_t)(byte == '}'))
        return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      depth--;
      pos++;
      continue;
    }
    if (byte == ',') {
      uint64_t cap;
      if (depth == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      stack[depth - 1].count++;
      cap = stack[depth - 1].is_object ? limits->members : limits->elements;
      if (stack[depth - 1].count + 1 > cap)
        return fail(diag,
                    stack[depth - 1].is_object ? BCIR_JER_MEMBERS_EXCEEDED
                                               : BCIR_JER_ELEMENTS_EXCEEDED,
                    pos, stack[depth - 1].count + 1);
      pos++;
      continue;
    }
    if (byte == ':') {
      pos++;
      continue;
    }
    if (byte == '"') {
      size_t end = pos;
      st = scan_string(data, len, pos, &ctx, &end);
      if (st != BCIR_JER_OK) return st;
      pos = end;
      counted++;
      if (counted > limits->nodes)
        return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      continue;
    }
    if (byte == '-' || is_digit(byte)) {
      size_t end = pos;
      st = scan_number(data, len, pos, &ctx, &end);
      if (st != BCIR_JER_OK) return st;
      pos = end;
      counted++;
      if (counted > limits->nodes)
        return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      continue;
    }
    {
      /* `true`, `false`, `null` -- and nothing else. Refusing here rather than downstream is
       * what keeps the non-JSON `NaN` and `Infinity` literals out of the bounded path: ECMA-404
       * clause 8 has no such token, and a reader that admits them has a second grammar. */
      size_t taken = match_literal(data, len, pos, 0);
      if (taken == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      st = spend(&ctx, (uint64_t)taken, pos);
      if (st != BCIR_JER_OK) return st;
      pos += taken;
      counted++;
      if (counted > limits->nodes)
        return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
    }
  }
  if (depth != 0) return fail(diag, BCIR_JER_MALFORMED, len, 0);
  if (nodes != 0) *nodes = counted;
  return BCIR_JER_OK;
}

/* --- stage 3: the grammar ---------------------------------------------------------------------
 *
 * An explicit state machine over the caller's stack, never the C stack: a document 64 deep
 * must cost 64 stack ENTRIES, not 64 return frames, or "depth" would be a limit on the
 * caller's thread rather than on the input. */

enum {
  ST_ARRAY_FIRST = 0,   /* just after `[`: a value, or `]` for the empty array */
  ST_ARRAY_VALUE = 1,   /* after an element: `,` or `]` */
  ST_ARRAY_NEXT = 2,    /* after `,`: a value, and here `]` IS a trailing comma */
  ST_OBJECT_FIRST = 3,  /* just after `{`: a member name, or `}` for the empty object */
  ST_OBJECT_COLON = 4,  /* after a name: `:` and nothing else */
  ST_OBJECT_MEMBER = 5, /* after `:`: the member's value */
  ST_OBJECT_VALUE = 6,  /* after a member value: `,` or `}` */
  ST_OBJECT_NEXT = 7    /* after `,`: a member name, and here `}` IS a trailing comma */
};

/* Whether the container's current state admits a value here. Naming it keeps the three
 * states that do (an array's first and post-comma positions, and an object's post-colon
 * one) from being a fall-through nobody can audit. */
static int expects_value(uint8_t state) {
  return state == ST_ARRAY_FIRST || state == ST_ARRAY_NEXT || state == ST_OBJECT_MEMBER;
}

typedef struct parse_ctx {
  scan_ctx scan;
  uint8_t *scratch;
  size_t scratch_cap;
  bcir_jer_sink sink;
  void *ctx;
} parse_ctx;

static bcir_jer_status emit(parse_ctx *p, bcir_jer_event event, size_t offset,
                            const uint8_t *text, size_t text_len) {
  int code;
  if (p->sink == 0) return BCIR_JER_OK;
  code = p->sink(p->ctx, event, offset, text, text_len);
  if (code == 0) return BCIR_JER_OK;
  (void)fail(p->scan.diag, BCIR_JER_SINK_REFUSED, offset, 0);
  if (p->scan.diag != 0) p->scan.diag->sink_code = code;
  return BCIR_JER_SINK_REFUSED;
}

/* Scan the string at `pos`, decode it into the scratch, and emit it as `event`. */
static bcir_jer_status emit_string(parse_ctx *p, const uint8_t *data, size_t len, size_t pos,
                                   bcir_jer_event event, size_t *end) {
  size_t stop = pos;
  size_t written = 0;
  bcir_jer_status st = scan_string(data, len, pos, &p->scan, &stop);
  if (st != BCIR_JER_OK) return st;
  /* `stop` is just past the closing quote, so the contents are (pos + 1, stop - 1). */
  st = bcir_jer_unescape(data + pos + 1, stop - pos - 2, p->scratch, p->scratch_cap,
                         &written, p->scan.diag);
  if (st != BCIR_JER_OK) {
    /* Re-anchor the diagnostic: `bcir_jer_unescape` reports offsets relative to the string
     * contents it was handed, and a caller needs one relative to the document. */
    if (p->scan.diag != 0 && p->scan.diag->offset != BCIR_JER_NO_OFFSET &&
        st != BCIR_JER_OVERFLOW)
      p->scan.diag->offset += pos + 1;
    return st;
  }
  *end = stop;
  return emit(p, event, pos, p->scratch, written);
}

bcir_jer_status bcir_jer_parse(const uint8_t *data, size_t len,
                               const bcir_jer_limits *limits,
                               bcir_jer_level *stack, size_t stack_entries,
                               uint8_t *scratch, size_t scratch_cap,
                               bcir_jer_sink sink, void *ctx,
                               bcir_jer_diag *diag) {
  parse_ctx p;
  size_t pos = 0;
  uint32_t depth = 0;
  uint64_t counted = 0;
  int done = 0;
  bcir_jer_status st;

  clear(diag);
  if (limits == 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (data == 0 && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (stack == 0 && limits->depth != 0)
    return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (scratch == 0 && scratch_cap != 0)
    return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  if (stack_entries < limits->depth)
    return fail(diag, BCIR_JER_OVERFLOW, BCIR_JER_NO_OFFSET, (uint64_t)limits->depth);
  if ((uint64_t)len > limits->input_bytes)
    return fail(diag, BCIR_JER_INPUT_TOO_LARGE, 0, (uint64_t)len);

  p.scan.limits = limits;
  p.scan.work = 0;
  p.scan.diag = diag;
  p.scratch = scratch;
  p.scratch_cap = scratch_cap;
  p.sink = sink;
  p.ctx = ctx;

  for (;;) {
    uint8_t byte;
    /* Skip white-space, charging it, so a document padded with a megabyte of spaces spends
     * the same budget as one padded with a megabyte of anything else. */
    while (pos < len && is_space(data[pos])) {
      st = spend(&p.scan, 1, pos);
      if (st != BCIR_JER_OK) return st;
      pos++;
    }
    if (done) break;
    if (pos >= len) return fail(diag, BCIR_JER_MALFORMED, len, 0);
    byte = data[pos];
    st = spend(&p.scan, 1, pos);
    if (st != BCIR_JER_OK) return st;

    /* Where are we? At depth 0 a value is the whole document; inside a container the state
     * says what is allowed next, and every "not allowed" arm is a MALFORMED carrying the
     * offset of the octet that was not allowed. */
    if (depth > 0) {
      bcir_jer_level *level = &stack[depth - 1];
      uint8_t closer = (uint8_t)(level->is_object ? '}' : ']');

      if (byte == closer) {
        /* Legal after a value, and legal when the container is still empty. NOT legal
         * straight after a comma -- `[1,]` and `{"a":1,}` are trailing commas, which
         * ECMA-404 does not admit and which several permissive readers quietly accept. */
        if (level->state != ST_ARRAY_VALUE && level->state != ST_OBJECT_VALUE &&
            level->state != ST_ARRAY_FIRST && level->state != ST_OBJECT_FIRST)
          return fail(diag, BCIR_JER_MALFORMED, pos, 0);
        st = emit(&p, level->is_object ? BCIR_JER_EV_OBJECT_END : BCIR_JER_EV_ARRAY_END,
                  pos, 0, 0);
        if (st != BCIR_JER_OK) return st;
        depth--;
        pos++;
        if (depth == 0) done = 1;
        else stack[depth - 1].state =
               (uint8_t)(stack[depth - 1].is_object ? ST_OBJECT_VALUE : ST_ARRAY_VALUE);
        continue;
      }
      if (level->state == ST_ARRAY_VALUE || level->state == ST_OBJECT_VALUE) {
        uint64_t cap;
        if (byte != ',') return fail(diag, BCIR_JER_MALFORMED, pos, 0);
        cap = level->is_object ? limits->members : limits->elements;
        level->count++;
        if (level->count + 1 > cap)
          return fail(diag, level->is_object ? BCIR_JER_MEMBERS_EXCEEDED
                                             : BCIR_JER_ELEMENTS_EXCEEDED,
                      pos, level->count + 1);
        level->state = (uint8_t)(level->is_object ? ST_OBJECT_NEXT : ST_ARRAY_NEXT);
        pos++;
        continue;
      }
      if (level->state == ST_OBJECT_COLON) {
        if (byte != ':') return fail(diag, BCIR_JER_MALFORMED, pos, 0);
        level->state = ST_OBJECT_MEMBER;
        pos++;
        continue;
      }
      if (level->state == ST_OBJECT_FIRST || level->state == ST_OBJECT_NEXT) {
        size_t end = pos;
        if (byte != '"') return fail(diag, BCIR_JER_MALFORMED, pos, 0);
        st = emit_string(&p, data, len, pos, BCIR_JER_EV_MEMBER_NAME, &end);
        if (st != BCIR_JER_OK) return st;
        counted++;
        if (counted > limits->nodes)
          return fail(diag, BCIR_JER_NODES_EXCEEDED, end, counted);
        level->state = ST_OBJECT_COLON;
        pos = end;
        continue;
      }
      if (!expects_value(level->state))
        return fail(diag, BCIR_JER_MALFORMED, pos, 0);
    }

    /* A value. */
    if (byte == '{' || byte == '[') {
      depth++;
      if (depth > limits->depth) return fail(diag, BCIR_JER_DEPTH_EXCEEDED, pos, depth);
      stack[depth - 1].count = 0;
      stack[depth - 1].is_object = (uint8_t)(byte == '{');
      stack[depth - 1].state = (uint8_t)(byte == '{' ? ST_OBJECT_FIRST : ST_ARRAY_FIRST);
      counted++;
      if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
      st = emit(&p, byte == '{' ? BCIR_JER_EV_OBJECT_BEGIN : BCIR_JER_EV_ARRAY_BEGIN,
                pos, 0, 0);
      if (st != BCIR_JER_OK) return st;
      pos++;
      continue;
    }
    if (byte == '"') {
      size_t end = pos;
      st = emit_string(&p, data, len, pos, BCIR_JER_EV_STRING, &end);
      if (st != BCIR_JER_OK) return st;
      pos = end;
    } else if (byte == '-' || is_digit(byte)) {
      size_t end = pos;
      st = scan_number(data, len, pos, &p.scan, &end);
      if (st != BCIR_JER_OK) return st;
      /* The RAW token, bounded inside `data`. Nothing here parses a double: a freestanding
       * reader that did would make "the same document" mean two values depending on libm. */
      st = emit(&p, BCIR_JER_EV_NUMBER, pos, data + pos, end - pos);
      if (st != BCIR_JER_OK) return st;
      pos = end;
    } else {
      bcir_jer_event event = BCIR_JER_EV_NULL;
      size_t taken = match_literal(data, len, pos, &event);
      if (taken == 0) return fail(diag, BCIR_JER_MALFORMED, pos, 0);
      st = spend(&p.scan, (uint64_t)taken, pos);
      if (st != BCIR_JER_OK) return st;
      st = emit(&p, event, pos, 0, 0);
      if (st != BCIR_JER_OK) return st;
      pos += taken;
    }
    counted++;
    if (counted > limits->nodes) return fail(diag, BCIR_JER_NODES_EXCEEDED, pos, counted);
    if (depth == 0) done = 1;
    else stack[depth - 1].state =
           (uint8_t)(stack[depth - 1].is_object ? ST_OBJECT_VALUE : ST_ARRAY_VALUE);
  }

  /* 4.2's ordering makes trailing input its own refusal rather than a silent truncation:
   * a peer that appended a second document must be told so, not served the first. */
  if (pos != len) return fail(diag, BCIR_JER_TRAILING_INPUT, pos, 0);
  return BCIR_JER_OK;
}

/* --- 3.3 framing ------------------------------------------------------------------------------ */

static uint32_t read_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
         ((uint32_t)p[3] << 24);
}

static uint64_t read_u64(const uint8_t *p) {
  uint64_t value = 0;
  int at;
  for (at = 7; at >= 0; at--) value = (value << 8) | (uint64_t)p[at];
  return value;
}

bcir_jer_status bcir_jer_unframe(const uint8_t *data, size_t len,
                                 bcir_jer_frame *out, bcir_jer_diag *diag) {
  uint32_t version;
  uint64_t declared;
  uint32_t crc;

  clear(diag);
  if (out == 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);
  out->version = 0;
  out->sequence = 0;
  out->generation = 0;
  out->payload = 0;
  out->payload_len = 0;
  if (data == 0 && len != 0) return fail(diag, BCIR_JER_INVALID, BCIR_JER_NO_OFFSET, 0);

  if (len < BCIR_JER_FRAME_HEADER_SIZE)
    return fail(diag, BCIR_JER_FRAME_MALFORMED, 0, BCIR_JER_FRAME_HEADER_SIZE);
  if (data[0] != 'B' || data[1] != 'J' || data[2] != 'E' || data[3] != 'R')
    return fail(diag, BCIR_JER_FRAME_MALFORMED, 0, 0);
  version = data[4];
  if (version != BCIR_JER_FRAME_VERSION)
    return fail(diag, BCIR_JER_FRAME_MALFORMED, 4, 0);
  /* The layout is `jer_bounded.py`'s `struct.Struct("<4sBxxxQQII")`, little-endian:
   *   0..3   magic "BJER"       16..23  generation : u64
   *   4      version : u8       24..27  payload length : u32
   *   5..7   padding            28..31  payload CRC-32 : u32 */
  declared = read_u32(data + 24);
  crc = read_u32(data + 28);
  {
    /* The declared length is attacker-controlled, so the comparison is done in a form that
     * cannot overflow: compare the REMAINDER against the claim rather than adding to it. */
    uint64_t carried = (uint64_t)(len - BCIR_JER_FRAME_HEADER_SIZE);
    if (carried != declared)
      return fail(diag, BCIR_JER_FRAME_MALFORMED, BCIR_JER_FRAME_HEADER_SIZE,
                  declared + BCIR_JER_FRAME_HEADER_SIZE);
  }
  if (bcir_crc32(data + BCIR_JER_FRAME_HEADER_SIZE, len - BCIR_JER_FRAME_HEADER_SIZE) != crc)
    return fail(diag, BCIR_JER_FRAME_INTEGRITY, BCIR_JER_FRAME_HEADER_SIZE, 0);

  out->version = version;
  out->sequence = read_u64(data + 8);
  out->generation = read_u64(data + 16);
  out->payload = data + BCIR_JER_FRAME_HEADER_SIZE;
  out->payload_len = len - BCIR_JER_FRAME_HEADER_SIZE;
  return BCIR_JER_OK;
}
