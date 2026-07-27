/*===- bcir_xer.c - freestanding X.693 XER lexical primitives ---------------===
 *
 * See bcir_xer.h for the contract. Every function here is total over its inputs and
 * touches no memory outside the buffers the caller passed.
 *===----------------------------------------------------------------------===*/
#include "bcir_xer.h"

/* X.680 Table 3 (12.15.5): the control characters that have no direct spelling in an
 * `xmlcstring`. The index is the character code; a NULL entry means the character is not
 * in the Table. Note the three holes at 9, 10 and 13 -- the Table's own NOTE excludes them,
 * and they appear literally, which is also what makes them usable as 8.1.4 white-space. */
static const char *const kControlElement[32] = {
  "nul", "soh", "stx", "etx", "eot", "enq", "ack", "bel", "bs", 0, 0, "vt", "ff", 0,
  "so", "si", "dle", "dc1", "dc2", "dc3", "dc4", "nak", "syn", "etb", "can", "em",
  "sub", "esc", "is4", "is3", "is2", "is1"
};

int bcir_xer_is_space(int c) {
  /* 8.1.4, exactly: HORIZONTAL TABULATION (9), LINE FEED (10), CARRIAGE RETURN (13),
   * SPACE (32). Nothing else, and in particular not VERTICAL TABULATION or FORM FEED,
   * which Table 3 spells as escapes instead. */
  return c == 9 || c == 10 || c == 13 || c == 32;
}

size_t bcir_xer_skip_space(const char *data, size_t len, size_t pos) {
  if (data == 0) return pos;
  while (pos < len && bcir_xer_is_space((unsigned char)data[pos])) pos++;
  return pos;
}

/* The characters an XER element name is built from. Every name XER can produce is a
 * `typereference` (X.680 12.2), an `identifier` (12.3) or an `xmlasn1typename` (Table 4),
 * so the repertoire is ASCII letters, digits and HYPHEN-MINUS plus the LOW LINE that
 * Table 4 and 14.2's "XML" guard introduce. COLON is scanned so a namespace-qualified name
 * can be refused for that reason rather than as a syntax error. */
static int name_start(int c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static int name_char(int c) {
  return name_start(c) || (c >= '0' && c <= '9') || c == '-' || c == '.' || c == ':';
}

static bcir_xer_status excluded(bcir_xer_tag *out, bcir_xer_excluded reason) {
  out->excluded = reason;
  return BCIR_XER_EXCLUDED;
}

bcir_xer_status bcir_xer_scan_tag(const char *data, size_t len, size_t pos,
                                  bcir_xer_tag *out) {
  size_t start, cursor;
  int closing = 0;

  if (out == 0) return BCIR_XER_INVALID;
  out->kind = BCIR_XER_TAG_START;
  out->name_off = 0;
  out->name_len = 0;
  out->end = pos;
  out->excluded = BCIR_XER_EXCL_NONE;
  if (data == 0 && len != 0) return BCIR_XER_INVALID;
  if (pos > len) return BCIR_XER_INVALID;
  if (pos == len) return BCIR_XER_TRUNCATED;
  if (data[pos] != '<') return BCIR_XER_MALFORMED;

  /* 8.1.2's NOTE, construct by construct. The order matters: "<!--" has to be tested
   * before the bare "<!" that catches a DOCTYPE. */
  if (len - pos >= 4 && data[pos + 1] == '!' && data[pos + 2] == '-' &&
      data[pos + 3] == '-')
    return excluded(out, BCIR_XER_EXCL_COMMENT);
  if (len - pos >= 9 && data[pos + 1] == '!' && data[pos + 2] == '[' &&
      data[pos + 3] == 'C' && data[pos + 4] == 'D' && data[pos + 5] == 'A' &&
      data[pos + 6] == 'T' && data[pos + 7] == 'A' && data[pos + 8] == '[')
    return excluded(out, BCIR_XER_EXCL_CDATA);
  if (len - pos >= 2 && data[pos + 1] == '!')
    return excluded(out, BCIR_XER_EXCL_DOCTYPE);
  if (len - pos >= 2 && data[pos + 1] == '?')
    return excluded(out, BCIR_XER_EXCL_PI);

  start = pos + 1;
  if (start < len && data[start] == '/') {
    closing = 1;
    start++;
  }
  if (start >= len) return BCIR_XER_TRUNCATED;
  if (!name_start((unsigned char)data[start])) return BCIR_XER_MALFORMED;
  cursor = start;
  while (cursor < len && name_char((unsigned char)data[cursor])) cursor++;
  out->name_off = start;
  out->name_len = cursor - start;

  /* A COLON anywhere in the name makes it a qualified name (16.7), which only an
   * EXTENDED-XER NAMESPACE instruction can put there. */
  {
    size_t at;
    for (at = start; at < cursor; at++)
      if (data[at] == ':') return excluded(out, BCIR_XER_EXCL_NAMESPACE);
  }

  /* A character that ends the name without being white-space or a tag terminator is a
   * character no XER element name may hold. Every name XER can produce is ASCII (12.2,
   * 12.3, Table 4), so this is where a UTF-8 letter in a tag name is refused -- separately
   * from the attribute case below, which is white-space-separated and means something
   * entirely different. */
  if (cursor < len && data[cursor] != '/' && data[cursor] != '>' &&
      !bcir_xer_is_space((unsigned char)data[cursor]))
    return BCIR_XER_MALFORMED;

  cursor = bcir_xer_skip_space(data, len, cursor);
  if (cursor >= len) return BCIR_XER_TRUNCATED;
  if (data[cursor] != '/' && data[cursor] != '>') {
    /* An attribute. A general parser would accept it and hand back a value stripped of the
     * qualification the sender attached, which is why this is named rather than ignored. */
    return excluded(out, BCIR_XER_EXCL_ATTRIBUTE);
  }
  if (data[cursor] == '/') {
    if (closing) return BCIR_XER_MALFORMED;          /* "</name/>" is not a tag */
    if (cursor + 1 >= len) return BCIR_XER_TRUNCATED;
    if (data[cursor + 1] != '>') return BCIR_XER_MALFORMED;
    out->kind = BCIR_XER_TAG_EMPTY;
    out->end = cursor + 2;
    return BCIR_XER_OK;
  }
  out->kind = closing ? BCIR_XER_TAG_END : BCIR_XER_TAG_START;
  out->end = cursor + 1;
  return BCIR_XER_OK;
}

/* --- X.680 12.15: the xmlcstring lexical item ------------------------------------------ */

bcir_xer_status bcir_xer_utf8_next(const uint8_t *data, size_t len, size_t pos,
                                   uint32_t *code, size_t *width) {
  uint32_t value;
  size_t need, at;

  if (code == 0 || width == 0) return BCIR_XER_INVALID;
  *code = 0;
  *width = 0;
  if (data == 0 && len != 0) return BCIR_XER_INVALID;
  if (pos >= len) return BCIR_XER_TRUNCATED;

  value = data[pos];
  if (value < 0x80u) {
    need = 0;
  } else if ((value & 0xE0u) == 0xC0u) {
    need = 1;
    value &= 0x1Fu;
  } else if ((value & 0xF0u) == 0xE0u) {
    need = 2;
    value &= 0x0Fu;
  } else if ((value & 0xF8u) == 0xF0u) {
    need = 3;
    value &= 0x07u;
  } else {
    return BCIR_XER_MALFORMED;                       /* a continuation or a 5-octet form */
  }
  if (len - pos <= need) return BCIR_XER_TRUNCATED;
  for (at = 1; at <= need; at++) {
    uint8_t next = data[pos + at];
    if ((next & 0xC0u) != 0x80u) return BCIR_XER_MALFORMED;
    value = (value << 6) | (uint32_t)(next & 0x3Fu);
  }
  /* Overlong forms, surrogates and anything above U+10FFFF are not characters. Rejecting
   * them here rather than passing them on is what stops two encoders disagreeing about
   * what a byte sequence means -- the classic source of a security-relevant decode
   * mismatch between a validator and a consumer. */
  if ((need == 1 && value < 0x80u) || (need == 2 && value < 0x800u) ||
      (need == 3 && value < 0x10000u))
    return BCIR_XER_MALFORMED;
  if (value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu))
    return BCIR_XER_MALFORMED;
  *code = value;
  *width = need + 1;
  return BCIR_XER_OK;
}

/* Append one octet, counting even when there is nowhere to put it so a NULL/0 call
 * measures the result exactly. */
static void put(char *out, size_t cap, size_t *written, int *over, char c) {
  if (*written < cap && out != 0) {
    out[*written] = c;
  } else {
    *over = 1;
  }
  (*written)++;
}

static void put_text(char *out, size_t cap, size_t *written, int *over, const char *text) {
  while (*text != '\0') put(out, cap, written, over, *text++);
}

bcir_xer_status bcir_xer_escape(const uint8_t *utf8, size_t len, char *out, size_t cap,
                                size_t *written) {
  size_t pos = 0;
  size_t produced = 0;
  int over = 0;

  if (written == 0) return BCIR_XER_INVALID;
  *written = 0;
  if (utf8 == 0 && len != 0) return BCIR_XER_INVALID;
  if (out == 0 && cap != 0) return BCIR_XER_INVALID;

  while (pos < len) {
    uint32_t code = 0;
    size_t width = 0;
    bcir_xer_status st = bcir_xer_utf8_next(utf8, len, pos, &code, &width);
    if (st != BCIR_XER_OK) {
      *written = produced;
      return st;
    }
    if (code == '&') {
      put_text(out, cap, &produced, &over, "&amp;");           /* 12.15.4 */
    } else if (code == '<') {
      put_text(out, cap, &produced, &over, "&lt;");
    } else if (code == '>') {
      put_text(out, cap, &produced, &over, "&gt;");
    } else if (code < 32u && kControlElement[code] != 0) {
      put(out, cap, &produced, &over, '<');                    /* 12.15.5 */
      put_text(out, cap, &produced, &over, kControlElement[code]);
      put_text(out, cap, &produced, &over, "/>");
    } else if (code == 9u || code == 10u || code == 13u ||
               (code >= 32u && code <= 0xD7FFu) ||
               (code >= 0xE000u && code <= 0xFFFDu) ||
               (code >= 0x10000u && code <= 0x10FFFFu)) {
      size_t at;                                               /* 12.15.1: verbatim */
      for (at = 0; at < width; at++)
        put(out, cap, &produced, &over, (char)utf8[pos + at]);
    } else {
      /* U+FFFE, U+FFFF and the C0 controls Table 3 omits. X.680 41.10's NOTE is explicit
       * that such values "cannot be transferred using XML Encoding Rules". */
      *written = produced;
      return BCIR_XER_UNREPRESENTABLE;
    }
    pos += width;
  }
  *written = produced;
  return over ? BCIR_XER_OVERFLOW : BCIR_XER_OK;
}

/* Encode one scalar as UTF-8 into the caller's buffer, counting like `put`. */
static void put_scalar(uint8_t *out, size_t cap, size_t *written, int *over,
                       uint32_t code) {
  uint8_t buffer[4];
  size_t width, at;

  if (code < 0x80u) {
    buffer[0] = (uint8_t)code;
    width = 1;
  } else if (code < 0x800u) {
    buffer[0] = (uint8_t)(0xC0u | (code >> 6));
    buffer[1] = (uint8_t)(0x80u | (code & 0x3Fu));
    width = 2;
  } else if (code < 0x10000u) {
    buffer[0] = (uint8_t)(0xE0u | (code >> 12));
    buffer[1] = (uint8_t)(0x80u | ((code >> 6) & 0x3Fu));
    buffer[2] = (uint8_t)(0x80u | (code & 0x3Fu));
    width = 3;
  } else {
    buffer[0] = (uint8_t)(0xF0u | (code >> 18));
    buffer[1] = (uint8_t)(0x80u | ((code >> 12) & 0x3Fu));
    buffer[2] = (uint8_t)(0x80u | ((code >> 6) & 0x3Fu));
    buffer[3] = (uint8_t)(0x80u | (code & 0x3Fu));
    width = 4;
  }
  for (at = 0; at < width; at++) {
    if (*written < cap && out != 0) {
      out[*written] = buffer[at];
    } else {
      *over = 1;
    }
    (*written)++;
  }
}

/* Match `text` at `pos`; returns 1 on a match. */
static int matches(const char *data, size_t len, size_t pos, const char *text) {
  size_t at = 0;
  while (text[at] != '\0') {
    if (pos + at >= len || data[pos + at] != text[at]) return 0;
    at++;
  }
  return 1;
}

bcir_xer_status bcir_xer_unescape(const char *data, size_t len, int allow_numeric,
                                  uint8_t *out, size_t cap, size_t *written) {
  size_t pos = 0;
  size_t produced = 0;
  int over = 0;

  if (written == 0) return BCIR_XER_INVALID;
  *written = 0;
  if (data == 0 && len != 0) return BCIR_XER_INVALID;
  if (out == 0 && cap != 0) return BCIR_XER_INVALID;

  while (pos < len) {
    unsigned char c = (unsigned char)data[pos];
    if (c == '>') {
      /* 12.15.2: ">" may appear only as "&gt;" or a numeric escape. */
      *written = produced;
      return BCIR_XER_MALFORMED;
    }
    if (c == '<') {
      /* Inside an `xmlcstring` the only markup that may appear is a Table 3 control
       * escape; anything else is a child element, which the caller's schema-directed
       * layer has to see rather than this one swallowing it. */
      size_t code;
      for (code = 0; code < 32u; code++) {
        const char *name = kControlElement[code];
        size_t name_len = 0;
        if (name == 0) continue;
        while (name[name_len] != '\0') name_len++;
        if (matches(data, len, pos + 1, name) &&
            matches(data, len, pos + 1 + name_len, "/>")) {
          put_scalar(out, cap, &produced, &over, (uint32_t)code);
          pos += name_len + 3;
          break;
        }
      }
      if (code < 32u) continue;
      *written = produced;
      return BCIR_XER_MALFORMED;
    }
    if (c != '&') {
      /* Validate as UTF-8 on the way through: an unescaped octet run that is not valid
       * UTF-8 is not a character string, and 8.1.3 makes the document UTF-8. */
      uint32_t code = 0;
      size_t width = 0;
      bcir_xer_status st =
          bcir_xer_utf8_next((const uint8_t *)data, len, pos, &code, &width);
      size_t at;
      if (st != BCIR_XER_OK) {
        *written = produced;
        return st;
      }
      for (at = 0; at < width; at++) {
        if (produced < cap && out != 0) {
          out[produced] = (uint8_t)(unsigned char)data[pos + at];
        } else {
          over = 1;
        }
        produced++;
      }
      pos += width;
      continue;
    }
    if (matches(data, len, pos, "&amp;")) {
      put_scalar(out, cap, &produced, &over, '&');
      pos += 5;
    } else if (matches(data, len, pos, "&lt;")) {
      put_scalar(out, cap, &produced, &over, '<');
      pos += 4;
    } else if (matches(data, len, pos, "&gt;")) {
      put_scalar(out, cap, &produced, &over, '>');
      pos += 4;
    } else if (pos + 1 < len && data[pos + 1] == '#') {
      /* X.680 12.15.8, deleted from CXER by 9.1.3. */
      size_t at = pos + 2;
      int hex = 0;
      uint32_t code = 0;
      int digits = 0;
      if (!allow_numeric) {
        *written = produced;
        return BCIR_XER_EXCLUDED;
      }
      if (at < len && (data[at] == 'x' || data[at] == 'X')) {
        hex = 1;
        at++;
      }
      while (at < len && data[at] != ';') {
        unsigned char d = (unsigned char)data[at];
        unsigned value;
        if (d >= '0' && d <= '9') {
          value = (unsigned)(d - '0');
        } else if (hex && d >= 'a' && d <= 'f') {
          value = (unsigned)(d - 'a') + 10u;
        } else if (hex && d >= 'A' && d <= 'F') {
          value = (unsigned)(d - 'A') + 10u;
        } else {
          *written = produced;
          return BCIR_XER_MALFORMED;
        }
        if (code > (0x7FFFFFFFu - value) / (hex ? 16u : 10u)) {
          *written = produced;                       /* refuse rather than wrap */
          return BCIR_XER_MALFORMED;
        }
        code = code * (hex ? 16u : 10u) + value;
        digits++;
        at++;
      }
      if (at >= len) {
        *written = produced;
        return BCIR_XER_TRUNCATED;
      }
      if (digits == 0 || code > 0x10FFFFu || (code >= 0xD800u && code <= 0xDFFFu)) {
        *written = produced;
        return BCIR_XER_MALFORMED;
      }
      put_scalar(out, cap, &produced, &over, code);
      pos = at + 1;
    } else {
      /* XER defines no general entity mechanism, so there is no "&nbsp;". Refusing it is
       * also what keeps a C peer out of the entity-expansion bug class entirely. */
      *written = produced;
      return BCIR_XER_MALFORMED;
    }
  }
  *written = produced;
  return over ? BCIR_XER_OVERFLOW : BCIR_XER_OK;
}
