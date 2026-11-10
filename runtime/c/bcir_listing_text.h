/*===- bcir_listing_text.h - the MC1 listings' one text writer ---------------===
 *
 * The chunked writer, the number spellings and the hexadecimal-dump row that the StreamPack listing
 * (bcir_sp_listing.c) and the BCAB listing (bcir_ab_listing.c) share, so the two print one dialect
 * from one implementation: f"{v:0Nx}", f"{v}", f"{v:<N}", ascii(text) and the dump row
 * f"{offset:08x}  {hex:<{width * 3 - 1}}  |{ascii}|". Freestanding, no heap: static inline over
 * <stddef.h>/<stdint.h>. Every name carries the bcir_lt_ prefix, since installed headers share a
 * namespace with their consumers.
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_LISTING_TEXT_H
#define BCIR_LISTING_TEXT_H

#include <stddef.h>
#include <stdint.h>

/* The most octets one sink call carries. */
#define BCIR_LISTING_CHUNK 1024u

/* Take `len` octets of output; return 0 when taken, nonzero to refuse (the format call then
 * stops and reports the refusal). */
typedef int (*bcir_sink_fn)(void *ctx, const char *text, size_t len);

typedef struct bcir_lt_out {
  char buf[BCIR_LISTING_CHUNK];
  size_t n;
  bcir_sink_fn sink;
  void *ctx;
  int failed; /* the sink refused: nothing more is offered to it */
} bcir_lt_out;

static inline void bcir_lt_open(bcir_lt_out *o, bcir_sink_fn sink, void *ctx) {
  o->n = 0; o->sink = sink; o->ctx = ctx; o->failed = 0;
}

static inline void bcir_lt_flush(bcir_lt_out *o) {
  if (!o->failed && o->n && o->sink(o->ctx, o->buf, o->n)) o->failed = 1;
  o->n = 0;
}

static inline void bcir_lt_put(bcir_lt_out *o, const char *s, size_t n) {
  while (n && !o->failed) {
    size_t room = sizeof o->buf - o->n;
    size_t take = n < room ? n : room;
    for (size_t i = 0; i < take; i++) o->buf[o->n + i] = s[i];
    o->n += take; s += take; n -= take;
    if (o->n == sizeof o->buf) bcir_lt_flush(o);
  }
}

static inline void bcir_lt_c(bcir_lt_out *o, char ch) { bcir_lt_put(o, &ch, 1); }

static inline void bcir_lt_s(bcir_lt_out *o, const char *s) {
  size_t n = 0;
  while (s[n]) n++;
  bcir_lt_put(o, s, n);
}

/* f"{v:0{digits}x}": lowercase, at least `digits` wide, never truncated. */
static inline void bcir_lt_hex(bcir_lt_out *o, uint64_t v, unsigned digits) {
  static const char hex[] = "0123456789abcdef";
  char t[16];
  unsigned n = 0;
  do { t[n++] = hex[v & 0xFu]; v >>= 4; } while (v && n < 16u);
  while (n < digits && n < 16u) t[n++] = '0';
  while (n) bcir_lt_c(o, t[--n]);
}

/* f"{v}" and f"{v:<width}" (left-aligned, space-padded, never truncated) for an unsigned value. */
static inline void bcir_lt_dec_left(bcir_lt_out *o, uint64_t v, unsigned width) {
  char t[20];
  unsigned n = 0, shown;
  do { t[n++] = (char)('0' + (int)(v % 10u)); v /= 10u; } while (v);
  shown = n;
  while (n) bcir_lt_c(o, t[--n]);
  for (; shown < width; shown++) bcir_lt_c(o, ' ');
}
static inline void bcir_lt_dec(bcir_lt_out *o, uint64_t v) { bcir_lt_dec_left(o, v, 0); }

/* f"{v}" for a signed value. */
static inline void bcir_lt_sdec(bcir_lt_out *o, int64_t v) {
  if (v < 0) {
    bcir_lt_c(o, '-');
    bcir_lt_dec(o, (uint64_t)0 - (uint64_t)v);
  } else {
    bcir_lt_dec(o, (uint64_t)v);
  }
}

/* ascii(text) for a str whose octets are well-formed UTF-8 (the callers' readers checked them):
 * repr's quote choice (" when the text holds ' and no ", else ') and escapes (the quote and the
 * backslash; \t \n \r by name; every other code point below U+0020 and U+007F by number), with
 * every code point past U+007E escaped as \xhh, \uhhhh or \Uhhhhhhhh. repr keeps a printable
 * non-ASCII character literal, and which characters are printable is the interpreter's Unicode
 * database, so the oracle spells quoted strings with ascii() and this is its twin. */
static inline void bcir_lt_quoted(bcir_lt_out *o, const char *text, size_t n) {
  const uint8_t *s = (const uint8_t *)text;
  int single = 0, dbl = 0;
  for (size_t i = 0; i < n; i++) {
    if (s[i] == '\'') single = 1;
    if (s[i] == '"') dbl = 1;
  }
  char quote = single && !dbl ? '"' : '\'';
  bcir_lt_c(o, quote);
  size_t i = 0;
  while (i < n) {
    uint32_t cp = s[i];
    size_t extra = cp < 0x80u ? 0u : cp < 0xE0u ? 1u : cp < 0xF0u ? 2u : 3u;
    if (extra) cp &= extra == 1u ? 0x1Fu : extra == 2u ? 0x0Fu : 0x07u;
    i++;
    for (size_t k = 0; k < extra && i < n; k++) cp = (cp << 6) | (uint32_t)(s[i++] & 0x3Fu);
    if (cp == (uint32_t)(uint8_t)quote || cp == '\\') {
      bcir_lt_c(o, '\\'); bcir_lt_c(o, (char)cp);
    } else if (cp == '\t') {
      bcir_lt_put(o, "\\t", 2);
    } else if (cp == '\n') {
      bcir_lt_put(o, "\\n", 2);
    } else if (cp == '\r') {
      bcir_lt_put(o, "\\r", 2);
    } else if (cp >= 0x20u && cp < 0x7Fu) {
      bcir_lt_c(o, (char)cp);
    } else if (cp < 0x100u) {
      bcir_lt_put(o, "\\x", 2); bcir_lt_hex(o, cp, 2);
    } else if (cp < 0x10000u) {
      bcir_lt_put(o, "\\u", 2); bcir_lt_hex(o, cp, 4);
    } else {
      bcir_lt_put(o, "\\U", 2); bcir_lt_hex(o, cp, 8);
    }
  }
  bcir_lt_c(o, quote);
}

/* The dump's rows over data[offset, offset + length), `width` octets a row (1..64). */
static inline void bcir_lt_rows(bcir_lt_out *o, const uint8_t *data, size_t offset, size_t length,
                                uint32_t width) {
  const size_t end = offset + length;
  for (size_t at = offset; at < end && !o->failed; at += width) {
    size_t n = end - at < width ? end - at : width;
    bcir_lt_hex(o, at, 8);
    bcir_lt_put(o, "  ", 2);
    for (size_t i = 0; i < n; i++) {
      if (i) bcir_lt_c(o, ' ');
      bcir_lt_hex(o, data[at + i], 2);
    }
    for (size_t pad = n * 3u - 1u; pad < (size_t)width * 3u - 1u; pad++) bcir_lt_c(o, ' ');
    bcir_lt_put(o, "  |", 3);
    for (size_t i = 0; i < n; i++) {
      uint8_t b = data[at + i];
      bcir_lt_c(o, b >= 32u && b < 127u ? (char)b : '.');
    }
    bcir_lt_put(o, "|\n", 2);
  }
}

#endif /* BCIR_LISTING_TEXT_H */
