/*===- bcir_diag.c - Clang-style diagnostic renderer (C twin of cfront/diagnostics.py) ===
 *
 * The source-location model + caret renderer. Every routine mirrors its diagnostics.py
 * sibling line-for-line so the rendered text is byte-identical across the two rails:
 *   line_col / _line_bounds / _caret_prefix / _snippet / render.
 *===----------------------------------------------------------------------===*/
#include "bcir_diag.h"

#include <string.h>
#include <stdio.h>

/* the byte offset of the last '\n' before `off` (exclusive), or -1 if none (the C twin of
 * source.rfind("\n", 0, off)). */
static int rfind_nl(const char *s, int off) {
  for (int i = off - 1; i >= 0; i--) if (s[i] == '\n') return i;
  return -1;
}
/* the byte offset of the first '\n' at or after `off`, or len if none (str.find("\n", off)). */
static int find_nl(const char *s, int len, int off) {
  for (int i = off; i < len; i++) if (s[i] == '\n') return i;
  return len;
}

void bcir_diag_line_col(const char *source, int offset, int *line, int *col) {
  int len = (int)strlen(source);
  if (offset < 0) offset = 0; else if (offset > len) offset = len;
  int ln = 1; for (int i = 0; i < offset; i++) if (source[i] == '\n') ln++;   /* count("\n",0,offset)+1 */
  int ls = rfind_nl(source, offset) + 1;                                       /* line start */
  *line = ln;
  *col = offset - ls + 1;                                                       /* 1-based column */
}

/* the [start, end) offsets of the line containing `offset` (end excludes the newline). */
static void line_bounds(const char *s, int len, int offset, int *lstart, int *lend) {
  if (offset < 0) offset = 0; else if (offset > len) offset = len;
  *lstart = rfind_nl(s, offset) + 1;
  *lend = find_nl(s, len, offset);
}

/* append `n` bytes of `p` to the snprintf-style cursor (w may exceed cap; only [w,cap) is written). */
static size_t put(char *out, size_t cap, size_t w, const char *p, size_t n) {
  for (size_t i = 0; i < n; i++) { if (w + i + 1 < cap) out[w + i] = p[i]; }
  return w + n;
}
static size_t putc1(char *out, size_t cap, size_t w, char c) { return put(out, cap, w, &c, 1); }
static size_t puts1(char *out, size_t cap, size_t w, const char *s) { return put(out, cap, w, s, strlen(s)); }

/* The source line for `span` plus a caret/underline line, appended as "<line>\n<underline>" (the two
 * _snippet entries, joined to the banner by the caller's '\n'). Mirrors diagnostics._snippet. */
static size_t snippet(char *out, size_t cap, size_t w, const char *s, int len, bcir_span span) {
  int lstart, lend; line_bounds(s, len, span.start, &lstart, &lend);
  int line_len = lend - lstart;
  int col = span.start - lstart;                                  /* 0-based caret column */
  int hi = span.end < lend ? span.end : lend;                     /* min(span.end, lend) */
  int width = hi - span.start; if (width < 1) width = 1;          /* max(1, ...) */
  w = put(out, cap, w, s + lstart, (size_t)line_len);             /* the source line */
  w = putc1(out, cap, w, '\n');
  for (int i = 0; i < col && i < line_len; i++)                   /* caret prefix: tabs as tabs */
    w = putc1(out, cap, w, s[lstart + i] == '\t' ? '\t' : ' ');
  w = putc1(out, cap, w, '^');
  for (int i = 0; i < width - 1; i++) w = putc1(out, cap, w, '~');
  return w;
}

/* Append one banner (the primary or a note). `first` guards the leading '\n' of the join. */
static size_t banner(char *out, size_t cap, size_t w, const char *s, int len, const char *filename,
                     const char *severity, const char *message, bcir_span span, int *first) {
  char head[64];
  if (!*first) w = putc1(out, cap, w, '\n');
  *first = 0;
  if (!span.has_span) {
    w = puts1(out, cap, w, filename); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, severity); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, message);
  } else {
    int line, col; bcir_diag_line_col(s, span.start, &line, &col);
    w = puts1(out, cap, w, filename);
    snprintf(head, sizeof head, ":%d:%d: ", line, col); w = puts1(out, cap, w, head);
    w = puts1(out, cap, w, severity); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, message);
    w = putc1(out, cap, w, '\n');
    w = snippet(out, cap, w, s, len, span);
  }
  return w;
}

size_t bcir_diag_render(const bcir_diag *d, const char *source, const char *filename,
                        char *out, size_t cap) {
  int len = (int)strlen(source);
  int first = 1;
  size_t w = banner(out, cap, 0, source, len, filename, d->severity, d->message, d->span, &first);
  for (int i = 0; i < d->n_notes; i++)
    w = banner(out, cap, w, source, len, filename, "note", d->notes[i].message, d->notes[i].span, &first);
  if (cap) out[w < cap ? w : cap - 1] = 0;
  return w;
}
