/*===- bcir_diag.c - Clang-style diagnostic renderer (C twin of cfront/diagnostics.py) ===
 *
 * The source-location model + caret renderer. Every routine mirrors its diagnostics.py
 * sibling line-for-line so the rendered text is byte-identical across the two rails:
 *   line_col / _line_bounds / _caret_prefix / _snippet / render.
 *===----------------------------------------------------------------------===*/
#include "bcir_diag.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

static int source_len(const char *source) {
  size_t n = strlen(source);
  return n > (size_t)INT_MAX ? INT_MAX : (int)n;
}

static int diag_shape_valid(const bcir_diag *d) {
  if (!d || !d->severity || !d->message || d->n_notes < 0 || d->n_fixits < 0 ||
      d->n_incstack < 0 || (d->n_notes && !d->notes) ||
      (d->n_fixits && !d->fixits) || (d->n_incstack && !d->incstack) ||
      (d->has_origin && !d->origin_file)) return 0;
  for (int i = 0; i < d->n_notes; i++) if (!d->notes[i].message) return 0;
  for (int i = 0; i < d->n_fixits; i++) if (!d->fixits[i].replacement) return 0;
  for (int i = 0; i < d->n_incstack; i++) if (!d->incstack[i].file) return 0;
  return 1;
}

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
  if (line) *line = 1;
  if (col) *col = 1;
  if (!source || !line || !col) return;
  int len = source_len(source);
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
  if (!p && n) return SIZE_MAX;
  if (out && cap && w < cap - 1u) {
    size_t room = cap - 1u - w;
    size_t copy = n < room ? n : room;
    if (copy) memcpy(out + w, p, copy);
  }
  return n > SIZE_MAX - w ? SIZE_MAX : w + n;
}
static size_t putc1(char *out, size_t cap, size_t w, char c) { return put(out, cap, w, &c, 1); }
static size_t puts1(char *out, size_t cap, size_t w, const char *s) {
  return s ? put(out, cap, w, s, strlen(s)) : SIZE_MAX;
}

/* The source line for `span` plus a caret/underline line, appended as "<line>\n<underline>" (the two
 * _snippet entries, joined to the banner by the caller's '\n'). Mirrors diagnostics._snippet. */
static size_t snippet(char *out, size_t cap, size_t w, const char *s, int len, bcir_span span) {
  if (span.start < 0) span.start = 0;
  if (span.start > len) span.start = len;
  if (span.end < span.start) span.end = span.start;
  if (span.end > len) span.end = len;
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

/* Append one banner (the primary or a note). `first` guards the leading '\n' of the join. `loc_file`
 * is the file printed in the banner; `loc_line < 0` resolves the line from the span (else it is the
 * override -- the origin line from the preprocessor line map). */
static size_t banner(char *out, size_t cap, size_t w, const char *s, int len, const char *loc_file,
                     const char *severity, const char *message, bcir_span span, int *first,
                     int loc_line) {
  char head[64];
  if (!*first) w = putc1(out, cap, w, '\n');
  *first = 0;
  if (!span.has_span) {
    w = puts1(out, cap, w, loc_file); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, severity); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, message);
  } else {
    int line, col; bcir_diag_line_col(s, span.start, &line, &col);
    if (loc_line >= 0) line = loc_line;                          /* origin override */
    w = puts1(out, cap, w, loc_file);
    snprintf(head, sizeof head, ":%d:%d: ", line, col); w = puts1(out, cap, w, head);
    w = puts1(out, cap, w, severity); w = puts1(out, cap, w, ": ");
    w = puts1(out, cap, w, message);
    w = putc1(out, cap, w, '\n');
    w = snippet(out, cap, w, s, len, span);
  }
  return w;
}

/* The Python repr() of a string (the form `diagnostics.render` prints a fix-it replacement in): single
 * quotes unless the string holds a `'` and no `"` (then double quotes), backslash + the quote char +
 * \n/\r/\t escaped, other control bytes as \xXX. Mirrors CPython's string repr for ASCII input. */
static size_t py_repr(char *o, size_t cap, size_t w, const char *s) {
  int has_sq = 0, has_dq = 0;
  for (const char *p = s; *p; p++) { if (*p == '\'') has_sq = 1; else if (*p == '"') has_dq = 1; }
  char q = (has_sq && !has_dq) ? '"' : '\'';
  w = putc1(o, cap, w, q);
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    unsigned char ch = *p;
    if (ch == '\\') w = puts1(o, cap, w, "\\\\");
    else if (ch == (unsigned char)q) { w = putc1(o, cap, w, '\\'); w = putc1(o, cap, w, q); }
    else if (ch == '\n') w = puts1(o, cap, w, "\\n");
    else if (ch == '\r') w = puts1(o, cap, w, "\\r");
    else if (ch == '\t') w = puts1(o, cap, w, "\\t");
    else if (ch < 0x20 || ch == 0x7f) { char b[8]; snprintf(b, sizeof b, "\\x%02x", ch); w = puts1(o, cap, w, b); }
    else w = putc1(o, cap, w, (char)ch);
  }
  return putc1(o, cap, w, q);
}

/* Append one diagnostic's Clang-layout render at cursor `w` (no NUL); returns the new cursor. The
 * report renderer joins several of these with '\n', the C twin of "\n".join(render(d) for d ...). */
static size_t render_one(char *out, size_t cap, size_t w, const char *source, const char *filename,
                         const bcir_diag *d) {
  int len = source_len(source);
  int first = 1;
  /* origin: print the "In file included from <file>:<line>:" frames (outermost first), then relocate
   * the primary banner to (origin_file, origin_line). The notes are NOT relocated (default filename). */
  const char *pfile = d->has_origin ? d->origin_file : filename;
  int pline = d->has_origin ? d->origin_line : -1;
  if (d->has_origin) {
    char head[64];
    for (int i = 0; i < d->n_incstack; i++) {
      if (!first) w = putc1(out, cap, w, '\n');
      first = 0;
      w = puts1(out, cap, w, "In file included from ");
      w = puts1(out, cap, w, d->incstack[i].file);
      snprintf(head, sizeof head, ":%d:", d->incstack[i].line); w = puts1(out, cap, w, head);
    }
  }
  w = banner(out, cap, w, source, len, pfile, d->severity, d->message, d->span, &first, pline);
  for (int i = 0; i < d->n_fixits; i++) {                 /* one-line suggested edits, under the caret */
    const bcir_fixit *fx = &d->fixits[i];
    const char *verb = fx->replacement[0] == '\0' ? "remove"
                       : (fx->span.start == fx->span.end ? "insert" : "replace with");
    if (!first) w = putc1(out, cap, w, '\n');
    first = 0;
    w = puts1(out, cap, w, "fix-it: "); w = puts1(out, cap, w, verb);
    w = putc1(out, cap, w, ' '); w = py_repr(out, cap, w, fx->replacement);
  }
  for (int i = 0; i < d->n_notes; i++)                    /* notes use the default filename (not origin) */
    w = banner(out, cap, w, source, len, filename, "note", d->notes[i].message, d->notes[i].span, &first, -1);
  return w;
}

size_t bcir_diag_render(const bcir_diag *d, const char *source, const char *filename,
                        char *out, size_t cap) {
  if (!out) cap = 0;
  if (cap) out[0] = 0;
  if (!source) source = "";
  if (!filename) filename = "";
  if (!diag_shape_valid(d)) return 0;
  size_t w = render_one(out, cap, 0, source, filename, d);
  if (cap) out[w < cap ? w : cap - 1] = 0;
  return w;
}

size_t bcir_diag_report_render(const bcir_diag *ds, int n, const char *source, const char *filename,
                               char *out, size_t cap) {
  if (!out) cap = 0;
  if (cap) out[0] = 0;
  if (!source) source = "";
  if (!filename) filename = "";
  if (n < 0 || (n && !ds)) return 0;
  for (int i = 0; i < n; i++) if (!diag_shape_valid(&ds[i])) return 0;
  size_t w = 0;
  for (int i = 0; i < n; i++) {                           /* "\n".join(render(d) for d in diagnostics) */
    if (i) w = putc1(out, cap, w, '\n');
    w = render_one(out, cap, w, source, filename, &ds[i]);
  }
  if (cap) out[w < cap ? w : cap - 1] = 0;
  return w;
}

/* --- JSON serialization (byte-identical to Python json.dumps(indent=2)) ----------------------- */

static size_t ind(char *o, size_t cap, size_t w, int depth) {       /* 2 spaces per nesting level */
  for (int i = 0; i < depth * 2; i++) w = putc1(o, cap, w, ' ');
  return w;
}
/* A JSON string literal with the standard escapes (ensure_ascii: control chars -> \u00XX). */
static size_t jstr(char *o, size_t cap, size_t w, const char *s) {
  w = putc1(o, cap, w, '"');
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    unsigned char ch = *p;
    switch (ch) {
      case '"':  w = puts1(o, cap, w, "\\\""); break;
      case '\\': w = puts1(o, cap, w, "\\\\"); break;
      case '\n': w = puts1(o, cap, w, "\\n"); break;
      case '\r': w = puts1(o, cap, w, "\\r"); break;
      case '\t': w = puts1(o, cap, w, "\\t"); break;
      case '\b': w = puts1(o, cap, w, "\\b"); break;
      case '\f': w = puts1(o, cap, w, "\\f"); break;
      default:
        if (ch < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", ch); w = puts1(o, cap, w, b); }
        else w = putc1(o, cap, w, (char)ch);
    }
  }
  return putc1(o, cap, w, '"');
}
static size_t jint(char *o, size_t cap, size_t w, int v) {
  char b[16]; snprintf(b, sizeof b, "%d", v); return puts1(o, cap, w, b);
}
/* the comma+newline+indent that precedes every object member after the first (json.dumps with an
 * indent uses ',' as the item separator and ': ' as the key separator). */
static size_t member(char *o, size_t cap, size_t w, int *first, int kd) {
  if (!*first) w = putc1(o, cap, w, ',');
  *first = 0;
  w = putc1(o, cap, w, '\n');
  return ind(o, cap, w, kd);
}
/* a nested `{ "start": N, "end": N }` range object whose closing brace sits at `depth`. */
static size_t jrange(char *o, size_t cap, size_t w, bcir_span span, int depth) {
  int rd = depth + 1, rf = 1;
  w = putc1(o, cap, w, '{');
  w = member(o, cap, w, &rf, rd); w = jstr(o, cap, w, "start"); w = puts1(o, cap, w, ": "); w = jint(o, cap, w, span.start);
  w = member(o, cap, w, &rf, rd); w = jstr(o, cap, w, "end");   w = puts1(o, cap, w, ": "); w = jint(o, cap, w, span.end);
  w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, depth);
  return putc1(o, cap, w, '}');
}
/* the location members (the C twin of diagnostics._loc): "file" always, then line/column/range for a
 * spanned diagnostic. `kd` is the member indent depth; `first` threads the comma state. */
static size_t jloc(char *o, size_t cap, size_t w, const char *s, const char *filename,
                   bcir_span span, int *first, int kd) {
  w = member(o, cap, w, first, kd); w = jstr(o, cap, w, "file"); w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, filename);
  if (span.has_span) {
    int line, col; bcir_diag_line_col(s, span.start, &line, &col);
    w = member(o, cap, w, first, kd); w = jstr(o, cap, w, "line");   w = puts1(o, cap, w, ": "); w = jint(o, cap, w, line);
    w = member(o, cap, w, first, kd); w = jstr(o, cap, w, "column"); w = puts1(o, cap, w, ": "); w = jint(o, cap, w, col);
    w = member(o, cap, w, first, kd); w = jstr(o, cap, w, "range");  w = puts1(o, cap, w, ": "); w = jrange(o, cap, w, span, kd);
  }
  return w;
}
/* one note object `{ "message": ..., <loc> }` whose closing brace sits at `depth`. */
static size_t jnote(char *o, size_t cap, size_t w, const bcir_note *nt, const char *s,
                    const char *filename, int depth) {
  int nd = depth + 1, nf = 1;
  w = putc1(o, cap, w, '{');
  w = member(o, cap, w, &nf, nd); w = jstr(o, cap, w, "message"); w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, nt->message);
  w = jloc(o, cap, w, s, filename, nt->span, &nf, nd);
  w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, depth);
  return putc1(o, cap, w, '}');
}
/* one include-chain frame `{ "file": ..., "line": N }` whose closing brace sits at `depth`. */
static size_t jincframe(char *o, size_t cap, size_t w, const bcir_incframe *fr, int depth) {
  int id = depth + 1, jf = 1;
  w = putc1(o, cap, w, '{');
  w = member(o, cap, w, &jf, id); w = jstr(o, cap, w, "file"); w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, fr->file);
  w = member(o, cap, w, &jf, id); w = jstr(o, cap, w, "line"); w = puts1(o, cap, w, ": "); w = jint(o, cap, w, fr->line);
  w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, depth);
  return putc1(o, cap, w, '}');
}
/* one fix-it object `{ "replacement": ..., "range": {...} }` whose closing brace sits at `depth`. */
static size_t jfixit(char *o, size_t cap, size_t w, const bcir_fixit *fx, int depth) {
  int fd = depth + 1, ff = 1;
  w = putc1(o, cap, w, '{');
  w = member(o, cap, w, &ff, fd); w = jstr(o, cap, w, "replacement"); w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, fx->replacement);
  w = member(o, cap, w, &ff, fd); w = jstr(o, cap, w, "range");       w = puts1(o, cap, w, ": "); w = jrange(o, cap, w, fx->span, fd);
  w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, depth);
  return putc1(o, cap, w, '}');
}
/* one diagnostic object whose closing brace sits at `depth` (the C twin of diagnostic_to_dict). */
static size_t jdiag(char *o, size_t cap, size_t w, const bcir_diag *d, const char *s,
                    const char *filename, int depth) {
  int kd = depth + 1, f = 1;
  w = putc1(o, cap, w, '{');
  w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "severity"); w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, d->severity);
  w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "message");  w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, d->message);
  w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "phase");    w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, d->phase ? d->phase : "");
  if (d->has_origin && d->span.has_span) {                /* relocate to the origin file/line + #include chain */
    int line, col; bcir_diag_line_col(s, d->span.start, &line, &col);   /* column from the source */
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "file");   w = puts1(o, cap, w, ": "); w = jstr(o, cap, w, d->origin_file);
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "line");   w = puts1(o, cap, w, ": "); w = jint(o, cap, w, d->origin_line);
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "column"); w = puts1(o, cap, w, ": "); w = jint(o, cap, w, col);
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "range");  w = puts1(o, cap, w, ": "); w = jrange(o, cap, w, d->span, kd);
    if (d->n_incstack > 0) {
      w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "includedFrom"); w = puts1(o, cap, w, ": [");
      for (int i = 0; i < d->n_incstack; i++) {
        if (i) w = putc1(o, cap, w, ',');
        w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd + 1);
        w = jincframe(o, cap, w, &d->incstack[i], kd + 1);
      }
      w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd); w = putc1(o, cap, w, ']');
    }
  } else {
    w = jloc(o, cap, w, s, filename, d->span, &f, kd);    /* default location (spanless or no origin) */
  }
  if (d->n_fixits > 0) {                                  /* fixits come before notes (dict order) */
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "fixits"); w = puts1(o, cap, w, ": [");
    for (int i = 0; i < d->n_fixits; i++) {
      if (i) w = putc1(o, cap, w, ',');
      w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd + 1);
      w = jfixit(o, cap, w, &d->fixits[i], kd + 1);
    }
    w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd); w = putc1(o, cap, w, ']');
  }
  if (d->n_notes > 0) {
    w = member(o, cap, w, &f, kd); w = jstr(o, cap, w, "notes"); w = puts1(o, cap, w, ": [");
    for (int i = 0; i < d->n_notes; i++) {
      if (i) w = putc1(o, cap, w, ',');
      w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd + 1);
      w = jnote(o, cap, w, &d->notes[i], s, filename, kd + 1);
    }
    w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, kd); w = putc1(o, cap, w, ']');
  }
  w = putc1(o, cap, w, '\n'); w = ind(o, cap, w, depth);
  return putc1(o, cap, w, '}');
}

size_t bcir_diag_to_json(const bcir_diag *ds, int n, const char *source, const char *filename,
                         char *out, size_t cap) {
  size_t w;
  if (!out) cap = 0;
  if (cap) out[0] = 0;
  if (!source) source = "";
  if (!filename) filename = "";
  if (n < 0 || (n && !ds)) return 0;
  for (int i = 0; i < n; i++) if (!diag_shape_valid(&ds[i])) return 0;
  if (n == 0) {
    w = puts1(out, cap, 0, "[]");
  } else {
    w = putc1(out, cap, 0, '[');
    for (int i = 0; i < n; i++) {
      if (i) w = putc1(out, cap, w, ',');
      w = putc1(out, cap, w, '\n'); w = ind(out, cap, w, 1);
      w = jdiag(out, cap, w, &ds[i], source, filename, 1);
    }
    w = putc1(out, cap, w, '\n'); w = putc1(out, cap, w, ']');
  }
  if (cap) out[w < cap ? w : cap - 1] = 0;
  return w;
}
