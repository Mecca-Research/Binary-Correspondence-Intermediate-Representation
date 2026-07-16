/*===- bcir_diag.h - Clang-grade source diagnostics (C twin of cfront/diagnostics.py) ===
 *
 * The source-location model + the renderer that anchors a caret under the offending
 * span -- the foundation the rest of the "Clang-grade diagnostics" surface builds on.
 *
 * A diagnostic carries a half-open [start, end) byte range into the (preprocessed)
 * translation-unit text; the renderer resolves an offset to a 1-based line/column and
 * prints the Clang layout: a `file:line:col: severity: message` banner, the source
 * line, and a `^~~~` underline. The output is byte-identical to diagnostics.render(),
 * so the two rails share one diagnostic format. Host tool (libc).
 *===----------------------------------------------------------------------===*/
#ifndef BCIR_DIAG_H
#define BCIR_DIAG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A half-open [start, end) range of byte offsets into the source. has_span == 0 is a spanless
 * (file-level) diagnostic -- a banner with no caret (the C twin of a `span is None`). */
typedef struct { int start; int end; int has_span; } bcir_span;

/* A secondary message attached to a diagnostic (an include frame, a macro-expansion site, a
 * "previous definition was here" back-reference); renders as a `note:` banner under the primary. */
typedef struct { const char *message; bcir_span span; } bcir_note;

/* A suggested edit: replace the source over `span` with `replacement`. An insertion has a zero-width
 * span; a deletion has an empty replacement. The renderer derives the verb (remove / insert /
 * replace with) from those, exactly as diagnostics.render does. */
typedef struct { bcir_span span; const char *replacement; } bcir_fixit;

/* One `#include` chain frame -- a (file, line) the offending file was included from. The renderer
 * prints these as Clang's "In file included from <file>:<line>:" notes (outermost first). */
typedef struct { const char *file; int line; } bcir_incframe;

/* One rendered-ready diagnostic. `phase` records the raising stage (lex/preprocess/parse/lower) for
 * machine-readable provenance (unused by the text renderer). */
typedef struct {
  const char *severity;       /* "error" | "warning" | "note" */
  const char *message;
  bcir_span span;
  const bcir_note *notes;     /* secondary banners (may be NULL when n_notes == 0) */
  int n_notes;
  const bcir_fixit *fixits;   /* suggested edits (may be NULL when n_fixits == 0) */
  int n_fixits;
  const char *phase;
  /* the preprocessor-line-map origin: when has_origin, the primary banner is relocated to
   * (origin_file, origin_line) and the #include chain is printed above it. The column + the source
   * snippet still come from the (preprocessed) `source`, mirroring diagnostics.render's `origin`. */
  int has_origin;
  const char *origin_file;
  int origin_line;
  const bcir_incframe *incstack;   /* outermost first (may be NULL when n_incstack == 0) */
  int n_incstack;
} bcir_diag;

/* The 1-based (line, column) of a byte offset, clamped into [0, len(source)]. Column counts bytes
 * from the line start (a tab is one column; the caret line reproduces leading tabs so it still
 * aligns). Mirrors diagnostics.line_col. */
void bcir_diag_line_col(const char *source, int offset, int *line, int *col);

/* Render `d` over `source` in the Clang text layout (banner + source line + caret underline, then a
 * banner per note), byte-identical to diagnostics.render() with no include-stack origin. snprintf
 * semantics: writes at most cap-1 bytes + NUL, returns the length that would be written.
 * Invalid borrowed pointers/count relationships return 0 and leave an empty output. */
size_t bcir_diag_render(const bcir_diag *d, const char *source, const char *filename,
                        char *out, size_t cap);

/* Render `n` diagnostics as one report (each in the Clang layout, joined by '\n'), the C twin of
 * DiagnosticReport.render() -- the output shape of a panic-mode parser-recovery run that reports
 * every error it resynchronizes past, not just the first. snprintf semantics; the complete
 * report shape is preflighted before output, so malformed input cannot leave a partial report. */
size_t bcir_diag_report_render(const bcir_diag *ds, int n, const char *source, const char *filename,
                               char *out, size_t cap);

/* Serialize `n` diagnostics as a JSON array (a `-fdiagnostics-format=json`-style machine-readable
 * feed), byte-identical to DiagnosticReport.to_json() (Python json.dumps(indent=2), no include-stack
 * origin / fix-its). Each object carries severity, message, phase, the file:line:column + byte range
 * (or just file for a spanless diagnostic), and any notes (each with its own location). Messages are
 * assumed ASCII (the front-end's diagnostics are); the standard JSON escapes are applied. snprintf
 * semantics. */
size_t bcir_diag_to_json(const bcir_diag *ds, int n, const char *source, const char *filename,
                         char *out, size_t cap);

#ifdef __cplusplus
}
#endif

#endif /* BCIR_DIAG_H */
