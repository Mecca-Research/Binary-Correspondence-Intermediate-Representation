/*===- test_diag.c - host harness for the BCIR diagnostic renderer ----------------===
 * Reads a source file and a tab-separated diagnostic spec on stdin, then prints the
 * Clang-layout text (default) or the JSON feed (--json). bcir/tests/test_c_cfront.py
 * feeds the SAME spec to diagnostics.render() / DiagnosticReport.to_json() and asserts
 * the two are byte-identical (the dual-rail diagnostic-format gate). Spec lines
 * (start == end == -1 -> a spanless / file-level banner). A line whose first field is a
 * severity starts a new diagnostic; "-" attaches a note and "+" a fix-it to the current
 * one (a primary's severity may itself be "note", so the sentinels can't be words):
 *     <severity>\t<start>\t<end>\t<message>      (a primary diagnostic)
 *     -\t<start>\t<end>\t<message>               (a note on the preceding primary)
 *     +\t<start>\t<end>\t<replacement>           (a fix-it on the preceding primary)
 *   usage: test_diag [--json] <source> <filename>
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bcir_diag.h"

#define MAXDIAGS 16
#define MAXNOTES 64
#define MAXFIXITS 64
#define MAXLINES (MAXDIAGS + MAXNOTES + MAXFIXITS)

/* split a line into (tag, start, end, message); message is the tail after the 3rd tab. The fields
 * point into the mutable `line` buffer (tabs are overwritten with NULs). */
static int parse_spec(char *line, char **tag, bcir_span *span, char **msg) {
  char *t1 = strchr(line, '\t'); if (!t1) return 0; *t1 = 0;
  char *t2 = strchr(t1 + 1, '\t'); if (!t2) return 0; *t2 = 0;
  char *t3 = strchr(t2 + 1, '\t'); if (!t3) return 0; *t3 = 0;
  *tag = line;
  int s = atoi(t1 + 1), e = atoi(t2 + 1);
  span->start = s; span->end = e; span->has_span = !(s == -1 && e == -1);
  *msg = t3 + 1;
  return 1;
}

int main(int argc, char **argv) {
  int json = 0, ai = 1;
  if (ai < argc && !strcmp(argv[ai], "--json")) { json = 1; ai++; }
  if (argc - ai < 2) { fprintf(stderr, "usage: %s [--json] <source> <filename>\n", argv[0]); return 2; }
  const char *path = argv[ai], *filename = argv[ai + 1];

  FILE *fp = fopen(path, "rb");
  if (!fp) { perror("fopen"); return 2; }
  static char src[1 << 16];
  size_t n = fread(src, 1, sizeof src - 1, fp); src[n] = 0; fclose(fp);

  static char lines[MAXLINES][1024];
  static bcir_diag diags[MAXDIAGS];
  static bcir_note notes[MAXNOTES];
  static bcir_fixit fixits[MAXFIXITS];
  int nd = 0, nnotes = 0, nfix = 0, cur = -1;
  for (int li = 0; li < MAXLINES && fgets(lines[li], sizeof lines[li], stdin); li++) {
    size_t L = strlen(lines[li]);
    while (L && (lines[li][L - 1] == '\n' || lines[li][L - 1] == '\r')) lines[li][--L] = 0;
    if (L == 0) { li--; continue; }
    char *tag; bcir_span sp; char *msg;
    if (!parse_spec(lines[li], &tag, &sp, &msg)) { fprintf(stderr, "bad spec: %s\n", lines[li]); return 2; }
    if (!strcmp(tag, "-")) {                                 /* "-" sentinel: a note (not a severity) */
      if (cur < 0 || nnotes >= MAXNOTES) { fprintf(stderr, "stray note\n"); return 2; }
      notes[nnotes].message = msg; notes[nnotes].span = sp;
      if (diags[cur].notes == NULL) diags[cur].notes = &notes[nnotes];
      diags[cur].n_notes++; nnotes++;
    } else if (!strcmp(tag, "+")) {                          /* "+" sentinel: a fix-it */
      if (cur < 0 || nfix >= MAXFIXITS) { fprintf(stderr, "stray fix-it\n"); return 2; }
      fixits[nfix].replacement = msg; fixits[nfix].span = sp;
      if (diags[cur].fixits == NULL) diags[cur].fixits = &fixits[nfix];
      diags[cur].n_fixits++; nfix++;
    } else if (nd < MAXDIAGS) {
      cur = nd++;
      memset(&diags[cur], 0, sizeof diags[cur]);
      diags[cur].severity = tag; diags[cur].message = msg; diags[cur].span = sp; diags[cur].phase = "parse";
    }
  }

  static char out[1 << 16];
  if (json) {
    bcir_diag_to_json(diags, nd, src, filename, out, sizeof out);
    fputs(out, stdout);
  } else {
    for (int i = 0; i < nd; i++) {
      if (i) fputc('\n', stdout);                           /* DiagnosticReport.render joins with '\n' */
      bcir_diag_render(&diags[i], src, filename, out, sizeof out);
      fputs(out, stdout);
    }
  }
  return 0;
}
