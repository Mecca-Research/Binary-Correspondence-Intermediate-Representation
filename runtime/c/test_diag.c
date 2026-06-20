/*===- test_diag.c - host harness for the BCIR diagnostic renderer ----------------===
 * Reads a source file (argv[1]) and a tab-separated diagnostic spec on stdin, then prints
 * bcir_diag_render's Clang-layout output. bcir/tests/test_c_cfront.py feeds the SAME spec
 * to diagnostics.render() and asserts the two are byte-identical (the dual-rail diagnostic
 * format gate). Spec lines (start == end == -1 -> a spanless / file-level banner):
 *     <severity>\t<start>\t<end>\t<message>      (line 1: the primary)
 *     note\t<start>\t<end>\t<message>            (zero or more notes)
 *===----------------------------------------------------------------------===*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bcir_diag.h"

#define MAXNOTES 16

/* split a line into (field0, start, end, message); message is the tail after the 3rd tab. Returns the
 * field-0 token in `tag` and fills span + msg (which point into the mutable line buffer). */
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
  if (argc < 3) { fprintf(stderr, "usage: %s <source> <filename>\n", argv[0]); return 2; }
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) { perror("fopen"); return 2; }
  static char src[1 << 16];
  size_t n = fread(src, 1, sizeof src - 1, fp); src[n] = 0; fclose(fp);

  static char lines[MAXNOTES + 1][1024];
  static bcir_note notes[MAXNOTES];
  bcir_diag d; memset(&d, 0, sizeof d); d.notes = notes; d.phase = "parse";
  int got_primary = 0, nn = 0;
  for (int li = 0; li <= MAXNOTES && fgets(lines[li], sizeof lines[li], stdin); li++) {
    size_t L = strlen(lines[li]); while (L && (lines[li][L - 1] == '\n' || lines[li][L - 1] == '\r')) lines[li][--L] = 0;
    if (L == 0) { li--; continue; }
    char *tag; bcir_span sp; char *msg;
    if (!parse_spec(lines[li], &tag, &sp, &msg)) { fprintf(stderr, "bad spec: %s\n", lines[li]); return 2; }
    if (!got_primary) { d.severity = tag; d.message = msg; d.span = sp; got_primary = 1; }
    else if (nn < MAXNOTES) { notes[nn].message = msg; notes[nn].span = sp; nn++; }
  }
  if (!got_primary) { fprintf(stderr, "no primary diagnostic\n"); return 2; }
  d.n_notes = nn;

  static char out[1 << 16];
  bcir_diag_render(&d, src, argv[2], out, sizeof out);
  fputs(out, stdout);
  return 0;
}
