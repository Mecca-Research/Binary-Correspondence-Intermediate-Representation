/*===- fuzz_xer.c - libFuzzer entry for the X.693 XER lexical primitives ----===
 *
 * The totality contract: for ANY input bytes and ANY offset, every bcir_xer_* entry point
 * returns a status and never reads outside the buffer nor writes outside the caller's
 * output. There is no "valid input" notion here -- these are the functions a C peer runs
 * BEFORE any type is consulted, on bytes an attacker chose, so the property under test is
 * that no escape, tag or UTF-8 sequence can walk a cursor out of bounds.
 *
 * The offsets are derived FROM the input rather than fixed, so the fuzzer reaches the
 * interior of a document instead of only its first octet, and the escape/unescape pair is
 * driven with a deliberately UNDERSIZED output buffer as well as an ample one -- the
 * measure-then-write path is where an off-by-one would otherwise hide.
 *===----------------------------------------------------------------------===*/
#include <stddef.h>
#include <stdint.h>

#include "bcir_xer.h"

#define SCRATCH 4096

static void drive(const char *data, size_t len, size_t pos) {
  static uint8_t wide[SCRATCH * 8];
  static uint8_t narrow[7];
  bcir_xer_tag tag;
  uint32_t code = 0;
  size_t width = 0;
  size_t written = 0;

  (void)bcir_xer_skip_space(data, len, pos);
  (void)bcir_xer_scan_tag(data, len, pos, &tag);
  (void)bcir_xer_utf8_next((const uint8_t *)data, len, pos, &code, &width);

  (void)bcir_xer_escape((const uint8_t *)data, len, (char *)wide, sizeof(wide), &written);
  /* A buffer far too small: the writer must still count exactly and report OVERFLOW
   * rather than running past `narrow`. */
  (void)bcir_xer_escape((const uint8_t *)data, len, (char *)narrow, sizeof(narrow),
                        &written);
  /* And a measuring call, whose whole point is that `out` is NULL. */
  (void)bcir_xer_escape((const uint8_t *)data, len, 0, 0, &written);

  (void)bcir_xer_unescape(data, len, 1, wide, sizeof(wide), &written);
  (void)bcir_xer_unescape(data, len, 0, wide, sizeof(wide), &written);
  (void)bcir_xer_unescape(data, len, 1, narrow, sizeof(narrow), &written);
  (void)bcir_xer_unescape(data, len, 1, 0, 0, &written);
}

/* Walk a whole document the way a decoder would: alternate tag scans and text runs until
 * the input is exhausted. The bound is what proves the walk terminates on hostile input
 * rather than spinning on a zero-width construct. */
static void walk(const char *data, size_t len) {
  size_t pos = 0;
  int guard = 0;
  while (pos < len && guard++ < 4096) {
    bcir_xer_tag tag;
    pos = bcir_xer_skip_space(data, len, pos);
    if (pos >= len) break;
    if (data[pos] != '<') {
      pos++;
      continue;
    }
    if (bcir_xer_scan_tag(data, len, pos, &tag) != BCIR_XER_OK) break;
    if (tag.end <= pos) break;                     /* a tag must consume something */
    pos = tag.end;
  }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  static char buffer[SCRATCH];
  size_t len = size > SCRATCH ? SCRATCH : size;
  size_t at;

  for (at = 0; at < len; at++) buffer[at] = (char)data[at];
  if (len == 0) {
    drive(buffer, 0, 0);
    return 0;
  }
  drive(buffer, len, 0);
  drive(buffer, len, (size_t)data[0] % (len + 1));
  drive(buffer, len, len);                          /* the at-end edge */
  drive(buffer, len, len + 1);                      /* and past it: must be INVALID */
  walk(buffer, len);
  return 0;
}
