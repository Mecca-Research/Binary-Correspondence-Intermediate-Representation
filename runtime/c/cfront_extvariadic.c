/* External variadic calls (#extvariadic): calling the printf / scanf family of <stdio.h> variadic
 * functions (snprintf / vsnprintf / ...). They are not defined in the unit and not lowered -- like a
 * <math.h> (libm) call they emit verbatim and stay opaque to the R18 call graph (a bcir_ twin / an
 * "undefined function" R18 diagnostic would be wrong), returning int. The read-only char[] format string
 * passes through as an argument. A vsnprintf-forwarding wrapper combines this with the variadic consumer
 * side: define `f(..., ...)`, walk a va_list, and hand the cursor to vsnprintf. Both rails agree
 * structurally and the emit is Clang-behaviour-equivalent (the real libc produces identical bytes). The
 * differential checks the formatted buffer AND the returned count. */

int ev_int(char *buf, int x)         { return snprintf(buf, 32, "%d", x); }
int ev_mix(char *buf, int x, long y) { return snprintf(buf, 48, "%d/%ld/%x", x, y, (unsigned)x); }
int ev_width(char *buf, int x)       { return snprintf(buf, 32, "[%06d]", x); }

/* a vsnprintf-forwarding wrapper -- a variadic function handing its own cursor to an external variadic */
int ev_fwd(char *buf, int sz, const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  int n = vsnprintf(buf, sz, fmt, ap);
  va_end(ap);
  return n;
}
int ev_call(char *buf, int a, int b) { return ev_fwd(buf, 32, "%d,%d", a, b); }
