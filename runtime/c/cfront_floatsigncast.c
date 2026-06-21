/* float / double -> SIGNED integer cast (#floatsigncast): a floating value converted to a signed
 * integer must use a SIGNED cast operator. Both rails canonicalized every integer cast to the unsigned
 * fixed-width spelling -- `(uint32_t)x` / `(uint8_t)x` -- which makes it a float -> *unsigned* conversion:
 * UB for a negative value and target-divergent (x86 wraps, aarch64 saturates to 0), and even on x86 a
 * sub-int signed target loses the sign (`(signed char)(-5.0f)` came out 251, not -5). The #395 aarch64
 * failure was the 32-bit instance of this. Now a float -> signed-int conversion emits the signed
 * fixed-width operator into a signed temp, so it is well-defined on every target. The cast temp is used
 * directly here (no signed named local to launder it), so the temp's own type carries the result. */
unsigned f2int(unsigned a, unsigned b) {
  return (unsigned)(int)((float)(a % 1000u) - (float)(b % 1000u));      /* float -> int, can be negative */
}
unsigned f2short(unsigned a, unsigned b) {
  return (unsigned)(short)((float)(a % 2000u) - (float)(b % 2000u));    /* float -> short */
}
unsigned f2schar(unsigned a, unsigned b) {
  return (unsigned)(signed char)((float)(a % 200u) - (float)(b % 200u)); /* float -> signed char */
}
unsigned d2long(unsigned a, unsigned b) {
  long r = (long)((double)(a % 100000u) - (double)(b % 100000u));        /* double -> long */
  return (unsigned)(r & 0xFFFFFFu);
}
unsigned f2uint(unsigned a, unsigned b) {
  return (unsigned)((float)(a % 1000u) + (float)(b % 1000u));            /* control: float -> unsigned stays unsigned */
}
