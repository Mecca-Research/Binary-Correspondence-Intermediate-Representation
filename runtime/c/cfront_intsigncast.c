/* Pure-integer narrowing cast to a SIGNED integer, result used directly (#intsigncast): the twin typed
 * every integer cast's result temp as unsigned, so a signed (sub-int) target lost its sign whenever the
 * cast value was used without first landing in a signed named local. `(signed char)(-5)` came out 251,
 * not -5; and `(int)u` read back unsigned, turning an arithmetic `>>` into a logical one. The oracle, which
 * types the cast temp with the target's true signedness, was correct. Now the twin matches: the integer
 * cast temp carries the target signedness (sub-int signed targets keep their sign; `(int)u` reads signed).
 * float -> signed casts (#floatsigncast) and unsigned targets are unchanged. */
unsigned i2schar(unsigned a, unsigned b) {
  return (unsigned)(signed char)((int)(a % 256u) - (int)(b % 256u));     /* direct use, can be negative */
}
unsigned i2short(unsigned a, unsigned b) {
  return (unsigned)(short)((int)(a % 60000u) - (int)(b % 60000u));
}
unsigned i2int_shr(unsigned a, unsigned b) {
  unsigned u = a - b;                                                     /* may wrap to a high value */
  return (unsigned)((int)u >> 3);                                         /* (int)u reads signed -> arithmetic >> */
}
unsigned i2schar_arith(unsigned a, unsigned b) {
  return (unsigned)((signed char)((int)(a % 256u) - (int)(b % 256u)) + 1000);   /* cast feeds further arithmetic */
}
unsigned i2long_div(unsigned a, unsigned b) {
  long x = (long)((int)(a % 100000u) - (int)(b % 100000u));               /* signed widen, then a signed / */
  return (unsigned)((x / 7) & 0xFFFFFFu);
}
unsigned i2uchar(unsigned a, unsigned b) {
  return (unsigned char)(a - b);                                         /* control: unsigned target unchanged */
}
