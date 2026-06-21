/* Integer promotion + float on unary `-` / `~` (#unarypromote): two miscompiles the dual-emit sweep
 * surfaced after the wide-operand fix (#longunary).
 *
 * (1) A sub-int unsigned operand promotes to *signed* int (§6.3.1.1), so `~(unsigned char)0` is `-1`,
 *     not `4294967295`, and `-(unsigned char)1 < 0` is true. The C twin kept the operand's unsignedness
 *     on the result temp, so the sign test / arithmetic-shift of the complement went the wrong way -- a
 *     silent twin-only miscompile (the oracle, via `promote_int`, was correct).
 *
 * (2) `-x` on a `float` / `double` stays floating (floats do not integer-promote). BOTH rails forced
 *     the result to a `uint32_t` temp, truncating `-2.5` to a huge integer -- a both-rail miscompile the
 *     dual-rail check alone could not see (only the differential vs Clang did). The float and double
 *     spellings even disagreed with each other (4-byte vs 8-byte truncation).
 *
 * Now both rails take the promoted operand type: a sub-int integer -> signed int, a float -> the same
 * float, a long -> 64-bit; `!` stays int. */
unsigned ucomplement(unsigned a, unsigned b) {
  unsigned char c = a + b;                       /* ~c promotes to signed int: ~c is always < 0 */
  return (~c) < 0 ? 1u : 0u;                      /* the twin returned 0 (unsigned ~c) -- now 1 */
}
unsigned unegsign(unsigned a, unsigned b) {
  unsigned char c = a | 1u;                       /* c >= 1, so -c (promoted) is negative */
  (void)b;
  return (-c) < 0 ? (unsigned)c : 999u;
}
unsigned ushortshift(unsigned a, unsigned b) {
  unsigned short s = a * b;                        /* ~s is a signed int; an arithmetic >> keeps the sign */
  return (unsigned)((~s) >> 4);
}
unsigned fnegate(unsigned a, unsigned b) {
  float x = (float)(a % 1000u);
  int r = (int)(-x - (float)(b % 1000u));          /* -x stays float; truncating it would blow up r */
  return (unsigned)r;
}
unsigned dnegate(unsigned a, unsigned b) {
  double x = (double)(a % 100000u);
  long r = (long)(-x + (double)(b % 100000u));     /* -x stays double (8-byte), not a uint32 */
  return (unsigned)(r & 0xFFFFFFu);
}
