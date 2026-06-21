/* Unary `-` / `~` on a wide (long / long long) operand (#longunary): the value model forced a unary op's
 * result to a 4-byte uint32, so negating a `long` -- `-(long)1` -- produced 4294967295 (a 32-bit -1)
 * that widened back to a POSITIVE long, breaking `x < 0` and abs on a long (silent miscompile, both
 * rails). Now both rails keep the promoted operand type, so a `long` negation / complement stays 64-bit. */
unsigned long_abs(unsigned a, unsigned b) {
  long x = (long)(a % 1000000u) - (long)(b % 1000000u);
  long n = -x;                                  /* unary minus on a long */
  return (unsigned)(x < 0 ? n : x);             /* |x| via the long sign test */
}
unsigned long_complement(unsigned a, unsigned b) {
  long long y = ~(long long)(a % 50000u) + (long long)(b % 50000u);   /* complement on long long */
  return (unsigned)(y < 0 ? -y : y) & 0xFFFFu;
}
