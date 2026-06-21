/* Signed bitfield read sign-extension (#signedbf): a `signed`/`int` bitfield of width N holds an
 * N-bit two's-complement value, so reading it must sign-extend from bit N-1 -- `int x:4` holding the bit
 * pattern 1111 reads -1, not 15. Both rails extracted the field as `(unit >> off) & mask` and stopped,
 * zero-extending every read: `int x:4 = -1` came back 15, and `(int x:8) < 0` was always false. (Unsigned
 * bitfields, which DO zero-extend, were already correct.) Now both rails carry the field's signedness on
 * the c.bf.get claim and emit `(value ^ sign_bit) - sign_bit` into a signed temp for a signed field. */
struct Sbf { int x : 4; int y : 12; unsigned u : 8; int w : 1; };

unsigned bf_read(unsigned a, unsigned b) {
  struct Sbf s; s.x = a; (void)b;
  return (unsigned)(s.x + 100);                  /* x sign-extended: 1111 -> -1 -> 99 */
}
unsigned bf_signtest(unsigned a, unsigned b) {
  struct Sbf s; s.y = (int)a - (int)b;
  return s.y < 0 ? 1u : 0u;                       /* a 12-bit signed field's sign test */
}
unsigned bf_arith(unsigned a, unsigned b) {
  struct Sbf s; s.y = (int)a - (int)b;
  return (unsigned)(s.y * 2 + 5);                 /* the sign-extended value feeds arithmetic */
}
unsigned bf_onebit(unsigned a, unsigned b) {
  struct Sbf s; s.w = a; (void)b;
  return (unsigned)(s.w);                          /* int w:1 holds 0 or -1 (not 0/1) */
}
unsigned bf_unsigned(unsigned a, unsigned b) {
  struct Sbf s; s.u = a + b;
  return s.u;                                      /* control: an unsigned field zero-extends */
}
