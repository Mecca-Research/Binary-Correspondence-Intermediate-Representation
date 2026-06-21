/* Signed sub-int storage read sign-extension (#signedload): reading a `signed char` / `short` from a
 * struct member, a member array, a local array, or a pointer indexes an N-byte two's-complement value
 * that must sign-extend to int -- `s.c = -5` reads -5, and `s.c < 0` is true. The C twin loaded a member
 * via a zero-extending `memcpy` into a `uint32_t` temp (dropping the sign), and `emit_index` typed the
 * array-element temp unsigned -- so a signed member/array read came back as a large positive and its sign
 * test was always false. (A plain `return arr[i]` happened to match by modular luck; a `< 0` or a sign-
 * dependent result did not.) The oracle, which carries the element's signedness on the load temp, was
 * correct. Now the twin types each load temp with the element's (width, signedness) and memcpy's the exact
 * width, so a signed sub-int read sign-extends. Unsigned elements (zero-extend) are unchanged. */
struct Smix { signed char c; short h; signed char a[4]; };

unsigned m_signtest(unsigned a, unsigned b) {
  struct Smix s; s.c = (int)a - (int)b;
  return s.c < 0 ? 1u : 0u;                       /* signed char member sign test */
}
unsigned m_value(unsigned a, unsigned b) {
  struct Smix s; s.h = (int)a - (int)b;
  return (unsigned)(s.h + 1000);                   /* short member sign-extended value */
}
unsigned m_arr_signtest(unsigned a, unsigned b) {
  struct Smix s; s.a[2] = (int)a - (int)b;
  return s.a[2] < 0 ? 7u : 3u;                     /* signed char member-array sign test */
}
unsigned loc_signtest(unsigned a, unsigned b) {
  signed char arr[4]; arr[1] = (int)a - (int)b;
  return arr[1] < 0 ? 1u : 0u;                     /* signed char local-array sign test */
}
unsigned loc_value(unsigned a, unsigned b) {
  short arr[2]; arr[0] = (int)a - (int)b;
  return (unsigned)(arr[0] * 2 + 5);               /* short local-array sign-extended arithmetic */
}
unsigned uc_control(unsigned a, unsigned b) {
  unsigned char arr[4]; arr[0] = a + b;
  return arr[0] > 200u ? 1u : 0u;                  /* control: an unsigned element zero-extends */
}
