/* Pointer arithmetic mutation (#ptrarith): p++ / ++p / p += n / p -= n on a pointer parameter -- the
 * buffer-walk / cursor idiom. The pointer-arithmetic result stays a pointer (a single c.ptradd /
 * c.ptrsub claim, emitted `p += n;`), so C scales by the element size; the old integer-typed desugar
 * (`uint32_t t = p + n;`) truncated the pointer. Both rails agree structurally + Clang-equivalent. */
unsigned sum3(unsigned *p) { unsigned s = *p; p++; s += *p; ++p; s += *p; return s; }
unsigned stride(unsigned *p, unsigned n) { p += n; unsigned a = *p; p -= 1u; return a * 10u + *p; }
unsigned scan(unsigned *p, unsigned k) { unsigned s = 0u; for (unsigned i = 0u; i < k; i++) { s += *p; p++; } return s; }
