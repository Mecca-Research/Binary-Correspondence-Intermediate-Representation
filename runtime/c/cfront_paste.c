/* Tokens the preprocessor must keep apart (CF-PASTE). Both cfront preprocessors re-spell each line from
 * its tokens, and kept a space only between two words, so two tokens whose spellings run together into
 * other tokens came out as those (maximal munch, C 6.4p4). `a + ++g` became `a+++g`, which is `(a++) + g`:
 * the unit lowered clean and computed another value. So did `-NEG(a)` (a decrement) and `a + INC b` (a
 * post-increment of `a`). `y / *p` became a comment opener and `a + +b` a parse error, so those units were
 * refused. Each function here computes what the original does, on both rails. */
#include <stdint.h>

#define NEG(x) -x
#define INC ++
#define ID(x) x

uint32_t ps_preinc(uint32_t a, uint32_t g) {
  return a + ++g;
}

uint32_t ps_predec(uint32_t a, uint32_t g) {
  return a - --g;
}

int32_t ps_negneg(int32_t a) {
  return -NEG(a);
}

int32_t ps_incmacro(int32_t a, int32_t b) {
  return a + INC b;
}

uint32_t ps_deref(uint32_t y, uint32_t *p) {
  return y / *p + ID(y) / ID(*p);
}

int32_t ps_unary(int32_t a, int32_t b) {
  return a + +b - -b + (a - -1);
}
