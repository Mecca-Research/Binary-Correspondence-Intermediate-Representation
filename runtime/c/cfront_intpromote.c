/* Integer promotions + usual arithmetic conversions (#intpromote, the C twin of
 * ctype_model.promote_int/usual_arith_int + lower._bin_result_type). Each temp is typed by its true
 * (width, signedness): a signed `int` divide / remainder / right-shift / comparison emits signed C
 * (not the old flat uint32_t), and a mixed-width op widens per §6.3.1.8. So the emitted C is
 * behaviour-equivalent to Clang over the FULL signed range -- the case the old unsigned-32 model got
 * wrong. The equivalence harness feeds full-range (negative) values; cfront_intpromote_check.c. */
int sdiv(int a, int b){ return b ? a / b : 0; }              /* signed division (truncates toward 0) */
int smod(int a, int b){ return b ? a % b : 0; }              /* signed remainder */
int sshr(int a){ return a >> 3; }                            /* arithmetic right shift (sign-extends) */
int scmp(int a, int b){ return (a < b) + 2*(a <= b) + 4*(a > b) + 8*(a >= b); }   /* signed compares */
unsigned udiv(unsigned a, unsigned b){ return b ? a / b : 0u; }  /* unsigned division (control) */
long wide(int a, long b){ return a + b; }                    /* int + long -> 64-bit UAC */
long umix(unsigned a, long b){ return a * b; }               /* unsigned * long -> long */
