/* Shift compound-assignment <<= / >>= (#shiftassign): the 3-char compound operators on scalar, member,
 * and array lvalues -- the bit-manipulation / fixed-point-scaling idiom. The lexer now emits <<= / >>=
 * as 3-char tokens; each lvalue store path desugars `lv OP= e` to `lv = lv OP e` (a c.bin.shl/shr + a
 * store) via the shared is_compound_op + compound_binop. Both rails agree + the emit is Clang-equivalent. */
struct Scl { unsigned w; };
unsigned sscale(unsigned x) { unsigned a = x; a <<= 2u; a >>= 1u; a <<= (x & 3u); return a; }
unsigned mscale(unsigned x) { struct Scl s = {x}; s.w <<= 3u; s.w >>= 2u; return s.w; }
unsigned ascale(unsigned x) { unsigned a[3] = {x, x + 1u, x + 2u}; unsigned i = x % 3u; a[i] <<= 2u; a[i] >>= 1u; return a[0] + a[1] + a[2]; }
