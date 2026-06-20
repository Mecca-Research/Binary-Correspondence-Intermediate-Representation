/* Store through a pointer (#ptrstore): `*p = v` / `*p OP= v` / `*(p + i) = v` -- writing a pointee, an
 * output parameter, or an MMIO location. The write counterpart of the deref load (cfront_deref.c): the
 * C twin previously parsed `*p` only as a read, so a store through a pointer failed ("parse error"). Now
 * both rails lower it to the same store -- offset-0 imm for `*p`, the indexed `p[i]` form for `*(p + i)`
 * -- so the emit is Clang-equivalent (verified on independent buffers since these mutate through p). */
void store_scaled(unsigned *p, unsigned v) { *p = v * 3u + 1u; }
void accum_at(unsigned *p, unsigned i, unsigned v) { *(p + i) += v & 0xFu; }
void set_then_bump(unsigned *out, unsigned v) { *out = v; *out += 7u; }
