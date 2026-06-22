/* GCC/Clang integer builtins (#builtins): the bit-manipulation family used in DSP / crypto / bitset
 * kernels -- __builtin_popcount(l/ll), clz(l/ll), ctz(l/ll), ffs, parity, __builtin_bswap16/32/64, and
 * __builtin_abs/labs/llabs. Not defined in the unit, they emit verbatim (the libm-call mold: opaque to
 * the R18 call graph, no bcir_ twin) with a fixed result type -- the bit counts + abs return int / long,
 * bswap the unsigned operand width. Previously a `bcir___builtin_popcount` twin was synthesized and R18
 * flagged it undefined. Both rails emit the same builtin; the real backend computes identically. (clz/ctz
 * are UB on 0, so the operands are forced non-zero; abs avoids INT_MIN.) */

int       bi_pop(unsigned x)            { return __builtin_popcount(x) * 2
                                               + (int)__builtin_popcountll((unsigned long long)x << 20); }
int       bi_clz(unsigned x)            { return __builtin_clz(x | 1u) * 100 + __builtin_ctz(x | 0x80000u); }
int       bi_ffs(int x)                 { return __builtin_ffs(x) * 10 + __builtin_parity((unsigned)x); }
unsigned  bi_bswap(unsigned x)          { return __builtin_bswap32(x) + (unsigned)__builtin_bswap16((unsigned short)x); }
long long bi_bswap64(unsigned long long x){ return (long long)__builtin_bswap64(x); }
long      bi_abs(int x)                 { return __builtin_abs(x) + __builtin_labs((long)x * 1000)
                                               + (long)__builtin_llabs((long long)x * 7); }
