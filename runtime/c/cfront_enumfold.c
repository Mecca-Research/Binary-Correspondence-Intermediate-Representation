/* CF-ENUMFOLD: an enumerator, a case label and an array dimension fold in C's own types on both rails -- each
 * operation in its promoted operands' type (C11 6.3.1.1, 6.3.1.8), a division truncating toward zero (6.5.5p6), a
 * comparison with an unsigned int operand unsigned, a shift in its promoted left operand's type, a cast converting
 * to its type, `?:` evaluating one arm in the arms' common type and `&&`/`||` their right operand only when the left
 * one does not decide -- and `long` as wide as the target makes it. The oracle had folded with unbounded integers
 * (`-7 / 2` was -4, `-7 % 2` 1, `~0u > 5` 0) and the twin in `long long` (`~0u > 5` 0), and the twin spelled a
 * negative enumerator as its 64-bit two's complement (`int32_t t = 18446744073709551613;`). A dimension that is an
 * integer constant expression makes a fixed array, not a variable-length one (6.7.6.2p4). */
#include <stdint.h>

/* division and remainder truncate toward zero; a remainder takes the dividend's sign */
enum ef_div { EF_Q = -7 / 2, EF_R = -7 % 2, EF_Q2 = 7 / -2, EF_R2 = 7 % -2, EF_Q3 = -7 / -2, EF_R3 = -7 % -2,
              EF_UQ = -7u / 2u };

/* a comparison in the usual arithmetic conversions: an unsigned int operand makes it unsigned, a type narrower
 * than int promotes to int, a long long holds every unsigned int; a long holds one only where it is 64 bits. A
 * constant is the first type of its list that holds it: past INT_MAX a decimal one is a long where long is 64 bits
 * and a long long where it is 32, and `0xFFFFFFFFL` a long or an unsigned long -- positive either way */
enum ef_cmp { EF_U = ~0u > 5, EF_NEG = -1 < 0u, EF_ULL = -1 < 0ull, EF_LL = -1LL < 0u, EF_U8 = (uint8_t)255 > -1,
              EF_U16 = (uint16_t)0 - 1 < 0, EF_EQ = 0xFFFFFFFFu == -1, EF_LONG = -1L < 1u,
              EF_HEXL = -1 < 0xFFFFFFFFL, EF_DEC = 2147483648 > 0, EF_HEXP = 0xFFFFFFFFL > 0 };

/* shifts: in the promoted left operand's type -- an unsigned one logical, a negative signed one arithmetic */
enum ef_shift { EF_SHL = 1 << 30, EF_SAR = -8 >> 1, EF_SHR = 0x80000000 >> 31, EF_USHL = 1u << 31 >> 28,
                EF_WIDE = 1ull << 40 >> 37, EF_NARROW = (uint8_t)1 << 9, EF_SIGNBIT = (int)(1u << 31) };

/* casts: to an unsigned type modulo 2^N, to a signed type as Clang and GCC convert, to _Bool whether nonzero */
enum ef_cast { EF_C8 = (uint8_t)300, EF_CS8 = (int8_t)200, EF_C16 = (uint16_t)-1, EF_CS16 = (int16_t)0x8000,
               EF_CINT = (int)4294967295u, EF_CBOOL = (_Bool)256, EF_CU32 = (uint32_t)-1 / 2u,
               EF_CL = (unsigned long)-1 > 0xFFFFFFFFu, EF_CLL = (int)(long long)-1 };

/* `?:` takes one arm, in the arms' common type; an operand C does not evaluate is not folded */
enum ef_cond { EF_SEL = EF_U ? (EF_NEG ? 10 : 20) : 30, EF_SELU = (1 ? -1 : 0u) > 0, EF_SELN = 0 ? 1 : 2 ? 3 : 4,
               EF_SELD = (0 ? 1u : -1) / 2, EF_AND = 0 && 1 / 0, EF_OR = 1 || 1 / 0, EF_DEAD = 1 ? 2 : 1 / 0,
               EF_DEADOVF = 0 ? 2147483647 + 1 : -5, EF_LOGIC = (EF_SEL && EF_DEADOVF) + (0 || 0) * 2 + !EF_SEL * 4
               + !0 * 8 };

/* an enumerator with no value counts on from the one before it, negative and at int's edges */
enum ef_run { EF_LO = -3, EF_LO1, EF_LO2, EF_ZERO, EF_MIN = -2147483647 - 1, EF_MIN1, EF_MAX = 2147483647 };

/* enumerators as array dimensions: a global, a member, a typedef, a designator */
enum { EF_N = EF_Q3 + 1 };
uint32_t ef_g[EF_N * 2] = {[EF_N] = 7u, [EF_Q3 * 2] = 9u};
struct ef_rec { uint8_t b[EF_N]; uint32_t k; };
typedef uint16_t ef_row[EF_N + 1];

/* every enumerator through a value: each is the constant C gives it, an int */
uint64_t ef_values(uint32_t s)
{
    uint64_t k = s;
    k = k * 31u + (uint64_t)EF_Q + (uint64_t)EF_R * 3u + (uint64_t)EF_Q2 * 5u + (uint64_t)EF_R2 * 7u;
    k = k * 31u + (uint64_t)EF_Q3 + (uint64_t)EF_R3 * 3u + (uint64_t)EF_UQ * 5u;
    k = k * 31u + (uint64_t)EF_U + (uint64_t)EF_NEG * 2u + (uint64_t)EF_ULL * 4u + (uint64_t)EF_LL * 8u;
    k = k * 31u + (uint64_t)EF_U8 + (uint64_t)EF_U16 * 2u + (uint64_t)EF_EQ * 4u + (uint64_t)EF_LONG * 8u;
    k = k * 31u + (uint64_t)EF_HEXL + (uint64_t)EF_DEC * 3u + (uint64_t)EF_HEXP * 5u;
    k = k * 31u + (uint64_t)EF_SHL + (uint64_t)EF_SAR * 3u + (uint64_t)EF_SHR * 5u + (uint64_t)EF_USHL * 7u;
    k = k * 31u + (uint64_t)EF_WIDE + (uint64_t)EF_NARROW * 3u + (uint64_t)EF_SIGNBIT;
    k = k * 31u + (uint64_t)EF_C8 + (uint64_t)EF_CS8 * 3u + (uint64_t)EF_C16 * 5u + (uint64_t)EF_CS16 * 7u;
    k = k * 31u + (uint64_t)EF_CINT + (uint64_t)EF_CBOOL * 3u + (uint64_t)EF_CU32 + (uint64_t)EF_CL * 5u;
    k = k * 31u + (uint64_t)EF_CLL;
    k = k * 31u + (uint64_t)EF_SEL + (uint64_t)EF_SELU * 3u + (uint64_t)EF_SELN * 5u + (uint64_t)EF_SELD;
    k = k * 31u + (uint64_t)EF_AND + (uint64_t)EF_OR * 3u + (uint64_t)EF_DEAD * 5u + (uint64_t)EF_DEADOVF * 7u;
    k = k * 31u + (uint64_t)EF_LOGIC;
    k = k * 31u + (uint64_t)EF_LO + (uint64_t)EF_LO1 * 3u + (uint64_t)EF_LO2 * 5u + (uint64_t)EF_ZERO;
    k = k * 31u + (uint64_t)EF_MIN + (uint64_t)EF_MIN1 * 3u + (uint64_t)EF_MAX * 5u + (uint64_t)EF_N;
    return k;
}

/* an enumerator in a comparison and in arithmetic is an int */
int32_t ef_signed(int32_t v)
{
    int32_t k = v < EF_Q ? EF_LO : EF_LO2;
    k += v / EF_Q2 + v % EF_Q;
    return k + (EF_MIN1 < v) + (v > EF_SIGNBIT) * 2 - EF_CS16 / 4;
}

/* case labels fold as enumerators do; a switch on a signed value reaches the negative ones */
int32_t ef_switch(int32_t v)
{
    switch (v) {
    case -7 / 2:
        return 1;
    case -7 % 2:
        return 2;
    case ~0u > 5:
        return 3;
    case EF_Q2 * 4:
        return 4;
    case (int8_t)200:
        return 5;
    case 0 ? 1 : 2 ? EF_LO2 * 7 : 3:
        return 6;
    case 1 << 4:
        return 7;
    case (uint8_t)300:
        return 8;
    case EF_MIN:
        return 9;
    case EF_MAX:
        return 10;
    case 0 && 1 / 0:
        return 11;
    case 1 ? 2 : 1 / 0:
        return 12;
    default:
        return v;
    }
}

/* ... and convert to the type of the value switched on: -1 is UINT32_MAX for a uint32_t */
uint32_t ef_uswitch(uint32_t u)
{
    switch (u) {
    case -1:
        return 1u;
    case -7 / 2:
        return 3u;
    case (unsigned char)-1:
        return 4u;
    case EF_UQ:
        return 5u;
    default:
        return u;
    }
}

/* case labels of a 64-bit switch: exact past LLONG_MAX */
uint32_t ef_wswitch(uint64_t w)
{
    switch (w) {
    case 0xFFFFFFFFFFFFFFFFu:
        return 1u;
    case 1ull << 63:
        return 2u;
    case -2LL:
        return 3u;
    case (uint64_t)EF_MIN:
        return 4u;
    default:
        return (uint32_t)w;
    }
}

/* enumerators as array dimensions: fixed arrays, as in C -- a local, a member, a typedef, a global */
uint32_t ef_dims(uint32_t s)
{
    uint32_t a[EF_N];
    uint8_t b[EF_N * 2 + 1];
    struct ef_rec r;
    ef_row row;
    a[EF_N - 1] = s;
    b[EF_N * 2] = (uint8_t)s;
    r.b[EF_N - 1] = (uint8_t)(s >> 8);
    r.k = s;
    row[EF_N] = (uint16_t)s;
    uint32_t k = a[EF_N - 1] + b[EF_N * 2] + r.b[EF_N - 1] + r.k + row[EF_N] + ef_g[EF_N] + ef_g[EF_Q3 * 2];
    return k + (uint32_t)sizeof a + (uint32_t)sizeof b + (uint32_t)sizeof r + (uint32_t)sizeof row
           + (uint32_t)sizeof ef_g;
}

uint64_t ef_entry(uint32_t s)
{
    uint64_t k = ef_values(s) ^ (uint64_t)ef_signed((int32_t)(s >> 1) - 7);
    k += (uint64_t)ef_switch((int32_t)(s % 64u) - 60) + ef_uswitch(s) * 3u + ef_wswitch((uint64_t)s << 31);
    return k ^ ef_dims(s);
}
