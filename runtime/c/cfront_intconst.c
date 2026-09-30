/* CF-INTCONST: an integer constant is its exact value, in its C11 6.4.4.1 type, on both rails -- in every base
 * (hexadecimal, octal, binary, decimal) and with every suffix (`u`, `l`, `ll` and their combinations), at the
 * edges of its type -- and a file-scope initializer folds in its operands' own types. The C twin had kept a
 * constant past LLONG_MAX as LLONG_MAX (`0xFFFFFFFFFFFFFFFFu` was 9223372036854775807) and read an octal constant
 * as decimal (`017` was 17); the oracle folded a global's initializer with unbounded integers (`-7 / 2` rendered
 * -4, `~0u` -1). `ic_types` and `ic_suffix` read each constant's type through a comparison with -1, which
 * converts to an unsigned type, and through `sizeof`; the oracle's linkable emit renders the globals. */
#include <stdint.h>

/* file-scope initializers: each operation in its operands' type, the result converted to the global's */
int32_t ic_gq = -7 / 2;                /* -3: C truncates toward zero */
int32_t ic_gr = -7 % 2;                /* -1: a remainder takes the dividend's sign */
int32_t ic_gm = 7 % -2;                /* 1 */
int64_t ic_gd = -9223372036854775807 / 4;
uint64_t ic_gw = ~0u;                  /* 4294967295: `~` of an unsigned int */
uint64_t ic_gww = ~0ull;               /* 18446744073709551615 */
int64_t ic_gn = ~0;                    /* -1 */
uint32_t ic_gsh = 1u << 31;
uint64_t ic_gsh64 = 1ull << 63;
int64_t ic_gsr = -8 >> 1;              /* -4: an arithmetic shift, on every target */
uint32_t ic_gusr = 0x80000000 >> 31;   /* 1: `0x80000000` is an unsigned int */
int64_t ic_gneg = -0x80000000;         /* 2147483648: `-` of an unsigned int wraps */
int64_t ic_glong = -2147483648;        /* -2147483648: `2147483648` is a long (a long long where long is 32 bits) */
uint64_t ic_gwrap = 0xFFFFFFFF + 1;    /* 0: unsigned int arithmetic wraps */
int64_t ic_gwide = 4294967295 + 1;     /* 4294967296: a decimal constant past INT_MAX is a long */
int32_t ic_gcmp = -1 < 0u;             /* 0: compared as unsigned int */
int32_t ic_gcmpl = -1 < 0x100000000;   /* 1: compared as a long */
int32_t ic_gsel = -1 < 0u ? 5 : 6;     /* 6 */
uint64_t ic_goct = 017 + 0777 + 0b101; /* 531 */
uint64_t ic_gdiv = 0xFFFFFFFFFFFFFFFFu / 3u;
int64_t ic_gmin = -9223372036854775807 - 1;
int32_t ic_gcast = (int8_t)300;        /* 44 */
uint32_t ic_gsz = sizeof(uint64_t) * 3u;
int16_t ic_garr[4] = {-7 / 2, 7 / -2, -7 % 3, ~0u >> 28};

/* hexadecimal: past INT_MAX an unsigned int, past UINT_MAX a long, past LONG_MAX an unsigned long */
uint64_t ic_hex(uint32_t s)
{
    uint64_t k = 0xFFFFFFFFFFFFFFFFu ^ s;
    k += 0x8000000000000000 >> (s & 63u);
    k ^= 0x7FFFFFFFFFFFFFFF + (uint64_t)s;
    k += 0xFFFFFFFFFFFFFFFF - s;
    k ^= 0x8000000000000001ULL >> 60;
    k += 0xFFFFFFFF + s;               /* an unsigned int: wraps at 2^32 */
    k ^= 0x100000000 + s;              /* a long */
    return k + 0xFFFFFFFFFFFFFFFFull / (s | 1u);
}

/* octal: its digits read in base 8 -- `017` is 15 */
uint64_t ic_octal(uint32_t s)
{
    uint64_t k = 017 + 0777 * 2 + 010 * s + 00 + 07u + 017L + 0123LL;
    k += 01777777777777777777777;      /* 2^64 - 1: an unsigned long */
    k ^= 0777777777777777777777;       /* 2^63 - 1: a long */
    k += 01000000000000000000000u;     /* 2^63 */
    k ^= 037777777777 + s;             /* 0xFFFFFFFF: an unsigned int, wraps */
    return k;
}

/* binary, up to 64 digits */
uint64_t ic_binary(uint32_t s)
{
    uint64_t k = 0b1010 + 0B11111111 + 0b0 + 0b1u + 0b11L + 0b100LL;
    k += 0b1111111111111111111111111111111111111111111111111111111111111111;   /* 2^64 - 1 */
    k ^= 0b1000000000000000000000000000000000000000000000000000000000000000;   /* 2^63 */
    k += 0b11111111111111111111111111111111 + s;   /* 0xFFFFFFFF: an unsigned int, wraps */
    return k * (s | 1u);
}

/* decimal: signed until `u`; past INT_MAX a long */
uint64_t ic_decimal(uint32_t s)
{
    uint64_t k = 2147483647 + (uint64_t)s;
    k += 2147483648 + 4294967295 + 4294967296;
    k ^= 9223372036854775807;
    k += 9223372036854775808u + 18446744073709551615u + 18446744073709551615ULL + s;
    return k ^ (4294967295u + s);      /* an unsigned int: wraps */
}

/* each constant's type, read through `-1 < c`: -1 converts to an unsigned type and compares above it */
uint32_t ic_types(uint32_t s)
{
    uint32_t k = 0u;
    k = k * 2u + (-1 < 0x7FFFFFFF);            /* int */
    k = k * 2u + (-1 < 0x80000000);            /* unsigned int */
    k = k * 2u + (-1 < 0xFFFFFFFF);            /* unsigned int */
    k = k * 2u + (-1 < 0x100000000);           /* long */
    k = k * 2u + (-1 < 0x7FFFFFFFFFFFFFFF);    /* long */
    k = k * 2u + (-1 < 0x8000000000000000);    /* unsigned long */
    k = k * 2u + (-1 < 2147483647);            /* int */
    k = k * 2u + (-1 < 2147483648);            /* long */
    k = k * 2u + (-1 < 4294967295);            /* long */
    k = k * 2u + (-1 < 9223372036854775807);   /* long */
    k = k * 2u + (-1 < 4294967295u);           /* unsigned int */
    k = k * 2u + (-1 < 4294967296u);           /* unsigned long */
    k = k * 2u + (-1 < 017);                   /* int */
    k = k * 2u + (-1 < 017777777777);          /* int */
    k = k * 2u + (-1 < 020000000000);          /* unsigned int */
    k = k * 2u + (-1 < 037777777777);          /* unsigned int */
    k = k * 2u + (-1 < 040000000000);          /* long */
    k = k * 2u + (-1 < 01000000000000000000000);   /* unsigned long */
    k = k * 2u + (-1 < 0b1111111111111111111111111111111);    /* int */
    k = k * 2u + (-1 < 0b10000000000000000000000000000000);   /* unsigned int */
    k = k * 2u + (-1 < 0b100000000000000000000000000000000);  /* long */
    return k ^ s;
}

/* each suffix spelling: its type's size, and whether -1 converts above it */
uint64_t ic_suffix(uint32_t s)
{
    uint64_t k = s;
    k = k * 31u + sizeof(1u) * 2u + (-1 < 1u);
    k = k * 31u + sizeof(1U) * 2u + (-1 < 1U);
    k = k * 31u + sizeof(1l) * 2u + (-1 < 1l);
    k = k * 31u + sizeof(1L) * 2u + (-1 < 1L);
    k = k * 31u + sizeof(1ll) * 2u + (-1 < 1ll);
    k = k * 31u + sizeof(1LL) * 2u + (-1 < 1LL);
    k = k * 31u + sizeof(1ul) * 2u + (-1 < 1ul);
    k = k * 31u + sizeof(1uL) * 2u + (-1 < 1uL);
    k = k * 31u + sizeof(1Ul) * 2u + (-1 < 1Ul);
    k = k * 31u + sizeof(1UL) * 2u + (-1 < 1UL);
    k = k * 31u + sizeof(1lu) * 2u + (-1 < 1lu);
    k = k * 31u + sizeof(1lU) * 2u + (-1 < 1lU);
    k = k * 31u + sizeof(1Lu) * 2u + (-1 < 1Lu);
    k = k * 31u + sizeof(1LU) * 2u + (-1 < 1LU);
    k = k * 31u + sizeof(1ull) * 2u + (-1 < 1ull);
    k = k * 31u + sizeof(1uLL) * 2u + (-1 < 1uLL);
    k = k * 31u + sizeof(1Ull) * 2u + (-1 < 1Ull);
    k = k * 31u + sizeof(1ULL) * 2u + (-1 < 1ULL);
    k = k * 31u + sizeof(1llu) * 2u + (-1 < 1llu);
    k = k * 31u + sizeof(1llU) * 2u + (-1 < 1llU);
    k = k * 31u + sizeof(1LLu) * 2u + (-1 < 1LLu);
    k = k * 31u + sizeof(1LLU) * 2u + (-1 < 1LLU);
    k = k * 31u + sizeof(0x1L) * 2u + (-1 < 0xFFFFFFFFL);
    k = k * 31u + sizeof(0x1LL) * 2u + (-1 < 0xFFFFFFFFFFFFFFFFLL);
    return k;
}

/* the edges in arithmetic: the twin had computed with LLONG_MAX for each constant past it */
uint64_t ic_edges(uint64_t x)
{
    uint64_t k = (x * 0x9E3779B97F4A7C15u) ^ (x / 0xFFFFFFFFFFFFFFFEu) ^ (x % 0x8000000000000001u);
    k ^= 0xFFFFFFFFFFFFFFFFu - x;
    return k + (x < 0x8000000000000000u) + (x >= 18446744073709551615u);
}

/* case labels in every base */
uint32_t ic_cases(uint32_t s)
{
    switch (s & 0x1Fu) {
    case 010:
        return 1u;
    case 0x10:
        return 2u;
    case 0b11:
        return 3u;
    case 017:
        return 4u;
    case 0B11111:
        return 5u;
    default:
        return s;
    }
}

/* an array's dimensions in octal and hexadecimal */
uint32_t ic_dims(uint32_t s)
{
    uint8_t a[010];
    uint16_t b[0x10];
    a[1] = (uint8_t)s;
    b[2] = (uint16_t)(s >> 8);
    return (uint32_t)sizeof a + (uint32_t)sizeof b + a[1] + b[2];
}

/* the file-scope initializers, as the globals hold them */
uint64_t ic_globals(uint32_t s)
{
    uint64_t k = (uint64_t)ic_gq + (uint64_t)ic_gr * 3u + (uint64_t)ic_gm * 5u + (uint64_t)ic_gd;
    k ^= ic_gw + ic_gww + (uint64_t)ic_gn + ic_gsh + ic_gsh64 + (uint64_t)ic_gsr + ic_gusr;
    k += (uint64_t)ic_gneg + (uint64_t)ic_glong + ic_gwrap + (uint64_t)ic_gwide + (uint64_t)ic_gcmp;
    k ^= (uint64_t)ic_gcmpl + (uint64_t)ic_gsel + ic_goct + ic_gdiv + (uint64_t)ic_gmin;
    k += (uint64_t)ic_gcast + ic_gsz + (uint64_t)ic_garr[0] + (uint64_t)ic_garr[3];
    return k ^ s;
}

uint64_t ic_entry(uint32_t s)
{
    uint64_t k = ic_hex(s) ^ ic_octal(s) ^ ic_binary(s) ^ ic_decimal(s);
    k += ic_types(s) + ic_suffix(s) + ic_edges(((uint64_t)s << 32) | s);
    return k ^ ic_cases(s) ^ ic_dims(s) ^ ic_globals(s);
}
