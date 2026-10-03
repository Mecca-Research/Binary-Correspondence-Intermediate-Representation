/* CF-CONSTEXPR2: the constant expressions CF-ENUMFOLD left, on both rails. Case labels each converted to the switch's
 * promoted type, none equal (C11 6.8.4.2p3, p5); character constants read once, a prefixed one of its prefix's type
 * (6.4.4.4p11, C23 6.4.4.5); an enumeration declared in a block, its constants and tag scoped to it (6.2.1p4); an
 * enumerator, `sizeof` and `_Alignof` of a type-name and a floating constant cast to an integer type wherever C takes
 * an integer constant expression -- a bit-field's width, `_BitInt(N)`, a compound literal's dimension, a row
 * pointer's (6.6p6); zero-length and flexible array members, which occupy no bytes. Each had been refused on one rail
 * or both, or read as a literal only -- the twin had laid `uint32_t a : W;` out as a member as wide as its type, and
 * a `uint32_t t[0]` member as one element (`sizeof` 8 for Clang's 4). Every value here is the same on the 64-bit
 * Linux hosts the suite runs the emits on; those the target's character types decide are the per-target test's. */
#include <stdint.h>

/* character constants: a multi-character one packs its last four bytes big-endian into an int; a prefixed one has
 * its prefix's type -- `u8` an unsigned char, `u` char16_t, `U` char32_t -- and the arithmetic it takes part in is
 * that type's: `U'a' - 98` is unsigned, `u'a' - 98` an int */
enum ce_char { CE_A = 'a', CE_AB = 'ab', CE_LAST4 = 'abcde', CE_ALL1 = '\377\377\377\377', CE_ESC = '\n' + '\'' * 2,
               CE_U8 = u8'\xff', CE_U16 = u'\xffff', CE_U32 = U'\xffffffff' > 0, CE_U8NEG = u8'a' - 98 < 0,
               CE_U16NEG = u'a' - 98 < 0, CE_U32NEG = U'a' - 98 < 0, CE_LBIG = L'\xff' };

/* sizeof and _Alignof of type-names -- a struct, a member array, a pointer -- and floating constants cast to integer
 * types, each rounded to its own type first (16777217.0f is 16777216), then truncated toward zero */
struct ce_pair { uint8_t tag; uint32_t v[3]; double d; };
enum ce_size { CE_SZ = sizeof(struct ce_pair), CE_AL = _Alignof(struct ce_pair), CE_ARR = sizeof(uint32_t[3][2]),
               CE_PTR = sizeof(uint8_t *) * 2, CE_F1 = (int)2.75, CE_F2 = (uint8_t)255.9, CE_F3 = (_Bool)0.5,
               CE_F4 = (int)16777217.0f - 16777216, CE_F5 = (int)0x1.8p3, CE_F6 = (int)(1.5e2), CE_F7 = (_Bool)0.0,
               CE_F8 = (uint16_t)65535.99, CE_MIX = (int)(sizeof(double) * 3) + (int)1e1 };

/* an enumerator as a bit-field's width, a compound literal's dimension and a row pointer's (a `_BitInt`'s width is
 * the per-target test's: GCC 13 has no `_BitInt`) */
enum ce_dims { CE_W = 3, CE_N = 4 };
struct ce_bits { uint32_t lo : CE_W; uint32_t mid : CE_W * 2; uint32_t : CE_W - 1; uint32_t hi : sizeof(uint16_t) * 4; };

/* zero-length and flexible array members: no bytes of their own, an element of their type where they are indexed */
typedef uint32_t ce_row[0];
struct ce_zhdr { uint32_t n; ce_row tail; };
struct ce_fhdr { uint8_t c; uint64_t tail[]; };

/* a switch on a uint8_t promotes to int: 255 and -1 are two labels; on a uint32_t, 4294967295u and -2 are too */
uint32_t ce_switch8(uint8_t v)
{
    switch (v) {
    case 255:
        return 1u;
    case -1:
        return 2u;
    case 'a':
        return 3u;
    case u8'b':
        return 4u;
    default:
        return 5u;
    }
}

uint32_t ce_switch32(uint32_t v)
{
    switch (v) {
    case 4294967295u:
        return 10u;
    case -2:
        return 20u;
    case CE_SZ:
        return 30u;
    case (int)7.9:
        return 40u;
    default:
        return v & 7u;
    }
}

/* prefixed character constants in run-time arithmetic: each its prefix's type */
uint32_t ce_chars(uint32_t x)
{
    uint32_t a = x * U'a';
    int32_t b = (int32_t)(x & 255u) - u8'\xff';
    int32_t c = (int32_t)(x & 255u) - u'b';
    return a + (uint32_t)b + (uint32_t)c + (uint32_t)(x < u'\xffff') + (uint32_t)CE_ALL1 + (uint32_t)CE_LAST4 +
           (uint32_t)CE_U32 + (uint32_t)CE_U8NEG * 2u + (uint32_t)CE_U16NEG * 4u + (uint32_t)CE_U32NEG * 8u +
           (uint32_t)CE_LBIG + (uint32_t)CE_ESC + (uint32_t)CE_AB + (uint32_t)CE_U16;
}

/* an enumeration declared in a block: its constants and its tag end with the block, each hiding an outer name of
 * its own from its enumerator on, and a local declared after it hiding it in turn */
uint32_t ce_block(uint32_t s)
{
    uint32_t CE_N = s & 7u;                        /* hides the file-scope CE_N */
    enum ce_local { CL_A = 3, CL_B = CL_A * 2 };
    uint32_t a[CL_B];
    a[CL_A] = s;
    enum ce_local tagged = CL_B;
    uint32_t r = a[CL_A] + (uint32_t)tagged + CE_N;
    {
        enum { CE_N = 9, CL_A = CE_N + 1 };        /* hides the local CE_N and the outer CL_A */
        r += (uint32_t)CE_N * 100u + (uint32_t)CL_A;
        {
            uint32_t CL_A = s >> 4;                /* a local hides the block's enumerator again */
            r += CL_A;
        }
        r += (uint32_t)CL_A * 1000u;
    }
    return r + (uint32_t)CL_A + CE_N;              /* the outer CL_A and the local CE_N again */
}

/* enumerators and sizeof as widths: the struct as Clang lays it out, each field truncated to its width */
uint32_t ce_bitfields(uint32_t s)
{
    struct ce_bits b;
    b.lo = s;
    b.mid = s >> 3;
    b.hi = s >> 9;
    return b.lo + b.mid * 8u + b.hi * 512u + (uint32_t)sizeof(struct ce_bits) * 100000u;
}

/* a compound literal of an enumerator's dimension, a row pointer of one */
static uint32_t ce_rowsum(uint32_t (*rows)[CE_N], uint32_t i)
{
    return rows[i & 1u][0] + rows[i & 1u][CE_N - 1];
}

uint32_t ce_literals(uint32_t s)
{
    uint32_t *p = (uint32_t[CE_N]){s, s + 1u, s + 2u, s + 3u};
    uint32_t m[2][CE_N] = {{s, 1u, 2u, 3u}, {4u, 5u, 6u, s ^ 9u}};
    uint32_t q[(int)2.5] = {s, s * 3u};
    return p[CE_N - 1] + ce_rowsum(m, s) + q[1] + (uint32_t)(sizeof q / sizeof q[0]);
}

/* zero-length and flexible members: the struct as Clang lays it out, the member indexed in place (the test's driver
 * gives it the storage) */
uint32_t ce_zero(const struct ce_zhdr *z, const struct ce_fhdr *f, uint32_t i)
{
    return z->n + z->tail[i & 1u] + (uint32_t)f->tail[i & 1u] + f->c + (uint32_t)sizeof(struct ce_zhdr) * 100u +
           (uint32_t)sizeof(struct ce_fhdr) * 1000u;
}

/* the entry: every function above from one value but `ce_zero` */
uint32_t constexpr2(uint32_t s)
{
    return ce_switch8((uint8_t)s) + ce_switch32(s) * 3u + ce_chars(s) + ce_block(s) + ce_bitfields(s) +
           ce_literals(s) + (uint32_t)CE_SZ + (uint32_t)CE_AL +
           (uint32_t)CE_ARR + (uint32_t)CE_PTR + (uint32_t)(CE_F1 + CE_F2 + CE_F3 + CE_F4 + CE_F5 + CE_F6 + CE_F7) +
           (uint32_t)CE_F8 + (uint32_t)CE_MIX + (uint32_t)CE_A;
}
