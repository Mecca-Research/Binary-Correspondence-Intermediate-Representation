/* CF-ENUMOBJ: an object of an enumerated type has the type the implementation makes the enumeration compatible with
 * (C11 6.7.2.2p4) -- on the System V targets GCC and Clang make one with no negative enumerator `unsigned int` and one
 * with a negative enumerator `int`; the MSVC ABI makes every one `int`. The enumeration constants stay `int` (6.4.4.3).
 * Both rails had typed every enum object `int`: `(c - 5) < 0` was 1 and `_Generic(c - 5, ...)` chose `int` where
 * Clang and GCC give 0 and `unsigned int` -- the same claim graph on both rails, so only the emit run against the
 * original showed it. Each function reads an enum object where the two types part: a comparison with 0 after a
 * subtraction, a widening conversion, `_Generic`, a division, a shift, a compound assignment, a 3-bit bit-field. */
#include <stdint.h>

enum col { RED, GREEN = 4, BLUE };   /* no negative enumerator: `unsigned int` on System V, `int` on MSVC */
enum sg { NEG = -1, POS = 1 };        /* a negative enumerator: `int` everywhere */
typedef enum { T_A = 2, T_B = 9 } tenum;   /* an untagged one, named by its typedef */
typedef enum col col_t;

static enum col gcol = GREEN;
enum sg gsg = NEG;

struct holder {
    enum col c;
    enum sg g;
    enum col bits : 3;   /* 4..7 read back as themselves where the enum is unsigned, negative where it is `int` */
};

/* a local: the subtraction, the comparison with 0, `_Generic`, a division and a shift in the enum's type */
static uint32_t eo_local(uint32_t s)
{
    enum col c = (s & 1u) ? GREEN : RED;
    uint32_t r = (uint32_t)((c - 5) < 0);
    r += _Generic(c - 5, int: 10u, unsigned int: 20u, default: 30u);
    r += (uint32_t)((c - 6) / 2 > 100) * 100u;
    r += (uint32_t)((c - 6) >> 30) * 1000u;
    return r;
}

/* file-scope objects, a static one stored from `s` and one of an enum with a negative enumerator */
static uint32_t eo_global(uint32_t s)
{
    gcol = (enum col)(s & 7u);
    uint32_t r = (uint32_t)((int64_t)(gcol - 5) < 0);
    r += (uint32_t)(gsg < 0) * 2u;
    r += _Generic(gcol, unsigned int: 4u, default: 8u);
    r += _Generic(gsg, int: 16u, default: 32u);
    return r;
}

/* a return type and a parameter type */
static enum col eo_pick(uint32_t s)
{
    return (s & 3u) == 3u ? BLUE : (s & 1u) ? GREEN : RED;
}

static int64_t eo_take(enum col c)
{
    return (int64_t)(c - 5) + (int64_t)c;
}

static uint32_t eo_calls(uint32_t s)
{
    int64_t w = eo_take(eo_pick(s)) + (int64_t)(eo_pick(s >> 1) - 5);
    return (uint32_t)(w >> 32) ^ (uint32_t)w;
}

/* members, an enum with a negative enumerator beside them, and a bit-field of the enum type */
static uint32_t eo_member(uint32_t s)
{
    struct holder h = {(enum col)(s & 7u), (s & 1u) ? NEG : POS, (enum col)(s & 7u)};
    uint32_t r = (uint32_t)((int64_t)(h.c - 5) < 0);
    r += (uint32_t)((h.g - 1) < 0) * 2u;
    r += (uint32_t)((int64_t)(h.bits - 5) < 0) * 4u;
    r += (uint32_t)((int64_t)h.bits + 8) * 8u;
    return r;
}

/* typedef names of an enum: one of a tagged enum, one of an untagged one */
static uint32_t eo_typedef(uint32_t s)
{
    col_t a = (col_t)(s & 7u);
    tenum b = (s & 1u) ? T_A : T_B;
    uint32_t r = (uint32_t)((int64_t)(a - 5) > 0);
    r += (uint32_t)((int64_t)(b - 5) > 0) * 2u;
    r += _Generic(b, unsigned int: 4u, default: 0u);
    r += _Generic(a, unsigned int: 8u, default: 0u);
    return r;
}

/* a conversion into the enum and out of it to a wider type; the enum's size */
static uint32_t eo_conv(uint32_t s)
{
    enum col c = (enum col)(0u - (s & 1u));   /* 0, or every bit set */
    int64_t w = c;                            /* widened from the enum's own type */
    return (uint32_t)(w >> 32) + (uint32_t)sizeof(enum col) + (uint32_t)sizeof c;
}

/* compound assignments and steps of an enum object */
static uint32_t eo_compound(uint32_t s)
{
    enum col c = RED;
    c -= (enum col)(s & 1u);   /* RED, or every bit set */
    int64_t w = c;
    c = (enum col)(s & 3u);
    c--;
    int64_t v = c;
    return (uint32_t)(w >> 40) ^ (uint32_t)(v >> 40) ^ (uint32_t)v;
}

/* a switch on an enum object, its case labels the enumerators */
static uint32_t eo_switch(uint32_t s)
{
    enum col c = eo_pick(s);
    switch (c) {
    case RED:
        return 1u;
    case GREEN:
        return 2u;
    case BLUE:
        return 3u;
    default:
        return 4u;
    }
}

/* a tag is no ordinary identifier (C11 6.2.3p1): an object named as the tag `col`, and an enumerator `shade` of one
 * enumeration beside the tag `shade` of another -- neither is the other */
enum shade { SH0, SH1 };
enum other { shade = 1 };
static uint32_t col = 7u;

static uint32_t eo_names(uint32_t s)
{
    enum shade h = (enum shade)(s & 1u);
    return col + (uint32_t)shade * 2u + (uint32_t)((int64_t)(h - 1) >> 40) * 4u;
}

/* the enum with a negative enumerator stays `int` */
static uint32_t eo_signed(uint32_t s)
{
    enum sg g = (s & 1u) ? NEG : POS;
    return (uint32_t)((int64_t)g < 0) + _Generic(g, int: 2u, default: 4u) + (uint32_t)(g - 2 < 0) * 8u;
}

uint32_t eo_entry(uint32_t s)
{
    return eo_local(s) + eo_global(s) * 3u + eo_calls(s) * 5u + eo_member(s) * 7u + eo_typedef(s) * 11u +
           eo_conv(s) * 13u + eo_compound(s) * 17u + eo_switch(s) * 19u + eo_signed(s) * 23u + eo_names(s) * 29u;
}
