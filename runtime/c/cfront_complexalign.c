/* A scalar member sits at its ABI alignment -- the oracle's `CType.align` -- on both rails (CF-CALIGN). A
 * `_Complex` member aligns to its ELEMENT: `float _Complex` to 4, `double _Complex` to 8, `long double
 * _Complex` to the long double's (16 on LP64). An `_Atomic` member takes the ABI's atomic promotion: a type
 * no wider than 16 bytes on a 64-bit target aligns to its size. The twin aligned every scalar member to its
 * whole size, so a `double _Complex` after an 8-byte member landed at 16 instead of 8 and every member after
 * it moved with it; `sizeof` and `_Alignof` took the same wrong answers. It also sized a typedef'd complex
 * twice (`cf` below was 16 bytes), and neither rail promoted an `_Atomic` member. Every member is written
 * before it is read, and every value, size and alignment is folded into the result. The 16-byte atomic
 * member is placed but never accessed: its access is a libatomic call the harness does not link. */
#include <stdint.h>

typedef float _Complex cf;

struct Mix {
    float _Complex z;           /* 0 */
    double _Complex w;          /* 8 -- the twin put it at 16 */
    uint32_t tag;               /* 24 */
};
struct Ld {
    uint8_t c;
    long double _Complex l;     /* the long double's alignment */
    uint16_t h;
};
struct Td {
    uint8_t c;
    cf z;                       /* a typedef'd complex: 8 bytes at 4 -- the twin sized it 16 */
    uint16_t h;
};
struct Am {
    uint8_t c;
    _Atomic float _Complex z;   /* promoted: at 8 */
    _Atomic double _Complex w;  /* promoted: at 16 on a 64-bit target; placed, not accessed */
    uint16_t h;                 /* 32 */
};

uint32_t complex_align(struct Mix *m, struct Ld *d, struct Td *t, struct Am *a, double _Complex w, float x)
{
    m->z = x;
    m->w = w;
    m->tag = 7u;
    d->c = 3u;
    d->l = w;
    d->h = 9u;
    t->c = 1u;
    t->z = x;
    t->h = 11u;
    a->c = 2u;
    a->z = x;
    a->h = 13u;
    cf az = a->z;
    uint32_t same = (uint32_t)(__real__ m->z == x) + 2u * (uint32_t)(__real__ m->w == __real__ w)
                    + 4u * (uint32_t)(__imag__ m->w == __imag__ w) + 8u * (uint32_t)(__real__ d->l == __real__ w)
                    + 16u * (uint32_t)(__imag__ d->l == __imag__ w) + 32u * (uint32_t)(__real__ t->z == x)
                    + 64u * (uint32_t)(__real__ az == x);
    uint32_t layout = (uint32_t)(_Alignof(double _Complex) + 3u * _Alignof(float _Complex)
                                 + 5u * _Alignof(long double _Complex) + 7u * sizeof(struct Mix)
                                 + 11u * sizeof(struct Ld) + 13u * sizeof(struct Td) + 17u * sizeof(struct Am)
                                 + 19u * _Alignof(_Atomic double _Complex) + 23u * sizeof(cf));
    return same + 128u * (m->tag + d->c + d->h + t->c + t->h + a->c + a->h) + 8192u * layout;
}
