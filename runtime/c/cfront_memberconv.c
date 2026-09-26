/* A store the emit spells as a byte copy -- a member, a member-array element, a field of an array of
 * structs, a bitfield, a store through a pointer, an initializer's member or element -- converts the value
 * to the slot's declared type, as C's assignment and initialization do (C23 6.5.17.2, 6.7.10): an integer
 * stored into a float slot becomes that number, a float stored into an integer slot is truncated toward
 * zero. Both rails once chose the stored bytes from the VALUE's type, so `s->f = v` copied the integer's
 * bits into the float and `s->si = x` the float's into the integer, a float stored into a bitfield did not
 * compile, and `(s->si += x)` yielded the unconverted sum. Every value is folded into the result, and every
 * slot is written before it is read, so the original and the emitted twin see the same memory. */
#include <stdint.h>

struct M {
    uint32_t tag;
    float f;
    double d;
    int32_t si;
    float arr[4];
    int32_t iarr[2];
    _Bool b;
    uint32_t lo : 4;
    uint32_t hi : 12;
};
struct E {
    float x;
    uint32_t n;
};
struct In {
    float g;
    _Bool on;
    uint32_t w : 10;
};
struct Out {
    int32_t k;
    struct In in;
    float row[3];
};
struct Link {
    struct M *m;
};

/* initializers: positional, designated, nested, and a two-dimensional local array */
static int32_t inits(int32_t neg, float x)
{
    struct Out o = { neg, { neg, x, x }, { 1, neg, 3 } };
    struct Out p = { .in.g = neg, .in.w = x, .row[1] = neg, .k = x };
    float grid[2][2] = { { 1, neg }, { neg, 4 } };
    return (int32_t)(o.in.g * 2.0f) + o.in.on + (int32_t)o.in.w + (int32_t)(o.row[1] * 3.0f)
           + (int32_t)o.row[2] + (int32_t)(p.in.g * 5.0f) + (int32_t)p.in.w + (int32_t)p.row[1] + p.k
           + (int32_t)(grid[0][1] + grid[1][0] * 2.0f + grid[1][1]);
}

uint32_t member_conv(struct M *s, struct E *e, float *pf, uint32_t u, float x)
{
    struct M loc;
    struct Link lk = { s };
    int32_t neg = (int32_t)u - 150;   /* -150 .. 49 */
    s->f = neg;                       /* int -> float member */
    s->d = u;                         /* unsigned -> double member */
    s->si = x - 500.0f;               /* float -> signed member, toward zero */
    s->tag = x;                       /* float -> unsigned member */
    s->arr[u & 3u] = neg;             /* int -> float member-array element */
    s->iarr[u & 1u] = x;              /* float -> int member-array element */
    s->iarr[u & 1u] += x;             /* int += float is float, converted back */
    e[u & 1u].x = neg;                /* int -> float field of an array of structs */
    e[u & 1u].n = x;                  /* float -> unsigned field of an array of structs */
    e[u & 1u].n += x;
    *pf = neg;                        /* int -> float through a pointer */
    s->b = x;                         /* float -> _Bool: nonzero is 1 */
    s->hi = x;                        /* float -> a bitfield */
    lk.m->lo = x / 100.0f;            /* float -> a bitfield, through a pointer member */
    loc.f = u;                        /* `.` on a local */
    loc.si = x;
    int32_t pv = (int32_t)((*pf = neg) * 2.0f);    /* a stored value, as a value */
    int32_t mv = (s->si = x);
    int32_t cv = (s->si += x);                     /* the converted sum, not the float */
    s->f += u & 7u;                                /* float += unsigned stays float */
    *(float *)(void *)(pf + 1) = neg;              /* int -> float through a cast pointer */
    int32_t sum = (int32_t)(s->f * 4.0f) + (int32_t)s->d + s->si + (int32_t)(s->arr[u & 3u] * 2.0f)
                  + s->iarr[u & 1u] + (int32_t)(e[u & 1u].x * 3.0f) + (int32_t)(*pf) + (int32_t)pf[1]
                  + (int32_t)loc.f + loc.si + pv + mv + cv + inits(neg, x);
    return (uint32_t)sum + s->tag + e[u & 1u].n + s->b + s->hi + s->lo;
}
