/* Complex struct members (#complexmember) -- a struct field of `_Complex` type. The C twin previously
 * typed a member load/store by SIZE: a 16-byte `double _Complex` field was read into a 16-byte `long double`
 * real temp (and stored by staging through `long double`), a SILENT miscompile (the real part survived, the
 * imaginary part was lost), because the twin's field record carried `is_float` but no `is_complex`. The
 * field record now carries `is_complex`, so a `_Complex` member loads into a complex temp and stores as the
 * complex pair -- matching the oracle, which already typed the member correctly. Exercised: member read,
 * native complex arithmetic on members, a member STORE (struct build), an `l`-free <complex.h> call on a
 * member, and a complex member round-tripped through a by-value struct returned from an in-unit call. */
#include <complex.h>

struct C { double _Complex z; double w; };

struct C cmake(double _Complex z, double w)      /* member STORE: build a struct with a complex field */
{
    struct C c;
    c.z = z;                                     /* store a complex value into a complex member */
    c.w = w;
    return c;
}

double cmag(struct C c)
{
    return cabs(c.z) + c.w;                       /* a <complex.h> call on a complex member -> real */
}

double _Complex centry(struct C a, struct C b, double _Complex v)
{
    struct C m = cmake(v, a.w);                   /* build via member stores, returned by value, re-read */
    return a.z * b.z + m.z + b.z / a.z;           /* read complex members + native complex arithmetic */
}
