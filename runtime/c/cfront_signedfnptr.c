/* A function-pointer member call whose target returns a SIGNED type now carries the return's signedness onto
 * the call RESULT (#signedfnptr), so a downstream arithmetic `>>` / `< 0` / `(long)`-widen sign-extends
 * correctly. The member (`c.call.imember`) and indirect call results were hardcoded UNSIGNED on both rails --
 * a both-rails miscompile. Self-contained: each funcptr struct member is set to a fixture function returning a
 * NEGATIVE value, then used sign-sensitively, so the fix is observable and Clang-equivalent. */

static int  negate(int x) { return -x - 1; }         /* a negative int for x >= 0 */
static int  dbl(int x)    { return -2 * x; }
static long lwiden(int x) { return -(long)x - 1; }    /* a negative long */

struct DI { int  (*op)(int); };
struct DL { long (*op)(int); };

int sf_shift(int x) {                                 /* signed int return -> arithmetic right shift */
    struct DI d;
    d.op = negate;
    return d.op(x) >> 1;                              /* a logical shift (unsigned) would be wrong */
}

int sf_abs(int x) {                                   /* signed int return -> the abs idiom (r < 0) */
    struct DI d;
    d.op = dbl;
    int r = d.op(x);
    return r < 0 ? -r : r;
}

long sf_wide(int x) {                                 /* a WIDE (long) funcptr return keeps its width */
    struct DL d;
    d.op = lwiden;
    return d.op(x) - 1;
}
