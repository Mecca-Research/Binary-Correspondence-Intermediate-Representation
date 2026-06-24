/* The comma operator (#comma): `a, b` evaluates `a` for its side effects, discards its value, and yields
 * `b`. Only the PRIMARY parenthesized form is the operator -- call arguments and initializer elements keep
 * their SEPARATOR comma (`f(a, b)`, `{a, b}`), so those are unaffected. Each function yields a deterministic
 * value (never an address), so the equivalence harness matches. */

unsigned cseq(unsigned a, unsigned b)
{
    return (a + 1u, b * 2u);                 /* discards a+1, yields b*2 */
}

unsigned cside(unsigned a, unsigned b)
{
    unsigned t = 0u;
    unsigned r = (t = a + 5u, t + b);        /* the assignment to t runs first, then yields t + b */
    return r;
}

unsigned cchain(unsigned a, unsigned b, unsigned c)
{
    return (a & 7u, b | 1u, c + 3u);         /* a chain: only c+3 is yielded */
}
