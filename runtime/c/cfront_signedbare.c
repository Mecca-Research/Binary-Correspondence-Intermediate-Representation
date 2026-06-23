/* Bare `signed` == `signed int` (#signedbare) -- `signed` with no base width keyword. The oracle handled
 * it, but the C twin treated `signed` as a pure modifier (it set the sign, never the size / `seen`), so a
 * bare `signed x` left the type unfinished and the twin FELL BACK -- a both-rails divergence. The twin now
 * mirrors `unsigned`: bare `signed` finalizes as `signed int`, while a following base (`signed char` /
 * `signed long`) still overrides the width. Negative values are built from unsigned params by subtraction
 * (like #signedcmp), so the SIGNEDNESS is genuinely exercised vs Clang: a signed `< 0` compare, an
 * arithmetic (sign-replicating) right shift, and signed division (rounds toward zero), where a wrongly-
 * unsigned `signed` would diverge. */

unsigned sbare_abs(unsigned a, unsigned b)
{
    signed x = (signed)(a % 4000u) - (signed)(b % 4000u);   /* a genuinely-negative signed int */
    signed m = x < 0 ? -x : x;                               /* |x| via a signed `< 0` compare */
    return (unsigned)m;
}

unsigned sbare_shift(unsigned a)
{
    signed x = (signed)(a % 8000u) - 4000;                   /* in [-4000, 3999] */
    return (unsigned)(x >> 2);                               /* arithmetic (sign-replicating) right shift */
}

unsigned sbare_entry(unsigned a, unsigned b)
{
    signed x = (signed)(a % 1000u) - (signed)(b % 1000u);
    signed y = (signed)(b % 100u) + 1;                       /* in [1, 100] -- never 0, so no div-by-zero */
    signed q = x / y;                                        /* signed division (toward zero) */
    signed cmp = (x < y) ? 1 : 0;                            /* signed compare */
    return (unsigned)(q + cmp + (x >> 1));
}
