/* Plain `char` struct members (#charmember): a member declared plain `char` (no signed/unsigned) has
 * IMPLEMENTATION-DEFINED signedness -- signed on x86-64, UNSIGNED on AArch64. The C twin must read such a
 * member back as `char` (impl-defined sign), NOT as `int8_t` (always signed): reading it as int8_t
 * sign-extends a high-bit value on AArch64 where the source zero-extends, so the emit diverged from Clang on
 * ARM. The field table now carries is_plain_char and the member/element load emits `char`, matching the
 * oracle and Clang on EVERY target. Exercised as a direct member, a member array element, and a nested-struct
 * member; `signed char` / `unsigned char` keep their fixed (int8_t / uint8_t) spelling for contrast. */

struct Inner { char ic; int x; };
struct Chars {
    int          tag;
    char         c;          /* plain char: impl-defined sign */
    signed char  sc;         /* always int8_t */
    unsigned char uc;        /* always uint8_t */
    char         buf[4];     /* a plain-char member array */
    struct Inner in;         /* a nested struct with a plain-char member */
};

long charmember_rmw(struct Chars *p, int v)
{
    p->tag   = v;
    p->c     = (char)(v * 3);          /* may land a high-bit (negative-on-x86 / >=128-on-arm) value */
    p->sc    = (signed char)(v - 5);
    p->uc    = (unsigned char)(v + 7);
    p->buf[0] = (char)v;
    p->buf[1] = (char)(v + 40);
    p->buf[2] = (char)(v - 40);
    p->buf[3] = (char)(v * 9);
    p->in.ic = (char)(v + 100);
    p->in.x  = v;
    return (long)p->tag + p->c + p->sc + p->uc
         + p->buf[0] + p->buf[1] + p->buf[2] + p->buf[3]
         + p->in.ic + p->in.x;
}
