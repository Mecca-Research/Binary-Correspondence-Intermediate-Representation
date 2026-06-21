/* Faithful char types (#chartypes): C has THREE distinct one-byte character types and the emit now
 * spells each faithfully, so the output is behaviour-equivalent on EVERY target -- not just where plain
 * `char` happens to be signed:
 *   - plain `char`     -> `char`           (signedness is implementation-defined: signed on x86, UNSIGNED on ARM)
 *   - `signed char`    -> always signed    (`signed char` on the oracle, int8_t on the twin)
 *   - `unsigned char`  -> always unsigned  (`unsigned char` / uint8_t)
 * Before, the oracle collapsed `signed char` -> `char` (so it zero-extended a negative pointee on ARM) and
 * the twin emitted int8_t for plain `char` (so it sign-extended on ARM) -- the two rails disagreed with the
 * source, and each other, on ARM. Now both keep the three distinct. Driven by a differential run under BOTH
 * -fsigned-char and -funsigned-char, so plain char's platform sign is exercised both ways. */

int      ct_plain_deref(char *p)          { return *p >> 1; }          /* plain char: platform-defined shift */
int      ct_signed_deref(signed char *p)  { return *p >> 1; }          /* always an arithmetic (sign) shift */
int      ct_unsigned_deref(unsigned char *p){ return *p >> 1; }        /* always a logical shift */
int      ct_plain_cmp(char c)             { return (c < 0) ? 100 : -100; }   /* plain char compare (platform) */
int      ct_signed_cmp(signed char c)     { return (c < 0) ? 100 : -100; }   /* always can be negative */
int      ct_plain_div(char *p)            { return *p / 3 + *p % 3; }   /* plain char divide/remainder */
int      ct_signed_div(signed char *p)    { return *p / 3 + *p % 3; }   /* always signed divide */
unsigned ct_unsigned_div(unsigned char *p){ return *p / 3u; }
char     ct_roundtrip(char *p)            { char x = *p; return x; }    /* a plain char local + return */
int      ct_plain_widen(char *p)          { return (int)(*p) + 1000; }  /* the platform widening of a char */
