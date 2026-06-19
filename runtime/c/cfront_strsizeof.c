/* sizeof a string literal -> the char-array length including the NUL, with escape sequences decoded
 * (a simple \c, an octal \NNN, or a hex \xHH.. each one byte). The literal is not materialized -- only
 * its size is folded to a constant -- so both rails just count bytes identically (str_bytes). */
uint32_t strsizes(uint32_t x)
{
    uint32_t a = sizeof "hello";      /* 6  (5 chars + NUL) */
    uint32_t b = sizeof("");          /* 1  (just the NUL) */
    uint32_t t = sizeof "tab\there";  /* 9  (\t is one byte) */
    uint32_t h = sizeof "\x41\x42";   /* 3  (two hex-escaped bytes + NUL) */
    return a + b + t + h + (x & 1);
}
