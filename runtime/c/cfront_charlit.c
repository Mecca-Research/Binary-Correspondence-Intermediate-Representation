/* Character constants: a single character is its byte value (a signed char), an escape sequence
 * decodes to one byte (simple \c, octal \NNN, hex \xHH), and a multi-character constant 'AB' packs
 * big-endian like Clang/GCC. Each is an `int` constant -- a c.const claim -- used in ordinary
 * integer arithmetic, so the emit renders the folded value and stays behaviour-equal to Clang. */
uint32_t classify(uint32_t x)
{
    uint32_t a  = 'A';        /* 65                          */
    uint32_t nl = '\n';       /* 10  (simple escape)         */
    uint32_t z  = '\x7a';     /* 122 ('z', hex escape)       */
    uint32_t o  = '\101';     /* 65  ('A', octal escape)     */
    uint32_t ab = 'AB';       /* 0x4142 = 16706 (multichar)  */
    return a + nl + z + o + ab + (x & 1);
}
