/* Compound assignment `name OP= expr` (the register / bit-manipulation idiom): each compound
 * assignment desugars to `name = name OP expr` -- a binary op plus a copy into the named storage --
 * identically on both rails, so it is Clang-behaviour-equivalent. */
uint32_t pack_bits(uint32_t x, uint32_t mask)
{
    uint32_t r = x;
    r |= mask;
    r &= 0xFFu;
    r += 3u;
    r ^= mask;
    return r;
}
