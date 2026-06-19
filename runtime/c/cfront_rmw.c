/* MMIO register read-modify-write: `dev->reg OP= expr` -- the set/clear-control-bits driver idiom.
 * A compound assignment to a (volatile) struct member is a read of the member, a binary op, then a
 * store back -- the read + store are ordered volatile MMIO accesses; both rails emit the same
 * load / op / store sequence, Clang-behaviour-equivalent. */
struct ctl { uint32_t enable; uint32_t mode; };

uint32_t configure(volatile struct ctl *dev, uint32_t bits)
{
    dev->enable |= bits;        /* set bits */
    dev->mode &= 15u;           /* clear all but the low nibble */
    return dev->enable;
}
