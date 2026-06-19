/* Bitfield compound-assignment `r->field OP= bits` -- the read-modify a multi-bit field in place
 * pattern. It composes the bitfield read + write paths: read the field (c.bf.get), apply the binary
 * op, re-insert the masked result (c.bf.set), and store the unit back. For a volatile pointee the
 * unit accesses are ordered MMIO. */
struct reg { uint32_t en : 1; uint32_t mode : 3; uint32_t prio : 4; };

uint32_t tweak(volatile struct reg *r, uint32_t b)
{
    r->mode |= b;
    r->prio &= 3u;
    return r->mode;
}
