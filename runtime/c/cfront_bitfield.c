/* MMIO bitfield write: setting a named bitfield within a (volatile) register -- the multi-bit
 * control-field driver pattern. `r->field = v` reads the storage unit, inserts the masked bits in
 * place (c.bf.set: `(old & ~(mask<<off)) | ((v & mask) << off)`), then stores the unit back; for a
 * volatile pointee the read + store are ordered MMIO accesses. Bitfield reads (c.bf.get) already
 * worked; this completes the write side. */
struct creg { uint32_t en : 1; uint32_t mode : 3; uint32_t prio : 4; };

uint32_t set_mode(volatile struct creg *r, uint32_t m)
{
    r->mode = m;
    return r->prio;
}
