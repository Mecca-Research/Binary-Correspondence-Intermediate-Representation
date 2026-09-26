/* Volatile indexed register accesses at every width: `p[i]` through a `volatile uint8_t *`,
 * `volatile uint16_t *` and `volatile uint32_t *`. Each is one access of exactly its element, at
 * p + i * sizeof *p -- the twin once wrote the byte store as a 4-byte store at p + 4*i, and the oracle
 * dropped the qualifier from its emitted C. The stores are observed through a PLAIN alias of the same
 * bytes, so a wider or misplaced write changes the fingerprint the entry returns; the loads read their
 * element back at its own width. */
uint32_t volatile_width(uint8_t *buf, uint32_t i, uint32_t v)
{
    volatile uint8_t *r8 = (volatile uint8_t *)buf;
    volatile uint16_t *r16 = (volatile uint16_t *)buf;
    volatile uint32_t *r32 = (volatile uint32_t *)buf;
    const uint8_t *plain = buf;
    uint32_t k = i & 3u;
    r8[k] = (uint8_t)v;                    /* bytes 0..3: one byte */
    r16[4u + k] = (uint16_t)(v >> 5);      /* bytes 8..15: one half-word */
    r32[4u + k] = v ^ 0x5a5a5a5au;         /* bytes 16..31: one word */
    uint32_t fp = 0u;
    for (uint32_t j = 0u; j < 40u; j++)
        fp = fp * 31u + plain[j];
    return fp + r8[k ^ 1u] + r16[4u + (k ^ 2u)] + r32[4u + (k ^ 3u)];
}
