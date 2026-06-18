/* Integer casts (type)expr: truncating casts to narrower fixed-width types (the register-packing /
 * field-masking pattern), plus an identity cast. In the uint32-tracked value model a cast to a
 * narrower unsigned type masks -- assigning the cast back into a 32-bit temp zero-extends -- which is
 * exactly Clang's integer-promotion result; both rails lower it to a `c.cast:<type>` claim and emit
 * `(type)expr`. */
uint32_t pack(uint32_t a, uint32_t b)
{
    uint32_t lo = (uint8_t)a;             /* keep the low 8 bits */
    uint32_t mid = (uint16_t)(a >> 4);    /* keep 16 bits of a shifted field */
    uint32_t wide = (uint32_t)b;          /* identity cast */
    return lo + mid + (wide & 0xFFu);
}
