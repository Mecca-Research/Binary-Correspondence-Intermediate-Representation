/* L3: pointers + arrays (GEP-equivalent indexing). Lowered by both rails. */
uint32_t l3_index(uint32_t *base, uint32_t i)
{
    uint32_t lo = base[i];
    uint32_t hi = base[i + 1];
    return lo + hi;
}
