/* C23 `[[reproducible]]` robustness (#reproducible / A1.3): the hint mixed with OTHER C23 attributes
 * in the same `[[ ... ]]` run -- a namespaced attribute (`gnu::const`) and the standard
 * `[[deprecated("msg")]]` with a string argument. The attribute scanner must consume the whole run,
 * act ONLY on `reproducible`/`unsequenced`, and DROP every attribute from the emit (so the C is
 * behaviour-equivalent under Clang). The entry `rp_entry` is the last function and carries the hint, so
 * the entry-keyed summary stays parity-identical; repro=2 (rp_helper + rp_entry). */
[[reproducible, gnu::const]] uint32_t rp_helper(uint32_t x)
{
    return (x * 2654435761u) >> 8;
}
[[deprecated("use rp_entry2"), reproducible]] uint32_t rp_entry(uint32_t a, uint32_t b)
{
    uint32_t h = rp_helper(a) + rp_helper(b);
    return h ^ (a + b);
}
