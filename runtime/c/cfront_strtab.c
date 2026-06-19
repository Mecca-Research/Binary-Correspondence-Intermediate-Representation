/* String-literal table: a literal longer than the old 32-byte resource-name cap now emits its full
 * spelling (the C twin keeps it in a side table, not the resource name), and identical literals in a
 * function share one read-only global (dedup). Both are observable only in the emit / resource set --
 * the claim graph summary is unchanged -- and stay Clang-behaviour-equivalent. */
uint32_t pick(uint32_t i)
{
    uint32_t a = "the quick brown fox jumps over the lazy dog"[i % 43];   /* > 32 bytes: full spelling */
    uint32_t b = "shared marker"[i & 7];     /* "shared marker" ...            */
    uint32_t c = "shared marker"[i & 3];     /* ... appears twice -> one global */
    return a + b + c;
}
