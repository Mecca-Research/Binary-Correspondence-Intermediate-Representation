/* String literals as values: a literal lowers to an anonymous read-only char[] global, and using it
 * decays to a const char* pointer. Indexing reads a byte; the bare literal is the pointer. The
 * emitter renders a reference to the anonymous global as the inline literal (Clang-equivalent), so no
 * synthesized global declaration is needed. (The literal's spelling is short -- the C twin carries it
 * in the resource-name field.) */
uint32_t strpick(uint32_t i)
{
    return "ABCD"[i & 3];          /* index a string literal -> the i-th byte */
}
