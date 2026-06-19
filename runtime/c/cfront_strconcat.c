/* Adjacent string literals concatenate (C translation phase 6): `sizeof` folds across the pieces, and
 * indexing reads from the joined bytes. The pieces stay adjacent in the emit (separated by a space)
 * rather than being merged into one decoded run, so a hex/octal escape never absorbs the next piece's
 * leading digit (`"\x41" "2"` is 'A' then '2', never `\x412`). Clang-behaviour-equivalent. */
uint32_t cat(uint32_t i)
{
    uint32_t n = sizeof("abc" "de");        /* 6 = "abcde" + NUL                              */
    uint32_t t = sizeof("tab\t" "x");       /* 6 = 4 + 1 + NUL  (\t stays one byte, no merge) */
    uint32_t h = sizeof("\x41" "2");        /* 3 = 'A' + '2' + NUL  (the escape-boundary case) */
    uint32_t b = "hi " "there"[i % 8];      /* index the 8-char concatenation "hi there"      */
    return n + t + h + b + (i & 1);
}
