/* Translation phase 3 (comment stripping). Comments in tricky positions -- inline between tokens, a
 * star-space-slash `* /` inside a block comment (NOT a close), a line comment, and an inert
 * #define INERT 1
 * directive that lives inside a comment so it must never fire. Both rails strip every comment to a
 * single space (keeping the spanned newlines) before the preprocessor, so tokens never glue and the
 * comment's punctuation is never re-tokenized. Parity + Clang behaviour-equivalence. */
unsigned commented(unsigned a /* p1 */, unsigned b)
{
    // a line comment between statements
    unsigned c = a /* a tricky * / still in the comment */ + b;
    return c /* trailing */ ;
}
