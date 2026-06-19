/* Wide / UTF literal prefixes (L, u, U, u8) on character and string literals. A prefixed character
 * constant keeps the (ASCII) code-point value; a prefixed string literal has the element width of its
 * character type, so `sizeof` scales (wchar_t / char32_t = 4, char16_t = 2, char / u8 = 1). The prefix
 * is preserved in the emit, so Clang sees the same literal -- behaviour-equivalent. */
uint32_t widelit(uint32_t x)
{
    uint32_t nl = L'\n';           /* 10  (code point; wchar_t constant)  */
    uint32_t a  = u'A';            /* 65  (code point; char16_t constant) */
    uint32_t sl = sizeof(L"hi");   /* 12  = 3 wchar_t                      */
    uint32_t su = sizeof(U"hi");   /* 12  = 3 char32_t                     */
    uint32_t s2 = sizeof(u"hi");   /* 6   = 3 char16_t                     */
    uint32_t s8 = sizeof(u8"hi");  /* 3   = 3 char                         */
    return nl + a + sl + su + s2 + s8 + (x & 1);
}
