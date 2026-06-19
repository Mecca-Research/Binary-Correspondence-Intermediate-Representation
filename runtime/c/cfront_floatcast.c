/* int and float conversions: explicit casts -- int to float (float)i, int to double (double)i, the
 * narrowing (float)d, and float to int (int)f (truncation) -- plus the implicit usual-arithmetic
 * conversions in a mixed int and float expression. A cast to a floating type yields a float temp;
 * (int)f truncates toward zero. The emit renders the real C cast, so the resident backend performs the
 * conversion and the result stays Clang-behaviour-equivalent. */
float convert(int i, float f)
{
    float a = (float)i;
    int   t = (int)f;
    double d = (double)i * 2.0;
    return a + (float)t + (float)d;
}
