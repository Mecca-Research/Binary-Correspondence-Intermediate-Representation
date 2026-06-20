/* Hexadecimal floating literals (C99/C23): a hex mantissa + a mandatory *binary* exponent
 * (0x1.8p3 == 12.0, 0x1p-1f == 0.5f). The binary-exponent sign survives preprocessing (the
 * pp-number spans p[+-]); the typed c.fconst keeps the f/F suffix at float width and a bare hex
 * literal (0xFF, no 'p') stays an integer. Parity with the oracle + Clang behaviour-equivalence. */
float hexf(float a)
{
    float h = a * 0x1p-1f;          /* 0.5f */
    return h + 0x1.4p2f;            /* 5.0f */
}

double hexd(double x, double y)
{
    double w = x * 0x1.8p3 + y;     /* 12.0 */
    return w - 0x1p-2;              /* 0.25 */
}
