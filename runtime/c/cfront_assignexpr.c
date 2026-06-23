/* Assignment as an EXPRESSION (#assignexpr): an assignment yields a value (the assigned value), so it can
 * appear as a sub-expression -- chained `a = b = c`, in a condition `if ((x = f()) > 0)` / `while ((n = n-1))`,
 * parenthesized `(a = v) + 1`, or compound `(a += 3) * 2`. The C twin previously had NO assignment in its
 * value grammar (only a statement form) and PARSE-errored on all of these, diverging from the oracle/Clang.
 * Now both rails support a NAMED-VARIABLE target (local/param/global) as an assignment-expression, mirroring
 * the statement forms and the oracle's _assign (right-associative; the value is the variable's new storage).
 * A member/array-lvalue assignment used as a value (`(p->x = v) + 1`) stays a follow-on -- both rails fall
 * back -- so this fixture uses only scalar variables. */

int assign_expr(int x, int y)
{
    int a, b, c;
    a = b = c = x;                       /* chained, right-associative: all get x */
    int sum = a + b + c;

    int d = (b = y) + 1;                 /* parenthesized assignment as a value */
    sum += d + b;

    int e = (a += 3) * 2;                /* compound assignment as a value */
    sum += e + a;

    int f = (c <<= 2) - 1;               /* shift-compound assignment as a value */
    sum += f + c;

    if ((a = x * 2) > 0)                 /* assignment in a condition */
        sum += a;
    else
        sum -= a;

    int n = (x & 7) + 4, t = 0;
    while ((n = n - 1) > 0)              /* assignment in a loop condition */
        t += n;
    sum += t + n;

    int g = 0;
    for (int i = 0; i < 4; i++)
        g = g + (b = b + i);             /* assignment-expression inside a larger expression (sequenced) */
    sum += g + b;

    return sum;
}
