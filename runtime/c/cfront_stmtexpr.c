/* GCC statement expressions (#stmtexpr): `({ s1; ...; e; })` -- a compound statement, in its own scope,
 * whose value is the last statement (an expression statement). The macro-hygiene idiom (a safe max(), a
 * local temporary). The prefix statements lower inline before the use site; the result is the last
 * expression's value. The twin has no AST, so it lowers the prefix statements in place, then rolls the
 * LAST statement back (the speculative undo typeof uses) and re-parses it as the value expression. Both
 * rails agree and the emit is Clang-behaviour-equivalent. */

/* a local temporary + a final value (the canonical use) */
int se_simple(int x)   { return ({ int y = x * 2; y + 3; }); }
/* the safe-max macro idiom (each operand evaluated once into a temporary) */
int se_max(int a, int b){ return ({ int m = a > b ? a : b; m * 10 + (a < b); }); }
/* embedded in a larger expression -- `({...}) + 100` */
int se_embed(int x)    { int z = ({ int t = x + 1; t * t; }) + 100; return z; }
/* control flow (a loop) inside the statement expression */
int se_loop(int x)     { return ({ int s = 0; for (int i = 1; i <= x; i++) s += i; s; }); }
/* nested statement expressions */
int se_nest(int x)     { return ({ int a = ({ int b = x; b + 1; }); a * 2; }); }
/* the statement expression is its own scope: the inner `y` shadows, then the outer `y` is restored */
int se_scope(int x)    { int y = 100; int r = ({ int y = x; y * 2; }); return r + y; }
/* a VOID statement expression: the last item is a control-flow statement, so the value is discarded */
int se_void(int x)     { int n = 0; ({ if (x & 1) n = x * 2; else { n = x + 1; } }); return n; }

/* Clang's GNU rule: a bare terminal bitfield retains its DECLARED type once the statement expression
 * ceases to be a bitfield expression. Ordinary reads and enclosing expressions still promote normally. */
struct se_bits { unsigned u : 3; signed int s : 3; };
unsigned se_bf_declared(unsigned v) { struct se_bits b; b.u=v; return -({ b.u; }); }
int se_bf_regular(unsigned v)       { struct se_bits b; b.u=v; return -(b.u); }
int se_bf_cast(unsigned v)          { struct se_bits b; b.u=v; return -({ (int)b.u; }); }
int se_bf_expr(unsigned v)          { struct se_bits b; b.u=v; return -({ b.u + 0; }); }
int se_bf_comma(unsigned v)         { struct se_bits b; b.u=v; return -({ (0, b.u); }); }
unsigned se_bf_paren(unsigned v)    { struct se_bits b; b.u=v; return -({ (b.u); }); }
unsigned se_bf_arrow(unsigned v)    { struct se_bits b, *p=&b; p->u=v; return -({ p->u; }); }
int se_bf_signed(int v)             { struct se_bits b; b.s=v; return -({ b.s; }); }
