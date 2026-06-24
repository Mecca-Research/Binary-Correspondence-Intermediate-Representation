/* Computed goto (#computedgoto): the GNU label-as-value `&&L` (a `void *`) and the indirect `goto *p`. A
 * void* holds a taken label address and `goto *p` dispatches to it. Both rails lower to the GNU forms, which
 * Clang compiles. Deterministic + bounded (no UB / unbounded loops). */

unsigned dispatch(unsigned x) {                  /* pick a label by a runtime value */
    void *p = &&odd;
    if ((x & 1u) == 0u) p = &&even;
    goto *p;
even: return x * 2u;
odd:  return x * 3u + 1u;
}

unsigned jumptable(unsigned i) {                 /* a local jump table indexed by a value */
    void *t[3];
    t[0] = &&a;
    t[1] = &&b;
    t[2] = &&c;
    goto *t[i % 3u];
a: return 100u;
b: return 200u;
c: return 300u;
}

unsigned threaded(unsigned n) {                  /* a computed-goto loop, bounded by a mask */
    unsigned s = 0u, i = 0u;
    void *back = &&body;
body:
    if (i < (n & 7u)) {
        s += i;
        i++;
        goto *back;
    }
    return s;
}
