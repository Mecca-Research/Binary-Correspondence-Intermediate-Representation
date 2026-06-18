/* sizeof -- folds to a compile-time integer constant (the type / operand's static size); the
 * operand is not evaluated. Both rails compute the same sizes (scalar table + struct layout). */
struct pair { uint32_t lo; uint32_t hi; };

uint32_t sizes(uint32_t x, uint32_t *p)
{
    uint32_t s1 = sizeof(uint32_t);     /* 4 */
    uint32_t s2 = sizeof(struct pair);  /* 8 */
    uint32_t s3 = sizeof(uint16_t);     /* 2 */
    uint32_t sv = sizeof(x);            /* 4  (x is uint32_t) */
    uint32_t sp = sizeof(p);            /* 8  (p is a pointer) */
    return s1 + s2 + s3 + sv + sp + (x & 7);
}
