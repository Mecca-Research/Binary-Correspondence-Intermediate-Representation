/* _Alignof / alignof (C11 / C23): folds to a compile-time constant -- the type's alignment, read
 * from the shared scalar/struct layout model (the operand is never evaluated, like sizeof). Only the
 * type-name form is valid C. */
struct mixed { uint8_t a; uint32_t b; };
uint32_t alignments(uint32_t x)
{
    uint32_t wa = _Alignof(uint32_t);      /* 4 */
    uint32_t ba = alignof(uint8_t);        /* 1 */
    uint32_t sa = _Alignof(struct mixed);  /* 4 (the widest member) */
    return x + wa + ba + sa;
}
