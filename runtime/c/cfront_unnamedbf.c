/* Unnamed & zero-width bitfields (#unnamedbf, C11 6.7.2.1): a bitfield with NO declarator -- `T : width;` --
 * positions the layout cursor but is not an accessible member. A non-zero unnamed bitfield `T : N` reserves N
 * padding bits (so the following field skips them); a ZERO-WIDTH `T : 0` forces the next bitfield to start at
 * the next storage-unit boundary of T. Crucially, an UNNAMED bitfield (padding or zero-width) does NOT raise
 * the struct's alignment -- only a NAMED bitfield's type does (Itanium/Clang) -- so `struct{char c; int :0;
 * char d;}` is align 1, size 5, with d at offset 4. Both rails reproduce this layout (the sizeof/offsetof
 * LAYOUT differential) and a read-modify-write of each NAMED field lands at the right offset. */

struct Reg {
    unsigned lo   : 5;        /* bits 0-4 */
    unsigned      : 3;        /* unnamed padding -> bits 5-7 skipped */
    unsigned hi   : 8;        /* bits 8-15 */
    unsigned      : 0;        /* zero-width -> `mid` starts a fresh storage unit (byte 4) */
    unsigned mid  : 12;       /* in the unit at byte 4 */
    unsigned char tail;       /* a plain member after the bitfield units */
};

struct Split {
    char     c;               /* offset 0 */
    int      : 0;             /* zero-width int: bumps the cursor to the next int boundary, NOT the struct align */
    char     d;               /* offset 4 (struct stays align 1, sizeof 5) */
};

int64_t unnamed_rmw(struct Reg *p, struct Split *q, int v)
{
    p->lo   = (unsigned)v;
    p->hi   = (unsigned)(v + 1);
    p->mid  = (unsigned)(v * 3);
    p->tail = (unsigned char)(v + 2);
    q->c    = (char)v;
    q->d    = (char)(v + 7);
    return (int64_t)p->lo + p->hi + p->mid + p->tail + q->c + q->d;
}
