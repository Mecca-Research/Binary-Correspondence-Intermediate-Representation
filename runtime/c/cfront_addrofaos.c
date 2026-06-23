/* Address-of an array-of-structs element FIELD inside a struct member (#addrofaos) -- `&s.arr[i].field` /
 * `&s->arr[i].field`, completing the address-of family. The field lands at `&base + member_off +
 * i*sizeof(elem) + field_off`, where the element STRIDE is the element STRUCT's size (not the field's) --
 * reusing the member-array row-major flatten plus the element struct's field offset (oracle `_lvalue`; twin
 * `member_arr_index` + the element-struct field lookup). (`&arr[i].field` on a PLAIN array/pointer base
 * stays a both-rails follow-on -- the twin's bare-array-param model doesn't reach it.) The entry is
 * POINTER-based (the harness can't seed a by-value struct with array members); the value-struct case rides
 * parity + the cross-target gate. Writes are IDEMPOTENT so the same-buffer equivalence harness is
 * deterministic. */

struct Cell { unsigned tag, val; };
struct Bank { struct Cell cell[4]; unsigned base; };

static void setk(unsigned *p, unsigned k) { *p = k; }

unsigned aos_value(struct Bank b, unsigned i)        /* &value-struct array-of-structs element field */
{
    setk(&b.cell[i & 3u].val, 7u);
    return b.cell[i & 3u].val;
}

unsigned aos_entry(struct Bank *bp, unsigned i)      /* the entry: &pointer-struct array-of-structs element field */
{
    setk(&bp->cell[i & 3u].val, 9u);
    setk(&bp->cell[i & 3u].tag, 3u);
    return bp->cell[i & 3u].val + bp->cell[i & 3u].tag + bp->base;
}
