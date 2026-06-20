/* Native struct member arrays (#memberarray): `s.arr[i]` -- a 1-D array member (a DMA buffer, a FIFO, a
 * packet body). The element lands at `&s + member_offset + i * elem_size` on both rails, carried as a
 * c.load / c.store with the index AND a (member byte offset, element size) imm -- distinct from a plain
 * `base[idx]` (no imm) and a plain member access (no index). Read, write, compound, and a narrowing
 * uint8 element store all match Clang. (Multi-dimensional member arrays and pointer members stay a
 * documented LLVM-fallback; a compound with a *computed* index re-evaluates it on the oracle -- a
 * pre-existing plain-array difference -- so the compound below uses a simple local-variable index.) */
struct Packet { unsigned hdr; unsigned body[6]; };
unsigned pkt_sum(unsigned i, unsigned a) {
  struct Packet p;
  p.hdr = a;
  for (unsigned t = 0u; t < 6u; t++) p.body[t] = a + t * 2u;   /* fill all -- simple-index stores */
  unsigned k = i % 6u;
  p.body[k] += a & 3u;                                         /* compound, simple (local) index */
  return p.body[i % 6u] + p.body[(i + 1u) % 6u] + p.hdr;       /* computed-index reads */
}
struct Buf { unsigned char data[8]; unsigned len; };
unsigned buf_pack(unsigned i, unsigned a) {
  struct Buf b;
  b.len = a;
  for (unsigned t = 0u; t < 8u; t++) b.data[t] = (unsigned char)(a + t);   /* uint8 narrowing stores */
  return (unsigned)b.data[i & 7u] + b.len;
}
/* A 2-D member array `g.cell[r][c]` (a small grid / matrix in a register block): the row-major index
 * flattens to `r*4 + c`, the element lands at `&g + cell_off + (r*4+c)*4`. */
struct Grid { unsigned cell[3][4]; unsigned tag; };
unsigned grid_pick(unsigned i, unsigned a) {
  struct Grid g;
  g.tag = a;
  for (unsigned r = 0u; r < 3u; r++)
    for (unsigned c = 0u; c < 4u; c++)
      g.cell[r][c] = a + r * 4u + c;                              /* 2-D member-array stores */
  return g.cell[i % 3u][i & 3u] + g.cell[(i + 1u) % 3u][0] + g.tag;   /* 2-D reads */
}
