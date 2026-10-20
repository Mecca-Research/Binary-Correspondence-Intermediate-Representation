"""PX-5 (LAYOUT): the layouts the native trainer's kernels keep, stated as GF(2) maps.

`bcir.kbcir.linear_layout` is the algebra; this module states, in it, every storage layout the
trainer's tensor unit (runtime/c/bcir_tensor.c) writes for its GEMMs -- each one a GF(2) tile under
an integer grid, the tail past the operand's shape the padding its packer zeroes -- so the C's
packers can be held to a statement rather than to a second spelling of their own index arithmetic
(`bcir/tests/test_native_layouts.py` reads the packers' outputs back and holds every element):

* the bf16 contract's A tile (`A_TILE`): 16 rows x 32 depths, row-major -- the AMX A tile register,
  64 bytes a row -- the in-tile offset's five low bits the depth, the next four the row;
* its B tile (`B_TILE`): 16 depth PAIRS x 16 columns, a column's two depths of a pair adjacent --
  TDPBF16PS's B operand -- offset bit 0 the depth's low bit, bits 1-4 the column, bits 5-8 the
  depth's other four: a bit permutation (the VNNI interleave), not a row-major tile;
* the panels and packs (`a_pack`, `b_pack`, and the panels BF16-PANELS writes, `a_panel`,
  `b_panel`, `at_panel`): those tiles under the grid the C orders them in -- a 16-row block's depth
  tiles one after another, block after block -- every element past the operand's rows, columns or
  depth +0.0;
* the conversion a writer makes when it derives a second panel from the first (`panel_second`):
  `b_panel(...)^-1 . a_panel(...)`, element for element;
* the AVX-512 blocked GEMM's B tile (`kb_tile`): a depth row of 48 floats -- three 16-lane vectors,
  the register image `kb512_block` loads -- lanes the GF(2) part (four bits), the three vectors the
  grid's (radix 3), the columns past the call's `nj` zero (the masked loads' padding);
* the 16 x 16 register transpose (`TRANSPOSE16`): (register, lane) to (lane, register), the
  conversion `kb512_transpose16` makes -- and its four stages (`TRANSPOSE16_STAGES`), each a bit
  permutation of an element's (register, lane): the pairs unpacked, the quads shuffled, the 128-bit
  lanes exchanged between registers four apart, then eight apart -- composed, the transpose;
* the A broadcast (`BROADCAST`): the 16 lanes of `_mm512_set1_ps` hold one element -- the kernel is
  every lane bit, the preimage of the element all 16 lanes.

Statements only: nothing here generates a layout (the roadmap's order: state, hold, then generate).
"""

from __future__ import annotations

from ...kbcir.linear_layout import LinearLayout, TiledLayout

__all__ = [
    "A_TILE",
    "B_TILE",
    "BROADCAST",
    "TRANSPOSE16",
    "TRANSPOSE16_STAGES",
    "a_pack",
    "a_panel",
    "at_panel",
    "b_pack",
    "b_panel",
    "kb_tile",
    "panel_second",
]

#: The bf16 A tile: in-tile offset (512 bf16) -> (row, depth): bits 0-4 the depth, 5-8 the row.
A_TILE = LinearLayout.from_bits(
    (("offset", 512),),
    (("row", 16), ("depth", 32)),
    {("offset", b): [("depth", b)] if b < 5 else [("row", b - 5)] for b in range(9)},
)
#: The bf16 B tile (VNNI): offset -> (depth, col): bit 0 the depth's low bit, bits 1-4 the column,
#: bits 5-8 the depth's bits 1-4.
B_TILE = LinearLayout.from_bits(
    (("offset", 512),),
    (("depth", 32), ("col", 16)),
    {("offset", b): [("depth", 0)] if b == 0 else [("col", b - 1)] if b < 5 else [("depth", b - 4)]
     for b in range(9)},
)  # fmt: skip
#: A block of 16 registers of 16 lanes: an element's register q and lane t.
_REGS = (("q", 16), ("t", 16))


def _stage(moves: str) -> LinearLayout:
    """A bit permutation of an element's (register q, lane t) in a block of 16 registers: `moves`
    names each bit that moves (`q0>t0`: the register's bit 0 becomes the lane's bit 0); every other
    bit stays where it is."""
    moved = dict(move.split(">") for move in moves.split())
    wiring = {}
    for name, _size in _REGS:
        for b in range(4):
            to = moved.get(f"{name}{b}", f"{name}{b}")
            wiring[(name, b)] = [(to[0], int(to[1:]))]
    return LinearLayout.from_bits(_REGS, _REGS, wiring)


#: The 16 x 16 register transpose: (register q, lane t) of the source to (register t, lane q).
TRANSPOSE16 = _stage("q0>t0 q1>t1 q2>t2 q3>t3 t0>q0 t1>q1 t2>q2 t3>q3")
#: `kb512_transpose16`'s stages in order, each an element's (register, lane) before the stage to its
#: (register, lane) after it: `_mm512_unpacklo/hi_ps` of register pairs; `_mm512_shuffle_ps` 0x44
#: and 0xEE of registers two apart; `_mm512_shuffle_f32x4` 0x88 and 0xDD of registers four apart;
#: then of registers eight apart. Each a 3-cycle of bits; composed, `TRANSPOSE16`.
TRANSPOSE16_STAGES = (
    _stage("q0>t0 t0>t1 t1>q0"),
    _stage("q0>q1 q1>t1 t1>q0"),
    _stage("q2>t3 t2>q2 t3>t2"),
    _stage("q3>t3 t2>q3 t3>t2"),
)
#: The A broadcast: the 16 lanes of a register holding one element (row, depth) of A.
BROADCAST = LinearLayout.from_bits(
    (("lane", 16),), (("row", 1), ("depth", 1)), {("lane", b): [] for b in range(4)}
)


def _blocks(extent: int, size: int) -> int:
    return -(-extent // size)


def a_pack(rows: int, depth: int) -> TiledLayout:
    """The bf16 A pack (and A panel) of a rows x depth operand: A tiles, a 16-row block's depth tiles
    in order, block after block; rows padded to 32 (the AMX instance's 32 x 32 accumulators), depth
    to 32, the padding +0.0."""
    return TiledLayout(
        A_TILE,
        (("row", 2 * _blocks(rows, 32)), ("depth", _blocks(depth, 32))),
        ("row", "depth"),
        (("row", rows), ("depth", depth)),
    )


def b_pack(depth: int, cols: int) -> TiledLayout:
    """The bf16 B pack (and B panel) of a depth x cols operand: B tiles, a 16-column block's depth
    tiles in order, block after block; columns padded to 32, depth to 32, the padding +0.0."""
    return TiledLayout(
        B_TILE,
        (("depth", _blocks(depth, 32)), ("col", 2 * _blocks(cols, 32))),
        ("col", "depth"),
        (("depth", depth), ("col", cols)),
    )


def a_panel(rows: int, cols: int) -> TiledLayout:
    """An r x c tensor's panel as A (BF16-PANELS): rows r, depth c."""
    return a_pack(rows, cols)


def b_panel(rows: int, cols: int) -> TiledLayout:
    """An r x c tensor's panel as B: depth r, columns c."""
    return b_pack(rows, cols)


def at_panel(rows: int, cols: int) -> TiledLayout:
    """An r x c tensor's panel as A^T: rows c, depth r -- A's tiles over the transpose."""
    return a_pack(cols, rows)


def panel_second(rows: int, cols: int, *, transposed: bool):
    """The conversion a panel writer makes from an r x c tensor's A panel to its second panel -- its
    B panel, or with `transposed` its A^T panel: an A-panel offset to the second panel's offset
    holding the same element (`b^-1 . a`, through the tensor's coordinates)."""
    first = a_panel(rows, cols)
    second = at_panel(rows, cols) if transposed else b_panel(rows, cols)

    def convert(offset: int) -> int:
        x = first.coords(offset)
        if transposed:
            return second.offset(row=x["depth"], depth=x["row"])
        return second.offset(depth=x["row"], col=x["depth"])

    return convert


def kb_tile(depths: int, cols: int) -> TiledLayout:
    """The AVX-512 blocked GEMM's packed B tile for `depths` depths of a column block of `cols`
    (at most 48) columns: each depth a row of three 16-lane vectors, the lanes the GF(2) part, the
    columns past `cols` zero (the masked loads' padding)."""
    if not 0 < cols <= 48:
        raise ValueError("a column block of 1..48 columns")
    lane = LinearLayout.from_bits((("offset", 16),), (("depth", 1), ("col", 16)),
                                  {("offset", b): [("col", b)] for b in range(4)})  # fmt: skip
    return TiledLayout(lane, (("depth", depths), ("col", 3)), ("depth", "col"),
                       (("depth", depths), ("col", cols)))  # fmt: skip
