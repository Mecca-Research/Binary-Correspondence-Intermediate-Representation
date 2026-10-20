"""PX-5 (LAYOUT): the trainer's kernel layouts held to the packers that keep them.

`bcir.hosted.models.native_layouts` states, as GF(2) maps under integer grids, the storage layouts
the tensor unit writes for its GEMMs. Each witness runs a packer -- the panel writers the library
exports, and in a build defining BCIR_DT_LAYOUT_PROBE the bf16 packs of every instance the host
has, the AVX-512 B tile and its 16 x 16 transpose -- on an operand whose elements name their own
coordinates, reads back what it wrote, and holds every element to the statement: the offset holds
the element the layout says it holds, and every padding offset holds zero. The algebra's own laws
are `test_linear_layout`'s; here the statements meet the C.
"""

import ctypes as ct
import tempfile
from pathlib import Path

from bcir.hosted.models.native_layouts import (
    A_TILE,
    B_TILE,
    BROADCAST,
    TRANSPOSE16,
    TRANSPOSE16_STAGES,
    a_pack,
    a_panel,
    at_panel,
    b_pack,
    b_panel,
    kb_tile,
    panel_second,
)
from bcir.kbcir.linear_layout import LinearLayout
from bcir.tests import native_decoder_fixtures as nf

#: The host's C compiler, or None: the witnesses' one capability (a fault sweep `--require`s it).
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()
_U16 = ct.POINTER(ct.c_uint16)
_F32 = ct.POINTER(ct.c_float)


_BUILT: list = []  # the probe's library and the directory that holds it, once built


def _library():
    """The layout probe's library (`native_decoder_fixtures.layout_library`), built once a process
    and shared by the witnesses that run C; None without a compiler or outside a source checkout --
    a documented skip."""
    if not _BUILT:
        held = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        path = nf.layout_library(Path(held.name) / "probe")
        _BUILT.extend([None if path is None else ct.CDLL(str(path)), held])
    return _BUILT[0]


def _bf16_value(bits: int) -> float:
    """A bfloat16 element as the binary32 it widens to."""
    return ct.c_float.from_buffer_copy((bits << 16).to_bytes(4, "little")).value


def _decode(buf, scale: float = 1.0) -> list[int | None]:
    """Each element of a bf16 buffer as the coordinate code it carries (value / scale - 1), or None
    for +0.0 -- the padding. An element the packer never wrote (the buffer is lent as 0xFFFF, a NaN)
    or any other value is a failure here, not a code."""
    out = []
    for o, bits in enumerate(buf):
        assert bits != 0xFFFF, f"offset {o} never written"
        if bits == 0:
            out.append(None)
            continue
        v = _bf16_value(bits) / scale
        assert v == v and v >= 1 and v == int(v), (o, hex(bits), v)
        out.append(int(v) - 1)
    return out


def test_the_tiles_are_the_amx_registers():
    """The A tile is a row-major 16 x 32 bfloat16 tile -- offset bits 0-4 the depth, 5-8 the row,
    the AMX tile register's 64-byte rows -- and the B tile the VNNI pair layout -- bit 0 the
    depth's low bit, bits 1-4 the column, 5-8 the depth's other bits -- each a bijection whose
    inverse is the packer's index function; the broadcast's kernel is every lane bit."""
    for o in range(512):
        assert A_TILE.apply(offset=o) == {"row": o >> 5, "depth": o & 31}
        b = B_TILE.apply(offset=o)
        assert (b["depth"] >> 1) * 32 + 2 * b["col"] + (b["depth"] & 1) == o
    assert A_TILE.bijective and B_TILE.bijective
    assert A_TILE.convert_to(A_TILE) == LinearLayout.identity((("offset", 512),))
    one, kernel = BROADCAST.preimage(row=0, depth=0)
    assert one == {"lane": 0} and len(kernel) == 4


def test_the_transpose_is_its_four_stages():
    """kb512_transpose16's four stages -- unpacklo/hi_ps, shuffle_ps 0x44/0xEE, shuffle_f32x4
    0x88/0xDD twice -- each a bit permutation of (register, lane), compose to the transpose."""
    total = TRANSPOSE16_STAGES[0]
    for stage in TRANSPOSE16_STAGES[1:]:
        total = stage.compose(total)
    assert total == TRANSPOSE16
    assert all(stage.bijective for stage in TRANSPOSE16_STAGES)


def _writer(lib, name, argtypes):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = argtypes, ct.c_int
    return fn


def _pad32(n: int) -> int:
    return -(-n // 32) * 32


def _instances(lib) -> list[int]:
    """The bf16 instances this build and host offer (1 the portable C, 4 AVX-512, 8 AMX), the
    portable C first: every host has it."""
    lib.bcir_tensor_mm_bf16_variants.restype = ct.c_int
    variants = [v for v in (1, 4, 8) if lib.bcir_tensor_mm_bf16_variants() & v]
    assert variants[0] == 1, variants
    return variants


def _panel_codes(lib, conv, swiglu, v2, rows, cols, code):
    """The panels a writer instance (`conv`; None the library's entry, the host's widest) makes of
    an r x c tensor whose elements carry `code`: the SiLU product's A and B panels at x = 32, u =
    code (SiLU(32) = 32 in binary32, so each element is 32 x code), or the SwiGLU VJP's adjoints as
    A and A^T at g = 32, u = 1, dy = code (dgate = code, dup = 32 x code) -- each decoded."""
    n = rows * cols
    big, one = (ct.c_float * n)(*([32.0] * n)), (ct.c_float * n)(*([1.0] * n))
    dy = (ct.c_float * n)(*code)
    bufs = [(ct.c_uint16 * n)(*([0xFFFF] * n)) for _ in range(4)]
    if not swiglu:
        u = (ct.c_float * n)(*code)
        if conv is None:
            fn = _writer(lib, "bcir_tensor_silu_mul_panels_pool",
                         [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_size_t, _F32, _F32, _U16, _U16,
                          ct.POINTER(ct.c_int)])  # fmt: skip
            assert fn(None, v2, rows, cols, big, u, bufs[0], bufs[1], ct.byref(ct.c_int(0))) == 1
        else:
            assert _probe_panels(lib)(conv, 0, v2, rows, cols, big, u, None, bufs[0], bufs[1],
                                      None, None) == 1  # fmt: skip
        return {"a": _decode(bufs[0], 32.0), "b": _decode(bufs[1], 32.0)}
    if conv is None:
        fn = _writer(lib, "bcir_tensor_swiglu_backward_panels_pool",
                     [ct.c_void_p, ct.c_int, ct.c_size_t, ct.c_size_t, _F32, _F32, _F32, _U16,
                      _U16, _U16, _U16])  # fmt: skip
        assert fn(None, v2, rows, cols, big, one, dy, *bufs) == 1
    else:
        assert _probe_panels(lib)(conv, 1, v2, rows, cols, big, one, dy, bufs[0], bufs[1],
                                  bufs[2], bufs[3]) == 1  # fmt: skip
    return {"dga": _decode(bufs[0]), "dgt": _decode(bufs[1]), "dua": _decode(bufs[2], 32.0),
            "dut": _decode(bufs[3], 32.0)}  # fmt: skip


def _probe_panels(lib):
    return _writer(lib, "bcir_tensor_probe_panels",
                   [ct.c_int, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, _F32, _F32, _F32, _U16,
                    _U16, _U16, _U16])  # fmt: skip


def test_the_panel_writers_keep_the_stated_panels():
    """The SiLU writer's panels (A and B) and the SwiGLU VJP's (each adjoint as A and as A^T), by
    every instance the host has and by the library's entries, on tensors whose elements carry their
    row and then their column: every instance writes the portable C's bytes and the entries the
    widest instance's; every element is the one `a_panel`, `b_panel` and `at_panel` say its offset
    holds; and each second panel is the first converted (`panel_second`), element for element."""
    lib = _library()
    if lib is None:
        return  # no C compiler, or no C units (an installed package): a documented skip
    instances = _instances(lib)
    for rows, cols in ((32, 32), (64, 96), (96, 32)):
        n = rows * cols
        codes = {
            "row": [1 + i // cols for i in range(n)],
            "col": [1 + i % cols for i in range(n)],
        }
        A, B, AT = a_panel(rows, cols), b_panel(rows, cols), at_panel(rows, cols)
        to_b = panel_second(rows, cols, transposed=False)
        to_t = panel_second(rows, cols, transposed=True)
        for v2 in (0, 1):
            for swiglu in (0, 1):
                got = {(conv, axis): _panel_codes(lib, conv, swiglu, v2, rows, cols, code)
                       for conv in [*instances, None] for axis, code in codes.items()}  # fmt: skip
                for axis in codes:
                    for conv in instances[1:]:
                        assert got[conv, axis] == got[1, axis], (conv, swiglu, v2, rows, cols)
                    assert got[None, axis] == got[instances[-1], axis], (swiglu, v2, rows, cols)
                held = {name: list(zip(got[1, "row"][name], got[1, "col"][name]))
                        for name in got[1, "row"]}  # fmt: skip
                for o in range(n):
                    x, t = A.coords(o), AT.coords(o)
                    if not swiglu:
                        y = B.coords(o)
                        assert held["a"][o] == (x["row"], x["depth"]), (rows, cols, v2, o)
                        assert held["b"][o] == (y["depth"], y["col"]), (rows, cols, v2, o)
                        assert held["b"][to_b(o)] == held["a"][o], (rows, cols, v2, o)
                        continue
                    for first, second in (("dga", "dgt"), ("dua", "dut")):
                        where = (first, second, rows, cols, v2, o)
                        assert held[first][o] == (x["row"], x["depth"]), where
                        assert held[second][o] == (t["depth"], t["row"]), where
                        assert held[second][to_t(o)] == held[first][o], where


def _packed_codes(pack, variant, which, trans, rows, cols, axis):
    """The probe's pack of an operand whose elements carry their first (`axis` "row") or second
    coordinate, stored as its packer reads it -- rows x cols, or transposed -- decoded."""
    code = [0.0] * (rows * cols)
    for i in range(rows):
        for j in range(cols):
            code[j * rows + i if trans else i * cols + j] = float(1 + (i if axis == "row" else j))
    layout = a_pack(rows, cols) if which == 0 else b_pack(rows, cols)
    size = _pad32(rows) * _pad32(cols)  # the probe's contract: the operand padded to 32 both ways
    assert layout.size == size, (which, rows, cols, layout.size, size)
    x = (ct.c_float * len(code))(*code)
    out = (ct.c_uint16 * size)(*([0xFFFF] * size))
    ld = rows if trans else cols
    assert pack(variant, which, trans, rows, cols, 1.0, x, ld, out) == 1, (variant, which, trans)
    return _decode(out)


def test_the_bf16_packs_keep_the_stated_packs_and_their_tails():
    """Every bf16 pack instance the host has (the portable C, AVX-512, AMX) packs A (as stored and
    transposed) and B (as stored and transposed) of shapes with ragged tails as `a_pack` and
    `b_pack` state: each element at its offset, every offset past the operand +0.0, and every
    instance the same bytes."""
    lib = _library()
    if lib is None:
        return
    pack = _writer(lib, "bcir_tensor_probe_bf16_pack",
                   [ct.c_int, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_float, _F32,
                    ct.c_size_t, _U16])  # fmt: skip
    variants = _instances(lib)
    for rows, cols in ((40, 70), (32, 32), (17, 96), (64, 33)):
        for which, trans in ((0, 0), (0, 1), (1, 0), (1, 1)):
            layout = a_pack(rows, cols) if which == 0 else b_pack(rows, cols)
            names = ("row", "depth") if which == 0 else ("depth", "col")
            got = {(v, axis): _packed_codes(pack, v, which, trans, rows, cols, axis)
                   for v in variants for axis in ("row", "col")}  # fmt: skip
            for v in variants[1:]:  # every instance the portable C's bytes
                assert got[v, "row"] == got[1, "row"] and got[v, "col"] == got[1, "col"], v
            for o in range(layout.size):
                held = (got[1, "row"][o], got[1, "col"][o])
                if layout.padding(o):
                    assert held == (None, None), (rows, cols, which, trans, o, held)
                else:
                    c = layout.coords(o)
                    assert held == (c[names[0]], c[names[1]]), (rows, cols, which, trans, o)


def test_the_avx512_tile_and_transpose_keep_their_layouts():
    """The AVX-512 blocked GEMM's packed B tile -- every column block width from 1 to 48, and full,
    offset and ragged blocks over two depth ranges, B as stored and transposed -- holds each element
    where `kb_tile` says, the columns past the block zero; and the 16 x 16 register transpose moves
    each element where `TRANSPOSE16` says. A host without AVX-512 answers 0: a documented skip."""
    lib = _library()
    if lib is None:
        return
    kb = _writer(lib, "bcir_tensor_probe_kb_pack",
                 [ct.c_int, ct.c_size_t, ct.c_size_t, _F32, ct.c_size_t, ct.c_size_t,
                  ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p])  # fmt: skip
    tr = _writer(lib, "bcir_tensor_probe_transpose16", [_F32, ct.c_size_t, _F32, ct.c_size_t])
    src = (ct.c_float * 256)(*[float(16 * q + t) for q in range(16) for t in range(16)])
    dst = (ct.c_float * 256)()
    if tr(src, 16, dst, 16) == 0:
        return  # no AVX-512 on this host: the instance is not run here
    for q in range(16):
        for t in range(16):
            y = TRANSPOSE16.apply(q=q, t=t)
            assert dst[16 * y["q"] + y["t"]] == src[16 * q + t]
    k, n = 40, 100
    cases = [(0, nj, 8, 16) for nj in range(1, 49)]  # every block width
    cases += [(jb, nj, lb, nl) for jb, nj in ((0, 48), (48, 48), (96, 4), (17, 31))
              for lb, nl in ((0, k), (8, 16))]  # fmt: skip
    for tb in (0, 1):
        b = (ct.c_float * (k * n))()
        for d in range(k):
            for j in range(n):
                b[j * k + d if tb else d * n + j] = float(1024 * d + j + 1)
        for jb, nj, lb, nl in cases:
            raw = (ct.c_float * (nl * 48 + 16))()
            base = ct.addressof(raw)
            aligned = (base + 63) & ~63
            tile = (ct.c_float * (nl * 48)).from_address(aligned)
            for i in range(nl * 48):
                tile[i] = -1.0
            assert kb(tb, k, n, b, k if tb else n, jb, nj, lb, nl, aligned) == 1
            layout = kb_tile(nl, nj)
            for o in range(nl * 48):
                c = layout.coords(o)
                want = float(1024 * (lb + c["depth"]) + jb + c["col"] + 1)
                want = 0.0 if layout.padding(o) else want
                assert tile[o] == want, (tb, jb, nj, lb, nl, o, tile[o], want)
