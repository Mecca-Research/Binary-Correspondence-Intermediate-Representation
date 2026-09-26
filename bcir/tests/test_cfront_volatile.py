"""CF-VOL: `volatile` carried through both cfront rails.

The rows (`volatile_fixtures.measure`, gated by `tools/perf/check_volatile.py`) grade every place a C
program puts `volatile`, against every access form, at every element width, lowered by both rails
and judged by Clang. This module holds what they rest on:

* the oracle's half, which needs only the interpreter and so runs from the installed package: every
  volatile access marked a device access, the qualifier reaching globals, members and casts, a
  loaded value being an ordinary value, and the two spellings the emit relies on;
* the judges, each shown firing on the defect it exists to catch (a gate nobody has seen fail is
  not a gate, L2) -- a dropped qualifier, a register written wider than its slot, a footprint
  without the device;
* the rows themselves at their bound, with the MMIO register idiom `*(volatile uint32_t *)ADDR`
  judged by what Clang emits, since no harness can hand a program that address.

The Clang-judged half needs Clang (and, for the rows, the checkout's runtime/c); without them it
is skipped here, except in a job that installed Clang for the suite (`BCIR_REQUIRE_LLVM`), where
the absence is a failure (L2).
"""

from __future__ import annotations

import os

from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.emit import _volatile_ptr
from bcir.frontends.cfront.lower import _pointer_spelling
from bcir.frontends.cfront.ctype_model import pointer, scalar
from bcir.model.lanes import Domain, Lane
from bcir.tests import volatile_fixtures as vf


def _unit(src: str):
    return compile_unit(src, check_clang=False)


def _clang() -> str | None:
    clang = vf.find_clang()
    if not clang:
        assert not os.environ.get("BCIR_REQUIRE_LLVM"), (
            "BCIR_REQUIRE_LLVM is set and no Clang judges the volatile forms"
        )
    return clang


# --- the oracle's half: runs anywhere ------------------------------------------------------------


def test_every_volatile_access_of_the_corpus_is_a_device_access_on_the_oracle():
    """In every form of the corpus, each claim that touches volatile storage is a device access
    (MMIO, lane H, barriered -- R3's law), and at least one claim carries the volatile bit: the
    access the program performs. The corpus is the rows' own, so this is the oracle's half of
    them, graded without Clang."""
    forms = 0
    for unit in vf.units():
        res = _unit(unit.source)
        assert res.is_clean, (unit.tag, [str(d) for d in res.diagnostics][:3])
        for entry in unit.entries:
            lf = res.lowered.functions[entry.name]
            touched = [
                c
                for c in lf.claims
                if any(
                    (r := lf.resources.get(x)) is not None and r.domain == Domain.MMIO
                    for x in (*c.rd, *c.wr)
                )
            ]
            assert touched, entry.name
            for c in touched:
                assert (c.domain, c.lane, c.hazard) == (Domain.MMIO, Lane.H, "barriered"), (
                    entry.name,
                    c.op,
                )
            assert any(c.volatile for c in lf.claims), entry.name
            forms += 1
    assert forms == len(vf.forms()) > 400  # every form examined, none skipped


def test_the_qualifier_reaches_globals_members_and_casts():
    """The parent lowered a volatile global, a file-scope pointer to volatile, a member pointer to
    volatile and a cast to one as ordinary memory: each is now an MMIO resource, and the access
    through it a volatile claim."""
    res = _unit(
        "#include <stdint.h>\n"
        "volatile uint32_t greg;\n"
        "volatile uint16_t *gvp;\n"
        "struct D { uint32_t pad; volatile uint8_t *regs; };\n"
        "uint32_t g(void) { return greg; }\n"
        "uint32_t p(uint32_t i) { return gvp[i & 3u]; }\n"
        "uint32_t m(struct D *d, uint32_t i) { return d->regs[i & 3u]; }\n"
        "uint32_t c(void *raw, uint32_t i) {"
        " volatile uint32_t *q = (volatile uint32_t *)raw; return q[i & 3u]; }\n"
    )
    assert res.is_clean, [str(d) for d in res.diagnostics]
    for fn in ("g", "p", "m", "c"):
        lf = res.lowered.functions[fn]
        vol = [c for c in lf.claims if c.volatile]
        assert vol and all(c.domain == Domain.MMIO for c in vol), (fn, lf.claims)


def test_a_loaded_value_is_an_ordinary_value():
    """Lvalue conversion drops the qualifier: the value a volatile load yields is a RAM temp, so
    `x |= 1` on it is plain arithmetic -- the parent typed it volatile and refused the unit (R3:
    an MMIO resource touched by a RAM claim)."""
    res = _unit(
        "#include <stdint.h>\n"
        "uint32_t f(volatile uint32_t *p) { uint32_t x = p[1]; x |= 1u; p[2] = x; return x; }\n"
    )
    assert res.is_clean, [str(d) for d in res.diagnostics]
    lf = res.lowered.functions["f"]
    load = next(c for c in lf.claims if c.op == "c.load")
    assert load.volatile and lf.resources[load.wr[0]].domain == Domain.RAM


def test_a_pointer_cast_names_its_target_faithfully():
    """A cast to a pointer yields a pointer of exactly the target type: its pointee's sign, a plain
    `char`, a float, a struct tag or `void`, behind its `volatile`. The parent width-named the
    pointee (dropping the qualifier and the sign) and gave the result a uint32 temp, truncating the
    address. The twin spells it byte-identically (the digest keeps the spelling)."""
    u16, i16 = scalar("uint16_t"), scalar("int16_t")
    assert _pointer_spelling(pointer(u16)) == "uint16_t *"
    assert _pointer_spelling(pointer(i16)) == "int16_t *"
    assert _pointer_spelling(pointer(scalar("char"))) == "char *"
    assert _pointer_spelling(pointer(scalar("double"))) == "double *"
    assert _pointer_spelling(pointer(scalar("void"))) == "void *"
    assert _pointer_spelling(pointer(pointer(u16))) == "uint16_t **"
    res = _unit(
        "#include <stdint.h>\nuint32_t f(void *raw) { return *(volatile uint32_t *)raw; }\n"
    )
    lf = res.lowered.functions["f"]
    cast = next(c for c in lf.claims if c.op.startswith("c.cast:"))
    assert cast.op == "c.cast:volatile uint32_t *"
    assert lf.rid_types[cast.wr[0]].kind == "pointer"
    assert lf.resources[cast.wr[0]].domain == Domain.MMIO


def test_a_volatile_slot_is_spelled_so_the_slot_is_what_is_volatile():
    """`volatile T *` for a scalar or aggregate slot; `T volatile *` when the slot is itself a
    pointer -- a qualifier in front of `T *` would make the pointee volatile, not the slot."""
    assert _volatile_ptr("uint32_t") == "volatile uint32_t *"
    assert _volatile_ptr("struct R") == "volatile struct R *"
    assert _volatile_ptr("uint8_t *") == "uint8_t * volatile *"


# --- the judges, each seen firing ----------------------------------------------------------------

_ORIGINAL = (
    "#include <stdint.h>\nuint32_t f(volatile uint8_t *p, uint32_t i, uint32_t v) {{ {body} }}\n"
)


def _judge_unit(body: str) -> vf.Unit:
    entry = vf.Entry("f", "param", "st", "u8", "uint8_t", "ptr", "", ())
    return vf.Unit("u8", _ORIGINAL.format(body=body), [entry])


def test_the_emit_judge_fires_on_a_dropped_qualifier():
    clang = _clang()
    if not clang:
        return
    unit = _judge_unit("p[i & 7u] = (uint8_t)v; return 0u;")
    good = "static uint32_t bcir_f(volatile uint8_t *p, uint32_t i, uint32_t v) { p[i & 7u] = (uint8_t)v; return 0u; }\n"
    bad = good.replace("volatile uint8_t *p", "uint8_t *p")
    assert vf.emit_mismatches(clang, unit, good) == set()
    assert vf.emit_mismatches(clang, unit, bad) == {"f"}


def test_the_behaviour_judge_fires_on_a_register_written_wider_than_its_slot():
    """The twin's original bug: a byte store written as a 32-bit store at `p + 4*i`."""
    clang = _clang()
    if not clang:
        return
    unit = _judge_unit("p[i & 7u] = (uint8_t)v; return 0u;")
    good = "static uint32_t bcir_f(volatile uint8_t *p, uint32_t i, uint32_t v) { p[i & 7u] = (uint8_t)v; return 0u; }\n"
    wide = (
        "static uint32_t bcir_f(volatile uint8_t *p, uint32_t i, uint32_t v)"
        " { ((volatile uint32_t *)p)[i & 7u] = (uint8_t)v; return 0u; }\n"
    )
    assert vf.behaviour_mismatches(clang, unit, good) == set()
    assert vf.behaviour_mismatches(clang, unit, wide) == {"f"}


def test_the_device_judge_fires_on_a_footprint_without_the_device():
    unit = _judge_unit("p[i & 7u] = (uint8_t)v; return 0u;")
    assert vf.device_missed(unit, "fn=f reads=*,f.p writes=*\n") == set()
    assert vf.device_missed(unit, "fn=f reads=f.p writes=*\n") == {"f"}
    assert vf.device_missed(unit, "") == {"f"}  # a function the report omits is missed too


# --- the rows and the register idiom -------------------------------------------------------------


def test_the_register_idiom_is_one_access_of_its_width_on_both_rails():
    """`*(volatile uint32_t *)ADDR` -- a device register at a fixed address -- lowers on both rails
    (the parent refused it on both) to exactly the original's one volatile access, at its width.
    No harness can hand a program that address, so Clang's emitted IR is the judge, and the rails'
    claim graphs must agree."""
    clang = _clang()
    if not clang:
        return
    src = (
        "#include <stdint.h>\n"
        "void w(uint32_t v) { *(volatile uint32_t *)0x40000000u = v; }\n"
        "uint32_t r(void) { return *(volatile uint32_t *)0x40000004u; }\n"
        "uint16_t h(uintptr_t base) { return *(volatile uint16_t *)(base + 2u); }\n"
    )
    entries = [vf.Entry(n, "cast", "st", "u32", "uint32_t", "", "", ()) for n in ("w", "r", "h")]
    unit = vf.Unit("reg", src, entries)
    py = vf.oracle(unit)
    assert not py.refused, py.refused
    assert vf.emit_mismatches(clang, unit, py.emitted) == set()
    ops = vf.volatile_ops(clang, src)
    assert ops["w"] == [("store", "i32")] and ops["r"] == [("load", "i32")]
    assert ops["h"] == [("load", "i16")]
    if not os.path.isdir(vf._runtime_c()):  # the installed package: no twin to build
        return
    from bcir.tests.escape_fixtures import build_bcir_cc  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as d:
        tw = vf.twin(unit, build_bcir_cc(d))
    assert not tw.refused, tw.refused
    assert tw.digest == py.digest
    assert vf.emit_mismatches(clang, unit, tw.emitted) == set()


def test_the_rows_are_at_their_bound():
    """Every form, both rails, every judge: 0 in every row (the gate's bound)."""
    if not vf.available():  # no Clang, or the installed package (no runtime/c to build the twin)
        assert not (os.environ.get("BCIR_REQUIRE_LLVM") and os.path.isdir(vf._runtime_c())), (
            "BCIR_REQUIRE_LLVM is set and no Clang grades the volatile rows"
        )
        return
    rows = vf.measure()
    assert rows == {row: 0.0 for row in vf.ROWS}, rows
