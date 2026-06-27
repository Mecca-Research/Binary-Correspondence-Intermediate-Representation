"""Phase 4 — segment 3 (optimizer/robustness): the LLVM-backend fallback contract.

`compile_with_fallback` is a total entry point: it returns a verified CompileResult, or a result whose
`needs_fallback` signals that a construct is outside BCIR's supported subset (or the unit is malformed)
so a driver routes it to the LLVM backend -- it never crashes. `compile_unit` keeps raising.
"""
from __future__ import annotations

from bcir.frontends.cfront import compile_unit, compile_with_fallback
from bcir.frontends.cfront.cparse import CParseError


def test_supported_program_does_not_fall_back():
    r = compile_with_fallback("int f(int x){ return x + 1; }\n", check_clang=False)
    assert not r.needs_fallback and r.fallback == "" and r.is_clean


def test_unsupported_construct_falls_back_with_a_reason():
    # a non-constant static initializer is outside the supported subset -> fall back, do not crash.
    r = compile_with_fallback("int g; int f(void){ static int x = g; return x; }\n", check_clang=False)
    assert r.needs_fallback and not r.is_clean
    assert r.fallback.startswith("lower:")                    # the rejecting stage is tagged


def test_malformed_unit_falls_back_at_the_parse_stage():
    r = compile_with_fallback("int f(void){ return 1 }\n", check_clang=False)
    assert r.needs_fallback and r.fallback.startswith("parse:")


def test_fallback_is_total_over_several_unsupported_constructs():
    # whatever the construct, the contract returns a CompileResult (never raises).
    for src in ("int f(void){ 5 = 3; return 0; }\n",           # not an lvalue
                "int f(void){ goto nowhere; return 0; }\n",     # a statement beyond the subset
                "@bad\n"):                                      # garbage
        r = compile_with_fallback(src, check_clang=False)
        assert isinstance(r, type(compile_with_fallback("int x;\n", check_clang=False)))
        # either it compiled or it asked for fallback -- but it returned, not crashed
        assert r.needs_fallback or r.is_clean


def test_bitint_supported_subset_does_not_fall_back():
    # the supported `_BitInt(N)` subset: same-type arithmetic (incl. a NON-standard 12-bit lane) over
    # locals/params/returns, and a `_BitInt` with an integer constant -- compiles clean, no fallback.
    for src in (
        "unsigned _BitInt(12) f(unsigned _BitInt(12) a, unsigned _BitInt(12) b){ return a + b; }\n",
        "_BitInt(20) f(_BitInt(20) a, _BitInt(20) b){ _BitInt(20) c = a * b; return c - (_BitInt(20))7; }\n",
        "signed _BitInt(8) f(signed _BitInt(8) a){ return a + (signed _BitInt(8))3 - a; }\n",
        "unsigned _BitInt(64) f(unsigned _BitInt(64) a, unsigned _BitInt(64) b){ return (a & b) << (unsigned _BitInt(64))1; }\n",
    ):
        r = compile_with_fallback(src, check_clang=False)
        assert not r.needs_fallback and r.is_clean, (src, r.fallback)


def test_bitint_unsupported_forms_route_to_fallback_not_miscompile():
    # the conservative boundary (the SAFETY CONTRACT): a form OUTSIDE the supported `_BitInt` subset must
    # route to fallback (a CLowerError / CParseError -> needs_fallback) rather than emit possibly-wrong code
    # that drops the exact width. Each of these is unsupported and MUST NOT compile to BCIR.
    cases = {
        # mixing a `_BitInt` with a standard integer VARIABLE in arithmetic (C23 does NOT promote `_BitInt`,
        # so the common type would have to be carried exactly -- only same-type is modeled).
        "unsigned _BitInt(8) f(unsigned _BitInt(8) a, int b){ return a + b; }\n": "lower",
        # mixing two DIFFERENT `_BitInt` widths in one expression.
        "_BitInt(8) f(_BitInt(8) a, _BitInt(16) b){ return a + b; }\n": "lower",
        # a `_BitInt` as a struct member (member layout + access not modeled in the subset).
        "struct S { _BitInt(12) x; }; _BitInt(12) f(struct S s){ return s.x; }\n": "lower",
        # widths outside the supported 2..64 range are rejected at parse.
        "_BitInt(100) f(_BitInt(100) a){ return a + a; }\n": "parse",
        "_BitInt(1) f(_BitInt(1) a){ return a; }\n": "parse",
    }
    for src, stage in cases.items():
        r = compile_with_fallback(src, check_clang=False)
        assert r.needs_fallback and not r.is_clean, ("expected fallback", src)
        assert r.fallback.startswith(stage + ":"), (src, r.fallback)


def test_compile_unit_keeps_its_raise_contract():
    # only the fallback wrapper degrades gracefully; compile_unit still raises (locked boundary).
    try:
        compile_unit("int f(void){ return 1 }\n", check_clang=False)
        raise AssertionError("compile_unit should raise on malformed input")
    except CParseError:
        pass
