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


def test_compile_unit_keeps_its_raise_contract():
    # only the fallback wrapper degrades gracefully; compile_unit still raises (locked boundary).
    try:
        compile_unit("int f(void){ return 1 }\n", check_clang=False)
        raise AssertionError("compile_unit should raise on malformed input")
    except CParseError:
        pass
