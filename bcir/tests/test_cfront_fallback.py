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
    # locals/params/returns, and a `_BitInt` with an explicitly-cast constant -- compiles clean, no fallback.
    for src in (
        "unsigned _BitInt(12) f(unsigned _BitInt(12) a, unsigned _BitInt(12) b){ return a + b; }\n",
        "_BitInt(20) f(_BitInt(20) a, _BitInt(20) b){ _BitInt(20) c = a * b; return c - (_BitInt(20))7; }\n",
        "signed _BitInt(8) f(signed _BitInt(8) a){ return a + (signed _BitInt(8))3 - a; }\n",
        "unsigned _BitInt(64) f(unsigned _BitInt(64) a, unsigned _BitInt(64) b){ return (a & b) << (unsigned _BitInt(64))1; }\n",
    ):
        r = compile_with_fallback(src, check_clang=False)
        assert not r.needs_fallback and r.is_clean, (src, r.fallback)


def test_bitint_mixed_width_where_bitint_wins_is_first_class():
    # NEW first-class subset (sub-feature A): a mixed-width `_BitInt` expression whose C23 6.3.1.8 result is
    # itself a `_BitInt(N)` -- the bit-precise operand WINS the rank (its width strictly exceeds the other
    # operand's post-promotion width). The modeled result type == Clang's (verified by the `_Generic`
    # differential in test_c_cfront.py); these compile clean, NO fallback.
    for src in (
        # `_BitInt(N>32)` + a standard int VARIABLE -> the `_BitInt` wins (N > 32 = width(int)).
        "_BitInt(33) f(_BitInt(33) a, int b){ return a + b; }\n",
        "unsigned _BitInt(64) f(unsigned _BitInt(64) a, unsigned b){ return a + b; }\n",
        "_BitInt(64) f(_BitInt(64) a, unsigned b){ return a * b; }\n",       # bi64 + uint -> bi64 (own sign)
        # a WIDER `_BitInt` mixed with a NARROWER `_BitInt` -> the wider wins.
        "_BitInt(16) f(_BitInt(16) a, _BitInt(8) b){ return a + b; }\n",
        "unsigned _BitInt(20) f(unsigned _BitInt(20) a, _BitInt(12) b){ return a | (unsigned _BitInt(20))b; }\n",
        # a `_BitInt(N>32)` + a bare integer constant whose literal type is narrower -> stays `_BitInt(N)`.
        "_BitInt(33) f(_BitInt(33) a){ return a + 5; }\n",
        "unsigned _BitInt(40) f(unsigned _BitInt(40) a){ return a * 3u; }\n",
    ):
        r = compile_with_fallback(src, check_clang=False)
        assert not r.needs_fallback and r.is_clean, (src, r.fallback)


def test_bitint_bitfield_is_first_class():
    # NEW first-class subset (sub-feature B): a `_BitInt(N)` BITFIELD `_BitInt(N) m : W` (2<=N<=64, 1<=W<=N)
    # packs into the `_BitInt(N)` storage unit (1/2/4/8 bytes) with the SAME LSB-first packing as a
    # standard-int bitfield of that storage size -- byte-identical to Clang (verified by the layout
    # differential in test_c_cfront.py). The field reads/writes its W bits typed `_BitInt(N)`.
    for src in (
        "struct S { _BitInt(12) a : 5; _BitInt(12) b : 6; }; _BitInt(12) f(struct S s){ return s.a + s.b; }\n",
        "struct S { unsigned _BitInt(12) a : 5; }; unsigned _BitInt(12) f(unsigned _BitInt(12) v){"
        " struct S s; s.a = v; return s.a; }\n",
        # a WIDE (W>32) `_BitInt` bitfield does NOT promote -- it stays `_BitInt(N)` in arithmetic.
        "struct S { unsigned _BitInt(64) x : 40; }; unsigned _BitInt(64) f(struct S s){"
        " return s.x + (unsigned _BitInt(64))1; }\n",
        # a `_BitInt` bitfield alongside a standard-int bitfield (mixed packing in one unit family).
        "struct S { unsigned _BitInt(12) a : 5; unsigned int b : 6; };"
        " unsigned _BitInt(12) f(struct S s){ return s.a; }\n",
    ):
        r = compile_with_fallback(src, check_clang=False)
        assert not r.needs_fallback and r.is_clean, (src, r.fallback)


def test_bitint_plain_member_does_not_fall_back():
    # a PLAIN (non-bitfield) `_BitInt(N)` struct/union MEMBER is now first-class: its `bitint` CType carries
    # the Clang storage width (so the layout matches), the member access loads/stores at that width typed
    # `_BitInt(N)`, and same-type arithmetic on the loaded value stays N-bit -- so these compile clean.
    for src in (
        # read a `_BitInt(12)` (non-standard width) member, same-type arithmetic, return a `_BitInt`.
        "struct S { int t; _BitInt(12) x; }; _BitInt(12) f(struct S s){ return s.x + (_BitInt(12))1; }\n",
        # write a `_BitInt` member of a local, read it back.
        "struct S { unsigned _BitInt(12) x; };"
        " unsigned _BitInt(12) f(unsigned _BitInt(12) v){ struct S s; s.x = v + (unsigned _BitInt(12))2; return s.x; }\n",
        # a `_BitInt(64)` member alongside a 12-bit member (mixed sub-word/word/8-byte layout).
        "struct S { _BitInt(12) lo; unsigned _BitInt(64) hi; };"
        " unsigned _BitInt(64) f(struct S s){ return s.hi + (unsigned _BitInt(64))9; }\n",
        # a `_BitInt` UNION member.
        "union U { _BitInt(20) a; int b; }; _BitInt(20) f(union U u){ return u.a; }\n",
    ):
        r = compile_with_fallback(src, check_clang=False)
        assert not r.needs_fallback and r.is_clean, (src, r.fallback)


def test_bitint_unsupported_forms_route_to_fallback_not_miscompile():
    # the conservative boundary (the SAFETY CONTRACT): a form OUTSIDE the supported `_BitInt` subset must
    # route to fallback (a CLowerError / CParseError -> needs_fallback) rather than emit possibly-wrong code
    # that drops the exact width. Each of these is unsupported and MUST NOT compile to BCIR.
    cases = {
        # THE PRECISE BOUNDARY (sub-feature A): a `_BitInt` mixed with a standard int / different `_BitInt`
        # whose C23 6.3.1.8 result is a STANDARD integer type stays fallback -- the bit-precise operand does
        # NOT win the rank (equal or LESSER width), so the result would be int/long/... whose long-vs-long-long
        # width-collapse in the value model could diverge from Clang's `_Generic` view. Correctness > coverage.
        #
        # `_BitInt(N<=32)` + int: int has the greater(-or-equal) rank -> result int (NOT a `_BitInt`).
        "unsigned _BitInt(8) f(unsigned _BitInt(8) a, int b){ return a + b; }\n": "lower",
        "_BitInt(32) f(_BitInt(32) a, int b){ return a + b; }\n": "lower",
        # `_BitInt(N)` + a standard int of EQUAL-OR-GREATER width (the standard wins the tie / the rank).
        "_BitInt(64) f(_BitInt(64) a, long b){ return a + b; }\n": "lower",
        "_BitInt(8) f(_BitInt(8) a, long b){ return a + b; }\n": "lower",
        # `_BitInt(N)` + a bare integer constant whose literal type out-ranks it -> result is the standard type.
        "_BitInt(8) f(_BitInt(8) a){ return a + 5; }\n": "lower",
        "_BitInt(20) f(_BitInt(20) a){ return a + 5L; }\n": "lower",
        # mixing a `_BitInt` MEMBER read with a standard integer variable where the std wins (same as above).
        "struct S { _BitInt(12) x; }; _BitInt(12) f(struct S s, int k){ return s.x + k; }\n": "lower",
        # a `_BitInt` mixed with a FLOATING type -> the result is a float, NOT a `_BitInt` (even a WIDE
        # `_BitInt(64)` -- the float width must not be mistaken for a standard-int rank). Route to fallback.
        "_BitInt(64) f(_BitInt(64) a, double b){ return a + b; }\n": "lower",
        "_BitInt(64) f(_BitInt(64) a, float b){ return a + b; }\n": "lower",
        # a `_BitInt` bitfield with width W outside 1..N (W>N is invalid C; Clang errors) -> fallback.
        "struct S { _BitInt(8) x : 9; }; _BitInt(8) f(struct S s){ return s.x; }\n": "lower",
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
