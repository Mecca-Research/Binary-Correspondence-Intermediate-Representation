"""Phase 4 — segment 3 (optimizer correctness): inter-procedural summary reuse (IPO) for the C
frontend. A function is planned once into a summary; a call whose actuals are cost-compatible with the
callee's formals reuses that cost instead of re-planning the body, so `ipo["reused"]` counts the saved
plans and `ipo["leaves"]` reflects plan-once (the body's leaves are counted a single time)."""

from __future__ import annotations

from bcir.frontends.cfront import compile_unit


def test_a_helper_is_planned_once_and_reused_per_call_site():
    src = (
        "int helper(int x){ return x*2 + 1; }\n"
        "int entry(int a){ return helper(a) + helper(a + 1) + helper(a + 2); }\n"
    )
    r = compile_unit(src, check_clang=False)
    assert r.is_clean
    assert r.ipo["reused"] == 3  # all three call sites reuse the summary
    assert r.ipo["leaves"] == 3  # plan-once: helper's body leaves counted once
    assert r.ipo["worst"] > 0 and r.ipo["expected"] > 0


def test_leaf_only_program_reuses_nothing():
    r = compile_unit("int f(int x){ return x + 1; }\n", check_clang=False)
    assert r.ipo["reused"] == 0 and r.ipo["leaves"] == 1


def test_distinct_helpers_each_reused():
    src = (
        "int a(int x){ return x + 1; }\n"
        "int b(int x){ return x * 3; }\n"
        "int e(int x){ return a(x) + a(x + 1) + b(x) + b(x + 1); }\n"
    )
    r = compile_unit(src, check_clang=False)
    assert r.is_clean and r.ipo["reused"] == 4  # two helpers, two call sites each


def test_ipo_path_does_not_mask_recursion():
    # the summary path must not hide an R18 violation -- recursion is still rejected.
    r = compile_unit(
        "int rec(int x){ return rec(x - 1); } int e(void){ return rec(5); }\n", check_clang=False
    )
    assert not r.r18_ok and any(d.law == "R18" for d in r.diagnostics)
