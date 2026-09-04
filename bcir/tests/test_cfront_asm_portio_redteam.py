"""H3 (residual) -- a MALFORMED-INPUT RED-TEAM over the cfront `c.asm` / `c.portio` lowering paths
(`_asm_stmt` ~lower.py:1718 and `_portio` ~lower.py:1968), on the **Python rail**.

WHY THIS IS PYTHON-RAIL-ONLY. GNU inline asm (`asm`/`__asm__`) and the port-I/O intrinsics
(`inb`/`inw`/`inl` / `outb`/`outw`/`outl`) are a **Python-frontend-only** feature: the C twin
(`runtime/c/bcir_cfront.c`) does NOT parse inline assembly or port I/O at all (its only `asm` token is
a comment noting it emits no per-ISA asm). So H1's C-twin sanitizer/fuzz sweep -- the malformed-input
robustness coverage the rest of the C subset gets -- does **not** reach `_asm_stmt` / `_portio`. The
happy-path suites (`test_c_cfront_asm.py`, `test_c_cfront_portio.py`) exercise these paths
HAPPY-PATH-mostly. This file closes that residual: it feeds malformed / adversarial asm + portio C
snippets through the SAME cfront entry points the happy-path suites use (`compile_unit` / `diagnose` /
`parse_unit`, from `bcir.frontends.cfront`) and asserts the ONE unifying robustness property:

    Every malformed snippet either (a) **lowers cleanly** (the template/edge is a trusted opaque
    pass-through -- a bad `%N` / mismatched operand is the downstream C compiler's job, not the
    frontend's), OR (b) **raises a clean cfront diagnostic** -- one of the four front-end exception
    types (`CParseError` / `CLexError` / `CPPError` / `CLowerError`) or a `ValueError` (the emit-stage
    diagnostic that `pipeline._FALLBACK_PHASE` recognizes). It must NEVER escape an **uncaught internal
    Python exception** -- `AttributeError` / `IndexError` / `KeyError` / `TypeError` / `RecursionError`
    -- which would be a real robustness bug (the frontend should raise an honest, located diagnostic, or
    route to the LLVM fallback, never crash on a deref of `None` or an out-of-range index).

It mirrors the existing red-team style (`test_area_b_redteam.py`): stdlib-only, fully deterministic
hand-chosen adversarial inputs (NO RNG -- a red-team targets specific boundaries), plain `def test_*`
functions so `bcir.tests.run_all` auto-collects them, and it MODIFIES NO production code -- it only
probes the existing lowering. Both rails are driven per snippet: `compile_unit` keeps its raise-on-error
contract (so a clean rejection surfaces as one of the allowed exception types), and `diagnose` is the
total report path (a malformed unit must come back as `not ok` with diagnostics, never a crash).

WHAT THIS RED-TEAM FOUND. Across the whole corpus below, every malformed asm + portio snippet degrades
HONESTLY -- it either lowers (the opaque-template / unknown-spelling cases) or raises a `CParseError` /
`CLowerError`. None escapes an uncaught internal exception. (If one ever does, the assertions here flag
it -- which is exactly the point of the hardening gate. The fix would belong in `_asm_stmt`/`_portio`,
not here.)
"""

from bcir.frontends.cfront import compile_unit, diagnose, parse_unit
from bcir.frontends.cfront.clex import CLexError
from bcir.frontends.cfront.cpp import CPPError
from bcir.frontends.cfront.cparse import CParseError
from bcir.frontends.cfront.lower import CLowerError

# The ALLOWED clean-diagnostic exception types -- a malformed unit MAY raise one of these (an honest,
# located front-end rejection). They mirror `pipeline._FALLBACK_PHASE` exactly: a stage-rejected unit is
# routed to the LLVM fallback by these same types. `ValueError` is the emit-stage diagnostic.
CLEAN_DIAGNOSTIC_TYPES = (CParseError, CLexError, CPPError, CLowerError, ValueError)

# The FORBIDDEN uncaught-internal exception types -- an uncaught one of these is a real robustness bug
# (a deref of None, an out-of-range index, a missing dict key, a bad type op, or unbounded recursion the
# parser's depth guard should have caught as a clean CParseError). The red-team FAILS if any escapes.
INTERNAL_BUG_TYPES = (AttributeError, IndexError, KeyError, TypeError, RecursionError)


def _assert_clean_or_diagnoses(src: str) -> str:
    """Drive one malformed snippet through BOTH cfront entry points and assert the unifying property:
    it lowers cleanly OR raises a clean cfront diagnostic, NEVER an uncaught internal exception.

    Returns a short outcome tag ("LOWERED" | "<ExceptionTypeName>") for the per-test assertions below to
    pin the ACTUAL behaviour (so this stays a regression gate, not just a no-crash smoke test)."""
    # (1) compile_unit keeps its raise-on-error contract: a clean rejection is one of the allowed types;
    #     an internal-bug type escaping here is the robustness failure this gate exists to catch.
    outcome: str
    try:
        compile_unit(src, check_clang=False)
        outcome = "LOWERED"
    except INTERNAL_BUG_TYPES as e:  # the bug we are hunting -- surface it loudly, NEVER swallow it
        raise AssertionError(
            f"UNCAUGHT INTERNAL EXCEPTION (robustness bug) from compile_unit on {src!r}: "
            f"{type(e).__name__}: {e} -- the frontend must raise a clean cfront diagnostic "
            f"(CParseError/CLowerError/...) or lower, never crash on this malformed input"
        ) from e
    except CLEAN_DIAGNOSTIC_TYPES as e:
        outcome = type(e).__name__
    # (2) diagnose is the TOTAL report path: it catches every front-end error and renders it, so it must
    #     NEVER itself raise -- and a malformed unit must come back `not ok` with at least one diagnostic
    #     (a clean unit reports ok). This guards the diagnostics rail the same way.
    try:
        rep = diagnose(src)
    except INTERNAL_BUG_TYPES as e:
        raise AssertionError(
            f"UNCAUGHT INTERNAL EXCEPTION (robustness bug) from diagnose on {src!r}: "
            f"{type(e).__name__}: {e} -- diagnose must be total (render the error), never crash"
        ) from e
    if outcome == "LOWERED":
        assert rep.ok, (src, "compile_unit lowered but diagnose reported errors", rep.diagnostics)
    else:
        assert not rep.ok and rep.diagnostics, (
            src,
            "compile_unit raised a clean diagnostic but diagnose reported ok",
            outcome,
        )
    return outcome


# =====================================================================================================
# CORPUS A -- inline assembly (`_asm_stmt`). The asm TEMPLATE is a TRUSTED OPAQUE pass-through: a bad
# `%N` ref, a mismatched operand count, a duplicate output, or an empty template is NOT validated by the
# frontend (it is re-emitted verbatim for the downstream C compiler to accept or reject) -> those LOWER.
# A syntactically broken asm, an `asm goto`, or a non-lvalue / non-scalar output is a clean DIAGNOSTIC.
# In every case: a clean outcome, never an uncaught internal exception.
# =====================================================================================================


def test_asm_template_more_pct_refs_than_operands_is_clean():
    # `%1`/`%5` referenced but only one input operand -- a count MISMATCH. The template is opaque/trusted
    # (BCIR owns only the calling side), so this LOWERS verbatim; the C compiler is what rejects a bad ref.
    out = _assert_clean_or_diagnoses(
        'unsigned f(unsigned x){ unsigned y=0; asm("mov %1,%5" : "=r"(y) : "r"(x)); return y; }'
    )
    assert out == "LOWERED", out


def test_asm_empty_template_is_clean():
    # an empty template `""` is a valid (degenerate) basic asm -> it LOWERS (implicitly volatile).
    assert _assert_clean_or_diagnoses('void f(void){ asm(""); }') == "LOWERED"


def test_asm_template_references_pct9_with_no_operand_is_clean():
    # `%9` with zero operands -- the dangling ref is the C compiler's concern, not the frontend's; the
    # opaque template LOWERS verbatim (no operand-count validation against the template body).
    assert _assert_clean_or_diagnoses('void f(void){ asm("%9"); }') == "LOWERED"


def test_asm_huge_operand_index_in_template_is_clean():
    # a pathologically large `%999999` index -- still just opaque template text -> LOWERS (no indexing of
    # an operand array by the parsed number, so no IndexError).
    assert _assert_clean_or_diagnoses('void f(void){ asm("%999999"); }') == "LOWERED"


def test_asm_empty_constraint_strings_are_clean():
    # empty constraint strings `""(y)` / `""(x)` -- malformed-but-parseable; the constraint is stored
    # verbatim (stripped of quotes) and LOWERS. (gcc would reject an empty constraint; the frontend does
    # not pre-validate constraint syntax -- it is part of the trusted opaque calling side.)
    assert (
        _assert_clean_or_diagnoses(
            'unsigned f(unsigned x){ unsigned y=0; asm("" : ""(y) : ""(x)); return y; }'
        )
        == "LOWERED"
    )


def test_asm_goto_is_a_clean_rejection_not_a_crash():
    # `asm goto` (label operands) is the DEFERRED form -- it must be the clean CLowerError rejection
    # (control flow not yet modeled), never a crash on the label list.
    out = _assert_clean_or_diagnoses('void f(void){ asm goto("" : : : : L); L: ; }')
    assert out == "CLowerError", out


def test_asm_volatile_memory_clobber_lowers_barriered():
    # the VALID barrier form `__asm__ __volatile__("" ::: "memory")` -- lowers cleanly (a barriered edge).
    # Included so the red-team also pins a known-good case as LOWERED (not a false-positive rejection).
    assert (
        _assert_clean_or_diagnoses('void f(void){ __asm__ __volatile__("" ::: "memory"); }')
        == "LOWERED"
    )


def test_asm_duplicate_output_operands_are_clean():
    # the SAME lvalue `y` as two output operands -- a duplicate write set. The frontend does not de-dup or
    # reject it (the constraints are opaque); it LOWERS (the C compiler resolves the tied registers).
    assert (
        _assert_clean_or_diagnoses(
            'unsigned f(unsigned x){ unsigned y=0; asm("" : "=r"(y), "=r"(y) : "r"(x)); return y; }'
        )
        == "LOWERED"
    )


def test_asm_zero_operands_zero_constraints_extended_is_clean():
    # an extended asm with ALL three sections empty `("" : : : )` -- 0 operands, 0 clobbers. Valid -> LOWERS
    # (and stays non-volatile: it is extended, not basic). No empty-section indexing crash.
    assert _assert_clean_or_diagnoses('void f(void){ asm("" : : : ); }') == "LOWERED"


def test_asm_output_to_non_scalar_lvalue_is_a_clean_diagnostic():
    # an output operand `"=r"(*p)` to a deref (non-scalar-local) lvalue -- a clean CLowerError (a member /
    # array / deref / bitfield / MMIO output lvalue is a follow-on), never an AttributeError on the lvalue.
    out = _assert_clean_or_diagnoses('void f(int *p){ asm("" : "=r"(*p) : ); }')
    assert out == "CLowerError", out


def test_asm_output_to_undeclared_name_is_a_clean_diagnostic():
    # an output operand naming an UNDECLARED identifier `zzz` -- not in the env -> a clean CLowerError (the
    # `_asm_output_rid` else-branch), never a KeyError on the env lookup.
    out = _assert_clean_or_diagnoses('void f(void){ asm("" : "=r"(zzz) : ); }')
    assert out == "CLowerError", out


def test_asm_output_to_a_literal_is_a_clean_diagnostic():
    # an output operand that is a non-lvalue literal `"=r"(5)` -- not a Name in the env -> a clean
    # CLowerError ("must be a scalar local variable"), never a TypeError treating the IntLit as an lvalue.
    out = _assert_clean_or_diagnoses('void f(void){ asm("" : "=r"(5) : ); }')
    assert out == "CLowerError", out


def test_asm_missing_close_paren_is_a_clean_parse_error():
    # a truncated asm `asm("nop" ;` (missing `)`) -- a clean CParseError, never a crash mid-parse.
    out = _assert_clean_or_diagnoses('void f(void){ asm("nop" ; }')
    assert out == "CParseError", out


def test_asm_constraint_with_no_operand_expr_is_a_clean_parse_error():
    # a constraint with no `(operand)` `"=r" :` -- a clean CParseError (expected `(`), not an index/crash.
    out = _assert_clean_or_diagnoses('void f(void){ asm("" : "=r" : ); }')
    assert out == "CParseError", out


def test_asm_only_colons_no_template_is_a_clean_parse_error():
    # `asm(:::)` -- no template string at all -> a clean CParseError (expected a string literal), no crash.
    out = _assert_clean_or_diagnoses("void f(void){ asm(:::); }")
    assert out == "CParseError", out


def test_asm_volatile_with_empty_parens_is_a_clean_parse_error():
    # `asm volatile()` -- empty parens, no template -> a clean CParseError, never an IndexError on an
    # empty operand-section list.
    out = _assert_clean_or_diagnoses("void f(void){ asm volatile(); }")
    assert out == "CParseError", out


def test_asm_plus_output_with_empty_operand_is_a_clean_parse_error():
    # a read-write `"+r"()` with an EMPTY operand `()` -- the `"+"`-is-also-a-read path must not crash on a
    # missing operand expr; it is a clean CParseError at the parse layer.
    out = _assert_clean_or_diagnoses(
        'unsigned f(unsigned x){ unsigned y=0; asm("" : "+r"() : ); return y; }'
    )
    assert out == "CParseError", out


def test_asm_input_operand_to_undeclared_name_is_a_clean_diagnostic():
    # an INPUT operand reading an undeclared identifier `"r"(zzz)` -- a clean CLowerError from the rvalue
    # lookup (an unknown identifier is rejected), never a KeyError.
    out = _assert_clean_or_diagnoses('void f(void){ asm("" : : "r"(zzz)); }')
    assert out == "CLowerError", out


# =====================================================================================================
# CORPUS B -- port-mapped I/O (`_portio`). The contract validated in `_portio`: arity (inb/inw/inl take
# EXACTLY 1 arg; outb/outw/outl take EXACTLY 2), the port operand must be integer, the value operand must
# be integer. A violation is a clean CLowerError. A non-portio spelling is NOT an intrinsic (it falls
# through to an ordinary call). `outb` is void -- using its result is handled (the void rid). In every
# case: a clean outcome, never an uncaught internal exception (notably no IndexError on `actuals[0]`).
# =====================================================================================================


def test_portio_inb_with_no_args_is_a_clean_diagnostic():
    # `inb()` -- zero args where exactly one is required. The arity check runs BEFORE `actuals[0]`, so it
    # is a clean CLowerError ("takes exactly 1 argument"), NEVER an IndexError on the empty actuals tuple.
    out = _assert_clean_or_diagnoses("unsigned f(void){ return inb(); }")
    assert out == "CLowerError", out


def test_portio_inb_with_two_args_is_a_clean_diagnostic():
    # `inb(0x60, 0x61)` -- two args where exactly one is required -> a clean CLowerError.
    out = _assert_clean_or_diagnoses("unsigned f(void){ return inb(0x60, 0x61); }")
    assert out == "CLowerError", out


def test_portio_outb_with_no_args_is_a_clean_diagnostic():
    # `outb()` -- zero args where exactly two are required. The arity check guards both `actuals[0]` and
    # `actuals[1]`, so it is a clean CLowerError, never an IndexError.
    out = _assert_clean_or_diagnoses("void f(void){ outb(); }")
    assert out == "CLowerError", out


def test_portio_outb_with_one_arg_is_a_clean_diagnostic():
    # `outb(v)` -- one arg where exactly two are required -> a clean CLowerError (the Linux value,port form).
    out = _assert_clean_or_diagnoses("void f(unsigned v){ outb(v); }")
    assert out == "CLowerError", out


def test_portio_outb_with_three_args_is_a_clean_diagnostic():
    # `outb(v, 1, 2)` -- three args where exactly two are required -> a clean CLowerError.
    out = _assert_clean_or_diagnoses("void f(unsigned v){ outb(v, 1, 2); }")
    assert out == "CLowerError", out


def test_portio_float_port_is_a_clean_diagnostic():
    # a floating-point port `inb(p)` with `double p` -- the port must be integer -> a clean CLowerError
    # (`_check_port_operand`), never a TypeError comparing a float CType.
    out = _assert_clean_or_diagnoses("unsigned f(double p){ return inb(p); }")
    assert out == "CLowerError", out


def test_portio_float_value_is_a_clean_diagnostic():
    # a floating-point value `outl(v, 0x1f0)` with `double v` -- the value must be integer -> a clean
    # CLowerError (`_check_value_operand`).
    out = _assert_clean_or_diagnoses("void f(double v){ outl(v, 0x1f0); }")
    assert out == "CLowerError", out


def test_portio_pointer_port_is_a_clean_diagnostic():
    # a pointer port `inb(p)` with `int *p` -- a non-integer, non-scalar port -> a clean CLowerError.
    out = _assert_clean_or_diagnoses("unsigned f(int *p){ return inb(p); }")
    assert out == "CLowerError", out


def test_portio_pointer_value_is_a_clean_diagnostic():
    # a pointer value `outb(p, 0x60)` with `int *p` -- a non-integer value -> a clean CLowerError.
    out = _assert_clean_or_diagnoses("void f(int *p){ outb(p, 0x60); }")
    assert out == "CLowerError", out


def test_portio_struct_port_is_a_clean_diagnostic():
    # an aggregate (struct) port `inb(s)` -- a non-integer aggregate operand -> a clean CLowerError, never
    # an AttributeError reaching into the aggregate CType.
    out = _assert_clean_or_diagnoses(
        "struct S { int a; }; unsigned f(struct S s){ return inb(s); }"
    )
    assert out == "CLowerError", out


def test_portio_struct_value_is_a_clean_diagnostic():
    # an aggregate (struct) value `outb(s, 0x60)` -> a clean CLowerError on the value operand.
    out = _assert_clean_or_diagnoses("struct S { int a; }; void f(struct S s){ outb(s, 0x60); }")
    assert out == "CLowerError", out


def test_portio_string_literal_port_is_a_clean_diagnostic():
    # a string-literal port `inb("hi")` -- not an integer -> a clean CLowerError.
    out = _assert_clean_or_diagnoses('unsigned f(void){ return inb("hi"); }')
    assert out == "CLowerError", out


def test_portio_unknown_spelling_is_not_an_intrinsic_and_lowers_as_a_call():
    # `inq(0x60)` -- NOT one of the six port intrinsics. It must NOT be force-fit into the port table
    # (no KeyError on `_PORTIO_IN[...]`); it falls through to an ordinary unresolved call edge and LOWERS.
    assert _assert_clean_or_diagnoses("unsigned f(void){ return inq(0x60); }") == "LOWERED"


def test_portio_nested_in_in_argument_is_clean():
    # a port read whose ARGUMENT is itself a port read `inb(inb(0x60))` -- the inner read lowers to a temp
    # (an integer), which is a valid port operand for the outer -> LOWERS (no recursion/aliasing crash).
    assert _assert_clean_or_diagnoses("unsigned f(void){ return inb(inb(0x60)); }") == "LOWERED"


def test_portio_nested_out_over_in_is_clean():
    # `outb(inb(0x60), 0x70)` -- the written value is itself a port read. Both port edges lower and share
    # the __ioport resource (ordered) -> LOWERS cleanly.
    assert _assert_clean_or_diagnoses("void f(void){ outb(inb(0x60), 0x70); }") == "LOWERED"


def test_portio_outb_result_used_as_a_value_is_clean():
    # `outb` is VOID -- assigning its result `unsigned y = outb(v, 0x60)` exercises the `_VOID_RID` return
    # path through an assignment. It must LOWER (the void rid is handled), never an internal crash typing a
    # void temp. (A real C compiler would warn "void value not ignored"; the frontend handles it cleanly.)
    assert (
        _assert_clean_or_diagnoses(
            "unsigned f(unsigned v){ unsigned y = outb(v, 0x60); return y; }"
        )
        == "LOWERED"
    )


def test_portio_outb_result_in_arithmetic_is_clean():
    # `outb(v, 0x60) + 1` -- the void result flows into arithmetic. The `_VOID_RID` path must LOWER (or
    # diagnose) cleanly, never a KeyError/TypeError using the void rid as an arithmetic operand.
    assert _assert_clean_or_diagnoses(
        "void f(unsigned v){ unsigned y = outb(v, 0x60) + 1; (void)y; }"
    ) in ("LOWERED", "CLowerError")


def test_portio_inb_result_ignored_as_void_statement_is_clean():
    # `inb(0x60);` as a bare statement (result discarded) -- the side-effecting read still lowers + emits
    # (never DCE'd). LOWERS cleanly.
    assert _assert_clean_or_diagnoses("void f(void){ inb(0x60); }") == "LOWERED"


def test_portio_deeply_nested_reads_are_a_clean_depth_diagnostic_not_recursion():
    # a deeply-nested chain `inb(inb(inb(...0x60...)))` (60 deep) -- the parser's depth guard must reject it
    # as a clean CParseError ("nesting too deep"), NEVER an uncaught RecursionError. This is the precise
    # case the brief flags: a pathological nest must degrade to a clean diagnostic, not blow the stack.
    expr = "0x60"
    for _ in range(60):
        expr = f"inb({expr})"
    out = _assert_clean_or_diagnoses(f"unsigned f(void){{ return {expr}; }}")
    assert out in ("CParseError", "LOWERED"), (
        out
    )  # a clean depth diagnostic (or lowers if under the guard)


# =====================================================================================================
# CORPUS C -- the parse layer behind these features (`parse_unit` directly), to confirm a malformed asm
# STATEMENT (not just the lowering) is a clean CParseError, never an internal parser crash. (The portio
# intrinsics reuse the ordinary call grammar, so their malformed-syntax cases are covered by the parse
# layer's existing call-expression handling and the lowering corpus above.)
# =====================================================================================================


def test_parse_unit_rejects_malformed_asm_cleanly():
    # `parse_unit` (the raw parser the happy-path suite uses) must reject a broken asm with a CParseError,
    # never an internal exception. Probe several broken forms.
    for bad in (
        'void f(void){ asm("nop" ; }',  # missing `)`
        "void f(void){ asm(:::); }",  # no template
        "void f(void){ asm volatile(); }",  # empty parens
        'void f(void){ asm("" : "=r" : ); }',  # constraint, no operand
    ):
        try:
            parse_unit(bad)
            raise AssertionError(f"parse_unit accepted malformed asm: {bad!r}")
        except INTERNAL_BUG_TYPES as e:
            raise AssertionError(
                f"UNCAUGHT INTERNAL EXCEPTION from parse_unit on {bad!r}: {type(e).__name__}: {e}"
            ) from e
        except CParseError:
            pass  # the clean, expected rejection


def test_parse_unit_accepts_asm_goto_grammar_then_lowering_rejects():
    # `asm goto` PARSES (grammar completeness -- `is_goto`/`goto_labels` recorded) but the LOWERING rejects
    # it (CLowerError). Confirm the two-layer split is clean end to end (parse ok, lower diagnoses).
    u = parse_unit('void f(int x){ asm goto("" : : "r"(x) : : L1, L2); L1: ; L2: ; }')
    st = u.funcs[0].body[0]
    assert type(st).__name__ == "AsmStmt" and st.is_goto is True
    out = _assert_clean_or_diagnoses(
        'void f(int x){ asm goto("" : : "r"(x) : : L1, L2); L1: ; L2: ; }'
    )
    assert out == "CLowerError", out


# =====================================================================================================
# A run_all-friendly explicit collector (the suite is auto-collected by `test_*` name in run_all; this
# mirrors the test_area_b_redteam convention for a direct `python -m ...` invocation).
# =====================================================================================================


def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    return len(fns)


if __name__ == "__main__":
    n = run_all()
    print(f"{n} passed")
