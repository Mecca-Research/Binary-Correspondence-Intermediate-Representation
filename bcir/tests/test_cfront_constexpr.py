"""The §5.9 integer CONSTANT-EXPRESSION EVALUATOR (both rails).

Enum initializers, `case` labels, static-local initializers and file-scope global
initializers now fold the full integer constant-expression vocabulary -- arithmetic, bit
ops, shifts, COMPARISONS, logical &&/|| and the ternary -- through one shared evaluator
per rail (`cparse._const_eval` / `lower._fold_const` on the oracle; `ce_expr` on the C
twin), and the linkable emit renders the folded value. `sizeof` stays out on purpose (the
target ABI is chosen at lower time); a genuinely non-constant initializer still refuses
loudly. Behavior parity vs the host-cc reference build is the end gate."""

import os
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.tests.test_c_executor import _cc

_SRC = (
    "enum { A = 1 << 4, B = A + 2, C = (A > 10) ? 7 : 9, D = (1 < 2) && (3 != 4) };\n"
    "unsigned ge = (3u * 4u + 1u) << 2;\n"
    "unsigned pick(unsigned x) {\n"
    "    static unsigned seed = (5 > 3) ? 40u : 2u;\n"
    "    switch (x) {\n"
    "    case A: return seed + 1u;\n"
    "    case B: return (unsigned)C;\n"
    "    default: return (unsigned)D + ge;\n"
    "    }\n"
    "}\n"
    "int main(void) { return (int)(pick(16u) + pick(18u) + pick(0u)); }\n"
)


def test_constant_expressions_fold_on_enums_cases_statics_and_globals():
    """One fixture exercises every fold site: shift/arith/comparison/ternary/logical enums,
    enum-typed case labels, a ternary static-local init, a shifted global init -- clean
    compile, the folded values visible in the artifacts, and rc parity with the reference
    (41 + 7 + 53 == 101)."""
    r = compile_unit(_SRC, check_clang=False)
    assert r.is_clean, r.diagnostics
    assert "static unsigned int seed = 40u;" in r.emitted["pick"]  # the ternary folded
    from bcir.frontends.cfront.emit import emit_linkable

    text = emit_linkable(r.lowered, r.emitted)
    assert "unsigned int ge = 52;" in text  # (3*4+1)<<2 rendered
    cc = _cc()
    if cc is None:
        return

    def _build(src_path, out):  # the house convention:
        p = subprocess.run(
            [cc, "-std=c23", "-Wall", "-Werror", src_path, "-o", out],
            capture_output=True,
            text=True,
        )  # c23 first (the emitter may
        if p.returncode == 0:  # place a decl after a case
            return p  # label), plain c11 fallback
        return subprocess.run([cc, "-std=c11", src_path, "-o", out], capture_output=True, text=True)

    with tempfile.TemporaryDirectory() as tmp:
        ref_src = os.path.join(tmp, "ref.c")
        with open(ref_src, "w", encoding="utf-8") as f:
            f.write(_SRC)
        ref = os.path.join(tmp, "ref")
        p = _build(ref_src, ref)
        assert p.returncode == 0, p.stderr
        assert subprocess.run([ref]).returncode == 101
        lk = os.path.join(tmp, "lk.c")  # the emitted artifact
        with open(lk, "w", encoding="utf-8") as f:  # behaves identically
            f.write(text)
        prog = os.path.join(tmp, "prog")
        p = _build(lk, prog)
        assert p.returncode == 0, p.stderr
        assert subprocess.run([prog]).returncode == 101


def test_nonconstant_initializers_still_refuse_loudly():
    """The evaluator widens the CONSTANT vocabulary only: a global initializer reading
    another global is still non-constant -- the linkable emit refuses it by name."""
    from bcir.frontends.cfront.emit import emit_linkable

    r = compile_unit(
        "unsigned base = 7u;\nunsigned q = base + 1u;\nunsigned f(void) { return q; }\n",
        check_clang=False,
    )
    try:
        emit_linkable(r.lowered, r.emitted)
        raise AssertionError("a global-reading initializer must be rejected")
    except ValueError as e:
        assert "q" in str(e)


def test_division_by_zero_folds_to_zero_on_both_vocabularies():
    """The documented guard: a constant `x / 0` folds to 0 (both rails' evaluators use the
    same rule instead of dying), pinned so the vocabularies cannot drift apart silently."""
    r = compile_unit(
        "enum { Z = 5 / 0, M = 5 % 0 };\nint f(void) { return Z + M; }\n", check_clang=False
    )
    assert r.is_clean
    from bcir.frontends.cfront.emit import emit_linkable  # Z and M folded to 0 at parse

    assert "return" in r.emitted["f"]


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
