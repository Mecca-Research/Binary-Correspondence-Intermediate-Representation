"""ASM1 — inline assembly as an ISA-neutral TRUSTED OPAQUE EDGE in the cfront C subset.

Inline `asm`/`__asm__` is modeled exactly like the `c.call.libm.void:` external-effect family: a trusted
opaque effect edge. BCIR owns the *calling side* (operand binding + constraints + clobber declaration +
ordering semantics); the assembly TEMPLATE is opaque/trusted and re-emitted VERBATIM (so the same asm
compiles on whatever target the C compiler targets — ISA-neutral pass-through). This gate covers:

  * lexing `asm` / `__asm__` (+ the `__volatile__` synonym) as keywords;
  * parsing the basic + extended GNU forms (outputs / inputs / clobbers / volatile / empty sections), and
    rejecting `asm goto` + a non-scalar output lvalue with an honest diagnostic;
  * lowering to a `c.asm:` vs `c.asm.volatile:` claim, with the right read/write operand binding;
  * volatile/basic asm is NOT dead-code-eliminated (a side-effecting edge), and a `"memory"`-clobber /
    volatile asm is an ordering BARRIER (the `barriered` hazard — never reordered/fused across);
  * the asm edge is off the legality value-path (no R-law verdict, `is_clean`);
  * VERBATIM re-emit (the emitted C carries the exact `__asm__(...)` template/constraints/clobbers, in the
    reserved `__asm__`/`__volatile__` spellings so it is valid even under `-std=c11 -pedantic`);
  * emit -> re-parse idempotence (the A3 round-trip analog);
  * gcc AND clang BOTH compile the emitted C, and a `"memory"` barrier preserves the observable value (the
    A2 differential analog). Toolchain-gated: self-skips when a compiler is absent, like the cfront suite.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit, diagnose, parse_unit
from bcir.frontends.cfront.clex import KEYWORDS, tokenize
from bcir.frontends.cfront.emit import emit_function

_GCC = shutil.which("gcc")
_CLANG = shutil.which("clang")
_CC = _CLANG or shutil.which("cc") or _GCC


# --- helpers ------------------------------------------------------------------------------------

def _lf(src: str, fn: str | None = None):
    """Compile `src` (no Clang check) and return the (result, lowered-function) for `fn` (default: the
    last function)."""
    r = compile_unit(src, check_clang=False)
    fns = r.lowered.functions
    lf = fns[fn] if fn else fns[next(reversed(fns))]
    return r, lf


def _asm_claims(lf):
    return [c for c in lf.claims if c.op.startswith("c.asm")]


def _emit_clean(lf) -> str:
    """The emitted function body (attestation comment stripped) — for re-parse + the compile harness."""
    import re

    return re.sub(r"/\*.*?\*/\n?", "", emit_function(lf), flags=re.S)


def _both_compile(c_text: str) -> str:
    """Compile `c_text` under BOTH gcc AND clang at -std=c11 -pedantic (the reserved-spelling lane) and run
    it; return "ok" iff every available compiler builds AND the program exits 0, "skip" if no compiler is
    present, else a "FAIL:<compiler>:<detail>" string. The emitted `__asm__`/`__volatile__` is valid ISO+GNU,
    so -pedantic must not warn it away."""
    ccs = [(n, p) for n, p in (("gcc", _GCC), ("clang", _CLANG)) if p]
    if not ccs:
        if not _CC:
            return "skip"
        ccs = [("cc", _CC)]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "asm.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(c_text)
        for name, cc in ccs:
            exe = os.path.join(d, f"asm_{name}")
            b = subprocess.run([cc, "-std=c11", "-pedantic", "-Wall", "-O2", src, "-o", exe],
                               capture_output=True, text=True)
            if b.returncode != 0:
                return f"FAIL:{name}:build:{(b.stderr or b.stdout).strip().splitlines()[-1:]}"
            run = subprocess.run([exe], capture_output=True, text=True)
            if run.returncode != 0:
                return f"FAIL:{name}:run:rc={run.returncode}:{run.stdout.strip()}"
    return "ok"


# --- (1) lexing ---------------------------------------------------------------------------------

def test_asm_keywords_are_registered():
    # only the DOUBLE-UNDERSCORE spellings are reserved keywords (implementation-reserved, benign). Plain
    # `asm` is NOT a keyword -- it stays a usable ISO-C identifier under -std=c11 (see
    # test_plain_asm_is_a_usable_identifier); the asm STATEMENT is recognized contextually.
    assert {"__asm__", "__volatile__"} <= KEYWORDS
    assert "asm" not in KEYWORDS
    assert "volatile" in KEYWORDS              # already a keyword (it lexes the volatile spelling)


def test_asm_lexes_as_an_identifier_token():
    # keywords + `asm` all lex as IDENT tokens (the parser decides their role).
    toks = [t.text for t in tokenize('asm __asm__ __volatile__ x') if t.kind == "IDENT"]
    assert toks == ["asm", "__asm__", "__volatile__", "x"]


def test_asm_string_template_lexes_with_escapes_intact():
    toks = [t for t in tokenize(r'asm("mov\t%0, %1")') if t.kind == "STRING"]
    assert toks and toks[0].text == r'"mov\t%0, %1"'


# --- (2) parsing --------------------------------------------------------------------------------

def test_parse_basic_asm():
    u = parse_unit('void f(void){ asm("nop"); }')
    s = u.funcs[0].body[0]
    assert type(s).__name__ == "AsmStmt"
    assert s.template == '"nop"' and s.outputs == () and s.inputs == () and s.clobbers == ()
    assert s.is_volatile is True               # a BASIC asm is implicitly volatile (§6.47.1)


def test_parse_both_keyword_spellings():
    for kw in ("asm", "__asm__"):
        s = parse_unit(f'void f(void){{ {kw}("nop"); }}').funcs[0].body[0]
        assert type(s).__name__ == "AsmStmt" and s.template == '"nop"'


def test_parse_volatile_qualifiers():
    for q in ("volatile", "__volatile__"):
        s = parse_unit(f'void f(void){{ asm {q}("" ::: "memory"); }}').funcs[0].body[0]
        assert s.is_volatile is True


def test_parse_extended_outputs_inputs_clobbers():
    s = parse_unit('unsigned f(unsigned x){ unsigned y=0; '
                   'asm("mov %1,%0" : "=r"(y) : "r"(x) : "cc"); return y; }').funcs[0].body[1]
    assert s.template == '"mov %1,%0"'
    assert len(s.outputs) == 1 and s.outputs[0][1] == '"=r"'
    assert len(s.inputs) == 1 and s.inputs[0][1] == '"r"'
    assert s.clobbers == ('"cc"',)
    assert s.is_volatile is False              # an extended asm without `volatile` is NOT volatile


def test_parse_empty_operand_sections_are_valid():
    s = parse_unit('void f(void){ asm("" : : : "memory"); }').funcs[0].body[0]
    assert s.outputs == () and s.inputs == () and s.clobbers == ('"memory"',)


def test_parse_symbolic_operand_names():
    s = parse_unit('unsigned f(unsigned x){ unsigned y=0; '
                   'asm("" : [out]"=r"(y) : [in]"r"(x)); return y; }').funcs[0].body[1]
    assert s.outputs[0][0] == "out" and s.inputs[0][0] == "in"


def test_parse_adjacent_template_concatenation():
    s = parse_unit('void f(void){ asm("nop\\n\\t" "nop"); }').funcs[0].body[0]
    assert "nop" in s.template and s.template.count('"') >= 2     # both literals kept (adjacency)


def test_parse_asm_goto_is_recorded():
    s = parse_unit('void f(int x){ asm goto("" : : "r"(x) : : L1, L2); L1: ; L2: ; }').funcs[0].body[0]
    assert s.is_goto is True and s.goto_labels == ("L1", "L2")


def test_plain_asm_is_a_usable_identifier():
    # `asm` is NOT a global keyword (only `__asm__`/`__volatile__` are reserved) -- it stays a usable ISO-C
    # identifier under -std=c11, recognized as a STATEMENT only when followed by `(` or a qualifier. (Regression
    # for the Clang-equivalence break where a plain-`asm` keyword rejected `int asm = 3;` etc.)
    # a local named `asm`:
    s = parse_unit('int f(void){ int asm = 3; return asm; }').funcs[0].body
    assert type(s[0]).__name__ == "Decl" and s[0].name == "asm"
    # a parameter named `asm`:
    g = parse_unit('int g(int asm){ return asm + 1; }').funcs[0]
    assert g.params[0].name == "asm"
    # `asm` assigned as a statement (not an asm-statement -- no following `(`/qualifier):
    h = parse_unit('int h(void){ int asm = 1; asm = 4; return asm; }').funcs[0].body
    assert type(h[1]).__name__ == "ExprStmt"
    # and an asm STATEMENT still parses when followed by `(` or a qualifier:
    a = parse_unit('void k(void){ asm("nop"); asm volatile("" ::: "memory"); }').funcs[0].body
    assert type(a[0]).__name__ == "AsmStmt" and type(a[1]).__name__ == "AsmStmt"


def test_asm_as_identifier_is_clang_equivalent():
    # the original regression: `int g(int asm)` must compile clean AND stay Clang-equivalent (asm is an
    # ordinary identifier, not a keyword). Self-skips cleanly without a compiler.
    r = compile_unit('int g(int asm){ return asm + 1; }', check_clang=True)
    assert r.is_clean
    assert r.equivalence == "match" or r.equivalence.startswith("skip"), r.equivalence


def test_missing_comma_clobber_is_a_clean_diagnostic():
    # FIX 3: a clobber is exactly ONE string token -- a missing-comma list `"cc" "memory"` must be a clean
    # CParseError (not silently merged into one clobber `cc" "memory`, which would miss the "memory" barrier).
    rep = diagnose('void f(void){ asm("" : : : "cc" "memory"); }')
    assert not rep.ok and rep.diagnostics


def test_missing_comma_constraint_is_a_clean_diagnostic():
    # FIX 3: same for a constraint (one string token) -- `"=r" "x"(y)` is not a merged constraint.
    rep = diagnose('unsigned f(unsigned x){ unsigned y=0; asm("" : "=r" "x"(y) : "r"(x)); return y; }')
    assert not rep.ok and rep.diagnostics


def test_non_goto_trailing_labels_section_is_a_clean_parse_error():
    # FIX 4: only `asm goto` has a 5th (labels) section -- a non-goto extended asm with a trailing `: labels`
    # is a clean syntax error at the parse layer (not a misleading downstream "asm goto" message).
    rep = diagnose('void f(int x){ asm("" : : "r"(x) : "cc" : L1); L1: ; }')
    assert not rep.ok and rep.diagnostics
    assert "asm goto" not in rep.diagnostics[0].message       # a syntax error, not the goto rejection


def test_malformed_asm_is_a_clean_diagnostic():
    rep = diagnose('void f(void){ asm("nop" ; }')               # missing `)`
    assert not rep.ok and rep.diagnostics                       # a diagnostic, not a crash


def test_asm_goto_is_rejected_with_an_honest_diagnostic():
    rep = diagnose('void f(void){ asm goto("" : : : : L); L: ; }')
    assert not rep.ok and "asm goto" in rep.diagnostics[0].message


def test_non_scalar_output_is_rejected_with_an_honest_diagnostic():
    rep = diagnose('void f(int *p){ asm("" : "=r"(*p) : ); }')
    assert not rep.ok and "scalar local" in rep.diagnostics[0].message


# --- (3) lowering -------------------------------------------------------------------------------

def test_basic_asm_lowers_to_c_asm_volatile():
    _r, lf = _lf('void f(void){ asm("nop"); }')
    cs = _asm_claims(lf)
    assert len(cs) == 1 and cs[0].op == "c.asm.volatile:"       # basic == implicitly volatile


def test_nonvolatile_extended_asm_lowers_to_c_asm():
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; asm("" : "=r"(y) : "r"(x)); return y; }')
    cs = _asm_claims(lf)
    assert len(cs) == 1 and cs[0].op == "c.asm:"               # not volatile -> plain c.asm:


def test_volatile_extended_asm_lowers_to_c_asm_volatile():
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; '
                 'asm volatile("" : "=r"(y) : "r"(x)); return y; }')
    assert _asm_claims(lf)[0].op == "c.asm.volatile:"


def test_operand_binding_read_and_write_sets():
    # `"=r"(y) : "r"(x)`: y (output) is a WRITE, x (input) is a READ.
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; asm("" : "=r"(y) : "r"(x)); return y; }')
    c = _asm_claims(lf)[0]
    yrid = next(rid for rid, _n, _ct in lf.locals if _n == "y")
    xrid = next(rid for n, rid, _ct in lf.params if n == "x")
    assert c.wr == (yrid,) and xrid in c.rd


def test_readwrite_output_is_both_read_and_written():
    # `"+r"(y)`: y is BOTH read and written.
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=x; asm("" : "+r"(y) : ); return y; }')
    c = _asm_claims(lf)[0]
    yrid = next(rid for rid, _n, _ct in lf.locals if _n == "y")
    assert yrid in c.rd and c.wr == (yrid,)


def test_asm_edge_is_off_the_legality_value_path():
    # a trusted opaque effect: no R-law verdict, the unit is clean (like c.call.libm.void:).
    r, _lfn = _lf('void f(void){ asm volatile("" ::: "memory"); }')
    assert r.is_clean and not r.diagnostics


# --- (4) barrier / not-eliminated semantics -----------------------------------------------------

def test_memory_clobber_is_a_barrier_hazard():
    _r, lf = _lf('void f(void){ asm("" ::: "memory"); }')
    assert _asm_claims(lf)[0].hazard == "barriered"


def test_volatile_asm_is_a_barrier_hazard():
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; asm volatile("" : "=r"(y) : "r"(x)); return y; }')
    assert _asm_claims(lf)[0].hazard == "barriered"


def test_plain_asm_without_memory_clobber_is_not_a_barrier():
    _r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; asm("" : "=r"(y) : "r"(x)); return y; }')
    assert _asm_claims(lf)[0].hazard == "unique"


def test_volatile_asm_with_unused_output_is_not_eliminated():
    # the cfront emit walks lf.body (every claim, source order) -- it never reorders or DROPS a claim, so a
    # side-effecting volatile asm survives even when its output is unused.
    r, lf = _lf('void f(void){ unsigned y=0; asm volatile("" : "=r"(y) : : "memory"); }')
    assert _asm_claims(lf), "the volatile asm claim was dropped from the lowered graph"
    assert "__asm__" in r.emitted["f"], "the volatile asm was eliminated from the emit"
    assert r.plans["f"].steps                                  # the plan still realizes the claim


def test_barrier_is_emitted_in_source_position_between_memory_ops():
    # the barrier must sit BETWEEN the store and the load in the emit (not hoisted/sunk across them).
    src = ('unsigned f(unsigned *p, unsigned x){ *p = x; '
           'asm volatile("" ::: "memory"); return *p; }')
    _r, lf = _lf(src)
    body = _emit_clean(lf)
    store_i = body.index("memcpy") if "memcpy" in body else body.index("*")
    asm_i = body.index("__asm__")
    load_i = body.rindex("return")
    assert store_i < asm_i < load_i, f"barrier not between store and load:\n{body}"


# --- (5) verbatim re-emit -----------------------------------------------------------------------

def test_basic_asm_emits_verbatim_reserved_spelling():
    r, _lfn = _lf('void f(void){ asm("nop"); }')
    body = r.emitted["f"]
    assert '__asm__ __volatile__ ("nop");' in body              # reserved spelling, valid under -pedantic
    assert "asm(" not in body.replace("__asm__", "")            # not the bare `asm` spelling


def test_memory_barrier_emits_verbatim():
    r, _lfn = _lf('void f(void){ asm volatile("" ::: "memory"); }')
    assert '__asm__ __volatile__ ("" :  :  : "memory");' in r.emitted["f"]


def test_extended_asm_emits_operands_and_clobbers_verbatim():
    r, lf = _lf('unsigned f(unsigned x){ unsigned y=0; '
                'asm("mov %1,%0" : "=r"(y) : "r"(x) : "cc"); return y; }')
    body = r.emitted["f"]
    assert '__asm__ ("mov %1,%0" : "=r" (y) : "r" (x) : "cc");' in body


def test_symbolic_names_emit_verbatim():
    r, _lfn = _lf('unsigned f(unsigned x){ unsigned y=0; '
                 'asm("" : [out]"=r"(y) : [in]"r"(x)); return y; }')
    body = r.emitted["f"]
    assert '[out] "=r" (y)' in body and '[in] "r" (x)' in body


# --- (6) emit -> re-parse idempotence (the A3 round-trip analog) ---------------------------------

def _asm_signature(lf):
    return [(c.op, len(c.rd), len(c.wr), c.hazard) for c in lf.claims if c.op.startswith("c.asm")]


def test_emit_reparse_idempotence():
    for src in (
        'void f(void){ asm("nop"); }',
        'void g(void){ asm volatile("" ::: "memory"); }',
        'unsigned h(unsigned x){ unsigned y=0; asm("" : "=r"(y) : "r"(x)); return y; }',
        'unsigned k(unsigned x){ unsigned y=x; asm volatile("" : "+r"(y) : : "cc"); return y; }',
        # FIX 2: an all-empty EXTENDED (non-volatile) asm must NOT collapse to the implicitly-volatile basic
        # form on re-emit -- so its `c.asm:`/`unique` signature is a fixed point, not `c.asm.volatile:`.
        'void m(void){ asm("nop" : :); }',
        'void n(void){ asm("x" : : :); }',
    ):
        _r1, lf1 = _lf(src)
        e1 = _emit_clean(lf1)
        _r2, lf2 = _lf(e1)
        e2 = _emit_clean(lf2)
        _r3, lf3 = _lf(e2)
        # the observable asm signature is a fixed point from the first emit onward.
        assert _asm_signature(lf2) == _asm_signature(lf3) == _asm_signature(lf1), (
            f"asm round-trip drift for {src!r}: {_asm_signature(lf1)} -> "
            f"{_asm_signature(lf2)} -> {_asm_signature(lf3)}")


def test_all_empty_extended_asm_stays_nonvolatile():
    # FIX 2: `asm("nop" : :)` is EXTENDED (has colon sections) -> non-volatile `c.asm:`/`unique`. It must
    # re-emit with its `:` sections (not as the basic `__asm__ ("nop");`, which re-parses to volatile).
    _r, lf = _lf('void f(void){ asm("nop" : :); }')
    c = _asm_claims(lf)[0]
    assert c.op == "c.asm:" and c.hazard == "unique"
    body = _emit_clean(lf)
    assert '__asm__ ("nop" :  :  : );' in body                  # the extended (3-colon) form, not the basic form
    assert '__asm__ __volatile__ ("nop");' not in body


# --- (7) gcc AND clang BOTH compile the emit; the barrier preserves the value (A2 analog) --------

_HARNESS = """#include <stdint.h>
#include <stdio.h>
#include <string.h>
{emit}
int main(void){{
  /* the empty-template `"=r"(out) : "0"(in)` form (a tied register) is an ISA-neutral copy; the `"memory"`
     barrier is a compiler ordering fence that must NOT change the observable value. Both hold under gcc+clang. */
  unsigned acc = 0;
  for (unsigned x = 0; x < 1000u; x++) {{
    if (bcir_copy(x) != x) return 1;            /* the "=r"/"r" copy is the identity */
    acc += bcir_barriered_sum(x, 7u);           /* the barrier-fenced store/load returns x+7 */
    if (bcir_barriered_sum(x, 7u) != x + 7u) return 2;
  }}
  bcir_just_a_barrier();                         /* a basic + a memory-barrier asm with no operands */
  (void)acc;
  puts("OK");
  return 0;
}}
"""

_PROG = """
unsigned copy(unsigned x){ unsigned y = 0; __asm__("" : "=r"(y) : "0"(x)); return y; }
unsigned barriered_sum(unsigned a, unsigned b){
  unsigned s = a + b;
  __asm__ __volatile__("" ::: "memory");        /* an ordering fence around the value */
  return s;
}
void just_a_barrier(void){
  __asm__("nop");
  __asm__ __volatile__("" ::: "memory");
}
"""


def test_emitted_asm_compiles_under_gcc_and_clang_and_preserves_value():
    r = compile_unit(_PROG, check_clang=False)
    emit = "\n\n".join(_emit_clean(lf) for lf in r.lowered.functions.values())
    verdict = _both_compile(_HARNESS.format(emit=emit))
    assert verdict in ("ok", "skip"), f"emitted asm differential failed: {verdict}"


def test_original_and_emit_are_behaviour_equivalent_under_clang():
    # the in-process Clang differential (the pipeline's own equivalence harness): the ORIGINAL asm program
    # and the emitted bcir_* agree on seeded inputs. Self-skips cleanly without a compiler.
    r = compile_unit('unsigned f(unsigned x){ unsigned y = x; '
                     '__asm__ __volatile__("" ::: "memory"); return y; }', check_clang=True)
    assert r.is_clean
    assert r.equivalence == "match" or r.equivalence.startswith("skip"), r.equivalence


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
