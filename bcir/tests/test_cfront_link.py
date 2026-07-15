"""Phase 3 LINKING, first slice: function PROTOTYPES + the cross-TU call edge (`c.call.tu:`)
+ the host-linker end-to-end.

A file-scope prototype (`T name(params);`) now parses on the oracle rail; a CALL to a
prototyped-but-undefined callee is a TYPED external edge the host linker resolves from a
sibling object -- like a libm edge it is opaque to the in-unit R18 call graph and emits
VERBATIM (external linkage, no `bcir_` rename) with an `extern` declaration so the emitted
TU compiles standalone; unlike libm it derives NO -l flag (a sibling TU is not a library).
A prototype whose definition lives in the SAME unit is just a forward declaration -- the
definition wins and the call stays a real R18 edge. The headline is the two-TU binary:
the bcir-EMITTED caller object, linked by the host linker against the callee TU with the
DERIVED --emit-link-flags, behaves exactly like the all-original reference build.
(The C-twin port -- prototypes in bcir_cfront.c -- and the linkable non-static emission
mode are the named next steps; see the master roadmap Phase 3.)"""

import os
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.linkflags import format_link_flags
from bcir.tests.test_c_executor import _cc
from bcir.toolchain import host_link_args

_MAIN = "double scale(double x);\nint main(void) { double v = scale(16.0); return (int)v; }\n"
_LIB = "#include <math.h>\ndouble scale(double x) { return sqrt(x) + 1.0; }\n"


def test_a_prototyped_callee_is_a_typed_cross_tu_edge():
    r = compile_unit(_MAIN, check_clang=False)
    assert r.is_clean and r.r18_ok                       # NOT an R18 undefined-function error
    ops = [c.op for c in r.lowered.functions["main"].claims]
    assert "c.call.tu:scale" in ops, ops
    assert "extern double scale(double);" in r.emitted["main"]   # the standalone declaration
    assert "scale(t" in r.emitted["main"]                # verbatim call: external linkage,
    assert "bcir_scale" not in r.emitted["main"]         # no bcir_ rename


def test_a_forward_declaration_stays_a_real_r18_edge():
    src = ("unsigned g(unsigned x);\n"
           "unsigned f(unsigned y) { return g(y) + 1u; }\n"
           "unsigned g(unsigned x) { return x * 2u; }\n")
    r = compile_unit(src, check_clang=False)
    assert r.is_clean and r.r18_ok
    ops = [c.op for c in r.lowered.functions["f"].claims]
    assert "c.call:g" in ops and not any(o.startswith("c.call.tu:") for o in ops)


def test_no_link_flag_derives_from_a_tu_edge():
    caller = compile_unit(_MAIN, check_clang=False)
    assert format_link_flags(caller.link_flags) in ("", "-")     # a sibling TU, not a library
    callee = compile_unit(_LIB, check_clang=False)
    assert "-lm" in format_link_flags(callee.link_flags)         # the callee's sqrt DOES derive -lm


def test_the_two_tu_binary_links_with_derived_flags_and_behaves():
    """THE HEADLINE: the bcir-emitted caller object + the callee TU, linked by the HOST
    linker with the derived flags, exits exactly like the all-original reference build."""
    cc = _cc()
    if cc is None:
        return
    caller = compile_unit(_MAIN, check_clang=False)
    callee = compile_unit(_LIB, check_clang=False)
    flags = [f for f in format_link_flags(callee.link_flags).split() if f.startswith("-l")]
    with tempfile.TemporaryDirectory() as tmp:
        lib = os.path.join(tmp, "lib.c")
        main_ref = os.path.join(tmp, "main.c")
        with open(lib, "w", encoding="utf-8") as f:
            f.write(_LIB)
        with open(main_ref, "w", encoding="utf-8") as f:
            f.write(_MAIN)
        ref = os.path.join(tmp, "ref")
        r = subprocess.run(host_link_args([cc, "-std=c11", main_ref, lib, "-o", ref, "-lm"]),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        rc_ref = subprocess.run([ref]).returncode

        emitted = os.path.join(tmp, "main_emit.c")      # the bcir artifact: emitted caller +
        with open(emitted, "w", encoding="utf-8") as f:  # a thin entry adapter in the same TU
            f.write("#include <stdint.h>\n" + caller.emitted["main"]
                    + "\nint main(void) { return (int)bcir_main(); }\n")
        prog = os.path.join(tmp, "prog")
        r = subprocess.run([cc, "-std=c11", emitted, lib, "-o", prog, *flags],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr               # the emitted TU compiles + LINKS standalone
        rc_bcir = subprocess.run([prog]).returncode
    assert rc_bcir == rc_ref == 5, (rc_bcir, rc_ref)     # sqrt(16)+1 == 5: behaviour equivalence


def test_linkable_emit_links_two_emitted_tus():
    """THE PHASE-3 CLOSER: BOTH TUs re-rendered by the linkable emit (external linkage, real
    names, derived includes) link TO EACH OTHER -- no original source in the final image --
    and the binary behaves exactly like the all-original reference build."""
    cc = _cc()
    if cc is None:
        return
    from bcir.frontends.cfront.emit import emit_linkable
    caller = compile_unit(_MAIN, check_clang=False)
    callee = compile_unit(_LIB, check_clang=False)
    with tempfile.TemporaryDirectory() as tmp:
        m = os.path.join(tmp, "m.c")
        li = os.path.join(tmp, "l.c")
        with open(m, "w", encoding="utf-8") as f:
            f.write(emit_linkable(caller.lowered, caller.emitted))
        with open(li, "w", encoding="utf-8") as f:
            f.write(emit_linkable(callee.lowered, callee.emitted))
        prog = os.path.join(tmp, "prog")
        r = subprocess.run(host_link_args(
            [cc, "-std=c11", "-Wall", "-Werror", m, li, "-o", prog, "-lm"]),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr                    # self-contained, warning-free
        assert subprocess.run([prog]).returncode == 5


def test_linkable_globals_define_declare_and_reject_nonconstant():
    from bcir.frontends.cfront.emit import emit_linkable
    user = compile_unit("extern unsigned base;\n"
                        "unsigned scaled(unsigned x) { return base * x + 1u; }\n",
                        check_clang=False)
    definer = compile_unit("unsigned base = 7u;\nunsigned tab[3] = {1u, 2u, 3u};\n"
                           "unsigned peek(void) { return tab[0] + base; }\n",
                           check_clang=False)
    u_text = emit_linkable(user.lowered, user.emitted)
    d_text = emit_linkable(definer.lowered, definer.emitted)
    assert "extern unsigned int base;" in u_text              # a declaration, never a definition
    assert "unsigned int base = 7;" in d_text                 # the defining TU renders the value
    assert "unsigned int tab[3] = {1, 2, 3};" in d_text
    assert "unsigned int scaled(unsigned int x)" in u_text    # non-static, real name
    assert "static" not in u_text.replace("_Static_assert", "")
    cc = _cc()
    if cc is not None:                                        # the pair genuinely links + runs
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.c")
            b = os.path.join(tmp, "b.c")
            main = os.path.join(tmp, "main.c")
            for path, text in ((a, u_text), (b, d_text)):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            with open(main, "w", encoding="utf-8") as f:
                f.write("extern unsigned scaled(unsigned);\n"
                        "int main(void){ return (int)scaled(6u); }\n")
            prog = os.path.join(tmp, "prog")
            from bcir.tests.test_c_executor import _RUNTIME_C
            r = subprocess.run([cc, "-std=c11", "-I", _RUNTIME_C, main, a, b,
                                os.path.join(_RUNTIME_C, "bcir_quarantine.c"), "-o", prog],
                               capture_output=True, text=True)   # the documented contract: a masked
            assert r.returncode == 0, r.stderr                   # access links bcir_quarantine.c
            assert subprocess.run([prog]).returncode == 43    # 7*6+1
    # Part VII A4: &x ADDRESS CONSTANTS and sizeof-bearing initializers now RENDER --
    # the SS5.9 evaluator consults the chosen ABI's layout oracle (sizeof folds at
    # LOWER time, where the ABI lives) and an address constant is the linker's
    # relocation, rendered by name. A bare `return g;` (zero claims) names the global
    # too (the raw-rid mis-render this test flushed out). Forward-referencing address
    # constants (illegal C: use before declaration) still refuse LOUDLY.
    fl = compile_unit("unsigned base = 7u;\nunsigned *pp = &base;\n"
                      "unsigned sz = sizeof(unsigned) * 4u;\n"
                      "unsigned geto(void) { return base; }\n", check_clang=False)
    a4 = emit_linkable(fl.lowered, fl.emitted)
    assert "unsigned int * pp = &base;" in a4                 # the relocation, by name
    assert "unsigned int sz = 16;" in a4                      # sizeof folded on the ABI
    assert "return base;" in a4                               # the bare global return
    if cc is not None:                                        # ... and it genuinely runs
        with tempfile.TemporaryDirectory() as tmp:
            unit = os.path.join(tmp, "unit.c")
            main = os.path.join(tmp, "main.c")
            with open(unit, "w", encoding="utf-8") as f:
                f.write(a4)
            with open(main, "w", encoding="utf-8") as f:
                f.write("extern unsigned *pp;\nextern unsigned sz;\n"
                        "int main(void){ return (int)(*pp + sz); }\n")
            prog = os.path.join(tmp, "prog")
            r = subprocess.run([cc, "-std=c11", "-Wall", "-Werror", main, unit,
                                "-o", prog], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert subprocess.run([prog]).returncode == 23    # *(&base) + 4*sizeof = 7+16
    fwd = compile_unit("unsigned *qq = &later;\nunsigned later = 3u;\n"
                       "unsigned geti(void) { return later; }\n", check_clang=False)
    try:
        emit_linkable(fwd.lowered, fwd.emitted)
        raise AssertionError("a forward-referencing address constant must refuse")
    except ValueError as e:
        assert "qq" in str(e)


def test_linkable_honors_source_static_functions_and_globals():
    """Source-`static` honoring: each TU keeps its OWN same-named static helper (internal
    linkage, real name) and a static global stays file-local -- the two TUs still link into
    one binary, which a stripped-static (exported) rendering could not do (duplicate symbol)."""
    from bcir.frontends.cfront.emit import emit_linkable
    tu_a = compile_unit("static unsigned mix(unsigned x) { return x * 3u; }\n"
                        "static unsigned seed = 5u;\n"
                        "unsigned fa(unsigned x) { return mix(x) + seed; }\n",
                        check_clang=False)
    tu_b = compile_unit("static unsigned mix(unsigned x) { return x + 100u; }\n"
                        "unsigned fb(unsigned x) { return mix(x); }\n",
                        check_clang=False)
    a_text = emit_linkable(tu_a.lowered, tu_a.emitted)
    b_text = emit_linkable(tu_b.lowered, tu_b.emitted)
    assert "static unsigned int mix(unsigned int x)" in a_text    # static KEPT, real name
    assert "static unsigned int seed = 5;" in a_text              # file-local global
    assert "\nunsigned int fa(unsigned int x)" in a_text          # the export is non-static
    assert "static unsigned int mix(unsigned int x)" in b_text
    cc = _cc()
    if cc is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        a = os.path.join(tmp, "a.c")
        b = os.path.join(tmp, "b.c")
        main = os.path.join(tmp, "main.c")
        for path, text in ((a, a_text), (b, b_text)):
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        with open(main, "w", encoding="utf-8") as f:
            f.write("extern unsigned fa(unsigned);\nextern unsigned fb(unsigned);\n"
                    "int main(void){ return (int)(fa(2u) + fb(1u)); }\n")
        prog = os.path.join(tmp, "prog")
        r = subprocess.run([cc, "-std=c11", "-Wall", "-Werror", main, a, b, "-o", prog],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr        # two same-named statics link CLEANLY
        assert subprocess.run([prog]).returncode == 112   # fa(2)=2*3+5=11, fb(1)=101


def test_linkable_static_forward_declaration_and_pointer_string_table():
    """Review-hardened edges: (1) the C static-forward-declaration idiom -- a static callee
    DEFINED AFTER its caller -- must yield a compilable artifact (the emit forward-declares
    every kept-static function); (2) a pointer-element string table (`char *tab[] = {..}`)
    must NOT take the char-array string path (it would size the array from the string bytes
    and mis-render) -- it refuses loudly, and the array keeps its init-tuple count."""
    from bcir.frontends.cfront.emit import emit_linkable
    fwd = compile_unit("static int r(int v);\nint s(int x) { return r(x); }\n"
                       "static int r(int x) { return x + 1; }\n", check_clang=False)
    text = emit_linkable(fwd.lowered, fwd.emitted)
    assert "static int r(int);" in text                       # the forward declaration
    cc = _cc()
    if cc is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fwd.c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            r = subprocess.run([cc, "-std=c11", "-Wall", "-Werror", "-c", path,
                                "-o", os.path.join(tmp, "fwd.o")],
                               capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
    ptab = compile_unit('char *tab[] = {"hi"};\nunsigned f4(void) { return 3u; }\n',
                        check_clang=False)
    _, ct, vals, _, _ = [g for g in ptab.lowered.globals_decl if g[0] == "tab"][0]
    assert ct.count == 1 and vals is None                     # tuple-sized, not string-sized
    try:
        emit_linkable(ptab.lowered, ptab.emitted)
        raise AssertionError("a pointer-element string table must be rejected")
    except ValueError as e:
        assert "tab" in str(e)


def test_linkable_renders_float_negative_and_string_initializers():
    """The renderer past the first slice: float spellings (suffix kept), signed integer
    constants, and string-initialized char arrays (sized from the LITERAL, not the init
    tuple) all render -- and the artifact is host-compilable stand-alone."""
    from bcir.frontends.cfront.emit import emit_linkable
    r = compile_unit("double dv = 1.5;\ndouble nv = -2.5;\nint off = -7;\n"
                     "char msg[] = \"hi\";\nfloat fs = 0.5f;\n"
                     "unsigned fz(unsigned x) { return x + 1u; }\n", check_clang=False)
    text = emit_linkable(r.lowered, r.emitted)
    assert "double dv = 1.5;" in text
    assert "double nv = -2.5;" in text
    assert "int off = -7;" in text
    assert 'char msg[3] = "hi";' in text                  # 2 code units + the NUL
    assert "float fs = 0.5f;" in text                     # the source spelling, suffix included
    cc = _cc()
    if cc is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "g.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        r2 = subprocess.run([cc, "-std=c11", "-Wall", "-Werror", "-c", src,
                             "-o", os.path.join(tmp, "g.o")], capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr


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
