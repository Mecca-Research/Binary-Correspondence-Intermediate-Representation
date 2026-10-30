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
Both rails render the linkable artifact: the oracle's `emit_linkable` and the twin's `bcir-cc
--linkable` (`bcir_cfront_compile_linkable`, CF-LINKTWIN), held to each other over the whole corpus
(`test_the_twins_linkable_artifact_is_the_oracles_over_the_corpus`) and linked to each other here."""

import glob
import os
import re
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit
from bcir.frontends.cfront.linkflags import format_link_flags
from bcir.tests.test_c_cfront import _C, _build_bcir_cc, _includes_for, _session_build_dir
from bcir.tests.test_c_executor import _cc
from bcir.toolchain import host_link_args


def _twin_artifact(src: str, name: str = "unit.c", target: "str | None" = None) -> tuple:
    """The twin's linkable artifact of `src` (`bcir-cc --linkable`, CF-LINKTWIN), for `target`'s data model (None: the
    host's): its exit status, the artifact and the diagnostics."""
    exe = _build_bcir_cc(_session_build_dir())
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        run = subprocess.run(
            [exe, "--linkable", *(("--target", target) if target else ()), "-I", _C, path],
            capture_output=True,
            text=True,
            timeout=300,
        )
    return run.returncode, run.stdout, run.stderr


def _artifacts(src: str) -> tuple:
    """Each rail's linkable artifact of `src` that both render: (rail, text)."""
    from bcir.frontends.cfront.emit import emit_linkable

    r = compile_unit(src, check_clang=False)
    rc, twin, err = _twin_artifact(src)
    assert rc == 0, f"the twin's linkable emit\n{err}"
    return (("oracle", emit_linkable(r.lowered, r.emitted)), ("twin", twin))


_MAIN = "double scale(double x);\nint main(void) { double v = scale(16.0); return (int)v; }\n"
_LIB = "#include <math.h>\ndouble scale(double x) { return sqrt(x) + 1.0; }\n"


def test_a_prototyped_callee_is_a_typed_cross_tu_edge():
    r = compile_unit(_MAIN, check_clang=False)
    assert r.is_clean and r.r18_ok  # NOT an R18 undefined-function error
    ops = [c.op for c in r.lowered.functions["main"].claims]
    assert "c.call.tu:scale" in ops, ops
    assert "extern double scale(double);" in r.emitted["main"]  # the standalone declaration
    assert "scale(t" in r.emitted["main"]  # verbatim call: external linkage,
    assert "bcir_scale" not in r.emitted["main"]  # no bcir_ rename


def test_a_forward_declaration_stays_a_real_r18_edge():
    src = (
        "unsigned g(unsigned x);\n"
        "unsigned f(unsigned y) { return g(y) + 1u; }\n"
        "unsigned g(unsigned x) { return x * 2u; }\n"
    )
    r = compile_unit(src, check_clang=False)
    assert r.is_clean and r.r18_ok
    ops = [c.op for c in r.lowered.functions["f"].claims]
    assert "c.call:g" in ops and not any(o.startswith("c.call.tu:") for o in ops)


def test_no_link_flag_derives_from_a_tu_edge():
    caller = compile_unit(_MAIN, check_clang=False)
    assert format_link_flags(caller.link_flags) in ("", "-")  # a sibling TU, not a library
    callee = compile_unit(_LIB, check_clang=False)
    assert "-lm" in format_link_flags(callee.link_flags)  # the callee's sqrt DOES derive -lm


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
        r = subprocess.run(
            host_link_args([cc, "-std=c11", main_ref, lib, "-o", ref, "-lm"]),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr
        rc_ref = subprocess.run([ref]).returncode

        emitted = os.path.join(tmp, "main_emit.c")  # the bcir artifact: emitted caller +
        with open(emitted, "w", encoding="utf-8") as f:  # a thin entry adapter in the same TU
            f.write(
                "#include <stdint.h>\n"
                + caller.emitted["main"]
                + "\nint main(void) { return (int)bcir_main(); }\n"
            )
        prog = os.path.join(tmp, "prog")
        r = subprocess.run(
            [cc, "-std=c11", emitted, lib, "-o", prog, *flags], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr  # the emitted TU compiles + LINKS standalone
        rc_bcir = subprocess.run([prog]).returncode
    assert rc_bcir == rc_ref == 5, (rc_bcir, rc_ref)  # sqrt(16)+1 == 5: behaviour equivalence


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
        r = subprocess.run(
            host_link_args([cc, "-std=c11", "-Wall", "-Werror", m, li, "-o", prog, "-lm"]),
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr  # self-contained, warning-free
        assert subprocess.run([prog]).returncode == 5


def test_linkable_globals_define_declare_and_reject_nonconstant():
    from bcir.frontends.cfront.emit import emit_linkable

    user = compile_unit(
        "extern unsigned base;\nunsigned scaled(unsigned x) { return base * x + 1u; }\n",
        check_clang=False,
    )
    definer = compile_unit(
        "unsigned base = 7u;\nunsigned tab[3] = {1u, 2u, 3u};\n"
        "unsigned peek(void) { return tab[0] + base; }\n",
        check_clang=False,
    )
    u_text = emit_linkable(user.lowered, user.emitted)
    d_text = emit_linkable(definer.lowered, definer.emitted)
    assert "extern unsigned int base;" in u_text  # a declaration, never a definition
    assert "unsigned int base = 7;" in d_text  # the defining TU renders the value
    assert "unsigned int tab[3] = {1, 2, 3};" in d_text
    assert "unsigned int scaled(unsigned int x)" in u_text  # non-static, real name
    assert "static" not in u_text.replace("_Static_assert", "")
    cc = _cc()
    if cc is not None:  # the pair genuinely links + runs
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, "a.c")
            b = os.path.join(tmp, "b.c")
            main = os.path.join(tmp, "main.c")
            for path, text in ((a, u_text), (b, d_text)):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            with open(main, "w", encoding="utf-8") as f:
                f.write(
                    "extern unsigned scaled(unsigned);\nint main(void){ return (int)scaled(6u); }\n"
                )
            prog = os.path.join(tmp, "prog")
            from bcir.tests.test_c_executor import _RUNTIME_C

            r = subprocess.run(
                [
                    cc,
                    "-std=c11",
                    "-I",
                    _RUNTIME_C,
                    main,
                    a,
                    b,
                    os.path.join(_RUNTIME_C, "bcir_quarantine.c"),
                    "-o",
                    prog,
                ],
                capture_output=True,
                text=True,
            )  # the documented contract: a masked
            assert r.returncode == 0, r.stderr  # access links bcir_quarantine.c
            assert subprocess.run([prog]).returncode == 43  # 7*6+1
    # Part VII A4: &x ADDRESS CONSTANTS and sizeof-bearing initializers now RENDER --
    # the SS5.9 evaluator consults the chosen ABI's layout oracle (sizeof folds at
    # LOWER time, where the ABI lives) and an address constant is the linker's
    # relocation, rendered by name. A bare `return g;` (zero claims) names the global
    # too (the raw-rid mis-render this test flushed out). Forward-referencing address
    # constants (illegal C: use before declaration) still refuse LOUDLY.
    fl = compile_unit(
        "unsigned base = 7u;\nunsigned *pp = &base;\n"
        "unsigned sz = sizeof(unsigned) * 4u;\n"
        "unsigned geto(void) { return base; }\n",
        check_clang=False,
    )
    a4 = emit_linkable(fl.lowered, fl.emitted)
    assert "unsigned int * pp = &base;" in a4  # the relocation, by name
    assert "unsigned int sz = 16;" in a4  # sizeof folded on the ABI
    assert "return base;" in a4  # the bare global return
    if cc is not None:  # ... and it genuinely runs
        with tempfile.TemporaryDirectory() as tmp:
            unit = os.path.join(tmp, "unit.c")
            main = os.path.join(tmp, "main.c")
            with open(unit, "w", encoding="utf-8") as f:
                f.write(a4)
            with open(main, "w", encoding="utf-8") as f:
                f.write(
                    "extern unsigned *pp;\nextern unsigned sz;\n"
                    "int main(void){ return (int)(*pp + sz); }\n"
                )
            prog = os.path.join(tmp, "prog")
            r = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Werror", main, unit, "-o", prog],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, r.stderr
            assert subprocess.run([prog]).returncode == 23  # *(&base) + 4*sizeof = 7+16
    fwd = compile_unit(
        "unsigned *qq = &later;\nunsigned later = 3u;\nunsigned geti(void) { return later; }\n",
        check_clang=False,
    )
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

    tu_a = compile_unit(
        "static unsigned mix(unsigned x) { return x * 3u; }\n"
        "static unsigned seed = 5u;\n"
        "unsigned fa(unsigned x) { return mix(x) + seed; }\n",
        check_clang=False,
    )
    tu_b = compile_unit(
        "static unsigned mix(unsigned x) { return x + 100u; }\n"
        "unsigned fb(unsigned x) { return mix(x); }\n",
        check_clang=False,
    )
    a_text = emit_linkable(tu_a.lowered, tu_a.emitted)
    b_text = emit_linkable(tu_b.lowered, tu_b.emitted)
    assert "static unsigned int mix(unsigned int x)" in a_text  # static KEPT, real name
    assert "static unsigned int seed = 5;" in a_text  # file-local global
    assert "\nunsigned int fa(unsigned int x)" in a_text  # the export is non-static
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
            f.write(
                "extern unsigned fa(unsigned);\nextern unsigned fb(unsigned);\n"
                "int main(void){ return (int)(fa(2u) + fb(1u)); }\n"
            )
        prog = os.path.join(tmp, "prog")
        r = subprocess.run(
            [cc, "-std=c11", "-Wall", "-Werror", main, a, b, "-o", prog],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr  # two same-named statics link CLEANLY
        assert subprocess.run([prog]).returncode == 112  # fa(2)=2*3+5=11, fb(1)=101


def test_linkable_static_forward_declaration_and_pointer_string_table():
    """Review-hardened edges: (1) the C static-forward-declaration idiom -- a static callee
    DEFINED AFTER its caller -- must yield a compilable artifact (the emit forward-declares
    every kept-static function); (2) a pointer-element string table (`char *tab[] = {..}`)
    must NOT take the char-array string path (it would size the array from the string bytes
    and mis-render): the array keeps its init-tuple count, and each literal renders as the
    address constant it is (C11 6.6p9), in an artifact that compiles (CF-LINKEMIT)."""
    from bcir.frontends.cfront.emit import emit_linkable

    fwd = compile_unit(
        "static int r(int v);\nint s(int x) { return r(x); }\n"
        "static int r(int x) { return x + 1; }\n",
        check_clang=False,
    )
    text = emit_linkable(fwd.lowered, fwd.emitted)
    assert "static int r(int);" in text  # the forward declaration
    cc = _cc()
    if cc is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fwd.c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            r = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Werror", "-c", path, "-o", os.path.join(tmp, "fwd.o")],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, r.stderr
    ptab = compile_unit(
        'char *tab[] = {"hi"};\nunsigned f4(void) { return 3u; }\n', check_clang=False
    )
    _, ct, vals, _, _ = [g for g in ptab.lowered.globals_decl if g[0] == "tab"][0]
    assert ct.count == 1 and vals == ('{"hi"}',)  # tuple-sized, not string-sized
    text = emit_linkable(ptab.lowered, ptab.emitted)
    assert 'tab[1] = {"hi"};' in text, text
    if cc is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ptab.c")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            r = subprocess.run(
                [cc, "-std=c11", "-Wall", "-Werror", "-c", path, "-o", os.path.join(tmp, "ptab.o")],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 0, r.stderr


def test_linkable_renders_float_negative_and_string_initializers():
    """The renderer past the first slice: float spellings (suffix kept), signed integer
    constants, and string-initialized char arrays (sized from the LITERAL, not the init
    tuple) all render -- and the artifact is host-compilable stand-alone."""
    from bcir.frontends.cfront.emit import emit_linkable

    r = compile_unit(
        "double dv = 1.5;\ndouble nv = -2.5;\nint off = -7;\n"
        'char msg[] = "hi";\nfloat fs = 0.5f;\n'
        "unsigned fz(unsigned x) { return x + 1u; }\n",
        check_clang=False,
    )
    text = emit_linkable(r.lowered, r.emitted)
    assert "double dv = 1.5;" in text
    assert "double nv = -2.5;" in text
    assert "int off = -7;" in text
    assert 'char msg[3] = "hi";' in text  # 2 code units + the NUL
    assert "float fs = 0.5f;" in text  # the source spelling, suffix included
    cc = _cc()
    if cc is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "g.c")
        with open(src, "w", encoding="utf-8") as f:
            f.write(text)
        r2 = subprocess.run(
            [cc, "-std=c11", "-Wall", "-Werror", "-c", src, "-o", os.path.join(tmp, "g.o")],
            capture_output=True,
            text=True,
        )
        assert r2.returncode == 0, r2.stderr


# CF-LINKEMIT (card 4): units whose linkable emit had not built alone -- a struct, a union, a nested struct or a typedef'd
# anonymous struct named; a `const` global and a `const` table of structs; tables of string pointers; address constants;
# a function designator in an initializer; the emit's own memcpy; `<complex.h>`, `<stdarg.h>` and `<stdatomic.h>` --
# each with the type definitions a unit sharing it would need, that unit's declarations of what it defines, and the
# values its driver prints of `s` (name, source, types, declarations, values).
_LINKEMIT_UNITS = (
    (
        "struct_member",
        "struct pt { uint32_t x, y; };\nstruct pt lk_gp = {1u, 2u};\n"
        "uint32_t lk_member(uint32_t s) { lk_gp.x = s; return lk_gp.x + lk_gp.y; }\n",
        "struct pt { uint32_t x, y; };\n",
        "extern struct pt lk_gp;\nuint32_t lk_member(uint32_t s);\n",
        ("lk_member(s)", "lk_gp.y"),
    ),
    (
        "const_global",
        "const uint32_t lk_k = 5u;\nstatic const uint32_t lk_t[3] = {1u, 2u, 3u};\n"
        "uint32_t lk_const(uint32_t s) { const uint32_t *p = &lk_k; return *p + lk_t[s % 3u]; }\n",
        "",
        "extern const uint32_t lk_k;\nuint32_t lk_const(uint32_t s);\n",
        ("lk_const(s)", "lk_k"),
    ),
    (
        "string_table",
        'const char *lk_tab[] = {"a", "bc"};\n'
        "uint32_t lk_tabf(uint32_t s) { return (uint32_t)lk_tab[s & 1u][0]; }\n",
        "",
        "extern const char *lk_tab[2];\nuint32_t lk_tabf(uint32_t s);\n",
        ("lk_tabf(s)", "(uint32_t)lk_tab[1][1]"),
    ),
    (
        "const_string_table",
        'const char *const lk_names[2] = {"x", "yz"};\n'
        "uint32_t lk_namef(uint32_t s) { return (uint32_t)lk_names[s & 1u][0]; }\n",
        "",
        "extern const char *const lk_names[2];\nuint32_t lk_namef(uint32_t s);\n",
        ("lk_namef(s)", "(uint32_t)lk_names[1][1]"),
    ),
    (
        "struct_array",
        "struct pt { uint32_t x, y; };\nstruct pt lk_ga[2] = {{1u, 2u}, {3u, 4u}};\n"
        "uint32_t lk_arr(uint32_t s) { return lk_ga[s & 1u].y; }\n",
        "struct pt { uint32_t x, y; };\n",
        "extern struct pt lk_ga[2];\nuint32_t lk_arr(uint32_t s);\n",
        ("lk_arr(s)", "lk_ga[0].x"),
    ),
    (
        "const_struct_table",
        "struct pt { uint32_t x, y; };\nconst struct pt lk_cpt[2] = {{1u, 2u}, {3u, 4u}};\n"
        "uint32_t lk_cs(uint32_t s) { const struct pt *p = &lk_cpt[s & 1u]; return p->x + lk_cpt[1].y; }\n",
        "struct pt { uint32_t x, y; };\n",
        "extern const struct pt lk_cpt[2];\nuint32_t lk_cs(uint32_t s);\n",
        ("lk_cs(s)",),
    ),
    (
        "union_global",
        "union un { uint32_t w; uint8_t b[4]; };\nunion un lk_gu = {7u};\n"
        "uint32_t lk_union(uint32_t s) { lk_gu.w = s; return lk_gu.b[0]; }\n",
        "union un { uint32_t w; uint8_t b[4]; };\n",
        "extern union un lk_gu;\nuint32_t lk_union(uint32_t s);\n",
        ("lk_union(s)", "lk_gu.w"),
    ),
    (
        "nested_struct",
        "struct in { uint16_t a, b; };\nstruct ot { struct in i; uint32_t c; };\nstruct ot lk_go = {{1u, 2u}, 3u};\n"
        "uint32_t lk_nest(uint32_t s) { return lk_go.i.b + lk_go.c + s; }\n",
        "struct in { uint16_t a, b; };\nstruct ot { struct in i; uint32_t c; };\n",
        "extern struct ot lk_go;\nuint32_t lk_nest(uint32_t s);\n",
        ("lk_nest(s)", "lk_go.i.a"),
    ),
    (
        "typedef_anonymous",
        "typedef struct { uint32_t a; uint32_t b; } pair_t;\npair_t lk_pair = {1u, 2u};\n"
        "uint32_t lk_pairf(uint32_t s) { return lk_pair.a + lk_pair.b * s; }\n",
        "typedef struct { uint32_t a; uint32_t b; } pair_t;\n",
        "extern pair_t lk_pair;\nuint32_t lk_pairf(uint32_t s);\n",
        ("lk_pairf(s)", "lk_pair.b"),
    ),
    (
        "address_constants",
        "uint32_t lk_buf[4] = {1u, 2u, 3u, 4u};\nuint32_t *lk_ptr = &lk_buf[2];\nuint32_t *const lk_cp = lk_buf;\n"
        "uint32_t lk_addr(uint32_t s) { return *lk_ptr + lk_cp[s & 3u]; }\n",
        "",
        "extern uint32_t lk_buf[4];\nextern uint32_t *lk_ptr;\nextern uint32_t *const lk_cp;\n"
        "uint32_t lk_addr(uint32_t s);\n",
        ("lk_addr(s)", "*lk_ptr + lk_cp[1]"),
    ),
    (
        "function_designator",
        "static uint32_t lk_inc(uint32_t v) { return v + 1u; }\nuint32_t (*lk_fp)(uint32_t) = lk_inc;\n"
        "uint32_t lk_call(uint32_t s) { return lk_fp(s) * 3u; }\n",
        "",
        "extern uint32_t (*lk_fp)(uint32_t);\nuint32_t lk_call(uint32_t s);\n",
        ("lk_call(s)", "lk_fp(s)"),
    ),
    (
        "struct_copy",
        "struct pt { uint32_t x, y; };\nstruct pt lk_q = {3u, 4u};\n"
        "uint32_t lk_copy(uint32_t s) { struct pt v = lk_q; v.x += s; return v.x + v.y; }\n",
        "struct pt { uint32_t x, y; };\n",
        "extern struct pt lk_q;\nuint32_t lk_copy(uint32_t s);\n",
        ("lk_copy(s)", "lk_q.x"),
    ),
    (
        "complex_parts",
        "#include <complex.h>\ndouble _Complex lk_z = 2.0;\n"
        "uint32_t lk_cplx(uint32_t s) { double _Complex z = lk_z * (double)(s & 7u);"
        " return (uint32_t)creal(z) + (uint32_t)cimag(z); }\n",
        "",
        "extern double _Complex lk_z;\nuint32_t lk_cplx(uint32_t s);\n",
        ("lk_cplx(s)",),
    ),
    (
        "variadic",
        "#include <stdarg.h>\n"
        "uint32_t lk_sum(uint32_t n, ...) { va_list ap; va_start(ap, n); uint32_t r = 0u;"
        " for (uint32_t i = 0u; i < n; i++) r += va_arg(ap, uint32_t); va_end(ap); return r; }\n"
        "uint32_t lk_var(uint32_t s) { return lk_sum(3u, s, 2u, 5u); }\n",
        "",
        "uint32_t lk_sum(uint32_t n, ...);\nuint32_t lk_var(uint32_t s);\n",
        ("lk_var(s)",),
    ),
    (
        "atomic_global",
        "#include <stdatomic.h>\n_Atomic uint32_t lk_at = 1u;\n"
        "uint32_t lk_atom(uint32_t s) { return atomic_fetch_add(&lk_at, s & 7u) + lk_at; }\n",
        "",
        "extern _Atomic uint32_t lk_at;\nuint32_t lk_atom(uint32_t s);\n",
        ("lk_atom(s)", "(uint32_t)lk_at"),
    ),
)
# A loop's `continue` label the emit places whether or not the body continues: unused, a `-Wall` warning both emits
# share (recorded in DEVELOPMENT_HISTORY's found-not-fixed list), named here for the one unit with a loop
_LINKEMIT_QUIET = {"variadic": ("-Wno-unused-label",)}
_LINKEMIT_GAPS = "0u, 1u, 2u, 3u, 7u, 255u, 256u, 65535u, 65536u, 0x12345678u, 0xFFFFFFFFu"


def test_linkable_emit_defines_what_the_source_defines():
    """CF-LINKEMIT, CF-LINKTWIN: each unit's linkable emit, on each rail (the oracle's and the twin's `--linkable`),
    (1) builds alone under `-std=c11 -Wall -Werror` -- its struct, union and typedef definitions as the source spells
    them, a `const` global `const`, a table of string pointers and an address constant (C11 6.6p9) rendered,
    `<string.h>` for its own memcpy and the headers its calls need; (2) still builds with another unit's declarations
    of what it defines after it, so each definition has the source's type (C11 6.2.7p2: a global defined without its
    `const` is another type); and (3) linked with a driver, prints what the original prints, under Clang and GCC. It
    had named every struct undefined, spelled a typedef'd anonymous struct `struct $anon0`, dropped `const`, refused
    `const char *tab[] = {"a"}`, and called memcpy, `va_start` and `atomic_fetch_add` undeclared."""
    from bcir.frontends.cfront.emit import emit_linkable

    runtime_c = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runtime", "c")
    compilers = [c for c in (shutil.which("clang"), shutil.which("gcc")) if c]
    if not compilers or not os.path.isfile(os.path.join(runtime_c, "bcir_quarantine.c")):
        return
    for name, body, types, decls, values in _LINKEMIT_UNITS:
        src = "#include <stdint.h>\n" + body
        r = compile_unit(src, check_clang=False)
        rc, twin, err = _twin_artifact(src, f"{name}.c")
        assert rc == 0, f"{name}: the twin's linkable emit\n{err}"
        for rail, linkable in (("oracle", emit_linkable(r.lowered, r.emitted)), ("twin", twin)):
            _linkemit_unit_runs(
                compilers, runtime_c, f"{name} ({rail})", name, src, linkable, types, decls, values
            )


def _linkemit_unit_runs(
    compilers, runtime_c, label_of, name, src, linkable, types, decls, values
) -> None:
    """`test_linkable_emit_defines_what_the_source_defines` of one rail's artifact of the unit `name`."""
    printed = " ".join(f'printf(" %u", (unsigned)({v}));' for v in values)
    driver = (
        f"#include <stdint.h>\n#include <stdio.h>\n{types}{decls}"
        f"static const uint32_t lk_in[] = {{{_LINKEMIT_GAPS}}};\n"
        "int main(void) {\n  for (unsigned n = 0; n < sizeof lk_in / sizeof lk_in[0]; n++) {\n"
        f"    uint32_t s = lk_in[n];\n    {printed}\n    putchar('\\n');\n  }}\n  return 0;\n}}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        for label, text in (
            ("linkable", linkable),
            ("declared", linkable + "\n" + decls),
            ("original", src),
            ("driver", driver),
        ):
            paths[label] = os.path.join(tmp, f"{label}.c")
            with open(paths[label], "w", encoding="utf-8") as f:
                f.write(text)
        for cc in compilers:
            for label in ("linkable", "declared"):
                b = subprocess.run(
                    [
                        cc,
                        "-std=c11",
                        "-Wall",
                        "-Werror",
                        *_LINKEMIT_QUIET.get(name, ()),
                        "-I",
                        runtime_c,
                    ]
                    + ["-c", paths[label], "-o", os.path.join(tmp, f"{label}.o")],
                    capture_output=True,
                    text=True,
                )
                assert b.returncode == 0, (
                    f"{label_of}: the {label} linkable emit does not build ({cc})\n{b.stderr}"
                )
            outs = []
            for label, parts in (
                ("original", [paths["original"]]),
                (
                    "linkable",
                    [
                        "-I",
                        runtime_c,
                        paths["linkable"],
                        os.path.join(runtime_c, "bcir_quarantine.c"),
                    ],
                ),
            ):
                prog = os.path.join(tmp, label)
                b = subprocess.run(
                    host_link_args(
                        [cc, "-std=c11", "-O2", *parts, paths["driver"], "-o", prog, "-lm"]
                    ),
                    capture_output=True,
                    text=True,
                )
                assert b.returncode == 0, (
                    f"{label_of}: the {label} program does not build ({cc})\n{b.stderr}"
                )
                outs.append(
                    subprocess.run([prog], capture_output=True, text=True, timeout=120).stdout
                )
            assert outs[0] and outs[1] == outs[0], (label_of, cc, outs[0][:200], outs[1][:200])


def _build(cc: str, tmp: str, files: dict, *flags: str, link: bool = True) -> "tuple":
    """Write `files` (name -> text) in `tmp` and build them with `cc`: (status, the program's path, diagnostics). A
    unit an artifact calls the quarantine from links `bcir_quarantine.c`."""
    paths = []
    for fname, text in files.items():
        path = os.path.join(tmp, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        paths.append(path)
    prog = os.path.join(tmp, "prog")
    cmd = [cc, "-std=c11", *flags, "-I", _C, *paths]
    cmd += (
        [os.path.join(_C, "bcir_quarantine.c"), "-o", prog, "-lm"] if link else ["-c", "-o", prog]
    )
    b = subprocess.run(host_link_args(cmd) if link else cmd, capture_output=True, text=True)
    return b.returncode, prog, b.stderr


def test_the_twins_linkable_artifacts_link_to_each_other_and_to_the_oracles():
    """CF-LINKTWIN: the twin's linkable artifacts (`bcir-cc --linkable`) of the caller and the callee link to each
    other, and each to the oracle's of the other, `-Wall -Werror` -- no original source in the image -- and every
    binary behaves as the all-original build does (sqrt(16) + 1 == 5)."""
    cc = _cc()
    if cc is None:
        return
    from bcir.frontends.cfront.emit import emit_linkable

    arts = {}
    for role, src in (("main", _MAIN), ("lib", _LIB)):
        r = compile_unit(src, check_clang=False)
        rc, twin, err = _twin_artifact(src, f"{role}.c")
        assert rc == 0, err
        arts[role] = {"oracle": emit_linkable(r.lowered, r.emitted), "twin": twin}
    for m, li in (("twin", "twin"), ("twin", "oracle"), ("oracle", "twin")):
        with tempfile.TemporaryDirectory() as tmp:
            files = {"m.c": arts["main"][m], "l.c": arts["lib"][li]}
            rc, prog, err = _build(cc, tmp, files, "-Wall", "-Werror")
            assert rc == 0, f"{m} caller, {li} callee: {err}"
            assert subprocess.run([prog]).returncode == 5, (m, li)


def test_the_twins_linkable_globals_define_declare_and_refuse():
    """CF-LINKTWIN: the twin defines a unit's globals as the oracle does -- an `extern` one declared, never defined; a
    definition with its initializer re-spelled, a table's braces kept; the address of a global declared before it by
    name; `sizeof` folded on the target -- and the pair links and runs (7*6+1, then *(&base) + 4*sizeof = 7+16). A
    global whose initializer names one declared after it (no constant C takes: the name is not yet declared) is refused
    by name, as the oracle refuses it, and so is one whose initializer reads an object."""
    cc = _cc()
    if cc is None:  # the twin is built, and the artifacts linked, with it
        return
    user_src = "extern unsigned base;\nunsigned scaled(unsigned x) { return base * x + 1u; }\n"
    def_src = (
        "unsigned base = 7u;\nunsigned tab[3] = {1u, 2u, 3u};\n"
        "unsigned peek(void) { return tab[0] + base; }\n"
    )
    rc, u_text, err = _twin_artifact(user_src)
    assert rc == 0, err
    rc, d_text, err = _twin_artifact(def_src)
    assert rc == 0, err
    assert re.search(r"^extern uint32_t base;$", u_text, re.M), u_text
    assert re.search(r"^uint32_t base = 7;$", d_text, re.M), d_text
    assert re.search(r"^uint32_t tab\[3\] = \{1, 2, 3\};$", d_text, re.M), d_text
    assert re.search(r"^uint32_t scaled\(uint32_t x\)$", u_text, re.M), (
        u_text
    )  # exported, its real name
    assert not re.search(r"^static ", u_text, re.M), u_text
    a4_src = (
        "unsigned base = 7u;\nunsigned *pp = &base;\n"
        "unsigned sz = sizeof(unsigned) * 4u;\n"
        "unsigned geto(void) { return base; }\n"
    )
    rc, a4, err = _twin_artifact(a4_src)
    assert rc == 0, err
    assert re.search(r"^uint32_t \* pp = &base;$", a4, re.M), a4  # the relocation, by name
    assert re.search(r"^uint32_t sz = 16;$", a4, re.M), a4  # sizeof folded on the ABI
    with tempfile.TemporaryDirectory() as tmp:
        main = "extern unsigned scaled(unsigned);\nint main(void){ return (int)scaled(6u); }\n"
        rc, prog, err = _build(
            cc, tmp, {"main.c": main, "a.c": u_text, "b.c": d_text}, "-Wall", "-Werror"
        )
        assert rc == 0, err
        assert subprocess.run([prog]).returncode == 43
    with tempfile.TemporaryDirectory() as tmp:
        main = (
            "extern unsigned *pp;\nextern unsigned sz;\nint main(void){ return (int)(*pp + sz); }\n"
        )
        rc, prog, err = _build(cc, tmp, {"main.c": main, "unit.c": a4}, "-Wall", "-Werror")
        assert rc == 0, err
        assert subprocess.run([prog]).returncode == 23
    # the forms an entry takes, each rendered as the oracle renders it: a string's pieces one space apart, an element's
    # address at two constant subscripts in their order, a constant's parentheses dropped -- around a floating one,
    # its sign and an address
    forms = (
        "#include <stdint.h>\n"
        "uint32_t lk_m[2][3] = {{1u, 2u, 3u}, {4u, 5u, 6u}};\nuint32_t *lk_mp = &lk_m[1][2];\n"
        'const char *lk_cat = "ab" "cd";\n'
        "double lk_pf = (1.5);\ndouble lk_nf = -(2.5);\nuint32_t *lk_pq = &(lk_m[0][1]);\n"
        "uint32_t lk_forms(void) { return (uint32_t)lk_cat[2] + *lk_mp + *lk_pq + (uint32_t)lk_pf"
        " + (uint32_t)-lk_nf; }\n"
    )
    spelled = {
        "lk_mp": "&lk_m[1][2]",
        "lk_cat": '"ab" "cd"',
        "lk_pf": "1.5",
        "lk_nf": "-2.5",
        "lk_pq": "&lk_m[0][1]",
    }
    for rail, text in _artifacts(forms):
        for gname, init in spelled.items():
            assert re.search(rf"^[^=\n]*\b{gname} = {re.escape(init)};$", text, re.M), (
                rail,
                gname,
                text,
            )
        with tempfile.TemporaryDirectory() as tmp:
            main = "#include <stdint.h>\nuint32_t lk_forms(void);\nint main(void){ return (int)lk_forms(); }\n"
            rc, prog, err = _build(cc, tmp, {"main.c": main, "unit.c": text}, "-Wall", "-Werror")
            assert rc == 0, (rail, err)
            assert subprocess.run([prog]).returncode == 99 + 6 + 2 + 1 + 2, (
                rail
            )  # 'c' + 6 + 2 + 1 + 2
    for src, gname in (
        (
            "unsigned *qq = &later;\nunsigned later = 3u;\nunsigned geti(void) { return later; }\n",
            "qq",
        ),
        ("unsigned k = 3u;\nunsigned j = k + 1u;\nunsigned getj(void) { return j; }\n", "j"),
    ):
        rc, out, err = _twin_artifact(src)
        assert rc == 1 and not out, (rc, out[:200])
        assert (
            f"linkable emit error: linkable emit: global '{gname}' has a non-renderable constant initializer"
            in err
        ), err
        from bcir.frontends.cfront.emit import emit_linkable

        r = compile_unit(src, check_clang=False)
        try:
            emit_linkable(r.lowered, r.emitted)
            raise AssertionError(f"the oracle rendered {gname}")
        except ValueError as e:
            assert f"global {gname!r} has a non-renderable constant initializer" in str(e), str(e)


def test_the_twins_linkable_honors_source_static_and_forward_declarations():
    """CF-LINKTWIN: the twin keeps a source-`static` function and global `static` under its real name -- two units
    each with a same-named static helper link into one binary (fa(2) + fb(1) = 11 + 101) -- declares every function
    before its globals, so a static callee defined after its caller compiles, and renders a pointer's string literal
    as the address constant it is, its table sized by its entries, not the string's bytes."""
    cc = _cc()
    if cc is None:  # the twin is built, and the artifacts linked, with it
        return
    a_src = (
        "static unsigned mix(unsigned x) { return x * 3u; }\n"
        "static unsigned seed = 5u;\n"
        "unsigned fa(unsigned x) { return mix(x) + seed; }\n"
    )
    b_src = "static unsigned mix(unsigned x) { return x + 100u; }\nunsigned fb(unsigned x) { return mix(x); }\n"
    rc, a_text, err = _twin_artifact(a_src)
    assert rc == 0, err
    rc, b_text, err = _twin_artifact(b_src)
    assert rc == 0, err
    assert re.search(r"^static uint32_t mix\(uint32_t x\)$", a_text, re.M), a_text
    assert re.search(r"^static uint32_t seed = 5;$", a_text, re.M), a_text
    assert re.search(r"^uint32_t fa\(uint32_t x\)$", a_text, re.M), a_text
    fwd_src = "static int r(int v);\nint s(int x) { return r(x); }\nstatic int r(int x) { return x + 1; }\n"
    rc, fwd, err = _twin_artifact(fwd_src)
    assert rc == 0, err
    assert re.search(r"^static int32_t r\(int32_t x\);$", fwd, re.M), (
        fwd
    )  # every function declared up front
    rc, ptab, err = _twin_artifact('char *tab[] = {"hi"};\nunsigned f4(void) { return 3u; }\n')
    assert rc == 0, err
    assert re.search(r'^char \* tab\[1\] = \{"hi"\};$', ptab, re.M), ptab
    rc, fl, err = _twin_artifact(
        "double dv = 1.5;\ndouble nv = -2.5;\nint off = -7;\n"
        'char msg[] = "hi";\nfloat fs = 0.5f;\nunsigned fz(unsigned x) { return x + 1u; }\n'
    )
    assert rc == 0, err
    for line in (
        "double dv = 1.5;",
        "double nv = -2.5;",
        "int32_t off = -7;",
        'char msg[3] = "hi";',
        "float fs = 0.5f;",
    ):
        assert re.search("^" + re.escape(line) + "$", fl, re.M), (line, fl)
    with tempfile.TemporaryDirectory() as tmp:
        main = (
            "extern unsigned fa(unsigned);\nextern unsigned fb(unsigned);\n"
            "int main(void){ return (int)(fa(2u) + fb(1u)); }\n"
        )
        rc, prog, err = _build(
            cc, tmp, {"main.c": main, "a.c": a_text, "b.c": b_text}, "-Wall", "-Werror"
        )
        assert rc == 0, err
        assert subprocess.run([prog]).returncode == 112
    for name, text in (("fwd", fwd), ("ptab", ptab), ("floats", fl)):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _prog, err = _build(cc, tmp, {f"{name}.c": text}, "-Wall", "-Werror", link=False)
            assert rc == 0, (name, err)


# CF-LINKTWIN: initializers whose value is the target's -- the sizes of `long`, a pointer, `long double`, `size_t` and a
# struct of a `long`, and `(unsigned long)-1` -- and what each target's data model makes of them, by global
_TARGET_INITS = (
    "#include <stdint.h>\n#include <stddef.h>\n"
    "unsigned long lk_l = sizeof(long);\n"
    "unsigned lk_ps = sizeof(void *);\n"
    "unsigned long long lk_ld = sizeof(long double);\n"
    "unsigned long lk_max = (unsigned long)-1;\n"
    "size_t lk_sz = sizeof(size_t) * 3u;\n"
    "struct lk_s { char c; long l; } lk_st = {1, 2};\n"
    "unsigned lk_sts = sizeof(struct lk_s);\n"
    "unsigned lk_get(unsigned s) { return s + lk_ps + (unsigned)lk_l + lk_sts + (unsigned)lk_ld + (unsigned)lk_sz; }\n"
)
_TARGET_VALUES = {
    "x86_64-linux": {
        "lk_l": "8",
        "lk_ps": "8",
        "lk_ld": "16",
        "lk_max": "18446744073709551615u",
        "lk_sz": "24",
        "lk_sts": "16",
    },
    "aarch64-linux": {
        "lk_l": "8",
        "lk_ps": "8",
        "lk_ld": "16",
        "lk_max": "18446744073709551615u",
        "lk_sz": "24",
        "lk_sts": "16",
    },
    "riscv64-linux": {
        "lk_l": "8",
        "lk_ps": "8",
        "lk_ld": "16",
        "lk_max": "18446744073709551615u",
        "lk_sz": "24",
        "lk_sts": "16",
    },
    "x86_64-windows": {
        "lk_l": "4",
        "lk_ps": "8",
        "lk_ld": "8",
        "lk_max": "4294967295",
        "lk_sz": "24",
        "lk_sts": "8",
    },
    "i386-linux": {
        "lk_l": "4",
        "lk_ps": "4",
        "lk_ld": "12",
        "lk_max": "4294967295",
        "lk_sz": "12",
        "lk_sts": "8",
    },
}


def test_the_twins_linkable_initializers_fold_on_each_target_as_the_oracles():
    """CF-LINKTWIN: on each of the five targets, each rail's linkable artifact defines the globals whose initializer
    is the target's -- `sizeof` of `long`, a pointer, `long double`, `size_t` and a struct of a `long`, and
    `(unsigned long)-1` -- with the value that target's data model gives it: the twin folds an initializer for the
    target it compiles for, as the oracle does."""
    if _cc() is None:  # the twin is built with it
        return
    from bcir.frontends.cfront.emit import emit_linkable

    for target, want in _TARGET_VALUES.items():
        rc, twin, err = _twin_artifact(_TARGET_INITS, target=target)
        assert rc == 0, (target, err)
        r = compile_unit(_TARGET_INITS, check_clang=False, target=target)
        for rail, text in (("oracle", emit_linkable(r.lowered, r.emitted)), ("twin", twin)):
            for gname, value in want.items():
                assert re.search(rf"^[^=\n]*\b{gname} = {re.escape(value)};$", text, re.M), (
                    target,
                    rail,
                    gname,
                    value,
                )
            assert re.search(r"^[^=\n]*\blk_st = \{1, 2\};$", text, re.M), (target, rail)


# CF-LINKTWIN: a static function that calls an exported one. The oracle's artifact kept `static` on every line of a
# static function's text, its forward declaration of the exported callee among them -- `static uint32_t g(uint32_t
# x);` after g's own declaration, which no compiler takes (C11 6.2.2p7); the twin's kept it only on lines declaring a
# static function. Both rails now keep it there alone.
_STATIC_CALLS_EXPORTED = (
    "#include <stdint.h>\n"
    "uint32_t g(uint32_t x) { return x + 1u; }\n"
    "static uint32_t h(uint32_t x) { return g(x) * 2u; }\n"
    "uint32_t k(uint32_t x) { return h(x) + 3u; }\n"
)


def test_a_static_function_calling_an_exported_one_links_on_both_rails():
    """CF-LINKTWIN: each rail's linkable artifact of a unit whose static function calls an exported one builds alone
    under `-Wall -Werror` and, linked with a driver, returns what the original does: k(4) = (4 + 1) * 2 + 3."""
    cc = _cc()
    if cc is None:
        return
    for rail, text in _artifacts(_STATIC_CALLS_EXPORTED):
        assert re.search(r"^static uint32_t h\(", text, re.M), (rail, text)
        assert not re.search(r"^static uint32_t (g|k)\(", text, re.M), (rail, text)
        with tempfile.TemporaryDirectory() as tmp:
            main = "#include <stdint.h>\nuint32_t k(uint32_t x);\nint main(void){ return (int)k(4u); }\n"
            rc, prog, err = _build(cc, tmp, {"main.c": main, "unit.c": text}, "-Wall", "-Werror")
            assert rc == 0, (rail, err)
            assert subprocess.run([prog]).returncode == 13, rail


# CF-LINKTWIN: names a C11 header defines as macros -- `bool` (<stdbool.h>), `alignas` and `alignof` (<stdalign.h>),
# keywords in C23 -- carried into the artifact by the source's types (and the oracle's own `bool` temps). The oracle's
# artifact named them without the header, and did not build as C11. (`true` and `false`, the header's too, neither
# front end reads as a constant outside `#if`.)
_C11_MACROS = (
    "#include <stdint.h>\n#include <stdbool.h>\n#include <stdalign.h>\n"
    "struct flags { bool on; alignas(8) uint32_t n; };\n"
    "struct flags lk_fl = {1, 3u};\n"
    "bool lk_any = 0;\n"
    "uint32_t lk_flags(uint32_t s) { lk_fl.n += s; return (uint32_t)(bool)(lk_fl.n & 2u) + lk_fl.on"
    " + (uint32_t)alignof(struct flags) + (uint32_t)lk_any; }\n"
)


# ... and a unit whose global's declaration alone names `bool`, on the oracle (its `_cname`; the twin spells the global
# `_Bool`): the oracle had read its headers off its functions and types, where none does (CF-LINKTWIN)
_C11_BOOL_GLOBAL = (
    "#include <stdint.h>\n#include <stdbool.h>\n"
    "bool lk_on = 1;\n"
    "uint32_t lk_flags(uint32_t s) { return s + (uint32_t)lk_on; }\n"
)
#: The names C11 spells as a header's macros, and the header each needs (`emit._C11_MACRO_HEADERS`'s, held here apart).
_C11_NAMES = (
    (re.compile(r"\bbool\b"), "#include <stdbool.h>\n"),
    (re.compile(r"\b(?:alignas|alignof)\b"), "#include <stdalign.h>\n"),
)


def test_the_linkable_emit_includes_the_header_of_bool_and_alignas_on_both_rails():
    """CF-LINKTWIN: each rail's linkable artifact of a C11 unit spelling `bool`, `alignas` and `alignof` -- in a
    struct it defines, a global and a function -- includes <stdbool.h> and <stdalign.h>; the artifact of a unit whose
    global's declaration alone names `bool` includes the header of every such name its text spells, wherever it
    spells it; and each builds alone under `-std=c11 -Wall -Werror` and returns what the original does."""
    cc = _cc()
    if cc is None:
        return
    from bcir.frontends.cfront.emit import _code_only

    for unit in (_C11_MACROS, _C11_BOOL_GLOBAL):
        for rail, text in _artifacts(unit):
            code = _code_only(text)
            for name_re, include in _C11_NAMES:
                if name_re.search(code):
                    assert include in text, (rail, include, text[:600])
            if unit is _C11_MACROS:  # its struct spells all three, on each rail
                assert "#include <stdbool.h>\n" in text and "#include <stdalign.h>\n" in text, (
                    rail,
                    text[:400],
                )
            with tempfile.TemporaryDirectory() as tmp:
                rc, _prog, err = _build(cc, tmp, {"unit.c": text}, "-Wall", "-Werror", link=False)
                assert rc == 0, (rail, err)
            with tempfile.TemporaryDirectory() as tmp:
                outs = []
                for label, files in (
                    ("original", {"o.c": unit}),
                    ("linkable", {"l.c": text}),
                ):
                    d = os.path.join(tmp, label)
                    os.mkdir(d)
                    driver = (
                        "#include <stdint.h>\n#include <stdio.h>\nuint32_t lk_flags(uint32_t s);\n"
                        'int main(void){ for (uint32_t s = 0; s < 6; s++) printf("%u ", (unsigned)lk_flags(s));'
                        " return 0; }\n"
                    )
                    rc, prog, err = _build(cc, d, {**files, "drv.c": driver})
                    assert rc == 0, (rail, label, err)
                    outs.append(subprocess.run([prog], capture_output=True, text=True).stdout)
                assert outs[0] and outs[1] == outs[0], (rail, outs)


# CF-LINKTWIN, the corpus: every fixture through both rails' linkable emit. The headers a fixture's original is built
# with (its tests prepend them); the errors an artifact may not have -- C's own and the ones C23 and GCC 14 made of
# implicit declarations and incompatible pointers -- its warnings being the source's own.
_ORIGINAL_HEADERS = (
    "stdint.h",
    "stdio.h",
    "string.h",
    "stddef.h",
    "stdlib.h",
    "math.h",
    "stdatomic.h",
    "stdarg.h",
    "stdbool.h",
)
_LINKABLE_ERRORS = (
    "-w",
    "-Werror=implicit-function-declaration",
    "-Werror=implicit-int",
    "-Werror=int-conversion",
    "-Werror=incompatible-pointer-types",
    "-Werror=return-type",
)
#: The corpus's functions whose type the linkable artifact does not keep, each with why -- an open case both rails
#: share, held as an exact set: a fix, or one more, changes it. A function whose type keeps a qualifier below a
#: parameter's or the return's top level is defined through a wrapper of its emitted function (CF-PARMQUALS); a
#: wrapper cannot pass a variadic function's `...` on, so such a function is defined with the emit's parameter
#: types, `char *fmt` for `const char *fmt`. Its artifact builds; another unit declaring it as its source does
#: declares another type (C11 6.2.7p2).
_LINKABLE_VARIADIC_OPEN = {
    ("cfront_extvariadic.c", "ev_fwd"): "a printf-style forwarder's `const char *fmt, ...`",
}
#: A wrapped function's definition in an artifact: its head, then `{`, the one call of its emitted function, `}`
_WRAPPER = re.compile(
    r"^((?:static )?[^\n;{}]*?\b(\w+)\([^\n;{}]*\))\n\{\n    (?:return )?(?:\([^\n]*?\))?bcir_\2\([^\n]*\);\n\}",
    re.M,
)


def _split_params(s: str) -> list:
    """A parameter list's parameters, split at the commas outside parentheses."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur else out


#: C's scalar type names as the twin spells them on the host's data model (LP64): the oracle keeps C's, `unsigned
#: int`, the twin the exact-width name, `uint32_t` -- the one type
_SCALAR_NAMES = (
    (r"\bunsigned\s+long\s+long(\s+int)?\b", "uint64_t"),
    (r"\b(signed\s+)?long\s+long(\s+int)?\b", "int64_t"),
    (r"\bunsigned\s+long(\s+int)?\b", "uint64_t"),
    (r"\b(signed\s+)?long(\s+int)?\b", "int64_t"),
    (r"\bunsigned\s+short(\s+int)?\b", "uint16_t"),
    (r"\b(signed\s+)?short(\s+int)?\b", "int16_t"),
    (r"\bunsigned\s+char\b", "uint8_t"),
    (r"\bsigned\s+char\b", "int8_t"),
    (r"\bunsigned(\s+int)?\b", "uint32_t"),
    (r"\b(signed\s+)?int\b", "int32_t"),
)


def _wrappers(artifact: str, lowered=None) -> dict:
    """The artifact's wrapped functions (CF-PARMQUALS): name -> its definition, spaces aside -- the rails spell a
    pointer's `*`s apart and together -- with C's scalar names as the twin spells them (`_SCALAR_NAMES`), and, given
    the unit's `lowered` functions, each function-pointer parameter
    as its position alone: the oracle spells its type inline, `RET (*f)(PARAMS)`, the twin by the typedef it names it
    by, the one type either way (the call, compared whole, names it)."""
    out = {}
    for m in _WRAPPER.finditer(artifact):
        text = m.group(0)
        for pat, exact in _SCALAR_NAMES:
            text = re.sub(pat, exact, text)
        name, text = m.group(2), re.sub(r"\s+", "", text)
        lf = lowered.functions.get(name) if lowered is not None else None
        fps = [k for k, (_n, _r, ct) in enumerate(lf.params) if ct.kind == "funcptr"] if lf else []
        if fps:
            head, rest = text.split("{", 1)
            pre, params = head.split(f"{name}(", 1)
            parts = _split_params(params[:-1])
            for k in fps:  # its name is the call's argument, which is compared whole
                parts[k] = f"<fp{k}>"
            text = f"{pre}{name}({','.join(parts)}){{{rest}"
        out[name] = text
    return out


def _original_std(clang: str, path: str) -> "str | None":
    """The C standard the fixture's original builds under, with its tests' headers: C11, else C2x; None when
    neither."""
    heads = [x for h in _ORIGINAL_HEADERS for x in ("-include", h)]
    for std in ("c11", "c2x"):
        b = subprocess.run(
            [clang, f"-std={std}", *_LINKABLE_ERRORS, *heads, "-I", _C, "-fsyntax-only", path],
            capture_output=True,
            text=True,
        )
        if b.returncode == 0:
            return std
    return None


def _builds_alone(clang: str, std: str, text: str, tmp: str) -> str:
    """'' when `text` builds alone under `std` with no error `_LINKABLE_ERRORS` makes one, else its first error."""
    path = os.path.join(tmp, "artifact.c")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    b = subprocess.run(
        [clang, f"-std={std}", *_LINKABLE_ERRORS, "-I", _C, "-fsyntax-only", path],
        capture_output=True,
        text=True,
    )
    return "" if b.returncode == 0 else (b.stderr.strip().splitlines() or ["?"])[0]


# CF-PARMQUALS: a unit whose functions take and return types qualified below their top level -- the API a driver's
# header declares (`const uint8_t *buf`) -- its header, and a caller that knows the functions only through it
_PQ_TYPES = "#include <stdint.h>\ntypedef uint32_t (*pq_cop)(const uint32_t *);\n"
_PQ_LIB = (
    _PQ_TYPES
    + r"""
static uint32_t pq_first(const uint32_t *p) { return p[0]; }
uint32_t pq_sum(const uint32_t *p, uint32_t n) {
  uint32_t s = 0u;
  for (uint32_t i = 0u; i < n; i++) s += p[i];
  return s;
}
uint32_t pq_names(const char *const *v, uint32_t n) { return (uint32_t)v[n - 1u][0]; }
const uint32_t *pq_at(const uint32_t *p, uint32_t i) { return p + i; }
const char *const *pq_next(const char *const *v) { return v + 1; }
void pq_copy(uint32_t *restrict d, const uint32_t *restrict s, uint32_t n) {
  for (uint32_t i = 0u; i < n; i++) d[i] = s[i];
}
uint32_t pq_apply(pq_cop f, const uint32_t *p) { return f(p) * 2u + pq_first(p); }
uint32_t pq_va(const uint32_t *p, uint32_t n, ...) { return p[0] + n; }
static uint32_t pq_last(const uint32_t *p, uint32_t n) { return p[n - 1u]; }
static uint32_t pq_tabf(const uint32_t *p) { return p[2] * 3u; }
static pq_cop pq_tab[1] = {pq_tabf};
uint32_t pq_entry(uint32_t s) {
  uint32_t a[3] = {s, s + 1u, s + 2u};
  uint32_t b[3];
  pq_cop g = pq_first;
  pq_copy(b, a, 3u);
  return pq_sum(b, 3u) + *pq_at(a, 2u) + pq_apply(g, a) + pq_apply(pq_first, b) + pq_va(a, 1u, s)
         + pq_last(a, 3u) + pq_tab[0](b);
}
"""
)
_PQ_HEADER = (
    _PQ_TYPES
    + """uint32_t pq_sum(const uint32_t *p, uint32_t n);
uint32_t pq_names(const char *const *v, uint32_t n);
const uint32_t *pq_at(const uint32_t *p, uint32_t i);
const char *const *pq_next(const char *const *v);
void pq_copy(uint32_t *restrict d, const uint32_t *restrict s, uint32_t n);
uint32_t pq_apply(pq_cop f, const uint32_t *p);
uint32_t pq_entry(uint32_t s);
"""
)
_PQ_MAIN = (
    _PQ_HEADER
    + r"""#include <stdio.h>
static const char *const pq_w[3] = {"ab", "cd", "ef"};
static uint32_t pq_second(const uint32_t *p) { return p[1]; }
int main(void) {
  uint32_t a[4] = {3u, 5u, 7u, 9u}, b[4];
  pq_copy(b, a, 4u);
  uint32_t r = pq_sum(b, 4u) + *pq_at(a, 3u) * 3u + pq_names(pq_w, 2u) + (uint32_t)pq_next(pq_w)[0][1];
  r += pq_apply(pq_second, a) * 5u + pq_entry(10u) * 7u;
  printf("%u\n", r);
  return 0;
}
"""
)
#: the functions `_PQ_LIB` defines through a wrapper: each whose type keeps a qualifier below a parameter's or the
#: return's top level -- `pq_apply`'s `const uint32_t *p` as well as a function pointer's own parameter -- but the
#: exported variadic `pq_va`, whose `...` a wrapper cannot pass on (`_LINKABLE_VARIADIC_OPEN`; `_PQ_HEADER` leaves it
#: out, since its definition keeps the emit's types), and the static `pq_last`, which
#: the unit only calls: no other unit declares it, and a wrapper of it would be a function nothing names. The static
#: `pq_first` is a pointer's value in a function, `pq_tabf` in a file-scope table
_PQ_WRAPPED = {
    "pq_first",
    "pq_sum",
    "pq_names",
    "pq_at",
    "pq_next",
    "pq_copy",
    "pq_apply",
    "pq_tabf",
}
_PQ_ERRORS = (
    "-Werror=incompatible-pointer-types",
    "-Werror=int-conversion",
    "-Werror=implicit-function-declaration",
    "-Werror=unused-function",
)


def test_the_linkable_artifact_keeps_each_functions_declared_type():
    """CF-PARMQUALS: each rail's linkable artifact of `_PQ_LIB` declares and defines every function with the type its
    source does -- `const uint32_t *p`, `const char *const *v`, `const uint32_t *restrict s`, a qualified return --
    through a wrapper of its emitted function, whose own parameters keep none (`_cname`). The emitted functions keep
    their `static bcir_` names, so the unit's own calls call them; a pointer to a function, `pq_cop g = pq_first;`,
    is the wrapper's, of the source's type, as is `pq_tabf` in a file-scope table. The exported variadic `pq_va` and the
    static `pq_last`, which the unit only calls, are defined as before, under their own names: no wrapper, and none
    unused. The two rails write the same wrappers, spaces aside. Then the artifact
    compiles together with the unit's header in one translation unit (a declaration of another type would be an
    error), links with a caller that knows the functions only through that header, and prints what the original's
    build prints. Without its qualifiers the header would conflict: the check reads the declarations."""
    cc = _cc()
    if cc is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        rc, prog, err = _build(cc, tmp, {"m.c": _PQ_MAIN, "l.c": _PQ_LIB}, *_PQ_ERRORS)
        assert rc == 0, err
        want = subprocess.run([prog], capture_output=True, text=True, timeout=60).stdout
    assert want.strip().isdigit(), want
    arts = dict(_artifacts(_PQ_LIB))
    lowered = compile_unit(_PQ_LIB, check_clang=False).lowered
    ow, tw = _wrappers(arts["oracle"], lowered), _wrappers(arts["twin"], lowered)
    assert set(ow) == _PQ_WRAPPED, sorted(ow)
    assert tw == ow, (tw, ow)
    for rail, art in arts.items():
        assert re.search(r"^static uint32_t bcir_pq_sum\(", art, re.M), (rail, art[:600])
        assert re.search(r"^uint32_t pq_va\(uint32_t ?\* ?p, uint32_t n, \.\.\.\)$", art, re.M), (
            rail
        )
        assert re.search(r"^static uint32_t pq_last\(uint32_t ?\* ?p, uint32_t n\)$", art, re.M), (
            rail
        )
        assert re.search(r"^\s*pq_cop g = pq_first;|^\s*g = pq_first;", art, re.M), rail
        with tempfile.TemporaryDirectory() as tmp:
            rc, _o, err = _build(
                cc, tmp, {"one.c": art + "\n" + _PQ_HEADER}, *_PQ_ERRORS, link=False
            )
            assert rc == 0, (rail, err)
            # the control: the header with one parameter's qualifier dropped is another type, and an error
            bad = _PQ_HEADER.replace(
                "uint32_t pq_sum(const uint32_t *p", "uint32_t pq_sum(uint32_t *p"
            )
            assert bad != _PQ_HEADER
            rc, _o, err = _build(cc, tmp, {"one.c": art + "\n" + bad}, *_PQ_ERRORS, link=False)
            assert rc != 0 and "pq_sum" in err, (rail, err[-400:])
            rc, prog, err = _build(cc, tmp, {"m.c": _PQ_MAIN, "l.c": art}, *_PQ_ERRORS)
            assert rc == 0, (rail, err)
            got = subprocess.run([prog], capture_output=True, text=True, timeout=60).stdout
            assert got == want, (rail, got, want)


def test_the_twins_linkable_artifact_is_the_oracles_over_the_corpus():
    """CF-LINKTWIN: every corpus fixture (`runtime/c/cfront_*.c`) through both rails' linkable emit -- the oracle's
    `emit_linkable` and the twin's `bcir-cc --linkable`. The two refuse the same fixtures, for the same global; where
    they render, the twin's artifact (1) defines the unit's types as the oracle's does, the same lines in the same
    order, right after its attestation; (2) holds each global the oracle's does, line for line -- its storage class,
    its thread storage, its name and dimensions, and its initializer re-spelled byte for byte -- as many times as the
    oracle's; (3) wraps the functions the oracle's wraps, each function whose type keeps a qualifier below a
    parameter's or the return's top level but a variadic one (`_LINKABLE_VARIADIC_OPEN`, exactly) and a static one
    the unit only calls, with the oracle's wrapper, spaces aside (CF-PARMQUALS); and (4) builds alone wherever the original does, under the original's C
    standard, as the oracle's does -- every one. The twin had emitted the unit's functions alone."""
    from bcir.frontends.cfront.emit import emit_linkable

    clang = shutil.which("clang")
    if clang is None:
        return
    exe = _build_bcir_cc(_session_build_dir())
    seen = {
        "units": 0,
        "with_types": 0,
        "with_globals": 0,
        "with_inits": 0,
        "refused": 0,
        "built": 0,
        "wrapped": 0,
        "called_only": 0,
    }
    fails = {"oracle": set(), "twin": set()}
    variadic_open: set = set()
    for path in sorted(glob.glob(os.path.join(_C, "cfront_*.c"))):
        fx = os.path.basename(path)
        src = open(path, encoding="utf-8").read()
        run = subprocess.run(
            [exe, "--linkable", "-I", _C, path], capture_output=True, text=True, timeout=300
        )
        try:
            r = compile_unit(src, check_clang=False, includes=_includes_for(fx))
        except Exception as e:  # noqa: BLE001 -- a unit the oracle refuses: the twin refuses it too
            assert run.returncode != 0, f"{fx}: the oracle refuses it ({e}), the twin renders it"
            continue
        if r.fallback:
            assert run.returncode != 0, (
                f"{fx}: the oracle falls back ({r.fallback}), the twin renders it"
            )
            continue
        seen["units"] += 1
        try:
            oracle = emit_linkable(r.lowered, r.emitted)
        except ValueError as e:
            seen["refused"] += 1
            gname = re.search(r"global '([^']+)'", str(e)).group(1)
            assert run.returncode == 1 and f"global '{gname}'" in run.stderr, (
                fx,
                str(e),
                run.stderr[-300:],
            )
            continue
        assert run.returncode == 0, (
            f"{fx}: the twin refuses what the oracle renders\n{run.stderr[-300:]}"
        )
        twin = run.stdout
        defs = list(r.lowered.type_defs)
        after = twin.split(" */\n", 1)[1]
        if defs:
            seen["with_types"] += 1
            block = "\n".join(defs) + "\n"
            assert after.startswith(block), (
                f"{fx}: the types\n{block}\n----\n{after[: len(block) + 200]}"
            )
            nxt = after[len(block) :].split("\n", 1)[0]
            assert not nxt.endswith(" ;"), f"{fx}: a type the oracle does not define: {nxt}"
        want: dict = {}
        threads = r.lowered.thread_globals
        for gname, _ct, vals, is_extern, is_static in r.lowered.globals_decl:
            tls = "_Thread_local " if gname in threads else ""
            if is_extern:
                pat = rf"^extern {tls}[^=\n]*\b{re.escape(gname)}(\[\d*\])*;$"
            else:
                store = "static " if is_static else ""
                init = f" = {re.escape(vals[0])}" if vals else ""
                seen["with_inits"] += bool(vals)
                pat = rf"^(?!static |extern |\s){store}{tls}[^=\n]*\b{re.escape(gname)}(\[\d*\])*{init};$"
                if store:
                    pat = rf"^{store}{tls}(?!static |extern )[^=\n]*\b{re.escape(gname)}(\[\d*\])*{init};$"
            want[pat] = want.get(pat, 0) + 1
        seen["with_globals"] += bool(want)
        for pat, n in want.items():
            got = len(re.findall(pat, twin, re.M))
            assert got == n, f"{fx}: {n} global line(s) /{pat}/, the twin's artifact has {got}"
        # the wrapped functions (CF-PARMQUALS): on both rails, each function whose type keeps a qualifier below a
        # parameter's or the return's top level, but a variadic one and a static one the unit only calls; the same
        # wrapper on both
        quals = r.lowered.fn_quals
        as_value = set(r.lowered.init_refs) | {
            lf.globals_used[rid]
            for lf in r.lowered.functions.values()
            for rid in lf.fn_types
            if rid in lf.globals_used
        }
        want_w = {
            n
            for n, lf in r.lowered.functions.items()
            if quals.get(n) and not lf.variadic and (not lf.static_fn or n in as_value)
        }
        variadic_open |= {
            (fx, n) for n, lf in r.lowered.functions.items() if quals.get(n) and lf.variadic
        }
        ow, tw = _wrappers(oracle, r.lowered), _wrappers(twin, r.lowered)
        assert set(ow) == want_w and set(tw) == want_w, (fx, sorted(want_w), sorted(ow), sorted(tw))
        assert ow == tw, (fx, ow, tw)
        seen["wrapped"] += len(want_w)
        seen["called_only"] += sum(
            1
            for n, lf in r.lowered.functions.items()
            if quals.get(n) and not lf.variadic and lf.static_fn and n not in as_value
        )
        std = _original_std(clang, path)
        if std is None:
            continue
        seen["built"] += 1
        with tempfile.TemporaryDirectory() as tmp:
            for rail, text in (("oracle", oracle), ("twin", twin)):
                err = _builds_alone(clang, std, text, tmp)
                if err:
                    fails[rail].add(fx)
                assert not err, f"{fx}: the {rail}'s artifact does not build alone ({std})\n{err}"
    assert fails == {"oracle": set(), "twin": set()}, fails
    assert variadic_open == set(_LINKABLE_VARIADIC_OPEN), sorted(variadic_open)
    # what the corpus exercises (2026-10-27: 228 units rendered and built, 118 with types, 45 with globals, 120
    # initializers; 2026-10-29: 6 wrapped functions in 4 units, and 5 static ones in 3 units left unwrapped, each
    # only called): a vacuous pass -- no fixture rendered, no global compared, no function wrapped, none left
    # unwrapped for being only called -- is no pass
    assert seen["units"] >= 220 and seen["built"] >= 220 and seen["wrapped"] >= 6, seen
    assert seen["called_only"] >= 5, seen
    assert seen["with_types"] >= 110 and seen["with_globals"] >= 42 and seen["with_inits"] >= 110, (
        seen
    )


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
