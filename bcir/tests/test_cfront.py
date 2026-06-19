"""The C frontend (Phase C.1 / C.1-MVP) — the staged conformance ladder L1–L4, each gated by the six
artifacts: a C source fixture, the lowered claim graph, the K_BCIR plan, the emitted C output, the
R1–R18 verifier checkpoint, and Clang behaviour-equivalence.

The Clang check is toolchain-gated (it compiles + runs the original fixture beside the emitted C),
so it self-skips in the quick tier and runs for real under c-runtime / thorough — the structural
artifacts (parse / lower / verify / plan / emit / explain) always run.
"""

import os
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront import compile_unit

_FIX = os.path.join(os.path.dirname(__file__), "..", "frontends", "cfront", "fixtures")
_CLANG = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _fixture(name: str) -> str:
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return f.read()


def _assert_six_artifacts(name: str, *, includes=None, embeds=None):
    r = compile_unit(_fixture(name), includes=includes, embeds=embeds)
    # (2) claim graph, (3) plan, (4) emitted C, (6a) R1–R18 verifier, explain — always.
    assert r.lowered.functions, f"{name}: no functions lowered"
    assert r.is_clean, f"{name}: R1–R18 not clean: {[(d.law, d.message) for d in r.diagnostics]}"
    for fn, lf in r.lowered.functions.items():
        assert r.plans[fn].steps, f"{name}:{fn}: empty plan"
        assert "static" in r.emitted[fn] and f"bcir_{fn}" in r.emitted[fn], f"{name}:{fn}: no C"
        assert r.explain[fn], f"{name}:{fn}: no explain text"
        # C.2 attestation stamped on every emitted function.
        att = r.attestation[fn]
        assert att["R18_callgraph_integrity"] == "clean"
        assert "attestation" in r.emitted[fn]
    # (5) Clang behaviour-equivalence — real under a toolchain, skipped cleanly otherwise.
    if _CLANG:
        assert r.behaviour_equivalent, f"{name}: not behaviour-equivalent ({r.equivalence})"
    else:
        assert r.equivalence.startswith("skip"), f"{name}: expected skip, got {r.equivalence}"
    return r


# --- L1: fixed-width integer expressions ---------------------------------------------------------

def test_L1_integer_expressions():
    r = _assert_six_artifacts("L1_int_expr.c")
    claims = r.lowered.functions["l1_compute"].claims
    ops = {c.op.rsplit(".", 1)[-1] for c in claims}
    assert {"add", "mul", "xor", "shl", "sub"} <= ops          # the source operators are present
    assert not any(c.op.startswith("c.call") for c in claims)  # no calls at L1


def test_L1_string_literal_sizeof():
    """A string literal lexes as a STRING token (escapes intact) and, in `sizeof`, folds to its
    char-array length (decoded bytes + the NUL) -- matching Clang. (The literal is not materialized
    as a value yet; that is the next slice.)"""
    from bcir.frontends.cfront.clex import tokenize
    assert [t.text for t in tokenize(r'x "a\tb" y') if t.kind == "STRING"] == [r'"a\tb"']

    def szof(lit: str) -> int:
        r = compile_unit(f"uint32_t f(void){{ return sizeof {lit}; }}", check_clang=False)
        return next(c.imm[0] for c in r.lowered.functions["f"].claims if c.op == "c.const")
    assert szof(r'"hello"') == 6
    assert szof('""') == 1
    assert szof(r'"tab\there"') == 9        # \t is one byte
    assert szof(r'"\x41\x42"') == 3         # hex escapes -> one byte each
    assert szof(r'"\0"') == 2               # octal NUL -> one byte


def test_L3_string_literal_value():
    """A string literal lowers to an anonymous read-only char[] global; using it decays to a pointer
    (indexing reads a byte; the bare literal is the pointer). The emitter renders the anonymous global
    as the inline literal, so the emit is Clang-equivalent with no synthesized declaration."""
    def eqok(r):                            # match when Clang is present, else cleanly skipped
        return r.equivalence == "match" or r.equivalence.startswith("skip")
    # indexing a string literal -> the i-th byte, via an anonymous global referenced by a load.
    r = compile_unit('uint32_t pick(uint32_t i){ return "ABCD"[i & 3]; }\n')
    lf = r.lowered.functions["pick"]
    assert r.is_clean and eqok(r)
    assert '"ABCD"' in lf.globals_used.values()                  # the anon global renders inline
    assert any(c.op == "c.load" for c in lf.claims)
    # the bare literal is a const char* pointer (return-only: rendered inline, no claims/decl).
    r2 = compile_unit('const char *msg(void){ return "hello"; }\n')
    lf2 = r2.lowered.functions["msg"]
    assert r2.is_clean and eqok(r2) and lf2.ret_type.kind == "pointer"
    assert '"hello"' in lf2.globals_used.values()


def test_L1_character_constants():
    """A character constant lexes as a CHAR token and folds to an `int` value: a single character is
    its byte value (signed char), an escape decodes to one byte (simple `\\c`, octal `\\NNN`, hex
    `\\xHH`), and a multi-character `'AB'` packs big-endian like Clang/GCC -- so it lowers to a
    `c.const` and stays behaviour-equal to Clang."""
    from bcir.frontends.cfront.clex import parse_char_literal, tokenize
    assert [t.text for t in tokenize(r"x 'A' '\n'") if t.kind == "CHAR"] == ["'A'", r"'\n'"]
    assert parse_char_literal("'A'") == 65
    assert parse_char_literal(r"'\n'") == 10 and parse_char_literal(r"'\t'") == 9
    assert parse_char_literal(r"'\x7a'") == 122 and parse_char_literal(r"'\101'") == 65
    assert parse_char_literal(r"'\0'") == 0
    assert parse_char_literal("'AB'") == 0x4142            # multi-character constant (big-endian pack)

    def eqok(r):
        return r.equivalence == "match" or r.equivalence.startswith("skip")
    r = compile_unit("uint32_t f(void){ return 'A' + '\\n' + 'AB'; }\n")
    lf = r.lowered.functions["f"]
    assert r.is_clean and eqok(r)
    assert sum(1 for c in lf.claims if c.op == "c.const") == 3   # three char constants, all folded


def test_L3_string_literal_dedup_and_table():
    """Identical string literals in a function share one anonymous global (dedup), and a literal
    longer than any fixed-size name buffer still materializes in full -- the spelling is held
    out-of-band, so the emit can inline it at any length and stay Clang-equivalent."""
    long_lit = "this string literal is definitely longer than thirty-two bytes"
    src = ('uint32_t f(uint32_t i){ return "kv"[i & 1] + "kv"[i & 1] + "%s"[i %% 7]; }\n' % long_lit)
    r = compile_unit(src, check_clang=False)
    lf = r.lowered.functions["f"]
    str_rids = [rid for rid in lf.resources if rid >= 970000]
    assert len(str_rids) == 2                                    # "kv" used twice -> one global (dedup)
    assert ('"%s"' % long_lit) in lf.globals_used.values()       # the long literal survives in full
    assert max(len(s) for s in lf.globals_used.values()) > 34    # past the old 32-byte name cap


def test_L7_string_literal_concatenation():
    """Adjacent string literals concatenate (C translation phase 6): `sizeof` folds across the pieces,
    and the pieces stay adjacent (not merged into a decoded run), so a hex/octal escape never absorbs
    the next piece's leading digit -- `"\\x41" "2"` is 'A' then '2', never `\\x412`."""
    from bcir.frontends.cfront.lower import _str_bytes

    def szof(lit: str) -> int:
        r = compile_unit("uint32_t f(void){ return sizeof %s; }" % lit, check_clang=False)
        f = r.lowered.functions["f"]
        return [c for c in f.claims if c.op == "c.const"][-1].imm[0]

    assert _str_bytes('"abc" "de"') == 5 and szof('"abc" "de"') == 6        # 5 chars + NUL
    assert _str_bytes(r'"tab\t" "x"') == 5                                  # \t stays one byte
    assert _str_bytes(r'"\x41" "2"') == 2                                   # escape boundary: 'A','2'
    assert szof(r'"\x41" "2"') == 3
    # the concatenation is one literal, so postfix `[]` indexes the joined bytes (8 chars).
    r = compile_unit('uint32_t g(uint32_t i){ return "hi " "there"[i % 8]; }\n')
    lg = r.lowered.functions["g"]
    assert r.is_clean and (r.equivalence == "match" or r.equivalence.startswith("skip"))
    assert '"hi " "there"' in lg.globals_used.values()                      # pieces kept adjacent


def test_L7_wide_and_utf_literal_prefixes():
    """Wide/UTF prefixes `L`/`u`/`U`/`u8` on character and string literals: a bare prefix letter is
    still an identifier; a prefixed character constant keeps its (ASCII) code-point value; a prefixed
    string literal has the element width of its character type, so `sizeof` scales."""
    from bcir.frontends.cfront.clex import parse_char_literal, tokenize

    kinds = {(t.kind, t.text) for t in tokenize('L"a" u8"b" u\'c\' U + Label')}
    assert ("STRING", 'L"a"') in kinds and ("STRING", 'u8"b"') in kinds and ("CHAR", "u'c'") in kinds
    assert ("IDENT", "U") in kinds and ("IDENT", "Label") in kinds          # bare prefix -> identifier
    assert parse_char_literal(r"L'\n'") == 10 and parse_char_literal("u'A'") == 65

    def szof(lit: str) -> int:
        r = compile_unit("uint32_t f(void){ return sizeof %s; }" % lit, check_clang=False)
        f = r.lowered.functions["f"]
        return [c for c in f.claims if c.op == "c.const"][-1].imm[0]

    assert szof('"hi"') == 3 and szof('u8"hi"') == 3       # char / char (3 units × 1 byte)
    assert szof('u"hi"') == 6                              # char16_t (× 2)
    assert szof('L"hi"') == 12 and szof('U"hi"') == 12     # wchar_t / char32_t (× 4)
    r = compile_unit("uint32_t f(void){ return L'\\n' + u'A'; }\n")
    assert r.is_clean and (r.equivalence == "match" or r.equivalence.startswith("skip"))


# --- L2: struct / union layout + member access ---------------------------------------------------

def test_L2_struct_member_access():
    r = _assert_six_artifacts("L2_struct.c")
    lf = r.lowered.functions["l2_sumsq"]
    loads = [c for c in lf.claims if c.op == "c.load"]
    assert loads, "L2 should lower member access to c.load claims"
    # the second field (y) is read at byte offset 4 (Clang-compatible struct layout).
    assert any(c.imm and c.imm[0] == 4 for c in loads), "expected a member load at offset 4"
    assert r.lowered.aggregates["point"].size == 8


# --- L3: pointers / arrays (GEP-equivalent indexing) ---------------------------------------------

def test_L3_pointer_array_indexing():
    r = _assert_six_artifacts("L3_ptr_array.c")
    lf = r.lowered.functions["l3_index"]
    indexed = [c for c in lf.claims if c.op == "c.load" and len(c.rd) == 2]
    assert indexed, "L3 should lower base[i] to an indexed (base, index) c.load"


def test_L3_array_of_row_pointer_declarator():
    """A pointer-to-array parameter `(*m)[8]` (the row-pointer a 2D array decays to) lowers like the
    2D array param: it decays to a flat element pointer with the inner extent recorded, and `m[i][j]`
    flattens row-major to `i*8 + j` -- so the outer index is scaled by the inner dim (8)."""
    src = ("uint32_t f(uint32_t (*m)[8], uint32_t i, uint32_t j){ return m[i & 7][j & 7]; }\n")
    r = compile_unit(src)
    lf = r.lowered.functions["f"]
    assert r.is_clean and (r.equivalence == "match" or r.equivalence.startswith("skip"))
    p0 = lf.params[0][2]
    assert p0.kind == "pointer" and p0.shape[1] == 8           # decayed row pointer, inner extent 8
    assert any(c.op == "c.const" and c.imm and c.imm[0] == 8 for c in lf.claims)   # the row stride


# --- L4: functions + the call graph -> R18 -------------------------------------------------------

def test_L4_call_graph_is_R18_clean():
    r = _assert_six_artifacts("L4_callgraph.c")
    assert set(r.lowered.functions) == {"l4_scale", "l4_main"}
    main = r.lowered.functions["l4_main"]
    assert len([c for c in main.claims if c.op.startswith("c.call:")]) == 2
    assert r.r18_ok


# --- L5: volatile / MMIO register map + bitfields ------------------------------------------------

def test_L5_mmio_register_map_and_bitfields():
    from bcir.model import Domain
    r = _assert_six_artifacts("L5_mmio_regmap.c")
    dec = r.lowered.functions["uart_decode"]
    # the volatile register pointer became an MMIO resource...
    assert any(res.domain == Domain.MMIO for res in dec.module.resources.values())
    # ...the MMIO load is ordered (barriered, not unique)...
    mmio_loads = [c for c in dec.claims if c.op == "c.load" and c.domain == Domain.MMIO]
    assert mmio_loads and all(c.hazard == "barriered" for c in mmio_loads)
    # ...and bitfields lowered to explicit mask/shift extract claims.
    assert [c for c in dec.claims if c.op == "c.bf.get"]
    cfg = r.lowered.aggregates["ctrl_bits"]
    assert cfg.field("enable")[2] == 0 and cfg.field("baud")[2] == 3   # bit offsets pack LSB-first


def test_L5_mmio_write_requires_barriered_hazard():
    from bcir.model import Domain
    r = compile_unit(_fixture("L5_mmio_regmap.c"), check_clang=False)
    cfg = r.lowered.functions["uart_configure"]
    stores = [c for c in cfg.claims if c.op == "c.store" and c.domain == Domain.MMIO]
    assert stores and all(c.hazard == "barriered" for c in stores)   # R3: MMIO write hazard
    assert r.is_clean                                                  # R3/R5 clean


# --- L6: control flow (branches + bounded loops) -------------------------------------------------

def test_L6_branches():
    r = _assert_six_artifacts("L6_branch.c")
    from bcir.frontends.cfront.lower import IfNode
    body = r.lowered.functions["l6_clamp"].body
    assert any(isinstance(n, IfNode) for n in body), "if should lower to an IfNode in the body tree"
    assert "if (" in r.emitted["l6_clamp"]


def test_L6_bounded_loop():
    r = _assert_six_artifacts("L6_loop.c")
    from bcir.frontends.cfront.lower import WhileNode
    body = r.lowered.functions["l6_weighted_sum"].body
    assert any(isinstance(n, WhileNode) for n in body), "while should lower to a WhileNode"
    assert "while (1)" in r.emitted["l6_weighted_sum"]


# --- L7: the preprocessor (macros / conditionals / include / #embed) -----------------------------

def test_L7_object_and_function_macros():
    r = _assert_six_artifacts("L7_macros.c")
    # the #if HW_REV>=2 branch was taken and FIELD()/MASK expanded into the claim graph.
    assert any(c.op == "c.bin.shr" for c in r.lowered.functions["l7_decode"].claims)


def test_L7_include_project_header():
    r = _assert_six_artifacts("L7_include.c", includes={"regmap.h": _fixture("regmap.h")})
    # REG_BASE (0x40000000) from the header must have reached the lowered constants.
    consts = {c.imm[0] for c in r.lowered.functions["l7_regaddr"].claims if c.op == "c.const"}
    assert 0x40000000 in consts


def test_L7_c23_embed_table():
    data = bytes((i * 37) & 0xFF for i in range(16))
    r = _assert_six_artifacts("L7_embed.c", embeds={"crc.bin": data})
    g = r.lowered.functions["l7_crc_step"].globals_used
    assert "crc_table" in g.values()                          # the #embed table is a referenced global
    assert r.lowered.aggregates == r.lowered.aggregates       # (no aggregates needed here)


def test_L7_preprocessor_unit():
    from bcir.frontends.cfront.cpp import preprocess
    out = preprocess("#define A 2\n#define SQ(x) ((x)*(x))\nint v = SQ(A+1);")
    assert "((2+1)*(2+1))" in out.replace(" ", "")            # arg expanded + rescanned
    assert preprocess("#if 1+1==2\nyes\n#else\nno\n#endif").strip() == "yes"
    assert preprocess("#ifdef X\na\n#elifndef Y\nb\n#endif").strip() == "b"   # C23 #elifndef
    assert preprocess("x\n#embed \"d\"\ny", embeds={"d": bytes([1, 2, 3])}).split() == \
        ["x", "1,", "2,", "3", "y"]


def test_L7_predefined_file_and_line():
    """__LINE__/__FILE__ are dynamic predefined macros: __LINE__ tracks the 1-based logical line
    (reflecting the macro *invocation* site, not the definition), __FILE__ the current file name,
    and both report as `defined` to #ifdef / defined(). __STDC_HOSTED__ is predefined too."""
    from bcir.frontends.cfront.cpp import Preprocessor, preprocess
    # __LINE__ counts logical lines from 1; __FILE__ is a string literal of the file name.
    assert preprocess("a __LINE__\nb __LINE__\nc __LINE__").split() == ["a", "1", "b", "2", "c", "3"]
    assert Preprocessor().process("x __FILE__", name="foo.c").strip() == 'x"foo.c"'
    # __LINE__ through an object / function macro reflects the invocation line, not the #define line.
    assert "intc=3;" in preprocess("#define L __LINE__\nq\nint c = L;").replace(" ", "")
    assert "intb=2;" in preprocess("#define ID(x) x\nint b = ID(__LINE__);").replace(" ", "")
    # both are `defined`, and usable in #if arithmetic.
    assert preprocess("#ifdef __LINE__\nyes\n#endif").strip() == "yes"
    assert preprocess("#if defined(__FILE__) && __LINE__ == 1\nok\n#endif").strip() == "ok"
    # __STDC_HOSTED__ is predefined (the hosted-C signal).
    assert preprocess("#if __STDC_HOSTED__\nhosted\n#endif").strip() == "hosted"
    # across an #include boundary __FILE__/__LINE__ switch to the header and restore on return.
    body = preprocess('t __LINE__ __FILE__\n#include "h"\nu __LINE__ __FILE__\n',
                      includes={"h": "in __LINE__ __FILE__"})
    assert body == 't 1"<source>"\nin 1"h"\nu 3"<source>"\n'


def test_L7_line_directive():
    """`#line N ["file"]` resets the presumed line number of the *following* line (and __FILE__ when a
    name is given); operands are macro-expanded; a #line-set name survives an #include and restores."""
    from bcir.frontends.cfront.cpp import preprocess
    # #line sets the NEXT line; numbering then counts up from there.
    assert preprocess("a __LINE__\n#line 100\nb __LINE__\nc __LINE__").split() == \
        ["a", "1", "b", "100", "c", "101"]
    # a file-name operand redirects __FILE__ too, and operands are macro-expanded.
    assert preprocess('#line 50 "foo.c"\nx __LINE__ __FILE__').strip() == 'x 50"foo.c"'
    assert preprocess("#define N 200\n#line N\nq __LINE__").split() == ["q", "200"]
    # a malformed #line is ignored (numbering just continues); an inactive-branch #line never fires.
    assert preprocess("p __LINE__\n#line\nq __LINE__").split() == ["p", "1", "q", "3"]
    assert preprocess("#if 0\n#line 999\n#endif\nr __LINE__").split() == ["r", "4"]
    # a #line-set name persists across an #include and is restored on return.
    body = preprocess('#line 7 "a.h"\nt __LINE__ __FILE__\n#include "i"\nu __LINE__ __FILE__\n',
                      includes={"i": "in __LINE__ __FILE__"})
    assert body == 't 7"a.h"\nin 1"i"\nu 9"a.h"\n'


def test_L7_predefined_date_and_time():
    """__DATE__/__TIME__ are predefined string macros, frozen by SOURCE_DATE_EPOCH (UTC) so builds
    are reproducible (a single-digit day is space-padded), and reported as `defined`."""
    import os
    from bcir.frontends.cfront.cpp import preprocess
    old = os.environ.get("SOURCE_DATE_EPOCH")
    try:
        os.environ["SOURCE_DATE_EPOCH"] = "1234567890"        # 2009-02-13 23:31:30 UTC
        assert preprocess("__DATE__ __TIME__").strip() == '"Feb 13 2009""23:31:30"'
        os.environ["SOURCE_DATE_EPOCH"] = "1577836800"        # 2020-01-01 00:00:00 UTC
        assert preprocess("d=__DATE__").strip() == 'd="Jan  1 2020"'    # single-digit day padded
        assert preprocess("#ifdef __TIME__\nyes\n#endif").strip() == "yes"
        assert preprocess("#if defined(__DATE__)\nok\n#endif").strip() == "ok"
    finally:
        if old is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = old


def test_L7_pragma_operator():
    """`_Pragma("...")` is a lowering no-op (like `#pragma`): it is recognized anywhere a token can
    appear — including when produced by a macro via `#` stringize — and consumed, emitting nothing."""
    from bcir.frontends.cfront.cpp import preprocess
    assert preprocess('int x; _Pragma("once") int y;').strip() == "int x;int y;"
    assert preprocess('a _Pragma("GCC diagnostic push") b').strip() == "a b"
    assert preprocess('p _Pragma("a(b)c") q').strip() == "p q"        # balanced parens consumed
    # produced by a macro (destringize) — still consumed after rescanning.
    assert preprocess("#define DO(x) _Pragma(#x)\nDO(message hi)\nz").strip() == "z"
    assert preprocess('#define PUSH _Pragma("pack(push,1)")\nPUSH\nint a;').strip() == "int a;"
    # a bare _Pragma not followed by '(' is left alone (degenerate).
    assert preprocess("_Pragma").strip() == "_Pragma"


def test_L7_file_macro_real_path():
    """__FILE__ reflects the translation unit's name when the driver supplies one; the default stays
    "<source>". `preprocess(name=...)` / `compile_unit(filename=...)` thread it. (__FILE__ inside
    compiled code awaits string-literal lexing; today it surfaces via the preprocessor / `-E`.)"""
    from bcir.frontends.cfront.cpp import preprocess
    from bcir.frontends.cfront import compile_unit
    assert preprocess("a __FILE__\nb __FILE__", name="proj/foo.c").strip() == \
        'a"proj/foo.c"\nb"proj/foo.c"'
    assert preprocess("x __FILE__").strip() == 'x"<source>"'                  # default unchanged
    # compile_unit accepts + forwards `filename` to the preprocessor (no crash; plumbing for __FILE__).
    compile_unit("unsigned int n(void){ return __LINE__; }\n", filename="k.c", check_clang=False)


def test_L7_has_feature_macros():
    """The `__has_*` feature-test operators in #if: `__has_attribute` reports the L8 ABI attributes
    (packed/aligned, GCC `__x__` spelling too) and nothing else; `__has_builtin`/`__has_c_attribute`
    report 0 (none supported yet). All `__has_*` operators are reported as `defined`."""
    from bcir.frontends.cfront.cpp import preprocess
    assert preprocess("#if __has_attribute(packed)\nY\n#else\nN\n#endif").strip() == "Y"
    assert preprocess("#if __has_attribute(__aligned__)\nY\n#else\nN\n#endif").strip() == "Y"
    assert preprocess("#if __has_attribute(deprecated)\nY\n#else\nN\n#endif").strip() == "N"
    assert preprocess("#if __has_builtin(__builtin_expect)\nY\n#else\nN\n#endif").strip() == "N"
    assert preprocess("#if __has_c_attribute(nodiscard)\nY\n#else\nN\n#endif").strip() == "N"
    # the standard defined-guarded idiom works (the operators report as `defined`).
    assert preprocess("#ifdef __has_attribute\nD\n#endif").strip() == "D"
    assert preprocess("#if defined(__has_attribute) && __has_attribute(aligned)\nOK\n#endif").strip() \
        == "OK"


def test_L7_has_include():
    """`__has_include` probes the header search path (resolved against the in-memory mount here),
    both the quoted and angle forms; the dual-rail C twin resolves the same names from disk."""
    from bcir.frontends.cfront.cpp import preprocess
    inc = {"there.h": "int x;"}
    assert preprocess('#if __has_include("there.h")\nY\n#else\nN\n#endif', includes=inc).strip() == "Y"
    assert preprocess("#if __has_include(<there.h>)\nY\n#else\nN\n#endif", includes=inc).strip() == "Y"
    assert preprocess('#if __has_include("gone.h")\nY\n#else\nN\n#endif', includes=inc).strip() == "N"
    assert preprocess('#if defined(__has_include) && __has_include("there.h")\nOK\n#endif',
                      includes=inc).strip() == "OK"


def test_L7_variadic_macros():
    """Variadic `#define M(...)` / `M(a, ...)`: __VA_ARGS__ expands to *all* trailing args
    (comma-joined), works empty, and through `#` stringize / `##` paste."""
    from bcir.frontends.cfront.cpp import preprocess
    assert preprocess("#define V(...) f(__VA_ARGS__)\nV(1,2,3)").strip() == "f(1,2,3)"
    assert preprocess("#define L(a, ...) g(a, __VA_ARGS__)\nL(x,1,2)").strip() == "g(x,1,2)"
    assert preprocess("#define E(a, ...) k(a, __VA_ARGS__)\nE(z)").strip() == "k(z,)"   # empty
    assert preprocess("#define S(...) #__VA_ARGS__\nS(1, 2, 3)").strip() == '"1,2,3"'   # stringize
    assert preprocess("#define P(...) x ## __VA_ARGS__\nP(1,2)").strip() == "x1,2"      # paste
    # C23 __VA_OPT__: the content appears iff __VA_ARGS__ is non-empty (the trailing-comma idiom).
    log = "#define LOG(fmt, ...) p(fmt __VA_OPT__(,) __VA_ARGS__)\n"
    assert preprocess(log + 'LOG("hi")').strip() == 'p("hi")'
    assert preprocess(log + 'LOG("hi", 1, 2)').strip() == 'p("hi",1,2)'
    assert preprocess("#define M(...) a __VA_OPT__(X) b\nM()\nM(1)").strip() == "a b\na X b"
    assert preprocess("#define E(...) z __VA_OPT__(Y)\nE(,)").strip() == "z Y"   # a comma is a token


# --- L8: ABI — struct return-by-value, packed/aligned, calling convention vs Clang ----------------

def _clang_layout(struct_src: str, tag: str, fields: list):
    """sizeof + offsetof of `struct tag` as *Clang* computes them (the ABI ground truth)."""
    probe = (f"#include <stdint.h>\n#include <stddef.h>\n#include <stdio.h>\n{struct_src}\n"
             f"int main(void){{ printf(\"%zu\", sizeof(struct {tag}));"
             + "".join(f' printf(" %zu", offsetof(struct {tag}, {f}));' for f in fields)
             + " return 0; }")
    with tempfile.TemporaryDirectory() as d:
        src, exe = os.path.join(d, "p.c"), os.path.join(d, "p")
        open(src, "w").write(probe)
        if subprocess.run([_CLANG, "-std=c23", src, "-o", exe],
                          capture_output=True).returncode != 0:
            subprocess.run([_CLANG, src, "-o", exe], capture_output=True, check=True)
        nums = [int(x) for x in subprocess.run([exe], capture_output=True, text=True).stdout.split()]
    return nums[0], dict(zip(fields, nums[1:]))


def test_L8_struct_return_by_value():
    r = _assert_six_artifacts("L8_struct_return.c")
    assert r.lowered.functions["l8_swap"].ret_type.is_aggregate    # returns a struct by value


def test_L8_packed_layout_matches_clang():
    r = _assert_six_artifacts("L8_packed.c")
    hdr = r.lowered.aggregates["wire_hdr"]
    # the frontend's packed layout (no padding): cmd@0, addr@1, len@5, size 7.
    assert hdr.field("addr")[1] == 1 and hdr.field("len")[1] == 5 and hdr.size == 7
    if _CLANG:                                                # cross-check against Clang's ABI
        src = ("struct __attribute__((packed)) wire_hdr "
               "{ uint8_t cmd; uint32_t addr; uint16_t len; };")
        size, offs = _clang_layout(src, "wire_hdr", ["cmd", "addr", "len"])
        assert size == hdr.size
        assert offs == {"cmd": hdr.field("cmd")[1], "addr": hdr.field("addr")[1],
                        "len": hdr.field("len")[1]}


def test_L8_natural_and_aligned_layout_matches_clang():
    src = ("#include <stdint.h>\n"
           "struct natural { uint8_t a; uint32_t b; uint16_t c; };\n"
           "struct __attribute__((aligned(16))) blk { uint32_t v; };\n"
           "uint32_t use(struct natural n) { return n.a; }\n")
    r = compile_unit(src, check_clang=False)
    nat, blk = r.lowered.aggregates["natural"], r.lowered.aggregates["blk"]
    assert nat.field("b")[1] == 4 and nat.size == 12          # natural alignment padding
    assert blk.align == 16 and blk.size == 16                 # forced alignment
    if _CLANG:
        size, offs = _clang_layout("struct natural { uint8_t a; uint32_t b; uint16_t c; };",
                                   "natural", ["a", "b", "c"])
        assert size == nat.size and offs["b"] == nat.field("b")[1]


# --- C.2: the self-check artifact + attestation --------------------------------------------------

def test_C2_selfcheck_artifact_runs():
    from bcir.frontends.cfront import emit_selfcheck
    r = compile_unit(_fixture("L5_mmio_regmap.c"))
    sc = emit_selfcheck(r)
    assert "int main(" in sc and "bcir_uart_decode" in sc
    if _CLANG:                                    # the artifact is a real, compilable self-check
        assert r.behaviour_equivalent


def test_R18_rejects_recursion():
    r = compile_unit("#include <stdint.h>\nuint32_t f(uint32_t n){ return f(n - 1); }\n",
                     check_clang=False)
    assert not r.r18_ok
    assert any(d.law == "R18" and "recurs" in d.message for d in r.diagnostics)


def test_R18_rejects_undefined_callee():
    r = compile_unit("#include <stdint.h>\nuint32_t g(uint32_t a){ return missing(a); }\n",
                     check_clang=False)
    assert not r.r18_ok
    assert any(d.law == "R18" and "undefined" in d.message for d in r.diagnostics)


# --- the frontend lowers to the SAME model the oracle verifies (the dual-rail invariant) ---------

def test_lowering_targets_the_real_claim_graph_model():
    from bcir.model import Claim, Module, Resource
    r = compile_unit(_fixture("L1_int_expr.c"), check_clang=False)
    m = r.lowered.functions["l1_compute"].module
    assert isinstance(m, Module)
    assert all(isinstance(res, Resource) for res in m.resources.values())
    assert all(isinstance(c, Claim) for p in m.phases for c in p.claims)
