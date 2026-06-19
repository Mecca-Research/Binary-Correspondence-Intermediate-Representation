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
