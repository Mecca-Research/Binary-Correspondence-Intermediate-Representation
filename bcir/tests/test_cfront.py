"""The C frontend (Phase C.1 / C.1-MVP) — the staged conformance ladder L1–L4, each gated by the six
artifacts: a C source fixture, the lowered claim graph, the K_BCIR plan, the emitted C output, the
R1–R18 verifier checkpoint, and Clang behaviour-equivalence.

The Clang check is toolchain-gated (it compiles + runs the original fixture beside the emitted C),
so it self-skips in the quick tier and runs for real under c-runtime / thorough — the structural
artifacts (parse / lower / verify / plan / emit / explain) always run.
"""

import os
import shutil

from bcir.frontends.cfront import compile_unit

_FIX = os.path.join(os.path.dirname(__file__), "..", "frontends", "cfront", "fixtures")
_CLANG = shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _fixture(name: str) -> str:
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return f.read()


def _assert_six_artifacts(name: str):
    r = compile_unit(_fixture(name))
    # (2) claim graph, (3) plan, (4) emitted C, (6a) R1–R18 verifier, explain — always.
    assert r.lowered.functions, f"{name}: no functions lowered"
    assert r.is_clean, f"{name}: R1–R18 not clean: {[(d.law, d.message) for d in r.diagnostics]}"
    for fn, lf in r.lowered.functions.items():
        assert r.plans[fn].steps, f"{name}:{fn}: empty plan"
        assert r.emitted[fn].strip().startswith("static"), f"{name}:{fn}: no emitted C"
        assert r.explain[fn], f"{name}:{fn}: no explain text"
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
