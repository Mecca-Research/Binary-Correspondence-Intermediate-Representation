"""C23 `_BitInt(N)` Q-fixed lane kernels: emit -> compile -> run, exact-width parity.

Roadmap §5 #2 ("more C23 where it pays"): the K_BCIR cost algebra is Q8 fixed point,
and the deterministic integer execution path is exactly where exact-width lanes pay.
`emit_qfixed_kernel_c` lowers a Q-fixed elementwise op with `_BitInt(N)` lanes and a
`_BitInt(2N)` product accumulator (the place a standard int would silently promote or
have no type at all), with a portable standard-int fallback selected by the
preprocessor -- so the same source builds bit-identically under C23 and C11.
"""

import shutil

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower.c_kernel import (
    compile_and_run_qfixed_c,
    emit_qfixed_kernel_c,
    emit_qfixed_selfcheck_c,
)
from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _no_cc() -> bool:
    return not (shutil.which("clang") or shutil.which("cc") or shutil.which("gcc"))


def _add_plan():
    m = vector_add(1024)
    return m, optimize(m, AVX, COOL)


def _mul_plan(n: int = 1024):
    m = Module(name="qmul")
    for rid in (1, 2, 3):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(n,)))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[
        Claim(id=1, opcode=Opcode.MUL, lane=Lane.U, stride_class=StrideClass.UNIT,
              count=n, rd=(1, 2), wr=(3,), op="vector.mul", domain=Domain.RAM)]))
    return m, optimize(m, AVX, COOL)


def test_qfixed_emits_bitint_with_a_portable_fallback():
    m, r = _mul_plan()
    c = emit_qfixed_kernel_c(m, r, "qk", lane_bits=16, frac_bits=8)
    assert "_BitInt(16) q_lane_t" in c          # exact 16-bit lane (C23 6.2.5)
    assert "_BitInt(32) q_acc_t" in c           # exact 2N-bit product accumulator
    assert "__STDC_VERSION__ >= 202311L" in c   # gated on C23
    assert "__BITINT_MAXWIDTH__" in c           # ... and the toolchain admitting 2N
    assert "typedef int16_t q_lane_t" in c      # the standard-int fallback
    assert ">> 8" in c                          # Q8 scaled multiply (the cost-model coupling)
    assert "restrict" in c                      # non-aliasing contract preserved


def test_qfixed_exact_nonstandard_width_has_no_standard_type():
    # A 12-/20-/24-bit Q-fixed lane has NO standard C integer type; _BitInt(N) is the
    # only exact representation -- the headline "where it pays".
    m, r = _mul_plan()
    for lane_bits in (12, 20, 24):
        c = emit_qfixed_kernel_c(m, r, "qk", lane_bits=lane_bits, frac_bits=lane_bits // 3)
        assert f"_BitInt({lane_bits}) q_lane_t" in c
        assert f"_BitInt({2 * lane_bits}) q_acc_t" in c


def test_qfixed_validates_format():
    m, r = _mul_plan()
    for bad in ((16, 0), (16, 16), (40, 8)):   # frac>=lane, frac=0, 2*lane>64
        try:
            emit_qfixed_kernel_c(m, r, "qk", lane_bits=bad[0], frac_bits=bad[1])
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


def test_selfcheck_is_self_contained_c():
    m, r = _mul_plan()
    sc = emit_qfixed_selfcheck_c(m, r, "qk", 16, 8)
    assert "int main(void)" in sc and "OK qk" in sc and "malloc" in sc


def test_qfixed_c23_and_c11_agree_bit_for_bit():
    """The _BitInt(N) build (C23) and the standard-int fallback (C11) must produce
    identical results -- the proof the exact-width lowering is portable, not lossy."""
    if _no_cc():
        return  # skip cleanly without a C compiler
    if not shutil.which("clang"):
        return  # _BitInt(N) + -std=c11/-std=c23 need a recent clang; gcc 13's c2x differs
    for label, (m, r) in (("ADD", _add_plan()), ("MUL", _mul_plan())):
        outs = {}
        for std in ("c23", "c11"):
            ok, out = compile_and_run_qfixed_c(m, r, "qk", lane_bits=16, frac_bits=8, std=std)
            assert ok, f"{label} {std}: {out}"
            outs[std] = out
        assert "bitint=1" in outs["c23"], "C23 must use _BitInt"
        assert "bitint=0" in outs["c11"], "C11 must use the fallback"


def test_qfixed_runs_for_mul_and_add_and_nonstandard_widths():
    if _no_cc() or not shutil.which("clang"):
        return
    m, r = _mul_plan()
    # exact 24-bit Q8.16 lane under C23 (no standard 24-bit type exists).
    ok, out = compile_and_run_qfixed_c(m, r, "qk", lane_bits=24, frac_bits=16, std="c23")
    assert ok, out
    assert "bitint=1" in out
