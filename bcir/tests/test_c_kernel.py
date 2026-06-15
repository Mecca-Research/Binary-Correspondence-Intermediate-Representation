"""The portable C23 kernel backend: emit -> compile -> run -> self-check, and the
R12 lowering contract (lane geometry, bounds, precision, non-aliasing)."""

import shutil

from bcir.codegen import emit_c_source
from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import compile_and_run_c, emit_kernel_c, emit_selfcheck_c
from bcir.verify import verify_c_lowering

AVX = TargetProfile.x86_avx512()


def _laws(diags):
    return {d.law for d in diags}


def _plan(theta=Theta.cool()):
    m = vector_add(1024)
    return m, optimize(m, AVX, theta)


# --- emission: clean, portable, C23 ----------------------------------------------

def test_kernel_is_c23_and_restrict_qualified():
    m, r = _plan()
    c = emit_kernel_c(m, r)
    assert c.startswith("/* BCIR -> portable C23 kernel")
    assert "const float * restrict A" in c          # the non-aliasing contract
    assert "static_assert(sizeof(float) == 4" in c  # C23 keyword, file scope
    assert "#pragma STDC FP_CONTRACT OFF" in c      # reproducible float
    assert "size_t n" in c                          # <stddef.h> trip count


def test_selected_width_drives_the_loop():
    # The default (no hardware width given): conservatively cap at the selected
    # width -- the literal geometry-encoding form, unchanged by the width-aware work.
    m, r = _plan(Theta.cool())                       # AVX-512 cool -> vec16
    assert "width=16" in emit_kernel_c(m, r)
    assert "for (; i + 16u <= n; i += 16u)" in emit_kernel_c(m, r)
    m, r = _plan(Theta.hot())                         # hot downclock -> vec8
    assert "width=8" in emit_kernel_c(m, r)
    assert "< 8u" in emit_kernel_c(m, r)


def test_scalar_width_emits_no_vector_loop():
    m, r = _plan()
    c = emit_kernel_c(m, r, width_override=1)
    assert "width=1" in c and "for (size_t i = 0; i < n; ++i)" in c
    assert "u <= n" not in c                          # no vector-width loop


# --- width-aware lowering: go-fast at the full lane, cap only when throttled ------

def test_full_hardware_lane_emits_an_idiomatic_uncapped_loop():
    # cool -> selected width 16 == AVX-512 widest lane -> go-fast: an idiomatic loop
    # the resident compiler vectorizes to >= the lane (and interleaves wider than a
    # hand-blocked loop could). No < 16u cap; geometry recorded in the head note.
    m, r = _plan(Theta.cool())
    c = emit_kernel_c(m, r, hw_width=AVX.vector_width)
    assert "width=16" in c and "full hardware lane" in c
    assert "16u" not in c                                       # not hand-blocked
    assert "for (size_t i = 0; i < n; ++i)" in c                # the idiomatic loop
    assert verify_c_lowering(m, r, c, hw_width=AVX.vector_width) == []   # R12 clean


def test_sub_maximal_width_keeps_the_throttle_cap():
    # hot -> selected width 8 < widest lane 16 -> a deliberate thermal/power throttle:
    # the loop MUST cap at 8 so the compiler cannot widen past the sub-maximal lane.
    m, r = _plan(Theta.hot())
    c = emit_kernel_c(m, r, hw_width=AVX.vector_width)
    assert "width=8" in c and "throttled lane" in c and "< 8u" in c
    assert verify_c_lowering(m, r, c, hw_width=AVX.vector_width) == []


def test_r12_rejects_a_full_lane_kernel_that_hides_a_sub_width_cap():
    # a go-fast kernel must not secretly cap below the lane (that throttles the backend).
    m, r = _plan(Theta.cool())
    good = emit_kernel_c(m, r, hw_width=AVX.vector_width)
    bad = good.replace("for (size_t i = 0; i < n; ++i)\n    C[i] = A[i] + B[i];",
                       "for (size_t j = 0; j < 8u; ++j) C[j] = A[j] + B[j];")
    assert "R12" in _laws(verify_c_lowering(m, r, bad, hw_width=AVX.vector_width))


def test_r12_rejects_a_throttle_kernel_with_the_cap_removed():
    # stripping the throttle cap lets the compiler widen past the thermal-limited lane.
    m, r = _plan(Theta.hot())
    bad = emit_kernel_c(m, r, hw_width=AVX.vector_width).replace("< 8u", "< n")
    assert "R12" in _laws(verify_c_lowering(m, r, bad, hw_width=AVX.vector_width))


def test_i32_kernel_uses_stdint_and_no_fp_pragma():
    m, r = _plan()
    c = emit_kernel_c(m, r, elem="i32")
    assert "#include <stdint.h>" in c and "int32_t" in c
    assert "FP_CONTRACT" not in c                     # integers do not contract


def test_codegen_fallback_delegates_to_the_c_backend():
    m, r = _plan()
    assert emit_c_source(m, r) == emit_kernel_c(m, r)


# --- AOT: compile with the host C compiler and self-check ------------------------

def test_kernel_compiles_and_self_checks():
    if shutil.which("clang") is None and shutil.which("cc") is None and shutil.which("gcc") is None:
        return  # skip cleanly when no C compiler is present
    for theta, elem in [(Theta.cool(), "f32"), (Theta.hot(), "f32"), (Theta.cool(), "i32")]:
        m, r = _plan(theta)
        ok, out = compile_and_run_c(m, r, elem=elem)
        assert ok, f"theta={theta} elem={elem}: {out}"
        assert "OK" in out


def test_selfcheck_exercises_the_tail():
    # the self-check calls the kernel at count and count+7 (non-divisible).
    m, r = _plan()
    sc = emit_selfcheck_c(m, r)
    assert "check(1024u)" in sc and "check(1024u + 7u)" in sc


# --- R12: the C lowering contract ------------------------------------------------

def test_clean_kernel_satisfies_R12():
    m, r = _plan()
    assert verify_c_lowering(m, r, emit_kernel_c(m, r)) == []


def test_tampered_width_is_R12():
    m, r = _plan()
    bad = emit_kernel_c(m, r).replace("width=16", "width=9")
    assert "R12" in _laws(verify_c_lowering(m, r, bad))


def test_dropped_restrict_is_R12():
    m, r = _plan()
    bad = emit_kernel_c(m, r).replace(" restrict", "")
    assert "R12" in _laws(verify_c_lowering(m, r, bad))


def test_wrong_op_is_R12():
    m, r = _plan()
    bad = emit_kernel_c(m, r).replace("] + B", "] - B")
    assert "R12" in _laws(verify_c_lowering(m, r, bad))


def test_missing_bounds_guard_is_R12():
    m, r = _plan()
    bad = emit_kernel_c(m, r).replace("<= n", "<= 0").replace("< n", "< 0")
    assert "R12" in _laws(verify_c_lowering(m, r, bad))


def test_missing_discharge_note_is_R12():
    m, r = _plan()
    bad = emit_kernel_c(m, r).split("\n", 1)[1]   # drop the head comment
    assert "R12" in _laws(verify_c_lowering(m, r, bad))
