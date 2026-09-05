"""LLVM lowering tests: the emitted kernel assembles and runs+self-checks."""

import shutil

from bcir.examples import fused_chain, histogram_gather, scan_chain, vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import compile_and_run, emit_kernel_ll
from bcir.lower.llvm import emit_harness_c, harness_trip_counts
from bcir.run import main as run_main


def _result():
    return vector_add(1024), optimize(vector_add(1024), TargetProfile.x86_avx512(), Theta.cool())


def test_emitted_kernel_is_legal_llvm():
    # The selected vec16 lowering must assemble under a modern llvm-as (opaque ptrs).
    if shutil.which("llvm-as") is None:
        return  # skip cleanly when the LLVM toolchain is absent
    import subprocess
    import tempfile
    import os

    module, res = _result()
    ll = emit_kernel_ll(module, res, fn_name="bcir_kernel")
    assert "<16 x float>" in ll  # cool AVX-512 -> vec16
    assert "width=16 elem=float epilogue=scalar" in ll.splitlines()[0]
    assert "%nvec = and i64 %n, -16" in ll  # the vector loop's bound (the tail contract)
    assert "%c = fadd float %a, %b" in ll  # the scalar epilogue
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.ll")
        with open(path, "w") as f:
            f.write(ll)
        out = subprocess.run(
            ["llvm-as", path, "-o", os.path.join(d, "k.bc")], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr


def test_lowering_compiles_and_runs():
    # End-to-end Milestones 5-7 in miniature: build with clang and self-check. The
    # harness drives the planned count, a non-divisible one, a sub-width one and zero
    # behind canaries (S0-8): the parent's vector loop wrote past n on the second.
    if shutil.which("clang") is None:
        return  # skip cleanly when clang is absent
    module, res = _result()
    ok, out = compile_and_run(module, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK bcir_kernel trips=1024, 1031, 15, 0" in out


def test_harness_drives_the_tail_contract():
    """S0-8: the self-check harness calls the kernel with every trip count that can
    expose an unmasked tail, each behind a canary region the kernel must not touch."""
    module, res = _result()
    assert harness_trip_counts(module, res) == (1024, 1031, 15, 0)
    harness = emit_harness_c(module, res, fn_name="bcir_kernel")
    assert "long trips[] = { 1024, 1031, 15, 0 };" in harness
    assert "CANARY" in harness and "wrote past n" in harness


def test_non_divisible_count_keeps_the_selected_width():
    """S0-8: a count that is not a multiple of the width is no longer legalized to
    scalar; the vector kernel keeps the selected width and the epilogue finishes the
    remainder (the runtime n is what the harness varies anyway)."""
    module = vector_add(1000)
    res = optimize(module, TargetProfile.x86_avx512(), Theta.cool())
    ll = emit_kernel_ll(module, res)
    assert "width=16 elem=float epilogue=scalar" in ll.splitlines()[0]
    assert "<16 x float>" in ll and "%nvec = and i64 %n, -16" in ll
    if shutil.which("clang") is not None:
        ok, out = compile_and_run(module, res)
        assert ok, out
        assert "OK bcir_kernel trips=1000, 1007, 15, 0" in out
    # a scalar selection carries no epilogue and no vector type
    scalar = emit_kernel_ll(module, res, width_override=1)
    assert "width=1 elem=float epilogue=none" in scalar.splitlines()[0]
    assert "x float>" not in scalar and "and i64" not in scalar


def test_non_power_of_two_width_is_refused():
    module, res = _result()
    try:
        emit_kernel_ll(module, res, width_override=12)
    except NotImplementedError as exc:
        assert "power-of-two" in str(exc)
    else:
        raise AssertionError("a width the mask cannot express was accepted")


def _assert_rejected(module):
    result = optimize(module, TargetProfile.x86_avx512(), Theta.cool())
    try:
        emit_kernel_ll(module, result)
    except NotImplementedError as exc:
        assert "single-claim elementwise LLVM AOT/JIT subset" in str(exc)
    else:
        raise AssertionError("partial LLVM lowering silently accepted an unsupported graph")


def test_multi_claim_graphs_are_rejected_instead_of_truncated():
    _assert_rejected(fused_chain(32))
    _assert_rejected(scan_chain(32))


def test_multi_claim_graph_is_rejected_even_if_realization_omits_a_claim():
    module = fused_chain(32)
    result = optimize(module, TargetProfile.x86_avx512(), Theta.cool())
    result.steps = result.steps[:1]
    try:
        emit_kernel_ll(module, result)
    except NotImplementedError as exc:
        assert "found 2" in str(exc)
    else:
        raise AssertionError("partial realization hid an executable graph claim")


def test_unsupported_single_claim_is_rejected():
    _assert_rejected(histogram_gather(32))


def test_requested_unsupported_cli_lowering_exits_nonzero():
    assert run_main(["fused_chain", "--emit-llvm"]) == 1
