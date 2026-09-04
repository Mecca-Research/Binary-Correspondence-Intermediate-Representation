"""LLVM lowering tests: the emitted kernel assembles and runs+self-checks."""

import shutil

from bcir.examples import fused_chain, histogram_gather, scan_chain, vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import compile_and_run, emit_kernel_ll
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
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "k.ll")
        with open(path, "w") as f:
            f.write(ll)
        out = subprocess.run(
            ["llvm-as", path, "-o", os.path.join(d, "k.bc")], capture_output=True, text=True
        )
        assert out.returncode == 0, out.stderr


def test_lowering_compiles_and_runs():
    # End-to-end Milestones 5-7 in miniature: build with clang and self-check.
    if shutil.which("clang") is None:
        return  # skip cleanly when clang is absent
    module, res = _result()
    ok, out = compile_and_run(module, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK" in out


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
