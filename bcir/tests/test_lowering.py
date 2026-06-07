"""LLVM lowering tests: the emitted kernel assembles and runs+self-checks."""

import shutil

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import compile_and_run, emit_kernel_ll


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
        out = subprocess.run(["llvm-as", path, "-o", os.path.join(d, "k.bc")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr


def test_lowering_compiles_and_runs():
    # End-to-end Milestones 5-7 in miniature: build with clang and self-check.
    if shutil.which("clang") is None:
        return  # skip cleanly when clang is absent
    module, res = _result()
    ok, out = compile_and_run(module, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK" in out
