"""CT5: in-process JIT execution of the lowered kernel via lli."""

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import jit_run
from bcir.toolchain import resolve_llvm_tools


def test_jit_runs_and_self_checks():
    # Same StreamPack lowering, run in-process via lli (vs the AOT clang path).
    if not resolve_llvm_tools("clang", "llvm-link", "lli", pipeline="JIT test").ok:
        return  # skip cleanly when the JIT toolchain is absent
    m = vector_add(1024)
    res = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    ok, out = jit_run(m, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK" in out
