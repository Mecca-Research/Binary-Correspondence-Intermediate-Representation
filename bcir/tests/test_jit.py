"""CT5: in-process JIT execution of the lowered kernel via lli."""

from shutil import which

from bcir.examples import vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.lower import jit_run


def test_jit_runs_and_self_checks():
    # Same StreamPack lowering, run in-process via lli (vs the AOT clang path).
    if not all(which(t) for t in ("clang", "lli", "llvm-link")) and \
       not all(which(t) for t in ("clang", "lli-18", "llvm-link-18")):
        return  # skip cleanly when the JIT toolchain is absent
    m = vector_add(1024)
    res = optimize(m, TargetProfile.x86_avx512(), Theta.cool())
    ok, out = jit_run(m, res, fn_name="bcir_kernel")
    assert ok, out
    assert "OK" in out
