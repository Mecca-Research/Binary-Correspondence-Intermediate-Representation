"""CT5: in-process JIT execution of the lowered kernel (WASM-like AOT+JIT).

AOT (`compile_and_run`) builds a native binary with clang. The JIT path reuses
the *same* StreamPack lowering: it emits the kernel + the C self-check harness,
compiles them to LLVM IR, links them, and runs the linked module in-process with
`lli`. One portable artifact, two backends (AOT and JIT), per target.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from shutil import which

from ..model import Module
from ..kbcir.realize import RealizationResult
from .llvm import emit_harness_c, emit_kernel_ll


def _tool(*names: str) -> str | None:
    for n in names:
        p = which(n)
        if p:
            return p
    return None


def jit_run(
    module: Module,
    result: RealizationResult,
    fn_name: str = "bcir_kernel",
    workdir: str | None = None,
) -> tuple[bool, str]:
    """JIT-execute the lowered kernel via lli; returns (ok, output).

    ok is True only if the linked module ran under lli and the C harness printed
    OK with exit code 0. Skips (returns False, reason) if a tool is missing.
    """
    clang = _tool("clang")
    llvm_link = _tool("llvm-link", "llvm-link-18")
    lli = _tool("lli", "lli-18")
    missing = [n for n, t in (("clang", clang), ("llvm-link", llvm_link), ("lli", lli)) if t is None]
    if missing:
        return False, f"missing tool(s) for JIT: {', '.join(missing)}"

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-jit-")
    try:
        kll = os.path.join(workdir, "kernel.ll")
        hc = os.path.join(workdir, "harness.c")
        hll = os.path.join(workdir, "harness.ll")
        linked = os.path.join(workdir, "linked.ll")
        with open(kll, "w") as f:
            f.write(emit_kernel_ll(module, result, fn_name))
        with open(hc, "w") as f:
            f.write(emit_harness_c(module, result, fn_name))

        steps = [
            [clang, "-O0", "-emit-llvm", "-S", hc, "-o", hll],
            [llvm_link, hll, kll, "-S", "-o", linked],
        ]
        for cmd in steps:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                return False, f"JIT prep failed ({cmd[0]}):\n{r.stdout}{r.stderr}"
        run = subprocess.run([lli, linked], capture_output=True, text=True)
        ok = run.returncode == 0 and "OK" in run.stdout
        return ok, run.stdout + run.stderr
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
