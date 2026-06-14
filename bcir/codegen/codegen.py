"""Per-target codegen driver: BCIR -> LLVM IR -> object/asm via llc, + a C fallback."""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from shutil import which

from ..model import Module, Opcode
from ..kbcir.realize import RealizationResult
from ..lower.c_kernel import emit_kernel_c
from ..lower.llvm import _find_elementwise, emit_kernel_ll
from .targets import CODEGEN_TARGETS, CodegenTarget

_C_OP = {Opcode.ADD: "+", Opcode.SUB: "-", Opcode.MUL: "*"}


@dataclass
class CodegenResult:
    ok: bool
    target: str
    artifact: bytes | str | None
    message: str


def _tool(*names: str) -> str | None:
    for n in names:
        p = which(n)
        if p:
            return p
    return None


def codegen(module: Module, result: RealizationResult, target_name: str,
            fn_name: str = "bcir_kernel", workdir: str | None = None) -> CodegenResult:
    """Generate a real artifact for `target_name` (an object or asm)."""
    if target_name not in CODEGEN_TARGETS:
        return CodegenResult(False, target_name, None, f"unknown target {target_name!r}")
    tgt = CODEGEN_TARGETS[target_name]
    llc = _tool("llc", "llc-18")
    if llc is None:
        return CodegenResult(False, target_name, None, "llc not found")

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-codegen-")
    try:
        ll = os.path.join(workdir, "kernel.ll")
        out = os.path.join(workdir, "kernel.out")
        with open(ll, "w") as f:
            f.write(emit_kernel_ll(module, result, fn_name, elem=tgt.elem, width_override=tgt.width))

        cmd = [llc, f"-mtriple={tgt.triple}", *tgt.extra_flags]
        if tgt.filetype == "obj":
            cmd += ["-filetype=obj"]
        cmd += [ll, "-o", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return CodegenResult(False, target_name, None,
                                 f"llc failed for {tgt.triple}:\n{r.stderr.strip()}")

        if tgt.filetype == "obj":
            with open(out, "rb") as f:
                data = f.read()
            objdump = _tool("llvm-objdump", "llvm-objdump-18")
            fmt = ""
            if objdump:
                fmt = subprocess.run([objdump, "-f", out], capture_output=True, text=True).stdout
            ok = (not tgt.object_format) or (tgt.object_format in fmt)
            return CodegenResult(ok, target_name, data,
                                 f"{len(data)} bytes, format ok" if ok
                                 else f"unexpected object format (want {tgt.object_format})")
        else:
            with open(out, "r") as f:
                text = f.read()
            ok = (not tgt.asm_marker) or (tgt.asm_marker in text)
            return CodegenResult(ok, target_name, text,
                                 "asm ok" if ok else f"missing marker {tgt.asm_marker!r}")
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def emit_c_source(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel") -> str:
    """Portable C-source fallback -- the universal bootstrap target (any C23
    compiler). Delegates to the C kernel backend (`lower.c_kernel.emit_kernel_c`):
    the K_BCIR-selected lane width drives the loop, pointers are restrict-qualified,
    and a scalar tail keeps it bounds-safe (R12: `verify.verify_c_lowering`)."""
    return emit_kernel_c(module, result, fn_name)


def codegen_c(module: Module, result: RealizationResult, fn_name: str = "bcir_kernel",
              workdir: str | None = None) -> CodegenResult:
    """Emit the C fallback and compile it to an object (proves it builds anywhere)."""
    cc = _tool("clang", "cc", "gcc")
    if cc is None:
        return CodegenResult(False, "c", None, "no C compiler")
    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-cgen-c-")
    try:
        src = os.path.join(workdir, "kernel.c")
        obj = os.path.join(workdir, "kernel.o")
        with open(src, "w") as f:
            f.write(emit_c_source(module, result, fn_name))
        r = None
        for std in ("-std=c23", "-std=c2x"):   # C23 (clang) then c2x (gcc 13)
            r = subprocess.run([cc, std, "-O2", "-c", src, "-o", obj],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
        ok = r is not None and r.returncode == 0 and os.path.exists(obj)
        data = open(obj, "rb").read() if ok else None
        return CodegenResult(ok, "c", data, "compiled" if ok else (r.stderr.strip() if r else "no compiler"))
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def codegen_all(module: Module, result: RealizationResult) -> dict:
    """Run codegen for every registered target + the C fallback (skips on tool/backend gaps)."""
    out = {name: codegen(module, result, name) for name in CODEGEN_TARGETS}
    out["c"] = codegen_c(module, result)
    return out
