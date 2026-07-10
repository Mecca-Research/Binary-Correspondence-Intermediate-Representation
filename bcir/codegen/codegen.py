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
from ..toolchain import resolve_llvm_tools
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
    requested = ("llc", "llvm-objdump") if tgt.filetype == "obj" else ("llc",)
    llvm = resolve_llvm_tools(*requested, pipeline=f"{target_name} codegen")
    if not llvm.ok:
        return CodegenResult(False, target_name, None, llvm.message)
    llc = llvm.paths["llc"]

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
            objdump = llvm.paths["llvm-objdump"]
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


# --- the native-object decision gate: one target end-to-end via the resident cc ----
#
# docs/BCIR_NATIVE_OBJECT_GATE.md: BCIR-native instruction selection (a hand-rolled
# ELF/relocation emitter) is DEFERRED -- the warranted path is to emit the kernel and
# let the *resident compiler* finish it to a real object. This closes that loop for one
# target end-to-end (eBPF integer-scalar, or x86-64 scalar) by compiling the emitted C
# straight to a relocatable object with `clang --target=...`, then proving the bytes are
# a genuine ELF object for the expected machine. eBPF is integer-only (no FP), so it
# takes the i32 kernel and `-ffreestanding` (no libc). EM_BPF = 247, EM_X86_64 = 62.
_OBJECT_TARGETS: dict[str, dict] = {
    "bpf":     {"triple": "bpf", "e_machine": 247, "elem": "i32", "freestanding": True},
    "x86_64":  {"triple": "x86_64-linux-gnu", "e_machine": 62, "elem": "i32", "freestanding": False},
    # EM_AARCH64 = 183: the ARM (Raspberry Pi 5) native-object target, alongside x86_64.
    "aarch64": {"triple": "aarch64-linux-gnu", "e_machine": 183, "elem": "i32", "freestanding": False},
}


def _elf_machine(obj: bytes) -> int | None:
    """The ELF e_machine of a relocatable object, or None if it is not ELF."""
    import struct
    if len(obj) < 20 or obj[:4] != b"\x7fELF":
        return None
    endi = "<" if obj[5] == 1 else ">"     # EI_DATA: 1 = little-endian
    return struct.unpack_from(endi + "H", obj, 18)[0]


def codegen_object_c(module: Module, result: RealizationResult, target: str = "bpf",
                     fn_name: str = "bcir_kernel", workdir: str | None = None) -> CodegenResult:
    """Compile the emitted C kernel to a REAL native object for `target` via the
    resident compiler -- the warranted slice of the native-object decision gate (no
    BCIR-native isel). Returns the object bytes; `ok` iff it is an ELF object for the
    target's expected machine (eBPF EM_BPF=247 / x86-64 EM_X86_64=62)."""
    if target not in _OBJECT_TARGETS:
        return CodegenResult(False, target, None, f"unknown object target {target!r}")
    spec = _OBJECT_TARGETS[target]
    cc = _tool("clang")  # need --target= cross support (clang, not cc/gcc)
    if cc is None:
        return CodegenResult(False, target, None, "clang not found (needed for --target)")
    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-obj-")
    try:
        src = os.path.join(workdir, "kernel.c")
        obj = os.path.join(workdir, "kernel.o")
        with open(src, "w") as f:
            f.write(emit_kernel_c(module, result, fn_name, elem=spec["elem"]))
        cmd = [cc, f"--target={spec['triple']}", "-O2", "-std=c23", "-c", src, "-o", obj]
        if spec["freestanding"]:
            cmd.insert(1, "-ffreestanding")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(obj):
            return CodegenResult(False, target, None,
                                 f"clang --target={spec['triple']} failed:\n{r.stderr.strip()}")
        data = open(obj, "rb").read()
        mach = _elf_machine(data)
        ok = mach == spec["e_machine"]
        return CodegenResult(ok, target, data,
                             f"{len(data)}-byte ELF object, e_machine={mach}" if ok
                             else f"unexpected object (e_machine={mach}, want {spec['e_machine']})")
    finally:
        if created:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)
