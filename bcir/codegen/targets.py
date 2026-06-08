"""Codegen target descriptors -- the data behind each bcir.target.lower_contract.

Adding a target is a data entry (the container stays open), not codegen code: a
triple, the `llc` knobs, the element type (eBPF is integer-only -- no FP), and how
to validate the artifact (object-format string for ELF targets, or an asm marker
for text targets like PTX).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CodegenTarget:
    name: str
    triple: str
    filetype: str            # "obj" (ELF) or "asm" (text, e.g. PTX/SPIR-V)
    object_format: str = ""  # expected llvm-objdump "file format" (obj targets)
    asm_marker: str = ""     # required substring in the asm (asm targets)
    elem: str = "f32"        # "f32" (float SIMD) or "i32" (eBPF: no FP)
    width: int | None = None  # force a vector/scalar width (eBPF: scalar)
    extra_flags: tuple = ()  # extra llc flags (e.g. -mcpu)
    note: str = ""


# The seeded targets. RISC-V uses the soft-float ABI by default; nvptx emits PTX;
# eBPF uses a scalar integer kernel; SPIR-V is best-effort (no backend in stock
# llc 18 -- codegen reports it cleanly when the target is unregistered).
CODEGEN_TARGETS = {
    "x86_64": CodegenTarget(
        "x86_64", "x86_64-unknown-linux-gnu", "obj", object_format="elf64-x86-64"),
    "aarch64": CodegenTarget(
        "aarch64", "aarch64-linux-gnu", "obj", object_format="elf64-littleaarch64",
        note="ARM cross-target"),
    "riscv64": CodegenTarget(
        "riscv64", "riscv64", "obj", object_format="elf64-littleriscv",
        note="RISC-V cross-target (soft-float ABI)"),
    "nvptx64": CodegenTarget(
        "nvptx64", "nvptx64-nvidia-cuda", "asm", asm_marker=".target",
        extra_flags=("-mcpu=sm_50",), note="NVIDIA GPU (PTX)"),
    "bpf": CodegenTarget(
        "bpf", "bpfel", "obj", object_format="elf64-bpf", elem="i32", width=1,
        note="eBPF (integer-only, scalar; verifier-friendly)"),
    "spirv64": CodegenTarget(
        "spirv64", "spirv64-unknown-unknown", "asm", asm_marker="OpEntryPoint",
        note="GPU SPIR-V (best-effort; needs a SPIR-V backend)"),
}
