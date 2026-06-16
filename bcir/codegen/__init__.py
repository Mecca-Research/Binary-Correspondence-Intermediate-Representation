"""Per-target codegen (Phase 9): the runnable side of bcir.target.lower_contract.

BCIR -> LLVM IR -> per-target object/asm via `llc` (ARM/RISC-V/GPU-PTX/eBPF), plus
a portable C-source fallback. Each `CodegenTarget` is the oracle realization of a
`bcir.target.lower_contract`: it picks the triple, the element type (eBPF has no FP),
and how to validate the produced artifact.
"""

from .codegen import (
    CodegenResult,
    codegen,
    codegen_all,
    codegen_c,
    codegen_object_c,
    emit_c_source,
)
from .targets import CODEGEN_TARGETS, CodegenTarget

__all__ = [
    "CodegenResult", "codegen", "codegen_all", "codegen_c", "codegen_object_c",
    "emit_c_source", "CODEGEN_TARGETS", "CodegenTarget",
]
