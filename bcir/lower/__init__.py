"""Lowering BCIR-4 -> BCIR-5: emit legal LLVM IR that compiles+runs via clang."""

from .llvm import compile_and_run, emit_harness_c, emit_kernel_ll

__all__ = ["compile_and_run", "emit_harness_c", "emit_kernel_ll"]
