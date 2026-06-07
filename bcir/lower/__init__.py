"""Lowering BCIR-4 -> BCIR-5: legal LLVM IR, run AOT (clang) or JIT (lli)."""

from .jit import jit_run
from .llvm import compile_and_run, emit_harness_c, emit_kernel_ll

__all__ = ["compile_and_run", "emit_harness_c", "emit_kernel_ll", "jit_run"]
