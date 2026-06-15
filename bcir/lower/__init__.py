"""Lowering BCIR-4 -> BCIR-5: legal LLVM IR (AOT clang / JIT lli / WASM), plus the
generic stackify foundation for the stack-machine bytecode targets."""

from .jit import jit_run
from .llvm import compile_and_run, emit_harness_c, emit_kernel_ll
from .c_kernel import (
    compile_and_run_c,
    emit_gather_kernel_c,
    emit_header_c,
    emit_kernel_c,
    emit_reduce_c,
    emit_selfcheck_c,
    emit_strided_c,
    find_reduce,
    find_strided,
)
from .memory_model import barrier_fence_ir, fence_ordering, hazard_to_ordering
from .stackify import BinOp, Const, Local, stackify, to_cil, to_jvm, to_wasm
from .wasm import compile_to_wasm, is_valid_wasm, run_wasm_node

__all__ = [
    "compile_and_run", "emit_harness_c", "emit_kernel_ll", "jit_run",
    "compile_and_run_c", "emit_gather_kernel_c", "emit_header_c", "emit_kernel_c",
    "emit_reduce_c", "emit_selfcheck_c", "emit_strided_c", "find_reduce", "find_strided",
    "compile_to_wasm", "run_wasm_node", "is_valid_wasm",
    "stackify", "to_wasm", "to_jvm", "to_cil", "Const", "Local", "BinOp",
    "hazard_to_ordering", "fence_ordering", "barrier_fence_ir",
]
