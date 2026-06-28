"""Lowering BCIR-4 -> BCIR-5: legal LLVM IR (AOT clang / JIT lli / WASM), plus the
generic stackify foundation for the stack-machine bytecode targets.

Imports are **lazy** (PEP 562): emitting a C kernel (`c_kernel` + `llvm`) does not
eagerly pull the JIT (`jit` -> `inspect`), the WASM/stack-machine path (`wasm`,
`stackify` -> `ast`), or the memory-ordering helpers. They resolve on first access.
`from bcir.lower.c_kernel import emit_kernel_c` and `from bcir.lower import jit_run`
both still work.
"""

from __future__ import annotations

import importlib
import importlib.util

_EXPORTS: dict[str, tuple[str, ...]] = {
    "jit": ("jit_run",),
    "llvm": ("compile_and_run", "emit_harness_c", "emit_kernel_ll"),
    "c_kernel": ("compile_and_run_c", "compile_and_run_compensated_c", "compile_and_run_qfixed_c",
                 "emit_compensated_reduce_c", "emit_compensated_selfcheck_c",
                 "emit_gather_kernel_c", "emit_header_c", "emit_kernel_c",
                 "emit_qfixed_kernel_c", "emit_qfixed_selfcheck_c", "emit_reduce_c",
                 "emit_selfcheck_c", "emit_strided_c", "find_reduce", "find_strided",
                 "SpatialBinding", "SpatialPlan", "optimize_spatial", "is_pim_target"),
    "inference": ("InferenceModel", "Layer", "emit_inference_kernel_c",
                  "compile_and_run_inference_c", "quantize_model"),
    "mlir": ("to_mlir", "plan_view", "PlanView", "ClaimView", "PathView"),
    "memory_model": ("barrier_fence_ir", "fence_ordering", "hazard_to_ordering",
                     "c_memory_order", "emit_ring_header_c"),
    "stackify": ("BinOp", "Const", "Local", "stackify", "to_cil", "to_jvm", "to_wasm"),
    "wasm": ("compile_to_wasm", "is_valid_wasm", "run_wasm_node"),
    "specialist": ("ShapeKey", "ShapeLedger", "SpecialistResult", "emit_specialist_c",
                   "specializable", "synthesize"),
}

_NAME_TO_MOD: dict[str, str] = {name: mod for mod, names in _EXPORTS.items() for name in names}

__all__ = sorted(_NAME_TO_MOD)

# `stackify` is both an exported function and a submodule name. Bind the function
# eagerly so the `stackify` submodule does not shadow it on later import (Python
# won't overwrite an existing parent attribute, but an unbound name would be).
from .stackify import stackify  # noqa: E402


def __getattr__(name: str):
    """Resolve an exported name (or a submodule) lazily on first access (PEP 562)."""
    mod = _NAME_TO_MOD.get(name)
    if mod is not None:
        value = getattr(importlib.import_module(f".{mod}", __name__), name)
        globals()[name] = value
        return value
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        submodule = importlib.import_module(f".{name}", __name__)
        globals()[name] = submodule
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
