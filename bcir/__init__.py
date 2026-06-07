"""BCIR — Binary Correspondence Intermediate Representation.

This package is the **executable conformance oracle** for BCIR (per
docs/BCIR_LANGREF.md). It realizes the central equation

    K_BCIR(G | H, Theta) = min_pi  sum_i  T_i (X) f_i(pi)  =  min_pi C_H(pi, Theta)

as a runnable optimizer: BCIR is the goal graph G; the K_BCIR cost engine scores
realization paths pi over an integer cost algebra under a substrate profile H and
live runtime state Theta; the GEM layer hydrates the selected plan into a
StreamPack; the lowering layer emits legal LLVM IR that compiles+runs via clang.

The MLIR dialect family under ``mlir/`` is the *law* (TableGen/ODS + an IRDL
projection); this package must agree with it (see docs/PARITY.md). The package is
deterministic: all hot-path math is integer/fixed-point.
"""

from .model import (
    Claim,
    Domain,
    Lane,
    Module,
    Opcode,
    Phase,
    Resource,
    StrideClass,
)

__all__ = [
    "Claim",
    "Domain",
    "Lane",
    "Module",
    "Opcode",
    "Phase",
    "Resource",
    "StrideClass",
]

__version__ = "0.2.0"
