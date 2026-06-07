"""Lane, stride-class, and memory-domain taxonomies.

Integer values are normative and MUST match the MLIR enum attributes in
``mlir/include/BCIR/BCIRAttrs.td`` exactly (see docs/PARITY.md).
"""

from __future__ import annotations

from enum import IntEnum


class Lane(IntEnum):
    """Execution-geometry lane (LangRef Sec. 7) — not a vector hint."""

    U = 0      # unit / constant stride (affine or stride-proven)
    UX = 1     # cacheline-local indexed
    T = 2      # tile (e.g. 16x16)
    GGG = 3    # full gather/scatter (always legal, must be minimized)
    A = 4      # atomic
    H = 5      # hazard / provenance / control (barriers, fences)


class StrideClass(IntEnum):
    """Access-pattern shape carried by a claim."""

    SCALAR = 0
    UNIT = 1
    STRIDED = 2
    CACHELINE = 3
    TILE = 4
    RANDOM = 5


class Domain(IntEnum):
    """Memory domain. Mapped onto a `MemoryHierarchy` tier in the cost model."""

    RAM = 0
    VRAM = 1
    NVM = 2
    MMIO = 3
    CXL = 4
    HBM = 5
