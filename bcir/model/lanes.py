"""Lane, stride-class, and memory-domain taxonomies.

Integer values are normative and MUST match the MLIR enum attributes in
``mlir/include/BCIR/BCIRAttrs.td`` exactly (see docs/PARITY.md).
"""

from __future__ import annotations

from enum import IntEnum


class Lane(IntEnum):
    """Execution-geometry lane (LangRef Sec. 7) — not a vector hint."""

    U = 0  # unit / constant stride (affine or stride-proven)
    UX = 1  # cacheline-local indexed
    T = 2  # tile (e.g. 16x16)
    GGG = 3  # full gather/scatter (always legal, must be minimized)
    A = 4  # atomic
    H = 5  # hazard / provenance / control (barriers, fences)


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


#: The device-ISOLATED domains (LangRef R3): an MMIO register is a distinct address space the
#: host cannot transparently substitute for a memory tier. RAM/HBM/VRAM/CXL/NVM are the
#: tiers a claim may stage across (`kbcir.device_manifest._MEM_TIERS`; the HAM fabric reads
#: NVM into VRAM by design, and the law rail's "NVM cell" isolation had no fixture -- the
#: structural corpus surfaced the divergence and the tier model won). A resource in an
#: isolated domain may be touched only by a claim that declares that domain; such a claim
#: may still carry host-memory operands (the value an MMIO write stores, the index a
#: register read uses) -- the shape every cfront MMIO access has. One predicate, both rails
#: (`BCIRPassSupport.h` `isIsolatedDomain`).
ISOLATED_DOMAINS: frozenset = frozenset({Domain.MMIO})
