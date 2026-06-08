"""Phase 8: the formal BCIR -> LLVM atomic-ordering memory model.

BCIR hazard contracts (LangRef R5) and explicit memory orderings map onto LLVM's
atomic memory orderings. This module is the normative mapping (mirrored by the
`BCIR_MemOrdering` enum in BCIRAttrs.td and the barrier lowering in BCIRPasses.cpp):

  - a claim's *hazard mode* implies the ordering its accesses need, and
  - an explicit *ordering* mnemonic maps 1:1 to an LLVM ordering, clamped to the
    fence-legal set (>= acquire) when emitted as a `fence`.
"""

from __future__ import annotations

# BCIR hazard contract -> the LLVM ordering its accesses require.
HAZARD_ORDERING = {
    "unique": "monotonic",     # no inter-lane sharing; atomicity only, no fences
    "atomic": "acq_rel",       # atomic read-modify-write: acquire + release
    "barriered": "seq_cst",    # an explicit full barrier
}

# BCIR MemOrdering mnemonic -> LLVM ordering (identity; matches BCIRAttrs.td).
LLVM_ORDERING = {
    "unordered": "unordered",
    "monotonic": "monotonic",
    "acquire": "acquire",
    "release": "release",
    "acq_rel": "acq_rel",
    "seq_cst": "seq_cst",
}

# Orderings legal on an LLVM `fence` (must be >= acquire).
_FENCE_LEGAL = {"acquire", "release", "acq_rel", "seq_cst"}


def hazard_to_ordering(hazard_mode: str) -> str:
    """The LLVM ordering implied by a BCIR hazard contract."""
    return HAZARD_ORDERING.get(hazard_mode, "seq_cst")


def fence_ordering(ordering: str = "seq_cst") -> str:
    """Map a BCIR ordering to a fence-legal LLVM ordering (clamps to seq_cst)."""
    o = LLVM_ORDERING.get(ordering, "seq_cst")
    return o if o in _FENCE_LEGAL else "seq_cst"


def barrier_fence_ir(ordering: str = "seq_cst") -> str:
    """The legal LLVM `fence` instruction for a BCIR barrier of the given ordering."""
    return f"fence {fence_ordering(ordering)}"
