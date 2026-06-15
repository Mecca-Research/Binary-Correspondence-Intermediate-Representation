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


# --- the C side of the zero-copy telemetry ring (atomic stores; no syscall) -------
#
# The Python `telemetry.TelemetryRing` was a single-process model. This emits the
# *C producer* for a real shared-memory ring: the kernel atomic-stores cycle counts
# and milestones to fixed offsets, and a Python reader mmaps the SAME region and
# samples it via pointer arithmetic -- no serialization, no syscall per sample, no
# stdio. Layout (little-endian, matches `telemetry.TelemetryRing._FMT` = 7 x i64):
#
#   [ 0] _Atomic uint64_t head        records written so far (monotonic publish ctr)
#   [ 8] uint64_t        capacity     record slots
#   [16] uint64_t        record_size  = 56 (7 x int64)
#   [24] uint64_t        reserved
#   [32] record[capacity]             each: claim_id, cycles, bytes, misses,
#                                            thermal, voltage, utilization (i64 LE)
#
# `head` is published with RELEASE ordering (`hazard_to_ordering("atomic")` -> acq_rel,
# narrowed to release for a store) so a reader that acquire-loads `head` sees a fully
# written record -- the same hazard->ordering law the rest of the lowering uses.

RING_HEADER_BYTES = 32
RING_RECORD_BYTES = 56          # 7 x int64, == struct.calcsize("<7q")
RING_FIELDS = ("claim_id", "cycles", "bytes", "misses", "thermal", "voltage", "utilization")

# LLVM ordering -> C11 stdatomic memory_order_*.
_C_MEMORDER = {
    "unordered": "memory_order_relaxed", "monotonic": "memory_order_relaxed",
    "acquire": "memory_order_acquire", "release": "memory_order_release",
    "acq_rel": "memory_order_acq_rel", "seq_cst": "memory_order_seq_cst",
}


def c_memory_order(ordering: str) -> str:
    """Map an LLVM/BCIR ordering to a C11 `memory_order_*` (for emitted atomics)."""
    return _C_MEMORDER.get(ordering, "memory_order_seq_cst")


def emit_ring_header_c(fn_prefix: str = "bcir_ring") -> str:
    """Emit the freestanding C for the shared-memory telemetry ring producer: an
    `init`, a `write` that atomic-publishes one record, and the layout constants. A
    Python reader (`telemetry.parse_shared_ring`) mmaps the same region and reads it
    with no syscall/serialization. The publish store uses the release ordering implied
    by the `atomic` hazard contract (`hazard_to_ordering`)."""
    # The producer publishes `head` with a RELEASE store (the release half of the
    # `atomic` hazard's acq_rel contract; the consumer does the acquire load), so a
    # reader that sees the bumped head sees the fully written record. A pure store
    # cannot carry acquire, hence release rather than acq_rel.
    assert hazard_to_ordering("atomic") == "acq_rel"
    publish = c_memory_order("release")
    fields = ", ".join(f"int64_t {f}" for f in RING_FIELDS)
    stores = "\n".join(f"  rec[{i}] = {f};" for i, f in enumerate(RING_FIELDS))
    return (
        "/* BCIR zero-copy telemetry ring (C producer; mmap-shared with Python). */\n"
        "#include <stddef.h>\n#include <stdint.h>\n#include <stdatomic.h>\n\n"
        f"enum {{ {fn_prefix.upper()}_HEADER = {RING_HEADER_BYTES}, "
        f"{fn_prefix.upper()}_RECORD = {RING_RECORD_BYTES} }};\n\n"
        f"void {fn_prefix}_init(void *base, uint64_t capacity) {{\n"
        "  uint64_t *h = (uint64_t *)base;\n"
        "  atomic_store_explicit((_Atomic uint64_t *)&h[0], 0u, memory_order_relaxed);\n"
        f"  h[1] = capacity; h[2] = {RING_RECORD_BYTES}u; h[3] = 0u;\n"
        "}\n\n"
        f"void {fn_prefix}_write(void *base, {fields}) {{\n"
        "  _Atomic uint64_t *head = (_Atomic uint64_t *)base;\n"
        "  uint64_t cap = ((uint64_t *)base)[1];\n"
        "  uint64_t h = atomic_load_explicit(head, memory_order_relaxed);\n"
        f"  int64_t *rec = (int64_t *)((char *)base + {RING_HEADER_BYTES}"
        f" + (h %% cap) * {RING_RECORD_BYTES});\n".replace("%%", "%")
        + stores + "\n"
        f"  atomic_store_explicit(head, h + 1u, {publish});  /* publish after record */\n"
        "}\n"
    )
