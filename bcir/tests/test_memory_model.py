"""Phase 8: BCIR -> LLVM atomic-ordering memory model tests."""

from bcir.lower import barrier_fence_ir, fence_ordering, hazard_to_ordering
from bcir.lower.memory_model import HAZARD_ORDERING, LLVM_ORDERING


def test_hazard_contract_to_ordering():
    assert hazard_to_ordering("unique") == "monotonic"
    assert hazard_to_ordering("atomic") == "acq_rel"
    assert hazard_to_ordering("barriered") == "seq_cst"
    assert hazard_to_ordering("???") == "seq_cst"          # safe default
    assert set(HAZARD_ORDERING) == {"unique", "atomic", "barriered"}


def test_ordering_maps_one_to_one_to_llvm():
    for mnem in ("unordered", "monotonic", "acquire", "release", "acq_rel", "seq_cst"):
        assert LLVM_ORDERING[mnem] == mnem


def test_fence_ordering_clamps_to_legal():
    # Fences require >= acquire; unordered/monotonic clamp to seq_cst.
    assert fence_ordering("acquire") == "acquire"
    assert fence_ordering("release") == "release"
    assert fence_ordering("acq_rel") == "acq_rel"
    assert fence_ordering("seq_cst") == "seq_cst"
    assert fence_ordering("monotonic") == "seq_cst"
    assert fence_ordering("unordered") == "seq_cst"


def test_barrier_fence_ir_is_legal_llvm():
    assert barrier_fence_ir("acquire") == "fence acquire"
    assert barrier_fence_ir() == "fence seq_cst"
    assert barrier_fence_ir("monotonic") == "fence seq_cst"
