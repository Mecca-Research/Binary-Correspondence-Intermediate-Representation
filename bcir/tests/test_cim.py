"""Compute-In-Memory awareness: dispatch large reductions to the memory controller.

A large reduction is offloaded to PIM (the bytes-not-moved beat PIM's compute
surcharge); a small one stays on the core; non-reductions are never offloaded.
Gains-only and deterministic."""

from bcir.gem import annotate_cim, cim_decision
from bcir.gem.cim import _PIM_COMPUTE_Q8
from bcir.examples import gather_reduce, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta

AVX = TARGETS["x86_avx512"]


def test_large_reduction_offloads_to_pim():
    d = cim_decision("reduce.gather", count=1 << 20, elem_bytes=4, h=AVX)
    assert d.offload and d.pim_cost < d.core_cost and d.saved > 0


def test_tiny_reduction_stays_on_the_core():
    # at small counts the 1-element result transfer + surcharge does not pay off.
    d = cim_decision("reduce.gather", count=2, elem_bytes=4, h=AVX)
    assert not d.offload and d.saved == 0


def test_non_reduction_is_never_offloaded():
    d = cim_decision("vector.add", count=1 << 20, elem_bytes=4, h=AVX)
    assert not d.offload


def test_annotate_marks_the_reduction_segment_pim():
    m = gather_reduce(1 << 20)
    r = optimize(m, AVX, Theta.cool())
    pack, decisions = annotate_cim(m, r, AVX)
    red = [s for s in pack.segments if s.opcode.startswith("reduce.")]
    assert red and all(s.dispatch == "pim" for s in red)  # offloaded
    assert any(d.offload and d.saved > 0 for d in decisions)


def test_annotation_leaves_elementwise_on_the_core():
    m = vector_add(1 << 20)
    r = optimize(m, AVX, Theta.cool())
    pack, decisions = annotate_cim(m, r, AVX)
    assert all(s.dispatch == "core" for s in pack.segments)  # nothing offloaded
    assert decisions == []  # no reduction to consider


def test_offload_threshold_tracks_the_surcharge_and_is_deterministic():
    assert _PIM_COMPUTE_Q8 > 256  # PIM compute is dearer
    a = cim_decision("reduce.gather", count=4096, elem_bytes=4, h=AVX)
    b = cim_decision("reduce.gather", count=4096, elem_bytes=4, h=AVX)
    assert a == b  # deterministic


def test_mlir_cim_dvfs_constants_are_in_sync():
    """Pins the constants -bcir-cim / -bcir-dvfs annotate in mlir/test/passes/{cim,dvfs}.mlir,
    so the law-rail FileCheck cannot silently rot from the oracle."""
    # CIM: a reduction over 4096 elems offloads (core 20480 > pim 10244); 1024 does not.
    big = cim_decision("reduce.add", 4096, 4, AVX)
    assert big.offload and big.core_cost == 20480 and big.pim_cost == 10244
    small = cim_decision("reduce.add", 1024, 4, AVX)
    assert not small.offload
    # DVFS: the vector_add phase is bandwidth-bound -> downclock 192.
    from bcir.gem.dvfs import classify, clock_for, phase_totals

    res = optimize(vector_add(1024), AVX, Theta.cool())
    ((compute, memory),) = phase_totals(vector_add(1024), res).values()
    assert (compute, memory) == (64, 3840)
    klass = classify(compute, memory)
    assert klass == "memory" and clock_for(klass, Theta.cool())[0] == 192
