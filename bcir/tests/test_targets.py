"""Target-open container + memory-hierarchy tests (CT1)."""

from bcir.examples import histogram_gather, vector_add, vector_add_hbm
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import HProfile, MemoryHierarchy, MemTier, TargetProfile, Theta
from bcir.model import Domain


def test_every_target_builds_and_has_widths():
    for name, h in TARGETS.items():
        assert h.widths()  # non-empty
        assert h.vector_width == max(h.widths())


def test_hprofile_is_back_compat_alias():
    assert HProfile is TargetProfile
    assert HProfile.x86_avx2().name == "x86-64-avx2"


def test_pi_star_differs_by_target():
    # The same vector_add graph realizes at a different lane width per target.
    expect = {"x86_avx512": 16, "x86_avx2": 8, "arm64_neon": 4, "nvidia_ptx": 32}
    for name, width in expect.items():
        res = optimize(vector_add(1024), TARGETS[name], Theta.cool())
        assert res.by_claim()[1000].width == width, f"{name} -> {res.by_claim()[1000].width}"


def test_hbm_is_cheaper_than_dram():
    h = TargetProfile.x86_avx512()
    dram = optimize(vector_add(1024), h, Theta.cool()).score
    hbm = optimize(vector_add_hbm(1024), h, Theta.cool()).score
    assert hbm < dram, f"HBM {hbm} should be cheaper than DRAM {dram}"


def test_ham_access_beats_flat_gather():
    h = TargetProfile.x86_avx2()
    flat = optimize(histogram_gather(1024, ham=False), h, Theta.cool()).score
    ham = optimize(histogram_gather(1024, ham=True), h, Theta.cool()).score
    assert ham < flat, f"HAM O(log n) {ham} should beat flat gather {flat}"


def test_memtier_enum_matches_the_mlir_law():
    # PARITY.md: MemTier values are normative and identical to BCIRAttrs.td.
    assert [(t.name, int(t)) for t in MemTier] == [
        ("L1", 0), ("L2", 1), ("L3", 2), ("DRAM", 3), ("HBM", 4), ("CXL", 5), ("SSD", 6)]
    hierarchy = MemoryHierarchy.default()
    for t in MemTier:
        assert hierarchy.by_name(t.name).mem_tier is t
    assert hierarchy.tier_for(Domain.RAM).mem_tier is MemTier.DRAM
    assert hierarchy.tier_for(Domain.HBM).mem_tier is MemTier.HBM
    assert hierarchy.tier_for(Domain.NVM).mem_tier is MemTier.SSD
