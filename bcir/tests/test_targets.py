"""Target-open container + memory-hierarchy tests (CT1)."""

from bcir.examples import histogram_gather, vector_add, vector_add_hbm
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import HProfile, TargetProfile, Theta


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
