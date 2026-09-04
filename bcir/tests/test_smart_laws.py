"""R14-R16 as bcir.verify functions -- dual-rail symmetry with the MLIR
-bcir-lower-to-llvm laws (CIM dispatch / DVFS clock / allocator tier). Each is
negative-tested per law, mirroring the gem_passes_neg.mlir discipline on the oracle
rail."""

import dataclasses

from bcir.examples import gather_reduce, vector_add
from bcir.gem import hydrate
from bcir.gem.cim import annotate_cim
from bcir.gem.dvfs import DVFSDecision, DVFSPlan, plan_dvfs
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.allocator import Placement
from bcir.kbcir.cost import MemTier, Theta, TargetProfile
from bcir.verify import verify_allocator, verify_cim, verify_dvfs, verify_smart_lowering

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


# --- R14: CIM/PIM dispatch legality ----------------------------------------------


def test_r14_pim_on_a_reduction_is_legal():
    pim = dataclasses.replace(TargetProfile.x86_avx512(), isa_features=frozenset({"pim"}))
    m = gather_reduce(1 << 20)
    pack, decisions = annotate_cim(m, optimize(m, pim, COOL), pim)
    assert any(d.offload for d in decisions)  # the reduction did offload
    assert verify_cim(pack) == []  # ... and that is legal (R14 clean)


def test_r14_pim_on_a_non_reduction_is_rejected():
    m = vector_add(1024)
    pack = hydrate(m, optimize(m, AVX, COOL))
    pack.segments = [dataclasses.replace(pack.segments[0], dispatch="pim")]  # add + pim: illegal
    diags = verify_cim(pack)
    assert [d.law for d in diags] == ["R14"]


# --- R15: DVFS clock legality ----------------------------------------------------


def test_r15_a_real_dvfs_plan_is_legal():
    m = vector_add(1 << 16)
    assert verify_dvfs(plan_dvfs(m, optimize(m, AVX, COOL), COOL)) == []


def test_r15_out_of_range_clock_is_rejected():
    plan = DVFSPlan(decisions=(DVFSDecision(0, 0, 0, "compute", 900, "x"),))
    assert [d.law for d in verify_dvfs(plan)] == ["R15"]


def test_r15_memory_bound_overclock_is_rejected():
    plan = DVFSPlan(decisions=(DVFSDecision(0, 0, 0, "memory", 320, "x"),))
    assert [d.law for d in verify_dvfs(plan)] == ["R15"]


# --- R16: allocator placement legality -------------------------------------------


def test_r16_a_fitting_placement_is_legal():
    # 1024 * 4 B = 4 KiB fits the 64 KiB L1 cap.
    m = vector_add(1024)
    assert verify_allocator(m, Placement(tiers={10: MemTier.L1}, moved=(10,))) == []


def test_r16_oversized_l1_placement_is_rejected():
    # 1<<20 floats = 4 MiB > the 64 KiB L1 cap.
    m = vector_add(1 << 20)
    diags = verify_allocator(m, Placement(tiers={10: MemTier.L1}, moved=(10,)))
    assert [d.law for d in diags] == ["R16"]


def test_r16_l2_cap_is_four_mib():
    m = vector_add(1 << 20)  # 4 MiB: == the L2 cap (legal), > L1 (illegal)
    assert verify_allocator(m, Placement(tiers={10: MemTier.L2}, moved=(10,))) == []
    assert [d.law for d in verify_allocator(m, Placement(tiers={10: MemTier.L1}, moved=(10,)))] == [
        "R16"
    ]


def test_r16_offchip_tiers_have_no_cap():
    m = vector_add(1 << 24)  # huge, but DRAM/HBM carry no static cap
    assert verify_allocator(m, Placement(tiers={10: MemTier.DRAM}, moved=())) == []


# --- the aggregator --------------------------------------------------------------


def test_verify_smart_lowering_runs_the_provided_laws():
    m = vector_add(1024)
    pack = hydrate(m, optimize(m, AVX, COOL))
    plan = plan_dvfs(m, optimize(m, AVX, COOL), COOL)
    placement = Placement(tiers={10: MemTier.L1}, moved=(10,))
    assert verify_smart_lowering(m, pack=pack, dvfs_plan=plan, placement=placement) == []
    # an injected R14 violation surfaces through the aggregator.
    pack.segments = [dataclasses.replace(pack.segments[0], dispatch="pim")]
    assert [d.law for d in verify_smart_lowering(m, pack=pack)] == ["R14"]
