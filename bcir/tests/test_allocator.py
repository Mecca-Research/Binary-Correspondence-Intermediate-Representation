"""The RL allocator (smart malloc): intent-aware placement, gains-only.

Hot ephemeral resources are promoted to the fast tier; cold/large ones go to the
bandwidth tier or stay put; a placement that wins nothing is a no-op (it never
slows the simple case). Deterministic, integer, opt-in."""

from bcir.kbcir import TARGETS
from bcir.kbcir.allocator import (
    DEFAULT_PLACER,
    FrozenPlacer,
    Placement,
    ResourceProfile,
    place,
    profile_resources,
    train_placer,
)
from bcir.kbcir.cost import MemTier
from bcir.model import Claim, Domain, Module, Opcode, Phase, Resource

AVX = TARGETS["x86_avx512"]


def _claim(cid, rd, wr, phase, n=64):
    return Claim(
        id=cid,
        opcode=Opcode.ADD,
        count=n,
        rd=tuple(rd),
        wr=tuple(wr),
        op="vector.add",
        domain=Domain.RAM,
    )


def _module():
    """R10: small, hit by 3 claims in one phase (hot ephemeral). R11: 4 MB, touched
    once in phase 0 and once in phase 3 (cold, durable, low frequency)."""
    m = Module(name="alloc")
    m.add_resource(Resource(rid=10, domain=Domain.RAM, shape=(64,)))  # 256 B
    m.add_resource(Resource(rid=11, domain=Domain.RAM, shape=(1 << 22,)))  # 16 MB
    m.add_resource(Resource(rid=12, domain=Domain.RAM, shape=(64,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                _claim(1, [10], [12], 0),
                _claim(2, [12], [10], 0),
                _claim(3, [10, 11], [12], 0),
            ],
        )
    )
    m.add_phase(Phase(phase_id=1, deps=(0,), claims=[_claim(4, [12], [12], 1)]))
    m.add_phase(Phase(phase_id=2, deps=(1,), claims=[_claim(5, [12], [12], 2)]))
    m.add_phase(Phase(phase_id=3, deps=(2,), claims=[_claim(6, [11], [12], 3)]))
    return m


# --- profiling: intent derived from the claim graph ------------------------------


def test_profiles_capture_lifespan_frequency_and_size():
    p = profile_resources(_module())
    assert p[10].ephemeral and p[10].lifespan == 1 and p[10].frequency == 3  # hot ephemeral
    assert not p[11].ephemeral and p[11].lifespan == 4  # durable (phase 0..3)
    assert p[11].size_bytes == (1 << 22) * 4  # 16 MB


# --- placement: hot -> fast tier, cold/large -> bandwidth tier --------------------


def test_hot_ephemeral_is_promoted_to_a_fast_tier():
    pl = place(_module(), AVX)
    assert pl.tier_of(10) in (MemTier.L1, MemTier.L2, MemTier.L3)  # promoted off DRAM
    assert 10 in pl.moved and "hot/ephemeral" in pl.rationale[10]


def test_cold_large_resource_goes_to_the_bandwidth_tier_not_sram():
    pl = place(_module(), AVX)
    assert pl.tier_of(11) == MemTier.HBM  # too big for SRAM -> bandwidth tier
    assert pl.tier_of(11) != MemTier.L1


def test_placement_is_deterministic():
    a, b = place(_module(), AVX), place(_module(), AVX)
    assert a.tiers == b.tiers and a.moved == b.moved


# --- gains-only: no relocation when nothing wins ---------------------------------


def test_no_gain_is_a_noop():
    # A single large resource already in the bandwidth tier (HBM domain): SRAM can't
    # fit it and HBM is already its tier, so there is no cheaper placement.
    m = Module(name="noop")
    m.add_resource(Resource(rid=20, domain=Domain.HBM, shape=(1 << 20,)))
    m.add_phase(Phase(phase_id=0, claims=[_claim(1, [20], [20], 0, n=1 << 20)]))
    pl = place(m, AVX)
    assert pl.is_noop and pl.moved == ()
    assert pl.tier_of(20) == MemTier.HBM  # unchanged


# --- the learned, frozen scorer (moegate freeze discipline) ----------------------


def test_trained_placer_learns_hot_vs_cold_and_is_frozen_q8():
    hot = ResourceProfile(
        rid=1, size_bytes=256, lifespan=1, frequency=8, declared_tier=MemTier.DRAM
    )
    cold = ResourceProfile(
        rid=2, size_bytes=1 << 22, lifespan=8, frequency=1, declared_tier=MemTier.DRAM
    )
    data = [(hot, 1), (cold, 0)] * 16
    fp = train_placer(data, gen=2)
    assert fp.gen == 2 and all(isinstance(w, int) for w in fp.wq)  # frozen Q8 ints
    assert fp.heat(hot) > fp.heat(cold)  # learned the signal
    # deterministic freeze: same data -> identical integer weights
    assert train_placer(data, gen=2).wq == fp.wq
    assert isinstance(DEFAULT_PLACER, FrozenPlacer) and DEFAULT_PLACER.heat(
        hot
    ) > DEFAULT_PLACER.heat(cold)
