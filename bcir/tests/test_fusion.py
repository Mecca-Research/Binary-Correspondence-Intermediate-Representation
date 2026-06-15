"""Multi-claim fusion: where the (max,+) overlap pricing and batching earn their
keep. Two independent claims sharing a read operand run as concurrent waves
(makespan < serial), and the StreamPack/batch groups them."""

from dataclasses import replace

from bcir.examples import fused_chain, histogram_gather, saxpy_strided, scan_chain, vector_add
from bcir.gem import hydrate, price_scheduled
from bcir.gem.overlap import optimize_scheduled
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def test_multi_claim_overlap_beats_the_serial_sum():
    # fused_chain: two independent vec16 claims -> they co-execute on distinct
    # affinity domains, so the (max,+) makespan is below the serial Sigma.
    m = fused_chain(1024)
    r = optimize(m, AVX, COOL)
    sp = price_scheduled(m, r, AVX, COOL)
    assert len(r.steps) == 2
    assert sp.makespan < sp.serial and sp.overlap_gain > 0


def test_single_claim_has_no_overlap():
    m = vector_add(1024)
    sp = price_scheduled(m, optimize(m, AVX, COOL), AVX, COOL)
    assert sp.overlap_gain == 0          # one claim: makespan == serial (the degenerate case)


def test_fused_chain_hydrates_two_segments():
    m = fused_chain(1024)
    pack = hydrate(m, optimize(m, AVX, COOL))
    assert len(pack.segments) == 2 and pack.provenance_ok()


def test_optimize_scheduled_is_r9_consistent():
    # the scheduled optimizer returns a serial-repriced plan (score == sum of step
    # costs) whose makespan does not exceed the serial bound.
    m = fused_chain(1024)
    r, sp = optimize_scheduled(m, AVX, COOL)
    assert sp.makespan <= sp.serial
    assert r.score == sum(s.cost for s in r.steps)


# --- scan: a dependency chain serializes (the counterpart to overlap) ------------

def test_dependent_chain_does_not_overlap():
    m = scan_chain(1024)
    sp = price_scheduled(m, optimize(m, AVX, COOL), AVX, COOL)
    assert len(m.phases[0].claims) == 2
    assert sp.overlap_gain == 0          # RAW dependency: no overlap, no fusion


# --- the fusion discount (distinct from overlap): shared operand, one bin --------

def _no_share(n=1024):
    m = Module(name="no_share")
    for rid, nm in ((60, "A"), (61, "B"), (62, "C"), (63, "E"), (64, "D"), (65, "F")):
        m.add_resource(Resource(rid=rid, shape=(n,), name=nm))
    c1 = Claim(id=6001, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(60, 61), wr=(62,), op="vector.add")
    c2 = Claim(id=6002, opcode=Opcode.ADD, lane=Lane.U, stride_class=StrideClass.UNIT,
               count=n, rd=(63, 65), wr=(64,), op="vector.add")
    m.add_phase(Phase(phase_id=0, deps=(), claims=[c1, c2]))
    return m


def test_fusion_discount_rewards_a_shared_operand_in_one_bin():
    # On a single affinity domain both claims share one bin (back-to-back). When
    # they share a read operand (A), the second reuses A's loaded lines -> the
    # memory fusion discount; an otherwise-identical no-share chain pays full price.
    h1 = replace(AVX, affinity_domains=1)
    shared = price_scheduled(fused_chain(1024), optimize(fused_chain(1024), h1, COOL), h1, COOL)
    noshare = price_scheduled(_no_share(1024), optimize(_no_share(1024), h1, COOL), h1, COOL)
    assert shared.makespan < noshare.makespan      # the fusion discount is real


# --- per-target parity beyond vector_add -----------------------------------------

def test_per_target_parity_saxpy_and_histogram():
    # pin the selected scores across targets (the worked-example matrix widened).
    saxpy = {t: optimize(saxpy_strided(1024), TARGETS[t], COOL).score
             for t in ("x86_avx512", "x86_avx2", "arm64_neon", "nvidia_ptx")}
    assert saxpy == {"x86_avx512": 71680, "x86_avx2": 71680,
                     "arm64_neon": 71680, "nvidia_ptx": 71680}
    histo = {t: optimize(histogram_gather(1024), TARGETS[t], COOL).score
             for t in ("x86_avx512", "nvidia_ptx")}
    assert histo == {"x86_avx512": 528384, "nvidia_ptx": 266240}
