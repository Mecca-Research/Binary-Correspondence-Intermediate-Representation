"""Multi-claim fusion: where the (max,+) overlap pricing and batching earn their
keep. Two independent claims sharing a read operand run as concurrent waves
(makespan < serial), and the StreamPack/batch groups them."""

from bcir.examples import fused_chain, vector_add
from bcir.gem import hydrate, price_scheduled
from bcir.gem.overlap import optimize_scheduled
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta

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
