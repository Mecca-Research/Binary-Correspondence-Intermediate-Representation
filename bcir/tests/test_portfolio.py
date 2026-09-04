"""L2 learning placement tests: the policy portfolio + the replay gate."""

from bcir.examples import vector_add
from bcir.kbcir import (
    PolicyPortfolio,
    ReplayCertificate,
    classify,
    episodes_from,
    replay_gate,
)
from bcir.kbcir.calibrate import EwmaCalibrator, adaptive_policy
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.weights import ENERGY, PERF, Policy
from bcir.telemetry import DataDNA

AVX = TargetProfile.x86_avx512()
EPISODES = [Theta.cool(), Theta.hot(), Theta.mem_bound()]


def test_selection_is_a_deterministic_class_table():
    # The portfolio selector mirrors the adaptive policy rule exactly -- a
    # plan-time table lookup, not an inference.
    p = PolicyPortfolio.default()
    for theta in EPISODES:
        assert p.select(theta).name == adaptive_policy(theta).name
    assert classify(Theta.cool()) == "nominal"
    assert classify(Theta.hot()) == "constrained"
    assert classify(Theta.mem_bound()) == "saturated"


def test_gate_admits_a_non_regressing_candidate():
    # A candidate identical to the incumbent can never regress: admitted.
    cert = replay_gate(vector_add(1024), AVX, PERF, PERF, EPISODES)
    assert cert.admitted and cert.regressions == 0 and cert.episodes == 3


def test_gate_rejects_a_regressing_candidate():
    # A thermal-obsessed gain schedule picks vec8 on a cool machine; judged
    # under the incumbent's metric that plan prices 9472 > 7808 -- regression.
    hotbias = Policy("hotbias", (0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0))
    cert = replay_gate(vector_add(1024), AVX, hotbias, PERF, [Theta.cool()])
    assert not cert.admitted and cert.regressions == 1


def test_promotion_requires_an_admitting_certificate():
    p = PolicyPortfolio.default()
    hotbias = Policy("hotbias", (0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0))
    bad = replay_gate(vector_add(1024), AVX, hotbias, PERF, [Theta.cool()])
    try:
        p.promote("latency", hotbias, bad)
        assert False, "expected the gate to hold"
    except ValueError:
        pass
    assert p.entries["latency"].policy is PERF  # the incumbent survives


def test_promotion_behind_the_gate_bumps_the_generation():
    # A retuned-but-equivalent schedule passes the gate; promotion swaps it in
    # with a bumped generation tag (frozen, certified -- never silent).
    p = PolicyPortfolio.default()
    retuned = Policy("latency", (2, 2, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1))  # == PERF weights
    cert = replay_gate(vector_add(1024), AVX, retuned, PERF, EPISODES)
    assert cert.admitted
    entry = p.promote("latency", retuned, cert)
    assert entry.certified and entry.gen == 2
    assert p.select(Theta.cool()) is retuned


def test_certificate_must_cover_the_promotion():
    p = PolicyPortfolio.default()
    forged = ReplayCertificate(candidate="other", incumbent="latency", episodes=3, regressions=0)
    try:
        p.promote("latency", ENERGY, forged)
        assert False, "expected a coverage check"
    except ValueError:
        pass


def test_gate_is_deterministic():
    a = replay_gate(vector_add(1024), AVX, ENERGY, PERF, EPISODES)
    b = replay_gate(vector_add(1024), AVX, ENERGY, PERF, EPISODES)
    assert a == b


def test_episodes_fold_from_logged_telemetry():
    # The gate replays Theta episodes folded from the data-DNA log -- the same
    # provenance chain the rehydrate loop uses.
    batches = [
        [DataDNA(segment_id="s0", claim_id=1000, thermal=80, utilization=70)],
        [DataDNA(segment_id="s0", claim_id=1000, thermal=10, utilization=20)],
    ]
    eps = episodes_from(batches, EwmaCalibrator(), Theta.cool())
    assert len(eps) == 2
    assert eps[0].thermal > eps[1].thermal >= 0
    cert = replay_gate(vector_add(1024), AVX, PERF, PERF, eps)
    assert cert.admitted
