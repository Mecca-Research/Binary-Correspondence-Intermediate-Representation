"""L3 regret ledger + policy provenance (law R13) tests."""

from dataclasses import replace

from bcir.examples import vector_add
from bcir.kbcir import (
    PolicyPortfolio,
    ReplayCertificate,
    boundary_report,
    ledger_from_episodes,
    measure_regret,
    replay_gate,
)
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.microbench import reference_table
from bcir.kbcir.portfolio import PortfolioEntry
from bcir.kbcir.weights import ENERGY, PERF, SAFE, THROUGHPUT, Policy
from bcir.verify import verify_provenance

AVX = TargetProfile.x86_avx512()
EPISODES = [Theta.cool(), Theta.hot(), Theta.mem_bound()]
HOTBIAS = Policy("hotbias", (0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0))


def _laws(diags):
    return {d.law for d in diags}


# --- the regret ledger ------------------------------------------------------------

def test_default_portfolio_has_zero_regret():
    # The seeded portfolio's class-table selection is hindsight-optimal among
    # its own certified entries on the standard episodes: the dashboard is calm.
    ledger = ledger_from_episodes(vector_add(1024), AVX, EPISODES,
                                  PolicyPortfolio.default())
    assert all(e.total_regret == 0 for e in ledger.entries.values())
    assert all(v.verdict == "keep" for v in boundary_report(ledger))


def test_regret_is_pinned_for_a_bad_rule():
    # hotbias deployed on a cool machine picks vec8; judged under the neutral
    # latency yardstick that plan prices 9472 vs the hindsight-best 7808:
    # regret 1664, attributed to the deployed rule.
    m = measure_regret(vector_add(1024), AVX, Theta.cool(), HOTBIAS,
                       [PERF, THROUGHPUT, ENERGY, SAFE])
    assert (m.deployed, m.best, m.regret) == (9472, 7808, 1664)
    assert m.best_rule == "latency"


def test_regret_is_nonnegative_by_construction():
    # The deployed rule is always in the comparison pool.
    for pol in (PERF, ENERGY, HOTBIAS):
        for theta in EPISODES:
            m = measure_regret(vector_add(1024), AVX, theta, pol, [PERF, ENERGY])
            assert m.regret >= 0


def test_ledger_books_accumulate_and_render():
    # Deploy hotbias in the nominal slot: the ledger books the sustained
    # regret and the boundary report flags exactly that rule for retuning.
    p = PolicyPortfolio.default()
    p.entries["latency"] = PortfolioEntry(policy=HOTBIAS, gen=1, certified=True)
    ledger = ledger_from_episodes(vector_add(1024), AVX,
                                  [Theta.cool(), Theta.cool()], p)
    e = ledger.entries["hotbias"]
    assert (e.episodes, e.total_regret, e.worst_regret, e.regret_rate) == (2, 3328, 1664, 1664)
    assert "hotbias" in ledger.dashboard()
    verdicts = {v.rule: v.verdict for v in boundary_report(ledger)}
    assert verdicts["hotbias"] == "retune"


def test_ledger_is_deterministic():
    p = PolicyPortfolio.default()
    a = ledger_from_episodes(vector_add(1024), AVX, EPISODES, p)
    b = ledger_from_episodes(vector_add(1024), AVX, EPISODES, p)
    assert a.entries == b.entries


# --- law R13: policy/table provenance ----------------------------------------------

def test_promoted_policy_without_certificate_is_R13():
    p = PolicyPortfolio.default()
    p.entries["latency"] = PortfolioEntry(policy=PERF, gen=2, certified=True)
    assert "R13" in _laws(verify_provenance(p))


def test_certified_promotion_satisfies_R13():
    # The full third-order loop, closed by hand: the ledger flags the slot, a
    # retuned candidate passes the L2 gate, the swap is certified -- and R13
    # can witness the whole chain.
    p = PolicyPortfolio.default()
    retuned = Policy("latency", PERF.base)
    cert = replay_gate(vector_add(1024), AVX, retuned, PERF, EPISODES)
    p.promote("latency", retuned, cert)
    assert p.entries["latency"].gen == 2
    assert verify_provenance(p, certificates=[cert]) == []
    assert "R13" in _laws(verify_provenance(p))  # drop the certificate: illegal


def test_forged_certificate_does_not_cover_the_swap():
    p = PolicyPortfolio.default()
    p.entries["latency"] = PortfolioEntry(policy=PERF, gen=2, certified=True)
    forged = ReplayCertificate(candidate="other", incumbent="latency",
                               episodes=3, regressions=0)
    assert "R13" in _laws(verify_provenance(p, certificates=[forged]))


def test_calibrated_profile_must_present_its_table():
    table = reference_table()
    h = table.apply(AVX)
    assert verify_provenance(None, h=h, table=table) == []
    assert "R13" in _laws(verify_provenance(None, h=h))  # table withheld


def test_table_drift_and_stale_generation_are_R13():
    table = reference_table()
    h = table.apply(AVX)
    drifted = replace(h, gather_penalty=99)
    assert "R13" in _laws(verify_provenance(None, h=drifted, table=table))
    stale = replace(h, cal_gen=2)
    assert "R13" in _laws(verify_provenance(None, h=stale, table=table))


def test_seeded_profile_needs_no_provenance():
    # cal_gen 0 means seeded constants: nothing to certify.
    assert verify_provenance(PolicyPortfolio.default(), h=AVX) == []
