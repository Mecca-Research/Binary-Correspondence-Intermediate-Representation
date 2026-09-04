"""The learned MoE gate: a GNN over the claim graph, trained on the regret ledger."""

from dataclasses import replace

from bcir.examples import histogram_gather, saxpy_strided, tiled_matmul, vector_add
from bcir.kbcir import (
    FrozenGate,
    PolicyPortfolio,
    TARGETS,
    freeze,
    gate_replay_gate,
    ledger_labels,
    train_gate,
)
from bcir.kbcir.cost import Theta
from bcir.kbcir.moegate import node_features, theta_features
from bcir.kbcir.weights import ENERGY, PERF, SAFE, THROUGHPUT
from bcir.verify import verify_provenance

AVX = TARGETS["x86_avx512"]
EXPERTS = [PERF, ENERGY]
COOL = Theta.cool()


def _laws(diags):
    return {d.law for d in diags}


# --- features -------------------------------------------------------------------


def test_node_and_theta_features_have_fixed_shapes():
    f = node_features(vector_add(1024).phases[0].claims[0])
    assert len(f) == 15 and f[0] == 1.0
    assert len(theta_features(Theta.hot())) == 5
    assert theta_features(Theta.hot())[0] == 0.8  # thermal 80 / 100


# --- training (exact softmax-CE SGD) --------------------------------------------

_SEPARABLE = [
    (vector_add(1024), COOL, 0),  # unit-stride graph -> expert 0
    (histogram_gather(1024), COOL, 1),  # gather graph      -> expert 1
]


def test_gate_fits_a_graph_separable_dataset():
    # Two distinct claim-graph shapes with distinct best experts: the GNN learns
    # to route by graph structure (100% training accuracy).
    g = train_gate(_SEPARABLE, EXPERTS, hidden=8, epochs=300, lr=0.2)
    assert g.route(vector_add(1024), COOL) == 0
    assert g.route(histogram_gather(1024), COOL) == 1


def test_gate_learns_a_theta_split():
    # Same graph, different Theta -> different best expert: the head learns the
    # Theta part of the routing too.
    samples = [(vector_add(1024), Theta.cool(), 0), (vector_add(1024), Theta.hot(), 1)]
    g = train_gate(samples, EXPERTS, hidden=8, epochs=400, lr=0.3)
    assert g.route(vector_add(1024), Theta.cool()) == 0
    assert g.route(vector_add(1024), Theta.hot()) == 1


def test_training_is_deterministic():
    a = train_gate(_SEPARABLE, EXPERTS, epochs=120)
    b = train_gate(_SEPARABLE, EXPERTS, epochs=120)
    assert a.W == b.W and a.b == b.b and a.U == b.U and a.c == b.c


def test_distribution_is_a_simplex():
    g = train_gate(_SEPARABLE, EXPERTS, epochs=50)
    p = g.distribution(histogram_gather(1024), COOL)
    assert abs(sum(p) - 1.0) < 1e-9 and all(0 <= x <= 1 for x in p)


def test_route_returns_a_portfolio_expert():
    g = train_gate(_SEPARABLE, EXPERTS, epochs=50)
    assert g.route_policy(vector_add(1024), COOL) in EXPERTS


# --- the frozen integer gate ----------------------------------------------------


def test_frozen_gate_matches_float_routing_on_clear_margins():
    g = train_gate(_SEPARABLE, EXPERTS, epochs=300, lr=0.2)
    fz = freeze(g)
    for m in (vector_add(1024), histogram_gather(1024)):
        assert fz.route(m, COOL) == g.route(m, COOL)


def test_frozen_route_is_deterministic_integer():
    fz = freeze(train_gate(_SEPARABLE, EXPERTS, epochs=120))
    # Two calls, identical integer result; and the route is a valid expert index.
    r1 = fz.route(histogram_gather(1024), COOL)
    r2 = fz.route(histogram_gather(1024), COOL)
    assert r1 == r2 and 0 <= r1 < fz.num_experts


def test_freeze_carries_a_stable_fingerprint_and_gen():
    fz = freeze(train_gate(_SEPARABLE, EXPERTS, epochs=120), gate_gen=3)
    assert fz.gate_gen == 3
    assert (
        fz.fingerprint
        == freeze(train_gate(_SEPARABLE, EXPERTS, epochs=120), gate_gen=3).fingerprint
    )


def test_multi_claim_graph_aggregates_neighbours():
    # A multi-claim module exercises the message-passing aggregation (not just
    # the single-node path); routing still returns a valid expert.
    g = train_gate(_SEPARABLE, EXPERTS, epochs=80)
    m = saxpy_strided(1024)
    assert 0 <= g.route(m, COOL) < len(EXPERTS)
    assert 0 <= freeze(g).route(tiled_matmul(256), COOL) < len(EXPERTS)


# --- trained on the regret ledger -----------------------------------------------


def test_ledger_labels_pick_the_hindsight_best_expert():
    thetas = [Theta.cool(), Theta.hot(), Theta.mem_bound()]
    samples, experts = ledger_labels(vector_add(1024), AVX, thetas, PolicyPortfolio.default())
    assert len(samples) == 3
    assert [e.name for e in experts] == ["energy", "latency", "safe", "throughput"]
    for _, _, label in samples:
        assert 0 <= label < len(experts)


def test_gate_trained_on_the_ledger_routes_validly():
    thetas = [Theta.cool(), Theta.hot(), Theta.mem_bound()]
    samples, experts = ledger_labels(vector_add(1024), AVX, thetas, PolicyPortfolio.default())
    g = train_gate(samples, experts, epochs=200)
    pol = g.route_policy(vector_add(1024), Theta.cool())
    assert pol in experts


# --- the replay gate (router deploys behind a no-regression certificate) --------


def test_gate_replay_gate_admits_a_non_regressing_router():
    # A gate trained on the ledger's own hindsight-best labels cannot regress
    # against the incumbent classify router on those episodes.
    thetas = [Theta.cool(), Theta.hot(), Theta.mem_bound()]
    samples, experts = ledger_labels(vector_add(1024), AVX, thetas, PolicyPortfolio.default())
    g = freeze(train_gate(samples, experts, epochs=300))
    cert = gate_replay_gate(vector_add(1024), AVX, thetas, g, PolicyPortfolio.default())
    assert cert.candidate == "moe_gate" and cert.incumbent == "classify"
    assert cert.admitted and cert.regressions == 0


def test_gate_replay_gate_rejects_a_regressing_router():
    # A gate hard-wired (via a one-class training set) to always route to the
    # energy specialist regresses on a cool machine (where latency wins).
    thetas = [Theta.cool()]
    bad_samples = [(vector_add(1024), Theta.cool(), 0)]  # always expert 0 = energy
    experts = [ENERGY, PERF]
    g = freeze(train_gate(bad_samples, experts, epochs=300))
    cert = gate_replay_gate(vector_add(1024), AVX, thetas, g, PolicyPortfolio.default())
    assert g.route(vector_add(1024), Theta.cool()) == 0  # routes to energy
    assert not cert.admitted and cert.regressions == 1


# --- R13: gate provenance --------------------------------------------------------


def test_admitting_gate_certificate_satisfies_R13():
    thetas = [Theta.cool(), Theta.hot(), Theta.mem_bound()]
    samples, experts = ledger_labels(vector_add(1024), AVX, thetas, PolicyPortfolio.default())
    g = freeze(train_gate(samples, experts, epochs=300))
    cert = gate_replay_gate(vector_add(1024), AVX, thetas, g, PolicyPortfolio.default())
    assert verify_provenance(None, gate_certificates=[cert]) == []


def test_deploying_a_regressing_gate_is_R13():
    from bcir.kbcir import ReplayCertificate

    # A gate proposed for deployment whose replay certificate did not pass
    # (regressions > 0): R13 forbids deploying it -- the network proposes, the
    # verifier disposes.
    regressing = ReplayCertificate(
        candidate="moe_gate", incumbent="classify", episodes=5, regressions=2
    )
    assert not regressing.admitted
    assert "R13" in _laws(verify_provenance(None, gate_certificates=[regressing]))
