"""Fuzzy / continuous MoE routing: during exploration data flows partially through
every expert (temperature-annealed soft weights); the choice hardens to one expert
only when a path dominates. The frozen gate exposes the same fuzzy weights as
deterministic Q8 integers."""

from bcir.examples import histogram_gather, vector_add
from bcir.kbcir import TARGETS, freeze, harden, train_gate
from bcir.kbcir.cost import Theta
from bcir.kbcir.weights import ENERGY, PERF

AVX = TARGETS["x86_avx512"]
EXPERTS = [PERF, ENERGY]
COOL = Theta.cool()
_SAMPLES = [(vector_add(1024), COOL, 0), (histogram_gather(1024), COOL, 1)]


def _gate():
    return train_gate(_SAMPLES, EXPERTS, hidden=4, epochs=200)


# --- continuous edge weights (the GNN, training/exploration form) ----------------


def test_fuzzy_route_is_a_distribution_over_all_experts():
    g = _gate()
    w = g.route_fuzzy(vector_add(1024), COOL, temperature=1.0)
    assert len(w) == len(EXPERTS) and all(x >= 0.0 for x in w)
    assert abs(sum(w) - 1.0) < 1e-9
    assert all(x > 0.0 for x in w)  # partial flow through *every* path


def test_temperature_anneals_from_soft_to_hard():
    g = _gate()
    m = vector_add(1024)
    soft = max(g.route_fuzzy(m, COOL, temperature=5.0))  # hot: flat / exploratory
    sharp = max(g.route_fuzzy(m, COOL, temperature=0.1))  # cold: peaked / exploitative
    assert sharp >= soft  # low temperature sharpens


def test_blend_pairs_experts_with_weights():
    g = _gate()
    blend = g.blend(vector_add(1024), COOL, temperature=1.0)
    assert [p for p, _ in blend] == EXPERTS
    assert abs(sum(w for _, w in blend) - 1.0) < 1e-9


def test_harden_collapses_only_when_a_path_dominates():
    g = _gate()
    m = vector_add(1024)
    hardened, idx = harden(g.route_fuzzy(m, COOL, temperature=0.05))  # peaked
    assert hardened and idx == g.route(m, COOL)  # collapses to the argmax expert
    # an even split never hardens (still exploring)
    assert harden([0.5, 0.5], threshold=0.7) == (False, None)


# --- the frozen gate's integer fuzzy distribution --------------------------------


def test_frozen_distribution_is_q8_normalized_and_matches_route():
    fg = freeze(_gate())
    m = vector_add(1024)
    d = fg.distribution(m, COOL)
    assert sum(d) == 256 and all(x >= 0 for x in d)  # exact Q8 simplex
    assert max(range(len(d)), key=lambda k: d[k]) == fg.route(m, COOL)  # argmax == route


def test_frozen_distribution_temperature_flattens_and_is_deterministic():
    fg = freeze(_gate())
    m = histogram_gather(1024)
    assert max(fg.distribution(m, COOL, temperature=1)) >= max(
        fg.distribution(m, COOL, temperature=8)
    )  # higher temp -> flatter
    assert fg.distribution(m, COOL) == fg.distribution(m, COOL)  # deterministic
