"""The temperature dial: soft log-sum-exp DP, exact at T=0, differentiable at T>0."""

from bcir.examples import PROGRAMS, saxpy_strided, vector_add
from bcir.kbcir import TARGETS, optimize, softselect, temperature_sweep
from bcir.kbcir.cost import Theta
from bcir.kbcir.softdp import free_energy, grad_free_energy_wrt_weights
from bcir.kbcir.weights import PERF, weights

AVX = TARGETS["x86_avx512"]


def test_zero_temperature_is_the_tropical_optimizer_exactly():
    # T=0 delegates to the integer optimizer: free energy == hard score (bit
    # exact), marginals one-hot on the chosen plan, MAP == the tropical plan.
    for name, build in PROGRAMS.items():
        for h in TARGETS.values():
            m = build()
            hard = optimize(m, h, Theta.cool())
            s = softselect(m, h, Theta.cool(), PERF, temperature=0.0)
            assert s.free_energy == float(hard.score), name
            assert s.hard_score == hard.score
            for step in hard.steps:
                assert s.map_by_claim[step.claim_id] == step.candidate.name
                assert s.marginals[step.claim_id][step.candidate.name] == 1.0


def test_vector_add_zero_temp_pins_7808():
    s = softselect(vector_add(1024), AVX, Theta.cool(), PERF, 0.0)
    assert s.free_energy == 7808.0 and s.hard_score == 7808
    assert s.map_by_claim[1000] == "vec16"
    assert s.marginals[1000] == {"vec16": 1.0}


def test_small_temperature_concentrates_on_the_hard_plan():
    # T tiny vs the candidate gap (vec16 7808 vs vec8 9472, gap 1664): the soft
    # distribution collapses onto vec16 and the free energy ~ the hard score.
    s = softselect(vector_add(1024), AVX, Theta.cool(), PERF, temperature=1.0)
    assert s.map_by_claim[1000] == "vec16"
    assert s.marginals[1000]["vec16"] > 0.999
    assert abs(s.free_energy - 7808.0) < 1.0


def test_free_energy_never_exceeds_the_hard_minimum():
    # softmin <= min for all T > 0: F_T <= hard_score, with the gap closing as T->0.
    m = vector_add(1024)
    for T in (1.0, 100.0, 1000.0, 10000.0):
        s = softselect(m, AVX, Theta.cool(), PERF, T)
        assert s.free_energy <= s.hard_score
        assert s.gap >= 0.0


def test_free_energy_rises_toward_the_minimum_as_temperature_falls():
    m = vector_add(1024)
    hot = free_energy(m, AVX, Theta.cool(), PERF, 10000.0)
    warm = free_energy(m, AVX, Theta.cool(), PERF, 1000.0)
    cold = free_energy(m, AVX, Theta.cool(), PERF, 10.0)
    assert hot < warm < cold <= 7808.0


def test_marginals_are_a_distribution():
    m = saxpy_strided(1024)
    s = softselect(m, AVX, Theta.cool(), PERF, temperature=500.0)
    for dist in s.marginals.values():
        assert abs(sum(dist.values()) - 1.0) < 1e-9
        assert all(0.0 <= p <= 1.0 for p in dist.values())


def test_higher_temperature_raises_entropy():
    # More temperature => flatter marginals => the top candidate's mass drops.
    m = vector_add(1024)
    cold = softselect(m, AVX, Theta.cool(), PERF, 500.0).marginals[1000]["vec16"]
    hot = softselect(m, AVX, Theta.cool(), PERF, 5000.0).marginals[1000]["vec16"]
    assert cold > hot


def test_expected_cost_dot_weights_equals_free_energy_at_zero():
    # At T=0 the expected cost vector IS the chosen path's cost, so its
    # scalarization reproduces the score exactly: E[C].w == F_0 == 7808.
    m = vector_add(1024)
    s = softselect(m, AVX, Theta.cool(), PERF, 0.0)
    assert s.expected_score(AVX, Theta.cool(), PERF) == 7808.0


def test_expected_score_upper_bounds_free_energy_jensen():
    # Gibbs/Jensen: E_pi[score] >= F_T (softmin <= mean), equality only at T=0.
    m = vector_add(1024)
    for T in (10.0, 1000.0, 10000.0):
        s = softselect(m, AVX, Theta.cool(), PERF, T)
        assert s.expected_score(AVX, Theta.cool(), PERF) >= s.free_energy - 1e-6


def test_analytic_gradient_is_the_expected_cost():
    m = vector_add(1024)
    s = softselect(m, AVX, Theta.cool(), PERF, 1000.0)
    assert grad_free_energy_wrt_weights(s) == s.expected_cost
    assert len(s.expected_cost) == 12


def _policy_with(base):
    from bcir.kbcir.weights import Policy
    return Policy("perf_bumped", tuple(base))


def test_gradient_matches_finite_difference():
    # dF/dw_d == E[C_d]: verify the analytic gradient against a central finite
    # difference in weight d (memory, dim 1). At cool Theta weights == policy.base,
    # so bumping the base bumps the effective weight by the same amount.
    from bcir.kbcir import cost as costmod
    m = vector_add(1024)
    T = 2000.0
    grad = grad_free_energy_wrt_weights(softselect(m, AVX, Theta.cool(), PERF, T))
    base = list(weights(AVX, Theta.cool(), 0, PERF))
    d = costmod.MEMORY
    eps = 1.0
    plus = list(base); plus[d] += eps
    minus = list(base); minus[d] -= eps
    f_plus = free_energy(m, AVX, Theta.cool(), _policy_with(plus), T)
    f_minus = free_energy(m, AVX, Theta.cool(), _policy_with(minus), T)
    fd = (f_plus - f_minus) / (2 * eps)
    assert abs(fd - grad[d]) < max(0.5, abs(grad[d]) * 0.02)


def test_temperature_sweep_is_monotone_and_bounded():
    sweep = temperature_sweep(vector_add(1024), AVX, Theta.cool())
    fs = [f for _, f in sweep]
    assert sweep[0][0] == 0.0 and fs[0] == 7808.0          # T=0 anchor
    assert all(f <= 7808.0 for f in fs)                    # F <= hard min
    assert all(a >= b for a, b in zip(fs, fs[1:]))         # rises as T falls
