"""Bayesian cost model + conformal error bars + ABC (the learned organ)."""

import math

from bcir.examples import histogram_gather, vector_add
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.bayescal import (
    BayesianCalibratedProfile,
    abc_calibrate,
    bayes_calibrate,
    conformal_delta,
    gaussian_update,
)
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.microbench import MicrobenchRaw, calibrate_from_raw
from bcir.verify import verify_provenance

AVX = TargetProfile.x86_avx512()


def _raw(random_ns):
    # stream/strided/compute fixed at the baseline; only the random regime varies.
    return MicrobenchRaw(stream_ns=100, strided_ns=100, random_ns=random_ns,
                         compute_ns=100, n=1024, repeats=5)


def _laws(diags):
    return {d.law for d in diags}


# --- conjugate-Gaussian (VI-exact) posterior ------------------------------------

def test_weak_prior_posterior_mean_is_the_sample_mean():
    post = gaussian_update([10.0, 20.0, 30.0], prior_mean=0.0, prior_var=1e12)
    assert abs(post.mean - 20.0) < 1e-3


def test_more_data_shrinks_the_posterior_variance():
    few = gaussian_update([10.0, 20.0, 30.0], prior_mean=0.0, obs_var=100.0)
    many = gaussian_update([10.0, 20.0, 30.0] * 10, prior_mean=0.0, obs_var=100.0)
    assert many.var < few.var          # evidence concentrates the posterior


def test_empty_observations_return_the_prior():
    post = gaussian_update([], prior_mean=42.0, prior_var=7.0)
    assert post.mean == 42.0 and post.var == 7.0


# --- split conformal prediction (distribution-free) -----------------------------

def test_conformal_quantile_is_pinned():
    res = [1, 2, 3, 4, 5, 6, 7, 8, 9]            # n=9
    assert conformal_delta(res, coverage=0.9) == 9   # ceil(10*0.9)=9 -> 9th smallest
    assert conformal_delta(res, coverage=0.8) == 8   # ceil(10*0.8)=8 -> 8th smallest
    assert conformal_delta(res, coverage=0.5) == 5   # ceil(10*0.5)=5 -> 5th


def test_higher_coverage_never_shrinks_the_interval():
    res = [3, 1, 4, 1, 5, 9, 2, 6]
    assert conformal_delta(res, 0.95) >= conformal_delta(res, 0.7)


def test_no_residuals_is_zero_delta():
    assert conformal_delta([], 0.9) == 0.0


# --- the frozen Bayesian table --------------------------------------------------

def test_bayes_table_posterior_mean_and_interval():
    # Five noisy random ratios around 16x: posterior mean ~ the average; the
    # gather_penalty interval brackets the point estimate.
    raws = [_raw(r) for r in (1500, 1550, 1600, 1650, 1700)]   # random_q8 = 15..17x
    bt = bayes_calibrate(AVX, raws, coverage=0.9, cal_gen=3)
    assert bt.cal_gen == 3
    assert 15 <= bt.gather_penalty <= 17               # posterior mean ratio ~16
    lo, hi = bt.gather_penalty_interval()
    assert lo <= bt.gather_penalty <= hi               # point inside its interval
    assert bt.random_delta_q8 >= 0


def test_zero_noise_collapses_to_the_point_table():
    # Identical measurements -> zero residuals -> zero delta -> the point table
    # is the degenerate case (exact recovery of microbench.calibrate_from_raw).
    raws = [_raw(1600)] * 6
    bt = bayes_calibrate(AVX, raws, cal_gen=1)
    point = calibrate_from_raw(_raw(1600), AVX)
    # 1600/100 = 16x ratio -> random_q8 = 16*256 = 4096 -> gather_penalty 16.
    assert bt.random_delta_q8 == 0
    assert bt.gather_penalty == point.gather_penalty == 16
    assert bt.point.random_q8 == point.random_q8


def test_bayes_table_applies_like_a_point_table():
    raws = [_raw(800)] * 4                              # 8x -> gather_penalty 8
    bt = bayes_calibrate(AVX, raws, cal_gen=2)
    h = bt.apply(AVX)
    assert h.gather_penalty == 8 and h.cal_gen == 2


def test_bayes_table_json_round_trips():
    bt = bayes_calibrate(AVX, [_raw(1600), _raw(1700)], coverage=0.8, cal_gen=5)
    assert BayesianCalibratedProfile.from_json(bt.to_json()) == bt


def test_calibrated_gather_penalty_reaches_the_planner():
    # The Bayesian posterior mean re-prices histogram_gather, exactly as the
    # point table does -- the learned organ feeds the certified rail.
    bt = bayes_calibrate(AVX, [_raw(800)] * 4, cal_gen=1)   # gather_penalty 8
    seeded = optimize(histogram_gather(1024), AVX, Theta.cool()).score
    learned = optimize(histogram_gather(1024), bt.apply(AVX), Theta.cool()).score
    assert learned < seeded


# --- R13: conformal-guarantee provenance ----------------------------------------

def test_well_formed_bayes_table_satisfies_R13():
    bt = bayes_calibrate(AVX, [_raw(1500), _raw(1600), _raw(1700)], cal_gen=1)
    h = bt.apply(AVX)
    assert verify_provenance(None, h=h, table=bt) == []


def test_negative_delta_is_R13():
    from dataclasses import replace
    bt = bayes_calibrate(AVX, [_raw(1500), _raw(1700)], cal_gen=1)
    bad = replace(bt, random_delta_q8=-5)
    assert "R13" in _laws(verify_provenance(None, h=bad.apply(AVX), table=bad))


def test_interval_from_a_single_sample_is_R13():
    from dataclasses import replace
    # A nonzero conformal delta claimed from one observation is not a guarantee.
    bt = bayes_calibrate(AVX, [_raw(1600)], cal_gen=1)        # samples=1, delta=0
    forged = replace(bt, random_delta_q8=500)
    assert forged.point.samples == 1
    assert "R13" in _laws(verify_provenance(None, h=forged.apply(AVX), table=forged))


def test_out_of_range_coverage_is_R13():
    from dataclasses import replace
    bt = bayes_calibrate(AVX, [_raw(1600)] * 4, cal_gen=1)
    bad = replace(bt, coverage_milli=1500)              # > 1.0 coverage: impossible
    assert "R13" in _laws(verify_provenance(None, h=bad.apply(AVX), table=bad))


# --- ABC: the GEM/optimize forward model as the simulator -----------------------

def test_abc_recovers_the_true_gather_penalty():
    # "True" gather_penalty 16 generates the observed histogram_gather score.
    # Propose a grid of tables; ABC accepts only those whose simulated score
    # matches -- the exact-16 table is in, far-off ones are out.
    m = histogram_gather(1024)
    true = calibrate_from_raw(_raw(1600), AVX)          # gather_penalty 16
    observed = optimize(m, true.apply(AVX), Theta.cool()).score
    proposals = [calibrate_from_raw(_raw(r * 100), AVX) for r in (2, 8, 16, 24, 64)]
    post = abc_calibrate(m, AVX, Theta.cool(), observed, proposals, epsilon=0)
    gps = post.gather_penalties()
    assert 16 in gps                                   # the truth is accepted
    assert 64 not in gps and 2 not in gps              # far-off proposals rejected


def test_abc_epsilon_widens_the_posterior():
    m = histogram_gather(1024)
    true = calibrate_from_raw(_raw(1600), AVX)
    observed = optimize(m, true.apply(AVX), Theta.cool()).score
    proposals = [calibrate_from_raw(_raw(r * 100), AVX) for r in (8, 12, 16, 20, 24)]
    tight = abc_calibrate(m, AVX, Theta.cool(), observed, proposals, epsilon=0)
    loose = abc_calibrate(m, AVX, Theta.cool(), observed, proposals, epsilon=10**9)
    assert len(loose.accepted) >= len(tight.accepted)
    assert len(loose.accepted) == len(proposals)       # everything within a huge eps


def test_abc_posterior_folds_into_a_bayesian_table():
    m = histogram_gather(1024)
    true = calibrate_from_raw(_raw(1600), AVX)
    observed = optimize(m, true.apply(AVX), Theta.cool()).score
    proposals = [calibrate_from_raw(_raw(r * 100), AVX) for r in (12, 14, 16, 18, 20)]
    post = abc_calibrate(m, AVX, Theta.cool(), observed, proposals, epsilon=10**9)
    bt = post.to_bayesian(AVX, coverage=0.9, cal_gen=4)
    assert bt.cal_gen == 4
    lo, hi = bt.gather_penalty_interval()
    assert lo <= bt.gather_penalty <= hi
    assert verify_provenance(None, h=bt.apply(AVX), table=bt) == []
