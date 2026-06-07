"""K_BCIR cost-algebra and optimizer tests (exact, deterministic)."""

from bcir.examples import histogram_gather, vector_add
from bcir.kbcir import optimize
from bcir.kbcir.cost import N, CostVector, TargetProfile, Theta
from bcir.kbcir.semiring import dag_shortest_path
from bcir.model import Lane


def test_cost_vector_is_12d():
    assert N == 12
    cv = CostVector.of(compute=3, memory=5)
    assert cv.as_dict()["compute"] == 3
    assert cv.as_dict()["memory"] == 5
    assert cv.dot((1,) * N) == 8


def test_couple_is_q8_scale():
    cv = CostVector.of(memory=1000)
    # x0.75 in Q8 == 192/256.
    assert cv.couple(tuple(192 if i == 1 else 256 for i in range(N))).as_dict()["memory"] == 750


def test_vector_add_cool_selects_vec16_score_7808():
    # The LangRef worked example: cool Theta on AVX-512 picks vec16 with score 7808.
    res = optimize(vector_add(1024), TargetProfile.x86_avx512(), Theta.cool())
    cand = res.by_claim()[1000]
    assert cand.width == 16
    assert cand.lane == Lane.U
    assert res.score == 7808


def test_vector_add_hot_replans_to_vec8():
    # A hot machine raises thermal/power weights -> wide SIMD downclocks -> pick vec8.
    res = optimize(vector_add(1024), TargetProfile.x86_avx512(), Theta.hot())
    assert res.by_claim()[1000].width == 8


def test_random_gather_is_never_bucketed():
    # Correctness law: a declared-RANDOM claim only gets the legal GGG gather lane,
    # never a cheaper UX cacheline bucket.
    res = optimize(histogram_gather(1024), TargetProfile.x86_avx2(), Theta.cool())
    cand = res.by_claim()[3000]
    assert cand.lane == Lane.GGG
    assert cand.name == "gather"


def test_semiring_min_plus_shortest_path():
    # 0 ->(1) 1 ->(1) 3 ; 0 ->(5) 2 ->(1) 3 ; min path cost = 2.
    adj = [[(1, 1), (2, 5)], [(3, 1)], [(3, 1)], []]
    dist, pred = dag_shortest_path(4, adj, source=0)
    assert dist[3] == 2
    assert pred[3] == 1
