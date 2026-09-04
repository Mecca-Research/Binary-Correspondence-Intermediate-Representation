"""G4 (Pillar 3a) -- a cache-line / memory-bank-conflict predictor as a FROZEN, informs-only cost SIGNAL.

The vision-alignment audit (Pillar 3a) flags that BCIR's ``memory`` cost is a STATIC Q8 formula with NO model
that PREDICTS cache-line utilization or memory-bank conflicts from a claim's access SHAPE. ``cache_predict.py``
builds that "Layer-1 AI" predictor in BCIR's disciplined form: a small, deterministic, analytic model that
computes a cache-line-utilization-waste + bank-conflict cost SIGNAL from a claim's (stride, element size,
count) and FROZEN Q8 cache/bank params, and feeds it into the cost model's plan ranking on the CONTENTION
axis -- STRICTLY off the legality path (it INFORMS which realization is cheapest, NEVER a legality verdict).

Covers:
  * the analytic model: a unit-stride sweep has 0 waste / 0 conflict (a clean no-op signal); a power-of-2
    bank-colliding stride (stride == a multiple of #banks) prices HIGHER bank-conflict contention than a
    coprime/conflict-free stride; a strided gather prices HIGHER cache-line waste than a unit sweep;
  * the FROZEN Q8 params (cacheline, banks == mem_channels) + reproducibility (same claim+params => same Q8
    cost, every time -- a hard BCIR requirement);
  * THE QUARANTINE PROOF (the gate that matters): feeding a conflict-prone and a conflict-free plan through
    the verifier yields IDENTICAL legality with AND without the contention signal -- the signal NEVER changes
    an R-law verdict, and it ONLY moves the CONTENTION cost axis;
  * OPT-IN / GAINS-ONLY: the default ``optimize`` path never sets CONTENTION, so every existing pinned
    conformance score is byte-identical (the signal is consulted only when a caller asks);
  * the proof-carrying ``ContentionPrediction`` record + the opt-in ``rank_realizations`` helper (a
    conflict-prone realization ranks more expensive; a conflict-free one is preferred).

Two-truth discipline (``twotruth.py``): a predicted cache/bank cost is a GRADED signal -- it informs plan
ranking but is quarantined from the R-laws. Deterministic + pure-Python on the oracle rail; no MLIR change.
"""

from dataclasses import replace

from bcir.model import Claim, Domain, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.kbcir.cost import CONTENTION, MEMORY, TargetProfile, Theta
from bcir.kbcir.realize import candidates_for, optimize
from bcir.kbcir.cache_predict import (
    AccessShape,
    CachePredictor,
    ContentionPrediction,
    predict,
    rank_realizations,
)
from bcir.verify import verify_plan

AVX = TargetProfile.x86_avx512()  # cacheline 64, elem_bytes 4, mem_channels(banks) 4
N = 1024


def _claim(sc, sk=1, n=N, cid=1):
    return Claim(
        id=cid,
        opcode=Opcode.ADD,
        lane=Lane.U,
        stride_class=sc,
        stride_k=sk,
        count=n,
        rd=(1,),
        wr=(2,),
    )


def _module(sc, sk=1):
    m = Module(name="g4")
    m.add_resource(Resource(rid=1, domain=Domain.RAM, shape=(N,), name="a"))
    m.add_resource(Resource(rid=2, domain=Domain.RAM, shape=(N,), name="b"))
    m.add_phase(Phase(phase_id=0, claims=[_claim(sc, sk)]))
    return m


# --- the FROZEN Q8 params + the access-shape mapping ----------------------------------------------


def test_predictor_geometry_matches_the_target_cache_and_banks():
    pred = CachePredictor.for_target(AVX)
    assert pred.cacheline == AVX.cacheline  # frozen cache-line size
    assert pred.banks == AVX.mem_channels  # banks == the concurrent-stream / channel count
    assert (
        pred._elems_per_line(AVX.elem_bytes) == AVX.cacheline // AVX.elem_bytes
    )  # 16 f32 per 64B line
    assert pred.gen == 0  # seeded analytic constants (a calibrated table bumps it)


def test_access_shape_reads_the_declared_stride():
    # UNIT/SCALAR/TILE/CACHELINE -> unit stride; STRIDED -> stride_k; RANDOM -> worst-case line-per-element.
    assert AccessShape.from_claim(_claim(StrideClass.UNIT), AVX).stride == 1
    assert AccessShape.from_claim(_claim(StrideClass.SCALAR), AVX).stride == 1
    assert AccessShape.from_claim(_claim(StrideClass.TILE), AVX).stride == 1
    assert AccessShape.from_claim(_claim(StrideClass.STRIDED, sk=7), AVX).stride == 7
    gather = AccessShape.from_claim(_claim(StrideClass.RANDOM), AVX)
    assert gather.stride == AVX.cacheline // AVX.elem_bytes  # a fresh line per element (16)


# --- the unit sweep is a clean no-op signal -------------------------------------------------------


def test_unit_sweep_has_zero_waste_zero_conflict_zero_cost():
    p = predict(_claim(StrideClass.UNIT), AVX)
    assert p.line_waste_q8 == 0 and p.bank_conflict_q8 == 0
    assert p.contention_cost == 0 and p.is_conflict_free and p.is_noop


def test_scalar_and_tile_are_also_conflict_free():
    for sc in (StrideClass.SCALAR, StrideClass.TILE, StrideClass.CACHELINE):
        p = predict(_claim(sc), AVX)
        assert p.contention_cost == 0 and p.is_noop, sc


# --- bank conflicts: a power-of-2 bank-colliding stride prices higher than a conflict-free one ----


def test_pow2_bank_colliding_stride_prices_higher_contention_than_conflict_free():
    # stride == #banks (4) collides on ONE bank -> maximal serialization; a coprime stride (3) reaches all
    # banks -> conflict-free. The colliding stride MUST price strictly higher bank-conflict contention.
    colliding = predict(_claim(StrideClass.STRIDED, sk=AVX.mem_channels), AVX)  # stride 4 == #banks
    coprime = predict(_claim(StrideClass.STRIDED, sk=3), AVX)  # 3 coprime to 4
    assert coprime.bank_conflict_q8 == 0  # gcd(3,4)=1 -> reaches all banks -> no conflict
    assert colliding.bank_conflict_q8 > 0  # gcd(4,4)=4 -> 1 bank -> serialized
    assert colliding.contention_cost > coprime.contention_cost


def test_bank_conflict_grows_with_the_shared_factor():
    # gcd(stride, banks) is the serialization degree: a stride sharing a larger factor with #banks conflicts
    # harder. With 4 banks: stride 1/3 (coprime) -> 0; stride 2 (gcd 2) -> mid; stride 4/8 (gcd 4) -> max.
    pred = CachePredictor(cacheline=64, banks=4)

    def conflict(s):
        return pred.bank_conflict_q8(AccessShape(stride=s, elem_bytes=4, count=N))

    assert conflict(1) == 0 and conflict(3) == 0  # coprime -> conflict-free
    assert 0 < conflict(2) < conflict(4)  # gcd 2 < gcd 4
    assert conflict(8) == conflict(4)  # a multiple of #banks also hits one bank


def test_multiple_of_bank_count_serializes():
    # the classic conflict: a stride that is a MULTIPLE of the bank count keeps hitting the same bank.
    for s in (4, 8, 12, 16):  # all multiples of 4 banks
        p = predict(_claim(StrideClass.STRIDED, sk=s), AVX)
        assert p.bank_conflict_q8 > 0, s


# --- cache-line waste: a strided gather wastes more lines than a unit sweep -----------------------


def test_strided_gather_prices_higher_cacheline_waste_than_unit_sweep():
    unit = predict(_claim(StrideClass.UNIT), AVX)
    gather = predict(_claim(StrideClass.RANDOM), AVX)  # a fresh line per element
    assert gather.line_waste_q8 > unit.line_waste_q8 == 0
    assert gather.contention_cost > unit.contention_cost
    # the gather is the worst case: one line per element -> waste == (elems_per_line - 1) in Q8.
    epl = AVX.cacheline // AVX.elem_bytes
    assert gather.line_waste_q8 == (epl - 1) * 256


def test_cacheline_waste_grows_with_stride_then_saturates():
    pred = CachePredictor(cacheline=64, banks=4)  # 16 f32 per line

    def waste(s):
        return pred.line_waste_q8(AccessShape(stride=s, elem_bytes=4, count=N))

    assert waste(1) == 0  # unit sweep: full utilization
    assert waste(2) < waste(4) < waste(16)  # more lines dragged per useful element
    assert waste(32) == waste(16)  # saturates at one line per element (epl=16)


def test_strided_access_prices_higher_than_unit_on_the_combined_signal():
    # the headline ranking: a strided/colliding access ranks COSTLIER than a contiguous sweep.
    unit = predict(_claim(StrideClass.UNIT), AVX).contention_cost
    strided = predict(_claim(StrideClass.STRIDED, sk=4), AVX).contention_cost
    gather = predict(_claim(StrideClass.RANDOM), AVX).contention_cost
    assert unit < strided < gather


def test_contention_scales_with_count():
    small = predict(_claim(StrideClass.STRIDED, sk=4, n=64), AVX).contention_cost
    big = predict(_claim(StrideClass.STRIDED, sk=4, n=4096), AVX).contention_cost
    assert big > small  # more elements -> more wasted lines / conflicts


# --- reproducibility: same claim + params => same Q8 cost, every time -----------------------------


def test_predictor_is_deterministic():
    c = _claim(StrideClass.STRIDED, sk=6)
    a = predict(c, AVX)
    b = predict(c, AVX)
    assert (a.line_waste_q8, a.bank_conflict_q8, a.contention_cost) == (
        b.line_waste_q8,
        b.bank_conflict_q8,
        b.contention_cost,
    )
    assert a == b


def test_frozen_params_fingerprint_is_stable():
    assert CachePredictor.for_target(AVX).fingerprint == CachePredictor.for_target(AVX).fingerprint
    assert (
        CachePredictor(cacheline=64, banks=4).fingerprint
        != CachePredictor(cacheline=128, banks=4).fingerprint
    )


def test_integer_only_no_floats_in_the_signal():
    p = predict(_claim(StrideClass.STRIDED, sk=4), AVX)
    assert isinstance(p.line_waste_q8, int) and isinstance(p.bank_conflict_q8, int)
    assert isinstance(p.contention_cost, int)


# --- THE QUARANTINE PROOF: the signal NEVER changes a legality verdict ----------------------------


def _augment_with_contention(module, result, pred):
    """Add the predicted CONTENTION cost to each plan step's cost vector (the opt-in consultation) and return
    the augmented RealizationResult -- the plan as a caller who CONSULTED the signal would see it."""
    claims = {c.id: c for ph in module.phases for c in ph.claims}
    steps, score = [], 0
    for s in result.steps:
        cv = pred.cost_vector(AccessShape.from_claim(claims[s.claim_id], AVX))
        cand = replace(s.candidate, base=s.candidate.base + cv)
        cost = s.cost + cv.v[CONTENTION]
        steps.append(replace(s, candidate=cand, cost=cost))
        score += cost
    return replace(result, steps=steps, score=score)


def test_quarantine_legality_is_identical_with_and_without_the_signal():
    # feed a conflict-prone AND a conflict-free plan through the verifier; both are equally legal regardless
    # of the contention signal -- the predicted cost NEVER becomes a legality verdict (the R-law quarantine).
    pred = CachePredictor.for_target(AVX)
    for sc, sk in ((StrideClass.UNIT, 1), (StrideClass.STRIDED, 4), (StrideClass.RANDOM, 1)):
        m = _module(sc, sk)
        # RANDOM/STRIDED legality: keep the lane the planner chose for the claim's geometry.
        res = optimize(m, AVX, Theta.cool())
        base_diags = [str(d) for d in verify_plan(m, res)]
        aug = _augment_with_contention(m, res, pred)
        aug_diags = [str(d) for d in verify_plan(m, aug)]
        assert base_diags == aug_diags, (
            sc,
            base_diags,
            aug_diags,
        )  # legality UNCHANGED by the signal


def test_quarantine_signal_only_moves_the_contention_axis():
    # the augmented plan differs from the base plan ONLY on the CONTENTION axis -- every other cost dim
    # (memory/compute/...) is byte-identical, so the signal cannot perturb the pinned memory/compute scores.
    pred = CachePredictor.for_target(AVX)
    m = _module(StrideClass.STRIDED, sk=4)
    res = optimize(m, AVX, Theta.cool())
    aug = _augment_with_contention(m, res, pred)
    for s_base, s_aug in zip(res.steps, aug.steps):
        for d in range(len(s_base.candidate.base.v)):
            if d == CONTENTION:
                assert s_aug.candidate.base.v[d] >= s_base.candidate.base.v[d]  # contention rose
            else:
                assert (
                    s_aug.candidate.base.v[d] == s_base.candidate.base.v[d]
                )  # everything else identical


def test_conflict_prone_and_conflict_free_plans_are_equally_legal():
    # the strongest form: a conflict-prone (strided) and a conflict-free (unit) module both verify clean,
    # signal or no signal -- the predictor ranks one costlier but neither is more/less LEGAL.
    pred = CachePredictor.for_target(AVX)
    m_free, m_prone = _module(StrideClass.UNIT), _module(StrideClass.STRIDED, sk=4)
    for m in (m_free, m_prone):
        res = optimize(m, AVX, Theta.cool())
        assert verify_plan(m, res) == []  # legal without the signal
        assert verify_plan(m, _augment_with_contention(m, res, pred)) == []  # ... and with it


# --- OPT-IN / GAINS-ONLY: the default path never sets CONTENTION (pinned scores unperturbed) -------


def test_default_optimize_never_sets_the_contention_axis():
    # the signal is opt-in: nothing in the default planner consults the predictor, so every candidate's
    # CONTENTION axis stays 0 and every existing pinned plan score is byte-identical.
    for sc, sk in ((StrideClass.UNIT, 1), (StrideClass.STRIDED, 4), (StrideClass.RANDOM, 1)):
        m = _module(sc, sk)
        res = optimize(m, AVX, Theta.cool())
        assert all(s.candidate.base.v[CONTENTION] == 0 for s in res.steps), (sc, sk)


def test_default_candidates_carry_no_contention():
    # realize.candidates_for (the default candidate generator) never produces a contention term -- the
    # predictor is a SEPARATE, additive contribution, not baked into the base costs.
    for sc in StrideClass:
        c = _claim(sc, sk=4)
        for cand in candidates_for(c, AVX):
            assert cand.base.v[CONTENTION] == 0, (sc, cand.name)


def test_predicting_a_claim_does_not_mutate_it_or_the_candidates():
    # consulting the signal is pure: the claim and its candidates are unchanged after a prediction + ranking.
    c = _claim(StrideClass.STRIDED, sk=4)
    before = candidates_for(c, AVX)
    before_costs = [tuple(cand.base.v) for cand in before]
    predict(c, AVX)
    rank_realizations(before, c, AVX)
    after_costs = [tuple(cand.base.v) for cand in before]
    assert before_costs == after_costs  # candidates untouched (no mutation)


# --- the proof-carrying record + the opt-in ranking helper ----------------------------------------


def test_prediction_record_is_proof_carrying():
    p = predict(_claim(StrideClass.STRIDED, sk=4), AVX)
    assert isinstance(p, ContentionPrediction)
    assert p.shape.stride == 4 and p.shape.count == N
    assert p.contention_cost > 0 and not p.is_noop and not p.is_conflict_free
    assert p.gen == 0


def test_rank_realizations_prefers_the_conflict_free_realization():
    # a STRIDED claim has a strided candidate AND a gather candidate; the contention signal ranks the gather
    # (worst-case cache-line waste) strictly costlier, so the strided realization is preferred.
    c = _claim(StrideClass.STRIDED, sk=4)
    cands = candidates_for(c, AVX)
    ranked = rank_realizations(cands, c, AVX)
    names = [r[0].name for r in ranked]
    # the gather candidate carries the worst contention -> it cannot rank first when a cheaper one exists.
    gather = next(r for r in ranked if r[0].lane == Lane.GGG)
    others = [r for r in ranked if r[0].lane != Lane.GGG]
    if others:
        assert gather[2] >= max(o[2] for o in others)  # gather's contention is the largest
        assert ranked[0][0].lane != Lane.GGG  # the cheapest realization is not the gather
    # the ranking is by (total, name): non-decreasing total.
    assert all(ranked[i][3] <= ranked[i + 1][3] for i in range(len(ranked) - 1))


def test_rank_realizations_is_a_noop_ranking_for_a_unit_sweep():
    # a unit-stride claim has 0 contention on every candidate, so ranking is by base cost alone (the signal
    # adds nothing) -- the gains-only shape: the predictor never makes the simple, contiguous case costlier.
    c = _claim(StrideClass.UNIT)
    ranked = rank_realizations(candidates_for(c, AVX), c, AVX)
    assert all(r[2] == 0 for r in ranked)  # every candidate has 0 contention
    # ranked cheapest-first by base cost (contention is a flat 0).
    assert all(ranked[i][1] <= ranked[i + 1][1] for i in range(len(ranked) - 1))


def test_ranking_does_not_change_legality_or_the_base_costs():
    # ranking is purely advisory: it returns (cand, base, contention, total) WITHOUT mutating any candidate's
    # cost vector, so the plan the default optimizer would build is unchanged.
    c = _claim(StrideClass.STRIDED, sk=4)
    cands = candidates_for(c, AVX)
    snapshot = [tuple(cand.base.v) for cand in cands]
    ranked = rank_realizations(cands, c, AVX)
    assert [tuple(cand.base.v) for cand in cands] == snapshot  # base costs untouched
    for cand, base, cont, total in ranked:
        assert total == base + cont  # the augmented total is base + signal


# --- target portability: the frozen geometry tracks the target -----------------------------------


def test_predictor_tracks_target_bank_count():
    # the conflict degree is the SERIALIZATION FACTOR g = gcd(stride, banks) -- the throughput lost vs the
    # machine's own ideal. A stride == the SMALLER bank count fully serializes the 4-bank CPU (gcd 4 -> 1
    # bank) but the 32-bank GPU still spreads it across more banks at a larger stride. We pin the geometry
    # dependence with a stride that is a full multiple of the GPU's bank count.
    gpu = TargetProfile.nvidia_ptx()  # mem_channels 32, cacheline 128
    cpu_pred = CachePredictor.for_target(AVX)  # banks 4
    gpu_pred = CachePredictor.for_target(gpu)  # banks 32
    # a stride == the GPU bank count (32) FULLY serializes the GPU (gcd 32,32 -> 1 bank) but the CPU saturates
    # at its own 4-bank ceiling (gcd 32,4 = 4 -> already 1 bank) -- the GPU's worst case is MORE serialized.
    shape32 = AccessShape(stride=32, elem_bytes=4, count=N)
    assert gpu_pred.bank_conflict_q8(shape32) > cpu_pred.bank_conflict_q8(shape32) > 0
    # a stride coprime to both is conflict-free everywhere (the geometry-independent conflict-free case).
    coprime = AccessShape(stride=3, elem_bytes=4, count=N)
    assert cpu_pred.bank_conflict_q8(coprime) == 0 and gpu_pred.bank_conflict_q8(coprime) == 0
