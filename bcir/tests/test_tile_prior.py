"""D3 (§8.2): learned cost-model priors at L1 (`kbcir/tile_prior.py`).

The accel-ranker precedent one level down: a logistic prior over CHEAP tile features
(no cost_of call to rank), trained offline on the exact tile search's own choices, frozen
to a Q8 integer table, used only to ORDER the search with a PROOF-gated early exit. Pins:
(1) the safety law -- guided == exhaustive on (fits_cache, bottleneck) over held-out
shapes, mismatches 0, while costing ~70% fewer nodes (gate at >=40%, measured 71%);
(2) the proof degrades honestly -- under a bandwidth-starved calibrated profile the
bottleneck floor is unreachable, the exit never fires, reduction is ~0 and the optimum
still matches (proposals reduce search only when they can PROVE it);
(3) the calibloop wiring -- retrained under the calibrated profile, the certificate holds
there too (the prior follows the measured cost model, never overrides the exact search);
(4) frozen determinism -- the Q8 table gives one stable order, twice."""

import dataclasses

from bcir.kbcir.cost import TargetProfile
from bcir.kbcir.tile_prior import (FrozenTilePrior, guided_plan_matmul, prior_samples,
                                   tile_prior_certificate, train_tile_prior)

AVX = TargetProfile.x86_avx512()
TRAIN = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (32, 64, 128),
         (128, 32, 16), (8, 256, 8), (64, 16, 256)]
HELD_OUT = [(48, 48, 48), (16, 128, 64), (256, 64, 32), (96, 24, 96),
            (8, 8, 512), (160, 160, 16)]


def _frozen(target=AVX):
    return train_tile_prior(prior_samples(TRAIN, target)).freeze()


def test_guided_search_matches_the_exact_optimum_with_real_reduction():
    cert = tile_prior_certificate(HELD_OUT, _frozen(), AVX)
    assert cert.admitted and cert.mismatches == 0, cert
    assert cert.checked == len(HELD_OUT)
    assert cert.reduction >= 0.40, f"reduction {cert.reduction:.1%} (measured 71%)"
    assert cert.nodes_guided < cert.nodes_exhaustive


def test_the_early_exit_is_a_proof_not_a_heuristic():
    """A bandwidth-starved calibration makes memory bind everywhere: the bottleneck floor
    is unreachable, so the exit NEVER fires -- reduction ~0, optimum still exact. The
    prior only reduces search when it can prove the stop."""
    starved = dataclasses.replace(AVX, mem_channels=1, name="starved")
    cert = tile_prior_certificate(HELD_OUT, _frozen(starved), starved)
    assert cert.admitted and cert.mismatches == 0, cert
    assert cert.nodes_guided == cert.nodes_exhaustive       # degenerated, honestly


def test_retrained_under_the_calibrated_profile_the_certificate_still_holds():
    """The calibloop wiring: hand the trainer the measured profile and the proposals
    follow the calibrated cost model -- the exact search (same profile) verifies every
    choice, so a recalibration can never be overridden by a stale prior."""
    boosted = dataclasses.replace(AVX, mem_channels=8, name="boosted")
    cert = tile_prior_certificate(HELD_OUT, _frozen(boosted), boosted)
    assert cert.admitted and cert.mismatches == 0, cert


def test_the_frozen_table_is_deterministic_and_integer():
    a, b = _frozen(), _frozen()
    assert a == b and all(isinstance(w, int) for w in a.wq)
    assert a.order(48, 48, 48, AVX) == b.order(48, 48, 48, AVX)
    plan1, n1 = guided_plan_matmul(48, 48, 48, AVX, a)
    plan2, n2 = guided_plan_matmul(48, 48, 48, AVX, a)
    assert plan1 == plan2 and n1 == n2
    # a zero prior is a legal (useless) prior: still exact, just no reduction guarantee.
    zero = FrozenTilePrior(wq=(0,) * 8)
    cert = tile_prior_certificate(HELD_OUT[:2], zero, AVX)
    assert cert.admitted and cert.mismatches == 0


def test_persisted_prior_round_trips_and_refuses_staleness():
    """D3 slice 2: the frozen table persists as a self-describing envelope tied to the
    calibration generation; a stale cal_gen / wrong target / newer schema refuses LOUDLY
    (a stale prior's search reduction is an unearned claim); the round-tripped prior keeps
    the certificate admitted."""
    import json
    from bcir.kbcir.tile_prior import load_tile_prior, save_tile_prior
    pri = _frozen()
    text = save_tile_prior(pri, target_name=AVX.name, cal_gen=3)
    d = json.loads(text)
    assert d["kind"] == "bcir.tile_prior" and d["schema"] == 1 and d["cal_gen"] == 3
    back = load_tile_prior(text, expect_target=AVX.name, expect_cal_gen=3)
    assert back == pri
    cert = tile_prior_certificate(HELD_OUT[:3], back, AVX)
    assert cert.admitted and cert.mismatches == 0
    for kwargs, msg in ((dict(expect_target=AVX.name, expect_cal_gen=4), "STALE"),
                        (dict(expect_target="other-host", expect_cal_gen=3), "retrain")):
        try:
            load_tile_prior(text, **kwargs)
            raise AssertionError(f"must refuse: {msg}")
        except ValueError as e:
            assert msg in str(e)
    newer = json.dumps({**d, "schema": 99})
    try:
        load_tile_prior(newer, expect_target=AVX.name, expect_cal_gen=3)
        raise AssertionError("a newer schema must be refused")
    except ValueError as e:
        assert "newer" in str(e)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
