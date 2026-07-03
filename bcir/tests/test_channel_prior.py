"""D3 slice 3: CHANNEL-CHOICE priors + per-shape-class tables (`kbcir/channel_prior.py`).

The tile-prior discipline applied to the tower's channel decision: a per-shape-class TABLE
(the `shape_class` key minted for exactly this) answers a trained class with ZERO per-channel
optimize() pricings; a miss prices every suitable channel (prior-ordered, exhaustively exact);
the certificate proves guided == exhaustive with mismatches 0 and counts the earned pricing
reduction; a poisoned table is CAUGHT; the persisted envelope ties to the TOWER's
(name, cal_gen) pairs and refuses staleness/re-towering/newer schemas loudly."""

import dataclasses
import os
import tempfile

from bcir.channels import CHANNELS
from bcir.kbcir.channel_prior import (ChannelPriorCertificate, FrozenChannelPrior,
                                      build_channel_table, channel_prior_certificate,
                                      guided_plan_channel, load_channel_prior,
                                      prior_channel_samples, save_channel_prior,
                                      train_channel_prior)
from bcir.kbcir.cost import Theta
from bcir.kbcir.tile_prior import shape_class

TOWER = [CHANNELS["x86_avx512"], CHANNELS["arm64_neon"], CHANNELS["nvidia_ptx"]]
COOL = Theta.cool()
TRAIN = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (32, 64, 128),
         (128, 32, 16), (8, 256, 8), (64, 16, 256), (256, 256, 64)]
HELD_OUT = [(24, 24, 24), (48, 48, 48), (160, 40, 20), (12, 300, 12),
            (96, 24, 96), (320, 320, 80)]                # same classes, different shapes


def _frozen():
    return train_channel_prior(prior_channel_samples(TRAIN, TOWER, COOL)).freeze()


def test_guided_choice_matches_the_exact_optimum_with_real_reduction():
    """The safety law + the earned win: over held-out shapes (whose CLASSES the table saw,
    whose shapes it did not) the guided choice equals the exhaustive argmin with mismatches
    0, at a measured 83% fewer per-channel optimize() pricings (gate >= 50%)."""
    table = build_channel_table(TRAIN, TOWER, COOL)
    cert = channel_prior_certificate(HELD_OUT, TOWER, COOL, table=table, prior=_frozen())
    assert cert.admitted, cert
    assert cert.mismatches == 0
    assert cert.checked == len(HELD_OUT)
    assert cert.reduction >= 0.50, cert                  # measured 0.83
    assert cert.priced_guided < cert.priced_exhaustive


def test_an_unseen_shape_class_degenerates_honestly():
    """A shape whose class the table never saw prices EVERY suitable channel (no zero-cost
    guess) and still returns the exhaustive argmin -- with or without the ordering prior."""
    table = build_channel_table(TRAIN, TOWER, COOL)
    assert shape_class(1024, 1024, 1024) not in table
    exact, ne = guided_plan_channel(1024, 1024, 1024, TOWER, COOL)
    ch, n = guided_plan_channel(1024, 1024, 1024, TOWER, COOL, table=table, prior=_frozen())
    assert n == ne == len(TOWER)                         # degenerated, honestly
    assert ch.name == exact.name
    # and an empty tower refuses rather than guessing.
    try:
        guided_plan_channel(8, 8, 8, [], COOL)
        raise AssertionError("an empty tower must be rejected")
    except ValueError as e:
        assert "no suitable channel" in str(e)


def test_a_poisoned_table_is_caught_by_the_certificate():
    """The certificate is a CHECK, not a trust: a table entry pointing at the wrong channel
    produces mismatches >= 1 and the certificate is NOT admitted (the AccelCertificate
    pattern -- a lying artifact never ships)."""
    table = build_channel_table(TRAIN, TOWER, COOL)
    exact, _ = guided_plan_channel(*HELD_OUT[0], TOWER, COOL)
    wrong = next(c.name for c in TOWER if c.name != exact.name)
    poisoned = dict(table)
    poisoned[shape_class(*HELD_OUT[0])] = wrong
    cert = channel_prior_certificate(HELD_OUT, TOWER, COOL, table=poisoned)
    assert cert.mismatches >= 1
    assert not cert.admitted


def test_the_frozen_prior_is_deterministic_and_integer():
    a, b = _frozen(), _frozen()
    assert a == b
    assert all(isinstance(v, int) for v in a.wq)
    # a zero prior is a legal (useless) prior: ordering degrades, exactness does not.
    zero = FrozenChannelPrior(wq=(0,) * 8)
    ch, n = guided_plan_channel(48, 48, 48, TOWER, COOL, prior=zero)
    exact, _ = guided_plan_channel(48, 48, 48, TOWER, COOL)
    assert ch.name == exact.name and n == len(TOWER)


def test_persisted_prior_round_trips_and_refuses_staleness():
    """The envelope law: kind/schema pinned; round-trip equality; a recalibrated channel
    (cal_gen bump) -> STALE; a different tower -> retrain; a newer schema -> upgrade."""
    import json
    table = build_channel_table(TRAIN, TOWER, COOL)
    pri = _frozen()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prior.json")
        save_channel_prior(path, pri, table, TOWER)
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        assert doc["kind"] == "bcir.channel_prior" and doc["schema"] == 1
        back, tback = load_channel_prior(path, TOWER)
        assert back == pri and tback == table
        # a recalibrated channel: same tower names, a bumped cal_gen -> STALE.
        bumped_prof = dataclasses.replace(TOWER[0].profile, cal_gen=7)
        bumped = [dataclasses.replace(TOWER[0], profile=bumped_prof)] + TOWER[1:]
        try:
            load_channel_prior(path, bumped)
            raise AssertionError("a recalibrated tower must refuse the stale prior")
        except ValueError as e:
            assert "STALE" in str(e)
        # a different tower -> retrain.
        try:
            load_channel_prior(path, TOWER[:2])
            raise AssertionError("a re-towered load must refuse")
        except ValueError as e:
            assert "retrain" in str(e)
        # a newer schema -> upgrade BCIR.
        doc["schema"] = 99
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        try:
            load_channel_prior(path, TOWER)
            raise AssertionError("a newer schema must refuse")
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
