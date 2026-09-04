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
from bcir.kbcir.channel_prior import (
    ChannelPriorCertificate,
    FrozenChannelPrior,
    build_channel_table,
    channel_prior_certificate,
    guided_plan_channel,
    load_channel_prior,
    prior_channel_samples,
    save_channel_prior,
    train_channel_prior,
)
from bcir.kbcir.cost import Theta
from bcir.kbcir.tile_prior import shape_class

TOWER = [CHANNELS["x86_avx512"], CHANNELS["arm64_neon"], CHANNELS["nvidia_ptx"]]
COOL = Theta.cool()
TRAIN = [
    (16, 16, 16),
    (32, 32, 32),
    (64, 64, 64),
    (32, 64, 128),
    (128, 32, 16),
    (8, 256, 8),
    (64, 16, 256),
    (256, 256, 64),
]
HELD_OUT = [
    (24, 24, 24),
    (48, 48, 48),
    (160, 40, 20),
    (12, 300, 12),
    (96, 24, 96),
    (320, 320, 80),
]  # same classes, different shapes


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
    assert cert.reduction >= 0.50, cert  # measured 0.83
    assert cert.priced_guided < cert.priced_exhaustive


def test_an_unseen_shape_class_degenerates_honestly():
    """A shape whose class the table never saw prices EVERY suitable channel (no zero-cost
    guess) and still returns the exhaustive argmin -- with or without the ordering prior."""
    table = build_channel_table(TRAIN, TOWER, COOL)
    assert shape_class(1024, 1024, 1024) not in table
    exact, ne = guided_plan_channel(1024, 1024, 1024, TOWER, COOL)
    ch, n = guided_plan_channel(1024, 1024, 1024, TOWER, COOL, table=table, prior=_frozen())
    assert n == ne == len(TOWER)  # degenerated, honestly
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


def test_persisted_prior_rejects_ambiguous_or_malformed_installable_state():
    import json

    table = build_channel_table(TRAIN, TOWER, COOL)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prior.json")
        save_channel_prior(path, _frozen(), table, TOWER)
        with open(path, encoding="utf-8") as f:
            good = json.load(f)

        bad_docs = []
        d = json.loads(json.dumps(good))
        d["wq"] = [0]
        bad_docs.append(d)
        d = json.loads(json.dumps(good))
        d["wq"][0] = True
        bad_docs.append(d)
        d = json.loads(json.dumps(good))
        d["table"] = {"1,2": TOWER[0].name}
        bad_docs.append(d)
        d = json.loads(json.dumps(good))
        d["table"] = {"1,2,3": "ghost"}
        bad_docs.append(d)
        d = json.loads(json.dumps(good))
        d["tower"].append(d["tower"][0])
        bad_docs.append(d)
        for bad in bad_docs:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bad, f)
            try:
                load_channel_prior(path, TOWER)
                raise AssertionError("malformed installable channel prior must be rejected")
            except ValueError:
                pass

        raw = json.dumps(good)
        raw = raw.replace('"schema": 1', '"schema": 1, "schema": 1', 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(raw)
        try:
            load_channel_prior(path, TOWER)
            raise AssertionError("duplicate keys must be rejected")
        except ValueError as e:
            assert "duplicate JSON key" in str(e)

        with open(path, "wb") as f:
            f.write(b" " * ((1 << 20) + 1))
        try:
            load_channel_prior(path, TOWER)
            raise AssertionError("oversized priors must be rejected")
        except ValueError as e:
            assert "exceeds" in str(e)

    for weights in ((0,), (0,) * 9, (False,) * 8):
        try:
            FrozenChannelPrior(weights)
            raise AssertionError("malformed in-memory priors must be rejected")
        except ValueError:
            pass


# --- D3.4: the table wired into orchestrate + calibrated towers -------------------------------


def _cal_tower():
    """A CALIBRATED tower (cal_gen 1 -- D3.4's second half): the host profile is
    strictly the cheapest gemm channel (mem_unit 1 vs the gpu's 2), so the exhaustive
    winner is unambiguous (no cost ties for the tie-break laws to disagree over)."""
    x86, gpu = CHANNELS["x86_avx512"], CHANNELS["nvidia_ptx"]
    host = dataclasses.replace(
        x86,
        name="host_cal",
        profile=dataclasses.replace(x86.profile, name="host_cal", mem_unit=1, cal_gen=1),
    )
    wide = dataclasses.replace(
        gpu,
        name="gpu_cal",
        profile=dataclasses.replace(gpu.profile, name="gpu_cal", mem_unit=2, cal_gen=1),
    )
    return [host, wide, CHANNELS["arm64_neon"]]


def _gemm_fixture(M, N, K):
    """A single-gemm module + its dims record (the orchestrate_guided interface: the
    module's author declares the gemm dimensions it already knows)."""
    from bcir.kbcir.calling_side import _gemm_claim
    from bcir.model import Domain, Module, Phase, Resource

    m = Module(name=f"gemm_{M}x{N}x{K}")
    for rid, nm in ((1, "A"), (2, "B"), (3, "C")):
        m.add_resource(Resource(rid=rid, domain=Domain.RAM, shape=(max(1, M * N),), name=nm))
    m.add_phase(Phase(phase_id=0, deps=(), claims=[_gemm_claim(M, N, K)]))
    return m, {9_300: (M, N, K)}


def test_d3_4_the_table_wires_into_orchestrate_with_zero_mismatches():
    """D3.4 first half: guided orchestrate == exhaustive orchestrate CLAIM FOR CLAIM
    (mismatches 0) at a measured >= 30% reduction in whole-module optimize() runs (a
    table hit prices ONE channel instead of the tower; an unseen class falls back to
    the full tower -- exactness is never delegated); a POISONED table changes
    placements and is CAUGHT by the certificate, never trusted."""
    from bcir.kbcir.channel_prior import orchestrate_guided, orchestrate_prior_certificate
    from bcir.kbcir.train_graph import TrainStepSpec, train_step_module

    tower = _cal_tower()
    table = build_channel_table(TRAIN, tower, COOL)
    fixtures = [_gemm_fixture(*s) for s in HELD_OUT[:3]]
    fixtures.append((train_step_module(TrainStepSpec()), {1: (8, 1, 4)}))  # unseen class
    cert = orchestrate_prior_certificate(fixtures, tower, COOL, table=table)
    assert cert.admitted, (cert.mismatches, cert.checked)
    assert cert.checked >= 9  # 3 gemms + the step's claims
    assert cert.reduction >= 0.30, cert.reduction
    _, priced = orchestrate_guided(fixtures[0][0], tower, COOL, table=table, dims=fixtures[0][1])
    assert priced == 1  # a hit prices ONE channel
    poisoned = {k: "gpu_cal" for k in table}  # strictly-worse winners
    bad = orchestrate_prior_certificate(fixtures[:3], tower, COOL, table=poisoned)
    assert not bad.admitted and bad.mismatches > 0  # caught, not trusted


def test_d3_4_calibrated_towers_hold_the_laws_and_the_uniformity_is_recorded():
    """D3.4 second half, MEASURED not asserted: under per-channel CALIBRATED profiles
    (cal_gen 1) the table builds, certifies (slice-3 certificate, 0 mismatches), and
    the staleness law bites (a prior saved under the stock tower refuses under the
    calibrated one). And the recorded truth: gemm class winners stay TOWER-UNIFORM
    under the L2 linear cost model -- the wrapped-gemm cost is memory-dominated and
    linear in the output budget, so one calibrated channel dominates EVERY size class;
    per-class divergence requires the L3 tile/cache model (`cost_of`'s cache-fitting
    term) -- the named follow-on, recorded, not hidden."""
    tower = _cal_tower()
    table = build_channel_table(TRAIN, tower, COOL)
    assert table and set(table.values()) == {"host_cal"}  # uniform: measured truth
    cert = channel_prior_certificate(HELD_OUT, tower, COOL, table=table)
    assert cert.admitted and cert.mismatches == 0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prior.json")
        pri = _frozen()
        save_channel_prior(path, pri, build_channel_table(TRAIN, TOWER, COOL), TOWER)
        try:
            load_channel_prior(path, tower)  # calibrated != stock
            raise AssertionError("a calibrated tower must refuse the stock prior")
        except ValueError as e:
            assert "retrain" in str(e) or "STALE" in str(e)


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
