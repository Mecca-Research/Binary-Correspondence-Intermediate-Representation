"""Provenance manifest + deterministic replay: the version-DAG spine (R13)."""

from dataclasses import replace

from bcir.examples import vector_add
from bcir.kbcir import (
    ProvenanceManifest,
    ProvenanceMismatch,
    TARGETS,
    build_manifest,
    manifest_for,
    optimize,
    reproduces,
)
from bcir.kbcir import replay as replay_plan
from bcir.kbcir.cost import Theta
from bcir.kbcir.microbench import reference_table
from bcir.verify import verify_manifest

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def _laws(diags):
    return {d.law for d in diags}


# --- the manifest is the commit hash of a plan ----------------------------------

def test_manifest_is_deterministic_and_pins_the_plan():
    m = vector_add(1024)
    a = build_manifest(m, AVX, COOL)
    b = build_manifest(m, AVX, COOL)
    assert a == b                                   # same inputs -> same commit
    assert a.score == 7808 and a.widths == ((1000, 16),)


def test_manifest_equality_implies_identical_plan():
    m = vector_add(1024)
    man = build_manifest(m, AVX, COOL)
    assert reproduces(man, m, AVX, COOL)            # same digest -> same plan
    plan = replay_plan(man, m, AVX, COOL)
    assert plan.score == man.score == optimize(m, AVX, COOL).score


def test_changed_input_changes_the_digest_and_fails_replay():
    m = vector_add(1024)
    cool = build_manifest(m, AVX, COOL)
    assert not reproduces(cool, m, AVX, Theta.hot())     # different commit
    try:
        replay_plan(cool, m, AVX, Theta.hot())
        assert False, "expected ProvenanceMismatch"
    except ProvenanceMismatch:
        pass


def test_diff_pinpoints_the_changed_component():
    m = vector_add(1024)
    cool = build_manifest(m, AVX, COOL)
    hot = build_manifest(m, AVX, Theta.hot())
    assert cool.diff(hot) == ["theta"]              # only Theta moved


def test_artifacts_are_part_of_the_commit():
    m = vector_add(1024)
    plain = build_manifest(m, AVX, COOL)
    tagged = build_manifest(m, AVX, COOL, artifacts=[("gate", 99), ("cal_gen", 3)])
    assert plain.digest != tagged.digest
    assert plain.diff(tagged) == ["artifacts"]
    assert plain.score == tagged.score              # same plan, different provenance


def test_json_round_trips():
    man = build_manifest(vector_add(1024), AVX, COOL, artifacts=[("gate", 7)])
    assert ProvenanceManifest.from_json(man.to_json()) == man


# --- the version DAG (immutable within a generation; branches across them) -------

def test_calibrated_vs_seeded_is_a_distinct_branch_same_value():
    # Same module/theta/policy under the seeded constants vs a calibrated table:
    # two distinct commits (the version DAG), each reproducing its own plan. The
    # plan VALUE is the same here (vector_add is unit-stride), the PROVENANCE is not.
    m = vector_add(1024)
    table = reference_table()
    seeded = build_manifest(m, AVX, COOL)
    calibrated = manifest_for(m, table.apply(AVX), COOL, table=table)
    assert seeded.digest != calibrated.digest
    assert "target" in seeded.diff(calibrated) and "artifacts" in seeded.diff(calibrated)
    assert seeded.score == calibrated.score == 7808
    assert reproduces(seeded, m, AVX, COOL)
    assert reproduces(calibrated, m, table.apply(AVX), COOL,
                      artifacts=calibrated.artifacts)


def test_manifest_for_assembles_artifact_tags():
    from bcir.gem import hydrate
    m = vector_add(1024)
    res = optimize(m, AVX, COOL)
    pack = hydrate(m, res)
    man = manifest_for(m, AVX, COOL, pack=pack)
    names = {n for n, _ in man.artifacts}
    assert {"topo_gen", "map_gen", "data_gen"} <= names


# --- R13: the manifest law -------------------------------------------------------

def test_faithful_manifest_satisfies_R13():
    m = vector_add(1024)
    man = build_manifest(m, AVX, COOL)
    assert verify_manifest(man, m, AVX, COOL) == []


def test_tampered_digest_is_R13():
    m = vector_add(1024)
    forged = replace(build_manifest(m, AVX, COOL), digest=123456789)
    assert "R13" in _laws(verify_manifest(forged, m, AVX, COOL))


def test_nonreproducible_score_is_R13():
    # A manifest whose recorded score does not match the replayed optimum: the
    # plan is not reproducible from its stated provenance.
    m = vector_add(1024)
    forged = replace(build_manifest(m, AVX, COOL), score=9999)
    assert "R13" in _laws(verify_manifest(forged, m, AVX, COOL))


def test_every_target_manifest_reproduces_and_verifies():
    m = vector_add(1024)
    for h in TARGETS.values():
        man = build_manifest(m, h, COOL)
        assert reproduces(man, m, h, COOL)
        assert verify_manifest(man, m, h, COOL) == []
