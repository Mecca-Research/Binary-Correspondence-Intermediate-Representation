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


# --- the law rail recomputes the digest (R13 digest recompute, -bcir-verify) ----

def _fnv_outer(components, artifacts):
    """An independent re-implementation of the FNV-1a chain that the C++ -bcir-verify
    R13 digest recompute mirrors (provenance._digest over the component hashes + the
    normalized (name, generation) artifact pairs). Pins the byte-level contract."""
    h, prime, mask = 14695981039346656037, 1099511628211, (1 << 64) - 1
    items = [str(c) for c in components]
    for name, gen in artifacts:
        items += [str(name), str(gen)]
    for it in items:
        for byte in it.encode("utf-8"):
            h = ((h ^ byte) * prime) & mask
        h = ((h ^ 0xFF) * prime) & mask
    return h & ((1 << 63) - 1)


def test_digest_is_the_fnv_chain_of_the_component_hashes():
    """The digest is exactly the FNV-1a chain of the four component hashes (+ artifacts) --
    the relation -bcir-verify recomputes so the law no longer trusts the declared digest."""
    man = build_manifest(vector_add(1024), AVX, COOL)
    comps = (man.m_module, man.m_target, man.m_theta, man.m_policy)
    assert _fnv_outer(comps, man.artifacts) == man.digest
    forged = replace(man, digest=man.digest ^ 0xABCD)
    assert _fnv_outer(comps, forged.artifacts) != forged.digest   # a tampered digest is caught


def test_mlir_verify_provenance_constants_are_in_sync():
    """The constants hard-coded in mlir/test/passes/verify_provenance.mlir are a real
    vector_add manifest's hashes; this pins them so the FileCheck cannot silently rot."""
    man = build_manifest(vector_add(1024), AVX, COOL)
    assert (man.m_module, man.m_target, man.m_theta, man.m_policy) == (
        7127522701151166272, 5864064355688965777, 1870846051561339781, 4048695575545564183)
    assert man.digest == 9201837206445197944          # the no-artifact case
    # the with-artifacts case folds (cal_gen, 4) and (map_gen, 2) in, sorted by name.
    arts = (("cal_gen", 4), ("map_gen", 2))
    assert _fnv_outer((man.m_module, man.m_target, man.m_theta, man.m_policy), arts) \
        == 3780911091132933688


def test_component_hashes_recompute_from_ir_primitives():
    """Each component hash is the FNV-1a chain over exactly the primitives the IR carries
    (resource/claim fields incl. the opcode; capability fields incl. name + scalable; policy
    name + UNFOLDED base) -- the relation -bcir-verify recomputes from the IR for the
    m_module / m_target / m_policy cross-checks (R13; BCIRVerifyPass.cpp hash*FromIR)."""
    from bcir.kbcir.provenance import hash_module, hash_target, hash_policy
    from bcir.kbcir.weights import PERF

    def fnv(scalars):
        h, prime, mask = 14695981039346656037, 1099511628211, (1 << 64) - 1
        for it in scalars:
            for b in str(it).encode("utf-8"):
                h = ((h ^ b) * prime) & mask
            h = ((h ^ 0xFF) * prime) & mask
        return h & ((1 << 63) - 1)

    m, H = vector_add(1024), AVX
    mod = [m.name, m.cacheline, m.align]
    for r in sorted(m.resources.values(), key=lambda r: r.rid):
        mod += [r.rid, int(r.domain), *r.shape, r.layout, r.align, r.access,
                r.priority, r.map_gen, r.data_gen]
    for ph in m.phases:
        mod += [ph.phase_id, *sorted(ph.deps)]
        for c in sorted(ph.claims, key=lambda c: c.id):
            mod += [c.id, int(c.opcode), int(c.lane), int(c.stride_class), c.count, c.stride_k,
                    *c.rd, *c.wr, c.hazard, int(c.domain), c.verify, c.bounds, c.op,
                    c.offset, c.cost_class]
    assert fnv(mod) == hash_module(m) == 7127522701151166272
    tgt = [H.name, H.triple, H.cacheline, H.elem_bytes, *sorted(H.lane_widths), H.warp,
           H.scalable, H.gather_penalty, H.mem_unit, H.base_overhead, H.thermal_density,
           H.power_density, H.per_op_heat, H.affinity_domains,
           getattr(H, "mem_channels", 4), getattr(H, "cal_gen", 0)]
    assert fnv(tgt) == hash_target(H) == 5864064355688965777
    assert fnv([PERF.name, *PERF.base]) == hash_policy(PERF) == 4048695575545564183
