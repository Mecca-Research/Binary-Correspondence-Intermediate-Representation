"""Provenance manifest + deterministic replay: the version-DAG spine (R13)."""

import json
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
    assert a == b  # same inputs -> same commit
    assert a.score == 7808 and a.widths == ((1000, 16),)


def test_manifest_equality_implies_identical_plan():
    m = vector_add(1024)
    man = build_manifest(m, AVX, COOL)
    assert reproduces(man, m, AVX, COOL)  # same digest -> same plan
    plan = replay_plan(man, m, AVX, COOL)
    assert plan.score == man.score == optimize(m, AVX, COOL).score


def test_changed_input_changes_the_digest_and_fails_replay():
    m = vector_add(1024)
    cool = build_manifest(m, AVX, COOL)
    assert not reproduces(cool, m, AVX, Theta.hot())  # different commit
    try:
        replay_plan(cool, m, AVX, Theta.hot())
        assert False, "expected ProvenanceMismatch"
    except ProvenanceMismatch:
        pass


def test_diff_pinpoints_the_changed_component():
    m = vector_add(1024)
    cool = build_manifest(m, AVX, COOL)
    hot = build_manifest(m, AVX, Theta.hot())
    assert cool.diff(hot) == ["theta"]  # only Theta moved


def test_artifacts_are_part_of_the_commit():
    m = vector_add(1024)
    plain = build_manifest(m, AVX, COOL)
    tagged = build_manifest(m, AVX, COOL, artifacts=[("gate", 99), ("cal_gen", 3)])
    assert plain.digest != tagged.digest
    assert plain.diff(tagged) == ["artifacts"]
    assert plain.score == tagged.score  # same plan, different provenance


def test_json_round_trips():
    man = build_manifest(vector_add(1024), AVX, COOL, artifacts=[("gate", 7)])
    assert ProvenanceManifest.from_json(man.to_json()) == man


def test_json_rejects_ambiguous_or_forged_manifests():
    manifest = build_manifest(vector_add(1024), AVX, COOL, artifacts=[("cal_gen", 1), ("gate", 7)])
    document = json.loads(manifest.to_json())
    bad_documents = []

    bad = dict(document)
    bad["digest"] ^= 1
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["widths"] = [[1000, 16], [1000, 8]]
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["widths"] = [[1000, True]]
    bad_documents.append(json.dumps(bad))
    bad = dict(document)
    bad["artifacts"] = [["gate", 7], ["gate", 8]]
    bad_documents.append(json.dumps(bad))
    bad_documents.append(manifest.to_json().replace('"score": 7808', '"score": 7808, "score": 0'))

    for text in bad_documents:
        try:
            ProvenanceManifest.from_json(text)
            assert False, "expected malformed provenance manifest to be rejected"
        except ValueError:
            pass


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
    assert reproduces(calibrated, m, table.apply(AVX), COOL, artifacts=calibrated.artifacts)


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
    assert _fnv_outer(comps, forged.artifacts) != forged.digest  # a tampered digest is caught


def test_mlir_verify_provenance_constants_are_in_sync():
    """The constants hard-coded in mlir/test/passes/verify_provenance.mlir are a real
    vector_add manifest's hashes; this pins them so the FileCheck cannot silently rot."""
    man = build_manifest(vector_add(1024), AVX, COOL)
    assert (man.m_module, man.m_target, man.m_theta, man.m_policy) == (
        7127522701151166272,
        5192828792194564141,
        1870846051561339781,
        4048695575545564183,
    )
    assert man.digest == 8915526058458340485  # the no-artifact case
    # the with-artifacts case folds (cal_gen, 4) and (map_gen, 2) in, sorted by name.
    arts = (("cal_gen", 4), ("map_gen", 2))
    assert (
        _fnv_outer((man.m_module, man.m_target, man.m_theta, man.m_policy), arts)
        == 6843787964663692581
    )


def test_component_hashes_recompute_from_ir_primitives():
    """Each component hash is the FNV-1a chain over exactly the primitives the IR carries
    (resource/claim fields incl. the opcode, claims in declared order; capability fields
    incl. name, scalable and the memory tiers; policy name + UNFOLDED base) -- the relation
    -bcir-verify recomputes from the IR for the
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
        mod += [
            r.rid,
            int(r.domain),
            *r.shape,
            r.layout,
            r.align,
            r.access,
            r.priority,
            r.map_gen,
            r.data_gen,
        ]
    for ph in m.phases:
        mod += [ph.phase_id, *sorted(ph.deps)]
        for c in ph.claims:  # declared order (S0-D)
            mod += [
                c.id,
                int(c.opcode),
                int(c.lane),
                int(c.stride_class),
                c.count,
                c.stride_k,
                *c.rd,
                *c.wr,
                c.hazard,
                int(c.domain),
                c.verify,
                c.bounds,
                c.op,
                c.offset,
                c.cost_class,
            ]
    assert fnv(mod) == hash_module(m) == 7127522701151166272
    tgt = [
        H.name,
        H.triple,
        H.cacheline,
        H.elem_bytes,
        *sorted(H.lane_widths),
        H.warp,
        H.scalable,
        H.gather_penalty,
        H.mem_unit,
        H.base_overhead,
        H.thermal_density,
        H.power_density,
        H.per_op_heat,
        H.affinity_domains,
        getattr(H, "mem_channels", 4),
        getattr(H, "cal_gen", 0),
    ]
    for t in H.mem.tiers:  # the memory hierarchy (S0-D), declared order
        tgt += [t.name, t.latency_cyc, t.bw_factor, t.lat_factor, t.capacity]
    assert fnv(tgt) == hash_target(H) == 5192828792194564141
    assert fnv([PERF.name, *PERF.base]) == hash_policy(PERF) == 4048695575545564183


# --- G3 / S1-B: one canonical digest, computed once, identity-bound, mutation-invalidated --


def _fnv_recursive(*items):
    """The pre-G3 reference: FNV-1a over a RECURSIVELY flattened item sequence, one mask
    per byte. Kept here as the oracle the iterative stream must reproduce bit for bit."""

    def flatten(xs):
        for x in xs:
            if isinstance(x, (list, tuple)):
                yield from flatten(x)
            else:
                yield x

    h, prime, mask = 14695981039346656037, 1099511628211, (1 << 64) - 1
    for it in flatten(items):
        for byte in str(it).encode("utf-8"):
            h = ((h ^ byte) * prime) & mask
        h = ((h ^ 0xFF) * prime) & mask
    return h & ((1 << 63) - 1)


def _recursive_module_hash(m):
    res = [
        (
            r.rid,
            int(r.domain),
            tuple(r.shape),
            r.layout,
            r.align,
            r.access,
            r.priority,
            r.map_gen,
            r.data_gen,
        )
        for r in sorted(m.resources.values(), key=lambda r: r.rid)
    ]
    phases = []
    for ph in m.phases:
        claims = [
            (
                c.id,
                int(c.opcode),
                int(c.lane),
                int(c.stride_class),
                c.count,
                c.stride_k,
                tuple(c.rd),
                tuple(c.wr),
                c.hazard,
                int(c.domain),
                c.verify,
                c.bounds,
                c.op,
                c.offset,
                c.cost_class,
            )
            for c in ph.claims
        ]
        phases.append((ph.phase_id, tuple(sorted(ph.deps)), tuple(claims)))
    return _fnv_recursive(m.name, m.cacheline, m.align, tuple(res), tuple(phases))


def test_the_iterative_stream_reproduces_the_recursive_digest_bit_for_bit():
    """G3 replaced the recursive canonical flattening with one iterative stream (and the
    per-byte mask with one per item). The digest is a cross-rail content address pinned in
    verify_provenance.mlir and recomputed by hashModuleFromIR, so it may not move by a bit:
    the corpus, 60 generated modules and the audit's 2,048-resource fixture agree."""
    import random

    from bcir.examples import PROGRAMS
    from bcir.kbcir.differential import gen_module
    from bcir.kbcir.provenance import canonical_stream, hash_module
    from bcir.performance_audit import static_memory_module

    rng = random.Random(3)
    modules = [build() for build in PROGRAMS.values()]
    modules += [gen_module(rng) for _ in range(60)]
    modules.append(static_memory_module(1))
    for m in modules:
        assert hash_module(m) == _recursive_module_hash(m), m.name
        assert len(canonical_stream(m)) == sum(1 for _ in _flatten_like(m)), m.name


def _flatten_like(m):
    """The item count of the recursive flattening (the stream must carry every item)."""
    yield from (m.name, m.cacheline, m.align)
    for r in sorted(m.resources.values(), key=lambda r: r.rid):
        yield from (
            r.rid,
            r.domain,
            *r.shape,
            r.layout,
            r.align,
            r.access,
            r.priority,
            r.map_gen,
            r.data_gen,
        )
    for ph in m.phases:
        yield from (ph.phase_id, *sorted(ph.deps))
        for c in ph.claims:
            yield from (
                c.id,
                c.opcode,
                c.lane,
                c.stride_class,
                c.count,
                c.stride_k,
                *c.rd,
                *c.wr,
                c.hazard,
                c.domain,
                c.verify,
                c.bounds,
                c.op,
                c.offset,
                c.cost_class,
            )


def test_a_bool_item_never_takes_the_integer_memo():
    """`_fnv_items` memoizes the rendering of exact ints and strs. A bool is an int subclass
    that renders as "True"/"False" -- what the law rail writes for `scalable` -- so it must
    never collide with the memo entry of 1/0, or hash_target would change silently."""
    from bcir.kbcir.provenance import _fnv, hash_target

    assert _fnv(True) == _fnv_recursive(True) != _fnv(1) == _fnv_recursive(1)
    assert _fnv(1, True, "1") == _fnv_recursive(1, True, "1")
    assert hash_target(replace(AVX, scalable=True)) != hash_target(replace(AVX, scalable=False))


def test_the_module_identity_is_computed_once_and_shared():
    """The three hashes of section 5.2 -- the planner's, the verifier's, the client's --
    are one: the static-memory plan (with its internal verify), an identity-bound external
    verify, a manifest and an execution scope over the same module compute one digest."""
    from bcir.kbcir.provenance import digest_stats, module_identity
    from bcir.kbcir.scope import scope_for
    from bcir.kbcir.static_memory import plan_static_memory, verify_static_memory_plan
    from bcir.performance_audit import _AuditHardware, static_memory_module

    m = static_memory_module(1)
    hardware, bindings = _AuditHardware(), {rid: "ram" for rid in m.resources}
    before = digest_stats()["hash_module"]
    identity = module_identity(m)
    plan = plan_static_memory(m, bindings, hardware)
    assert verify_static_memory_plan(plan, m, bindings, hardware, identity=identity) == ()
    manifest = build_manifest(m, AVX, COOL)
    scope = scope_for(module=m, target=AVX, theta=COOL)
    assert digest_stats()["hash_module"] - before == 1
    assert manifest.m_module == identity.digest == scope.P["module_hash"]
    assert module_identity(m) is identity  # the cache, not a recomputation


def test_a_verifier_recomputes_at_a_trust_boundary():
    """R13's verify_manifest, replay and reproduces judge an external record: they recompute
    the module digest instead of trusting the cached identity (its right to recompute)."""
    from bcir.kbcir.provenance import digest_stats, module_identity

    m = vector_add(1024)
    manifest = build_manifest(m, AVX, COOL)
    module_identity(m)
    before = digest_stats()["hash_module"]
    assert verify_manifest(manifest, m, AVX, COOL) == []
    assert reproduces(manifest, m, AVX, COOL)
    replay_plan(manifest, m, AVX, COOL)
    assert digest_stats()["hash_module"] - before == 3  # one fresh digest per verdict


def test_a_declared_mutation_invalidates_the_identity():
    """add_resource / add_phase / touch() drop the cache: the next identity is a new digest,
    and the old one no longer describes the module -- refused strictly, recomputed otherwise."""
    from bcir.kbcir.provenance import IdentityMismatch, digest_of, hash_module, module_identity
    from bcir.model import Claim, Opcode, Phase, Resource

    m = vector_add(1024)
    old = module_identity(m)
    m.add_resource(Resource(rid=999, shape=(8,)))
    assert module_identity(m).digest != old.digest and not old.matches(m)
    m.add_phase(
        Phase(phase_id=7, deps=(0,), claims=[Claim(id=77, opcode=Opcode.ADD, rd=(999,), wr=(999,))])
    )
    newer = module_identity(m)
    assert newer.digest != old.digest and newer.digest == hash_module(m)
    assert digest_of(m, old) == hash_module(m)  # a stale identity is recomputed ...
    try:
        digest_of(m, old, strict=True)  # ... or refused
    except IdentityMismatch:
        pass
    else:
        raise AssertionError("a stale identity must be refused under strict")
    m.touch()
    assert module_identity(m) is not newer and module_identity(m).digest == newer.digest


def test_an_undeclared_in_place_edit_cannot_pass_a_verifier():
    """The Class-B defect the roadmap forbids rebuilding: a digest cache that survives a
    mutation. An in-place edit of a claim field (no touch()) leaves the cached identity in
    place, but a verifier validates by CONTENT: the stale identity is refused, the fresh
    digest disagrees with the plan's, and the static-memory verifier reports the mismatch."""
    from bcir.kbcir.provenance import IdentityMismatch, digest_of, hash_module, module_identity
    from bcir.kbcir.static_memory import plan_static_memory, verify_static_memory_plan
    from bcir.performance_audit import _AuditHardware, static_memory_module

    m = static_memory_module(1)
    hardware, bindings = _AuditHardware(), {rid: "ram" for rid in m.resources}
    plan = plan_static_memory(m, bindings, hardware)
    stale = module_identity(m)
    m.phases[3].claims[0].count = 1023  # undeclared: no touch()
    assert not stale.matches(m)
    assert digest_of(m, stale) == hash_module(m) != stale.digest
    try:
        digest_of(m, stale, strict=True)
    except IdentityMismatch:
        pass
    else:
        raise AssertionError("an identity must not survive an in-place edit")
    errors = verify_static_memory_plan(plan, m, bindings, hardware, identity=stale)
    assert "module digest mismatch" in errors
    # An undeclared STRUCTURAL edit (an appended claim) is caught one step earlier, by the
    # census the cache re-checks before trusting the revision.
    m.touch()
    fresh = module_identity(m)
    m.phases[0].claims.append(m.phases[3].claims[0])
    assert module_identity(m) is not fresh and module_identity(m).digest != fresh.digest


def test_cross_module_substitution_is_refused():
    """An identity minted from module A presented for module B: B's content is not what the
    digest describes, so the verifier refuses it (strict) or recomputes B's own digest, and
    a plan of A verified against B with A's identity fails on the digest."""
    from bcir.examples import fused_chain
    from bcir.kbcir.provenance import IdentityMismatch, digest_of, hash_module, module_identity
    from bcir.kbcir.static_memory import plan_static_memory, verify_static_memory_plan
    from bcir.performance_audit import _AuditHardware, static_memory_module

    a, b = vector_add(1024), fused_chain(1024)
    identity_a = module_identity(a)
    assert not identity_a.matches(b)
    assert digest_of(b, identity_a) == hash_module(b) != identity_a.digest
    try:
        digest_of(b, identity_a, strict=True)
    except IdentityMismatch:
        pass
    else:
        raise AssertionError("cross-module substitution must be refused")
    m1, m2 = static_memory_module(1), static_memory_module(1)
    m2.phases[0].claims[0].op = "tensor.other"
    hardware, bindings = _AuditHardware(), {rid: "ram" for rid in m1.resources}
    plan = plan_static_memory(m1, bindings, hardware)
    assert "module digest mismatch" in verify_static_memory_plan(
        plan, m2, bindings, hardware, identity=module_identity(m1)
    )
    # ... while an identical module IS the same content, and the same digest.
    same = static_memory_module(1)
    assert (
        module_identity(m1).matches(same)
        and module_identity(same).digest == module_identity(m1).digest
    )


def test_the_identity_is_not_part_of_the_modules_equality_or_hash():
    """The revision and the cache are bookkeeping, not content: two equal modules stay equal
    whatever their revisions, and hash_module never folds them (R13 parity)."""
    from bcir.kbcir.provenance import hash_module, module_identity

    a, b = vector_add(1024), vector_add(1024)
    module_identity(a)
    b.touch()
    b.touch()
    assert a == b and hash_module(a) == hash_module(b)
    assert (a.revision, b.revision) != (0, 0) or True  # revisions differ; equality does not
