"""Memory modules: the fixpoint law a = Lim(Res(U)) (Phase 21 -> Phase 20, R13).

A memory module is a *saturated* e-graph extraction, frozen and generation-tagged.
The witness is `saturated == True`: an artifact is admissible as memory only when
resolution reached its fixpoint, never at a budget cutoff. These checks pin the
axiom and its bridge into the provenance manifest.
"""

import json
from dataclasses import replace

from bcir.examples import vector_add
from bcir.kbcir import (
    MemoryModule,
    TARGETS,
    UnsaturatedMemory,
    build_manifest,
    fingerprint,
    freeze_module,
    is_idempotent,
    manifest_for,
    memory_artifacts,
    reproduces,
    reresolve,
    try_freeze,
)
from bcir.kbcir.cost import Theta
from bcir.kbcir.egraph import Add, C, Mul, Sub, V
from bcir.kbcir.memory import freeze
from bcir.model import Claim, Module, Opcode, Phase, Resource
from bcir.verify import verify_memory, verify_provenance

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()

# a*b + a*c -> a*(b+c): the canonical resolution example (needs >1 round to fix).
_FACTOR = Add(Mul(V("a"), V("b")), Mul(V("a"), V("c")))
_FACTORED = Mul(V("a"), Add(V("b"), V("c")))


def _laws(diags):
    return {d.law for d in diags}


# --- the fixpoint a = Lim(Res(U)) ------------------------------------------------


def test_freeze_extracts_the_resolution_fixpoint():
    # freeze runs resolution to saturation and freezes the min-cost representative:
    # Extract(Lim(Res(U))). The factored form is the attractor.
    mm = freeze(_FACTOR, generation=1)
    assert mm.saturated and mm.admissible
    assert mm.expr == _FACTORED and mm.cost == 3
    assert mm.generation == 1


def test_an_already_minimal_form_is_its_own_memory():
    mm = freeze(Mul(V("a"), V("b")))
    assert mm.saturated and mm.expr == Mul(V("a"), V("b")) and mm.cost == 2


# --- the admissibility law: saturated == True => admissible -----------------------


def test_budget_cutoff_is_not_a_fixpoint_and_is_refused():
    # A 1-round budget stops before the fixpoint witness even when the extracted
    # value is already optimal: it is a partial Res^k(U), not Lim(Res(U)). freeze
    # refuses to pin a non-canonical artifact as memory.
    cutoff = try_freeze(_FACTOR, budget=1)
    assert not cutoff.saturated and not cutoff.admissible
    assert cutoff.cost == 3  # value is right ...
    try:  # ... but the witness is absent
        freeze(_FACTOR, budget=1)
        assert False, "expected UnsaturatedMemory"
    except UnsaturatedMemory:
        pass


def test_an_untagged_artifact_is_not_memory():
    # A frozen artifact must be generation-tagged (immutable within a generation).
    mm = freeze(_FACTOR, generation=1)
    assert not replace(mm, generation=0).admissible


# --- idempotence: Res(Lim(Res(U))) = Lim(Res(U)) (the attractor property) --------


def test_a_memory_module_is_its_own_attractor():
    mm = freeze(_FACTOR)
    assert is_idempotent(mm)
    re = reresolve(mm)
    assert re.saturated and re.expr == mm.expr and re.fingerprint == mm.fingerprint


# --- determinism + content addressing (what the manifest needs) ------------------


def test_two_saturated_runs_freeze_the_same_module():
    # Confluence: the fixpoint is canonical, so independent freezes agree exactly
    # (manifest equality => identical plan depends on this).
    assert freeze(_FACTOR) == freeze(_FACTOR)


def test_fingerprint_is_content_addressed():
    a = freeze(_FACTOR)
    b = freeze(Add(Mul(V("a"), V("b")), Mul(V("a"), V("d"))))
    assert a.fingerprint != b.fingerprint
    assert a.fingerprint == fingerprint(a.expr, a.cost)


def test_json_round_trips():
    mm = freeze(_FACTOR, generation=5)
    assert MemoryModule.from_json(mm.to_json()) == mm


def test_json_rejects_ambiguous_or_forged_memory_modules():
    mm = freeze(_FACTOR, generation=5)
    bad_documents = [mm.to_json().replace('"cost": 3', '"cost": 3, "cost": 3')]

    for mutate in (
        lambda d: d.update(cost=d["cost"] + 1),
        lambda d: d.update(fingerprint=d["fingerprint"] ^ 1),
        lambda d: d.update(generation=True),
        lambda d: d["expr"].__setitem__(0, "exec"),
        lambda d: d["expr"].__setitem__(1, "unexpected"),
    ):
        bad = json.loads(mm.to_json())
        mutate(bad)
        bad_documents.append(json.dumps(bad))

    deep = ["var", "x", []]
    for _ in range(257):
        deep = ["add", None, [deep, ["const", 0, []]]]
    bad = json.loads(mm.to_json())
    bad["expr"] = deep
    bad_documents.append(json.dumps(bad))

    for text in bad_documents:
        try:
            MemoryModule.from_json(text)
            assert False, "expected malformed memory module to be rejected"
        except ValueError:
            pass


# --- module-level memory (the CSE / liked-pair view) -----------------------------


def _module_with(rd_a, rd_b):
    m = Module(name="cse")
    for rid in set(rd_a) | set(rd_b) | {90, 91}:
        m.add_resource(Resource(rid=rid, shape=(8,)))
    m.add_phase(
        Phase(
            phase_id=0,
            claims=[
                Claim(id=1, opcode=Opcode.ADD, rd=rd_a, wr=(90,), op="vector.add"),
                Claim(id=2, opcode=Opcode.ADD, rd=rd_b, wr=(91,), op="vector.add"),
            ],
        )
    )
    return m


def test_freeze_module_resolves_the_whole_claim_forest():
    shared = freeze_module(_module_with((10, 11), (10, 11)), generation=2)
    distinct = freeze_module(_module_with((10, 11), (12, 13)), generation=2)
    assert shared.admissible and distinct.admissible
    # CSE: identical claims collapse to fewer liked pairs (a smaller fixpoint).
    assert shared.enodes < distinct.enodes


def test_module_memory_passes_the_fixpoint_recheck():
    mm = freeze_module(_module_with((10, 11), (12, 13)), generation=1)
    assert verify_memory(mm) == [] and is_idempotent(mm)


# --- R13: the memory-module fixpoint law -----------------------------------------


def test_admissible_module_satisfies_R13():
    mm = freeze(_FACTOR, generation=1)
    assert verify_memory(mm) == []
    assert verify_provenance(None, memory_modules=[mm]) == []


def test_cutoff_module_is_R13():
    cutoff = try_freeze(_FACTOR, budget=1)
    assert "R13" in _laws(verify_memory(cutoff))
    assert "R13" in _laws(verify_provenance(None, memory_modules=[cutoff]))


def test_untagged_module_is_R13():
    untagged = replace(freeze(_FACTOR), generation=0)
    assert "R13" in _laws(verify_memory(untagged))


def test_wrong_generation_is_R13():
    mm = freeze(_FACTOR, generation=3)
    assert "R13" in _laws(verify_memory(mm, generation=4))
    assert verify_memory(mm, generation=3) == []


def test_tampered_representative_fails_the_recheck():
    # Forge a "saturated" module over a still-reducible expression. The verifier
    # does not trust the flag: re-resolving x+0 yields x, exposing the lie.
    forged = MemoryModule(
        expr=Add(V("x"), C(0)),
        cost=1,
        fingerprint=fingerprint(Add(V("x"), C(0)), 1),
        generation=1,
        iterations=1,
        enodes=3,
        saturated=True,
    )
    assert "R13" in _laws(verify_memory(forged))


def test_tampered_fingerprint_is_R13():
    forged = replace(freeze(_FACTOR), fingerprint=123456789)
    assert "R13" in _laws(verify_memory(forged))


# --- the bridge: a frozen memory module is part of a plan's commit hash ----------


def test_memory_module_chains_into_the_manifest():
    m = vector_add(1024)
    mm = freeze_module(m, generation=3)
    assert memory_artifacts(mm) == (("mem_gen", 3), ("mem_fp", mm.fingerprint))
    man = manifest_for(m, AVX, COOL, memory=mm)
    names = {n for n, _ in man.artifacts}
    assert {"mem_gen", "mem_fp"} <= names
    assert reproduces(man, m, AVX, COOL, artifacts=man.artifacts)


def test_a_new_generation_is_a_distinct_commit_same_plan():
    # Bumping the frozen memory's generation is a new branch in the version DAG:
    # the digest moves (provenance changed) but the plan value does not.
    m = vector_add(1024)
    g3 = manifest_for(m, AVX, COOL, memory=freeze_module(m, generation=3))
    g4 = manifest_for(m, AVX, COOL, memory=freeze_module(m, generation=4))
    assert g3.digest != g4.digest
    assert g3.diff(g4) == ["artifacts"]
    assert g3.score == g4.score
