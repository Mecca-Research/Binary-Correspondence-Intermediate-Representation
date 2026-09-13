"""G13 / S2-E: the workload component W, the measured-candidate corpus, the replay-gate integration.

The two gates the roadmap names, held as exact properties: a scope digest separates two
workloads on one program (plans for W1 and W2 carry distinct certificate digests, and `diff`
names W), and a policy promoted by the corpus never loses to the incumbent on the logged
episodes (the measured replay gate replays every logged episode or refuses; the portfolio
refuses a certificate over a subset). Around them: the workload is a validated, digested
declaration held to its module; the corpus is append-only and content-addressed (tampering is
refused on read); evidence carries raw samples and the host's attestation; the dispatch law's
measured rail runs only over a declared W and existing evidence; the class ladder holds a
measured claim to its scope and to the two-target rule; and the corpus informs, never decides
-- the plan is `optimize`'s and the verifier's verdict does not move.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

from bcir.examples import PROGRAMS, vector_add
from bcir.gem.dispatch import DispatchRequest, dispatch, solve_measured
from bcir.gem.exact import certify_measured, certify_schedule, certify_selection
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.matmul import plan_matmul
from bcir.kbcir.measured import (
    TENANCIES,
    MeasuredCorpus,
    MeasuredPlan,
    assignment_digest,
    from_schedule_artifact,
    measure_plans,
    pooled_median,
    read_corpus,
    replay_measured,
    scope_key,
    theta_key,
    write_corpus,
)
from bcir.kbcir.portfolio import PolicyPortfolio, ReplayCertificate, replay_gate
from bcir.kbcir.schedule_artifact import measure_schedule_candidates
from bcir.kbcir.scope import (
    MEASURED_COMPONENTS,
    UNDECLARED,
    certificate_class_allowed,
    scope_for,
)
from bcir.kbcir.weights import ENERGY, PERF, POLICIES, SAFE, THROUGHPUT
from bcir.kbcir.workload import WORKLOAD_CLASSES, Workload, verify_workload, workload_for
from bcir.tests.sweep_fixtures import random_module
from bcir.verify import verify_plan

H = TARGETS["x86_avx2"]
EPISODES = (Theta.cool(), Theta.hot(), Theta.mem_bound())
COMMIT = "5b45fdca"


def _work(result) -> int:
    """A bounded body whose cost follows the plan's widths (the stand-in for executing it)."""
    n = sum(max(1, step.candidate.width) for step in result.steps)
    acc = 0
    for i in range(n * 50):
        acc += i & 3
    return acc


def _corpus(module, workload, policies=None, episodes=EPISODES, repeats=3) -> MeasuredCorpus:
    corpus = MeasuredCorpus()
    for theta in episodes:
        for entry in measure_plans(
            module,
            H,
            theta,
            policies or tuple(POLICIES.values()),
            _work,
            workload=workload,
            source_commit=COMMIT,
            repeats=repeats,
        ):
            corpus.append(entry)
    return corpus


def _refused(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


# --- the workload is a declaration -----------------------------------------------------------


def test_a_workload_is_a_declared_validated_and_digested_model():
    m = vector_add(256)
    w = workload_for(m, service_level="latency", latency_ns=500_000)
    assert w.shapes == ((10, (256,)), (11, (256,)), (12, (256,)))
    assert w.classify() == "interactive" and len(w.digest()) == 64
    assert Workload.from_dict(json.loads(w.to_json())) == w
    assert workload_for(m, batch=8).classify() == "batch"
    assert workload_for(m, concurrency=2).classify() == "batch"
    assert workload_for(m).classify() == "nominal"
    assert set(WORKLOAD_CLASSES) == {"nominal", "interactive", "batch"}
    assert not verify_workload(w, m)
    # refusals: no floats or bools, consistent service levels and distributions, ordered
    # unique shapes with positive extents, a known version
    for bad in (
        dict(batch=0),
        dict(batch=1.5),
        dict(batch=True),
        dict(service_level="latency"),
        dict(latency_ns=5),
        dict(throughput_per_s=1),
        dict(distribution="dynamic"),
        dict(expected_counts=((1, 1),)),
        dict(shapes=((2, (4,)), (1, (4,)))),
        dict(shapes=((1, (4,)), (1, (4,)))),
        dict(shapes=((1, (0,)),)),
        dict(version="WorkloadV0"),
    ):
        assert _refused(Workload, **bad), bad
    assert _refused(Workload.from_dict, {**w.component(), "extra": 1})
    # the declaration is held to the module it describes
    assert verify_workload(Workload(shapes=((10, (256,)), (11, (256,)))), m) == [
        "resource 12 has no declared shape"
    ]
    wrong = Workload(shapes=((10, (128,)), (11, (256,)), (12, (256,))))
    assert "declared (128,)" in verify_workload(wrong, m)[0]
    assert verify_workload(workload_for(m, expected_counts={1000: 5}), m) == [
        "claim 1000 is not dynamic: its count is static"
    ]
    dyn = vector_add(256)
    dyn.phases[0].claims[0] = replace(dyn.phases[0].claims[0], dynamic=True)
    dyn.touch()
    assert not verify_workload(workload_for(dyn, expected_counts={1000: 100}), dyn)
    assert (
        "above its declared bound"
        in verify_workload(workload_for(dyn, expected_counts={1000: 1000}), dyn)[0]
    )
    assert "not in the module" in verify_workload(workload_for(dyn, expected_counts={7: 1}), dyn)[0]


# --- gate 1: a scope digest separates two workloads ----------------------------------------------


def test_a_scope_digest_separates_two_workloads_on_one_program():
    """Plans for W1 and W2 on one program carry distinct digests, on every certificate."""
    for name, build in sorted(PROGRAMS.items()):
        m = build()
        r = optimize(m, H, Theta.cool(), PERF)
        w1 = workload_for(m, service_level="latency", latency_ns=1_000_000)
        w2 = workload_for(m, batch=8, service_level="throughput", throughput_per_s=1_000)
        w3 = workload_for(m)
        scopes = [
            certify_schedule(m, r, H, Theta.cool(), PERF, requested="TMSAO-4", workload=w).scope
            for w in (w1, w2, w3)
        ]
        assert len(set(scopes)) == 3, name
        again = certify_schedule(m, r, H, Theta.cool(), PERF, requested="TMSAO-4", workload=w1)
        assert again.scope == scopes[0], name  # the same workload is the same scope
        selection = [
            certify_selection(m, r, H, Theta.cool(), PERF, workload=w).scope for w in (w1, w2)
        ]
        assert selection[0] != selection[1] and selection[0] != scopes[0], name
    s1 = scope_for(m, H, Theta.cool(), PERF, workload=w1)
    s2 = scope_for(m, H, Theta.cool(), PERF, workload=w2)
    assert s1.diff(s2) == ("W",) and s1.W == w1.component() and "W" not in s1.undeclared()
    assert "W" in scope_for(m, H, Theta.cool(), PERF).undeclared()


def test_the_ladder_holds_a_measured_claim_to_its_scope_and_the_two_target_rule():
    m = vector_add(256)
    full = scope_for(
        m, H, Theta.cool(), PERF, workload=workload_for(m), measurement={"protocol": "raw"}
    )
    evidence = {"incumbent": True, "search_coverage": 1.0, "prediction_interval": (1, 2, 3, 0)}
    assert certificate_class_allowed(full, evidence)[0] == "TMSAO-3"
    assert MEASURED_COMPONENTS == ("P", "H", "W", "M")
    for missing in ("W", "M"):
        klass, reason = certificate_class_allowed(replace(full, **{missing: UNDECLARED}), evidence)
        assert klass == "TMSAO-4" and missing in reason and "nobody wrote down" in reason
    klass, reason = certificate_class_allowed(
        full, {"incumbent": True, "search_coverage": 1.0, "tenancy": "virtualized"}
    )
    assert klass == "TMSAO-4" and "two-target rule" in reason and "virtualized" in reason
    klass, reason = certificate_class_allowed(full, {"incumbent": True, "search_coverage": 0.0})
    assert klass == "TMSAO-4" and "nothing to rank" in reason
    klass, reason = certificate_class_allowed(
        full, {"incumbent": True, "search_coverage": 0.0, "stale": 2}
    )
    assert klass == "TMSAO-4" and "stale" in reason
    # G0's rule stands: a measurement is a real statement without an objective relation
    assert certificate_class_allowed(replace(full, O=UNDECLARED), evidence)[0] == "TMSAO-3"


# --- the corpus -----------------------------------------------------------------------------------


def test_the_corpus_is_append_only_and_content_addressed():
    m = random_module(0)
    w = workload_for(m, batch=4)
    corpus = _corpus(m, w, policies=(PERF, ENERGY), episodes=EPISODES[:2], repeats=2)
    assert len(corpus) == 4 and len(set(corpus.chain)) == 4 and corpus.head == corpus.chain[-1]
    assert _refused(corpus.append, corpus.entries[0])  # evidence is counted once
    assert _refused(corpus.append, "not evidence")
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "corpus.json"
        write_corpus(path, corpus)
        again = read_corpus(path)
    assert again.head == corpus.head and again.entries == corpus.entries
    assert again.extends(corpus) and corpus.extends(again)
    # append-only across versions: a longer history extends, a rewritten one does not
    longer = MeasuredCorpus(corpus.entries)
    longer.append(replace(corpus.entries[0], source_commit="abcdef0"))
    assert longer.extends(corpus) and not corpus.extends(longer)
    assert not MeasuredCorpus(corpus.entries[1:]).extends(corpus)
    # tampering is refused on read: altered, removed, reordered, a forged head or chain
    d = json.loads(corpus.to_json())
    first = d["entries"][0]
    altered = {**first, "samples": [{**first["samples"][0], "wall_ns": 1}] + first["samples"][1:]}
    for label, doc in (
        ("altered", {**d, "entries": [altered] + d["entries"][1:]}),
        ("removed", {**d, "entries": d["entries"][1:], "chain": d["chain"][1:]}),
        ("reordered", {**d, "entries": d["entries"][::-1]}),
        ("forged head", {**d, "head": "0" * 64}),
        ("forged chain", {**d, "chain": d["chain"][:-1] + ["0" * 64]}),
        ("unknown field", {**d, "extra": 1}),
        ("unknown schema", {**d, "schema": "bcir.measured_corpus.v0"}),
    ):
        assert _refused(MeasuredCorpus.from_dict, doc), label
    assert _refused(MeasuredPlan.from_dict, {**first, "extra": 1})
    assert _refused(MeasuredPlan.from_dict, {**first, "tenancy": "silicon"})
    assert _refused(MeasuredPlan.from_dict, {**first, "samples": []})


def test_evidence_carries_raw_samples_counters_and_the_host_attestation():
    m = random_module(0)
    w = workload_for(m, batch=4)
    entries = measure_plans(
        m, H, Theta.cool(), (PERF, ENERGY), _work, workload=w, source_commit=COMMIT, repeats=4
    )
    program, target, digest = scope_key(m, H, w)
    for entry in entries:
        assert (entry.program, entry.target, entry.workload) == (program, target, digest)
        assert entry.theta == theta_key(Theta.cool())
        assert len(entry.samples) == 4 and all(s.wall_ns >= 1 for s in entry.samples)
        assert entry.tenancy in TENANCIES and entry.signals
        assert entry.plan == assignment_digest(optimize(m, H, Theta.cool(), POLICIES[entry.policy]))
        low, median, high, mad = entry.stats()
        assert low <= median <= high and mad >= 0 and entry.median_wall_ns == median
        assert entry.silicon == (entry.tenancy == "bare-metal" and entry.hardware_pmu)
        assert MeasuredPlan.from_dict(json.loads(entry.to_json())) == entry
    assert {entry.policy for entry in entries} == {"latency", "energy"}
    assert entries[0].plan != entries[1].plan  # the fixture's policies select different plans
    assert _refused(
        measure_plans,
        m,
        H,
        Theta.cool(),
        (PERF,),
        _work,
        workload=w,
        source_commit=COMMIT,
        repeats=0,
    )
    assert _refused(
        measure_plans, m, H, Theta.cool(), (PERF, PERF), _work, workload=w, source_commit=COMMIT
    )
    assert _refused(
        measure_plans, m, H, Theta.cool(), (PERF,), None, workload=w, source_commit=COMMIT
    )
    # B1's matmul evidence enters the same corpus, with what B1 knew and nothing more
    target_profile = TargetProfile.x86_avx2()
    analytic = plan_matmul(2, 2, 2, target_profile)
    artifact = measure_schedule_candidates(
        2,
        2,
        2,
        target_profile,
        [analytic],
        lambda _plan: sum(range(64)),
        workload="gemm",
        source_commit=COMMIT,
        repeats=1,
    )
    rows = from_schedule_artifact(artifact, w)
    assert len(rows) == 1 and rows[0].target == artifact.target_sha256
    assert rows[0].tenancy == "unproven" and rows[0].policy.startswith("tile:")
    corpus = MeasuredCorpus(entries)
    corpus.append(rows[0])
    assert len(corpus) == 3
    assert corpus.physical_targets() <= 1  # B1's rows are unproven; this host's are one target


# --- gate 2: the replay gate over the corpus ------------------------------------------------------


def test_a_policy_promoted_by_the_corpus_never_loses_on_the_logged_episodes():
    for seed in (0, 1, 5):
        m = random_module(seed)
        w = workload_for(m, batch=4)
        corpus = _corpus(m, w)
        program, target, digest = scope_key(m, H, w)
        logged = corpus.episodes(program, target, digest)
        assert logged == tuple(sorted((theta_key(t) for t in EPISODES), key=int))
        assert corpus.census(program, target, digest) == ("energy", "latency", "safe", "throughput")
        for candidate in (ENERGY, THROUGHPUT, SAFE):
            cert = replay_measured(corpus, program, target, digest, candidate, PERF)
            assert cert.corpus == corpus.head and cert.logged == cert.episodes == 3
            assert cert.covered
            # the verdict is the corpus's own numbers, episode by episode
            losses = 0
            for episode in logged:
                mine = pooled_median(
                    corpus.lookup(program, target, digest, theta=episode, policy=candidate)
                )
                theirs = pooled_median(
                    corpus.lookup(program, target, digest, theta=episode, policy=PERF)
                )
                losses += mine > theirs
            assert cert.regressions == losses and cert.admitted == (losses == 0)
            portfolio = PolicyPortfolio.default()
            if cert.admitted:
                entry = portfolio.promote(PERF.name, candidate, cert)
                assert entry.certified and entry.gen == 2
            else:
                assert _refused(portfolio.promote, PERF.name, candidate, cert)
        # a certificate over a subset of the log is refused, whatever it says
        subset = ReplayCertificate(ENERGY.name, PERF.name, 1, 0, corpus=corpus.head, logged=3)
        assert not subset.admitted and not subset.covered
        assert _refused(PolicyPortfolio.default().promote, PERF.name, ENERGY, subset)
        # a candidate without evidence on a logged episode is refused rather than judged
        hot = theta_key(Theta.hot())
        partial = MeasuredCorpus(
            e for e in corpus.entries if not (e.policy == "energy" and e.theta == hot)
        )
        assert _refused(replay_measured, partial, program, target, digest, ENERGY, PERF)
        assert _refused(
            replay_measured,
            corpus,
            program,
            target,
            workload_for(m, batch=8).digest(),
            ENERGY,
            PERF,
        )
    # the analytic gate is unchanged: no corpus, no coverage requirement
    cert = replay_gate(vector_add(1024), H, PERF, PERF, list(EPISODES))
    assert cert.admitted and cert.corpus == "" and cert.logged == 0 and cert.covered


def test_the_portfolio_selects_by_runtime_and_workload_class():
    p = PolicyPortfolio.default()
    m = vector_add(256)
    interactive = workload_for(m, service_level="latency", latency_ns=1_000)
    batch = workload_for(m, batch=8)
    nominal = workload_for(m)
    for theta in EPISODES:
        assert p.select(theta) is p.select(theta, None)  # the historical rule
    assert p.select(Theta.cool(), interactive).name == "latency"
    assert p.select(Theta.cool(), batch).name == "throughput"
    assert p.select(Theta.cool(), nominal).name == "latency"
    # a runtime constraint outranks the workload
    assert p.select(Theta.hot(), batch).name == "energy"
    assert p.select(Theta.mem_bound(), interactive).name == "throughput"


# --- the dispatch law's measured rail --------------------------------------------------------------


def test_the_dispatch_law_runs_the_measured_rail_only_over_a_declared_workload_and_evidence():
    m = random_module(0)
    w = workload_for(m, batch=4)
    pols = list(POLICIES.values())
    n = sum(len(p.claims) for p in m.phases)
    d = dispatch(DispatchRequest("selection", n, "TMSAO-3", 5))
    assert d.rail == "fast" and "W" in d.reason
    d = dispatch(DispatchRequest("selection", n, "TMSAO-3", 5, workload=w.digest()))
    assert d.rail == "fast" and "no measured plan" in d.reason
    d = dispatch(DispatchRequest("schedule", n, "TMSAO-3", 5, workload=w.digest(), evidence=3))
    assert d.rail == "fast" and "whole plans" in d.reason
    d = dispatch(DispatchRequest("selection", n, "TMSAO-3", 5, workload=w.digest(), evidence=3))
    assert (d.rail, d.solver, d.units, d.expected) == (
        "measured",
        "measured_best",
        "samples",
        "TMSAO-3",
    )
    for kind in ("path", "selection"):
        req = DispatchRequest(kind, n, "TMSAO-3", 5, workload=w.digest(), evidence=1)
        assert dispatch(req).rail == "measured"
    # the other classes do not read the workload
    assert (
        dispatch(
            DispatchRequest("selection", n, "TMSAO-4", 5, workload=w.digest(), evidence=3)
        ).rail
        == "fast"
    )
    assert (
        dispatch(
            DispatchRequest("selection", n, "TMSAO-2", 5, workload=w.digest(), evidence=3)
        ).rail
        == "proof"
    )
    for kwargs in (dict(workload=""), dict(evidence=-1), dict(evidence=True)):
        assert _refused(DispatchRequest, "selection", n, "TMSAO-3", 5, **kwargs), kwargs
    corpus = _corpus(m, w, episodes=EPISODES[:1])
    program, target, digest = scope_key(m, H, w)
    evidence = len(corpus.lookup(program, target, digest))
    request = DispatchRequest("selection", n, "TMSAO-3", 5, workload=w.digest(), evidence=evidence)
    (policy, result), record = solve_measured(request, corpus, m, H, Theta.cool(), pols, workload=w)
    assert record.stop_reason == "measured" and record.bound_source == "corpus median"
    assert record.spent == evidence * 3
    assert assignment_digest(result) == assignment_digest(optimize(m, H, Theta.cool(), policy))
    medians = {
        pol.name: pooled_median(
            corpus.lookup(program, target, digest, theta=theta_key(Theta.cool()), policy=pol)
        )
        for pol in pols
    }
    assert medians[policy.name] == min(medians.values())
    assert record.granted == "TMSAO-4"  # one host is not two attested silicon targets
    # without evidence the fast rail places the incumbent's plan once
    (policy, result), record = solve_measured(
        DispatchRequest("selection", n, "TMSAO-3", 5, workload=w.digest()),
        corpus,
        m,
        H,
        Theta.cool(),
        pols,
        workload=w,
        incumbent=SAFE,
    )
    assert policy is SAFE and record.stop_reason == "heuristic" and record.decision.rail == "fast"
    # stale evidence -- plans the planner no longer selects -- is not used
    stale = MeasuredCorpus(replace(e, plan="0" * 64) for e in corpus.entries)
    (policy, result), record = solve_measured(
        request, stale, m, H, Theta.cool(), pols, workload=w, incumbent=PERF
    )
    assert policy is PERF and record.stop_reason == "heuristic"


def test_the_measured_certificate_is_bound_to_its_scope_and_the_corpus_informs_never_decides():
    m = random_module(0)
    w = workload_for(m, batch=4)
    pols = list(POLICIES.values())
    corpus = _corpus(m, w, episodes=EPISODES[:2])
    cert = certify_measured(m, H, Theta.cool(), w, corpus, pols)
    assert cert.dispatch.decision.rail == "measured" and cert.dispatch.stop_reason == "measured"
    assert cert.coverage == {"measured": 4, "stale": 0, "census": 4}
    assert cert.corpus == corpus.head and cert.episodes == 2 and cert.samples == 3
    assert cert.interval[0] <= cert.interval[1] <= cert.interval[2] and cert.interval[3] >= 0
    assert cert.median_ns == cert.interval[1]
    assert cert.plan == assignment_digest(optimize(m, H, Theta.cool(), POLICIES[cert.policy]))
    if cert.physical_targets >= 2:  # pragma: no cover - only on attested silicon, two targets
        assert cert.klass == "TMSAO-3"
    else:
        assert cert.klass == "TMSAO-4" and "two-target rule" in cert.statement
    assert cert.to_dict()["class"] == cert.klass == cert.to_dict()["dispatch"]["granted"]
    # W and M are inside the scope: the same workload is the same scope, another is another
    assert certify_measured(m, H, Theta.cool(), w, corpus, pols).scope == cert.scope
    other = certify_measured(m, H, Theta.cool(), workload_for(m, batch=8), corpus, pols)
    assert other.scope != cert.scope and other.dispatch.decision.rail == "fast"
    assert "nothing to rank" in other.statement and other.coverage["measured"] == 0
    # the ladder grants TMSAO-3 to attested silicon on two targets and to nothing less (the
    # corpus is data: these entries CLAIM bare metal, which is what the rule reads)
    silicon = MeasuredCorpus(
        replace(
            e, tenancy="bare-metal", signals="pmu=cpu", hardware_pmu=True, cycles=1, instructions=1
        )
        for e in corpus.entries
    )
    one = certify_measured(m, H, Theta.cool(), w, silicon, pols)
    assert one.klass == "TMSAO-4" and one.physical_targets == 1
    for e in corpus.entries[:2]:
        silicon.append(
            replace(
                e,
                target="ab" * 32,
                tenancy="bare-metal",
                signals="pmu=cpu",
                hardware_pmu=True,
                cycles=1,
                instructions=1,
            )
        )
    two = certify_measured(m, H, Theta.cool(), w, silicon, pols)
    assert two.klass == "TMSAO-3" and two.physical_targets == 2
    assert (
        two.dispatch.granted == "TMSAO-3" and two.statement == "best measured admitted realization"
    )
    # stale evidence: the fast rail's plan, and the reason names the staleness
    stale = MeasuredCorpus(replace(e, plan="0" * 64) for e in corpus.entries)
    c = certify_measured(m, H, Theta.cool(), w, stale, pols)
    assert c.coverage == {"measured": 0, "stale": 4, "census": 4} and "stale" in c.statement
    assert c.dispatch.stop_reason == "heuristic"
    # informs, never decides: every plan the rail returns is the planner's under a policy and
    # the verifier's verdict on it does not depend on the corpus
    for theta in EPISODES[:2]:
        for pol in pols:
            r = optimize(m, H, theta, pol)
            assert verify_plan(m, r, H, theta=theta, policy=pol) == []
    (policy, result), _record = solve_measured(
        DispatchRequest(
            "selection",
            sum(len(p.claims) for p in m.phases),
            "TMSAO-3",
            5,
            workload=w.digest(),
            evidence=len(corpus.entries),
        ),
        corpus,
        m,
        H,
        Theta.cool(),
        pols,
        workload=w,
    )
    assert result == optimize(m, H, Theta.cool(), policy)
    assert verify_plan(m, result, H, theta=Theta.cool(), policy=policy) == []
