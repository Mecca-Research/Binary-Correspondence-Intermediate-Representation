"""G6 / S2-D: typed regions and the objective registry -- affine first, opaque as the fallback.

Two exact laws carry the slice. Every region expands conservatively to claims: the region
graph of any module, expanded, is that module claim for claim, and the plan `optimize` selects
is unchanged. And an objective is admitted to the registry only with its laws proved -- closure
under the declared overflow policy, identities, associativity, the commutativity and idempotence
of `select`, distributivity where claimed, and the strict order `select` realizes -- with the
two names the law rail shares reproducing the tree's own paths: min-plus the planner's shortest
path, max-plus the exact scheduler's critical path.
"""

from __future__ import annotations

import random

from bcir.examples import PROGRAMS
from bcir.gem.exact import certify_selection
from bcir.gem.dispatch import DispatchRequest, dispatch, solve_path
from bcir.kbcir import TARGETS, optimize
from bcir.kbcir.cost import Theta
from bcir.kbcir.differential import gen_module
from bcir.kbcir.objectives import (
    I64_MAX,
    INF,
    LAW_RAIL_NAMES,
    Objective,
    ObjectiveError,
    dag_best_path,
    dominates,
    frontier,
    objective,
    registry,
    verify_objective,
)
from bcir.kbcir.realize import _flatten
from bcir.kbcir.regions import (
    REFUSALS,
    Region,
    affine_refusal,
    expand,
    module_floor,
    region_floor,
    region_graph,
    verify_region,
)
from bcir.kbcir.semiring import dag_shortest_path
from bcir.kbcir.weights import ENERGY, PERF, POLICIES
from bcir.model import Claim, Lane, Module, Opcode, Phase, Resource, StrideClass
from bcir.tests.sweep_fixtures import general_fixture, random_module


def _corpus():
    out = [(name, build()) for name, build in sorted(PROGRAMS.items())]
    out.append(("general2x3", general_fixture(2, 3)))
    out += [(f"gen{seed}", gen_module(random.Random(seed))) for seed in range(20)]
    out += [(f"rand{seed}", random_module(seed)) for seed in range(20)]
    return out


# --- the objective registry ---------------------------------------------------------------


def test_every_registry_entry_proves_its_laws_under_both_overflow_policies():
    for policy in ("checked", "saturate"):
        entries = registry(policy)
        assert set(entries) == {
            "min_plus",
            "max_plus",
            "min_max",
            "boolean",
            "lexicographic",
            "pareto",
        }
        for name, entry in entries.items():
            assert verify_objective(entry) == [], name
            assert entry.overflow == policy
    assert set(LAW_RAIL_NAMES) <= set(registry())
    for bad in ("plus_times", "tropical", ""):
        try:
            objective(bad)
        except ObjectiveError:
            continue
        raise AssertionError(f"unknown objective {bad!r} admitted")
    try:
        Objective("x", "int", min, max, 0, 0, lambda a, b: a < b, overflow="wrap")
    except ObjectiveError:
        pass
    else:
        raise AssertionError("an undeclared overflow policy was accepted")


def test_a_lawless_operator_is_refused():
    """'Semiring' is not a label: an operator without its laws does not enter the registry."""
    broken = Objective(
        "broken",
        "int",
        lambda a, b: a - b,
        lambda a, b: min(a, b),
        0,
        INF,
        lambda a, b: a < b,
        frozenset({"distributive", "commutative_combine"}),
    )
    problems = verify_objective(broken)
    assert any("associative" in p for p in problems) and any("commutative" in p for p in problems)
    wrong_order = Objective(
        "wrong", "int", lambda a, b: a + b, lambda a, b: min(a, b), 0, INF, lambda a, b: a > b
    )
    assert any("realize" in p for p in verify_objective(wrong_order))


def test_overflow_is_a_policy_not_an_accident():
    checked, saturating = objective("min_plus"), objective("min_plus", "saturate")
    try:
        checked.combine(I64_MAX, 1)
    except ObjectiveError:
        pass
    else:
        raise AssertionError("the checked policy let a value overflow i64")
    assert saturating.combine(I64_MAX, 1) == I64_MAX
    assert checked.combine(5, INF) == INF  # the identities live outside the range by design


def test_min_plus_is_the_planners_path_and_max_plus_the_critical_path():
    rng = random.Random(3)
    min_plus, max_plus = objective("min_plus"), objective("max_plus")
    for _ in range(200):
        n = rng.randint(2, 30)
        adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for u in range(n - 1):
            for v in range(u + 1, min(n, u + 1 + rng.randint(1, 4))):
                if rng.random() < 0.7:
                    adj[u].append((v, rng.randrange(0, 100)))
        assert dag_best_path(n, adj, min_plus) == dag_shortest_path(n, adj)
        longest, _pred = dag_best_path(n, adj, max_plus)
        # the longest path is never below the shortest, and unreachable nodes stay at zero
        shortest, _p = dag_shortest_path(n, adj)
        for v in range(n):
            if shortest[v] == INF:
                assert longest[v] == -INF
            else:
                assert longest[v] >= shortest[v]
    # the exact scheduler's critical path IS max-plus over the hazard DAG
    from bcir.gem.concurrency import _topo_phase_ids
    from bcir.gem.exact import _critical_path
    from bcir.gem.schedule import _PhaseDispatch, _streams, durations_from, phase_hazards

    checked = 0
    for _name, module in _corpus()[:24]:
        h = TARGETS["x86_avx512"]
        dur = durations_from(optimize(module, h, Theta.cool(), PERF))
        if not dur:
            continue
        hazards = phase_hazards(module)
        pmap = module.phase_map()
        for pid in _topo_phase_ids(module):
            claims = sorted(pmap[pid].claims, key=lambda c: c.id)
            d = _PhaseDispatch(claims, hazards[pid], *_streams(h), True)
            ids = [c.id for c in claims]
            preds = {cid: tuple(p for p in d.preds_of[cid] if p in d.claim_by_id) for cid in ids}
            durations = {cid: max(0, dur.get(cid, 1)) for cid in ids}
            head = _critical_path(ids, preds, durations)
            index = {cid: i + 1 for i, cid in enumerate(ids)}
            adj = [[] for _ in range(len(ids) + 1)]
            for cid in ids:
                if not preds[cid]:
                    adj[0].append((index[cid], durations[cid]))
                for p in preds[cid]:
                    adj[index[p]].append((index[cid], durations[cid]))
            best, _ = dag_best_path(len(ids) + 1, adj, max_plus)
            assert all(best[index[cid]] == head[cid] for cid in ids)
            checked += 1
    assert checked >= 20


def test_the_other_objectives_behave_as_their_laws_say():
    min_max, boolean, lex, pareto = (
        objective("min_max"),
        objective("boolean"),
        objective("lexicographic"),
        objective("pareto"),
    )
    # a path costs its bottleneck; the least bottleneck wins
    adj = [[(1, 5), (2, 9)], [(3, 7)], [(3, 1)], []]
    best, pred = dag_best_path(4, adj, min_max)
    assert best[3] == 7 and pred[3] == 1
    # reachability
    adj = [[(1, True), (2, False)], [(3, False)], [(3, True)], []]
    best, _ = dag_best_path(4, adj, boolean)
    assert best == [True, True, False, False]
    # lexicographic: the first component decides, the second breaks ties
    adj = [[(1, (1, 9)), (2, (1, 2))], [(3, (0, 0))], [(3, (0, 5))], []]
    best, pred = dag_best_path(4, adj, lex)
    assert best[3] == (1, 7) and pred[3] == 2
    # pareto: a frontier of non-dominated vectors
    assert (
        dominates((1, 2), (2, 2))
        and not dominates((1, 2), (1, 2))
        and not dominates((3, 1), (1, 3))
    )
    assert frontier([(1, 3), (3, 1), (2, 2), (4, 4), (1, 3)]) == frozenset({(1, 3), (3, 1), (2, 2)})
    a, b = frozenset({(1, 3), (3, 1)}), frozenset({(2, 2)})
    assert pareto.select(a, b) == frozenset({(1, 3), (3, 1), (2, 2)})
    assert pareto.combine(a, b) == frozenset({(3, 5), (5, 3)})
    assert pareto.better(frozenset({(1, 1)}), frozenset({(2, 2), (1, 3)}))


# --- regions ------------------------------------------------------------------------------


def test_every_region_expands_conservatively_to_the_module_and_the_plan_is_unchanged():
    kinds: dict[str, int] = {}
    for name, module in _corpus():
        graph = region_graph(module)
        for region in graph.regions:
            assert verify_region(region, module) == [], (name, region)
            kinds[region.kind] = kinds.get(region.kind, 0) + 1
        assert set(graph.by_claim()) == {c.id for p in module.phases for c in p.claims}, name
        expanded = expand(graph, module)
        assert [(pid, c.id) for pid, c in expanded] == [
            (pid, c.id) for pid, c in _flatten(module)
        ], name
        for tname in ("x86_avx512", "x86_avx2"):
            h = TARGETS[tname]
            before = optimize(module, h, Theta.cool(), PERF)
            rebuilt = Module(name=module.name, cacheline=module.cacheline, align=module.align)
            for resource in module.resources.values():
                rebuilt.add_resource(resource)
            for phase in module.phases:
                rebuilt.add_phase(
                    Phase(
                        phase_id=phase.phase_id,
                        deps=phase.deps,
                        claims=[c for pid, c in expanded if pid == phase.phase_id],
                        event=phase.event,
                    )
                )
            after = optimize(rebuilt, h, Theta.cool(), PERF)
            assert [(s.claim_id, s.candidate, s.cost) for s in after.steps] == [
                (s.claim_id, s.candidate, s.cost) for s in before.steps
            ], (name, tname)
    assert kinds["affine"] >= 40 and kinds["opaque"] >= 40


def test_affine_refusals_are_named_and_the_verifier_refuses_a_forged_region():
    module = general_fixture(1, 2)  # four unit-stride claims: one affine region
    graph = region_graph(module)
    assert [r.kind for r in graph.regions] == ["affine"] and graph.regions[0].size == 4
    region = graph.regions[0]
    assert len(region.maps) == 12 and all(m.stride == 1 for m in region.maps)
    for claim, why in (
        (
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=8,
                rd=(1,),
                wr=(2,),
                dynamic=True,
            ),
            "dynamic-trip-count",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=8,
                rd=(1,),
                wr=(2,),
                volatile=True,
            ),
            "volatile",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.BARRIER,
                lane=Lane.H,
                stride_class=StrideClass.SCALAR,
                count=1,
                hazard="barriered",
            ),
            "fence",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=8,
                rd=(1,),
                wr=(2,),
                hazard="atomic",
            ),
            "atomic",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.GGG_LOAD,
                lane=Lane.GGG,
                stride_class=StrideClass.RANDOM,
                count=8,
                rd=(1,),
                wr=(2,),
            ),
            "sparse",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.UX,
                stride_class=StrideClass.CACHELINE,
                count=8,
                rd=(1,),
                wr=(2,),
            ),
            "cacheline-indexed",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.T_MACC,
                lane=Lane.T,
                stride_class=StrideClass.TILE,
                count=8,
                rd=(1,),
                wr=(2,),
            ),
            "tile",
        ),
        (
            Claim(
                id=9,
                opcode=Opcode.ADD,
                lane=Lane.U,
                stride_class=StrideClass.UNIT,
                count=10**9,
                rd=(1,),
                wr=(2,),
            ),
            "extent",
        ),
    ):
        assert affine_refusal(claim, module) == why, why
        assert why in REFUSALS
    # forged regions: non-consecutive claims, a foreign claim, wrong maps, a model on an opaque one
    ids = region.claim_ids
    assert verify_region(Region("affine", 0, (ids[0], ids[2])), module)
    assert verify_region(Region("affine", 0, (ids[0], 999)), module)
    assert verify_region(Region("affine", 0, ids), module) == [
        "access maps are not the claims' maps"
    ]
    assert verify_region(Region("opaque", 0, ids, maps=region.maps), module)
    sparse = PROGRAMS["histogram_gather"]()
    sparse_graph = region_graph(sparse)
    assert all(r.kind == "opaque" and r.refusal == "sparse" for r in sparse_graph.regions)
    try:
        Region("affine", 0, ())
    except ValueError:
        pass
    else:
        raise AssertionError("an empty region was accepted")
    try:
        expand(sparse_graph, module)  # a graph of another module
    except ValueError:
        pass
    else:
        raise AssertionError("a foreign region graph expanded")


def test_the_region_floor_is_never_above_what_the_plan_scores_and_is_tighter_where_no_read_is_shared():
    for name, module in _corpus():
        for tname in ("x86_avx512", "x86_avx2"):
            h = TARGETS[tname]
            for theta in (Theta.cool(), Theta(thermal=80, power=70)):
                for policy in POLICIES.values():
                    result = optimize(module, h, theta, policy)
                    floor = module_floor(module, h, theta, policy)
                    assert floor <= result.score, (name, tname, policy.name, floor, result.score)
                    steps = {s.claim_id: s.cost for s in result.steps}
                    for region in region_graph(module).regions:
                        assert region_floor(region, module, h, theta, policy) <= sum(
                            steps[cid] for cid in region.claim_ids
                        ), (name, region.kind)

    # two unit claims: sharing a read, the successor may be discounted; on disjoint reads
    # the floor knows no discount is possible and is tighter by exactly the discount
    def pair(shared: bool) -> Module:
        m = Module(name="pair")
        for rid in (1, 2, 3, 4, 5):
            m.add_resource(Resource(rid=rid, shape=(4096,)))
        first = Claim(
            id=1,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=4096,
            rd=(1, 2),
            wr=(3,),
            op="vector.add",
        )
        second = Claim(
            id=2,
            opcode=Opcode.ADD,
            lane=Lane.U,
            stride_class=StrideClass.UNIT,
            count=4096,
            rd=(1 if shared else 4, 5),
            wr=(3,),
            op="vector.add",
        )
        m.add_phase(Phase(phase_id=0, claims=[first, second]))
        return m

    h = TARGETS["x86_avx512"]
    shared, disjoint = pair(True), pair(False)
    floor_shared, floor_disjoint = (
        module_floor(shared, h, Theta.cool(), PERF),
        module_floor(disjoint, h, Theta.cool(), PERF),
    )
    score_shared, score_disjoint = (
        optimize(shared, h, Theta.cool(), PERF).score,
        optimize(disjoint, h, Theta.cool(), PERF).score,
    )
    assert floor_shared < floor_disjoint  # the discount is possible only with the shared read
    assert (
        floor_shared == score_shared and floor_disjoint == score_disjoint
    )  # and both floors are met


def test_the_selection_certificate_is_exact_and_records_the_path_rail():
    for name, build in sorted(PROGRAMS.items()):
        module = build()
        h = TARGETS["x86_avx512"]
        result = optimize(module, h, Theta.cool(), PERF)
        certificate = certify_selection(module, result, h, Theta.cool(), PERF)
        assert (
            certificate.klass == "TMSAO-1"
            and certificate.optimum == certificate.score == certificate.lower_bound
        ), name
        assert certificate.structural_floor <= certificate.score and certificate.coupling_cost >= 0
        body = certificate.to_dict()
        assert body["objective"] == "min_plus" and body["dispatch"]["kind"] == "path"
        assert body["dispatch"]["solver"] == "optimize" and body["dispatch"]["granted"] == "TMSAO-1"
        assert sum(body["regions"].values()) == len(region_graph(module).regions)
        assert len(certificate.scope) == 64
    module = PROGRAMS["vector_add"]()
    plain = certify_selection(module, optimize(module, h, Theta.cool(), PERF), h)
    assert plain.klass == "TMSAO-4" and "declared" in plain.statement
    # the path rail: a DP closes at any size; a heuristic request records the fast rail
    assert dispatch(DispatchRequest("path", 10**6)).expected == "TMSAO-1"
    result, record = solve_path(
        DispatchRequest("path", 3, "TMSAO-4"), module, h, Theta.cool(), PERF
    )
    assert (
        record.decision.rail == "fast"
        and record.granted == "TMSAO-4"
        and record.spent >= len(result.steps)
    )
