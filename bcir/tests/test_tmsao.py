"""TMSAO data-path performance and secondary correctness regressions.

The timing audit is informational; these tests pin equivalence, bounded structural
work, deep-graph safety, and fail-closed data handling without scheduler-sensitive
wall-clock assertions.
"""

from __future__ import annotations

import random

from bcir.examples import vector_add
from bcir.frontends.models.assessment import BankEnvelope, HardwareEnvelope
from bcir.gem import concurrency
from bcir.gem.async_tokens import async_plan
from bcir.gem.concurrency import _conflict, schedule_concurrent
from bcir.gem.execute import execute
from bcir.gem.schedule import schedule_eft
from bcir.gem.streampack import hydrate
from bcir.kbcir.allocator import live_intervals, pool_plan, profile_resources
from bcir.kbcir.classical import KnnModel, knn_classify, knn_neighbours, squared_distance
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.realize import ChosenStep, RealizationResult, optimize
from bcir.kbcir.static_memory import plan_static_memory
from bcir.kbcir.unsupervised import (EmbeddingTable, embedding_lookup, k_fold_indices,
                                     kmeans_fit, kmeans_inertia, minmax_fit,
                                     standard_scaler_fit)
from bcir.model import (Claim, Lane, Module, Opcode, Phase, Resource, StrideClass,
                        phase_graph_has_cycle, topological_phase_ids)
from bcir.verify import verify


AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _claim(claim_id: int, rd=(), wr=(), *, cost_class="bandwidth") -> Claim:
    return Claim(claim_id, Opcode.ADD, Lane.U, StrideClass.UNIT, 8,
                 rd=tuple(rd), wr=tuple(wr), cost_class=cost_class)


def _random_claims(seed: int, count: int) -> list[Claim]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        reads = tuple(sorted(rng.sample(range(1, 17), rng.randint(0, 3))))
        writes = tuple(sorted(rng.sample(range(1, 17), rng.randint(0, 2))))
        rows.append(_claim(1000 + index, reads, writes,
                           cost_class=rng.choice(("compute", "bandwidth"))))
    return rows


def _module_with_claims(claims: list[Claim]) -> Module:
    module = Module("adversarial")
    for rid in range(1, 17):
        module.add_resource(Resource(rid, shape=(8,)))
    module.add_phase(Phase(0, (), claims))
    return module


def _legacy_predecessors(claims: list[Claim]) -> dict[int, list[int]]:
    return {claim.id: [prior.id for prior in claims[:index] if _conflict(prior, claim)]
            for index, claim in enumerate(claims)}


def _legacy_waves(claims: list[Claim]) -> dict[int, int]:
    wave_of: dict[int, int] = {}
    for index, claim in enumerate(claims):
        wave = 0
        for prior in claims[:index]:
            if _conflict(prior, claim):
                wave = max(wave, wave_of[prior.id] + 1)
        wave_of[claim.id] = wave
    return wave_of


def _legacy_eft(claims: list[Claim], predecessors: dict[int, list[int]],
                durations: dict[int, int]):
    pending = list(claims)
    finish_of: dict[int, int] = {}
    domain_free = [0]
    resident = [set()]
    slots = []
    while pending:
        ready = [claim for claim in pending
                 if all(prior in finish_of for prior in predecessors[claim.id])]
        claim = sorted(ready, key=lambda row: (-durations[row.id], row.id))[0]
        pending.remove(claim)
        start = max(domain_free[0],
                    max((finish_of[prior] for prior in predecessors[claim.id]), default=0))
        finish = start + durations[claim.id]
        domain_free[0] = finish
        resident[0].update(claim.rd); resident[0].update(claim.wr)
        finish_of[claim.id] = finish
        slots.append((claim.id, 0, start, finish))
    return slots


def test_deep_reversed_phase_dag_is_iterative_across_every_execution_rail():
    count = 2048
    module = Module("deep-reversed")
    for phase_id in reversed(range(count)):
        claim = Claim(10_000 + phase_id, Opcode.NOP, Lane.H, StrideClass.SCALAR)
        module.add_phase(Phase(phase_id, (phase_id - 1,) if phase_id else (), [claim]))
    assert not phase_graph_has_cycle(module)
    assert topological_phase_ids(module) == list(range(count))
    assert verify(module) == []
    assert execute(module).phase_order == list(range(count))
    result = optimize(module, AVX, COOL)
    assert len(result.steps) == count
    assert [row.phase_id for row in result.steps] == list(range(count))


def test_deep_phase_cycle_is_detected_without_recursion():
    count = 4096
    module = Module("deep-cycle")
    for phase_id in reversed(range(count)):
        dependency = phase_id - 1 if phase_id else count - 1
        module.add_phase(Phase(phase_id, (dependency,), []))
    assert phase_graph_has_cycle(module)
    assert any(row.law == "R4" and "cycle" in row.message for row in verify(module))


def test_resource_indexed_waves_and_tokens_match_pairwise_semantics():
    for seed in range(100):
        claims = _random_claims(seed, 1 + seed % 48)
        module = _module_with_claims(claims)
        waves = schedule_concurrent(module)
        wave_of = {claim_id: wave
                   for wave, members in enumerate(waves.waves) for claim_id in members}
        assert wave_of == _legacy_waves(claims)
        assert async_plan(module).awaits == _legacy_predecessors(claims)


def test_ready_heap_eft_matches_full_pending_scan():
    for seed in range(100):
        claims = _random_claims(1000 + seed, 1 + seed % 40)
        module = _module_with_claims(claims)
        durations = {claim.id: ((claim.id * 17 + seed) % 31) + 1 for claim in claims}
        expected = _legacy_eft(claims, _legacy_predecessors(claims), durations)
        actual = [(row.claim_id, row.domain, row.start, row.finish)
                  for row in schedule_eft(module, durations).slots]
        assert actual == expected


def test_independent_wave_and_token_planning_touch_each_claim_once():
    count = 1024
    claims = [_claim(1000 + index, (index + 1,), (index + 1,)) for index in range(count)]
    module = Module("independent")
    for rid in range(1, count + 1):
        module.add_resource(Resource(rid, shape=(8,)))
    module.add_phase(Phase(0, (), claims))
    original = concurrency._claim_accesses
    touches = 0

    def counted(claim):
        nonlocal touches
        touches += 1
        return original(claim)

    concurrency._claim_accesses = counted
    try:
        schedule_concurrent(module)
        assert touches == count
        touches = 0
        async_plan(module)
        assert touches == count
    finally:
        concurrency._claim_accesses = original


def test_schedulers_fail_closed_on_duplicate_claim_identity():
    module = _module_with_claims([_claim(7, (1,), (2,)), _claim(7, (3,), (4,))])
    for function, arguments in ((schedule_concurrent, (module,)),
                                (async_plan, (module,)),
                                (schedule_eft, (module, {7: 1}))):
        try:
            function(*arguments)
            raise AssertionError("duplicate claim identity must not produce an ambiguous schedule")
        except ValueError:
            pass


def _out_of_order_lifetime_module() -> Module:
    module = Module("out-of-order-lifetimes")
    module.add_resource(Resource(1, shape=(16,)))
    module.add_resource(Resource(2, shape=(16,)))
    module.add_phase(Phase(2, (1,), [_claim(12, (1,), (1,))]))
    module.add_phase(Phase(0, (), [_claim(10, (1,), (1,))]))
    module.add_phase(Phase(1, (0,), [_claim(11, (2,), (2,))]))
    return module


def _hardware() -> HardwareEnvelope:
    bank = BankEnvelope("ram", "RAM", "host", 1 << 30, 1 << 30, 1, 1)
    return HardwareEnvelope("audit-host", "x86_64", (bank,))


def test_topological_liveness_prevents_unsafe_static_aliasing():
    module = _out_of_order_lifetime_module()
    assert live_intervals(module) == {1: (0, 2), 2: (1, 1)}
    assert profile_resources(module)[1].lifespan == 3
    assert pool_plan(module).peak_bytes == 128
    plan = plan_static_memory(module, {1: "ram", 2: "ram"}, _hardware())
    offsets = {row.rid: row.offset for row in plan.allocations}
    assert offsets[1] != offsets[2]


def _legacy_pool(module: Module):
    intervals = live_intervals(module)
    sizes = {rid: module.resource(rid).count * module.resource(rid).elem_bytes
             for rid in intervals}
    arenas = []
    for rid in sorted(intervals, key=lambda value: (intervals[value][0], value)):
        lo, hi = intervals[rid]
        slot = next((row for row in arenas if row["last"] < lo), None)
        if slot is None:
            arenas.append({"members": [rid], "last": hi, "size": sizes[rid]})
        else:
            slot["members"].append(rid); slot["last"] = hi
            slot["size"] = max(slot["size"], sizes[rid])
    return tuple((tuple(row["members"]), row["size"]) for row in arenas)


def test_heap_pooling_matches_original_first_available_arena_semantics():
    for seed in range(100):
        rng = random.Random(50_000 + seed)
        phase_count = rng.randint(1, 16)
        module = Module(f"pool-{seed}")
        phases = [Phase(index, (index - 1,) if index else (), [])
                  for index in range(phase_count)]
        for rid in range(1, rng.randint(2, 64)):
            module.add_resource(Resource(rid, elem_bytes=rng.choice((1, 2, 4, 8)),
                                         shape=(rng.randint(1, 65),)))
            for phase_id in rng.sample(range(phase_count),
                                       rng.randint(1, min(phase_count, 5))):
                phases[phase_id].claims.append(_claim(200_000 + rid * 32 + phase_id,
                                                      (rid,), (rid,)))
        module.phases.extend(phases)
        expected = _legacy_pool(module)
        actual = pool_plan(module)
        assert tuple((row.members, row.size_bytes) for row in actual.pools) == expected


def _legacy_static_offsets(module: Module) -> dict[int, int]:
    intervals = live_intervals(module)
    placed = []
    offsets = {}
    for rid in sorted(intervals, key=lambda item: (intervals[item][0], item)):
        resource = module.resource(rid)
        size = resource.count * resource.elem_bytes
        lo, hi = intervals[rid]
        alignment = max(resource.align, 64)
        active = sorted((row for row in placed
                         if not (row[5] < lo or hi < row[4])), key=lambda row: (row[1], row[0]))
        offset = 0
        for row in active:
            offset = (offset + alignment - 1) & ~(alignment - 1)
            if offset + size <= row[1]:
                break
            offset = max(offset, row[2])
        offset = (offset + alignment - 1) & ~(alignment - 1)
        placed.append((rid, offset, offset + size, size, lo, hi))
        offsets[rid] = offset
    return offsets


def test_static_free_list_matches_the_original_first_fit_layout():
    for seed in range(100):
        rng = random.Random(seed)
        phase_count = rng.randint(1, 12)
        resource_count = rng.randint(1, 48)
        module = Module(f"static-{seed}")
        phases = [Phase(index, (index - 1,) if index else (), [])
                  for index in range(phase_count)]
        bindings = {}
        for rid in range(1, resource_count + 1):
            module.add_resource(Resource(rid, elem_bytes=rng.choice((1, 2, 4, 8)),
                                         shape=(rng.randint(1, 33),),
                                         align=rng.choice((64, 128, 256))))
            bindings[rid] = "ram"
            touched = rng.sample(range(phase_count), rng.randint(1, min(phase_count, 4)))
            for phase_id in touched:
                phases[phase_id].claims.append(_claim(100_000 + rid * 32 + phase_id,
                                                      (rid,), (rid,)))
        for phase in phases:
            module.add_phase(phase)
        expected = _legacy_static_offsets(module)
        plan = plan_static_memory(module, bindings, _hardware())
        assert {row.rid: row.offset for row in plan.allocations} == expected


def _legacy_kmeans(X, n_samples, n_feat, k, init_indices, n_iter):
    values = [float(value) for value in X]
    centroids = [values[init_indices[c] * n_feat + j]
                 for c in range(k) for j in range(n_feat)]
    labels = [0] * n_samples

    def assign(point):
        best = 0
        best_distance = squared_distance(point, list(centroids[:n_feat]))
        for centroid in range(1, k):
            distance = squared_distance(
                point, list(centroids[centroid * n_feat:(centroid + 1) * n_feat]))
            if distance < best_distance:
                best, best_distance = centroid, distance
        return best

    for _ in range(n_iter):
        for sample in range(n_samples):
            labels[sample] = assign(values[sample * n_feat:(sample + 1) * n_feat])
        sums = [0.0] * (k * n_feat); counts = [0] * k
        for sample, centroid in enumerate(labels):
            counts[centroid] += 1
            for feature in range(n_feat):
                sums[centroid * n_feat + feature] += values[sample * n_feat + feature]
        updated = list(centroids)
        for centroid in range(k):
            if counts[centroid]:
                for feature in range(n_feat):
                    updated[centroid * n_feat + feature] = (
                        sums[centroid * n_feat + feature] / counts[centroid])
        centroids = updated
    for sample in range(n_samples):
        labels[sample] = assign(values[sample * n_feat:(sample + 1) * n_feat])
    return centroids, labels


def test_flat_kmeans_matches_slice_reference_over_odd_data_patterns():
    for seed in range(100):
        rng = random.Random(seed)
        samples = rng.randint(1, 24); features = rng.randint(1, 8)
        clusters = rng.randint(1, min(samples, 6)); iterations = rng.randint(0, 8)
        values = [rng.randint(-4, 4) + rng.random()
                  for _ in range(samples * features)]
        initial = rng.sample(range(samples), clusters)
        actual = kmeans_fit(values, samples, features, clusters, initial, iterations)
        assert actual == _legacy_kmeans(
            values, samples, features, clusters, initial, iterations)
        assert kmeans_inertia(values, samples, features, *actual) >= 0.0


def test_unlabeled_inputs_reject_silent_numeric_coercion_and_nonfinite_values():
    table = EmbeddingTable((0.0, 1.0, 2.0, 3.0), 2, 2)
    for bad in ([0.5], [True], ["1"]):
        try:
            embedding_lookup(table, bad, 2)
            raise AssertionError(f"embedding id {bad!r} must fail")
        except ValueError:
            pass
    values = [0.0, 0.0, 1.0, 1.0]
    for iterations in (True, 1.5):
        try:
            kmeans_fit(values, 2, 2, 1, [0], iterations)
            raise AssertionError("fractional/Boolean iteration count must fail")
        except ValueError:
            pass
    try:
        kmeans_fit([0.0, float("nan"), 1.0, 1.0], 2, 2, 1, [0], 1)
        raise AssertionError("non-finite unlabeled data must fail")
    except ValueError:
        pass
    try:
        kmeans_inertia(values, 2, 2, [0.0, 0.0], [0, 0.5])
        raise AssertionError("fractional cluster labels must fail")
    except ValueError:
        pass


def test_classical_and_preprocessing_inputs_fail_closed_on_odd_numeric_patterns():
    model = KnnModel((0.0, 0.0, 1.0, 1.0), (0.0, 1.0), 2)
    for bad_k in (True, 1.5, "1"):
        try:
            knn_neighbours(model, [0.0, 0.0], bad_k)
            raise AssertionError("KNN k must be an exact integer")
        except ValueError:
            pass
    for bad_query in ([float("nan"), 0.0], [float("inf"), 0.0]):
        try:
            knn_neighbours(model, bad_query, 1)
            raise AssertionError("KNN non-finite queries must fail")
        except ValueError:
            pass
    fractional_labels = KnnModel((0.0, 1.0), (0.5, 1.0), 1)
    try:
        knn_classify(fractional_labels, [0.0], 1)
        raise AssertionError("fractional KNN class labels must not be truncated")
    except ValueError:
        pass
    for function in (standard_scaler_fit, minmax_fit):
        for dimensions in ((True, 1), (1, 1.5)):
            try:
                function([0.0], *dimensions)
                raise AssertionError("scaler dimensions must be exact integers")
            except ValueError:
                pass
        try:
            function([0.0, float("nan")], 2, 1)
            raise AssertionError("non-finite scaler data must fail")
        except ValueError:
            pass
    for arguments in ((True, 2, 0), (4, 2.0, 0), (4, 2, 1.5)):
        try:
            k_fold_indices(*arguments)
            raise AssertionError("fold dimensions and seed must be exact integers")
        except ValueError:
            pass


def test_hydrate_refuses_partial_unknown_duplicate_and_phase_mismatched_plans():
    module = vector_add(16)
    result = optimize(module, AVX, COOL)
    step = result.steps[0]
    malformed = (
        RealizationResult([], 0),
        RealizationResult([ChosenStep(999_999, step.phase_id, step.candidate, step.cost)], step.cost),
        RealizationResult([step, step], result.score * 2),
        RealizationResult([ChosenStep(step.claim_id, step.phase_id + 1,
                                      step.candidate, step.cost)], result.score),
    )
    for plan in malformed:
        try:
            hydrate(module, plan)
            raise AssertionError("malformed/partial plan must not hydrate")
        except ValueError:
            pass
