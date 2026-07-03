"""D1 first slice: training as a PLANNED claim graph (`kbcir/train_graph.py`).

One training step is six chained phases (forward -> activation -> loss -> reduce -> backward ->
update), each a first-class claim: the step verifies under the R-laws, prices under the tropical
optimizer, composes over steps via `kbcir.compose` (a run is a Seq -- series-summed cost),
respects an RCSP budget (feasible-or-not BEFORE execution), and is R13-deterministic. The step
count bridges structurally to the M3 loop (`steps_for` == epochs * batches-per-epoch). Execution
stays `training.train`; this is its planning/verification shadow (prototype-then-port)."""

import ast

from bcir.kbcir import TARGETS
from bcir.kbcir.cost import Theta
from bcir.kbcir.rcsp import Budget, Infeasible
from bcir.kbcir.realize import optimize
from bcir.kbcir.train_graph import (N_STAGES, TrainStepSpec, plan_train_run, steps_for,
                                    train_run_region, train_step_module)
from bcir.verify import verify, verify_smart_lowering

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def test_train_step_module_is_law_clean():
    m = train_step_module(TrainStepSpec())
    assert verify(m) == [], verify(m)                          # R1-R8 static laws
    assert verify_smart_lowering(m) == []                      # R14-R22 advisory laws (incl. gem seams)


def test_train_step_prices_and_realizes_in_stage_order():
    m = train_step_module(TrainStepSpec())
    res = optimize(m, AVX, COOL)
    assert res.score > 0
    phases = [s.phase_id for s in res.steps]
    assert phases == sorted(phases), phases                    # forward before update, always
    assert len(phases) == N_STAGES


def test_train_step_hash_is_deterministic_r13():
    from bcir.kbcir.provenance import hash_module
    a = hash_module(train_step_module(TrainStepSpec()))
    b = hash_module(train_step_module(TrainStepSpec()))
    assert a == b and a != 0


def test_composed_run_is_the_series_sum_of_steps():
    spec = TrainStepSpec()
    one = plan_train_run(spec, 1, AVX, COOL)
    three = plan_train_run(spec, 3, AVX, COOL)
    assert one.leaves == N_STAGES and three.leaves == 3 * N_STAGES
    assert three.worst_cost == 3 * one.worst_cost              # Seq = series sum, no branch spread
    assert three.expected_cost == three.worst_cost


def test_budget_feasibility_is_decided_before_execution():
    spec = TrainStepSpec()
    one = plan_train_run(spec, 1, AVX, COOL)
    # a generous cap on every dimension: feasible, same plan shape.
    generous = Budget(caps=tuple((i, 10 ** 12) for i in range(12)))
    assert plan_train_run(spec, 2, AVX, COOL, budget=generous).leaves == 2 * N_STAGES
    # an absurd compute cap: the run is INFEASIBLE -- known before a single step executes.
    try:
        plan_train_run(spec, 2, AVX, COOL, budget=Budget(caps=((0, 0),)))
        assert False, "a zero compute budget must be infeasible"
    except Infeasible:
        pass
    assert one.worst_cost > 0


def test_steps_for_bridges_to_the_m3_loop_shape():
    # 24 examples, batch 8, 5 epochs -> 15 optimizer steps; the planned run has one update
    # stage per step -- the structural bridge between the plan and what training.train runs.
    assert steps_for(24, 8, 5) == 15
    region = train_run_region(TrainStepSpec(), steps_for(24, 8, 5))
    assert len(region.parts) == 15


def test_train_graph_touches_no_verifier_and_emits_no_diagnostic():
    # cost-side module (the two-truth quarantine): no verify import, nothing graded.
    import bcir.kbcir.train_graph as tg
    bound = set(vars(tg))
    assert not (bound & {"verify", "verify_smart_lowering", "Diagnostic", "Graded", "decide"})
    tree = ast.parse(open(tg.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)


# --- D1 step 2: the plan IS the execution path -------------------------------------------------

def _toy_logistic(n=24, nf=4, seed=0xD1E2):
    import random
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        cls = rng.random() < 0.5
        base = 1.0 if cls else -1.0
        X.append([base + rng.uniform(-0.4, 0.4) for _ in range(nf)])
        y.append(1.0 if cls else 0.0)
    return X, y


def test_hydrated_pack_carries_the_plan_with_r10_r11_clean():
    from bcir.kbcir.train_graph import hydrate_train_step, train_step_module
    from bcir.verify import verify_pack
    spec = TrainStepSpec()
    pack, result = hydrate_train_step(spec, AVX, COOL)
    assert len(pack.segments) == N_STAGES and pack.provenance_ok()
    assert verify_pack(train_step_module(spec), pack) == []
    assert [seg.claim_id for seg in pack.segments] == [s.claim_id for s in result.steps]


def test_executor_dispatches_the_planned_stage_order():
    from bcir.kbcir.train_graph import train_planned
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    run = train_planned(spec, X, y, [0.0] * spec.n_params, epochs=1, lr=0.5, h=AVX, theta=COOL)
    assert run.exec_orders and all(o == [1, 2, 3, 4, 5, 6] for o in run.exec_orders)
    assert [seg.claim_id for seg in run.pack.segments] == run.exec_orders[0]


def test_one_executed_step_matches_an_independent_reference():
    # the GEM-dispatched kernels compute EXACTLY the closed-form logistic step.
    import math
    from bcir.kbcir.train_graph import train_planned
    spec = TrainStepSpec(n_features=2, batch=4)
    X = [[0.5, -0.2], [1.0, 0.3], [-0.7, 0.9], [0.1, -1.1]]
    y = [1.0, 1.0, 0.0, 0.0]
    w0 = [0.1, -0.2, 0.05]
    run = train_planned(spec, X, y, w0, epochs=1, lr=0.3, h=AVX, theta=COOL)
    # the reference step, computed independently of the executor path
    z = [sum(X[i][j] * w0[j] for j in range(2)) + w0[2] for i in range(4)]
    act = [1.0 / (1.0 + math.exp(-v)) for v in z]
    g = [sum((act[i] - y[i]) * X[i][j] for i in range(4)) / 4 for j in range(2)]
    g.append(sum(act[i] - y[i] for i in range(4)) / 4)
    want = [w0[j] - 0.3 * g[j] for j in range(3)]
    assert all(abs(a - b) < 1e-12 for a, b in zip(run.weights, want)), (run.weights, want)


def test_planned_training_genuinely_converges():
    # THE HEADLINE: real training where every step is dispatched by the GEM executor over the
    # planned claim graph -- gated by the shared convergence gate, like every E-demo.
    from bcir.kbcir.train_graph import train_planned
    from bcir.tests._convergence import assert_converged
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    run = train_planned(spec, X, y, [0.0] * spec.n_params, epochs=120, lr=0.8, h=AVX, theta=COOL)
    assert_converged(run.losses, final_below=0.05, max_ratio=0.1, name="D1 planned training")


def test_per_epoch_manifests_are_distinct_and_replayable():
    from bcir.kbcir.provenance import replay
    from bcir.kbcir.train_graph import train_planned, train_step_module
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    run = train_planned(spec, X, y, [0.0] * spec.n_params, epochs=3, lr=0.5, h=AVX, theta=COOL)
    digests = [m.digest for m in run.manifests]
    assert len(set(digests)) == 3                              # the epoch tag moves the digest
    assert run.manifests[0].diff(run.manifests[1]) == ["artifacts"]
    m = train_step_module(spec)
    for e, man in enumerate(run.manifests):                    # each epoch replays exactly
        arts = (("epoch", e), ("topo_gen", run.pack.topo_gen),
                ("map_gen", run.pack.map_gen), ("data_gen", run.pack.data_gen))
        assert replay(man, m, AVX, COOL, artifacts=arts).score > 0


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


def test_pipelined_run_respects_raw_and_overlaps_the_metrics_tail():
    """D1 step 5: over the multi-step run module the token DAG keeps the weight-critical
    path exact -- step i+1's forward starts at or after step i's update (RAW on W) -- while
    the METRICS tail (reduce) genuinely overlaps the update on another domain."""
    from bcir.kbcir.train_graph import schedule_train_run
    spec = TrainStepSpec()
    cert, sched = schedule_train_run(spec, 4, AVX, COOL)
    assert cert.admitted, cert
    for s in range(3):
        upd = sched.slot_of(s * 10 + 6)
        fwd_next = sched.slot_of((s + 1) * 10 + 1)
        red = sched.slot_of(s * 10 + 4)
        assert fwd_next.start >= upd.finish, (s, fwd_next, upd)     # RAW on W: never violated
        assert red.start < upd.finish, (s, red, upd)                # the metrics tail overlaps
        assert red.domain != upd.domain                             # ... on its own domain


def test_overlap_win_is_real_and_scales_with_steps():
    """The certificate's three disciplines order (pipelined <= barriered <= serial), the win
    is real (measured ~34% makespan reduction, gate at 25%), and it scales linearly with the
    step count (each step contributes the same loss-tail overlap)."""
    from bcir.kbcir.train_graph import schedule_train_run
    spec = TrainStepSpec()
    c1, _ = schedule_train_run(spec, 1, AVX, COOL)
    c8, _ = schedule_train_run(spec, 8, AVX, COOL)
    assert c1.admitted and c8.admitted
    assert c8.pipelined <= 0.75 * c8.barriered, (c8.pipelined, c8.barriered)
    assert c8.overlap_win == 8 * c1.overlap_win > 0                 # per-step win, exactly
