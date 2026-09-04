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
from bcir.kbcir.train_graph import (
    N_STAGES,
    TrainStepSpec,
    plan_train_run,
    steps_for,
    train_run_region,
    train_step_module,
)
from bcir.verify import verify, verify_smart_lowering

AVX = TARGETS["x86_avx512"]
COOL = Theta.cool()


def test_train_step_module_is_law_clean():
    m = train_step_module(TrainStepSpec())
    assert verify(m) == [], verify(m)  # R1-R8 static laws
    assert verify_smart_lowering(m) == []  # R14-R22 advisory laws (incl. gem seams)


def test_train_step_prices_and_realizes_in_stage_order():
    m = train_step_module(TrainStepSpec())
    res = optimize(m, AVX, COOL)
    assert res.score > 0
    phases = [s.phase_id for s in res.steps]
    assert phases == sorted(phases), phases  # forward before update, always
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
    assert three.worst_cost == 3 * one.worst_cost  # Seq = series sum, no branch spread
    assert three.expected_cost == three.worst_cost


def test_budget_feasibility_is_decided_before_execution():
    spec = TrainStepSpec()
    one = plan_train_run(spec, 1, AVX, COOL)
    # a generous cap on every dimension: feasible, same plan shape.
    generous = Budget(caps=tuple((i, 10**12) for i in range(12)))
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
    assert len(set(digests)) == 3  # the epoch tag moves the digest
    assert run.manifests[0].diff(run.manifests[1]) == ["artifacts"]
    m = train_step_module(spec)
    for e, man in enumerate(run.manifests):  # each epoch replays exactly
        arts = (
            ("epoch", e),
            ("topo_gen", run.pack.topo_gen),
            ("map_gen", run.pack.map_gen),
            ("data_gen", run.pack.data_gen),
        )
        assert replay(man, m, AVX, COOL, artifacts=arts).score > 0


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
        assert fwd_next.start >= upd.finish, (s, fwd_next, upd)  # RAW on W: never violated
        assert red.start < upd.finish, (s, red, upd)  # the metrics tail overlaps
        assert red.domain != upd.domain  # ... on its own domain


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
    assert c8.overlap_win == 8 * c1.overlap_win > 0  # per-step win, exactly


def test_stream_step_is_law_clean_hydrates_and_updates_once():
    """D1 step 6: the STREAMED step (micro-batch streams within one step) is R-law clean,
    hydrates to an R10/R11-clean StreamPack like any program (the autodiff-closure enabler),
    keeps exactly ONE weight-update claim (mean-of-equal-split-means == the batch mean), and
    its combine genuinely awaits every stream's gradient."""
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.train_graph import train_stream_module
    from bcir.verify import verify_pack

    spec = TrainStepSpec()  # batch 8
    m = train_stream_module(spec, 4)  # 4 streams x micro-batch 2
    assert verify(m) == [], verify(m)
    assert verify_smart_lowering(m) == []
    claims = [c for ph in m.phases for c in ph.claims]
    w_writers = [c for c in claims if 2 in c.wr]  # rid 2 == the shared W
    assert len(w_writers) == 1 and w_writers[0].op == "gem.opt_step:sgd"
    combine = [c for c in claims if c.op == "reduce.grad_mean"]
    assert len(combine) == 1
    assert combine[0].rd == tuple(s * 10 + 8 for s in range(4))  # every stream's GRAD
    result = optimize(m, AVX, COOL)
    pack = hydrate(m, result, plan=m.name)
    assert pack.provenance_ok()
    assert verify_pack(m, pack) == []
    # validation: ragged micro-batches refuse loudly.
    try:
        train_stream_module(spec, 3)  # 8 % 3 != 0
        raise AssertionError("ragged micro-batches must be rejected")
    except ValueError as e:
        assert "divisible" in str(e)


def test_stream_step_overlaps_streams_and_serializes_only_on_the_update():
    """The token DAG runs the streams CONCURRENTLY (disjoint RID bands; W is read-shared,
    and read-read never conflicts): with 2 streams both forwards start at t=0 on different
    domains, the combine starts only after every stream's backward, and the single update
    is last. The certificate's win is real -- measured ~61% makespan reduction with
    4 streams; gate at 25% (the measured-then-pinned discipline)."""
    from bcir.kbcir.train_graph import schedule_stream_step

    spec = TrainStepSpec()
    cert, sched = schedule_stream_step(spec, 2, AVX, COOL)
    assert cert.admitted, cert
    f0, f1 = sched.slot_of(1), sched.slot_of(11)  # both streams' forwards
    assert f0.start == 0 and f1.start == 0  # truly concurrent from t=0
    assert f0.domain != f1.domain
    comb, upd = sched.slot_of(21), sched.slot_of(22)
    for s in (0, 1):
        assert comb.start >= sched.slot_of(s * 10 + 5).finish  # awaits every backward
    assert upd.start >= comb.finish  # the single update is last
    c4, _ = schedule_stream_step(spec, 4, AVX, COOL)
    assert c4.admitted
    assert c4.pipelined <= 0.75 * c4.barriered, (c4.pipelined, c4.barriered)


def test_streamed_gradients_average_to_the_full_batch_gradient():
    """The single-update law's numeric half: splitting a batch into equal micro-batches and
    averaging the per-stream MEAN gradients equals the full-batch mean gradient (<= 1e-12
    -- float summation order differs, exact in real arithmetic). The logistic/BCE gradient
    is the same closed form train_planned executes."""
    import random
    from bcir.kbcir.recurrent import sigmoid

    rng = random.Random(0xD16)
    nf, b, streams = 4, 8, 4
    X = [[rng.uniform(-1, 1) for _ in range(nf)] for _ in range(b)]
    y = [float(rng.randint(0, 1)) for _ in range(b)]
    w = [rng.uniform(-0.5, 0.5) for _ in range(nf + 1)]

    def grad(rows, ys):  # the closed-form mean gradient
        n = len(rows)
        act = [sigmoid(sum(r[j] * w[j] for j in range(nf)) + w[nf]) for r in rows]
        g = [sum((act[i] - ys[i]) * rows[i][j] for i in range(n)) / n for j in range(nf)]
        g.append(sum(act[i] - ys[i] for i in range(n)) / n)
        return g

    full = grad(X, y)
    mb = b // streams
    parts = [grad(X[s * mb : (s + 1) * mb], y[s * mb : (s + 1) * mb]) for s in range(streams)]
    combined = [sum(p[j] for p in parts) / streams for j in range(nf + 1)]
    assert all(abs(a - c) <= 1e-12 for a, c in zip(full, combined))


def test_d1_8_the_stream_count_is_a_plan_decision():
    """D1 step 8: the optimizer CHOOSES the stream count from the cost model -- the
    sweep covers every batch divisor, the chosen count's pipelined makespan is the
    swept minimum (measured here: the frontier is genuinely non-monotonic, so the
    choice is a real decision, not a formality), it never loses to the unstreamed
    step, and max_streams caps the sweep honestly."""
    from bcir.kbcir.train_graph import plan_stream_count

    spec = TrainStepSpec(batch=8)
    plan = plan_stream_count(spec, AVX, COOL)
    assert [s for s, _ in plan.swept] == [1, 2, 4, 8]  # every divisor priced
    assert plan.admitted
    assert plan.certificate.streams == plan.streams
    assert plan.certificate.pipelined == min(mk for _, mk in plan.swept)
    assert plan.certificate.pipelined <= dict(plan.swept)[1]  # never worse than 1x
    assert len({mk for _, mk in plan.swept}) > 1  # the sweep discriminates
    capped = plan_stream_count(spec, AVX, COOL, max_streams=2)
    assert capped.streams in (1, 2)
    assert [s for s, _ in capped.swept] == [1, 2]


# --- specification-faithful numerics and tail batches ---------------------------------------


def test_train_step_spec_rejects_unsupported_or_incoherent_choices():
    for kwargs, needle in (
        ({"n_features": 0}, "n_features"),
        ({"batch": 0}, "batch"),
        ({"activation": "softmax"}, "activation"),
        ({"loss": "hinge"}, "loss"),
        ({"activation": "relu", "loss": "bce"}, "sigmoid"),
        ({"optimizer": "lion"}, "optimizer"),
    ):
        try:
            TrainStepSpec(**kwargs)
            raise AssertionError(f"expected TrainStepSpec refusal for {kwargs}")
        except ValueError as exc:
            assert needle in str(exc), (kwargs, exc)


def test_activation_loss_logit_gradients_match_finite_difference():
    from bcir.kbcir.train_graph import _activation_and_derivative, _loss_and_dz

    eps = 1e-6
    for activation, loss in (
        ("sigmoid", "bce"),
        ("relu", "mse"),
        ("sigmoid", "mse"),
        ("tanh", "mse"),
        ("gelu", "mse"),
    ):
        for z, target in ((-0.7, 0.2), (0.4, 0.8)):

            def value(at):
                pred = _activation_and_derivative(activation, at)[0]
                return _loss_and_dz(loss, activation, at, pred, target)[0]

            pred = _activation_and_derivative(activation, z)[0]
            analytic = _loss_and_dz(loss, activation, z, pred, target)[1]
            numeric = (value(z + eps) - value(z - eps)) / (2.0 * eps)
            assert abs(analytic - numeric) <= 2e-7 * max(1.0, abs(analytic), abs(numeric)), (
                activation,
                loss,
                z,
                analytic,
                numeric,
            )


def test_stateful_optimizer_resources_and_reference_trajectories():
    from bcir.kbcir.train_graph import _OptimizerRuntime
    from bcir.lower.optimizers import adam_step, momentum_step, rmsprop_step, sgd_step

    grads = ([0.2, -0.4, 0.1], [-0.3, 0.5, -0.2], [0.1, 0.2, -0.6])
    for name in ("sgd", "momentum", "rmsprop", "adam"):
        runtime = _OptimizerRuntime(name, 3)
        got = [0.5, -0.25, 0.75]
        want = list(got)
        state1 = state2 = None
        step = 0
        for grad in grads:
            runtime.update(got, list(grad), 0.05)
            if name == "sgd":
                want = sgd_step(want, grad, 0.05)
            elif name == "momentum":
                if state1 is None:
                    state1 = [0.0] * 3
                want, state1 = momentum_step(want, grad, state1, 0.05)
            elif name == "rmsprop":
                if state1 is None:
                    state1 = [0.0] * 3
                want, state1 = rmsprop_step(want, grad, state1, 0.05)
            else:
                if state1 is None:
                    state1 = [0.0] * 3
                if state2 is None:
                    state2 = [0.0] * 3
                want, state1, state2, step = adam_step(want, grad, state1, state2, step, 0.05)
            assert got == want

        module = train_step_module(TrainStepSpec(n_features=2, optimizer=name))
        names = {resource.name for resource in module.resources.values()}
        update = next(c for phase in module.phases for c in phase.claims if c.id == 6)
        if name == "sgd":
            assert not (names & {"VELOCITY", "SQ_AVG", "ADAM_M", "ADAM_V", "ADAM_STEP"})
        elif name == "momentum":
            assert "VELOCITY" in names and 90_001 in update.rd and 90_001 in update.wr
        elif name == "rmsprop":
            assert "SQ_AVG" in names and 90_001 in update.rd and 90_001 in update.wr
        else:
            assert {"ADAM_M", "ADAM_V", "ADAM_STEP"} <= names
            assert {90_001, 90_002, 90_003} <= set(update.rd) & set(update.wr)
        assert verify(module) == [], verify(module)


def test_nondefault_spec_changes_real_training_not_only_provenance():
    from bcir.kbcir.train_graph import train_planned

    X = [[0.5, -0.2], [1.0, 0.3], [-0.7, 0.9], [0.1, -1.1]]
    y = [1.0, 1.0, 0.0, 0.0]
    w0 = [0.1, -0.2, 0.05]
    default = train_planned(TrainStepSpec(2, 4), X, y, w0, epochs=2, lr=0.03, h=AVX, theta=COOL)
    alternate = train_planned(
        TrainStepSpec(2, 4, "tanh", "mse", "adam"), X, y, w0, epochs=2, lr=0.03, h=AVX, theta=COOL
    )
    assert alternate.weights != default.weights
    assert alternate.losses != default.losses


def test_planned_tail_executes_and_streamed_tail_is_exact_or_refuses_before_training():
    from bcir.kbcir.train_graph import train_planned, train_streamed

    spec = TrainStepSpec(n_features=2, batch=4, activation="tanh", loss="mse", optimizer="adam")
    X = [[0.2, -0.1], [0.4, 0.3], [-0.2, 0.5], [0.1, -0.4], [0.7, 0.2], [0.3, -0.2]]
    y = [0.1, 0.7, 0.3, 0.8, 0.2, 0.6]
    w0 = [0.1, -0.2, 0.05]
    planned = train_planned(spec, X, y, w0, epochs=2, lr=0.01, h=AVX, theta=COOL)
    streamed = train_streamed(spec, 2, X, y, w0, epochs=2, lr=0.01, h=AVX, theta=COOL)
    assert len(planned.exec_orders) == len(streamed.exec_orders) == 4  # full + tail, twice
    assert all(abs(a - b) <= 1e-12 for a, b in zip(planned.weights, streamed.weights))
    assert all(abs(a - b) <= 1e-12 for a, b in zip(planned.losses, streamed.losses))

    original = list(w0)
    try:
        train_streamed(spec, 2, X[:5], y[:5], w0, epochs=1, lr=0.01, h=AVX, theta=COOL)
        raise AssertionError("a remainder of one cannot split into two equal streams")
    except ValueError as exc:
        assert "remainder" in str(exc) and "no training" in str(exc)
    assert w0 == original


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
