"""D1 step 3 (ML/AI roadmap §8.2): the train step as a NO-PYTHON hot artifact.

Step 1 made a training step a planned claim graph; step 2 made the GEM executor dispatch
real training over it. This step closes the loop D1 promised: the SAME hydrated train-step
pack, encoded to the binary StreamPack ABI (`bcir.abi.encode`), executes through the C
executor (`runtime/c/bcir_exec.c`, driven by the `test_exec` harness) and must reproduce the
oracle executor's dispatch order, phase order and per-phase telemetry -- so one step of
training is now an artifact a C driver runs end to end with no Python in the loop. The
strongest pin ties the BINARY artifact to the REAL trajectory: the C dispatch order equals
the claim order every executed step of a convergent `train_planned` run actually used.
Toolchain-gated: each test returns early when no C compiler is on PATH."""

import tempfile

from bcir.abi import encode
from bcir.gem import execute
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.train_graph import TrainStepSpec, hydrate_train_step, train_step_module
from bcir.tests.test_c_executor import _build_harness, _cc, _run

AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def test_c_executor_runs_the_train_step_pack():
    """The headline: the binary-encoded train-step pack dispatches through bcir_exec with
    the oracle's exact order ([1..6], one stage per phase) + phase order + telemetry."""
    if _cc() is None:
        return
    spec = TrainStepSpec()
    m = train_step_module(spec)
    pack, _result = hydrate_train_step(spec, AVX, COOL)
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        c_order, c_phases = _run(exe, tmp, encode(pack))
    er = execute(m)
    assert c_order == er.order == [1, 2, 3, 4, 5, 6], (c_order, er.order)
    assert [p[0] for p in c_phases] == er.phase_order
    assert c_phases == [(ps.phase_id, ps.scheduled, ps.executed) for ps in er.phases]
    assert all(sched == 1 and ex == 1 for _pid, sched, ex in c_phases)   # one stage per phase


def test_binary_dispatch_order_is_the_real_training_order():
    """The C executor's order over the encoded pack == the dispatch order of EVERY executed
    step of a real GEM-driven training run (the D1 step-2 path) -- the binary artifact and
    the training trajectory are the same plan, not lookalikes."""
    if _cc() is None:
        return
    from bcir.kbcir.train_graph import train_planned
    from bcir.tests.test_train_graph import _toy_logistic
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    run = train_planned(spec, X, y, [0.0] * spec.n_params, epochs=2, lr=0.5, h=AVX, theta=COOL)
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        c_order, _phases = _run(exe, tmp, encode(run.pack))
    assert run.exec_orders and all(o == c_order for o in run.exec_orders), \
        (c_order, run.exec_orders[:2])


def test_cross_shape_train_packs_keep_parity():
    """Parity is not a fixed-shape accident: wider features / bigger batches re-plan, and
    the C executor still matches the oracle over each encoded pack."""
    if _cc() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        exe = _build_harness(tmp)
        for spec in (TrainStepSpec(n_features=2, batch=4),
                     TrainStepSpec(n_features=16, batch=32)):
            pack, _result = hydrate_train_step(spec, AVX, COOL)
            c_order, c_phases = _run(exe, tmp, encode(pack))
            er = execute(train_step_module(spec))
            assert c_order == er.order, (spec, c_order, er.order)
            assert c_phases == [(ps.phase_id, ps.scheduled, ps.executed) for ps in er.phases]


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
