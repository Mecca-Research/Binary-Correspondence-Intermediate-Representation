"""D1 step 4 (ML/AI roadmap §8.2): the train step COMPUTES in C -- training as a C artifact.

Steps 1-3 made a training step a planned, hydrated, binary-executable artifact whose
dispatch runs with no Python. This step puts the six NUMERIC stage kernels behind the C
executor's per-claim callback (`runtime/c/bcir_train.c`, kernel-for-kernel twin of the
oracle's `_step_kernels`) and runs the whole training loop in C (`test_train.c`): binary
pack in, per-epoch loss curve + trained weights out. The differential gate: the C loss
curve and final weights match the oracle's `train_planned` to float round-off (same
accumulation order, same guarded sigmoid, same eps-clamped BCE -- on a shared libm the two
rails agree to the last bit), the first step's dispatch order is the executor's [1..6], and
the C-run curve passes the SAME shared convergence gate as every E-demo -- real training,
no Python in the loop. Toolchain-gated: returns early when no C compiler is on PATH."""

import os
import subprocess
import tempfile

from bcir.abi import encode
from bcir.kbcir.cost import TargetProfile, Theta
from bcir.kbcir.train_graph import TrainStepSpec, train_planned
from bcir.tests.test_c_executor import _RUNTIME_C, _cc
from bcir.tests.test_train_graph import _toy_logistic

AVX = TargetProfile.x86_avx512()
COOL = Theta.cool()


def _build(tmp):
    exe = os.path.join(tmp, "test_train")
    srcs = [os.path.join(_RUNTIME_C, f)
            for f in ("test_train.c", "bcir_train.c", "bcir_exec.c", "bcir_runtime.c")]
    r = subprocess.run([_cc(), "-std=c11", "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C,
                        *srcs, "-o", exe, "-lm"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def _run_c_training(tmp, exe, run, spec, X, y, w0, epochs, lr):
    """Feed the oracle-encoded pack + the exact dataset to the C harness; parse its curve."""
    pack_path = os.path.join(tmp, "step.pack")
    with open(pack_path, "wb") as f:
        f.write(encode(run.pack))
    data_path = os.path.join(tmp, "data.txt")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write(f"{spec.n_features} {spec.batch} {len(X)} {epochs} {lr!r}\n")
        for row in X:
            f.write(" ".join(repr(float(v)) for v in row) + "\n")
        f.write(" ".join(repr(float(v)) for v in y) + "\n")
        f.write(" ".join(repr(float(v)) for v in w0) + "\n")
    r = subprocess.run([exe, pack_path, data_path], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    losses, order, weights = [], [], []
    for line in r.stdout.strip().splitlines():
        if line.startswith("epoch "):
            losses.append(float(line.split()[1]))
        elif line.startswith("order:"):
            order = [int(t) for t in line.split()[1:]]
        elif line.startswith("weights:"):
            weights = [float(t) for t in line.split()[1:]]
    return losses, order, weights


def test_c_training_matches_the_oracle_curve_and_weights():
    """The differential gate: same pack, same data, same loop -- the C rail's per-epoch
    losses and trained weights equal train_planned's to float round-off, and the six
    stages were sequenced by the EXECUTOR ([1..6] recorded in the first step)."""
    if _cc() is None:
        return
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    w0 = [0.0] * spec.n_params
    run = train_planned(spec, X, y, w0, epochs=12, lr=0.5, h=AVX, theta=COOL)
    with tempfile.TemporaryDirectory() as tmp:
        losses, order, weights = _run_c_training(tmp, _build(tmp), run, spec, X, y, w0, 12, 0.5)
    assert order == [1, 2, 3, 4, 5, 6], order
    assert len(losses) == len(run.losses) == 12
    assert all(abs(c - p) <= 1e-12 for c, p in zip(losses, run.losses)), \
        [(c, p) for c, p in zip(losses, run.losses) if abs(c - p) > 1e-12]
    assert len(weights) == spec.n_params
    assert all(abs(c - p) <= 1e-12 for c, p in zip(weights, run.weights)), (weights, run.weights)


def test_c_training_genuinely_converges():
    """THE HEADLINE: real convergence where every stage is C -- the binary pack dispatched
    by bcir_exec, the numerics computed by bcir_train.c -- gated by the SAME shared
    convergence gate as the oracle's D1 run (and equal to it to float round-off)."""
    if _cc() is None:
        return
    from bcir.tests._convergence import assert_converged
    spec = TrainStepSpec()
    X, y = _toy_logistic()
    w0 = [0.0] * spec.n_params
    run = train_planned(spec, X, y, w0, epochs=120, lr=0.8, h=AVX, theta=COOL)
    with tempfile.TemporaryDirectory() as tmp:
        losses, _order, _weights = _run_c_training(tmp, _build(tmp), run, spec, X, y, w0, 120, 0.8)
    assert_converged(losses, final_below=0.05, max_ratio=0.1, name="D1 step-4 C-kernel training")
    assert abs(losses[-1] - run.losses[-1]) <= 1e-12, (losses[-1], run.losses[-1])


def test_c_streamed_training_matches_both_oracle_paths():
    """D1 step 7: the STREAMED step trains in C -- the binary stream pack dispatched by
    bcir_exec, the per-stream kernels + gradient combine + SINGLE update computed by
    bcir_stream_kernel. Gates: (1) the C curve and trained weights equal the oracle's
    train_streamed to float round-off; (2) BOTH equal the plain train_planned run to
    <= 1e-12 (the mean-of-equal-split-means identity, now end-to-end on the C rail);
    (3) the first step's dispatch order is the executor's per-stream bands then combine
    then update -- exactly the streamed claim graph's phase order."""
    if _cc() is None:
        return
    from bcir.abi import encode as abi_encode
    from bcir.gem.streampack import hydrate
    from bcir.kbcir.realize import optimize
    from bcir.kbcir.train_graph import train_stream_module, train_streamed
    spec = TrainStepSpec()                                   # batch 8
    streams = 4
    X, y = _toy_logistic()
    w0 = [0.0] * spec.n_params
    planned = train_planned(spec, X, y, w0, epochs=12, lr=0.5, h=AVX, theta=COOL)
    streamed = train_streamed(spec, streams, X, y, w0, epochs=12, lr=0.5, h=AVX, theta=COOL)
    m = train_stream_module(spec, streams)
    pack = hydrate(m, optimize(m, AVX, COOL), plan=m.name)
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "test_train_stream")
        srcs = [os.path.join(_RUNTIME_C, f)
                for f in ("test_train_stream.c", "bcir_train.c", "bcir_exec.c",
                          "bcir_runtime.c")]
        r = subprocess.run([_cc(), "-std=c11", "-O2", "-Wall", "-Wextra", "-I", _RUNTIME_C,
                            *srcs, "-o", exe, "-lm"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        pack_path = os.path.join(tmp, "stream.pack")
        with open(pack_path, "wb") as f:
            f.write(abi_encode(pack))
        data_path = os.path.join(tmp, "data.txt")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(f"{spec.n_features} {spec.batch} {len(X)} 12 {streams} {0.5!r}\n")
            for row in X:
                f.write(" ".join(repr(float(v)) for v in row) + "\n")
            f.write(" ".join(repr(float(v)) for v in y) + "\n")
            f.write(" ".join(repr(float(v)) for v in w0) + "\n")
        r = subprocess.run([exe, pack_path, data_path], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        losses, order, weights = [], [], []
        for line in r.stdout.strip().splitlines():
            if line.startswith("epoch "):
                losses.append(float(line.split()[1]))
            elif line.startswith("order:"):
                order = [int(t) for t in line.split()[1:]]
            elif line.startswith("weights:"):
                weights = [float(t) for t in line.split()[1:]]
    want_order = [s * 10 + k for s in range(streams) for k in range(1, 6)]
    want_order += [streams * 10 + 1, streams * 10 + 2]
    assert order == want_order == streamed.exec_orders[0], (order, streamed.exec_orders[0])
    assert len(losses) == 12
    assert all(abs(c - p) <= 1e-12 for c, p in zip(losses, streamed.losses))
    assert all(abs(c - p) <= 1e-12 for c, p in zip(weights, streamed.weights))
    assert all(abs(c - p) <= 1e-12 for c, p in zip(losses, planned.losses))   # the identity,
    assert all(abs(c - p) <= 1e-12 for c, p in zip(weights, planned.weights))  # C end-to-end


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
