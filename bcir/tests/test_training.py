"""M3 (ML Tier-1, Pillar 5b / B4) -- the supervised TRAINING LOOP: the capstone correctness gate.

``bcir.kbcir.training`` ties the trio together: a model forward (the B3 ``Tape`` autodiff), the M1 loss
library, the existing autodiff backward, and the M2 optimizers, into an epoch / mini-batch training loop with
a dataset abstraction, train/val split, eval metrics, and an early-stop hook. The headline: logistic
regression AND a small MLP trained end-to-end to high accuracy on toy datasets.

Covers:
  * Dataset / batching: ``minibatches`` is a deterministic permutation (same seed -> same order; every sample
    once per epoch; correct batch sizes incl. a ragged last batch); ``train_val_split`` is disjoint + covers
    all + deterministic.
  * Metrics: ``accuracy`` / ``mse_metric`` / ``binary_f1`` match an INDEPENDENT hand-computed value on a
    fixed (preds, labels), incl. a confusion-matrix anchor for F1.
  * LOGISTIC REGRESSION end-to-end (the headline): a linearly-separable 2-D binary set; ``train`` with
    BCE + Adam reaches >= ~95% train (>= ~90% val) within a bounded epoch count; the train loss is
    (near-)monotone decreasing; deterministic (same seed -> identical final loss + params).
  * MLP end-to-end: a 2-layer MLP (hidden relu + output) on a NON-linearly-separable XOR-like set, trained
    with BCE + Adam, reaches high train accuracy CLEARLY above a linear model's ceiling on that set (proving
    the hidden layer learns the nonlinearity). Bounded epochs, deterministic.
  * Softmax-CE path: a multiclass linear head trained to high accuracy (proves the transcendental softmax-CE
    seed composes through the loop).
  * MSE closed-set path: a linear regression converges to the known minimizer (the closed-set loss path).
  * Early stop: a run that plateaus on val stops early (fewer epochs than the cap) -- the hook fires.
  * The two-truth invariant: the training module imports no verifier / emits no Diagnostic; the model graphs
    it builds are ordinary closed-set Tape DAGs.

All deterministic, pure-Python (no numpy). The loop is the oracle generalization of
``autodiff_kernel.oracle_train``."""

import ast
import math

from bcir.kbcir.training import (
    Dataset,
    EarlyStop,
    TrainResult,
    accuracy,
    binary_f1,
    linear_model,
    make_linearly_separable,
    make_xor,
    minibatches,
    mlp_model,
    mlp_param_names,
    mse_metric,
    train,
    train_val_split,
    _lcg_permutation,
    _predict,
)


# A shared, deterministic MLP init: small positive hidden biases keep the relu units active at start (avoids
# dead-unit stalls), weights from the same LCG used throughout the module. Reproducible by construction.
def _mlp_init(names):
    state = 12345

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (state >> 33) / float(1 << 31) - 1.0

    return [0.1 if nm.startswith("b1_") else rnd() for nm in names]


# === Dataset / batching: deterministic permutation, full coverage, ragged last batch ===================


def test_minibatches_is_a_deterministic_permutation_covering_all_samples_once():
    ds = Dataset(
        tuple((float(i), float(-i)) for i in range(10)), tuple(float(i % 2) for i in range(10))
    )
    # same seed -> same order (the determinism the loop relies on).
    b1 = list(minibatches(ds, 3, seed=42))
    b2 = list(minibatches(ds, 3, seed=42))
    flat1 = [tuple(x) for xb, _ in b1 for x in xb]
    flat2 = [tuple(x) for xb, _ in b2 for x in xb]
    assert flat1 == flat2, "same seed must yield the same batch order"
    # a DIFFERENT seed gives a different order (it is a real shuffle, not the identity).
    b3 = [tuple(x) for xb, _ in minibatches(ds, 3, seed=7) for x in xb]
    assert b3 != flat1, "a different seed should reshuffle"
    # every sample appears EXACTLY ONCE per epoch (a permutation).
    seen = sorted(flat1)
    assert seen == sorted(tuple(row) for row in ds.X), (
        "an epoch must cover every sample exactly once"
    )
    assert len(flat1) == len(ds), "no sample dropped or duplicated"


def test_minibatches_batch_sizes_including_a_ragged_last_batch():
    ds = Dataset(tuple((float(i),) for i in range(10)), tuple(0.0 for _ in range(10)))
    sizes = [len(xb) for xb, _ in minibatches(ds, 3, seed=1)]
    # 10 samples, batch 3 -> [3, 3, 3, 1] (the last is the ragged remainder).
    assert sizes == [3, 3, 3, 1], sizes
    # x and y batches stay aligned in length.
    for xb, yb in minibatches(ds, 4, seed=1):
        assert len(xb) == len(yb)
    # batch_size >= n -> a single full-dataset batch.
    full = list(minibatches(ds, 100, seed=1))
    assert len(full) == 1 and len(full[0][0]) == 10


def test_lcg_permutation_is_a_valid_deterministic_permutation():
    for seed in (0, 1, 5, 999):
        p = _lcg_permutation(12, seed)
        assert sorted(p) == list(range(12)), (seed, p)  # a valid permutation of 0..11
        assert _lcg_permutation(12, seed) == p  # reproducible


def test_train_val_split_is_disjoint_covers_all_and_deterministic():
    ds = Dataset(
        tuple((float(i), float(i * i)) for i in range(20)), tuple(float(i % 2) for i in range(20))
    )
    tr, va = train_val_split(ds, 0.25, seed=3)
    assert len(va) == 5 and len(tr) == 15, (len(tr), len(va))  # round(0.25*20) = 5
    tr_set = {tuple(r) for r in tr.X}
    va_set = {tuple(r) for r in va.X}
    assert tr_set.isdisjoint(va_set), "train and val must be disjoint"
    assert tr_set | va_set == {tuple(r) for r in ds.X}, "split must cover every example"
    # deterministic: same seed -> same partition.
    tr2, va2 = train_val_split(ds, 0.25, seed=3)
    assert {tuple(r) for r in va2.X} == va_set
    # frac=0 -> all training, no val.
    tr0, va0 = train_val_split(ds, 0.0, seed=3)
    assert len(tr0) == 20 and va0 is None


# === Metrics: match an INDEPENDENT hand computation (incl. a confusion-matrix anchor for F1) ===========


def test_accuracy_binary_and_multiclass_match_hand_computation():
    # BINARY: scalar probabilities thresholded at 0.5. preds-as-class: 1,0,1,0 ; truth: 1,0,0,0 -> 3/4 right.
    assert accuracy([0.9, 0.1, 0.6, 0.3], [1, 0, 0, 0]) == 0.75
    # the 0.5 boundary: exactly 0.5 predicts class 1 (>= threshold).
    assert accuracy([0.5, 0.49], [1, 0]) == 1.0
    # MULTICLASS: argmax of each probability vector. preds argmax: 1,0,2 ; truth: 1,0,2 -> all right.
    assert accuracy([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1], [0.2, 0.3, 0.5]], [1, 0, 2]) == 1.0
    # an argmax tie breaks on the LOWEST index (deterministic).
    assert accuracy([[0.5, 0.5]], [0]) == 1.0 and accuracy([[0.5, 0.5]], [1]) == 0.0


def test_mse_metric_matches_hand_computation():
    # (1/n) sum (p - t)^2 : ((1-1)^2 + (2-0)^2 + (3-3)^2)/3 = 4/3.
    assert abs(mse_metric([1.0, 2.0, 3.0], [1.0, 0.0, 3.0]) - 4.0 / 3.0) < 1e-12


def test_binary_f1_matches_a_hand_computed_confusion_matrix():
    # preds (thresholded 0.5): 1,1,0,1,0 ; truth: 1,0,0,1,1.
    #   tp = 2 (idx 0, 3), fp = 1 (idx 1), fn = 1 (idx 4), tn = 1 (idx 2).
    res = binary_f1([0.9, 0.8, 0.2, 0.7, 0.1], [1, 0, 0, 1, 1])
    assert (res["tp"], res["fp"], res["fn"], res["tn"]) == (2, 1, 1, 1), res
    # precision = tp/(tp+fp) = 2/3 ; recall = tp/(tp+fn) = 2/3 ; f1 = 2/3.
    assert abs(res["precision"] - 2.0 / 3.0) < 1e-12
    assert abs(res["recall"] - 2.0 / 3.0) < 1e-12
    assert abs(res["f1"] - 2.0 / 3.0) < 1e-12
    # degenerate guards: no positive predictions -> precision 0, f1 0 (no divide-by-zero).
    z = binary_f1([0.1, 0.2], [1, 0])
    assert z["precision"] == 0.0 and z["f1"] == 0.0 and z["recall"] == 0.0


# === LOGISTIC REGRESSION end-to-end (the headline) =====================================================


def _train_logreg(seed=3):
    ds = make_linearly_separable(80, seed=1)
    tr, va = train_val_split(ds, 0.25, seed=7)
    res = train(
        linear_model,
        [0.0, 0.0, 0.0],
        tr,
        loss="bce",
        optimizer="adam",
        epochs=80,
        batch_size=16,
        lr=0.1,
        val=va,
        metrics=("accuracy", "f1"),
        seed=seed,
    )
    return tr, va, res


def test_logistic_regression_reaches_high_accuracy_end_to_end():
    tr, va, res = _train_logreg()
    # >= ~95% train accuracy and >= ~90% val accuracy within the bounded epoch count (the headline target).
    tr_acc = res.train_metrics["accuracy"][-1]
    va_acc = res.val_metrics["accuracy"][-1]
    assert tr_acc >= 0.95, ("train accuracy too low", tr_acc)
    assert va_acc >= 0.90, ("val accuracy too low", va_acc)
    # the model actually learned (the F1 is high too, not just accuracy on an imbalanced set).
    assert res.train_metrics["f1"][-1] >= 0.95, res.train_metrics["f1"][-1]
    # the loss came down a long way (training reduced the objective).
    assert res.train_loss[-1] < 0.1 * res.train_loss[0], (res.train_loss[0], res.train_loss[-1])


def test_logistic_regression_loss_is_near_monotone_decreasing():
    _, _, res = _train_logreg()
    losses = res.train_loss
    # NEAR-monotone: no single-epoch rise exceeds 1% of the initial loss (a sane lr keeps the descent clean).
    rises = [b - a for a, b in zip(losses, losses[1:]) if b > a]
    max_rise = max(rises) if rises else 0.0
    assert max_rise < 0.01 * losses[0], (max_rise, losses[0])


def test_logistic_regression_is_deterministic_across_runs():
    # same seed -> identical final loss AND identical final params (byte-for-byte reproducible).
    _, _, r1 = _train_logreg(seed=5)
    _, _, r2 = _train_logreg(seed=5)
    assert r1.train_loss[-1] == r2.train_loss[-1], (r1.train_loss[-1], r2.train_loss[-1])
    assert r1.params == r2.params, "same seed must produce identical final parameters"
    # a different seed shuffles differently -> a (slightly) different trajectory (it is a real shuffle).
    _, _, r3 = _train_logreg(seed=6)
    assert r3.train_loss != r1.train_loss or r3.params != r1.params


# === MLP end-to-end on a NON-linearly-separable set (the hidden layer learns) ==========================


def _xor_split():
    xds = make_xor(30, seed=2)
    return train_val_split(xds, 0.25, seed=5)


def test_mlp_clears_the_linear_ceiling_on_a_nonlinear_set():
    # The headline of the MLP demo: on an XOR-like set a LINEAR model is near chance (~50%), but a 2-layer MLP
    # with a hidden relu learns the nonlinearity and reaches high accuracy -- clearly above the linear ceiling.
    xtr, xva = _xor_split()

    # --- the linear model's ceiling on this set (it cannot separate XOR) ---
    lin = train(
        linear_model,
        [0.0, 0.0, 0.0],
        xtr,
        loss="bce",
        optimizer="adam",
        epochs=400,
        batch_size=16,
        lr=0.05,
        seed=4,
    )
    lin_preds = _predict(linear_model, lin.params, ["w0", "w1", "w2"], xtr.X, "bce")
    lin_acc = accuracy(lin_preds, xtr.y)
    assert lin_acc < 0.8, ("a linear model should NOT solve XOR", lin_acc)  # the ceiling

    # --- the 2-layer MLP (hidden relu via select, output logit) ---
    nh = 12
    names = mlp_param_names(2, nh)
    mdl = mlp_model(2, nh)
    p0 = _mlp_init(names)
    res = train(
        mdl,
        p0,
        xtr,
        loss="bce",
        optimizer="adam",
        epochs=400,
        batch_size=16,
        lr=0.05,
        metrics=("accuracy",),
        seed=4,
        param_names=names,
    )
    mlp_acc = res.train_metrics["accuracy"][-1]
    # the MLP reaches high accuracy ...
    assert mlp_acc >= 0.9, ("MLP should solve XOR", mlp_acc)
    # ... CLEARLY above the linear ceiling (the hidden layer is doing real work).
    assert mlp_acc > lin_acc + 0.1, (mlp_acc, lin_acc)
    # the loss dropped substantially.
    assert res.train_loss[-1] < 0.3 * res.train_loss[0], (res.train_loss[0], res.train_loss[-1])


def test_mlp_training_is_deterministic():
    xtr, _ = _xor_split()
    nh = 12
    names = mlp_param_names(2, nh)
    mdl = mlp_model(2, nh)
    p0 = _mlp_init(names)
    r1 = train(
        mdl,
        p0,
        xtr,
        loss="bce",
        optimizer="adam",
        epochs=120,
        batch_size=16,
        lr=0.05,
        seed=4,
        param_names=names,
    )
    r2 = train(
        mdl,
        list(p0),
        xtr,
        loss="bce",
        optimizer="adam",
        epochs=120,
        batch_size=16,
        lr=0.05,
        seed=4,
        param_names=names,
    )
    assert r1.train_loss == r2.train_loss, "MLP training must be deterministic given the seed"
    assert r1.params == r2.params


# === Softmax-CE multiclass path (the transcendental softmax-CE seed composes) ==========================


def _softmax_linear_model(K, d):
    """A K-class linear head: logits_k = dot(W_k, x), built per example -- the loss="softmax_ce" shape (a
    K-length list of logit nodes per example). Param names W{k}_{j}."""

    def model(tape, names, X):
        Wvars = [[tape.var(f"W{k}_{j}") for j in range(d)] for k in range(K)]
        out = []
        for row in X:
            xc = [tape.const(row[j]) for j in range(d)]
            out.append([tape.dot(tuple(Wvars[k]), tuple(xc)) for k in range(K)])
        return out

    return model


def _three_class_dataset(n=60, seed=7):
    centers = [(2.0, 0.0), (-1.0, 2.0), (-1.0, -2.0)]
    state = seed & 0xFFFFFFFFFFFFFFFF or 1

    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (state >> 33) / float(1 << 31) - 1.0

    X, y = [], []
    for i in range(n):
        c = i % 3
        cx, cy = centers[c]
        X.append((cx + 0.3 * rnd(), cy + 0.3 * rnd()))
        y.append(float(c))
    return Dataset(tuple(X), tuple(y))


def test_softmax_ce_multiclass_reaches_high_accuracy():
    ds = _three_class_dataset()
    names = [f"W{k}_{j}" for k in range(3) for j in range(2)]
    mdl = _softmax_linear_model(3, 2)
    res = train(
        mdl,
        [0.0] * 6,
        ds,
        loss="softmax_ce",
        optimizer="adam",
        epochs=150,
        batch_size=16,
        lr=0.1,
        metrics=("accuracy",),
        seed=2,
        param_names=names,
    )
    assert res.train_metrics["accuracy"][-1] >= 0.95, res.train_metrics["accuracy"][-1]
    assert res.train_loss[-1] < 0.1 * res.train_loss[0], (res.train_loss[0], res.train_loss[-1])


# === MSE closed-set path: a linear regression converges to the known minimizer ========================


def test_mse_closed_set_regression_converges_to_the_known_minimizer():
    # fit y = 2x + 1 (exactly realizable): the MSE-into-Tape loss is differentiated by the EXISTING
    # autodiff.grad (no seeding), and the loop drives it to the minimizer (w, b) = (2, 1).
    def reg_model(tape, names, X):
        w = tape.var("w0")
        b = tape.var("w1")
        return [tape.add(tape.mul(w, tape.const(row[0])), b) for row in X]

    ds = Dataset(
        tuple((float(i),) for i in range(-5, 6)), tuple(2.0 * i + 1.0 for i in range(-5, 6))
    )
    res = train(
        reg_model,
        [0.0, 0.0],
        ds,
        loss="mse",
        optimizer="adam",
        epochs=600,
        batch_size=4,
        lr=0.1,
        metrics=("mse",),
        seed=1,
        param_names=["w0", "w1"],
    )
    assert abs(res.params[0] - 2.0) < 1e-2 and abs(res.params[1] - 1.0) < 1e-2, res.params
    assert res.train_metrics["mse"][-1] < 1e-4, res.train_metrics["mse"][-1]


# === Early stop: a run that plateaus on val stops before the cap (the hook fires) ======================


def test_early_stop_fires_when_val_plateaus():
    ds = make_linearly_separable(60, seed=1)
    tr, va = train_val_split(ds, 0.3, seed=7)
    cap = 200
    es = EarlyStop(patience=3, min_delta=1e-5)
    res = train(
        linear_model,
        [0.0, 0.0, 0.0],
        tr,
        loss="bce",
        optimizer="adam",
        epochs=cap,
        batch_size=16,
        lr=0.2,
        val=va,
        early_stop=es,
        seed=3,
    )
    # the run stopped EARLY (fewer epochs than the cap) once the val loss plateaued -- the hook fired.
    assert res.early_stopped, "early stop should fire on a converging run"
    assert res.epochs_run < cap, (res.epochs_run, cap)
    assert es.stopped_epoch >= 0
    # the returned params are the BEST-so-far snapshot (the early-stop restore).
    assert len(res.params) == 3


def test_early_stop_does_not_fire_when_still_improving():
    # with a generous patience and a still-improving run, training uses the full epoch budget (the hook is a
    # convergence heuristic, not a forced cut).
    ds = make_linearly_separable(60, seed=1)
    tr, va = train_val_split(ds, 0.3, seed=7)
    es = EarlyStop(patience=1000, min_delta=1e-9)
    res = train(
        linear_model,
        [0.0, 0.0, 0.0],
        tr,
        loss="bce",
        optimizer="adam",
        epochs=10,
        batch_size=16,
        lr=0.05,
        val=va,
        early_stop=es,
        seed=3,
    )
    assert not res.early_stopped and res.epochs_run == 10


# === TrainResult carries the history ====================================================================


def test_train_result_carries_per_epoch_history():
    _, _, res = _train_logreg()
    assert isinstance(res, TrainResult)
    assert len(res.train_loss) == res.epochs_run
    assert len(res.val_loss) == res.epochs_run  # a val set was supplied
    assert len(res.train_metrics["accuracy"]) == res.epochs_run
    assert res.final_train_loss == res.train_loss[-1]
    assert res.final_val_loss == res.val_loss[-1]


# === The two-truth invariant: training is cost/optimization-side, never a verdict ======================


def test_training_touches_no_verifier_and_emits_no_diagnostic():
    import bcir.kbcir.training as tr_mod

    # (1) no verifier / diagnostic / graded name is bound in the live module namespace.
    bound = set(vars(tr_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    # (2) no verifier module is imported (parse the AST so a docstring mention can't trip it).
    tree = ast.parse(open(tr_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # (3) the TrainResult is a plain numeric carrier -- no graded/confidence wrapper on the outputs.
    _, _, res = _train_logreg()
    assert isinstance(res.params, list) and all(isinstance(p, float) for p in res.params)
    assert all(isinstance(x, float) for x in res.train_loss)


def test_model_graphs_are_closed_set_and_verify_clean():
    # the model builds ordinary closed-set Tape DAGs (the autodiff-kernel lowerable set) -- never a
    # transcendental node, never a verifier object. The closed primitive set IS what keeps the gradient honest.
    from bcir.kbcir.autodiff import Tape
    from bcir.lower.autodiff_kernel import _LOWERABLE

    t = Tape()
    out = linear_model(t, ["w0", "w1", "b"], [(1.0, 2.0), (-1.0, 0.5)])
    ops = {t.node(nid).op for nid in range(t.node_count())}
    assert ops <= _LOWERABLE, ("model graph escaped the closed lowerable set", ops)
    assert out  # the model produced nodes
    # the MLP graph too (its relu is the select primitive, still in the closed set).
    t2 = Tape()
    names = mlp_param_names(2, 3)
    mlp_model(2, 3)(t2, names, [(1.0, 2.0)])
    ops2 = {t2.node(nid).op for nid in range(t2.node_count())}
    assert ops2 <= _LOWERABLE, ("MLP graph escaped the closed lowerable set", ops2)
    assert "select" in ops2, "the MLP hidden relu should lower to the closed-set select primitive"


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
