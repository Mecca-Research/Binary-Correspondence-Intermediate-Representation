"""M3 (ML Tier-1, roadmap Pillar 5b / Phase B4): the SUPERVISED TRAINING LOOP -- the capstone of the trio.

This is the third and FINAL piece of the ML Tier-1 trio (M1 losses -> M2 optimizers -> M3 the training
loop). M1 (``bcir.kbcir.losses``) built the loss head on the B3 autodiff so a single forward -> loss ->
backward -> parameter-gradient pass runs end-to-end; M2 (``bcir.lower.optimizers``) generalized the minimal
SGD move into the adaptive optimizer family (momentum / RMSprop / Adam, each a side-effect-free
``(params, state) -> (params', state')`` reference rule). What was MISSING is the thing that turns those two
single-step primitives into actual learning: an epoch / mini-batch TRAINING LOOP -- a dataset abstraction, a
deterministic shuffle, a train/val split, eval metrics, and an early-stop hook -- that drives a model to high
accuracy. This module is that loop, and its capstone demonstration (in ``bcir/tests/test_training.py``) is
**logistic regression AND a small MLP trained end-to-end to high accuracy** on toy datasets. That completes
the trio that turns BCIR's matmul + activation + autodiff into supervised learning.

HOW THE LOOP COMPOSES M1 + M2 + AUTODIFF (the crux -- it is the same two-path structure M1 is built around,
forced by the autodiff closure property). A ``model`` is a callable that, given the current ``params`` and one
batch, BUILDS a fresh forward pass: a ``Tape`` plus, per example, the model's scalar prediction node (a
closed-set loss like MSE) OR its raw LOGITS node(s) (a transcendental loss like BCE / softmax-CE). The loop
then takes ONE of two paths per batch -- exactly the M1 split:

  1. CLOSED-SET-LOSS path (MSE). The loss is expressible ENTIRELY in the Tape primitives, so the loop builds
     ``loss_node = mse(tape, preds, targets)`` INTO the same Tape and calls the EXISTING ``autodiff.grad`` on
     it -- the gradient ``dL/dparam`` comes straight out, no seeding. This is the elegant reuse path: the loop
     adds nothing, it just differentiates the model->loss DAG M1 already proved correct.

  2. TRANSCENDENTAL-LOSS path (BCE-with-logits, softmax-CE). The loss VALUE needs ``log``/``exp`` and so is
     NOT a closed-set Tape node; but its gradient w.r.t. the LOGITS is the famous closed form that DOES live
     in the closed set (``sigmoid(z) - y`` ; ``softmax(z) - onehot``). M1 returns that as ``grad_logits``. The
     loop then CHAINS it into the model's backward: for each logit node ``k`` it runs ``autodiff.grad`` on
     ``logit_k`` to get ``d(logit_k)/dparam`` and accumulates ``dL/dparam += grad_logits[k] * d(logit_k)/dparam``
     (the seed is the closed-form derivative; the transcendental lives ONLY in the monitored loss value, never
     in the gradient that flows into the parameters). This is exactly the composition M1's
     ``test_logistic_regression_composition_grad_via_seed_matches_finite_difference`` already proved for a
     single example -- M3 just sums it over a mini-batch.

Once the loop has the per-parameter gradient (either path), it drives the weight update with the M2 optimizer
rule selected by name (``"sgd"`` / ``"momentum"`` / ``"rmsprop"`` / ``"adam"`` + hypers), managing that
optimizer's per-parameter state (velocity / squared-grad EMA / Adam's m,v,t) across steps. So the loop is the
generalization of ``bcir.lower.autodiff_kernel.oracle_train`` (the existing single-DAG forward->backward->SGD
reference) to epochs, mini-batches, an arbitrary M1 loss, an arbitrary M2 optimizer, a held-out validation
split, eval metrics, and early stopping -- the deployable training API, in the smallest honest form.

THE TWO-TRUTH NOTE (LangRef §13, the cost/optimization-side quarantine -- the SAME boundary M1/M2 sit on).
Training is a COST / OPTIMIZATION-side activity: it decides WHICH legal plan's weights fit the data best -- it
is NEVER an R-law legality verdict. This module touches no verifier and emits no ``Diagnostic`` (a test pins
this). The model graphs it builds are ordinary ``Tape`` DAGs in the closed primitive set; the gradient that
reaches the parameters never carries a transcendental. It sits exactly where the losses and optimizers sit:
off the legality path, on the graded/cost side.

DETERMINISM. Given a ``seed``, the loop is fully reproducible: the mini-batch shuffle is a deterministic
permutation keyed by ``(seed, epoch)`` (a tiny stdlib LCG -- no ``random``, no numpy, dependency-free), the
train/val split is a deterministic partition, and the parameter init is the caller's (``params0``). Two runs
with the same seed produce byte-identical loss curves and final parameters (a test pins this). No numpy --
plain lists / tuples / ``math`` only, to stay dependency-free and match the repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .autodiff import Tape, evaluate, grad
from .losses import (binary_cross_entropy_with_logits, mse, mse_value,
                     softmax_cross_entropy)
from ..lower.optimizers import adam_step, momentum_step, rmsprop_step, sgd_step


# === a deterministic, dependency-free shuffle (no random / numpy -- a tiny reproducible LCG) ============

def _lcg_permutation(n: int, seed: int) -> list[int]:
    """A DETERMINISTIC permutation of ``range(n)`` keyed by ``seed`` -- a Fisher-Yates shuffle whose random
    stream is a small stdlib linear-congruential generator (the Numerical Recipes / glibc constants). No
    ``random`` module, no numpy: the same ``(n, seed)`` always yields the SAME order (a test pins this), which
    is what makes an epoch's mini-batch order reproducible. The LCG is for SHUFFLE order only -- never for any
    legality-relevant choice (it is cost/optimization-side)."""
    idx = list(range(n))
    state = (int(seed) & 0xFFFFFFFFFFFFFFFF) or 0x9E3779B97F4A7C15   # avoid a degenerate all-zero state
    for i in range(n - 1, 0, -1):
        # glibc-style 64-bit LCG step; take the high bits (better-distributed than the low ones).
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        j = (state >> 33) % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    return idx


# === the dataset / batching abstraction (plain lists/tuples, deterministic) =============================

@dataclass(frozen=True)
class Dataset:
    """A small in-memory supervised dataset: a list of feature rows ``X`` (each a tuple/list of floats) and a
    parallel list of labels ``y`` (a float per row -- a 0/1 binary label, a class index, or a regression
    target). Plain Python containers, no numpy. Frozen so it is a stable, hashable-by-identity carrier the
    loop can split / batch without mutating."""

    X: tuple
    y: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", tuple(tuple(float(v) for v in row) for row in self.X))
        object.__setattr__(self, "y", tuple(float(t) for t in self.y))
        if len(self.X) != len(self.y):
            raise ValueError(f"Dataset: X/y length mismatch ({len(self.X)} != {len(self.y)})")
        if not self.X:
            raise ValueError("Dataset: needs at least one example")
        n_feat = len(self.X[0])
        for i, row in enumerate(self.X):
            if len(row) != n_feat:
                raise ValueError(f"Dataset: row {i} has {len(row)} features, expected {n_feat}")

    def __len__(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return len(self.X[0])


def minibatches(dataset: Dataset, batch_size: int, *, seed: int = 0):
    """Yield ``(X_batch, y_batch)`` tuples covering every example EXACTLY ONCE in a DETERMINISTIC shuffled
    order (one epoch). The order is the ``_lcg_permutation`` keyed by ``seed`` -- so the same seed yields the
    same batch order (a test pins this), and every sample appears once per epoch. The last batch is RAGGED
    (smaller) when ``len(dataset)`` is not a multiple of ``batch_size``. ``batch_size`` >= len yields a single
    full-dataset batch (full-batch gradient descent)."""
    if batch_size < 1:
        raise ValueError(f"minibatches: batch_size must be >= 1; got {batch_size}")
    n = len(dataset)
    order = _lcg_permutation(n, seed)
    for start in range(0, n, batch_size):
        chunk = order[start:start + batch_size]
        xb = tuple(dataset.X[i] for i in chunk)
        yb = tuple(dataset.y[i] for i in chunk)
        yield xb, yb


def train_val_split(dataset: Dataset, frac: float, *, seed: int = 0) -> tuple[Dataset, Dataset]:
    """Partition ``dataset`` into ``(train, val)`` where ``val`` holds ``round(frac * n)`` examples, drawn by a
    DETERMINISTIC shuffle keyed by ``seed``. The two halves are DISJOINT and together COVER every example (a
    test pins both), and the split is reproducible (same seed -> same partition). ``frac`` is the validation
    fraction in ``[0, 1)``; ``frac == 0`` returns ``(dataset, empty-but-valid-shaped)`` -- handled by the loop
    treating an empty val as "no validation"."""
    if not (0.0 <= frac < 1.0):
        raise ValueError(f"train_val_split: frac must be in [0, 1); got {frac}")
    n = len(dataset)
    n_val = round(frac * n)
    if n_val >= n:
        raise ValueError(f"train_val_split: frac={frac} leaves no training data (n_val={n_val} of {n})")
    order = _lcg_permutation(n, seed)
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    train = Dataset(tuple(dataset.X[i] for i in train_idx), tuple(dataset.y[i] for i in train_idx))
    if n_val == 0:
        return train, None
    val = Dataset(tuple(dataset.X[i] for i in val_idx), tuple(dataset.y[i] for i in val_idx))
    return train, val


# === eval metrics (reference impls, deterministic, stdlib only) =========================================

def accuracy(preds, labels) -> float:
    """Classification accuracy -- the fraction of correct predictions. Two modes, inferred from the shape of
    ``preds``:

      * MULTICLASS: each ``preds[i]`` is a probability/score VECTOR (a list) -> the prediction is its argmax
        (ties broken on the lowest index, deterministic); ``labels[i]`` is the integer true class.
      * BINARY: each ``preds[i]`` is a scalar probability in [0, 1] -> the prediction is ``1`` iff ``p >= 0.5``
        (the standard 0.5 threshold); ``labels[i]`` is the 0/1 label.

    Returns the fraction in [0, 1]. Deterministic; hand-verifiable on a fixed (preds, labels)."""
    preds = list(preds)
    labels = list(labels)
    if len(preds) != len(labels) or not preds:
        raise ValueError(f"accuracy: bad lengths preds={len(preds)} labels={len(labels)}")
    correct = 0
    for p, y in zip(preds, labels):
        if isinstance(p, (list, tuple)):
            pred_cls = max(range(len(p)), key=lambda k: p[k])     # argmax, ties -> lowest index
        else:
            pred_cls = 1 if p >= 0.5 else 0
        if pred_cls == int(round(y)):
            correct += 1
    return correct / len(preds)


def mse_metric(preds, targets) -> float:
    """Mean-squared-error metric over scalar predictions/targets -- the regression eval. Reuses the M1
    ``losses.mse_value`` reference so the metric and the training loss are the SAME number (no second MSE)."""
    return mse_value(list(preds), list(targets))


def binary_f1(preds, labels, *, threshold: float = 0.5) -> dict:
    """Precision / recall / F1 for binary classification, plus the full confusion-matrix counts. Each
    ``preds[i]`` is a scalar probability (thresholded at ``threshold``, default 0.5) or already a 0/1 label;
    ``labels[i]`` is the 0/1 truth. Returns ``{"precision","recall","f1","tp","fp","fn","tn"}``.

    Definitions (the textbook ones, hand-verifiable against a confusion matrix):
      precision = tp / (tp + fp)   (0 if no positive predictions),
      recall    = tp / (tp + fn)   (0 if no actual positives),
      f1        = 2*precision*recall / (precision + recall)   (0 if both are 0).
    Deterministic; stdlib only."""
    preds = list(preds)
    labels = list(labels)
    if len(preds) != len(labels) or not preds:
        raise ValueError(f"binary_f1: bad lengths preds={len(preds)} labels={len(labels)}")
    tp = fp = fn = tn = 0
    for p, y in zip(preds, labels):
        pred = 1 if (float(p) >= threshold) else 0
        truth = int(round(y))
        if pred == 1 and truth == 1:
            tp += 1
        elif pred == 1 and truth == 0:
            fp += 1
        elif pred == 0 and truth == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# === the loss spec (which M1 loss + which of the two composition paths) =================================

# A loss is named; the loop dispatches on the name to pick the closed-set vs transcendental-seed path. The
# model callable's contract (its return shape) is tied to the loss kind:
#   "mse"        -> CLOSED-SET path. model returns a list of scalar PREDICTION node ids (one per example).
#   "bce"        -> TRANSCENDENTAL path. model returns a list of scalar LOGIT node ids (one per example).
#   "softmax_ce" -> TRANSCENDENTAL path. model returns a list of K-length LOGIT-node lists (one row/example).
_CLOSED_SET_LOSSES = ("mse",)
_TRANSCENDENTAL_LOSSES = ("bce", "softmax_ce")
_LOSSES = _CLOSED_SET_LOSSES + _TRANSCENDENTAL_LOSSES


# === the optimizer state manager (drives the M2 reference rules across steps) ===========================

class _OptState:
    """Holds + advances one M2 optimizer's per-parameter state across training steps, dispatching by name to
    the M2 reference rule. ``step(params, grads) -> params'`` applies ONE M2 update and rolls the state
    forward (velocity / squared-grad EMA / Adam's m,v,t) -- the loop calls it once per mini-batch. This is the
    repeated-step LOOP M2's docstring said M3 would wire; M2 delivers the single deployable step, M3 manages
    its state over many."""

    def __init__(self, name: str, n: int, lr: float, **hp) -> None:
        if name not in ("sgd", "momentum", "rmsprop", "adam"):
            raise ValueError(f"unknown optimizer {name!r}; choose sgd/momentum/rmsprop/adam")
        self.name = name
        self.lr = float(lr)
        self.hp = hp
        self.velocity = [0.0] * n
        self.sq_avg = [0.0] * n
        self.m = [0.0] * n
        self.v = [0.0] * n
        self.t = 0

    def step(self, params, grads):
        if self.name == "sgd":
            return sgd_step(params, grads, self.lr)
        if self.name == "momentum":
            new_p, self.velocity = momentum_step(params, grads, self.velocity, self.lr,
                                                 beta=self.hp.get("beta", 0.9))
            return new_p
        if self.name == "rmsprop":
            new_p, self.sq_avg = rmsprop_step(params, grads, self.sq_avg, self.lr,
                                             beta=self.hp.get("beta", 0.9), eps=self.hp.get("eps", 1e-8))
            return new_p
        # adam
        new_p, self.m, self.v, self.t = adam_step(params, grads, self.m, self.v, self.t, self.lr,
                                                  beta1=self.hp.get("beta1", 0.9),
                                                  beta2=self.hp.get("beta2", 0.999),
                                                  eps=self.hp.get("eps", 1e-8))
        return new_p


# === the per-batch forward -> loss -> gradient (the M1 two-path composition) ============================

def _batch_loss_and_grad(model, params, param_names, xb, yb, loss):
    """ONE mini-batch: build the model's forward on this batch, compute the M1 loss, and the per-parameter
    gradient -- via the closed-set path (MSE) or the transcendental-seed path (BCE / softmax-CE). Returns
    ``(loss_value, grad_dict)`` where ``grad_dict`` maps each param name to the MEAN gradient over the batch
    (averaged so the step size is batch-size-independent, the standard convention). This is the function that
    composes M1 + autodiff; the loop calls it, then hands ``grad_dict`` to the M2 optimizer."""
    env = dict(zip(param_names, params))
    tape = Tape()
    out = model(tape, param_names, xb)         # the model builds the forward into `tape`; returns node(s)
    nb = len(xb)

    if loss == "mse":
        # CLOSED-SET path: `out` is a list of scalar prediction node ids, one per example. Build the MSE loss
        # node INTO the same tape over all batch predictions vs their targets, then the EXISTING autodiff.grad
        # differentiates the whole model->loss DAG -- no seeding. (mse already scales by 1/nb -> mean loss.)
        pred_nids = tuple(out)
        loss_node = mse(tape, pred_nids, list(yb))
        loss_val = evaluate(tape, loss_node, env)
        g = grad(tape, loss_node, env).grads
        return loss_val, {name: g.get(name, 0.0) for name in param_names}

    # TRANSCENDENTAL-SEED path: the loss value is monitored via M1; the gradient is seeded by grad_logits and
    # chained through each logit's own backward. We accumulate dL/dparam = sum_logit grad_logits[k] *
    # d(logit_k)/dparam, summed over the batch, then divide by nb for the mean.
    grad_acc = {name: 0.0 for name in param_names}
    total_loss = 0.0
    if loss == "bce":
        # `out` is a list of scalar logit node ids, one per example (binary).
        for logit_node, y in zip(out, yb):
            z = evaluate(tape, logit_node, env)
            lval, grad_logits = binary_cross_entropy_with_logits([z], [y])  # mean over 1 -> the per-example loss
            seed = grad_logits[0]                                            # dL/dz = sigmoid(z) - y
            dlogit = grad(tape, logit_node, env).grads                       # d(logit)/dparam, seed 1.0
            for name in param_names:
                grad_acc[name] += seed * dlogit.get(name, 0.0)
            total_loss += lval
        loss_val = total_loss / nb
        return loss_val, {name: grad_acc[name] / nb for name in param_names}

    # softmax_ce: `out` is a list of K-length logit-node lists, one row per example; yb holds the class index.
    for logit_row, y in zip(out, yb):
        zvec = [evaluate(tape, ln, env) for ln in logit_row]
        K = len(zvec)
        onehot = [1.0 if k == int(round(y)) else 0.0 for k in range(K)]
        lval, grad_logits = softmax_cross_entropy(zvec, onehot)             # grad = softmax(z) - onehot
        for k, ln in enumerate(logit_row):
            seed = grad_logits[k]
            dlogit = grad(tape, ln, env).grads                              # d(logit_k)/dparam
            for name in param_names:
                grad_acc[name] += seed * dlogit.get(name, 0.0)
        total_loss += lval
    loss_val = total_loss / nb
    return loss_val, {name: grad_acc[name] / nb for name in param_names}


# === predictions (for metrics) -- run the model forward and read off the probability/class ==============

def _predict(model, params, param_names, X, loss):
    """Run the model forward over ``X`` and return a list of PREDICTIONS in the shape the metrics expect:
      * "mse"        -> a scalar prediction per example (regression),
      * "bce"        -> a scalar SIGMOID PROBABILITY per example (binary -> accuracy/F1 threshold at 0.5),
      * "softmax_ce" -> a K-length PROBABILITY VECTOR per example (multiclass -> accuracy via argmax).
    Built fresh on a tape (no gradient needed for eval -- just the forward), deterministic."""
    env = dict(zip(param_names, params))
    tape = Tape()
    out = model(tape, param_names, X)
    if loss == "mse":
        return [evaluate(tape, nid, env) for nid in out]
    if loss == "bce":
        return [1.0 / (1.0 + math.exp(-evaluate(tape, nid, env))) for nid in out]
    # softmax_ce: softmax the logit row (stable max-subtraction) -> a probability vector per example.
    preds = []
    for logit_row in out:
        z = [evaluate(tape, ln, env) for ln in logit_row]
        m = max(z)
        ex = [math.exp(v - m) for v in z]
        s = sum(ex) or 1.0
        preds.append([e / s for e in ex])
    return preds


# === the early-stop hook (patience on the validation loss) ==============================================

@dataclass
class EarlyStop:
    """A patience-based early-stop hook on the VALIDATION loss. After each epoch the loop calls ``update``
    with the current val loss; training stops when the val loss has not improved (by more than ``min_delta``)
    for ``patience`` consecutive epochs. ``best_loss`` / ``best_params`` snapshot the best-so-far model. A
    pure cost-side convergence heuristic (never a verdict)."""

    patience: int = 5
    min_delta: float = 1e-4
    best_loss: float = math.inf
    bad_epochs: int = 0
    best_params: list = field(default_factory=list)
    stopped_epoch: int = -1

    def update(self, val_loss: float, params, epoch: int) -> bool:
        """Record this epoch's val loss; return True iff training should STOP now. Snapshots ``best_params``
        whenever the loss improves."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_params = list(params)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False


# === the result carrier =================================================================================

@dataclass
class TrainResult:
    """The outcome of :func:`train`: the per-epoch history (train loss, optional val loss, and each requested
    metric on train + val) plus the final parameters and the epoch count actually run (< the cap if early
    stop fired). Everything plain numeric -- no graded/verdict wrapper (training is cost-side)."""

    params: list                         # the final (or best, if early-stopped) parameter vector
    train_loss: list = field(default_factory=list)     # per-epoch mean training loss
    val_loss: list = field(default_factory=list)       # per-epoch validation loss ([] if no val set)
    train_metrics: dict = field(default_factory=dict)  # metric name -> per-epoch list on the train set
    val_metrics: dict = field(default_factory=dict)    # metric name -> per-epoch list on the val set
    epochs_run: int = 0                  # epochs actually executed (< the cap if early stop fired)
    early_stopped: bool = False

    @property
    def final_train_loss(self) -> float:
        return self.train_loss[-1] if self.train_loss else math.inf

    @property
    def final_val_loss(self) -> float:
        return self.val_loss[-1] if self.val_loss else math.inf


# === the metric drivers (apply a requested metric name to a set of predictions) =========================

def _apply_metric(name: str, preds, labels, loss) -> float:
    """Compute one requested metric by name on a set of predictions (in the :func:`_predict` shape). Supported
    names: ``"accuracy"`` (binary threshold / multiclass argmax), ``"mse"`` (regression), ``"f1"`` (binary
    F1). Returns a single float."""
    if name == "accuracy":
        return accuracy(preds, labels)
    if name == "mse":
        return mse_metric(preds, labels)
    if name == "f1":
        return binary_f1(preds, labels)["f1"]
    raise ValueError(f"unknown metric {name!r}; choose accuracy/mse/f1")


# === the training loop ==================================================================================

def train(model, params0, dataset: Dataset, *, loss: str, optimizer: str = "adam", epochs: int = 100,
          batch_size: int = 32, lr: float = 0.01, val: Dataset | None = None, metrics=(),
          early_stop: EarlyStop | None = None, seed: int = 0, param_names=None, **opt_hp) -> TrainResult:
    """Train ``model`` end-to-end with mini-batch gradient descent -- the M3 capstone API.

    Composes M1 + M2 + the B3 autodiff into an epoch / mini-batch loop:
      forward (the ``model`` builds a ``Tape``) -> M1 ``loss`` -> ``autodiff.grad`` (with the M1 closed-form
      seed for a transcendental loss) -> M2 ``optimizer`` step -> accumulate per-epoch train/val loss + metrics
      -> optional early stop. It is the oracle generalization of ``autodiff_kernel.oracle_train``.

    Arguments:
      * ``model(tape, param_names, X) -> nodes`` -- a callable that, GIVEN the param-name list and a batch of
        feature rows ``X``, builds the forward pass into ``tape`` and returns the per-example output node(s):
        a scalar prediction node per example for ``loss="mse"``; a scalar LOGIT node per example for
        ``loss="bce"``; a K-length list of logit nodes per example for ``loss="softmax_ce"``. The model reads
        its parameters by the names in ``param_names`` (it calls ``tape.var(name)``); the loop supplies the
        current values via the env.
      * ``params0`` -- the initial parameter VALUES (a list of floats; the init is the caller's, so determinism
        is the caller's seed + this list).
      * ``loss`` -- ``"mse"`` (closed-set) / ``"bce"`` / ``"softmax_ce"`` (transcendental seed).
      * ``optimizer`` -- ``"sgd"`` / ``"momentum"`` / ``"rmsprop"`` / ``"adam"`` (the M2 rule); ``**opt_hp``
        passes its hypers (beta / beta1 / beta2 / eps).
      * ``epochs`` / ``batch_size`` / ``lr`` -- the loop schedule.
      * ``val`` -- an optional held-out :class:`Dataset` for per-epoch validation loss/metrics + early stop.
      * ``metrics`` -- an iterable of metric names (``"accuracy"`` / ``"mse"`` / ``"f1"``) tracked per epoch.
      * ``early_stop`` -- an optional :class:`EarlyStop` patience hook on the val loss.
      * ``seed`` -- the deterministic shuffle seed (per-epoch order is keyed by ``(seed, epoch)``).
      * ``param_names`` -- the parameter NAMES the model uses (defaults to ``"w0".."w{n-1}"``); the ABI between
        ``params0`` (values) and the model (which reads ``tape.var(name)``).

    Returns a :class:`TrainResult` with the per-epoch history + final params. DETERMINISTIC given the seed:
    two runs with the same seed + ``params0`` produce identical loss curves and final parameters."""
    if loss not in _LOSSES:
        raise ValueError(f"unknown loss {loss!r}; choose from {_LOSSES}")
    params = [float(p) for p in params0]
    n = len(params)
    if param_names is None:
        param_names = [f"w{i}" for i in range(n)]
    param_names = list(param_names)
    if len(param_names) != n:
        raise ValueError(f"train: param_names/params0 length mismatch ({len(param_names)} != {n})")

    opt = _OptState(optimizer, n, lr, **opt_hp)
    result = TrainResult(params=params)
    metric_names = tuple(metrics)
    for m in metric_names:
        result.train_metrics[m] = []
        result.val_metrics[m] = []

    for epoch in range(int(epochs)):
        # one epoch: every example once, in the deterministic shuffled order keyed by (seed, epoch) so each
        # epoch reshuffles reproducibly. Accumulate the example-weighted mean training loss over the batches.
        epoch_loss_sum = 0.0
        epoch_count = 0
        for xb, yb in minibatches(dataset, batch_size, seed=seed * 1_000_003 + epoch):
            batch_loss, gdict = _batch_loss_and_grad(model, params, param_names, xb, yb, loss)
            grads = [gdict[name] for name in param_names]
            params = opt.step(params, grads)
            epoch_loss_sum += batch_loss * len(xb)     # weight by batch size for the true epoch mean
            epoch_count += len(xb)
        result.params = params
        result.train_loss.append(epoch_loss_sum / max(1, epoch_count))

        # per-epoch metrics on the train set (and val, if present).
        if metric_names:
            tr_preds = _predict(model, params, param_names, dataset.X, loss)
            for m in metric_names:
                result.train_metrics[m].append(_apply_metric(m, tr_preds, dataset.y, loss))

        # validation loss + metrics (a fresh forward over the whole val set, full-batch).
        v_loss = None
        if val is not None and len(val) > 0:
            v_loss, _ = _batch_loss_and_grad(model, params, param_names, val.X, val.y, loss)
            result.val_loss.append(v_loss)
            if metric_names:
                v_preds = _predict(model, params, param_names, val.X, loss)
                for m in metric_names:
                    result.val_metrics[m].append(_apply_metric(m, v_preds, val.y, loss))

        result.epochs_run = epoch + 1

        # early stop on the val loss (patience). On stop, restore the best-so-far snapshot.
        if early_stop is not None and v_loss is not None:
            if early_stop.update(v_loss, params, epoch):
                result.early_stopped = True
                if early_stop.best_params:
                    result.params = list(early_stop.best_params)
                break

    return result


# === reference toy datasets + reference models (for the demos / tests) ==================================

def make_linearly_separable(n: int = 80, *, seed: int = 0) -> Dataset:
    """A small linearly-separable 2-D binary dataset: two Gaussian-ish blobs (one per class) generated by a
    DETERMINISTIC LCG (no numpy), placed so a line cleanly separates them. Class 0 around (-2, -2), class 1
    around (+2, +2); the spread is small enough to stay separable. The headline logistic-regression demo
    trains on this and must reach ~100% accuracy."""
    state = (int(seed) & 0xFFFFFFFFFFFFFFFF) or 0x1234567
    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (state >> 33) / float(1 << 31) - 1.0          # ~ uniform in [-1, 1)
    X, y = [], []
    for i in range(n):
        cls = i % 2
        cx, cy = (2.0, 2.0) if cls == 1 else (-2.0, -2.0)
        X.append((cx + 0.6 * rnd(), cy + 0.6 * rnd()))
        y.append(float(cls))
    return Dataset(tuple(X), tuple(y))


def make_xor(n_per_quadrant: int = 25, *, seed: int = 0) -> Dataset:
    """An XOR-like NON-linearly-separable 2-D binary dataset: four clusters at the corners (++, --) = class 1,
    (+-, -+) = class 0 -- the canonical problem a LINEAR model cannot solve (its accuracy ceiling is ~50-75%),
    but a 2-layer MLP with a hidden relu can. Deterministic LCG. The MLP demo trains on this and must clear
    the linear ceiling, proving the hidden layer learns the nonlinearity."""
    state = (int(seed) & 0xFFFFFFFFFFFFFFFF) or 0x7654321
    def rnd():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return (state >> 33) / float(1 << 31) - 1.0
    X, y = [], []
    centers = [((1.5, 1.5), 1.0), ((-1.5, -1.5), 1.0), ((1.5, -1.5), 0.0), ((-1.5, 1.5), 0.0)]
    for (cx, cy), cls in centers:
        for _ in range(n_per_quadrant):
            X.append((cx + 0.5 * rnd(), cy + 0.5 * rnd()))
            y.append(cls)
    return Dataset(tuple(X), tuple(y))


def linear_model(tape: Tape, param_names, X):
    """A logistic-regression / linear model: per example, the LOGIT ``z = w0*x0 + w1*x1 + ... + b`` built in
    the Tape (a ``dot`` + a bias add -- the closed-set primitives). Parameters: ``w0..w{d-1}`` then ``b`` (so
    ``param_names`` is length ``d+1``). Returns one scalar logit node per example -- the ``loss="bce"`` shape.
    The reference model for the logistic-regression headline demo."""
    d = len(param_names) - 1                                  # last name is the bias
    wvars = tuple(tape.var(param_names[i]) for i in range(d))
    bvar = tape.var(param_names[d])
    out = []
    for row in X:
        xconst = tuple(tape.const(row[i]) for i in range(d))
        out.append(tape.add(tape.dot(wvars, xconst), bvar))
    return out


def mlp_param_names(n_in: int, n_hidden: int) -> list:
    """The parameter-name ABI for :func:`mlp_model`: a hidden layer (``n_hidden`` x ``n_in`` weights + hidden
    biases) then a single output unit (``n_hidden`` weights + 1 bias). Names are flat and deterministic so the
    caller can build a matching ``params0`` init vector."""
    names = []
    for h in range(n_hidden):
        for i in range(n_in):
            names.append(f"W1_{h}_{i}")
        names.append(f"b1_{h}")
    for h in range(n_hidden):
        names.append(f"W2_{h}")
    names.append("b2")
    return names


def mlp_model(n_in: int, n_hidden: int):
    """Build a 2-layer MLP forward function: a hidden layer with a RELU (the differentiable ``select(h, h, 0)``
    -- relu is ``max(0, h)``, exactly the autodiff ``select`` shape, so the gradient flows through the closed
    set) and a single linear output unit producing the binary LOGIT. Returns a ``model(tape, names, X)``
    callable in the ``loss="bce"`` shape (one scalar logit node per example).

    Layer shapes mirror ``lower.inference.Layer`` (a dense ``out x in`` matmul + bias, then activation): the
    hidden pre-activation ``a_h = dot(W1_h, x) + b1_h``, relu'd to ``r_h = select(a_h, a_h, 0)``, then the
    output ``z = dot(W2, r) + b2``. The hidden relu is what lets the MLP clear a linear model's ceiling on a
    non-linearly-separable set (the MLP demo proves this)."""
    def model(tape: Tape, param_names, X):
        # bind every parameter var once (hash-consed, so reused across examples).
        W1 = [[tape.var(f"W1_{h}_{i}") for i in range(n_in)] for h in range(n_hidden)]
        b1 = [tape.var(f"b1_{h}") for h in range(n_hidden)]
        W2 = [tape.var(f"W2_{h}") for h in range(n_hidden)]
        b2 = tape.var("b2")
        zero = tape.const(0.0)
        out = []
        for row in X:
            xconst = [tape.const(row[i]) for i in range(n_in)]
            hidden = []
            for h in range(n_hidden):
                pre = tape.add(tape.dot(tuple(W1[h]), tuple(xconst)), b1[h])
                # relu = max(0, pre) via the differentiable select (cond=pre>0 ? pre : 0) -- closed set.
                hidden.append(tape.select(pre, pre, zero))
            z = tape.add(tape.dot(tuple(W2), tuple(hidden)), b2)
            out.append(z)
        return out
    return model
