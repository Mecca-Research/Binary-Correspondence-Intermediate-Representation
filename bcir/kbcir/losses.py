"""M1 (ML Tier-1, roadmap Pillar 5b / Phase B4): a LOSS-FUNCTION LIBRARY built on the B3 autodiff.

This is the reference oracle that turns BCIR's existing matmul + activation + reverse-mode autodiff +
SGD into trainable supervised models: it adds the loss head so a full training step -- forward -> loss ->
backward -> parameter gradients -- runs end-to-end. M1 unlocks logistic regression (binary cross-entropy)
and multiclass classification (softmax cross-entropy) on top of the existing ``Tape`` autodiff; MSE
unlocks the regression head. Pure-Python oracle, OFF the legality path: a loss is a cost / optimization-side
quantity (which legal plan trains best), NEVER an R-law verdict -- the two-truth quarantine (LangRef §13,
``bcir/kbcir/twotruth.py``) holds. This module touches no verifier and emits no Diagnostic.

THE TWO-PATH DESIGN (the crux -- it is forced by the autodiff closure property).
``bcir.kbcir.autodiff``'s ``Tape`` has a CLOSED differentiable primitive set
``{const, var, neg, add, sub, mul, div, dot, select}``. There is **no exp / log** in it: transcendentals
ride the trusted ``c.call.libm:`` FFI edge (the activation.py / B5 pattern) and are NOT re-differentiable
by this machinery. The autodiff module's own honest-limitation note (its module docstring, boundary (a))
states it directly: *"CLOSURE is required. A primitive whose VJP is NOT in the primitive set (e.g. one
needing exp/log/a transcendental) ... breaks the closure: the gradient would not be a DAG in this set and
could not be re-differentiated by this machinery."* So losses split into two paths, and the split is not a
convenience -- it is exactly that closure boundary:

  1. CLOSED-SET LOSS (MSE) -- expressible ENTIRELY in the Tape primitives. ``MSE = (1/n)·dot(e, e)`` with
     ``e_i = pred_i - target_i`` is just sub + dot + a scale-by-const, all in the closed set. We build the
     loss INTO the Tape as nodes whose root is the scalar loss; then the EXISTING ``autodiff.grad`` (and the
     EXISTING ``bcir.lower.autodiff_kernel.emit_autodiff_kernel_c``) differentiates / lowers the whole
     model -> loss DAG for free -- it is just more closed-set nodes. This is the elegant path: reuse, don't
     reinvent. The reference ``mse_value`` / ``mse_grad`` are provided too (an independent recompute + the
     closed-form ``dL/dpred_i = (2/n)(pred_i - target_i)`` for verification and for the seed path).

  2. TRANSCENDENTAL LOSS (softmax cross-entropy, BCE-with-logits) -- the forward VALUE needs ``log`` / ``exp``
     (computed here with ``math.log`` / ``math.exp``, the oracle's stand-in for the libm edge, exactly as
     ``activation.py`` does for sigmoid / softmax). It therefore CANNOT be a closed-set Tape node with a
     re-differentiable backward. BUT each of these losses has a famous CLOSED-FORM gradient w.r.t. its logits
     that DOES live in the closed set:

         d(softmax_CE)/d(logits) = softmax(logits) - onehot(target)
         d(BCE_with_logits)/d(logits) = sigmoid(logits) - target

     So we return ``(loss_value, grad_logits)`` where ``grad_logits`` is that closed-form vector. It SEEDS
     the model's existing backward pass: the autodiff backward seed for a scalar output is normally 1.0; here
     the seed for the LOGITS layer is ``grad_logits`` (one seed per logit). The transcendental lives ONLY in
     the monitored loss value; the GRADIENT that flows into the parameters stays entirely in the closed set,
     so the model autodiff differentiates it end-to-end with no exp/log node ever entering the Tape. This is
     the standard trick (the softmax-CE / sigmoid-CE fused gradients every framework uses) and it is what
     keeps the training gradient honest under BCIR's closure rule.

REUSE. The closed-form gradients reuse the EXISTING G1 activation references
(``bcir.kbcir.activation.softmax_reference`` and ``sigmoid_reference``) -- there is no second softmax/sigmoid
here. The numerically-stable forward forms mirror the activation module's stability discipline:
softmax-CE subtracts ``max(logits)`` before ``exp`` (log-sum-exp), and BCE-with-logits uses the
``max(z,0) - z·y + log(1 + exp(-|z|))`` form that never overflows for large ``|z|``.

Deterministic, dependency-free. A research/oracle organ (kept cold). It MAY be consumed by the test harness'
``autodiff_kernel.compile_and_run_grad_c`` (which compiles emitted C in a tempdir -- a harness, not a repo
C-file change). Quantized-input bridge (the ``activation.activation_via_bridge`` pattern) is OPTIONAL for M1
and NOT added: losses operate on f32 predictions/logits.
"""

from __future__ import annotations

import math

from .activation import sigmoid_reference, softmax_reference


# === Path 1: the closed-set loss (MSE) -- built INTO the Tape, differentiated by the existing autodiff ===


def mse(tape, pred_nids, targets) -> int:
    """Build mean-squared-error as CLOSED-SET Tape nodes over the model's prediction node ids and return
    the scalar loss node id.

    ``MSE = (1/n) * sum_i (pred_i - target_i)^2 = (1/n) * dot(e, e)`` where ``e_i = pred_i - target_i``.
    Every step is a Tape primitive: ``sub`` to form each error ``e_i``, ``dot(e, e)`` for the sum of squares
    (the matmul-substrate primitive), then ``mul`` by the constant ``1/n``. Because the result is rooted in
    the SAME closed primitive set the model lives in, the EXISTING ``autodiff.grad(tape, loss, env)``
    differentiates the whole model -> loss DAG end-to-end, and ``emit_autodiff_kernel_c`` lowers it to C for
    free -- no new backward rule, no new lowering. This is the elegant closed-set path.

    ``pred_nids`` is a tuple of Tape node ids (the model's scalar outputs); ``targets`` is a same-length
    iterable of constant floats (the labels). Returns the loss node id."""
    pred_nids = tuple(pred_nids)
    targets = [float(t) for t in targets]
    n = len(pred_nids)
    if n == 0:
        raise ValueError("mse: need at least one prediction node")
    if len(targets) != n:
        raise ValueError(f"mse: pred/target length mismatch ({n} != {len(targets)})")
    # e_i = pred_i - target_i (a const target node per label).
    errs = tuple(tape.sub(p, tape.const(t)) for p, t in zip(pred_nids, targets))
    sq = tape.dot(errs, errs)  # sum_i e_i^2 -- the closed-set sum of products
    return tape.mul(tape.const(1.0 / n), sq)  # scale by 1/n -> the mean


def mse_value(pred, target) -> float:
    """Reference MSE value: ``(1/n) * sum_i (pred_i - target_i)^2``. An independent stdlib recompute used to
    cross-check the Tape-built loss and to drive the seed-path verification."""
    pred = [float(p) for p in pred]
    target = [float(t) for t in target]
    if len(pred) != len(target) or not pred:
        raise ValueError(f"mse_value: bad lengths pred={len(pred)} target={len(target)}")
    n = len(pred)
    return sum((p - t) * (p - t) for p, t in zip(pred, target)) / n


def mse_grad(pred, target) -> list:
    """Reference closed-form MSE gradient w.r.t. the predictions: ``dL/dpred_i = (2/n)(pred_i - target_i)``.
    This is the closed-form anchor for the closed-set path and the seed (``grad_pred``) a caller would feed
    into a model's backward if the predictions were a separate layer. (For the pure closed-set MSE-into-Tape
    path this is just a cross-check; the Tape's own autodiff already produces it.)"""
    pred = [float(p) for p in pred]
    target = [float(t) for t in target]
    if len(pred) != len(target) or not pred:
        raise ValueError(f"mse_grad: bad lengths pred={len(pred)} target={len(target)}")
    n = len(pred)
    return [(2.0 / n) * (p - t) for p, t in zip(pred, target)]


# === Path 2: the transcendental losses -- (loss_value, grad_logits) where grad_logits SEEDS the backward ===


def softmax_cross_entropy(logits, target_onehot) -> tuple:
    """Softmax cross-entropy: return ``(loss_value, grad_logits)``.

    Forward (numerically stable, the log-sum-exp form -- subtract ``max(logits)`` before ``exp`` so no
    overflow): ``loss = -sum_k onehot_k * log(softmax(logits)_k)``. The forward value needs ``log`` / ``exp``
    (transcendentals), so it is NOT a closed-set Tape node -- it is the monitored loss value only.

    Gradient (the famous closed-form, fully in the closed set): ``grad_logits = softmax(logits) - onehot``.
    This vector SEEDS the model's existing backward pass (one seed per logit), so the parameter gradient
    flows through the model autodiff with no exp/log ever entering the Tape. The softmax is the EXISTING G1
    ``activation.softmax_reference`` (no second softmax here).

    ``logits`` is a length-K score vector; ``target_onehot`` is a same-length one-hot label (or any
    probability vector that sums to 1). Returns ``(loss_value: float, grad_logits: list[float])``."""
    logits = [float(z) for z in logits]
    onehot = [float(y) for y in target_onehot]
    if len(logits) != len(onehot) or not logits:
        raise ValueError(
            f"softmax_cross_entropy: bad lengths logits={len(logits)} onehot={len(onehot)}"
        )
    # softmax via the existing G1 reference (axis_len = K: a single row). It already does the stable
    # max-subtraction internally.
    probs = softmax_reference(logits, len(logits))
    # loss = -sum_k onehot_k * log(p_k); log of the stable softmax probability (always in (0, 1]).
    loss = -sum(y * math.log(p) for y, p in zip(onehot, probs) if y != 0.0)
    grad_logits = [p - y for p, y in zip(probs, onehot)]
    return loss, grad_logits


def binary_cross_entropy_with_logits(logits, target) -> tuple:
    """Binary cross-entropy computed directly from logits (the numerically-stable fused form): return
    ``(loss_value, grad_logits)``.

    Forward (stable, never overflows for large ``|z|``): for each logit ``z`` and label ``y`` in {0,1},
    ``bce(z, y) = max(z, 0) - z*y + log(1 + exp(-|z|))``. This is the algebraically-exact rewrite of
    ``-[y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]`` that avoids computing ``sigmoid`` then ``log`` (which
    overflows / underflows for large ``|z|``). The total loss is the MEAN over the logits. ``log`` / ``exp``
    make it transcendental -> the monitored loss value only.

    Gradient (closed form, in the closed set): ``grad = (sigmoid(z) - y) / n`` per logit (the mean scales by
    ``1/n``). This SEEDS the model backward. The sigmoid is the EXISTING G1 ``activation.sigmoid_reference``.

    ``logits`` and ``target`` are same-length vectors (``target`` in {0,1} per element, the standard binary
    label). Returns ``(loss_value: float, grad_logits: list[float])``."""
    logits = [float(z) for z in logits]
    target = [float(y) for y in target]
    if len(logits) != len(target) or not logits:
        raise ValueError(
            f"binary_cross_entropy_with_logits: bad lengths "
            f"logits={len(logits)} target={len(target)}"
        )
    n = len(logits)
    # stable per-element BCE: max(z,0) - z*y + log(1 + exp(-|z|)).
    per = [max(z, 0.0) - z * y + math.log1p(math.exp(-abs(z))) for z, y in zip(logits, target)]
    loss = sum(per) / n
    probs = sigmoid_reference(logits)  # the existing G1 sigmoid reference
    grad_logits = [(p - y) / n for p, y in zip(probs, target)]
    return loss, grad_logits


def hinge(scores, target) -> tuple:
    """Multiclass / binary hinge (SVM-style) loss: return ``(loss_value, grad)``.

    Binary hinge for a single score ``s`` and label ``y`` in {-1, +1}: ``L = max(0, 1 - y*s)``; the gradient
    is ``-y`` when the margin is violated (``y*s < 1``) and ``0`` otherwise (the subgradient, a.e. -- the
    same a.e. convention the autodiff ``select`` VJP uses). The total loss is the MEAN over the scores, so the
    gradient scales by ``1/n``.

    This is included as a CLEAN closed-form ``(value, grad)`` pair on the same transcendental-style seed path,
    even though hinge itself uses no transcendental: its value is a ``max`` (piecewise linear, exactly the
    ``select`` shape) and its subgradient is the closed-form seed. ``scores`` and ``target`` (labels in
    {-1, +1}) are same-length vectors. Returns ``(loss_value: float, grad: list[float])``."""
    scores = [float(s) for s in scores]
    target = [float(y) for y in target]
    if len(scores) != len(target) or not scores:
        raise ValueError(f"hinge: bad lengths scores={len(scores)} target={len(target)}")
    n = len(scores)
    margins = [1.0 - y * s for s, y in zip(scores, target)]
    loss = sum(m if m > 0.0 else 0.0 for m in margins) / n
    # subgradient: -y/n where the margin is violated (m > 0), else 0 (a.e. convention at the kink m == 0).
    grad = [(-y / n) if m > 0.0 else 0.0 for m, y in zip(margins, target)]
    return loss, grad
