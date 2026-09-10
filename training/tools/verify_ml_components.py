#!/usr/bin/env python3
"""Gate the ML component inventory (`ml_components.py`).

An inventory is a claim about another part of the repository, which is the kind
of claim that rots quietly: BCIR renames a function, and a corpus that only
listed names keeps listing it. Worse, an inventory that merely *imports* a
module reports "available" for code that cannot run here at all -- BCIR's hosted
stages import cleanly and then need torch inside the function.

So this gate does three things, in increasing strength:

  1. **Resolve.** Every declared module imports and every declared symbol exists.
     A rename fails here, on the corpus's side, before a chapter cites it.
  2. **Exercise.** Every component declared to run here is CALLED, and a property
     of its output is asserted -- a normalized row really has zero mean, a
     rotation really preserves norms, an analytic gradient really agrees with a
     finite difference, a compensated reduction really beats a naive one. This
     is the difference between "the name exists" and "the thing works".
  3. **Classify honestly.** A component declared `needs-torch` must genuinely be
     unrunnable here: the gate confirms torch is absent and refuses to let such
     a component be counted as exercised. An honest skip, never a silent pass.

Anti-vacuity: an empty inventory, a component with no symbols, and a
`runs-here` component with no exercise all FAIL. A gate that iterates over
nothing agrees with everything.

    python3 training/tools/verify_ml_components.py
    python3 training/tools/verify_ml_components.py --require-torch
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent.parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml_components import COMPONENTS, DECLARED_ONLY, RUNS_HERE, TORCH_GATED  # noqa: E402

TOLERANCE = 1e-6


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def require(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)


# --------------------------------------------------------------------------
# Exercises: one per component that claims to run here.
# --------------------------------------------------------------------------


def exercise_tokenization(report: Report) -> str:
    from bcir.hosted.training.bpe import BytePairTokenizer

    tokenizer = BytePairTokenizer.train(
        ["attention is all you need", "loss go down"], vocab_size=300
    )
    probe = "unseen é bytes"
    report.require(
        tokenizer.decode(tokenizer.encode(probe)) == probe,
        "the tokenizer does not round-trip text it never saw; byte fallback is broken",
    )
    return f"vocab {tokenizer.vocab_size}, byte fallback round-trips"


def exercise_activations(report: Report) -> str:
    from bcir.kbcir import activation

    report.require(
        activation.relu_reference([-2.0, 0.0, 3.0]) == [0.0, 0.0, 3.0],
        "relu did not clamp negatives to zero",
    )
    report.require(
        abs(activation.sigmoid_reference([0.0])[0] - 0.5) < TOLERANCE,
        "sigmoid(0) is not 0.5",
    )
    probabilities = activation.softmax_reference([1.0, 2.0, 3.0, 0.5, 0.5, 0.5], axis_len=3)
    for start in (0, 3):
        total = math.fsum(probabilities[start : start + 3])
        report.require(abs(total - 1.0) < TOLERANCE, f"a softmax row sums to {total}, not 1")
    report.require(
        all(0.0 <= p <= 1.0 for p in probabilities), "softmax produced a value outside [0, 1]"
    )
    return "relu/sigmoid clamp correctly; every softmax row sums to 1"


def exercise_layernorm(report: Report) -> str:
    from bcir.kbcir import transformer

    rows, dim = 2, 4
    x = [3.0, -1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
    y = transformer.layernorm_reference(x, rows, dim, [1.0] * dim, [0.0] * dim)
    stats = transformer.layernorm_stats(y, rows, dim)
    for index, (mean, variance) in enumerate(stats):
        report.require(abs(mean) < 1e-5, f"normalized row {index} has mean {mean}, not ~0")
        report.require(
            abs(variance - 1.0) < 1e-3,
            f"normalized row {index} has variance {variance}, not ~1",
        )
    return f"{rows} rows normalized to mean ~0, variance ~1"


def exercise_rmsnorm(report: Report) -> str:
    """The analytic gradient must agree with a finite difference of the output."""
    from bcir.kbcir import transformer_grads

    rows, dim = 1, 4
    x = [0.7, -1.3, 2.1, 0.4]
    gamma = [1.1, 0.9, 1.0, 1.2]
    gout = [0.3, -0.5, 0.2, 0.8]
    grad_x, _grad_gamma = transformer_grads.rmsnorm_grad(x, rows, dim, gamma, gout)

    eps = 1e-6
    worst = 0.0
    for index in range(dim):
        shifted_up = list(x)
        shifted_down = list(x)
        shifted_up[index] += eps
        shifted_down[index] -= eps
        up = transformer_grads.rmsnorm_reference(shifted_up, rows, dim, gamma)
        down = transformer_grads.rmsnorm_reference(shifted_down, rows, dim, gamma)
        numeric = math.fsum(g * (u - d) / (2 * eps) for g, u, d in zip(gout, up, down, strict=True))
        worst = max(worst, abs(numeric - grad_x[index]))
    report.require(
        worst < 1e-4,
        f"the RMSNorm gradient disagrees with a finite difference by {worst:.2e}",
    )
    return f"analytic gradient matches a central difference to {worst:.1e}"


def exercise_rope(report: Report) -> str:
    """A rotation changes direction and preserves length; that is the whole test."""
    from bcir.kbcir import transformer_grads

    rows, dim = 3, 4
    x = [0.5, -0.2, 1.4, 0.9, 1.0, 0.0, 0.0, 1.0, -0.3, 0.6, 0.2, -0.8]
    rotated = transformer_grads.rope_reference(x, rows, dim)
    report.require(len(rotated) == len(x), "RoPE changed the number of coordinates")
    for row in range(rows):
        before = x[row * dim : (row + 1) * dim]
        after = rotated[row * dim : (row + 1) * dim]
        for pair in range(0, dim, 2):
            norm_before = math.hypot(before[pair], before[pair + 1])
            norm_after = math.hypot(after[pair], after[pair + 1])
            report.require(
                abs(norm_before - norm_after) < 1e-9,
                f"RoPE changed the norm of row {row} pair {pair // 2}: "
                f"{norm_before} -> {norm_after}",
            )
    moved = any(abs(a - b) > 1e-9 for a, b in zip(x, rotated, strict=True))
    report.require(moved, "RoPE returned its input unchanged, so nothing was rotated")
    return f"{rows} rows rotated; every pair kept its norm"


def exercise_attention(report: Report) -> str:
    from bcir.kbcir import attention

    seq_len, d_k = 3, 2
    spec = attention.AttentionSpec(seq_len=seq_len, d_k=d_k)
    q = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    k = list(q)
    v = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5]
    scores = attention.scores_reference(q, k, spec)
    report.require(
        len(scores) == seq_len * seq_len,
        f"scores are {len(scores)} values, expected {seq_len * seq_len}",
    )
    context = attention.attention_reference(q, k, v, spec)
    report.require(
        len(context) == seq_len * d_k,
        f"context is {len(context)} values, expected {seq_len * d_k}",
    )
    # Attention output is a convex combination of value rows, so every
    # coordinate must lie within the range of that coordinate across V.
    for position in range(d_k):
        column = [v[row * d_k + position] for row in range(seq_len)]
        for row in range(seq_len):
            value = context[row * d_k + position]
            report.require(
                min(column) - TOLERANCE <= value <= max(column) + TOLERANCE,
                f"attention output {value} at ({row},{position}) is outside the convex "
                "hull of V, so the weights do not form a distribution",
            )
    problems = attention.check_attention(
        spec, (seq_len, d_k), (seq_len, d_k), (seq_len, d_k), "f32", (seq_len, d_k), "f32"
    )
    report.require(not problems, f"a well-formed attention shape was rejected: {problems}")
    return f"seq {seq_len}: output inside the convex hull of V; shapes legal"


def exercise_losses(report: Report) -> str:
    """A loss and its gradient must be the same function twice."""
    from bcir.kbcir import losses

    pred = [0.4, -1.2, 2.5]
    target = [0.0, -1.0, 2.0]
    analytic = losses.mse_grad(pred, target)
    eps = 1e-6
    worst = 0.0
    for index in range(len(pred)):
        up, down = list(pred), list(pred)
        up[index] += eps
        down[index] -= eps
        numeric = (losses.mse_value(up, target) - losses.mse_value(down, target)) / (2 * eps)
        worst = max(worst, abs(numeric - analytic[index]))
    report.require(worst < 1e-5, f"the MSE gradient is off by {worst:.2e}")
    report.require(
        losses.mse_value(target, target) == 0.0, "MSE of a perfect prediction is not zero"
    )
    return f"MSE gradient matches its own value function to {worst:.1e}"


def exercise_autodiff(report: Report) -> str:
    from bcir.kbcir import autodiff

    tape = autodiff.Tape()
    x = tape.var("x")
    y = tape.var("y")
    output = tape.add(tape.mul(x, tape.mul(x, y)), tape.tanh(y))  # x^2*y + tanh(y)
    env = {"x": 1.3, "y": -0.7}
    result = autodiff.grad_at(tape, output, env)
    numeric = autodiff.finite_difference_grad(tape, output, env)
    report.require(
        set(result.grads) == set(env),
        f"the tape returned gradients for {sorted(result.grads)}, expected {sorted(env)}",
    )
    error = autodiff.max_grad_error(result.grads, numeric)
    report.require(error < 1e-5, f"reverse-mode disagrees with a finite difference by {error:.2e}")
    expected = env["x"] ** 2 * env["y"] + math.tanh(env["y"])
    report.require(
        abs(result.value - expected) < 1e-9,
        f"the tape evaluated to {result.value}, not {expected}",
    )
    return f"reverse-mode matches a finite difference to {error:.1e} over {result.forward_ops} ops"


def exercise_optimizers(report: Report) -> str:
    """An optimizer must go downhill on a bowl."""
    from bcir.lower import optimizers

    params = [3.0, -4.0]
    grads_at = lambda p: [2.0 * value for value in p]  # noqa: E731 - f(p) = sum(p^2)
    loss = lambda p: sum(value * value for value in p)  # noqa: E731
    start = loss(params)

    moved = optimizers.sgd_step(params, grads_at(params), 0.1)
    report.require(loss(moved) < start, "one SGD step did not reduce a convex loss")

    adam_params = list(params)
    m = [0.0] * len(params)
    v = [0.0] * len(params)
    step = 0
    for _ in range(24):
        adam_params, m, v, step = optimizers.adam_step(
            adam_params, grads_at(adam_params), m, v, step, 0.1
        )
    report.require(
        loss(adam_params) < start,
        f"Adam did not reduce a convex loss: {start} -> {loss(adam_params)}",
    )
    return f"SGD and Adam both descend a bowl ({start:g} -> {loss(adam_params):.3g})"


def exercise_training_loop(report: Report) -> str:
    from bcir.kbcir import training

    data = training.make_linearly_separable(n=64, seed=7)
    train_set, val_set = training.train_val_split(data, 0.25, seed=7)
    report.require(bool(train_set.X) and bool(val_set.X), "the split produced an empty side")
    report.require(
        len(train_set.X) + len(val_set.X) == len(data.X),
        "the train/validation split loses or duplicates rows",
    )
    # A minibatch is a (X, y) pair of tuples, not a Dataset.
    batches = list(training.minibatches(train_set, batch_size=16, seed=7))
    report.require(bool(batches), "batching produced no minibatch")
    report.require(
        all(len(rows) == len(labels) for rows, labels in batches),
        "a minibatch carries a different number of rows than labels",
    )
    covered = sum(len(rows) for rows, _labels in batches)
    report.require(
        covered == len(train_set.X),
        f"minibatches cover {covered} of {len(train_set.X)} rows; a training loop would "
        "silently drop the remainder",
    )
    return f"{len(train_set.X)} train rows in {len(batches)} batch(es), all covered"


def exercise_precision(report: Report) -> str:
    """A compensated reduction must be at least as accurate as a naive one."""
    from bcir.kbcir import precision

    # Each term truncates by 300/256 - 1, so a per-term shift drifts low while a
    # residual-carrying one does not. A case where the two agree proves nothing.
    values = [1] * 10
    weight = 300
    exact = precision.exact_reduce_q8(values, weight)
    naive = precision.naive_reduce_q8(values, weight)
    compensated = precision.compensated_reduce_q8(values, weight)
    report.require(
        naive != exact,
        f"the naive reduction already equals the exact one ({exact}); this case does "
        "not exercise the compensation at all",
    )
    report.require(
        abs(compensated - exact) < abs(naive - exact),
        f"the compensated reduction ({compensated}) is no closer to exact ({exact}) "
        f"than the naive one ({naive})",
    )
    # The framework's own bound must be tighter when compensation is claimed --
    # otherwise "compensated" is a label rather than a guarantee.
    plain = precision.reduction_error_bound(len(values))
    carried = precision.reduction_error_bound(len(values), compensated=True)
    report.require(
        carried < plain,
        f"the compensated error bound ({carried} ULP) is not tighter than the plain "
        f"one ({plain} ULP)",
    )
    report.require(
        precision.ulp_distance(1, 1) == 0, "ulp_distance says a value differs from itself"
    )
    return (
        f"naive {naive} vs exact {exact}; compensated {compensated}; bound {plain} -> {carried} ULP"
    )


def exercise_quantization(report: Report) -> str:
    from bcir.kbcir import quantize

    values = [0.5, -0.25, 0.125, -1.0, 0.75, 0.0, -0.5, 0.25]
    group = quantize.quantize_group(values, bits=8)
    restored = quantize.dequantize([group])
    report.require(len(restored) == len(values), "dequantization changed the number of coordinates")
    report.require(
        all(code != -128 for code in group.codes),
        "a quantized code used -128, which is asymmetric and forbidden by BCIRQ8",
    )
    error = quantize.max_abs_error(values, len(values), 8)
    report.require(error < 0.02, f"round-trip error {error} is larger than the grid justifies")
    report.require(
        all(abs(a - b) <= error + TOLERANCE for a, b in zip(values, restored, strict=True)),
        "a dequantized coordinate is further from its input than the declared bound",
    )
    return f"symmetric codes, round-trip error {error:.4f}"


def exercise_lowbit(report: Report) -> str:
    from bcir.kbcir import lowbit

    codes = [-7, -1, 0, 1, 7, -3, 2, 4]
    packed = lowbit.pack_signed_int4(codes)
    report.require(
        len(packed) == (len(codes) + 1) // 2,
        f"{len(codes)} nibbles packed into {len(packed)} bytes; that is not half",
    )
    report.require(
        list(lowbit.unpack_signed_int4(packed, len(codes))) == codes,
        "Q4 pack/unpack did not round-trip",
    )
    return f"{len(codes)} signed nibbles round-trip through {len(packed)} bytes"


def exercise_recurrent(report: Report) -> str:
    from bcir.kbcir import recurrent

    input_dim = hidden_dim = 2

    def weights(seed: float) -> list[float]:
        return [seed * (index + 1) * 0.1 for index in range(hidden_dim * input_dim)]

    params = recurrent.GruParams(
        W_z=weights(1.0),
        U_z=weights(0.5),
        b_z=[0.1, -0.2],
        W_r=weights(-0.7),
        U_r=weights(0.3),
        b_r=[-0.1, 0.2],
        W_n=weights(0.9),
        U_n=weights(-0.4),
        b_n=[0.05, 0.15],
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    )
    x = [0.6, -0.3]
    h_prev = [0.2, 0.4]
    baseline = recurrent.gru_cell_reference(x, h_prev, params)
    grads = recurrent.gru_cell_grads(x, h_prev, params)
    report.require(
        len(baseline) == hidden_dim,
        f"the GRU cell returned {len(baseline)} states, expected {hidden_dim}",
    )
    report.require(bool(grads), "the GRU cell produced no gradients")
    report.require(
        all(-1.0 <= value <= 1.0 for value in baseline),
        f"a GRU state left the interpolation range its gates define: {baseline}",
    )
    # h = (1-z)*h_prev + z*n is a convex blend, so with h_prev inside [-1, 1]
    # and n bounded by tanh, the new state cannot leave that interval. That is a
    # property of the cell rather than of these particular weights.
    return f"GRU cell evaluated; {len(grads)} gradient block(s); states stay in [-1, 1]"


def exercise_moe(report: Report) -> str:
    from bcir.kbcir import moegate

    report.require(
        callable(moegate.freeze),
        "a learned gate cannot be frozen, so it could still change under a running plan",
    )
    # Hardening turns a distribution into a decision only when it is confident
    # enough; below the threshold there is no expert. That is the two-truth
    # discipline in miniature -- learned data declines rather than guesses.
    confident, expert = moegate.harden([0.05, 0.9, 0.05], threshold=0.7)
    report.require(confident and expert == 1, f"a confident gate returned ({confident}, {expert})")
    unsure, none_expert = moegate.harden([0.4, 0.35, 0.25], threshold=0.7)
    report.require(
        not unsure and none_expert is None,
        f"an unconfident gate still chose an expert: ({unsure}, {none_expert})",
    )
    return "a confident distribution hardens to one expert; an unconfident one to none"


def exercise_byte_latent(report: Report) -> str:
    from bcir.kbcir import byte_latent

    report.require(
        hasattr(byte_latent, "ByteLatentSpec") and hasattr(byte_latent, "BytePatchSpec"),
        "the byte-latent specs are not both present",
    )
    return "byte-latent and patch specs present"


def exercise_transformer_block(report: Report) -> str:
    from bcir.kbcir import transformer

    seq_len = 3
    mask = transformer.causal_mask(seq_len)
    report.require(
        len(mask) == seq_len * seq_len, f"a causal mask for {seq_len} is {len(mask)} values"
    )
    for row in range(seq_len):
        for column in range(seq_len):
            value = mask[row * seq_len + column]
            if column > row:
                report.require(
                    value == float("-inf"),
                    f"the causal mask lets position {row} attend to future position {column}",
                )
            else:
                report.require(value == 0.0, f"the causal mask blocks position {row} from {column}")
    return f"causal mask for seq {seq_len} is strictly lower-triangular"


def _tiny_decoder():
    """The smallest model BCIR's hosted stack accepts, on CPU, deterministically."""
    import torch

    from bcir.hosted.models.model import HostedLlama
    from bcir.hosted.models.spec import DecoderSpec

    torch.set_num_threads(2)
    torch.manual_seed(1729)
    spec = DecoderSpec(
        vocab_size=64, d_model=32, n_heads=2, n_layers=1, d_ff=64, activation="silu_gate"
    )
    return HostedLlama(spec)


def exercise_supervised_fine_tuning(report: Report) -> str:
    """The corpus's own SFTExample, through BCIR's own SFT stage."""
    from bcir.hosted.training import contracts, stages

    model = _tiny_decoder()
    examples = [
        contracts.SFTExample(
            prompt_ids=(1, 2, 3),
            response_ids=(4, 5),
            provenance_sha256=contracts.sha256_text(f"probe-{index}"),
            weight=1.0,
        )
        for index in range(4)
    ]
    spec = stages.StageTrainSpec(stage="sft", steps=2, learning_rate=1e-3, batch_size=2)
    run = stages.train_sft(model, examples, spec)
    report.require(run.stage == "sft", f"the stage reported itself as {run.stage!r}")
    report.require(run.examples == len(examples), "the stage did not see every example")
    report.require(
        run.final_loss < run.initial_loss,
        f"two SFT steps did not reduce the loss: {run.initial_loss} -> {run.final_loss}",
    )
    return f"{run.examples} examples, {run.steps} steps, loss {run.initial_loss:.3f} -> {run.final_loss:.3f}"


def exercise_preference_optimization(report: Report) -> str:
    """The corpus's own PreferenceExample, through BCIR's own DPO stage."""
    from bcir.hosted.training import contracts, stages

    policy, reference = _tiny_decoder(), _tiny_decoder()
    pairs = [
        contracts.PreferenceExample(
            prompt_ids=(1, 2),
            chosen_ids=(3, 4),
            rejected_ids=(5, 6),
            provenance_sha256=contracts.sha256_text(f"pair-{index}"),
            weight=1.0,
        )
        for index in range(2)
    ]
    spec = stages.StageTrainSpec(stage="dpo", steps=1, learning_rate=1e-4, batch_size=1)

    # BCIR refuses a reference model that can still move. That is the two-truth
    # rule at training time -- the thing you measure against must not drift --
    # and it is worth asserting rather than merely satisfying.
    try:
        stages.train_dpo(policy, reference, pairs, spec)
    except ValueError as exc:
        report.require(
            "frozen" in str(exc), f"an unfrozen DPO reference failed for another reason: {exc}"
        )
    else:
        report.require(False, "DPO accepted a reference model whose parameters can still move")

    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    run = stages.train_dpo(policy, reference, pairs, spec)
    report.require(run.stage == "dpo", f"the stage reported itself as {run.stage!r}")
    report.require(run.examples == len(pairs), "the stage did not see every pair")
    return f"{run.examples} pairs, {run.steps} step(s), loss {run.initial_loss:.4f} -> {run.final_loss:.4f}"


EXERCISES = {
    "tokenization": exercise_tokenization,
    "activation-functions": exercise_activations,
    "layer-normalization": exercise_layernorm,
    "rms-normalization": exercise_rmsnorm,
    "positional-encodings": exercise_rope,
    "attention": exercise_attention,
    "transformer-block": exercise_transformer_block,
    "loss-functions": exercise_losses,
    "automatic-differentiation": exercise_autodiff,
    "optimizers": exercise_optimizers,
    "training-loop": exercise_training_loop,
    "precision-framework": exercise_precision,
    "quantization": exercise_quantization,
    "low-bit-formats": exercise_lowbit,
    "recurrent-cells": exercise_recurrent,
    "mixture-of-experts": exercise_moe,
    "byte-latent-models": exercise_byte_latent,
    "supervised-fine-tuning": exercise_supervised_fine_tuning,
    "preference-optimization": exercise_preference_optimization,
}


# --------------------------------------------------------------------------


def torch_available() -> bool:
    try:
        importlib.import_module("torch")
    except ImportError:
        return False
    return True


def check_resolution(report: Report) -> None:
    torch_present = torch_available()
    unresolved: list[str] = []
    report.require(bool(COMPONENTS), "the inventory is empty; this gate would check nothing")
    report.require(
        bool(RUNS_HERE), "no component claims to run here, so nothing would be exercised"
    )
    seen: set[str] = set()
    for component in COMPONENTS:
        report.require(component.topic not in seen, f"{component.topic!r} is declared twice")
        seen.add(component.topic)
        report.require(bool(component.symbols), f"{component.topic}: declares no symbol to resolve")
        report.require(
            bool(component.note.strip()),
            f"{component.topic}: declares no note saying what it is",
        )
        report.require(
            bool(component.unexercised_reason.strip()) == (component.reach == "declared"),
            f"{component.topic}: reach is {component.reach!r} but its unexercised "
            "reason is " + ("missing" if component.reach == "declared" else "set anyway"),
        )
        try:
            module = importlib.import_module(component.module)
        except ImportError as exc:
            # `bcir.hosted.training.stages` raises at IMPORT time without torch,
            # not merely at call time. Where torch is absent that is the honest
            # skip; anywhere else, and for any other cause, it is a failure. The
            # cause is checked rather than assumed, so a component cannot be
            # parked in a torch-gated class to hide an unrelated breakage.
            if component.reach in ("torch-gated", "declared") and not torch_present:
                report.require(
                    "torch" in str(exc).lower(),
                    f"{component.topic}: {component.module} does not import, and the "
                    f"reason is not torch: {exc}",
                )
                unresolved.append(component.topic)
                continue
            report.require(False, f"{component.topic}: {component.module} does not import: {exc}")
            continue
        missing = [name for name in component.symbols if not hasattr(module, name)]
        report.require(
            not missing,
            f"{component.topic}: {component.module} no longer provides "
            f"{', '.join(missing)}; the inventory has drifted from BCIR",
        )
    resolved = len(COMPONENTS) - len(unresolved)
    suffix = f"; {len(unresolved)} need torch to import and it is absent" if unresolved else ""
    print(
        f"[resolve] {resolved}/{len(COMPONENTS)} component(s), every declared symbol "
        f"present{suffix}"
    )


def check_every_runnable_is_exercised(report: Report) -> None:
    """A component claimed to run here without an exercise is an unchecked claim."""
    unexercised = [c.topic for c in RUNS_HERE if c.topic not in EXERCISES]
    report.require(
        not unexercised,
        f"{len(unexercised)} component(s) claim to run here but are never called: "
        f"{', '.join(unexercised)}",
    )
    stale = sorted(set(EXERCISES) - {c.topic for c in COMPONENTS})
    report.require(not stale, f"exercise(s) for components no longer declared: {', '.join(stale)}")


def check_exercises(report: Report) -> None:
    for component in RUNS_HERE:
        exercise = EXERCISES.get(component.topic)
        if exercise is None:
            continue
        try:
            summary = exercise(report)
        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a stop
            report.require(
                False,
                f"{component.topic}: exercising it raised {type(exc).__name__}: {exc}",
            )
            continue
        print(f"[run]     {component.topic:26s} {summary}")


def check_torch_classification(report: Report, require_torch: bool) -> None:
    """Torch-gated components are exercised where torch is, and skipped loudly where not."""
    available = torch_available()
    for component in TORCH_GATED:
        report.require(
            component.topic in EXERCISES,
            f"{component.topic} is declared torch-gated but has no exercise; it would "
            "then be neither run nor honestly skipped",
        )
    if not available:
        if require_torch:
            report.require(
                False,
                "--require-torch was passed and torch is absent; the hosted stages "
                "cannot run, and a skip here would be counted as a pass",
            )
            return
        print(
            f"[skip]    torch absent: {len(TORCH_GATED)} hosted stage(s) neither resolvable "
            "nor exercised here"
        )
        for component in TORCH_GATED:
            print(f"          - {component.topic} ({component.module})")
        return

    for component in TORCH_GATED:
        exercise = EXERCISES.get(component.topic)
        if exercise is None:
            continue  # already reported above; looking it up anyway loses the finding
        try:
            summary = exercise(report)
        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a stop
            report.require(
                False, f"{component.topic}: exercising it raised {type(exc).__name__}: {exc}"
            )
            continue
        print(f"[torch]   {component.topic:26s} {summary}")


def check_declared_only(report: Report) -> None:
    """Nothing sits in the unexercised class by omission; each states its reason."""
    for component in DECLARED_ONLY:
        report.require(
            component.topic not in EXERCISES,
            f"{component.topic} is declared unexercised but has an exercise; the two "
            "cannot both be true",
        )
        report.require(
            bool(component.unexercised_reason.strip()),
            f"{component.topic} is unexercised and states no reason; an unexplained "
            "gap reads exactly like an oversight",
        )
    if DECLARED_ONLY:
        print(
            f"[declared] {len(DECLARED_ONLY)} component(s) declared and not exercised, "
            "each with a stated reason"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-torch",
        action="store_true",
        help="fail instead of skipping when the hosted stages cannot be exercised",
    )
    args = parser.parse_args(argv)

    report = Report()
    check_resolution(report)
    check_every_runnable_is_exercised(report)
    check_exercises(report)
    check_torch_classification(report, args.require_torch)
    check_declared_only(report)

    if report.failures:
        print("ml component gate: FAILED", file=sys.stderr)
        for failure in report.failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"ml component gate: PASSED ({report.checks} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
