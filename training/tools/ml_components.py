#!/usr/bin/env python3
"""What BCIR already provides for machine learning, as data rather than prose.

The corpus's rule is that a subject teaches what a gate can check. Before an
`ml/` subject can be written, the honest question is which training-time
components *exist in this repository*, because a chapter about attention that
BCIR does not implement would be a chapter no gate can back.

This file is that answer, recorded so it can be checked rather than believed.
[`verify_ml_components.py`](verify_ml_components.py) resolves every symbol below
and, for the ones it claims run here, calls them and asserts a property. An
inventory that only imports proves that names exist, not that anything works.

**Three reachability classes, and the differences matter.**

`runs-here` is BCIR's tensor-framework-free substrate under `bcir/kbcir/` and
`bcir/lower/`: flat Python lists with explicit `rows`/`dim`, no torch, no numpy.
Exercisable on any host, which is why the corpus can gate it unconditionally.

`torch-gated` is the hosted stack under `bcir/hosted/training/stages.py`, which
needs torch to *import*, not merely to run: the module raises
`ModuleNotFoundError("hosted alignment stages require PyTorch")` at import time.
An earlier draft of this file claimed the opposite, having checked on a host
where torch happened to be installed -- the module imported, so the dependency
looked deferred. CI, which has no torch on the corpus runner, said otherwise
within two minutes. These are exercised where torch is present and skipped,
loudly and by name, where it is not.

`declared` is resolved but not exercised, each with a stated reason. Nothing is
in this class by omission: the gate requires the reason, and a component cannot
be quietly parked here.

An earlier note in this corpus said the whole training-time list needed torch
and was therefore out of reach. Measured, that was wrong twice: the substrate
needs no torch at all, and where torch IS installed the hosted SFT and DPO
stages run on the corpus's own example contracts.

The lesson from getting the import boundary wrong is the one this corpus keeps
relearning: a property measured on one host is a property of that host until a
second one disagrees.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

Reach = Literal["runs-here", "torch-gated", "declared"]


@dataclasses.dataclass(frozen=True)
class Component:
    """One declared capability, and where in BCIR it actually lives."""

    topic: str
    module: str
    symbols: tuple[str, ...]
    reach: Reach
    note: str
    # Required exactly when `reach` is "declared", and empty otherwise. A field
    # the gate can check beats a note it would have to read: grepping a prose
    # note for the phrase "not exercised" passes on any note that happens to
    # contain it, which is a witness asserting a substring rather than a fact.
    unexercised_reason: str = ""


# The list the corpus was asked to account for, plus what the survey found
# beside it. Order is the order a curriculum would take them in.
COMPONENTS: tuple[Component, ...] = (
    Component(
        topic="tokenization",
        module="bcir.hosted.training.bpe",
        symbols=("BytePairTokenizer",),
        reach="runs-here",
        note=(
            "Deterministic byte-fallback BPE. Already wired: the corpus trains it on "
            "its own prepared documents in export_training_examples.py."
        ),
    ),
    Component(
        topic="activation-functions",
        module="bcir.kbcir.activation",
        symbols=(
            "ActivationSpec",
            "relu_reference",
            "sigmoid_reference",
            "tanh_reference",
            "gelu_reference",
            "softmax_reference",
            "plan_activation",
            "check_activation",
            "is_exact",
            "libm_edges",
        ),
        reach="runs-here",
        note=(
            "References plus a planner and a legality check. `is_exact` and "
            "`libm_edges` are the part a textbook omits: which activations are exactly "
            "representable and where a libm implementation may legally differ."
        ),
    ),
    Component(
        topic="layer-normalization",
        module="bcir.kbcir.transformer",
        symbols=("layernorm_reference", "layernorm_stats"),
        reach="runs-here",
        note="LayerNorm with a statistics reader, so the invariant is checkable per row.",
    ),
    Component(
        topic="rms-normalization",
        module="bcir.kbcir.transformer_grads",
        symbols=("rmsnorm_reference", "rmsnorm_grad"),
        reach="runs-here",
        note="RMSNorm and its analytic gradient, checkable against a finite difference.",
    ),
    Component(
        topic="positional-encodings",
        module="bcir.kbcir.transformer_grads",
        symbols=("rope_reference", "rope_grad"),
        reach="runs-here",
        note=(
            "Rotary position embedding and its gradient. A rotation preserves the norm "
            "of every pair it turns, which is the property the gate asserts."
        ),
    ),
    Component(
        topic="attention",
        module="bcir.kbcir.attention",
        symbols=(
            "AttentionSpec",
            "attention_reference",
            "scores_reference",
            "plan_attention",
            "check_attention",
            "cost_vector",
            "bottleneck",
        ),
        reach="runs-here",
        note=(
            "Scaled dot-product attention as a reference, a plan, a legality check and "
            "a cost vector -- the same legality-then-cost order the rest of BCIR uses."
        ),
    ),
    Component(
        topic="transformer-block",
        module="bcir.kbcir.transformer",
        symbols=(
            "TransformerBlockSpec",
            "TransformerBlockParams",
            "transformer_block_reference",
            "multihead_attention_reference",
            "feedforward_reference",
            "swiglu_reference",
            "causal_mask",
            "check_transformer",
        ),
        reach="runs-here",
        note="A whole block: multi-head attention, SwiGLU feed-forward, causal masking.",
    ),
    Component(
        topic="loss-functions",
        module="bcir.kbcir.losses",
        symbols=(
            "mse",
            "mse_value",
            "mse_grad",
            "hinge",
            "softmax_cross_entropy",
            "binary_cross_entropy_with_logits",
        ),
        reach="runs-here",
        note="Value and gradient together, so a loss can be checked against its own slope.",
    ),
    Component(
        topic="automatic-differentiation",
        module="bcir.kbcir.autodiff",
        symbols=(
            "Tape",
            "grad",
            "grad_at",
            "hessian",
            "finite_difference_grad",
            "max_grad_error",
            "registry_completeness",
        ),
        reach="runs-here",
        note=(
            "Reverse-mode on an explicit tape, with finite-difference agreement and a "
            "registry-completeness check -- the differentiable ops are enumerable."
        ),
    ),
    Component(
        topic="optimizers",
        module="bcir.lower.optimizers",
        symbols=(
            "sgd_step",
            "momentum_step",
            "rmsprop_step",
            "adam_step",
            "reference_optimizer_trajectory",
            "emit_adam_step_c",
        ),
        reach="runs-here",
        note=(
            "SGD, momentum, RMSProp and Adam as pure update rules -- and each one also "
            "emits C, so the Python reference and the shipped kernel are the same rule. "
            "AdamW's decoupled decay is the hosted stack's (torch) optimizer, not this."
        ),
    ),
    Component(
        topic="training-loop",
        module="bcir.kbcir.training",
        symbols=("Dataset", "EarlyStop", "TrainResult", "train", "minibatches", "train_val_split"),
        reach="runs-here",
        note=(
            "A seeded loop over the rules above: batching, a validation split, early "
            "stopping and metrics, with no tensor framework underneath."
        ),
    ),
    Component(
        topic="precision-framework",
        module="bcir.kbcir.precision",
        symbols=(
            "Interval",
            "exact_reduce_q8",
            "naive_reduce_q8",
            "compensated_reduce_q8",
            "accuracy_bound",
            "quantization_error_bound",
            "reduction_error_bound",
            "ulp_distance",
            "meets_tolerance",
        ),
        reach="runs-here",
        note=(
            "Not a dtype menu: interval arithmetic, ULP distance, and proved bounds on "
            "quantization and reduction error. This is what lets a precision claim be "
            "a gate rather than a preference."
        ),
    ),
    Component(
        topic="quantization",
        module="bcir.kbcir.quantize",
        symbols=(
            "QGroup",
            "quantize_group",
            "quantize_per_group",
            "dequantize",
            "quantized_dot",
            "integer_dot",
            "max_abs_error",
            "accumulator_bits",
        ),
        reach="runs-here",
        note="BCIRQ8: grouped power-of-two scales, symmetric codes, exact integer dots.",
    ),
    Component(
        topic="low-bit-formats",
        module="bcir.kbcir.lowbit",
        symbols=(
            "PackedQ4Tensor",
            "pack_signed_int4",
            "unpack_signed_int4",
            "calibrated_q4q8_dot",
            "SmoothQuantPolicy",
            "calibrate_smoothquant",
            "admit_low_bit_format",
        ),
        reach="runs-here",
        note="Q4 packing and SmoothQuant calibration, with an admission check per format.",
    ),
    Component(
        topic="recurrent-cells",
        module="bcir.kbcir.recurrent",
        symbols=(
            "RnnParams",
            "LstmParams",
            "GruParams",
            "gru_cell_reference",
            "gru_cell_grads",
            "gru_unroll",
            "check_recurrent",
        ),
        reach="runs-here",
        note="LSTM and GRU cells with analytic gradients and a Jacobian cross-check.",
    ),
    Component(
        topic="mixture-of-experts",
        module="bcir.kbcir.moegate",
        symbols=("GNNGate", "FrozenGate", "train_gate", "freeze", "harden", "node_features"),
        reach="runs-here",
        note=(
            "A learned gate that is frozen and hardened before use -- the two-truth "
            "discipline: learned data may rank, never decide legality in flight."
        ),
    ),
    Component(
        topic="byte-latent-models",
        module="bcir.kbcir.byte_latent",
        symbols=(
            "ByteLatentSpec",
            "BytePatchSpec",
            "ByteVocabularySpec",
            "ByteSpeculationSpec",
            "LearnedPatchPolicySpec",
        ),
        reach="runs-here",
        note="Byte-level patching and speculation specs (the BLT/MambaByte direction).",
    ),
    Component(
        topic="supervised-fine-tuning",
        module="bcir.hosted.training.stages",
        symbols=("train_sft", "train_reasoning_sft"),
        reach="torch-gated",
        note=(
            "The hosted stack, and the consumer of this corpus's own `SFTExample`s. "
            "Measured here: 4 examples, 2 steps, loss 4.254 -> 3.965 in ~15s on CPU."
        ),
    ),
    Component(
        topic="preference-optimization",
        module="bcir.hosted.training.stages",
        symbols=("train_dpo", "train_reward_model", "HostedRewardModel"),
        reach="torch-gated",
        note=(
            "DPO and reward modelling -- the RLAIF half, fed by verifier-decided pairs. "
            "`train_dpo` refuses a reference model whose parameters are not frozen, "
            "which is the two-truth rule applied to a training stage."
        ),
    ),
    Component(
        topic="policy-optimization",
        module="bcir.hosted.training.stages",
        symbols=("train_ppo", "HostedValueModel", "generalized_advantage_estimate"),
        reach="declared",
        note="PPO with a value model and GAE -- the RLHF half.",
        unexercised_reason=(
            "needs a rollout and a reward source the corpus does not produce, so a "
            "cheap smoke run would prove the import rather than the algorithm"
        ),
    ),
    Component(
        topic="embedding-distillation",
        module="bcir.hosted.training.stages",
        symbols=("train_embedding_distillation", "HostedEmbeddingStudent"),
        reach="torch-gated",
        note=(
            "The stage that would give this corpus a LEARNED embedding provider, "
            "replacing the lexical baseline the retrieval evaluation still reports. "
            "It needs no external teacher: the targets are a cosine Gram matrix, and "
            "the corpus's own lexical provider produces the vectors it is built from. "
            "An earlier note here said this needed 'a teacher model this repository "
            "does not ship' -- measured, that was false, and the code constructs and "
            "calls no teacher at all."
        ),
    ),
    Component(
        topic="bounded-reasoning-search",
        module="bcir.hosted.training.reasoning",
        symbols=("SearchBudget", "SearchResult", "Verification", "bounded_reasoning_search"),
        reach="declared",
        note="Verified reasoning search under an explicit budget.",
        unexercised_reason="needs a generator, which is the model the corpus has not trained",
    ),
)

RUNS_HERE = tuple(c for c in COMPONENTS if c.reach == "runs-here")
TORCH_GATED = tuple(c for c in COMPONENTS if c.reach == "torch-gated")
DECLARED_ONLY = tuple(c for c in COMPONENTS if c.reach == "declared")
