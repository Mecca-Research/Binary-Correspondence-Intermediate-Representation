"""G1: gem.activation as a planned claim -- the elementwise tensor-op family relu/sigmoid/tanh/gelu/
softmax, modeled exactly like gem.matmul (kbcir.matmul) is.

An activation is a tensor op carrying shape/rank/dtype metadata. Like a matmul, its RESULT is fixed by a
single reference (this module verifies every realization against that reference), while its REALIZATION --
lane width, the elementwise loop -- is a free choice the K_BCIR cost model plans (the dual-semiring
tile/loop search of matmul.py collapses here to a width/loop search, since an elementwise op has no inner
reduction to tile except softmax's last-axis reduce). This is the foundational prerequisite for matmul+
activation fusion (G2), baked-weights inference (G5), and training (G6).

THE TWO-TRUTH / QUARANTINE BOUNDARY (the load-bearing design constraint -- see also kbcir.precision,
kbcir.quantize, and the B5 emit_blas_gemm_c precedent): float/transcendental math must NOT sit on the
deterministic legality path. We split the family in two:

  * ``relu`` is integer/Q-fixed CLEAN: ``max(0, x)`` is a pure comparison/select, deterministic and EXACT
    (0 ULP) on any dtype -- it never touches a transcendental, so it lowers as a plain integer/float
    ``max`` with NO accuracy contract. It is the activation analog of the exact integer matmul path.

  * ``sigmoid``/``tanh``/``gelu``/``softmax`` need a transcendental (exp / tanh). They are handled the way
    BCIR already handles every float / transcendental seam: routed through the TRUSTED ``c.call.libm:``
    FFI edge (``expf``/``tanhf``), exactly as B5's ``emit_blas_gemm_c`` wraps the trusted ``cblas_sgemm``
    through the same edge, and exactly as the cfront rail lowers a ``<math.h>`` call (``c.call.libm:expf``;
    see bcir/frontends/cfront/lower.py). WHY the libm edge and NOT a frozen Q8 LUT: the libm edge is the
    EXISTING, consistent BCIR pattern for a trusted-but-opaque float kernel (B5 set it; the cfront rail
    uses it for every math.h function), it keeps the transcendental fully OFF the deterministic rail
    (libm is opaque to the legality laws, like any external edge), and -- crucially -- it reuses the SAME
    certification story B5 already established: the trusted external (here expf/tanhf, there sgemm) is
    treated as EXACT, so when the activation consumes quantized ``_BitInt(N)`` lanes the only certified
    error is the Q8<->f32<->Q8 bridge step at the boundary, R17-bounded by
    ``precision.quantization_error_bound`` (1 ULP at the realized per-group grid) -- identical to the B5
    bound. A frozen Q8 LUT is the alternative the design note sanctions, but it would introduce a NEW
    frozen-table artifact + its own approximation bound; the libm edge needs neither and matches what the
    repo already ships, so it is the consistent choice. (The dequantized-input reference below makes the
    bridge the SOLE error source, exactly like matmul.gemm_via_bridge.)

Scope: this is the oracle realization + planner + verification + the dequantized-bridge reference. Wiring
the chosen plan into an MLIR ``gem.activation`` law op is the separate G1-MLIR follow-up (kept separate,
as matmul.py's MLIR wiring was)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cost import COMPUTE, MEMORY, CostVector, TargetProfile
from .precision import quantization_error_bound
from .quantize import dequantize, quantize_per_group

# The activation family. ``exact`` activations are integer/Q-fixed clean (0 ULP, no transcendental, no
# accuracy contract); the rest route a transcendental through the trusted ``c.call.libm:`` edge.
ACTIVATIONS = ("relu", "sigmoid", "tanh", "gelu", "softmax")
EXACT_ACTIVATIONS = ("relu",)  # max(0,x): deterministic, 0 ULP
TRANSCENDENTAL_ACTIVATIONS = ("sigmoid", "tanh", "gelu", "softmax")  # need expf/tanhf via libm

# Which trusted libm symbol each transcendental activation's C kernel calls through the c.call.libm:
# edge (the B5 wrap pattern). relu needs none (pure max). gelu uses tanh's erf-free tanh approximation.
_LIBM_EDGE = {
    "sigmoid": ("expf",),
    "tanh": ("tanhf",),
    "gelu": ("tanhf",),  # the standard tanh-approximation GELU (no erf): 0.5x(1+tanh(...))
    "softmax": ("expf",),
}


def is_exact(kind: str) -> bool:
    """True iff `kind` lowers exactly (0 ULP) with no transcendental -- only ``relu``. The complement
    needs the ``c.call.libm:`` edge and is only R17-certified through the Q8 boundary bridge."""
    _check_kind(kind)
    return kind in EXACT_ACTIVATIONS


def libm_edges(kind: str) -> tuple[str, ...]:
    """The trusted libm symbols `kind`'s C kernel calls through the ``c.call.libm:`` FFI edge (empty for
    the exact ``relu``). The link rail adds ``-lm`` for these (cf. frontends/cfront/linkflags.py)."""
    _check_kind(kind)
    return _LIBM_EDGE.get(kind, ())


def _check_kind(kind: str) -> None:
    if kind not in ACTIVATIONS:
        raise ValueError(f"unknown activation {kind!r}; expected one of {ACTIVATIONS}")


# ---- the math: a single reference per activation (every realization must agree) ------------------
# These are the definition. relu is exact integer/float; the transcendental four are computed with the
# trusted math functions (the oracle's stand-in for the libm edge the C kernel calls), so the C kernel --
# which calls the SAME libm -- reproduces them to within float round-off. A softmax is the canonical
# reduce-max -> exp -> normalize over the last axis (numerically stable: subtract the row max first).

_E = 2.718281828459045


def relu_reference(x: list[float]) -> list[float]:
    """relu(x) = max(0, x), elementwise -- the EXACT integer/Q-fixed-clean activation (0 ULP, no
    transcendental). The single source of truth the relu realization is checked against."""
    return [v if v > 0.0 else 0.0 for v in x]


def sigmoid_reference(x: list[float]) -> list[float]:
    """sigmoid(x) = 1/(1+exp(-x)), elementwise. Transcendental -> the trusted expf edge."""
    return [1.0 / (1.0 + math.exp(-v)) for v in x]


def tanh_reference(x: list[float]) -> list[float]:
    """tanh(x), elementwise. Transcendental -> the trusted tanhf edge."""
    return [math.tanh(v) for v in x]


def gelu_reference(x: list[float]) -> list[float]:
    """gelu(x), the standard tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))) -- the form
    used by GPT/BERT, computed through the same tanhf edge the C kernel calls (no erf needed)."""
    c = math.sqrt(2.0 / math.pi)
    return [0.5 * v * (1.0 + math.tanh(c * (v + 0.044715 * v * v * v))) for v in x]


def softmax_reference(x: list[float], axis_len: int) -> list[float]:
    """softmax over the LAST axis: reduce-max -> exp(x - max) -> normalize, per row of `axis_len`
    elements (the numerically-stable form: subtract the row max before exp, so no overflow). `x` is a
    flat row-major tensor whose last axis has length `axis_len`; len(x) must be a multiple of it."""
    if axis_len < 1:
        raise ValueError(f"softmax axis_len must be >= 1; got {axis_len}")
    if len(x) % axis_len != 0:
        raise ValueError(f"softmax: len(x)={len(x)} not a multiple of axis_len={axis_len}")
    out = [0.0] * len(x)
    for r in range(0, len(x), axis_len):
        row = x[r : r + axis_len]
        m = max(row)  # reduce-max (the stability shift)
        ex = [math.exp(v - m) for v in row]  # exp over the shifted row
        s = sum(ex) or 1.0  # normalize (sum is exact-positive, never 0)
        for j in range(axis_len):
            out[r + j] = ex[j] / s
    return out


def activation_reference(kind: str, x: list[float], axis_len: int | None = None) -> list[float]:
    """Dispatch to the activation's reference. `axis_len` is required for softmax (the last-axis length);
    ignored for the elementwise four."""
    _check_kind(kind)
    if kind == "relu":
        return relu_reference(x)
    if kind == "sigmoid":
        return sigmoid_reference(x)
    if kind == "tanh":
        return tanh_reference(x)
    if kind == "gelu":
        return gelu_reference(x)
    # softmax
    return softmax_reference(x, axis_len if axis_len is not None else len(x))


def activation_via_bridge(
    kind: str, x: list[float], group_size: int, bits: int, axis_len: int | None = None
) -> list[float]:
    """The Q8<->float32<->Q8 bridge wrapped around the TRUSTED activation (integrate, don't reinvent) --
    the activation analog of matmul.gemm_via_bridge. The input arrives as per-group quantized storage; the
    bridge dequantizes to float32, the trusted activation (relu's exact max, or the libm-edge expf/tanhf
    for the rest) is applied, and BCIR owns the boundary quantization. So the oracle reference for the
    quantized path is the activation of the *round-tripped* inputs -- the SOLE error vs the true activation
    is the INPUT quantization (R17-certified by quantization_error_bound), the activation itself being
    exact (relu) or trusted (the libm transcendentals). Mirrors B5 exactly."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return activation_reference(kind, dx, axis_len)


# ---- the planned-claim shape metadata (mirrors matmul.TilePlan) ----------------------------------


@dataclass(frozen=True)
class ActivationSpec:
    """A gem.activation as a FIRST-CLASS planned claim: the op kind + its tensor shape/dtype metadata,
    the activation analog of matmul.TilePlan's shape carrier. `shape` is the row-major tensor extent;
    `axis_len` is the softmax reduction axis (the last dim) -- 0/unused for the elementwise four. `dtype`
    is the contracted element type ("f32" or "i32"). `exact` records the quarantine side: relu is 0-ULP
    integer/Q-fixed clean, the rest carry a transcendental routed to the trusted libm edge."""

    kind: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    axis_len: int = 0  # softmax last-axis length (0 for the elementwise activations)
    width: int = 1  # the K_BCIR-selected lane width (the realization choice)

    @property
    def count(self) -> int:
        """Total element count = product of the shape (rank-agnostic, like Resource.count)."""
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def exact(self) -> bool:
        return self.kind in EXACT_ACTIVATIONS


# ---- shape/dtype well-formedness (mirrors matmul's op-level validation; NOT a new global R-law) ---
# gem.matmul validates its op shape inline (cost_of asserts dims, plan_matmul bounds the tiles, the
# verifier never added an R22). We do the SAME here: a small op-level check the oracle rail calls
# explicitly, NOT a numbered global R-law across all rails / LangRef / gen_status (that is the heavier,
# separate change the task forbids). It checks input/output shape + dtype consistency for an activation.

_DTYPES = ("f32", "i32")


def check_activation(
    spec: ActivationSpec,
    in_shape: tuple[int, ...],
    in_dtype: str,
    out_shape: tuple[int, ...],
    out_dtype: str,
) -> list[str]:
    """Op-level well-formedness for a gem.activation claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape gem.matmul uses inline, lifted into one named checker so the
    oracle rail and the tests can call it. NOT a globally-numbered R-law (matches gem.matmul, which adds
    none). The laws, all of which an activation must satisfy:

      1. an activation is ELEMENTWISE-SHAPED: input and output tensors have the SAME shape (every
         activation, including softmax, is shape-preserving -- softmax reduces only for the normalizer,
         the output tensor is the input's shape);
      2. dtype is PRESERVED: out_dtype == in_dtype == the spec's declared dtype, and is a known dtype;
      3. the spec's own shape matches the input shape (the claim's metadata is consistent);
      4. an EXACT activation (relu) may be i32 (integer/Q-fixed) OR f32; a TRANSCENDENTAL activation
         needs a float result, so it must be f32 (its libm edge returns float) -- never i32;
      5. softmax declares a valid last-axis length that divides the tensor and equals the last dim;
         the elementwise activations declare axis_len == 0 (no reduction axis)."""
    errs: list[str] = []
    if spec.kind not in ACTIVATIONS:
        errs.append(f"unknown activation {spec.kind!r}")
        return errs
    # (2) dtype known + preserved.
    for nm, dt in (("in", in_dtype), ("out", out_dtype), ("spec", spec.dtype)):
        if dt not in _DTYPES:
            errs.append(f"{spec.kind}: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"{spec.kind}: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    if spec.dtype != in_dtype:
        errs.append(f"{spec.kind}: spec dtype {spec.dtype!r} != input dtype {in_dtype!r}")
    # (1) shape-preserving (elementwise).
    if tuple(in_shape) != tuple(out_shape):
        errs.append(
            f"{spec.kind}: not shape-preserving (in {tuple(in_shape)} != out {tuple(out_shape)})"
        )
    # (3) spec metadata consistent with the input shape.
    if tuple(spec.shape) != tuple(in_shape):
        errs.append(f"{spec.kind}: spec shape {tuple(spec.shape)} != input shape {tuple(in_shape)}")
    # (4) the quarantine dtype rule: a transcendental needs a float result.
    if spec.kind in TRANSCENDENTAL_ACTIVATIONS and in_dtype != "f32":
        errs.append(
            f"{spec.kind}: transcendental activation needs f32 (the libm edge returns float), "
            f"got {in_dtype!r}"
        )
    # (5) the softmax axis.
    if spec.kind == "softmax":
        if not in_shape:
            errs.append("softmax: needs a non-empty (>=1-D) shape for its last-axis reduction")
        else:
            last = in_shape[-1]
            if spec.axis_len != last:
                errs.append(f"softmax: axis_len {spec.axis_len} != last shape dim {last}")
            if spec.axis_len < 1 or (spec.count % spec.axis_len != 0):
                errs.append(
                    f"softmax: axis_len {spec.axis_len} must be >=1 and divide count {spec.count}"
                )
    elif spec.axis_len != 0:
        errs.append(
            f"{spec.kind}: elementwise activation must declare axis_len 0, got {spec.axis_len}"
        )
    return errs


# ---- the cost / realization plug-in (mirrors matmul.cost_of / cost_vector / plan_*) ---------------


def cost_of(spec: ActivationSpec, width: int, target: TargetProfile) -> tuple[int, int]:
    """The analytic (compute, mem) cost of realizing an activation at lane `width`, in target units --
    the same roofline shape matmul.cost_of uses, specialized to an elementwise op.

    compute: count ops / (vector_width * fma) -- the elementwise throughput term. A transcendental costs
             more per element than relu's single max; we model that as a small per-op multiplier so the
             cost model can SEE that gelu is heavier than relu (the planner ranks them honestly).
    mem: bytes streamed (1 read + 1 write per element) / bandwidth. softmax re-reads the row for its
         reduce-max + normalize passes (3 streams over the axis), so it carries more traffic."""
    n = max(1, spec.count)
    thr = max(1, target.vector_width * (2 if target.fma else 1))
    # per-element compute weight: relu == 1 (a max); a transcendental ~ a libm call we model at 8x; gelu
    # is heavier still (a cube + a tanh), softmax adds the exp.
    op_weight = {"relu": 1, "sigmoid": 8, "tanh": 8, "gelu": 12, "softmax": 8}[spec.kind]
    compute = ((n * op_weight) + thr - 1) // thr
    streams = 3 if spec.kind == "softmax" else 2  # softmax: max + exp/sum + normalize re-reads
    bw = max(1, target.mem_unit * target.mem_channels)
    mem = (n * streams + bw - 1) // bw
    return compute, mem


def cost_vector(spec: ActivationSpec, *, quant_bits: int = 0) -> CostVector:
    """Express an activation plan's cost in the production 12-dim K_BCIR CostVector -- the production
    wiring that lets a planned gem.activation be SCORED by the same algebra as every other claim (exactly
    as matmul.cost_vector does). The roofline terms land on compute/memory; a quantized realization carries
    the R17 Q8-bridge bound on the accuracy axis. A relu is EXACT, so it carries NO accuracy cost even
    when its inputs are quantized lanes (its own op adds 0 ULP -- the only error a quantized relu has is
    the bridge step itself, the same +1 ULP the bridge always costs; a transcendental is identical because
    the libm kernel is treated as trusted/exact). So the accuracy axis is the bridge bound iff quantized."""
    width = spec.width if spec.width >= 1 else 1
    compute, mem = cost_of(spec, width, TargetProfile.for_host())
    acc = quantization_error_bound() if quant_bits > 0 else 0
    return CostVector.of(compute=compute, memory=mem, accuracy=acc)


def bottleneck(cv: CostVector) -> int:
    """The max,+ roofline bottleneck of an activation cost vector: the binding resource among compute and
    memory (identical to matmul.bottleneck -- the SOTA-scan finding that the bottleneck is a max, not a
    min,+ sum). Elementwise activations are memory-bound; the heavy transcendentals shift compute up."""
    return max(cv.v[COMPUTE], cv.v[MEMORY])


def _divisor_widths(target: TargetProfile, count: int) -> list[int]:
    """Candidate lane widths to cost: the target's available lane widths, capped at the element count
    (a width wider than the tensor is pointless). Deterministic + small (mirrors matmul._divisor_tiles)."""
    return sorted({w for w in target.widths() if w <= max(1, count)} | {1})


def plan_activation(
    kind: str,
    shape: tuple[int, ...],
    dtype: str = "f32",
    axis_len: int = 0,
    target: TargetProfile | None = None,
) -> ActivationSpec:
    """K_BCIR's deterministic lane-width choice for an activation -- the elementwise analog of
    matmul.plan_matmul. For every candidate width: compute the roofline BOTTLENECK = max(compute, mem)
    (the max,+ step) and select the minimum-bottleneck width (the min,+ step), ties broken on the WIDER
    lane then a stable key so the plan is fully reproducible. Returns the chosen ActivationSpec."""
    _check_kind(kind)
    target = target or TargetProfile.for_host()
    base = ActivationSpec(kind=kind, shape=tuple(shape), dtype=dtype, axis_len=axis_len)
    best: ActivationSpec | None = None
    best_key: tuple | None = None
    for w in _divisor_widths(target, base.count):
        compute, mem = cost_of(base, w, target)
        bn = max(compute, mem)  # max,+ : the binding roofline resource
        # min,+ selection: min bottleneck, then prefer the WIDER lane (more throughput), stable.
        key = (bn, -w)
        if best is None or key < best_key:  # type: ignore[operator]
            best, best_key = ActivationSpec(kind, tuple(shape), dtype, axis_len, w), key
    assert best is not None
    return best
