"""G7: gem.conv as a planned claim -- a 2-D convolution, modeled EXACTLY like gem.matmul (kbcir.matmul)
and gem.activation (kbcir.activation).

The vision-alignment audit (docs/VISION_ALIGNMENT_AUDIT.md, Pillar 5c) flags that ``gem.matmul`` was the
only first-class tensor op -- no conv. This module adds the convolution. Like a matmul, a conv's RESULT is
fixed by a single reference (this module verifies every realization against the naive direct convolution),
while its REALIZATION -- the lowering STRATEGY (direct vs im2col+gemm) and the tile/loop order of the
underlying matmul -- is a free choice the K_BCIR cost model plans. The realization does NOT change the
result; it changes the cost, and K_BCIR picks it by the same deterministic dual-semiring analytic search
gem.matmul uses.

WHY 2-D (the dimension decision, stated honestly). A 2-D conv is the load-bearing vision/CNN primitive and
-- crucially for this design -- it is a STRUCTURED MATMUL: the im2col (patch-gather) reshaping turns a
direct conv into ``C[out_pix x C_out] = patches[out_pix x (C_in*Kh*Kw)] @ W[(C_in*Kh*Kw) x C_out]``, so its
realization can be PRICED THROUGH THE EXISTING matmul cost model with NO new roofline terms (the task's
core requirement: "a conv is a structured matmul; reuse matmul.cost_of"). A 1-D conv would be simpler but
exercises the im2col=matmul equivalence less convincingly; 2-D is the more useful and more representative
op, so it is the deliverable. Single-group, stride/padding parameterized, NO dilation and NO grouped/
depthwise channels (those are the deferred follow-up, exactly as matmul/activation deferred their MLIR
ports).

THE im2col=matmul EQUIVALENCE (the load-bearing identity). The direct convolution

    out[oc, y, x] = sum_{ic, ky, kx} in[ic, y*s + ky - pad, x*s + kx - pad] * W[oc, ic, ky, kx]   (+ zero-pad)

is exactly the matmul ``P @ Wm`` where ``P`` is the im2col patch matrix (one row per output pixel, one
column per (ic,ky,kx) tap) and ``Wm`` is the weight matrix (rows = taps, cols = output channels). Both
forms accumulate the IDENTICAL set of products, so for integer/exact inputs the direct conv and the
im2col+gemm are BIT-IDENTICAL -- which is what makes the lowering strategy a free, verified choice the cost
model may optimize (the same property tiling has for a matmul).

THE Q8 BRIDGE (R17). ``conv_via_bridge`` is the conv analog of ``matmul.gemm_via_bridge`` /
``activation.activation_via_bridge``: the inputs arrive as per-group quantized storage, the bridge
dequantizes them to float32, the TRUSTED conv runs on the round-tripped values, and BCIR owns the boundary
quantization -- so the SOLE error vs the true conv is the INPUT quantization (R17-certified by
``precision.quantization_error_bound``), the conv arithmetic itself being exact. Mirrors B5 exactly.

Scope: this is the oracle realization + planner + verification + the dequantized-bridge reference. Wiring
the chosen plan into an MLIR ``gem.conv`` law op is the separate G7-MLIR follow-up (kept separate, exactly
as matmul.py / activation.py kept their MLIR wiring separate)."""

from __future__ import annotations

from dataclasses import dataclass

from . import matmul
from .cost import COMPUTE, MEMORY, CostVector, TargetProfile
from .precision import quantization_error_bound
from .quantize import dequantize, quantize_per_group

# The conv lowering strategies the K_BCIR search ranks. Both reproduce the reference bit-for-bit; they
# differ only in cost (direct keeps no patch buffer; im2col materializes the patches but reuses the highly
# tiled/blocked gemm path). "auto" is never stored -- the planner resolves to one of these two.
STRATEGIES = ("direct", "im2col")


# ---- the planned-claim shape metadata (mirrors matmul.TilePlan / activation.ActivationSpec) -----


@dataclass(frozen=True)
class ConvSpec:
    """A gem.conv as a FIRST-CLASS planned claim: a single-group 2-D convolution's shape/dtype metadata --
    the conv analog of matmul.TilePlan's shape carrier. Layout is NCHW-flat (the input is C_in*H*W
    row-major, the weights are C_out*C_in*Kh*Kw row-major). ``dtype`` is the contracted element type
    ("f32" or "i32"). ``strategy`` records the K_BCIR-selected lowering (direct vs im2col+gemm); the inner
    ``tile`` is the chosen matmul TilePlan for the im2col gemm (None until planned / for the direct form)."""

    in_c: int  # C_in  (input channels)
    in_h: int  # H     (input height)
    in_w: int  # W     (input width)
    out_c: int  # C_out (output channels / filters)
    kh: int  # kernel height
    kw: int  # kernel width
    stride: int = 1
    pad: int = 0
    dtype: str = "f32"
    strategy: str = "im2col"  # the K_BCIR-selected lowering (direct | im2col)
    tile: "matmul.TilePlan | None" = None  # the chosen matmul plan for the im2col gemm

    @property
    def out_h(self) -> int:
        """Output height: floor((H + 2*pad - Kh)/stride) + 1 (the standard valid-after-padding formula)."""
        return (self.in_h + 2 * self.pad - self.kh) // self.stride + 1

    @property
    def out_w(self) -> int:
        return (self.in_w + 2 * self.pad - self.kw) // self.stride + 1

    @property
    def out_pixels(self) -> int:
        """Number of output spatial positions = out_h * out_w (the M of the im2col gemm)."""
        return self.out_h * self.out_w

    @property
    def taps(self) -> int:
        """Number of taps per output element = C_in*Kh*Kw (the K of the im2col gemm -- the reduction)."""
        return self.in_c * self.kh * self.kw

    @property
    def gemm_dims(self) -> tuple[int, int, int]:
        """The (M, N, K) of the equivalent im2col gemm: M=out_pixels, N=out_c, K=taps. This is what makes a
        conv a STRUCTURED MATMUL -- its realization is priced through the existing matmul cost model."""
        return self.out_pixels, self.out_c, self.taps

    @property
    def in_count(self) -> int:
        return self.in_c * self.in_h * self.in_w

    @property
    def weight_count(self) -> int:
        return self.out_c * self.in_c * self.kh * self.kw

    @property
    def out_count(self) -> int:
        return self.out_c * self.out_pixels


# ---- the math: a single reference + the verified realizations (all must agree) -------------------
# The direct convolution is the definition / single source of truth. The im2col+gemm realization
# accumulates the IDENTICAL products (just reshaped), so for integer inputs it is bit-identical -- the
# property the verifier checks (what makes the lowering strategy a free choice the cost model optimizes).


def _in_at(x, ic: int, iy: int, ix: int, spec: ConvSpec) -> float:
    """One (possibly zero-padded) input element ``in[ic, iy, ix]`` from the NCHW-flat buffer. An index
    outside the H x W frame reads 0.0 (the implicit zero padding) -- never an out-of-bounds access."""
    if iy < 0 or iy >= spec.in_h or ix < 0 or ix >= spec.in_w:
        return 0.0
    return x[(ic * spec.in_h + iy) * spec.in_w + ix]


def conv2d_reference(x, w, spec: ConvSpec) -> list[float]:
    """The naive DIRECT 2-D convolution -- the single source of truth every realization is checked against.
    ``x`` is C_in*H*W row-major; ``w`` is C_out*C_in*Kh*Kw row-major; the result is C_out*out_h*out_w
    row-major. Cross-correlation (the ML convention: no kernel flip), single group, zero padding."""
    oh, ow = spec.out_h, spec.out_w
    out = [0.0] * (spec.out_c * oh * ow)
    for oc in range(spec.out_c):
        for oy in range(oh):
            for ox in range(ow):
                s = 0.0
                for ic in range(spec.in_c):
                    for ky in range(spec.kh):
                        iy = oy * spec.stride + ky - spec.pad
                        for kx in range(spec.kw):
                            ix = ox * spec.stride + kx - spec.pad
                            wv = w[((oc * spec.in_c + ic) * spec.kh + ky) * spec.kw + kx]
                            s += _in_at(x, ic, iy, ix, spec) * wv
                out[(oc * oh + oy) * ow + ox] = s
    return out


def im2col(x, spec: ConvSpec) -> list[float]:
    """Build the im2col patch matrix ``P`` (row-major out_pixels x taps): row p = output pixel p, column t =
    the (ic, ky, kx) tap, so ``P[p, t] = in[ic, oy*s+ky-pad, ox*s+kx-pad]`` (zero-padded). This is the
    reshaping that turns the conv into a plain matmul ``P @ Wm`` -- the structured-matmul identity."""
    oh, ow = spec.out_h, spec.out_w
    taps = spec.taps
    p = [0.0] * (spec.out_pixels * taps)
    for oy in range(oh):
        for ox in range(ow):
            row = oy * ow + ox
            t = 0
            for ic in range(spec.in_c):
                for ky in range(spec.kh):
                    iy = oy * spec.stride + ky - spec.pad
                    for kx in range(spec.kw):
                        ix = ox * spec.stride + kx - spec.pad
                        p[row * taps + t] = _in_at(x, ic, iy, ix, spec)
                        t += 1
    return p


def weights_as_matrix(w, spec: ConvSpec) -> list[float]:
    """Reshape the conv weights ``W[C_out,C_in,Kh,Kw]`` into the gemm matrix ``Wm`` (row-major taps x
    out_c): row t = the (ic,ky,kx) tap, column oc = output channel. So ``Wm[t, oc] = W[oc, ic, ky, kx]`` --
    the transpose-into-K-major layout the im2col matmul ``P @ Wm`` consumes."""
    taps = spec.taps
    wm = [0.0] * (taps * spec.out_c)
    for oc in range(spec.out_c):
        t = 0
        for ic in range(spec.in_c):
            for ky in range(spec.kh):
                for kx in range(spec.kw):
                    wm[t * spec.out_c + oc] = w[
                        ((oc * spec.in_c + ic) * spec.kh + ky) * spec.kw + kx
                    ]
                    t += 1
    return wm


def conv2d_im2col(x, w, spec: ConvSpec, plan: "matmul.TilePlan | None" = None) -> list[float]:
    """The im2col+gemm realization: reshape to patches ``P`` and weights ``Wm``, run the (tiled) matmul
    ``Pm = P @ Wm`` (M=out_pixels, N=out_c, K=taps), then scatter ``Pm`` back to the C_out*out_h*out_w
    layout. Because it accumulates the IDENTICAL products as the direct form, the result is bit-identical
    for integer inputs (the verification property). ``plan`` is the matmul tile/loop choice (the realization
    detail the cost model picks); None uses the untiled reference matmul (same result)."""
    m, n, k = spec.gemm_dims
    p = im2col(x, spec)
    wm = weights_as_matrix(w, spec)
    pm = (
        matmul.matmul_tiled(p, wm, m, n, k, plan)
        if plan is not None
        else matmul.matmul_reference(p, wm, m, n, k)
    )
    # scatter Pm[out_pixel, oc] -> out[oc, oy, ox] (the gemm is pixel-major; the conv output is channel-major)
    oh, ow = spec.out_h, spec.out_w
    out = [0.0] * (spec.out_c * oh * ow)
    for row in range(m):
        for oc in range(n):
            out[oc * m + row] = pm[row * n + oc]
    return out


def conv2d_realize(x, w, spec: ConvSpec) -> list[float]:
    """Run the conv via the spec's K_BCIR-SELECTED realization (``spec.strategy`` + ``spec.tile``). Both
    strategies reproduce the reference exactly; this is the single entry the verifier drives so a test can
    assert "the planned realization == the naive reference" the way test_matmul does for the chosen tile."""
    if spec.strategy == "direct":
        return conv2d_reference(x, w, spec)
    return conv2d_im2col(x, w, spec, spec.tile)


def conv_via_bridge(x, w, spec: ConvSpec, group_size: int, bits: int) -> list[float]:
    """The Q8<->float32<->Q8 bridge wrapped around the TRUSTED conv (integrate, don't reinvent) -- the conv
    analog of matmul.gemm_via_bridge / activation.activation_via_bridge. The input + weights arrive as
    per-group quantized storage; the bridge dequantizes them to float32, the trusted direct conv runs on the
    round-tripped values, and BCIR owns the boundary quantization. So the oracle reference for the quantized
    path is the conv of the *round-tripped* inputs -- the SOLE error vs the true conv is the INPUT
    quantization (R17-certified by quantization_error_bound), the conv arithmetic itself being exact.
    Mirrors B5 exactly."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    dw = dequantize(quantize_per_group(w, group_size, bits))
    return conv2d_reference(dx, dw, spec)


# ---- shape/dtype well-formedness (mirrors activation.check_activation; NOT a new global R-law) ----
# gem.matmul / gem.activation validate their op shape inline and add NO globally-numbered R-law. We do the
# SAME here: a small op-level check the oracle rail calls explicitly, NOT a numbered global R-law across all
# rails / LangRef / gen_status (the heavier change the task forbids).

_DTYPES = ("f32", "i32")


def check_conv(
    spec: ConvSpec,
    in_shape: tuple[int, ...],
    in_dtype: str,
    w_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    out_dtype: str,
) -> list[str]:
    """Op-level well-formedness for a gem.conv claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape gem.matmul/gem.activation use inline, lifted into one named
    checker so the oracle rail and the tests can call it. NOT a globally-numbered R-law (matches
    gem.matmul/gem.activation, which add none). The laws a conv must satisfy:

      1. all extents are POSITIVE and the kernel fits the padded input (out_h, out_w >= 1);
      2. dtype is a KNOWN dtype and PRESERVED: out_dtype == in_dtype == the spec's declared dtype;
      3. the input shape is the NCHW-flat (C_in, H, W), the weight shape is (C_out, C_in, Kh, Kw), and the
         output shape is (C_out, out_h, out_w) -- the claim's metadata is internally consistent;
      4. stride >= 1 and pad >= 0 (a conv with stride 0 or negative pad is ill-formed)."""
    errs: list[str] = []
    # (4) hyperparameter sanity (checked first; out_h/out_w below assume stride >= 1).
    if spec.stride < 1:
        errs.append(f"conv: stride must be >= 1; got {spec.stride}")
    if spec.pad < 0:
        errs.append(f"conv: pad must be >= 0; got {spec.pad}")
    # (1) positive extents + the kernel fits the padded frame.
    for nm, v in (
        ("in_c", spec.in_c),
        ("in_h", spec.in_h),
        ("in_w", spec.in_w),
        ("out_c", spec.out_c),
        ("kh", spec.kh),
        ("kw", spec.kw),
    ):
        if v < 1:
            errs.append(f"conv: {nm} must be >= 1; got {v}")
    if errs:
        return errs  # later checks assume the extents are usable
    if spec.kh > spec.in_h + 2 * spec.pad or spec.kw > spec.in_w + 2 * spec.pad:
        errs.append(
            f"conv: kernel {spec.kh}x{spec.kw} does not fit the padded input "
            f"{spec.in_h + 2 * spec.pad}x{spec.in_w + 2 * spec.pad}"
        )
    if spec.out_h < 1 or spec.out_w < 1:
        errs.append(f"conv: output is empty (out_h={spec.out_h}, out_w={spec.out_w})")
    # (2) dtype known + preserved.
    for nm, dt in (("in", in_dtype), ("weight", out_dtype), ("spec", spec.dtype)):
        if dt not in _DTYPES:
            errs.append(f"conv: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"conv: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    if spec.dtype != in_dtype:
        errs.append(f"conv: spec dtype {spec.dtype!r} != input dtype {in_dtype!r}")
    # (3) shapes consistent with the spec's metadata.
    if tuple(in_shape) != (spec.in_c, spec.in_h, spec.in_w):
        errs.append(
            f"conv: input shape {tuple(in_shape)} != (C_in,H,W) {(spec.in_c, spec.in_h, spec.in_w)}"
        )
    if tuple(w_shape) != (spec.out_c, spec.in_c, spec.kh, spec.kw):
        errs.append(
            f"conv: weight shape {tuple(w_shape)} != (C_out,C_in,Kh,Kw) "
            f"{(spec.out_c, spec.in_c, spec.kh, spec.kw)}"
        )
    if tuple(out_shape) != (spec.out_c, spec.out_h, spec.out_w):
        errs.append(
            f"conv: output shape {tuple(out_shape)} != (C_out,out_h,out_w) "
            f"{(spec.out_c, spec.out_h, spec.out_w)}"
        )
    return errs


# ---- the cost / realization plug-in (reuses matmul.cost_of -- a conv IS a structured matmul) ------


def cost_of(
    spec: ConvSpec, strategy: str, target: TargetProfile, tile: "matmul.TilePlan | None" = None
) -> tuple[int, int, bool]:
    """The analytic (compute, mem, fits_cache) cost of a conv realization, PRICED THROUGH THE EXISTING
    matmul roofline (matmul.cost_of) -- the conv-is-a-structured-matmul wiring the task requires (no new
    roofline terms). M,N,K are the im2col gemm dims (out_pixels, out_c, taps); the strategy only changes how
    that gemm's traffic is realized:

      * ``im2col``: the gemm runs in the chosen ``tile`` (the matmul tile/loop plan); its (compute, mem,
        fits) is exactly ``matmul.cost_of`` at that tile -- the conv reuses the whole blocked-gemm reuse
        model. (The patch buffer's extra traffic is the im2col materialization; modeled as the gemm's A
        reads, which is what reading the patches costs.)
      * ``direct``: no patch buffer is materialized, so there is NO im2col write/re-read of P -- but the
        direct loop CANNOT block the reduction the way the gemm tile does (it streams the input window per
        output pixel), so it is modeled as the UNTILED gemm (tile = whole M,N,K). Same compute (identical
        MAC count -- a conv's MACs are fixed), more memory traffic on a large reduction -> the planner can
        SEE when blocking the im2col gemm beats the direct stream (and when it does not)."""
    m, n, k = spec.gemm_dims
    if strategy == "direct":
        return matmul.cost_of(m, n, k, m, n, k, target)  # untiled: the direct stream, no blocking
    pl = tile or matmul.plan_matmul(m, n, k, target)
    return matmul.cost_of(m, n, k, pl.tile_m, pl.tile_n, pl.tile_k, target)


def cost_vector(spec: ConvSpec, *, quant_bits: int = 0) -> CostVector:
    """Express a conv plan's cost in the production 12-dim K_BCIR CostVector -- the production wiring that
    lets a planned gem.conv be SCORED by the same algebra as every other claim (exactly as
    matmul.cost_vector / activation.cost_vector). The roofline terms (from matmul.cost_of, via cost_of)
    land on compute/memory; a quantized realization carries the R17 Q8-bridge bound on the accuracy axis."""
    compute, mem, _ = cost_of(spec, spec.strategy, TargetProfile.for_host(), spec.tile)
    acc = quantization_error_bound() if quant_bits > 0 else 0
    return CostVector.of(compute=compute, memory=mem, accuracy=acc)


def bottleneck(cv: CostVector) -> int:
    """The max,+ roofline bottleneck of a conv cost vector: the binding resource among compute and memory
    (identical to matmul.bottleneck -- the SOTA-scan finding that the bottleneck is a max, not a min,+
    sum). A small-kernel large-spatial conv is memory-bound; a many-channel conv shifts compute up."""
    return max(cv.v[COMPUTE], cv.v[MEMORY])


def plan_conv(
    in_c: int,
    in_h: int,
    in_w: int,
    out_c: int,
    kh: int,
    kw: int,
    stride: int = 1,
    pad: int = 0,
    dtype: str = "f32",
    target: TargetProfile | None = None,
) -> ConvSpec:
    """K_BCIR's deterministic conv-lowering choice -- the conv analog of matmul.plan_matmul. For each
    strategy (direct, im2col): compute its roofline BOTTLENECK = max(compute, mem) (the max,+ step), prefer
    a cache-fitting realization, and select the minimum-bottleneck strategy (the min,+ step); ties break on
    a stable key so the plan is fully reproducible. The im2col strategy carries the inner matmul plan (its
    own tile/loop search, run by matmul.plan_matmul). Returns the chosen ConvSpec (strategy + inner tile)."""
    target = target or TargetProfile.for_host()
    base = ConvSpec(in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype)
    m, n, k = base.gemm_dims
    inner = matmul.plan_matmul(m, n, k, target)  # the im2col gemm's own tile/loop plan
    best: ConvSpec | None = None
    best_key: tuple | None = None
    for strat in STRATEGIES:
        tile = inner if strat == "im2col" else None
        compute, mem, fits = cost_of(base, strat, target, tile)
        bn = max(compute, mem)  # max,+ : the binding roofline resource
        # min,+ selection: cache-fitting first, then min bottleneck, then prefer im2col (the blocked gemm
        # reuse) on a tie -- all deterministic.
        key = (not fits, bn, 0 if strat == "im2col" else 1)
        if best is None or key < best_key:  # type: ignore[operator]
            best = ConvSpec(in_c, in_h, in_w, out_c, kh, kw, stride, pad, dtype, strat, tile)
            best_key = key
    assert best is not None
    return best
