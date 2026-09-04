"""G5 (roadmap task #52): the baked-weights end-to-end inference kernel emitter.

The VISION_ALIGNMENT_AUDIT (Pillar 5a) flags that BCIR has every piece -- ``#embed`` Q8 tables
(``bcir.abi.q8_tables``), ``gem.matmul`` (``bcir.kbcir.matmul`` / ``lower.c_kernel.emit_blas_gemm_c``),
``gem.activation`` (``bcir.kbcir.activation`` / ``emit_activation_kernel_c``), and the G2 fused
matmul+activation epilogue (``emit_matmul_activation_c``) -- but emits NO end-to-end inference kernel
with a trained model's weights baked in. This module is that emitter.

THE VISION (Pillar 5a): once a model is trained its weights are CONSTANT. So bake them straight into the
emitted C as ``static const`` arrays (compiled in -- no runtime load, no parsing, the constants live in
.rodata where the optimizer can fold them) and run the forward pass at the hardware limit. The emitted
function is ONE self-contained ``void fn(const float *x, float *y)`` that computes, layer by layer,

    h_{l+1} = activation_l( W_l @ h_l + b_l )

reusing the SAME inline-activation epilogue the G2 fused kernel (``c_kernel._inline_activation_expr``)
uses -- the activation is applied to each output element AS IT IS PRODUCED, so no per-layer intermediate
is materialized within a layer (the matmul product never round-trips to memory before the activation).

LAYOUT. A dense layer is a matrix-VECTOR product: ``W`` is row-major ``out x in``, ``h`` is a length-``in``
column vector, ``y[o] = act( sum_i W[o,i]*h[i] + b[o] )``. That is exactly ``emit_matmul_activation_c``'s
fused ``C[MxN] = act(A[MxK] @ B[KxN])`` with ``M=out, K=in, N=1`` (A=W, B=h) -- the bias is the only
addition, folded into the accumulator ``s`` before the inline activation. Each layer writes into a small
``static`` scratch buffer; the next layer reads it (the only cross-LAYER buffers, sized to the widest
hidden vector). Softmax, being a last-axis ROW reduction, can NOT be an inline per-element epilogue (the
G2 scope-out carries over): it is emitted as a SEPARATE final normalizing pass over the last layer's
output, never fused -- and only as the final layer.

BAKING. Each layer's W (and b) is a ``static const float`` array. SMALL tables (<= ``_EMBED_THRESHOLD``
elements) are emitted as a brace-enclosed float literal -- readable, and the constant folder sees through
it. LARGE tables follow the ``q8_tables.py`` pattern: a little-endian ``float`` blob baked with C23
``#embed`` (ISO/IEC 9899:2023 6.10.3) when the toolchain supports it, with the byte-identical fallback
array selected by ``__has_embed`` (the nested-``#if`` probe q8_tables.py established) so the SAME source
builds on pre-``#embed`` toolchains. Either way the weights are COMPILED IN.

QUANTIZED PATH (R17). An f32-baked model is EXACT -- the emitted C reproduces the Python forward pass
bit-for-bit for an all-relu (integer/exact) network, and to float round-off when a transcendental
activation is in the chain (the kernel calls the SAME libm the oracle reference does). A Q8-QUANTIZED
model bakes the round-tripped (Q8<->f32<->Q8 bridged) weights instead; the inference output's accuracy is
then bounded by the bridge step -- ``precision.quantization_error_bound`` (R17), exactly as B5 / the
activation bridge bound it. ``quantize_model`` produces that quantized model (reusing ``quantize.py``);
``InferenceModel.r17_bound`` reports the certified bound (0 for an exact f32 model).

SCOPE (honest). Supported: a chain of DENSE (fully-connected) layers, each with an optional bias and one
activation from the elementwise family relu / sigmoid / tanh / gelu (fused inline), plus softmax as the
FINAL layer only (a separate non-fused normalizing pass). DEFERRED to G7: conv, attention, residual /
skip connections, batched (M>1) inference, and weight quantization finer than per-tensor groups. The
emitted attestation comment records the layers, activations, the parameter count, and the R17 bound.

This is the ORACLE realization + emitter + verification (the reference forward pass, the compile-and-run
gate, the R17 assertion). Wiring it into an MLIR ``gem.inference`` law op is the separate G5-MLIR
follow-up (kept separate, exactly as matmul.py / activation.py kept their MLIR wiring separate)."""

from __future__ import annotations

from ..toolchain import host_link_args

import struct
from dataclasses import dataclass, field

from ..kbcir.activation import ACTIVATIONS, activation_reference, libm_edges
from ..kbcir.precision import quantization_error_bound
from ..kbcir.quantize import dequantize, quantize_per_group
from .c_kernel import _inline_activation_expr

# Tables with more than this many elements bake via C23 #embed (with the __has_embed fallback array, the
# q8_tables.py pattern); smaller ones use a readable brace-enclosed float literal. The threshold is a
# readability/portability knob, NOT a correctness one -- both bake the identical constants.
_EMBED_THRESHOLD = 256

# Activations that fuse as an inline per-element epilogue (the G2 set: relu exact + the transcendental
# three). softmax is a last-axis row reduction -> a separate final pass, never fused, final layer only.
_FUSIBLE = ("relu", "sigmoid", "tanh", "gelu")


# --- the model / layer representation (a simple, frozen Python carrier) ---------------------------


@dataclass(frozen=True)
class Layer:
    """One DENSE layer of a trained model: a row-major ``out x in`` weight matrix ``W`` (flat,
    ``W[o*in + i]``), an optional length-``out`` bias ``b``, and an ``activation`` kind. The forward pass
    of this layer is ``y[o] = activation( sum_i W[o,i]*x[i] + b[o] )`` over a length-``in`` input ``x``.

    Frozen + all-data, so a model is hashable / content-addressable (the weights ARE the model). ``W`` and
    ``b`` are stored as plain tuples of floats (the trained constants, baked verbatim into the emit)."""

    n_in: int
    n_out: int
    W: tuple[float, ...]  # row-major out x in, W[o*n_in + i]
    b: tuple[float, ...] = ()  # length n_out, or empty for no bias
    activation: str = "relu"

    def __post_init__(self) -> None:
        if self.n_in < 1 or self.n_out < 1:
            raise ValueError(f"layer dims must be >= 1; got n_in={self.n_in} n_out={self.n_out}")
        if len(self.W) != self.n_out * self.n_in:
            raise ValueError(
                f"W has {len(self.W)} elements, expected n_out*n_in = {self.n_out * self.n_in}"
            )
        if self.b and len(self.b) != self.n_out:
            raise ValueError(f"bias has {len(self.b)} elements, expected n_out = {self.n_out}")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.activation!r}; expected one of {ACTIVATIONS}"
            )

    @property
    def has_bias(self) -> bool:
        return bool(self.b)

    @property
    def param_count(self) -> int:
        return len(self.W) + len(self.b)

    def forward(self, x: list[float]) -> list[float]:
        """This layer's reference forward pass over a length-``n_in`` input -- the single source of truth
        the emitted C must reproduce. ``y = activation(W @ x + b)``; softmax reduces over the whole output
        vector (its only legal place is as the final layer)."""
        if len(x) != self.n_in:
            raise ValueError(f"input length {len(x)} != layer n_in {self.n_in}")
        pre = []
        for o in range(self.n_out):
            s = 0.0
            base = o * self.n_in
            for i in range(self.n_in):
                s += self.W[base + i] * x[i]
            if self.b:
                s += self.b[o]
            pre.append(s)
        axis = self.n_out if self.activation == "softmax" else None
        return activation_reference(self.activation, pre, axis)


@dataclass(frozen=True)
class InferenceModel:
    """An ordered chain of dense ``Layer`` s -- a small trained model whose weights are constant and so can
    be BAKED into the emitted kernel. ``quantized_bits`` records whether the weights are Q8-quantized
    (> 0) -- an f32 model is exact, a quantized one carries the R17 bridge bound. ``name`` rides the
    attestation comment."""

    layers: tuple[Layer, ...]
    quantized_bits: int = 0
    name: str = "mlp"
    # the per-tensor quant group size used when quantized (0 = not quantized); informational/attestation.
    quant_group_size: int = 0

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("an inference model needs at least one layer")
        for a, c in zip(self.layers, self.layers[1:]):
            if a.n_out != c.n_in:
                raise ValueError(
                    f"layer chain dim mismatch: a layer outputs {a.n_out} but the next "
                    f"expects {c.n_in}"
                )
        # softmax is the last-axis row reduction -- legal ONLY as the final layer (never an inline
        # epilogue, never mid-chain where its output feeds another matmul as if elementwise).
        for li, layer in enumerate(self.layers):
            if layer.activation == "softmax" and li != len(self.layers) - 1:
                raise ValueError(
                    f"softmax may only be the FINAL layer (it is a row reduction, not a "
                    f"fusible elementwise epilogue); found it at layer {li}"
                )

    @property
    def n_in(self) -> int:
        return self.layers[0].n_in

    @property
    def n_out(self) -> int:
        return self.layers[-1].n_out

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers)

    @property
    def activations(self) -> tuple[str, ...]:
        return tuple(l.activation for l in self.layers)

    @property
    def is_exact(self) -> bool:
        """An f32-baked model is exact (no bridge); a Q8-quantized one carries the R17 bound."""
        return self.quantized_bits == 0

    @property
    def r17_bound(self) -> int:
        """The R17 accuracy bound at the Q8 boundary (ULPs at the realized per-group grid) for a quantized
        model; 0 for an exact f32 model. This is ``precision.quantization_error_bound`` -- the SAME bound
        B5 / the activation bridge certify (the weights are bridged once at bake time; the kernel math is
        then exact integer-or-trusted-libm, so the bridge step is the sole certified error source)."""
        return 0 if self.is_exact else quantization_error_bound()

    def forward(self, x: list[float]) -> list[float]:
        """The end-to-end reference forward pass: each layer in turn (the truth the emitted C reproduces --
        bit-exact for an all-relu f32 model, to float round-off with a transcendental)."""
        h = list(x)
        for layer in self.layers:
            h = layer.forward(h)
        return h


# --- the Q8-quantized path (reuse quantize.py; bake the bridged weights) --------------------------


def quantize_model(model: InferenceModel, group_size: int = 0, bits: int = 8) -> InferenceModel:
    """Return a model whose every layer's W and b are the Q8<->f32<->Q8 BRIDGED (round-tripped) weights --
    i.e. quantized to per-group ``bits``-wide codes and dequantized back to float32. Baking the bridged
    weights makes the emitted kernel's output bounded by the bridge step alone (R17,
    ``quantization_error_bound``), exactly as ``matmul.gemm_via_bridge`` / ``activation_via_bridge`` do:
    the constants the C sees ARE the dequantized values, so the bridge is the only error source and the
    kernel math itself adds none. ``group_size`` 0 means per-tensor (the whole W / b is one group)."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2; got {bits}")
    new_layers = []
    for layer in model.layers:
        gs_w = group_size or len(layer.W)
        wq = tuple(dequantize(quantize_per_group(list(layer.W), gs_w, bits)))
        if layer.b:
            gs_b = group_size or len(layer.b)
            bq = tuple(dequantize(quantize_per_group(list(layer.b), gs_b, bits)))
        else:
            bq = ()
        new_layers.append(Layer(layer.n_in, layer.n_out, wq, bq, layer.activation))
    return InferenceModel(
        tuple(new_layers), quantized_bits=bits, name=model.name, quant_group_size=group_size
    )


# --- baking helpers: a static const float table, literal (small) or #embed (large) ----------------


def _float_blob(values) -> bytes:
    """The values as a little-endian float32 blob (the bytes a ``#embed`` bakes -- the table IS the bytes,
    no codegen-of-a-C-array, no drift; the q8_tables.py discipline)."""
    return b"".join(struct.pack("<f", float(v)) for v in values)


def _fc(v) -> str:
    """A C float32 literal for ``v``, round-trip-exact (``%.9g``) and ALWAYS suffix-safe: a bare integer
    print like ``1`` would make ``1f`` an invalid constant (a float suffix needs a ``.``/``e``), so force a
    ``.0`` when the formatted text has no decimal point or exponent. The value is unchanged."""
    s = f"{float(v):.9g}"
    if not any(ch in s for ch in ".eEnN"):  # no '.', no exponent, not inf/nan -> a bare integer
        s += ".0"
    return s + "f"


def _literal_array(name: str, values) -> str:
    """A readable ``static const float NAME[] = { ... };`` literal for a SMALL baked table. ``%.9g`` is
    the round-trip-exact float32 print width, so the baked constant equals the trained one bit-for-bit."""
    body = ", ".join(_fc(v) for v in values)
    return f"static const float {name}[{len(values)}] = {{{body}}};\n"


def _embed_array(name: str, values, blob_files: dict[str, bytes]) -> str:
    """A LARGE baked table via C23 ``#embed`` (with the ``__has_embed`` fallback, the q8_tables.py
    pattern). The float blob is registered in ``blob_files`` (the caller writes ``<name>.bin`` next to the
    .c so ``#embed`` finds it); the fallback is the byte-identical literal array. The bytes are reinterpreted
    as ``float`` through a union so a strict-aliasing build stays well-defined."""
    blob = _float_blob(values)
    blob_files[f"{name}.bin"] = blob
    fallback_bytes = ",\n  ".join(
        ", ".join(f"0x{byte:02x}" for byte in blob[i : i + 12]) for i in range(0, len(blob), 12)
    )
    n = len(values)
    macro = f"BCIR_INFER_EMBED_{name.upper()}"
    return (
        f"/* {name}: {n} baked weights via C23 #embed (q8_tables.py pattern); "
        f"__has_embed selects the byte-identical fallback on a pre-#embed toolchain. */\n"
        f"#if defined(__has_embed)\n"
        f'#  if __has_embed("{name}.bin")\n'
        f"#    define {macro} 1\n"
        f"#  endif\n"
        f"#endif\n"
        f"#if defined({macro}) && {macro}\n"
        f'static const unsigned char {name}_bytes[] = {{\n#embed "{name}.bin"\n}};\n'
        f"#else\n"
        f"#define {macro} 0\n"
        f"static const unsigned char {name}_bytes[] = {{\n  {fallback_bytes}\n}};\n"
        f"#endif\n"
        f'_Static_assert(sizeof({name}_bytes) == {n} * 4, "baked {name} must be {n} little-endian '
        f'float32");\n'
        f"static inline float {name}(unsigned i) {{\n"
        f"  union {{ unsigned char b[4]; float f; }} u;\n"
        f"  u.b[0] = {name}_bytes[i*4+0]; u.b[1] = {name}_bytes[i*4+1];\n"
        f"  u.b[2] = {name}_bytes[i*4+2]; u.b[3] = {name}_bytes[i*4+3];\n"
        f"  return u.f;\n"
        f"}}\n"
    )


@dataclass
class _EmitContext:
    """Carries the blob files an emit needs written beside the .c (for the ``#embed`` tables). Empty for an
    all-literal (small-model) emit."""

    blob_files: dict[str, bytes] = field(default_factory=dict)


# --- the kernel emitter ---------------------------------------------------------------------------


def emit_inference_kernel_c(
    model: InferenceModel, fn_name: str = "bcir_infer", _ctx: _EmitContext | None = None
) -> str:
    """Emit ONE self-contained C function ``void fn_name(const float *x, float *y)`` that BAKES every
    layer's weights as ``static const`` arrays and computes the forward pass end-to-end, reusing the G2
    fused inline-activation epilogue per layer. ``x`` is the length-``model.n_in`` input; ``y`` is the
    length-``model.n_out`` output. Self-contained: only libm for transcendental activations (link ``-lm``;
    the B1 link rail derives ``-lm`` for such a unit from its ``c.call.libm:`` edges).

    The emitted attestation comment is proof-carrying: it records the layer shapes, activations, parameter
    count, and -- if the model is Q8-quantized -- the R17 bound the baked-bridged weights are certified
    within. ``_ctx`` collects any ``#embed`` blob files the caller must write beside the .c (see
    ``compile_and_run_inference_c``)."""
    ctx = _ctx if _ctx is not None else _EmitContext()
    # which transcendentals appear -> whether we need <math.h> and which libm edges to attest.
    needs_math = any(l.activation in ("sigmoid", "tanh", "gelu", "softmax") for l in model.layers)
    edges = sorted({e for l in model.layers for e in libm_edges(l.activation)})

    # --- the attestation header (proof-carrying / honest scope) ---
    shape_chain = " -> ".join([str(model.n_in)] + [str(l.n_out) for l in model.layers])
    acts = ", ".join(
        f"L{li}:{l.activation}{'+bias' if l.has_bias else ''}" for li, l in enumerate(model.layers)
    )
    quant_note = (
        f"Q8-quantized (bits={model.quantized_bits}, per-{'tensor' if not model.quant_group_size else f'group{model.quant_group_size}'}); "
        f"R17 accuracy bound = {model.r17_bound} ULP at the realized per-group grid "
        f"(quantization_error_bound -- the SOLE certified error; the kernel math is exact-or-"
        f"trusted-libm)"
        if not model.is_exact
        else "f32-baked: EXACT (no Q8 bridge -- bit-exact to the reference for all-relu, float "
        "round-off with a transcendental)"
    )
    libm_note = (
        f" libm edges: {', '.join(edges)} (link -lm)." if edges else " no libm (clean rail)."
    )
    head = (
        f"/* BCIR -> G5 baked-weights inference kernel '{model.name}' (Pillar 5a; once trained the weights\n"
        f" * are CONSTANT -> baked as static const, compiled in, no runtime load). Forward pass:\n"
        f" *   h_{{l+1}} = activation_l(W_l @ h_l + b_l),  reusing the G2 fused inline-activation epilogue\n"
        f" *   (the activation is applied per output element as produced -- no per-layer intermediate).\n"
        f" * shape: {shape_chain};  layers: {acts};  params: {model.param_count}.\n"
        f" * {quant_note}.{libm_note}\n"
        f" * scope: dense layers + relu/sigmoid/tanh/gelu (fused) and softmax (final layer, separate pass);\n"
        f" *        conv/attention/residual/batched are G7. */\n"
        "#include <stddef.h>\n" + ("#include <math.h>\n" if needs_math else "")
    )

    # --- bake every layer's W and b (literal for small; #embed for large) ---
    baked = []
    embedded_flags = []  # per layer: (W_is_embedded, b_is_embedded)
    for li, layer in enumerate(model.layers):
        w_embed = len(layer.W) > _EMBED_THRESHOLD
        if w_embed:
            baked.append(_embed_array(f"W{li}", layer.W, ctx.blob_files))
        else:
            baked.append(_literal_array(f"W{li}", layer.W))
        b_embed = False
        if layer.has_bias:
            b_embed = len(layer.b) > _EMBED_THRESHOLD
            if b_embed:
                baked.append(_embed_array(f"b{li}", layer.b, ctx.blob_files))
            else:
                baked.append(_literal_array(f"b{li}", layer.b))
        embedded_flags.append((w_embed, b_embed))

    # --- the cross-layer scratch buffers (the only buffers between layers; sized to the hidden widths) ---
    # One static buffer per INTERNAL layer boundary: h{li} holds layer li's output, read by layer li+1.
    # The final layer writes `y` directly, so its output needs no buffer -> declare h0 .. h{N-2} only (a
    # net with a single layer declares none). This keeps the emit -Wall/-Wextra clean (no unused buffer).
    scratch = "".join(
        f"  static float h{li}[{model.layers[li].n_out}];\n" for li in range(len(model.layers) - 1)
    )

    body_lines = [f"void {fn_name}(const float *x, float *y) {{"]
    body_lines.append(scratch)
    # source of layer 0 is x; thereafter the previous scratch buffer; the LAST layer writes y directly.
    for li, layer in enumerate(model.layers):
        src = "x" if li == 0 else f"h{li - 1}"
        dst = "y" if li == len(model.layers) - 1 else f"h{li}"
        w_embed, b_embed = embedded_flags[li]
        w_acc = f"W{li}(o * {layer.n_in} + i)" if w_embed else f"W{li}[o * {layer.n_in} + i]"
        b_acc = f"b{li}(o)" if b_embed else f"b{li}[o]"
        bias_add = f"      s += {b_acc};\n" if layer.has_bias else ""
        if layer.activation == "softmax":
            # the final softmax: a separate (non-fused) pass. First the affine pre-activation into dst, then
            # the stable reduce-max -> exp -> normalize over the whole output vector (the last axis).
            body_lines.append(
                f"  /* layer {li}: dense (W{li} @ {src} + b) -> softmax (FINAL, separate non-fused pass). */\n"
                f"  for (size_t o = 0; o < {layer.n_out}; ++o) {{\n"
                f"    float s = 0.0f;\n"
                f"    for (size_t i = 0; i < {layer.n_in}; ++i) s += {w_acc} * {src}[i];\n"
                f"{bias_add}"
                f"    {dst}[o] = s;\n"
                f"  }}\n"
                f"  {{ float m = {dst}[0];\n"
                f"    for (size_t o = 1; o < {layer.n_out}; ++o) if ({dst}[o] > m) m = {dst}[o];\n"
                f"    float sum = 0.0f;\n"
                f"    for (size_t o = 0; o < {layer.n_out}; ++o) {{ {dst}[o] = expf({dst}[o] - m); "
                f"sum += {dst}[o]; }}  /* c.call.libm:expf */\n"
                f"    if (sum == 0.0f) sum = 1.0f;\n"
                f"    for (size_t o = 0; o < {layer.n_out}; ++o) {dst}[o] /= sum; }}\n"
            )
        else:
            # the fused inline-activation epilogue (the G2 building block): act applied to s in-register.
            expr, _ = _inline_activation_expr(layer.activation, "s")
            body_lines.append(
                f"  /* layer {li}: FUSED dense + {layer.activation} (W{li} @ {src}"
                f"{' + b' if layer.has_bias else ''}); activation inline per element. */\n"
                f"  for (size_t o = 0; o < {layer.n_out}; ++o) {{\n"
                f"    float s = 0.0f;\n"
                f"    for (size_t i = 0; i < {layer.n_in}; ++i) s += {w_acc} * {src}[i];\n"
                f"{bias_add}"
                f"    {dst}[o] = {expr};   /* inline activation epilogue -- no separate pass */\n"
                f"  }}\n"
            )
    body_lines.append("}\n")
    return head + "\n" + "".join(baked) + "\n" + "".join(body_lines)


# --- compile + run the emitted kernel against the Python reference (the gate that matters) ---------


def _which(name: str) -> str | None:
    from shutil import which

    return which(name)


def compile_and_run_inference_c(
    model: InferenceModel, x: list[float], fn_name: str = "bcir_infer", workdir: str | None = None
) -> tuple[bool, list[float], str]:
    """Emit the baked-weights kernel + a tiny ``main`` that runs it on ``x`` and prints ``y``, compile
    (writing any ``#embed`` blobs beside the .c), run, and return ``(ok, y_values, combined_output)``.
    The caller compares ``y_values`` to ``model.forward(x)`` (bit-exact for all-relu f32, float round-off
    with a transcendental). Links ``-lm`` (the trusted libm edge) so transcendental activations resolve --
    the B1 link rail derives the same ``-lm`` for a unit with these ``c.call.libm:`` edges."""
    import os
    import subprocess
    import tempfile

    cc = _which("clang") or _which("cc") or _which("gcc")
    if cc is None:
        return False, [], "no C compiler on PATH"
    if len(x) != model.n_in:
        raise ValueError(f"input length {len(x)} != model n_in {model.n_in}")

    ctx = _EmitContext()
    kernel = emit_inference_kernel_c(model, fn_name, _ctx=ctx)
    xinit = ", ".join(_fc(v) for v in x)
    main = (
        f"\n#include <stdio.h>\nint main(void) {{\n"
        f"  const float x[{model.n_in}] = {{{xinit}}};\n"
        f"  float y[{model.n_out}];\n"
        f"  {fn_name}(x, y);\n"
        f'  for (int i = 0; i < {model.n_out}; ++i) printf("%.9g ", y[i]);\n'
        f"  return 0;\n}}\n"
    )

    created = workdir is None
    workdir = workdir or tempfile.mkdtemp(prefix="bcir-infer-")
    try:
        src = os.path.join(workdir, "infer.c")
        exe = os.path.join(workdir, "infer")
        with open(src, "w") as f:
            f.write(kernel + main)
        for fname, blob in ctx.blob_files.items():  # the #embed blobs, beside the .c
            with open(os.path.join(workdir, fname), "wb") as f:
                f.write(blob)
        build = None
        for std in ("-std=c23", "-std=c2x", "-std=c11"):
            build = subprocess.run(
                host_link_args([cc, std, "-O2", "-Wall", "-Wextra", src, "-lm", "-o", exe]),
                capture_output=True,
                text=True,
            )
            if build.returncode == 0:
                break
        if build is None or build.returncode != 0:
            return (
                False,
                [],
                "inference build failed:\n" + (build.stdout + build.stderr if build else ""),
            )
        run = subprocess.run([exe], capture_output=True, text=True)
        if run.returncode != 0:
            return False, [], "inference run failed:\n" + run.stdout + run.stderr
        ys = [float(t) for t in run.stdout.split()]
        return True, ys, run.stdout + run.stderr
    finally:
        if created:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)
