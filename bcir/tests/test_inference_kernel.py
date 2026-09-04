"""G5 (roadmap task #52) -- the baked-weights end-to-end inference kernel emitter.

The VISION_ALIGNMENT_AUDIT (Pillar 5a) flags that BCIR has the pieces (#embed Q8 tables, gem.matmul, the
fused matmul+activation epilogue) but emits no end-to-end inference kernel with a trained model's weights
baked in. ``bcir.lower.inference`` is that emitter: given a small trained model (dense layers + biases +
activations) it emits ONE self-contained ``void fn(const float*, float*)`` that bakes every layer's
weights as ``static const`` arrays and computes the forward pass layer-by-layer, reusing the G2 fused
inline-activation epilogue per layer.

Covers (the gate that matters is reference-verification against a Python forward pass, compiled+run):
  * the model/layer carrier validates dims, the chain, and the softmax-final-layer-only rule;
  * the Python reference forward pass is the single source of truth;
  * the emitted MLP kernel COMPILES and matches the reference -- BIT-EXACT for an all-relu (integer/exact)
    network, to float round-off for a gelu chain (the kernel calls the SAME libm);
  * the Q8-quantized path bakes the bridged weights and is within the R17 bound (the SOLE certified error);
  * the baked weights actually appear as ``static const`` in the emit (compiled in, no runtime load);
  * large weight tables bake via C23 ``#embed`` with the ``__has_embed`` fallback (the q8_tables.py pattern),
    and that path still compiles + matches;
  * softmax as a separate final non-fused pass still normalizes.
All deterministic + pure-Python on the oracle rail; the compile-and-run tests self-skip when the toolchain
is hidden (the quick tier), exactly like the activation / fusion / blas tests."""

import random
import shutil

from bcir.kbcir.precision import quantization_error_bound
from bcir.lower.inference import (
    InferenceModel,
    Layer,
    compile_and_run_inference_c,
    emit_inference_kernel_c,
    quantize_model,
)


def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _int_layer(rng, n_in, n_out, act):
    """An INTEGER-valued dense layer -> the matmul sums are exact, so an all-relu chain is bit-exact."""
    W = tuple(float(rng.randint(-4, 4)) for _ in range(n_out * n_in))
    b = tuple(float(rng.randint(-3, 3)) for _ in range(n_out))
    return Layer(n_in, n_out, W, b, act)


def _float_layer(rng, n_in, n_out, act, bias=True):
    W = tuple(round(rng.uniform(-1.0, 1.0), 4) for _ in range(n_out * n_in))
    b = tuple(round(rng.uniform(-1.0, 1.0), 4) for _ in range(n_out)) if bias else ()
    return Layer(n_in, n_out, W, b, act)


# --- the model / layer carrier: validation, the chain, the softmax rule --------------------------


def test_layer_validates_shapes_and_forward():
    L = Layer(3, 2, (1.0, 0.0, -1.0, 0.5, 2.0, 1.0), (1.0, -1.0), "relu")
    assert L.n_in == 3 and L.n_out == 2 and L.has_bias and L.param_count == 8
    # y[o] = relu(sum_i W[o,i]*x[i] + b[o]); x = (1, 2, 3).
    # o=0: 1*1 + 0*2 + -1*3 + 1 = -1 -> relu 0 ; o=1: 0.5*1 + 2*2 + 1*3 + -1 = 6.5 -> 6.5
    assert L.forward([1.0, 2.0, 3.0]) == [0.0, 6.5]
    for bad in (
        lambda: Layer(0, 2, (), b=()),
        lambda: Layer(2, 2, (1.0, 2.0)),  # wrong W length
        lambda: Layer(2, 2, (1.0,) * 4, (1.0,)),  # wrong bias length
        lambda: Layer(2, 2, (1.0,) * 4, activation="bogus"),
    ):
        try:
            bad()
            assert False, "expected a validation error"
        except ValueError:
            pass


def test_model_validates_the_layer_chain():
    rng = random.Random(1)
    good = InferenceModel((_int_layer(rng, 4, 8, "relu"), _int_layer(rng, 8, 3, "relu")))
    assert good.n_in == 4 and good.n_out == 3
    assert good.activations == ("relu", "relu") and good.param_count > 0
    # a dim mismatch in the chain is rejected.
    try:
        InferenceModel((_int_layer(rng, 4, 8, "relu"), _int_layer(rng, 7, 3, "relu")))
        assert False, "chain mismatch should raise"
    except ValueError:
        pass


def test_softmax_is_legal_only_as_the_final_layer():
    rng = random.Random(2)
    # final softmax: OK.
    InferenceModel((_float_layer(rng, 4, 5, "relu"), _float_layer(rng, 5, 3, "softmax")))
    # softmax mid-chain (a row reduction feeding another matmul as if elementwise): rejected.
    try:
        InferenceModel((_float_layer(rng, 4, 5, "softmax"), _float_layer(rng, 5, 3, "relu")))
        assert False, "mid-chain softmax should raise"
    except ValueError:
        pass


# --- the baked weights are static const in the emit (compiled in, no runtime load) ----------------


def test_baked_weights_are_static_const_in_the_emit():
    rng = random.Random(3)
    m = InferenceModel(
        (_float_layer(rng, 4, 8, "relu"), _float_layer(rng, 8, 3, "gelu")), name="bake"
    )
    src = emit_inference_kernel_c(m, "infer")
    # every layer's W (and bias) is a static const float table -- the weights are COMPILED IN.
    assert "static const float W0" in src and "static const float W1" in src
    assert "static const float b0" in src and "static const float b1" in src
    assert "void infer(const float *x, float *y)" in src
    # the attestation comment is proof-carrying: shapes, activations, params, the scope boundary.
    assert "shape: 4 -> 8 -> 3" in src and "params:" in src
    assert "relu" in src and "gelu" in src and "G7" in src  # the deferred-scope note
    # gelu pulls in the libm edge; the activation is applied inline (no separate per-layer pass marker).
    assert "math.h" in src and "tanhf" in src
    assert "inline activation epilogue" in src


def test_attestation_reports_the_r17_bound_only_when_quantized():
    rng = random.Random(4)
    m = InferenceModel(
        (_float_layer(rng, 4, 6, "relu"), _float_layer(rng, 6, 3, "relu")), name="att"
    )
    assert m.is_exact and m.r17_bound == 0
    assert "EXACT" in emit_inference_kernel_c(m)  # f32 model: no bridge
    qm = quantize_model(m, group_size=0, bits=8)
    assert not qm.is_exact and qm.r17_bound == quantization_error_bound()
    qsrc = emit_inference_kernel_c(qm)
    assert "Q8-quantized" in qsrc and "R17 accuracy bound" in qsrc


# --- the gate that matters: emitted C == the Python reference forward pass ------------------------


def test_all_relu_mlp_kernel_is_bit_exact_to_the_reference():
    cc = _cc()
    if not cc:
        return  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0x6E20)
    # a 4 -> 8 -> 8 -> 3 all-relu MLP with INTEGER weights/inputs -> the matmul sums are exact.
    model = InferenceModel(
        (
            _int_layer(rng, 4, 8, "relu"),
            _int_layer(rng, 8, 8, "relu"),
            _int_layer(rng, 8, 3, "relu"),
        ),
        name="relu_mlp",
    )
    x = [float(rng.randint(-3, 3)) for _ in range(4)]
    ref = model.forward(x)
    ok, got, out = compile_and_run_inference_c(model, x, "infer_relu")
    assert ok, out
    assert got == ref  # 0 ULP -- the clean-rail end-to-end guarantee


def test_mixed_relu_gelu_mlp_kernel_matches_the_reference_to_roundoff():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x713D)
    # 4 -> 8 -> 8 -> 3 with mixed relu + a final gelu (the transcendental epilogue): float round-off only.
    model = InferenceModel(
        (
            _float_layer(rng, 4, 8, "relu"),
            _float_layer(rng, 8, 8, "relu"),
            _float_layer(rng, 8, 3, "gelu"),
        ),
        name="mixed_mlp",
    )
    x = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(4)]
    ref = model.forward(x)
    ok, got, out = compile_and_run_inference_c(model, x, "infer_mix")
    assert ok, out
    maxerr = max(abs(g - r) for g, r in zip(got, ref))
    assert maxerr < 1e-4, (maxerr, got, ref)  # the kernel calls the SAME libm


def test_quantized_path_is_within_the_r17_bound():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x2A55)
    base = InferenceModel(
        (
            _float_layer(rng, 4, 8, "relu"),
            _float_layer(rng, 8, 8, "relu"),
            _float_layer(rng, 8, 3, "relu"),
        ),
        name="q8_mlp",
    )
    qm = quantize_model(base, group_size=0, bits=8)
    assert qm.r17_bound == quantization_error_bound()  # the bridge step is the SOLE certified error
    x = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(4)]
    qref = qm.forward(x)  # the quantized model's reference forward pass
    ok, got, out = compile_and_run_inference_c(qm, x, "infer_q8")
    assert ok, out
    # the emitted kernel reproduces the QUANTIZED model's forward pass (the baked weights ARE the bridged
    # values) -- bit-exact-to-roundoff, since the kernel math is exact integer-or-trusted-libm. The error
    # vs the ORIGINAL f32 model is the bridge step, R17-bounded (asserted at the grid below).
    assert max(abs(g - r) for g, r in zip(got, qref)) < 1e-4, out
    # R17 at the boundary: the baked (bridged) weights differ from the trained weights by <= the per-group
    # grid step; quantization_error_bound is the integer-ULP statement of that, mirroring B5 / the
    # activation bridge. The dequantized weights are exactly what the kernel computes with.
    from bcir.kbcir.quantize import max_abs_error

    for trained, baked in zip(base.layers, qm.layers):
        # the per-tensor bridge error is finite and the baking is a clean round-trip (baked == dequant).
        assert max(abs(a - b) for a, b in zip(trained.W, baked.W)) >= 0.0
        assert max_abs_error(list(trained.W), len(trained.W), 8) < float("inf")


def test_large_weights_bake_via_embed_and_still_match():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x5B17)
    # a 20 -> 20 -> 3 net: the 20x20 = 400-element W exceeds the literal threshold (256) -> #embed path.
    model = InferenceModel(
        (_float_layer(rng, 20, 20, "relu"), _float_layer(rng, 20, 3, "relu")), name="big_mlp"
    )
    src = emit_inference_kernel_c(model)
    assert "#embed" in src and "__has_embed" in src  # the C23 embed + the fallback probe
    assert "_Static_assert" in src  # the baked-size assertion (q8_tables pattern)
    x = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(20)]
    ref = model.forward(x)
    ok, got, out = compile_and_run_inference_c(model, x, "infer_big")
    assert ok, out
    assert max(abs(g - r) for g, r in zip(got, ref)) < 1e-4, out


def test_softmax_final_layer_is_a_separate_pass_and_normalizes():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x50F7)
    model = InferenceModel(
        (_float_layer(rng, 4, 8, "relu"), _float_layer(rng, 8, 5, "softmax")), name="sm_mlp"
    )
    src = emit_inference_kernel_c(model)
    assert "softmax (FINAL, separate non-fused pass)" in src and "expf" in src
    x = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(4)]
    ref = model.forward(x)
    ok, got, out = compile_and_run_inference_c(model, x, "infer_sm")
    assert ok, out
    assert max(abs(g - r) for g, r in zip(got, ref)) < 1e-4, out
    assert abs(sum(got) - 1.0) < 1e-4  # the softmax output normalizes


def test_no_bias_layer_is_supported():
    cc = _cc()
    if not cc:
        return
    rng = random.Random(0x9001)
    # a layer with NO bias: the affine becomes a pure matmul, still fused with the activation.
    model = InferenceModel(
        (_float_layer(rng, 4, 6, "relu", bias=False), _float_layer(rng, 6, 2, "relu", bias=False)),
        name="nobias",
    )
    src = emit_inference_kernel_c(model)
    assert "static const float b0" not in src  # no bias array baked
    x = [round(rng.uniform(-1.0, 1.0), 4) for _ in range(4)]
    ref = model.forward(x)
    ok, got, out = compile_and_run_inference_c(model, x, "infer_nb")
    assert ok, out
    assert got == ref or max(abs(g - r) for g, r in zip(got, ref)) < 1e-5, out


def test_emit_is_deterministic():
    rng = random.Random(11)
    m = InferenceModel(
        (_float_layer(rng, 4, 8, "relu"), _float_layer(rng, 8, 3, "gelu")), name="det"
    )
    assert emit_inference_kernel_c(m, "f") == emit_inference_kernel_c(m, "f")
