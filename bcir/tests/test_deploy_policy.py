"""BCIR Deployment Policy (COMP-1 / P1-P6), LUT4-LANES, and BCIRQ8 v2 gates.

Exercises:
- P1-P6 policy contracts, width gate (`min_lut4_dim=512`), selective P3
  promotion, fail-closed refusal, Q8 bounded-error verification, Dual-SVID +
  Joint-ITQ factorization (both pure-Python stdlib and optional NumPy path),
  and P6 ``bcir.deploy_baseline.v1`` schema validation.
- ``BCIRQ8 v1`` bit-for-bit compatibility and ``BCIRQ8 v2`` container-level
  enforcement (MLP-only ``LUT4_1B``, finite binary16 scales, canonical zero
  padding, and 3-level CRC32) in both Python and C.
- ``LUT4-LANES`` native C kernels (`portable=1`, `AVX2=2`, `AVX-512=4`,
  0 bit drift, column slicing, ``mb4`` micro-batched prefill, f64 & f32).
- End-to-end ``bcir-llama`` (1T vs 3T ``DECODE-THREADS``) and ``bcir-qualify``
  (GEM vs monolithic ``bcir_llama``) bit-for-bit parity on ``BCIRQ8 v2``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from unittest import mock

from bcir.asn1 import streampack as pack_asn1
from bcir.frontends.models.decode import (
    DecoderSpec,
    DecoderWeights,
    LayerWeights,
    next_token_logits,
    reference_decode,
)
from bcir.frontends.models.decoder_gem import run_decoder_pack
from bcir.frontends.models.decoder_program import decoder_program
from bcir.tests.decoder_fixtures import planned_pack
from bcir.frontends.models.deploy_policy import (
    CONTRACT_BCIRQ8_G32,
    CONTRACT_COMP1_LUT4_1B_1P,
    CONTRACT_COMP1_LUT4_1B_RES,
    CONTRACT_FP32_REF,
    DEPLOY_BASELINE_SCHEMA,
    DeployBaselineRecord,
    DeploymentPolicy,
    EvalMetrics,
    PrecisionContract,
    Q8BoundReport,
    export_hybrid_decoder,
    factorize_lut4_matrix,
    load_baseline_record,
    qualify_deployment,
    save_baseline_record,
    validate_baseline_record,
    verify_q8_error_bound,
)
from bcir.frontends.models.weights_io import (
    DIRECTORY_ENTRY_SIZE,
    FLAG_HYBRID_LUT4_MLP,
    HEADER_SIZE,
    TENSOR_EMBEDDING,
    TENSOR_ENC_LUT4_1B,
    TENSOR_G_ATTN,
    TENSOR_LM_HEAD,
    TENSOR_W_DOWN,
    TENSOR_W_GATE,
    TENSOR_W_K,
    TENSOR_W_O,
    TENSOR_W_Q,
    TENSOR_W_UP,
    TENSOR_W_V,
    VERSION,
    VERSION_V2,
    Lut4Path,
    Lut4Tensor,
    encode_fp16_le,
    lut4_matvec_reference,
    lut4_packed_bytes,
    pack_lut4_signs,
    read_q8_decoder,
    read_q8_decoder_v2,
    unpack_lut4_signs,
    write_q8_decoder,
)
from bcir.kbcir.native_ai import NativeAIKernels
from bcir.kbcir.unsupervised import EmbeddingTable
from bcir.abi import encode
from bcir.toolchain import build_units, host_c_compiler, host_link_args

_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME = _ROOT / "runtime" / "c"
_CC = host_c_compiler()

_HASHES = {
    "model": "11" * 32,
    "config": "22" * 32,
    "tokenizer": "33" * 32,
}
_TOK_IDS = {"bos": 1, "eos": 2, "pad": 0, "context_length": 64}


def _synth_values(count: int, seed: int) -> tuple[float, ...]:
    return tuple(
        round(
            0.28 * math.sin((i + 1) * 0.37 + seed * 1.13) + 0.09 * math.cos((i + 3) * 0.19 - seed),
            6,
        )
        for i in range(count)
    )


def _synth_decoder(
    *,
    vocab_size: int = 24,
    d_model: int = 32,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    n_layers: int = 2,
    d_ff: int = 48,
    tied: bool = False,
) -> tuple[DecoderSpec, DecoderWeights]:
    spec = DecoderSpec(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        rope_base=10000.0,
        activation="silu_gate",
        n_kv_heads=n_kv_heads,
        tied_embeddings=tied,
    )
    kvd = spec.kv_dim
    emb = EmbeddingTable(
        table=_synth_values(vocab_size * d_model, 1), n_vocab=vocab_size, dim=d_model
    )
    layers = []
    for lyr in range(n_layers):
        s = 10 + 17 * lyr
        layers.append(
            LayerWeights(
                g_attn=tuple(1.0 + 0.05 * math.sin(i + s) for i in range(d_model)),
                w_q=_synth_values(d_model * d_model, s + 1),
                w_k=_synth_values(d_model * kvd, s + 2),
                w_v=_synth_values(d_model * kvd, s + 3),
                w_o=_synth_values(d_model * d_model, s + 4),
                g_ff=tuple(1.0 + 0.05 * math.cos(i + s) for i in range(d_model)),
                w1=_synth_values(d_model * d_ff, s + 5),
                b1=(),
                w2=_synth_values(d_ff * d_model, s + 6),
                b2=(),
                w_gate=_synth_values(d_model * d_ff, s + 7),
            )
        )
    g_final = tuple(1.0 + 0.03 * math.sin(i + 99) for i in range(d_model))
    lm_head = () if tied else _synth_values(vocab_size * d_model, 101)
    weights = DecoderWeights(embedding=emb, layers=tuple(layers), g_final=g_final, lm_head=lm_head)
    return spec, weights


def _repair_bcirq8_crcs(buf: bytearray) -> bytes:
    """Recompute tensor, body, and header CRCs on a mutated BCIRQ8 buffer."""
    tensor_count = struct.unpack_from("<I", buf, 80)[0]
    dir_off = struct.unpack_from("<Q", buf, 88)[0]
    for idx in range(tensor_count):
        eoff = dir_off + idx * DIRECTORY_ENTRY_SIZE
        eflags = buf[eoff + 5]
        reserved = struct.unpack_from("<H", buf, eoff + 6)[0]
        dim0, dim1, elems, groups = struct.unpack_from("<IIII", buf, eoff + 8)
        exp_off, code_off = struct.unpack_from("<QQ", buf, eoff + 32)
        if eflags == TENSOR_ENC_LUT4_1B:
            r0 = groups & 0xFFFF
            r1 = (groups >> 16) & 0xFFFF
            ranks = [r0] if reserved == 1 else [r0, r1]
            exp_len = sum(2 * (dim0 + r + dim1) for r in ranks)
            code_len = sum(lut4_packed_bytes(r, dim0) + lut4_packed_bytes(dim1, r) for r in ranks)
        else:
            exp_len = groups * 2
            code_len = elems
        crc = zlib.crc32(buf[exp_off : exp_off + exp_len])
        crc = zlib.crc32(buf[code_off : code_off + code_len], crc) & 0xFFFFFFFF
        struct.pack_into("<I", buf, eoff + 24, crc)
    body_crc = zlib.crc32(buf[dir_off:]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 112, body_crc)
    struct.pack_into("<I", buf, 116, 0)
    hdr_crc = zlib.crc32(buf[:HEADER_SIZE]) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 116, hdr_crc)
    return bytes(buf)


def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} to be raised")


def _build_llama_cli(directory: Path) -> Path:
    assert _CC is not None
    exe = directory / ("bcir-llama.exe" if os.name == "nt" else "bcir-llama")
    sources = (
        "bcir_q8_model.c",
        "bcir_decode.c",
        "bcir_ai_kernels.c",
        "bcir_llama.c",
        "bcir_llama_cli.c",
        "bcir_tensor.c",
    )
    cmd = [
        _CC,
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        *(["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else []),
        "-I",
        str(_RUNTIME),
        *(str(_RUNTIME / s) for s in sources),
        *([] if sys.platform == "win32" else ["-pthread"]),
        "-o",
        str(exe),
        *host_link_args(["-lm"]),
    ]
    build_units(cmd, "bcir-llama v2 build", timeout=600)
    return exe


def _build_qualify_cli(directory: Path) -> Path:
    assert _CC is not None
    exe = directory / ("bcir-qualify.exe" if os.name == "nt" else "bcir-qualify")
    sources = (
        "bcir_qualify_cli.c",
        "bcir_decoder_gem.c",
        "bcir_llama.c",
        "bcir_q8_model.c",
        "bcir_q4_kernel.c",
        "bcir_ai_kernels.c",
        "bcir_decode.c",
        "bcir_exec.c",
        "bcir_runtime.c",
        "bcir_asn1_streampack.c",
        "bcir_asn1.c",
        "bcir_sha256.c",
    )
    cmd = [
        _CC,
        "-std=c11",
        "-O2",
        "-ffp-contract=off",
        "-Wall",
        "-Wextra",
        "-Werror",
        *(["-D_CRT_SECURE_NO_WARNINGS"] if os.name == "nt" else []),
        "-I",
        str(_RUNTIME),
        *(str(_RUNTIME / s) for s in sources),
        "-o",
        str(exe),
        *host_link_args(["-lm"]),
    ]
    build_units(cmd, "bcir-qualify v2 build", timeout=600)
    return exe


def test_p1_precision_contract_validation():
    for contract in (
        CONTRACT_FP32_REF,
        CONTRACT_BCIRQ8_G32,
        CONTRACT_COMP1_LUT4_1B_RES,
        CONTRACT_COMP1_LUT4_1B_1P,
    ):
        rt = PrecisionContract.from_dict(contract.to_dict())
        assert rt == contract

    # Attention must NEVER be 1-bit LUT4
    _assert_raises(
        ValueError,
        PrecisionContract,
        name="bad_attn",
        weight_store="bcirq8_v2_hybrid_lut4_mlp",
        attention_weight_format="lut4_1b",
        mlp_weight_format="lut4_1b_2p",
        scale_dtype="pow2_i16+fp16_le",
        lut4_paths=2,
        lut4_target_bpw=1.0,
    )
    # Hybrid store requires lut4_paths in {1, 2}
    _assert_raises(
        ValueError,
        PrecisionContract,
        name="bad_paths",
        weight_store="bcirq8_v2_hybrid_lut4_mlp",
        attention_weight_format="q8_g32",
        mlp_weight_format="lut4_1b_2p",
        scale_dtype="pow2_i16+fp16_le",
        lut4_paths=0,
    )


def test_p2_policy_width_gate_and_rank_solver():
    policy = DeploymentPolicy(min_lut4_dim=512, target_mlp_bpw=1.0, default_lut4_paths=2)
    narrow = DecoderSpec(vocab_size=256, d_model=128, n_heads=4, n_layers=2, d_ff=384, n_kv_heads=2)
    wide = DecoderSpec(vocab_size=256, d_model=512, n_heads=8, n_layers=2, d_ff=1536, n_kv_heads=2)

    ok_n, reason_n = policy.admits_lut4(narrow)
    assert not ok_n and "width gate" in reason_n

    ok_w, _ = policy.admits_lut4(wide)
    assert ok_w
    for allowed_tid in (TENSOR_W_GATE, TENSOR_W_UP, TENSOR_W_DOWN):
        assert policy.admits_lut4(wide, allowed_tid)[0]
    for protected_tid in (
        TENSOR_W_Q,
        TENSOR_W_K,
        TENSOR_W_V,
        TENSOR_W_O,
        TENSOR_EMBEDDING,
        TENSOR_G_ATTN,
        TENSOR_LM_HEAD,
    ):
        assert not policy.admits_lut4(wide, protected_tid)[0]

    for din, dout in ((512, 1536), (1536, 512), (768, 2048), (2048, 768)):
        r2 = policy.solve_rank(din, dout, paths=2)
        assert len(r2) == 2 and r2[0] >= 1 and r2[1] >= 1
        bpw2 = policy.effective_mlp_bpw(din, dout, r2)
        assert 0.5 < bpw2 <= 1.10, (din, dout, r2, bpw2)
        r1 = policy.solve_rank(din, dout, paths=1)
        assert len(r1) == 1 and r1[0] >= 1
        bpw1 = policy.effective_mlp_bpw(din, dout, r1)
        assert 0.5 < bpw1 <= 1.10, (din, dout, r1, bpw1)


def test_p3_p4_qualification_and_fail_closed():
    policy = DeploymentPolicy(
        min_lut4_dim=512,
        max_q8_ce_regression=0.02,
        max_hybrid_ce_regression=0.20,
        max_recall_drop=0.0,
        allow_p3_promotion=True,
    )
    wide = DecoderSpec(vocab_size=256, d_model=512, n_heads=8, n_layers=4, d_ff=1536, n_kv_heads=2)
    narrow = DecoderSpec(vocab_size=256, d_model=128, n_heads=4, n_layers=2, d_ff=384, n_kv_heads=2)
    fp32 = EvalMetrics(ce=1.50, ppl=math.exp(1.50), exact_recall=1.0)
    q8_ok = EvalMetrics(ce=1.505, ppl=math.exp(1.505), exact_recall=1.0)
    hyb_ok = EvalMetrics(ce=1.62, ppl=math.exp(1.62), exact_recall=1.0)
    hyb_bad = EvalMetrics(ce=1.85, ppl=math.exp(1.85), exact_recall=1.0)
    bound_ok = Q8BoundReport(True, 12, 6, 1000, 0, 1e-4, 0.99, "id=11:layer=0")
    bound_bad = Q8BoundReport(False, 12, 6, 1000, 3, 0.5, 2.1, "id=11:layer=0")

    # 1. Both pass -> ADMITTED_HYBRID_V2
    v = qualify_deployment(
        wide, policy, fp32_metrics=fp32, q8_metrics=q8_ok, hybrid_metrics=hyb_ok, q8_bound=bound_ok
    )
    assert v.status == "ADMITTED_HYBRID_V2" and v.contract == CONTRACT_COMP1_LUT4_1B_RES

    # 2. Narrow width gate -> PROMOTED_Q8_V1
    v_narrow = qualify_deployment(
        narrow,
        policy,
        fp32_metrics=fp32,
        q8_metrics=q8_ok,
        hybrid_metrics=hyb_ok,
        q8_bound=bound_ok,
    )
    assert v_narrow.status == "PROMOTED_Q8_V1" and v_narrow.contract == CONTRACT_BCIRQ8_G32

    # 3. Hybrid fails -> P3 selective layer promotion rescues it
    v_sel = qualify_deployment(
        wide,
        policy,
        fp32_metrics=fp32,
        q8_metrics=q8_ok,
        hybrid_metrics=hyb_bad,
        q8_bound=bound_ok,
        per_layer_hybrid_ce_delta={0: 0.22, 1: 0.05, 2: 0.04, 3: 0.04},
    )
    assert v_sel.status == "PROMOTED_SELECTIVE_V2" and v_sel.promoted_layers == (0,)

    # 4. Hybrid fails without sufficient selective rescue -> PROMOTED_Q8_V1
    v_full = qualify_deployment(
        wide, policy, fp32_metrics=fp32, q8_metrics=q8_ok, hybrid_metrics=hyb_bad, q8_bound=bound_ok
    )
    assert v_full.status == "PROMOTED_Q8_V1" and v_full.promoted_layers == (0, 1, 2, 3)

    # 5. allow_p3_promotion=False -> REFUSED on width gate or hybrid failure
    strict = DeploymentPolicy(min_lut4_dim=512, allow_p3_promotion=False)
    assert (
        qualify_deployment(
            narrow,
            strict,
            fp32_metrics=fp32,
            q8_metrics=q8_ok,
            hybrid_metrics=hyb_ok,
            q8_bound=bound_ok,
        ).status
        == "REFUSED"
    )
    assert (
        qualify_deployment(
            wide,
            strict,
            fp32_metrics=fp32,
            q8_metrics=q8_ok,
            hybrid_metrics=hyb_bad,
            q8_bound=bound_ok,
        ).status
        == "REFUSED"
    )

    # 6. Hard fail-closed gates (NEVER bypassed by promotion)
    assert (
        qualify_deployment(
            wide,
            policy,
            fp32_metrics=fp32,
            q8_metrics=q8_ok,
            hybrid_metrics=hyb_ok,
            q8_bound=bound_bad,
        ).status
        == "REFUSED"
    )
    assert (
        qualify_deployment(
            wide,
            policy,
            fp32_metrics=fp32,
            q8_metrics=q8_ok,
            hybrid_metrics=hyb_ok,
            q8_bound=bound_ok,
            lane_max_bit_drift=1,
        ).status
        == "REFUSED"
    )
    q8_bad = EvalMetrics(ce=1.60, exact_recall=1.0)
    assert (
        qualify_deployment(
            wide,
            policy,
            fp32_metrics=fp32,
            q8_metrics=q8_bad,
            hybrid_metrics=hyb_ok,
            q8_bound=bound_ok,
        ).status
        == "REFUSED"
    )
    q8_recall_drop = EvalMetrics(ce=1.501, exact_recall=0.95)
    assert (
        qualify_deployment(
            wide,
            policy,
            fp32_metrics=fp32,
            q8_metrics=q8_recall_drop,
            hybrid_metrics=hyb_ok,
            q8_bound=bound_ok,
        ).status
        == "REFUSED"
    )


def test_p5_q8_error_bound_and_dual_svid_itq_factorization():
    spec, weights = _synth_decoder(
        vocab_size=24, d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=48
    )
    policy_wide = DeploymentPolicy(
        min_lut4_dim=16, target_mlp_bpw=2.0, default_lut4_paths=2, itq_iters=10, svid_iters=8
    )
    policy_narrow = DeploymentPolicy(min_lut4_dim=512, target_mlp_bpw=1.0)

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        p_v1 = d / "model_v1.bcirq8"
        p_v2 = d / "model_v2.bcirq8"
        meta1, luts1, rep1 = export_hybrid_decoder(
            p_v1, spec, weights, policy=policy_narrow, source_hashes=_HASHES, tokenizer_ids=_TOK_IDS
        )
        assert (
            meta1.version == VERSION
            and meta1.lut4_tensor_count == 0
            and rep1.promoted_to_v1
            and not luts1
        )
        bound1 = verify_q8_error_bound(spec, weights, p_v1)
        assert bound1.passed and bound1.violation_count == 0 and bound1.lut4_tensors_skipped == 0
        assert bound1.max_normalized_error <= 1.0 + 1e-12

        meta2, luts2, rep2 = export_hybrid_decoder(
            p_v2, spec, weights, policy=policy_wide, source_hashes=_HASHES, tokenizer_ids=_TOK_IDS
        )
        assert (
            meta2.version == VERSION_V2 and meta2.lut4_tensor_count == 6 and not rep2.promoted_to_v1
        )
        assert len(luts2) == 6 and meta2.file_size < meta1.file_size
        bound2 = verify_q8_error_bound(spec, weights, p_v2)
        assert bound2.passed and bound2.violation_count == 0 and bound2.lut4_tensors_skipped == 6
        assert bound2.max_normalized_error <= 1.0 + 1e-12

        # Perturbing a reference weight beyond 2^(e-1) must fail the Q8 bound check
        bad_layers = list(weights.layers)
        bad_wq = tuple(v + 5.0 if i == 0 else v for i, v in enumerate(bad_layers[0].w_q))
        bad_layers[0] = LayerWeights(
            g_attn=bad_layers[0].g_attn,
            w_q=bad_wq,
            w_k=bad_layers[0].w_k,
            w_v=bad_layers[0].w_v,
            w_o=bad_layers[0].w_o,
            g_ff=bad_layers[0].g_ff,
            w1=bad_layers[0].w1,
            b1=(),
            w2=bad_layers[0].w2,
            b2=(),
            w_gate=bad_layers[0].w_gate,
        )
        bad_weights = DecoderWeights(
            embedding=weights.embedding,
            layers=tuple(bad_layers),
            g_final=weights.g_final,
            lm_head=weights.lm_head,
        )
        bound_bad = verify_q8_error_bound(spec, bad_weights, p_v2)
        assert not bound_bad.passed and bound_bad.violation_count >= 1

    # Verify pure-Python stdlib factorization path explicitly (with numpy hidden)
    small_w = _synth_values(8 * 12, 77)
    with mock.patch.dict(sys.modules, {"numpy": None}):
        lt_1p, err_1p = factorize_lut4_matrix(
            small_w,
            tensor_id=TENSOR_W_GATE,
            layer=0,
            in_dim=8,
            out_dim=12,
            ranks=(3,),
            itq_iters=5,
            svid_iters=4,
        )
        lt_2p, err_2p = factorize_lut4_matrix(
            small_w,
            tensor_id=TENSOR_W_GATE,
            layer=0,
            in_dim=8,
            out_dim=12,
            ranks=(3, 2),
            itq_iters=5,
            svid_iters=4,
        )
    assert len(lt_1p.paths) == 1 and len(lt_2p.paths) == 2
    assert err_2p <= err_1p + 1e-6


def test_p6_baseline_record_schema_and_validation():
    policy = DeploymentPolicy()
    verdict = qualify_deployment(
        DecoderSpec(vocab_size=256, d_model=512, n_heads=8, n_layers=2, d_ff=1536, n_kv_heads=2),
        policy,
        fp32_metrics=EvalMetrics(ce=1.40),
        q8_metrics=EvalMetrics(ce=1.402),
        hybrid_metrics=EvalMetrics(ce=1.50),
        q8_bound=Q8BoundReport(True, 10, 6, 100, 0, 1e-4, 0.9, "id=11:layer=0"),
    ).to_dict()
    assert verdict["status"] == "ADMITTED_HYBRID_V2"
    record = DeployBaselineRecord(
        created_utc="2026-11-24T00:00:00Z",
        bcir_git_sha="deadbeef",
        host={"cpu": "test", "isa": "avx2"},
        policy=policy,
        contracts={
            "fp32": CONTRACT_FP32_REF,
            "q8": CONTRACT_BCIRQ8_G32,
            "comp1": CONTRACT_COMP1_LUT4_1B_RES,
        },
        models=({"name": "wide512", "verdict": dict(verdict)},),
    ).to_dict()

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "baseline.json"
        save_baseline_record(p, record)
        loaded = load_baseline_record(p)
        assert (
            loaded["schema"] == DEPLOY_BASELINE_SCHEMA and loaded["models"][0]["name"] == "wide512"
        )

        # Duplicate JSON key rejected
        dup_path = Path(tmp) / "dup.json"
        dup_path.write_text(
            p.read_text(encoding="utf-8").replace('"schema":', '"schema": "x", "schema":', 1),
            encoding="utf-8",
        )
        _assert_raises(ValueError, load_baseline_record, dup_path)

        # NaN constant rejected
        nan_path = Path(tmp) / "nan.json"
        nan_text = p.read_text(encoding="utf-8").replace(": 0.02,", ": NaN,", 1)
        assert ": NaN," in nan_text
        nan_path.write_text(nan_text, encoding="utf-8")
        _assert_raises(ValueError, load_baseline_record, nan_path)

    # Admitted model with lane_zero_drift=False rejected
    bad_rec = json.loads(json.dumps(record))
    bad_rec["models"][0]["verdict"]["lane_zero_drift"] = False
    _assert_raises(ValueError, validate_baseline_record, bad_rec)


def test_bcirq8_v2_container_enforces_mlp_only_and_v1_preserved():
    spec, weights = _synth_decoder(
        vocab_size=20, d_model=16, n_heads=2, n_kv_heads=1, n_layers=1, d_ff=24
    )
    policy = DeploymentPolicy(
        min_lut4_dim=8, target_mlp_bpw=2.0, default_lut4_paths=2, itq_iters=6, svid_iters=4
    )

    # Protected tensor ids refused at Lut4Tensor constructor level
    v_p = pack_lut4_signs([1] * (4 * 16), 4, 16)
    u_p = pack_lut4_signs([1] * (16 * 4), 16, 4)
    lp = Lut4Path(
        4,
        v_p,
        u_p,
        encode_fp16_le([0.1] * 16),
        encode_fp16_le([0.2] * 4),
        encode_fp16_le([0.3] * 16),
    )
    for protected_tid in (
        TENSOR_W_Q,
        TENSOR_W_K,
        TENSOR_W_V,
        TENSOR_W_O,
        TENSOR_EMBEDDING,
        TENSOR_LM_HEAD,
        TENSOR_G_ATTN,
    ):
        _assert_raises(ValueError, Lut4Tensor, protected_tid, 0, 16, 16, (lp,))

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        p_v1 = d / "v1.bcirq8"
        p_v2 = d / "v2.bcirq8"
        meta1 = write_q8_decoder(p_v1, spec, weights, source_hashes=_HASHES, tokenizer_ids=_TOK_IDS)
        assert meta1.version == VERSION and meta1.lut4_tensor_count == 0
        s1, w1, luts1, m1 = read_q8_decoder_v2(p_v1)
        assert (
            s1 == spec
            and not luts1
            and m1.version == VERSION
            and w1.g_final == read_q8_decoder(p_v1)[1].g_final
        )

        meta2, luts2, _ = export_hybrid_decoder(
            p_v2, spec, weights, policy=policy, source_hashes=_HASHES, tokenizer_ids=_TOK_IDS
        )
        assert meta2.version == VERSION_V2 and meta2.lut4_tensor_count == 3
        s2, _w2, luts2_rd, m2 = read_q8_decoder_v2(p_v2)
        assert s2 == spec and m2.version == VERSION_V2 and luts2_rd == luts2

        # Raw binary tampering tests (with CRCs repaired to hit semantic gates!)
        raw_v2 = p_v2.read_bytes()
        # (a) v2 without FLAG_HYBRID_LUT4_MLP
        bad_flag = bytearray(raw_v2)
        cur_flags = struct.unpack_from("<I", bad_flag, 16)[0]
        struct.pack_into("<I", bad_flag, 16, cur_flags & ~FLAG_HYBRID_LUT4_MLP)
        p_bad_flag = d / "bad_flag.bcirq8"
        p_bad_flag.write_bytes(_repair_bcirq8_crcs(bad_flag))
        _assert_raises(ValueError, read_q8_decoder_v2, p_bad_flag)

        # (b) v2 with 0 LUT4 tensors (upgraded v1 header with FLAG_HYBRID_LUT4_MLP set)
        bad_zero = bytearray(p_v1.read_bytes())
        struct.pack_into("<H", bad_zero, 8, VERSION_V2)
        struct.pack_into(
            "<I", bad_zero, 16, struct.unpack_from("<I", bad_zero, 16)[0] | FLAG_HYBRID_LUT4_MLP
        )
        p_bad_zero = d / "bad_zero.bcirq8"
        p_bad_zero.write_bytes(_repair_bcirq8_crcs(bad_zero))
        _assert_raises(ValueError, read_q8_decoder_v2, p_bad_zero)

        # (c) NaN binary16 scale inside a LUT4 tensor
        dir_off = struct.unpack_from("<Q", raw_v2, 88)[0]
        # Entry 7 is W_GATE (0:EMB, 1:G_ATTN, 2:WQ, 3:WK, 4:WV, 5:WO, 6:G_FF, 7:W_GATE)
        gate_eoff = dir_off + 7 * DIRECTORY_ENTRY_SIZE
        assert struct.unpack_from("<H", raw_v2, gate_eoff)[0] == TENSOR_W_GATE
        gate_scales_off = struct.unpack_from("<Q", raw_v2, gate_eoff + 32)[0]
        gate_packed_off = struct.unpack_from("<Q", raw_v2, gate_eoff + 40)[0]

        bad_nan = bytearray(raw_v2)
        struct.pack_into("<H", bad_nan, gate_scales_off, 0x7E00)  # binary16 quiet NaN
        p_bad_nan = d / "bad_nan.bcirq8"
        p_bad_nan.write_bytes(_repair_bcirq8_crcs(bad_nan))
        _assert_raises(ValueError, read_q8_decoder_v2, p_bad_nan)

        # (d) Nonzero padding row bit in packed LUT4 matrix (r < 32 so high nibble has padding rows!)
        bad_pad = bytearray(raw_v2)
        bad_pad[gate_packed_off + 15] |= 0xF0  # row 31 is padding since rank < 32
        p_bad_pad = d / "bad_pad.bcirq8"
        p_bad_pad.write_bytes(_repair_bcirq8_crcs(bad_pad))
        _assert_raises(ValueError, read_q8_decoder_v2, p_bad_pad)

        # (e) Tampered Attention W_Q entry carrying LUT4_1B flag!
        wq_eoff = dir_off + 2 * DIRECTORY_ENTRY_SIZE
        assert struct.unpack_from("<H", raw_v2, wq_eoff)[0] == TENSOR_W_Q
        bad_attn = bytearray(raw_v2)
        bad_attn[wq_eoff + 5] = TENSOR_ENC_LUT4_1B
        struct.pack_into("<H", bad_attn, wq_eoff + 6, 1)
        struct.pack_into("<I", bad_attn, wq_eoff + 20, 4)  # r0=4, r1=0
        p_bad_attn = d / "bad_attn.bcirq8"
        p_bad_attn.write_bytes(_repair_bcirq8_crcs(bad_attn))
        _assert_raises(ValueError, read_q8_decoder_v2, p_bad_attn)

        # Verify the C loader ALSO rejects every single one of those tampered files!
        if _CC is not None:
            llama_exe = _build_llama_cli(d)
            for bad_path in (p_bad_flag, p_bad_zero, p_bad_nan, p_bad_pad, p_bad_attn):
                r = subprocess.run(
                    [
                        str(llama_exe),
                        "--model",
                        str(bad_path),
                        "--prompt-ids",
                        "1,2",
                        "--max-new",
                        "1",
                    ],
                    capture_output=True,
                    text=True,
                )
                assert r.returncode != 0, f"C loader unexpectedly accepted {bad_path.name}"


def test_native_lut4_lanes_parity_columns_and_mb4():
    if _CC is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        with NativeAIKernels.build(tmp, cc=_CC) as kernels:
            mask = kernels.lut4_instances()
            assert mask & 1
            admitted = [inst for inst in (1, 2, 4) if mask & inst]

            for in_dim, out_dim, ranks in ((17, 29, (5,)), (64, 96, (12, 4)), (131, 73, (19, 7))):
                paths = []
                for idx, r in enumerate(ranks):
                    v_signs = [1 if ((i * 7 + idx * 13) % 3) else -1 for i in range(r * in_dim)]
                    u_signs = [-1 if ((i * 5 + idx * 11) % 2) else 1 for i in range(out_dim * r)]
                    v_py = pack_lut4_signs(v_signs, r, in_dim)
                    u_py = pack_lut4_signs(u_signs, out_dim, r)
                    assert kernels.lut4_pack_signs(v_signs, rows=r, cols=in_dim) == v_py
                    assert kernels.lut4_pack_signs(u_signs, rows=out_dim, cols=r) == u_py
                    assert kernels.lut4_validate_packed(v_py, rows=r, cols=in_dim)
                    assert unpack_lut4_signs(v_py, r, in_dim) == tuple(v_signs)
                    paths.append(
                        Lut4Path(
                            r,
                            v_py,
                            u_py,
                            encode_fp16_le([0.12 * math.sin(i + idx + 1) for i in range(in_dim)]),
                            encode_fp16_le([0.25 * math.cos(i + idx + 2) for i in range(r)]),
                            encode_fp16_le(
                                [0.18 * math.sin(i * 0.3 + idx) for i in range(out_dim)]
                            ),
                        )
                    )
                paths_t = tuple(paths)
                x = tuple(0.31 * math.sin(i * 0.41 + 0.2) for i in range(in_dim))
                ref = lut4_matvec_reference(x, in_dim, out_dim, paths_t)
                ref_bits = [struct.pack("<d", v) for v in ref]

                # 0-bit-drift across all admitted instances (0, 1, 2, 4) and Python oracle
                for inst in (None, 0, *admitted):
                    got = kernels.lut4_matvec(x, paths_t, output_count=out_dim, instance=inst)
                    assert [struct.pack("<d", v) for v in got] == ref_bits, (in_dim, out_dim, inst)

                # Column slicing parity
                c0, c1 = out_dim // 5, (4 * out_dim) // 5
                for inst in (None, 0, *admitted):
                    got_col = kernels.lut4_matvec(
                        x, paths_t, output_count=out_dim, instance=inst, columns=(c0, c1)
                    )
                    assert [struct.pack("<d", v) for v in got_col] == ref_bits[c0:c1]

                # mb4 micro-batch prefill parity (7 tokens: 4 + 3 tail)
                batch = tuple(
                    tuple(0.25 * math.cos((t + 1) * (i + 1) * 0.13) for i in range(in_dim))
                    for t in range(7)
                )
                ref_batch = [
                    [
                        struct.pack("<d", v)
                        for v in lut4_matvec_reference(row, in_dim, out_dim, paths_t)
                    ]
                    for row in batch
                ]
                for inst in (0, *admitted):
                    got_mb4 = kernels.lut4_matmul_mb4(
                        batch, paths_t, output_count=out_dim, instance=inst
                    )
                    for t in range(7):
                        assert [struct.pack("<d", v) for v in got_mb4[t]] == ref_batch[t]

                # FP32 LUT4-LANES 0-bit-drift across portable/AVX2/AVX-512 & mb4
                f32_p = [
                    struct.pack("<f", v)
                    for v in kernels.lut4_matvec_f32(x, paths_t, output_count=out_dim, instance=1)
                ]
                for inst in (0, *admitted):
                    got_f32 = [
                        struct.pack("<f", v)
                        for v in kernels.lut4_matvec_f32(
                            x, paths_t, output_count=out_dim, instance=inst
                        )
                    ]
                    assert got_f32 == f32_p
                    got_f32_col = [
                        struct.pack("<f", v)
                        for v in kernels.lut4_matvec_f32(
                            x, paths_t, output_count=out_dim, instance=inst, columns=(c0, c1)
                        )
                    ]
                    assert got_f32_col == f32_p[c0:c1]
                mb4_f32_p = kernels.lut4_matmul_mb4_f32(
                    batch, paths_t, output_count=out_dim, instance=1
                )
                for inst in (0, *admitted):
                    mb4_f32 = kernels.lut4_matmul_mb4_f32(
                        batch, paths_t, output_count=out_dim, instance=inst
                    )
                    for t in range(7):
                        assert [struct.pack("<f", v) for v in mb4_f32[t]] == [
                            struct.pack("<f", v) for v in mb4_f32_p[t]
                        ]


def test_bcirq8_v2_c_llama_threads_and_gem_qualify_parity():
    if _CC is None:
        return
    prompt = [1, 3, 5]
    max_new = 3
    policy = DeploymentPolicy(
        min_lut4_dim=16, target_mlp_bpw=2.0, default_lut4_paths=2, itq_iters=8, svid_iters=6
    )

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        llama_exe = _build_llama_cli(d)
        qual_exe = _build_qualify_cli(d)

        for tied, promoted in ((False, ()), (True, ()), (False, (0,))):
            spec, weights = _synth_decoder(
                vocab_size=24, d_model=32, n_heads=4, n_kv_heads=2, n_layers=2, d_ff=48, tied=tied
            )
            model_path = d / f"hybrid-tied{int(tied)}-prom{len(promoted)}.bcirq8"
            meta, luts, rep = export_hybrid_decoder(
                model_path,
                spec,
                weights,
                policy=policy,
                source_hashes=_HASHES,
                tokenizer_ids=_TOK_IDS,
                promoted_layers=promoted,
            )
            assert meta.version == VERSION_V2 and not rep.promoted_to_v1
            assert len(luts) == 3 * (spec.n_layers - len(promoted))

            # Python reference decode over reconstructed weights
            rd_spec, rd_weights, _rd_luts, rd_meta = read_q8_decoder_v2(model_path)
            assert rd_meta.artifact_sha256 == meta.artifact_sha256
            py_ids = reference_decode(prompt, rd_spec, rd_weights, max_new)
            py_logits = next_token_logits(prompt + py_ids[:-1], rd_spec, rd_weights)

            # C bcir-llama 1T vs 3T DECODE-THREADS: bit-for-bit identical!
            logits_1t = d / f"logits-1t-{int(tied)}-{len(promoted)}.f64"
            logits_3t = d / f"logits-3t-{int(tied)}-{len(promoted)}.f64"
            r1 = subprocess.run(
                [
                    str(llama_exe),
                    "--model",
                    str(model_path),
                    "--prompt-ids",
                    ",".join(map(str, prompt)),
                    "--max-new",
                    str(max_new),
                    "--threads",
                    "1",
                    "--logits-out",
                    str(logits_1t),
                ],
                capture_output=True,
                text=True,
            )
            assert r1.returncode == 0, r1.stderr
            r3 = subprocess.run(
                [
                    str(llama_exe),
                    "--model",
                    str(model_path),
                    "--prompt-ids",
                    ",".join(map(str, prompt)),
                    "--max-new",
                    str(max_new),
                    "--threads",
                    "3",
                    "--logits-out",
                    str(logits_3t),
                ],
                capture_output=True,
                text=True,
            )
            assert r3.returncode == 0, r3.stderr
            assert r1.stdout == r3.stdout
            raw_1t = logits_1t.read_bytes()
            raw_3t = logits_3t.read_bytes()
            assert raw_1t == raw_3t, "DECODE-THREADS 1T vs 3T drift on BCIRQ8 v2!"
            c_logits = struct.unpack(f"<{spec.vocab_size}d", raw_1t)
            assert max(abs(a - b) for a, b in zip(c_logits, py_logits)) <= 1e-9
            c_ids = json.loads(r1.stdout)["generated_ids"]
            assert c_ids == py_ids

            # C bcir-qualify: GEM vs monolithic bcir_llama bit-for-bit parity on BOTH native & DER packs!
            prog = decoder_program(spec, len(prompt), max_new)
            pack, _ = planned_pack(prog)
            native_bytes = encode(pack)
            der_bytes = pack_asn1.encode_pack(pack)
            pack_bin = d / f"pack-{int(tied)}-{len(promoted)}.bin"
            pack_der = d / f"pack-{int(tied)}-{len(promoted)}.der"
            pack_bin.write_bytes(native_bytes)
            pack_der.write_bytes(der_bytes)
            oracle = run_decoder_pack(pack, rd_spec, rd_weights, prompt)

            for der_flag, ppath in (("--native", pack_bin), ("--der", pack_der)):
                q_logits = d / f"qlogits-{int(tied)}-{len(promoted)}.f64"
                rq = subprocess.run(
                    [
                        str(qual_exe),
                        str(model_path),
                        str(ppath),
                        der_flag,
                        "--prompt",
                        ",".join(map(str, prompt)),
                        "--reps",
                        "2",
                        "--logits-out",
                        str(q_logits),
                    ],
                    capture_output=True,
                    text=True,
                )
                assert rq.returncode == 0, (der_flag, rq.stderr, rq.stdout)
                rep_q = json.loads(rq.stdout)
                assert rep_q["status"] == "PASS", rep_q
                assert rep_q["logits"]["bitwise_identical"] is True
                assert rep_q["logits"]["gem_runs_identical"] is True
                assert rep_q["tokens"]["gem"] == rep_q["tokens"]["monolithic"] == oracle.tokens
                assert rep_q["program"]["native_sha256"] == hashlib.sha256(native_bytes).hexdigest()
                assert q_logits.read_bytes() == raw_1t


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"PASS {_name}")
