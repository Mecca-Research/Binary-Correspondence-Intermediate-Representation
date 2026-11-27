"""End-to-end Python ABI 6 MIXER-EXT tests (mixers, upgrades, Muon/Q8-Muon, checkpointing)."""

from __future__ import annotations

import ctypes as ct
import math
import tempfile
from pathlib import Path

from bcir.hosted.models.mixer_ext import (
    EXT_IGNORE,
    MixerExtSpec,
    NativeMixerExt,
    OptimizerConfig,
    PrecisionPolicy,
)
from bcir.toolchain import host_c_compiler

_CC = host_c_compiler()
_LIB_DIR: tempfile.TemporaryDirectory[str] | None = None
_LIB_PATH: Path | None = None


def _get_lib() -> Path | None:
    global _LIB_DIR, _LIB_PATH
    if _CC is None:
        return None
    if _LIB_PATH is None:
        _LIB_DIR = tempfile.TemporaryDirectory(prefix="bcir-mixer-ext-test-")
        _LIB_PATH = NativeMixerExt.build(_LIB_DIR.name, cc=_CC)
    return _LIB_PATH


def _synth_data(n: int, vocab: int, seed: int = 1337) -> tuple[list[int], list[int]]:
    r = seed
    toks, tgts = [], []
    for _ in range(n):
        r = (r * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        toks.append((r >> 32) % vocab)
        r = (r * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        tgts.append((r >> 32) % vocab)
    return toks, tgts


def test_abi6_verifiers_and_identities():
    lib_path = _get_lib()
    if lib_path is None:
        return
    lib = NativeMixerExt.load_library(lib_path)
    assert lib.bcir_mixer_ext_abi_version() == 6

    d = ct.c_double(1.0)
    assert lib.bcir_deltanet_verify_wy_vs_recurrent(2, 16, 2, 8, 4, ct.byref(d)) == 0
    assert d.value < 1e-5
    assert lib.bcir_kda_verify_wy_vs_recurrent(2, 16, 2, 8, 4, ct.byref(d)) == 0
    assert d.value < 1e-5
    assert lib.bcir_gdn2_verify_wy_vs_recurrent(2, 16, 2, 8, 4, ct.byref(d)) == 0
    assert d.value < 1e-5
    assert lib.bcir_mla_verify_absorption(12, 2, 8, 8, 4, ct.byref(d)) == 0
    assert d.value < 1e-6
    assert lib.bcir_mla_verify_absorption(12, 2, 8, 8, 0, ct.byref(d)) == 0
    assert d.value < 1e-6
    m_err, g_err = ct.c_double(1.0), ct.c_double(1.0)
    assert lib.bcir_mhc_verify_birkhoff(2, ct.byref(m_err), ct.byref(g_err)) == 0
    assert m_err.value < 1e-5 and g_err.value < 1e-4
    ns_err = ct.c_double(1.0)
    assert lib.bcir_newton_schulz_verify(None, 16, 24, ct.byref(ns_err)) == 0
    assert ns_err.value < 0.35


def test_hybrid_gdn2_gated_mla_train_and_checkpoint_roundtrip():
    lib_path = _get_lib()
    if lib_path is None:
        return
    spec = MixerExtSpec(
        vocab=24,
        width=32,
        heads=4,
        kvheads=2,
        layers=4,
        ff=64,
        batch=2,
        time=12,
        layer_mixers=("gdn2", "kda", "deltanet", "mla"),
        layer_use_rope=(False, False, False, False),
        mla_dc=16,
        mla_dr=4,
        gate_mode="g1_headwise",
        qk_norm=True,
        canon_mode="canon_ac",
        mhc_streams=2,
        online_softmax=True,
        gate_rank=8,
        seed=7,
        threads=2,
    )
    pol = PrecisionPolicy(opt_kind="muon_q8_bf16_grad")
    opt = OptimizerConfig(lr=3e-3, grad_clip=3.0)
    stride = spec.batch * spec.time
    toks, tgts = _synth_data(6 * stride, spec.vocab)

    with NativeMixerExt(lib_path, spec, pol) as model:
        ms = model.mem_stats()
        assert ms["persistent_bytes"] < ms["n_params"] * 8.0
        ev_first = model.train_steps(toks[: 3 * stride], tgts[: 3 * stride], steps=3, opt=opt)
        assert all(math.isfinite(e["loss"]) and e["status"] == 0 for e in ev_first)
        tel = model.telemetry()
        assert tel["muon_ns_calls"] > 0
        assert tel["fp64_wy_solves"] > 0

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "hybrid_step3"
            digest = model.save_checkpoint(ckpt, step=3, extra={"tag": "abi6"})
            assert len(digest) == 64

            # Continue uninterrupted for steps 4..6 on the original model
            ev_cont = model.train_steps(toks[3 * stride :], tgts[3 * stride :], steps=3, opt=opt)
            assert all(math.isfinite(e["loss"]) and e["status"] == 0 for e in ev_cont)
            weights_uninterrupted = model.weights_bytes()
            eval_before = model.evaluate(toks, tgts, n_batches=2)
            assert math.isfinite(eval_before["ce"]) and eval_before["nonfinite"] == 0

            # Restore step-3 weights + optimizer state into a fresh model and continue steps 4..6
            spec_resume = MixerExtSpec(
                vocab=spec.vocab,
                width=spec.width,
                heads=spec.heads,
                kvheads=spec.kvheads,
                layers=spec.layers,
                ff=spec.ff,
                batch=spec.batch,
                time=spec.time,
                layer_mixers=spec.layer_mixers,
                layer_use_rope=spec.layer_use_rope,
                mla_dc=spec.mla_dc,
                mla_dr=spec.mla_dr,
                gate_mode=spec.gate_mode,
                qk_norm=spec.qk_norm,
                canon_mode=spec.canon_mode,
                mhc_streams=spec.mhc_streams,
                online_softmax=spec.online_softmax,
                gate_rank=spec.gate_rank,
                skip_init=True,
                threads=1,
            )
            with NativeMixerExt(lib_path, spec_resume, pol) as restored:
                meta = restored.load_checkpoint(ckpt)
                assert meta["step"] == 3 and meta["extra"]["tag"] == "abi6"
                assert "opt_state_sha256" in meta
                ev_resumed = restored.train_steps(
                    toks[3 * stride :], tgts[3 * stride :], steps=3, opt=opt
                )
                assert [e["loss"] for e in ev_resumed] == [e["loss"] for e in ev_cont]
                assert [e["step"] for e in ev_resumed] == [e["step"] for e in ev_cont]
                assert restored.weights_bytes() == weights_uninterrupted
                eval_after = restored.evaluate(toks, tgts, n_batches=2)
                assert eval_after["ce"] == eval_before["ce"]
                assert eval_after["accuracy"] == eval_before["accuracy"]


def test_optimizer_memory_frontier_and_fail_closed():
    lib_path = _get_lib()
    if lib_path is None:
        return
    spec = MixerExtSpec(
        vocab=32,
        width=64,
        heads=4,
        kvheads=2,
        layers=2,
        ff=128,
        batch=2,
        time=8,
        layer_mixers=("gdn2", "mla"),
        layer_use_rope=(False, True),
        online_softmax=True,
    )
    bytes_by_opt = {}
    for kind in (
        "adamw_fp32",
        "adamw_adafactor_m2",
        "adamw_q8_moments_bf16_grad",
        "muon_q8_bf16_grad",
    ):
        with NativeMixerExt(lib_path, spec, PrecisionPolicy(opt_kind=kind)) as model:
            ms = model.mem_stats()
            bytes_by_opt[kind] = ms["persistent_bytes"] / ms["n_params"]
    assert bytes_by_opt["muon_q8_bf16_grad"] < bytes_by_opt["adamw_q8_moments_bf16_grad"]
    assert bytes_by_opt["adamw_q8_moments_bf16_grad"] < bytes_by_opt["adamw_adafactor_m2"]
    assert bytes_by_opt["adamw_adafactor_m2"] < bytes_by_opt["adamw_fp32"]

    # All-masked target batch must fail closed
    toks, _ = _synth_data(spec.batch * spec.time, spec.vocab)
    masked = [EXT_IGNORE] * (spec.batch * spec.time)
    with NativeMixerExt(lib_path, spec, PrecisionPolicy()) as model:
        try:
            model.train_steps(toks, masked, steps=1, opt=OptimizerConfig())
        except ValueError:
            pass
        else:
            raise AssertionError("all-masked target batch was not refused")


TESTS = (
    test_abi6_verifiers_and_identities,
    test_hybrid_gdn2_gated_mla_train_and_checkpoint_roundtrip,
    test_optimizer_memory_frontier_and_fail_closed,
)
