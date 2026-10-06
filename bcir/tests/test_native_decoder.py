"""Dependency-free native planning, kernels, learning and checkpoint gates.

Framework differentials run separately in the hosted-model CI job. The quick
runner hides compilers; pure layout/configuration contracts still execute.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native import (
    NativeAdamW,
    NativeDecoder,
    parameter_layout,
    _checkpoint_digest,
)


def _spec():
    return DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1)


def _reject(fn):
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError("invalid input accepted")


def test_native_parameter_layout_and_optimizer_contracts():
    layout = parameter_layout(_spec())
    assert sum(row[3] for row in layout) == 1320
    assert "lm_head.weight" not in {row[0] for row in layout}
    assert len(layout) == 20
    for settings in (
        {"lr": math.inf},
        {"epsilon": 0},
        {"beta1": 1},
        {"beta2": -1},
        {"grad_clip": -1},
        {"lr": True},
        {"weight_decay": math.nan},
    ):
        _reject(lambda settings=settings: NativeAdamW(**settings))
    NativeAdamW(lr=0, grad_clip=0, beta1=0, beta2=0)


def test_native_decoder_learning_resume_memory_limits_and_corruption():
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        root = Path(temporary)
        lib = NativeDecoder.build(root, cc=cc)
        trainer = NativeDecoder(lib, _spec())
        _reject(lambda: NativeDecoder(lib, _spec(), max_memory_bytes=100))
        _reject(lambda: NativeDecoder(lib, _spec(), context_length=2**32))
        _reject(lambda: trainer.backward([1, 2, 3, 16], [2, 3, 4, 5]))
        inputs, targets = [1, 2, 3, 4], [2, 3, 4, 5]
        opt = NativeAdamW(lr=0.01, weight_decay=0.01)
        events = trainer.run(inputs * 40, targets * 40, steps=40, optimizer=opt)
        assert events[-1]["loss"] < events[0]["loss"] * 0.5
        path = root / "checkpoint.bcirdt"
        trainer.save(path)
        other = NativeDecoder.resume(lib, path)
        assert other.state.step == 40
        assert trainer.run(inputs * 2, targets * 2, steps=2, optimizer=opt) == other.run(
            inputs * 2, targets * 2, steps=2, optimizer=opt
        )
        for name in ("weights", "moment1", "moment2"):
            assert bytes(trainer.buffers[name]) == bytes(other.buffers[name])
        before = bytes(trainer.buffers["weights"])
        trainer.buffers["gradients"][-1] = math.nan
        _reject(lambda: trainer.update(opt))
        assert bytes(trainer.buffers["weights"]) == before
        original = path.read_bytes()
        changed = bytearray(original)
        changed[-1] ^= 0x80
        path.write_bytes(changed)
        _reject(lambda: NativeDecoder.resume(lib, path))
        size = struct.unpack("<I", original[8:12])[0]
        meta = json.loads(original[12 : 12 + size])
        altered = dict(meta)
        altered["decoder"] = dict(meta["decoder"], rope_base=1234.0)
        header = json.dumps(altered).encode()
        path.write_bytes(
            original[:8] + struct.pack("<I", len(header)) + header + original[12 + size :]
        )
        _reject(lambda: NativeDecoder.resume(lib, path))
        header = original[12 : 12 + size].decode().rstrip("}") + ',"step":0}'
        encoded = header.encode()
        path.write_bytes(
            original[:8] + struct.pack("<I", len(encoded)) + encoded + original[12 + size :]
        )
        _reject(lambda: NativeDecoder.resume(lib, path))
        payload = bytearray(original[12 + size :])
        payload[:4] = struct.pack("<f", math.nan)
        meta["sha256"] = _checkpoint_digest(meta, payload)
        header = json.dumps(meta).encode()
        path.write_bytes(original[:8] + struct.pack("<I", len(header)) + header + payload)
        _reject(lambda: NativeDecoder.resume(lib, path))
        path.write_bytes(original + b"x")
        _reject(lambda: NativeDecoder.resume(lib, path))
        staged = trainer.state_dict()
        staged["model.norm.weight"][0] = math.inf
        _reject(lambda: trainer.load_state_dict(staged))
        assert bytes(trainer.buffers["weights"]) == before


def test_native_c_harness_c11_and_c23():
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        return
    root = Path(__file__).resolve().parents[2]
    c = root / "runtime" / "c"
    from bcir.toolchain import host_link_args

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        for standard in ("c11", "c2x"):
            exe = Path(temporary) / f"gate-{standard}"
            cmd = [
                cc,
                f"-std={standard}",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-ffp-contract=off",
                "-I",
                str(c),
                str(c / "test_decoder_train.c"),
                str(c / "bcir_tensor.c"),
                str(c / "bcir_decoder_train.c"),
                "-o",
                str(exe),
            ]
            subprocess.run(
                cmd + host_link_args(["-lm"]), check=True, capture_output=True, timeout=120
            )
            result = subprocess.run(
                [str(exe)], check=True, capture_output=True, text=True, timeout=60
            )
            assert result.stdout.count("PASS") == 3


def test_native_provider_c_abi_keeps_frameworks_off_execution_path():
    cc = shutil.which("g++") or shutil.which("clang++")
    if not cc:
        return
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
        src = '#include "bcir_decoder_train.h"\nstatic_assert(sizeof(float)==4);\nint f(){return BCIR_DT_OK;}\n'
        subprocess.run(
            [
                cc,
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-I",
                str(root / "runtime" / "c"),
                "-x",
                "c++",
                "-c",
                "-o",
                str(Path(temporary) / "boundary.o"),
                "-",
            ],
            input=src,
            text=True,
            capture_output=True,
            check=True,
            timeout=60,
        )
