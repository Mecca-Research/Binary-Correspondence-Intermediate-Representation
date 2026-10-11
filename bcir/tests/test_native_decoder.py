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


#: The compiler the NDT-GEM witnesses build with (`bcir.toolchain.host_c_compiler`), or None: the
#: capability their fault table (tools/testing/faults/ndt-gem.json) requires.
_CC = __import__("bcir.toolchain", fromlist=["host_c_compiler"]).host_c_compiler()


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
            assert result.stdout.count("PASS") == 6


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


# --- NDT-GEM: the step held to its GEM+ program (bcir/tests/native_decoder_fixtures.py) -----


def _ndt_library(tmp, sources=None):
    from bcir.tests import native_decoder_fixtures as nf

    path = nf.build_library(tmp, sources=sources)
    return None if path is None else __import__("ctypes").CDLL(str(path))


def test_the_compiled_plan_and_gemm_are_the_programs():
    """Both exact rows at zero: the C plan's arenas are the program's over the corpus, and
    bcir_tensor_mm is its binary32 contract bit for bit on all 20 cases."""
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        lib = _ndt_library(tmp)
        if lib is None:
            return  # no C compiler: the native rows are a skip, never a pass
        assert nf.plan_mismatch(lib) == 0
        assert nf.mm_mismatch(lib) == 0
        assert len(nf.MM_CASES) == 20 and len(nf.PLAN_CORPUS) == 6


def test_the_rows_fire_on_wrong_kernels_and_plans():
    """Wrong copies of the C units, built from copies of their sources (never by editing the
    tree): the packed B^T tile read without its depth offset, the register block accumulated in
    descending order or fused into one rounding, A read untransposed whatever ta says; the plan's
    workspace or cache count off by one span. Each turns its row red."""
    from bcir.tests import native_decoder_fixtures as nf

    if nf.compiler() is None:
        return
    tensor, train = nf.source("bcir_tensor.c"), nf.source("bcir_decoder_train.c")
    mutants = (
        ("mm", "bcir_tensor.c", tensor,
         "x->b[(jb+j)*x->ldb+lb+l]", "x->b[(jb+j)*x->ldb+l]"),
        ("mm", "bcir_tensor.c", tensor,
         "for (l=0;l<nl;l++) {\n    const float *bl=bt+l*nr;",
         "for (l=nl;l-->0;) {\n    const float *bl=bt+l*nr;"),
        ("mm", "bcir_tensor.c", tensor,
         "acc[r][j]+=av[r]*bl[j];", "acc[r][j]=fmaf(av[r],bl[j],acc[r][j]);"),
        ("mm", "bcir_tensor.c", tensor,
         "x->a[x->ta ? (lb+l)*x->lda+i+r : (i+r)*x->lda+lb+l]", "x->a[(i+r)*x->lda+lb+l]"),
        ("plan", "bcir_decoder_train.c", train,
         "!term(&p.scratch,5,nd)", "!term(&p.scratch,6,nd)"),
        ("plan", "bcir_decoder_train.c", train,
         "!term(&la,3,nf)", "!term(&la,4,nf)"),
    )  # fmt: skip
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for i, (row, unit, text, old, new) in enumerate(mutants):
            assert text.count(old) == 1, old
            lib = _ndt_library(Path(tmp) / str(i), {unit: text.replace(old, new)})
            fired = nf.mm_mismatch(lib) if row == "mm" else nf.plan_mismatch(lib)
            assert fired > 0, f"train.{row}.mismatch cannot see {new!r}"


def test_the_step_floor_is_measured_beside_the_step():
    """`train.step.ms`'s floor is G26's compute term at the best rate the same library reaches on
    the step's own GEMM shapes, in the same run: positive and below the step on a small bench."""
    from bcir.tests import native_decoder_fixtures as nf

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = nf.build_library(tmp)
        if path is None:
            return
        small = (DecoderSpec(64, 32, 4, 1, 64, activation="silu_gate", n_kv_heads=2), 2, 16)
        rows = nf.measure_step(path, small)
        assert 0 < rows["train.step.ms.floor"] < rows["train.step.ms"], rows


def test_the_harness_rows_are_the_programs_numbers():
    """`train.memory.transient`'s frozen baseline is the bench program's own ratio, and every
    NDT-GEM row the harness declares is one the fixtures measure -- a change to the program or
    the C plan cannot leave the harness quoting a stale number."""
    import sys

    from bcir.tests import native_decoder_fixtures as nf

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.perf.gemplus_baseline import METRICS

    rows = {m.key: m for m in METRICS if m.group == "train"}
    assert set(rows) == {"train.plan.mismatch", "train.mm.mismatch", "train.memory.transient",
                         "train.step.ms"}  # fmt: skip
    ratio, held, bound, incumbent = nf.transient_ratio()
    assert rows["train.memory.transient"].baseline == ratio == held / bound
    assert (held, bound, incumbent) == (999_424, 753_664, 753_664)
    assert rows["train.mm.mismatch"].what.startswith(f"of {len(nf.MM_CASES)} GEMM cases")
    pairs = len(nf.PLAN_CORPUS) * 2  # the parent stated activations and scratch only in C
    assert rows["train.plan.mismatch"].baseline == pairs
