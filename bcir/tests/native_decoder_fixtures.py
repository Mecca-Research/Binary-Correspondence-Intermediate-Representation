"""The NDT-GEM fixtures and grader: the native training step held to its GEM+ program.

`bcir.hosted.models.native_program.training_step_program` states the C trainer's step from the
spec alone. One function per row family grades it against the C rail; the tests, the harness
(`tools/perf/gemplus_baseline.py --group train`) and the fault table call them:

    train.plan.mismatch        (spec, arena) pairs over `plan_corpus()` where the program's arena
                               sum differs from `bcir_decoder_make_plan`'s count
    train.mm.mismatch          `MM_CASES` whose native `bcir_tensor_mm` output differs in any bit
                               from `mm_reference` -- an independent binary32 emulation of the
                               kernel's contract (beta first, then alpha*a times b added for k
                               ascending, every operation rounded, no contraction)
    train.memory.transient     what the C plan reserves for activations and workspace, over the
                               G26 concurrent-live bound of the same values (exact; no compiler)
    train.step.ms              one native training step at `BENCH` (wall), and its floor in the
                               same run: the step's counted work at the best rate the same kernel
                               reaches on any of the step's own GEMM shapes, timed alone
                               (`NativeProgram.roofline`, compute term) -- a bound on this host's
                               kernel, TMSAO-3, never on the silicon

Without a C compiler the native rows are not measured -- a skip, never a pass.
"""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path

from bcir.frontends.models.decode import DecoderSpec
from bcir.hosted.models.native_program import ARENAS, training_step_program

_C = Path(__file__).resolve().parents[2] / "runtime" / "c"
_FP = ct.POINTER(ct.c_float)

#: The plan corpus: tied and untied, MHA, GQA and one shared head, one to three layers, context
#: one, a batch above one, and widths that cross the kernels' 32-wide tiles.
PLAN_CORPUS = (
    (DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", n_kv_heads=1), 1, 4),
    (DecoderSpec(16, 8, 2, 2, 16, activation="silu_gate", tied_embeddings=False), 2, 3),
    (DecoderSpec(32, 16, 4, 1, 24, activation="silu_gate", n_kv_heads=2), 1, 1),
    (DecoderSpec(50, 24, 3, 3, 40, activation="silu_gate", n_kv_heads=1), 3, 5),
    (DecoderSpec(97, 64, 8, 2, 160, activation="silu_gate", n_kv_heads=4,
                 tied_embeddings=False), 2, 33),
    (DecoderSpec(256, 64, 4, 2, 192, activation="silu_gate", n_kv_heads=2), 2, 32),
)  # fmt: skip

#: The step the wall row times (also the program the memory row prices).
BENCH = (DecoderSpec(256, 64, 4, 2, 192, activation="silu_gate", n_kv_heads=2), 2, 32)

#: (ta, tb, m, n, k, alpha, beta): every transpose pair, the edges of the 32-wide tiles, the
#: trainer's three call forms and a beta that scales.
MM_CASES = tuple(
    (ta, tb, m, n, k, alpha, beta)
    for ta in (0, 1)
    for tb in (0, 1)
    for (m, n, k), alpha, beta in (
        ((1, 1, 1), 1.0, 0.0),
        ((3, 33, 5), 1.0, 1.0),
        ((33, 31, 65), 0.75, 0.0),
        ((32, 32, 32), -1.25, 0.5),
        ((7, 65, 40), 1.0, 1.0),
    )
)


def compiler() -> str | None:
    from bcir.toolchain import host_c_compiler

    return host_c_compiler()


def build_library(directory, *, sources: dict[str, str] | None = None) -> Path | None:
    """The trainer as a shared library, with the build contract `NativeDecoder.build` uses --
    or, given `sources` (a unit's name -> its text), with those texts in place of the tree's
    units (a wrong copy, never an edit of the tree). None without a compiler."""
    cc = compiler()
    if cc is None:
        return None
    from bcir.hosted.models.native import NativeDecoder

    if not sources:
        return NativeDecoder.build(directory, cc=cc)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    units = []
    for name in ("bcir_tensor.c", "bcir_decoder_train.c"):
        if name in sources:
            (target / name).write_text(sources[name], encoding="utf-8", newline="\n")
            units.append(str(target / name))
        else:
            units.append(str(_C / name))
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    out = target / ("bcir_decoder_train_mutant" + suffix)
    from bcir.toolchain import host_link_args

    cmd = [cc, "-std=c11", "-O3", "-ffp-contract=off", "-DBCIR_DT_BUILD_SHARED", "-shared"]
    if sys.platform != "win32":
        cmd.append("-fPIC")
    cmd += ["-I", str(_C), *units, "-o", str(out)]
    run = subprocess.run(cmd + host_link_args(["-lm"]), capture_output=True, text=True,
                         timeout=180)  # fmt: skip
    assert run.returncode == 0, run.stderr[-2000:]
    return out


def source(name: str) -> str:
    """A tree unit's text (`bcir_tensor.c`, `bcir_decoder_train.c`), for a wrong copy of it."""
    with open(_C / name, encoding="utf-8") as handle:
        return handle.read()


# --- train.plan.mismatch --------------------------------------------------------------------


def native_plan(lib, spec: DecoderSpec, batch: int, time_: int) -> dict[str, int]:
    """`bcir_decoder_make_plan`'s three counts, spread over the six arenas they size."""
    from bcir.hosted.models.native import _Plan, _Spec

    shape = _Spec(spec.vocab_size, spec.d_model, spec.n_heads, spec.kv_heads, spec.n_layers,
                  spec.d_ff, batch, time_, int(spec.tied_embeddings), spec.rope_base,
                  spec.rms_norm_eps)  # fmt: skip
    plan = _Plan()
    fn = lib.bcir_decoder_make_plan
    fn.argtypes, fn.restype = [ct.POINTER(_Spec), ct.POINTER(_Plan)], ct.c_int
    assert fn(ct.byref(shape), ct.byref(plan)) == 0, (spec, batch, time_)
    out = dict.fromkeys(ARENAS[:4], plan.parameters)
    out.update(activations=plan.activations, scratch=plan.scratch)
    return out


def plan_mismatch(lib, corpus=PLAN_CORPUS) -> int:
    """(spec, arena) pairs where the program and the C plan disagree."""
    bad = 0
    for spec, batch, time_ in corpus:
        mine = training_step_program(spec, batch, time_).arena_elements()
        theirs = native_plan(lib, spec, batch, time_)
        bad += sum(mine[a] != theirs[a] for a in ARENAS)
    return bad


# --- train.mm.mismatch ----------------------------------------------------------------------


def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _operands(m, n, k, seed):
    """Deterministic binary32 operands with mixed signs and magnitudes (no NaN, no overflow)."""
    state = seed * 2654435761 % 2**32 or 1

    def nxt():
        nonlocal state
        state = (state * 1103515245 + 12345) % 2**31
        return _f32((state / 2**31 - 0.5) * (8.0 if state & 1 else 0.125))

    return (
        [nxt() for _ in range(m * k)],
        [nxt() for _ in range(k * n)],
        [nxt() for _ in range(m * n)],
    )


def mm_reference(ta, tb, m, n, k, alpha, a, b, beta, c):
    """The kernel's contract in binary32, one rounded operation at a time (a double holds the
    exact sum or product of two floats only up to its own rounding, and 53 >= 2 * 24 + 2, so
    rounding that double to binary32 is the binary32 operation -- no double-rounding error)."""
    alpha, beta = _f32(alpha), _f32(beta)
    out = [0.0 if beta == 0.0 else _f32(beta * x) for x in c]
    for i in range(m):
        for j in range(n):
            acc = out[i * n + j]
            for l in range(k):
                av = _f32(alpha * a[l * m + i if ta else i * k + l])
                acc = _f32(acc + _f32(av * b[j * k + l if tb else l * n + j]))
            out[i * n + j] = acc
    return out


def mm_native(lib, ta, tb, m, n, k, alpha, a, b, beta, c):
    fn = lib.bcir_tensor_mm
    fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                   ct.c_float, _FP, _FP, ct.c_float, _FP]  # fmt: skip
    fn.restype = None
    ca, cb, cc = (ct.c_float * len(a))(*a), (ct.c_float * len(b))(*b), (ct.c_float * len(c))(*c)
    fn(None, ta, tb, m, n, k, alpha, ca, cb, beta, cc)
    return list(cc)


def mm_mismatch(lib, cases=MM_CASES) -> int:
    """Cases whose native output differs in any bit from the reference."""
    bad = 0
    for seed, (ta, tb, m, n, k, alpha, beta) in enumerate(cases, 1):
        a, b, c = _operands(m, n, k, seed)
        want = mm_reference(ta, tb, m, n, k, alpha, a, b, beta, c)
        got = mm_native(lib, ta, tb, m, n, k, alpha, a, b, beta, c)
        bad += struct.pack(f"<{len(want)}f", *want) != struct.pack(f"<{len(got)}f", *got)
    return bad


# --- train.memory.transient -----------------------------------------------------------------


def transient_ratio(bench=BENCH) -> tuple[float, int, int, int]:
    """(C bytes / bound, C bytes, the G26 bound, GEM+'s incumbent layout) for the bench step."""
    program = training_step_program(*bench)
    held, bound = program.transient_bytes(), program.transient_lower_bound()
    return held / bound, held, bound, program.transient_incumbent().extent


# --- train.step.ms and its floor ------------------------------------------------------------


def _best_ms(fn, repeats):
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        took = (time.perf_counter() - start) * 1000
        best = took if best is None else min(best, took)
    return best


def _gemm_ms(fn, shape, repeats):
    """The best of `repeats` timings of `calls` back-to-back GEMMs of `shape`, and `calls`
    (enough to amortize the ctypes call on a tiny shape)."""
    ta, tb, m, n, k = shape
    a, b, c = (ct.c_float * (m * k))(), (ct.c_float * (k * n))(), (ct.c_float * (m * n))()
    for buf in (a, b):
        for i in range(len(buf)):
            buf[i] = ((i * 37) % 17 - 8) / 16.0
    calls = max(1, 200_000 // (m * n * k))

    def run():
        for _ in range(calls):
            fn(None, ta, tb, m, n, k, 1.0, a, b, 0.0, c)

    return _best_ms(run, repeats), calls


def peak_rate(lib, program, repeats=7) -> tuple[float, tuple]:
    """The best work units per second `bcir_tensor_mm` reaches on any GEMM shape the step
    issues, each timed alone (best of `repeats`), and that shape."""
    fn = lib.bcir_tensor_mm
    fn.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                   ct.c_float, _FP, _FP, ct.c_float, _FP]  # fmt: skip
    fn.restype = None
    best, which = 0.0, None
    for shape in program.gemm_shapes():
        ms, calls = _gemm_ms(fn, shape, repeats)
        _ta, _tb, m, n, k = shape
        rate = 2 * m * n * k * calls / (ms / 1000)
        if rate > best:
            best, which = rate, shape
    return best, which


def step_ms(path, bench=BENCH, steps=4, repeats=9) -> float:
    """The median wall time of one `bcir_decoder_run` step at `bench`, over `repeats` runs of
    `steps` steps each on one trainer, after one untimed run (the step's work does not depend
    on the values it trains, so the runs time the same kernels on warm arenas)."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder

    spec, batch, time_ = bench
    n = batch * time_
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n * steps)]
    targets = [(7 * i + 4) % spec.vocab_size for i in range(n * steps)]
    trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_)
    opt = NativeAdamW(lr=1e-3)
    trainer.run(tokens, targets, steps=steps, optimizer=opt)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        trainer.run(tokens, targets, steps=steps, optimizer=opt)
        samples.append((time.perf_counter() - start) * 1000 / steps)
    return statistics.median(samples)


def measure_step(path, bench=BENCH) -> dict[str, float]:
    """`train.step.ms` and its floor, measured in the same run on the same library (`path`:
    the shared library `build_library` wrote)."""
    program = training_step_program(*bench)
    value = step_ms(path, bench)
    rate, _shape = peak_rate(ct.CDLL(str(path)), program)
    seconds, _terms = program.roofline(Fraction(rate))
    return {"train.step.ms": value, "train.step.ms.floor": float(seconds) * 1000}


# --- every row --------------------------------------------------------------------------------


def measure(*, timed: bool = True) -> dict[str, float]:
    """Every NDT-GEM row this host can measure: the memory row always, the native rows with a
    compiler (and the timed one only when `timed`)."""
    ratio = transient_ratio()[0]
    out = {"train.memory.transient": ratio}
    tmp = tempfile.mkdtemp(prefix="bcir-ndt-")
    try:
        path = build_library(tmp)
        if path is None:
            return out
        lib = ct.CDLL(str(path))
        out["train.plan.mismatch"] = float(plan_mismatch(lib))
        out["train.mm.mismatch"] = float(mm_mismatch(lib))
        if timed:
            out.update(measure_step(path))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


__all__ = [
    "BENCH",
    "MM_CASES",
    "PLAN_CORPUS",
    "build_library",
    "measure",
    "measure_step",
    "mm_mismatch",
    "mm_reference",
    "plan_mismatch",
    "source",
    "transient_ratio",
]


if __name__ == "__main__":  # pragma: no cover - a manual probe
    print(measure())
    os._exit(0)
