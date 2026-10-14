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
    train.mm.fused.mismatch    `MM_CASES` whose native `bcir_tensor_mm_fused` output differs in any bit
                               from `mm_reference_fused` -- the fused contract (v2, NDT-FMA) as an
                               exact rational emulation rounded once per term
    train.exp.mismatch         inputs among the 2^24 `exp_sweep` sweeps (every binary32 in [-104, 89]
                               at stride 2^8) where two instances of the BCIR exponential differ
    train.exp.ulp.max          the largest distance, in ulps, of the BCIR exponential from exp in
                               double rounded to binary32 over the same inputs (stated: 1)
    train.state.threads.mismatch  thread counts among `THREAD_COUNTS[1:]` whose state after three
                               steps of `THREADS_BENCH`, or the losses and norms those steps
                               reported, differ in any bit from the one-thread run's on the same
                               library, weights, tokens and rates (NDT-THREADS), under v1 and
                               under the reduce contract
    train.norm.reduce.mismatch  of the clipping norms' squares and roots three steps of
                               `THREADS_BENCH` compute under v1 and under the reduce contract,
                               those that differ in any bit from `grad_norm_squared_reference`
                               and `grad_norm_reference` -- the oracle's emulation of each
                               contract from the gradients the backward left -- plus the counts
                               of `REDUCE_CHUNK_COUNTS` where the C chunk rule and `reduce_chunk`
                               disagree (NDT-REDUCE)

Without a C compiler the native rows are not measured -- a skip, never a pass.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import importlib.util
import math
import os
import shutil
import statistics
import re
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

#: The blocked contract's cases (NDT-KBLOCK): the 20 above and twenty deeper ones that cross the
#: split rule's three cases (one block, half-and-half, whole blocks then halves), an alpha the
#: binary32 product does not scale exactly, and beta 1 (the trainer's dW accumulations).
MM_KBLOCK_CASES = MM_CASES + tuple(
    (ta, tb, m, n, k, alpha, beta)
    for ta in (0, 1)
    for tb in (0, 1)
    for (m, n, k), alpha, beta in (
        ((2, 3, 1000), 1.0, 1.0),
        ((1, 2, 2048), 1.0, 0.0),
        ((3, 2, 449), 1.0, 1.0),
        ((2, 2, 897), 0.5, 1.0),
        ((2, 2, 100), 1.1, 1.0),
    )
)

#: The step the threads row trains (NDT-THREADS): 80 tokens (one 64-row GEMM task and a 16-row
#: tail), T=40 (past the attention backward's 32-row chunks), GQA groups of two, a width, a KV
#: width, an FF width and a vocabulary off every 32- and 64-wide block edge; three steps.
THREADS_BENCH = (DecoderSpec(250, 72, 4, 2, 200, activation="silu_gate", n_kv_heads=2), 2, 40)
THREAD_COUNTS = (1, 2, 3, 4)


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
        cmd += ["-fPIC", "-pthread"]
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


def round_f32(x: Fraction) -> float:
    """The binary32 nearest the rational x, ties to even, exactly -- never through a double, whose
    own rounding would resolve a tie wrongly (1 + 2^-24 + 2^-60 rounds to 1 + 2^-23 in binary32
    but to the tie 1 + 2^-24 in double first, and from there to 1). Overflow is the signed
    infinity; the result may be subnormal or zero."""
    if x == 0:
        return 0.0
    sign = -1.0 if x < 0 else 1.0
    a = abs(x)
    n, d = a.numerator, a.denominator
    e = n.bit_length() - d.bit_length()  # 2^(e-1) < a < 2^(e+1)
    if a < Fraction(2) ** e:
        e -= 1  # 2^e <= a < 2^(e+1)
    scale = max(e - 23, -149)  # the quantum: 2^(e-23), floored at the subnormal quantum
    num, den = (n, d << scale) if scale >= 0 else (n << -scale, d)
    q, r = divmod(num, den)
    if 2 * r > den or (2 * r == den and q & 1):
        q += 1
    if q >> 24:  # rounding carried into the next binade
        q >>= 1
        scale += 1
    if scale + 23 > 127:
        return sign * math.inf
    return sign * math.ldexp(q, scale)  # exact: q < 2^24 and the exponent is in range


def mm_reference_fused(ta, tb, m, n, k, alpha, a, b, beta, c):
    """The fused contract (v2): beta first, then for k ascending acc = round(acc + av * b) rounded
    ONCE, av the binary32 product alpha*a as in v1 -- exact rationals, so no double stands in."""
    alpha, beta = _f32(alpha), _f32(beta)
    out = [0.0 if beta == 0.0 else _f32(beta * x) for x in c]
    for i in range(m):
        for j in range(n):
            acc = Fraction(out[i * n + j])
            for l in range(k):
                av = Fraction(_f32(alpha * a[l * m + i if ta else i * k + l]))
                acc = Fraction(round_f32(acc + av * Fraction(b[j * k + l if tb else l * n + j])))
            out[i * n + j] = float(acc)
    return out


def kblock_split(k: int, q: int = 448, u: int = 16) -> list[int]:
    """The blocked contract's depth split (bcir_tensor_kblock_split): while terms remain, a
    whole block of q when at least 2q remain, else half the remainder rounded up to a
    multiple of u when more than q remain, else the remainder."""
    out, ls = [], 0
    while ls < k:
        rem = k - ls
        length = q if rem >= 2 * q else (rem // 2 + u - 1) // u * u if rem > q else rem
        out.append(length)
        ls += length
    return out


def mm_reference_kblock(ta, tb, m, n, k, alpha, a, b, beta, c):
    """The blocked fused contract (v3): beta first; for each block of `kblock_split(k)` in
    order, P = the fused chain of its terms from zero (exact rationals, rounded once per
    term, a unscaled), then c = round(c + round(alpha*P)) -- two binary32 operations."""
    alpha, beta = _f32(alpha), _f32(beta)
    out = [0.0 if beta == 0.0 else _f32(beta * x) for x in c]
    blocks = kblock_split(k)
    for i in range(m):
        for j in range(n):
            acc, l0 = out[i * n + j], 0
            for length in blocks:
                p = Fraction(0)
                for l in range(l0, l0 + length):
                    av = Fraction(a[l * m + i if ta else i * k + l])
                    p = Fraction(round_f32(p + av * Fraction(b[j * k + l if tb else l * n + j])))
                acc = _f32(acc + _f32(alpha * float(p)))
                l0 += length
            out[i * n + j] = acc
    return out


_MM_SIG = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
           ct.c_float, _FP, _FP, ct.c_float, _FP]  # fmt: skip


def mm_native(lib, ta, tb, m, n, k, alpha, a, b, beta, c, *, fused=False, kblock=False,
              variant=None):  # fmt: skip
    """The contract's default entry (the widest instance the host runs), or with `variant` one
    named instance (1 generic, 2 AVX2, 4 AVX-512) through the contract's `_variant` entry,
    which refuses -- returns 0 and writes nothing -- an instance the host lacks."""
    name = (
        "bcir_tensor_mm_kblock" if kblock else "bcir_tensor_mm_fused" if fused else "bcir_tensor_mm"
    )
    fn = getattr(lib, name + ("_variant" if variant is not None else ""))
    fn.argtypes = ([ct.c_int] if variant is not None else [ct.c_void_p]) + _MM_SIG[1:]
    fn.restype = ct.c_int if variant is not None else None
    ca, cb, cc = (ct.c_float * len(a))(*a), (ct.c_float * len(b))(*b), (ct.c_float * len(c))(*c)
    ran = fn(variant if variant is not None else None, ta, tb, m, n, k, alpha, ca, cb, beta, cc)
    if variant is not None and ran != variant:
        raise RuntimeError(f"instance {variant} is not one this host runs")
    return list(cc)


def mm_variants(lib, *, fused=False, kblock=False) -> list[int]:
    """The instances the host runs under a contract, as the unit reports them (1, 2, 4)."""
    name = (
        "bcir_tensor_mm_kblock" if kblock else "bcir_tensor_mm_fused" if fused else "bcir_tensor_mm"
    )
    fn = getattr(lib, name + "_variants")
    fn.argtypes = []
    fn.restype = ct.c_int
    mask = fn()
    return [v for v in (1, 2, 4) if mask & v]


def mm_mismatch(lib, cases=MM_CASES, *, fused=False, kblock=False) -> int:
    """Cases whose native output differs in any bit from the reference of its contract. Under
    the blocked contract every instance the host runs is held to it (NDT-BLOCK: the AVX-512 and
    AVX2 instances are their own code, not the generic block under a wider target), so a case
    counts once per instance that differs."""
    bad = 0
    reference = mm_reference_kblock if kblock else mm_reference_fused if fused else mm_reference
    variants = mm_variants(lib, kblock=True) if kblock else [None]
    for seed, (ta, tb, m, n, k, alpha, beta) in enumerate(cases, 1):
        a, b, c = _operands(m, n, k, seed)
        want = struct.pack(f"<{m * n}f", *reference(ta, tb, m, n, k, alpha, a, b, beta, c))
        for variant in variants:
            got = mm_native(lib, ta, tb, m, n, k, alpha, a, b, beta, c, fused=fused,
                            kblock=kblock, variant=variant)  # fmt: skip
            bad += want != struct.pack(f"<{len(got)}f", *got)
    return bad


def mm_fused_mismatch(lib, cases=MM_CASES) -> int:
    """`train.mm.fused.mismatch`: the fused kernel against its exact emulation."""
    return mm_mismatch(lib, cases, fused=True)


def mm_kblock_mismatch(lib, cases=MM_KBLOCK_CASES) -> int:
    """`train.mm.kblock.mismatch`: every instance of the blocked kernel the host runs against
    its exact emulation, and the C split rule against the Python one on the cases' depths and
    the rule's edges."""
    split = lib.bcir_tensor_kblock_split
    split.argtypes = [ct.c_size_t, ct.POINTER(ct.c_size_t), ct.c_size_t]
    split.restype = ct.c_size_t
    bad = 0
    for k in sorted({case[4] for case in cases} | {0, 447, 448, 449, 896, 897, 1024, 3000}):
        lengths = (ct.c_size_t * 64)()
        count = split(k, lengths, 64)
        bad += list(lengths[:count]) != kblock_split(k)
    return bad + mm_mismatch(lib, cases, kblock=True)


def openblas_library() -> Path | None:
    """The scipy_openblas32 wheel's shared library, when the wheel is importable here."""
    spec = importlib.util.find_spec("scipy_openblas32")
    if spec is None:
        return None
    module = importlib.import_module("scipy_openblas32")
    for name in ("libscipy_openblas.so", "libscipy_openblas.dylib", "libscipy_openblas.dll"):
        path = Path(module.get_lib_dir()) / name
        if path.exists():
            return path
    return None


def provider_mismatch(lib, shapes=None) -> int | None:
    """`train.mm.provider.differences`: OpenBLAS's sgemm (scipy_openblas32, one thread) against
    the blocked contract over the comparison decoder's step shapes at beta 0 and 1 -- the
    admission differential `NativeDecoder.use_cblas` runs; None where the wheel is not
    importable. A host-dependent count: 0 where the wheel's target computes the contract (the
    SkylakeX kernels of the 2026-10-12 Intel host), 14 of 30 on PR #817's AMD EPYC host, whose
    OpenBLAS blocks differently -- reported, never graded off the baseline host."""
    from bcir.hosted.models.native import _CBLAS, provider_mismatch as differential
    from bcir.hosted.models.native_program import training_step_program

    path = openblas_library()
    if path is None:
        return None
    if shapes is None:
        spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
        shapes = tuple(dict.fromkeys(training_step_program(spec, 8, 128).gemm_shapes()))
    blas = ct.CDLL(str(path))
    blas.scipy_openblas_set_num_threads.argtypes = [ct.c_int]
    blas.scipy_openblas_set_num_threads(1)
    slot = ct.cast(blas.scipy_cblas_sgemm, _CBLAS)
    return differential(lib, ct.cast(ct.pointer(slot), ct.c_void_p), shapes)


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


def provider_unsound(path, shapes=None) -> int | None:
    """`train.mm.provider.unsound`: the admission verdict `NativeDecoder.use_cblas` gives
    OpenBLAS on the comparison decoder, against the differential it is defined by -- 1 where
    the provider is admitted though a case differs, refused though none does, or refused with
    a count other than the differential's; 0 on a host that admits (the Intel host) and on one
    that refuses (PR #817's AMD host) alike; None where the wheel is not importable. The exact
    row every host grades: the guard is sound wherever it runs, whatever the wheel computes."""
    from bcir.hosted.models.native import NativeDecoder

    blas = openblas_library()
    if blas is None:
        return None
    lib = ct.CDLL(str(path))
    differences = provider_mismatch(lib, shapes)
    spec = DecoderSpec(260, 384, 8, 6, 1024, activation="silu_gate", n_kv_heads=4)
    trainer = NativeDecoder(path, spec, batch_size=8, context_length=128, contract="kblock")
    try:
        trainer.use_cblas(blas, symbol="scipy_cblas_sgemm")
    except ValueError as refusal:
        counted = re.search(r"(\d+) of (\d+) GEMM cases differ", str(refusal))
        unsound = (
            trainer.backend != "portable-c"
            or trainer.provider_admission is not None
            or counted is None
            or int(counted.group(1)) != differences
            or differences == 0
        )
    else:
        admitted = trainer.provider_admission
        unsound = (
            admitted is None
            or admitted["mismatch"] != 0
            or differences != 0
            or not trainer.backend.startswith("cblas:")
        )
    finally:
        trainer.close()
    return int(unsound)


def measure_step(path, bench=BENCH) -> dict[str, float]:
    """`train.step.ms` and its floor, measured in the same run on the same library (`path`:
    the shared library `build_library` wrote)."""
    program = training_step_program(*bench)
    value = step_ms(path, bench)
    rate, _shape = peak_rate(ct.CDLL(str(path)), program)
    seconds, _terms = program.roofline(Fraction(rate))
    return {"train.step.ms": value, "train.step.ms.floor": float(seconds) * 1000}


# --- train.exp.mismatch, train.exp.ulp.max ----------------------------------------------------


def exp_sweep(lib, stride: int = 256) -> tuple[int, int, int]:
    """(inputs swept, inputs where two instances differ, max ulp from exp in double rounded to
    binary32) of `bcir_tensor_exp_sweep` at `stride`."""
    fn = lib.bcir_tensor_exp_sweep
    fn.argtypes = [ct.c_uint, ct.POINTER(ct.c_size_t), ct.POINTER(ct.c_uint)]
    fn.restype = ct.c_size_t
    mismatch, ulp = ct.c_size_t(), ct.c_uint()
    swept = fn(stride, ct.byref(mismatch), ct.byref(ulp))
    return swept, mismatch.value, ulp.value


# --- train.state.threads.mismatch -----------------------------------------------------------


def state_digest(trainer) -> str:
    """SHA-256 over the weights, both moments, the step and the bias powers."""
    digest = hashlib.sha256()
    for name in ("weights", "moment1", "moment2"):
        digest.update(ct.string_at(trainer.buffers[name], trainer.state.plan.parameters * 4))
    digest.update(struct.pack("<Qdd", trainer.state.step, trainer.state.beta1_power,
                              trainer.state.beta2_power))  # fmt: skip
    return digest.hexdigest()


def thread_states(
    path,
    bench=THREADS_BENCH,
    steps=3,
    counts=THREAD_COUNTS,
    *,
    contract="v1",
    refusals=False,
    huge_pages=True,
) -> dict[int, str]:
    """The state digest after `steps` steps of `bench` -- the same seeded weights, tokens and
    rates -- with the loss and the clipping norm every step reported folded in (the norm is the
    one reduction a partition could reach, NDT-REDUCE, and the float rounding of the update
    absorbs its last bits for a while, so the state alone would not see a chunk that followed the
    thread count), on a trainer of each thread count in `counts` (one thread is the serial
    trainer); each trainer's pool is released before the next is made. With `refusals`, a trainer
    that refuses a step is a state of its own -- "refused: " and the trainer's report -- where it
    would otherwise raise: the row's verdict on a library whose pool refuses what the serial
    trainer admits (a dropped task's finiteness report since NDT-SCAN), never a traceback.
    `huge_pages` is the trainers' state placement (NDT-PAGES), which no state may depend on."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder

    spec, batch, time_ = bench
    n = batch * time_
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n * steps)]
    targets = [(7 * i + 4) % spec.vocab_size for i in range(n * steps)]
    rates = [1e-3 * (1 - 0.1 * s) for s in range(steps)]
    out = {}
    for threads in counts:
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                threads=threads, contract=contract,
                                huge_pages=huge_pages)  # fmt: skip
        try:
            try:
                events = trainer.run(tokens, targets, steps=steps,
                                     optimizer=NativeAdamW(lr=1e-3), learning_rates=rates)  # fmt: skip
            except ValueError as exc:
                if not (refusals and str(exc).startswith("native run stopped after ")):
                    raise
                out[threads] = f"refused: {exc}"
                continue
            digest = hashlib.sha256(state_digest(trainer).encode())
            for event in events:
                digest.update(struct.pack("<dd", event["loss"], event["grad_norm"]))
            out[threads] = digest.hexdigest()
        finally:
            trainer.close()
    return out


def threads_mismatch(
    path, bench=THREADS_BENCH, steps=3, counts=THREAD_COUNTS, contracts=("v1", "reduce")
) -> int:
    """Thread counts after the first whose state differs from the first count's, summed over
    `contracts` (v1, and the reduce contract whose chunked norm is the one reduction the pool
    could reach, NDT-REDUCE). A refused run is a state that differs, and a refused first count --
    no serial state to hold the others to -- counts once itself, so a library that refuses
    everywhere is a finding, not an agreement."""
    bad = 0
    for contract in contracts:
        states = thread_states(path, bench, steps, counts, contract=contract, refusals=True)
        first = states[counts[0]]
        bad += sum(states[t] != first for t in counts[1:]) + first.startswith("refused: ")
    return bad


# --- train.norm.reduce.mismatch (NDT-REDUCE) --------------------------------------------------


def reduce_chunk(count: int) -> int:
    """The reduce contract's chunk rule (bcir_decoder_reduce_chunk): 1,024 doubled until the
    chunks number at most 1,024 -- 1,024 up to 2^20 elements, 16,384 for the 9.84M decoder."""
    chunk = 1024
    while -(-count // chunk) > 1024:
        chunk *= 2
    return chunk


def grad_norm_squared_reference(gradients, scale=1.0, *, reduce=False) -> float:
    """The clipping norm's square under its two contracts, every operation an IEEE double one as
    the C update performs it: v1's index-order chain over every element, or the reduce contract's
    chunk chains from zero added in chunk order from zero (the chunks of `reduce_chunk`). The
    update reports the square root; `bcir_decoder_gradient_norm` reports this sum, which the row
    compares before the root rounds a difference of the chunks' order away."""
    if not reduce:
        norm = 0.0
        for x in gradients:
            g = x * scale
            norm += g * g
        return norm
    chunk, norm = reduce_chunk(len(gradients)), 0.0
    for start in range(0, len(gradients), chunk):
        part = 0.0
        for x in gradients[start : start + chunk]:
            g = x * scale
            part += g * g
        norm += part
    return norm


def grad_norm_reference(gradients, scale=1.0, *, reduce=False) -> float:
    """The clipping norm the update reports under either contract: the square's root."""
    return math.sqrt(grad_norm_squared_reference(gradients, scale, reduce=reduce))


#: The counts the chunk rule is held on, both rails: the first doubling, the record's decoder.
REDUCE_CHUNK_COUNTS = (1, 1023, 1024, 1025, 1 << 20, (1 << 20) + 1, 9_836_928, 1 << 31)


def norm_mismatch(path, bench=THREADS_BENCH, steps=3) -> int:
    """NDT-REDUCE's row: the clipping norm's square the trainer computes under v1 and under the
    reduce contract (`bcir_decoder_gradient_norm`, the update's own first pass) and the norm each
    update then reports, against the oracle's emulation of that contract from the gradients the
    backward left, bit for bit (2 * steps cases of the threads bench, each its square and its
    root), plus the counts of `REDUCE_CHUNK_COUNTS` on which the C chunk rule and the Python one
    disagree."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder

    spec, batch, time_ = bench
    n = batch * time_
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n * steps)]
    targets = [(7 * i + 4) % spec.vocab_size for i in range(n * steps)]
    lib = ct.CDLL(str(path))
    lib.bcir_decoder_reduce_chunk.argtypes = [ct.c_size_t]
    lib.bcir_decoder_reduce_chunk.restype = ct.c_size_t
    lib.bcir_decoder_gradient_norm.argtypes = [ct.c_void_p, ct.c_double, ct.POINTER(ct.c_double)]
    lib.bcir_decoder_gradient_norm.restype = ct.c_int
    bad = sum(lib.bcir_decoder_reduce_chunk(c) != reduce_chunk(c) for c in REDUCE_CHUNK_COUNTS)
    for contract in ("v1", "reduce"):
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                contract=contract)  # fmt: skip
        try:
            count = trainer.state.plan.parameters
            for step in range(steps):
                at = slice(step * n, (step + 1) * n)
                trainer.backward(tokens[at], targets[at])
                want = grad_norm_squared_reference(trainer.buffers["gradients"][:count],
                                                   reduce=contract == "reduce")  # fmt: skip
                squared = ct.c_double(-1.0)
                rc = lib.bcir_decoder_gradient_norm(ct.byref(trainer.state), 1.0,
                                                    ct.byref(squared))  # fmt: skip
                bad += rc != 0 or squared.value != want
                bad += trainer.update(NativeAdamW(lr=1e-3)) != math.sqrt(want)
        finally:
            trainer.close()
    return bad


# --- every row --------------------------------------------------------------------------------


def corpus_unselected() -> int | None:
    """TRAIN-RAIL's row, `train.corpus.unselected`: the tracked Markdown files of docs/ and
    training/ that the train rail's recipe does not select, read by tools/models/train_rail.py
    without a training run; None outside a source checkout (the wheel ships neither tools/ nor
    the trees), where the row is not measured rather than read as zero."""
    root = Path(__file__).resolve().parents[2]
    tool = root / "tools" / "models" / "train_rail.py"
    if not tool.is_file() or not (root / "training").is_dir() or not (root / "docs").is_dir():
        return None
    spec = importlib.util.spec_from_file_location("bcir_tools_train_rail", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    selected = module.selected_by_recipe(root, module.DEFAULT_RECIPE)
    return len(module.unselected(root, selected))


def measure(*, timed: bool = True) -> dict[str, float]:
    """Every NDT-GEM row this host can measure: the memory row always, the corpus row in a source
    checkout, the native rows with a compiler (and the timed one only when `timed`)."""
    ratio = transient_ratio()[0]
    out = {"train.memory.transient": ratio}
    unselected = corpus_unselected()
    if unselected is not None:
        out["train.corpus.unselected"] = float(unselected)
    tmp = tempfile.mkdtemp(prefix="bcir-ndt-")
    try:
        path = build_library(tmp)
        if path is None:
            return out
        lib = ct.CDLL(str(path))
        out["train.plan.mismatch"] = float(plan_mismatch(lib))
        out["train.mm.mismatch"] = float(mm_mismatch(lib))
        out["train.mm.fused.mismatch"] = float(mm_fused_mismatch(lib))
        out["train.mm.kblock.mismatch"] = float(mm_kblock_mismatch(lib))
        provider = provider_mismatch(lib)
        if provider is not None:
            out["train.mm.provider.differences"] = float(provider)
            out["train.mm.provider.unsound"] = float(provider_unsound(path))
        swept, mismatch, ulp = exp_sweep(lib)
        assert swept > 8_000_000, swept
        out["train.exp.mismatch"] = float(mismatch)
        out["train.exp.ulp.max"] = float(ulp)
        out["train.state.threads.mismatch"] = float(threads_mismatch(path))
        out["train.norm.reduce.mismatch"] = float(norm_mismatch(path))
        if timed:
            out.update(measure_step(path))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


__all__ = [
    "BENCH",
    "MM_CASES",
    "PLAN_CORPUS",
    "THREAD_COUNTS",
    "THREADS_BENCH",
    "build_library",
    "corpus_unselected",
    "exp_sweep",
    "measure",
    "measure_step",
    "mm_fused_mismatch",
    "mm_kblock_mismatch",
    "MM_KBLOCK_CASES",
    "kblock_split",
    "mm_reference_kblock",
    "provider_mismatch",
    "provider_unsound",
    "openblas_library",
    "mm_mismatch",
    "mm_variants",
    "mm_reference",
    "mm_reference_fused",
    "plan_mismatch",
    "round_f32",
    "source",
    "state_digest",
    "thread_states",
    "threads_mismatch",
    "reduce_chunk",
    "grad_norm_reference",
    "grad_norm_squared_reference",
    "REDUCE_CHUNK_COUNTS",
    "norm_mismatch",
    "transient_ratio",
]


if __name__ == "__main__":  # pragma: no cover - a manual probe
    print(measure())
    os._exit(0)
