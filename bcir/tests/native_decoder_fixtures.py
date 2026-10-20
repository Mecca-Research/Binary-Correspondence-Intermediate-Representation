"""The NDT-GEM fixtures and grader: the native training step held to its GEM+ program.

`bcir.hosted.models.native_program.training_step_program` states the C trainer's step from the
spec alone. One function per row family grades it against the C rail; the tests, the harness
(`tools/perf/gemplus_baseline.py --group train`) and the fault table call them:

    train.plan.mismatch        (spec, arena) pairs over `plan_corpus()` where the program's arena
                               sum differs from `bcir_decoder_make_plan`'s count
    train.plan.remat.mismatch  (spec, recompute set) pairs over the same corpus and `REMAT_SETS`
                               (cut to each spec's layers, every distinct set once) where the
                               program's activation sum under the set differs from
                               `bcir_decoder_activation_count` (REMAT, GEO-3)
    train.mm.mismatch         `MM_CASES` whose native `bcir_tensor_mm` output differs in any bit
                               from `mm_reference` -- an independent binary32 emulation of the
                               kernel's contract (beta first, then alpha*a times b added for k
                               ascending, every operation rounded, no contraction)
    train.memory.transient     what a trainer lends for the bench step's activation and scratch
                               values placed by their lifetimes (PX-4, `placed_transient`), over
                               the G26 concurrent-live bound of the values it places (exact; no
                               compiler); its frozen baseline the C plan's spans over the step
                               program's bound (`transient_ratio`)
    train.step.ms              one native training step at `BENCH` (wall), and its floor in the
                               same run: the step's counted work at the best rate the same kernel
                               reaches on any of the step's own GEMM shapes, timed alone
                               (`NativeProgram.roofline`, compute term) -- a bound on this host's
                               kernel, TMSAO-3, never on the silicon
    train.mm.fused.mismatch    `MM_CASES` whose native `bcir_tensor_mm_fused` output differs in any bit
                               from `mm_reference_fused` -- the fused contract (v2, NDT-FMA) as an
                               exact rational emulation rounded once per term
    train.mm.bf16.mismatch     `MM_BF16_CASES` whose output through any bf16 instance the host runs
                               (portable, AVX-512, AMX admitted) differs in any bit from
                               `mm_reference_bf16` -- the bf16 contract (NDT-BF16) as exact
                               rationals: operands rounded to bf16, 32-term blocks of two lanes
                               from +0, the flushes where the AMX unit flushes
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
    train.plan.executed.mismatch  cases of `trace_corpus()` whose launch trace (a build defining
                               BCIR_DT_TRACE_PROBE) differs from the launches the step program
                               states under the case's decisions (`native_launch.run_launches`),
                               or dropped a record (PX-1)
    train.plan.program.mismatch  program runs (`program_runs`: every trace case from the program
                               of the C step's decisions, the state cases under each of the
                               planner's other choices) that stopped, trained another state or
                               events than `bcir_decoder_run_accum` on the case, or traced other
                               launches than the program states (PX-2)
    train.plan.pack.mismatch   pack runs (`pack_runs`: the program runs' configurations, each as a
                               StreamPack and ExecutionPlanV1 the trainer lowers) whose lowering was
                               refused or is not the program's words, that stopped, or that trained
                               another state or events than the C step on the case (PX-2b)
    train.plan.place.mismatch  the same runs with every transient value placed by its lifetime and
                               run on a trainer lending exactly the placement's arena
                               (`place_runs`): those refused, lowered to other words than the
                               program's at the placed places, stopped, placed below G26's bound
                               (live values sharing bytes), or training another state or events
                               than the C step (PX-4)

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
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple
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
#: binary32 product does not scale exactly -- in one K block and across three, since the first
#: block's epilogue (which scales c, NDT-BETA) and a later block's are each their own code -- and
#: beta 1 (the trainer's dW accumulations).
MM_KBLOCK_CASES = MM_CASES + tuple(
    (ta, tb, m, n, k, alpha, beta)
    for ta in (0, 1)
    for tb in (0, 1)
    for (m, n, k), alpha, beta in (
        ((2, 3, 1000), 1.0, 1.0),
        ((1, 2, 2048), 1.0, 0.0),
        ((3, 2, 449), 1.0, 1.0),
        ((2, 2, 897), 1.1, 1.0),
        ((2, 2, 100), 1.1, 1.0),
    )
)

#: The step the threads row trains (NDT-THREADS): 80 tokens (one 64-row GEMM task and a 16-row
#: tail), T=40 (past the attention backward's 32-row chunks), GQA groups of two, a width, a KV
#: width, an FF width and a vocabulary off every 32- and 64-wide block edge; three steps.
THREADS_BENCH = (DecoderSpec(250, 72, 4, 2, 200, activation="silu_gate", n_kv_heads=2), 2, 40)
#: One reduce chunk (728 parameters, tied): the clipping norm under the reduce contract is the
#: chunk loop's chain alone -- on a vector host the update pairs whole chunks (NDT-UPDATE), and the
#: threads bench's 133 chunks reach that loop only in their short last one.
ONE_CHUNK_BENCH = (DecoderSpec(16, 8, 2, 1, 16, activation="silu_gate", n_kv_heads=1), 2, 8)
#: A step whose SwiGLU block admits the bf16 panels (BF16-PANELS): 32 rows (two sequences of 16), a
#: 64-wide model over a 128-wide hidden activation -- every GEMM the block's outputs feed whole
#: 32-tiles -- so under a bf16 contract the trainer writes the hidden activation and the gate and up
#: adjoints as the GEMMs' panels wherever their slots are 64-byte aligned (the huge-page
#: placement's are; a ctypes array's alignment is the allocator's).
PANELS_BENCH = (DecoderSpec(64, 64, 2, 2, 128, activation="silu_gate", n_kv_heads=1), 2, 16)
#: GEO-3 (REMAT, WS-PLAN): four layers, so a recompute set keeps some slots and shares one, on the
#: SwiGLU panels' grid (64 rows, width 64, FF 128), so a bf16 workspace has three tiers.
REMAT_BENCH = (DecoderSpec(97, 64, 4, 4, 128, activation="silu_gate", n_kv_heads=2), 2, 32)
#: The recompute sets the parity and the state witnesses take: none, every layer, every other one,
#: the highest alone (no slot freed), and all but the highest.
REMAT_SETS = (0, 0xF, 0x5, 0x8, 0x7)
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
    from bcir.toolchain import build_units, host_link_args

    cmd = [cc, "-std=c11", "-O3", "-ffp-contract=off", "-DBCIR_DT_BUILD_SHARED", "-shared"]
    if sys.platform != "win32":
        cmd += ["-fPIC", "-pthread"]
    cmd += ["-I", str(_C), *units, "-o", str(out)]
    # a copy the compiler refuses is BuildFailed (check_tests.py's `build` finding), never a test's
    build_units(cmd + host_link_args(["-lm"]), "native decoder build of a copy", timeout=180)
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


def native_activation_count(lib, spec: DecoderSpec, batch: int, time_: int, recompute: int) -> int:
    """`bcir_decoder_activation_count` for the plan of `spec` under a recompute set (REMAT)."""
    from bcir.hosted.models.native import _Plan, _Spec

    shape = _Spec(spec.vocab_size, spec.d_model, spec.n_heads, spec.kv_heads, spec.n_layers,
                  spec.d_ff, batch, time_, int(spec.tied_embeddings), spec.rope_base,
                  spec.rms_norm_eps)  # fmt: skip
    plan = _Plan()
    make, count = lib.bcir_decoder_make_plan, lib.bcir_decoder_activation_count
    make.argtypes, make.restype = [ct.POINTER(_Spec), ct.POINTER(_Plan)], ct.c_int
    count.argtypes, count.restype = [ct.POINTER(_Plan), ct.c_uint64], ct.c_size_t
    assert make(ct.byref(shape), ct.byref(plan)) == 0, (spec, batch, time_)
    return count(ct.byref(plan), recompute)


def recompute_plan_mismatch(lib, corpus=PLAN_CORPUS, sets=REMAT_SETS) -> int:
    """(spec, recompute set) pairs where the program's activation sum under the set differs from
    `bcir_decoder_activation_count` (REMAT): the sets cut to each spec's layers, every distinct
    one counted once."""
    bad = 0
    for spec, batch, time_ in corpus:
        for recompute in sorted({mask & ((1 << spec.n_layers) - 1) for mask in sets}):
            mine = training_step_program(spec, batch, time_, recompute).arena_elements()
            bad += mine["activations"] != native_activation_count(
                lib, spec, batch, time_, recompute
            )
    return bad


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


# --- train.mm.bf16.mismatch (NDT-BF16) ----------------------------------------------------------

#: The bf16 contract's cases: the fused cases and depths across one, two and five 32-term blocks,
#: an inexact alpha (A is scaled before its bf16 rounding) and beta 1.
MM_BF16_CASES = MM_CASES + tuple(
    (ta, tb, m, n, k, alpha, beta)
    for ta in (0, 1)
    for tb in (0, 1)
    for (m, n, k), alpha, beta in (
        ((2, 3, 160), 1.0, 1.0),
        ((17, 2, 96), 1.1, 0.0),
    )
)


def _round_bits(x: Fraction, p: int) -> Fraction:
    """The nonzero rational x rounded to `p` significant bits, ties to even, the exponent unbounded."""
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if a < Fraction(2) ** e:
        e -= 1  # 2^e <= a < 2^(e+1)
    quantum = Fraction(2) ** (e - p + 1)
    scaled = a / quantum
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2):
        whole += 1
    v = whole * quantum
    return v if x > 0 else -v


def bf16_round(x: float) -> float:
    """The bf16 contract's operand rounding of a finite binary32: a subnormal or zero to its zero,
    else eight significant bits to nearest, ties to even, past the largest finite to infinity."""
    if x == 0.0 or abs(x) < 2.0**-126:
        return math.copysign(0.0, x)
    v = _round_bits(Fraction(x), 8)
    return math.copysign(math.inf, x) if abs(v) >= 2**128 else float(v)


_TINY = Fraction(2) ** -126


def _lane(e: float, a: float, b: float) -> float:
    """One step of a bf16 lane: e + a*b exactly, the binary32 nearest it -- or the zero of its sign
    when its 24-bit rounding with the exponent unbounded is below 2^-126 (tiny after rounding: the
    flush MXCSR's FTZ and the AMX unit apply)."""
    x = Fraction(e) + Fraction(a) * Fraction(b)
    if x == 0:  # an exact zero: -0 only from -0 plus a -0 product
        product_negative = (math.copysign(1.0, a) < 0) != (math.copysign(1.0, b) < 0)
        return -0.0 if math.copysign(1.0, e) < 0 and product_negative else 0.0
    if abs(_round_bits(x, 24)) < _TINY:
        return -0.0 if x < 0 else 0.0
    return round_f32(x)


def _flush(x: float) -> float:
    return math.copysign(0.0, x) if x != 0.0 and abs(x) < 2.0**-126 else x


def mm_reference_bf16(ta, tb, m, n, k, alpha, a, b, beta, c):
    """The bf16 contract (NDT-BF16) in exact rationals: beta first (the rounded product, zero under
    beta 0); every operand rounded to bf16 -- A after its binary32 product with alpha; the depth in
    blocks of 32 from l = 0, padded with +0 terms; per block an even and an odd lane from +0, each
    term's fma rounded once and flushed when tiny, then c = daz(c) + (E + O), both sums binary32
    operations flushed when subnormal."""
    alpha, beta = _f32(alpha), _f32(beta)
    out = [0.0 if beta == 0.0 else _f32(beta * x) for x in c]
    ra = [bf16_round(_f32(alpha * v)) for v in a]
    rb = [bf16_round(v) for v in b]
    for i in range(m):
        for j in range(n):
            acc = out[i * n + j]
            for l0 in range(0, k, 32):
                lanes = [0.0, 0.0]
                for l in range(l0, l0 + 32):
                    av = (ra[l * m + i] if ta else ra[i * k + l]) if l < k else 0.0
                    bv = (rb[j * k + l] if tb else rb[l * n + j]) if l < k else 0.0
                    lanes[l % 2] = _lane(lanes[l % 2], av, bv)
                acc = _flush(_f32(_flush(acc) + _flush(_f32(lanes[0] + lanes[1]))))
            out[i * n + j] = acc
    return out


def mm_bf16_variants(lib) -> list[int]:
    """The bf16 contract's instances this host runs (1 portable, 4 AVX-512, 8 AMX admitted)."""
    fn = lib.bcir_tensor_mm_bf16_variants
    fn.argtypes, fn.restype = [], ct.c_int
    mask = fn()
    return [v for v in (1, 4, 8) if mask & v]


def mm_bf16_mismatch(lib, cases=MM_BF16_CASES) -> int:
    """`train.mm.bf16.mismatch`: of `cases`, through every bf16 instance the host runs with the
    workspace whole, those whose output differs in any bit from `mm_reference_bf16` -- a case once
    per instance that differs."""
    fn = lib.bcir_tensor_mm_bf16_variant
    fn.argtypes = [ct.c_int] + _MM_SIG[1:] + [ct.c_void_p, ct.c_size_t]
    fn.restype = ct.c_int
    size = lib.bcir_tensor_mm_bf16_workspace
    size.argtypes, size.restype = [ct.c_size_t] * 3, ct.c_size_t
    bad = 0
    for seed, (ta, tb, m, n, k, alpha, beta) in enumerate(cases, 1):
        a, b, c = _operands(m, n, k, seed)
        want = struct.pack(f"<{m * n}f", *mm_reference_bf16(ta, tb, m, n, k, alpha, a, b, beta, c))
        bytes_ = size(m, n, k)
        workspace = (ct.c_ubyte * max(bytes_, 1))()
        for variant in mm_bf16_variants(lib):
            ca, cb = (ct.c_float * len(a))(*a), (ct.c_float * len(b))(*b)
            cc = (ct.c_float * len(c))(*c)
            ran = fn(variant, ta, tb, m, n, k, alpha, ca, cb, beta, cc, workspace, bytes_)
            bad += ran != variant or want != struct.pack(f"<{m * n}f", *cc)
    return bad


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
    """(C bytes / bound, C bytes, the G26 bound, GEM+'s incumbent layout) for the bench step: the
    C plan's fixed spans, the row's frozen baseline."""
    program = training_step_program(*bench)
    held, bound = program.transient_bytes(), program.transient_lower_bound()
    return held / bound, held, bound, program.transient_incumbent().extent


def placed_transient(bench=BENCH) -> tuple[float, int, int, bool]:
    """PX-4: (placed bytes / bound, the placed transient arena's bytes, G26's bound over the values
    it places, whether every part's layout is a proved optimum) for the bench step's pack, its
    transient values placed by their lifetimes (`native_pack.place_by_lifetime`): what a trainer
    lends for them, the row `train.memory.transient` reads."""
    from bcir.hosted.models.native_pack import place_by_lifetime, training_pack

    placed = place_by_lifetime(training_pack(*bench, contract=0))
    return placed.extent / placed.bound, placed.extent, placed.bound, placed.optimal


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
    recompute=0,
    workspace_bytes=None,
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
    `huge_pages` is the trainers' state placement (NDT-PAGES), which no state may depend on, nor
    may `recompute` and `workspace_bytes` (GEO-3: the recompute set, the bf16 workspace)."""
    from bcir.hosted.models.native import NativeAdamW, NativeDecoder

    spec, batch, time_ = bench
    n = batch * time_
    tokens = [(7 * i + 3) % spec.vocab_size for i in range(n * steps)]
    targets = [(7 * i + 4) % spec.vocab_size for i in range(n * steps)]
    rates = [1e-3 * (1 - 0.1 * s) for s in range(steps)]
    out = {}
    for threads in counts:
        trainer = NativeDecoder(path, spec, batch_size=batch, context_length=time_,
                                threads=threads, contract=contract, huge_pages=huge_pages,
                                recompute=recompute, workspace_bytes=workspace_bytes)  # fmt: skip
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


# --- train.plan.executed.mismatch (PX-1) ----------------------------------------------------------

#: The trace corpus's contracts: every GEMM entry BCIR has (v1, fused, blocked, bf16), the
#: exponential, the reduce pass, and the fused and bf16 attention.
TRACE_CONTRACTS = (
    "v1", "fused+exp", "kblock+exp+reduce", "bf16", "bf16+exp+attnbf16", "kblock+attn",
)  # fmt: skip


@dataclass(frozen=True)
class TraceCase:
    """One `bcir_decoder_run_accum` call to trace: a trainer and the call's shape."""

    spec: DecoderSpec
    batch: int
    time: int
    contract: str
    recompute: int = 0
    workspace: str = "need"  # the bf16 workspace's tier: "need", "panels" or "floor"
    threads: int = 1
    steps: int = 1
    accumulation: int = 1
    provider: str = ""  # "", "mm" (a CBLAS sgemm) or "mm+silu" (and a SiLU)
    preflight: bool = False  # an update whose bound fails (epsilon 1e-45), so the preflight runs


def trace_corpus() -> tuple[TraceCase, ...]:
    """Every spec of the plan corpus under the six contracts, three recompute sets (none, every
    layer, the lowest) and two call shapes (one step; two steps of two micro-batches -- FACTS-RUN's
    skipped scan and the accumulation's scans); the panels' bench under both bf16 contracts at the
    workspace's three tiers, at one thread and four; a provider's GEMM, and its GEMM and SiLU,
    under v1 and the blocked contract; and an update whose bound fails."""
    cases = []
    for spec, batch, time_ in PLAN_CORPUS:
        for contract in TRACE_CONTRACTS:
            for recompute in dict.fromkeys((0, (1 << spec.n_layers) - 1, 1)):
                for steps, accumulation in ((1, 1), (2, 2)):
                    cases.append(TraceCase(spec, batch, time_, contract, recompute,
                                           steps=steps, accumulation=accumulation))  # fmt: skip
    spec, batch, time_ = PANELS_BENCH
    for contract in ("bf16", "bf16+exp+attnbf16"):
        for workspace in ("need", "panels", "floor"):
            for threads in (1, 4):
                cases.append(TraceCase(spec, batch, time_, contract, 0b10, workspace, threads))
    spec, batch, time_ = PLAN_CORPUS[1]  # untied: the provider's zero fill and the head's own VJP
    for contract in ("v1", "kblock+exp"):
        for provider in ("mm", "mm+silu"):
            cases.append(TraceCase(spec, batch, time_, contract, provider=provider, steps=2,
                                   accumulation=2))  # fmt: skip
    cases.append(TraceCase(spec, batch, time_, "v1", preflight=True))
    return tuple(cases)


def checkout_library(directory) -> Path | None:
    """The trainer as `build_library` builds it, or None without a compiler or outside a source
    checkout (an installed package ships no C units: the one absence that is expected there)."""
    if not (_C / "bcir_decoder_train.c").is_file():
        return None
    return build_library(directory)


def trace_library(directory) -> Path | None:
    """The trainer built with its launch trace (BCIR_DT_TRACE_PROBE), from the tree's unit as it
    stands -- a fault the sweep injected included. None without a compiler, or outside a source
    checkout (an installed package ships no C units: the one absence that is expected there)."""
    if not (_C / "bcir_decoder_train.c").is_file():
        return None
    unit = "#define BCIR_DT_TRACE_PROBE 1\n" + source("bcir_decoder_train.c")
    return build_library(directory, sources={"bcir_decoder_train.c": unit})


class TraceRun(NamedTuple):
    """One traced call: the launches its plan states, the records the probe held (launch, record),
    the records it had no room for, the state the call left (`state_digest`) and its events."""

    planned: list
    traced: list
    dropped: int
    state: str
    events: tuple = ()


def _case_trainer(path, case: TraceCase, transient=None):
    """The trainer `case` runs on -- its contract, recompute set, threads, workspace tier and
    provider; its transient arenas the C plan's, or as `transient` lends them (PX-4) -- and the
    state's decisions as `run_launches` and `step_program` take them."""
    from bcir.hosted.models import native as nat
    from bcir.hosted.models.native import NativeDecoder
    from bcir.hosted.models.native_launch import SPACES

    common = dict(batch_size=case.batch, context_length=case.time, contract=case.contract,
                  recompute=case.recompute, threads=case.threads)  # fmt: skip
    workspace = None
    if case.workspace != "need":
        probe = NativeDecoder(path, case.spec, **common)
        memory = probe.plan_memory
        probe.close()
        # the panels' tier where it lies at or above the floor (memory_plan's tiers): under the
        # bf16 attention the floor holds its regions, which can exceed what the panel groups pack
        workspace = memory.workspace_floor
        if case.workspace == "panels":
            workspace = max(memory.workspace_panels, workspace)
    trainer = NativeDecoder(path, case.spec, workspace_bytes=workspace, transient=transient,
                            **common)  # fmt: skip
    try:
        lib = trainer.lib
        if case.provider:
            gemm = lib.bcir_tensor_mm
            gemm.argtypes = [ct.c_void_p, ct.c_int, ct.c_int, ct.c_size_t, ct.c_size_t, ct.c_size_t,
                             ct.c_float, _FP, _FP, ct.c_float, _FP]  # fmt: skip
            gemm.restype = None
            callback = nat._CBLAS(
                lambda order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc: gemm(
                    None, ta == 112, tb == 112, m, n, k, alpha, a, b, beta, c
                )
            )
            trainer.use_cblas_function(callback, name="trace", admit=False)
            if case.provider == "mm+silu":
                silu = ct.cast(lib.bcir_tensor_silu, nat._SILU)
                provider = trainer.state.provider
                trainer.state.provider = nat._Provider(provider.ctx, provider.mm, silu)
                trainer._provider_refs += (silu,)
        decisions = dict(
            contract=nat._CONTRACTS[case.contract], recompute=trainer.recompute,
            workspace_bytes=trainer.workspace_bytes,
            panels_need=trainer.plan_memory.workspace_panels,
            alignment={SPACES[name]: ct.addressof(buf) % 64 for name, buf in trainer.buffers.items()},
            provider_mm=bool(case.provider), provider_silu=case.provider == "mm+silu",
            steps=case.steps, accumulation=case.accumulation,
        )  # fmt: skip
    except BaseException:
        trainer.close()
        raise
    return trainer, decisions


def _case_inputs(case: TraceCase):
    """The case's tokens, targets and optimizer (an update whose bound fails where it asks)."""
    from bcir.hosted.models.native import NativeAdamW

    n = case.batch * case.time * case.steps * case.accumulation
    tokens = [(7 * i + 3) % case.spec.vocab_size for i in range(n)]
    targets = [(5 * i + 1) % case.spec.vocab_size for i in range(n)]
    return tokens, targets, NativeAdamW(epsilon=1e-45) if case.preflight else NativeAdamW()


def _probe(lib):
    from bcir.hosted.models.native_launch import RECORD_WORDS

    lib.bcir_decoder_trace_probe_begin.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    lib.bcir_decoder_trace_probe_begin.restype = None
    lib.bcir_decoder_trace_probe_end.argtypes = [ct.POINTER(ct.c_size_t)]
    lib.bcir_decoder_trace_probe_end.restype = ct.c_size_t
    lib.bcir_decoder_trace_probe_words.argtypes = []
    lib.bcir_decoder_trace_probe_words.restype = ct.c_size_t
    assert lib.bcir_decoder_trace_probe_words() == RECORD_WORDS


def trace_case(path, case: TraceCase, capacity: int | None = None, *, trace: bool = True):
    """Run `case` on the trace build at `path` (a `TraceRun`). `capacity` lends the probe that
    many records instead of room for twice the plan's; `trace=False` runs `path` as a build
    without the probe -- the library itself -- and holds no records."""
    from bcir.hosted.models.native_launch import RECORD_WORDS, read_trace, run_launches

    trainer, decisions = _case_trainer(path, case)
    try:
        lib = trainer.lib
        if trace:
            _probe(lib)
        planned = run_launches(case.spec, case.batch, case.time, **decisions)
        if capacity is None:
            capacity = 2 * sum(len(launch.records) for launch in planned) + 64
        words = (ct.c_uint64 * (capacity * RECORD_WORDS))()
        tokens, targets, optimizer = _case_inputs(case)
        if not trace:
            events = trainer.run(tokens, targets, steps=case.steps, optimizer=optimizer,
                                 gradient_accumulation=case.accumulation)  # fmt: skip
            return TraceRun(planned, [], 0, state_digest(trainer), events)
        lib.bcir_decoder_trace_probe_begin(ct.addressof(trainer.state), ct.addressof(words),
                                           capacity)  # fmt: skip
        try:
            events = trainer.run(tokens, targets, steps=case.steps, optimizer=optimizer,
                                 gradient_accumulation=case.accumulation)  # fmt: skip
        finally:
            dropped = ct.c_size_t()
            count = lib.bcir_decoder_trace_probe_end(ct.byref(dropped))
        return TraceRun(planned, read_trace(words, count), dropped.value, state_digest(trainer),
                        events)  # fmt: skip
    finally:
        trainer.close()


class ProgramRun(NamedTuple):
    """One program run: the program, the records its run traced (launch, record), the records it
    had no room for, the state it left, its events, and its status (0, or the C's refusal)."""

    program: list
    traced: list
    dropped: int
    state: str
    events: tuple
    status: int


def program_case(path, case: TraceCase, choices: dict | None = None) -> ProgramRun:
    """Run `case` from its program (`native_launch.step_program` under the case's state and the
    planner's `choices`) on the trace build at `path`, its launches traced."""
    from bcir.hosted.models.native import NativeRunStopped
    from bcir.hosted.models.native_launch import (
        RECORD_WORDS,
        program_words,
        read_trace,
        step_program,
    )

    trainer, decisions = _case_trainer(path, case)
    try:
        lib = trainer.lib
        _probe(lib)
        program = step_program(
            case.spec, case.batch, case.time, **(decisions | dict(choices or {}))
        )
        capacity = 2 * sum(len(launch.records) for launch in program) + 64
        words = (ct.c_uint64 * (capacity * RECORD_WORDS))()
        tokens, targets, optimizer = _case_inputs(case)
        events, status = (), 0
        lib.bcir_decoder_trace_probe_begin(ct.addressof(trainer.state), ct.addressof(words),
                                           capacity)  # fmt: skip
        try:
            events = trainer.run_program(program_words(program), tokens, targets,
                                         steps=case.steps, optimizer=optimizer,
                                         gradient_accumulation=case.accumulation)  # fmt: skip
        except NativeRunStopped as stopped:
            status = stopped.status
        finally:
            dropped = ct.c_size_t()
            count = lib.bcir_decoder_trace_probe_end(ct.byref(dropped))
        return ProgramRun(program, read_trace(words, count), dropped.value, state_digest(trainer),
                          events, status)  # fmt: skip
    finally:
        trainer.close()


def executed_mismatch(path, cases=None) -> tuple[int, list[str]]:
    """`train.plan.executed.mismatch`: the cases of `cases` (the trace corpus) whose trace differs
    from their planned launches, or dropped a record; and what differed, case by case."""
    from bcir.hosted.models.native_launch import launch_mismatches

    bad, notes = 0, []
    for case in trace_corpus() if cases is None else cases:
        run = trace_case(path, case)
        problems = launch_mismatches(run.planned, run.traced)
        if run.dropped:
            problems.append(f"{run.dropped} records dropped")
        if problems:
            bad += 1
            notes.append(f"{case}: {problems[0]}")
    return bad, notes


# --- train.plan.program.mismatch (PX-2) -----------------------------------------------------------


def state_cases(corpus) -> list[int]:
    """One case of the corpus per decision set the state could depend on -- contract, recompute
    or not, workspace tier, threads, provider, a failing bound -- its multi-step one where the
    corpus has it."""
    picked: dict[tuple, int] = {}
    for i, case in enumerate(corpus):
        key = (case.contract, bool(case.recompute), case.workspace, case.threads, case.provider,
               case.preflight)  # fmt: skip
        if key not in picked or case.steps > corpus[picked[key]].steps:
            picked[key] = i
    return sorted(picked.values())


def program_choices(case: TraceCase) -> tuple[dict, ...]:
    """The planner's choices a program takes otherwise than the C step does (PX-2): the linears
    alone (no group), a linear's VJPs as two calls (no pair), the SwiGLU block without its panels,
    every scan the run's facts leave out, and every layer recomputed (one cache slot: the set every
    state's arena holds)."""
    return ({"group": False}, {"pair": False}, {"panels": False}, {"facts": False},
            {"recompute": (1 << case.spec.n_layers) - 1})  # fmt: skip


def program_runs(path, cases=None):
    """Every program run `train.plan.program.mismatch` reads, as (case, choices, the C step's
    `TraceRun`, the program's `ProgramRun`): every case of `cases` (the trace corpus) from the
    program that takes the C step's own decisions, and its state cases under each of
    `program_choices`."""
    corpus = trace_corpus() if cases is None else tuple(cases)
    picked = set(state_cases(corpus))
    for i, case in enumerate(corpus):
        step = trace_case(path, case)
        yield case, {}, step, program_case(path, case)
        if i in picked:
            for choices in program_choices(case):
                yield case, choices, step, program_case(path, case, choices)


def program_run_count(cases=None) -> int:
    """How many runs `program_runs` makes of `cases` (the trace corpus): the row's baseline."""
    corpus = trace_corpus() if cases is None else tuple(cases)
    return len(corpus) + sum(len(program_choices(corpus[i])) for i in state_cases(corpus))


def program_problems(step: TraceRun, run: ProgramRun) -> list[str]:
    """What a program run did otherwise than it must: stopped, dropped a record, traced other
    launches than its program states, or left another state or events than the C step's `step`
    on the same case. Empty when the run is the step its program states."""
    from bcir.hosted.models.native_launch import launch_mismatches, program_launches

    problems = launch_mismatches(program_launches(run.program), run.traced)
    if run.status:
        problems.insert(0, f"stopped with status {run.status}")
    if run.dropped:
        problems.append(f"{run.dropped} records dropped")
    if run.state != step.state or run.events != step.events:
        problems.append("the state or the events are not the C step's")
    return problems


def program_mismatch(path, cases=None) -> tuple[int, int, list[str]]:
    """`train.plan.program.mismatch`: of the program runs (`program_runs`), those that stopped,
    whose state or events differ from the C step's on the same case (`bcir_decoder_run_accum`), or
    whose trace differs from the launches its program states or dropped a record; the runs; and
    what differed, run by run."""
    bad, runs, notes = 0, 0, []
    for case, choices, step, run in program_runs(path, cases):
        runs += 1
        problems = program_problems(step, run)
        if problems:
            bad += 1
            notes.append(f"{case} {choices}: {problems[0]}")
    return bad, runs, notes


# --- train.plan.pack.mismatch (PX-2b) ------------------------------------------------------------


def pack_library(directory) -> Path | None:
    """The trainer with the PX-2b pack lowering (`NativeDecoder.build(..., pack=True)`), from the
    tree's units as they stand -- a fault the sweep injected included. None without a compiler, or
    outside a source checkout (an installed package ships no C units: the one expected absence)."""
    if not (_C / "bcir_decoder_pack.c").is_file():
        return None
    cc = compiler()
    if cc is None:
        return None
    from bcir.hosted.models.native import NativeDecoder

    return NativeDecoder.build(directory, cc=cc, pack=True)


class PackRun(NamedTuple):
    """One pack run: the training pack, the words its lowering wrote, the state the run left, its
    events and its status -- 0, the run's stop status, or -1 when the lowering refused the pack
    (`why` then its reason, `native_pack.REASONS`) -- and, placed by lifetime (PX-4), the
    `native_pack.Placed` it ran on."""

    pack: object
    words: tuple
    state: str
    events: tuple
    status: int
    why: int = 0
    placed: object = None


def pack_case(path, case: TraceCase, choices: dict | None = None, *, placements=None,
              run: bool = True, place: bool = False) -> PackRun:  # fmt: skip
    """`case` from its pack (PX-2b): the program `step_program` states for the case's state and the
    planner's `choices`, stated as a module (`native_pack.training_pack`), planned, hydrated and
    placed (`pack_artifacts`: the C step's own places, or `placements`), lowered by the trainer of
    the case on the pack build at `path` (`bcir_decoder_lower_pack`) and, with `run`, run. With
    `place` (PX-4) the transient values are placed by their lifetimes (`place_by_lifetime`) and the
    pack lowered and run on a trainer of the case that lends exactly the placement's arena."""
    from bcir.hosted.models.native import NativeRunStopped, PackRefused
    from bcir.hosted.models.native_pack import pack_artifacts, place_by_lifetime, training_pack

    trainer, decisions = _case_trainer(path, case)
    try:
        tp = training_pack(case.spec, case.batch, case.time, **(decisions | dict(choices or {})))
        placed = place_by_lifetime(tp) if place else None
        if placed is not None:
            placements = placed.placements
            trainer.close()
            trainer, _ = _case_trainer(path, case, transient=(placed.floats, 0))
        pack, plan, _, _ = pack_artifacts(tp, placements=placements)
        try:
            words = trainer.lower_pack(pack, plan)
        except PackRefused as refused:
            return PackRun(tp, (), "", (), -1, refused.why, placed)
        if not run:
            return PackRun(tp, tuple(words), "", (), 0, 0, placed)
        tokens, targets, optimizer = _case_inputs(case)
        events, status = (), 0
        try:
            events = trainer.run_program(words, tokens, targets, steps=case.steps,
                                         optimizer=optimizer,
                                         gradient_accumulation=case.accumulation)  # fmt: skip
        except NativeRunStopped as stopped:
            status = stopped.status
        return PackRun(tp, tuple(words), state_digest(trainer), events, status, 0, placed)
    finally:
        trainer.close()


def pack_runs(path, cases=None):
    """Every pack run `train.plan.pack.mismatch` reads, as (case, choices, the C step's `TraceRun`
    on the same build, the `PackRun`): the configurations `program_runs` reads -- every case of
    `cases` (the trace corpus) from the program of the C step's own decisions, its state cases
    under each of `program_choices` -- each as a pack."""
    corpus = trace_corpus() if cases is None else tuple(cases)
    picked = set(state_cases(corpus))
    for i, case in enumerate(corpus):
        step = trace_case(path, case, trace=False)
        yield case, {}, step, pack_case(path, case)
        if i in picked:
            for choices in program_choices(case):
                yield case, choices, step, pack_case(path, case, choices)


def _ran_as_the_step(step: TraceRun, run: PackRun, problems: list[str]) -> list[str]:
    """`problems`, then what a lowered run did otherwise than the C step's `step` on its case: it
    stopped, or it trained another state or events -- the one check the pack and the placed runs'
    comparisons share."""
    if run.status:
        problems.append(f"stopped with status {run.status}")
    if run.state != step.state or run.events != step.events:
        problems.append("the state or the events are not the C step's")
    return problems


def pack_problems(step: TraceRun, run: PackRun) -> list[str]:
    """What a pack run did otherwise than it must: a lowering refused or other than its program's
    words, a run that stopped, or another state or events than the C step's `step`."""
    if run.status == -1:
        return [f"the lowering refused the pack (why {run.why})"]
    problems = []
    if run.words != run.pack.words:
        problems.append("the lowering is not the program's words")
    return _ran_as_the_step(step, run, problems)


def place_runs(path, cases=None):
    """PX-4: `pack_runs`' configurations, each pack placed by its values' lifetimes and lowered and
    run on a trainer lending exactly the placement's arena (`pack_case(..., place=True)`), as
    (case, choices, the C step's `TraceRun` on the full build, the placed `PackRun`)."""
    corpus = trace_corpus() if cases is None else tuple(cases)
    picked = set(state_cases(corpus))
    for i, case in enumerate(corpus):
        step = trace_case(path, case, trace=False)
        yield case, {}, step, pack_case(path, case, place=True)
        if i in picked:
            for choices in program_choices(case):
                yield case, choices, step, pack_case(path, case, choices, place=True)


def place_problems(step: TraceRun, run: PackRun) -> list[str]:
    """What a placed pack run did otherwise than it must: a lowering refused or other than the
    program's words at the placement's places (`native_pack.placed_words`), a placement below G26's
    bound -- smaller than any layout of its values over their lifetimes, so live values share
    bytes -- or a run that stopped or trained another state or events than the C step's `step`. A
    placement above its bound is the layout's gap to the bound, not a divergence (`place_mismatch`
    returns it)."""
    from bcir.hosted.models.native_pack import placed_words

    if run.status == -1:
        return [f"the lowering refused the placed pack (why {run.why})"]
    problems = []
    if run.words != placed_words(run.pack, run.placed.placements):
        problems.append("the lowering is not the program's words at the placed places")
    if run.placed.extent < run.placed.bound:
        problems.append(f"placed at {run.placed.extent} bytes, below its {run.placed.bound} bound")
    return _ran_as_the_step(step, run, problems)


def place_mismatch(path, cases=None) -> tuple[int, int, list[str], list[tuple[int, bool]]]:
    """`train.plan.place.mismatch`: of the placed pack runs (`place_runs`), those refused, lowered
    to other words than the program's at the placed places, stopped, placed below their bound or
    training another state or events than the C step on the same case; the runs; what differed,
    run by run; and each run's layout as (its bytes above G26's bound, whether every part's
    layout is a proved optimum) -- 0 and proved is TMSAO-1, the rest a bounded gap."""
    bad, runs, notes, gaps = 0, 0, [], []
    for case, choices, step, run in place_runs(path, cases):
        runs += 1
        problems = place_problems(step, run)
        if problems:
            bad += 1
            notes.append(f"{case} {choices}: {problems[0]}")
        if run.status != -1:
            gaps.append((run.placed.extent - run.placed.bound, run.placed.optimal))
    return bad, runs, notes, gaps


def pack_mismatch(path, cases=None) -> tuple[int, int, list[str]]:
    """`train.plan.pack.mismatch`: of the pack runs (`pack_runs`), those whose lowering was refused
    or is not their program's words, that stopped, or that trained another state or events than
    the C step on the same case; the runs; and what differed, run by run."""
    bad, runs, notes = 0, 0, []
    for case, choices, step, run in pack_runs(path, cases):
        runs += 1
        problems = pack_problems(step, run)
        if problems:
            bad += 1
            notes.append(f"{case} {choices}: {problems[0]}")
    return bad, runs, notes


#: The fuzz harness's state (runtime/c/fuzz_decoder_train.c): its spec and shape, the contracts its
#: configuration byte's low three bits name (bit 3: every layer recomputed), its workspace.
FUZZ_STATE = PLAN_CORPUS[0]
FUZZ_CONTRACTS = (0, 3, 4, 14, 16, 82, 32, 36)
FUZZ_WORKSPACE = 1 << 20


def renumber_jobs(records: list[list[int]]) -> list[list[int]]:
    """A program's records (lists of words) with their jobs numbered from 1 again, in place -- a
    GEMM job's calls keep one number between them -- as an edit that adds or drops records needs,
    so the check meets the edit rather than a launch out of order. Returns the records."""
    number, last = 0, None
    for r in records:
        if not (r[0] == 5 and last == (5, r[1])):  # a GEMM call of the job before
            number += 1
        last = (r[0], r[1])
        r[1] = number
    return records


def fuzz_input(config: int, steps: int, accumulation: int, words) -> bytes:
    """A fuzz input: the configuration byte, the shape byte, the program's words little-endian."""
    shape = (steps - 1) | (accumulation - 1) << 1
    return bytes([config, shape]) + struct.pack(f"<{len(words)}Q", *words)


def fuzz_program(config: int, steps: int = 1, accumulation: int = 1) -> list[int]:
    """The planner's program for the fuzz harness's state under `config` (its words)."""
    from bcir.hosted.models.native_launch import program_words, step_program

    spec, batch, time_ = FUZZ_STATE
    contract = FUZZ_CONTRACTS[config & 7]
    return program_words(step_program(
        spec, batch, time_, contract=contract, recompute=3 if config & 8 else 0,
        workspace_bytes=FUZZ_WORKSPACE if contract & 16 else 0, steps=steps,
        accumulation=accumulation,
    ))  # fmt: skip


#: PX-4: the configuration bits the two fuzz harnesses read for the transient arenas -- lent at
#: what the program reaches in each (FUZZ_PLACED), then one float short (FUZZ_SHORT) -- and the
#: configurations whose planner programs they are seeded with placed by lifetime.
FUZZ_PLACED, FUZZ_SHORT = 16, 32
FUZZ_PLACED_CONFIGS = (0, 4, 8, 13)


def fuzz_reach(words, space: int, most: int) -> int:
    """The floats of `space` (5 activations, 6 scratch) a program's operands reach -- each one's
    offset and bytes there rounded up to a float, counting only operands inside the arena's `most`
    floats: the harnesses' `reach` (runtime/c/fuzz_decoder_train.c, fuzz_decoder_pack.c)."""
    from bcir.hosted.models.native_launch import RECORD_WORDS

    floats = 0
    for base in range(0, len(words) - len(words) % RECORD_WORDS, RECORD_WORDS):
        for q in range(10):
            space_, offset, nbytes = words[base + 11 + 3 * q : base + 14 + 3 * q]
            if space_ == space and offset <= 4 * most and nbytes <= 4 * most - offset:
                floats = max(floats, -(-(offset + nbytes) // 4))
    return floats


def fuzz_lent(words, config: int, plan) -> tuple[int, int] | None:
    """The transient arenas' floats a harness lends under `config` for `words` (`plan` the state's
    C plan: its activation and scratch counts): None for the trainer's own, else what the words
    reach -- the activations up to the plan's whole transient span, the room the harnesses
    allocate, since a placement holds the scratch values there too -- one float short under
    FUZZ_SHORT."""
    if not config & FUZZ_PLACED:
        return None
    acts = fuzz_reach(words, 5, plan.activations + plan.scratch)
    scratch = fuzz_reach(words, 6, plan.scratch)
    if config & FUZZ_SHORT:
        if acts:
            acts -= 1
        elif scratch:
            scratch -= 1
    return acts, scratch


def fuzz_training_pack(config: int, steps: int = 1, accumulation: int = 1):
    """The planner's training pack for the fuzz harnesses' state under `config` -- the one both the
    pack seeds (`fuzz_pack`) and the placed program seeds (`fuzz_placed`) are made from."""
    from bcir.hosted.models.native_pack import training_pack

    spec, batch, time_ = FUZZ_STATE
    contract = FUZZ_CONTRACTS[config & 7]
    return training_pack(spec, batch, time_, contract=contract, recompute=3 if config & 8 else 0,
                         workspace_bytes=FUZZ_WORKSPACE if contract & 16 else 0, steps=steps,
                         accumulation=accumulation)  # fmt: skip


def fuzz_placed(config: int, steps: int = 1, accumulation: int = 1):
    """The planner's training pack for the fuzz harnesses' state under `config`, and its transient
    values placed by lifetime (`native_pack.place_by_lifetime`)."""
    from bcir.hosted.models.native_pack import place_by_lifetime

    tp = fuzz_training_pack(config, steps, accumulation)
    return tp, place_by_lifetime(tp)


def fuzz_seeds() -> list[tuple[str, bytes, bool]]:
    """The fuzz harness's seed corpus as (name, input, whether the check admits it): the
    planner's program for every configuration at one step, two of them at two steps of two
    micro-batches, the planner's programs of `FUZZ_PLACED_CONFIGS` placed by lifetime on the arenas
    they reach (PX-4) and one of them a float short, near misses of the check's rules on the
    first, and the kept layers' program on the recompute set's arena -- so a campaign starts inside
    the record grammar and inside its refusals."""
    from bcir.hosted.models.native_pack import placed_words

    out = []
    for config in range(16):
        shapes = ((1, 1), (2, 2)) if config in (0, 4) else ((1, 1),)
        for steps, accumulation in shapes:
            words = fuzz_program(config, steps, accumulation)
            out.append((f"c{config}_{steps}x{accumulation}",
                        fuzz_input(config, steps, accumulation, words), True))  # fmt: skip
    for config in FUZZ_PLACED_CONFIGS:
        tp, placed = fuzz_placed(config)
        words = placed_words(tp, placed.placements)
        out.append((f"placed_c{config}", fuzz_input(config | FUZZ_PLACED, 1, 1, words), True))
        if config == 0:
            out.append(("placed_short", fuzz_input(config | FUZZ_PLACED | FUZZ_SHORT, 1, 1, words),
                        False))  # fmt: skip
    words, w = fuzz_program(0), 48
    records = [list(words[i : i + w]) for i in range(0, len(words), w)]
    rms = next(i for i, r in enumerate(records) if r[0] == 4)
    gemm = next(i for i, r in enumerate(records) if r[0] == 5)
    loss = next(i for i, r in enumerate(records) if r[0] == 15)
    misses = {
        "unscanned": renumber_jobs([list(r) for r in records[1:]]),  # weights read, not known
        "eps": [r if i != rms else r[:5] + [r[5] + 1] + r[6:] for i, r in enumerate(records)],
        "alpha": [r if i != gemm else r[:8] + [0x40000000] + r[9:] for i, r in enumerate(records)],
        "twoloss": renumber_jobs(
            [list(r) for r in records[: loss + 1] + [records[loss]] + records[loss + 1 :]]
        ),
    }
    for name, edited in misses.items():
        out.append((f"miss_{name}", fuzz_input(0, 1, 1, [x for r in edited for x in r]), False))
    # the kept layers' program on the recompute set's trainer (PX-4): its caches past the arena the
    # trainer lends -- the recompute set's activations -- though inside the room the harness
    # allocates for a placement, so only a harness lending the set's own arena refuses it
    out.append(("miss_recompute_arena", fuzz_input(8, 1, 1, fuzz_program(0)), False))
    return out


def fuzz_pack_input(config: int, steps: int, accumulation: int, pack: bytes, plan: bytes) -> bytes:
    """A pack fuzz input (runtime/c/fuzz_decoder_pack.c): the configuration byte, the shape byte,
    the pack's length (u32, little-endian), the pack, then the plan."""
    shape = (steps - 1) | (accumulation - 1) << 1
    return bytes([config, shape]) + struct.pack("<I", len(pack)) + pack + plan


def fuzz_pack_parts(data: bytes) -> tuple[int, int, int, bytes, bytes]:
    """A pack fuzz input's parts: (configuration, steps, accumulation, pack, plan)."""
    (n,) = struct.unpack_from("<I", data, 2)
    return data[0], 1 + (data[1] & 1), 1 + (data[1] >> 1 & 1), data[6 : 6 + n], data[6 + n :]


def fuzz_pack(config: int, steps: int = 1, accumulation: int = 1):
    """The planner's pack and plan for the fuzz harness's state under `config` (bytes, bytes, and
    the training pack and the two objects for an edit)."""
    from bcir.hosted.models.native_pack import pack_artifacts

    tp = fuzz_training_pack(config, steps, accumulation)
    return (tp, *pack_artifacts(tp))


def fuzz_pack_seeds() -> list[tuple[str, bytes, bool]]:
    """The pack fuzz harness's seed corpus as (name, input, whether the lowering admits it): the
    planner's pack for every configuration at one step, two of them at two steps of two
    micro-batches, the planner's packs of `FUZZ_PLACED_CONFIGS` placed by lifetime and run on the
    arenas their programs reach (PX-4) -- one of them a float short, which lowers and whose run is
    refused -- and near misses of the lowering's refusals on the first: two live values on one
    span (the alias law), a name outside the vocabulary, a value no lifetime places, a claim past
    its value's lifetime, a block not the claim's."""
    import copy
    import dataclasses

    from bcir.abi.streampack_abi import encode
    from bcir.hosted.models.native_pack import pack_artifacts
    from bcir.tests.plan_fixtures import raw_encode

    out = []
    for config in range(16):
        shapes = ((1, 1), (2, 2)) if config in (0, 4) else ((1, 1),)
        for steps, accumulation in shapes:
            _tp, pack, plan, _, _ = fuzz_pack(config, steps, accumulation)
            out.append((f"c{config}_{steps}x{accumulation}",
                        fuzz_pack_input(config, steps, accumulation, pack, plan), True))  # fmt: skip
    for config in FUZZ_PLACED_CONFIGS:
        tp, placed = fuzz_placed(config)
        pack, plan, _, _ = pack_artifacts(tp, placements=placed.placements)
        out.append((f"placed_c{config}", fuzz_pack_input(config | FUZZ_PLACED, 1, 1, pack, plan),
                    True))  # fmt: skip
        if config == 0:
            out.append(("placed_short",
                        fuzz_pack_input(config | FUZZ_PLACED | FUZZ_SHORT, 1, 1, pack, plan), True))  # fmt: skip
    tp, pack_bytes, plan_bytes, pack, plan = fuzz_pack(0)
    claims = {c.id: c for ph in tp.module.phases for c in ph.claims}
    rms = next(i for i, seg in enumerate(pack.segments) if seg.opcode == "train.rms")
    x0, n1 = claims[pack.segments[rms].claim_id].rd[0], claims[pack.segments[rms].claim_id].wr[0]
    gemm = next(i for i, seg in enumerate(pack.segments) if seg.opcode.startswith("train.gemm."))
    lt = {x.rid: x for x in plan.lifetimes}
    spans = [x for x in plan.lifetimes if x.last_phase > x.first_phase and
             x.bank in ("activations", "scratch")]  # fmt: skip

    def with_lifetimes(edit):  # without the wire laws: the encoder refuses what the C must
        q = copy.deepcopy(plan)
        q.lifetimes = sorted(edit(list(q.lifetimes)), key=lambda x: x.rid)
        return raw_encode(q)

    def with_segment(i, block=None, **fields):
        p = dataclasses.replace(pack, segments=list(pack.segments), blocks=list(pack.blocks))
        if fields:
            p.segments[i] = dataclasses.replace(p.segments[i], **fields)
        if block:
            p.blocks[i] = dataclasses.replace(p.blocks[i], **block)
        return encode(p)

    misses = {
        "alias": (pack_bytes, with_lifetimes(lambda ls: [
            dataclasses.replace(x, offset=lt[x0].offset) if x.rid == n1 else x for x in ls])),
        "op": (with_segment(gemm, opcode="train.gemm.solo.nt"), plan_bytes),
        "rid": (pack_bytes, with_lifetimes(lambda ls: [x for x in ls if x.rid != spans[0].rid])),
        "live": (pack_bytes, with_lifetimes(lambda ls: [
            dataclasses.replace(x, last_phase=x.last_phase - 1, first_tick=0, last_tick=0)
            if x.rid == spans[0].rid else x for x in ls])),
        "block": (with_segment(gemm, block={"count": pack.blocks[gemm].count + 1}), plan_bytes),
    }  # fmt: skip
    for name, (pk, pl) in misses.items():
        out.append((f"miss_{name}", fuzz_pack_input(0, 1, 1, pk, pl), False))
    return out


def measure(*, timed: bool = True) -> dict[str, float]:
    """Every NDT-GEM row this host can measure: the memory row always, the corpus row in a source
    checkout, the native rows with a compiler (and the timed one only when `timed`)."""
    ratio = placed_transient()[0]  # PX-4: the placed arena (the C plan's: `transient_ratio`)
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
        out["train.plan.remat.mismatch"] = float(recompute_plan_mismatch(lib))
        out["train.mm.mismatch"] = float(mm_mismatch(lib))
        out["train.mm.fused.mismatch"] = float(mm_fused_mismatch(lib))
        out["train.mm.kblock.mismatch"] = float(mm_kblock_mismatch(lib))
        out["train.mm.bf16.mismatch"] = float(mm_bf16_mismatch(lib))
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
        traced = trace_library(Path(tmp) / "trace")
        if traced is not None:  # no C units to trace (an installed package): not measured
            out["train.plan.executed.mismatch"] = float(executed_mismatch(traced)[0])
            out["train.plan.program.mismatch"] = float(program_mismatch(traced)[0])
        packed = pack_library(Path(tmp) / "pack")
        if packed is not None:  # no C units to lower with (an installed package): not measured
            out["train.plan.pack.mismatch"] = float(pack_mismatch(packed)[0])
            out["train.plan.place.mismatch"] = float(place_mismatch(packed)[0])
        if timed:
            out.update(measure_step(path))
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


__all__ = [
    "BENCH",
    "MM_CASES",
    "PANELS_BENCH",
    "PLAN_CORPUS",
    "REMAT_BENCH",
    "REMAT_SETS",
    "native_activation_count",
    "recompute_plan_mismatch",
    "THREAD_COUNTS",
    "THREADS_BENCH",
    "build_library",
    "corpus_unselected",
    "exp_sweep",
    "measure",
    "measure_step",
    "mm_bf16_mismatch",
    "mm_bf16_variants",
    "mm_fused_mismatch",
    "mm_kblock_mismatch",
    "mm_reference_bf16",
    "bf16_round",
    "MM_BF16_CASES",
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
    "TRACE_CONTRACTS",
    "TraceCase",
    "TraceRun",
    "executed_mismatch",
    "trace_case",
    "trace_corpus",
    "trace_library",
    "checkout_library",
    "FUZZ_CONTRACTS",
    "FUZZ_PLACED",
    "FUZZ_PLACED_CONFIGS",
    "FUZZ_SHORT",
    "FUZZ_STATE",
    "FUZZ_WORKSPACE",
    "fuzz_input",
    "fuzz_lent",
    "fuzz_pack",
    "fuzz_pack_input",
    "fuzz_pack_parts",
    "fuzz_pack_seeds",
    "fuzz_placed",
    "fuzz_reach",
    "renumber_jobs",
    "fuzz_program",
    "fuzz_seeds",
    "fuzz_training_pack",
    "PackRun",
    "pack_case",
    "pack_library",
    "pack_mismatch",
    "pack_problems",
    "pack_runs",
    "place_mismatch",
    "place_problems",
    "place_runs",
    "placed_transient",
    "ProgramRun",
    "program_case",
    "program_choices",
    "program_mismatch",
    "program_problems",
    "program_run_count",
    "program_runs",
    "state_cases",
]


if __name__ == "__main__":  # pragma: no cover - a manual probe
    print(measure())
    os._exit(0)
