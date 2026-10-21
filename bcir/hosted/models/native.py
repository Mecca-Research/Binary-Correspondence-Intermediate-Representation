"""Opt-in FP32 decoder trainer; all tensor execution and updates live in C.

Only stdlib at this boundary. PyTorch is an independent reference, never an
execution fallback. Checkpoints are bounded hashed JSON + FP32 arrays, not pickle.
The native ABI admits selective C++/accelerator providers.
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import mmap
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
from array import array
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from ...frontends.models.decode import DecoderSpec
from ...toolchain import BuildFailed, host_link_args

_ROOT = Path(__file__).resolve().parents[3]
_C = _ROOT / "runtime" / "c"
_UNITS = ("bcir_tensor.h", "bcir_tensor.c", "bcir_decoder_train.h", "bcir_decoder_train.c")
#: PX-2b: the pack lowering, and the StreamPack / ExecutionPlanV1 verifiers and GEM's executor it
#: runs on (`NativeDecoder.build(..., pack=True)`).
_PACK_UNITS = (
    "bcir_decoder_pack.h", "bcir_decoder_pack.c", "bcir_runtime.h", "bcir_runtime.c",
    "bcir_exec.h", "bcir_exec.c", "bcir_streampack.h", "bcir_execution_plan.h",
)  # fmt: skip
_FP = ct.POINTER(ct.c_float)
_UP = ct.POINTER(ct.c_uint32)
_SZ = ct.c_size_t
_MM = ct.CFUNCTYPE(
    None, ct.c_void_p, ct.c_int, ct.c_int, _SZ, _SZ, _SZ, ct.c_float, _FP, _FP, ct.c_float, _FP
)
_SILU = ct.CFUNCTYPE(None, ct.c_void_p, _SZ, _FP, _FP)
_CBLAS = ct.CFUNCTYPE(
    None,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_int,
    ct.c_float,
    _FP,
    ct.c_int,
    _FP,
    ct.c_int,
    ct.c_float,
    _FP,
    ct.c_int,
)
_MAGIC = b"BCIRDT1\0"


def provider_mismatch(lib, slot, shapes) -> int:
    """The blocked contract's admission differential (NDT-KBLOCK): over `shapes` (ta, tb, m, n,
    k), at beta 0 and 1 with alpha 1 -- the trainer's calls -- the provider behind `slot` (a
    pointer to a CBLAS sgemm function pointer, as `bcir_tensor_cblas_mm` reads it) against
    `bcir_tensor_mm_kblock` on deterministic operands, bit for bit; returns the cases that
    differ. Empty `shapes` is refused: a differential over nothing admits nothing."""
    shapes = tuple(shapes)
    if not shapes:
        raise ValueError("the admission differential needs at least one GEMM shape")
    sig = [ct.c_void_p, ct.c_int, ct.c_int, _SZ, _SZ, _SZ, ct.c_float, _FP, _FP, ct.c_float, _FP]
    provider, kblock = lib.bcir_tensor_cblas_mm, lib.bcir_tensor_mm_kblock
    provider.argtypes = kblock.argtypes = sig
    provider.restype = kblock.restype = None
    state = 0x2545F491
    pattern = []
    for _ in range(4096):  # one block of mixed signs and magnitudes, tiled with an offset
        state = (state * 1103515245 + 12345) % 2**31
        pattern.append((state / 2**31 - 0.5) * (8.0 if state & 1 else 0.125))

    def operand(count, offset):
        out = (ct.c_float * count)()
        out[:] = (pattern[offset:] + pattern * (count // 4096 + 1))[:count]
        return out

    bad = 0
    for index, (ta, tb, m, n, k) in enumerate(shapes):
        a, b = operand(m * k, (7 * index) % 4096), operand(k * n, (13 * index + 5) % 4096)
        start = operand(m * n, (29 * index + 11) % 4096)
        for beta in (0.0, 1.0):
            want, got = (ct.c_float * (m * n))(), (ct.c_float * (m * n))()
            ct.memmove(want, start, 4 * m * n)
            ct.memmove(got, start, 4 * m * n)
            kblock(None, ta, tb, m, n, k, 1.0, a, b, beta, want)
            provider(slot, ta, tb, m, n, k, 1.0, a, b, beta, got)
            bad += bytes(want) != bytes(got)
    return bad


#: The contracts a state trains under (ABI 3): v1 rounds each GEMM term's product and its addition
#: and takes exp from libm; "fused" rounds the two once (one IEEE fmaf per term, NDT-FMA); "exp" puts
#: SiLU, the SwiGLU VJP and the softmax on the BCIR exponential (NDT-EXP); "fused+exp" both;
#: "kblock" is the blocked fused GEMM contract (v3, NDT-KBLOCK: the depth split by one rule into
#: blocks of up to 448, each block's fused chain from zero added to c in order -- the arithmetic
#: OpenBLAS's sgemm performs, so such a provider can be admitted under it by `use_cblas`'s
#: differential), "kblock+exp" both. Under v1, "fused" and "exp" a provider's GEMM keeps its own
#: arithmetic; under "kblock" it is admitted only where it is the contract's, bit for bit.
#: "bf16" (ABI 4, NDT-BF16) puts BCIR's GEMMs under the bf16 contract (bcir_tensor.h): operands
#: rounded to bfloat16 and accumulated in binary32 as the AMX unit's TDPBF16PS accumulates them --
#: the AMX, AVX-512 and portable instances one result, bit for bit -- with a workspace for the
#: packed operands the trainer lends from its own mapping (`workspace_bytes`); "bf16+exp" with the
#: BCIR exponential. A CBLAS provider computes in binary32, so none is admitted under it.
#: "reduce" (NDT-REDUCE) puts the clipping norm under a fixed-shape reduction -- the gradient in
#: chunks of `reduce_chunk` elements, each chunk's squares summed in index order in double from
#: zero, the chunk sums added in chunk order from zero -- so the update's one pass runs on the pool;
#: "<contract>+reduce" names it with any of the seven above. "attn" (ATTN-FMA) puts the attention's
#: products under the fused contract -- every multiply-add of the scores, the context and the
#: backward's products one fmaf, the same terms in the same order -- and "<contract>+attn" names it
#: with any of the above ("reduce+attn", "kblock+exp+reduce+attn", ...). "attnbf16" (BF16-ATTN)
#: puts them on bfloat16 operands instead -- each product the bf16 GEMM contract's over the depth
#: tiles the causal mask reaches, the softmax and the score adjoint v1's -- and is named only with a
#: bf16 GEMM contract ("bf16+attnbf16", ..., "bf16+exp+reduce+attnbf16"): it packs into the same
#: workspace, which `workspace_bytes` then sizes for it too, and excludes "+attn". Whatever the
#: contract, the loss, the gamma adjoints and the embedding scatter run on the pool in the serial
#: order: every loss and state is the serial one, bit for bit, at every thread count.
_CONTRACTS = {"v1": 0, "fused": 1, "exp": 2, "fused+exp": 3, "kblock": 4, "kblock+exp": 6,
              "bf16": 16, "bf16+exp": 18}  # fmt: skip
_CONTRACTS.update({("reduce" if name == "v1" else name + "+reduce"): bits | 8
                   for name, bits in list(_CONTRACTS.items())})  # fmt: skip
_CONTRACTS.update({("attn" if name == "v1" else name + "+attn"): bits | 32
                   for name, bits in list(_CONTRACTS.items())})  # fmt: skip
_CONTRACTS.update({name + "+attnbf16": bits | 64 for name, bits in list(_CONTRACTS.items())
                   if bits & 16 and not bits & 32})  # fmt: skip
_KBLOCK = 4
_REDUCE = 8
_BF16 = 16
_ATTN = 32
_ATTNBF16 = 64
#: The contract names `NativeDecoder(contract=...)` admits, sorted; a recipe names one of them.
CONTRACTS = tuple(sorted(_CONTRACTS))


class _Spec(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32)
        for n in ("vocab", "width", "heads", "kvheads", "layers", "ff", "batch", "time", "tied")
    ] + [("rope_base", ct.c_double), ("rms_eps", ct.c_double)]


class _Plan(ct.Structure):
    _fields_ = [("spec", _Spec)] + [(n, _SZ) for n in ("parameters", "activations", "scratch")]


class _Provider(ct.Structure):
    _fields_ = [("ctx", ct.c_void_p), ("mm", _MM), ("silu", _SILU)]


class _State(ct.Structure):
    _fields_ = (
        [("plan", _Plan)]
        + [
            (n, _FP)
            for n in ("weights", "gradients", "moment1", "moment2", "activations", "scratch")
        ]
        + [
            (n + "_capacity", _SZ)
            for n in ("weight", "gradient", "moment1", "moment2", "activation", "scratch")
        ]
        + [
            ("step", ct.c_uint64),
            ("beta1_power", ct.c_double),
            ("beta2_power", ct.c_double),
            ("provider", _Provider),
            ("pool", ct.c_void_p),  # ABI 2 (NDT-THREADS): the pool the kernels run on
            ("contract", ct.c_uint32),  # ABI 3 (NDT-FMA): the GEMM contract, _CONTRACTS
            ("workspace", ct.c_void_p),  # ABI 4 (NDT-BF16): the bf16 GEMMs' packed operands
            ("workspace_bytes", _SZ),
            ("recompute", ct.c_uint64),  # ABI 5 (REMAT): layers the backward recomputes
        ]
    )


class _Optimizer(ct.Structure):
    _fields_ = [
        (n, ct.c_double) for n in ("lr", "beta1", "beta2", "epsilon", "weight_decay", "grad_clip")
    ]


class _Event(ct.Structure):
    _fields_ = [("loss", ct.c_double), ("grad_norm", ct.c_double), ("step", ct.c_uint64)]


@dataclass(frozen=True)
class NativeAdamW:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def __post_init__(self):
        if any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in asdict(self).values()
        ):
            raise ValueError("AdamW values must be finite numbers")
        if (
            self.lr < 0
            or not 0 <= self.beta1 < 1
            or not 0 <= self.beta2 < 1
            or self.epsilon <= 0
            or self.weight_decay < 0
            or self.grad_clip < 0
        ):
            raise ValueError("invalid AdamW configuration")


def parameter_layout(spec: DecoderSpec):
    """Named HF-layout spans; tied embeddings appear once."""
    d, k, f, v = spec.d_model, spec.kv_dim, spec.d_ff, spec.vocab_size
    tensors = [("model.embed_tokens.weight", (v, d))]
    for layer in range(spec.n_layers):
        prefix = f"model.layers.{layer}."
        tensors.extend(
            (prefix + name, shape)
            for name, shape in (
                ("input_layernorm.weight", (d,)),
                ("self_attn.q_proj.weight", (d, d)),
                ("self_attn.k_proj.weight", (k, d)),
                ("self_attn.v_proj.weight", (k, d)),
                ("self_attn.o_proj.weight", (d, d)),
                ("post_attention_layernorm.weight", (d,)),
                ("mlp.gate_proj.weight", (f, d)),
                ("mlp.up_proj.weight", (f, d)),
                ("mlp.down_proj.weight", (d, f)),
            )
        )
    tensors.append(("model.norm.weight", (d,)))
    if not spec.tied_embeddings:
        tensors.append(("lm_head.weight", (v, d)))
    out, offset = [], 0
    for name, shape in tensors:
        n = math.prod(shape)
        out.append((name, shape, offset, n))
        offset += n
    return tuple(out)


class NativeRunStopped(ValueError):
    """A native run that stopped early: its C status (1 invalid, 2 capacity, 3 non-finite) and the
    steps it completed, which the state keeps."""

    def __init__(self, message: str, status: int, completed: int):
        super().__init__(message)
        self.status, self.completed = status, completed


class PackRefused(ValueError):
    """A pack and plan `bcir_decoder_lower_pack` refused (PX-2b): its C status (1 refused, 2 a
    buffer short) and `why`, the verifiers' bcir_status or a lowering reason
    (`native_pack.REASONS` names both)."""

    def __init__(self, message: str, status: int, why: int):
        super().__init__(message)
        self.status, self.why = status, why


def _check(rc: int):
    if rc:
        reason = {
            1: "invalid shape/IDs/alias/state",
            2: "capacity/overflow",
            3: "non-finite arithmetic",
        }.get(rc, rc)
        raise ValueError(f"native decoder refused operation: {reason}")


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate checkpoint key {key!r}")
        out[key] = value
    return out


def _checkpoint_digest(meta, payload):
    identity = {key: value for key, value in meta.items() if key != "sha256"}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    digest.update(payload)
    return digest.hexdigest()


#: NDT-PAGES: the transparent huge page the state's mapping is aligned to (the PMD size of
#: x86-64 and of a 4 KB arm64 kernel; under a larger one fewer of the mapping's pages are huge).
_HUGE_PAGE = 2 << 20


def _state_buffers(rows, huge_pages):
    """The trainer's state buffers, zeroed, one per `rows` entry (name, floats), in its order.

    NDT-PAGES: where Python exposes MADV_HUGEPAGE (Linux) they share one private anonymous
    mapping advised for transparent huge pages before any page is touched -- the first buffer
    on a 2 MB boundary, each next 64 bytes past the last one's end -- so the GEMMs' operand
    walks miss the TLB once a 2 MB page rather than once a 4 KB one. Private, because a shared
    anonymous mapping is shmem, whose huge pages follow another switch. Elsewhere, or with
    `huge_pages` false, each buffer is a ctypes array. No kernel's arithmetic depends on an
    address, so the placement never reaches a value. Returns the buffers and whether the kernel
    took the advice: one without THP refuses it (EINVAL) and the mapping holds 4 KB pages."""
    if not huge_pages or not hasattr(mmap, "MADV_HUGEPAGE"):
        return {name: (ct.c_float * n)() for name, n in rows}, False
    offsets, end = [], 0
    for _name, n in rows:
        end = (end + 63) & ~63
        offsets.append(end)
        end += 4 * n
    mapping = mmap.mmap(-1, end + _HUGE_PAGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        mapping.madvise(mmap.MADV_HUGEPAGE)
        advised = True
    except OSError:
        advised = False
    lead = -ct.addressof(ct.c_char.from_buffer(mapping)) % _HUGE_PAGE
    # each array holds the mapping (from_buffer keeps its source), so it lives as long as they do
    buffers = {
        name: (ct.c_float * n).from_buffer(mapping, lead + offset)
        for (name, n), offset in zip(rows, offsets)
    }
    return buffers, advised


_SIZE_MAX = 2 ** (8 * ct.sizeof(_SZ)) - 1


def _bind_plan(lib):
    """The C plan's byte counts (WS-PLAN, REMAT) on a loaded library: every figure the trainer
    admits or asks is the C's own, read here, never re-derived."""
    plan = ct.POINTER(_Plan)
    for name, args in (
        ("workspace_bytes", [plan, ct.c_uint32]),
        ("workspace_need", [plan, ct.c_uint32, ct.c_uint]),
        ("workspace_floor", [plan, ct.c_uint32, ct.c_uint]),
        ("workspace_panels", [plan, ct.c_uint32]),
        ("activation_count", [plan, ct.c_uint64]),
    ):
        fn = getattr(lib, "bcir_decoder_" + name)
        fn.argtypes, fn.restype = args, _SZ


@dataclass(frozen=True)
class MemoryPlan:
    """A trainer's arenas under a recompute set and a bf16 workspace (GEO-3), in bytes, every
    count the C plan's (bcir_decoder_activation_count, bcir_decoder_workspace_need/_panels/_floor).
    Each plan trains the state the all-kept trainer at the exact need trains, bit for bit: the
    choice moves memory and time, never a value."""

    recompute: int  # the layers the backward recomputes, bit l for layer l
    workspace_bytes: int  # the bf16 workspace lent (0 without a bf16 GEMM contract)
    activation_count: int  # floats of the activation arena under the recompute set
    workspace_need: int  # every GEMM group whole (the SwiGLU block's panels where admitted)
    workspace_panels: int  # the least at which the SwiGLU block keeps its panels; 0 none
    workspace_floor: int  # the least the state admits on its pool
    recomputes: int  # layer forwards the backward re-runs a step (the highest set layer is free)
    persistent_bytes: int  # weights, gradients and the two moments
    activation_bytes: int
    scratch_bytes: int
    total_bytes: int


def memory_plan(lib, plan, contract, threads, budget, *, recompute=None, workspace_bytes=None):
    """The arenas of one trainer (GEO-3: per-layer caching coupled with packing-aware workspace
    planning). With `recompute` given: that set and `workspace_bytes` (None: the exact need),
    refused (ValueError) where the C refuses them -- a bit past the last layer, a workspace below
    the floor or lent without a bf16 contract. With `recompute` None: the plan this module
    chooses within `budget` bytes, among the choices that keep every value, in the order the
    measured frontier prices them (docs/research/measurements/2026-10-19/geo3): the fewest
    recomputed layers (the lowest ones; which layers is no matter of cost), each with the
    workspace at the step's need and then at the SwiGLU panels' need, and the floor only after
    every set. On both measured shapes the panels' tier cost no step time the rounds resolved
    (1.3-2.1 MiB freed), a re-run layer 1-11% of a step (10.5-25 MiB) and the floor 32-75%
    (1.7-4.1 MiB). A declared policy, no optimality claim (TMSAO-4); micro-batch accumulation --
    cheaper than recomputation at one measured shape, dearer at the other, and never the same
    state -- is the caller's."""
    _bind_plan(lib)
    if contract not in _CONTRACTS:
        raise ValueError(f"contract must be one of {sorted(_CONTRACTS)}")
    # ctypes wraps an out-of-range integer into its C type silently (2^64 reaches the C as the
    # empty set), so a set or a size outside its type is refused here, never handed over
    if recompute is not None and (type(recompute) is not int or not 0 <= recompute < 2**64):
        raise ValueError("recompute must be a layer bit set in [0, 2^64)")
    if workspace_bytes is not None and (
        type(workspace_bytes) is not int or not 0 <= workspace_bytes < _SIZE_MAX
    ):
        raise ValueError("workspace_bytes must be None or a size in [0, SIZE_MAX)")
    bits = _CONTRACTS[contract]
    p = ct.byref(plan)
    need = lib.bcir_decoder_workspace_need(p, bits, threads)
    floor = lib.bcir_decoder_workspace_floor(p, bits, threads)
    panels = lib.bcir_decoder_workspace_panels(p, bits)
    if _SIZE_MAX in (need, floor, panels):
        raise ValueError("native plan's workspace overflows size_t or the plan is refused")
    layers = plan.spec.layers

    def arenas(mask, workspace):
        count = lib.bcir_decoder_activation_count(p, mask)
        if count == _SIZE_MAX:
            raise ValueError(f"recompute set {mask:#x} names a layer past the plan's {layers}")
        named = sum(1 for layer in range(min(layers, 64)) if (mask >> layer) & 1)
        persistent, acts, scratch = 16 * plan.parameters, 4 * count, 4 * plan.scratch
        return MemoryPlan(
            mask, workspace, count, need, panels, floor, max(named - 1, 0), persistent, acts,
            scratch, persistent + acts + scratch + workspace,
        )  # fmt: skip

    if recompute is not None:
        if workspace_bytes is None:
            workspace_bytes = need
        if not need and workspace_bytes:
            raise ValueError("a workspace is lent only under a bf16 GEMM contract")
        if workspace_bytes < floor:
            raise ValueError(f"workspace {workspace_bytes} below the floor {floor}")
        return arenas(recompute, workspace_bytes)
    if type(budget) is not int or budget < 1:
        raise ValueError("the budget must be a positive integer")
    tiers = [need]
    if floor <= panels < need:
        tiers.append(panels)
    sets = [0] + [(1 << r) - 1 for r in range(2, min(layers, 64) + 1)]
    # the frontier's prices: the panels' tier about free, then a re-run layer, the floor dearest
    order = [(mask, workspace) for mask in sets for workspace in tiers]
    if floor < tiers[-1]:
        order += [(mask, floor) for mask in sets]
    least = None
    for mask, workspace in order:
        chosen = arenas(mask, workspace)
        if chosen.total_bytes <= budget:
            return chosen
        least = chosen
    raise ValueError(
        f"no recompute set and workspace fit {budget} bytes: the least is {least.total_bytes}"
    )


class NativeDecoder:
    """Caller-owned arenas planned once; native forward/backward/AdamW/run ABI.

    Explicit build, no framework fallback. The memory budget bounds arena allocations,
    not all process memory. Serialize access to each training state.

    `threads` is the trainer's deterministic pool (NDT-THREADS): BCIR's kernels run on it in
    partitions fixed by the shapes, so the state after any number of steps is bit-identical at
    every thread count and to the one-thread (serial) trainer; `close()` releases the threads.
    A CBLAS provider threads its own GEMM as it is configured; the pool runs the rest.
    `contract` selects the arithmetic contracts (`"v1"`, the default of every gate and record;
    `"fused"`, one IEEE fmaf per GEMM term; `"exp"`, the BCIR exponential under SiLU, the SwiGLU
    VJP and the softmax; `"fused+exp"`; `"kblock"`; `"bf16"`, the GEMMs on bfloat16 operands
    accumulated as AMX accumulates them, with `workspace_bytes` of packing workspace counted in
    the budget; any of them `+reduce`, and any of those `+attn`, the attention's products fused,
    or, with a bf16 GEMM contract, `+attnbf16`, the attention's products on bfloat16 operands,
    `CONTRACTS`); the checkpoint names it and `resume` restores it.
    `huge_pages` (NDT-PAGES) puts the state on one mapping advised for 2 MB transparent huge
    pages where the host has them (`_state_buffers`); `self.huge_pages` says whether the kernel
    took the advice. The state is the same either way, bit for bit; the mapping adds at most one
    huge page to what the budget bounds.
    `transient` (PX-4) lends the activation and scratch arenas at the given float counts in place
    of the C plan's -- what a placement of a program's values needs (`native_pack.place_by_lifetime`,
    `Placed.floats`): such a trainer runs programs and packs placed there (`run_program`,
    `run_pack`), every operand held inside the lent arenas, and the fixed step refuses it
    (BCIR_DT_CAPACITY), whose spans are the plan's.
    """

    @staticmethod
    def build(directory, *, cc=None, pack=False, probe=None) -> Path:
        """The trainer as a shared library under `directory`; with `pack`, also the PX-2b pack
        lowering and what it runs on (`bcir_decoder_lower_pack`, `lower_pack`). `probe` "time"
        (PX-3) builds, under a name of its own, the same units with the job clock a measurement
        reads (BCIR_DT_TIME_PROBE: `native_choose.timed_run`): the same code between two clock
        reads a job, so the state it trains is the library's bit for bit."""
        if probe not in (None, "time"):
            raise ValueError('probe must be None or "time"')
        compiler = cc or os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("native decoder training requires a C11 compiler")
        units = _UNITS + (_PACK_UNITS if pack else ())
        missing = [u for u in units if not (_C / u).is_file()]
        if missing:  # the C units ship with the repository, not with the wheel
            raise RuntimeError(
                f"native decoder sources not found under {_C} ({', '.join(missing)}): build "
                "from a BCIR source checkout"
            )
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = (
            ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
        )
        stem = "bcir_decoder_pack" if pack else "bcir_decoder_train"
        target = target_dir / (stem + ("_time" if probe else "") + suffix)
        fd, name = tempfile.mkstemp(prefix=".decoder-", suffix=suffix, dir=target_dir)
        os.close(fd)
        temporary = Path(name)
        try:
            command = [
                compiler,
                "-std=c11",
                "-O3",
                "-ffp-contract=off",
                # sqrt without errno, so the AdamW passes vectorize; results are unchanged
                "-fno-math-errno",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-DBCIR_DT_BUILD_SHARED",
                "-shared",
            ]
            if probe == "time":
                command.append("-DBCIR_DT_TIME_PROBE")
            if sys.platform != "win32":
                # the pool's threads (NDT-THREADS) are pthreads here; Win32 threads need no flag
                command += ["-fPIC", "-pthread"]
            command += ["-I", str(_C), *(str(_C / u) for u in units if u.endswith(".c"))]
            command += ["-o", str(temporary)]
            result = subprocess.run(
                command + host_link_args(["-lm"]), capture_output=True, text=True, timeout=120
            )
            if result.returncode:  # a RuntimeError, as it always was
                raise BuildFailed("native decoder build", result.stderr)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def __init__(
        self,
        library,
        spec: DecoderSpec,
        *,
        batch_size=1,
        context_length=4,
        seed=1729,
        max_memory_bytes=512 * 1024 * 1024,
        threads=1,
        contract="v1",
        huge_pages=True,
        recompute=0,
        workspace_bytes=None,
        transient=None,
    ):
        if not isinstance(spec, DecoderSpec) or spec.activation != "silu_gate":
            raise ValueError("native decoder requires Llama/SwiGLU DecoderSpec")
        if (
            any(
                type(x) is not int or not 1 <= x <= 0xFFFFFFFF
                for x in (
                    batch_size,
                    context_length,
                    spec.vocab_size,
                    spec.d_model,
                    spec.n_heads,
                    spec.n_layers,
                    spec.d_ff,
                )
            )
            or type(max_memory_bytes) is not int
            or max_memory_bytes < 1
        ):
            raise ValueError("batch, context and memory budget must be positive integers")
        if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFF:
            raise ValueError("seed must be in [0, 2^31-1]")
        if type(threads) is not int or not 1 <= threads <= 4096:
            raise ValueError("threads must be an integer in [1, 4096]")
        if contract not in _CONTRACTS:
            raise ValueError(f"contract must be one of {sorted(_CONTRACTS)}")
        if type(huge_pages) is not bool:
            raise ValueError("huge_pages must be bool")
        if recompute != "auto" and (type(recompute) is not int or not 0 <= recompute < 2**64):
            raise ValueError('recompute must be "auto" or a layer bit set in [0, 2^64)')
        if workspace_bytes is not None and (
            type(workspace_bytes) is not int or workspace_bytes < 0
        ):
            raise ValueError("workspace_bytes must be None or a non-negative integer")
        if recompute == "auto" and workspace_bytes is not None:
            raise ValueError('recompute="auto" plans the workspace too: pass no workspace_bytes')
        if transient is not None and (
            type(transient) is not tuple
            or len(transient) != 2
            or any(type(x) is not int or not 0 <= x < 2**62 for x in transient)
        ):
            raise ValueError("transient must be None or (activation floats, scratch floats)")
        if transient is not None and recompute == "auto":
            raise ValueError(
                "transient lends the arenas a placement sized for one program: pass the recompute "
                'set it was placed under, not "auto"'
            )
        self.contract = contract
        self.spec, self.batch_size, self.context_length = spec, batch_size, context_length
        self.library = Path(library).resolve()
        self.lib = ct.CDLL(str(self.library))
        self._bind()
        self.threads = threads
        size = self.lib.bcir_tensor_pool_size(threads)
        if not size:
            rail = {1: "pthreads", 2: "Win32 threads"}.get(
                self.lib.bcir_tensor_pool_rails(), "no thread rail"
            )
            raise RuntimeError(
                f"native decoder pool of {threads} thread(s) refused ({rail} on this host)"
            )
        # the pool lives in this buffer, aligned by hand to 64 bytes, for as long as the trainer
        self._pool_memory = (ct.c_ubyte * (size + 64))()
        base = ct.addressof(self._pool_memory)
        aligned = (base + 63) & ~63
        self._pool = self.lib.bcir_tensor_pool_init(
            ct.c_void_p(aligned), size + 64 - (aligned - base), threads
        )
        if not self._pool:
            raise RuntimeError(f"native decoder pool of {threads} thread(s) could not start")
        plan = _Plan()
        shape = _Spec(
            spec.vocab_size,
            spec.d_model,
            spec.n_heads,
            spec.kv_heads,
            spec.n_layers,
            spec.d_ff,
            batch_size,
            context_length,
            int(spec.tied_embeddings),
            spec.rope_base,
            spec.rms_norm_eps,
        )
        _check(self.lib.bcir_decoder_make_plan(ct.byref(shape), ct.byref(plan)))
        # GEO-3: the arenas a recompute set and a workspace size need (memory_plan); "auto" plans
        # both within the budget, every choice the state the all-kept plan-sized trainer reaches
        if recompute == "auto":
            chosen = memory_plan(self.lib, plan, contract, threads, max_memory_bytes)
            recompute, workspace_bytes = chosen.recompute, chosen.workspace_bytes
        self.plan_memory = memory_plan(
            self.lib, plan, contract, threads, None, recompute=recompute,
            workspace_bytes=workspace_bytes,
        )  # fmt: skip
        self.recompute = recompute
        # NDT-BF16 / WS-PLAN: the bf16 GEMMs' workspace, by default the step's exact need
        self.workspace_bytes = self.plan_memory.workspace_bytes
        self.activation_count = self.plan_memory.activation_count
        self.memory_bytes = self.plan_memory.total_bytes
        # PX-4: the transient arenas a placement lends, counted in place of the plan's
        self.transient = transient
        lent = (self.activation_count, plan.scratch) if transient is None else transient
        self.memory_bytes += 4 * (sum(lent) - self.activation_count - plan.scratch)
        if self.memory_bytes > max_memory_bytes:
            raise ValueError(
                f"native plan needs {self.memory_bytes} bytes, budget {max_memory_bytes}"
            )
        self.layout = parameter_layout(spec)
        if sum(row[3] for row in self.layout) != plan.parameters:
            raise RuntimeError("native/Python parameter layout mismatch")
        self.buffers, self.huge_pages = _state_buffers(
            (
                ("weights", plan.parameters),
                ("gradients", plan.parameters),
                ("moment1", plan.parameters),
                ("moment2", plan.parameters),
                ("activations", lent[0]),
                ("scratch", lent[1]),
            )
            + ((("workspace", (self.workspace_bytes + 3) // 4),) if self.workspace_bytes else ()),
            huge_pages,
        )
        arenas = ("weights", "gradients", "moment1", "moment2", "activations", "scratch")
        workspace = self.buffers.get("workspace")
        self.state = _State(
            plan,
            *(self.buffers[n] for n in arenas),
            plan.parameters,
            plan.parameters,
            plan.parameters,
            plan.parameters,
            lent[0],
            lent[1],
            0,
            1.0,
            1.0,
            _Provider(),
            self._pool,
            _CONTRACTS[contract],
            ct.cast(workspace, ct.c_void_p) if workspace is not None else None,
            self.workspace_bytes,
            recompute,
        )
        rng = random.Random(seed)
        for name, _, offset, count in self.layout:
            norm = "layernorm" in name or name == "model.norm.weight"
            for i in range(count):
                self.buffers["weights"][offset + i] = 1.0 if norm else rng.gauss(0.0, 0.02)
        self.backend = "portable-c"
        self._provider_refs = ()
        self.provider_admission = None

    def close(self):
        """Release the pool's threads; a later call runs serially (the same state). Idempotent."""
        pool = getattr(self, "_pool", None)
        if pool:
            self._pool = None
            if hasattr(self, "state"):
                self.state.pool = None
            self.lib.bcir_tensor_pool_stop(pool)
            self._pool_memory = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass

    def _bind(self):
        self.lib.bcir_decoder_abi_version.argtypes = []
        self.lib.bcir_decoder_abi_version.restype = ct.c_uint32
        if self.lib.bcir_decoder_abi_version() != 5:
            raise ValueError("unsupported native decoder ABI")
        _bind_plan(self.lib)
        for name, args, result in (
            ("rails", [], ct.c_int),
            ("size", [ct.c_uint], _SZ),
            ("init", [ct.c_void_p, _SZ, ct.c_uint], ct.c_void_p),
            ("stop", [ct.c_void_p], None),
            ("threads", [ct.c_void_p], ct.c_uint),
        ):
            fn = getattr(self.lib, "bcir_tensor_pool_" + name)
            fn.argtypes, fn.restype = args, result
        sp = ct.POINTER(_State)
        functions = {
            "make_plan": [ct.POINTER(_Spec), ct.POINTER(_Plan)],
            "forward": [sp, _UP, _SZ],
            "loss_backward": [sp, _UP, _UP, _SZ, ct.c_int, ct.POINTER(ct.c_double)],
            "update": [sp, ct.POINTER(_Optimizer), ct.c_double, ct.POINTER(ct.c_double)],
            "run": [
                sp,
                ct.POINTER(_Optimizer),
                _UP,
                _UP,
                _SZ,
                ct.POINTER(ct.c_double),
                _SZ,
                ct.POINTER(_Event),
                _SZ,
                ct.POINTER(_SZ),
            ],
        }
        for name, args in functions.items():
            fn = getattr(self.lib, "bcir_decoder_" + name)
            fn.argtypes, fn.restype = args, ct.c_int
        self.lib.bcir_decoder_run_accum.argtypes = (
            functions["run"][:7] + [_SZ] + functions["run"][7:]
        )
        self.lib.bcir_decoder_run_accum.restype = ct.c_int

    def use_cblas(self, library, *, symbol="cblas_sgemm", admit=True):
        """Explicit LP64 CBLAS integration; caller configures provider threads (the trainer's
        own pool keeps running every kernel but the GEMM).

        Native adapter calls native BLAS directly, without Python callbacks. Missing
        library/symbol is an error. No implicit framework or backend switch.
        """
        if (
            max(
                self.batch_size * self.context_length,
                self.spec.d_model,
                self.spec.d_ff,
                self.spec.vocab_size,
            )
            > 0x7FFFFFFF
        ):
            raise ValueError("LP64 CBLAS requires dimensions within INT_MAX")
        lib = ct.CDLL(str(library))
        self.use_cblas_function(ct.cast(getattr(lib, symbol), _CBLAS), name=symbol, keep=lib,
                                admit=admit)  # fmt: skip

    def use_cblas_function(self, function, *, name="cblas_sgemm", keep=None, admit=True):
        """`use_cblas` for a CBLAS sgemm the caller already holds as a ctypes function (a
        library's symbol, or a callback). Under the blocked contract the provider is admitted
        by a differential over this trainer's GEMM shapes (`provider_mismatch`): every output
        bit for bit the contract's, at beta 0 and 1, or the provider is refused and the state
        keeps BCIR's kernels. `admit=False` installs it without the differential (for measuring
        the differential itself); under the other contracts a provider's GEMM keeps its own
        arithmetic and `provider_admission` is None."""
        if admit and _CONTRACTS[self.contract] & _BF16:
            raise ValueError(
                f"provider {name} not admitted under the {self.contract} contract: a CBLAS sgemm "
                "computes in binary32, never the bf16 contract's operands"
            )
        slot = ct.cast(function, _CBLAS)
        fn = ct.cast(self.lib.bcir_tensor_cblas_mm, _MM)
        self.state.provider = _Provider(ct.cast(ct.pointer(slot), ct.c_void_p), fn, _SILU())
        self._provider_refs = (keep, function, slot, fn)
        self.backend = f"cblas:{name}"
        self.provider_admission = None
        if admit and _CONTRACTS[self.contract] & _KBLOCK:
            shapes = self.gemm_shapes()
            bad = provider_mismatch(self.lib, self.state.provider.ctx, shapes)
            if bad:
                self.state.provider = _Provider()
                self._provider_refs = ()
                self.backend = "portable-c"
                raise ValueError(
                    f"provider {name} not admitted under the {self.contract} contract: "
                    f"{bad} of {2 * len(shapes)} GEMM cases differ from bcir_tensor_mm_kblock"
                )
            self.provider_admission = {
                "contract": self.contract, "provider": name, "cases": 2 * len(shapes),
                "mismatch": 0,
            }  # fmt: skip

    def gemm_shapes(self):
        """The distinct (ta, tb, m, n, k) of this trainer's step, from the step program."""
        from .native_program import training_step_program

        program = training_step_program(self.spec, self.batch_size, self.context_length)
        return tuple(dict.fromkeys(program.gemm_shapes()))

    def provider_mismatch(self):
        """The admission differential of the installed provider: GEMM cases (this trainer's
        shapes at beta 0 and 1) whose provider output differs from bcir_tensor_mm_kblock."""
        if not self.state.provider.ctx:
            raise ValueError("no CBLAS provider is installed")
        return provider_mismatch(self.lib, self.state.provider.ctx, self.gemm_shapes())

    def _tokens(self, values, *, steps=1):
        n = self.batch_size * self.context_length * steps
        if len(values) != n or any(
            type(v) is not int or not 0 <= v < self.spec.vocab_size for v in values
        ):
            raise ValueError("flat token array has wrong shape or out-of-vocabulary IDs")
        return (ct.c_uint32 * n)(*values)

    def forward(self, tokens):
        x = self._tokens(tokens)
        _check(self.lib.bcir_decoder_forward(ct.byref(self.state), x, len(x)))
        n = self.batch_size * self.context_length * self.spec.vocab_size
        offset = self.activation_count - n  # REMAT: the logits are the arena's last floats
        return array("f", self.buffers["activations"][offset : offset + n])

    def backward(self, tokens, targets, *, accumulate=False):
        if type(accumulate) is not bool:
            raise ValueError("accumulate must be bool")
        x, y, loss = self._tokens(tokens), self._tokens(targets), ct.c_double()
        _check(
            self.lib.bcir_decoder_loss_backward(
                ct.byref(self.state), x, y, len(x), int(accumulate), ct.byref(loss)
            )
        )
        return loss.value

    def update(self, optimizer=NativeAdamW(), *, gradient_scale=1.0):
        if not isinstance(optimizer, NativeAdamW):
            raise ValueError("optimizer must be NativeAdamW")
        opt, norm = _Optimizer(**asdict(optimizer)), ct.c_double()
        _check(
            self.lib.bcir_decoder_update(
                ct.byref(self.state), ct.byref(opt), gradient_scale, ct.byref(norm)
            )
        )
        return norm.value

    def run(
        self,
        tokens,
        targets,
        *,
        steps,
        optimizer=NativeAdamW(),
        learning_rates=None,
        gradient_accumulation=1,
    ):
        if type(steps) is not int or steps < 1 or not isinstance(optimizer, NativeAdamW):
            raise ValueError("run requires positive steps and NativeAdamW")
        if type(gradient_accumulation) is not int or gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be a positive integer")
        x, y = (
            self._tokens(tokens, steps=steps * gradient_accumulation),
            self._tokens(targets, steps=steps * gradient_accumulation),
        )
        rates = None
        if learning_rates is not None:
            if len(learning_rates) != steps:
                raise ValueError("learning rate count must equal steps")
            rates = (ct.c_double * steps)(*learning_rates)
        opt, events, done = _Optimizer(**asdict(optimizer)), (_Event * steps)(), _SZ()
        rc = self.lib.bcir_decoder_run_accum(
            ct.byref(self.state),
            ct.byref(opt),
            x,
            y,
            len(x),
            rates,
            steps,
            gradient_accumulation,
            events,
            steps,
            ct.byref(done),
        )
        if rc:
            raise NativeRunStopped(
                f"native run stopped after {done.value} complete steps (status {rc})",
                rc,
                done.value,
            )
        return tuple({"step": e.step, "loss": e.loss, "grad_norm": e.grad_norm} for e in events)

    def run_program(
        self,
        program,
        tokens,
        targets,
        *,
        steps,
        optimizer=NativeAdamW(),
        learning_rates=None,
        gradient_accumulation=1,
    ):
        """`run` from a launch program (PX-2, `bcir_decoder_run_program`): `program` the words
        `native_launch.program_words` spells -- a planner's decisions for this state -- each
        record the call the step makes for it. The C checks the whole program before anything
        runs (a refusal is status 1, nothing written); the run's outputs are `run`'s. An
        `array('Q')` is lent to the C as it is; any other sequence is checked (ints in
        [0, 2**64), so no bool or float) and converted once a call."""
        if type(steps) is not int or steps < 1 or not isinstance(optimizer, NativeAdamW):
            raise ValueError("run_program requires positive steps and NativeAdamW")
        if type(gradient_accumulation) is not int or gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be a positive integer")
        from .native_launch import RECORD_WORDS

        code = None
        if isinstance(program, array) and program.typecode == "Q" and program.itemsize == 8:
            code = program
        else:
            words = list(program)
            if not set(map(type, words)) - {int}:
                try:
                    code = array("Q", words)  # the range check, in C: OverflowError outside it
                except OverflowError:
                    code = None
        if code is None or len(code) % RECORD_WORDS:
            raise ValueError(f"a program is records of {RECORD_WORDS} unsigned 64-bit words")
        fn = getattr(self.lib, "bcir_decoder_run_program", None)
        if fn is None:
            raise ValueError("this native decoder library has no launch-program runner (PX-2)")
        fn.argtypes = [
            ct.POINTER(_State), ct.POINTER(_Optimizer), ct.POINTER(ct.c_uint64), _SZ, _UP, _UP,
            _SZ, ct.POINTER(ct.c_double), _SZ, _SZ, ct.POINTER(_Event), _SZ, ct.POINTER(_SZ),
        ]  # fmt: skip
        fn.restype = ct.c_int
        x, y = (
            self._tokens(tokens, steps=steps * gradient_accumulation),
            self._tokens(targets, steps=steps * gradient_accumulation),
        )
        rates = None
        if learning_rates is not None:
            if len(learning_rates) != steps:
                raise ValueError("learning rate count must equal steps")
            rates = (ct.c_double * steps)(*learning_rates)
        records = len(code) // RECORD_WORDS
        view = (ct.c_uint64 * len(code)).from_buffer(code) if code else (ct.c_uint64 * 1)()
        opt, events, done = _Optimizer(**asdict(optimizer)), (_Event * steps)(), _SZ()
        rc = fn(ct.byref(self.state), ct.byref(opt), view, records, x, y, len(x), rates, steps,
                gradient_accumulation, events, steps, ct.byref(done))  # fmt: skip
        if rc:
            raise NativeRunStopped(
                f"native program run stopped after {done.value} complete steps (status {rc})",
                rc,
                done.value,
            )
        return tuple({"step": e.step, "loss": e.loss, "grad_norm": e.grad_norm} for e in events)

    def program_decisions(self, *, steps=1, gradient_accumulation=1) -> dict:
        """The decisions a program this trainer runs is stated under, as `native_launch.
        step_program` and `native_pack.training_pack` take them (PX-3): its contract, recompute
        set, workspace and the panels' need, its arenas' alignment, whether a provider holds its
        GEMMs or its SiLU, and the call's steps and accumulation -- every one read off the trainer,
        so a program stated from them is one its `run_program` admits."""
        from .native_launch import SPACES

        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if type(gradient_accumulation) is not int or gradient_accumulation < 1:
            raise ValueError("gradient_accumulation must be a positive integer")
        provider = self.state.provider
        return dict(
            contract=_CONTRACTS[self.contract], recompute=self.recompute,
            workspace_bytes=self.workspace_bytes, panels_need=self.plan_memory.workspace_panels,
            alignment={SPACES[name]: ct.addressof(buf) % 64 for name, buf in self.buffers.items()},
            provider_mm=bool(provider.mm), provider_silu=bool(provider.silu),
            steps=steps, accumulation=gradient_accumulation,
            threads=self.threads, instances=self.instance_families(),
        )  # fmt: skip

    def instance_families(self) -> dict:
        """PX-3b (DISPATCH): the instances this host admits for the trainer's jobs, as masks of
        bcir_tensor.h's bits (1 portable, 2 AVX2, 4 AVX-512, 8 AMX): "gemm" its GEMM contract's
        family (bcir_tensor_mm_contract_variants), "attention" its attention's products' (the bf16
        family under +attnbf16, the fused GEMM's under +attn, v1's otherwise)."""
        lib, bits = self.lib, _CONTRACTS[self.contract]
        for name in ("mm_variants", "mm_fused_variants", "mm_bf16_variants"):
            fn = getattr(lib, "bcir_tensor_" + name)
            fn.argtypes, fn.restype = [], ct.c_int
        lib.bcir_tensor_mm_contract_variants.argtypes = [ct.c_int]
        lib.bcir_tensor_mm_contract_variants.restype = ct.c_int
        gemm = 3 if bits & _BF16 else 2 if bits & _KBLOCK else 1 if bits & 1 else 0
        attention = (lib.bcir_tensor_mm_bf16_variants() if bits & _ATTNBF16 else
                     lib.bcir_tensor_mm_fused_variants() if bits & _ATTN else
                     lib.bcir_tensor_mm_variants())  # fmt: skip
        return {
            "gemm": int(lib.bcir_tensor_mm_contract_variants(gemm)),
            "attention": int(attention),
        }

    def lower_pack(self, pack: bytes, plan: bytes) -> array:
        """PX-2b: the launch program a StreamPack and its ExecutionPlanV1 state for this trainer
        (`native_pack.pack_artifacts` mints them), as the array('Q') `run_program` runs: the two
        verified by their own laws (the plan's alias law among them), then each claim GEM's
        executor dispatches lowered to the record that runs it, every operand at the place its
        value's lifetime gives and live there (`bcir_decoder_lower_pack`). A refusal raises
        `PackRefused` with the C's reason; a library built without `pack=True` has no lowering."""
        from .native_launch import RECORD_WORDS
        from .native_pack import REASONS

        fn = getattr(self.lib, "bcir_decoder_lower_pack", None)
        sizes = getattr(self.lib, "bcir_decoder_pack_sizes", None)
        if fn is None or sizes is None:
            raise ValueError("this native decoder library has no pack lowering (PX-2b): build it "
                             "with pack=True")  # fmt: skip
        if type(pack) not in (bytes, bytearray) or type(plan) not in (bytes, bytearray):
            raise ValueError("a pack and its plan are bytes")
        pack, plan = bytes(pack), bytes(plan)
        sizes.argtypes = [ct.c_char_p, _SZ, ct.c_char_p, _SZ, ct.POINTER(_SZ), ct.POINTER(_SZ),
                          ct.POINTER(ct.c_int)]  # fmt: skip
        sizes.restype = ct.c_int
        records, work_bytes, why = _SZ(), _SZ(), ct.c_int()
        rc = sizes(pack, len(pack), plan, len(plan), ct.byref(records), ct.byref(work_bytes),
                   ct.byref(why))  # fmt: skip
        if rc == 0:
            work = (ct.c_uint64 * max(1, -(-work_bytes.value // 8)))()
            program = array("Q", bytes(8 * RECORD_WORDS * records.value))
            view = (ct.c_uint64 * len(program)).from_buffer(program) if program else None
            fn.argtypes = [ct.POINTER(_State), ct.c_char_p, _SZ, ct.c_char_p, _SZ, ct.c_void_p, _SZ,
                           ct.POINTER(ct.c_uint64), _SZ, ct.POINTER(_SZ), ct.POINTER(ct.c_int)]  # fmt: skip
            fn.restype = ct.c_int
            done = _SZ()
            rc = fn(ct.byref(self.state), pack, len(pack), plan, len(plan), work, ct.sizeof(work),
                    view, records.value, ct.byref(done), ct.byref(why))  # fmt: skip
            if rc == 0:
                return program
        raise PackRefused(
            f"pack refused: {REASONS.get(why.value, why.value)} (status {rc})", rc, why.value
        )

    def run_pack(
        self,
        pack: bytes,
        plan: bytes,
        tokens,
        targets,
        *,
        steps,
        optimizer=NativeAdamW(),
        learning_rates=None,
        gradient_accumulation=1,
    ):
        """`run` from a StreamPack and its plan: `lower_pack`, then `run_program` (PX-2b)."""
        return self.run_program(self.lower_pack(pack, plan), tokens, targets, steps=steps,
                                optimizer=optimizer, learning_rates=learning_rates,
                                gradient_accumulation=gradient_accumulation)  # fmt: skip

    def state_dict(self, *, gradients=False):
        buf = self.buffers["gradients" if gradients else "weights"]
        return {name: array("f", buf[off : off + n]) for name, _, off, n in self.layout}

    def decoder_weights(self):
        """Existing strict HF ingest handles transpose/RoPE permutation for Q8 export."""
        from ...frontends.models.hf_ingest import weights_from_tensors

        values = self.state_dict()
        return weights_from_tensors(
            self.spec, {name: ("F32", shape, values[name]) for name, shape, _, _ in self.layout}
        )

    def load_state_dict(self, values):
        expected = {name for name, *_ in self.layout}
        keys = set(values)
        if self.spec.tied_embeddings and "lm_head.weight" in keys:
            if "model.embed_tokens.weight" not in keys or list(values["lm_head.weight"]) != list(
                values["model.embed_tokens.weight"]
            ):
                raise ValueError("tied head and embedding differ")
            keys.remove("lm_head.weight")
        if keys != expected:
            raise ValueError("parameter inventory differs from native plan")
        staged = array("f")
        for name, _, _, n in self.layout:
            row = array("f", values[name])
            if len(row) != n or any(not math.isfinite(x) for x in row):
                raise ValueError(f"invalid parameter span {name}")
            staged.extend(row)
        ct.memmove(self.buffers["weights"], staged.tobytes(), len(staged) * 4)

    def save(self, path):
        """Atomic weights/moments/counter/bias-power checkpoint, between updates.

        SHA-256 detects corruption, not authenticity. Scratch/gradients recompute.
        Optimizer options and following rate schedule remain caller inputs.
        """
        payload = bytearray()
        for name in ("weights", "moment1", "moment2"):
            values = array("f", self.buffers[name])
            if any(not math.isfinite(x) for x in values) or (
                name == "moment2" and any(x < 0 for x in values)
            ):
                raise ValueError("cannot checkpoint non-finite state")
            if sys.byteorder != "little":
                values.byteswap()
            payload.extend(values.tobytes())
        meta = {
            "schema": "bcir.native_decoder.v2",
            "decoder": asdict(self.spec),
            "contract": self.contract,
            "batch_size": self.batch_size,
            "context_length": self.context_length,
            "step": self.state.step,
            "beta1_power": self.state.beta1_power,
            "beta2_power": self.state.beta2_power,
            "sha256": "",
        }
        meta["sha256"] = _checkpoint_digest(meta, payload)
        header = json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".native-checkpoint-", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(_MAGIC + struct.pack("<I", len(header)) + header + payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, target)
        finally:
            Path(tmp).unlink(missing_ok=True)

    @classmethod
    def resume(
        cls,
        library,
        path,
        *,
        max_memory_bytes=512 * 1024 * 1024,
        threads=1,
        huge_pages=True,
        recompute=0,
    ):
        with Path(path).open("rb") as stream:
            prefix = stream.read(12)
            if len(prefix) != 12 or prefix[:8] != _MAGIC:
                raise ValueError("invalid native checkpoint magic")
            size = struct.unpack("<I", prefix[8:])[0]
            if not 1 <= size <= 65536:
                raise ValueError("invalid native checkpoint header size")
            meta = json.loads(stream.read(size), object_pairs_hook=_unique_object)
            if not isinstance(meta, dict) or set(meta) != {
                "schema",
                "decoder",
                "batch_size",
                "context_length",
                "step",
                "beta1_power",
                "beta2_power",
                "sha256",
                "contract",
            }:
                raise ValueError("invalid native checkpoint fields")
            if meta["schema"] != "bcir.native_decoder.v2":
                raise ValueError("unsupported native checkpoint schema")
            if meta["contract"] not in _CONTRACTS:
                raise ValueError("unknown native checkpoint contract")
            if not isinstance(meta["decoder"], dict) or set(meta["decoder"]) != {
                f.name for f in fields(DecoderSpec)
            }:
                raise ValueError("checkpoint decoder has missing or unknown fields")
            if type(meta["step"]) is not int or not 0 <= meta["step"] < 2**64:
                raise ValueError("invalid checkpoint step")
            for key in ("beta1_power", "beta2_power"):
                v = meta[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
                    raise ValueError("invalid checkpoint bias power")
                if meta["step"] == 0 and v != 1:
                    raise ValueError("initial checkpoint bias powers must be one")
            obj = cls(
                library,
                DecoderSpec(**meta["decoder"]),
                batch_size=meta["batch_size"],
                context_length=meta["context_length"],
                max_memory_bytes=max_memory_bytes,
                threads=threads,
                contract=meta["contract"],
                huge_pages=huge_pages,
                recompute=recompute,  # REMAT: the arena's plan, never the state's: not saved
            )
            expected = obj.state.plan.parameters * 12
            payload = stream.read(expected + 1)
            if len(payload) != expected or _checkpoint_digest(meta, payload) != meta["sha256"]:
                raise ValueError("native checkpoint payload size/digest mismatch")
        values = array("f")
        values.frombytes(payload)
        if sys.byteorder != "little":
            values.byteswap()
        n = obj.state.plan.parameters
        if any(not math.isfinite(v) for v in values) or any(v < 0 for v in values[2 * n :]):
            raise ValueError("invalid native checkpoint numeric state")
        for i, name in enumerate(("weights", "moment1", "moment2")):
            ct.memmove(obj.buffers[name], values[i * n : (i + 1) * n].tobytes(), n * 4)
        obj.state.step = meta["step"]
        obj.state.beta1_power, obj.state.beta2_power = meta["beta1_power"], meta["beta2_power"]
        return obj
