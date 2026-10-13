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
from ...toolchain import host_link_args

_ROOT = Path(__file__).resolve().parents[3]
_C = _ROOT / "runtime" / "c"
_UNITS = ("bcir_tensor.h", "bcir_tensor.c", "bcir_decoder_train.h", "bcir_decoder_train.c")
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
#: "reduce" (NDT-REDUCE) puts the clipping norm under a fixed-shape reduction -- the gradient in
#: chunks of `reduce_chunk` elements, each chunk's squares summed in index order in double from
#: zero, the chunk sums added in chunk order from zero -- so the update's one pass runs on the pool;
#: "<contract>+reduce" names it with any of the five above. Whatever the contract, the loss, the
#: gamma adjoints and the embedding scatter run on the pool in the serial order: every loss and
#: state is the serial one, bit for bit, at every thread count.
_CONTRACTS = {"v1": 0, "fused": 1, "exp": 2, "fused+exp": 3, "kblock": 4, "kblock+exp": 6}
_CONTRACTS.update({("reduce" if name == "v1" else name + "+reduce"): bits | 8
                   for name, bits in list(_CONTRACTS.items())})  # fmt: skip
_KBLOCK = 4
_REDUCE = 8


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
    VJP and the softmax; `"fused+exp"`); the checkpoint names it and `resume` restores it.
    """

    @staticmethod
    def build(directory, *, cc=None) -> Path:
        compiler = cc or os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("native decoder training requires a C11 compiler")
        missing = [u for u in _UNITS if not (_C / u).is_file()]
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
        target = target_dir / ("bcir_decoder_train" + suffix)
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
            if sys.platform != "win32":
                # the pool's threads (NDT-THREADS) are pthreads here; Win32 threads need no flag
                command += ["-fPIC", "-pthread"]
            command += [
                "-I",
                str(_C),
                str(_C / "bcir_tensor.c"),
                str(_C / "bcir_decoder_train.c"),
                "-o",
                str(temporary),
            ]
            result = subprocess.run(
                command + host_link_args(["-lm"]), capture_output=True, text=True, timeout=120
            )
            if result.returncode:
                raise RuntimeError(f"native decoder build failed:\n{result.stderr}")
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
        self.memory_bytes = (4 * plan.parameters + plan.activations + plan.scratch) * 4
        if self.memory_bytes > max_memory_bytes:
            raise ValueError(
                f"native plan needs {self.memory_bytes} bytes, budget {max_memory_bytes}"
            )
        self.layout = parameter_layout(spec)
        if sum(row[3] for row in self.layout) != plan.parameters:
            raise RuntimeError("native/Python parameter layout mismatch")
        self.buffers = {
            name: (ct.c_float * n)()
            for name, n in (
                ("weights", plan.parameters),
                ("gradients", plan.parameters),
                ("moment1", plan.parameters),
                ("moment2", plan.parameters),
                ("activations", plan.activations),
                ("scratch", plan.scratch),
            )
        }
        self.state = _State(
            plan,
            *(self.buffers[n] for n in self.buffers),
            plan.parameters,
            plan.parameters,
            plan.parameters,
            plan.parameters,
            plan.activations,
            plan.scratch,
            0,
            1.0,
            1.0,
            _Provider(),
            self._pool,
            _CONTRACTS[contract],
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
        if self.lib.bcir_decoder_abi_version() != 3:
            raise ValueError("unsupported native decoder ABI")
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
        offset = self.state.plan.activations - n
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
            raise ValueError(f"native run stopped after {done.value} complete steps (status {rc})")
        return tuple({"step": e.step, "loss": e.loss, "grad_norm": e.grad_norm} for e in events)

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
    def resume(cls, library, path, *, max_memory_bytes=512 * 1024 * 1024, threads=1):
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
