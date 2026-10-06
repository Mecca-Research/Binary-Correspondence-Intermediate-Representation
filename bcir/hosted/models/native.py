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
    """

    @staticmethod
    def build(directory, *, cc=None) -> Path:
        compiler = cc or os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("native decoder training requires a C11 compiler")
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
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-DBCIR_DT_BUILD_SHARED",
                "-shared",
            ]
            if sys.platform != "win32":
                command.append("-fPIC")
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
        self.spec, self.batch_size, self.context_length = spec, batch_size, context_length
        self.library = Path(library).resolve()
        self.lib = ct.CDLL(str(self.library))
        self._bind()
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
        )
        rng = random.Random(seed)
        for name, _, offset, count in self.layout:
            norm = "layernorm" in name or name == "model.norm.weight"
            for i in range(count):
                self.buffers["weights"][offset + i] = 1.0 if norm else rng.gauss(0.0, 0.02)
        self.backend = "portable-c"
        self._provider_refs = ()

    def _bind(self):
        self.lib.bcir_decoder_abi_version.argtypes = []
        self.lib.bcir_decoder_abi_version.restype = ct.c_uint32
        if self.lib.bcir_decoder_abi_version() != 1:
            raise ValueError("unsupported native decoder ABI")
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

    def use_cblas(self, library, *, symbol="cblas_sgemm"):
        """Explicit LP64 CBLAS integration; caller configures provider threads.

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
        slot = ct.cast(getattr(lib, symbol), _CBLAS)
        fn = ct.cast(self.lib.bcir_tensor_cblas_mm, _MM)
        self.state.provider = _Provider(ct.cast(ct.pointer(slot), ct.c_void_p), fn, _SILU())
        self._provider_refs = (lib, slot, fn)
        self.backend = f"cblas:{symbol}"

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
            "schema": "bcir.native_decoder.v1",
            "decoder": asdict(self.spec),
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
    def resume(cls, library, path, *, max_memory_bytes=512 * 1024 * 1024):
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
            }:
                raise ValueError("invalid native checkpoint fields")
            if meta["schema"] != "bcir.native_decoder.v1":
                raise ValueError("unsupported native checkpoint schema")
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
