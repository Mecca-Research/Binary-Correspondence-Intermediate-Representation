"""Native ABI 6 MIXER-EXT sequence-mixer trainer and precision-policy engine.

Pure stdlib Python binding over BCIR's ISO C11 runtime (`bcir_tensor.c`,
`bcir_precision.c`, `bcir_mixer_ext.c`). Supports per-layer pluggable sequence
mixers (`gqa`, `swa`, `deltanet`, `kda`, `mla`, `gdn2`), orthogonal 2025-2026
architectural upgrades (`gate_mode` G1 SDPA sigmoid gate, `qk_norm`, `canon_mode`
Canon-A/C causal conv1d k=4, `mhc_streams` DeepSeek mHC Birkhoff streams,
`online_softmax` TRAIN-OPT-5 O(T) LSE tiled attention), and all 7 native
optimizers (`adamw_fp32`, `adamw_q8_moments`, `adamw_q8_moments_bf16_grad`,
`adamw_adafactor_m2`, `adamw_adafactor_bf16_grad`, `muon_fp32`,
`muon_q8_bf16_grad`).
"""

from __future__ import annotations

import ctypes as ct
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

from ...toolchain import BuildFailed, host_link_args

_ROOT = Path(__file__).resolve().parents[3]
_C = _ROOT / "runtime" / "c"
_UNITS = (
    "bcir_tensor.h",
    "bcir_tensor.c",
    "bcir_precision.h",
    "bcir_precision.c",
    "bcir_mixer_ext.h",
    "bcir_mixer_ext.c",
)

EXT_MAX_LAYERS = 64
EXT_IGNORE = 0xFFFFFFFF

MIXER_KINDS: dict[str, int] = {
    "gqa": 0,
    "swa": 1,
    "deltanet": 2,
    "kda": 3,
    "mla": 4,
    "gdn2": 5,
}

GATE_MODES: dict[str, int] = {
    "none": 0,
    "g1_headwise": 1,
    "g1_lowrank": 2,
}

CANON_MODES: dict[str, int] = {
    "off": 0,
    "canon_a": 1,
    "canon_ac": 2,
}

GEMM_KINDS: dict[str, int] = {
    "fp32_kblock": 0,
    "bf16_ndt": 1,
    "bf16_vdp": 2,
    "fp8_e4m3_blk32": 3,
    "fp8_e5m2_blk32": 4,
    "mxfp4_e2m1_blk32": 5,
    "bcirq8_g32": 6,
}

OPTIMIZER_KINDS: dict[str, int] = {
    "adamw_fp32": 0,
    "adamw_q8_moments": 1,
    "adamw_q8_moments_bf16_grad": 2,
    "adamw_adafactor_m2": 3,
    "adamw_adafactor_bf16_grad": 4,
    "muon_fp32": 5,
    "muon_q8_bf16_grad": 6,
}

INIT_POLICIES: dict[str, int] = {
    "legacy_002": 0,
    "residual_scaled": 1,
    "mup_v1": 2,
}

FP64_GUARD_MODES: dict[str, int] = {
    "off": 0,
    "conditional": 1,
    "always": 2,
}

_CKPT_SCHEMA = "bcir.mixer_ext.checkpoint.v1"


class _PrecPolicy(ct.Structure):
    _fields_ = [
        ("attn_fwd_gemm", ct.c_int),
        ("attn_bwd_gemm", ct.c_int),
        ("mlp_fwd_gemm", ct.c_int),
        ("mlp_bwd_gemm", ct.c_int),
        ("opt_kind", ct.c_int),
        ("fp64_guard_mode", ct.c_int),
        ("rmsnorm_cond_thresh", ct.c_double),
        ("softmax_logit_thresh", ct.c_double),
        ("wy_cond_thresh", ct.c_double),
        ("use_vdp_portable", ct.c_int),
    ]


class _PrecTelemetry(ct.Structure):
    _fields_ = [
        ("fp64_norm_promotions", ct.c_uint64),
        ("fp64_softmax_promotions", ct.c_uint64),
        ("fp64_wy_solves", ct.c_uint64),
        ("nonfinite_events", ct.c_uint64),
        ("grad_clip_hits", ct.c_uint64),
        ("total_steps", ct.c_uint64),
        ("q8_m2_floor_clamps", ct.c_uint64),
        ("gemm_calls_attn", ct.c_uint64),
        ("gemm_calls_mlp", ct.c_uint64),
        ("muon_ns_calls", ct.c_uint64),
        ("max_grad_norm", ct.c_double),
        ("max_softmax_logit", ct.c_double),
        ("min_rms_observed", ct.c_double),
        ("max_wy_offdiag", ct.c_double),
        ("max_abs_act", ct.c_double),
        ("first_tok_attn_mean", ct.c_double),
    ]


class _ExtSpec(ct.Structure):
    _fields_ = [
        ("vocab", ct.c_uint32),
        ("width", ct.c_uint32),
        ("heads", ct.c_uint32),
        ("kvheads", ct.c_uint32),
        ("layers", ct.c_uint32),
        ("ff", ct.c_uint32),
        ("batch", ct.c_uint32),
        ("time", ct.c_uint32),
        ("tied", ct.c_uint32),
        ("rope_base", ct.c_double),
        ("rms_eps", ct.c_double),
        ("layer_mixer", ct.c_uint32 * EXT_MAX_LAYERS),
        ("layer_use_rope", ct.c_uint32 * EXT_MAX_LAYERS),
        ("swa_window", ct.c_uint32),
        ("delta_chunk", ct.c_uint32),
        ("mla_dc", ct.c_uint32),
        ("mla_dr", ct.c_uint32),
        ("init_policy", ct.c_int),
        ("mup_d_base", ct.c_uint32),
        ("seed", ct.c_uint64),
        ("threads", ct.c_uint32),
        ("skip_init", ct.c_int),
        ("gate_mode", ct.c_uint32),
        ("qk_norm", ct.c_uint32),
        ("canon_mode", ct.c_uint32),
        ("mhc_streams", ct.c_uint32),
        ("online_softmax", ct.c_uint32),
        ("gate_rank", ct.c_uint32),
    ]


class _MemStats(ct.Structure):
    _fields_ = [
        ("n_params", ct.c_size_t),
        ("attn_params", ct.c_size_t),
        ("mlp_params", ct.c_size_t),
        ("nonlin_params", ct.c_size_t),
        ("weight_bytes", ct.c_size_t),
        ("opt_only_bytes", ct.c_size_t),
        ("persistent_bytes", ct.c_size_t),
        ("activation_bytes", ct.c_size_t),
        ("mixer_prob_bytes", ct.c_size_t),
        ("scratch_bytes", ct.c_size_t),
        ("total_bytes", ct.c_size_t),
        ("infer_kv_bytes_per_tok", ct.c_size_t),
        ("infer_rec_state_bytes", ct.c_size_t),
        ("init_ms", ct.c_double),
    ]


class _StepMetrics(ct.Structure):
    _fields_ = [
        ("loss", ct.c_double),
        ("grad_norm", ct.c_double),
        ("step_ms", ct.c_double),
        ("clipped", ct.c_int),
        ("status", ct.c_int),
        ("step", ct.c_uint64),
        ("supervised_tokens", ct.c_uint64),
        ("fp64_promotions", ct.c_uint64),
        ("nonfinite_events", ct.c_uint64),
    ]


class _EvalResult(ct.Structure):
    _fields_ = [
        ("ce", ct.c_double),
        ("accuracy", ct.c_double),
        ("tokens", ct.c_uint64),
        ("nonfinite", ct.c_uint64),
        ("eval_ms", ct.c_double),
    ]


class _AdamWParams(ct.Structure):
    _fields_ = [
        ("lr", ct.c_double),
        ("beta1", ct.c_double),
        ("beta2", ct.c_double),
        ("epsilon", ct.c_double),
        ("weight_decay", ct.c_double),
        ("grad_clip", ct.c_double),
    ]


class _TensorInfo(ct.Structure):
    _fields_ = [
        ("offset", ct.c_size_t),
        ("rows", ct.c_size_t),
        ("cols", ct.c_size_t),
        ("layer", ct.c_int),
        ("region", ct.c_int),
        ("name", ct.c_char * 48),
    ]


class _AdaGCConfig(ct.Structure):
    _fields_ = [
        ("enable", ct.c_int),
        ("beta", ct.c_double),
        ("lambda_mult", ct.c_double),
        ("abs_floor", ct.c_double),
        ("warmup_steps", ct.c_uint64),
    ]


class _TensorOptTelemetry(ct.Structure):
    _fields_ = [
        ("last_grad_norm", ct.c_double),
        ("ema_grad_norm", ct.c_double),
        ("last_adagc_scale", ct.c_double),
        ("grad_mom_cos", ct.c_double),
        ("lr_mult", ct.c_double),
        ("adagc_clip_hits", ct.c_uint64),
        ("kahan_enabled", ct.c_int),
    ]


@dataclass(frozen=True)
class MixerExtSpec:
    vocab: int
    width: int
    heads: int
    kvheads: int
    layers: int
    ff: int
    batch: int
    time: int
    layer_mixers: tuple[str, ...] = ()
    layer_use_rope: tuple[bool, ...] = ()
    tied: bool = False
    rope_base: float = 10000.0
    rms_eps: float = 1e-5
    swa_window: int = 32
    delta_chunk: int = 16
    mla_dc: int = 0
    mla_dr: int = 0
    init_policy: str = "mup_v1"
    mup_d_base: int = 128
    seed: int = 42
    threads: int = 1
    skip_init: bool = False
    gate_mode: str = "none"
    qk_norm: bool = False
    canon_mode: str = "off"
    mhc_streams: int = 1
    online_softmax: bool = False
    gate_rank: int = 0

    def to_c(self) -> _ExtSpec:
        if not 1 <= self.layers <= EXT_MAX_LAYERS:
            raise ValueError(f"layers must be in [1, {EXT_MAX_LAYERS}]")
        mixers = self.layer_mixers or ("gqa",) * self.layers
        ropes = self.layer_use_rope or (True,) * self.layers
        if len(mixers) != self.layers or len(ropes) != self.layers:
            raise ValueError("layer_mixers and layer_use_rope must match layers count")
        if self.init_policy not in INIT_POLICIES:
            raise ValueError(f"unknown init_policy {self.init_policy!r}")
        if self.gate_mode not in GATE_MODES:
            raise ValueError(f"unknown gate_mode {self.gate_mode!r}")
        if self.canon_mode not in CANON_MODES:
            raise ValueError(f"unknown canon_mode {self.canon_mode!r}")
        if self.mhc_streams not in (0, 1, 2, 4):
            raise ValueError("mhc_streams must be 0, 1, 2, or 4")
        c = _ExtSpec()
        c.vocab = self.vocab
        c.width = self.width
        c.heads = self.heads
        c.kvheads = self.kvheads
        c.layers = self.layers
        c.ff = self.ff
        c.batch = self.batch
        c.time = self.time
        c.tied = 1 if self.tied else 0
        c.rope_base = self.rope_base
        c.rms_eps = self.rms_eps
        for i, (mx, rp) in enumerate(zip(mixers, ropes)):
            if mx not in MIXER_KINDS:
                raise ValueError(f"unknown mixer kind {mx!r}")
            c.layer_mixer[i] = MIXER_KINDS[mx]
            c.layer_use_rope[i] = 1 if rp else 0
        c.swa_window = self.swa_window
        c.delta_chunk = self.delta_chunk
        c.mla_dc = self.mla_dc
        c.mla_dr = self.mla_dr
        c.init_policy = INIT_POLICIES[self.init_policy]
        c.mup_d_base = self.mup_d_base
        c.seed = self.seed
        c.threads = self.threads
        c.skip_init = 1 if self.skip_init else 0
        c.gate_mode = GATE_MODES[self.gate_mode]
        c.qk_norm = 1 if self.qk_norm else 0
        c.canon_mode = CANON_MODES[self.canon_mode]
        c.mhc_streams = self.mhc_streams
        c.online_softmax = 1 if self.online_softmax else 0
        c.gate_rank = self.gate_rank
        return c


@dataclass(frozen=True)
class PrecisionPolicy:
    attn_fwd_gemm: str = "fp32_kblock"
    attn_bwd_gemm: str = "fp32_kblock"
    mlp_fwd_gemm: str = "fp32_kblock"
    mlp_bwd_gemm: str = "fp32_kblock"
    opt_kind: str = "adamw_fp32"
    fp64_guard_mode: str = "conditional"
    rmsnorm_cond_thresh: float = 1e-4
    softmax_logit_thresh: float = 25.0
    wy_cond_thresh: float = 0.0
    use_vdp_portable: bool = False

    def to_c(self) -> _PrecPolicy:
        for name in (self.attn_fwd_gemm, self.attn_bwd_gemm, self.mlp_fwd_gemm, self.mlp_bwd_gemm):
            if name not in GEMM_KINDS:
                raise ValueError(f"unknown gemm kind {name!r}")
        if self.opt_kind not in OPTIMIZER_KINDS:
            raise ValueError(f"unknown opt_kind {self.opt_kind!r}")
        if self.fp64_guard_mode not in FP64_GUARD_MODES:
            raise ValueError(f"unknown fp64_guard_mode {self.fp64_guard_mode!r}")
        return _PrecPolicy(
            GEMM_KINDS[self.attn_fwd_gemm],
            GEMM_KINDS[self.attn_bwd_gemm],
            GEMM_KINDS[self.mlp_fwd_gemm],
            GEMM_KINDS[self.mlp_bwd_gemm],
            OPTIMIZER_KINDS[self.opt_kind],
            FP64_GUARD_MODES[self.fp64_guard_mode],
            self.rmsnorm_cond_thresh,
            self.softmax_logit_thresh,
            self.wy_cond_thresh,
            1 if self.use_vdp_portable else 0,
        )


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 3.0

    def to_c(self) -> _AdamWParams:
        vals = asdict(self)
        if any(not math.isfinite(v) for v in vals.values()):
            raise ValueError("optimizer hyperparameters must be finite")
        return _AdamWParams(
            self.lr, self.beta1, self.beta2, self.epsilon, self.weight_decay, self.grad_clip
        )


@dataclass(frozen=True)
class AdaGCConfig:
    enable: bool = True
    beta: float = 0.98
    lambda_mult: float = 1.05
    abs_floor: float = 1e-4
    warmup_steps: int = 1

    def to_c(self) -> _AdaGCConfig:
        if (
            not math.isfinite(self.beta)
            or not math.isfinite(self.lambda_mult)
            or not math.isfinite(self.abs_floor)
        ):
            raise ValueError("AdaGC parameters must be finite")
        return _AdaGCConfig(
            1 if self.enable else 0,
            self.beta,
            self.lambda_mult,
            self.abs_floor,
            max(1, int(self.warmup_steps)),
        )


def _bind_lib(lib: ct.CDLL) -> None:
    lib.bcir_mixer_ext_abi_version.argtypes = []
    lib.bcir_mixer_ext_abi_version.restype = ct.c_uint32
    lib.bcir_ext_create.argtypes = [ct.POINTER(_ExtSpec), ct.POINTER(_PrecPolicy)]
    lib.bcir_ext_create.restype = ct.c_void_p
    lib.bcir_ext_destroy.argtypes = [ct.c_void_p]
    lib.bcir_ext_destroy.restype = None
    lib.bcir_ext_get_mem_stats.argtypes = [ct.c_void_p, ct.POINTER(_MemStats)]
    lib.bcir_ext_get_mem_stats.restype = None
    lib.bcir_ext_get_telemetry.argtypes = [ct.c_void_p, ct.POINTER(_PrecTelemetry)]
    lib.bcir_ext_get_telemetry.restype = None
    lib.bcir_ext_reset_telemetry.argtypes = [ct.c_void_p]
    lib.bcir_ext_reset_telemetry.restype = None
    lib.bcir_ext_reset_opt.argtypes = [ct.c_void_p]
    lib.bcir_ext_reset_opt.restype = None
    lib.bcir_ext_opt_state_bytes.argtypes = [ct.c_void_p]
    lib.bcir_ext_opt_state_bytes.restype = ct.c_size_t
    lib.bcir_ext_save_opt_state.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    lib.bcir_ext_save_opt_state.restype = ct.c_int
    lib.bcir_ext_load_opt_state.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t]
    lib.bcir_ext_load_opt_state.restype = ct.c_int
    lib.bcir_ext_weights_ptr.argtypes = [ct.c_void_p]
    lib.bcir_ext_weights_ptr.restype = ct.POINTER(ct.c_float)
    lib.bcir_ext_param_count.argtypes = [ct.c_void_p]
    lib.bcir_ext_param_count.restype = ct.c_size_t
    lib.bcir_ext_tensor_count.argtypes = [ct.c_void_p]
    lib.bcir_ext_tensor_count.restype = ct.c_size_t
    lib.bcir_ext_tensor_info_at.argtypes = [ct.c_void_p, ct.c_size_t, ct.POINTER(_TensorInfo)]
    lib.bcir_ext_tensor_info_at.restype = ct.c_int
    lib.bcir_ext_configure_adagc.argtypes = [ct.c_void_p, ct.POINTER(_AdaGCConfig)]
    lib.bcir_ext_configure_adagc.restype = ct.c_int
    lib.bcir_ext_configure_kahan.argtypes = [ct.c_void_p, ct.c_int, ct.c_int]
    lib.bcir_ext_configure_kahan.restype = ct.c_int
    lib.bcir_ext_enable_kahan_tensor.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_int]
    lib.bcir_ext_enable_kahan_tensor.restype = ct.c_int
    lib.bcir_ext_set_tensor_lr_mult.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_double]
    lib.bcir_ext_set_tensor_lr_mult.restype = ct.c_int
    lib.bcir_ext_enable_tensor_telemetry.argtypes = [ct.c_void_p, ct.c_int]
    lib.bcir_ext_enable_tensor_telemetry.restype = ct.c_int
    lib.bcir_ext_get_tensor_telemetry.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.POINTER(_TensorOptTelemetry),
    ]
    lib.bcir_ext_get_tensor_telemetry.restype = ct.c_int
    lib.bcir_ext_forward.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
        ct.c_uint32,
        ct.POINTER(ct.c_float),
    ]
    lib.bcir_ext_forward.restype = ct.c_int
    lib.bcir_ext_evaluate.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.c_size_t,
        ct.c_uint32,
        ct.POINTER(_EvalResult),
    ]
    lib.bcir_ext_evaluate.restype = ct.c_int
    lib.bcir_ext_train_loop.argtypes = [
        ct.c_void_p,
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_uint32),
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(_AdamWParams),
        ct.POINTER(_StepMetrics),
    ]
    lib.bcir_ext_train_loop.restype = ct.c_int
    for fn_name in (
        "bcir_deltanet_verify_wy_vs_recurrent",
        "bcir_kda_verify_wy_vs_recurrent",
        "bcir_gdn2_verify_wy_vs_recurrent",
        "bcir_mla_verify_absorption",
    ):
        fn = getattr(lib, fn_name)
        fn.argtypes = [ct.c_uint32] * 5 + [ct.POINTER(ct.c_double)]
        fn.restype = ct.c_int
    lib.bcir_mhc_verify_birkhoff.argtypes = [
        ct.c_uint32,
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    lib.bcir_mhc_verify_birkhoff.restype = ct.c_int
    lib.bcir_newton_schulz_verify.argtypes = [
        ct.c_void_p,
        ct.c_size_t,
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    ]
    lib.bcir_newton_schulz_verify.restype = ct.c_int


class NativeMixerExt:
    """Managed handle to BCIR's ABI 6 MIXER-EXT native C11 trainer."""

    @staticmethod
    def build(directory: str | Path, *, cc: str | None = None) -> Path:
        compiler = cc or os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
        if not compiler:
            raise RuntimeError("native MIXER-EXT training requires a C11 compiler")
        missing = [u for u in _UNITS if not (_C / u).is_file()]
        if missing:
            raise RuntimeError(
                f"MIXER-EXT sources not found under {_C} ({', '.join(missing)}): "
                "build from a BCIR source checkout"
            )
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = (
            ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
        )
        target = target_dir / f"bcir_mixer_ext{suffix}"
        fd, name = tempfile.mkstemp(prefix=".mixer_ext-", suffix=suffix, dir=target_dir)
        os.close(fd)
        temporary = Path(name)
        try:
            command = [
                compiler,
                "-std=c11",
                "-O3",
                "-ffp-contract=off",
                "-fno-math-errno",
                "-Wall",
                "-Wextra",
                "-Wpedantic",
                "-Werror",
                "-DBCIR_DT_BUILD_SHARED",
                "-shared",
            ]
            if sys.platform != "win32":
                command += ["-fPIC", "-pthread"]
            command += ["-I", str(_C), *(str(_C / u) for u in _UNITS if u.endswith(".c"))]
            command += ["-o", str(temporary)]
            result = subprocess.run(
                command + host_link_args(["-lm"]), capture_output=True, text=True, timeout=120
            )
            if result.returncode:
                raise BuildFailed("native MIXER-EXT build", result.stderr)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    @staticmethod
    def load_library(library: str | Path) -> ct.CDLL:
        lib = ct.CDLL(str(Path(library).resolve()))
        _bind_lib(lib)
        return lib

    def __init__(
        self,
        library: str | Path,
        spec: MixerExtSpec,
        policy: PrecisionPolicy | None = None,
    ) -> None:
        self.spec = spec
        self.policy = policy or PrecisionPolicy()
        self.lib = self.load_library(library)
        if self.lib.bcir_mixer_ext_abi_version() != 6:
            raise RuntimeError("expected BCIR MIXER-EXT ABI version 6")
        c_spec = spec.to_c()
        c_pol = self.policy.to_c()
        self._handle = self.lib.bcir_ext_create(ct.byref(c_spec), ct.byref(c_pol))
        if not self._handle:
            raise ValueError("bcir_ext_create refused specification or policy")

    def close(self) -> None:
        if getattr(self, "_handle", None):
            self.lib.bcir_ext_destroy(self._handle)
            self._handle = None

    def __enter__(self) -> NativeMixerExt:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    @property
    def n_params(self) -> int:
        return int(self.lib.bcir_ext_param_count(self._handle))

    def mem_stats(self) -> dict[str, int | float]:
        ms = _MemStats()
        self.lib.bcir_ext_get_mem_stats(self._handle, ct.byref(ms))
        return {name: getattr(ms, name) for name, _ in _MemStats._fields_}

    def telemetry(self) -> dict[str, int | float]:
        tel = _PrecTelemetry()
        self.lib.bcir_ext_get_telemetry(self._handle, ct.byref(tel))
        return {name: getattr(tel, name) for name, _ in _PrecTelemetry._fields_}

    def reset_telemetry(self) -> None:
        self.lib.bcir_ext_reset_telemetry(self._handle)

    def reset_optimizer(self) -> None:
        self.lib.bcir_ext_reset_opt(self._handle)

    def tensor_layout(self) -> tuple[dict[str, int | str], ...]:
        count = int(self.lib.bcir_ext_tensor_count(self._handle))
        out = []
        info = _TensorInfo()
        for i in range(count):
            if self.lib.bcir_ext_tensor_info_at(self._handle, i, ct.byref(info)) != 0:
                raise RuntimeError(f"tensor_info_at failed for index {i}")
            out.append(
                {
                    "name": info.name.decode("utf-8"),
                    "offset": int(info.offset),
                    "rows": int(info.rows),
                    "cols": int(info.cols),
                    "layer": int(info.layer),
                    "region": int(info.region),
                }
            )
        return tuple(out)

    def configure_adagc(self, cfg: AdaGCConfig) -> None:
        c_cfg = cfg.to_c()
        rc = self.lib.bcir_ext_configure_adagc(self._handle, ct.byref(c_cfg))
        if rc != 0:
            raise ValueError(f"bcir_ext_configure_adagc failed with status {rc}")

    def configure_kahan(self, *, embed: bool = True, norm: bool = True) -> None:
        rc = self.lib.bcir_ext_configure_kahan(self._handle, 1 if embed else 0, 1 if norm else 0)
        if rc != 0:
            raise ValueError(f"bcir_ext_configure_kahan failed with status {rc}")

    def enable_kahan_tensor(self, idx: int, enable: bool = True) -> None:
        rc = self.lib.bcir_ext_enable_kahan_tensor(self._handle, int(idx), 1 if enable else 0)
        if rc != 0:
            raise ValueError(f"bcir_ext_enable_kahan_tensor failed with status {rc}")

    def set_tensor_lr_mult(self, idx: int, lr_mult: float) -> None:
        rc = self.lib.bcir_ext_set_tensor_lr_mult(self._handle, int(idx), float(lr_mult))
        if rc != 0:
            raise ValueError(f"bcir_ext_set_tensor_lr_mult failed with status {rc}")

    def enable_tensor_telemetry(self, enable: bool = True) -> None:
        rc = self.lib.bcir_ext_enable_tensor_telemetry(self._handle, 1 if enable else 0)
        if rc != 0:
            raise ValueError(f"bcir_ext_enable_tensor_telemetry failed with status {rc}")

    def tensor_telemetry(self) -> tuple[dict[str, int | float | str], ...]:
        layout = self.tensor_layout()
        out = []
        tel = _TensorOptTelemetry()
        for i, meta in enumerate(layout):
            if self.lib.bcir_ext_get_tensor_telemetry(self._handle, i, ct.byref(tel)) != 0:
                raise RuntimeError(f"get_tensor_telemetry failed for index {i}")
            out.append(
                {
                    "index": i,
                    "name": meta["name"],
                    "last_grad_norm": float(tel.last_grad_norm),
                    "ema_grad_norm": float(tel.ema_grad_norm),
                    "last_adagc_scale": float(tel.last_adagc_scale),
                    "grad_mom_cos": float(tel.grad_mom_cos),
                    "lr_mult": float(tel.lr_mult),
                    "adagc_clip_hits": int(tel.adagc_clip_hits),
                    "kahan_enabled": bool(tel.kahan_enabled),
                }
            )
        return tuple(out)

    def weights_bytes(self) -> bytes:
        n = self.n_params
        ptr = self.lib.bcir_ext_weights_ptr(self._handle)
        return ct.string_at(ptr, n * 4)

    def load_weights_bytes(self, raw: bytes) -> None:
        n = self.n_params
        if len(raw) != n * 4:
            raise ValueError(f"weight payload size {len(raw)} != expected {n * 4}")
        ptr = self.lib.bcir_ext_weights_ptr(self._handle)
        ct.memmove(ptr, raw, len(raw))

    def opt_state_bytes(self) -> bytes:
        nbytes = int(self.lib.bcir_ext_opt_state_bytes(self._handle))
        if nbytes <= 0:
            raise RuntimeError("optimizer state size unavailable")
        buf = ct.create_string_buffer(nbytes)
        rc = self.lib.bcir_ext_save_opt_state(self._handle, buf, nbytes)
        if rc != 0:
            raise ValueError(f"bcir_ext_save_opt_state failed with status {rc}")
        return buf.raw

    def load_opt_state_bytes(self, raw: bytes) -> None:
        buf = ct.create_string_buffer(raw, len(raw))
        rc = self.lib.bcir_ext_load_opt_state(self._handle, buf, len(raw))
        if rc != 0:
            raise ValueError(f"bcir_ext_load_opt_state failed with status {rc}")

    @staticmethod
    def _ckpt_stem_and_json(path: str | Path) -> tuple[str, Path]:
        base = Path(path)
        stem = base.stem if base.suffix in (".json", ".bin") else base.name
        return stem, base.parent / f"{stem}.json"

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        step: int = 0,
        extra: dict | None = None,
        include_opt_state: bool = True,
        _abort_before_commit: bool = False,
    ) -> str:
        """Save crash-atomic SHA-256-verified JSON metadata, FP32 weights, and optimizer state.

        Writes new generation-tagged `.bin`/`.opt.bin` payloads first, commits `.json` via
        `os.replace`, and unlinks the prior generation's payloads only after commit so an
        interruption mid-overwrite preserves the last valid checkpoint generation.
        """
        stem, json_path = self._ckpt_stem_and_json(path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        prev_meta: dict[str, object] | None = None
        prev_gen = 0
        if json_path.exists():
            try:
                loaded = json.loads(json_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prev_meta = loaded
                    raw_gen = loaded.get("generation", 0)
                    prev_gen = int(raw_gen) if isinstance(raw_gen, int) else 0
            except (OSError, ValueError, TypeError):
                prev_meta = None
                prev_gen = 0
        gen = prev_gen + 1

        payload = self.weights_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        gen_prefix = f"{stem}.g{gen}_{digest[:8]}"
        bin_path = json_path.parent / f"{gen_prefix}.bin"
        meta: dict[str, object] = {
            "schema": _CKPT_SCHEMA,
            "abi_version": 6,
            "generation": gen,
            "step": int(step),
            "n_params": self.n_params,
            "weights_sha256": digest,
            "weights_file": bin_path.name,
            "spec": asdict(self.spec),
            "policy": asdict(self.policy),
            "extra": extra or {},
        }
        tmp_bin = json_path.parent / f".{gen_prefix}.bin.tmp"
        tmp_bin.write_bytes(payload)
        os.replace(tmp_bin, bin_path)
        new_files = {bin_path.name}
        if _abort_before_commit:
            raise RuntimeError("simulated mid-save interruption before metadata commit")
        if include_opt_state:
            opt_path = json_path.parent / f"{gen_prefix}.opt.bin"
            opt_payload = self.opt_state_bytes()
            opt_digest = hashlib.sha256(opt_payload).hexdigest()
            tmp_opt = json_path.parent / f".{gen_prefix}.opt.bin.tmp"
            tmp_opt.write_bytes(opt_payload)
            os.replace(tmp_opt, opt_path)
            meta["opt_state_file"] = opt_path.name
            meta["opt_state_sha256"] = opt_digest
            meta["opt_state_bytes"] = len(opt_payload)
            new_files.add(opt_path.name)
        tmp_json = json_path.parent / f".{stem}.g{gen}.json.tmp"
        tmp_json.write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(tmp_json, json_path)
        if prev_meta is not None:
            for key in ("weights_file", "opt_state_file"):
                old_name = prev_meta.get(key)
                if isinstance(old_name, str) and old_name not in new_files:
                    (json_path.parent / old_name).unlink(missing_ok=True)
        return digest

    def load_checkpoint(self, path: str | Path, *, load_opt_state: bool = True) -> dict:
        _, json_path = self._ckpt_stem_and_json(path)
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        if meta.get("schema") != _CKPT_SCHEMA:
            raise ValueError(f"unsupported checkpoint schema {meta.get('schema')!r}")
        bin_path = json_path.parent / meta["weights_file"]
        payload = bin_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != meta["weights_sha256"]:
            raise ValueError("checkpoint weights SHA-256 mismatch")
        self.load_weights_bytes(payload)
        if load_opt_state and "opt_state_file" in meta:
            opt_path = json_path.parent / meta["opt_state_file"]
            opt_payload = opt_path.read_bytes()
            opt_digest = hashlib.sha256(opt_payload).hexdigest()
            if opt_digest != meta["opt_state_sha256"]:
                raise ValueError("checkpoint optimizer state SHA-256 mismatch")
            self.load_opt_state_bytes(opt_payload)
        return meta

    def train_steps(
        self,
        tokens: list[int] | array,
        targets: list[int] | array,
        *,
        steps: int,
        opt: OptimizerConfig,
        lrs: list[float] | None = None,
        active_times: list[int] | None = None,
    ) -> list[dict[str, int | float]]:
        stride = self.spec.batch * self.spec.time
        if len(tokens) < steps * stride or len(targets) < steps * stride:
            raise ValueError("token/target buffer shorter than steps * batch * time")
        tk_arr = (ct.c_uint32 * len(tokens))(*tokens)
        tg_arr = (ct.c_uint32 * len(targets))(*targets)
        at_ptr = None
        if active_times is not None:
            if len(active_times) < steps:
                raise ValueError("active_times shorter than steps")
            at_arr = (ct.c_uint32 * steps)(*active_times[:steps])
            at_ptr = at_arr
        lr_ptr = None
        if lrs is not None:
            if len(lrs) < steps:
                raise ValueError("lrs shorter than steps")
            lr_arr = (ct.c_double * steps)(*lrs[:steps])
            lr_ptr = lr_arr
        c_opt = opt.to_c()
        events = (_StepMetrics * steps)()
        rc = self.lib.bcir_ext_train_loop(
            self._handle, tk_arr, tg_arr, at_ptr, lr_ptr, steps, ct.byref(c_opt), events
        )
        if rc != 0:
            raise ValueError(f"bcir_ext_train_loop failed with status {rc}")
        return [
            {name: getattr(events[i], name) for name, _ in _StepMetrics._fields_}
            for i in range(steps)
        ]

    def evaluate(
        self,
        tokens: list[int] | array,
        targets: list[int] | array,
        *,
        n_batches: int,
        active_time: int | None = None,
    ) -> dict[str, int | float]:
        act_t = self.spec.time if active_time is None else active_time
        need = n_batches * self.spec.batch * act_t
        if len(tokens) < need or len(targets) < need:
            raise ValueError("eval token/target buffer too short")
        tk_arr = (ct.c_uint32 * need)(*tokens[:need])
        tg_arr = (ct.c_uint32 * need)(*targets[:need])
        res = _EvalResult()
        rc = self.lib.bcir_ext_evaluate(
            self._handle, tk_arr, tg_arr, n_batches, act_t, ct.byref(res)
        )
        if rc != 0:
            raise ValueError(f"bcir_ext_evaluate failed with status {rc}")
        return {name: getattr(res, name) for name, _ in _EvalResult._fields_}
