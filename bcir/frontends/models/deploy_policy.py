"""BCIR Deployment Policy Qualification Engine (COMP-1 / P1-P6).

Implements the explicit deployment precision law derived from the BCIR
precision-frontier, LittleBit sub-1-bit, and model-architecture studies:

- **P1 (Explicit Precision Contract):** every deployable artifact carries an
  explicit :class:`PrecisionContract` naming weight store, Attention format,
  MLP format, activation/accumulation/KV-cache dtypes, and scale format.
- **P2 (Hybrid MLP-Only Width-Gated Law):** ``LUT4_1B`` packed-1-bit factorized
  weights are permitted ONLY on MLP projections (``W_GATE``, ``W_UP``,
  ``W_DOWN``) when ``min(d_model, d_ff) >= min_lut4_dim`` (default ``512``).
  Attention (``W_Q/W_K/W_V/W_O``), embeddings, RMSNorms, and ``lm_head`` stay
  in deterministic per-group symmetric Q8.
- **P3 (Selective Promotion / Fail-Closed Fallback):** models failing the
  hybrid qualification budget either selectively promote sensitive MLP layers
  to Q8 (``PROMOTED_SELECTIVE_V2``), promote the whole model to ``BCIRQ8 v1``
  (``PROMOTED_Q8_V1``), or refuse deployment (``REFUSED``) when promotion is
  disabled or hard numerical/determinism gates fail.
- **P4 (Two-Stage Qualification):** uniform Q8 / Q8-Attention is qualified
  separately before admitting any hybrid ``LUT4_1B`` MLP compression.
- **P5 (Deterministic Bounded-Error Verification):** verifies the exact
  per-group Q8 error bound ``|w - q * 2^e| <= 2^(e - 1)`` on serialized
  ``BCIRQ8`` bytes and enforces 0 bit drift across SIMD kernel instances.
- **P6 (Versioned Baseline Record):** emits and strictly validates
  ``bcir.deploy_baseline.v1`` JSON records for reproducible regression checks.

This module is cold-quarantined (:mod:`tools.perf.import_graph`) and pure
Python stdlib by default; ``numpy`` is imported lazily inside factorization
helpers when available for fast SVD on wide matrices.
"""

from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .decode import DecoderSpec, DecoderWeights
from .weights_io import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    DEFAULT_MAX_DECODED_ELEMENTS,
    DIRECTORY_ENTRY_SIZE,
    HEADER_SIZE,
    LUT4_ALLOWED_TENSOR_IDS,
    LUT4_MAX_PATHS,
    LUT4_MAX_RANK,
    TENSOR_EMBEDDING,
    TENSOR_ENC_LUT4_1B,
    TENSOR_ENC_Q8,
    TENSOR_G_ATTN,
    TENSOR_G_FF,
    TENSOR_G_FINAL,
    TENSOR_LM_HEAD,
    TENSOR_W_DOWN,
    TENSOR_W_GATE,
    TENSOR_W_K,
    TENSOR_W_O,
    TENSOR_W_Q,
    TENSOR_W_UP,
    TENSOR_W_V,
    Lut4Path,
    Lut4Tensor,
    Q8ArtifactMetadata,
    _DIRECTORY,
    _HEADER_PREFIX,
    decode_fp16_le,
    encode_fp16_le,
    lut4_packed_bytes,
    pack_lut4_signs,
    write_q8_decoder,
)

if TYPE_CHECKING:
    from ...kbcir.native_ai import NativeAIKernels

DEPLOY_BASELINE_SCHEMA = "bcir.deploy_baseline.v1"

_VALID_WEIGHT_STORES = frozenset({"fp32", "bcirq8_v1", "bcirq8_v2_hybrid_lut4_mlp"})
_VALID_ATTN_FORMATS = frozenset({"fp32", "q8_g32"})
_VALID_MLP_FORMATS = frozenset({"fp32", "q8_g32", "lut4_1b_1p", "lut4_1b_2p"})
_VALID_ACT_DTYPES = frozenset({"fp64", "fp32"})
_VALID_SCALE_DTYPES = frozenset({"none", "pow2_i16", "fp16_le", "pow2_i16+fp16_le"})
_VALID_VERDICT_STATUSES = frozenset(
    {"ADMITTED_HYBRID_V2", "PROMOTED_SELECTIVE_V2", "PROMOTED_Q8_V1", "REFUSED"}
)

_FP16_MAX_SAFE = 60000.0
_FP16_MIN_NORM = 1e-4


@dataclass(frozen=True)
class PrecisionContract:
    """P1 explicit deployment precision contract."""

    name: str
    weight_store: str
    attention_weight_format: str
    mlp_weight_format: str
    activation_dtype: str = "fp64"
    accumulation_dtype: str = "fp64"
    kv_cache_dtype: str = "fp64"
    scale_dtype: str = "pow2_i16"
    group_size: int = 32
    lut4_paths: int = 0
    lut4_target_bpw: float = 0.0
    deterministic_zero_drift: bool = True
    bounded_q8_error: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("PrecisionContract.name must be a non-empty string")
        if self.weight_store not in _VALID_WEIGHT_STORES:
            raise ValueError(f"unsupported weight_store: {self.weight_store!r}")
        if self.attention_weight_format not in _VALID_ATTN_FORMATS:
            raise ValueError(
                f"attention_weight_format {self.attention_weight_format!r} refused "
                "(Attention must remain in fp32 or q8_g32, never 1-bit LUT4)"
            )
        if self.mlp_weight_format not in _VALID_MLP_FORMATS:
            raise ValueError(f"unsupported mlp_weight_format: {self.mlp_weight_format!r}")
        if (
            self.activation_dtype not in _VALID_ACT_DTYPES
            or self.accumulation_dtype not in _VALID_ACT_DTYPES
            or self.kv_cache_dtype not in _VALID_ACT_DTYPES
        ):
            raise ValueError("activation/accumulation/kv_cache dtypes must be fp64 or fp32")
        if self.scale_dtype not in _VALID_SCALE_DTYPES:
            raise ValueError(f"unsupported scale_dtype: {self.scale_dtype!r}")
        if type(self.group_size) is not int or not 1 <= self.group_size <= 0xFFFF:
            raise ValueError("group_size must be in [1, 65535]")
        if type(self.lut4_paths) is not int or not 0 <= self.lut4_paths <= LUT4_MAX_PATHS:
            raise ValueError(f"lut4_paths must be in [0, {LUT4_MAX_PATHS}]")
        if not math.isfinite(self.lut4_target_bpw) or self.lut4_target_bpw < 0.0:
            raise ValueError("lut4_target_bpw must be finite and >= 0")
        if self.weight_store == "bcirq8_v2_hybrid_lut4_mlp":
            if self.mlp_weight_format not in ("lut4_1b_1p", "lut4_1b_2p"):
                raise ValueError("BCIRQ8 v2 hybrid contract requires a lut4_1b MLP format")
            if self.lut4_paths not in (1, 2):
                raise ValueError("BCIRQ8 v2 hybrid contract requires lut4_paths in {1, 2}")
            if self.attention_weight_format != "q8_g32":
                raise ValueError("BCIRQ8 v2 hybrid contract requires q8_g32 Attention")
        elif self.lut4_paths != 0:
            raise ValueError("non-hybrid precision contract must have lut4_paths == 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrecisionContract":
        return cls(**dict(data))


CONTRACT_FP32_REF = PrecisionContract(
    name="fp32_reference",
    weight_store="fp32",
    attention_weight_format="fp32",
    mlp_weight_format="fp32",
    activation_dtype="fp64",
    accumulation_dtype="fp64",
    kv_cache_dtype="fp64",
    scale_dtype="none",
    group_size=32,
    lut4_paths=0,
    lut4_target_bpw=0.0,
    deterministic_zero_drift=True,
    bounded_q8_error=False,
)

CONTRACT_BCIRQ8_G32 = PrecisionContract(
    name="bcirq8_v1_q8_g32",
    weight_store="bcirq8_v1",
    attention_weight_format="q8_g32",
    mlp_weight_format="q8_g32",
    activation_dtype="fp64",
    accumulation_dtype="fp64",
    kv_cache_dtype="fp64",
    scale_dtype="pow2_i16",
    group_size=32,
    lut4_paths=0,
    lut4_target_bpw=0.0,
    deterministic_zero_drift=True,
    bounded_q8_error=True,
)

CONTRACT_COMP1_LUT4_1B_RES = PrecisionContract(
    name="bcirq8_v2_comp1_lut4_1b_2p",
    weight_store="bcirq8_v2_hybrid_lut4_mlp",
    attention_weight_format="q8_g32",
    mlp_weight_format="lut4_1b_2p",
    activation_dtype="fp64",
    accumulation_dtype="fp64",
    kv_cache_dtype="fp64",
    scale_dtype="pow2_i16+fp16_le",
    group_size=32,
    lut4_paths=2,
    lut4_target_bpw=1.0,
    deterministic_zero_drift=True,
    bounded_q8_error=True,
)

CONTRACT_COMP1_LUT4_1B_1P = PrecisionContract(
    name="bcirq8_v2_comp1_lut4_1b_1p",
    weight_store="bcirq8_v2_hybrid_lut4_mlp",
    attention_weight_format="q8_g32",
    mlp_weight_format="lut4_1b_1p",
    activation_dtype="fp64",
    accumulation_dtype="fp64",
    kv_cache_dtype="fp64",
    scale_dtype="pow2_i16+fp16_le",
    group_size=32,
    lut4_paths=1,
    lut4_target_bpw=1.0,
    deterministic_zero_drift=True,
    bounded_q8_error=True,
)


@dataclass(frozen=True)
class DeploymentPolicy:
    """P2/P3/P4 BCIR deployment policy configuration."""

    min_lut4_dim: int = 512
    protect_attention: bool = True
    q8_group_size: int = 32
    target_mlp_bpw: float = 1.0
    default_lut4_paths: int = 2
    residual_rank_fraction: float = 0.25
    max_q8_ce_regression: float = 0.02
    max_hybrid_ce_regression: float = 0.20
    max_recall_drop: float = 0.0
    allow_p3_promotion: bool = True
    require_zero_lane_drift: bool = True
    require_q8_bound: bool = True
    itq_iters: int = 25
    svid_iters: int = 20

    def __post_init__(self) -> None:
        if type(self.min_lut4_dim) is not int or self.min_lut4_dim < 1:
            raise ValueError("min_lut4_dim must be a positive integer")
        if type(self.q8_group_size) is not int or not 1 <= self.q8_group_size <= 0xFFFF:
            raise ValueError("q8_group_size must be in [1, 65535]")
        if not math.isfinite(self.target_mlp_bpw) or self.target_mlp_bpw <= 0.0:
            raise ValueError("target_mlp_bpw must be finite and positive")
        if type(self.default_lut4_paths) is not int or self.default_lut4_paths not in (1, 2):
            raise ValueError("default_lut4_paths must be 1 or 2")
        if (
            not math.isfinite(self.residual_rank_fraction)
            or not 0.0 < self.residual_rank_fraction < 1.0
        ):
            raise ValueError("residual_rank_fraction must be in (0, 1)")
        for field_name in ("max_q8_ce_regression", "max_hybrid_ce_regression", "max_recall_drop"):
            val = getattr(self, field_name)
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{field_name} must be finite and >= 0")
        if type(self.itq_iters) is not int or self.itq_iters < 0:
            raise ValueError("itq_iters must be an integer >= 0")
        if type(self.svid_iters) is not int or self.svid_iters < 1:
            raise ValueError("svid_iters must be an integer >= 1")

    def admits_lut4(self, spec: DecoderSpec, tensor_id: int | None = None) -> tuple[bool, str]:
        """Return ``(admitted, reason)`` under P2 width and Attention-protection gates."""
        if not isinstance(spec, DecoderSpec):
            raise ValueError("spec must be a DecoderSpec")
        if tensor_id is not None and tensor_id not in LUT4_ALLOWED_TENSOR_IDS:
            return (
                False,
                f"tensor_id={tensor_id} is protected (LUT4_1B is permitted only on MLP tensors)",
            )
        min_dim = min(spec.d_model, spec.d_ff)
        if min_dim < self.min_lut4_dim:
            return (
                False,
                f"width gate refused LUT4_1B: min(d_model={spec.d_model}, d_ff={spec.d_ff})="
                f"{min_dim} < min_lut4_dim={self.min_lut4_dim}",
            )
        return (
            True,
            f"admitted LUT4_1B MLP: min(d_model={spec.d_model}, d_ff={spec.d_ff})="
            f"{min_dim} >= {self.min_lut4_dim}",
        )

    def solve_rank(
        self,
        in_dim: int,
        out_dim: int,
        *,
        target_bpw: float | None = None,
        paths: int | None = None,
    ) -> tuple[int, ...]:
        """Compute per-path factorized ranks meeting the effective bit-per-weight budget."""
        if type(in_dim) is not int or type(out_dim) is not int or in_dim < 1 or out_dim < 1:
            raise ValueError("matrix dimensions must be positive integers")
        bpw = self.target_mlp_bpw if target_bpw is None else float(target_bpw)
        n_paths = self.default_lut4_paths if paths is None else int(paths)
        if not math.isfinite(bpw) or bpw <= 0.0:
            raise ValueError("target_bpw must be finite and positive")
        if n_paths not in (1, 2):
            raise ValueError("paths must be 1 or 2")
        max_rank = min(in_dim, out_dim, LUT4_MAX_RANK)
        budget_bits = bpw * in_dim * out_dim
        fixed_scale_bits = 16.0 * n_paths * (in_dim + out_dim)
        per_rank_bits = float(in_dim + out_dim + 16)
        raw_total = int(math.floor((budget_bits - fixed_scale_bits) / per_rank_bits))
        total_rank = max(n_paths, min(max_rank, raw_total))
        if n_paths == 1 or max_rank < 2:
            return (max(1, min(max_rank, total_rank)),)
        r2 = max(1, int(round(total_rank * self.residual_rank_fraction)))
        r1 = max(1, total_rank - r2)
        if r1 + r2 > max_rank:
            r2 = max(1, min(r2, max_rank // 2))
            r1 = max(1, max_rank - r2)
        return (r1, r2)

    @staticmethod
    def effective_mlp_bpw(in_dim: int, out_dim: int, ranks: Sequence[int]) -> float:
        """Exact on-disk bits per logical weight for a packed LUT4_1B MLP tensor."""
        scales_bytes = sum(2 * (in_dim + r + out_dim) for r in ranks)
        packed_bytes = sum(
            lut4_packed_bytes(r, in_dim) + lut4_packed_bytes(out_dim, r) for r in ranks
        )
        return 8.0 * (scales_bytes + packed_bytes) / float(in_dim * out_dim)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Q8BoundReport:
    """P5 bounded-error verification record: ``|w - q * 2^e| <= 2^(e - 1)``."""

    passed: bool
    tensors_checked: int
    lut4_tensors_skipped: int
    elements_checked: int
    violation_count: int
    max_abs_error: float
    max_normalized_error: float
    worst_tensor: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reference_tensor_map(
    spec: DecoderSpec, weights: DecoderWeights
) -> dict[tuple[int, int], tuple[float, ...]]:
    ref: dict[tuple[int, int], tuple[float, ...]] = {
        (TENSOR_EMBEDDING, -1): tuple(weights.embedding.table),
        (TENSOR_G_FINAL, -1): tuple(weights.g_final),
    }
    if not spec.tied_embeddings:
        ref[(TENSOR_LM_HEAD, -1)] = tuple(weights.lm_head)
    for layer, lw in enumerate(weights.layers):
        ref[(TENSOR_G_ATTN, layer)] = tuple(lw.g_attn)
        ref[(TENSOR_W_Q, layer)] = tuple(lw.w_q)
        ref[(TENSOR_W_K, layer)] = tuple(lw.w_k)
        ref[(TENSOR_W_V, layer)] = tuple(lw.w_v)
        ref[(TENSOR_W_O, layer)] = tuple(lw.w_o)
        ref[(TENSOR_G_FF, layer)] = tuple(lw.g_ff)
        ref[(TENSOR_W_GATE, layer)] = tuple(lw.w_gate)
        ref[(TENSOR_W_UP, layer)] = tuple(lw.w1)
        ref[(TENSOR_W_DOWN, layer)] = tuple(lw.w2)
    return ref


def verify_q8_error_bound(
    spec: DecoderSpec,
    ref_weights: DecoderWeights,
    artifact_path: os.PathLike | str,
    *,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
) -> Q8BoundReport:
    """Verify exact per-group Q8 bound ``|w - q * 2^e| <= 2^(e - 1)`` on disk.

    Inspects the serialized ``BCIRQ8`` (v1 or v2) directory and payloads
    directly against ``ref_weights``. Every Q8 tensor element must satisfy
    ``|w - ldexp(q, e)| <= ldexp(0.5, e)``. ``LUT4_1B`` MLP tensors in v2
    artifacts are counted separately and skipped by the Q8 bound verifier.
    """
    raw = Path(artifact_path).read_bytes()
    if len(raw) < HEADER_SIZE or len(raw) > max_bytes:
        raise ValueError(f"BCIRQ8 file size {len(raw)} is outside [HEADER_SIZE, {max_bytes}]")
    fields = _HEADER_PREFIX.unpack_from(raw)
    group_size = fields[5]
    tensor_count = fields[20]
    directory_offset = fields[22]
    ref_map = _reference_tensor_map(spec, ref_weights)

    tensors_checked = 0
    lut4_skipped = 0
    elements_checked = 0
    violation_count = 0
    max_abs_err = 0.0
    max_norm_err = 0.0
    worst_tensor = ""
    view = memoryview(raw)

    # Optional NumPy vectorization keeps wide 50M checks under 100ms while preserving stdlib fallback.
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None

    for idx in range(tensor_count):
        off = directory_offset + idx * DIRECTORY_ENTRY_SIZE
        (
            tensor_id,
            layer,
            _rank,
            eflags,
            _reserved,
            _dim0,
            _dim1,
            count,
            groups,
            _crc,
            _reserved2,
            exp_off,
            code_off,
        ) = _DIRECTORY.unpack_from(raw, off)
        key = (tensor_id, layer)
        if eflags == TENSOR_ENC_LUT4_1B:
            lut4_skipped += 1
            continue
        if eflags != TENSOR_ENC_Q8:
            raise ValueError(f"unexpected tensor encoding flag {eflags} at entry {idx}")
        if key not in ref_map or len(ref_map[key]) != count:
            raise ValueError(f"reference weights missing or shape mismatch for tensor {key}")
        ref_vals = ref_map[key]
        tensors_checked += 1
        elements_checked += count
        label = f"id={tensor_id}:layer={layer}"
        if np is not None:
            w = np.asarray(ref_vals, dtype=np.float64)
            exps = np.frombuffer(view[exp_off : exp_off + groups * 2], dtype="<i2").astype(np.int32)
            codes = np.frombuffer(view[code_off : code_off + count], dtype=np.int8).astype(
                np.float64
            )
            e_elem = np.repeat(exps, group_size)[:count]
            scales = np.ldexp(np.ones(count, dtype=np.float64), e_elem)
            deq = codes * scales
            abs_err = np.abs(w - deq)
            half_step = 0.5 * scales
            # Allow tiny FP rounding tolerance on the half-step bound itself (exact for powers of 2).
            bad = abs_err > half_step * (1.0 + 1e-12)
            bad_cnt = int(np.count_nonzero(bad))
            if bad_cnt:
                violation_count += bad_cnt
            t_max_abs = float(np.max(abs_err)) if count else 0.0
            norm = np.where(half_step > 0.0, abs_err / half_step, 0.0)
            t_max_norm = float(np.max(norm)) if count else 0.0
            if t_max_abs > max_abs_err:
                max_abs_err = t_max_abs
            if t_max_norm > max_norm_err:
                max_norm_err = t_max_norm
                worst_tensor = label
        else:
            exps = tuple(
                v for (v,) in struct.iter_unpack("<h", view[exp_off : exp_off + groups * 2])
            )
            codes = view[code_off : code_off + count]
            for i, w_val in enumerate(ref_vals):
                e = exps[i // group_size]
                q = codes[i]
                if q >= 128:
                    q -= 256
                deq = math.ldexp(float(q), e)
                half_step = math.ldexp(0.5, e)
                err = abs(float(w_val) - deq)
                if err > half_step * (1.0 + 1e-12):
                    violation_count += 1
                if err > max_abs_err:
                    max_abs_err = err
                norm = err / half_step if half_step > 0.0 else 0.0
                if norm > max_norm_err:
                    max_norm_err = norm
                    worst_tensor = label

    return Q8BoundReport(
        passed=(violation_count == 0 and tensors_checked > 0),
        tensors_checked=tensors_checked,
        lut4_tensors_skipped=lut4_skipped,
        elements_checked=elements_checked,
        violation_count=violation_count,
        max_abs_error=max_abs_err,
        max_normalized_error=max_norm_err,
        worst_tensor=worst_tensor,
    )


# ---------------------------------------------------------------------------
# Dual-SVID + Joint-ITQ Factorizer (stdlib Jacobi SVD + lazy NumPy fast path)
# ---------------------------------------------------------------------------


def _clamp_fp16_scales(values: Sequence[float]) -> list[float]:
    out = []
    for v in values:
        f = float(v)
        if not math.isfinite(f):
            f = 0.0
        elif f > _FP16_MAX_SAFE:
            f = _FP16_MAX_SAFE
        elif f < -_FP16_MAX_SAFE:
            f = -_FP16_MAX_SAFE
        out.append(f)
    return out


def _stdlib_jacobi_eig(
    sym: list[list[float]], max_sweeps: int = 40
) -> tuple[list[float], list[list[float]]]:
    """Symmetric eigendecomposition ``sym = V diag(w) V^T`` in pure Python."""
    n = len(sym)
    a = [row[:] for row in sym]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        max_off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) > max_off:
                    max_off = abs(apq)
                if abs(apq) <= 1e-14:
                    continue
                app, aqq = a[p][p], a[q][q]
                tau = (aqq - app) / (2.0 * apq)
                t = (
                    (1.0 / (tau + math.hypot(1.0, tau)))
                    if tau >= 0.0
                    else (-1.0 / (-tau + math.hypot(1.0, tau)))
                )
                c = 1.0 / math.hypot(1.0, t)
                s = t * c
                a[p][p] = app - t * apq
                a[q][q] = aqq + t * apq
                a[p][q] = 0.0
                a[q][p] = 0.0
                for k in range(n):
                    if k != p and k != q:
                        akp, akq = a[k][p], a[k][q]
                        a[k][p] = a[p][k] = c * akp - s * akq
                        a[k][q] = a[q][k] = s * akp + c * akq
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
        if max_off <= 1e-13:
            break
    eigvals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda idx: eigvals[idx], reverse=True)
    sorted_vals = [eigvals[i] for i in order]
    sorted_vecs = [[v[row][col] for col in order] for row in range(n)]
    return sorted_vals, sorted_vecs


def _stdlib_svd_top_r(
    mat: list[list[float]], rank: int
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Top-r SVD of ``mat [m, n]`` using symmetric Jacobi on the smaller Gram matrix."""
    m = len(mat)
    n = len(mat[0])
    if m <= n:
        gram = [[sum(mat[i][k] * mat[j][k] for k in range(n)) for j in range(m)] for i in range(m)]
        vals, vecs = _stdlib_jacobi_eig(gram)
        s = [math.sqrt(max(0.0, vals[k])) for k in range(rank)]
        u = [[vecs[i][k] for k in range(rank)] for i in range(m)]
        vt = []
        for k in range(rank):
            sk = s[k]
            if sk > 1e-12:
                vt.append([sum(u[i][k] * mat[i][j] for i in range(m)) / sk for j in range(n)])
            else:
                vt.append([1.0 if j == (k % n) else 0.0 for j in range(n)])
        return u, s, vt
    gram = [[sum(mat[k][i] * mat[k][j] for k in range(m)) for j in range(n)] for i in range(n)]
    vals, vecs = _stdlib_jacobi_eig(gram)
    s = [math.sqrt(max(0.0, vals[k])) for k in range(rank)]
    vt = [[vecs[j][k] for j in range(n)] for k in range(rank)]
    u = [[0.0] * rank for _ in range(m)]
    for k in range(rank):
        sk = s[k]
        if sk > 1e-12:
            for i in range(m):
                u[i][k] = sum(mat[i][j] * vt[k][j] for j in range(n)) / sk
        else:
            u[k % m][k] = 1.0
    return u, s, vt


def _stdlib_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    m, k, n = len(a), len(b), len(b[0])
    return [[sum(a[i][t] * b[t][j] for t in range(k)) for j in range(n)] for i in range(m)]


def _stdlib_transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _pack_signs_fast(signs_2d, rows: int, cols: int) -> bytes:
    """Fast LUT4 sign packer (NumPy vectorized when available, bit-identical to stdlib)."""
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        flat = [int(v) for row in signs_2d for v in row]
        return pack_lut4_signs(flat, rows, cols)

    arr = np.asarray(signs_2d, dtype=np.int8)
    if arr.shape != (rows, cols):
        raise ValueError(f"sign array shape {arr.shape} != {(rows, cols)}")
    rows32 = (rows + 31) & ~31
    bpg = rows32 >> 1
    groups4 = (cols + 3) >> 2
    neg = np.zeros((rows32, groups4 * 4), dtype=np.uint8)
    neg[:rows, :cols] = (arr < 0).astype(np.uint8)
    neg = neg.reshape(rows32, groups4, 4)
    nibs = (neg[:, :, 0] | (neg[:, :, 1] << 1) | (neg[:, :, 2] << 2) | (neg[:, :, 3] << 3)).astype(
        np.uint8
    )  # [rows32, groups4]
    nibs = nibs.reshape(rows32 // 32, 2, 16, groups4)
    packed = nibs[:, 0, :, :] | (nibs[:, 1, :, :] << 4)  # [n_blk32, 16, groups4]
    packed = np.transpose(packed, (2, 0, 1)).reshape(groups4 * bpg)
    return packed.tobytes()


def joint_itq_rotate(
    u_cont,
    v_cont,
    *,
    iters: int = 25,
) -> tuple[Any, Any]:
    """Joint-ITQ orthogonal rotation aligning both ``U [out, r]`` and ``V [r, in]``.

    Finds orthogonal ``R [r, r]`` minimizing
    ``||U R - sign(U R)||_F^2 + ||R^T V - sign(R^T V)||_F^2`` while keeping the
    exact product ``(U R)(R^T V) = U V``.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None

    if np is not None:
        u = np.asarray(u_cont, dtype=np.float64)
        v = np.asarray(v_cont, dtype=np.float64)
        r = u.shape[1]
        rot = np.eye(r, dtype=np.float64)
        for _ in range(iters):
            ur = u @ rot
            rv = rot.T @ v
            bu = np.where(ur >= 0.0, 1.0, -1.0)
            bv = np.where(rv >= 0.0, 1.0, -1.0)
            # Maximize tr(rot^T (u^T bu + v bv^T)) -> SVD(C) = P S Q^T => rot = P Q^T.
            cov = u.T @ bu + v @ bv.T
            p, _s, qt = np.linalg.svd(cov, full_matrices=False)
            rot = p @ qt
        return u @ rot, rot.T @ v

    # Pure-Python fallback
    u_py = [[float(x) for x in row] for row in u_cont]
    v_py = [[float(x) for x in row] for row in v_cont]
    r = len(u_py[0])
    rot_py = [[1.0 if i == j else 0.0 for j in range(r)] for i in range(r)]
    for _ in range(iters):
        ur = _stdlib_matmul(u_py, rot_py)
        rot_t = _stdlib_transpose(rot_py)
        rv = _stdlib_matmul(rot_t, v_py)
        bu = [[1.0 if x >= 0.0 else -1.0 for x in row] for row in ur]
        bv = [[1.0 if x >= 0.0 else -1.0 for x in row] for row in rv]
        c1 = _stdlib_matmul(_stdlib_transpose(u_py), bu)
        c2 = _stdlib_matmul(v_py, _stdlib_transpose(bv))
        cov_py = [[c1[i][j] + c2[i][j] for j in range(r)] for i in range(r)]
        p_mat, _s, qt_mat = _stdlib_svd_top_r(cov_py, r)
        rot_py = _stdlib_matmul(p_mat, qt_mat)
    return _stdlib_matmul(u_py, rot_py), _stdlib_matmul(_stdlib_transpose(rot_py), v_py)


def dual_svid_path(
    mat_out_by_in,
    rank: int,
    *,
    itq_iters: int = 25,
    svid_iters: int = 20,
) -> tuple[Lut4Path, Any]:
    """Factorize ``M [out_dim, in_dim]`` into one binary16-scaled ``Lut4Path``.

    Returns ``(lut4_path, approx_mat_out_by_in)`` where ``approx_mat`` uses the
    exact binary16-rounded scales so multi-path residual factorization fits the
    true on-disk residual.
    """
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None

    if np is not None:
        m = np.asarray(mat_out_by_in, dtype=np.float64)
        out_dim, in_dim = m.shape
        if not 1 <= rank <= min(out_dim, in_dim, LUT4_MAX_RANK):
            raise ValueError(f"invalid LUT4 rank {rank} for shape {(out_dim, in_dim)}")
        # SVD + Joint-ITQ initialization
        u_svd, s_svd, vt_svd = np.linalg.svd(m, full_matrices=False)
        u_r = u_svd[:, :rank]
        s_r = np.maximum(s_svd[:rank], 0.0)
        vt_r = vt_svd[:rank, :]
        sqrt_s = np.sqrt(np.maximum(s_r, 1e-12))
        u_cont = u_r * sqrt_s[None, :]
        v_cont = sqrt_s[:, None] * vt_r
        if itq_iters > 0 and rank > 1:
            u_rot, v_rot = joint_itq_rotate(u_cont, v_cont, iters=itq_iters)
        else:
            u_rot, v_rot = u_cont, v_cont
        u_sign = np.where(u_rot >= 0.0, 1.0, -1.0)
        v_sign = np.where(v_rot >= 0.0, 1.0, -1.0)

        # Dual-SVID alternating scale optimization for diag(h) @ U @ diag(l) @ V @ diag(g)
        h = np.mean(np.abs(u_rot), axis=1)
        g = np.mean(np.abs(v_rot), axis=0)
        h = np.where(h > 1e-8, h, 1e-4)
        g = np.where(g > 1e-8, g, 1e-4)
        lam = np.ones(rank, dtype=np.float64)

        for _ in range(svid_iters):
            # Update lam [r] given h, g
            # Approx_ij = h_i g_j sum_k l_k U_ik V_kj
            # Least-squares diagonal update (fast & numerically stable)
            hu = h[:, None] * u_sign  # [out, r]
            vg = v_sign * g[None, :]  # [r, in]
            # Solve ( (hu^T hu) * (vg vg^T) ) lam = diag(hu^T M vg^T)
            gram = (hu.T @ hu) * (vg @ vg.T)
            rhs = np.sum(hu * (m @ vg.T), axis=0)
            ridge = 1e-8 * (float(np.trace(gram)) / max(rank, 1) + 1e-12)
            lam = np.linalg.solve(gram + ridge * np.eye(rank), rhs)

            # Update h [out] given g, lam
            b_mat = (u_sign * lam[None, :]) @ (v_sign * g[None, :])  # [out, in]
            denom_h = np.sum(b_mat * b_mat, axis=1)
            h = np.where(
                denom_h > 1e-16, np.sum(m * b_mat, axis=1) / np.maximum(denom_h, 1e-16), 0.0
            )

            # Update g [in] given h, lam
            a_mat = (h[:, None] * u_sign * lam[None, :]) @ v_sign  # [out, in]
            denom_g = np.sum(a_mat * a_mat, axis=0)
            g = np.where(
                denom_g > 1e-16, np.sum(m * a_mat, axis=0) / np.maximum(denom_g, 1e-16), 0.0
            )

            # Re-balance scale norms across (h, g, lam) so all three stay well inside FP16 dynamic range
            h_rms = float(np.sqrt(np.mean(h * h) + 1e-18))
            g_rms = float(np.sqrt(np.mean(g * g) + 1e-18))
            if h_rms > _FP16_MIN_NORM:
                h = h / h_rms
                lam = lam * h_rms
            if g_rms > _FP16_MIN_NORM:
                g = g / g_rms
                lam = lam * g_rms

        g_fp16 = encode_fp16_le(_clamp_fp16_scales(g.tolist()))
        l_fp16 = encode_fp16_le(_clamp_fp16_scales(lam.tolist()))
        h_fp16 = encode_fp16_le(_clamp_fp16_scales(h.tolist()))
        g_q = np.asarray(decode_fp16_le(g_fp16), dtype=np.float64)
        l_q = np.asarray(decode_fp16_le(l_fp16), dtype=np.float64)
        h_q = np.asarray(decode_fp16_le(h_fp16), dtype=np.float64)
        approx = (h_q[:, None] * u_sign * l_q[None, :]) @ (v_sign * g_q[None, :])
        v_packed = _pack_signs_fast(v_sign.astype(np.int8), rank, in_dim)
        u_packed = _pack_signs_fast(u_sign.astype(np.int8), out_dim, rank)
        return Lut4Path(rank, v_packed, u_packed, g_fp16, l_fp16, h_fp16), approx

    # Pure-Python stdlib fallback
    m_py = [[float(x) for x in row] for row in mat_out_by_in]
    out_dim = len(m_py)
    in_dim = len(m_py[0])
    if not 1 <= rank <= min(out_dim, in_dim, LUT4_MAX_RANK):
        raise ValueError(f"invalid LUT4 rank {rank} for shape {(out_dim, in_dim)}")
    u_r, s_r, vt_r = _stdlib_svd_top_r(m_py, rank)
    sqrt_s = [math.sqrt(max(sv, 1e-12)) for sv in s_r]
    u_cont = [[u_r[i][k] * sqrt_s[k] for k in range(rank)] for i in range(out_dim)]
    v_cont = [[sqrt_s[k] * vt_r[k][j] for j in range(in_dim)] for k in range(rank)]
    if itq_iters > 0 and rank > 1:
        u_rot, v_rot = joint_itq_rotate(u_cont, v_cont, iters=itq_iters)
    else:
        u_rot, v_rot = u_cont, v_cont
    u_sign = [[1 if x >= 0.0 else -1 for x in row] for row in u_rot]
    v_sign = [[1 if x >= 0.0 else -1 for x in row] for row in v_rot]
    h = [max(sum(abs(x) for x in row) / rank, 1e-4) for row in u_rot]
    g = [max(sum(abs(v_rot[k][j]) for k in range(rank)) / rank, 1e-4) for j in range(in_dim)]
    lam = [1.0] * rank

    for _ in range(svid_iters):
        # Update h
        b_mat = [
            [
                sum(u_sign[i][k] * lam[k] * v_sign[k][j] * g[j] for k in range(rank))
                for j in range(in_dim)
            ]
            for i in range(out_dim)
        ]
        for i in range(out_dim):
            num = sum(m_py[i][j] * b_mat[i][j] for j in range(in_dim))
            den = sum(b_mat[i][j] * b_mat[i][j] for j in range(in_dim))
            h[i] = num / den if den > 1e-16 else 0.0
        # Update g
        a_mat = [
            [
                sum(h[i] * u_sign[i][k] * lam[k] * v_sign[k][j] for k in range(rank))
                for j in range(in_dim)
            ]
            for i in range(out_dim)
        ]
        for j in range(in_dim):
            num = sum(m_py[i][j] * a_mat[i][j] for i in range(out_dim))
            den = sum(a_mat[i][j] * a_mat[i][j] for i in range(out_dim))
            g[j] = num / den if den > 1e-16 else 0.0
        # Update lam coordinate-wise
        for k in range(rank):
            # residual without rank k
            num = 0.0
            den = 0.0
            for i in range(out_dim):
                hi_uik = h[i] * u_sign[i][k]
                for j in range(in_dim):
                    basis = hi_uik * v_sign[k][j] * g[j]
                    other = sum(
                        h[i] * u_sign[i][t] * lam[t] * v_sign[t][j] * g[j]
                        for t in range(rank)
                        if t != k
                    )
                    num += (m_py[i][j] - other) * basis
                    den += basis * basis
            lam[k] = num / den if den > 1e-16 else 0.0
        h_rms = math.sqrt(sum(v * v for v in h) / out_dim + 1e-18)
        g_rms = math.sqrt(sum(v * v for v in g) / in_dim + 1e-18)
        if h_rms > _FP16_MIN_NORM:
            h = [v / h_rms for v in h]
            lam = [v * h_rms for v in lam]
        if g_rms > _FP16_MIN_NORM:
            g = [v / g_rms for v in g]
            lam = [v * g_rms for v in lam]

    g_fp16 = encode_fp16_le(_clamp_fp16_scales(g))
    l_fp16 = encode_fp16_le(_clamp_fp16_scales(lam))
    h_fp16 = encode_fp16_le(_clamp_fp16_scales(h))
    g_q = decode_fp16_le(g_fp16)
    l_q = decode_fp16_le(l_fp16)
    h_q = decode_fp16_le(h_fp16)
    approx_py = [
        [
            sum(h_q[i] * u_sign[i][k] * l_q[k] * v_sign[k][j] * g_q[j] for k in range(rank))
            for j in range(in_dim)
        ]
        for i in range(out_dim)
    ]
    v_packed = pack_lut4_signs([x for row in v_sign for x in row], rank, in_dim)
    u_packed = pack_lut4_signs([x for row in u_sign for x in row], out_dim, rank)
    return Lut4Path(rank, v_packed, u_packed, g_fp16, l_fp16, h_fp16), approx_py


def factorize_lut4_matrix(
    weight_in_by_out: Sequence[float],
    *,
    tensor_id: int,
    layer: int,
    in_dim: int,
    out_dim: int,
    ranks: Sequence[int],
    itq_iters: int = 25,
    svid_iters: int = 20,
) -> tuple[Lut4Tensor, float]:
    """Factorize a row-major ``[in_dim, out_dim]`` MLP weight into a ``Lut4Tensor``.

    Returns ``(lut4_tensor, relative_frobenius_error)``.
    """
    if tensor_id not in LUT4_ALLOWED_TENSOR_IDS:
        raise ValueError(
            f"LUT4_1B factorization refused for protected tensor_id={tensor_id} (MLP only)"
        )
    if len(weight_in_by_out) != in_dim * out_dim:
        raise ValueError(f"weight length {len(weight_in_by_out)} != {in_dim * out_dim}")
    rank_tuple = tuple(int(r) for r in ranks)
    if not 1 <= len(rank_tuple) <= LUT4_MAX_PATHS:
        raise ValueError(f"ranks must contain 1..{LUT4_MAX_PATHS} entries")

    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        np = None

    if np is not None:
        # W is [in_dim, out_dim]; y = x @ W = W^T @ x so factorize M = W^T [out_dim, in_dim]
        m_orig = np.asarray(weight_in_by_out, dtype=np.float64).reshape(in_dim, out_dim).T.copy()
        residual = m_orig.copy()
        paths: list[Lut4Path] = []
        for r in rank_tuple:
            path, approx = dual_svid_path(residual, r, itq_iters=itq_iters, svid_iters=svid_iters)
            paths.append(path)
            residual = residual - approx
        norm_orig = float(np.linalg.norm(m_orig))
        rel_err = float(np.linalg.norm(residual)) / norm_orig if norm_orig > 0.0 else 0.0
        return Lut4Tensor(tensor_id, layer, in_dim, out_dim, tuple(paths)), rel_err

    m_orig_py = [
        [float(weight_in_by_out[i * out_dim + j]) for i in range(in_dim)] for j in range(out_dim)
    ]
    residual_py = [row[:] for row in m_orig_py]
    paths_py: list[Lut4Path] = []
    for r in rank_tuple:
        path, approx = dual_svid_path(residual_py, r, itq_iters=itq_iters, svid_iters=svid_iters)
        paths_py.append(path)
        residual_py = [
            [residual_py[j][i] - approx[j][i] for i in range(in_dim)] for j in range(out_dim)
        ]
    norm_orig = math.sqrt(sum(x * x for row in m_orig_py for x in row))
    norm_res = math.sqrt(sum(x * x for row in residual_py for x in row))
    rel_err = norm_res / norm_orig if norm_orig > 0.0 else 0.0
    return Lut4Tensor(tensor_id, layer, in_dim, out_dim, tuple(paths_py)), rel_err


@dataclass(frozen=True)
class ExportReport:
    """Summary of one policy-driven BCIRQ8 v1/v2 export."""

    version: int
    promoted_to_v1: bool
    width_gate_reason: str
    lut4_tensor_count: int
    promoted_layers: tuple[int, ...]
    mean_mlp_rel_err: float
    effective_mlp_bpw: float
    file_size: int
    artifact_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def export_hybrid_decoder(
    path: os.PathLike | str,
    spec: DecoderSpec,
    weights: DecoderWeights,
    *,
    policy: DeploymentPolicy,
    source_hashes: Mapping[str, str],
    tokenizer_ids: Mapping[str, int],
    context_length: int | None = None,
    promoted_layers: Sequence[int] = (),
    lut4_overrides: Mapping[tuple[int, int], Lut4Tensor] | None = None,
    max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    max_elements: int = DEFAULT_MAX_DECODED_ELEMENTS,
    native_kernels: NativeAIKernels | None = None,
) -> tuple[Q8ArtifactMetadata, dict[tuple[int, int], Lut4Tensor], ExportReport]:
    """Export a decoder under ``policy`` as ``BCIRQ8 v2`` (hybrid) or ``BCIRQ8 v1`` (promoted)."""
    admitted, reason = policy.admits_lut4(spec)
    promoted_set = frozenset(int(lyr) for lyr in promoted_layers)
    if any(lyr < 0 or lyr >= spec.n_layers for lyr in promoted_set):
        raise ValueError("promoted_layers contains an out-of-range layer index")

    if not admitted or len(promoted_set) == spec.n_layers:
        if not admitted and not policy.allow_p3_promotion:
            raise ValueError(
                f"deployment policy refused LUT4_1B and P3 promotion is disabled: {reason}"
            )
        meta = write_q8_decoder(
            path,
            spec,
            weights,
            group_size=policy.q8_group_size,
            source_hashes=source_hashes,
            tokenizer_ids=tokenizer_ids,
            context_length=context_length,
            max_bytes=max_bytes,
            max_elements=max_elements,
            native_kernels=native_kernels,
            lut4_tensors=None,
        )
        rep = ExportReport(
            version=meta.version,
            promoted_to_v1=True,
            width_gate_reason=reason,
            lut4_tensor_count=0,
            promoted_layers=tuple(range(spec.n_layers)),
            mean_mlp_rel_err=0.0,
            effective_mlp_bpw=8.0 + 16.0 / policy.q8_group_size,
            file_size=meta.file_size,
            artifact_sha256=meta.artifact_sha256,
        )
        return meta, {}, rep

    overrides = dict(lut4_overrides or {})
    for key, lt in overrides.items():
        if key[0] not in LUT4_ALLOWED_TENSOR_IDS or lt.tensor_id not in LUT4_ALLOWED_TENSOR_IDS:
            raise ValueError(f"lut4_overrides refused protected tensor {key}")

    lut4_map: dict[tuple[int, int], Lut4Tensor] = {}
    rel_errs: list[float] = []
    total_lut4_bits = 0.0
    total_lut4_elems = 0
    d, f = spec.d_model, spec.d_ff

    for layer, lw in enumerate(weights.layers):
        if layer in promoted_set:
            continue
        mlp_specs = (
            (TENSOR_W_GATE, d, f, lw.w_gate),
            (TENSOR_W_UP, d, f, lw.w1),
            (TENSOR_W_DOWN, f, d, lw.w2),
        )
        for tid, in_dim, out_dim, data in mlp_specs:
            key = (tid, layer)
            if key in overrides:
                lt = overrides[key]
                lut4_map[key] = lt
            else:
                ranks = policy.solve_rank(in_dim, out_dim)
                lt, rel_err = factorize_lut4_matrix(
                    data,
                    tensor_id=tid,
                    layer=layer,
                    in_dim=in_dim,
                    out_dim=out_dim,
                    ranks=ranks,
                    itq_iters=policy.itq_iters,
                    svid_iters=policy.svid_iters,
                )
                lut4_map[key] = lt
                rel_errs.append(rel_err)
            total_lut4_bits += 8.0 * lt.total_bytes
            total_lut4_elems += in_dim * out_dim

    meta = write_q8_decoder(
        path,
        spec,
        weights,
        group_size=policy.q8_group_size,
        source_hashes=source_hashes,
        tokenizer_ids=tokenizer_ids,
        context_length=context_length,
        max_bytes=max_bytes,
        max_elements=max_elements,
        native_kernels=native_kernels,
        lut4_tensors=lut4_map,
    )
    rep = ExportReport(
        version=meta.version,
        promoted_to_v1=False,
        width_gate_reason=reason,
        lut4_tensor_count=len(lut4_map),
        promoted_layers=tuple(sorted(promoted_set)),
        mean_mlp_rel_err=(sum(rel_errs) / len(rel_errs)) if rel_errs else 0.0,
        effective_mlp_bpw=(total_lut4_bits / total_lut4_elems) if total_lut4_elems else 0.0,
        file_size=meta.file_size,
        artifact_sha256=meta.artifact_sha256,
    )
    return meta, lut4_map, rep


# ---------------------------------------------------------------------------
# P3 & P4 Qualification Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMetrics:
    """Measured accuracy/throughput witness for one precision regime."""

    ce: float
    ppl: float | None = None
    token_acc: float | None = None
    exact_recall: float | None = None
    decode_tok_s: float | None = None
    lane_max_bit_drift: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.ce) or self.ce < 0.0:
            raise ValueError(f"EvalMetrics.ce must be finite and >= 0, got {self.ce!r}")
        if self.ppl is not None and (not math.isfinite(self.ppl) or self.ppl < 1.0):
            raise ValueError(f"EvalMetrics.ppl must be finite and >= 1, got {self.ppl!r}")
        if self.token_acc is not None and (
            not math.isfinite(self.token_acc) or not 0.0 <= self.token_acc <= 1.0
        ):
            raise ValueError(f"EvalMetrics.token_acc must be in [0, 1], got {self.token_acc!r}")
        if self.exact_recall is not None and (
            not math.isfinite(self.exact_recall) or not 0.0 <= self.exact_recall <= 1.0
        ):
            raise ValueError(
                f"EvalMetrics.exact_recall must be in [0, 1], got {self.exact_recall!r}"
            )
        if self.decode_tok_s is not None and (
            not math.isfinite(self.decode_tok_s) or self.decode_tok_s <= 0.0
        ):
            raise ValueError("EvalMetrics.decode_tok_s must be finite and > 0")
        if type(self.lane_max_bit_drift) is not int or self.lane_max_bit_drift < 0:
            raise ValueError("EvalMetrics.lane_max_bit_drift must be an integer >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationVerdict:
    """Deterministic P1-P5 qualification verdict."""

    status: str
    contract: PrecisionContract
    reasons: tuple[str, ...]
    promoted_layers: tuple[int, ...]
    q8_bound_passed: bool
    lane_zero_drift: bool
    q8_ce_delta: float
    hybrid_ce_delta: float | None
    recall_delta: float | None

    def __post_init__(self) -> None:
        if self.status not in _VALID_VERDICT_STATUSES:
            raise ValueError(f"invalid QualificationVerdict.status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "contract": self.contract.to_dict(),
            "reasons": list(self.reasons),
            "promoted_layers": list(self.promoted_layers),
            "q8_bound_passed": self.q8_bound_passed,
            "lane_zero_drift": self.lane_zero_drift,
            "q8_ce_delta": self.q8_ce_delta,
            "hybrid_ce_delta": self.hybrid_ce_delta,
            "recall_delta": self.recall_delta,
        }


def qualify_deployment(
    spec: DecoderSpec,
    policy: DeploymentPolicy,
    *,
    fp32_metrics: EvalMetrics,
    q8_metrics: EvalMetrics,
    hybrid_metrics: EvalMetrics | None = None,
    q8_bound: Q8BoundReport | None = None,
    per_layer_hybrid_ce_delta: Mapping[int, float] | None = None,
    lane_max_bit_drift: int = 0,
) -> QualificationVerdict:
    """Run the two-stage P1-P5 deployment qualification decision procedure.

    1. Hard fail-closed gates: Q8 bounded-error check and 0-bit-drift SIMD lane check.
    2. P4 Stage 1: Uniform Q8 / Q8-Attention accuracy and recall qualification.
    3. P2 Width Gate: Refuse ``LUT4_1B`` below ``policy.min_lut4_dim`` (promote to Q8 v1).
    4. P4 Stage 2: Hybrid ``COMP-1`` MLP qualification; on regression, apply P3 selective
       layer promotion or whole-model promotion to ``BCIRQ8 v1``, or fail closed if
       ``allow_p3_promotion`` is False.
    """
    if not isinstance(spec, DecoderSpec) or not isinstance(policy, DeploymentPolicy):
        raise ValueError("qualify_deployment requires typed DecoderSpec and DeploymentPolicy")
    reasons: list[str] = []
    q8_bound_ok = True if q8_bound is None else bool(q8_bound.passed)
    worst_drift = max(
        lane_max_bit_drift,
        q8_metrics.lane_max_bit_drift,
        hybrid_metrics.lane_max_bit_drift if hybrid_metrics is not None else 0,
    )
    zero_drift_ok = worst_drift == 0
    q8_ce_delta = q8_metrics.ce - fp32_metrics.ce
    hybrid_ce_delta = None if hybrid_metrics is None else (hybrid_metrics.ce - fp32_metrics.ce)

    q8_recall_delta: float | None = None
    if fp32_metrics.exact_recall is not None and q8_metrics.exact_recall is not None:
        q8_recall_delta = fp32_metrics.exact_recall - q8_metrics.exact_recall
    hybrid_recall_delta: float | None = None
    if (
        fp32_metrics.exact_recall is not None
        and hybrid_metrics is not None
        and hybrid_metrics.exact_recall is not None
    ):
        hybrid_recall_delta = fp32_metrics.exact_recall - hybrid_metrics.exact_recall

    # Hard fail-closed checks (cannot be bypassed by P3 promotion).
    if policy.require_q8_bound and not q8_bound_ok:
        reasons.append(
            f"P5 Q8 error bound failed: violations={q8_bound.violation_count if q8_bound else 'unknown'}"
        )
    if policy.require_zero_lane_drift and not zero_drift_ok:
        reasons.append(f"P5 SIMD lane determinism failed: max_bit_drift={worst_drift}")
    if q8_ce_delta > policy.max_q8_ce_regression:
        reasons.append(
            f"P4 Stage-1 Q8 CE regression {q8_ce_delta:+.4f} > limit {policy.max_q8_ce_regression:.4f}"
        )
    if q8_recall_delta is not None and q8_recall_delta > policy.max_recall_drop:
        reasons.append(
            f"P4 Stage-1 Q8 exact recall dropped by {q8_recall_delta:.4f} > {policy.max_recall_drop:.4f}"
        )
    if reasons:
        return QualificationVerdict(
            status="REFUSED",
            contract=CONTRACT_BCIRQ8_G32,
            reasons=tuple(reasons),
            promoted_layers=(),
            q8_bound_passed=q8_bound_ok,
            lane_zero_drift=zero_drift_ok,
            q8_ce_delta=q8_ce_delta,
            hybrid_ce_delta=hybrid_ce_delta,
            recall_delta=q8_recall_delta,
        )

    hybrid_contract = (
        CONTRACT_COMP1_LUT4_1B_RES if policy.default_lut4_paths == 2 else CONTRACT_COMP1_LUT4_1B_1P
    )

    # P2 Width gate
    admitted, width_reason = policy.admits_lut4(spec)
    if not admitted:
        if policy.allow_p3_promotion:
            return QualificationVerdict(
                status="PROMOTED_Q8_V1",
                contract=CONTRACT_BCIRQ8_G32,
                reasons=(width_reason, "P3 promoted narrow model to BCIRQ8 v1 (Q8 g=32)"),
                promoted_layers=tuple(range(spec.n_layers)),
                q8_bound_passed=q8_bound_ok,
                lane_zero_drift=zero_drift_ok,
                q8_ce_delta=q8_ce_delta,
                hybrid_ce_delta=hybrid_ce_delta,
                recall_delta=q8_recall_delta,
            )
        return QualificationVerdict(
            status="REFUSED",
            contract=hybrid_contract,
            reasons=(width_reason, "P3 promotion disabled"),
            promoted_layers=(),
            q8_bound_passed=q8_bound_ok,
            lane_zero_drift=zero_drift_ok,
            q8_ce_delta=q8_ce_delta,
            hybrid_ce_delta=hybrid_ce_delta,
            recall_delta=q8_recall_delta,
        )

    # P4 Stage 2: Hybrid COMP-1 qualification
    if hybrid_metrics is None:
        if policy.allow_p3_promotion:
            return QualificationVerdict(
                status="PROMOTED_Q8_V1",
                contract=CONTRACT_BCIRQ8_G32,
                reasons=("no hybrid COMP-1 metrics supplied; P3 promoted to BCIRQ8 v1",),
                promoted_layers=tuple(range(spec.n_layers)),
                q8_bound_passed=q8_bound_ok,
                lane_zero_drift=zero_drift_ok,
                q8_ce_delta=q8_ce_delta,
                hybrid_ce_delta=None,
                recall_delta=q8_recall_delta,
            )
        return QualificationVerdict(
            status="REFUSED",
            contract=hybrid_contract,
            reasons=("no hybrid COMP-1 metrics supplied and P3 promotion disabled",),
            promoted_layers=(),
            q8_bound_passed=q8_bound_ok,
            lane_zero_drift=zero_drift_ok,
            q8_ce_delta=q8_ce_delta,
            hybrid_ce_delta=None,
            recall_delta=q8_recall_delta,
        )

    assert hybrid_ce_delta is not None
    hybrid_fail: list[str] = []
    if hybrid_ce_delta > policy.max_hybrid_ce_regression:
        hybrid_fail.append(
            f"P4 Stage-2 hybrid CE regression {hybrid_ce_delta:+.4f} > limit "
            f"{policy.max_hybrid_ce_regression:.4f}"
        )
    if hybrid_recall_delta is not None and hybrid_recall_delta > policy.max_recall_drop:
        hybrid_fail.append(
            f"P4 Stage-2 hybrid exact recall dropped by {hybrid_recall_delta:.4f} > "
            f"{policy.max_recall_drop:.4f}"
        )

    if not hybrid_fail:
        return QualificationVerdict(
            status="ADMITTED_HYBRID_V2",
            contract=hybrid_contract,
            reasons=(
                width_reason,
                f"P4 Stage-1 Q8 CE delta {q8_ce_delta:+.4f} <= {policy.max_q8_ce_regression:.4f}",
                f"P4 Stage-2 hybrid CE delta {hybrid_ce_delta:+.4f} <= {policy.max_hybrid_ce_regression:.4f}",
            ),
            promoted_layers=(),
            q8_bound_passed=q8_bound_ok,
            lane_zero_drift=zero_drift_ok,
            q8_ce_delta=q8_ce_delta,
            hybrid_ce_delta=hybrid_ce_delta,
            recall_delta=hybrid_recall_delta
            if hybrid_recall_delta is not None
            else q8_recall_delta,
        )

    # P3 Selective promotion or full promotion to BCIRQ8 v1
    if not policy.allow_p3_promotion:
        return QualificationVerdict(
            status="REFUSED",
            contract=hybrid_contract,
            reasons=tuple(hybrid_fail + ["P3 promotion disabled"]),
            promoted_layers=(),
            q8_bound_passed=q8_bound_ok,
            lane_zero_drift=zero_drift_ok,
            q8_ce_delta=q8_ce_delta,
            hybrid_ce_delta=hybrid_ce_delta,
            recall_delta=hybrid_recall_delta,
        )

    # Try selective layer promotion when per-layer CE attributions are provided and recall passed.
    if per_layer_hybrid_ce_delta and (
        hybrid_recall_delta is None or hybrid_recall_delta <= policy.max_recall_drop
    ):
        ranked = sorted(
            (
                (int(lyr), float(delta))
                for lyr, delta in per_layer_hybrid_ce_delta.items()
                if 0 <= int(lyr) < spec.n_layers
                and math.isfinite(float(delta))
                and float(delta) > 0.0
            ),
            key=lambda item: (-item[1], item[0]),
        )
        promoted: list[int] = []
        remaining = hybrid_ce_delta
        for lyr, delta in ranked:
            if len(promoted) + 1 >= spec.n_layers:
                break
            promoted.append(lyr)
            remaining -= delta
            if remaining <= policy.max_hybrid_ce_regression:
                promoted_tuple = tuple(sorted(promoted))
                return QualificationVerdict(
                    status="PROMOTED_SELECTIVE_V2",
                    contract=hybrid_contract,
                    reasons=tuple(
                        hybrid_fail
                        + [
                            f"P3 selectively promoted layers {promoted_tuple} to Q8 "
                            f"(estimated remaining CE delta {remaining:+.4f})"
                        ]
                    ),
                    promoted_layers=promoted_tuple,
                    q8_bound_passed=q8_bound_ok,
                    lane_zero_drift=zero_drift_ok,
                    q8_ce_delta=q8_ce_delta,
                    hybrid_ce_delta=hybrid_ce_delta,
                    recall_delta=hybrid_recall_delta,
                )

    return QualificationVerdict(
        status="PROMOTED_Q8_V1",
        contract=CONTRACT_BCIRQ8_G32,
        reasons=tuple(hybrid_fail + ["P3 promoted full model to BCIRQ8 v1 (Q8 g=32)"]),
        promoted_layers=tuple(range(spec.n_layers)),
        q8_bound_passed=q8_bound_ok,
        lane_zero_drift=zero_drift_ok,
        q8_ce_delta=q8_ce_delta,
        hybrid_ce_delta=hybrid_ce_delta,
        recall_delta=q8_recall_delta,
    )


# ---------------------------------------------------------------------------
# P6 Versioned Baseline Record (bcir.deploy_baseline.v1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeployBaselineRecord:
    """Typed P6 versioned deployment baseline checkpoint (``bcir.deploy_baseline.v1``)."""

    created_utc: str
    bcir_git_sha: str
    host: Mapping[str, Any]
    policy: DeploymentPolicy
    contracts: Mapping[str, PrecisionContract]
    models: tuple[Mapping[str, Any], ...]
    kernel_benchmarks: tuple[Mapping[str, Any], ...] = ()
    summary: Mapping[str, Any] = ()  # type: ignore[assignment]
    schema: str = DEPLOY_BASELINE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.schema,
            "created_utc": self.created_utc,
            "bcir_git_sha": self.bcir_git_sha,
            "host": dict(self.host),
            "policy": self.policy.to_dict(),
            "contracts": {k: v.to_dict() for k, v in self.contracts.items()},
            "models": [dict(m) for m in self.models],
            "kernel_benchmarks": [dict(b) for b in self.kernel_benchmarks],
            "summary": dict(self.summary) if isinstance(self.summary, Mapping) else {},
        }
        validate_baseline_record(payload)
        return payload


def _check_json_finite(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float at {path}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string JSON key at {path}")
            _check_json_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_json_finite(v, f"{path}[{i}]")


def validate_baseline_record(record: Mapping[str, Any]) -> None:
    """Strictly validate a ``bcir.deploy_baseline.v1`` P6 baseline checkpoint."""
    if not isinstance(record, Mapping):
        raise ValueError("baseline record must be a JSON object")
    if record.get("schema") != DEPLOY_BASELINE_SCHEMA:
        raise ValueError(
            f"baseline record schema {record.get('schema')!r} != {DEPLOY_BASELINE_SCHEMA!r}"
        )
    required = ("schema", "created_utc", "bcir_git_sha", "host", "policy", "contracts", "models")
    for field in required:
        if field not in record:
            raise ValueError(f"baseline record missing required key {field!r}")
    _check_json_finite(record)
    if not isinstance(record["created_utc"], str) or not record["created_utc"].strip():
        raise ValueError("baseline record created_utc must be a non-empty string")
    if not isinstance(record["bcir_git_sha"], str) or not record["bcir_git_sha"].strip():
        raise ValueError("baseline record bcir_git_sha must be a non-empty string")
    if not isinstance(record["host"], Mapping):
        raise ValueError("baseline record host must be an object")
    if not isinstance(record["policy"], Mapping):
        raise ValueError("baseline record policy must be an object")
    DeploymentPolicy(**dict(record["policy"]))
    contracts = record["contracts"]
    if not isinstance(contracts, Mapping) or not contracts:
        raise ValueError("baseline record contracts must be a non-empty mapping")
    for name, cdata in contracts.items():
        if not isinstance(cdata, Mapping):
            raise ValueError(f"baseline contract {name!r} must be an object")
        PrecisionContract.from_dict(cdata)
    models = record["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("baseline record models must be a non-empty list")
    seen_names: set[str] = set()
    for idx, entry in enumerate(models):
        if not isinstance(entry, Mapping):
            raise ValueError(f"baseline models[{idx}] must be an object")
        mname = entry.get("name")
        if not isinstance(mname, str) or not mname.strip():
            raise ValueError(f"baseline models[{idx}] missing name")
        if mname in seen_names:
            raise ValueError(f"duplicate model entry {mname!r} in baseline record")
        seen_names.add(mname)
        verdict = entry.get("verdict")
        if not isinstance(verdict, Mapping):
            raise ValueError(f"baseline models[{idx}] missing verdict object")
        status = verdict.get("status")
        if status not in _VALID_VERDICT_STATUSES:
            raise ValueError(f"baseline models[{idx}] has invalid verdict status {status!r}")
        if status != "REFUSED":
            if verdict.get("lane_zero_drift") is not True:
                raise ValueError(
                    f"admitted/promoted model {mname!r} must record lane_zero_drift=True"
                )
            if verdict.get("q8_bound_passed") is not True:
                raise ValueError(
                    f"admitted/promoted model {mname!r} must record q8_bound_passed=True"
                )


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate JSON key {k!r} in baseline record")
        out[k] = v
    return out


def _reject_nonfinite_constant(token: str) -> Any:
    raise ValueError(f"non-finite JSON constant {token!r} forbidden in baseline record")


def load_baseline_record(path: os.PathLike | str) -> dict[str, Any]:
    """Load and strictly validate a P6 ``bcir.deploy_baseline.v1`` JSON record."""
    text = Path(path).read_text(encoding="utf-8")
    parsed = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_nonfinite_constant,
    )
    validate_baseline_record(parsed)
    return parsed


def save_baseline_record(path: os.PathLike | str, record: Mapping[str, Any]) -> None:
    """Validate and atomically write a P6 ``bcir.deploy_baseline.v1`` JSON record."""
    validate_baseline_record(record)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "CONTRACT_BCIRQ8_G32",
    "CONTRACT_COMP1_LUT4_1B_1P",
    "CONTRACT_COMP1_LUT4_1B_RES",
    "CONTRACT_FP32_REF",
    "DEPLOY_BASELINE_SCHEMA",
    "DeployBaselineRecord",
    "DeploymentPolicy",
    "EvalMetrics",
    "ExportReport",
    "PrecisionContract",
    "Q8BoundReport",
    "QualificationVerdict",
    "dual_svid_path",
    "export_hybrid_decoder",
    "factorize_lut4_matrix",
    "joint_itq_rotate",
    "load_baseline_record",
    "qualify_deployment",
    "save_baseline_record",
    "validate_baseline_record",
    "verify_q8_error_bound",
]
