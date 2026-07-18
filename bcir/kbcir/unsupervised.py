"""E6 (ML-breadth) -- UNSUPERVISED LEARNING + THE DATA PIPELINE (K-means / scalers / cross-validation /
autoencoder / embedding lookup), the SIXTH slice of the ML-breadth ladder after E1 (OLS), E2 (PCA), E3 (the
Transformer block), E4 (RNN/LSTM/GRU), E5 (classical-ML predict).

THE HONEST FRAMING -- the TRAIN vs PREDICT / FIT vs TRANSFORM split (the same research finding E5 surfaced and
E7's capstone cites). Unsupervised learning + the preprocessing pipeline split into two halves with OPPOSITE
structure, exactly as classical ML did:

  * FIT / TRAIN -- the bounded iterative or statistical pass over the DATA. K-means FIT is Lloyd's iteration (a
    bounded optimization loop: assign -> recompute centroids, repeated); a scaler FIT is a per-feature mean/std
    or min/max estimate; an autoencoder's WEIGHTS come from a training loop (M3, off this module). These have a
    convergence loop / a reduction over the whole training set -- like training, they are bounded-iterative or
    statistical, NOT a single fixed-shape dataflow. K-means FIT lives here because it is BOUNDED + DETERMINISTIC
    (fixed init_indices + fixed n_iter), so it has no data-dependent control flow -- but the note for E7 is that
    its ITERATIVE nature is the seam between this slice's fit half and the fixed-shape predict half.

  * PREDICT / TRANSFORM -- the OPPOSITE: a FIXED, BAKED set of constants makes the inference step a deterministic,
    fixed-shape kernel (the G5 baked-weights inference pattern + the Area-B "integrate, don't reinvent"
    discipline). K-means ASSIGN over baked centroids is the nearest-centroid kernel (exact -- it REUSES
    ``classical.squared_distance``, no transcendental); a scaler TRANSFORM over a baked ``(mean, std)`` /
    ``(min, max)`` is exact division (the ``sqrt`` at FIT time bakes the std, so transform is
    transcendental-free); the autoencoder FORWARD over baked weights is a matmul + activation composition; the
    embedding lookup over a baked table is an exact row gather.

WHAT E6 REUSES (do NOT reinvent distances / matmul / activations / losses):
  * K-means nearest-centroid assignment REUSES ``classical.squared_distance`` (the E5 exact squared-Euclidean) --
    ranking on squared distance is identical to ranking on distance (``sqrt`` is monotone), so assignment needs
    NO transcendental (the predict/assignment kernel).
  * the autoencoder REUSES ``matmul.matmul_reference`` (encode/decode projections), the G1 ``activation``
    references (the hidden activation), and the M1 ``losses.mse_value`` (the reconstruction error).
  * the cross-validation folds REUSE ``training._lcg_permutation`` (the M3 deterministic LCG Fisher-Yates
    shuffle) -- NO new RNG.
The only transcendental this module introduces is ``StandardScaler``'s ``sqrt`` (population std) AT FIT TIME --
it bakes into the std so the transform kernel is exact.

EACH ORGAN: a plain-float reference (the source of truth), an INDEPENDENT verifier (a defining property
recomputed by a fully independent path), and (for the kernels that consume a feature vector) a ``*_via_bridge``
Q8 round-trip of the input = the R17 input bound.

THE QUARANTINE BOUNDARY (mirrors ols.py / pca.py / transformer.py / recurrent.py / classical.py). This module
is OFF the legality path: it imports NO verifier, emits NO ``Diagnostic``, returns plain floats / plain ints /
plain lists (no graded/confidence wrapper). K-means assign / scaler transform / embedding lookup are EXACT
(squared-distance ranking / exact division / row gather -- no transcendental); the StandardScaler ``sqrt`` and
a transcendental autoencoder activation ride the trusted ``math``/libm edge AT FIT/FORWARD time only.
Deterministic, dependency-free (stdlib ``math`` + the REUSED BCIR oracle modules). A research/oracle organ.

LAYOUT (row-major / flat throughout). A data matrix ``X`` is ``n_samples x n_feat`` row-major
(``X[i*n_feat + j]`` is sample i feature j); a single point ``x`` is length ``n_feat``. Centroids are
``k x n_feat`` row-major; an autoencoder's ``W_enc`` is ``n_in x n_hidden`` row-major and ``W_dec`` is
``n_hidden x n_in`` row-major; an embedding table is ``n_vocab x dim`` row-major. All references return FRESH
values and do NOT mutate their inputs (side-effect-free oracle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .activation import activation_reference
from .classical import _squared_distance_at
from .losses import mse_value
from .matmul import matmul_reference
from .quantize import dequantize, quantize_per_group
from .training import _lcg_permutation


# ============================================================================================================
# 1. K-MEANS -- ASSIGN is the exact nearest-centroid kernel (REUSES classical.squared_distance); FIT is the
#    bounded, deterministic Lloyd's iteration; INERTIA is the objective + the monotone-non-increasing verifier.
# ============================================================================================================

def _kmeans_assign_flat(centroids, point, point_offset: int, k: int, n_feat: int) -> int:
    """Nearest centroid over validated flat storage, with no temporary row slices."""
    best_c = 0
    best_d = _squared_distance_at(point, point_offset, centroids, 0, n_feat)
    for centroid in range(1, k):
        distance = _squared_distance_at(
            point, point_offset, centroids, centroid * n_feat, n_feat)
        if distance < best_d:
            best_d = distance
            best_c = centroid
    return best_c


def _finite(values, field: str) -> None:
    try:
        finite = all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} entries must all be finite numbers") from exc
    if not finite:
        raise ValueError(f"{field} entries must all be finite")


def kmeans_assign(centroids: list[float], x: list[float], k: int, n_feat: int) -> int:
    """The nearest-centroid index for a point ``x`` -- the K-means PREDICT/ASSIGN kernel. REUSES the E5
    ``classical.squared_distance`` (exact squared Euclidean) to score every centroid; ranking on squared
    distance is IDENTICAL to ranking on distance (``sqrt`` is monotone), so assignment needs NO transcendental
    -- pure subtract/multiply/add + comparisons. DETERMINISTIC tie-break: the LOWEST centroid index wins on a
    distance tie. ``centroids`` is ``k x n_feat`` row-major (``centroids[c*n_feat + j]``); ``x`` is length
    ``n_feat``. Returns the assigned cluster id in ``[0, k)``. This is the fixed-shape kernel the C twin
    (``emit_kmeans_assign_c``) reproduces integer-exactly (it returns the same argmin). Side-effect-free."""
    if type(k) is not int or k < 1:
        raise ValueError(f"kmeans k must be >= 1; got {k}")
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"kmeans n_feat must be >= 1; got {n_feat}")
    if len(x) != n_feat:
        raise ValueError(f"kmeans point must be length n_feat = {n_feat}; got {len(x)}")
    if len(centroids) != k * n_feat:
        raise ValueError(f"kmeans centroids must be k*n_feat = {k * n_feat}; got {len(centroids)}")
    xf = [float(v) for v in x]
    _finite(xf, "kmeans point")
    _finite(centroids, "kmeans centroids")
    return _kmeans_assign_flat(centroids, xf, 0, k, n_feat)


def kmeans_fit(X: list[float], n_samples: int, n_feat: int, k: int,
               init_indices: list[int], n_iter: int) -> tuple[list[float], list[int]]:
    """Bounded, DETERMINISTIC Lloyd's iteration -- the K-means FIT (the bounded-iterative half, like training).
    Fixed ``init_indices`` choose the initial centroids from ``X`` (row ``init_indices[c]`` is centroid c's
    seed), a fixed ``n_iter`` bounds the loop, so the whole fit is fully reproducible (no randomness). Each
    iteration: ASSIGN every point to its nearest centroid (via :func:`kmeans_assign`, REUSING the exact squared
    distance) -> RECOMPUTE each centroid as the mean of its assigned points. An EMPTY cluster (no point
    assigned) is handled DETERMINISTICALLY: keep its PREVIOUS centroid (no division by zero, no random
    re-seed). Returns ``(centroids, labels)`` -- ``centroids`` is ``k x n_feat`` row-major (the final
    centroids), ``labels`` is length ``n_samples`` (each point's final cluster id). ``X`` is ``n_samples x
    n_feat`` row-major. Side-effect-free (X / init_indices are not mutated)."""
    if type(k) is not int or k < 1:
        raise ValueError(f"kmeans k must be >= 1; got {k}")
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"kmeans n_feat must be >= 1; got {n_feat}")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError(f"kmeans n_samples must be >= 1; got {n_samples}")
    if len(X) != n_samples * n_feat:
        raise ValueError(f"kmeans X must be n_samples*n_feat = {n_samples * n_feat}; got {len(X)}")
    if k > n_samples:
        raise ValueError(f"kmeans k must be <= n_samples = {n_samples}; got {k}")
    if len(init_indices) != k:
        raise ValueError(f"kmeans init_indices must have k = {k} entries; got {len(init_indices)}")
    if type(n_iter) is not int or n_iter < 0:
        raise ValueError(f"kmeans n_iter must be >= 0; got {n_iter}")
    for c, idx in enumerate(init_indices):
        if type(idx) is not int or not (0 <= idx < n_samples):
            raise ValueError(f"kmeans init_indices[{c}] = {idx} out of range [0, {n_samples})")
    Xf = [float(v) for v in X]
    _finite(Xf, "kmeans X")
    # the initial centroids: the chosen seed rows of X (a fresh copy, so X is never mutated).
    centroids = [Xf[init_indices[c] * n_feat + j] for c in range(k) for j in range(n_feat)]
    labels = [0] * n_samples
    for _ in range(n_iter):
        # ASSIGN every point to its nearest centroid (the exact squared-distance kernel).
        for i in range(n_samples):
            labels[i] = _kmeans_assign_flat(centroids, Xf, i * n_feat, k, n_feat)
        # RECOMPUTE each centroid as the mean of its assigned points; an empty cluster keeps its old centroid.
        sums = [0.0] * (k * n_feat)
        counts = [0] * k
        for i in range(n_samples):
            c = labels[i]
            counts[c] += 1
            base = i * n_feat
            cbase = c * n_feat
            for j in range(n_feat):
                sums[cbase + j] += Xf[base + j]
        new_centroids = list(centroids)
        for c in range(k):
            if counts[c] == 0:
                continue                                        # EMPTY cluster -> keep the previous centroid
            cbase = c * n_feat
            for j in range(n_feat):
                new_centroids[cbase + j] = sums[cbase + j] / counts[c]
        converged = new_centroids == centroids
        centroids = new_centroids
        if converged:
            break
    # a final assign so the returned labels match the returned centroids (also covers n_iter == 0).
    for i in range(n_samples):
        labels[i] = _kmeans_assign_flat(centroids, Xf, i * n_feat, k, n_feat)
    return centroids, labels


def kmeans_inertia(X: list[float], n_samples: int, n_feat: int,
                   centroids: list[float], labels: list[int]) -> float:
    """The K-means OBJECTIVE (within-cluster sum of squares):
    ``inertia = Sum_i squared_distance(X_i, centroid[label_i])`` -- the quantity Lloyd's iteration minimizes.
    REUSES ``classical.squared_distance`` (exact). ``X`` is ``n_samples x n_feat`` row-major; ``centroids`` is
    ``k x n_feat`` row-major; ``labels`` is length ``n_samples``. This is also the INDEPENDENT-verifier hook:
    inertia is MONOTONE NON-INCREASING across Lloyd iterations (a defining property -- both the assign step and
    the centroid-recompute step can only lower or hold the within-cluster sum of squares), so a test runs
    :func:`kmeans_fit` for increasing ``n_iter`` and asserts the inertia never rises. Side-effect-free."""
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"kmeans n_feat must be >= 1; got {n_feat}")
    if type(n_samples) is not int or n_samples < 0:
        raise ValueError(f"kmeans n_samples must be >= 0; got {n_samples}")
    if len(X) != n_samples * n_feat:
        raise ValueError(f"kmeans X must be n_samples*n_feat = {n_samples * n_feat}; got {len(X)}")
    if len(labels) != n_samples:
        raise ValueError(f"kmeans labels must have n_samples = {n_samples} entries; got {len(labels)}")
    Xf = [float(v) for v in X]
    _finite(Xf, "kmeans X")
    _finite(centroids, "kmeans centroids")
    acc = 0.0
    for i in range(n_samples):
        if type(labels[i]) is not int:
            raise ValueError(f"kmeans label {labels[i]!r} is not an integer cluster id")
        c = labels[i]
        if not (0 <= c * n_feat < len(centroids)) or (c + 1) * n_feat > len(centroids):
            raise ValueError(f"kmeans label {c} out of range for the centroid table")
        acc += _squared_distance_at(Xf, i * n_feat, centroids, c * n_feat, n_feat)
    return acc


def kmeans_assign_via_bridge(centroids: list[float], x: list[float], k: int, n_feat: int,
                             group_size: int, bits: int) -> int:
    """E6: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED K-means assign -- the point ``x`` arrives as
    per-group quantized storage; the bridge dequantizes it and the trusted :func:`kmeans_assign` runs on the
    round-tripped point. The SOLE certified error is the INPUT round-trip (R17-certified by
    ``quantization_error_bound``); the distance ranking is exact. Keep the bridged point off an exact
    equidistant tie so the round-trip never flips the cluster. Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return kmeans_assign(centroids, dx, k, n_feat)


# ============================================================================================================
# 2. PREPROCESSING SCALERS -- FIT produces a BAKED transform (the sqrt is at fit time); TRANSFORM is the exact
#    fixed kernel. INDEPENDENT verifiers: standardized training data has mean ~ 0 / std ~ 1; min-max lies in
#    [0, 1].
# ============================================================================================================

@dataclass(frozen=True)
class StandardScaler:
    """A baked StandardScaler: per-feature ``mean`` / ``std`` (the constants FIT produced). The TRANSFORM
    ``(x - mean) / std`` is then exact division (the ``sqrt`` that produced ``std`` was a FIT-time
    transcendental; the baked std makes the transform transcendental-free). ``mean`` / ``std`` are length
    ``n_feat``; ``std`` entries are all > 0 (a constant feature, std 0, is rejected at FIT time). Validated in
    ``__post_init__``."""

    mean: tuple[float, ...]
    std: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mean:
            raise ValueError("StandardScaler needs at least one feature")
        if len(self.std) != len(self.mean):
            raise ValueError(f"StandardScaler mean/std length mismatch ({len(self.mean)} != {len(self.std)})")
        _finite(self.mean, "StandardScaler mean")
        _finite(self.std, "StandardScaler std")
        if any(float(s) <= 0.0 for s in self.std):
            raise ValueError("StandardScaler std entries must all be > 0 (a constant feature has std 0; "
                             "drop it or use a different scaler)")

    @property
    def n_feat(self) -> int:
        return len(self.mean)


def standard_scaler_fit(X: list[float], n_samples: int, n_feat: int) -> StandardScaler:
    """FIT a StandardScaler: per-feature ``(mean, std)`` where ``std = sqrt(population variance)`` (variance
    ``/n_samples`` -- the population convention, NOT the sample ``/(n-1)``). The ``sqrt`` is the ONLY
    transcendental, and it is at FIT TIME -- it bakes into ``std`` so the later transform is exact. ``X`` is
    ``n_samples x n_feat`` row-major. A CONSTANT feature (zero variance -> std 0) is REJECTED here (the
    ``StandardScaler`` dataclass raises) because ``(x - mean) / 0`` is undefined -- the caller must drop a
    constant feature or pick MinMax. Side-effect-free."""
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"scaler n_feat must be >= 1; got {n_feat}")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError(f"scaler n_samples must be >= 1; got {n_samples}")
    if len(X) != n_samples * n_feat:
        raise ValueError(f"scaler X must be n_samples*n_feat = {n_samples * n_feat}; got {len(X)}")
    Xf = [float(v) for v in X]
    _finite(Xf, "scaler X")
    mean = [0.0] * n_feat
    for i in range(n_samples):
        base = i * n_feat
        for j in range(n_feat):
            mean[j] += Xf[base + j]
    mean = [m / n_samples for m in mean]
    var = [0.0] * n_feat
    for i in range(n_samples):
        base = i * n_feat
        for j in range(n_feat):
            d = Xf[base + j] - mean[j]
            var[j] += d * d
    var = [v / n_samples for v in var]                          # POPULATION variance (/n_samples)
    std = [math.sqrt(v) for v in var]                           # the SOLE transcendental, at FIT time -> baked
    return StandardScaler(tuple(mean), tuple(std))


def standard_scaler_transform(x: list[float], mean, std) -> list[float]:
    """TRANSFORM a point by a baked StandardScaler: ``out[j] = (x[j] - mean[j]) / std[j]`` per feature -- EXACT
    division (the baked mean/std make this transcendental-free; the ``sqrt`` was paid at fit time). ``x`` /
    ``mean`` / ``std`` are equal length. Returns a fresh list (side-effect-free)."""
    if len(x) != len(mean) or len(x) != len(std):
        raise ValueError(f"scaler transform length mismatch (x={len(x)} mean={len(mean)} std={len(std)})")
    _finite(x, "scaler transform x")
    _finite(mean, "scaler transform mean")
    _finite(std, "scaler transform std")
    out = []
    for j in range(len(x)):
        if float(std[j]) <= 0.0:
            raise ValueError(f"scaler transform: std[{j}] must be > 0; got {std[j]}")
        out.append((float(x[j]) - float(mean[j])) / float(std[j]))
    return out


@dataclass(frozen=True)
class MinMaxScaler:
    """A baked MinMaxScaler: per-feature ``mn`` / ``mx`` (the constants FIT produced). The TRANSFORM
    ``(x - mn) / (mx - mn)`` maps the training range to ``[0, 1]`` -- exact division (guarded ``mx > mn``).
    ``mn`` / ``mx`` are length ``n_feat`` with ``mx > mn`` per feature. Validated in ``__post_init__``."""

    mn: tuple[float, ...]
    mx: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.mn:
            raise ValueError("MinMaxScaler needs at least one feature")
        if len(self.mx) != len(self.mn):
            raise ValueError(f"MinMaxScaler mn/mx length mismatch ({len(self.mn)} != {len(self.mx)})")
        _finite(self.mn, "MinMaxScaler min")
        _finite(self.mx, "MinMaxScaler max")
        if any(float(self.mx[j]) <= float(self.mn[j]) for j in range(len(self.mn))):
            raise ValueError("MinMaxScaler needs mx > mn per feature (a constant feature has mx == mn; "
                             "drop it or use a different scaler)")

    @property
    def n_feat(self) -> int:
        return len(self.mn)


def minmax_fit(X: list[float], n_samples: int, n_feat: int) -> MinMaxScaler:
    """FIT a MinMaxScaler: per-feature ``(min, max)`` over the training set. NO transcendental at all (just a
    min / max reduction). A CONSTANT feature (``mx == mn``) is REJECTED here (the ``MinMaxScaler`` dataclass
    raises) because ``(x - mn) / (mx - mn)`` would divide by zero. ``X`` is ``n_samples x n_feat`` row-major.
    Side-effect-free."""
    if type(n_feat) is not int or n_feat < 1:
        raise ValueError(f"scaler n_feat must be >= 1; got {n_feat}")
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError(f"scaler n_samples must be >= 1; got {n_samples}")
    if len(X) != n_samples * n_feat:
        raise ValueError(f"scaler X must be n_samples*n_feat = {n_samples * n_feat}; got {len(X)}")
    Xf = [float(v) for v in X]
    _finite(Xf, "scaler X")
    mn = list(Xf[0:n_feat])
    mx = list(Xf[0:n_feat])
    for i in range(1, n_samples):
        base = i * n_feat
        for j in range(n_feat):
            v = Xf[base + j]
            if v < mn[j]:
                mn[j] = v
            if v > mx[j]:
                mx[j] = v
    return MinMaxScaler(tuple(mn), tuple(mx))


def minmax_transform(x: list[float], mn, mx) -> list[float]:
    """TRANSFORM a point by a baked MinMaxScaler: ``out[j] = (x[j] - mn[j]) / (mx[j] - mn[j])`` per feature --
    EXACT division (guarded ``mx > mn``). A training-set point lands in ``[0, 1]``; an out-of-range query may
    fall outside (that is the correct MinMax behaviour). Returns a fresh list (side-effect-free)."""
    if len(x) != len(mn) or len(x) != len(mx):
        raise ValueError(f"scaler transform length mismatch (x={len(x)} mn={len(mn)} mx={len(mx)})")
    _finite(x, "scaler transform x")
    _finite(mn, "scaler transform min")
    _finite(mx, "scaler transform max")
    out = []
    for j in range(len(x)):
        denom = float(mx[j]) - float(mn[j])
        if denom <= 0.0:
            raise ValueError(f"scaler transform: mx[{j}] must be > mn[{j}]; got mx={mx[j]} mn={mn[j]}")
        out.append((float(x[j]) - float(mn[j])) / denom)
    return out


# ============================================================================================================
# 3. CROSS-VALIDATION FOLDS -- a deterministic partition of [0, n) via the M3 LCG shuffle (REUSED, not
#    reinvented). INDEPENDENT verifier: the folds are a genuine partition (disjoint + cover) with balanced sizes.
# ============================================================================================================

def k_fold_indices(n_samples: int, n_folds: int, seed: int) -> list[list[int]]:
    """A deterministic K-FOLD partition of ``[0, n_samples)`` into ``n_folds`` index lists. REUSES the M3
    ``training._lcg_permutation`` (the LCG Fisher-Yates shuffle keyed by ``seed`` -- no new RNG, no ``random``,
    no numpy): shuffle ``range(n_samples)`` once, then deal the shuffled indices round-robin-free into
    contiguous balanced chunks (the first ``n_samples % n_folds`` folds get one extra element, so fold sizes
    differ by at most 1). The result is a genuine PARTITION: the folds are pairwise DISJOINT and their union is
    exactly ``{0, ..., n_samples - 1}`` (the independent verifier checks this). Same ``(n_samples, n_folds,
    seed)`` -> the same folds (a test pins reproducibility). Side-effect-free."""
    if type(n_samples) is not int or n_samples < 1:
        raise ValueError(f"k_fold n_samples must be >= 1; got {n_samples}")
    if type(n_folds) is not int or n_folds < 2:
        raise ValueError(f"k_fold n_folds must be >= 2; got {n_folds}")
    if type(seed) is not int:
        raise ValueError(f"k_fold seed must be an integer; got {seed!r}")
    if n_folds > n_samples:
        raise ValueError(f"k_fold n_folds must be <= n_samples = {n_samples}; got {n_folds}")
    order = _lcg_permutation(n_samples, seed)                   # REUSE the M3 deterministic LCG shuffle
    base = n_samples // n_folds
    extra = n_samples % n_folds                                 # the first `extra` folds get one more element
    folds: list[list[int]] = []
    start = 0
    for f in range(n_folds):
        size = base + (1 if f < extra else 0)
        folds.append(sorted(order[start:start + size]))         # sorted within a fold (stable, readable)
        start += size
    return folds


def k_fold_is_partition(folds: list[list[int]], n_samples: int) -> bool:
    """INDEPENDENT verifier for :func:`k_fold_indices`: confirm the folds are a GENUINE PARTITION of
    ``[0, n_samples)`` by a fully independent recompute -- the folds are pairwise DISJOINT (no index appears
    twice, checked by total count == union size) and their UNION is exactly ``{0, ..., n_samples - 1}`` (no
    index missing, no index out of range). Returns True iff both hold. This is the defining property a CV split
    must have (every sample is in exactly one validation fold). Side-effect-free."""
    seen: set[int] = set()
    total = 0
    for fold in folds:
        for idx in fold:
            seen.add(idx)
            total += 1
    return total == len(seen) and seen == set(range(n_samples))


# ============================================================================================================
# 4. AUTOENCODER -- a COMPOSITION (REUSES matmul + activation + the M1 MSE). FORWARD encodes then decodes;
#    reconstruction_error is the MSE. INDEPENDENT verifier: tied-identity weights reconstruct the input (err ~ 0).
# ============================================================================================================

def autoencoder_forward(x: list[float], n_in: int, n_hidden: int, W_enc, b_enc, W_dec, b_dec,
                        activation: str = "relu") -> list[float]:
    """Autoencoder FORWARD -- a COMPOSITION of ops BCIR already ships (do NOT reinvent matmul / activation):

        h     = activation(x @ W_enc + b_enc)        (encode: REUSE matmul + the G1 activation reference)
        recon = h @ W_dec + b_dec                    (decode: REUSE matmul; linear output)

    ``x`` is length ``n_in``; ``W_enc`` is ``n_in x n_hidden`` row-major; ``b_enc`` is length ``n_hidden``;
    ``W_dec`` is ``n_hidden x n_in`` row-major; ``b_dec`` is length ``n_in``. The matmuls go through
    ``matmul.matmul_reference`` (a ``1 x n_in`` @ ``n_in x n_hidden`` and a ``1 x n_hidden`` @ ``n_hidden x
    n_in``), the activation through ``activation.activation_reference``. Returns the length-``n_in``
    reconstruction. ``activation`` in ``{"relu","sigmoid","tanh","gelu","softmax"}`` -- ``relu`` is exact; the
    rest ride the trusted libm edge (and need f32). Side-effect-free."""
    if type(n_in) is not int or type(n_hidden) is not int or n_in < 1 or n_hidden < 1:
        raise ValueError(f"autoencoder dims must be >= 1; got n_in={n_in} n_hidden={n_hidden}")
    if len(x) != n_in:
        raise ValueError(f"autoencoder x must be length n_in = {n_in}; got {len(x)}")
    if len(W_enc) != n_in * n_hidden:
        raise ValueError(f"W_enc must be n_in*n_hidden = {n_in * n_hidden}; got {len(W_enc)}")
    if len(b_enc) != n_hidden:
        raise ValueError(f"b_enc must be length n_hidden = {n_hidden}; got {len(b_enc)}")
    if len(W_dec) != n_hidden * n_in:
        raise ValueError(f"W_dec must be n_hidden*n_in = {n_hidden * n_in}; got {len(W_dec)}")
    if len(b_dec) != n_in:
        raise ValueError(f"b_dec must be length n_in = {n_in}; got {len(b_dec)}")
    xf = [float(v) for v in x]
    # encode: x (1 x n_in) @ W_enc (n_in x n_hidden) -> pre (1 x n_hidden); REUSE matmul_reference.
    pre = matmul_reference(xf, [float(v) for v in W_enc], 1, n_hidden, n_in)
    pre = [pre[j] + float(b_enc[j]) for j in range(n_hidden)]
    # activation: the hidden non-linearity; REUSE the G1 reference (softmax over the single hidden row).
    h = activation_reference(activation, pre, n_hidden if activation == "softmax" else None)
    # decode: h (1 x n_hidden) @ W_dec (n_hidden x n_in) -> recon (1 x n_in); REUSE matmul_reference.
    recon = matmul_reference(h, [float(v) for v in W_dec], 1, n_in, n_hidden)
    return [recon[j] + float(b_dec[j]) for j in range(n_in)]


def reconstruction_error(x: list[float], recon: list[float]) -> float:
    """The autoencoder RECONSTRUCTION ERROR: the mean-squared error between the input and its reconstruction.
    REUSES the M1 ``losses.mse_value`` (no second MSE here). ``x`` and ``recon`` are equal length. The
    INDEPENDENT-verifier property: with TIED IDENTITY weights (``W_enc = W_dec = I``, zero bias, an
    identity-ish activation -- ``relu`` on a positive input, or linear) the reconstruction EQUALS the input, so
    the error is ~ 0; a non-trivial autoencoder (compressing through a bottleneck) has error > 0.
    Side-effect-free."""
    return mse_value(list(x), list(recon))


def autoencoder_via_bridge(x: list[float], n_in: int, n_hidden: int, W_enc, b_enc, W_dec, b_dec,
                           group_size: int, bits: int, activation: str = "relu") -> list[float]:
    """E6: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED autoencoder forward -- the INPUT ``x``
    arrives as per-group quantized storage; the bridge dequantizes it and the trusted
    :func:`autoencoder_forward` runs on the round-tripped input (the baked weights are trusted/exact). The SOLE
    certified error is the INPUT round-trip (R17). Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return autoencoder_forward(dx, n_in, n_hidden, W_enc, b_enc, W_dec, b_dec, activation)


# ============================================================================================================
# 5. EMBEDDING LOOKUP -- a baked n_vocab x dim table; exact row gather. INDEPENDENT verifier: the lookup equals
#    the direct row slice; an out-of-range id raises.
# ============================================================================================================

@dataclass(frozen=True)
class EmbeddingTable:
    """A baked embedding table: ``n_vocab`` rows of ``dim`` floats, row-major (``table[v*dim + d]`` is the d-th
    component of vocabulary id v). The constants a trained embedding layer bakes; the lookup is a pure row
    gather (exact, no arithmetic). Validated in ``__post_init__``."""

    table: tuple[float, ...]
    n_vocab: int
    dim: int

    def __post_init__(self) -> None:
        if type(self.n_vocab) is not int or self.n_vocab < 1:
            raise ValueError(f"embedding n_vocab must be >= 1; got {self.n_vocab}")
        if type(self.dim) is not int or self.dim < 1:
            raise ValueError(f"embedding dim must be >= 1; got {self.dim}")
        if len(self.table) != self.n_vocab * self.dim:
            raise ValueError(f"embedding table must be n_vocab*dim = {self.n_vocab * self.dim}; "
                             f"got {len(self.table)}")
        _finite(self.table, "embedding table")

    def row(self, vid: int) -> list[float]:
        """The raw row slice for id ``vid`` -- the direct-slice path the INDEPENDENT verifier compares against
        :func:`embedding_lookup`. Validates the id range."""
        if type(vid) is not int or not (0 <= vid < self.n_vocab):
            raise ValueError(f"embedding id {vid} out of range [0, {self.n_vocab})")
        return list(self.table[vid * self.dim:(vid + 1) * self.dim])


def embedding_lookup(table: EmbeddingTable, ids: list[int], dim: int) -> list[float]:
    """EMBEDDING LOOKUP: gather the rows for a list of integer ids -- ``out[t*dim + d] = table[ids[t]][d]`` for
    each id in ``ids``, EXACT (a pure row gather; no arithmetic, no transcendental). Returns a flat
    ``len(ids) x dim`` row-major list (the gathered rows concatenated, in id order). Each id is validated
    against the vocabulary range; an OUT-OF-RANGE id raises (the INDEPENDENT verifier checks both the
    direct-slice agreement and the out-of-range raise). ``dim`` must match the table's ``dim``.
    Side-effect-free."""
    if type(dim) is not int or dim != table.dim:
        raise ValueError(f"embedding_lookup dim {dim} != table dim {table.dim}")
    out: list[float] = []
    for vid in ids:
        if type(vid) is not int or not (0 <= vid < table.n_vocab):
            raise ValueError(f"embedding id {vid} out of range [0, {table.n_vocab})")
        start = vid * dim
        out.extend(table.table[start:start + dim])
    return out


# ============================================================================================================
# Op-level well-formedness (mirrors check_classical). D2: promoted to the R22/R23 law surface
# via verify.verify_ml_spec (caller-passed messages; no verifier import here).
# ============================================================================================================

_DTYPES = ("f32", "i32")
_KINDS = ("kmeans", "scaler", "autoencoder", "embedding")
# the activations that ride a transcendental (need f32); relu is exact. (mirrors activation.TRANSCENDENTAL_*.)
_TRANSCENDENTAL_ACTS = ("sigmoid", "tanh", "gelu", "softmax")


def check_unsupervised(kind: str, *, n_feat: int, n_unit: int, in_dtype: str, out_dtype: str,
                       activation: str = "relu") -> list[str]:
    """Op-level well-formedness for an unsupervised / pipeline claim, returned as a list of error strings (empty
    == well-formed) -- the shape/dtype-law shape, lifted into one named checker. NOT a globally-numbered R-law
    (matches ``check_classical`` / ``check_recurrent``). ``kind`` in
    ``{"kmeans","scaler","autoencoder","embedding"}``; ``n_feat`` the feature / vector dimension; ``n_unit`` the
    organ's primary extent (k for K-means, n_feat again for a scaler, n_hidden for the autoencoder, n_vocab for
    the embedding); ``activation`` matters only for ``kind == "autoencoder"``. The laws:

      1. ``kind`` is a KNOWN unsupervised organ;
      2. extents are POSITIVE: ``n_feat`` and ``n_unit`` both >= 1;
      3. dtype is a KNOWN dtype and PRESERVED: ``out_dtype == in_dtype``;
      4. THE QUARANTINE DTYPE RULE: an organ that uses a TRANSCENDENTAL at RUNTIME needs an f32 result. K-means
         ASSIGN (squared-distance ranking), a scaler TRANSFORM (the ``sqrt`` was at FIT time -> the transform is
         exact division), and the embedding LOOKUP (a row gather) are EXACT -- they may be i32 OR f32. An
         AUTOENCODER with a TRANSCENDENTAL activation (sigmoid/tanh/gelu/softmax) uses the libm edge at forward
         time, so it needs f32; an autoencoder with the exact ``relu`` may be i32 or f32;
      5. an autoencoder declares a known activation."""
    errs: list[str] = []
    if kind not in _KINDS:
        errs.append(f"unsupervised: kind {kind!r} is not a known organ {_KINDS}")
    for nm, v in (("n_feat", n_feat), ("n_unit", n_unit)):
        if v < 1:
            errs.append(f"unsupervised: {nm} must be >= 1; got {v}")
    for nm, dt in (("in", in_dtype), ("out", out_dtype)):
        if dt not in _DTYPES:
            errs.append(f"unsupervised: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"unsupervised: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    # the quarantine dtype rule: only a transcendental-activation autoencoder uses a runtime transcendental.
    uses_transcendental = (kind == "autoencoder" and activation in _TRANSCENDENTAL_ACTS)
    if uses_transcendental and in_dtype != "f32":
        errs.append(f"unsupervised: an autoencoder with a {activation!r} activation uses a transcendental "
                    f"(the libm edge returns float), so it needs f32, got {in_dtype!r}")
    if kind == "autoencoder" and activation not in ("relu",) + _TRANSCENDENTAL_ACTS:
        errs.append(f"unsupervised: autoencoder activation {activation!r} is not a known activation")
    return errs
