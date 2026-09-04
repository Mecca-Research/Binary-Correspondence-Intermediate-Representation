"""E5 (ML-breadth) -- CLASSICAL-ML PREDICT-PATH wraps (KNN / decision tree / SVM / Naive Bayes), the FIFTH slice
of the ML-breadth ladder after E1 (OLS), E2 (PCA), E3 (the Transformer block), E4 (RNN/LSTM/GRU).

THE HONEST FRAMING -- the TRAIN vs PREDICT split (the research finding the E7 capstone cites).
Classical ML splits SHARPLY into two halves with OPPOSITE structure:

  * TRAINING -- decision-tree INDUCTION (the greedy recursive split search over the data), the SVM dual
    QUADRATIC-PROGRAM solve, Naive-Bayes FITTING (per-class mean/variance estimation over a training set), KNN
    "fit" (it has none -- it stores the training set). The non-trivial trainers (tree induction, the SVM QP) are
    ITERATIVE / COMBINATORIAL optimization: no fixed dataflow, data-dependent control flow, convergence loops,
    variable-length intermediate structures. That is a POOR fit for BCIR's fixed-shape, planned-claim model --
    it belongs in a LIBRARY / Python (scikit-learn, libsvm), NOT as a BCIR claim. BCIR does not try to own it.

  * PREDICT / INFERENCE -- the OPPOSITE. Once trained, the model is a FIXED, BAKED set of constants: a decision
    tree is its (threshold, feature, child) arrays; an SVM is its support vectors + dual coefficients (alpha_i *
    y_i) + bias; a Gaussian-NB is its per-class log-prior + per-feature mean/variance; a KNN is its stored
    training set. PREDICT over a baked model is a DETERMINISTIC, FIXED-SHAPE kernel -- exactly the **G5
    baked-weights inference pattern** (`bcir/lower/inference.py::emit_inference_kernel_c`) combined with the
    Area-B "integrate, don't reinvent" discipline. So E5 is the classical-ML PREDICT path AS baked-model kernels.

BCIR owns the CALLING side (the row-major layout, the baked params, the Q8 feature boundary -- the R17 input
bound); the transcendentals ride the trusted ``c.call.libm:`` edge (RBF-SVM's ``exp`` -> ``expf``, Gaussian-NB's
``log`` -> ``logf``, both already ``-lm`` -- NO linkflags change); the rest is EXACT arithmetic (dot products,
squared distances, threshold comparisons, argmax / majority vote). ``libsvm`` is the canonical external library
for the SVM decision function in the "integrate, don't reinvent" framing, but we emit the decision function
DIRECTLY (it IS what libsvm's ``svm_predict`` computes) -- self-contained, no libsvm dependency, with only the
RBF ``exp`` on the libm edge.

THE FOUR PREDICTORS (each: a plain-float reference = the source of truth, an INDEPENDENT verifier, and a
``*_via_bridge`` Q8 round-trip of the FEATURE vector = the R17 input bound):

  1. KNN -- baked training set ``(X_train, y_train)``, query ``x``, ``k``. SQUARED Euclidean distance (exact; no
     sqrt needed to RANK -- ranking on squared distance is IDENTICAL to ranking on distance, since sqrt is
     monotone, so classification needs NO transcendental), take the ``k`` smallest (tie-break: lowest index),
     majority vote (``knn_classify``; tie-break: lowest class id) or mean of the k targets (``knn_regress``).
  2. DECISION TREE -- a baked flat-array tree (``feature`` / ``threshold`` / ``left`` / ``right`` / ``leaf_value``;
     a leaf is signalled by ``feature[node] == -1``). Traverse from the root: at an internal node go ``left`` if
     ``x[feature] <= threshold`` else ``right``, until a leaf. EXACT (comparisons + leaf labels; NO transcendental).
  3. SVM -- baked support vectors ``SV``, dual coeffs ``alpha_y`` (= alpha_i * y_i), bias ``b``. Linear decision
     ``Sum_i alpha_y[i] * (SV[i] . x) + b`` (exact dot products); RBF decision
     ``Sum_i alpha_y[i] * exp(-gamma * ||x - SV[i]||^2) + b`` (the ``exp`` is the transcendental -> libm ``expf``).
     ``svm_predict`` = ``sign(decision)`` -> class.
  4. NAIVE BAYES -- Gaussian NB: baked ``log_prior[c]`` + per-class per-feature ``mean[c][j]`` / ``var[c][j]``.
     ``log_posterior(x)[c] = log_prior[c] - 0.5 * Sum_j [ (x_j - mean[c][j])^2 / var[c][j] + log(2*pi*var[c][j]) ]``
     (the ``log`` is the transcendental -> libm ``logf``). The ``log(2*pi*var)`` normaliser is DATA-INDEPENDENT so
     it can be PRECOMPUTED / BAKED -- the predict kernel then needs only the data-dependent quadratic term, with
     the ``log`` baked-constant. ``nb_predict`` = argmax_c log_posterior.

THE QUARANTINE BOUNDARY (mirrors ols.py / pca.py / transformer.py / recurrent.py). This module is OFF the
legality path: it imports NO verifier, emits NO ``Diagnostic``, returns plain floats / plain ints (no
graded/confidence wrapper). KNN and the decision tree are EXACT (integer/float-exact -- comparisons, dot products,
argmax: no transcendental at all); RBF-SVM and Gaussian-NB ride the trusted libm edge for their single
transcendental (``exp`` / ``log``). Deterministic, dependency-free (stdlib ``math`` only). A research/oracle organ.

LAYOUT (row-major / flat throughout). A training matrix ``X_train`` is ``n_train x n_feat`` row-major
(``X_train[i*n_feat + j]`` is sample i feature j); a query ``x`` is length ``n_feat``. SVM support vectors ``SV``
are ``n_sv x n_feat`` row-major; Gaussian-NB ``mean`` / ``var`` are ``n_class x n_feat`` row-major. All references
return FRESH values and do NOT mutate their inputs (side-effect-free oracle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain

from .quantize import dequantize, quantize_per_group


# ============================================================================================================
# 1. KNN PREDICT -- baked training set; squared-distance ranking (no sqrt); majority vote / mean.
# ============================================================================================================


@dataclass(frozen=True)
class KnnModel:
    """A baked KNN model: the stored training set is the model (KNN has no separate trained parameters -- its
    "fit" is just storing ``(X_train, y_train)``). ``X_train`` is ``n_train x n_feat`` row-major
    (``X_train[i*n_feat + j]``); ``y_train`` is length ``n_train`` (class ids for classification, real targets for
    regression). ``n_feat`` is the feature count. Validated in ``__post_init__``."""

    X_train: tuple[float, ...]
    y_train: tuple[float, ...]
    n_feat: int

    def __post_init__(self) -> None:
        if type(self.n_feat) is not int or self.n_feat < 1:
            raise ValueError(f"knn n_feat must be >= 1; got {self.n_feat}")
        if not self.y_train:
            raise ValueError("knn needs at least one training sample")
        if len(self.X_train) != len(self.y_train) * self.n_feat:
            raise ValueError(
                f"knn X_train must be n_train*n_feat = {len(self.y_train) * self.n_feat}; "
                f"got {len(self.X_train)}"
            )
        try:
            finite = all(math.isfinite(float(value)) for value in chain(self.X_train, self.y_train))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("knn training values must be finite numbers") from exc
        if not finite:
            raise ValueError("knn training values must be finite numbers")

    @property
    def n_train(self) -> int:
        return len(self.y_train)


def _squared_distance_at(a, a_offset: int, b, b_offset: int, length: int) -> float:
    """Squared Euclidean distance over two flat, validated spans without slices."""
    acc = 0.0
    for index in range(length):
        delta = float(a[a_offset + index]) - float(b[b_offset + index])
        acc += delta * delta
    return acc


def squared_distance(a: list[float], b: list[float]) -> float:
    """The EXACT squared Euclidean distance ``Sum_j (a_j - b_j)^2`` between two equal-length feature vectors. NO
    sqrt: ranking on squared distance is IDENTICAL to ranking on distance (sqrt is monotone increasing on
    [0, inf)), so KNN classification / regression needs NO transcendental at all -- the squared distance both
    ranks the neighbours and is exact arithmetic. Used by the calling side; the C twin computes the same sum."""
    if len(a) != len(b):
        raise ValueError(f"squared_distance: length mismatch ({len(a)} != {len(b)})")
    try:
        if any(not math.isfinite(float(value)) for value in chain(a, b)):
            raise ValueError("squared_distance inputs must be finite")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("squared_distance inputs must be finite numbers") from exc
    return _squared_distance_at(a, 0, b, 0, len(a))


def _validate_knn_query(model: KnnModel, x, k) -> list[float]:
    if not isinstance(model, KnnModel):
        raise ValueError("knn model must be a KnnModel")
    if len(x) != model.n_feat:
        raise ValueError(f"knn query must be length n_feat = {model.n_feat}; got {len(x)}")
    if type(k) is not int or not 1 <= k <= model.n_train:
        raise ValueError(
            f"knn k must be an integer in [1, n_train] = [1, {model.n_train}]; got {k!r}"
        )
    try:
        values = [float(value) for value in x]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("knn query entries must be finite numbers") from exc
    if any(not math.isfinite(value) for value in values):
        raise ValueError("knn query entries must be finite numbers")
    return values


def knn_neighbours(model: KnnModel, x: list[float], k: int) -> list[int]:
    """The indices of the ``k`` nearest training samples to query ``x``, by SQUARED Euclidean distance, with a
    DETERMINISTIC tie-break: ties on distance break by LOWEST training index. Returns a length-``k`` list of
    training indices (closest first). The squared distance is exact (no sqrt -- monotone, so the same ranking),
    so this is fully deterministic / reproducible. ``x`` must be length ``n_feat``; ``k`` in ``[1, n_train]``."""
    nf = model.n_feat
    xf = _validate_knn_query(model, x, k)

    def scored():
        return (
            (_squared_distance_at(xf, 0, model.X_train, i * nf, nf), i)
            for i in range(model.n_train)
        )

    # Keep only k rows when k is small; heapq preserves the tuple's deterministic
    # (distance, index) ordering and avoids materializing/sorting the whole corpus.
    import heapq

    nearest = heapq.nsmallest(k, scored()) if k * 4 <= model.n_train else sorted(scored())[:k]
    return [i for _, i in nearest]


def knn_classify(model: KnnModel, x: list[float], k: int) -> int:
    """KNN CLASSIFICATION: majority vote over the ``k`` nearest neighbours' class ids, with a DETERMINISTIC
    tie-break -- a tie in vote count breaks by LOWEST class id. Returns the predicted class id (int). The class
    ids come from ``y_train`` (interpreted as integers). Exact: distances are squared-Euclidean (no sqrt), the
    vote is integer counting -- no transcendental, fully reproducible. Side-effect-free."""
    idxs = knn_neighbours(model, x, k)
    counts: dict[int, int] = {}
    for i in idxs:
        raw = model.y_train[i]
        if isinstance(raw, bool) or not float(raw).is_integer():
            raise ValueError(f"knn class label {raw!r} is not an exact integer")
        cls = int(raw)
        counts[cls] = counts.get(cls, 0) + 1
    best_count = max(counts.values())
    # tie-break: lowest class id among the classes achieving the max count.
    return min(cls for cls, c in counts.items() if c == best_count)


def knn_regress(model: KnnModel, x: list[float], k: int) -> float:
    """KNN REGRESSION: the mean of the ``k`` nearest neighbours' targets (``y_train`` as real values). Exact
    (squared-distance ranking, then an average -- no transcendental). Side-effect-free."""
    idxs = knn_neighbours(model, x, k)
    return sum(float(model.y_train[i]) for i in idxs) / float(len(idxs))


def knn_neighbours_independent(model: KnnModel, x: list[float], k: int) -> list[int]:
    """INDEPENDENT verifier for ``knn_neighbours``: recompute the k nearest by a FULLY independent path -- an
    O(n^2) selection (repeatedly scan for the current minimum (distance, index) among the not-yet-selected,
    rather than one sort). Same deterministic tie-break (lowest index). Must return the SAME neighbour set +
    order as ``knn_neighbours`` (which uses one ``sort``); a genuine cross-check that the ranking is correct."""
    nf = model.n_feat
    xf = _validate_knn_query(model, x, k)
    dists = [
        squared_distance(xf, list(model.X_train[i * nf : (i + 1) * nf]))
        for i in range(model.n_train)
    ]
    chosen: list[int] = []
    remaining = set(range(model.n_train))
    for _ in range(k):
        # the current minimum (distance, index) over the remaining set -- index is the tie-break.
        best = min(remaining, key=lambda i: (dists[i], i))
        chosen.append(best)
        remaining.discard(best)
    return chosen


def knn_classify_via_bridge(
    model: KnnModel, x: list[float], k: int, group_size: int, bits: int
) -> int:
    """E5: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED KNN classify -- the FEATURE vector ``x``
    arrives as per-group quantized storage; the bridge dequantizes it and the trusted ``knn_classify`` runs on
    the round-tripped query. The SOLE certified error is the INPUT round-trip (R17-certified by
    ``quantization_error_bound``); the distance / vote arithmetic is exact. Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return knn_classify(model, dx, k)


# ============================================================================================================
# 2. DECISION-TREE PREDICT -- a baked flat-array tree; exact threshold traversal (NO transcendental).
# ============================================================================================================


@dataclass(frozen=True)
class TreeModel:
    """A baked decision tree as FLAT arrays (the constants a trained tree bakes; CART / scikit-learn shape). For
    each node id ``n``: ``feature[n]`` is the split feature index (``-1`` signals a LEAF); ``threshold[n]`` the
    split threshold; ``left[n]`` / ``right[n]`` the child node ids (unused for a leaf); ``leaf_value[n]`` the
    leaf's prediction (class id or regression value, used only at a leaf). ``n_feat`` is the feature count; node
    0 is the root. Validated in ``__post_init__`` (consistent array lengths, children in range, root present)."""

    feature: tuple[int, ...]
    threshold: tuple[float, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    leaf_value: tuple[float, ...]
    n_feat: int

    def __post_init__(self) -> None:
        n = len(self.feature)
        if n < 1:
            raise ValueError("tree needs at least one node (the root)")
        if self.n_feat < 1:
            raise ValueError(f"tree n_feat must be >= 1; got {self.n_feat}")
        for nm, arr in (
            ("threshold", self.threshold),
            ("left", self.left),
            ("right", self.right),
            ("leaf_value", self.leaf_value),
        ):
            if len(arr) != n:
                raise ValueError(f"tree {nm} must have {n} entries (one per node); got {len(arr)}")
        for node in range(n):
            if self.feature[node] == -1:
                continue  # a leaf -- children / feature unused
            if not (0 <= self.feature[node] < self.n_feat):
                raise ValueError(
                    f"tree node {node}: feature {self.feature[node]} out of range "
                    f"[0, {self.n_feat})"
                )
            for child in (self.left[node], self.right[node]):
                if not (0 <= child < n):
                    raise ValueError(f"tree node {node}: child {child} out of range [0, {n})")

    @property
    def n_nodes(self) -> int:
        return len(self.feature)


def tree_predict(model: TreeModel, x: list[float]) -> float:
    """ITERATIVE decision-tree predict: traverse from the root (node 0). At an INTERNAL node go ``left`` if
    ``x[feature] <= threshold`` else ``right``; stop at a LEAF (``feature[node] == -1``) and return its
    ``leaf_value``. EXACT -- pure comparisons + a leaf-constant return, NO transcendental (integer/float-exact,
    0 ULP). ``x`` must be length ``n_feat``. A depth bound (= n_nodes) guards against a malformed cyclic tree
    (a well-formed tree's children always have a strictly larger node id in the CART layout, but the bound makes
    the traversal terminate even on a hand-built degenerate tree). Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"tree query must be length n_feat = {model.n_feat}; got {len(x)}")
    node = 0
    for _ in range(model.n_nodes + 1):  # bounded: at most n_nodes internal hops to a leaf
        if model.feature[node] == -1:
            return float(model.leaf_value[node])
        if float(x[model.feature[node]]) <= float(model.threshold[node]):
            node = model.left[node]
        else:
            node = model.right[node]
    raise ValueError("tree_predict: traversal exceeded the depth bound (malformed / cyclic tree)")


def tree_predict_recursive(
    model: TreeModel, x: list[float], node: int = 0, depth: int = 0
) -> float:
    """INDEPENDENT verifier for ``tree_predict``: a separate RECURSIVE traversal (vs the iterative array walk),
    on the SAME baked tree -- it must return the SAME leaf value. The two reach the leaf by independent control
    flow (recursion vs a bounded loop), so agreement cross-checks the traversal. A depth bound (= n_nodes) raises
    on a malformed cyclic tree. Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"tree query must be length n_feat = {model.n_feat}; got {len(x)}")
    if depth > model.n_nodes:
        raise ValueError("tree_predict_recursive: depth bound exceeded (malformed / cyclic tree)")
    if model.feature[node] == -1:
        return float(model.leaf_value[node])
    if float(x[model.feature[node]]) <= float(model.threshold[node]):
        return tree_predict_recursive(model, x, model.left[node], depth + 1)
    return tree_predict_recursive(model, x, model.right[node], depth + 1)


def tree_leaves_reachable(model: TreeModel) -> set[int]:
    """INDEPENDENT structural verifier: the set of LEAF node ids reachable from the root by a bounded BFS over
    the (left, right) child edges -- used to validate the tree is well-formed (children in range, acyclic within
    the bound, every traversal lands on a leaf). Returns the reachable leaf node ids (a non-empty set for any
    valid tree -- the root alone is a leaf for a stump). Bounded to ``n_nodes`` expansions (no infinite loop on a
    degenerate tree)."""
    seen: set[int] = set()
    leaves: set[int] = set()
    frontier = [0]
    expansions = 0
    while frontier and expansions <= model.n_nodes:
        node = frontier.pop()
        expansions += 1
        if node in seen:
            continue
        seen.add(node)
        if model.feature[node] == -1:
            leaves.add(node)
        else:
            frontier.append(model.left[node])
            frontier.append(model.right[node])
    return leaves


def tree_predict_via_bridge(model: TreeModel, x: list[float], group_size: int, bits: int) -> float:
    """E5: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED tree predict -- the FEATURE vector ``x`` is
    round-tripped through the bridge then the trusted ``tree_predict`` runs. The SOLE certified error is the
    INPUT round-trip (R17); the threshold comparisons are exact. NOTE the boundary subtlety (honest): a feature
    EXACTLY on a threshold could in principle flip a branch after the round-trip -- keep the bridged test points
    AWAY from exact ties (as E1-E4 keep decision-boundary points off exact ties). Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return tree_predict(model, dx)


# ============================================================================================================
# 3. SVM DECISION-FUNCTION PREDICT -- baked support vectors + dual coeffs + bias; linear or RBF kernel.
# ============================================================================================================


@dataclass(frozen=True)
class SvmModel:
    """A baked SVM model (what a trained libsvm / scikit-learn SVC bakes for prediction -- the dual QP solve that
    PRODUCED it is the training half, off BCIR's path). ``SV`` is the support vectors ``n_sv x n_feat`` row-major
    (``SV[i*n_feat + j]``); ``alpha_y`` is the per-SV dual coefficient ALREADY MULTIPLIED by the label
    (``alpha_y[i] = alpha_i * y_i``), length ``n_sv``; ``b`` is the bias. ``n_feat`` is the feature count.
    Validated in ``__post_init__``."""

    SV: tuple[float, ...]
    alpha_y: tuple[float, ...]
    b: float
    n_feat: int

    def __post_init__(self) -> None:
        if self.n_feat < 1:
            raise ValueError(f"svm n_feat must be >= 1; got {self.n_feat}")
        if not self.alpha_y:
            raise ValueError("svm needs at least one support vector")
        if len(self.SV) != len(self.alpha_y) * self.n_feat:
            raise ValueError(
                f"svm SV must be n_sv*n_feat = {len(self.alpha_y) * self.n_feat}; got {len(self.SV)}"
            )

    @property
    def n_sv(self) -> int:
        return len(self.alpha_y)


def svm_decision_linear(model: SvmModel, x: list[float]) -> float:
    """The LINEAR-kernel SVM decision function ``f(x) = Sum_i alpha_y[i] * (SV[i] . x) + b`` (EXACT dot products
    + a sum + the bias; NO transcendental). This is exactly what libsvm's ``svm_predict`` computes for a linear
    kernel -- emitted DIRECTLY (self-contained, no libsvm dependency). ``x`` must be length ``n_feat``. The sign
    of the result is the class (``svm_predict``). Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"svm query must be length n_feat = {model.n_feat}; got {len(x)}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    acc = float(model.b)
    for i in range(model.n_sv):
        dot = 0.0
        base = i * nf
        for j in range(nf):
            dot += float(model.SV[base + j]) * xf[j]
        acc += float(model.alpha_y[i]) * dot
    return acc


def svm_decision_rbf(model: SvmModel, x: list[float], gamma: float) -> float:
    """The RBF-kernel SVM decision function ``f(x) = Sum_i alpha_y[i] * exp(-gamma * ||x - SV[i]||^2) + b``. The
    squared distance ``||x - SV[i]||^2`` and the dual-weighted sum are EXACT; the only transcendental is the
    ``exp`` -- it rides the trusted ``c.call.libm:`` edge (``expf`` -> ``-lm`` in the C twin). This is exactly
    libsvm's RBF ``svm_predict``, emitted directly. ``expf(-gamma*d^2)`` UNDERFLOWS to 0 for a far support vector
    (large ``d``) -- fine (a distant SV contributes ~0); there is no overflow (the argument is <= 0). ``gamma > 0``
    is the RBF width. ``x`` must be length ``n_feat``. Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"svm query must be length n_feat = {model.n_feat}; got {len(x)}")
    if gamma <= 0.0:
        raise ValueError(f"rbf gamma must be > 0; got {gamma}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    acc = float(model.b)
    for i in range(model.n_sv):
        sv = list(model.SV[i * nf : (i + 1) * nf])
        d2 = squared_distance(xf, sv)
        acc += float(model.alpha_y[i]) * math.exp(
            -float(gamma) * d2
        )  # exp: the sole transcendental (libm edge)
    return acc


def svm_predict(decision: float) -> int:
    """The SVM class from a decision value: ``sign(decision)`` -> ``+1`` (decision >= 0) or ``-1`` (decision < 0).
    The two-class libsvm convention. (Keep test points AWAY from ``decision == 0`` so float32-vs-float64 rounding
    never flips the class -- the boundary itself is measure-zero.)"""
    return 1 if decision >= 0.0 else -1


def svm_decision_linear_independent(model: SvmModel, x: list[float]) -> float:
    """INDEPENDENT verifier for ``svm_decision_linear``: recompute the decision by a different summation order --
    accumulate the per-SV contributions ``alpha_y[i] * (SV[i] . x)`` into a list and sum the list (vs the running
    accumulator), with the bias added last. Same value to float round-off; a genuine cross-check of the dot-sum."""
    if len(x) != model.n_feat:
        raise ValueError(f"svm query must be length n_feat = {model.n_feat}; got {len(x)}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    terms = []
    for i in range(model.n_sv):
        dot = sum(float(model.SV[i * nf + j]) * xf[j] for j in range(nf))
        terms.append(float(model.alpha_y[i]) * dot)
    return sum(terms) + float(model.b)


def svm_decision_rbf_independent(model: SvmModel, x: list[float], gamma: float) -> float:
    """INDEPENDENT verifier for ``svm_decision_rbf``: recompute the kernel sum term-by-term into a list (the
    squared distance via an explicit per-feature loop, the ``exp`` of ``-gamma*d2``), then sum + bias. Same value
    to float round-off -- a genuine cross-check of the RBF kernel sum."""
    if len(x) != model.n_feat:
        raise ValueError(f"svm query must be length n_feat = {model.n_feat}; got {len(x)}")
    if gamma <= 0.0:
        raise ValueError(f"rbf gamma must be > 0; got {gamma}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    terms = []
    for i in range(model.n_sv):
        d2 = 0.0
        for j in range(nf):
            d = xf[j] - float(model.SV[i * nf + j])
            d2 += d * d
        terms.append(float(model.alpha_y[i]) * math.exp(-float(gamma) * d2))
    return sum(terms) + float(model.b)


def svm_predict_linear_via_bridge(
    model: SvmModel, x: list[float], group_size: int, bits: int
) -> int:
    """E5: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED linear-SVM predict -- the FEATURE vector ``x``
    is round-tripped, the trusted ``svm_decision_linear`` runs, and the sign is the class. The SOLE certified
    error is the INPUT round-trip (R17). Keep the bridged point off the decision boundary so the round-trip never
    flips the sign. Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return svm_predict(svm_decision_linear(model, dx))


# ============================================================================================================
# 4. NAIVE-BAYES PREDICT -- Gaussian NB; baked log-prior + per-class mean/var; the log(2*pi*var) baked.
# ============================================================================================================


@dataclass(frozen=True)
class GaussianNbModel:
    """A baked Gaussian-Naive-Bayes model (what fitting produces; the fit = per-class mean/variance estimation,
    the training half off BCIR's path). ``log_prior`` is length ``n_class`` (the baked per-class log prior
    ``log P(c)``); ``mean`` / ``var`` are ``n_class x n_feat`` row-major (``mean[c*n_feat + j]`` = feature j's
    mean for class c; ``var`` likewise, the per-class per-feature variance, ``var > 0``). ``n_feat`` is the
    feature count. Validated in ``__post_init__`` (positive variances, consistent shapes)."""

    log_prior: tuple[float, ...]
    mean: tuple[float, ...]
    var: tuple[float, ...]
    n_feat: int

    def __post_init__(self) -> None:
        if self.n_feat < 1:
            raise ValueError(f"nb n_feat must be >= 1; got {self.n_feat}")
        nc = len(self.log_prior)
        if nc < 1:
            raise ValueError("nb needs at least one class")
        if len(self.mean) != nc * self.n_feat:
            raise ValueError(
                f"nb mean must be n_class*n_feat = {nc * self.n_feat}; got {len(self.mean)}"
            )
        if len(self.var) != nc * self.n_feat:
            raise ValueError(
                f"nb var must be n_class*n_feat = {nc * self.n_feat}; got {len(self.var)}"
            )
        if any(v <= 0.0 for v in self.var):
            raise ValueError(
                "nb var entries must all be > 0 (a Gaussian likelihood needs a positive variance)"
            )

    @property
    def n_class(self) -> int:
        return len(self.log_prior)

    def baked_log_norm(self) -> tuple[float, ...]:
        """The DATA-INDEPENDENT normaliser term ``log(2*pi*var[c][j])`` per class/feature -- it depends only on
        the baked variances, NOT on the query, so it is PRECOMPUTED / BAKED. The predict kernel then needs only
        the data-dependent quadratic term ``(x_j - mean)^2 / var`` plus this baked constant (the only ``log`` in
        Gaussian-NB predict is thus a bake-time constant; the runtime kernel adds NO new ``log``). Returns the
        ``n_class x n_feat`` row-major baked log-normaliser."""
        return tuple(math.log(2.0 * math.pi * v) for v in self.var)


def gaussian_nb_log_posterior(model: GaussianNbModel, x: list[float]) -> list[float]:
    """The Gaussian-NB log-posterior per class (up to the shared evidence constant, which argmax ignores):

        log_posterior[c] = log_prior[c] - 0.5 * Sum_j [ (x_j - mean[c][j])^2 / var[c][j] + log(2*pi*var[c][j]) ]

    Returns a length-``n_class`` list. The ``log(2*pi*var)`` normaliser is DATA-INDEPENDENT -> taken from the
    baked ``baked_log_norm`` (so the only ``log`` is a bake-time constant; in the C twin nothing logs at runtime
    -- the predict kernel needs only the quadratic term). The quadratic term and the sum are EXACT; ``var > 0`` is
    guaranteed by the model. ``x`` must be length ``n_feat``. Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"nb query must be length n_feat = {model.n_feat}; got {len(x)}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    log_norm = model.baked_log_norm()  # the baked log(2*pi*var) -- data-independent constant
    out = []
    for c in range(model.n_class):
        acc = 0.0
        base = c * nf
        for j in range(nf):
            d = xf[j] - float(model.mean[base + j])
            acc += (d * d) / float(model.var[base + j]) + float(log_norm[base + j])
        out.append(float(model.log_prior[c]) - 0.5 * acc)
    return out


def nb_predict(model: GaussianNbModel, x: list[float]) -> int:
    """Gaussian-NB classification: ``argmax_c log_posterior(x)[c]``, with a DETERMINISTIC tie-break -- a tie in
    log-posterior breaks by LOWEST class id. Returns the predicted class id (int). Side-effect-free."""
    lp = gaussian_nb_log_posterior(model, x)
    best = max(lp)
    return min(c for c, v in enumerate(lp) if v == best)


def gaussian_nb_log_posterior_independent(model: GaussianNbModel, x: list[float]) -> list[float]:
    """INDEPENDENT verifier for ``gaussian_nb_log_posterior``: recompute each class's log-posterior from the FULL
    Gaussian log-likelihood ``Sum_j log N(x_j; mean, var)`` directly, where
    ``log N(x;mu,s2) = -0.5*log(2*pi*s2) - 0.5*(x-mu)^2/s2`` -- the per-feature log of the Gaussian density,
    computed inline (NOT via the baked-normaliser path), then ``log_prior[c]`` added. Must equal
    ``gaussian_nb_log_posterior`` to float round-off -- a genuine cross-check that the baked-normaliser
    factoring is exactly the full Gaussian log-likelihood. Side-effect-free."""
    if len(x) != model.n_feat:
        raise ValueError(f"nb query must be length n_feat = {model.n_feat}; got {len(x)}")
    nf = model.n_feat
    xf = [float(v) for v in x]
    out = []
    for c in range(model.n_class):
        acc = float(model.log_prior[c])
        base = c * nf
        for j in range(nf):
            mu = float(model.mean[base + j])
            s2 = float(model.var[base + j])
            # the full per-feature Gaussian log-density (independent of the baked-normaliser factoring).
            acc += -0.5 * math.log(2.0 * math.pi * s2) - 0.5 * (xf[j] - mu) ** 2 / s2
        out.append(acc)
    return out


def nb_predict_via_bridge(
    model: GaussianNbModel, x: list[float], group_size: int, bits: int
) -> int:
    """E5: the Q8<->float32<->Q8 bridge wrapped around the TRUSTED Gaussian-NB predict -- the FEATURE vector
    ``x`` is round-tripped, the trusted ``nb_predict`` runs on the round-tripped query. The SOLE certified error
    is the INPUT round-trip (R17); the quadratic term is exact and the ``log`` normaliser is baked. Keep the
    bridged point away from a class-decision tie so the round-trip never flips the argmax. Side-effect-free."""
    dx = dequantize(quantize_per_group(x, group_size, bits))
    return nb_predict(model, dx)


# ============================================================================================================
# Op-level well-formedness (mirrors check_recurrent). D2: promoted to the R22/R23 law surface
# via verify.verify_ml_spec (caller-passed messages; no verifier import here).
# ============================================================================================================

_DTYPES = ("f32", "i32")
_KINDS = ("knn", "tree", "svm", "nb")


def check_classical(
    kind: str,
    *,
    n_feat: int,
    n_param: int,
    in_dtype: str,
    out_dtype: str,
    svm_kernel: str = "linear",
) -> list[str]:
    """Op-level well-formedness for a classical-ML PREDICT claim, returned as a list of error strings (empty ==
    well-formed) -- the shape/dtype-law shape, lifted into one named checker. NOT a globally-numbered R-law
    (matches ``check_recurrent`` / ``check_transformer``). ``kind`` in ``{"knn","tree","svm","nb"}``; ``n_feat``
    the feature count; ``n_param`` the predictor's primary extent (n_train for KNN, n_nodes for tree, n_sv for
    SVM, n_class for NB); ``svm_kernel`` ("linear" / "rbf") matters only for ``kind == "svm"``. The laws:

      1. ``kind`` is a KNOWN classical predictor;
      2. extents are POSITIVE: ``n_feat`` and ``n_param`` both >= 1;
      3. dtype is a KNOWN dtype and PRESERVED: ``out_dtype == in_dtype``;
      4. THE QUARANTINE DTYPE RULE: an RBF-SVM (its ``exp``) and Gaussian-NB (its ``log``) use a TRANSCENDENTAL
         (the libm edge returns float), so they need an f32 result -- never i32. KNN and the decision tree are
         EXACT (squared-distance ranking / threshold comparisons -- no transcendental), so they are allowed i32
         OR f32. A LINEAR SVM is exact dot products (no transcendental) so it too may be i32 or f32; only the RBF
         SVM is constrained to f32."""
    errs: list[str] = []
    if kind not in _KINDS:
        errs.append(f"classical: kind {kind!r} is not a known predictor {_KINDS}")
    for nm, v in (("n_feat", n_feat), ("n_param", n_param)):
        if v < 1:
            errs.append(f"classical: {nm} must be >= 1; got {v}")
    for nm, dt in (("in", in_dtype), ("out", out_dtype)):
        if dt not in _DTYPES:
            errs.append(f"classical: {nm} dtype {dt!r} is not a known dtype {_DTYPES}")
    if in_dtype != out_dtype:
        errs.append(f"classical: dtype not preserved (in {in_dtype!r} != out {out_dtype!r})")
    # the quarantine dtype rule: a transcendental predictor (RBF-SVM exp / Gaussian-NB log) needs f32.
    uses_transcendental = (kind == "nb") or (kind == "svm" and svm_kernel == "rbf")
    if uses_transcendental and in_dtype != "f32":
        which = "Gaussian-NB (log)" if kind == "nb" else "RBF-SVM (exp)"
        errs.append(
            f"classical: {which} uses a transcendental (the libm edge returns float), so it needs f32, "
            f"got {in_dtype!r}"
        )
    if kind == "svm" and svm_kernel not in ("linear", "rbf"):
        errs.append(f"classical: svm kernel {svm_kernel!r} is not a known kernel ('linear'/'rbf')")
    return errs
