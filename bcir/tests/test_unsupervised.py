"""E6 (ML-breadth) -- UNSUPERVISED LEARNING + THE DATA PIPELINE (K-means / scalers / cross-validation /
autoencoder / embedding lookup): the correctness gate for the SIXTH ML-breadth slice after E1 (OLS), E2 (PCA),
E3 (the Transformer block), E4 (RNN/LSTM/GRU), E5 (classical-ML predict).

THE FIT vs PREDICT/TRANSFORM split (the research finding E7 cites). Unsupervised learning + preprocessing split
sharply: FIT/TRAIN (Lloyd's iteration, a scaler's mean/std estimate) is bounded-iterative/statistical; the
PREDICT/TRANSFORM step over a BAKED model (nearest-centroid assign, the exact scaler transform, the autoencoder
forward, the embedding gather) is a deterministic fixed-shape kernel = the G5 baked-weights pattern. E6 REUSES
BCIR's existing pieces: K-means assign reuses ``classical.squared_distance``; the autoencoder reuses
``matmul.matmul_reference`` + the G1 ``activation`` references + the M1 ``losses.mse_value``; the CV folds reuse
the M3 ``training._lcg_permutation``. The only new transcendental is StandardScaler's FIT-time ``sqrt`` (baked).

Covers: K-means converges on a 2-cluster toy set with MONOTONE-NON-INCREASING inertia + a known final
assignment, deterministic empty-cluster handling, and the exact nearest-centroid kernel (tie-break lowest
index); StandardScaler's standardized training data has per-feature mean ~ 0 / std ~ 1 (the independent
recompute) and the transform is exact division of the baked mean/std; MinMaxScaler maps training data into
[0,1]; the K-fold split is a genuine partition (disjoint + cover, balanced, reproducible); the tied-identity
autoencoder reconstructs the input (error ~ 0) while a real bottleneck has error > 0; the embedding lookup
equals the direct row slice and an out-of-range id raises; the Q8 bridges equal the kernel of the round-tripped
input; the emitted K-means-assign C kernel (EXACT, no libm) compiles + runs + matches the reference
integer-exactly; the K-means kernel needs NO ``-lm`` (no linkflags change); ``check_unsupervised`` accepts valid
+ rejects malformed claims (NOT a new global R-law); and the module imports no verifier / emits no Diagnostic
(the two-truth quarantine, AST-inspected). All deterministic + pure-Python on the oracle rail.
"""

import ast
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.toolchain import host_link_args

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.precision import quantization_error_bound
from bcir.kbcir.unsupervised import (EmbeddingTable, MinMaxScaler, StandardScaler,
                                     autoencoder_forward, autoencoder_via_bridge,
                                     check_unsupervised, embedding_lookup, k_fold_indices,
                                     k_fold_is_partition, kmeans_assign, kmeans_assign_via_bridge,
                                     kmeans_fit, kmeans_inertia, minmax_fit, minmax_transform,
                                     reconstruction_error, standard_scaler_fit,
                                     standard_scaler_transform)
from bcir.lower.c_kernel import emit_kmeans_assign_c
from bcir.tests._convergence import assert_converged


# ============================================================================================================
# 1. K-MEANS -- the exact nearest-centroid assign kernel; the deterministic Lloyd fit; the monotone inertia.
# ============================================================================================================

def _two_cluster_toy():
    """A 2-D toy with two well-separated clusters: cluster A near (0, 0), cluster B near (10, 10). Six points,
    three per cluster, off any exact equidistant tie."""
    X = [0.0, 0.0,  1.0, 0.0,  0.0, 1.0,                       # cluster A (near origin)
         10.0, 10.0,  11.0, 10.0,  10.0, 11.0]                # cluster B (near (10,10))
    return X, 6, 2                                             # X, n_samples, n_feat


def test_kmeans_assign_is_exact_nearest_centroid():
    # two centroids at (0,0) and (10,10); a point near the origin assigns to cluster 0, near (10,10) to 1.
    centroids = [0.0, 0.0,  10.0, 10.0]
    assert kmeans_assign(centroids, [0.2, 0.1], k=2, n_feat=2) == 0
    assert kmeans_assign(centroids, [9.8, 10.1], k=2, n_feat=2) == 1
    # squared distance is exact: (3,4) from origin is 25, from (10,10) is bigger -> cluster 0.
    assert kmeans_assign(centroids, [3.0, 4.0], k=2, n_feat=2) == 0


def test_kmeans_assign_tie_break_is_lowest_index():
    # a point exactly equidistant from two centroids -> the LOWEST centroid index wins (strict < keeps index 0).
    centroids = [-1.0,  1.0]                                   # 1-D centroids at -1 and +1
    assert kmeans_assign(centroids, [0.0], k=2, n_feat=1) == 0  # equidistant (d=1 each) -> lower index 0


def test_kmeans_fit_converges_on_the_toy_set():
    X, n, nf = _two_cluster_toy()
    # seed centroids with one point from each true cluster (indices 0 and 3) -> Lloyd recovers the two clusters.
    centroids, labels = kmeans_fit(X, n, nf, k=2, init_indices=[0, 3], n_iter=10)
    # the three origin-cluster points share one label, the three (10,10) points share the other.
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]
    # the centroids are the cluster means: A ~ ((0+1+0)/3, (0+0+1)/3), B ~ ((10+11+10)/3, (10+10+11)/3).
    # find which returned centroid is the origin cluster (label of point 0).
    ca = labels[0]
    base = ca * nf
    assert abs(centroids[base] - 1.0 / 3.0) < 1e-9 and abs(centroids[base + 1] - 1.0 / 3.0) < 1e-9


def test_kmeans_inertia_is_monotone_non_increasing_over_iterations():
    # the defining K-means property: inertia never RISES as Lloyd iterates (a genuine independent property).
    X, n, nf = _two_cluster_toy()
    prev = math.inf
    for it in range(0, 8):
        centroids, labels = kmeans_fit(X, n, nf, k=2, init_indices=[0, 5], n_iter=it)
        inertia = kmeans_inertia(X, n, nf, centroids, labels)
        assert inertia <= prev + 1e-12, (it, inertia, prev)    # non-increasing (small fp slack)
        prev = inertia
    # converged inertia is small (the two tight clusters have low within-cluster scatter).
    assert prev < 5.0


def test_kmeans_fit_handles_an_empty_cluster_deterministically():
    # 3 points, k=3 (k <= n_samples), but SEED two centroids at the SAME point (init_indices [0, 0, 2] -> the
    # first two centroids are both row 0). The deterministic lowest-index tie-break routes every equidistant
    # point to centroid 0, so centroid 1 wins NOTHING -> a genuine EMPTY cluster. It must keep its previous
    # centroid (no 0/0 NaN, no crash) and leave the fit fully deterministic.
    X = [0.0, 0.0,  0.1, 0.0,  10.0, 10.0]
    centroids, labels = kmeans_fit(X, 3, 2, k=3, init_indices=[0, 0, 2], n_iter=5)
    assert len(centroids) == 3 * 2 and len(labels) == 3
    # the empty cluster kept its previous centroid -> every component is finite (no 0/0 NaN, no crash).
    assert all(math.isfinite(c) for c in centroids)
    # n_iter=1 pins the documented behaviour DIRECTLY: after one assign->recompute, cluster 1 (the duplicate
    # seed at (0,0)) wins no point, so it KEEPS its previous centroid (still (0,0)) rather than NaN.
    c1, _ = kmeans_fit(X, 3, 2, k=3, init_indices=[0, 0, 2], n_iter=1)
    assert c1[2:4] == [0.0, 0.0]                               # centroid 1 unchanged from its seed (empty kept)
    # re-running gives the identical result (fully deterministic, no randomness).
    c2, l2 = kmeans_fit(X, 3, 2, k=3, init_indices=[0, 0, 2], n_iter=5)
    assert centroids == c2 and labels == l2


def test_kmeans_assign_bridge_is_assign_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    centroids = [0.0, 0.0,  10.0, 10.0]
    x = [0.234, 0.123]                                         # well inside cluster 0, off any tie
    dx = dequantize(quantize_per_group(x, 2, 8))
    assert kmeans_assign_via_bridge(centroids, x, 2, 2, 2, 8) == kmeans_assign(centroids, dx, 2, 2)


def test_kmeans_validates():
    for fn, args in [(kmeans_assign, ([0.0, 0.0], [0.0], 1, 2)),         # point length mismatch
                     (kmeans_assign, ([0.0, 0.0], [0.0, 0.0], 0, 2)),    # k < 1
                     (kmeans_fit, ([0.0, 0.0], 1, 2, 2, [0], 1)),        # k > n_samples
                     (kmeans_fit, ([0.0], 1, 1, 1, [5], 1))]:            # init index out of range
        try:
            fn(*args)
            assert False, f"{fn.__name__}{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 2. SCALERS -- StandardScaler (mean ~ 0 / std ~ 1 independent recompute); MinMaxScaler ([0,1]); exact transform.
# ============================================================================================================

def test_standard_scaler_standardizes_training_data_to_mean0_std1():
    # the INDEPENDENT verifier: transform the training data with the fitted scaler, then recompute the
    # per-feature mean / std DIRECTLY -- they must be ~ 0 / ~ 1.
    rng = random.Random(0xE601)
    n, nf = 50, 3
    X = [rng.uniform(-5, 5) for _ in range(n * nf)]
    sc = standard_scaler_fit(X, n, nf)
    transformed = [standard_scaler_transform(X[i * nf:(i + 1) * nf], sc.mean, sc.std) for i in range(n)]
    for j in range(nf):
        col = [transformed[i][j] for i in range(n)]
        mean = sum(col) / n
        var = sum((v - mean) ** 2 for v in col) / n            # population variance, matching the fit
        assert abs(mean) < 1e-9, (j, mean)                     # standardized mean ~ 0
        assert abs(math.sqrt(var) - 1.0) < 1e-9, (j, math.sqrt(var))  # standardized std ~ 1


def test_standard_scaler_transform_is_exact_division_of_baked_mean_std():
    # the baked mean/std make the transform exact (x-mean)/std -- a hand-checkable value.
    sc = StandardScaler(mean=(2.0,), std=(4.0,))
    assert standard_scaler_transform([10.0], sc.mean, sc.std) == [(10.0 - 2.0) / 4.0]   # exactly 2.0


def test_standard_scaler_rejects_a_constant_feature():
    # a constant feature has population variance 0 -> std 0 -> the dataclass rejects (division undefined).
    try:
        standard_scaler_fit([3.0, 3.0, 3.0], 3, 1)             # all identical -> std 0
        assert False, "a constant feature should raise"
    except ValueError:
        pass


def test_minmax_scaler_maps_training_data_into_unit_interval():
    # the INDEPENDENT verifier: every transformed training value lies in [0, 1], and the per-feature min/max
    # map to exactly 0 / 1.
    rng = random.Random(0xE602)
    n, nf = 40, 2
    X = [rng.uniform(-3, 7) for _ in range(n * nf)]
    sc = minmax_fit(X, n, nf)
    transformed = [minmax_transform(X[i * nf:(i + 1) * nf], sc.mn, sc.mx) for i in range(n)]
    for j in range(nf):
        col = [transformed[i][j] for i in range(n)]
        assert all(-1e-12 <= v <= 1.0 + 1e-12 for v in col), (j, min(col), max(col))   # in [0,1]
        assert abs(min(col) - 0.0) < 1e-12 and abs(max(col) - 1.0) < 1e-12             # min->0, max->1


def test_minmax_transform_is_exact():
    sc = MinMaxScaler(mn=(2.0,), mx=(6.0,))
    assert minmax_transform([4.0], sc.mn, sc.mx) == [(4.0 - 2.0) / (6.0 - 2.0)]        # exactly 0.5


def test_minmax_rejects_a_constant_feature():
    try:
        minmax_fit([5.0, 5.0], 2, 1)                            # mx == mn -> reject
        assert False, "a constant feature should raise"
    except ValueError:
        pass


# ============================================================================================================
# 3. CROSS-VALIDATION FOLDS -- the partition (disjoint + cover), balance, and reproducibility.
# ============================================================================================================

def test_k_fold_is_a_genuine_partition_disjoint_and_covering():
    # the INDEPENDENT verifier: the folds are pairwise disjoint and their union is exactly {0,...,n-1}.
    for n, kf in [(10, 5), (13, 4), (7, 7), (100, 6)]:
        folds = k_fold_indices(n, kf, seed=0xE603)
        assert len(folds) == kf
        assert k_fold_is_partition(folds, n), (n, kf, folds)   # disjoint + cover
        # also recompute the union/total directly as a second independent cross-check.
        all_idx = [i for fold in folds for i in fold]
        assert sorted(all_idx) == list(range(n))               # genuine partition (each index exactly once)


def test_k_fold_sizes_are_balanced():
    # fold sizes differ by at most 1 (the standard balanced K-fold property).
    folds = k_fold_indices(13, 4, seed=7)                       # 13 = 4 + 3 + 3 + 3
    sizes = sorted(len(f) for f in folds)
    assert max(sizes) - min(sizes) <= 1
    assert sizes == [3, 3, 3, 4]


def test_k_fold_is_deterministic_for_a_seed():
    # same (n, n_folds, seed) -> identical folds (reuses the M3 LCG, no randomness).
    a = k_fold_indices(20, 5, seed=123)
    b = k_fold_indices(20, 5, seed=123)
    assert a == b
    # a different seed generally gives a different partition (the shuffle differs).
    c = k_fold_indices(20, 5, seed=124)
    assert a != c


def test_k_fold_validates():
    for args in [(1, 2, 0),          # n_folds > n_samples
                 (10, 1, 0),         # n_folds < 2
                 (0, 2, 0)]:         # n_samples < 1
        try:
            k_fold_indices(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 4. AUTOENCODER -- the composition; tied-identity reconstruction (~ 0) vs a real bottleneck (> 0).
# ============================================================================================================

def _identity_flat(n):
    """The n x n identity matrix as a flat row-major list."""
    return [1.0 if i == j else 0.0 for i in range(n) for j in range(n)]


def test_autoencoder_tied_identity_reconstructs_the_input():
    # the INDEPENDENT verifier: W_enc = W_dec = I (n_in == n_hidden), zero bias, relu on a POSITIVE input ->
    # the reconstruction EQUALS the input (relu is identity on positives), so the error is ~ 0.
    n = 4
    x = [0.5, 1.5, 2.5, 3.5]                                   # all positive -> relu is identity
    W = _identity_flat(n)
    recon = autoencoder_forward(x, n, n, W, [0.0] * n, W, [0.0] * n, activation="relu")
    assert all(abs(r - xi) < 1e-12 for r, xi in zip(recon, x)), (recon, x)
    assert reconstruction_error(x, recon) < 1e-12              # error ~ 0


def test_autoencoder_nontrivial_has_positive_error():
    # a real autoencoder with a BOTTLENECK (n_hidden < n_in) and arbitrary weights does NOT reconstruct exactly.
    rng = random.Random(0xE604)
    n_in, n_hidden = 5, 2
    x = [rng.uniform(0.1, 2.0) for _ in range(n_in)]
    W_enc = [rng.uniform(-1, 1) for _ in range(n_in * n_hidden)]
    W_dec = [rng.uniform(-1, 1) for _ in range(n_hidden * n_in)]
    recon = autoencoder_forward(x, n_in, n_hidden, W_enc, [0.0] * n_hidden, W_dec, [0.0] * n_in, "relu")
    assert reconstruction_error(x, recon) > 0.0                # a non-trivial AE has reconstruction error


def test_autoencoder_reuses_matmul_and_activation():
    # cross-check the forward against a hand-built matmul + relu (proves it really composes the reused ops).
    from bcir.kbcir.activation import relu_reference
    from bcir.kbcir.matmul import matmul_reference
    x = [1.0, -2.0]
    n_in, n_hidden = 2, 3
    W_enc = [0.5, 0.0, 1.0,  -1.0, 2.0, 0.5]                   # 2 x 3
    b_enc = [0.1, 0.2, 0.3]
    W_dec = [1.0, 0.0,  0.0, 1.0,  0.5, 0.5]                   # 3 x 2
    b_dec = [0.0, 0.0]
    recon = autoencoder_forward(x, n_in, n_hidden, W_enc, b_enc, W_dec, b_dec, "relu")
    pre = matmul_reference(x, W_enc, 1, n_hidden, n_in)
    pre = [pre[j] + b_enc[j] for j in range(n_hidden)]
    h = relu_reference(pre)
    expect = matmul_reference(h, W_dec, 1, n_in, n_hidden)
    expect = [expect[j] + b_dec[j] for j in range(n_in)]
    assert all(abs(r - e) < 1e-12 for r, e in zip(recon, expect)), (recon, expect)


def test_autoencoder_bridge_is_forward_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    n = 3
    x = [0.31, 0.62, 0.93]
    W = _identity_flat(n)
    dx = dequantize(quantize_per_group(x, n, 8))
    bridged = autoencoder_via_bridge(x, n, n, W, [0.0] * n, W, [0.0] * n, n, 8, "relu")
    direct = autoencoder_forward(dx, n, n, W, [0.0] * n, W, [0.0] * n, "relu")
    assert bridged == direct


def test_autoencoder_validates():
    n = 3
    W = _identity_flat(n)
    for args in [([0.0, 0.0], n, n, W, [0.0] * n, W, [0.0] * n),          # x length mismatch
                 ([0.0] * n, n, n, W[:-1], [0.0] * n, W, [0.0] * n)]:     # W_enc length mismatch
        try:
            autoencoder_forward(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 5. EMBEDDING LOOKUP -- the row gather equals the direct slice; an out-of-range id raises.
# ============================================================================================================

def _toy_embedding():
    # a 4 x 2 table: row v is (v*10, v*10+1).
    return EmbeddingTable(table=(0.0, 1.0,  10.0, 11.0,  20.0, 21.0,  30.0, 31.0), n_vocab=4, dim=2)


def test_embedding_lookup_equals_the_direct_row_slice():
    # the INDEPENDENT verifier: the gather equals the direct row slices concatenated.
    t = _toy_embedding()
    ids = [2, 0, 3, 2]
    got = embedding_lookup(t, ids, dim=2)
    expect = [v for vid in ids for v in t.row(vid)]            # the direct-slice path
    assert got == expect
    assert got == [20.0, 21.0,  0.0, 1.0,  30.0, 31.0,  20.0, 21.0]   # hand value


def test_embedding_lookup_out_of_range_id_raises():
    t = _toy_embedding()
    for bad in ([4], [-1], [0, 99]):
        try:
            embedding_lookup(t, bad, dim=2)
            assert False, f"id {bad} should raise"
        except ValueError:
            pass


def test_embedding_validates():
    for args in [((0.0, 1.0), 0, 2),          # n_vocab < 1
                 ((0.0, 1.0), 1, 0),          # dim < 1
                 ((0.0, 1.0, 2.0), 2, 2)]:    # table length mismatch
        try:
            EmbeddingTable(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass
    # dim mismatch at lookup.
    t = _toy_embedding()
    try:
        embedding_lookup(t, [0], dim=3)
        assert False
    except ValueError:
        pass


# ============================================================================================================
# 6. The R17 bridge bound is the input round-trip (1 ULP), as every wrap.
# ============================================================================================================

def test_r17_bound_is_the_input_roundtrip():
    assert quantization_error_bound() == 1                      # 1 ULP, as E1-E5


# ============================================================================================================
# 7. Op-level well-formedness (check_unsupervised; NOT a new global R-law).
# ============================================================================================================

def test_check_unsupervised_accepts_well_formed_claims():
    assert check_unsupervised("kmeans", n_feat=2, n_unit=3, in_dtype="f32", out_dtype="f32") == []
    assert check_unsupervised("kmeans", n_feat=2, n_unit=3, in_dtype="i32", out_dtype="i32") == []   # assign exact -> i32 ok
    assert check_unsupervised("scaler", n_feat=4, n_unit=4, in_dtype="i32", out_dtype="i32") == []   # transform exact -> i32 ok
    assert check_unsupervised("embedding", n_feat=8, n_unit=100, in_dtype="i32", out_dtype="i32") == []  # gather exact
    assert check_unsupervised("autoencoder", n_feat=5, n_unit=2, in_dtype="i32", out_dtype="i32",
                              activation="relu") == []                                                # relu exact -> i32 ok
    assert check_unsupervised("autoencoder", n_feat=5, n_unit=2, in_dtype="f32", out_dtype="f32",
                              activation="sigmoid") == []


def test_check_unsupervised_rejects_malformed_claims():
    assert any("not a known organ" in e
               for e in check_unsupervised("foo", n_feat=2, n_unit=2, in_dtype="f32", out_dtype="f32"))
    assert any(">= 1" in e for e in check_unsupervised("kmeans", n_feat=0, n_unit=2, in_dtype="f32", out_dtype="f32"))
    assert any(">= 1" in e for e in check_unsupervised("kmeans", n_feat=2, n_unit=0, in_dtype="f32", out_dtype="f32"))
    assert any("not preserved" in e
               for e in check_unsupervised("scaler", n_feat=2, n_unit=2, in_dtype="f32", out_dtype="i32"))
    assert any("not a known dtype" in e
               for e in check_unsupervised("kmeans", n_feat=2, n_unit=2, in_dtype="f64", out_dtype="f64"))
    # the quarantine dtype rule: a transcendental-activation autoencoder needs f32.
    assert any("needs f32" in e for e in check_unsupervised("autoencoder", n_feat=2, n_unit=2, in_dtype="i32",
                                                            out_dtype="i32", activation="tanh"))
    # an unknown autoencoder activation is rejected.
    assert any("not a known activation" in e
               for e in check_unsupervised("autoencoder", n_feat=2, n_unit=2, in_dtype="f32",
                                           out_dtype="f32", activation="swish"))


# ============================================================================================================
# 8. The emitted K-means-assign C kernel compiles + reproduces the reference (self-skip when no toolchain).
# ============================================================================================================

def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _farr(v):
    return "{" + ", ".join(f"{f:.7f}f" for f in v) + "}"


def _compile_run(kernel, main, libm):
    cc = _cc()
    with tempfile.TemporaryDirectory() as dd:
        src = os.path.join(dd, "k.c")
        open(src, "w").write(kernel + main)
        exe = os.path.join(dd, "k")
        cmd = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", src]
        if libm:
            cmd.append("-lm")
        cmd += ["-o", exe]
        bld = subprocess.run(host_link_args(cmd), capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [int(t) for t in out.stdout.split()]


def test_kmeans_assign_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return                                                  # quick tier hides the toolchain -> self-skip
    k, nf = 3, 2
    centroids = [0.0, 0.0,  10.0, 10.0,  -5.0, 5.0]
    queries = [[0.3, 0.1], [9.7, 10.2], [-4.8, 5.1], [3.0, 4.0]]   # all off exact ties
    kernel = emit_kmeans_assign_c(k, nf, "km_assign")
    decls = "".join(f"  float X{i}[{nf}]={_farr(q)};\n" for i, q in enumerate(queries))
    calls = "".join(f'  printf("%d\\n", km_assign(X{i}, C));\n' for i in range(len(queries)))
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float C[{k * nf}]={_farr(centroids)};\n"
            f"{decls}{calls}  return 0;\n}}\n")
    got = _compile_run(kernel, main, libm=False)                # the assign kernel is EXACT -- no libm needed
    ref = [kmeans_assign(centroids, q, k, nf) for q in queries]
    assert got == ref, (got, ref)                               # INTEGER-exact (the same argmin cluster id)


def test_kmeans_c_emit_routes_and_validates():
    # the K-means assign kernel is EXACT (no math.h, no libm) -- the EXACT half of the Area-B pattern.
    src = emit_kmeans_assign_c(3, 2, "ka")
    assert "math.h" not in src and "expf" not in src and "sqrtf" not in src   # EXACT -- no transcendental
    assert "c.call.libm:" not in src                                          # mints no libm edge
    assert "int ka(const float *restrict x," in src                          # returns an int cluster id
    for bad in [(0, 2), (2, 0)]:
        try:
            emit_kmeans_assign_c(*bad)
            assert False, f"emit_kmeans_assign_c{bad} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 9. The link-flag rule: the K-means kernel needs NO -lm (it is exact); the REUSED edges are unchanged.
# ============================================================================================================

def test_kmeans_kernel_needs_no_libm_and_reused_rules_unchanged():
    # the K-means assign kernel is EXACT -- it mints no libm edge, so it needs no -lm. No linkflags change.
    src = emit_kmeans_assign_c(2, 2, "ka")
    assert "-lm" not in src and "math.h" not in src
    # no regression on the existing libm/library rules (E1-E5 + libc + unknown).
    assert library_for_callee("sqrtf") == "-lm"                # StandardScaler's fit-time sqrt would map here
    assert library_for_callee("expf") == "-lm"                 # an autoencoder sigmoid/tanh activation edge
    assert library_for_callee("tanhf") == "-lm"
    assert library_for_callee("LAPACKE_ssyev") == "-llapack"   # E2 PCA still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"      # B5 BLAS still maps
    assert library_for_callee("free") == NO_FLAG               # libc-implicit, known
    assert library_for_callee("totally_unknown_fn") is None    # unknown-callee policy unchanged


# ============================================================================================================
# the two-truth quarantine: the unsupervised module imports no verifier / emits no Diagnostic.
# ============================================================================================================

def test_unsupervised_touches_no_verifier_and_emits_no_diagnostic():
    # the unsupervised organs are cost-side / oracle quantities (off the legality path), never an R-law verdict.
    # AST-inspect the module: no verify / Diagnostic / Graded names bound, no verifier module imported (exactly
    # as test_ols / test_pca / test_recurrent / test_classical).
    import bcir.kbcir.unsupervised as un_mod
    bound = set(vars(un_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(un_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain ints / plain floats / plain lists (no graded/confidence wrapper).
    assert isinstance(kmeans_assign([0.0, 0.0, 10.0, 10.0], [0.1, 0.0], 2, 2), int)
    assert isinstance(kmeans_inertia([0.0, 0.0], 1, 2, [0.0, 0.0], [0]), float)
    assert isinstance(standard_scaler_transform([1.0], (0.0,), (1.0,)), list)
    assert isinstance(k_fold_indices(6, 3, 0), list)
    assert isinstance(embedding_lookup(_toy_embedding(), [0], 2), list)


def test_autoencoder_training_drives_reconstruction_loss_down():
    # E6 CONVERGENCE (H4) -- the STRONGEST demo: a relu autoencoder is fully CLOSED-SET (matmul + bias + relu ->
    # add / mul / select), so EVERY weight (encoder + decoder) trains end-to-end through the Tape autodiff, not
    # just a readout. Train a 4->2->4 autoencoder by Adam to reconstruct low-rank positive data; the
    # reconstruction MSE must DROP substantially (the bottleneck can capture the 2-D structure).
    from bcir.kbcir.training import Dataset, train

    rng = random.Random(0xE6C0)
    n_in, n_hidden = 4, 2
    u = [1.0, 0.5, 0.2, 0.8]
    v = [0.3, 1.0, 0.9, 0.1]
    inputs = []
    for _ in range(12):                                          # each input lies near a 2-D subspace...
        a, b = rng.uniform(0.2, 1.2), rng.uniform(0.2, 1.2)
        inputs.append([a * u[i] + b * v[i] + 0.3 for i in range(n_in)])   # ...+0.3 keeps it positive for relu
    # flatten to (input, output-dim) scalar-regression examples: target is the reconstructed component.
    rows, ys = [], []
    for x in inputs:
        for d in range(n_in):
            rows.append(tuple(x) + (float(d),))
            ys.append(x[d])

    we = [f"We{i}_{j}" for i in range(n_in) for j in range(n_hidden)]   # encoder W (n_in x n_hidden)
    be = [f"be{j}" for j in range(n_hidden)]
    wd = [f"Wd{j}_{i}" for j in range(n_hidden) for i in range(n_in)]   # decoder W (n_hidden x n_in)
    bd = [f"bd{i}" for i in range(n_in)]
    names = we + be + wd + bd

    def model(tape, param_names, X):
        var = {nm: tape.var(nm) for nm in param_names}
        zero = tape.const(0.0)
        out = []
        for row in X:
            x = row[:n_in]
            d = int(row[n_in])
            h = []
            for j in range(n_hidden):                            # encode: h = relu(x @ W_enc + b_enc)
                z = var[f"be{j}"]
                for i in range(n_in):
                    z = tape.add(z, tape.mul(var[f"We{i}_{j}"], tape.const(x[i])))
                h.append(tape.select(z, z, zero))                # relu (closed-set)
            z = var[f"bd{d}"]                                    # decode the d-th component: recon[d]
            for j in range(n_hidden):
                z = tape.add(z, tape.mul(var[f"Wd{j}_{d}"], h[j]))
            out.append(z)
        return out

    # POSITIVE init: the data is positive (the +0.3 offset), so a positive encoder init keeps every relu unit
    # active -- a symmetric +/-0.5 init strands a dead unit and plateaus at ~1.5e-2 instead of converging.
    p0 = [rng.uniform(0.1, 0.6) for _ in names]
    res = train(model, p0, Dataset(tuple(rows), tuple(ys)), loss="mse", optimizer="adam",
                epochs=400, lr=0.05, seed=1, param_names=names)
    # the shared convergence gate (the audit finding: the old gate required only a 2x drop while the measured
    # run achieved 70x -- a lax gate is not a convergence demo). The data is exactly rank-2 + a positive
    # offset, so near-ZERO reconstruction is achievable and is now asserted ABSOLUTELY (measured ~1e-8).
    assert_converged(res.train_loss, final_below=1e-6, max_ratio=0.01, name="E6 autoencoder")


def test_kmeans_reaches_a_fixpoint():
    # E6 K-means FIXPOINT (the audit gap): convergence was previously INFERRED from the inertia plateau;
    # assert the fixpoint itself -- one more Lloyd iteration changes NEITHER the centroids NOR the labels.
    X, n, nf = _two_cluster_toy()
    c8, l8 = kmeans_fit(X, n, nf, k=2, init_indices=[0, 5], n_iter=8)
    c9, l9 = kmeans_fit(X, n, nf, k=2, init_indices=[0, 5], n_iter=9)
    assert l8 == l9, (l8, l9)
    assert all(abs(a - b) < 1e-12 for a, b in zip(c8, c9)), (c8, c9)


def test_kmeans_inertia_strictly_decreases_before_convergence():
    # Strengthen the K-means convergence evidence: on well-separated data the inertia must STRICTLY drop over the
    # first Lloyd iterations (not merely non-increase) before it settles -- proof of a real optimization step,
    # not a no-op. (Complements test_kmeans_inertia_is_monotone_non_increasing_over_iterations.)
    X, n, nf = _two_cluster_toy()
    prev, drops = None, 0
    for it in range(0, 4):
        centroids, labels = kmeans_fit(X, n, nf, k=2, init_indices=[0, 5], n_iter=it)
        inertia = kmeans_inertia(X, n, nf, centroids, labels)
        if prev is not None and inertia < prev - 1e-9:
            drops += 1
        prev = inertia
    assert drops >= 1, "inertia never strictly dropped over early Lloyd iterations"


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    tests = sorted(n for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n)))
    passed = failed = 0
    for name in tests:
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
