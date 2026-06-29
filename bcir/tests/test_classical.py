"""E5 (ML-breadth) -- CLASSICAL-ML PREDICT-PATH wraps (KNN / decision tree / SVM / Naive Bayes): the correctness
gate for the FIFTH ML-breadth slice after E1 (OLS), E2 (PCA), E3 (the Transformer block), E4 (RNN/LSTM/GRU).

THE TRAIN vs PREDICT split (the research finding E7 cites). Classical ML's non-trivial TRAINING (tree induction,
the SVM QP solve, NB fitting) is iterative/combinatorial -- a poor fit for BCIR's fixed-shape claim model
(library/Python). PREDICT over a BAKED model is the opposite: a deterministic, fixed-shape kernel = the G5
baked-weights pattern. E5 is the PREDICT path. Each predictor has a plain-float reference (the source of truth),
an INDEPENDENT verifier (recomputed by a fully independent path), and a ``*_via_bridge`` Q8 round-trip of the
FEATURE vector (the R17 input bound).

Covers: KNN classifies a known query + regresses the neighbour mean, with the independent O(n^2)-selection
verifier agreeing on the neighbour set; the decision tree returns a known leaf, with the recursive verifier +
the structural leaf-reachability check; the linear SVM's sign is correct on a separable toy set + the RBF
decision matches an independent kernel-sum, both routed through (linear: exact) / (RBF: the libm expf edge);
Gaussian-NB picks the right class on well-separated Gaussians with the baked log(2*pi*var) normaliser, vs an
independent full-log-likelihood recompute; the Q8 bridge equals the predict of the round-tripped feature; the
emitted RBF-SVM (transcendental, expf -> -lm) + decision-tree (exact, no libm) C kernels compile + run + match;
the expf/logf -> -lm link rule is unchanged (NO linkflags change); check_classical accepts valid + rejects
malformed claims (NOT a new global R-law); and the module imports no verifier / emits no Diagnostic (the
two-truth quarantine, AST-inspected). All deterministic + pure-Python on the oracle rail.
"""

import ast
import math
import os
import random
import shutil
import subprocess
import tempfile

from bcir.frontends.cfront.linkflags import NO_FLAG, library_for_callee
from bcir.kbcir.classical import (GaussianNbModel, KnnModel, SvmModel, TreeModel,
                                  check_classical, gaussian_nb_log_posterior,
                                  gaussian_nb_log_posterior_independent, knn_classify,
                                  knn_classify_via_bridge, knn_neighbours,
                                  knn_neighbours_independent, knn_regress, nb_predict,
                                  nb_predict_via_bridge, squared_distance, svm_decision_linear,
                                  svm_decision_linear_independent, svm_decision_rbf,
                                  svm_decision_rbf_independent, svm_predict,
                                  svm_predict_linear_via_bridge, tree_leaves_reachable,
                                  tree_predict, tree_predict_recursive, tree_predict_via_bridge)
from bcir.kbcir.precision import quantization_error_bound
from bcir.lower.c_kernel import emit_svm_rbf_predict_c, emit_tree_predict_c


# ============================================================================================================
# 1. KNN PREDICT -- squared-distance ranking; majority vote / mean; the independent neighbour-set verifier.
# ============================================================================================================

def test_squared_distance_is_exact_and_monotone_rank():
    # squared distance is exact arithmetic; ranking on it == ranking on distance (sqrt monotone) -- no sqrt.
    assert squared_distance([0.0, 0.0], [3.0, 4.0]) == 25.0      # exact (3^2 + 4^2)
    assert squared_distance([1.0], [1.0]) == 0.0
    # ranking equivalence: d2 order == d order on a few points.
    pts = [[2.0, 0.0], [0.0, 1.0], [3.0, 4.0]]
    q = [0.0, 0.0]
    d2 = [squared_distance(q, p) for p in pts]
    d = [math.sqrt(v) for v in d2]
    assert sorted(range(3), key=lambda i: d2[i]) == sorted(range(3), key=lambda i: d[i])


def test_knn_classifies_a_known_query():
    # Two well-separated clusters: class 0 near the origin, class 1 near (10,10). A query at (0.2,0.1) is class 0.
    X = [0.0, 0.0,  1.0, 0.0,  0.0, 1.0,   10.0, 10.0,  11.0, 10.0,  10.0, 11.0]
    y = [0, 0, 0, 1, 1, 1]
    m = KnnModel(tuple(X), tuple(float(v) for v in y), n_feat=2)
    assert knn_classify(m, [0.2, 0.1], k=3) == 0
    assert knn_classify(m, [10.2, 9.9], k=3) == 1
    # k=1 nearest is the closest single point.
    assert knn_classify(m, [0.9, 0.05], k=1) == 0
    # the 3 nearest to (0.2,0.1) are the three class-0 points.
    nbrs = knn_neighbours(m, [0.2, 0.1], k=3)
    assert sorted(nbrs) == [0, 1, 2]


def test_knn_regresses_the_neighbour_mean():
    X = [0.0,  1.0,  2.0,  3.0,  10.0]
    y = [0.0, 10.0, 20.0, 30.0, 100.0]
    m = KnnModel(tuple(X), tuple(y), n_feat=1)
    # the 3 nearest to x=1.0 are samples 0,1,2 (values 0,10,20) -> mean 10.
    assert knn_regress(m, [1.0], k=3) == 10.0


def test_knn_independent_verifier_agrees_on_the_neighbour_set():
    # the O(n^2)-selection verifier must return the SAME neighbour set + order as the sort-based path.
    rng = random.Random(0xE501)
    nf, nt = 4, 30
    X = [rng.uniform(-5, 5) for _ in range(nt * nf)]
    y = [rng.randint(0, 2) for _ in range(nt)]
    m = KnnModel(tuple(X), tuple(float(v) for v in y), n_feat=nf)
    for _ in range(20):
        q = [rng.uniform(-5, 5) for _ in range(nf)]
        for k in (1, 3, 5, 7):
            a = knn_neighbours(m, q, k)
            b = knn_neighbours_independent(m, q, k)
            assert a == b, (k, a, b)                            # same set AND same deterministic order


def test_knn_deterministic_tie_break_lowest_index_then_lowest_class():
    # two training points equidistant from the query -> the lower index wins the neighbour slot.
    X = [-1.0,  1.0]            # x=0 query is equidistant (d=1) from both
    y = [5, 2]
    m = KnnModel(tuple(X), tuple(float(v) for v in y), n_feat=1)
    assert knn_neighbours(m, [0.0], k=1) == [0]                 # lowest index on a distance tie
    # class-vote tie (k=2, one of each class) -> lowest class id (2 < 5).
    assert knn_classify(m, [0.0], k=2) == 2


def test_knn_bridge_is_classify_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    rng = random.Random(0xE502)
    nf, nt = 3, 20
    X = [rng.uniform(-1, 1) for _ in range(nt * nf)]
    y = [rng.randint(0, 1) for _ in range(nt)]
    m = KnnModel(tuple(X), tuple(float(v) for v in y), n_feat=nf)
    q = [rng.uniform(-1, 1) for _ in range(nf)]
    dq = dequantize(quantize_per_group(q, nf, 8))
    assert knn_classify_via_bridge(m, q, 3, nf, 8) == knn_classify(m, dq, 3)


def test_knn_model_validates():
    for args in [((0.0,) * 4, (0.0, 0.0), 0),          # n_feat < 1
                 ((0.0,) * 3, (0.0, 0.0), 2),          # X_train length mismatch
                 ((), (), 2)]:                         # no training samples
        try:
            KnnModel(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 2. DECISION-TREE PREDICT -- exact threshold traversal; recursive + structural independent verifiers.
# ============================================================================================================

def _toy_tree():
    """A hand-built 5-node tree over 2 features:
        node 0: if x[0] <= 0.5 -> node 1 else node 2
        node 1: if x[1] <= 0.5 -> node 3 (leaf 10) else node 4 (leaf 20)
        node 2: leaf 30
    """
    return TreeModel(
        feature=(0, 1, -1, -1, -1),
        threshold=(0.5, 0.5, 0.0, 0.0, 0.0),
        left=(1, 3, 0, 0, 0),
        right=(2, 4, 0, 0, 0),
        leaf_value=(0.0, 0.0, 30.0, 10.0, 20.0),
        n_feat=2,
    )


def test_tree_returns_a_known_leaf():
    t = _toy_tree()
    assert tree_predict(t, [0.2, 0.2]) == 10.0                  # x0<=0.5, x1<=0.5 -> leaf 3
    assert tree_predict(t, [0.2, 0.9]) == 20.0                  # x0<=0.5, x1>0.5  -> leaf 4
    assert tree_predict(t, [0.9, 0.0]) == 30.0                  # x0>0.5           -> leaf 2
    # a stump (single leaf root) returns its value for any input.
    stump = TreeModel((-1,), (0.0,), (0,), (0,), (42.0,), 1)
    assert tree_predict(stump, [123.0]) == 42.0


def test_tree_recursive_verifier_matches_the_iterative_walk():
    t = _toy_tree()
    rng = random.Random(0xE503)
    for _ in range(200):
        x = [rng.uniform(-1, 2), rng.uniform(-1, 2)]
        assert tree_predict(t, x) == tree_predict_recursive(t, x)   # independent control flow, same leaf


def test_tree_structural_leaf_reachability():
    t = _toy_tree()
    leaves = tree_leaves_reachable(t)
    assert leaves == {2, 3, 4}                                  # exactly the three leaf node ids
    # every leaf the traversal can land on is in the reachable set (covered by sweeping inputs).
    rng = random.Random(0xE504)
    landed = set()
    for _ in range(500):
        x = [rng.uniform(-1, 2), rng.uniform(-1, 2)]
        # find which leaf node the prediction came from by matching leaf_value uniqueness.
        v = tree_predict(t, x)
        for n in leaves:
            if t.leaf_value[n] == v:
                landed.add(n)
    assert landed == leaves                                     # all three leaves are genuinely reachable


def test_tree_bridge_is_predict_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    t = _toy_tree()
    # keep the feature away from the exact thresholds so the round-trip never flips a branch.
    x = [0.123, 0.789]
    dx = dequantize(quantize_per_group(x, 2, 8))
    assert tree_predict_via_bridge(t, x, 2, 8) == tree_predict(t, dx)


def test_tree_model_validates():
    # child out of range.
    try:
        TreeModel((0, -1), (0.5, 0.0), (5, 0), (1, 0), (0.0, 9.0), 1)
        assert False
    except ValueError:
        pass
    # feature out of range.
    try:
        TreeModel((3, -1), (0.5, 0.0), (1, 0), (1, 0), (0.0, 9.0), 2)
        assert False
    except ValueError:
        pass
    # inconsistent array length.
    try:
        TreeModel((-1,), (0.0, 0.0), (0,), (0,), (1.0,), 1)
        assert False
    except ValueError:
        pass


# ============================================================================================================
# 3. SVM DECISION-FUNCTION PREDICT -- linear (exact) + RBF (the expf libm edge); independent kernel-sums.
# ============================================================================================================

def test_svm_linear_sign_is_correct_on_a_separable_toy_set():
    # A linearly-separable problem: SVs at (-1,0) [class -1] and (1,0) [class +1]; the decision boundary is x0=0.
    # alpha_y = (-c, +c) for some c>0 gives f(x) ~ proportional to x0 -> sign tracks the side of the boundary.
    m = SvmModel(SV=(-1.0, 0.0,  1.0, 0.0), alpha_y=(-0.5, 0.5), b=0.0, n_feat=2)
    # point on the + side (well away from the boundary x0=0).
    assert svm_predict(svm_decision_linear(m, [2.0, 0.3])) == 1
    # point on the - side.
    assert svm_predict(svm_decision_linear(m, [-2.0, -0.3])) == -1
    # the decision value of the linear kernel: f(x) = -0.5*(SV0.x) + 0.5*(SV1.x) + 0 = 0.5*(x0) - (-0.5)*(x0)=x0.
    assert abs(svm_decision_linear(m, [3.0, 0.0]) - 3.0) < 1e-12


def test_svm_linear_independent_verifier_matches():
    rng = random.Random(0xE505)
    nf, nsv = 5, 8
    m = SvmModel(tuple(rng.uniform(-2, 2) for _ in range(nsv * nf)),
                 tuple(rng.uniform(-1, 1) for _ in range(nsv)), rng.uniform(-1, 1), nf)
    for _ in range(30):
        x = [rng.uniform(-2, 2) for _ in range(nf)]
        a = svm_decision_linear(m, x)
        b = svm_decision_linear_independent(m, x)
        assert abs(a - b) < 1e-9, (a, b)


def test_svm_rbf_decision_matches_an_independent_kernel_sum():
    rng = random.Random(0xE506)
    nf, nsv = 4, 6
    m = SvmModel(tuple(rng.uniform(-1, 1) for _ in range(nsv * nf)),
                 tuple(rng.uniform(-1, 1) for _ in range(nsv)), rng.uniform(-1, 1), nf)
    gamma = 0.7
    for _ in range(30):
        x = [rng.uniform(-1, 1) for _ in range(nf)]
        a = svm_decision_rbf(m, x, gamma)
        b = svm_decision_rbf_independent(m, x, gamma)
        assert abs(a - b) < 1e-9, (a, b)


def test_svm_rbf_known_decision_value():
    # one SV at the origin, alpha_y=2, b=0.5, gamma=1: f(x) = 2*exp(-||x||^2) + 0.5.
    m = SvmModel(SV=(0.0, 0.0), alpha_y=(2.0,), b=0.5, n_feat=2)
    f = svm_decision_rbf(m, [1.0, 0.0], gamma=1.0)
    assert abs(f - (2.0 * math.exp(-1.0) + 0.5)) < 1e-12
    # at the SV itself the kernel is exp(0)=1 -> f = 2 + 0.5.
    assert abs(svm_decision_rbf(m, [0.0, 0.0], gamma=1.0) - 2.5) < 1e-12
    # far away: exp underflows toward 0 -> f -> b.
    assert abs(svm_decision_rbf(m, [100.0, 100.0], gamma=1.0) - 0.5) < 1e-12


def test_svm_predict_sign_convention():
    assert svm_predict(0.0) == 1                                # >= 0 -> +1
    assert svm_predict(1e-9) == 1
    assert svm_predict(-1e-9) == -1


def test_svm_rbf_rejects_nonpositive_gamma():
    m = SvmModel((0.0, 0.0), (1.0,), 0.0, 2)
    for g in (0.0, -1.0):
        try:
            svm_decision_rbf(m, [0.0, 0.0], g)
            assert False
        except ValueError:
            pass


def test_svm_linear_bridge_is_predict_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    m = SvmModel(SV=(-1.0, 0.0,  1.0, 0.0), alpha_y=(-0.5, 0.5), b=0.0, n_feat=2)
    x = [2.0, 0.3]                                              # well off the boundary -> sign stable
    dx = dequantize(quantize_per_group(x, 2, 8))
    assert svm_predict_linear_via_bridge(m, x, 2, 8) == svm_predict(svm_decision_linear(m, dx))


def test_svm_model_validates():
    for args in [((0.0, 0.0), (1.0,), 0.0, 0),         # n_feat < 1
                 ((0.0, 0.0, 0.0), (1.0,), 0.0, 2),    # SV length mismatch
                 ((), (), 0.0, 2)]:                    # no support vectors
        try:
            SvmModel(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 4. NAIVE-BAYES PREDICT -- Gaussian; baked log(2*pi*var); the independent full-log-likelihood verifier.
# ============================================================================================================

def _two_gaussian_nb():
    """Two well-separated 1-D Gaussian classes: class 0 ~ N(0, 1), class 1 ~ N(10, 1), equal priors."""
    return GaussianNbModel(
        log_prior=(math.log(0.5), math.log(0.5)),
        mean=(0.0, 10.0),
        var=(1.0, 1.0),
        n_feat=1,
    )


def test_nb_picks_the_right_class_on_well_separated_gaussians():
    m = _two_gaussian_nb()
    assert nb_predict(m, [0.3]) == 0                            # near class-0 mean
    assert nb_predict(m, [9.7]) == 1                            # near class-1 mean
    assert nb_predict(m, [4.9]) == 0                            # just below the midpoint (5) -> class 0
    assert nb_predict(m, [5.1]) == 1                            # just above the midpoint -> class 1


def test_nb_multifeature_well_separated():
    # 2 classes, 3 features; class 0 centred at origin, class 1 at (5,5,5).
    m = GaussianNbModel(log_prior=(math.log(0.5), math.log(0.5)),
                        mean=(0.0, 0.0, 0.0,  5.0, 5.0, 5.0),
                        var=(1.0, 1.0, 1.0,  1.0, 1.0, 1.0), n_feat=3)
    assert nb_predict(m, [0.1, -0.2, 0.3]) == 0
    assert nb_predict(m, [4.8, 5.1, 5.2]) == 1


def test_nb_independent_full_log_likelihood_verifier_matches():
    # the baked-normaliser log-posterior == the full inline Gaussian log-likelihood recompute.
    rng = random.Random(0xE507)
    nc, nf = 3, 4
    m = GaussianNbModel(tuple(math.log(1.0 / nc) for _ in range(nc)),
                        tuple(rng.uniform(-3, 3) for _ in range(nc * nf)),
                        tuple(rng.uniform(0.5, 2.0) for _ in range(nc * nf)), nf)
    for _ in range(30):
        x = [rng.uniform(-3, 3) for _ in range(nf)]
        a = gaussian_nb_log_posterior(m, x)
        b = gaussian_nb_log_posterior_independent(m, x)
        assert all(abs(ai - bi) < 1e-9 for ai, bi in zip(a, b)), (a, b)


def test_nb_baked_log_norm_is_data_independent():
    # the baked log(2*pi*var) depends only on the variances -- it's a constant the predict kernel never recomputes.
    m = _two_gaussian_nb()
    ln = m.baked_log_norm()
    assert len(ln) == m.n_class * m.n_feat
    assert all(abs(ln[i] - math.log(2.0 * math.pi * m.var[i])) < 1e-15 for i in range(len(ln)))


def test_nb_bridge_is_predict_of_the_roundtrip():
    from bcir.kbcir.quantize import dequantize, quantize_per_group
    m = _two_gaussian_nb()
    x = [0.3]                                                   # clearly class 0, off the decision midpoint
    dx = dequantize(quantize_per_group(x, 1, 8))
    assert nb_predict_via_bridge(m, x, 1, 8) == nb_predict(m, dx)


def test_nb_model_validates():
    for args in [((math.log(0.5),), (0.0,), (1.0,), 0),        # n_feat < 1
                 ((math.log(0.5),), (0.0, 0.0), (1.0,), 1),    # mean length mismatch
                 ((math.log(0.5),), (0.0,), (0.0,), 1)]:       # var <= 0
        try:
            GaussianNbModel(*args)
            assert False, f"{args} should raise"
        except ValueError:
            pass


# ============================================================================================================
# 5. The R17 bridge bound is the input round-trip (1 ULP), as every wrap.
# ============================================================================================================

def test_r17_bound_is_the_input_roundtrip():
    assert quantization_error_bound() == 1                      # 1 ULP, as E1-E4


# ============================================================================================================
# 6. Op-level well-formedness (check_classical; NOT a new global R-law).
# ============================================================================================================

def test_check_classical_accepts_well_formed_claims():
    assert check_classical("knn", n_feat=4, n_param=30, in_dtype="f32", out_dtype="f32") == []
    assert check_classical("tree", n_feat=2, n_param=5, in_dtype="i32", out_dtype="i32") == []   # tree exact -> i32 ok
    assert check_classical("knn", n_feat=3, n_param=10, in_dtype="i32", out_dtype="i32") == []   # knn exact -> i32 ok
    assert check_classical("svm", n_feat=5, n_param=8, in_dtype="i32", out_dtype="i32",
                           svm_kernel="linear") == []                                            # linear svm exact -> i32 ok
    assert check_classical("svm", n_feat=5, n_param=8, in_dtype="f32", out_dtype="f32",
                           svm_kernel="rbf") == []
    assert check_classical("nb", n_feat=3, n_param=2, in_dtype="f32", out_dtype="f32") == []


def test_check_classical_rejects_malformed_claims():
    assert any("not a known predictor" in e
               for e in check_classical("foo", n_feat=2, n_param=2, in_dtype="f32", out_dtype="f32"))
    assert any(">= 1" in e for e in check_classical("knn", n_feat=0, n_param=2, in_dtype="f32", out_dtype="f32"))
    assert any(">= 1" in e for e in check_classical("knn", n_feat=2, n_param=0, in_dtype="f32", out_dtype="f32"))
    assert any("not preserved" in e
               for e in check_classical("tree", n_feat=2, n_param=3, in_dtype="f32", out_dtype="i32"))
    assert any("not a known dtype" in e
               for e in check_classical("knn", n_feat=2, n_param=2, in_dtype="f64", out_dtype="f64"))
    # the quarantine dtype rule: RBF-SVM (exp) and Gaussian-NB (log) need f32.
    assert any("needs f32" in e for e in check_classical("svm", n_feat=2, n_param=2, in_dtype="i32",
                                                         out_dtype="i32", svm_kernel="rbf"))
    assert any("needs f32" in e for e in check_classical("nb", n_feat=2, n_param=2, in_dtype="i32",
                                                         out_dtype="i32"))
    # an unknown svm kernel is rejected.
    assert any("not a known kernel" in e for e in check_classical("svm", n_feat=2, n_param=2, in_dtype="f32",
                                                                  out_dtype="f32", svm_kernel="poly"))


# ============================================================================================================
# 7. The emitted classical-predict C kernels compile + reproduce the reference (self-skip when no toolchain).
# ============================================================================================================

def _cc():
    return shutil.which("clang") or shutil.which("cc") or shutil.which("gcc")


def _farr(v):
    return "{" + ", ".join(f"{f:.7f}f" for f in v) + "}"


def _iarr(v):
    return "{" + ", ".join(str(int(f)) for f in v) + "}"


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
        bld = subprocess.run(cmd, capture_output=True, text=True)
        assert bld.returncode == 0, bld.stderr
        out = subprocess.run([exe], capture_output=True, text=True)
        return [float(t) for t in out.stdout.split()]


def test_svm_rbf_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return                                                  # quick tier hides the toolchain -> self-skip
    rng = random.Random(0xE508)
    nf, nsv = 4, 5
    m = SvmModel(tuple(rng.uniform(-1, 1) for _ in range(nsv * nf)),
                 tuple(rng.uniform(-1, 1) for _ in range(nsv)), rng.uniform(-1, 1), nf)
    gamma = 0.6
    x = [rng.uniform(-1, 1) for _ in range(nf)]
    kernel = emit_svm_rbf_predict_c(nsv, nf, "svm_rbf")
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  float X[{nf}]={_farr(x)};\n"
            f"  float SV[{nsv * nf}]={_farr(m.SV)};\n"
            f"  float AY[{nsv}]={_farr(m.alpha_y)};\n"
            f"  float f = svm_rbf(X, SV, AY, {m.b:.7f}f, {gamma:.7f}f);\n"
            f'  printf("%.7f\\n", f);\n  return 0;\n}}\n')
    got = _compile_run(kernel, main, libm=True)[0]
    ref = svm_decision_rbf(m, x, gamma)
    # the C kernel calls the SAME libm expf; float32 vs the oracle's float64 differ by round-off only.
    assert abs(got - ref) < 1e-3, (got, ref)


def test_tree_c_kernel_matches_the_reference():
    cc = _cc()
    if not cc:
        return
    t = _toy_tree()
    kernel = emit_tree_predict_c(t.n_nodes, t.n_feat, "tree_p")
    # exercise all three leaves with three queries (compiled into one main that prints all three).
    queries = [[0.2, 0.2], [0.2, 0.9], [0.9, 0.0]]
    decls = "".join(f"  float X{i}[{t.n_feat}]={_farr(q)};\n" for i, q in enumerate(queries))
    calls = "".join(f'  printf("%.7f\\n", tree_p(X{i}, FE, TH, LE, RI, LV));\n' for i in range(len(queries)))
    main = (f"\n#include <stdio.h>\nint main(void){{\n"
            f"  int FE[{t.n_nodes}]={_iarr(t.feature)};\n"
            f"  float TH[{t.n_nodes}]={_farr(t.threshold)};\n"
            f"  int LE[{t.n_nodes}]={_iarr(t.left)};\n"
            f"  int RI[{t.n_nodes}]={_iarr(t.right)};\n"
            f"  float LV[{t.n_nodes}]={_farr(t.leaf_value)};\n"
            f"{decls}{calls}  return 0;\n}}\n")
    got = _compile_run(kernel, main, libm=False)                # the tree is EXACT -- no libm needed
    ref = [tree_predict(t, q) for q in queries]
    assert got == ref, (got, ref)                               # integer/float-exact (not merely round-off)


def test_classical_c_emit_routes_and_validates():
    # the RBF-SVM kernel rides the libm expf edge; the tree kernel is exact (no math.h, no libm).
    svm_src = emit_svm_rbf_predict_c(3, 2, "sr")
    assert "math.h" in svm_src and "expf" in svm_src and "c.call.libm:" in svm_src
    assert "float sr(const float *restrict x," in svm_src
    tree_src = emit_tree_predict_c(5, 2, "tp")
    assert "math.h" not in tree_src and "expf" not in tree_src   # the tree is exact -- NO transcendental, NO libm
    assert "c.call.libm:" not in tree_src
    assert "float tp(const float *restrict x," in tree_src
    for bad in [(0, 2), (2, 0)]:
        for fn in (emit_svm_rbf_predict_c, emit_tree_predict_c):
            try:
                fn(*bad)
                assert False, f"{fn.__name__}{bad} should raise"
            except ValueError:
                pass


# ============================================================================================================
# 8. The link-flag rule: expf/logf -> -lm (REUSES the existing libm rule; no linkflags change).
# ============================================================================================================

def test_classical_link_flag_rule_reuses_the_existing_libm_rule():
    # the RBF-SVM expf + the Gaussian-NB logf MATCH the existing libm rule -- NO linkflags change needed.
    assert library_for_callee("expf") == "-lm"                 # the RBF kernel edge
    assert library_for_callee("logf") == "-lm"                 # the Gaussian-NB normaliser edge (baked, but logf maps)
    assert library_for_callee("exp") == "-lm"                  # the double forms too
    assert library_for_callee("log") == "-lm"
    # no regression on the other rules (E1-E4 + libc + unknown).
    assert library_for_callee("sqrtf") == "-lm"                # E3 layernorm still maps
    assert library_for_callee("tanhf") == "-lm"                # E4 LSTM still maps
    assert library_for_callee("LAPACKE_ssyev") == "-llapack"   # E2 PCA still maps
    assert library_for_callee("cblas_sgemm") == "-lcblas"      # B5 BLAS still maps
    assert library_for_callee("free") == NO_FLAG               # libc-implicit, known
    assert library_for_callee("totally_unknown_fn") is None    # unknown-callee policy unchanged


# ============================================================================================================
# the two-truth quarantine: the classical module imports no verifier / emits no Diagnostic.
# ============================================================================================================

def test_classical_touches_no_verifier_and_emits_no_diagnostic():
    # the classical predictors are cost-side / oracle quantities (off the legality path), never an R-law verdict.
    # AST-inspect the module: no verify / Diagnostic / Graded names bound, no verifier module imported (exactly
    # as test_ols / test_pca / test_recurrent).
    import bcir.kbcir.classical as cl_mod
    bound = set(vars(cl_mod))
    assert not (bound & {"verify_quarantine", "Diagnostic", "Graded", "decide"}), sorted(bound)
    tree = ast.parse(open(cl_mod.__file__, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any("verify" in m for m in imported), sorted(imported)
    # the returned values are plain floats / plain ints (no graded/confidence wrapper).
    m = _two_gaussian_nb()
    assert isinstance(nb_predict(m, [0.3]), int)
    assert isinstance(gaussian_nb_log_posterior(m, [0.3]), list)
    assert isinstance(svm_decision_linear(SvmModel((1.0,), (1.0,), 0.0, 1), [2.0]), float)


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
